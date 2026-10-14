"""A document used to pin memory for its whole length, and nothing bounded it.

`ocr.max_concurrency` gates how many pages are BEING READ. `ocr_document` used to
rasterise every page of a document up front and hold the list until the response
was written, so the pages waiting on that semaphore were resident too. The real
figure was (in-flight documents x their page count), which no setting bounded.

Production evidence, 2026-10-14: OOM-killed at 90.7 GB with max_concurrency=1024,
then again at 92.7 GB with it at 128. Eight times smaller, half a gigabyte worse --
which is the proof that the semaphore was never what held the memory.

Two mechanisms, tested here:
  * `codai/ocr/budget.py` charges a document for its resident bytes BEFORE the
    pages exist, because afterwards the memory is already spent.
  * `_read_windowed` reads a window of pages at a time, so residency is
    O(window) rather than O(document).
"""

import asyncio
import io
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codai.ocr import budget as bud                            # noqa: E402
from codai.ocr import base as bse                              # noqa: E402
from codai.ocr.base import page_estimate, render_pages, load_pages   # noqa: E402


def _pdf(n_pages: int, w: int = 595, h: int = 842) -> bytes:
    """A real n-page PDF (A4 points by default) via the shipped rasteriser dep."""
    pdfium = pytest.importorskip("pypdfium2")
    doc = pdfium.PdfDocument.new()
    for _ in range(n_pages):
        doc.new_page(w, h)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# --------------------------------------------------------------- the budget #
def test_a_reservation_that_fits_is_admitted_immediately():
    b = bud.DocumentBudget(1000)

    async def _run():
        async with b.reserve(400):
            return b.used_bytes
    assert asyncio.run(_run()) == 400


def test_the_budget_is_returned_after_the_block():
    b = bud.DocumentBudget(1000)

    async def _run():
        async with b.reserve(400):
            pass
        return b.used_bytes
    assert asyncio.run(_run()) == 0


def test_a_second_document_waits_until_the_first_releases():
    """The whole point: the second one must NOT be admitted beside the first."""
    b = bud.DocumentBudget(1000)
    order = []

    async def _hold():
        async with b.reserve(700):
            order.append("first-in")
            await asyncio.sleep(0.05)
            order.append("first-out")

    async def _wait():
        await asyncio.sleep(0.01)          # let _hold take the budget first
        async with b.reserve(700):
            order.append("second-in")

    async def _run():
        await asyncio.gather(_hold(), _wait())
    asyncio.run(_run())
    assert order == ["first-in", "first-out", "second-in"]
    assert b.waits == 1


def test_two_documents_that_both_fit_run_together():
    b = bud.DocumentBudget(1000)
    peak = []

    async def _one():
        async with b.reserve(400):
            peak.append(b.used_bytes)
            await asyncio.sleep(0.02)

    async def _run():
        await asyncio.gather(_one(), _one())
    asyncio.run(_run())
    assert max(peak) == 800, "two 400-byte documents must coexist under 1000"
    assert b.waits == 0


def test_a_document_larger_than_the_whole_budget_still_runs():
    """A gate that can refuse forever is a deadlock with a polite error message."""
    b = bud.DocumentBudget(1000)

    async def _run():
        async with b.reserve(50_000):
            return b.used_bytes
    assert asyncio.run(_run()) == 1000, "clamped to the budget, not rejected"


def test_an_oversized_document_runs_alone_then_lets_the_next_in():
    b = bud.DocumentBudget(1000)
    seen = []

    async def _big():
        async with b.reserve(50_000):
            seen.append("big")
            await asyncio.sleep(0.03)

    async def _small():
        await asyncio.sleep(0.01)
        async with b.reserve(100):
            seen.append("small")

    async def _run():
        await asyncio.gather(_big(), _small())
    asyncio.run(_run())
    assert seen == ["big", "small"]


def test_the_budget_is_returned_when_the_document_raises():
    b = bud.DocumentBudget(1000)

    async def _run():
        with pytest.raises(ValueError):
            async with b.reserve(900):
                raise ValueError("bad scan")
        return b.used_bytes
    assert asyncio.run(_run()) == 0


def test_the_budget_is_returned_when_the_client_disconnects():
    """Cancellation is the case a naive implementation leaks: a disconnected
    client would shrink the budget permanently, one document at a time."""
    b = bud.DocumentBudget(1000)

    async def _run():
        async def _hold():
            async with b.reserve(900):
                await asyncio.sleep(10)
        t = asyncio.create_task(_hold())
        await asyncio.sleep(0.02)
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass
        return b.used_bytes
    assert asyncio.run(_run()) == 0


def test_a_cancelled_waiter_does_not_wedge_the_queue():
    """Someone waiting, then giving up, must not leave the next one blocked."""
    b = bud.DocumentBudget(1000)
    got = []

    async def _run():
        async def _hold():
            async with b.reserve(900):
                await asyncio.sleep(0.08)

        async def _waiter():
            async with b.reserve(900):
                got.append("waiter")

        async def _later():
            async with b.reserve(900):
                got.append("later")

        h = asyncio.create_task(_hold())
        await asyncio.sleep(0.01)
        w = asyncio.create_task(_waiter())
        await asyncio.sleep(0.01)
        w.cancel()
        await asyncio.gather(h, _later(), return_exceptions=True)
    asyncio.run(_run())
    assert got == ["later"]


def test_high_water_reports_the_peak():
    b = bud.DocumentBudget(1000)

    async def _run():
        async with b.reserve(300):
            async with b.reserve(400):
                pass
    asyncio.run(_run())
    assert b.high_water == 700


def test_resize_changes_the_ceiling_without_dropping_reservations():
    b = bud.DocumentBudget(1000)

    async def _run():
        async with b.reserve(400):
            b.resize(2000)
            assert b.used_bytes == 400
            return b.total_bytes
    assert asyncio.run(_run()) == 2000


# ------------------------------------------------------- the derived ceiling #
def test_an_auto_budget_prefers_the_cgroup_limit_over_host_ram(monkeypatch):
    """A 2 GB container on a 128 GB host that sizes itself from the host has no
    budget at all -- the container dies at its own limit, not the host's."""
    monkeypatch.setattr(bud, "_cgroup_limit_bytes", lambda: 8 * 1024**3)
    monkeypatch.setattr(bud, "_total_memory_bytes", lambda: 512 * 1024**3)
    assert bud.auto_budget_bytes() == int(8 * 1024**3 * bud.AUTO_FRACTION)


def test_an_auto_budget_uses_host_ram_when_uncontained(monkeypatch):
    monkeypatch.setattr(bud, "_cgroup_limit_bytes", lambda: None)
    monkeypatch.setattr(bud, "_total_memory_bytes", lambda: 64 * 1024**3)
    assert bud.auto_budget_bytes() == int(64 * 1024**3 * bud.AUTO_FRACTION)


def test_an_auto_budget_never_goes_below_the_floor(monkeypatch):
    """Below the floor a single ordinary document would serialise everything."""
    monkeypatch.setattr(bud, "_cgroup_limit_bytes", lambda: 256 * 1024**2)
    monkeypatch.setattr(bud, "_total_memory_bytes", lambda: 256 * 1024**2)
    assert bud.auto_budget_bytes() == bud.MIN_BUDGET_BYTES


def test_an_unknowable_host_still_gets_a_ceiling(monkeypatch):
    monkeypatch.setattr(bud, "_cgroup_limit_bytes", lambda: None)
    monkeypatch.setattr(bud, "_total_memory_bytes", lambda: None)
    assert bud.auto_budget_bytes() == bud.MIN_BUDGET_BYTES


def test_a_cgroup_sentinel_is_not_read_as_a_two_exabyte_limit(tmp_path):
    """cgroup v1 writes a huge sentinel for 'unlimited'; believing it is the same
    as having no ceiling at all."""
    f = tmp_path / "limit_in_bytes"
    f.write_text(str((1 << 63) - 1))
    assert bud._cgroup_limit_bytes([(str(f), None)]) is None


def test_the_cgroup_v2_unlimited_literal_is_understood(tmp_path):
    f = tmp_path / "memory.max"
    f.write_text("max\n")
    assert bud._cgroup_limit_bytes([(str(f), "max")]) is None


def test_a_real_cgroup_limit_is_read(tmp_path):
    f = tmp_path / "memory.max"
    f.write_text("2147483648\n")
    assert bud._cgroup_limit_bytes([(str(f), "max")]) == 2147483648


def test_a_missing_cgroup_file_is_not_an_error():
    assert bud._cgroup_limit_bytes([("/nope/does/not/exist", "max")]) is None


def test_the_configured_ceiling_overrides_the_derived_one():
    bud.reset_budget()
    try:
        b = bud.get_budget(2048)          # MB
        assert b.total_bytes == 2048 * 1024 * 1024
    finally:
        bud.reset_budget()


def test_zero_means_derive_it():
    bud.reset_budget()
    try:
        b = bud.get_budget(0)
        assert b.total_bytes == bud.auto_budget_bytes()
    finally:
        bud.reset_budget()


def test_changing_the_setting_resizes_the_live_budget():
    bud.reset_budget()
    try:
        bud.get_budget(1024)
        b = bud.get_budget(4096)
        assert b.total_bytes == 4096 * 1024 * 1024
    finally:
        bud.reset_budget()


# ------------------------------------------------------- pricing a document #
def test_a_pdf_is_priced_without_rendering_it():
    """The estimate has to come from the page tree: rendering to find out how
    much rendering costs is the bug."""
    data = _pdf(5)
    n, est = page_estimate(data, filename="x.pdf", dpi=250)
    assert n == 5
    per_page = est / n
    # A4 at 250 DPI is ~6.0 Mpixels; RGB is 3 bytes each -> ~17.3 MB, times the
    # measured PAGE_OVERHEAD (1.6) for decode transients -> ~27-28 MB.
    assert 26 * 1024**2 < per_page < 30 * 1024**2, per_page


def test_the_price_scales_with_dpi():
    data = _pdf(1)
    _, lo = page_estimate(data, filename="x.pdf", dpi=100)
    _, hi = page_estimate(data, filename="x.pdf", dpi=300)
    assert hi > lo * 8, "9x the pixels at 3x the DPI"


def test_the_price_scales_with_page_count():
    one = page_estimate(_pdf(1), filename="x.pdf", dpi=200)[1]
    ten = page_estimate(_pdf(10), filename="x.pdf", dpi=200)[1]
    assert 9 * one < ten < 11 * one


def test_an_undecodable_document_is_still_charged_something():
    """Charging an unknown document nothing is how an unbounded one slips past."""
    n, est = page_estimate(b"not a document at all", filename="x.pdf", dpi=200)
    assert n >= 1
    assert est > 0


def test_a_plain_image_is_priced_from_its_header():
    Image = pytest.importorskip("PIL.Image", reason="Pillow is a base dep")
    buf = io.BytesIO()
    Image.new("RGB", (1000, 2000)).save(buf, format="PNG")
    n, est = page_estimate(buf.getvalue(), filename="x.png", dpi=200)
    assert n == 1
    assert est == int(1000 * 2000 * 3 * bse.PAGE_OVERHEAD)


# ------------------------------------------------------------ page windowing #
def test_render_pages_is_lazy():
    """Nothing may be rendered until it is asked for, or windowing buys nothing."""
    data = _pdf(6)
    g = render_pages(data, filename="x.pdf", dpi=72)
    first = next(g)
    assert first.size[0] > 0
    assert len(list(g)) == 5, "the rest arrive one at a time, after the first"


def test_render_pages_agrees_with_load_pages():
    data = _pdf(4)
    streamed = [im.size for im in render_pages(data, filename="x.pdf", dpi=72)]
    whole = [im.size for im in load_pages(data, filename="x.pdf", dpi=72)]
    assert streamed == whole


def test_abandoning_a_document_stops_rendering_it():
    data = _pdf(50)
    g = render_pages(data, filename="x.pdf", dpi=72)
    next(g)
    g.close()                    # a client disconnected after page one
    with pytest.raises(StopIteration):
        next(g)


def test_a_multi_frame_image_yields_one_page_per_frame():
    Image = pytest.importorskip("PIL.Image", reason="Pillow is a base dep")
    buf = io.BytesIO()
    frames = [Image.new("RGB", (40, 40), c) for c in ("red", "green", "blue")]
    frames[0].save(buf, format="TIFF", save_all=True, append_images=frames[1:])
    got = list(render_pages(buf.getvalue(), filename="x.tiff", dpi=72))
    assert len(got) == 3


def test_an_undecodable_image_still_raises_through_the_generator():
    from codai.ocr.base import OcrError
    g = render_pages(b"junk", filename="x.png", dpi=72)
    with pytest.raises(OcrError) as e:
        next(g)
    assert e.value.status == 400


# ---------------------------------------- the document never renders ahead #
class _Cfg:
    """Just the fields the windowed read touches."""
    max_concurrency = 8
    max_resident_mb = 64
    page_window = 4
    blank_page_max_chars = 0        # measuring is a separate concern
    blank_page_ink = 0.0
    default_engine = "paddle"
    fallback_engines = "off"


def _manager(monkeypatch, n_pages, cfg=None, engine_for=None):
    """A manager whose rendering and engines are fakes, wired so the test can see
    exactly how far ahead the document was rendered when each window was read.

    Returns (manager, trace) where trace["pulled"] is the running count of pages
    rendered and trace["windows"] the (base, size, pulled_at_that_moment) triples.
    """
    from codai.ocr import manager as om
    from codai.ocr.base import OcrPage

    cfg = cfg or _Cfg()
    trace = {"pulled": 0, "windows": [], "reserved": []}

    def _fake_render(data, filename="", content_type="", dpi=200,
                     raster_workers=0, chunk=8):
        trace["chunk"] = chunk
        trace["raster_workers"] = raster_workers
        for i in range(n_pages):
            trace["pulled"] += 1
            yield f"page-{i}"                     # a stand-in for a PIL image
    monkeypatch.setattr(om, "render_pages", _fake_render)
    monkeypatch.setattr(om, "page_estimate",
                        lambda data, filename="", content_type="", dpi=200:
                        (n_pages, n_pages * 1024 * 1024))
    monkeypatch.setattr(om, "_wait_thermal_safe", lambda: None)

    class _Budget:
        total_bytes = 1 << 40

        def reserve(self, nbytes):
            trace["reserved"].append(nbytes)
            import contextlib

            @contextlib.asynccontextmanager
            async def _cm():
                yield
            return _cm()
    monkeypatch.setattr(om, "get_budget", lambda *a, **k: _Budget())

    m = om.OcrManager()
    m._require_cfg = lambda: cfg
    m.engine_dpi = lambda e: 200
    m.resolve_engine = lambda e: e or "paddle"

    async def _ocr_pages(images, engine=None, lang=None, base=0):
        trace["windows"].append((base, len(images), trace["pulled"]))
        name = (engine_for or (lambda b: engine or "paddle"))(base)
        return name, [OcrPage(index=base + i, text=f"t{base + i}")
                      for i in range(len(images))]
    m.ocr_pages = _ocr_pages

    async def _detect(images, pages, detect, base=0):
        return ([{"page": base}], [])
    m._detect_all = _detect
    return m, trace


def test_the_document_is_never_rendered_further_than_one_window_ahead(monkeypatch):
    """The regression. Before 0.3.65 every page existed before the first was read."""
    m, trace = _manager(monkeypatch, n_pages=40)
    out = asyncio.run(m.ocr_document(b"%PDF-x", filename="x.pdf"))
    assert out["num_pages"] == 40
    for base, size, pulled in trace["windows"]:
        assert pulled - base <= 4, (
            f"window at page {base} had {pulled - base} pages resident; "
            "the window is 4")


def test_every_page_is_still_read(monkeypatch):
    m, trace = _manager(monkeypatch, n_pages=40)
    out = asyncio.run(m.ocr_document(b"%PDF-x", filename="x.pdf"))
    assert trace["pulled"] == 40
    assert [p["index"] for p in out["pages"]] == list(range(40))


def test_the_last_window_may_be_short(monkeypatch):
    m, trace = _manager(monkeypatch, n_pages=10)       # window 4 -> 4, 4, 2
    asyncio.run(m.ocr_document(b"%PDF-x", filename="x.pdf"))
    assert [size for _, size, _ in trace["windows"]] == [4, 4, 2]


def test_a_single_page_document_is_one_window(monkeypatch):
    m, trace = _manager(monkeypatch, n_pages=1)
    out = asyncio.run(m.ocr_document(b"%PDF-x", filename="x.pdf"))
    assert [size for _, size, _ in trace["windows"]] == [1]
    assert out["num_pages"] == 1


def test_text_is_joined_in_page_order_across_windows(monkeypatch):
    m, _ = _manager(monkeypatch, n_pages=6)
    out = asyncio.run(m.ocr_document(b"%PDF-x", filename="x.pdf"))
    assert out["text"] == "\n\n".join(f"t{i}" for i in range(6))


def test_detections_are_numbered_by_document_page_not_window_page(monkeypatch):
    """A stamp on page 9 must not be reported as page 1 because it was the first
    page of the third window."""
    m, _ = _manager(monkeypatch, n_pages=12)
    out = asyncio.run(m.ocr_document(b"%PDF-x", filename="x.pdf"))
    assert [d["page"] for d in out["stamps"]] == [0, 4, 8]


def test_blank_pages_are_numbered_by_document_page(monkeypatch):
    from codai.ocr import manager as om

    class _BlankCfg(_Cfg):
        pass
    m, _ = _manager(monkeypatch, n_pages=12, cfg=_BlankCfg())
    # Every page of every window reads as blank -> indices must be 0..11, not
    # 0,1,2,3 repeated three times.
    monkeypatch.setattr(om, "_mark_blank_pages",
                        lambda cfg, images, pages: list(range(len(pages))))
    out = asyncio.run(m.ocr_document(b"%PDF-x", filename="x.pdf"))
    assert out["blank_pages"] == list(range(12))


def test_the_engine_is_sticky_across_windows(monkeypatch):
    """`ocr_pages` falls back per call, so without stickiness a document could be
    read by a different engine in every window and report only the last."""
    asked = []

    def _engine_for(base):
        return "doctr"                     # the first window fell back to doctr
    m, _ = _manager(monkeypatch, n_pages=12, engine_for=_engine_for)
    real = m.ocr_pages

    async def _spy(images, engine=None, lang=None, base=0):
        asked.append(engine)
        return await real(images, engine=engine, lang=lang, base=base)
    m.ocr_pages = _spy
    out = asyncio.run(m.ocr_document(b"%PDF-x", filename="x.pdf", engine="paddle"))
    assert out["engine"] == "doctr"
    assert asked == ["paddle", "doctr", "doctr"], (
        "after the first window chose an engine the rest must ask for it")


def test_the_reservation_is_window_sized_not_document_sized(monkeypatch):
    """Charging whole documents would queue work that never coexists."""
    m, trace = _manager(monkeypatch, n_pages=400)
    asyncio.run(m.ocr_document(b"%PDF-x", filename="x.pdf"))
    per_page = 1024 * 1024
    assert trace["reserved"] == [per_page * (4 + 2)], (
        "a 400-page document must reserve its window (+ render headroom), "
        "not 400 pages")


def test_a_short_document_reserves_only_what_it_has(monkeypatch):
    m, trace = _manager(monkeypatch, n_pages=2)
    asyncio.run(m.ocr_document(b"%PDF-x", filename="x.pdf"))
    assert trace["reserved"] == [1024 * 1024 * (2 + 2)]


# ------------------------------------------------------- the window setting #
def test_the_window_follows_max_concurrency_when_unset():
    from codai.ocr.manager import OcrManager

    class _C:
        page_window = 0
        max_concurrency = 12
    assert OcrManager()._page_window(_C()) == 12


def test_the_window_is_capped_so_it_cannot_reintroduce_the_problem():
    from codai.ocr.manager import OcrManager

    class _C:
        page_window = 0
        max_concurrency = 1024          # the value that was live when it OOMed
    assert OcrManager()._page_window(_C()) == 32


def test_an_explicit_window_wins():
    from codai.ocr.manager import OcrManager

    class _C:
        page_window = 3
        max_concurrency = 1024
    assert OcrManager()._page_window(_C()) == 3


def test_a_nonsense_window_never_yields_zero():
    from codai.ocr.manager import OcrManager

    class _C:
        page_window = None
        max_concurrency = 0
    assert OcrManager()._page_window(_C()) >= 1


# ------------------------------------------- a setting that actually arrives #
#
# Today's two production restarts were both "the setting did not apply": one
# because `max_concurrency` was never the right lever, one because the value was
# written to a config file nothing reads. A knob that exists in the dataclass and
# nowhere else is indistinguishable from a knob that does not work, so each of
# these has to be present at every layer it travels through.
_NEW_SETTINGS = ("max_resident_mb", "page_window")


def _read(rel):
    return (ROOT / rel).read_text()


@pytest.mark.parametrize("field", _NEW_SETTINGS)
def test_the_setting_exists_on_the_config_object(field):
    from codai.config import OcrConfig
    assert hasattr(OcrConfig(), field)


@pytest.mark.parametrize("field", _NEW_SETTINGS)
def test_the_setting_is_persisted(field):
    assert f'"{field}": self.config.ocr.{field}' in _read("codai/config.py")


@pytest.mark.parametrize("field", _NEW_SETTINGS)
def test_the_admin_api_reports_the_setting(field):
    assert f'"{field}": c.ocr.{field}' in _read("codai/admin/routes.py")


@pytest.mark.parametrize("field", _NEW_SETTINGS)
def test_the_admin_api_accepts_the_setting(field):
    src = _read("codai/admin/routes.py")
    assert f'if "{field}" in d: o.{field} = _i("{field}"' in src


@pytest.mark.parametrize("field", _NEW_SETTINGS)
def test_the_settings_page_loads_and_saves_the_setting(field):
    html = _read("codai/admin/templates/settings.html")
    el = "s-ocr-" + field.replace("_", "-")
    assert f"id=\"{el}\"" in html, f"no control for {field}"
    assert f"getElementById('{el}').value = ocr.{field}" in html
    assert f"{field}: parseInt(document.getElementById('{el}').value)" in html


def test_zero_survives_the_settings_page():
    """`|| 0` is correct here and `|| 64` would not be: 0 is the meaningful
    'derive it' value, so an emptied field must mean derive, not a default."""
    html = _read("codai/admin/templates/settings.html")
    for field in _NEW_SETTINGS:
        el = "s-ocr-" + field.replace("_", "-")
        line = [l for l in html.splitlines()
                if f"{field}: parseInt(document.getElementById('{el}')" in l][0]
        assert line.strip().endswith("|| 0,"), line


def test_the_budget_is_read_from_the_config_not_hardcoded():
    """The gate must follow the operator's ceiling, or it is just a constant."""
    src = _read("codai/ocr/manager.py")
    assert 'get_budget(int(getattr(cfg, "max_resident_mb", 0) or 0))' in src


def test_the_estimate_does_not_undercharge_a_real_page():
    """The factor exists because the pixel arithmetic under-predicted by 40-55%,
    and a budget that under-charges admits more than it meant to. Measured in
    tests/../codai/ocr/base.py's docstring; asserted here so a later edit that
    drops the factor has to face the number."""
    assert bse.PAGE_OVERHEAD >= 1.56, (
        "the worst measured ratio was 1.56; anything lower under-charges")


# -------------------------------------------- pdfium is not thread-safe #
#
# 0.3.65 moved rendering into `asyncio.to_thread` so a long PDF would stop
# stalling the event loop. That was correct in intent and broke OCR in
# production within the hour: until then `load_pages` ran ON the loop, which
# serialised every rasterisation BY ACCIDENT, and removing the accident let
# documents into pdfium concurrently. pdfium permits one caller at a time;
# concurrent use corrupts its state and reports
#
#     failed to rasterise PDF (pypdfium2): PDFium: Data format error
#
# on documents that are perfectly valid -- so the failure accuses the input.
# Measured before the fix: 1 document OK, 8 concurrent = 8 failures, 24 = 24.
#
# The lesson in test form: this path is only ever exercised concurrently, so
# testing it sequentially tests the one case that cannot fail.
def test_many_documents_rasterise_concurrently_without_corrupting_pdfium():
    import pypdfium2 as pdfium
    from codai.ocr.manager import _drain

    doc = pdfium.PdfDocument.new()
    for _ in range(4):
        doc.new_page(595, 842)
    buf = io.BytesIO()
    doc.save(buf)
    data = buf.getvalue()

    async def _one(k):
        page_estimate(data, filename=f"d{k}.pdf", dpi=100)
        g = render_pages(data, filename=f"d{k}.pdf", dpi=100)
        n = 0
        while True:
            imgs = await asyncio.to_thread(_drain, g, 2)
            if not imgs:
                break
            for im in imgs:
                im.tobytes()[:16]          # touch the pixels, as an engine would
            n += len(imgs)
        return n

    async def _run():
        return await asyncio.gather(*[_one(i) for i in range(16)],
                                    return_exceptions=True)
    got = asyncio.run(_run())
    bad = [g for g in got if isinstance(g, BaseException)]
    assert not bad, f"concurrent rasterisation failed: {bad[:3]}"
    assert got == [4] * 16, got


def test_the_whole_document_path_is_also_serialised():
    """`load_pages` is still used as the pdf2image fallback and by callers
    outside the manager, and it reaches the same unsafe library."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument.new()
    for _ in range(3):
        doc.new_page(595, 842)
    buf = io.BytesIO()
    doc.save(buf)
    data = buf.getvalue()

    async def _run():
        return await asyncio.gather(
            *[asyncio.to_thread(load_pages, data, f"w{i}.pdf", "", 100)
              for i in range(12)], return_exceptions=True)
    got = asyncio.run(_run())
    bad = [g for g in got if isinstance(g, BaseException)]
    assert not bad, f"concurrent load_pages failed: {bad[:3]}"
    assert [len(g) for g in got] == [3] * 12


def test_the_lock_is_held_around_every_pdfium_call():
    """Source-anchored: a later edit that adds a pdfium call outside the lock
    reintroduces a bug whose symptom blames the document, not the code."""
    src = (ROOT / "codai/ocr/base.py").read_text()
    for marker in ("pdfium.PdfDocument(data)", "page.render(scale=scale)",
                   "get_size()"):
        assert marker in src
    # Every PdfDocument construction sits under a `with _PDFIUM_LOCK:`
    for chunk in src.split("pdfium.PdfDocument(data)")[:-1]:
        assert "_PDFIUM_LOCK" in chunk.rsplit("try:", 1)[-1], (
            "a PdfDocument is built without holding _PDFIUM_LOCK")


def test_the_pool_chunk_matches_the_page_window(monkeypatch):
    """A pooled read borrows a worker per chunk and re-opens the document each
    time, so a chunk smaller than the window pays that cost for nothing."""
    m, trace = _manager(monkeypatch, n_pages=40)
    asyncio.run(m.ocr_document(b"%PDF-x", filename="x.pdf"))
    assert trace["chunk"] == 4, trace.get("chunk")      # _Cfg.page_window


def test_the_raster_worker_setting_reaches_the_renderer(monkeypatch):
    class _C(_Cfg):
        raster_workers = 3
    m, trace = _manager(monkeypatch, n_pages=8, cfg=_C())
    asyncio.run(m.ocr_document(b"%PDF-x", filename="x.pdf"))
    assert trace["raster_workers"] == 3
