"""Where the OCR pipeline runs, chosen by the IMAGE and nothing else.

    coderai-vllm      the pod serves the model; THIS box rasterises each page
                      and sends it as a base64 PNG (~1.06 MB per A4 page at
                      250 DPI, and again for every resumption round)
    coderai-vllm-ocr  the pod carries surya's client venv and pypdfium2, so it
                      is sent a ONE-PAGE PDF (~474 KB, 1.7 ms to split here
                      against ~105 ms to rasterise) and reads it on the card,
                      with its own vLLM over loopback

A separate `pod_ocr_pipeline` flag was built first and removed: two switches
that can disagree is the invisible mismatch that sent three rounds of blame at a
pod image. Choosing the image IS choosing the mode.

Two properties these tests exist to hold:

* page-level fan-out is unchanged -- one page is one request either way, so no
  pod owns a document and every pod in the pool shares the work;
* the pod-side path is an OPTIMISATION, so anything that stops it working falls
  back to the local pipeline rather than failing a document.
"""

import asyncio
import io
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codai.api import runpod_worker as rw                      # noqa: E402
from codai.ocr import manager as om                            # noqa: E402
from codai.ocr.base import OcrError, iter_pdf_page_pdfs, pdf_page_count  # noqa: E402


def _pdf(n=3):
    pdfium = pytest.importorskip("pypdfium2")
    d = pdfium.PdfDocument.new()
    for _ in range(n):
        d.new_page(595, 842)
    b = io.BytesIO()
    d.save(b)
    return b.getvalue()


# ------------------------------------------------------- the image decides #
@pytest.mark.parametrize("image,expected", [
    ("ghcr.io/nextime/coderai-vllm-ocr:0.3.69", True),
    ("ghcr.io/nextime/coderai-vllm-ocr:latest", True),
    ("my-mirror.example.com/coderai-vllm-ocr:pinned", True),
    ("ghcr.io/nextime/coderai-vllm:latest", False),
    ("ghcr.io/nextime/coderai-ocr:latest", False),
    ("vllm/vllm-openai:latest", False),
    ("", False),
])
def test_only_the_pipeline_image_runs_the_pipeline(image, expected):
    assert rw.image_runs_ocr_pipeline(image) is expected


def test_a_private_mirror_or_a_pinned_tag_still_matches():
    """Matched on the name, because the decision is made before a pod exists."""
    assert rw.image_runs_ocr_pipeline("reg.internal:5000/ns/coderai-vllm-ocr:0.3.69")


def test_there_is_no_second_switch():
    """The flag that could disagree with the image is gone, everywhere."""
    for rel in ("codai/api/runpod_worker.py", "codai/admin/routes.py",
                "codai/admin/templates/models.html"):
        src = (ROOT / rel).read_text()
        assert "pod_ocr_pipeline" not in src.replace("pod_ocr_pipeline()", ""), rel


def test_pod_runs_ocr_pipeline_reads_the_models_image():
    class _M:
        image = "ghcr.io/nextime/coderai-vllm-ocr:latest"
    assert rw.pod_runs_ocr_pipeline(_M()) is True

    class _N:
        image = "ghcr.io/nextime/coderai-vllm:latest"
    assert rw.pod_runs_ocr_pipeline(_N()) is False


# --------------------------------------------------- what the pod is told #
def _plan_env(image, served="datalab-to/surya-ocr-2"):
    """The env a real _plan() produces for a pod on `image`.

    Built from the actual dataclass rather than a stub: _plan reads a dozen
    fields (health_path, volumes, venv_on_volume...) and a hand-rolled fake
    tests the fake instead of the code."""
    m = rw.RunpodModelConfig()
    m.image = image
    m.port = 8000
    plan = rw._plan("coderai", image, "", m, "tok", {}, served,
                    None, capability="text")
    return plan.get("env") or {}


def test_a_pipeline_pod_is_told_to_switch_ocr_on(monkeypatch):
    class _O:
        dpi = 200
        surya_dpi = 250
        surya_max_tokens = 3072
        surya_max_continuations = 8
        surya_repeat_max_repeats = 0
        detect_mode = "off"
        detect_model_path = ""
    monkeypatch.setattr(rw, "_ocr_cfg", lambda: _O())
    env = _plan_env("ghcr.io/nextime/coderai-vllm-ocr:latest")
    assert env["CODERAI_OCR_ENABLED"] == "1"
    assert env["CODERAI_OCR_SURYA_ENABLED"] == "1"
    assert env["CODERAI_OCR_DEFAULT_ENGINE"] == "surya"
    assert env["CODERAI_OCR_SURYA_ACCEPT_LICENSE"] == "1"


def test_it_drives_its_own_model_over_loopback(monkeypatch):
    """`model` mode, so the pod's own gateway mints surya an attach token --
    no bearer hand-wired into the env, and a resumed page is a local round
    trip."""
    class _O:
        dpi = 0
    monkeypatch.setattr(rw, "_ocr_cfg", lambda: _O())
    env = _plan_env("ghcr.io/nextime/coderai-vllm-ocr:latest",
                    served="datalab-to/surya-ocr-2")
    assert env["CODERAI_OCR_SURYA_SERVE"] == "model"
    assert env["CODERAI_OCR_SURYA_MODEL_ID"] == "datalab-to/surya-ocr-2"


def test_it_is_pointed_at_the_baked_venv(monkeypatch):
    """Not told to BUILD one on a rented machine."""
    monkeypatch.setattr(rw, "_ocr_cfg", lambda: type("O", (), {"dpi": 0})())
    env = _plan_env("ghcr.io/nextime/coderai-vllm-ocr:latest")
    assert env["CODERAI_OCR_SURYA_VENV"] == "/opt/coderai/venvs/surya"


def test_the_fallback_chain_names_nothing_the_image_lacks(monkeypatch):
    """This image carries NO docTR; `auto` would route a failed page to an
    engine that is not there."""
    monkeypatch.setattr(rw, "_ocr_cfg", lambda: type("O", (), {"dpi": 0})())
    env = _plan_env("ghcr.io/nextime/coderai-vllm-ocr:latest")
    assert env["CODERAI_OCR_FALLBACK_ENGINES"] == "off"


def test_the_tuning_travels(monkeypatch):
    """A pod reading pages at surya's own defaults is indistinguishable from the
    'pre-fix pod image' that got blamed three times for orchestrator bugs. Every
    one of these was set in response to a measured failure."""
    class _O:
        dpi = 200
        surya_dpi = 250
        surya_max_tokens = 3072
        surya_max_continuations = 4
        surya_context_tokens = 18432
        surya_repeat_max_repeats = 6
        surya_repeat_window = 400
        surya_repeat_scaling = 2.5
        surya_langs = "it"
        max_concurrency = 128
        page_window = 8
        surya_repair_accents = True
        detect_mode = "off"
        detect_model_path = ""
    monkeypatch.setattr(rw, "_ocr_cfg", lambda: _O())
    env = _plan_env("ghcr.io/nextime/coderai-vllm-ocr:latest")
    assert env["CODERAI_OCR_SURYA_DPI"] == "250"
    assert env["CODERAI_OCR_SURYA_MAX_TOKENS"] == "3072"
    assert env["CODERAI_OCR_SURYA_MAX_CONTINUATIONS"] == "4"
    assert env["CODERAI_OCR_SURYA_CONTEXT_TOKENS"] == "18432"
    assert env["CODERAI_OCR_SURYA_REPEAT_MAX_REPEATS"] == "6"
    assert env["CODERAI_OCR_SURYA_REPEAT_WINDOW"] == "400"
    assert env["CODERAI_OCR_SURYA_REPEAT_SCALING"] == "2.5"
    assert env["CODERAI_OCR_SURYA_LANGS"] == "it"
    assert env["CODERAI_OCR_SURYA_REPAIR_ACCENTS"] == "1"


def test_a_plain_vllm_pod_is_told_nothing_about_ocr(monkeypatch):
    """coderai-vllm must behave exactly as it does today."""
    monkeypatch.setattr(rw, "_ocr_cfg", lambda: type("O", (), {"dpi": 0})())
    env = _plan_env("ghcr.io/nextime/coderai-vllm:latest")
    assert "CODERAI_OCR_SURYA_SERVE" not in env
    assert env.get("CODERAI_OCR_ENABLED") is None


def test_the_config_seeding_can_read_every_var_the_plan_writes():
    """A var the plan sends and the config never reads is a setting that
    silently does not apply -- the exact shape of two bugs already today."""
    plan_src = (ROOT / "codai/api/runpod_worker.py").read_text()
    cfg_src = (ROOT / "codai/config.py").read_text()
    block = plan_src.split("if pod_runs_ocr_pipeline(mcfg, image):")[1] \
                    .split('if cap == "ocr":')[0]
    import re
    sent = set(re.findall(r'"(CODERAI_OCR_[A-Z_]+)"', block))
    assert sent, "no env vars found in the pipeline block"
    # config.py builds two families from an f-string over the engine name --
    # CODERAI_OCR_<ENGINE>_ENABLED and _VENV -- so the literal never appears.
    dynamic = ("_ENABLED", "_VENV")
    for var in sorted(sent):
        if var.endswith(dynamic):
            assert 'CODERAI_OCR_{name.upper()}' in cfg_src, (
                f"{var} relies on the per-engine pattern and that pattern is gone")
            continue
        assert var in cfg_src, f"{var} is sent to the pod but never read"


# ----------------------------------------------------- the per-page split #
def test_each_page_leaves_as_its_own_pdf():
    data = _pdf(4)
    parts = list(iter_pdf_page_pdfs(data))
    assert len(parts) == 4
    assert all(pdf_page_count(p) == 1 for p in parts)


def test_splitting_is_far_cheaper_than_rasterising():
    """The point of the whole design: 1.7 ms/page measured against ~105 ms."""
    import time
    data = _pdf(8)
    t0 = time.time()
    list(iter_pdf_page_pdfs(data))
    per_page = (time.time() - t0) / 8
    assert per_page < 0.05, f"{per_page*1000:.0f} ms/page"


def test_an_abandoned_document_stops_splitting():
    g = iter_pdf_page_pdfs(_pdf(40))
    next(g)
    g.close()
    with pytest.raises(StopIteration):
        next(g)


# ------------------------------------------- the pipeline choice at runtime #
class _Cfg:
    surya_serve = "model"
    surya_model_id = "datalab-to/surya-ocr-2"
    max_concurrency = 8
    page_window = 4
    blank_page_max_chars = 0
    default_engine = "surya"
    fallback_engines = "off"


def test_the_model_mode_and_image_together_select_the_pod_path(monkeypatch):
    monkeypatch.setattr(rw, "_model_entry", lambda mid: {
        "runpod": {"image": "ghcr.io/nextime/coderai-vllm-ocr:latest"}})
    assert om.pod_pipeline_model(_Cfg(), "surya") == "datalab-to/surya-ocr-2"


def test_a_plain_vllm_image_keeps_the_local_path(monkeypatch):
    monkeypatch.setattr(rw, "_model_entry", lambda mid: {
        "runpod": {"image": "ghcr.io/nextime/coderai-vllm:latest"}})
    assert om.pod_pipeline_model(_Cfg(), "surya") == ""


def test_a_non_model_serve_mode_keeps_the_local_path(monkeypatch):
    class _C(_Cfg):
        surya_serve = "local"
    monkeypatch.setattr(rw, "_model_entry", lambda mid: {
        "runpod": {"image": "ghcr.io/nextime/coderai-vllm-ocr:latest"}})
    assert om.pod_pipeline_model(_C(), "surya") == ""


def test_another_engine_never_takes_the_surya_pod_path(monkeypatch):
    monkeypatch.setattr(rw, "_model_entry", lambda mid: {
        "runpod": {"image": "ghcr.io/nextime/coderai-vllm-ocr:latest"}})
    assert om.pod_pipeline_model(_Cfg(), "chandra") == ""
    assert om.pod_pipeline_model(_Cfg(), "olmocr") == ""


def test_a_missing_model_entry_keeps_the_local_path(monkeypatch):
    monkeypatch.setattr(rw, "_model_entry", lambda mid: None)
    assert om.pod_pipeline_model(_Cfg(), "surya") == ""



def _fake_placement(monkeypatch, kind="pod", on_choose=None):
    """Patch the REAL placement module's functions.

    Not `sys.modules`: `from codai.models import placement` resolves `placement`
    as an ATTRIBUTE of the already-imported package, so a sys.modules entry is
    bypassed whenever something else imported it first. These tests passed alone
    and failed in the full suite until this was fixed -- the classic shape of an
    order-dependent test.
    """
    from codai.models import placement as real

    class _Lease:
        def __init__(self):
            self.kind = kind
            self.base = "https://1.2.3.4:8000/v1"
            self.api_key = "k"

        def release(self):
            pass

    def _choose(mid, spill_after=None):
        if on_choose is not None:
            on_choose()
        return _Lease()
    monkeypatch.setattr(real, "choose", _choose)
    monkeypatch.setattr(real, "spill_threshold", lambda mid: 0)


# --------------------------------------------- it falls back, never fails #
def test_no_pod_means_the_local_pipeline(monkeypatch):
    """`placement.choose` answering "local" means nothing is rented. The whole
    document is then read locally -- not half one way and half the other, which
    is the hardest kind of result to explain."""
    mgr = om.OcrManager()
    mgr._require_cfg = lambda: _Cfg()
    mgr._sem = asyncio.Semaphore(4)

    _fake_placement(monkeypatch, kind="local")

    out = asyncio.run(mgr._read_on_pods(
        _pdf(3), "d.pdf", "application/pdf", "m",
        dpi=250, page_opts=None, detect=None, cfg=_Cfg()))
    assert out is None


def test_a_pod_error_is_not_swallowed(monkeypatch):
    """A pod that answers 500 is a real failure: raise it rather than quietly
    re-reading the document locally and hiding a broken fleet."""
    mgr = om.OcrManager()
    mgr._require_cfg = lambda: _Cfg()
    mgr._sem = asyncio.Semaphore(4)

    _fake_placement(monkeypatch)
    monkeypatch.setattr(om, "_post_page_to_pod",
                        lambda *a, **k: (_ for _ in ()).throw(
                            OcrError("pod said 500", status=500)))
    with pytest.raises(OcrError):
        asyncio.run(mgr._read_on_pods(
            _pdf(2), "d.pdf", "application/pdf", "m",
            dpi=250, page_opts=None, detect=None, cfg=_Cfg()))


def test_pages_come_back_in_document_order(monkeypatch):
    mgr = om.OcrManager()
    mgr._require_cfg = lambda: _Cfg()
    mgr._sem = asyncio.Semaphore(4)

    _fake_placement(monkeypatch)

    seen = []

    def _post(base, key, blob, name, dpi, opts, detect, engine="surya"):
        seen.append(name)
        return {"engine": engine, "pages": [{"text": name, "width": 1, "height": 2}]}
    monkeypatch.setattr(om, "_post_page_to_pod", _post)
    name, pages, blank, stamps, sigs = asyncio.run(mgr._read_on_pods(
        _pdf(5), "d.pdf", "application/pdf", "m",
        dpi=250, page_opts=None, detect=None, cfg=_Cfg()))
    assert name == "surya"
    assert [p.index for p in pages] == [0, 1, 2, 3, 4]
    assert [p.text for p in pages] == [f"{i}.pdf" for i in range(5)]
    assert sorted(seen) == sorted(f"{i}.pdf" for i in range(5))


def test_one_page_is_one_request_so_fan_out_survives(monkeypatch):
    """Not one document to one pod: `placement.choose` is called per page, so
    every pod in the pool shares the work."""
    mgr = om.OcrManager()
    mgr._require_cfg = lambda: _Cfg()
    mgr._sem = asyncio.Semaphore(8)
    chose = {"n": 0}
    _fake_placement(monkeypatch, on_choose=lambda: chose.__setitem__("n", chose["n"] + 1))
    monkeypatch.setattr(om, "_post_page_to_pod",
                        lambda *a, **k: {"engine": "surya", "pages": [{"text": "x"}]})
    asyncio.run(mgr._read_on_pods(_pdf(7), "d.pdf", "application/pdf", "m",
                                  dpi=250, page_opts=None, detect=None, cfg=_Cfg()))
    assert chose["n"] == 7, "the upstream must be chosen per PAGE"


def test_a_single_image_is_sent_whole(monkeypatch):
    """Nothing to split; it goes as it arrived."""
    Image = pytest.importorskip("PIL.Image")
    buf = io.BytesIO()
    Image.new("RGB", (40, 40)).save(buf, format="PNG")
    mgr = om.OcrManager()
    mgr._require_cfg = lambda: _Cfg()
    mgr._sem = asyncio.Semaphore(4)

    _fake_placement(monkeypatch)
    sent = []
    monkeypatch.setattr(om, "_post_page_to_pod",
                        lambda b, k, blob, name, *a: sent.append(len(blob)) or
                        {"engine": "surya", "pages": [{"text": "x"}]})
    asyncio.run(mgr._read_on_pods(buf.getvalue(), "x.png", "image/png", "m",
                                  dpi=250, page_opts=None, detect=None, cfg=_Cfg()))
    assert len(sent) == 1


# ------------------------------------------------------- the pod response #
def test_the_diagnostics_survive_the_boundary():
    """repeat_detected and friends are the whole reason a caller can tell a
    looping page from a blank one. Dropping them here would make a pod-read page
    less explicable than a locally-read one, for nothing."""
    page = om._page_from_pod({"text": "t", "width": 10, "height": 20,
                              "repeat_detected": True, "repeat_chars": 4096,
                              "model_attempts": [{"why": "x"}]}, 3)
    assert page.index == 3
    assert page.text == "t"
    assert page.meta["repeat_detected"] is True
    assert page.meta["repeat_chars"] == 4096
    assert page.meta["model_attempts"] == [{"why": "x"}]


def test_a_bare_page_needs_no_diagnostics():
    page = om._page_from_pod({"text": "t"}, 0)
    assert page.text == "t"
    assert page.meta == {}


def test_the_pod_url_is_built_from_the_lease_base(monkeypatch):
    from codai.api import pod_http
    seen = {}

    class _R:
        status_code = 200

        def json(self):
            return {"engine": "surya", "pages": []}
    monkeypatch.setattr(pod_http, "post",
                        lambda url, **kw: seen.update(url=url, kw=kw) or _R())
    om._post_page_to_pod("https://1.2.3.4:8000/v1", "tok", b"%PDF-x", "0.pdf",
                         250, {"max_tokens": 3072}, None)
    assert seen["url"] == "https://1.2.3.4:8000/v1/ocr"
    assert seen["kw"]["headers"]["Authorization"] == "Bearer tok"
    assert seen["kw"]["data"]["engine"] == "surya"
    assert seen["kw"]["data"]["dpi"] == "250"
    assert seen["kw"]["data"]["max_tokens"] == "3072"


def test_it_goes_through_pod_http_not_requests(monkeypatch):
    """A pod serves HTTPS with THIS install's CA on a bare IP; plain requests
    fails that twice over (see tests/test_olmocr_pod_tls.py)."""
    src = (ROOT / "codai/ocr/manager.py").read_text()
    fn = src.split("def _post_page_to_pod")[1].split("def _page_from_pod")[0]
    assert "pod_http.post(" in fn
    assert "requests.post(" not in fn


def test_an_unreachable_pod_is_a_503(monkeypatch):
    from codai.api import pod_http
    monkeypatch.setattr(pod_http, "post",
                        lambda url, **kw: (_ for _ in ()).throw(OSError("refused")))
    with pytest.raises(OcrError) as e:
        om._post_page_to_pod("https://1.2.3.4:8000/v1", "", b"x", "0.pdf",
                             0, {}, None)
    assert e.value.status == 503


# ------------------------------------------- which path ran must be visible #
def test_a_silent_fallback_would_make_a_test_prove_nothing():
    """An install configured for the pod-side pipeline that falls back on every
    request looks EXACTLY like a success. Without a line in the log an operator
    would conclude the new path worked when it had never run -- so both the
    fallback and the success say which one happened."""
    src = (ROOT / "codai/ocr/manager.py").read_text()
    fn = src.split("async def _read_on_pods")[1].split("\n    def ")[0]
    assert "but {_n} of {len(parts)} page(s) had no pod" in fn
    assert "ON THE PODS" in fn


def test_the_fallback_message_names_the_cause_not_just_the_effect():
    src = (ROOT / "codai/ocr/manager.py").read_text()
    fn = src.split("async def _read_on_pods")[1].split("\n    def ")[0]
    assert "placement answered 'local'" in fn
    assert "nothing rented" in fn


# ------------------------------- all three VLM engines, not just surya #
#
# surya needed a 1.6 GB client venv baked into the image. chandra and olmOCR in
# served mode speak only HTTP (PIL + requests, already in the main venv), so for
# them the pod side costs nothing beyond the rasteriser. Leaving them on the
# orchestrator while surya moved was an asymmetry, not a design.
class _MultiCfg:
    surya_serve = "model"
    surya_model_id = "datalab-to/surya-ocr-2"
    chandra_serve = "model"
    chandra_model_id = "datalab-to/chandra-ocr-2"
    olmocr_serve = "model"
    olmocr_model_id = "allenai/olmOCR-2-7B-1025-FP8"
    default_engine = "surya"
    max_concurrency = 8
    page_window = 4
    blank_page_max_chars = 0
    fallback_engines = "off"


@pytest.mark.parametrize("engine,expect", [
    ("surya", "datalab-to/surya-ocr-2"),
    ("chandra", "datalab-to/chandra-ocr-2"),
    ("olmocr", "allenai/olmOCR-2-7B-1025-FP8"),
])
def test_every_vlm_engine_can_take_the_pod_path(monkeypatch, engine, expect):
    monkeypatch.setattr(rw, "_model_entry", lambda mid: {
        "runpod": {"image": "ghcr.io/nextime/coderai-vllm-ocr:latest"}})
    assert om.pod_pipeline_model(_MultiCfg(), engine) == expect


@pytest.mark.parametrize("engine", ["surya", "chandra", "olmocr"])
def test_a_plain_vllm_image_keeps_every_engine_local(monkeypatch, engine):
    monkeypatch.setattr(rw, "_model_entry", lambda mid: {
        "runpod": {"image": "ghcr.io/nextime/coderai-vllm:latest"}})
    assert om.pod_pipeline_model(_MultiCfg(), engine) == ""


def test_a_non_vlm_engine_never_takes_the_pod_path(monkeypatch):
    """paddle and docTR run in-process; there is no model server to drive."""
    monkeypatch.setattr(rw, "_model_entry", lambda mid: {
        "runpod": {"image": "ghcr.io/nextime/coderai-vllm-ocr:latest"}})
    for eng in ("paddle", "doctr", "nonsense"):
        assert om.pod_pipeline_model(_MultiCfg(), eng) == ""


def test_each_engine_reads_its_own_serve_mode(monkeypatch):
    """chandra on `server` mode must not be dragged onto the pod path by
    surya's `model` mode."""
    class _C(_MultiCfg):
        chandra_serve = "server"
    monkeypatch.setattr(rw, "_model_entry", lambda mid: {
        "runpod": {"image": "ghcr.io/nextime/coderai-vllm-ocr:latest"}})
    assert om.pod_pipeline_model(_C(), "chandra") == ""
    assert om.pod_pipeline_model(_C(), "surya") == "datalab-to/surya-ocr-2"


def test_the_dispatch_tells_the_pod_WHICH_engine(monkeypatch):
    """Hardcoded "surya" sent a chandra page to surya, which the pod would have
    served with the wrong model -- and the text would have come back looking
    plausible."""
    from codai.api import pod_http
    seen = {}

    class _R:
        status_code = 200

        def json(self):
            return {"engine": "chandra", "pages": []}
    monkeypatch.setattr(pod_http, "post",
                        lambda url, **kw: seen.update(kw) or _R())
    om._post_page_to_pod("https://1.2.3.4:8000/v1", "t", b"%PDF-x", "0.pdf",
                         250, {}, None, "chandra")
    assert seen["data"]["engine"] == "chandra"


def test_the_pod_env_is_shaped_by_the_served_model_not_hardcoded():
    src = (ROOT / "codai/api/runpod_worker.py").read_text()
    blk = src.split("if pod_runs_ocr_pipeline(mcfg, image):")[1] \
             .split('if cap == "ocr":')[0]
    assert 'env[f"CODERAI_OCR_{_eng.upper()}_ENABLED"]' in blk
    assert 'env[f"CODERAI_OCR_{_eng.upper()}_SERVE"]' in blk
    assert '"CODERAI_OCR_SURYA_ENABLED"' not in blk, "still hardcoded to surya"


def test_only_surya_is_pointed_at_a_baked_venv():
    """chandra and olmOCR have none -- they speak HTTP from the main venv, which
    is why the pod side is nearly free for them."""
    src = (ROOT / "codai/api/runpod_worker.py").read_text()
    blk = src.split("if pod_runs_ocr_pipeline(mcfg, image):")[1] \
             .split('if cap == "ocr":')[0]
    assert 'if _eng == "surya":' in blk
    venv = blk.split('CODERAI_OCR_SURYA_VENV')[0]
    assert venv.rstrip().endswith('if _eng == "surya":') or 'if _eng == "surya"' in venv


def test_each_engine_inherits_its_own_licence_decision():
    """surya is GPL, chandra's weights are a modified OpenRAIL-M. A pod cannot
    accept either for itself."""
    src = (ROOT / "codai/api/runpod_worker.py").read_text()
    blk = src.split("if pod_runs_ocr_pipeline(mcfg, image):")[1] \
             .split('if cap == "ocr":')[0]
    assert "CODERAI_OCR_SURYA_ACCEPT_LICENSE" in blk
    assert "CODERAI_OCR_CHANDRA_ACCEPT_LICENSE" in blk
