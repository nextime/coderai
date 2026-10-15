"""A page can be RUINED without anything ever failing.

Reported from a 1,000-document run: ~97% came out `troncato`. Nothing raised,
nothing was logged as an error, and `fallback_engines` never fired -- because
the engine answered 200 every time. Its decode had fallen into a loop, run to
the token cap emitting the same fragment, and the resumer correctly refused to
continue a reply that was already repeating itself. The flag was set, the page
was returned, and the index swallowed it.

The same pages read cleanly on another engine (38,055 characters against a
truncated stub). So the recovery was available the whole time and nothing
reached for it: `fallback_engines` only fires on an exception, and there was
none.

These tests hold the shape of the fix:

* a degraded page is re-read on ANOTHER engine, not retried on the one that
  produced it (that retry is what set the flag);
* only degraded pages are re-read, so a clean document costs nothing;
* the replacement is kept only if it is NOT degraded in turn;
* the page keeps its own index and says who re-read it and why;
* it is OFF on a pod, because only the orchestrator can reach another engine.
"""

import asyncio
import io
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codai.ocr import manager as om                            # noqa: E402
from codai.ocr.base import OcrPage                             # noqa: E402


def _pdf(n=3):
    pdfium = pytest.importorskip("pypdfium2")
    d = pdfium.PdfDocument.new()
    for _ in range(n):
        d.new_page(595, 842)
    b = io.BytesIO()
    d.save(b)
    return b.getvalue()


def _looping(i, text="riga riga riga"):
    return OcrPage(index=i, text=text, meta={"repeat_detected": True})


def _truncated(i, text="meta pagina"):
    return OcrPage(index=i, text=text,
                   meta={"model_attempts": {"truncated": True, "complete": False}})


def _clean(i, text="una pagina intera, letta per bene."):
    return OcrPage(index=i, text=text)


class _Cfg:
    retry_degraded = "auto"
    retry_degraded_max_pages = 0
    fallback_engines = "auto"


def _mgr(cfg=None):
    m = om.OcrManager.__new__(om.OcrManager)
    m._cfg = cfg or _Cfg()
    m._sem = None
    m._require_cfg = lambda: m._cfg
    return m


# ------------------------------------------------------- what counts as bad #
def test_a_looping_page_is_degraded():
    assert om.OcrManager._page_degraded(_looping(0)) == "looping"


def test_a_truncated_page_is_degraded():
    assert om.OcrManager._page_degraded(_truncated(0)) == "truncated"


def test_a_clean_page_is_not():
    assert om.OcrManager._page_degraded(_clean(0)) == ""


def test_a_short_page_is_not_degraded():
    """A short page has four causes and they need opposite handling. Only the
    two the engine itself flags are re-read; `unreadable_report` keeps the rest,
    because re-reading a genuinely blank page spends a card to be told again
    that it is blank."""
    assert om.OcrManager._page_degraded(OcrPage(index=0, text="")) == ""
    assert om.OcrManager._page_degraded(OcrPage(index=0, text="x")) == ""


def test_a_resumed_but_complete_page_is_not_degraded():
    """`resumed` without `truncated` means the reply ran past the cap and WAS
    finished. That page is whole and must not be read twice."""
    p = OcrPage(index=0, text="lunga ma intera",
                meta={"model_attempts": {"resumed": 3, "calls": 4}})
    assert om.OcrManager._page_degraded(p) == ""


def test_degradation_survives_a_meta_that_is_not_a_dict():
    p = OcrPage(index=0, text="x", meta={"model_attempts": "oops"})
    assert om.OcrManager._page_degraded(p) == ""


# ------------------------------------------------------------- which engines #
def test_auto_uses_the_fallback_order():
    m = _mgr()
    m._fallback_order = lambda failed: ["chandra", "doctr"]
    assert m._retry_degraded_order("surya") == ["chandra", "doctr"]


def test_off_means_keep_what_the_first_engine_produced():
    cfg = _Cfg()
    cfg.retry_degraded = "off"
    m = _mgr(cfg)
    m._fallback_order = lambda failed: ["chandra"]
    assert m._retry_degraded_order("surya") == []


def test_an_explicit_order_never_includes_the_engine_that_failed():
    """Re-reading on the same engine is the retry that already produced the
    flag -- chandra raises the temperature itself and still looped."""
    cfg = _Cfg()
    cfg.retry_degraded = "surya, chandra"
    m = _mgr(cfg)
    m._fallback_order = lambda failed: []
    order = m._retry_degraded_order("surya")
    assert "surya" not in order


# ------------------------------------------------------------ the re-read #
def _run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


def _wire(m, answers, seen):
    """`answers` maps engine name -> callable(i) -> OcrPage or None."""
    async def _read(blobs, engine, **kw):
        seen.append((engine, len(blobs)))
        return [answers[engine](i) for i in range(len(blobs))]
    m._read_page_blobs = _read


def test_only_the_degraded_pages_are_re_read():
    m = _mgr()
    m._fallback_order = lambda f: ["chandra"]
    pages = [_clean(0), _looping(1), _clean(2)]
    seen = []
    _wire(m, {"chandra": lambda i: _clean(99)}, seen)
    note = _run(m._retry_degraded(_pdf(3), "a.pdf", "application/pdf", "surya",
                                  pages, dpi=200, page_opts={}, detect=None,
                                  lang="", cfg=m._cfg, window=8))
    assert seen == [("chandra", 1)], seen
    assert note and "chandra" in note
    assert pages[0].text == "una pagina intera, letta per bene."
    assert pages[1].text == "una pagina intera, letta per bene."


def test_a_clean_document_costs_nothing():
    m = _mgr()
    m._fallback_order = lambda f: ["chandra"]
    seen = []
    _wire(m, {"chandra": lambda i: _clean(0)}, seen)
    pages = [_clean(0), _clean(1)]
    note = _run(m._retry_degraded(_pdf(2), "a.pdf", "application/pdf", "surya",
                                  pages, dpi=200, page_opts={}, detect=None,
                                  lang="", cfg=m._cfg, window=8))
    assert seen == [] and note == ""


def test_the_replaced_page_keeps_its_own_index():
    """A page re-read on its own always comes back as page 0. Callers index
    into `pages` by page number, so the original index has to survive."""
    m = _mgr()
    m._fallback_order = lambda f: ["chandra"]
    pages = [_clean(0), _clean(1), _looping(2)]
    _wire(m, {"chandra": lambda i: _clean(0)}, [])
    _run(m._retry_degraded(_pdf(3), "a.pdf", "application/pdf", "surya", pages,
                           dpi=200, page_opts={}, detect=None, lang="",
                           cfg=m._cfg, window=8))
    assert [p.index for p in pages] == [0, 1, 2]


def test_the_replaced_page_says_who_read_it_and_why():
    m = _mgr()
    m._fallback_order = lambda f: ["chandra"]
    pages = [_looping(0)]
    _wire(m, {"chandra": lambda i: _clean(0)}, [])
    _run(m._retry_degraded(_pdf(1), "a.pdf", "application/pdf", "surya", pages,
                           dpi=200, page_opts={}, detect=None, lang="",
                           cfg=m._cfg, window=8))
    got = pages[0].meta["degraded_retry"]
    assert got["from"] == "surya" and got["to"] == "chandra"
    assert got["reason"] == "looping"
    assert "200" in got["why"]


def test_a_replacement_that_is_also_degraded_is_refused():
    """Keeping a second truncated page would hide the problem behind a
    different engine's name."""
    m = _mgr()
    m._fallback_order = lambda f: ["chandra"]
    pages = [_looping(0, "originale")]
    _wire(m, {"chandra": lambda i: _truncated(0, "altrettanto rotta")}, [])
    note = _run(m._retry_degraded(_pdf(1), "a.pdf", "application/pdf", "surya",
                                  pages, dpi=200, page_opts={}, detect=None,
                                  lang="", cfg=m._cfg, window=8))
    assert note == ""
    assert pages[0].text == "originale"
    assert "degraded_retry" not in pages[0].meta
    assert pages[0].meta["repeat_detected"] is True


def test_the_chain_is_walked_until_one_engine_reads_the_page():
    m = _mgr()
    m._fallback_order = lambda f: ["olmocr", "chandra"]
    pages = [_looping(0)]
    seen = []
    _wire(m, {"olmocr": lambda i: _truncated(0),
              "chandra": lambda i: _clean(0)}, seen)
    _run(m._retry_degraded(_pdf(1), "a.pdf", "application/pdf", "surya", pages,
                           dpi=200, page_opts={}, detect=None, lang="",
                           cfg=m._cfg, window=8))
    assert [e for e, _ in seen] == ["olmocr", "chandra"]
    assert pages[0].meta["degraded_retry"]["to"] == "chandra"


def test_a_later_engine_is_only_asked_for_the_pages_still_broken():
    m = _mgr()
    m._fallback_order = lambda f: ["olmocr", "chandra"]
    pages = [_looping(0), _looping(1)]
    seen = []
    _wire(m, {"olmocr": lambda i: _clean(0) if i == 0 else _truncated(0),
              "chandra": lambda i: _clean(0)}, seen)
    _run(m._retry_degraded(_pdf(2), "a.pdf", "application/pdf", "surya", pages,
                           dpi=200, page_opts={}, detect=None, lang="",
                           cfg=m._cfg, window=8))
    assert seen == [("olmocr", 2), ("chandra", 1)], seen


def test_an_engine_that_raises_outright_does_not_lose_the_document():
    m = _mgr()
    m._fallback_order = lambda f: ["olmocr", "chandra"]
    pages = [_looping(0, "quel che c'e")]

    async def _read(blobs, engine, **kw):
        if engine == "olmocr":
            raise RuntimeError("no pod")
        return [_clean(0)]
    m._read_page_blobs = _read
    note = _run(m._retry_degraded(_pdf(1), "a.pdf", "application/pdf", "surya",
                                  pages, dpi=200, page_opts={}, detect=None,
                                  lang="", cfg=m._cfg, window=8))
    assert "chandra" in note


def test_the_cap_bounds_what_one_document_may_spend():
    cfg = _Cfg()
    cfg.retry_degraded_max_pages = 2
    m = _mgr(cfg)
    m._fallback_order = lambda f: ["chandra"]
    pages = [_looping(i) for i in range(5)]
    seen = []
    _wire(m, {"chandra": lambda i: _clean(0)}, seen)
    _run(m._retry_degraded(_pdf(5), "a.pdf", "application/pdf", "surya", pages,
                           dpi=200, page_opts={}, detect=None, lang="",
                           cfg=cfg, window=8))
    assert seen == [("chandra", 2)], seen
    assert om.OcrManager._page_degraded(pages[4]) == "looping"


def test_a_single_image_is_re_read_whole():
    """No PDF to split: the one page IS the document."""
    m = _mgr()
    m._fallback_order = lambda f: ["chandra"]
    pages = [_looping(0)]
    seen = []
    _wire(m, {"chandra": lambda i: _clean(0)}, seen)
    note = _run(m._retry_degraded(b"\x89PNG\r\n\x1a\n...", "p.png", "image/png",
                                  "surya", pages, dpi=200, page_opts={},
                                  detect=None, lang="", cfg=m._cfg, window=8))
    assert seen == [("chandra", 1)] and "chandra" in note


# ------------------------------------------------------------ on the pods #
def test_a_pod_never_re_reads_its_own_degraded_page():
    """Only the orchestrator can reach another engine's pods. A pod would
    re-read on the engine that just produced the flag."""
    from codai.api import runpod_worker as rw
    src = (ROOT / "codai/api/runpod_worker.py").read_text()
    assert 'env["CODERAI_OCR_RETRY_DEGRADED"] = "off"' in src


def test_the_knob_exists_and_defaults_to_auto():
    from codai.config import OcrConfig
    assert OcrConfig().retry_degraded == "auto"
    assert OcrConfig().retry_degraded_max_pages == 0


# ------------------------------------------------- the inferred early stop #
class _ShortCfg(_Cfg):
    retry_short_pages_below = 0.25
    retry_short_pages_min_doc = 4


def _long(i, n=30000):
    return OcrPage(index=i, text="x" * n)


def test_short_page_detection_is_off_by_default():
    """It is INFERRED, not attested. A corpus with real title pages would pay
    for a second read of every one of them."""
    from codai.config import OcrConfig
    assert OcrConfig().retry_short_pages_below == 0.0
    pages = [_long(0), _long(1), _long(2), OcrPage(index=3, text="x" * 100)]
    assert om.OcrManager._short_pages(pages, _Cfg()) == {}


def test_a_page_an_eighth_the_length_of_its_neighbours_is_short():
    pages = [_long(0), _long(1), _long(2), OcrPage(index=3, text="x" * 4046)]
    got = om.OcrManager._short_pages(pages, _ShortCfg())
    assert list(got) == [3]
    assert "4046 chars" in got[3] and "median of 30000" in got[3]


def test_the_measure_is_the_documents_own_median_not_a_character_count():
    """A document of uniformly short pages has no short page. A threshold in
    characters would call every page of it an early stop."""
    pages = [OcrPage(index=i, text="x" * 900) for i in range(6)]
    assert om.OcrManager._short_pages(pages, _ShortCfg()) == {}


def test_a_two_page_document_has_no_median_worth_trusting():
    pages = [_long(0), OcrPage(index=1, text="x" * 10)]
    assert om.OcrManager._short_pages(pages, _ShortCfg()) == {}


def test_a_page_already_explained_as_blank_is_not_an_early_stop():
    pages = [_long(0), _long(1), _long(2),
             OcrPage(index=3, text="", meta={"blank": True, "ink": 0.0001})]
    assert om.OcrManager._short_pages(pages, _ShortCfg()) == {}


def test_a_short_page_is_only_replaced_by_a_LONGER_one():
    """The complaint was length. A second engine stopping just as early has
    told us the page really is short."""
    m = _mgr(_ShortCfg())
    m._fallback_order = lambda f: ["chandra"]
    pages = [_long(0), _long(1), _long(2), OcrPage(index=3, text="x" * 4046)]
    _wire(m, {"chandra": lambda i: OcrPage(index=0, text="y" * 900)}, [])
    note = _run(m._retry_degraded(_pdf(4), "a.pdf", "application/pdf", "olmocr",
                                  pages, dpi=200, page_opts={}, detect=None,
                                  lang="", cfg=m._cfg, window=8))
    assert note == ""
    assert pages[3].text == "x" * 4046


def test_a_short_page_IS_replaced_when_the_other_engine_reads_more():
    m = _mgr(_ShortCfg())
    m._fallback_order = lambda f: ["chandra"]
    pages = [_long(0), _long(1), _long(2), OcrPage(index=3, text="x" * 4046)]
    _wire(m, {"chandra": lambda i: OcrPage(index=0, text="y" * 30983)}, [])
    note = _run(m._retry_degraded(_pdf(4), "a.pdf", "application/pdf", "olmocr",
                                  pages, dpi=200, page_opts={}, detect=None,
                                  lang="", cfg=m._cfg, window=8))
    assert "chandra" in note
    assert pages[3].text == "y" * 30983
    assert pages[3].meta["degraded_retry"]["reason"].startswith("short")


def test_a_looping_page_is_replaced_on_its_own_terms_not_on_length():
    """Length is only the test for a page flagged SHORT. A looping page emits
    the same fragment to the cap, so it is usually the LONGER of the two."""
    m = _mgr(_ShortCfg())
    m._fallback_order = lambda f: ["chandra"]
    pages = [_long(0), _long(1), _long(2), _looping(3, "riga " * 4000)]
    _wire(m, {"chandra": lambda i: OcrPage(index=0, text="buono")}, [])
    note = _run(m._retry_degraded(_pdf(4), "a.pdf", "application/pdf", "surya",
                                  pages, dpi=200, page_opts={}, detect=None,
                                  lang="", cfg=m._cfg, window=8))
    assert "chandra" in note and pages[3].text == "buono"
