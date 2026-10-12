"""Accented letters came back from surya as U+FFFD, and it looked like an
encoding bug. It was not one.

U+FFFD is a real entry in surya-ocr-2's vocabulary: id 65421 of 65425, the only
id that decodes to it, with ``<unk>`` a separate token at 65424. The tokenizer is
``WordLevel`` and character-level for Latin text, so ``a`` with an accent is ONE
token and an accent cannot be split across two -- no partial-UTF-8 artefact is
possible, and per-token decoding is byte-identical to decoding the whole string.
Surya reads text only from ``message.content``, and coderai's gateway forwards
the upstream body as raw bytes. So the model PREDICTED "I cannot read this
glyph", and the pipeline delivered that faithfully.

Two things follow, and these are the tests for both:

1. It is a resolution problem. ``scale_to_fit`` caps surya's input at
   3072x2048 = 6.29 Mpixels and is called with DEFAULTS ONLY, so no surya
   setting raises it. An A4 page reaches that at ~255 DPI, so the shared
   ``ocr.dpi = 200`` was handing the model 3.89 of 6.29 Mpixels -- 62% -- and a
   diacritic is ~3 pixels tall at 200 DPI.
2. It must stop being silent. ``piu`` with the accent eaten indexes and searches
   wrong for ever; for judicial documents a known unknown beats a quiet
   corruption.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codai.ocr.base import (UNREADABLE, OcrLine, OcrPage, count_unreadable,   # noqa: E402
                            unreadable_offsets, unreadable_report)
from codai.ocr import manager as om                                           # noqa: E402


# --------------------------------------------------------------- the marker #
def test_the_marker_is_the_replacement_character():
    assert UNREADABLE == "�"
    assert UNREADABLE.encode("utf-8") == b"\xef\xbf\xbd"


def test_counting_and_locating():
    text = "pi� facolt�"
    assert count_unreadable(text) == 2
    assert unreadable_offsets(text) == [2, 10]


def test_clean_text_is_clean():
    assert count_unreadable("perché così più") == 0
    assert unreadable_offsets("perché") == []


def test_empty_and_none_are_safe():
    assert count_unreadable("") == 0
    assert count_unreadable(None) == 0
    assert unreadable_offsets(None) == []


# ------------------------------------------------ reporting, without noise #
def test_a_clean_line_is_byte_for_byte_what_it_always_was():
    """The key is omitted when there is nothing to say, so every existing
    consumer of a clean result sees exactly the response it saw before."""
    d = OcrLine(text="perché", bbox=[0, 0, 10, 10], conf=0.9).to_dict()
    assert d == {"text": "perché", "bbox": [0, 0, 10, 10], "conf": 0.9}
    assert "unreadable" not in d


def test_a_dirty_line_says_so():
    d = OcrLine(text="pi�", bbox=[0, 0, 1, 1], conf=0.4).to_dict()
    assert d["unreadable"] == 1


def test_a_clean_page_does_not_grow_a_key():
    d = OcrPage(index=0, text="tutto bene").to_dict()
    assert "unreadable" not in d


def test_a_dirty_page_says_so():
    d = OcrPage(index=0, text="pi� e facolt�").to_dict()
    assert d["unreadable"] == 2


def test_a_clean_document_reports_nothing_at_all():
    assert unreadable_report([OcrPage(index=0, text="va bene")]) == {}
    assert unreadable_report([]) == {}
    assert unreadable_report(None) == {}


def test_the_document_report_counts_locates_and_scores():
    page = OcrPage(
        index=3,
        text="la pi� alta facolt�",
        lines=[OcrLine(text="la pi� alta", bbox=[1, 2, 3, 4], conf=0.41),
               OcrLine(text="facolt�", bbox=[5, 6, 7, 8], conf=0.88)],
    )
    r = unreadable_report([page])
    assert r["count"] == 2
    assert r["pages"] == [{"page": 3, "count": 2}]
    assert [s["line"] for s in r["spots"]] == [0, 1]
    # The confidence carried is the LINE's -- the finest surya reports.
    assert r["spots"][0]["conf"] == 0.41
    assert r["spots"][1]["conf"] == 0.88
    assert r["spots"][0]["bbox"] == [1, 2, 3, 4]
    # Context makes a report readable without fetching the page back.
    assert "pi�" in r["spots"][0]["context"]


def test_the_report_counts_pages_separately():
    r = unreadable_report([OcrPage(index=0, text="pi�"),
                           OcrPage(index=1, text="ok"),
                           OcrPage(index=2, text="a�b�")])
    assert r["count"] == 3
    assert r["pages"] == [{"page": 0, "count": 1}, {"page": 2, "count": 2}]


def test_a_page_of_pure_noise_cannot_return_a_megabyte():
    """The count already says how bad it is; the located spots are capped."""
    from codai.ocr.base import UNREADABLE_MAX_SPOTS
    line = OcrLine(text=UNREADABLE * 500, bbox=[0, 0, 1, 1], conf=0.1)
    page = OcrPage(index=0, text=UNREADABLE * 500, lines=[line])
    r = unreadable_report([page])
    assert r["count"] == 500
    assert len(r["spots"]) == UNREADABLE_MAX_SPOTS
    assert r["truncated"] == 500 - UNREADABLE_MAX_SPOTS


def test_the_text_itself_is_left_alone():
    """Rewriting or stripping the marker would destroy the one thing that says
    WHERE the loss is. The report is additive."""
    page = OcrPage(index=0, text="pi�")
    assert page.to_dict()["text"] == "pi�"


# --------------------------------------------------------- per-engine DPI #
class _Cfg:
    def __init__(self, **kw):
        self.dpi = 200
        self.surya_dpi = 250
        self.default_engine = "surya"
        for k, v in kw.items():
            setattr(self, k, v)


def _mgr(cfg, resolved="surya"):
    m = om.OcrManager.__new__(om.OcrManager)
    m._require_cfg = lambda: cfg
    m.resolve_engine = lambda requested: resolved
    return m


def test_surya_gets_its_own_dpi():
    assert _mgr(_Cfg()).engine_dpi("surya") == 250


def test_paddle_is_left_on_the_shared_default():
    """paddle and docTR have their own optima and are not surya's business."""
    assert _mgr(_Cfg(), resolved="paddle").engine_dpi("paddle") == 200


def test_zero_means_use_the_shared_default():
    assert _mgr(_Cfg(surya_dpi=0)).engine_dpi("surya") == 200


def test_a_missing_per_engine_field_is_not_an_error():
    cfg = _Cfg()
    del cfg.surya_dpi
    assert _mgr(cfg).engine_dpi("surya") == 200


def test_an_unresolvable_engine_falls_back_rather_than_raising_here():
    """The engine path owns that error and words it properly; rasterising must
    not be where an unknown engine surfaces."""
    from codai.ocr.base import OcrError
    m = om.OcrManager.__new__(om.OcrManager)
    cfg = _Cfg()
    m._require_cfg = lambda: cfg

    def _boom(requested):
        raise OcrError("unknown OCR engine 'nope'", status=400)
    m.resolve_engine = _boom
    assert m.engine_dpi("nope") == 200


def test_an_explicit_request_dpi_still_wins():
    import inspect
    src = inspect.getsource(om.OcrManager.ocr_document)
    assert "int(dpi) if dpi else self.engine_dpi(engine)" in src


def test_the_default_saturates_an_a4_page_without_overshooting_much():
    """The ceiling is surya's, hard-coded and called with defaults:
    3072x2048 = 6291456 pixels. This is the arithmetic that chose 250, kept here
    so a later edit to surya_dpi has to face it.
    """
    cap = 3072 * 2048
    a4_w_in, a4_h_in = 8.27, 11.69
    px = lambda dpi: int(a4_w_in * dpi) * int(a4_h_in * dpi)   # noqa: E731
    assert px(200) / cap < 0.63            # the old default wasted 37%
    assert 0.9 < px(250) / cap <= 1.0      # the new one saturates it
    assert px(300) > cap                   # beyond here surya downscales again
