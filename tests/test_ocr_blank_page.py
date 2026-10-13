"""Telling an empty page from a page the engine could not read.

A four-page court PDF came back with 47 characters and was reported as an OCR
failure, with a sample sent over for diagnosis. It was not an OCR failure. Every
page was a ~900-byte CCITT Group 4 image -- which is what an EMPTY page costs,
a couple of bytes per all-white row -- carrying 0.13% ink, all of it a diagonal
scan edge and a diagonal "Corte di Cassazione - copia non ufficiale" watermark.
The 47 characters were the watermark. The engine read the page perfectly.

Three outcomes are otherwise indistinguishable in a short result:

    the model could not read it  -> needs a better model
    the reply was cut off        -> needs a bigger budget (model_attempts)
    the page is blank            -> needs the document fixed upstream

Confusing them sends an operator looking in entirely the wrong place, which is
what happened here, for most of a day. So every short page now carries the one
measurement that separates them, for every engine, taken from an image already
in memory.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw                                    # noqa: E402

from codai.config import OcrConfig                                  # noqa: E402
from codai.ocr.base import OcrPage, ink_coverage                    # noqa: E402
from codai.ocr.manager import _mark_blank_pages                     # noqa: E402


class _Cfg:
    def __init__(self, **kw):
        self.ocr = OcrConfig()
        for k, v in kw.items():
            setattr(self.ocr, k, v)


def _blank(w=827, h=1169):
    return Image.new("RGB", (w, h), "white")


def _watermarked():
    """What an empty scanned court page actually looks like: a page edge, a few
    specks, and a diagonal watermark."""
    img = _blank()
    d = ImageDraw.Draw(img)
    d.line((0, 4, 500, 12), fill="black", width=3)        # the scan edge
    for x, y in ((700, 30), (712, 48), (60, 1150)):
        d.ellipse((x, y, x + 3, y + 3), fill="black")     # specks
    d.text((700, 400), "Corte di Cassazione - copia non ufficiale", fill="#cccccc")
    return img


def _inked(fraction=0.06):
    img = _blank()
    d = ImageDraw.Draw(img)
    rows = int(img.height * fraction)
    d.rectangle((0, 0, img.width, rows), fill="black")
    return img


def _page(text=""):
    p = OcrPage(index=0, text=text, width=827, height=1169)
    return p


# ------------------------------------------------------------ the measurement #
def test_a_blank_page_measures_almost_no_ink():
    assert ink_coverage(_blank()) < 0.0001


def test_the_real_shape_of_an_empty_scan_is_still_under_the_floor():
    """Edge, specks and watermark together. The measured original was 0.13%;
    the floor is 0.5%, so the gap is real rather than a hair's breadth."""
    ink = ink_coverage(_watermarked())
    assert 0 < ink < OcrConfig().blank_page_ink


def test_a_page_with_text_on_it_is_well_above_the_floor():
    assert ink_coverage(_inked(0.06)) > OcrConfig().blank_page_ink * 5


def test_an_unmeasurable_image_says_so_rather_than_guessing():
    """-1, not 0: "I could not look" must not read as "there is nothing there"."""
    assert ink_coverage(object()) == -1.0


# ------------------------------------------------------------- the verdict #
def test_an_empty_page_with_a_short_result_is_called_blank():
    cfg = _Cfg()
    pages = [_page("Corte di Cassazione - copia non ufficiale")]
    blank = _mark_blank_pages(cfg, [_watermarked()], pages)
    assert blank == [0]
    info = pages[0].meta["page_ink"]
    assert info["blank"] is True
    assert "NOT the engine failing to read it" in info["why"]


def test_an_inked_page_with_a_short_result_is_NOT_called_blank():
    """The opposite verdict matters just as much: there IS something there, so
    a short result is the engine's failure and worth chasing."""
    cfg = _Cfg()
    pages = [_page("71 characters of nothing much at all, truncated mid-word")]
    blank = _mark_blank_pages(cfg, [_inked(0.06)], pages)
    assert blank == []
    info = pages[0].meta["page_ink"]
    assert info["blank"] is False
    assert "engine failing to read it" in info["why"]


def test_a_page_that_read_normally_is_not_measured_at_all():
    """No explanation is needed where nothing is unexplained, and the page stays
    byte-identical to what it always returned."""
    cfg = _Cfg()
    pages = [_page("x" * 5000)]
    assert _mark_blank_pages(cfg, [_blank()], pages) == []
    assert "page_ink" not in pages[0].meta


def test_the_floor_and_the_text_length_are_configurable():
    cfg = _Cfg(blank_page_ink=0.9)
    pages = [_page("short")]
    assert _mark_blank_pages(cfg, [_inked(0.06)], pages) == [0]   # 6% < 90%
    cfg2 = _Cfg(blank_page_max_chars=0)
    pages2 = [_page("short")]
    assert _mark_blank_pages(cfg2, [_blank()], pages2) == []
    assert "page_ink" not in pages2[0].meta


def test_a_missing_image_is_skipped_not_guessed():
    cfg = _Cfg()
    pages = [_page("short"), _page("short")]
    assert _mark_blank_pages(cfg, [_blank()], pages) == [0]


def test_it_never_raises_into_the_ocr_path():
    """A diagnostic that can break OCR is worse than no diagnostic."""
    cfg = _Cfg()
    pages = [_page("short")]
    assert _mark_blank_pages(cfg, [object()], pages) == []
