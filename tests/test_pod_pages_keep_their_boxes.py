"""A page read on a pod came back with no geometry at all.

`/v1/ocr` on a pod answers with exactly what a local read answers: `OcrPage.to_dict`
always emits `lines`, `regions` and `tables`. The orchestrator then rebuilt the
page key by key --

    page = OcrPage(index=index, text=..., width=..., height=...)
    meta = dict(d.get("meta") or {})
    for k in ("repeat_detected", ..., "confidence", "confidence_measured", ...):
        if k in d: meta[k] = d[k]

-- and never copied the three arrays. So every pod-read page arrived with
`lines == []` for every engine, and no request parameter could change it,
because nothing was switched off: the boxes were put on the wire and dropped on
arrival.

Reported from a client probing for them, and the shape of the report is the
lesson. `conf` came through (it is one of the names in that list) while the
boxes did not, so the evidence read as "the engines don't emit boxes on these
pods" and sent the search towards the OCR engines and a `detect` parameter that
has nothing to do with text lines. One boundary's key-by-key copy looked, from
outside, exactly like a missing engine feature.

Both boundaries now share one reassembly, and diagnostics ride by exclusion.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codai.ocr.base import OcrLine, OcrPage, OcrRegion, page_from_dict


def _pod_reply(index=0):
    """What a pod really sends: OcrPage.to_dict of a page with geometry."""
    page = OcrPage(index=index, text="SENTENZA\nn. 1234/2026", width=1700, height=2200)
    page.lines = [OcrLine(text="SENTENZA", bbox=[120.0, 90.0, 980.0, 160.0], conf=0.91),
                  OcrLine(text="n. 1234/2026", bbox=[120.0, 180.0, 760.0, 240.0], conf=0.88)]
    page.regions = [OcrRegion(label="title", bbox=[100.0, 80.0, 1000.0, 250.0],
                              conf=0.95, order=3)]
    page.tables = [{"bbox": [100.0, 400.0, 1600.0, 900.0], "html": "<table></table>"}]
    page.meta = {"confidence": 0.0, "confidence_measured": False,
                 "model_attempts": 2, "repeat_detected": True}
    return page.to_dict()


# ── the defect ───────────────────────────────────────────────────────────────

def test_the_lines_survive_the_pod_boundary():
    got = page_from_dict(_pod_reply(), index=7)
    assert len(got.lines) == 2, 'the boxes are still being dropped on arrival'
    assert got.lines[0].text == "SENTENZA"
    assert got.lines[0].bbox == [120.0, 90.0, 980.0, 160.0]
    assert got.lines[0].conf == 0.91


def test_the_regions_and_tables_survive_too():
    got = page_from_dict(_pod_reply(), index=0)
    assert len(got.regions) == 1 and got.regions[0].label == "title"
    assert len(got.tables) == 1


def test_the_reading_order_survives():
    """List position is the same as reading order only until a page has two
    columns, which is when it matters."""
    got = page_from_dict(_pod_reply(), index=0)
    assert got.regions[0].order == 3


def test_the_page_pixel_size_survives():
    """A box is meaningless without the frame it was measured in: a client
    scaling it onto a rendered image needs the page's own width and height."""
    got = page_from_dict(_pod_reply(), index=0)
    assert (got.width, got.height) == (1700, 2200)


def test_the_callers_index_wins_over_the_dicts():
    """A pod answering a single page numbers it 0; the per-page dispatch knows
    the real document position."""
    assert page_from_dict(_pod_reply(index=0), index=42).index == 42


def test_the_dicts_index_is_used_when_the_caller_gives_none():
    assert page_from_dict(_pod_reply(index=5)).index == 5


# ── what already worked must keep working ────────────────────────────────────

def test_the_diagnostics_still_arrive():
    """They were the one thing the old copy did carry; this must not trade the
    boxes for them."""
    got = page_from_dict(_pod_reply(), index=0)
    for k, v in (("confidence", 0.0), ("confidence_measured", False),
                 ("model_attempts", 2), ("repeat_detected", True)):
        assert got.meta.get(k) == v, k


def test_a_zero_confidence_is_not_mistaken_for_absent():
    """`confidence: 0.0` with `confidence_measured: false` means nothing was
    measured, and both halves have to cross for a caller to tell that from a
    model that was certain the page was bad."""
    got = page_from_dict(_pod_reply(), index=0)
    assert "confidence" in got.meta and got.meta["confidence"] == 0.0
    assert got.meta["confidence_measured"] is False


def test_an_unnamed_diagnostic_arrives_without_anyone_adding_it():
    """Forwarded by exclusion, which is the point: the old copy would have had
    to name this, and naming is what lost accents_repaired and model_attempts
    before it."""
    d = _pod_reply()
    d["some_future_diagnostic"] = {"n": 3}
    assert page_from_dict(d, index=0).meta.get("some_future_diagnostic") == {"n": 3}


def test_a_page_field_never_leaks_into_meta():
    got = page_from_dict(_pod_reply(), index=0)
    for k in ("index", "text", "width", "height", "lines", "regions", "tables", "meta"):
        assert k not in got.meta, k


# ── a reply that genuinely has no geometry is still fine ─────────────────────

def test_an_engine_without_boxes_is_not_broken_by_this():
    """olmOCR returns no boxes by design; that must stay an empty list, not an
    error."""
    got = page_from_dict({"index": 0, "text": "plain", "width": 10, "height": 20,
                          "lines": [], "regions": [], "tables": []})
    assert got.lines == [] and got.regions == [] and got.tables == []
    assert got.text == "plain"


def test_a_missing_key_does_not_raise():
    got = page_from_dict({"text": "only text"})
    assert got.lines == [] and got.width == 0 and got.text == "only text"


def test_a_malformed_line_is_skipped_not_fatal():
    """A page is worth more than a strict parse of one bad entry."""
    got = page_from_dict({"text": "x", "lines": ["not a dict",
                                                 {"text": "ok", "bbox": [1, 2, 3, 4]}]})
    assert len(got.lines) == 1 and got.lines[0].text == "ok"


def test_a_short_bbox_is_padded_rather_than_crashing():
    got = page_from_dict({"text": "x", "lines": [{"text": "a", "bbox": [1, 2]}]})
    assert len(got.lines[0].bbox) == 4


# ── one implementation, not two ──────────────────────────────────────────────

def test_both_boundaries_use_the_shared_reassembly():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mgr = open(os.path.join(root, 'codai/ocr/manager.py')).read()
    sub = open(os.path.join(root, 'codai/ocr/subprocess_engine.py')).read()
    i = mgr.index('def _page_from_pod')
    assert 'page_from_dict(' in mgr[i:i + 1600], (
        'the pod boundary reassembles pages on its own again')
    j = sub.index('def _page_from_dict')
    assert 'page_from_dict(' in sub[j:j + 900]


def test_the_pod_boundary_no_longer_copies_diagnostics_by_name():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mgr = open(os.path.join(root, 'codai/ocr/manager.py')).read()
    i = mgr.index('def _page_from_pod')
    body = mgr[i:i + 1600]
    assert '"repeat_detected", "repeat_chars"' not in body, (
        'the key-by-key copy is back, and it is what dropped the boxes')
