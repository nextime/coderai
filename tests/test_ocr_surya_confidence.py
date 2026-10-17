"""A page-level confidence for surya, in the shape the other engines use.

chandra and olmOCR each set `meta.confidence` and `meta.confidence_measured`
from the mean of `exp(logprob)` over tokens. surya set NEITHER: its predictor
reports confidence per LINE, so a caller switching engines got a different kind
of answer, and nothing could tell "no line reported a confidence" from "the
model was certain it was unsure".

That distinction is not academic. On the night the accent bug was found, surya
reported ~96 % per line while every accent in the text was a replacement
character -- the model was confident and RIGHT to be, because it emitted the
correct tokens and the loss happened after it. A 0.0 that means "nothing
measured this" being read as a verdict has already produced one confidently
wrong conclusion in this subsystem.
"""

import pytest

from codai.ocr.workers.ocr_worker import _page_confidence


# ------------------------------------------------------------- the aggregate #

def test_nothing_reported_is_not_a_confidence_of_zero():
    """The case the flag exists for."""
    assert _page_confidence([(10, None), (20, None)]) == (0.0, False)
    assert _page_confidence([]) == (0.0, False)
    assert _page_confidence(None) == (0.0, False)


def test_a_real_zero_is_reported_as_measured():
    """Indistinguishable from the above without the second value."""
    assert _page_confidence([(10, 0.0)]) == (0.0, True)


def test_it_is_weighted_by_characters_not_by_lines():
    """chandra's and olmOCR's number is a mean over TOKENS, i.e. over text. A
    plain mean over lines would let a two-word line count as much as a full
    one."""
    weighted = _page_confidence([(100, 0.95), (2, 0.1)])[0]
    plain_mean = (0.95 + 0.1) / 2
    assert weighted > plain_mean
    assert weighted == pytest.approx((100 * 0.95 + 2 * 0.1) / 102, abs=1e-4)


def test_lines_without_a_confidence_are_skipped_not_counted_as_zero():
    assert _page_confidence([(10, 0.8), (10, None)]) == (0.8, True)


def test_empty_lines_fall_back_to_a_plain_mean():
    """Zero characters everywhere must not divide by zero, and the
    confidences themselves are still real."""
    assert _page_confidence([(0, 0.5), (0, 0.9)]) == (0.7, True)


def test_a_uniform_page_reports_that_value():
    assert _page_confidence([(10, 0.9), (30, 0.9)]) == (0.9, True)


def test_strings_and_rubbish_are_not_confidences():
    assert _page_confidence([(10, "0.9"), (5, None), (5, {})]) == (0.0, False)


# ---------------------------------------------- it reaches the page and meta #

def test_the_worker_sets_both_keys_on_the_page():
    import inspect

    from codai.ocr.workers.ocr_worker import SuryaWorker
    src = inspect.getsource(SuryaWorker.ocr)
    assert 'page["confidence"], page["confidence_measured"] = _page_confidence(' in src


def test_the_raw_value_is_collected_not_the_coerced_one():
    """`float(conf or 0.0)` is what the per-line field needs and what would
    destroy the distinction if the aggregate read it."""
    import inspect

    from codai.ocr.workers.ocr_worker import SuryaWorker
    src = inspect.getsource(SuryaWorker.ocr)
    assert "_confs.append((len(str(text)), conf))" in src
    assert "_confs.append((len(text), conf))" in src


def test_both_keys_survive_the_reassembly_into_meta():
    from codai.ocr.subprocess_engine import SubprocessOcrEngine
    page = SubprocessOcrEngine._page_from_dict({
        "index": 0, "text": "ciao", "width": 10, "height": 10,
        "lines": [{"text": "ciao", "bbox": [0, 0, 1, 1], "conf": 0.9}],
        "confidence": 0.9, "confidence_measured": True,
    })
    assert page.meta["confidence"] == 0.9
    assert page.meta["confidence_measured"] is True


def test_a_false_measured_flag_is_not_swallowed_as_empty():
    """The reassembly skips keys whose value is None/""/[]/{} -- False must not
    be mistaken for one of those, or the honest case disappears."""
    from codai.ocr.subprocess_engine import SubprocessOcrEngine
    page = SubprocessOcrEngine._page_from_dict({
        "index": 0, "text": "x", "width": 1, "height": 1, "lines": [],
        "confidence": 0.0, "confidence_measured": False,
    })
    assert page.meta.get("confidence_measured") is False
    assert page.meta.get("confidence") == 0.0


def test_the_keys_cross_the_pod_boundary():
    """A page read on a rented pod must report the same shape as a local one."""
    import inspect

    from codai.ocr import manager
    src = inspect.getsource(manager._page_from_pod)
    assert "confidence" in src and "confidence_measured" in src


def test_all_three_engines_now_report_the_same_shape():
    import inspect

    from codai.ocr import chandra, olmocr
    for mod in (chandra, olmocr):
        src = inspect.getsource(mod)
        assert "confidence_measured" in src
    from codai.ocr.workers import ocr_worker
    assert "confidence_measured" in inspect.getsource(ocr_worker.SuryaWorker.ocr)
