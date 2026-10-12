"""The byte-level trace: whose fault is an unreadable character?

A marker in the finished text says nothing about the cause, and I got this wrong once
by arguing that it did. surya-ocr-2's vocabulary is BYTE-level: `a`-grave is two tokens
(UTF-8 C3 A0), and decoding EITHER byte alone yields exactly one replacement character.
So one marker per accented letter is equally the signature of

  * the model emitting a broken byte sequence  -> a faithful rendering of a wrong
    answer, nothing we can fix;
  * the model emitting a correct sequence that our serving stack mis-assembled ->
    the model was right and the bug is ours.

Only the tokens tell them apart. These tests pin the discriminator, and the two
properties that make it safe to ship: it is silent on a clean page, and it can never
break OCR to produce a diagnostic.
"""

import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest                                                      # noqa: E402


def _worker_module():
    """Load the worker the way it actually runs: as a standalone file.

    It is deliberately self-contained (stdlib + PIL + one OCR library, never
    coderai) because it executes inside an isolated venv, so it is not importable
    as ``codai.ocr.workers.ocr_worker`` in that venv at all. Loading it by path is
    what keeps this test honest about the thing being shipped.
    """
    path = ROOT / "codai" / "ocr" / "workers" / "ocr_worker.py"
    spec = importlib.util.spec_from_file_location("ocr_worker_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


W = _worker_module()


def _tok(token, byts, logprob=-0.01):
    return {"token": token, "bytes": byts, "logprob": logprob}


# ------------------------------------------------------------- the two verdicts #
def test_a_lone_lead_byte_is_reported_as_broken():
    """The model-is-wrong case: C3 with no continuation. `a`-grave never arrives."""
    calls = [[_tok("p", [112]), _tok("i", [105]), _tok("Ã", [195]), _tok(" ", [32])]]
    found = W._trace_findings(calls)
    assert len(found) == 1
    assert found[0]["token"]["bytes"] == [195]
    assert found[0]["kind"] == "invalid-utf8"


def test_a_complete_byte_pair_is_not_reported():
    """The stack-is-wrong case must NOT be confused with it: C3 A0 is a correct
    `a`-grave, and a trace that flagged it would cry wolf on every accent."""
    calls = [[_tok("Ã", [195]), _tok("ł", [160]), _tok("a", [97])]]
    assert W._trace_findings(calls) == []


def test_a_three_byte_character_split_across_tokens_is_not_reported():
    """An em dash is E2 80 94 over three tokens. It survived in production while
    every two-byte accent died, so it must read as healthy here."""
    calls = [[_tok("â", [226]), _tok("Ģ", [128]), _tok("Ķ", [148])]]
    assert W._trace_findings(calls) == []


def test_a_token_that_is_itself_the_marker_is_reported():
    """The third possibility: the SERVER already decoded to U+FFFD before we saw
    it, so the bytes arrive as the replacement character's own encoding."""
    calls = [[_tok("�", [239, 191, 189])]]
    found = W._trace_findings(calls)
    assert len(found) == 1
    assert "replacement character" in found[0]["why"]


def test_a_trailing_lead_byte_at_the_very_end_is_reported():
    """Nothing follows, so nothing can complete it — the commonest shape of a
    truncated generation."""
    calls = [[_tok("a", [97]), _tok("Ã", [195])]]
    assert len(W._trace_findings(calls)) == 1


def test_a_lead_byte_followed_by_ascii_is_reported():
    """C3 then 'a': the model started an accent and finished with something that
    cannot continue one."""
    found = W._trace_findings([[_tok("Ã", [195]), _tok("a", [97])]])
    assert len(found) == 1
    assert found[0]["kind"] == "invalid-utf8"


def test_a_four_byte_character_is_healthy():
    """F0 9F 93 84 — outside Italian, but the scan must not invent a fault."""
    assert W._trace_findings([[_tok("a", [240]), _tok("b", [159]),
                               _tok("c", [147]), _tok("d", [132])]]) == []


def test_a_sound_stream_with_no_markers_yields_nothing():
    """The decisive negative: if the bytes are good and the text still lost
    characters, the loss happened AFTER this point — and an empty trace on an
    unreadable page is what convicts the serving stack."""
    assert W._trace_findings([[_tok("p", [112]), _tok("Ã", [195]),
                               _tok("ł", [160])]]) == []


def test_one_finding_per_token_not_per_byte():
    """A three-byte token that is entirely wrong must not report three times."""
    found = W._trace_findings([[_tok("?", [195, 195, 195])]])
    assert len(found) == 1


def test_a_continuation_byte_with_no_lead_is_reported():
    calls = [[_tok("ł", [160]), _tok("a", [97])]]
    found = W._trace_findings(calls)
    assert len(found) == 1
    assert found[0]["token"]["bytes"] == [160]


# ------------------------------------------------------------------- the shape #
def test_each_finding_carries_its_neighbours():
    """A lead byte is only wrong in the light of what followed it, so a finding
    without context cannot be judged."""
    calls = [[_tok("a", [97]), _tok("b", [98]), _tok("Ã", [195]),
              _tok("c", [99]), _tok("d", [100])]]
    found = W._trace_findings(calls)
    ctx = found[0]["context"]
    assert [c["token"] for c in ctx] == ["a", "b", "Ã", "c", "d"]


def test_context_is_clipped_at_the_edges():
    calls = [[_tok("Ã", [195])]]
    assert len(W._trace_findings(calls)[0]["context"]) == 1


def test_findings_say_which_call_and_which_token():
    """A page is several blocks, each its own request; "token 4" is meaningless
    without which block it was token 4 of."""
    calls = [[_tok("a", [97])], [_tok("a", [97]), _tok("Ã", [195])]]
    found = W._trace_findings(calls)
    assert found[0]["call"] == 1
    assert found[0]["index"] == 1


def test_the_logprob_comes_through_so_confidence_is_visible():
    """A broken byte the model was SURE about is a different problem from one it
    guessed at."""
    calls = [[_tok("Ã", [195], logprob=-0.0001)]]
    assert W._trace_findings(calls)[0]["token"]["logprob"] == -0.0001


def test_findings_are_capped():
    """A page of pure noise must not return its entire token stream."""
    calls = [[_tok("Ã", [195]) for _ in range(W.TRACE_MAX + 25)]]
    assert len(W._trace_findings(calls)) == W.TRACE_MAX


def test_pydantic_style_objects_are_accepted_too():
    """The OpenAI client hands back model objects, not dicts, depending on version."""
    class _Entry:
        token, bytes, logprob = "Ã", [195], -0.2
    assert len(W._trace_findings([[_Entry()]])) == 1


def test_missing_bytes_are_not_an_anomaly():
    """A server that reports no bytes is not evidence of a broken byte."""
    calls = [[{"token": "a", "logprob": -0.1}]]
    assert W._trace_findings(calls) == []


def test_no_calls_is_no_findings():
    assert W._trace_findings([]) == []
    assert W._trace_findings(None) == []
    assert W._trace_findings([None, []]) == []


# ------------------------------------------------- it must never break the OCR #
def test_the_trace_is_off_unless_asked_for():
    w = W.SuryaWorker({})
    assert w._trace is False
    assert W.SuryaWorker({"token_trace": True})._trace is True


def test_installing_the_hook_without_surya_does_not_raise():
    """The worker runs in an isolated venv; a surya too old to have the backend
    module must degrade to "no diagnostic", never to "no OCR"."""
    w = W.SuryaWorker({"token_trace": True})
    w._install_trace_hook()          # surya is not installed in the test venv


def test_a_clean_page_reports_nothing(monkeypatch):
    """The whole response shape has to stay byte-identical when nothing is wrong."""
    w = W.SuryaWorker({"token_trace": True})
    w._mode = "predictor"
    w._calls.append([_tok("Ã", [195])])          # an anomaly is present...
    monkeypatch.setattr(w, "_predict", lambda img: [
        {"blocks": [{"html": "<p>tutto bene</p>", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
                     "confidence": 0.9, "reading_order": 0}]}])
    from PIL import Image
    page = w.ocr(Image.new("RGB", (10, 10)))
    # ...but the text has no marker, so there is nothing to diagnose.
    assert "unreadable_trace" not in page


def test_an_unreadable_page_carries_the_diagnosis(monkeypatch):
    w = W.SuryaWorker({"token_trace": True})
    w._mode = "predictor"
    from PIL import Image

    def _predict(img):
        # The hook would have filled this during the real call.
        w._calls.append([_tok("p", [112]), _tok("i", [105]), _tok("Ã", [195])])
        return [{"blocks": [{"html": "<p>pi�</p>",
                             "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
                             "confidence": 0.4, "reading_order": 0}]}]
    monkeypatch.setattr(w, "_predict", _predict)
    page = w.ocr(Image.new("RGB", (10, 10)))
    tr = page["unreadable_trace"]
    assert tr["conclusive"] is True
    assert tr["findings"][0]["token"]["bytes"] == [195]


def test_the_trace_is_cleared_between_pages(monkeypatch):
    """Otherwise page 100 of a document reports page 1's anomalies."""
    w = W.SuryaWorker({"token_trace": True})
    w._mode = "predictor"
    from PIL import Image
    monkeypatch.setattr(w, "_predict", lambda img: [
        {"blocks": [{"html": "<p>ok</p>", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
                     "confidence": 0.9, "reading_order": 0}]}])
    w._calls.append([_tok("Ã", [195])])
    w.ocr(Image.new("RGB", (10, 10)))
    assert w._calls == []


def test_analysis_failure_is_reported_not_raised(monkeypatch):
    """A diagnostic that can break a page would be worse than no diagnostic."""
    w = W.SuryaWorker({"token_trace": True})
    w._mode = "predictor"
    from PIL import Image

    def _boom(calls):
        raise RuntimeError("bad trace")
    monkeypatch.setattr(W, "_trace_report", _boom)

    def _predict(img):
        return [{"blocks": [{"html": "<p>pi�</p>",
                             "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
                             "confidence": 0.4, "reading_order": 0}]}]
    monkeypatch.setattr(w, "_predict", _predict)
    page = w.ocr(Image.new("RGB", (10, 10)))
    tr = page["unreadable_trace"]
    assert tr["conclusive"] is False
    assert "bad trace" in tr["why"]


# --------------------------------------------- an empty trace must not lie #
def test_missing_logprobs_are_not_reported_as_a_clean_byte_stream():
    """The flaw this envelope exists to close. With no logprobs there is nothing
    to analyse, and a bare empty finding list would read as "the bytes were sound,
    so the serving path is at fault" — a confident verdict from missing data."""
    rep = W._trace_report([None, None])
    assert rep["findings"] == []
    assert rep["conclusive"] is False
    assert "no logprobs" in rep["why"]
    assert rep["with_logprobs"] == 0


def test_a_hook_that_never_fired_is_also_inconclusive():
    rep = W._trace_report([])
    assert rep["conclusive"] is False
    assert "did not" in rep["why"]


def test_a_sound_stream_with_logprobs_present_blames_the_serving_path():
    """The real verdict, and the one that would mean we can fix this ourselves."""
    rep = W._trace_report([[_tok("p", [112]), _tok("\u00c3", [195]),
                            _tok("\u0142", [160])]])
    assert rep["conclusive"] is True
    assert rep["findings"] == []
    assert "serving path" in rep["why"]


def test_a_broken_stream_blames_the_model():
    rep = W._trace_report([[_tok("p", [112]), _tok("\u00c3", [195])]])
    assert rep["conclusive"] is True
    assert rep["kinds"] == {"invalid-utf8": 1}
    assert "the model is at fault" in rep["why"]


def test_the_envelope_always_counts_the_calls():
    rep = W._trace_report([None, [_tok("a", [97])], []])
    assert rep["calls"] == 3
    assert rep["with_logprobs"] == 1


# ------------------------------------------------------ and out through the API #
def test_the_report_carries_the_trace():
    from codai.ocr.base import OcrPage, OcrLine, unreadable_report
    page = OcrPage(index=0, text="pi\ufffd")
    page.lines = [OcrLine(text="pi\ufffd", bbox=[0, 0, 1, 1], conf=0.4)]
    page.meta["unreadable_trace"] = {
        "calls": 1, "with_logprobs": 1, "conclusive": True,
        "findings": [{"index": 2, "kind": "invalid-utf8",
                      "token": {"bytes": [195]}, "why": "x", "context": []}],
    }
    rep = unreadable_report([page])
    assert rep["count"] == 1
    assert rep["trace"][0]["findings"][0]["token"]["bytes"] == [195]
    assert rep["trace"][0]["page"] == 0
    assert rep["blame"] == "the model"


def test_the_blame_line_says_the_serving_path_when_the_bytes_were_sound():
    from codai.ocr.base import OcrPage, unreadable_report
    page = OcrPage(index=0, text="pi\ufffd")
    page.meta["unreadable_trace"] = {"calls": 1, "with_logprobs": 1,
                                     "conclusive": True, "findings": []}
    assert unreadable_report([page])["blame"] == "the serving path"


def test_the_blame_line_says_unknown_when_the_trace_could_not_be_read():
    """Rather than defaulting to either culprit."""
    from codai.ocr.base import OcrPage, unreadable_report
    page = OcrPage(index=0, text="pi\ufffd")
    page.meta["unreadable_trace"] = {"calls": 2, "with_logprobs": 0,
                                     "conclusive": False, "findings": []}
    assert unreadable_report([page])["blame"] == "unknown"


def test_a_clean_document_has_no_report_at_all():
    from codai.ocr.base import OcrPage, unreadable_report
    page = OcrPage(index=0, text="tutto bene")
    page.meta["unreadable_trace"] = {"conclusive": True, "findings": []}
    assert unreadable_report([page]) == {}


def test_the_trace_is_not_repeated_on_every_page():
    """It is reported once, document-wide. Repeating it per page would put the
    same findings in the response twice."""
    from codai.ocr.base import OcrPage
    page = OcrPage(index=0, text="pi\ufffd")
    page.meta["unreadable_trace"] = {"findings": [{"index": 2}]}
    page.meta["primary_language"] = "it"
    d = page.to_dict()
    assert "unreadable_trace" not in d.get("meta", {})
    assert d["meta"]["primary_language"] == "it"      # other metadata survives


def test_a_page_whose_only_meta_was_the_trace_reports_no_meta():
    from codai.ocr.base import OcrPage
    page = OcrPage(index=0, text="x")
    page.meta["unreadable_trace"] = {"findings": [{"index": 1}]}
    assert "meta" not in page.to_dict()


def test_toggling_the_flag_rebuilds_the_pool():
    """The hook patches surya's backend at LOAD, so a change that did not
    invalidate the pool would appear to save and do nothing."""
    from codai.config import OcrConfig
    from codai.ocr.manager import OcrManager
    off, on = OcrConfig(), OcrConfig()
    on.surya_token_trace = True
    assert OcrManager._pool_params(off) != OcrManager._pool_params(on)


def test_the_engine_tells_the_worker_at_load():
    from codai.config import OcrConfig
    from codai.ocr.surya import SuryaEngine
    cfg = OcrConfig()
    cfg.surya_token_trace = True
    assert SuryaEngine(cfg)._worker_opts()["token_trace"] is True
    assert SuryaEngine(OcrConfig())._worker_opts()["token_trace"] is False
