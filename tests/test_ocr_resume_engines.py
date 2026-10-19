"""chandra and olmocr resume a cut-off page, the same way surya does.

Both engines stopped at their configured cap and returned whatever they had.
What they had is the worse half of the problem:

  * chandra emits HTML, so a cut-off reply is unclosed markup — it parses to a
    fragment of the page, or to nothing;
  * olmocr opens with YAML front matter, so a reply cut off before the body
    closes parses to metadata and NO TEXT AT ALL — indistinguishable, in the
    finished page, from a model that could not read it.

These tests exercise each engine's own ask path: that the continuation really
goes out as a prefilled assistant turn with BOTH vLLM flags, that the text so
far is what gets sent, and that the page carries the same `model_attempts`
report surya produces. The rules themselves are tested in
tests/test_ocr_resume_shared.py.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codai.config import OcrConfig                                 # noqa: E402
from codai.ocr import chandra as ch                                # noqa: E402
from codai.ocr import olmocr as ol                                 # noqa: E402


class _Reply:
    """A pod_http response."""

    def __init__(self, body, status=200):
        self._body = body
        self.status_code = status
        self.text = "ok"

    def json(self):
        return self._body


def _completion(content, finish="stop", done=10, prompt=4000):
    return {"choices": [{"message": {"content": content},
                         "finish_reason": finish}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": done}}


def _server(monkeypatch, *bodies):
    """pod_http.post answering with each body in turn; records every payload."""
    from codai.api import pod_http
    sent = []

    def _post(url, json=None, headers=None, timeout=None, **kw):
        sent.append(json)
        i = min(len(sent) - 1, len(bodies) - 1)
        return _Reply(bodies[i])

    monkeypatch.setattr(pod_http, "post", _post)
    return sent


# --------------------------------------------------------------- chandra

def _chandra(**kw):
    cfg = OcrConfig()
    cfg.enabled = True
    cfg.chandra_enabled = True
    cfg.chandra_accept_license = True
    cfg.chandra_serve = "server"
    cfg.chandra_server_url = "http://h:8000"
    cfg.chandra_max_retries = 0
    for k, v in kw.items():
        setattr(cfg, k, v)
    eng = ch.ChandraEngine(cfg)
    eng.load()
    return eng


def test_chandra_continues_a_reply_that_hit_the_cap(monkeypatch):
    eng = _chandra(chandra_max_tokens=100, chandra_max_continuations=4)
    sent = _server(monkeypatch,
                   _completion("<p>first", finish="length", done=100),
                   _completion(" second</p>", finish="stop", done=20))
    text, conf, info = eng._ask("data:,", 0.0, 0.1)
    assert text == "<p>first second</p>"
    assert len(sent) == 2
    assert info["resumed"] == 1 and "the page is whole" in info["why"]


def test_chandra_sends_the_text_so_far_as_a_prefilled_assistant_turn(monkeypatch):
    eng = _chandra(chandra_max_tokens=100, chandra_max_continuations=2)
    sent = _server(monkeypatch,
                   _completion("<p>first", finish="length", done=100),
                   _completion("rest</p>", finish="stop", done=5))
    eng._ask("data:,", 0.0, 0.1)
    last = sent[1]["messages"][-1]
    assert last["role"] == "assistant" and last["content"] == "<p>first"


def test_chandra_sends_both_vllm_flags_or_the_model_starts_over(monkeypatch):
    """With add_generation_prompt left at its default the chat template appends a
    fresh assistant turn and the model transcribes the page AGAIN. That read as a
    model repeating itself for a whole day."""
    eng = _chandra(chandra_max_tokens=100, chandra_max_continuations=2)
    sent = _server(monkeypatch,
                   _completion("<p>a", finish="length", done=100),
                   _completion("b</p>", finish="stop", done=5))
    eng._ask("data:,", 0.0, 0.1)
    assert sent[1]["continue_final_message"] is True
    assert sent[1]["add_generation_prompt"] is False
    # …and the first call must NOT carry them: there is nothing to continue.
    assert "continue_final_message" not in sent[0]


def test_chandra_asks_for_a_segment_not_a_whole_new_budget(monkeypatch):
    eng = _chandra(chandra_max_tokens=100, chandra_max_continuations=2,
                   chandra_context_tokens=4300)
    sent = _server(monkeypatch,
                   _completion("<p>a", finish="length", done=100, prompt=4000),
                   _completion("b</p>", finish="stop", done=5))
    eng._ask("data:,", 0.0, 0.1)
    # 4300 window - (4000 prompt + 100 produced) = 200 left, under the 100 cap's
    # own ceiling, so the segment is what fits.
    assert sent[1]["max_tokens"] == 100


def test_chandra_does_not_resume_a_finished_reply(monkeypatch):
    eng = _chandra(chandra_max_tokens=100, chandra_max_continuations=4)
    sent = _server(monkeypatch, _completion("<p>done</p>", finish="stop", done=20))
    text, _conf, info = eng._ask("data:,", 0.0, 0.1)
    assert len(sent) == 1 and text == "<p>done</p>"
    assert info == {}            # nothing to report, so the page carries nothing


def test_chandra_with_resumption_off_marks_the_page_instead(monkeypatch):
    eng = _chandra(chandra_max_tokens=100, chandra_max_continuations=0)
    sent = _server(monkeypatch, _completion("<p>cut", finish="length", done=100))
    _text, _conf, info = eng._ask("data:,", 0.0, 0.1)
    assert len(sent) == 1
    assert info["truncated"] is True and info["complete"] is False
    assert "chandra_max_continuations" in info["why"]
    assert info["max_continuations"] == 0


def test_chandras_report_reaches_the_page(monkeypatch):
    """The whole point of the report is that a caller sees it."""
    eng = _chandra(chandra_max_tokens=100, chandra_max_continuations=2)
    _server(monkeypatch,
            _completion('<div data-label="Text"><p>a', finish="length", done=100),
            _completion('b</p></div>', finish="stop", done=5))
    from PIL import Image
    page = eng.recognize_image(Image.new("RGB", (80, 40), "white"))
    assert page.meta["model_attempts"]["resumed"] == 1


def test_a_request_overrides_chandras_configured_budgets(monkeypatch):
    from codai.ocr import base
    eng = _chandra(chandra_max_tokens=100, chandra_max_continuations=8)
    sent = _server(monkeypatch, _completion("<p>cut", finish="length", done=64))
    tok = base.PAGE_OPTIONS.set({"max_tokens": 64, "max_continuations": 0})
    try:
        _text, _conf, info = eng._ask("data:,", 0.0, 0.1)
    finally:
        base.PAGE_OPTIONS.reset(tok)
    assert sent[0]["max_tokens"] == 64          # the request's cap went out
    assert len(sent) == 1                       # and its "no resumption" held
    assert info["max_continuations"] == 0


# ---------------------------------------------------------------- olmocr

FRONT = ("---\nprimary_language: it\nis_rotation_valid: true\n"
         "rotation_correction: 0\nis_table: false\nis_diagram: false\n---\n")


def _olmocr(**kw):
    cfg = OcrConfig(enabled=True, olmocr_enabled=True, default_engine="olmocr")
    cfg.olmocr_serve = "server"
    cfg.olmocr_server_url = "http://h:8000"
    for k, v in kw.items():
        setattr(cfg, k, v)
    eng = ol.OlmOcrEngine(cfg)
    eng.load()
    return eng


def test_olmocr_continues_a_reply_that_hit_the_cap(monkeypatch):
    eng = _olmocr(olmocr_max_tokens=100, olmocr_max_continuations=4)
    sent = _server(monkeypatch,
                   _completion(FRONT + "La domanda", finish="length", done=100),
                   _completion(" e' fondata.", finish="stop", done=20))
    from PIL import Image
    text, _conf, info = eng._ask(Image.new("RGB", (80, 40), "white"))
    assert "La domanda e' fondata." in text
    assert len(sent) == 2 and info["resumed"] == 1


def test_olmocr_sends_the_prefix_and_both_flags(monkeypatch):
    from PIL import Image
    eng = _olmocr(olmocr_max_tokens=100, olmocr_max_continuations=2)
    sent = _server(monkeypatch,
                   _completion(FRONT + "half", finish="length", done=100),
                   _completion(" rest", finish="stop", done=5))
    eng._ask(Image.new("RGB", (80, 40), "white"))
    last = sent[1]["messages"][-1]
    assert last["role"] == "assistant" and last["content"].endswith("half")
    assert sent[1]["continue_final_message"] is True
    assert sent[1]["add_generation_prompt"] is False


def test_olmocrs_front_matter_makes_truncation_worse_so_it_is_reported(monkeypatch):
    """A reply cut off before the body closes parses to metadata and no text —
    which is why this engine needed resumption most."""
    from PIL import Image
    eng = _olmocr(olmocr_max_tokens=100, olmocr_max_continuations=0)
    _server(monkeypatch, _completion("---\nprimary_language: it\n",
                                     finish="length", done=100))
    page = eng.recognize_image(Image.new("RGB", (80, 40), "white"))
    assert page.meta["model_attempts"]["truncated"] is True
    assert "INCOMPLETE" in page.meta["model_attempts"]["why"]


def test_olmocr_reports_nothing_for_a_page_that_fitted(monkeypatch):
    from PIL import Image
    eng = _olmocr(olmocr_max_tokens=4096, olmocr_max_continuations=4)
    _server(monkeypatch, _completion(FRONT + "tutto il testo", finish="stop",
                                     done=50))
    page = eng.recognize_image(Image.new("RGB", (80, 40), "white"))
    assert "model_attempts" not in page.meta
    assert page.text.strip() == "tutto il testo"


def test_a_request_overrides_olmocrs_configured_budgets(monkeypatch):
    from PIL import Image

    from codai.ocr import base
    eng = _olmocr(olmocr_max_tokens=4096, olmocr_max_continuations=0)
    sent = _server(monkeypatch,
                   _completion(FRONT + "a", finish="length", done=512),
                   _completion("b", finish="stop", done=5))
    tok = base.PAGE_OPTIONS.set({"max_tokens": 512, "max_continuations": 3})
    try:
        eng._ask(Image.new("RGB", (80, 40), "white"))
    finally:
        base.PAGE_OPTIONS.reset(tok)
    # Resumption was OFF in the config and the request turned it on.
    assert sent[0]["max_tokens"] == 512 and len(sent) == 2


def test_both_engines_share_one_implementation():
    """Three copies of these rules is how they drift apart; the module is the
    point of the change."""
    for mod in (ROOT / "codai/ocr/chandra.py", ROOT / "codai/ocr/olmocr.py"):
        src = mod.read_text()
        # `loopguard` rides alongside it since the early abort landed. What
        # this pins is that neither engine grows its own resumption.
        assert ("from codai.ocr import resume" in src
                or "from codai.ocr import loopguard, resume" in src)
        assert "resume.resume(" in src and "resume.report(" in src
