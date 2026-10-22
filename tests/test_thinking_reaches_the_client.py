"""A model's thinking must arrive AS THINKING, and while it is thinking.

2026-10-22, from one pod, on one set of weights, behind two aliases:

  qwen38        -> clean answer
  diogene-alpha -> "The user is asking to list hearings from 2026-10-21 to..."

The entire chain of thought was delivered to the client as the answer. The gate
was a regex on the request's model NAME -- `qwen3` matches "qwen38" and does
not match "diogene-alpha" -- so the same template got opposite treatment.

Three separate breaks on the way to the client, all fixed here:
  * the gate asked the name instead of the model;
  * the buffered formatter took no `reasoning`, so a thought that WAS separated
    got dropped -- no leak, but no thinking either;
  * a proxy backend's text contract carried neither the remote's reasoning
    field (buffered) nor its reasoning deltas (streamed).
"""
import json
import types

import pytest

from codai.models import chat_style
from codai.models.parser import OpenAIFormatter
from codai.backends.runpod import RunpodBackend
from codai.api import vllm_worker as vw


PREFILL_TPL = ("{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}"
               "{{- '<think>\\n' }}{%- endif %}")
PLAIN_TPL = "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"


# ------------------------------------------------- does this model think at all #

@pytest.fixture
def model(monkeypatch):
    state = {"entry": {}, "template": ""}

    def _entry_for(name):
        return state["entry"]
    monkeypatch.setattr("codai.models.manager._model_entry_for", _entry_for)
    monkeypatch.setattr(chat_style, "chat_template_text", lambda p: state["template"])

    def _set(**kw):
        state.update(kw)
        chat_style.forget()
    return _set


def test_a_prefilled_think_tag_means_the_model_thinks(model):
    """THE BUG: this is read from the template now, so two aliases of one model
    cannot disagree about it."""
    model(entry={"path": "org/qwen3.8"}, template=PREFILL_TPL)
    assert chat_style.model_thinks("diogene-alpha") is True
    assert chat_style.model_thinks("qwen38") is True


def test_a_template_with_no_thinking_says_so(model):
    model(entry={"path": "org/plain"}, template=PLAIN_TPL)
    assert chat_style.model_thinks("whatever") is False


def test_an_adapter_is_judged_by_the_base_it_adapts(model):
    model(entry={"loras": [{"name": "a", "path": "sha256:ab"}],
                 "runpod": {"hf_repo": "org/qwen3.8"}}, template=PREFILL_TPL)
    assert chat_style.model_thinks("diogene-alpha") is True


def test_an_unreadable_template_is_not_an_answer(model):
    """It must fall through to the name, not assert "does not think"."""
    model(entry={}, template="")
    assert chat_style.model_thinks("org/unknown") is False


# ------------------------------------------- the buffered response carries it #

def test_the_formatter_sends_the_reasoning():
    """THE BUG: format_litellm_full had no `reasoning` parameter, so a thought
    the caller had separated out was then thrown away."""
    out = OpenAIFormatter("m").format_litellm_full(
        text="Ecco le udienze", prompt_tokens=1, completion_tokens=2,
        reasoning="I need to call the tool.")
    msg = out["choices"][0]["message"]
    assert msg["reasoning"] == "I need to call the tool."
    assert msg["reasoning_content"] == "I need to call the tool."
    assert msg["content"] == "Ecco le udienze"


def test_no_reasoning_adds_no_fields():
    msg = OpenAIFormatter("m").format_litellm_full(
        text="ciao", prompt_tokens=1, completion_tokens=1)["choices"][0]["message"]
    assert not msg.get("reasoning")
    assert not msg.get("reasoning_content")


def test_reasoning_survives_beside_a_tool_call():
    out = OpenAIFormatter("m").format_litellm_full(
        text="", prompt_tokens=1, completion_tokens=2,
        tool_calls=[{"id": "c1", "type": "function",
                     "function": {"name": "f", "arguments": "{}"}}],
        reasoning="thinking first")
    choice = out["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["reasoning"] == "thinking first"


# ------------------------------------------------------ across the proxy hop #

def _answer(message):
    r = types.SimpleNamespace()
    r.status_code, r.ok = 200, True
    r.json = lambda: {"choices": [{"message": message, "finish_reason": "stop"}],
                      "usage": {"prompt_tokens": 1, "completion_tokens": 2}}
    r.text, r.headers = json.dumps(message), {}
    return r


def _backend(monkeypatch, message):
    b = RunpodBackend.__new__(RunpodBackend)
    b._model_id = b._served_name = "diogene-alpha"
    b._last_usage, b._last_logprobs = {}, None
    b._last_tool_calls = b._last_reasoning = None
    b._url, b._headers, b._pool = "https://pod:1/v1", {}, None
    monkeypatch.setattr(b, "_acquire_endpoint",
                        lambda: ("https://pod:1/v1", lambda: None, ""))
    monkeypatch.setattr("codai.api.pod_http.post", lambda *a, **k: _answer(message))
    return b


def test_a_remotes_separated_thought_is_not_lost(monkeypatch):
    """The pod strips the tags and sends the thought in its own field, so there
    is nothing left in the text for the caller to find."""
    b = _backend(monkeypatch, {"role": "assistant", "content": "Ecco",
                               "reasoning_content": "I will call the tool."})
    assert b.generate_chat([{"role": "user", "content": "x"}]) == "Ecco"
    assert b.get_last_reasoning() == "I will call the tool."


def test_the_other_spelling_is_accepted_too(monkeypatch):
    b = _backend(monkeypatch, {"role": "assistant", "content": "Ecco",
                               "reasoning": "thought"})
    b.generate_chat([{"role": "user", "content": "x"}])
    assert b.get_last_reasoning() == "thought"


def test_it_does_not_carry_over_to_the_next_call(monkeypatch):
    b = _backend(monkeypatch, {"role": "assistant", "content": "a",
                               "reasoning_content": "first"})
    b.generate_chat([{"role": "user", "content": "x"}])
    monkeypatch.setattr("codai.api.pod_http.post",
                        lambda *a, **k: _answer({"role": "assistant", "content": "b"}))
    b.generate_chat([{"role": "user", "content": "x"}])
    assert b.get_last_reasoning() is None


def test_a_generating_backend_reports_none():
    from codai.backends.base import ModelBackend
    assert ModelBackend.get_last_reasoning(object()) is None


def test_the_manager_forwards_it():
    from codai.models.manager import ModelManager
    m = ModelManager.__new__(ModelManager)
    m.backend = types.SimpleNamespace(get_last_reasoning=lambda: "t")
    assert m.get_last_reasoning() == "t"
    m.backend = types.SimpleNamespace()
    assert m.get_last_reasoning() is None


# ------------------------------------------- streamed: live, not in one lump #

def _sse(deltas):
    """A remote's SSE body, one chunk per delta."""
    out = []
    for d in deltas:
        out.append("data: " + json.dumps({"choices": [{"delta": d,
                                                       "finish_reason": None}]}))
    out.append("data: [DONE]")
    return "\n".join(out) + "\n"


def _drain(backend, body):
    """Run _stream against a canned SSE body and collect what it yields."""
    import asyncio

    class _Resp:
        status_code, ok = 200, True
        text, headers = "", {}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def iter_content(self, chunk_size=8192):
            yield body.encode()

    import codai.api.pod_http as ph
    orig = ph.post
    ph.post = lambda *a, **k: _Resp()
    try:
        async def go():
            return [c async for c in backend._stream("u", {}, "delta")]
        return asyncio.run(go())
    finally:
        ph.post = orig


def _stream_backend():
    b = RunpodBackend.__new__(RunpodBackend)
    b._last_usage, b._last_logprobs = {}, None
    b._last_tool_calls = b._last_reasoning = None
    b._headers = {}
    b._auth = lambda bearer="": {}
    b._check = lambda r: None
    return b


def test_streamed_thinking_is_retagged_so_it_can_be_split_live():
    """THE BUG: delta.reasoning_content was dropped, so a client streaming a
    thinking model saw nothing until the answer began. The tags go back on, in
    order, as the deltas arrive -- not collected and emitted at the end."""
    b = _stream_backend()
    chunks = _drain(b, _sse([{"reasoning_content": "I need "},
                             {"reasoning_content": "the dates."},
                             {"content": "Ecco "},
                             {"content": "le udienze"}]))
    assert chunks == ["<think>I need ", "the dates.", "</think>Ecco ", "le udienze"]
    assert "".join(chunks).startswith("<think>")
    assert b.get_last_reasoning() == "I need the dates."


def test_the_opening_tag_is_sent_once():
    b = _stream_backend()
    chunks = _drain(b, _sse([{"reasoning_content": "a"}, {"reasoning_content": "b"},
                             {"reasoning_content": "c"}]))
    assert "".join(chunks).count("<think>") == 1


def test_an_all_thinking_stream_leaves_the_tag_open():
    """A thought followed by a tool call and no content at all. The caller's
    gate flushes an unclosed thought as reasoning, so closing it here would be
    wrong -- there is no content for it to precede."""
    b = _stream_backend()
    chunks = _drain(b, _sse([{"reasoning_content": "just thinking"}]))
    assert chunks == ["<think>just thinking"]
    assert "</think>" not in "".join(chunks)


def test_a_stream_with_no_thinking_is_untouched():
    b = _stream_backend()
    assert _drain(b, _sse([{"content": "ciao"}, {"content": " mondo"}])) == \
        ["ciao", " mondo"]
    assert b.get_last_reasoning() is None


def test_a_remote_that_already_sends_tags_is_not_double_tagged():
    """Some remotes do not separate the thought at all: it arrives inside
    content, tags and all, and the caller's gate has always handled that."""
    b = _stream_backend()
    chunks = _drain(b, _sse([{"content": "thinking"}, {"content": "</think>"},
                             {"content": "answer"}]))
    assert chunks == ["thinking", "</think>", "answer"]


# --------------------------------- thinking is on unless the request says no #

def _req(**kw):
    from codai.pydantic.textrequest import ChatCompletionRequest as TextRequest
    base = {"model": "m", "messages": [{"role": "user", "content": "x"}]}
    base.update(kw)
    return TextRequest(**base)


def test_unset_sends_nothing_so_the_model_keeps_its_default():
    """A thinking model thinks by default; coderai must not quietly ask it to
    stop just because the caller said nothing."""
    from codai.api.text import _chat_passthrough
    assert _req().enable_thinking is None
    out = _chat_passthrough(_req()) or {}
    assert "chat_template_kwargs" not in out


def test_an_explicit_false_reaches_the_chat_template():
    """THE BUG: coderai consumed enable_thinking and no backend forwarded it,
    so a request could not turn thinking off on a served model at all."""
    from codai.api.text import _chat_passthrough
    out = _chat_passthrough(_req(enable_thinking=False))
    assert out["chat_template_kwargs"] == {"enable_thinking": False}


def test_an_explicit_true_is_forwarded_too():
    from codai.api.text import _chat_passthrough
    out = _chat_passthrough(_req(enable_thinking=True))
    assert out["chat_template_kwargs"] == {"enable_thinking": True}


def test_a_callers_own_template_kwargs_are_not_overwritten():
    from codai.api.text import _chat_passthrough
    out = _chat_passthrough(_req(enable_thinking=False,
                                 chat_template_kwargs={"enable_thinking": True,
                                                       "x": 1}))
    assert out["chat_template_kwargs"] == {"enable_thinking": True, "x": 1}


def test_the_pod_payload_carries_it():
    """The local vLLM backend has forwarded passthrough for releases; a POD got
    none of it, so this could not reach a pod-served model."""
    b = RunpodBackend.__new__(RunpodBackend)
    b._model_id = b._served_name = "qwen38"
    payload = b._chat_payload([{"role": "user", "content": "x"}], 100, 0.0, 1.0,
                              None, False,
                              passthrough={"chat_template_kwargs":
                                           {"enable_thinking": False}})
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_passthrough_cannot_retarget_a_pod():
    b = RunpodBackend.__new__(RunpodBackend)
    b._model_id = b._served_name = "qwen38"
    msgs = [{"role": "user", "content": "x"}]
    payload = b._chat_payload(msgs, 100, 0.0, 1.0, None, False,
                              passthrough={"model": "someone-elses",
                                           "stream": True, "messages": []})
    assert payload["model"] == "qwen38"
    assert payload["stream"] is False
    assert payload["messages"] is msgs


# --------------------------- the SERVER must separate it: the tag is a token #

class _Cfg:
    tool_call_parser = "auto"
    reasoning_parser = "auto"


def test_a_thinking_model_launches_with_a_reasoning_parser(tmp_path):
    """THE ROOT CAUSE: </think> is a SPECIAL TOKEN and vLLM decodes with
    skip_special_tokens, so the tag is deleted before the text exists. A pod
    returned the thought and the answer concatenated with no separator of any
    kind -- nothing downstream can split on a boundary that was removed."""
    d = tmp_path / "m"; d.mkdir()
    (d / "chat_template.jinja").write_text(
        "{%- if add_generation_prompt %}{{- '<think>\\n' }}{%- endif %}"
        "<tool_call>\n<function=f>\n<parameter=k>v</parameter>")
    assert vw._reasoning_args(_Cfg(), {}, str(d)) == ["--reasoning-parser", "qwen3"]


def test_a_model_that_does_not_think_gets_no_parser(tmp_path):
    """A parser looking for a tag that is never coming."""
    d = tmp_path / "m"; d.mkdir()
    (d / "chat_template.jinja").write_text(
        "<tool_call>\n<function=f>\n<parameter=k>v</parameter>")
    assert vw._reasoning_args(_Cfg(), {}, str(d)) == []


def test_an_explicit_reasoning_parser_wins(tmp_path):
    d = tmp_path / "m"; d.mkdir()
    (d / "chat_template.jinja").write_text("<think>")
    cfg = _Cfg(); cfg.reasoning_parser = "deepseek_r1"
    assert vw._reasoning_args(cfg, {}, str(d)) == ["--reasoning-parser", "deepseek_r1"]


def test_off_leaves_the_channels_joined(tmp_path):
    d = tmp_path / "m"; d.mkdir()
    (d / "chat_template.jinja").write_text("<think>")
    cfg = _Cfg(); cfg.reasoning_parser = "off"
    assert vw._reasoning_args(cfg, {}, str(d)) == []


def test_an_unreadable_template_adds_no_flag(tmp_path):
    assert vw._reasoning_args(_Cfg(), {}, str(tmp_path / "nope")) == []


def test_the_families_each_get_their_own(tmp_path):
    from codai.models.chat_style import REASONING_PARSERS, vllm_reasoning_parser
    import codai.models.chat_style as cs
    for style, tpl, want in (
            ("glm_argkey", "<think><tool_call>x<arg_key>k</arg_key>", "glm45"),
            ("mistral_tokens", "<think>[TOOL_CALLS]", "mistral"),
    ):
        cs.forget()
        cs._CACHE["p"] = tpl
        assert vllm_reasoning_parser("p") == want, style
