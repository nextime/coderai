"""An answer cut off at the token limit must not report that it stopped.

2026-10-23, measured against the production pod: a reasoning model asked for 64
tokens spent all 64 inside its thought, returned NO content, and the response
said `finish_reason: "stop"`.

    max_tokens=64    finish=stop  content=0 chars ''  reasoning=262 chars
    max_tokens=1500  finish=stop  content=13 chars 'XMU-5635-FB'  reasoning=448

Two layers were each sure the other knew better. The proxy hardcoded
`finish_reason = "stop"` and never read what the pod reported; the formatter
then hardcoded it again, so even a caller that worked the reason out had it
overwritten on the way to the client. vLLM had said "length" all along.

Why it matters more than a label: "length" is the one reason a client can act
on. An answer truncated mid-sentence is incomplete, and told it stopped
normally the client has nothing to retry on and no sign anything is missing. On
a reasoning model the thought is generated FIRST and spends the same budget, so
a tight limit yields an empty answer that claims to have finished -- which is
how this was found, as a needle test scoring 0% for reasons that had nothing to
do with what it was measuring.
"""
import asyncio
import json

import pytest

from codai.api.text import _finish_reason
from codai.backends.base import ModelBackend, new_call_sink
from codai.backends.runpod import RunpodBackend
from codai.backends.vllm import VllmBackend
from codai.models.parser import OpenAIFormatter


# ----------------------------------------------------------- the pod's answer #

def _answer(finish_reason, content="half a sentence that stops mid-", reasoning=None):
    msg = {"role": "assistant", "content": content}
    if reasoning:
        msg["reasoning_content"] = reasoning
    return {"choices": [{"index": 0, "message": msg,
                         "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 7855, "completion_tokens": 64}}


class _Resp:
    status_code = 200
    ok = True
    headers = {}
    url = "https://pod:1/v1/chat/completions"

    def __init__(self, payload):
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _sse(*objs):
    body = ("\n".join("data: " + json.dumps(o) for o in objs)
            + "\ndata: [DONE]\n").encode()

    class _R(_Resp):
        def __init__(self):
            self.text = body.decode()

        def iter_content(self, chunk_size=8192):
            return [body]

        def json(self):
            return {}
    return _R()


def _runpod(monkeypatch, response):
    b = RunpodBackend.__new__(RunpodBackend)
    b._model_id = b._served_name = "qwen38"
    b._ctx = 122880
    b._last_usage = {}
    b._last_logprobs = b._last_tool_calls = b._last_reasoning = None
    b._last_finish_reason = None
    b._url, b._headers, b._pool, b._mode = "https://pod:1/v1", {}, None, "serverless"
    monkeypatch.setattr("codai.api.pod_http.post", lambda *a, **k: response)
    return b


def _vllm(monkeypatch, response):
    b = VllmBackend.__new__(VllmBackend)
    b._model_id = b._served_name = "qwen38"
    b._ctx = 122880
    b._last_usage = {}
    b._last_logprobs = b._last_tool_calls = b._last_reasoning = None
    b._last_finish_reason = None
    b._url = "http://127.0.0.1:41721"
    monkeypatch.setattr("requests.post", lambda *a, **k: response)
    return b


@pytest.mark.parametrize("make", [_runpod, _vllm])
def test_the_remote_said_length_and_the_proxy_keeps_it(monkeypatch, make):
    """THE BUG: the pod's own word for it was read and thrown away."""
    b = make(monkeypatch, _Resp(_answer("length")))
    new_call_sink()
    b.generate_chat([{"role": "user", "content": "hi"}], max_tokens=64)
    assert b.get_last_finish_reason() == "length"


@pytest.mark.parametrize("make", [_runpod, _vllm])
def test_a_normal_answer_still_says_stop(monkeypatch, make):
    b = make(monkeypatch, _Resp(_answer("stop", content="done.")))
    new_call_sink()
    b.generate_chat([{"role": "user", "content": "hi"}], max_tokens=1500)
    assert b.get_last_finish_reason() == "stop"


@pytest.mark.parametrize("make", [_runpod, _vllm])
def test_the_streamed_path_carries_it_too(monkeypatch, make):
    """The streamed answer is the one production actually uses."""
    b = make(monkeypatch, _sse(
        {"choices": [{"index": 0, "delta": {"content": "half a "}}]},
        {"choices": [{"index": 0, "delta": {"content": "sentence that"}},
                     ]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]}))
    new_call_sink()

    async def _run():
        return [c async for c in b.generate_chat_stream(
            [{"role": "user", "content": "hi"}], max_tokens=64)]
    asyncio.run(_run())
    assert b.get_last_finish_reason() == "length"


def test_a_backend_that_does_not_know_says_nothing():
    """A local backend generates text and is not told a reason; None lets the
    caller fall back to the budget rather than invent authority it lacks."""
    assert ModelBackend.get_last_finish_reason(object()) is None


# --------------------------------------------------------------- the decision #

class _Mgr:
    def __init__(self, reported=None, raises=False):
        self._reported, self._raises = reported, raises

    def get_last_finish_reason(self):
        if self._raises:
            raise RuntimeError("backend is gone")
        return self._reported


def test_what_the_backend_reports_wins():
    assert _finish_reason(_Mgr("length"), 10, 1500) == "length"
    assert _finish_reason(_Mgr("stop"), 1500, 1500) == "stop"


def test_without_a_report_the_budget_decides():
    """A local backend that used every token it was allowed was cut off."""
    assert _finish_reason(_Mgr(), 64, 64) == "length"
    assert _finish_reason(_Mgr(), 65, 64) == "length"
    assert _finish_reason(_Mgr(), 63, 64) == "stop"


def test_nothing_is_inferred_without_a_limit():
    """A long answer is not a truncated one; with no limit set there is no
    evidence either way, and guessing would mislabel every long completion."""
    assert _finish_reason(_Mgr(), 100000, None) == "stop"
    assert _finish_reason(_Mgr(), 0, 0) == "stop"


def test_a_broken_backend_does_not_break_the_answer():
    assert _finish_reason(_Mgr(raises=True), 10, 1500) == "stop"
    assert _finish_reason(object(), 10, 1500) == "stop"


# ---------------------------------------------------------------- the formatter #

def test_the_formatter_no_longer_overwrites_the_reason():
    """THE SECOND HALF OF THE BUG: the caller's reason was computed, passed in,
    and replaced with "stop" here."""
    out = OpenAIFormatter("qwen38").format_full(
        "half a sentence that stops mid-", 7855, 64, finish_reason="length")
    assert out["choices"][0]["finish_reason"] == "length"


def test_the_formatter_still_defaults_to_stop():
    out = OpenAIFormatter("qwen38").format_full("done.", 10, 3)
    assert out["choices"][0]["finish_reason"] == "stop"


def test_a_tool_call_is_still_a_tool_call():
    """tool_calls describes the answer's SHAPE and must not be overruled by a
    truncation that happened after the call was fully formed."""
    calls = [{"id": "c1", "type": "function",
              "function": {"name": "f", "arguments": "{}"}}]
    out = OpenAIFormatter("qwen38").format_full(
        "", 10, 3, tool_calls=calls, finish_reason="length")
    assert out["choices"][0]["finish_reason"] == "tool_calls"


def test_the_final_stream_chunk_carries_it():
    out = OpenAIFormatter("qwen38").format_chunk(
        "", is_final=True, usage={"prompt_tokens": 1, "completion_tokens": 64,
                                  "total_tokens": 65}, finish_reason="length")
    assert out["choices"][0]["finish_reason"] == "length"
    mid = OpenAIFormatter("qwen38").format_chunk("x", is_final=False,
                                                 finish_reason="length")
    assert mid["choices"][0]["finish_reason"] is None, "only the last chunk says why"


# ------------------------------------------------------- the production shape #

@pytest.mark.parametrize("make", [_runpod, _vllm])
def test_the_empty_thinking_answer_is_reported_as_truncated(monkeypatch, make):
    """What was actually observed: all 64 tokens spent in the thought, no
    content, and the client told the model had stopped on its own."""
    b = make(monkeypatch, _Resp(_answer("length", content="",
                                        reasoning="x" * 262)))
    new_call_sink()
    content = b.generate_chat([{"role": "user", "content": "find the code"}],
                              max_tokens=64)
    assert content == ""
    assert b.get_last_finish_reason() == "length"
    out = OpenAIFormatter("qwen38").format_full(
        content, 7855, 64, reasoning=b.get_last_reasoning(),
        finish_reason=b.get_last_finish_reason())
    choice = out["choices"][0]
    assert choice["finish_reason"] == "length", (
        "an empty answer that spent its whole budget thinking must say why")
    assert (choice["message"].get("reasoning_content") or "").startswith("x")
