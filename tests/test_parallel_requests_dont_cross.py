"""Two requests in flight to one pod must not read each other's answer.

The orchestrator talks to a pod in parallel by design -- the front admits 256,
the pool hands the same pod to every caller and only grows when one is loaded,
the HTTP client keeps 64-256 connections per pod, and the blocking call runs in
a 256-worker executor (measured on the production install 2026-10-22; Python's
own default there would have been 20).

What was NOT parallel-safe is what a backend cannot return in the text.
`generate_chat` returns a string, so usage, logprobs, the tool calls the remote
parsed and the thought it separated travel beside it and are read by the caller
AFTER the call (`get_last_*`). Those were attributes of the backend, and one
backend object serves every concurrent request for its model: each call began by
resetting them, so a second request arriving between the first one's answer and
its read blanked the first one's tool call -- or handed it the second one's.
With real parallelism that is not a rare interleaving; it is the common case,
and the symptom is the one already seen in production: an empty message where
the pod had answered perfectly well.

The store is per CALL now (codai/backends/base.py). These tests drive two calls
through ONE backend with the answers deliberately interleaved.
"""
import asyncio
import json
import threading
import types

import pytest

from codai.backends import base
from codai.backends.runpod import RunpodBackend


def _answer(message, usage=None):
    r = types.SimpleNamespace()
    r.status_code = 200
    r.ok = True
    r.headers = {}
    r.json = lambda: {"choices": [{"message": message,
                                   "finish_reason": "tool_calls"}],
                      "usage": usage or {"prompt_tokens": 1,
                                         "completion_tokens": 2,
                                         "total_tokens": 3}}
    r.text = json.dumps(message)
    return r


def _backend(monkeypatch, answer_for):
    """One backend object, as a model's manager holds it, answering per request."""
    b = RunpodBackend.__new__(RunpodBackend)
    b._model_id = b._served_name = "qwen38"
    b._ctx = 32768
    b._last_usage = {}
    b._last_logprobs = None
    b._last_tool_calls = None
    b._last_reasoning = None
    b._url = "https://pod:1/v1"
    b._headers = {}
    b._pool = None
    b._mode = "serverless"

    def _post(url, json=None, **kw):
        return _answer(*answer_for(json))
    monkeypatch.setattr("codai.api.pod_http.post", _post)
    return b


CALL_A = [{"id": "call_a", "type": "function",
           "function": {"name": "agenda_udienze", "arguments": "{}"}}]
CALL_B = [{"id": "call_b", "type": "function",
           "function": {"name": "cerca_sentenza", "arguments": "{}"}}]


def test_two_calls_keep_their_own_tool_call_and_thought(monkeypatch):
    """THE BUG: run A, let B start and reset the attributes, then read A."""
    gate = threading.Event()

    def answer_for(payload):
        who = payload["messages"][0]["content"]
        if who == "A":
            gate.wait(2)                 # A's answer waits for B to have run
            return ({"role": "assistant", "content": None,
                     "tool_calls": CALL_A, "reasoning_content": "thinking A"},
                    {"prompt_tokens": 11, "completion_tokens": 1, "total_tokens": 12})
        return ({"role": "assistant", "content": None,
                 "tool_calls": CALL_B, "reasoning_content": "thinking B"},
                {"prompt_tokens": 22, "completion_tokens": 2, "total_tokens": 24})

    b = _backend(monkeypatch, answer_for)
    seen = {}

    def run(tag):
        # Exactly what the request path does: a per-call store, then the
        # blocking call, then the reads.
        base.new_call_sink()
        b.generate_chat([{"role": "user", "content": tag}])
        seen[tag] = (b.get_last_tool_calls(), b.get_last_reasoning(),
                     b.get_last_usage())

    ta = threading.Thread(target=run, args=("A",))
    tb = threading.Thread(target=run, args=("B",))
    ta.start()
    tb.start()
    tb.join(5)                            # B finishes first, resetting the state
    gate.set()
    ta.join(5)

    assert seen["A"][0] == CALL_A, "A read another request's tool call"
    assert seen["B"][0] == CALL_B
    assert seen["A"][1] == "thinking A"
    assert seen["B"][1] == "thinking B"
    assert seen["A"][2]["prompt_tokens"] == 11
    assert seen["B"][2]["prompt_tokens"] == 22


def test_a_caller_without_a_store_still_works(monkeypatch):
    """Everything that is not the chat path -- the OCR pipeline reading
    logprobs, a test, a script -- never opens a store and must keep the old
    behaviour: the backend's own attributes."""
    b = _backend(monkeypatch, lambda payload: (
        {"role": "assistant", "content": None, "tool_calls": CALL_A,
         "reasoning_content": "thought"},))
    assert b.generate_chat([{"role": "user", "content": "hi"}]) == ""
    assert b.get_last_tool_calls() == CALL_A
    assert b.get_last_reasoning() == "thought"
    assert b.get_last_usage()["total_tokens"] == 3


def test_the_store_travels_into_a_worker_thread(monkeypatch):
    """asyncio.to_thread copies the context, and the copy SHARES the dict, so a
    write from the thread is visible to the awaiting request. That is the whole
    mechanism -- if it ever stopped holding, every answer would come back
    without its tool calls."""
    b = _backend(monkeypatch, lambda payload: (
        {"role": "assistant", "content": None, "tool_calls": CALL_B},))

    async def _run():
        base.new_call_sink()
        await asyncio.to_thread(b.generate_chat, [{"role": "user", "content": "x"}])
        return b.get_last_tool_calls()
    assert asyncio.run(_run()) == CALL_B


def test_a_fresh_store_does_not_inherit_the_previous_answer(monkeypatch):
    """A stale tool call attached to the next request would answer a question
    nobody asked -- the per-call guarantee the attributes used to give by
    resetting, which is what made them race."""
    b = _backend(monkeypatch, lambda payload: (
        {"role": "assistant", "content": None, "tool_calls": CALL_A},))
    base.new_call_sink()
    b.generate_chat([{"role": "user", "content": "1"}])
    assert b.get_last_tool_calls() == CALL_A

    monkeypatch.setattr("codai.api.pod_http.post",
                        lambda *a, **k: _answer({"role": "assistant",
                                                 "content": "plain text"}))
    base.new_call_sink()
    b.generate_chat([{"role": "user", "content": "2"}])
    assert b.get_last_tool_calls() is None


# ------------------------------------------------------------- streamed calls #

def _sse(chunks):
    body = ("".join("data: " + json.dumps(c) + "\n" for c in chunks)
            + "data: [DONE]\n").encode()

    class _R:
        status_code = 200
        ok = True
        headers = {}
        url = "https://pod:1/v1/chat/completions"
        text = body.decode()

        def iter_content(self, chunk_size=8192):
            return [body]

        def json(self):
            return {}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False
    return _R()


def test_two_streams_do_not_merge_their_thoughts(monkeypatch):
    """Reasoning used to be appended to one attribute on the shared backend, so
    two concurrent streams produced one interleaved thought -- and the reader
    runs on its own thread, which inherits no context variables, so the store
    has to be captured when the stream is opened."""
    b = _backend(monkeypatch, lambda payload: ({"role": "assistant"},))

    def _post(url, json=None, **kw):
        tag = json["messages"][0]["content"]
        return _sse([{"choices": [{"index": 0, "delta": {
            "reasoning_content": f"thought {tag}"}}]},
            {"choices": [{"index": 0, "delta": {"content": f"answer {tag}"},
                          "finish_reason": "stop"}]}])
    monkeypatch.setattr("codai.api.pod_http.post", _post)

    async def one(tag):
        base.new_call_sink()
        out = []
        async for chunk in b._stream("https://pod:1/v1/chat/completions",
                                     {"messages": [{"role": "user",
                                                    "content": tag}]},
                                     delta_key="delta"):
            out.append(chunk)
            await asyncio.sleep(0)        # let the other stream interleave
        return "".join(out), b.get_last_reasoning()

    async def both():
        return await asyncio.gather(one("A"), one("B"))
    (text_a, think_a), (text_b, think_b) = asyncio.run(both())

    assert think_a == "thought A"
    assert think_b == "thought B"
    assert "answer A" in text_a and "answer B" in text_b
