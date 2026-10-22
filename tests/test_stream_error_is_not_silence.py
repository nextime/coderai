"""A remote that fails mid-stream must not look like a remote that finished.

2026-10-22, production: a conversation longer than the pod's context window was
answered, over the streaming path, with HTTP 200, one final chunk carrying
`finish_reason: "stop"`, the PREVIOUS request's usage, and no content. No error
anywhere -- a successful, empty completion.

The pod had reported the failure properly. A remote serving an OpenAI API
answers 200 as soon as the stream opens, so anything that goes wrong afterwards
can only be said inside the stream, as an event: coderai does exactly that, and
deliberately, so a client treats it as an error instead of as assistant text.

But that event carries no `choices`, and the proxy's SSE reader looked only at
`choices[0].delta`. So the event was parsed, found to contain no text, and
dropped -- then the stream ended and the caller reported an empty answer. Every
mid-stream failure from a pod arrived this way, not just this one.
"""
import asyncio
import json

import pytest

from codai.backends.base import RemoteStreamError
from codai.backends.runpod import RunpodBackend
from codai.backends.vllm import VllmBackend


#: What the pod really sent (a context-window refusal, relayed as coderai does).
ERROR_EVENT = {"error": {"message": ("context_length_exceeded: 32769 tokens "
                                     "requested, limit 32768"),
                         "type": "invalid_request_error",
                         "code": "context_length_exceeded"}}


def _sse(*objs):
    """A streaming response body, as `requests` hands it over."""
    lines = []
    for o in objs:
        lines.append("data: " + json.dumps(o) if not isinstance(o, str) else o)
    body = ("\n".join(lines) + "\ndata: [DONE]\n").encode()

    class _R:                      # a context manager, as requests' own is
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


def _delta(text):
    return {"choices": [{"index": 0, "delta": {"content": text}}]}


def _drain(gen):
    async def _run():
        out = []
        async for chunk in gen:
            out.append(chunk)
        return out
    return asyncio.run(_run())


def _runpod(monkeypatch, response):
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
    monkeypatch.setattr("codai.api.pod_http.post", lambda *a, **k: response)
    return b


def _vllm(monkeypatch, response):
    b = VllmBackend.__new__(VllmBackend)
    b._model_id = b._served_name = "qwen38"
    b._ctx = 32768
    b._last_usage = {}
    b._last_logprobs = None
    b._last_tool_calls = None
    b._last_reasoning = None
    b._url = "http://127.0.0.1:41721"
    monkeypatch.setattr("requests.post", lambda *a, **k: response)
    return b


# ------------------------------------------------------------------- the pod #

def test_an_error_event_from_a_pod_is_raised(monkeypatch):
    """THE BUG: this came back as a stream that simply ended, and the request
    was reported to the client as an empty answer."""
    b = _runpod(monkeypatch, _sse(ERROR_EVENT))
    with pytest.raises(RemoteStreamError) as got:
        _drain(b._stream("https://pod:1/v1/chat/completions", {}, delta_key="delta"))
    assert "32768" in str(got.value)          # the numbers survive the raise


def test_an_error_after_some_text_still_raises(monkeypatch):
    """Half an answer followed by a failure is not a complete answer. The text
    already sent is kept -- the caller has streamed it to the client -- and the
    failure is then reported rather than swallowed."""
    b = _runpod(monkeypatch, _sse(_delta("Buon"), _delta("giorno"), ERROR_EVENT))
    seen = []

    async def _run():
        async for chunk in b._stream("https://pod:1/v1/chat/completions", {},
                                     delta_key="delta"):
            seen.append(chunk)
    with pytest.raises(RemoteStreamError):
        asyncio.run(_run())
    assert "".join(seen) == "Buongiorno"


def test_a_normal_stream_is_untouched(monkeypatch):
    b = _runpod(monkeypatch, _sse(_delta("ciao"),
                                  {"choices": [{"index": 0, "delta": {},
                                                "finish_reason": "stop"}]}))
    assert "".join(_drain(b._stream("https://pod:1/v1/chat/completions", {},
                                    delta_key="delta"))) == "ciao"


def test_an_error_with_no_message_still_fails_loudly(monkeypatch):
    """A remote that reports `{"error": {}}` has still reported an error; the
    one thing that must not happen is a silent success."""
    b = _runpod(monkeypatch, _sse({"error": {}}))
    with pytest.raises(RemoteStreamError):
        _drain(b._stream("https://pod:1/v1/chat/completions", {}, delta_key="delta"))


def test_a_literal_null_error_field_is_not_a_failure(monkeypatch):
    """Some OpenAI-compatible servers put `"error": null` in every ordinary
    chunk. Reading the key's presence alone would fail every one of their
    streams; reading it as truthy would let `{"error": {}}` pass as a success.
    Neither is acceptable, so it is "present and not None"."""
    b = _runpod(monkeypatch, _sse({"error": None, "choices": [
        {"index": 0, "delta": {"content": "ciao"}}]}))
    assert "".join(_drain(b._stream("https://pod:1/v1/chat/completions", {},
                                    delta_key="delta"))) == "ciao"


def test_an_error_as_a_bare_string_is_handled(monkeypatch):
    b = _runpod(monkeypatch, _sse({"error": "boom"}))
    with pytest.raises(RemoteStreamError) as got:
        _drain(b._stream("https://pod:1/v1/chat/completions", {}, delta_key="delta"))
    assert "boom" in str(got.value)


# ---------------------------------------------------------- the local server #

def test_a_local_vllm_error_event_is_raised_too(monkeypatch):
    b = _vllm(monkeypatch, _sse(ERROR_EVENT))
    monkeypatch.setattr(b, "_raise_for_status", lambda *a, **k: None)
    with pytest.raises(RemoteStreamError):
        _drain(b._stream("http://127.0.0.1:41721/v1/chat/completions", {},
                         delta_key="delta"))


def test_the_overflow_reaches_the_client_as_an_overflow(monkeypatch):
    """What the caller does with it: the raised text must still read as a
    context overflow, because that is what decides between an error event and a
    compact-and-retry."""
    from codai.api import context_overflow as co
    b = _runpod(monkeypatch, _sse(ERROR_EVENT))
    with pytest.raises(RemoteStreamError) as got:
        _drain(b._stream("https://pod:1/v1/chat/completions", {}, delta_key="delta"))
    assert co.is_overflow(str(got.value))
    assert co.parse(str(got.value))["limit"] == 32768
