"""A pod answers with a tool call; the client must not get an empty message.

2026-10-21: every tool-using request to a pod-served model came back as
`content: ""`, `finish_reason: "stop"`, no tool_calls -- an empty response --
while the pod had generated a perfectly good call. Asked directly, the pod
answered `{"content": null, "tool_calls": [...]}`; through the orchestrator the
same request answered nothing.

Cause: a proxy backend's `generate_chat` returns a STRING. The remote owns the
chat template, so it is the side that recognises the call and reports it
structurally -- and `message.content or ""` threw it away. The caller then had
an empty string, found no call to parse in it, and answered the client with an
empty assistant message.

The calls now travel beside the text, like usage and logprobs already did.
"""
import json
import types

import pytest

from codai.backends.base import ModelBackend, merge_tool_call_deltas
from codai.backends.runpod import RunpodBackend
from codai.backends.vllm import VllmBackend


CALL = [{"id": "call_1", "type": "function",
         "function": {"name": "agenda_udienze",
                      "arguments": '{"dal": "2026-10-21", "al": "2026-11-21"}'}}]


def _answer(message, usage=None):
    """A remote's non-streaming chat answer."""
    r = types.SimpleNamespace()
    r.status_code = 200
    r.ok = True
    r.json = lambda: {"choices": [{"message": message, "finish_reason": "tool_calls"}],
                      "usage": usage or {"prompt_tokens": 7, "completion_tokens": 9}}
    r.text = json.dumps(message)
    r.headers = {}
    return r


# ------------------------------------------------------------ the pod's answer #

def _runpod(monkeypatch, message):
    b = RunpodBackend.__new__(RunpodBackend)
    b._model_id = b._served_name = "diogene-alpha"
    b._ctx = 32768
    b._last_usage = {}
    b._last_logprobs = None
    b._last_tool_calls = None
    b._url = "https://pod:1/v1"
    b._headers = {}
    b._pool = None
    b._mode = "serverless"
    monkeypatch.setattr(b, "_acquire_endpoint",
                        lambda: ("https://pod:1/v1", lambda: None, ""))
    monkeypatch.setattr("codai.api.pod_http.post",
                        lambda *a, **k: _answer(message))
    return b


def test_a_pods_tool_call_is_not_lost(monkeypatch):
    """THE BUG: content is null, so the string contract carried nothing."""
    b = _runpod(monkeypatch, {"role": "assistant", "content": None, "tool_calls": CALL})
    assert b.generate_chat([{"role": "user", "content": "hi"}]) == ""
    assert b.get_last_tool_calls() == CALL


def test_text_answers_are_unchanged(monkeypatch):
    b = _runpod(monkeypatch, {"role": "assistant", "content": "Buongiorno"})
    assert b.generate_chat([{"role": "user", "content": "hi"}]) == "Buongiorno"
    assert b.get_last_tool_calls() is None


def test_a_later_text_answer_does_not_inherit_the_call(monkeypatch):
    """The side channel is per-call, like get_last_usage: a stale tool call
    attached to the next request would answer a question nobody asked."""
    b = _runpod(monkeypatch, {"role": "assistant", "content": None, "tool_calls": CALL})
    b.generate_chat([{"role": "user", "content": "hi"}])
    monkeypatch.setattr("codai.api.pod_http.post",
                        lambda *a, **k: _answer({"role": "assistant", "content": "ciao"}))
    assert b.generate_chat([{"role": "user", "content": "hi"}]) == "ciao"
    assert b.get_last_tool_calls() is None


def test_the_same_holds_for_a_local_vllm_server(monkeypatch):
    b = VllmBackend.__new__(VllmBackend)
    b._model_id = "qwen38"
    b._ctx = 32768
    b._last_usage = {}
    b._last_logprobs = None
    b._last_tool_calls = None
    b._url = "http://127.0.0.1:8111"
    monkeypatch.setattr(b, "_raise_for_status", lambda *a, **k: None)
    monkeypatch.setattr("requests.post",
                        lambda *a, **k: _answer({"role": "assistant", "content": None,
                                                 "tool_calls": CALL}))
    assert b.generate_chat([{"role": "user", "content": "hi"}]) == ""
    assert b.get_last_tool_calls() == CALL


def test_a_generating_backend_reports_none():
    """A local backend keeps the call inside its text, where the parser finds
    it; answering None leaves that path exactly as it was."""
    assert ModelBackend.get_last_tool_calls(object()) is None


# -------------------------------------------------------- streamed fragments #

def test_fragments_are_assembled():
    acc = None
    acc = merge_tool_call_deltas(acc, [{"index": 0, "id": "call_1", "type": "function",
                                        "function": {"name": "agenda_udienze",
                                                     "arguments": '{"dal": '}}])
    acc = merge_tool_call_deltas(acc, [{"index": 0, "function": {"arguments": '"2026-10-21"}'}}])
    assert acc == [{"id": "call_1", "type": "function",
                    "function": {"name": "agenda_udienze",
                                 "arguments": '{"dal": "2026-10-21"}'}}]
    assert json.loads(acc[0]["function"]["arguments"])["dal"] == "2026-10-21"


def test_two_interleaved_calls_keep_their_own_arguments():
    """Keyed by index, never by arrival order: concatenating the arguments of
    the wrong call yields valid JSON for a call nobody made."""
    acc = None
    for frag in ([{"index": 0, "id": "a", "function": {"name": "one", "arguments": '{"x":'}}],
                 [{"index": 1, "id": "b", "function": {"name": "two", "arguments": '{"y":'}}],
                 [{"index": 0, "function": {"arguments": "1}"}}],
                 [{"index": 1, "function": {"arguments": "2}"}}]):
        acc = merge_tool_call_deltas(acc, frag)
    assert [c["function"]["name"] for c in acc] == ["one", "two"]
    assert acc[0]["function"]["arguments"] == '{"x":1}'
    assert acc[1]["function"]["arguments"] == '{"y":2}'


def test_a_server_that_omits_the_index_still_works():
    acc = merge_tool_call_deltas(None, [{"id": "a", "function": {"name": "one",
                                                                 "arguments": "{}"}}])
    assert acc[0]["function"]["name"] == "one"


def test_nothing_in_nothing_out():
    assert merge_tool_call_deltas(None, None) is None
    assert merge_tool_call_deltas(None, []) is None
    assert merge_tool_call_deltas(CALL, None) == CALL


def test_a_streamed_delta_is_collected_off_the_wire():
    """The generator yields text; a streamed call has none, so it has to be
    picked up from the delta itself."""
    b = RunpodBackend.__new__(RunpodBackend)
    b._last_tool_calls = None
    b._last_usage = {}
    b._last_logprobs = None
    chunk = {"choices": [{"delta": {"tool_calls": [
        {"index": 0, "id": "call_1", "type": "function",
         "function": {"name": "agenda_udienze", "arguments": "{}"}}]}}]}
    delta = (chunk["choices"][0].get("delta") or {})
    b._last_tool_calls = merge_tool_call_deltas(b._last_tool_calls, delta.get("tool_calls"))
    assert b.get_last_tool_calls()[0]["function"]["name"] == "agenda_udienze"


# ------------------------------------------------------------ the manager hop #

def test_the_manager_forwards_them():
    from codai.models.manager import ModelManager
    m = ModelManager.__new__(ModelManager)
    m.backend = types.SimpleNamespace(get_last_tool_calls=lambda: CALL)
    assert m.get_last_tool_calls() == CALL


def test_the_manager_survives_a_backend_without_the_method():
    from codai.models.manager import ModelManager
    m = ModelManager.__new__(ModelManager)
    m.backend = types.SimpleNamespace()
    assert m.get_last_tool_calls() is None


# -------------------------------------------------- teaching the RIGHT pod #

def test_teaching_a_pod_uses_that_pods_token(monkeypatch):
    """Same class as the v0.4.59 401: pods in one pool carry different tokens,
    so the pool's own key would 401 against a pod that does not share it -- and
    the request is then lost to the very 404 teaching was meant to cure."""
    import types as _t
    b = RunpodBackend.__new__(RunpodBackend)
    b._model_id = b._served_name = "diogene-alpha"
    b._pool = _t.SimpleNamespace(api_key="pool-key")
    seen = {}
    monkeypatch.setattr("codai.api.remote_gateway.teach_pod",
                        lambda root, key, name: seen.update(key=key, name=name) or True)
    assert b._teach("https://pod:1", "its-own-key") is True
    assert seen == {"key": "its-own-key", "name": "diogene-alpha"}


def test_without_a_pod_token_the_pools_key_is_used(monkeypatch):
    import types as _t
    b = RunpodBackend.__new__(RunpodBackend)
    b._model_id = b._served_name = "diogene-alpha"
    b._pool = _t.SimpleNamespace(api_key="pool-key")
    seen = {}
    monkeypatch.setattr("codai.api.remote_gateway.teach_pod",
                        lambda root, key, name: seen.update(key=key) or True)
    b._teach("https://pod:1")
    assert seen["key"] == "pool-key"
