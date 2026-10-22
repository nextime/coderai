"""`function.arguments` crosses an HTTP hop as a JSON STRING, not an object.

2026-10-21, qwen38 from the verti container, on the turn AFTER a tool call:

  the pod answered 500: 400 Bad Request from .../v1/chat/completions
  3 validation errors:
    {'type': 'string_type',
     'loc': ('body', 0, 'ChatCompletionMessageFunctionToolCallParam',
             'function', 'arguments'), ...}

coderai hands its backends a MAPPING on purpose -- GGUF chat templates render
arguments with `|items` and llama.cpp raises "Can only get item pairs from a
mapping" on a string. The OpenAI schema says string. Both are right for their
own side, so a proxy backend converts at the edge.

Only reachable once a client replays an assistant turn that HAS tool calls,
which is why it stayed hidden while those calls were being dropped (v0.4.60).
"""
import json
import types

import pytest

from codai.backends.base import messages_for_the_wire
from codai.backends.runpod import RunpodBackend
from codai.backends.vllm import VllmBackend


def _turn(arguments):
    return [
        {"role": "user", "content": "udienze?"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "agenda_udienze", "arguments": arguments}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "[]"},
    ]


def _args(messages):
    return messages[1]["tool_calls"][0]["function"]["arguments"]


def test_a_mapping_becomes_a_json_string():
    """THE BUG: this went out as an object and the server refused the body."""
    out = messages_for_the_wire(_turn({"dal": "2026-10-21", "al": "2026-11-21"}))
    assert _args(out) == '{"dal": "2026-10-21", "al": "2026-11-21"}'
    assert json.loads(_args(out))["dal"] == "2026-10-21"


def test_a_string_is_left_exactly_as_it_is():
    """Already on-the-wire shape: re-encoding would double-escape it."""
    out = messages_for_the_wire(_turn('{"dal": "2026-10-21"}'))
    assert _args(out) == '{"dal": "2026-10-21"}'


def test_no_arguments_becomes_an_empty_object():
    """Not the string "null", which a server reads as an argument object."""
    assert _args(messages_for_the_wire(_turn(None))) == "{}"


def test_the_callers_list_is_not_mutated():
    """generate_chat retries with the same list; a converted-in-place message
    would be converted twice."""
    msgs = _turn({"dal": "2026-10-21"})
    messages_for_the_wire(msgs)
    assert _args(msgs) == {"dal": "2026-10-21"}


def test_messages_without_tool_calls_pass_through():
    msgs = [{"role": "user", "content": "ciao"}]
    assert messages_for_the_wire(msgs) == msgs
    assert messages_for_the_wire([]) == []
    assert messages_for_the_wire(None) == []


def test_a_pydantic_tool_call_is_accepted():
    class _Fn:
        def model_dump(self):
            return {"id": "c1", "type": "function",
                    "function": {"name": "f", "arguments": {"a": 1}}}
    out = messages_for_the_wire([{"role": "assistant", "tool_calls": [_Fn()]}])
    assert out[0]["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'


def test_unserialisable_arguments_do_not_fail_the_request():
    out = messages_for_the_wire(_turn({"x": object()}))
    assert _args(out) == "{}"


# ------------------------------------------------- the payloads that go out #

@pytest.mark.parametrize("make", [
    lambda: (RunpodBackend.__new__(RunpodBackend), "runpod"),
    lambda: (VllmBackend.__new__(VllmBackend), "vllm"),
])
def test_both_proxy_payloads_carry_a_string(make):
    b, _name = make()
    b._model_id = b._served_name = "qwen38"
    payload = b._chat_payload(_turn({"dal": "2026-10-21"}), 100, 0.0, 1.0, None, False)
    args = payload["messages"][1]["tool_calls"][0]["function"]["arguments"]
    assert isinstance(args, str) and json.loads(args) == {"dal": "2026-10-21"}


def test_the_other_http_backends_too():
    """ds4 and ktransformers post the same OpenAI body to a local server."""
    from codai.backends.ds4 import Ds4Backend
    from codai.backends.ktransformers import KtransformersBackend
    for cls in (Ds4Backend, KtransformersBackend):
        b = cls.__new__(cls)
        b._model_id = b._served_name = "m"
        payload = b._chat_payload(_turn({"dal": "x"}), 100, 0.0, 1.0, None, False)
        args = payload["messages"][1]["tool_calls"][0]["function"]["arguments"]
        assert isinstance(args, str), cls.__name__


def test_a_conversation_with_nothing_to_convert_is_handed_back_as_is():
    """Identity, not a copy: every request goes through this, OCR's
    page-per-request traffic included, and tests assert the payload carries the
    caller's own list (tests/test_chat_passthrough.py)."""
    msgs = [{"role": "user", "content": "ciao"}]
    assert messages_for_the_wire(msgs) is msgs
    already = _turn('{"dal": "x"}')
    assert messages_for_the_wire(already) is already


def test_the_payload_keeps_the_callers_list_when_untouched():
    b = RunpodBackend.__new__(RunpodBackend)
    b._model_id = b._served_name = "qwen38"
    msgs = [{"role": "user", "content": "ciao"}]
    assert b._chat_payload(msgs, 100, 0.0, 1.0, None, False)["messages"] is msgs
