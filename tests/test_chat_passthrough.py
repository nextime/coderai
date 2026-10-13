"""A proxy that forwards only the fields it recognises decides, on the server's
behalf, which of the server's features exist.

The bug this covers, measured 2026-10-13 on a live install: coderai's OCR page
resumption sends `continue_final_message` and `add_generation_prompt: False` so
the model carries on the exact string it was cut off in. The vLLM backend built
its request from an allow-list, so neither flag was ever forwarded. What reached
vLLM was a conversation ending in an assistant turn with `add_generation_prompt`
left at its default TRUE -- and surya-ocr-2's chat template then appends a fresh
`<|im_start|>assistant` after the prefilled text, so the model started a NEW
answer about the same image instead of continuing.

It did not fail loudly. Every resumption produced a restart, the no-progress
guard in `_continue_reply` caught it as "came back repeating itself" and threw
it away, so resumption had never once added a correct continuation while
appearing to work.
"""
import sys
import types
import pytest


def _backend():
    """A VllmBackend instance with nothing loaded — only payload building."""
    from codai.backends.vllm import VllmBackend
    b = VllmBackend.__new__(VllmBackend)
    b._served_name = "m"
    b._model_id = "m"
    b._cfg = types.SimpleNamespace(return_token_ids=False)
    return b


MSGS = [{"role": "user", "content": "read it"},
        {"role": "assistant", "content": "<block>half a page"}]


def test_the_resumption_flags_reach_the_payload():
    p = _backend()._chat_payload(
        MSGS, 2048, 0.0, 0.1, None, False,
        passthrough={"continue_final_message": True,
                     "add_generation_prompt": False})
    assert p["continue_final_message"] is True
    # The one that matters most: left at its default the template appends a new
    # assistant turn and the model restarts the page.
    assert p["add_generation_prompt"] is False


def test_an_unknown_field_is_forwarded_rather_than_dropped():
    """Not an allow-list. A field coderai has never heard of is between the
    caller and the server — vLLM grows request fields faster than any list
    here can be updated."""
    p = _backend()._chat_payload(
        MSGS, 2048, 0.0, 0.1, None, False,
        passthrough={"chat_template_kwargs": {"enable_thinking": False},
                     "truncate_prompt_tokens": 4096,
                     "some_field_invented_next_year": "x"})
    assert p["chat_template_kwargs"] == {"enable_thinking": False}
    assert p["truncate_prompt_tokens"] == 4096
    assert p["some_field_invented_next_year"] == "x"


def test_no_passthrough_changes_nothing():
    a = _backend()._chat_payload(MSGS, 2048, 0.0, 0.1, None, False)
    b = _backend()._chat_payload(MSGS, 2048, 0.0, 0.1, None, False,
                                 passthrough=None)
    c = _backend()._chat_payload(MSGS, 2048, 0.0, 0.1, None, False,
                                 passthrough={})
    assert a == b == c


@pytest.mark.parametrize("field,value", [("model", "someone-elses-model"),
                                         ("messages", []),
                                         ("stream", True)])
def test_the_requests_structure_is_not_overridable(field, value):
    """`passthrough` carries sampling and template options, not the request's
    shape: retargeting the model or flipping streaming under the response
    parser is not a feature, and a caller must not reach them this way."""
    p = _backend()._chat_payload(MSGS, 2048, 0.0, 0.1, None, False,
                                 passthrough={field: value})
    assert p[field] != value or field == "messages" and p[field] is MSGS
    assert p["model"] == "m"
    assert p["messages"] is MSGS
    assert p["stream"] is False


# --- what the API layer decides to forward -----------------------------------


def _req(**extra):
    from codai.pydantic.textrequest import ChatCompletionRequest
    return ChatCompletionRequest(model="m", messages=[
        {"role": "user", "content": "hi"}], **extra)


def test_unknown_request_fields_become_the_passthrough():
    from codai.api.text import _chat_passthrough
    out = _chat_passthrough(_req(continue_final_message=True,
                                 add_generation_prompt=False))
    assert out == {"continue_final_message": True,
                   "add_generation_prompt": False}


def test_fields_coderai_acts_on_are_not_forwarded_twice():
    """By EXCLUSION: coderai names what it consumes, not what it forwards, so a
    field invented after this code was written is forwarded by default rather
    than born broken."""
    from codai.api.text import _chat_passthrough
    assert _chat_passthrough(_req(temperature=0.5, max_tokens=99, seed=7,
                                  logprobs=True)) is None


def test_a_plain_request_forwards_nothing():
    from codai.api.text import _chat_passthrough
    assert _chat_passthrough(_req()) is None


def test_an_uninspectable_request_is_still_served():
    from codai.api.text import _chat_passthrough
    assert _chat_passthrough(object()) is None


# --- the manager only hands it to a backend that can take it -----------------


def test_a_backend_without_the_parameter_is_not_broken_by_it():
    from codai.models.manager import ModelManager

    def narrow(messages, max_tokens=None, temperature=0.7, top_p=1.0,
               stop=None, tools=None, response_format=None):
        return ""

    assert ModelManager._passthrough_kwargs(narrow, {"a": 1}) == {}


def test_a_backend_that_takes_it_gets_it():
    from codai.models.manager import ModelManager

    def wide(messages, max_tokens=None, temperature=0.7, top_p=1.0, stop=None,
             tools=None, response_format=None, passthrough=None):
        return ""

    assert ModelManager._passthrough_kwargs(wide, {"a": 1}) == {"passthrough": {"a": 1}}
    # Nothing to forward means nothing is added, so the richest rung of the
    # signature ladder is not tried for no reason.
    assert ModelManager._passthrough_kwargs(wide, None) == {}
    assert ModelManager._passthrough_kwargs(wide, {}) == {}


def test_a_kwargs_backend_counts_as_accepting_it():
    from codai.models.manager import ModelManager

    def anything(messages, **kw):
        return ""

    assert ModelManager._passthrough_kwargs(anything, {"a": 1}) == {"passthrough": {"a": 1}}


# --- the server's own explanation must survive the trip up -------------------


def test_an_error_carries_what_the_server_said():
    """`requests` reports only "400 Client Error: Bad Request for url: ...".
    vLLM puts the reason in the BODY, and it was being discarded — a whole
    diagnosis cycle was spent guessing at a 400 that had already explained
    itself."""
    import requests
    from codai.backends.vllm import VllmBackend

    class _R:
        status_code = 400
        reason = "Bad Request"
        url = "http://127.0.0.1:53601/v1/chat/completions"

        @staticmethod
        def json():
            return {"message": "This model's maximum context length is 32768 "
                                "tokens. However, you requested 40864 tokens."}

    with pytest.raises(requests.HTTPError) as e:
        VllmBackend._raise_for_status(_R(), "chat/completions")
    assert "32768" in str(e.value) and "40864" in str(e.value)
    assert "chat/completions" in str(e.value)


def test_a_server_that_explains_nothing_says_so():
    import requests
    from codai.backends.vllm import VllmBackend

    class _R:
        status_code = 500
        reason = "Internal Server Error"
        url = "http://x/v1/chat/completions"
        text = ""

        @staticmethod
        def json():
            raise ValueError("not json")

    with pytest.raises(requests.HTTPError) as e:
        VllmBackend._raise_for_status(_R())
    assert "no explanation" in str(e.value)


def test_a_success_raises_nothing():
    from codai.backends.vllm import VllmBackend

    class _R:
        status_code = 200

    VllmBackend._raise_for_status(_R())
