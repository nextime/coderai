# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""A page read in-process reported no measured confidence.

chandra and olmOCR each have three ways to reach their model: an HTTP server, a
rented pod, and coderai's own in-process chat handler. The first two ask for
per-token logprobs and turn them into a page confidence. The third returned
``conf=None`` with a comment saying the in-process handler could not surface
them.

It can. codai/api/text.py forwards ``request.logprobs`` to the backend (the
vLLM proxy supports it, ``get_last_logprobs``) and attaches what comes back to
``choices[0].logprobs`` -- the non-streaming chat path and the raw path both do.
Nothing asked.

And the in-process path is the one that runs where it matters: on an OCR pod
the pipeline and its vLLM share a process, so the dispatcher picks it for every
page. Hence Digesta seeing ``confidence_measured: false`` from chandra while
surya, whose own client talks to vLLM over HTTP, reported 0.99 for the same
pages.

A 0.0 that means "not measured" is indistinguishable from a 0.0 that means "the
model was certain it was unsure" -- see base.mean_token_prob -- so the flag
matters as much as the number.
"""

import json

import pytest

from codai.ocr import chandra as C
from codai.ocr import olmocr as O


MODULES = [pytest.param(C, id="chandra"), pytest.param(O, id="olmocr")]


# --------------------------------------------------------------------------- #
#  the reply is read the same way for its text and for its numbers
# --------------------------------------------------------------------------- #

class _Pydanticish:
    def __init__(self, data):
        self._data = data

    def model_dump(self):
        return self._data


class _JSONResponseish:
    def __init__(self, data):
        self.body = json.dumps(data).encode()


REPLY = {
    "choices": [{
        "message": {"content": "una pagina"},
        "logprobs": {"content": [{"logprob": -0.01}, {"logprob": -0.02}]},
    }],
    "usage": {"prompt_tokens": 7, "completion_tokens": 2},
}


@pytest.mark.parametrize("mod", MODULES)
@pytest.mark.parametrize("wrap", [dict, _Pydanticish, _JSONResponseish])
def test_every_reply_shape_yields_both_text_and_logprobs(mod, wrap):
    """The in-process handler returns a dict or a pydantic model; FastAPI hands
    back a JSONResponse. All three have to read the same."""
    result = wrap(REPLY) if wrap is not dict else REPLY
    assert mod._content_of(result) == "una pagina"
    from codai.ocr.base import logprobs_of
    assert logprobs_of(mod._as_dict(result)) == REPLY["choices"][0]["logprobs"]["content"]


@pytest.mark.parametrize("mod", MODULES)
def test_an_unreadable_reply_is_not_guessed_at(mod):
    assert mod._as_dict(object()) is None
    assert mod._content_of(object()) == ""


# --------------------------------------------------------------------------- #
#  the number itself
# --------------------------------------------------------------------------- #

def test_the_confidence_is_the_mean_token_probability():
    """The same computation surya's own client does, so the two numbers are
    comparable rather than two differently-defined "confidences"."""
    import math

    from codai.ocr.base import logprobs_of, mean_token_prob
    got = mean_token_prob(logprobs_of(REPLY))
    want = (math.exp(-0.01) + math.exp(-0.02)) / 2
    assert got == pytest.approx(want)


def test_a_reply_without_logprobs_is_not_measured():
    """None, never 0.0: the two are indistinguishable to a caller that gates on
    confidence, and that confusion has already produced one confidently wrong
    verdict in this subsystem."""
    from codai.ocr.base import logprobs_of, mean_token_prob
    assert mean_token_prob(logprobs_of({"choices": [{"message": {"content": "x"}}]})) is None


# --------------------------------------------------------------------------- #
#  wiring: asked for, and read back
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("mod,src", [
    pytest.param(C, "codai/ocr/chandra.py", id="chandra"),
    pytest.param(O, "codai/ocr/olmocr.py", id="olmocr"),
])
def test_the_in_process_request_asks_for_logprobs(mod, src):
    text = open(src).read()
    i = text.index("def _ask_local")
    body = text[i:text.index("\n    def ", i + 10)]
    assert 'extra["logprobs"] = True' in body, \
        "the in-process path has to ask, like the HTTP path does"
    # ...and before the request is built, or the flag never reaches it.
    assert body.index('extra["logprobs"] = True') < body.index("ChatCompletionRequest(")


def _code_of(src, func):
    """A method's EXECUTABLE lines: no docstring, no comments.

    These tests assert that a line no longer appears, and the comments explain
    what changed by quoting it -- so a check against the raw text matches the
    documentation and fails for the wrong reason.
    """
    text = open(src).read()
    i = text.index(f"def {func}")
    body = text[i:text.index("\n    def ", i + 10)]
    first = body.find('"""')
    if first != -1:
        end = body.find('"""', first + 3)
        body = body[:first] + (body[end + 3:] if end != -1 else "")
    return "\n".join(ln for ln in body.splitlines()
                     if not ln.strip().startswith("#"))


@pytest.mark.parametrize("mod,src", [
    pytest.param(C, "codai/ocr/chandra.py", id="chandra"),
    pytest.param(O, "codai/ocr/olmocr.py", id="olmocr"),
])
def test_the_in_process_path_no_longer_hardcodes_no_confidence(mod, src):
    body = _code_of(src, "_ask_local")
    assert "conf=None" not in body, "this is what reported it as unmeasured"
    assert "mean_token_prob(logprobs_of(_as_dict(result)))" in body


def test_the_handler_really_does_forward_and_return_them():
    """The claim this fix rests on: text.py passes request.logprobs to the
    backend and puts what comes back on the choice. If either half goes, the
    ask above becomes silent again."""
    text = open("codai/api/text.py").read()
    assert "logprobs=request.logprobs" in text
    assert "top_logprobs=request.top_logprobs" in text
    assert 'formatted_response["choices"][0]["logprobs"] = _lp' in text
    assert "get_last_logprobs()" in text
