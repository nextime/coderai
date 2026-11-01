# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""A pre-closed think block means the model does NOT think.

datalab-to/chandra-ocr-2 is an OCR fine-tune of a Qwen3.5 base, and its chat
template still mentions ``<think>``. It was therefore read as a thinking model
and its pod launched with ``--reasoning-parser qwen3``. That parser assumes the
stream starts inside the block and ends at ``</think>``; chandra's template
pre-fills a CLOSED and EMPTY block in the generation prompt, so the model emits
no such tag and the parser filed every page of transcription as
``reasoning_content``, returning ``content`` empty.

The pod answered 200 OK twenty-eight times and read nothing: no error anywhere,
a healthy pod, and $3.34 burned. The model config had not changed -- the
``:latest`` pod image had moved to one carrying the reasoning auto-detection.
"""

import pytest

from codai.models import chat_style as C


# The shapes that matter, reduced to the lines the decision turns on.

CHANDRA = """\
{%- for message in messages %}
    {%- if message.role == 'assistant' %}
        {%- if '</think>' in content %}
            {%- set reasoning_content = content.split('</think>')[0] %}
        {%- endif %}
        {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
    {{- '<think>\\n\\n</think>\\n\\n' }}
{%- endif %}"""

QWEN_SWITCHABLE = """\
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\\n\\n</think>\\n\\n' }}
    {%- endif %}
{%- endif %}"""

OPEN_PREFILL = """\
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
    {{- '<think>\\n' }}
{%- endif %}"""

NO_THINK = """\
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
{%- endif %}"""


@pytest.fixture
def tpl(monkeypatch):
    """Feed a template straight to the detector."""
    def _use(text):
        monkeypatch.setattr(C, "chat_template_text", lambda _p: text)
        return "some/model"
    return _use


def test_an_unconditionally_preclosed_block_is_not_thinking(tpl):
    """chandra: the block is shut before the model emits a token, so there is
    no </think> coming and a parser aimed at one hides the whole answer."""
    assert C.thinks(tpl(CHANDRA)) is False


def test_a_switchable_template_still_thinks(tpl):
    """Qwen3.5-9B pre-closes the block too -- but only in its
    thinking-DISABLED branch, so the model can still think when asked. Reading
    the pre-closed pair without checking for the switch would silence every
    real thinking model, which is far worse than the bug being fixed."""
    assert C.thinks(tpl(QWEN_SWITCHABLE)) is True


def test_an_open_prefill_is_the_leaking_form_and_still_thinks(tpl):
    """The case the parser exists for: body then a BARE </think>."""
    assert C.thinks(tpl(OPEN_PREFILL)) is True


def test_a_template_with_no_tags_thinks_nothing(tpl):
    assert C.thinks(tpl(NO_THINK)) is False


def test_an_empty_template_is_not_guessed_at(tpl):
    assert C.thinks(tpl("")) is False


def test_a_past_assistant_turn_is_not_the_prefill(tpl):
    """Scoped to the generation prompt on purpose: these templates re-render a
    past assistant turn that HAD reasoning using the same tag pair, and that
    says nothing about what the model is about to emit. The real chandra
    template carries three `<think>`; this reduction keeps two, and in both
    only the LAST one is the prefill."""
    assert CHANDRA.count("<think>") >= 2
    assert C._preclosed_empty_think(CHANDRA) is True
    # The pair belonging to the replayed turn is before the generation prompt,
    # so on its own it must not decide anything.
    head = CHANDRA[:CHANDRA.rfind("add_generation_prompt")]
    assert "<think>" in head
    assert C._preclosed_empty_think(head) is False


@pytest.mark.parametrize("gap", ["", "\\n\\n", "\\n", "  ", "\\t", "\n\n"])
def test_whitespace_between_the_tags_is_still_empty(gap):
    """The template source is Jinja, so a newline is as often the two-character
    escape as a real one."""
    text = ("{%- if add_generation_prompt %}{{- '<think>" + gap
            + "</think>' }}{%- endif %}")
    assert C._preclosed_empty_think(text) is True


def test_content_between_the_tags_is_not_empty():
    """A block with a body in it is a template REPLAYING a thought, not
    suppressing one."""
    text = ("{%- if add_generation_prompt %}"
            "{{- '<think>\\n' + reasoning_content + '\\n</think>' }}{%- endif %}")
    assert C._preclosed_empty_think(text) is False


def test_no_reasoning_parser_is_passed_for_such_a_model(monkeypatch):
    """The whole point: thinks() False means vllm_reasoning_parser() returns
    nothing, so the pod launches without --reasoning-parser."""
    monkeypatch.setattr(C, "chat_template_text", lambda _p: CHANDRA)
    assert C.vllm_reasoning_parser("datalab-to/chandra-ocr-2") == ""
