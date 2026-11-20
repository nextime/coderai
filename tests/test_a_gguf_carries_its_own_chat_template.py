# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""A locally-served GGUF's chat template is INSIDE the file.

``chat_template_text`` had two sources -- a model directory and an HF repo id --
and a ``.gguf`` is neither. So every locally-served GGUF answered "no template",
and :func:`thinks` fell through to matching the model's NAME. This catalogue
shows the cost: ``qwen38`` matches the name regex and ``redteam`` does not, so
the SAME Qwen3.8 template separated its thinking under one alias and streamed
raw ``<think>`` into ``content`` under the other -- which is what a client that
no longer parses inline tags (opencode since v1.1.46) renders as plain text.

The second half of this file guards the trap the first half opens. Making these
templates readable turns the reasoning gate ON for whatever they claim, and the
gate closes on a ``</think>``-family tag. gemma-4's template defines
``enable_thinking`` (defaulting it to false) but marks its thought
``<|think|>`` / ``<|channel>thought``. Reported as thinking, its close tag never
arrives, so every token files as reasoning and ``content`` comes back EMPTY --
answering 200 OK with nothing in it, the chandra-ocr-2 failure from the other
direction.
"""

from __future__ import annotations

import struct

import pytest


# --------------------------------------------------------------------------- #
# a minimal GGUF writer, so the test does not need a real model

def _s(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _kv_str(key: str, val: str) -> bytes:
    return _s(key) + struct.pack("<I", 8) + _s(val)


def _kv_u32(key: str, val: int) -> bytes:
    return _s(key) + struct.pack("<I", 4) + struct.pack("<I", val)


def _kv_str_array(key: str, vals) -> bytes:
    out = _s(key) + struct.pack("<I", 9) + struct.pack("<I", 8)
    out += struct.pack("<Q", len(vals))
    for v in vals:
        out += _s(v)
    return out


def _kv_f32_array(key: str, vals) -> bytes:
    out = _s(key) + struct.pack("<I", 9) + struct.pack("<I", 6)
    out += struct.pack("<Q", len(vals))
    for v in vals:
        out += struct.pack("<f", v)
    return out


def write_gguf(path, kvs: bytes, n_kv: int, magic: bytes = b"GGUF") -> str:
    with open(path, "wb") as f:
        f.write(magic)
        f.write(struct.pack("<I", 3))      # version
        f.write(struct.pack("<Q", 0))      # tensor_count
        f.write(struct.pack("<Q", n_kv))
        f.write(kvs)
    return str(path)


QWEN_TPL = (
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- if enable_thinking is defined and enable_thinking is false %}"
    "{{- '<think>\\n\\n</think>\\n\\n' }}"
    "{%- else %}{{- '<think>\\n' }}{%- endif %}{%- endif %}"
)

GEMMA4_TPL = (
    "{%- set enable_thinking = enable_thinking if enable_thinking is defined else false -%}"
    "{%- if enable_thinking -%}{{- '<|think|>\\n' -}}{%- endif -%}"
    "{%- if add_generation_prompt -%}{{- '<|turn>model\\n' -}}"
    "{%- if not enable_thinking -%}{{- '<|channel>thought\\n<channel|>' -}}{%- endif -%}"
    "{%- endif -%}"
)


@pytest.fixture(autouse=True)
def _clean_cache():
    from codai.models.chat_style import forget
    forget()
    yield
    forget()


# --------------------------------------------------------------------------- #
# the template is read out of the file

def test_the_template_is_read_from_the_gguf(tmp_path):
    from codai.models.chat_style import chat_template_text
    p = write_gguf(tmp_path / "m.gguf",
                   _kv_str("general.architecture", "qwen35")
                   + _kv_str("tokenizer.chat_template", QWEN_TPL), 2)
    assert chat_template_text(p) == QWEN_TPL


def test_a_gguf_that_thinks_is_reported_as_thinking(tmp_path):
    """THE BUG. False here leaves the reasoning gate off, and the proxy's own
    re-added <think> tags are streamed to the client as content."""
    from codai.models.chat_style import thinks
    p = write_gguf(tmp_path / "m.gguf",
                   _kv_str("tokenizer.chat_template", QWEN_TPL), 1)
    assert thinks(p) is True


def test_the_template_is_found_after_the_token_array(tmp_path):
    """The real key order: ``tokenizer.ggml.tokens`` holds ~250k strings and
    sits BEFORE the template, so a walker that cannot step over an array never
    reaches it."""
    from codai.models.chat_style import chat_template_text
    kvs = (_kv_str("general.architecture", "qwen35")
           + _kv_str_array("tokenizer.ggml.tokens", [f"t{i}" for i in range(5000)])
           + _kv_f32_array("tokenizer.ggml.scores", [0.5] * 5000)
           + _kv_u32("qwen35.block_count", 64)
           + _kv_str("tokenizer.chat_template", QWEN_TPL))
    p = write_gguf(tmp_path / "m.gguf", kvs, 5)
    assert chat_template_text(p) == QWEN_TPL


def test_a_gguf_without_a_template_is_not_an_error(tmp_path):
    from codai.models.chat_style import chat_template_text, thinks
    p = write_gguf(tmp_path / "m.gguf", _kv_str("general.architecture", "llama"), 1)
    assert chat_template_text(p) == ""
    assert thinks(p) is False


def test_a_file_that_is_not_a_gguf_is_not_an_error(tmp_path):
    from codai.models.chat_style import chat_template_text
    p = tmp_path / "notes.txt"
    p.write_text("not a model")
    assert chat_template_text(str(p)) == ""


def test_a_truncated_gguf_is_not_an_error(tmp_path):
    """A header that promises more keys than the file holds must not raise while
    a parser is being chosen."""
    from codai.models.chat_style import chat_template_text
    p = write_gguf(tmp_path / "m.gguf", _kv_str("general.architecture", "qwen35"), 9)
    assert chat_template_text(p) == ""


def test_an_unknown_value_type_stops_rather_than_guesses(tmp_path):
    """Past an unrecognised type the byte offsets mean nothing, so a reader that
    kept going would return whatever happened to decode as a string."""
    from codai.models.chat_style import chat_template_text
    kvs = (_s("weird.key") + struct.pack("<I", 99) + b"\x00" * 8
           + _kv_str("tokenizer.chat_template", QWEN_TPL))
    p = write_gguf(tmp_path / "m.gguf", kvs, 2)
    assert chat_template_text(p) == ""


# --------------------------------------------------------------------------- #
# a readable template must not arm a gate that cannot close

def test_a_switch_without_a_gateable_tag_is_not_thinking(tmp_path):
    """gemma-4: `enable_thinking` is defined (false by default) and the thought
    is marked `<|think|>`, which the streaming gate does not close on. True
    here files every token as reasoning and returns an empty answer."""
    from codai.models.chat_style import thinks
    p = write_gguf(tmp_path / "m.gguf",
                   _kv_str("tokenizer.chat_template", GEMMA4_TPL), 1)
    assert "enable_thinking" in GEMMA4_TPL
    assert thinks(p) is False


def test_a_bare_prefill_without_the_switch_is_thinking(tmp_path):
    """LFM2.5: the generation prompt pre-fills `<think>` and there is no
    switch, so the model emits a body and a bare `</think>`."""
    from codai.models.chat_style import thinks
    tpl = ("{%- if add_generation_prompt -%}"
           "{{- \"<|im_start|>assistant\\n<think>\" -}}{%- endif -%}")
    p = write_gguf(tmp_path / "m.gguf", _kv_str("tokenizer.chat_template", tpl), 1)
    assert thinks(p) is True


def test_a_preclosed_block_with_no_switch_is_not_thinking(tmp_path):
    """chandra-ocr-2: the block is shut before the model emits a token, so its
    output holds no close tag at all."""
    from codai.models.chat_style import thinks
    tpl = ("{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}"
           "{{- '<think>\\n\\n</think>\\n\\n' }}{%- endif %}")
    p = write_gguf(tmp_path / "m.gguf", _kv_str("tokenizer.chat_template", tpl), 1)
    assert thinks(p) is False


def test_every_claimed_tag_is_one_the_streaming_gate_closes_on():
    """The two lists are in different modules and must not drift: a tag that
    makes `thinks` say yes but that `_gate_reasoning` never matches is an
    answer that never gets emitted."""
    from codai.models.chat_style import _THOUGHT_TAGS
    from codai.api.text import _THINK_CLOSE_RE
    closers = [t for t in _THOUGHT_TAGS if t.startswith("</")]
    assert closers, "no closing form listed"
    for tag in closers:
        assert _THINK_CLOSE_RE.search(tag), f"{tag} is not one the gate closes on"


# --------------------------------------------------------------------------- #
# the dispatch order

def test_a_directory_still_wins_over_the_gguf_reader(tmp_path):
    """A model served from a directory keeps reading tokenizer_config.json --
    the GGUF path is an ADDITION, not a replacement."""
    from codai.models.chat_style import chat_template_text
    d = tmp_path / "model"
    d.mkdir()
    (d / "chat_template.jinja").write_text(QWEN_TPL)
    assert chat_template_text(str(d)) == QWEN_TPL


def test_a_repo_id_is_not_sent_to_the_gguf_reader(monkeypatch, tmp_path):
    """A name that is not a path on disk must still go to the repo lookup."""
    import codai.models.chat_style as cs
    seen = {}

    def _fake_repo(name):
        seen["name"] = name
        return QWEN_TPL

    monkeypatch.setattr(cs, "_from_repo", _fake_repo)
    assert cs.chat_template_text("Qwen/Qwen3-8B") == QWEN_TPL
    assert seen["name"] == "Qwen/Qwen3-8B"
