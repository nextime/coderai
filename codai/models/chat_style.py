# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Which tool-call syntax a model speaks — read from the model, not its name.

A tool call is TEXT the model emits in the format it was trained on, and the
chat template is where that format is written down: the template is what turns
a tools array into the prompt, so it states, in the model's own words, what the
model will emit back. A name does not. Two models called "Qwen3" can speak
different syntaxes — Qwen2.5 and the Hermes finetunes put JSON inside
``<tool_call>``, current Qwen3 releases emit ``<function=``/``<parameter=`` XML
— and a parser chosen for the wrong one of those fails on every call it sees.

So the style is resolved here, once per model, and both parser choices read it:
vLLM's ``--tool-call-parser`` (codai.api.vllm_worker) and coderai's own response
parser (codai.models.parser.ModelParserDispatcher).

Precedence is the same in both places, strongest first:

1. an explicit per-model or global setting — an operator who has looked at the
   output is better evidence than anything inferred, and "off" must stay off;
2. this module's reading of the model's template;
3. the model name.

The name stays as the last resort because a template is not always reachable: a
GGUF file carries its own, a repo may be private, a pod may be offline. Not
knowing is survivable — a wrong guess that silently produces wrong tool
arguments is not — so an unreadable template changes nothing.
"""

import re
import json
import os
import threading
from typing import Optional

#: style id -> (vLLM --tool-call-parser, coderai parser class name)
STYLES = {
    "qwen_xml": ("qwen3_xml", "QwenParser"),
    "hermes_json": ("hermes", "QwenParser"),
    "glm_argkey": ("glm45", "GLMParser"),
    "mistral_tokens": ("mistral", "MistralParser"),
    "llama_pythontag": ("llama3_json", "LlamaParser"),
    "deepseek_tokens": ("deepseek_v3", "DeepSeekParser"),
}

#: style id -> the vLLM ``--reasoning-parser`` that separates this family's
#: thought from its answer. Separate from STYLES because the two questions are
#: separate: a model can emit tool calls and never think (llama_pythontag has
#: no entry here), and the reasoning parser is only passed when the template
#: says the model thinks at all.
REASONING_PARSERS = {
    "qwen_xml": "qwen3",
    "hermes_json": "qwen3",
    "glm_argkey": "glm45",
    "deepseek_tokens": "deepseek_r1",
    "mistral_tokens": "mistral",
}

_CACHE: dict = {}
_LOCK = threading.Lock()


def _read(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except Exception:
        return ""


def _from_dir(model_path: str) -> str:
    out = _read(os.path.join(model_path, "chat_template.jinja"))
    if out:
        return out
    try:
        tc = json.loads(_read(os.path.join(model_path, "tokenizer_config.json")) or "{}")
    except Exception:
        return ""
    return str(tc.get("chat_template") or "")


def _from_repo(model_path: str) -> str:
    # Nothing here may raise, and nothing may be answered from the cache alone:
    # on a rented pod nothing is cached the first time, and a cache-only lookup
    # would report "no template" for every model it is asked about.
    try:
        from codai.api.embeddings import _fetch_repo_file
        got = _fetch_repo_file(model_path, "chat_template.jinja")
        if got:
            return _read(got)
        got = _fetch_repo_file(model_path, "tokenizer_config.json")
        if not got:
            return ""
        return str(json.loads(_read(got) or "{}").get("chat_template") or "")
    except Exception as exc:
        print(f"[chat-style] could not read the chat template for "
              f"{model_path!r}: {type(exc).__name__}: {exc}", flush=True)
        return ""


#: GGUF metadata value types -> the fixed byte width of a scalar, for SKIPPING
#: a value we are not asked about. Strings (8) and arrays (9) are variable and
#: handled in the walker.
_GGUF_SCALAR_WIDTH = {0: 1, 1: 1, 7: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4,
                      10: 8, 11: 8, 12: 8}


def _from_gguf(model_path: str) -> str:
    """The chat template a GGUF carries in ``tokenizer.chat_template``.

    A GGUF is a FILE -- not a directory and not a repo id -- so neither
    :func:`_from_dir` nor :func:`_from_repo` can see inside one, and a local
    ``.gguf`` is the commonest way a model is served here. Left unread, every
    locally-served GGUF answered "no template", and :func:`thinks` then fell
    through to matching the model's NAME. The two spellings in this catalogue
    show what that costs: `qwen38` matches the name regex and `redteam` does
    not, so the same Qwen3.8 template split its thinking under one alias and
    streamed raw ``<think>`` into ``content`` under the other.

    Arrays are SEEKED past rather than decoded. ``tokenizer.ggml.tokens`` is a
    single array of ~250k strings in these files and it sits BEFORE the
    template, so materialising it to reach the key costs half a million reads
    per lookup; the walker only needs each element's length to step over it.

    Returns '' for anything that is not a readable GGUF, so a caller cannot
    tell "no template" from "could not be read" -- deliberate: both mean the
    detection has to fall back, and neither is an error worth raising while
    choosing a parser.
    """
    import struct
    try:
        with open(model_path, "rb") as f:
            if f.read(4) != b"GGUF":
                return ""
            f.seek(4 + 8, 1)                      # version, tensor_count
            n_kv = struct.unpack("<Q", f.read(8))[0]

            def _skip_str():
                n = struct.unpack("<Q", f.read(8))[0]
                f.seek(n, 1)

            def _read_str():
                n = struct.unpack("<Q", f.read(8))[0]
                return f.read(n).decode("utf-8", "replace")

            def _skip_val(vt):
                if vt == 8:
                    _skip_str()
                    return True
                if vt in _GGUF_SCALAR_WIDTH:
                    f.seek(_GGUF_SCALAR_WIDTH[vt], 1)
                    return True
                if vt == 9:
                    et = struct.unpack("<I", f.read(4))[0]
                    cnt = struct.unpack("<Q", f.read(8))[0]
                    if et == 8:
                        for _ in range(cnt):
                            _skip_str()
                    elif et in _GGUF_SCALAR_WIDTH:
                        f.seek(_GGUF_SCALAR_WIDTH[et] * cnt, 1)
                    elif et == 9:
                        return False       # nested arrays: stop, do not guess
                    else:
                        return False
                    return True
                return False               # unknown type: offsets are no longer trustworthy

            for _ in range(n_kv):
                key = _read_str()
                vt = struct.unpack("<I", f.read(4))[0]
                if key == "tokenizer.chat_template" and vt == 8:
                    return _read_str()
                if not _skip_val(vt):
                    return ""
    except Exception:
        return ""
    return ""


def chat_template_text(model_path: str) -> str:
    """The model's chat template, or '' when it cannot be read.

    Cached: a model's template cannot change under us, and this is consulted
    while choosing a parser — which happens per request on some paths.
    """
    model_path = (model_path or "").strip()
    if not model_path:
        return ""
    with _LOCK:
        if model_path in _CACHE:
            return _CACHE[model_path]
    # A GGUF is a file, so it is tried BEFORE the repo lookup -- which would
    # otherwise ask the Hub for a repo named by a filesystem path, fail, and
    # report that a model with an embedded template has none.
    if os.path.isdir(model_path):
        text = _from_dir(model_path)
    elif os.path.isfile(model_path):
        text = _from_gguf(model_path)
    else:
        text = _from_repo(model_path)
    with _LOCK:
        _CACHE[model_path] = text
    return text


def tool_call_style(model_path: str) -> str:
    """A key of :data:`STYLES` naming what this model emits, or ''.

    Ordered by how specific the marker is. ``<function=`` with ``<parameter=``
    is the Qwen XML form; a bare ``<tool_call>`` wrapping JSON is the Hermes
    form, and the two are told apart by which of those the template contains —
    not by which came first in the file.
    """
    tpl = chat_template_text(model_path)
    if not tpl:
        return ""
    if "<function=" in tpl and "<parameter=" in tpl:
        return "qwen_xml"
    if "<arg_key>" in tpl or "<arg_value>" in tpl:
        return "glm_argkey"
    if "[TOOL_CALLS]" in tpl:
        return "mistral_tokens"
    if "<|python_tag|>" in tpl:
        return "llama_pythontag"
    if "<｜tool▁calls▁begin｜>" in tpl or "<|tool_calls_begin|>" in tpl:
        return "deepseek_tokens"
    if "<tool_call>" in tpl:
        return "hermes_json"
    return ""


#: ``<think>`` closed again with nothing but whitespace between. The template
#: source is Jinja, so the newlines are the two-character escapes ``\n`` as
#: often as they are real ones.
_PRECLOSED_EMPTY_THINK = re.compile(r"<think>(?:\\[nrt]|\s)*</think>")


#: Thought markers whose CLOSING form the streaming reasoning gate in
#: codai.api.text._gate_reasoning splits on. A template whose thought is marked
#: some other way cannot be gated, so it must not be reported as thinking.
_THOUGHT_TAGS = ("<think>", "</think>", "<thinking>", "</thinking>",
                 "<thought>", "</thought>", "<|begin_of_thought|>")


def _preclosed_empty_think(tpl: str) -> bool:
    """Whether the GENERATION PROMPT shuts an empty think block.

    Scoped to the generation prompt because the same tag pair appears earlier in
    these templates for an unrelated reason -- re-rendering a past assistant
    turn that had reasoning of its own -- and that says nothing about what the
    model is about to emit. chandra-ocr-2 has three ``<think>`` in total and
    only the last one is the prefill.
    """
    i = tpl.rfind("add_generation_prompt")
    if i < 0:
        return False
    return bool(_PRECLOSED_EMPTY_THINK.search(tpl[i:]))


def thinks(model_path: str) -> bool:
    """True when this model emits a reasoning block of its own.

    Read from the template for the same reason the tool-call style is: a name
    does not say. Measured 2026-10-22 — `qwen38` and `diogene-alpha` are the
    SAME weights behind two aliases, and a name regex (`qwen3`) matched one and
    not the other, so one answered cleanly and the other sent its entire chain
    of thought to the client as the answer.

    A template that pre-fills the opening tag in its generation prompt --

        {%- if add_generation_prompt %}
            {{- '<|im_start|>assistant\\n' }}
            {{- '<think>\\n' }}

    -- makes the model emit the thought body and a BARE ``</think>``, with no
    opening tag anywhere in the output. That is the form that leaks when
    nothing is looking for it.

    A template that pre-fills a CLOSED and EMPTY one instead --

        {%- if add_generation_prompt %}
            {{- '<|im_start|>assistant\\n' }}
            {{- '<think>\\n\\n</think>\\n\\n' }}

    -- is saying the opposite: the block is shut before the model emits a
    token, so its output contains no ``</think>`` at all. Measured on
    datalab-to/chandra-ocr-2, an OCR fine-tune of a Qwen3.5 base whose template
    still mentions the tags. It was read as a thinking model, launched with
    ``--reasoning-parser qwen3``, and that parser -- which assumes the stream
    starts INSIDE the block and ends at the tag -- filed every page of
    transcription as ``reasoning_content`` and returned ``content`` empty. The
    pod answered 200 OK twenty-eight times and read nothing.

    ``enable_thinking`` is what separates the two, and it is checked first: a
    template carrying that switch pre-closes the block only in its
    thinking-disabled branch (Qwen3.5-9B does exactly this), so the model can
    still think when asked. Without the switch, a pre-closed block in the
    generation prompt is unconditional and the model never thinks.
    """
    tpl = chat_template_text(model_path)
    if not tpl:
        return False
    # A tag the reasoning gate can actually SPLIT ON has to be in the template,
    # and it is required before anything else is considered. The switch alone is
    # not enough: gemma-4's template defines `enable_thinking` (defaulting it to
    # false) and marks the thought with `<|think|>` / `<|channel>thought`, which
    # is not the `<think>` form _gate_reasoning closes on. Answering True there
    # arms a gate whose close tag never arrives, so EVERY token files as
    # reasoning and `content` comes back empty -- the chandra-ocr-2 failure
    # again, from the other direction.
    if not any(t in tpl for t in _THOUGHT_TAGS):
        return False
    if "enable_thinking" in tpl:
        # Switchable: thinking is the caller's choice, so the parser is needed.
        return True
    if _preclosed_empty_think(tpl):
        return False
    return True


def template_path_for_model(model_name: str) -> str:
    """Where to read ``model_name``'s template from, or ''.

    The base an adapter adapts (a LoRA has no template of its own), else the
    entry's own path, else the name — a repo id often enough to be worth
    trying. Resolved here so the tool-call style, the parser choice and the
    reasoning check all ask about the same file.
    """
    name = str(model_name or "").strip()
    if not name:
        return ""
    try:
        from codai.models.parser import _family_name
        base = _family_name(name)
    except Exception:
        base = name
    try:
        from codai.models.manager import _model_entry_for
        entry = _model_entry_for(base) or _model_entry_for(name) or {}
        block = entry.get("runpod") if isinstance(entry.get("runpod"), dict) else {}
        return (str(entry.get("path") or "").strip()
                or str(block.get("hf_repo") or "").strip()
                or base or name)
    except Exception:
        return base or name


def model_thinks(model_name: str) -> bool:
    """:func:`thinks`, for a model named the way a request names it."""
    return thinks(template_path_for_model(model_name))


def vllm_reasoning_parser(model_path: str) -> str:
    """The vLLM ``--reasoning-parser`` for this model, or '' for none.

    Why the SERVER must do this and not us: the closing ``</think>`` is a
    SPECIAL TOKEN. vLLM decodes with skip_special_tokens, so the tag never
    appears in the text at all -- measured 2026-10-22, a Qwen3.8 pod returned
    the thought and the answer concatenated with no separator of any kind, and
    no amount of parsing downstream can find a boundary that was deleted before
    the text was formed. Launched with a reasoning parser, vLLM splits the two
    at the TOKEN level and reports `reasoning_content` as its own field.

    Only for a model whose template says it thinks: a parser wrapped around a
    model that does not would be looking for a tag that is never coming.
    """
    if not thinks(model_path):
        return ""
    return REASONING_PARSERS.get(tool_call_style(model_path), "")


def vllm_tool_parser(model_path: str) -> str:
    """The vLLM ``--tool-call-parser`` this model's template calls for, or ''."""
    return STYLES.get(tool_call_style(model_path), ("", ""))[0]


def coderai_parser_name(model_path: str) -> str:
    """The coderai parser class this model's template calls for, or ''."""
    return STYLES.get(tool_call_style(model_path), ("", ""))[1]


def forget(model_path: Optional[str] = None) -> None:
    """Drop cached templates — one model's, or all of them."""
    with _LOCK:
        if model_path is None:
            _CACHE.clear()
        else:
            _CACHE.pop(model_path, None)
