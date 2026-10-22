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
    text = _from_dir(model_path) if os.path.isdir(model_path) else _from_repo(model_path)
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
