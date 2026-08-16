# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Per-family prompt builders for the multi-family colibri engine (v1.5.0).

colibri v1.5.0 ships one C engine per model family, and each expects a DIFFERENT
serve payload even though they share the ``DATA``/``DONE`` mux framing (see the
project memory ``project_colibri_v15_families``):

* **GLM-5.2** (``colibri.c``) — a fully rendered XTML string. Lives in
  :mod:`codai.backends.colibri` (:func:`render_chat`), unchanged.
* **DeepSeek-V4** (``deepseek_v4.c``) — a fully rendered prompt string using the
  official DeepSeek markers. Built here by :func:`render_deepseek_chat`, byte-matching
  the engine's ``coli_v4_prompt_build`` for the system+user case.
* **Kimi-K3** (``kimi_k3.c``) — NOT a rendered string: a length-framed ``K3CHAT1``
  wire blob that the engine turns into its XTML template itself. Built here by
  :func:`build_kimi_wire`, byte-matching the engine's ``chat_build_wire`` parser
  (fixture ``c/tests/fixtures/kimi_chat_wire.txt``).

The family is chosen by the backend from the container's ``config.json``; this module
only owns the payload byte-format.
"""

import json
from typing import Dict, List, Optional

# --- DeepSeek-V4 markers (deepseek_v4.c: v4_bos/v4_user/v4_assistant) ----------- #
# NB: the bars are U+FF5C (｜) and U+2581 (▁), NOT ASCII '|' / '_'.
DS_BOS = "<｜begin▁of▁sentence｜>"
DS_EOS = "<｜end▁of▁sentence｜>"
DS_USER = "<｜User｜>"
DS_ASSISTANT = "<｜Assistant｜>"


def _content_text(content) -> str:
    """Flatten OpenAI message content (string or list of text parts) to a string.

    Kept in sync with :func:`codai.backends.colibri._content_text`.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in ("text", "input_text"):
                t = part.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "".join(parts)
    return str(content)


def _assistant_reasoning(message: Dict) -> str:
    """Return an assistant turn's separate reasoning text, if the client sent one.

    OpenAI-compatible clients surface chain-of-thought under a few keys; we accept the
    common ones. Empty string when the turn carries only visible content.
    """
    for key in ("reasoning_content", "reasoning"):
        v = message.get(key)
        if isinstance(v, str) and v:
            return v
    return ""


# --------------------------------------------------------------------------- #
# DeepSeek-V4 — rendered prompt string
# --------------------------------------------------------------------------- #
def render_deepseek_chat(messages: List[Dict], enable_thinking: bool = False) -> str:
    """Render the DeepSeek-V4 chat prompt colibri's ``deepseek_v4`` engine expects.

    For a single system+user turn this byte-matches the engine's own
    ``coli_v4_prompt_build`` (``{bos}{system}{User}{user}{Assistant}`` + ``</think>``
    for non-thinking or ``<think>`` for thinking). Multi-turn history follows the
    official DeepSeek template: assistant turns are closed with the end-of-sentence
    marker, and the trailing ``<｜Assistant｜>`` opens the turn to generate.
    """
    if not isinstance(messages, list) or not messages:
        raise ValueError("`messages` must be a non-empty array.")
    # System text is concatenated and placed right after BOS (matches the C helper's
    # single leading-system layout).
    system = "".join(_content_text(m.get("content")) for m in messages
                     if isinstance(m, dict) and m.get("role") in ("system", "developer"))
    out = [DS_BOS, system]
    for m in messages:
        if not isinstance(m, dict):
            raise ValueError("Each message must be an object.")
        role = m.get("role")
        if role in ("system", "developer"):
            continue
        if role == "user":
            out.append(DS_USER + _content_text(m.get("content")))
        elif role == "assistant":
            out.append(DS_ASSISTANT + _content_text(m.get("content")) + DS_EOS)
        elif role == "tool":
            # DeepSeek has no distinct tool turn in this template; fold tool output
            # into the user channel so the model still sees it.
            out.append(DS_USER + _content_text(m.get("content")))
        else:
            raise ValueError(f"Unsupported message role: {role!r}.")
    out.append(DS_ASSISTANT + ("<think>" if enable_thinking else "</think>"))
    return "".join(out)


# --------------------------------------------------------------------------- #
# Kimi-K3 — K3CHAT1 length-framed wire blob
# --------------------------------------------------------------------------- #
def _wire_msg(role: str, text: str) -> bytes:
    b = text.encode("utf-8")
    return (f"M {role} {len(b)}\n").encode("utf-8") + b


def _wire_assistant(reason: str, text: str) -> bytes:
    rb = reason.encode("utf-8")
    tb = text.encode("utf-8")
    return (f"A {len(rb)} {len(tb)}\n").encode("utf-8") + rb + tb


# --------------------------------------------------------------------------- #
# Kimi-K3 — rendered XTML string (for engines that tokenize a prompt string, e.g.
# the patched FareedKhan-dev/kimi-k3-in-c, as opposed to colibri's K3CHAT1 wire).
# Byte-reproduces colibri kimi_k3.c chat_build/chat_message/chat_assistant output.
# --------------------------------------------------------------------------- #
KIMI_OPEN, KIMI_CLOSE, KIMI_SEP, KIMI_EOM = "<|open|>", "<|close|>", "<|sep|>", "<|end_of_msg|>"


def _k_open(tag: str, role: Optional[str] = None) -> str:
    s = KIMI_OPEN + tag
    if role is not None:
        s += ' role="' + role + '"'
    return s + KIMI_SEP


def _k_close(tag: str) -> str:
    return KIMI_CLOSE + tag + KIMI_SEP


def render_kimi_xtml(messages: List[Dict], enable_thinking: bool = True) -> str:
    """Render the Kimi-K3 XTML chat prompt as a STRING (the engine tokenizes it).

    Mirrors colibri's ``kimi_k3.c`` builders: system/user/tool turns render as
    ``<|open|>message role="…"<|sep|>TEXT<|close|>message<|sep|><|end_of_msg|>``;
    assistant turns wrap the reply in ``<|open|>response<|sep|>…<|close|>response<|sep|>``
    (with a leading ``<|open|>think<|sep|>…<|close|>think<|sep|>`` when the turn carried
    reasoning); and the prompt ends by opening the assistant message plus its structural
    think/response channel to generate. The four XTML control tokens must exist in the
    engine's tokenizer.
    """
    if not isinstance(messages, list) or not messages:
        raise ValueError("`messages` must be a non-empty array.")
    out: List[str] = []
    for m in messages:
        if not isinstance(m, dict):
            raise ValueError("Each message must be an object.")
        role = m.get("role")
        text = _content_text(m.get("content"))
        if role in ("system", "developer"):
            out.append(_k_open("message", "system") + text + _k_close("message") + KIMI_EOM)
        elif role == "user":
            out.append(_k_open("message", "user") + text + _k_close("message") + KIMI_EOM)
        elif role == "tool":
            out.append(_k_open("message", "user") + text + _k_close("message") + KIMI_EOM)
        elif role == "assistant":
            reason = _assistant_reasoning(m)
            s = _k_open("message", "assistant")
            if reason:
                s += _k_open("think") + reason + _k_close("think")
            s += _k_open("response") + text + _k_close("response")
            s += _k_close("message") + KIMI_EOM
            out.append(s)
        else:
            raise ValueError(f"Unsupported message role: {role!r}.")
    out.append(_k_open("message", "assistant")
               + _k_open("think" if enable_thinking else "response"))
    return "".join(out)


def build_kimi_wire(messages: List[Dict], enable_thinking: bool = True) -> bytes:
    """Build the Kimi-K3 ``K3CHAT1`` wire payload for colibri's ``kimi_k3`` engine.

    Byte-matches the engine's ``chat_build_wire`` parser: a ``K3CHAT1`` header, then
    one framed directive per history message —

      * ``M <role> <nbytes>\\n<content>``   (system / user / assistant-without-reasoning;
        ``developer`` is normalised to ``system`` by the engine, but we send it as-is
        which the engine maps)
      * ``A <nreason> <ntext>\\n<reason><text>``  (assistant turn carrying reasoning)

    and a terminal ``G <0|1>\\n`` that sets the thinking channel and opens the
    assistant turn to generate. ``nbytes`` are exact UTF-8 byte counts.

    Returns raw ``bytes`` (the payload is length-framed, so the caller must send these
    bytes verbatim — do not re-encode).
    """
    if not isinstance(messages, list) or not messages:
        raise ValueError("`messages` must be a non-empty array.")
    out = [b"K3CHAT1\n"]
    for m in messages:
        if not isinstance(m, dict):
            raise ValueError("Each message must be an object.")
        role = m.get("role")
        if role in ("system", "developer", "user"):
            out.append(_wire_msg("system" if role == "developer" else role,
                                 _content_text(m.get("content"))))
        elif role == "assistant":
            reason = _assistant_reasoning(m)
            text = _content_text(m.get("content"))
            if reason:
                out.append(_wire_assistant(reason, text))
            else:
                out.append(_wire_msg("assistant", text))
        elif role == "tool":
            # No Kimi tool turn in the wire format; surface tool output as a user msg.
            out.append(_wire_msg("user", _content_text(m.get("content"))))
        else:
            raise ValueError(f"Unsupported message role: {role!r}.")
    out.append((f"G {1 if enable_thinking else 0}\n").encode("utf-8"))
    return b"".join(out)
