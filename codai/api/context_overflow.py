# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Reading a "does not fit the context window" refusal, and answering one.

Two jobs, both about the same two numbers -- the window, and what was asked of
it.

**Parsing.** Every engine words the refusal differently, and the numbers in it
are the only exact ones in the whole pipeline: coderai's own prompt size is a
chars/4 estimate, while the engine has tokenized the prompt. So when a request
is refused, the refusal is worth more than the estimate that let it through --
it says what the window really is (which can disagree with the configured
``ctx``) and exactly how far over the prompt went. :func:`parse` extracts both,
and :func:`remember_limit` keeps the window so the next request is compacted
against the served truth instead of the configured guess.

**Answering.** The numbers go FIRST in the message. This is not cosmetic. A
refusal crosses as many as three processes -- vLLM inside the pod, the pod's own
coderai, then this one -- and each wraps the one below it with its own prefix,
status line and URL. A client that reads a truncated prefix of the result (300,
400, 1000 characters: every client picks a cap, and the pod proxy URL alone can
be 120 characters) was getting the apologies and none of the figures, so it
could not tell an over-long conversation from any other 500 and had nothing to
retry with. Front-loading them means the first clause carries everything a
caller needs to shrink the request and try again:

    context_length_exceeded: 35000 tokens requested, limit 32768 (34000 prompt
    + 1000 completion) for model 'qwen38' -- ...

A caller matching on ``context_length_exceeded`` or reading the two integers
sees them whatever the cap, and the OpenAI-style ``code`` is the same string, so
matching on either works.
"""

from __future__ import annotations

import re
import threading
from typing import Dict, Optional

#: The OpenAI error code for this condition, and the token clients match on.
CODE = "context_length_exceeded"

#: Phrasings seen in the wild, each yielding what it knows. Order matters only
#: in that the first match wins, so the most specific patterns come first.
_PATTERNS = (
    # vLLM / OpenAI: the richest phrasing, and the one a pod answers with.
    re.compile(
        r"maximum context length is (?P<limit>\d+) tokens.{0,40}?"
        r"you requested (?P<requested>\d+) tokens?"
        r"(?:\D{0,40}?(?P<prompt>\d+) in the messages"
        r"\D{0,40}?(?P<completion>\d+) in the completion)?",
        re.IGNORECASE | re.DOTALL),
    # vLLM V1 engine, prompt-only phrasing.
    re.compile(
        r"(?:decoder )?prompt \(length (?P<prompt>\d+)\) is longer than the "
        r"maximum model length of (?P<limit>\d+)", re.IGNORECASE),
    # llama-cpp-python.
    re.compile(
        r"requested tokens \((?P<requested>\d+)\) exceed context window of "
        r"(?P<limit>\d+)", re.IGNORECASE),
    # coderai's own pre-flight refusal (see codai.api.text), so a relayed
    # message from a downstream coderai parses as well as an engine's.
    re.compile(
        r"(?P<requested>\d+) tokens requested, limit (?P<limit>\d+)",
        re.IGNORECASE),
    re.compile(
        r"~?(?P<prompt>\d+) prompt tokens vs n_ctx=(?P<limit>\d+)",
        re.IGNORECASE),
)

#: Text that means "it did not fit" even when no pattern yields numbers. Kept
#: deliberately broad: treating an overflow as an ordinary 500 costs a retry
#: that would have worked, while the reverse costs one pointless compaction.
_MARKERS = (
    "maximum context length", "context length", "context window",
    "context size", "available context", "exceed context", "exceeds n_ctx",
    "n_ctx", "maximum model length",
    "kv cache is full", "context shift is disabled", "too many tokens",
    "reduce the length of the messages", CODE,
)

#: ``{model: limit}`` learned from refusals. Process-local on purpose: it is a
#: correction to a configured number, not a configuration, and it must not
#: outlive a pod whose window really did change.
_LEARNED: Dict[str, int] = {}
_LOCK = threading.Lock()


def parse(text) -> Optional[dict]:
    """``{limit, requested, prompt, completion}`` from a refusal, or None.

    Only the keys the message actually stated are present, except that
    ``requested`` is derived from prompt+completion (or prompt alone) when the
    engine reported the parts but not the total -- a caller comparing "asked"
    against "allowed" should not have to know which phrasing it got.
    """
    s = str(text or "")
    if not s:
        return None
    for pat in _PATTERNS:
        m = pat.search(s)
        if not m:
            continue
        got = {}
        for key, raw in (m.groupdict() or {}).items():
            if raw is None:
                continue
            try:
                got[key] = int(raw)
            except (TypeError, ValueError):
                continue
        if "requested" not in got:
            if "prompt" in got:
                got["requested"] = got["prompt"] + got.get("completion", 0)
        if got.get("limit") or got.get("requested"):
            return got
    return None


def is_overflow(text) -> bool:
    """Whether this text is an engine refusing a request for being too long."""
    low = str(text or "").lower()
    if not low:
        return False
    if any(m in low for m in _MARKERS):
        return True
    # "exceeds the model's 32768 token limit" and similar: a token count and an
    # exceedance, which together are not said about anything else.
    return "token" in low and ("exceed" in low or "too long" in low)


def remember_limit(model: str, limit) -> Optional[int]:
    """Record the window an engine says it has, and return it.

    The served window and the configured one disagree in practice, and only the
    served one can refuse a request. Learning it from the refusal costs nothing
    -- no probe, no extra request -- and it is what makes a retry land inside
    the window rather than one estimate closer to it.
    """
    try:
        n = int(limit or 0)
    except (TypeError, ValueError):
        return None
    name = str(model or "").strip()
    if not name or n <= 0:
        return None
    with _LOCK:
        _LEARNED[name] = n
    return n


def learned_limit(model: str) -> Optional[int]:
    with _LOCK:
        return _LEARNED.get(str(model or "").strip())


def learn_from(model: str, text) -> Optional[dict]:
    """Parse a refusal and remember the limit it states. Returns the parse."""
    info = parse(text)
    if info and info.get("limit"):
        remember_limit(model, info["limit"])
    return info


def forget(model: Optional[str] = None) -> None:
    with _LOCK:
        if model is None:
            _LEARNED.clear()
        else:
            _LEARNED.pop(str(model or "").strip(), None)


def message(info: Optional[dict] = None, model: str = "",
            compacted: bool = False, raw: str = "") -> str:
    """The client-facing refusal, numbers first. See the module docstring.

    ``raw`` (the engine's own words) is appended rather than replaced: it can
    name a cause the numbers do not, and by then it is past the figures that a
    truncating client needs.
    """
    info = info or {}
    limit = info.get("limit")
    requested = info.get("requested")
    bits = []
    if requested and limit:
        bits.append(f"{requested} tokens requested, limit {limit}")
    elif limit:
        bits.append(f"over the {limit}-token limit")
    elif requested:
        bits.append(f"{requested} tokens requested")
    else:
        bits.append("the request is longer than the context window")
    parts = []
    if info.get("prompt"):
        parts.append(f"{info['prompt']} prompt")
    if info.get("completion"):
        parts.append(f"{info['completion']} completion")
    if parts:
        bits.append("(" + " + ".join(parts) + ")")
    if model:
        bits.append(f"for model '{model}'")
    head = f"{CODE}: " + " ".join(bits)
    tail = ("even after auto-compaction. Shorten the conversation, lower "
            "max_tokens, or raise the model's context size."
            if compacted else
            "The conversation is too long for this model's context window. "
            "Shorten it, lower max_tokens, enable auto-compaction for this "
            "model, or raise its context size.")
    out = f"{head} — {tail}"
    raw = str(raw or "").strip()
    if raw and raw not in out:
        out += f" (engine said: {raw})"
    return out
