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

"""
Text generation endpoints for the codai API.
"""

import asyncio
import json
import logging
import time
import uuid
from typing import AsyncGenerator, Dict, List, Optional

# Engine debug/log lines go out via a single os.write(1) syscall instead of the
# buffered, lock-guarded builtin print() — lower overhead on the generation hot
# path. Shadows print() for this whole module.
from codai.api.fastlog import fast_print as print  # noqa: A004

from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)

# Import from codai modules
from codai.models.manager import ModelManager, WhisperServerManager, MultiModelManager, model_manager, multi_model_manager
from codai.queue.manager import QueueManager, queue_manager
from codai.tasks import task_registry
from codai.api.prompt_cache import prompt_cache_manager
from codai.pydantic.textrequest import ChatCompletionRequest, ToolFunction, Tool
from codai.models.parser import filter_malformed_content, filter_repetition, format_tools_for_prompt, cleanup_control_tokens, OpenAIFormatter, ModelParserAdapter, ToolCallParser

# Import global state from state module
from codai.api.state import (
    set_global_args as _set_global_args,
    get_global_debug,
    set_global_debug as _set_global_debug,
    get_global_system_prompt,
    set_global_system_prompt as _set_global_system_prompt,
    get_global_tools_closer_prompt,
    set_global_tools_closer_prompt as _set_global_tools_closer_prompt,
    get_grammar_guided_gen,
    set_grammar_guided_gen as _set_grammar_guided_gen,
)

# Global reference to be set by coderai
global_args = None


# =============================================================================
# Helper Functions
# =============================================================================

def set_global_args(args):
    """Set global args from coderai."""
    global global_args
    global_args = args
    # Also set in the state module so other modules can access it
    _set_global_args(args)


def set_global_debug(debug: bool):
    """Set the global debug flag (via state module)."""
    _set_global_debug(debug)


def set_global_system_prompt(prompt):
    """Set the global system prompt (via state module)."""
    _set_global_system_prompt(prompt)


def set_global_tools_closer_prompt(tools_closer: bool):
    """Set the global tools-closer-prompt flag (via state module)."""
    _set_global_tools_closer_prompt(tools_closer)


def _conversation_session_key(request, http_request=None) -> Optional[str]:
    """Derive a stable per-conversation key for instance/KV-cache affinity.

    Prefers an explicit identifier the client supplies (the OpenAI ``user`` field
    or an ``X-Session-Id`` header). Otherwise falls back to a hash of the stable
    opening of the conversation (system prompt + first user turn for chat, or the
    prompt head for completions) — stable across the turns of one conversation,
    distinct between conversations. Returns None if nothing usable is available
    (callers then fall back to least-busy routing). Never raises.
    """
    try:
        # 1) Explicit id wins.
        if http_request is not None:
            sid = http_request.headers.get('x-session-id')
            if sid:
                return f"sid:{sid}"
        uid = getattr(request, 'user', None)
        if uid:
            return f"user:{uid}"

        # 2) Hash the stable opening of the conversation.
        import hashlib
        parts = []
        msgs = getattr(request, 'messages', None)
        if msgs:
            first_user_seen = False
            for m in msgs:
                role = getattr(m, 'role', None) or (m.get('role') if isinstance(m, dict) else None)
                content = getattr(m, 'content', None) or (m.get('content') if isinstance(m, dict) else None)
                if not isinstance(content, str):
                    content = str(content)
                if role == 'system':
                    parts.append(f"system:{content}")
                elif role == 'user' and not first_user_seen:
                    parts.append(f"user:{content}")
                    first_user_seen = True
                    break
        else:
            prompt = getattr(request, 'prompt', None)
            if isinstance(prompt, list):
                prompt = prompt[0] if prompt else ''
            if prompt:
                parts.append(str(prompt)[:1024])
        if not parts:
            return None
        digest = hashlib.sha256("\n".join(parts).encode('utf-8', 'ignore')).hexdigest()[:16]
        return f"hash:{digest}"
    except Exception:
        return None


def set_grammar_guided_gen(enabled: bool):
    """Set the grammar-guided generation flag (via state module)."""
    _set_grammar_guided_gen(enabled)


def _debug_requests_enabled() -> bool:
    """True when --debug-requests is set (full client<->API payload logging)."""
    return bool(getattr(global_args, 'debug_requests', False)) if global_args else False


def _clip_for_log(s, limit: int = 4000) -> str:
    """Bound a string for debug printing so a huge (e.g. runaway/prompt-echoing)
    generation can't be dumped in full.

    The engine writes stdout to a PIPE drained by the front; a multi-megabyte
    synchronous print fills the pipe and BLOCKS the print() call. These debug
    dumps run on the event-loop thread (inside the streaming generator), so a
    blocked print freezes the whole engine and the front's health poll flips it
    to 'not responding'. Keeping the head and tail bounds the write while still
    showing both ends of the output."""
    try:
        s = s if isinstance(s, str) else str(s)
    except Exception:
        return "<unprintable>"
    if len(s) <= limit:
        return s
    head = limit * 3 // 4
    tail = limit - head
    return f"{s[:head]}\n… [clipped {len(s) - limit} chars] …\n{s[-tail:]}"


class _ToolCallStreamGate:
    """Hold back streamed content once a tool-call marker appears, so a model's
    tool call (e.g. gemma's ``<|tool_call>call:NAME{…}``) isn't leaked to the client
    as assistant message content before it's parsed. The FULL text is still
    accumulated by the caller for end-of-stream tool extraction; this only decides
    what is safe to emit as a visible content delta.

    Markers cover gemma's special tokens plus the common tag formats. ``feed()``
    returns the text safe to emit for each chunk: everything up to the first marker,
    minus a small tail that could be a marker split across chunk boundaries. After a
    marker is seen, it emits nothing. ``flush()`` releases any held-back tail when no
    marker ever appeared."""
    MARKERS = ("<|tool_call>", "<|tool_response>", "<|tool_call|>", "<tool_call>",
               "<tool>", "<function=", "<|tool|>", "<|function_call>")
    _MAXLEN = max(len(m) for m in MARKERS)

    def __init__(self):
        self.buf = ""
        self.emitted = 0
        self.started = False

    def feed(self, chunk: str) -> str:
        self.buf += chunk
        if self.started:
            return ""
        earliest = None
        for m in self.MARKERS:
            i = self.buf.find(m, self.emitted)
            if i != -1 and (earliest is None or i < earliest):
                earliest = i
        if earliest is not None:
            out = self.buf[self.emitted:earliest]
            self.emitted = earliest
            self.started = True
            return out
        # No full marker yet: hold back a trailing run that could be the start of a
        # marker split across chunks (e.g. a chunk ending in "<|tool_c").
        safe = len(self.buf)
        maxtail = min(self._MAXLEN - 1, len(self.buf) - self.emitted)
        for k in range(maxtail, 0, -1):
            suffix = self.buf[-k:]
            if any(m.startswith(suffix) for m in self.MARKERS):
                safe = len(self.buf) - k
                break
        out = self.buf[self.emitted:safe]
        self.emitted = safe
        return out

    def flush(self) -> str:
        if self.started:
            return ""
        out = self.buf[self.emitted:]
        self.emitted = len(self.buf)
        return out


def _summarize_tool_calls(tool_calls):
    """Compact one-line-per-call view of OpenAI tool_calls (dict or pydantic)."""
    out = []
    for tc in (tool_calls or []):
        fn = (tc.get('function') if isinstance(tc, dict) else getattr(tc, 'function', None)) or {}
        name = fn.get('name', '') if isinstance(fn, dict) else getattr(fn, 'name', '')
        args = fn.get('arguments', '') if isinstance(fn, dict) else getattr(fn, 'arguments', '')
        if not isinstance(args, str):
            try:
                args = json.dumps(args)
            except Exception:
                args = str(args)
        out.append(f"{name}({args})")
    return out


def log_request_exchange(request):
    """Dump the incoming chat request (messages + tools) when --debug-requests.

    Shows exactly what an agentic client (opencode, etc.) sends each turn —
    including whether it replays prior assistant tool_calls and `role:tool`
    results — so tool-call loops can be diagnosed from the wire, not guesswork."""
    if not _debug_requests_enabled():
        return
    try:
        print(f"\n{'#'*80}\n# >>> REQUEST  model={getattr(request, 'model', '?')}  "
              f"stream={getattr(request, 'stream', False)}  "
              f"tools={len(getattr(request, 'tools', None) or [])}\n{'#'*80}")
        for i, m in enumerate(getattr(request, 'messages', []) or []):
            role = getattr(m, 'role', '?')
            content = getattr(m, 'content', '') or ''
            if isinstance(content, list):
                content = json.dumps(content)
            line = f"[{i}] {role}: {str(content)[:2000]}"
            tcs = getattr(m, 'tool_calls', None)
            if tcs:
                line += "  tool_calls=" + json.dumps(_summarize_tool_calls(tcs))
            tcid = getattr(m, 'tool_call_id', None)
            if tcid:
                line += f"  tool_call_id={tcid}"
            name = getattr(m, 'name', None)
            if name:
                line += f"  name={name}"
            print(line)
        tools = getattr(request, 'tools', None) or []
        if tools:
            names = []
            for t in tools:
                fn = t.get('function', {}) if isinstance(t, dict) else getattr(t, 'function', None)
                names.append((fn.get('name') if isinstance(fn, dict) else getattr(fn, 'name', '?')))
            print(f"# tools offered: {names}")
        print(f"{'#'*80}\n", flush=True)
    except Exception as e:
        print(f"[debug-requests] failed to log request: {e}", flush=True)


def log_response_exchange(content, tool_calls=None, finish_reason=None,
                          streamed=False, stage="pre-format"):
    """Dump the assistant message coderai *extracted* (content + tool_calls) when
    --debug-requests. This is the model's decision **before** the OpenAI formatter
    runs — pair it with :func:`log_response_payload` to see what the client gets."""
    if not _debug_requests_enabled():
        return
    try:
        tag = "STREAM" if streamed else "RESPONSE"
        print(f"\n{'#'*80}\n# <<< {tag} [{stage}]  finish_reason={finish_reason}\n{'#'*80}")
        if content:
            print(f"content: {str(content)[:2000]}")
        if tool_calls:
            for c in _summarize_tool_calls(tool_calls):
                print(f"tool_call: {c}")
        if not content and not tool_calls:
            print("(empty)")
        print(f"{'#'*80}\n", flush=True)
    except Exception as e:
        print(f"[debug-requests] failed to log response: {e}", flush=True)


def log_response_payload(payload, streamed=False):
    """Dump the exact payload the client receives (post OpenAI-formatter) when
    --debug-requests — the SSE chunk dict for streaming or the full JSON body for
    non-streaming. This is the ground truth of what opencode actually parses, so a
    formatter that rewrites/drops tool_calls or content is caught here."""
    if not _debug_requests_enabled():
        return
    try:
        tag = "STREAM CHUNK" if streamed else "RESPONSE BODY"
        print(f"\n{'#'*80}\n# <<< {tag} [post-format, sent to client]\n{'#'*80}")
        print(json.dumps(payload, indent=2, default=str)[:4000])
        print(f"{'#'*80}\n", flush=True)
    except Exception as e:
        print(f"[debug-requests] failed to log payload: {e}", flush=True)


# =============================================================================
# Router and Endpoints
# =============================================================================

router = APIRouter()


def _normalize_vision_content(content: list) -> list:
    """Normalize an OpenAI multipart message content list to the shape the
    llama.cpp multimodal (mmproj) handler expects: text parts as
    ``{"type":"text","text":...}`` and images as
    ``{"type":"image_url","image_url":{"url": ...}}``. The url may be an http(s)
    link or a ``data:image/...;base64,...`` URI — both are accepted. Unknown
    parts are dropped to a text placeholder so nothing crashes the handler."""
    norm = []
    for item in content:
        if not isinstance(item, dict):
            norm.append({"type": "text", "text": str(item)})
            continue
        t = item.get("type")
        if t == "text" and "text" in item:
            norm.append({"type": "text", "text": item["text"]})
        elif t in ("image_url", "input_image"):
            iu = item.get("image_url") if t == "image_url" else item.get("image")
            url = iu.get("url") if isinstance(iu, dict) else iu
            if url:
                norm.append({"type": "image_url", "image_url": {"url": url}})
        elif "text" in item:
            norm.append({"type": "text", "text": str(item["text"])})
        else:
            norm.append({"type": "text", "text": f"[{t or 'unknown'} content]"})
    return norm


def _normalize_tool_call_arguments(tool_calls):
    """Return tool_calls with each ``function.arguments`` as a dict (mapping)
    rather than a JSON string. OpenAI/Kilo send arguments as a JSON STRING, but
    several GGUF chat templates (e.g. Qwen) render them with ``arguments|items``,
    which requires a mapping — otherwise llama.cpp raises "Can only get item pairs
    from a mapping" while applying the template. A dict also serializes correctly
    for templates that use ``arguments|tojson``, so this is safe either way."""
    out = []
    for tc in (tool_calls or []):
        if hasattr(tc, "model_dump"):
            tc = tc.model_dump()
        if not isinstance(tc, dict):
            out.append(tc)
            continue
        tc = dict(tc)
        fn = tc.get("function")
        if isinstance(fn, dict) and isinstance(fn.get("arguments"), str):
            try:
                parsed = json.loads(fn["arguments"] or "{}")
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                fn = dict(fn)
                fn["arguments"] = parsed
                tc["function"] = fn
        out.append(tc)
    return out


def _estimate_tokens(messages) -> int:
    """Cheap prompt-size estimate (≈ chars/4 + per-message overhead). Good enough
    to decide whether to auto-compact; not an exact tokenizer count."""
    total = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            total += len(c)
        elif isinstance(c, list):
            for it in c:
                if isinstance(it, dict) and isinstance(it.get("text"), str):
                    total += len(it["text"])
        if m.get("tool_calls"):
            try:
                total += len(json.dumps(m["tool_calls"]))
            except Exception:
                pass
        total += 16
    return int(total / 4) + 8


def _compact_messages(messages, n_ctx, pct, strategy, summary_text=None, target=None):
    """Shrink an over-long message list to fit ``target`` tokens (default ~65% of
    n_ctx), keeping system messages and the most recent turns. Returns
    (new_messages, info|None). Strategies:
      - drop_oldest : keep only system + the recent tail that fits.
      - keep_head_tail: also keep the first user turn (context anchor) + a note.
      - summarize   : keep_head_tail, but replace the dropped middle with an LLM
                      summary (``summary_text``) when available, else a count note.
    Returns info=None (no change) when not over threshold or nothing can be dropped.
    """
    if not messages or not n_ctx or n_ctx <= 0:
        return messages, None
    try:
        pct = float(pct)
    except (TypeError, ValueError):
        pct = 85.0
    pct = min(99.0, max(50.0, pct))
    est = _estimate_tokens(messages)
    if target is None:
        target = int(n_ctx * 0.65)
        if est < n_ctx * pct / 100.0:
            return messages, None
    else:
        target = max(1, int(target))
        if est <= target:
            return messages, None

    sys_msgs = [m for m in messages if m.get("role") == "system"]
    body = [m for m in messages if m.get("role") != "system"]
    running = _estimate_tokens(sys_msgs)

    # Track membership by object identity — message dicts can be value-equal
    # (e.g. duplicate "try again" turns or identical tool results).
    tail = []
    tail_ids = set()
    for m in reversed(body):
        t = _estimate_tokens([m])
        if tail and running + t > target:
            break
        tail.insert(0, m)
        tail_ids.add(id(m))
        running += t
    # Don't start the kept tail on an orphaned tool result (its assistant
    # tool_calls turn would have been dropped) — that breaks chat templates.
    while tail and tail[0].get("role") == "tool":
        running -= _estimate_tokens([tail[0]])
        tail_ids.discard(id(tail[0]))
        tail.pop(0)

    head = []
    head_ids = set()
    if strategy in ("keep_head_tail", "summarize"):
        first_user = next((m for m in body if m.get("role") == "user"), None)
        if first_user is not None and id(first_user) not in tail_ids:
            head = [first_user]
            head_ids.add(id(first_user))

    dropped = [m for m in body if id(m) not in head_ids and id(m) not in tail_ids]
    if not dropped:
        return messages, None

    notes = []
    if strategy == "summarize" and summary_text:
        notes.append({"role": "system",
                      "content": "[Summary of earlier conversation]\n" + summary_text})
    elif strategy in ("keep_head_tail", "summarize"):
        notes.append({"role": "system",
                      "content": f"[Note: {len(dropped)} earlier message(s) omitted to fit the context window.]"})

    new = sys_msgs + head + notes + tail
    # Guarantee the CURRENT request (the last message) survives intact and is the
    # very last message after compaction — the compacted history/summary precedes
    # it, then the actual query is repeated at the end.
    last_msg = messages[-1] if messages else None
    if last_msg is not None and (not new or id(new[-1]) != id(last_msg)):
        new = [m for m in new if id(m) != id(last_msg)] + [last_msg]
    return new, {"dropped": len(dropped), "strategy": strategy,
                 "before_tokens": est, "after_tokens": _estimate_tokens(new),
                 "n_ctx": n_ctx}


async def _summarize_one(manager, text: str, max_tokens: int = 400):
    prompt = [
        {"role": "system", "content": "Summarize the following conversation "
         "concisely, preserving key facts, decisions, code, file paths and open "
         "tasks. Output only the summary."},
        {"role": "user", "content": text},
    ]
    out = await asyncio.to_thread(
        manager.generate_chat, messages=prompt, max_tokens=max_tokens, temperature=0.2)
    return (out or "").strip()


def _summary_chunk_chars(compact_n_ctx: int) -> int:
    """Per-chunk char budget for the summarizer so each summarization prompt fits
    the SUMMARIZING model's own context. Leaves headroom for the summary system
    prompt (~120 tok) and the generated summary (~500 tok); ~4 chars/token with a
    0.75 safety factor."""
    usable = max(int(compact_n_ctx or 0) - 700, 512)
    return max(2000, int(usable * 4 * 0.75))


async def _summarize_for_compact(manager, messages, keep_recent: int = 2,
                                 compact_n_ctx: int = 8192, progress=None):
    """Best-effort map-reduce summary of the older turns using ``manager`` (which
    may be a DIFFERENT model than the one serving the request): CHUNK the history
    to fit ``compact_n_ctx``, summarize each chunk, then iteratively reduce the
    combined chunk summaries until they fit one chunk. ``progress`` is an optional
    async callable(str) used to stream status to the client. Returns a summary
    string or None (caller falls back to a count note)."""
    async def _emit(msg):
        if progress:
            try:
                await progress(msg)
            except Exception:
                pass
    try:
        body = [m for m in messages if m.get("role") != "system"]
        older = body[:-keep_recent] if len(body) > keep_recent else body
        if not older:
            return None
        lines = []
        for m in older:
            c = m.get("content")
            if isinstance(c, list):
                c = " ".join(it.get("text", "") for it in c if isinstance(it, dict))
            lines.append(f"{m.get('role', '?')}: {str(c)}")
        text = "\n".join(lines)
        chunk_chars = _summary_chunk_chars(compact_n_ctx)
        chunks = [text[i:i + chunk_chars] for i in range(0, len(text), chunk_chars)] or [text]
        # Map → Reduce, looping the reduce until the combined summaries fit one chunk.
        level = 0
        while True:
            total = len(chunks)
            await _emit(f"summarizing {total} chunk(s) of earlier context…")
            summaries = []
            for i, ch in enumerate(chunks):
                await _emit(f"summarizing chunk {i + 1}/{total}…")
                s = await _summarize_one(manager, ch)
                if s:
                    summaries.append(s)
            if not summaries:
                return None
            if len(summaries) == 1:
                return summaries[0]
            combined = "\n".join(summaries)
            if len(combined) <= chunk_chars or level >= 3:
                await _emit("combining chunk summaries…")
                final = await _summarize_one(manager, combined[:chunk_chars], max_tokens=500)
                return final or combined
            # Still too big — reduce another level.
            chunks = [combined[i:i + chunk_chars] for i in range(0, len(combined), chunk_chars)]
            level += 1
    except Exception as e:
        print(f"[auto-compact] summary generation failed: {e}", flush=True)
        return None


async def _slice_oversized(manager, messages, target, compact_n_ctx, progress=None):
    """Last-resort layer of auto-compaction: when a SINGLE message is itself larger
    than the compaction ``target`` (so dropping other turns can't make the prompt
    fit), shrink that message in place — keep a verbatim head + tail (so the actual
    instruction / code survives) and replace the bulk middle with an LLM summary
    produced by ``manager``. Returns a new message list; best-effort, leaving any
    message that can't be summarized untouched."""
    async def _emit(msg):
        if progress:
            try:
                await progress(msg)
            except Exception:
                pass
    # No single message should occupy more than ~60% of the prompt target.
    per_msg_cap = max(512, int(target * 0.6))
    chunk_chars = _summary_chunk_chars(compact_n_ctx)
    out = []
    for m in messages:
        c = m.get("content")
        if (m.get("role") == "system" or not isinstance(c, str)
                or _estimate_tokens([m]) <= per_msg_cap):
            out.append(m)
            continue
        try:
            orig_tok = _estimate_tokens([m])
            cap_chars = per_msg_cap * 4
            head = c[:int(cap_chars * 0.35)]
            tail = c[-int(cap_chars * 0.15):] if int(cap_chars * 0.15) else ""
            middle = c[len(head):len(c) - len(tail)] if tail else c[len(head):]
            await _emit(f"summarizing a large {m.get('role', '?')} message (~{orig_tok} tok)…")
            chunks = [middle[i:i + chunk_chars]
                      for i in range(0, len(middle), chunk_chars)] or [middle]
            parts = []
            for ch in chunks:
                s = await _summarize_one(manager, ch)
                if s:
                    parts.append(s)
            summary = "\n".join(parts).strip() or "[content omitted]"
            nm = dict(m)
            nm["content"] = (f"{head}\n\n[… {orig_tok}-token message sliced to fit the "
                             f"context window; middle summarized:\n{summary}\n…]\n\n{tail}")
            out.append(nm)
        except Exception as e:
            print(f"[auto-compact] slice failed: {e}", flush=True)
            out.append(m)
    return out


def _model_max_tokens(request):
    """Model-level max_tokens cap, or None when unset.

    Per-model models.json ``max_tokens`` wins; otherwise the node-wide
    ``models.max_tokens``. This is the authority for a reply's length: the
    client's request is honored only when it is smaller (see _clamp_max_tokens)."""
    try:
        from codai.models.manager import multi_model_manager as _mmm
        _cc = _mmm._config_for_model(getattr(request, "model", None) or "") or {}
    except Exception:
        _cc = {}
    _raw = _cc.get("_raw_cfg") if isinstance(_cc, dict) else None
    if not isinstance(_raw, dict):
        _raw = {}
    _v = None
    if isinstance(_cc, dict) and _cc.get("max_tokens") is not None:
        _v = _cc.get("max_tokens")
    elif _raw.get("max_tokens") is not None:
        _v = _raw.get("max_tokens")
    else:
        try:
            from codai.admin.routes import config_manager as _cm
            if _cm is not None and getattr(_cm, "config", None) is not None:
                _v = getattr(_cm.config.models, "max_tokens", None)
        except Exception:
            _v = None
    try:
        _v = int(_v) if _v is not None else None
    except (TypeError, ValueError):
        _v = None
    return _v if (_v and _v > 0) else None


def _clamp_max_tokens(request):
    """Resolve the reply's max_tokens to a concrete value (never None).

    The model-level cap (per-model models.json "max_tokens", else global
    models.max_tokens) is the authority: a client's value is honored only when it
    is SMALLER; a larger or absent request uses the model-level value. When no
    model-level cap is configured, the client's value is kept as-is, and an absent
    value falls back to 2048. Crucially we never leave it None — the GGUF backend
    treats a missing max_tokens as its tiny 512-token default and truncates the
    reply mid-sentence."""
    cur = getattr(request, "max_tokens", None)
    try:
        cur = int(cur) if cur is not None else None
    except (TypeError, ValueError):
        cur = None
    if cur is not None and cur <= 0:
        cur = None
    cap = _model_max_tokens(request)
    if cap:
        request.max_tokens = min(cur, cap) if cur else cap
    else:
        request.max_tokens = cur if cur else 2048


def _resolve_compaction(request, current_manager):
    """Resolve effective auto-compaction settings for a request by merging the
    per-model config over the global ``compaction`` defaults. Returns a plan dict
    or None when compaction is disabled. The over-threshold decision is made later
    against the live token estimate (see ``_auto_compact_events``)."""
    try:
        from codai.models.manager import multi_model_manager as _mmm
        _cc = _mmm._config_for_model(getattr(request, "model", None) or "") or {}
    except Exception:
        _mmm = None
        _cc = {}
    # ``multi_model_manager.config`` stores a runtime-kwargs dict (built by
    # build_runtime_kwargs), NOT the raw models.json entry — the per-model
    # ``auto_compact*`` keys are only preserved under ``_raw_cfg``. Read from the
    # runtime dict first, then fall back to the raw entry, so the flags aren't
    # silently lost (which disabled compaction entirely).
    _raw = _cc.get("_raw_cfg") if isinstance(_cc, dict) else None
    if not isinstance(_raw, dict):
        _raw = {}

    def _mc(key, default):
        if isinstance(_cc, dict) and key in _cc:
            return _cc[key]
        if key in _raw:
            return _raw[key]
        return default

    _g = None
    try:
        from codai.admin.routes import config_manager as _cm
        if _cm is not None and getattr(_cm, "config", None) is not None:
            _g = _cm.config.compaction
    except Exception:
        _g = None

    def _gv(attr, default):
        return getattr(_g, attr, default) if _g is not None else default

    enabled = _mc("auto_compact", _gv("enabled", False))
    if not enabled:
        return None
    pct = _mc("auto_compact_pct", _gv("pct", 85)) or 85
    strategy = (_mc("auto_compact_strategy", None) or _gv("strategy", "drop_oldest") or "drop_oldest").strip()
    compact_model = (_mc("auto_compact_model", None) or _gv("model", "") or "").strip()
    tol_pct = _mc("auto_compact_tolerance_pct", _gv("tolerance_pct", 20))
    try:
        tol_pct = max(0.0, float(tol_pct))
    except (TypeError, ValueError):
        tol_pct = 20.0
    min_output = _mc("auto_compact_min_output", _gv("min_output", 512)) or 512
    try:
        safety = max(1.0, float(_mc("auto_compact_estimate_safety", _gv("estimate_safety", 1.15))))
    except (TypeError, ValueError):
        safety = 1.15

    try:
        n_ctx = current_manager.get_context_size() if current_manager else 0
    except Exception:
        n_ctx = 0

    # NOTE: the summarizer model (``compact_model``) is resolved LAZILY in
    # _auto_compact_events, only when the prompt is actually over threshold — so a
    # configured separate model isn't loaded on every (under-threshold) request.
    # ``max_tokens`` (the reply reservation) is counted against the window too —
    # the model writes its reply into the same n_ctx as the prompt.
    return {
        "pct": float(pct), "strategy": strategy, "n_ctx": n_ctx,
        "compact_model": compact_model, "current_manager": current_manager,
        "max_tokens": getattr(request, "max_tokens", None),
        "tolerance": 1.0 + tol_pct / 100.0, "min_output": int(min_output),
        "safety": safety,
    }


def _resolve_compact_manager(plan):
    """Lazily pick the manager that performs summarization for ``plan`` and its
    context size. Returns (manager, name, compact_n_ctx). Falls back to the
    request's own model when no separate model is configured or it can't load."""
    current_manager = plan.get("current_manager")
    compact_manager = current_manager
    try:
        compact_name = getattr(current_manager, "model_name", None) or "the model"
    except Exception:
        compact_name = "the model"
    compact_model = plan.get("compact_model")
    if compact_model:
        try:
            from codai.models.manager import multi_model_manager as _mmm
            _cand = _mmm.get_model_for_request(compact_model)
            if _cand is not None and getattr(_cand, "backend", None) is not None:
                compact_manager = _cand
                compact_name = compact_model
        except Exception:
            pass
    try:
        compact_n_ctx = compact_manager.get_context_size() if compact_manager else 0
    except Exception:
        compact_n_ctx = 0
    return compact_manager, compact_name, (compact_n_ctx or plan.get("n_ctx") or 4096)


async def _auto_compact_events(plan, messages):
    """Drive auto-compaction for ``plan`` (from ``_resolve_compaction``), yielding
    ('status', text) progress events and finally one ('done', messages, info,
    error) event. ``error`` is a string when the request still overflows after
    compaction (caller decides whether to raise or stream it), else None. When the
    prompt is under threshold, yields only the terminal ('done', messages, None,
    None)."""
    n_ctx = plan["n_ctx"]
    pct = plan["pct"]
    strategy = plan["strategy"]
    mt = int(plan.get("max_tokens") or 0)
    tol = float(plan.get("tolerance") or 1.20)
    min_out = int(plan.get("min_output") or 512)
    safety = float(plan.get("safety") or 1.15)
    if not n_ctx:
        yield ("done", messages, None, None)
        return

    # The reply is generated INTO the same window as the prompt, so the real
    # constraint is prompt + max_tokens ≤ n_ctx. Four layers, cheapest first:
    #   1. fits as-is                          → do nothing
    #   2. overflow within ``tol`` of n_ctx    → trim max_tokens to fit (lossless)
    #   3. overflow beyond tolerance & prompt   → compact history (drop/summarize)
    #      itself over target
    #   4. a single message still over target  → slice that message (summarize it)
    # ``pest`` inflates the cheap chars/4 estimate by ``safety`` for every
    # physical-fit decision — the raw estimate undercounts token-dense code/JSON
    # prompts, and trimming to the exact n_ctx edge off an undercount still
    # overflows the backend.
    prompt_est = _estimate_tokens(messages)
    pest = int(prompt_est * safety)
    if pest + mt <= n_ctx:
        yield ("done", messages, None, None)
        return

    tol_ctx = int(n_ctx * tol)
    # Keep room for a real reply: reserve the requested output (capped at half the
    # window so the prompt isn't squeezed to nothing); the kept prompt targets the
    # rest, never more than ~65% of n_ctx.
    reserve = max(min_out, min(mt or min_out, int(n_ctx * 0.5)))
    target = min(int(n_ctx * 0.65), max(min_out, n_ctx - reserve))

    info = None
    compact_manager = compact_n_ctx = None
    # Layer 3 — only when the overflow is beyond tolerance AND compacting history
    # can actually help (the prompt itself exceeds the target; if it's small and a
    # huge max_tokens is the problem, layer 2's trim handles it).
    if pest + mt > tol_ctx and pest > target:
        summary = None
        if strategy == "summarize":
            compact_manager, compact_name, compact_n_ctx = _resolve_compact_manager(plan)
            via = f" via {compact_name}" if compact_name and compact_name != "the model" else ""
            yield ("status", f"🗜 Compacting context (~{prompt_est}+{mt} tok > {int(tol * 100)}% "
                             f"of {n_ctx}) using '{strategy}'{via}…\n")
            # Bridge the summarizer's progress callback to this generator through a
            # queue so status lines stream to the client LIVE while it summarizes
            # (summarization can take minutes on a large model).
            _q: asyncio.Queue = asyncio.Queue()
            _DONE = object()

            async def _cb(msg):
                await _q.put(f"  • {msg}\n")

            async def _run():
                try:
                    return await _summarize_for_compact(
                        compact_manager, messages,
                        compact_n_ctx=compact_n_ctx, progress=_cb)
                finally:
                    await _q.put(_DONE)

            _task = asyncio.create_task(_run())
            while True:
                _ev = await _q.get()
                if _ev is _DONE:
                    break
                yield ("status", _ev)
            summary = await _task
        else:
            yield ("status", f"🗜 Compacting context (~{prompt_est}+{mt} tok > {int(tol * 100)}% "
                             f"of {n_ctx}) using '{strategy}'…\n")
        messages, info = _compact_messages(messages, n_ctx, pct, strategy, summary, target=target)
        if info:
            yield ("status", f"✅ Context compacted: dropped {info['dropped']} message(s), "
                             f"~{info['before_tokens']}→{info['after_tokens']} tokens.\n")
        prompt_est = _estimate_tokens(messages)
        pest = int(prompt_est * safety)

        # Layer 4 — a single message is still bigger than the target; slice it.
        if pest > target:
            if compact_manager is None:
                compact_manager, _cn, compact_n_ctx = _resolve_compact_manager(plan)
            yield ("status", "✂ A single message exceeds the target; slicing it…\n")
            _q2: asyncio.Queue = asyncio.Queue()
            _DONE2 = object()

            async def _cb2(msg):
                await _q2.put(f"  • {msg}\n")

            async def _run2():
                try:
                    return await _slice_oversized(
                        compact_manager, messages, target, compact_n_ctx, progress=_cb2)
                finally:
                    await _q2.put(_DONE2)

            _t2 = asyncio.create_task(_run2())
            while True:
                _ev = await _q2.get()
                if _ev is _DONE2:
                    break
                yield ("status", _ev)
            messages = await _t2
            prompt_est = _estimate_tokens(messages)
            pest = int(prompt_est * safety)

    # Layer 2 — trim the reply reservation so prompt + output fits the window.
    if mt and pest + mt > n_ctx:
        eff = max(min_out, n_ctx - pest)
        if eff < mt:
            plan["effective_max_tokens"] = eff
            yield ("status", f"✂ Reducing max_tokens {mt}→{eff} to fit the {n_ctx}-token window.\n")

    # Final guard: even a minimal reply won't fit — the prompt alone is too big.
    err = None
    if pest + min_out > n_ctx:
        err = ("The request is too large for this model's context window "
               f"(~{prompt_est} prompt tokens vs n_ctx={n_ctx}) even after "
               "auto-compaction. Shorten the latest message or increase the "
               "model's context size (n_ctx).")
    yield ("done", messages, info, err)


@router.post("/v1/chat/completions", summary="Chat completions")
async def chat_completions(request: ChatCompletionRequest, http_request: Request = None):
    """Chat completions endpoint with streaming and tool support."""
    log_request_exchange(request)

    # Check if we should use litellm backend
    parser_type = getattr(global_args, 'parser', 'auto') if global_args else 'auto'
    
    if parser_type == 'litellm':
        # Use LiteLLM backend
        from codai.openai.litellm import get_litellm_backend, LITELLM_AVAILABLE
        
        if not LITELLM_AVAILABLE:
            raise HTTPException(
                status_code=500,
                detail="LiteLLM is not installed. Run: pip install litellm"
            )
        
        # Check for API key in request - litellm requires an API key
        # If not provided, use a fake key to allow the request to proceed
        api_key = None
        
        # Try to get API key from request body
        if hasattr(request, 'api_key') and request.api_key:
            api_key = request.api_key
        
        # If no API key in body, try to get from Authorization header
        if not api_key:
            auth_header = http_request.headers.get('Authorization', '') if http_request else ''
            if auth_header.startswith('Bearer '):
                api_key = auth_header[7:]  # Extract token after 'Bearer '
        
        if not api_key:
            raise HTTPException(
                status_code=401,
                detail="An API key is required for the LiteLLM backend. "
                       "Provide an 'Authorization: Bearer <key>' header.",
            )
        
        # Determine the base URL for litellm to connect to
        api_base = None

        if request.model and request.model.startswith('ollama:'):
            client_host = "127.0.0.1"
            if http_request:
                host_header = http_request.headers.get('host', '')
                if host_header:
                    if ':' in host_header:
                        client_host = host_header.split(':')[0]
                    else:
                        client_host = host_header
            port = getattr(global_args, 'port', 11434) if global_args else 11434
            api_base = f"http://{client_host}:{port}/v1"
        else:
            if http_request:
                host_header = http_request.headers.get('host', '')
                if host_header:
                    if ':' in host_header:
                        parts = host_header.split(':')
                        client_host = parts[0]
                        server_port = parts[1] if len(parts) > 1 else str(getattr(global_args, 'port', 6745))
                    else:
                        client_host = host_header
                        server_port = str(getattr(global_args, 'port', 6745))
                else:
                    client_host = http_request.client.host if http_request.client else "127.0.0.1"
                    server_port = str(getattr(global_args, 'port', 6745))
            else:
                client_host = "127.0.0.1"
                server_port = str(getattr(global_args, 'port', 6745))
            use_https = getattr(global_args, 'https', False) or getattr(global_args, 'pubkey', None)
            protocol = "https" if use_https else "http"
            api_base = f"{protocol}://{client_host}:{server_port}/v1"
        
        # Get or create litellm backend
        litellm_backend = get_litellm_backend(
            model=request.model,
            api_key=api_key,
            api_base=api_base,
            context_window=8192,  # Default, can be made configurable
            model_manager=multi_model_manager  # Pass for alias resolution
        )
        
        # Get the tool_parser from multi_model_manager for model-specific parsing
        tool_parser = multi_model_manager.tool_parser if hasattr(multi_model_manager, 'tool_parser') else None
        
        # Convert messages to dict format
        messages_dict = []
        for msg in request.messages:
            msg_dict = {"role": msg.role, "content": msg.content or ""}
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                msg_dict["tool_calls"] = msg.tool_calls
            if hasattr(msg, 'tool_call_id') and msg.tool_call_id:
                msg_dict["tool_call_id"] = msg.tool_call_id
            messages_dict.append(msg_dict)
        
        # Prepare tools if provided
        tools_dict = None
        if request.tools:
            tools_dict = request.tools
        
        # Generate response
        try:
            if request.stream:
                # Streaming response
                
                async def generate():
                    try:
                        async for chunk in await litellm_backend.chat_completion(
                            messages=messages_dict,
                            model=request.model,
                            temperature=request.temperature,
                            top_p=request.top_p,
                            max_tokens=request.max_tokens,
                            stop=request.stop,
                            tools=tools_dict,
                            tool_choice=request.tool_choice,
                            stream=True,
                            tool_parser=tool_parser,
                        ):
                            if 'qwen' in request.model.lower():
                                content = chunk.get('choices', [{}])[0].get('delta', {}).get('content', '')
                                tool_calls = chunk.get('choices', [{}])[0].get('delta', {}).get('tool_calls', [])
                                if not tool_calls and content:
                                    tool_calls = litellm_backend.parse_qwen_tool_calls(content)
                                    if tool_calls:
                                        content = litellm_backend.strip_tool_tags(content)
                                        chunk['choices'][0]['delta']['content'] = content
                                        chunk['choices'][0]['delta']['tool_calls'] = tool_calls
                            yield f"data: {json.dumps(chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                    except Exception as e:
                        # Send error chunk then [DONE] so clients don't hang waiting
                        yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'internal_error'}})}\n\n"
                        yield "data: [DONE]\n\n"
                
                from fastapi.responses import StreamingResponse
                return StreamingResponse(generate(), media_type="text/event-stream")
            else:
                # Non-streaming response
                response = await litellm_backend.chat_completion(
                    messages=messages_dict,
                    model=request.model,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_tokens=request.max_tokens,
                    stop=request.stop,
                    tools=tools_dict,
                    tool_choice=request.tool_choice,
                    stream=False,
                    tool_parser=tool_parser,
                )
                
                # Handle Qwen tool calls
                if 'qwen' in request.model.lower() and 'choices' in response:
                    msg = response['choices'][0].get('message', {})
                    content = msg.get('content', '')
                    tool_calls = msg.get('tool_calls', [])
                    
                    if not tool_calls and content:
                        tool_calls = litellm_backend.parse_qwen_tool_calls(content)
                        if tool_calls:
                            msg['content'] = litellm_backend.strip_tool_tags(content)
                            msg['tool_calls'] = tool_calls
                            response['choices'][0]['message'] = msg
                
                # Add rate limit headers
                headers = {}
                if 'usage' in response:
                    headers = litellm_backend.get_rate_limit_headers(
                        prompt_tokens=response.get('usage', {}).get('prompt_tokens', 0),
                        completion_tokens=response.get('usage', {}).get('completion_tokens', 0)
                    )
                
                from fastapi.responses import JSONResponse
                return JSONResponse(content=response, headers=headers)
        
        except Exception as e:
            # Handle litellm errors
            error_response = {
                "error": {
                    "message": str(e),
                    "type": "internal_error",
                    "code": 500
                }
            }
            from fastapi.responses import JSONResponse
            return JSONResponse(content=error_response, status_code=500)
    
    # Continue with original implementation for 'auto' parser
    # Get the model for this request
    requested_model = request.model

    # Resolve and load the model, waiting if another model is currently loading.
    # Retries up to ~5 minutes (60 × 5s) so requests queue behind long video loads
    # rather than failing immediately with "No model loaded".
    _MAX_WAIT_TRIES = 60
    _model_key = None
    _instance_idx = None
    mm = None
    model_info = {}

    for _attempt in range(_MAX_WAIT_TRIES):
        # Fail fast on a corrupted CUDA context — retrying 60× is pointless.
        if getattr(multi_model_manager, 'cuda_context_poisoned', False):
            raise HTTPException(status_code=503, detail=(
                "CUDA context corrupted by an earlier device-side assert "
                f"({multi_model_manager.cuda_poison_reason}). Restart coderai to recover."))

        # If another model is loading, yield the event loop and wait for it to finish.
        if not multi_model_manager._model_ready_event.is_set():
            print(f"Text model '{requested_model}': waiting for model load to complete "
                  f"(attempt {_attempt + 1}/{_MAX_WAIT_TRIES})…")
            await asyncio.to_thread(
                multi_model_manager._model_ready_event.wait, 30.0
            )
            await asyncio.sleep(0)

        # In a thread: request_model may block waiting for a busy model to go
        # idle before evicting it; blocking the event loop here would deadlock.
        model_info = await asyncio.to_thread(
            multi_model_manager.request_model,
            requested_model,
            "text",
        )
        if model_info.get('error'):
            # CUDA-poison errors are unrecoverable → 503; others (unknown model) → 404.
            _status = 503 if 'CUDA context corrupted' in str(model_info['error']) else 404
            raise HTTPException(status_code=_status, detail=model_info['error'])

        _model_key = model_info.get('model_key')
        _candidate = None
        _session_key = _conversation_session_key(request, http_request)
        _acq = multi_model_manager.acquire_model_instance(
            _model_key, session_key=_session_key) if _model_key else None
        if _acq:
            _instance_idx, _candidate = _acq
            # Guard against stale pool entries (model evicted but pool not cleared)
            if hasattr(_candidate, 'backend') and _candidate.backend is None:
                multi_model_manager.release_model_instance(_model_key, _instance_idx)
                _instance_idx = None
                _candidate = None
        if _candidate is None:
            _candidate = multi_model_manager.get_model_for_request(requested_model)
        if _candidate is None and model_manager.backend is not None:
            _candidate = model_manager
        # Validate the candidate has a working backend before accepting it
        if _candidate is not None:
            if hasattr(_candidate, 'backend') and _candidate.backend is None:
                _candidate = None
        if _candidate is not None:
            mm = _candidate
            break

        _load_err = None
        if _model_key:
            _load_err = getattr(multi_model_manager, '_last_load_errors', {}).get(_model_key)
        if _load_err:
            raise HTTPException(status_code=503, detail=(
                f"Model '{requested_model}' failed to load: {_load_err}"))

        print(f"Text model '{requested_model}' not ready, retrying in 5s "
              f"(attempt {_attempt + 1}/{_MAX_WAIT_TRIES})…")
        await asyncio.sleep(5)

    def _release_instance():
        if _instance_idx is not None and _model_key:
            multi_model_manager.release_model_instance(_model_key, _instance_idx)

    if mm is None:
        _release_instance()
        raise HTTPException(status_code=503,
                            detail=f"Model '{requested_model}' could not be loaded after waiting. "
                                   "Another model may be using all available VRAM.")
    current_manager = mm

    # Does the resolved (loaded) model accept images? True only when an mmproj
    # projector was loaded into the llama.cpp backend (see VulkanBackend). When
    # set, multipart image content is preserved end-to-end instead of being
    # flattened to a text placeholder, so the multimodal handler can see it.
    _vision_ok = bool(getattr(getattr(current_manager, 'backend', None), 'supports_vision', False))

    # Inject system prompt if --system-prompt flag was provided
    messages = request.messages
    global_system_prompt = get_global_system_prompt()
    if global_system_prompt is not None:
        # Get the custom system prompt text
        if global_system_prompt is True:
            # Default system prompt
            system_addon = "You are a helpful assistant."
        else:
            # Custom system prompt provided as argument
            system_addon = str(global_system_prompt)
        
        # Check if there's already a system message
        system_found = False
        for i, msg in enumerate(messages):
            if msg.role == "system":
                # Chain the custom system prompt at the START of existing system message
                from codai.pydantic.textrequest import ChatMessage
                messages[i] = ChatMessage(role="system", content=system_addon + "\n\n" + msg.content)
                system_found = True
                break
        
        if not system_found:
            # No existing system message, use the custom one
            from codai.pydantic.textrequest import ChatMessage
            messages = [ChatMessage(role="system", content=system_addon)] + list(messages)
    
    # Enable thinking/reasoning mode if requested via API parameter OR CLI flag
    force_reasoning_args = getattr(global_args, 'force_reasoning', None) if global_args else None
    
    enable_thinking_api = getattr(request, 'enable_thinking', False)
    
    # Parse force_reasoning: can be list (from CLI) or string (legacy)
    if isinstance(force_reasoning_args, str):
        # Legacy: convert string to list
        if force_reasoning_args == "both":
            force_reasoning_args = ["inject", "stop"]
        elif force_reasoning_args == "stop":
            force_reasoning_args = ["stop"]
        elif force_reasoning_args == "inject":
            force_reasoning_args = ["inject"]
        elif force_reasoning_args == "all":
            # 'all' enables all reasoning methods
            force_reasoning_args = ["chat", "inject", "prompt", "mock", "raw", "twopass"]
        else:
            force_reasoning_args = []
    elif not force_reasoning_args:
        force_reasoning_args = []
    
    # Combine CLI args with API param
    # 'chat' from CLI enables API reasoning param
    reasoning_enabled = enable_thinking_api or (len(force_reasoning_args) > 0)

    # Whether to suppress the reasoning channel (drop the model's thinking from the
    # response). A REQUEST override wins, via a standard field — OpenRouter's
    # `reasoning: {"exclude": true}`, OpenAI-style `reasoning_effort: "none"`, or a
    # plain `suppress_reasoning` bool — else the per-model `suppress_reasoning`
    # config. Default: surface reasoning as a separate `reasoning`/`reasoning_content`.
    _suppress_reasoning = None
    _req_reasoning = getattr(request, "reasoning", None)
    if isinstance(_req_reasoning, dict) and "exclude" in _req_reasoning:
        _suppress_reasoning = bool(_req_reasoning.get("exclude"))
    if _suppress_reasoning is None:
        _eff = getattr(request, "reasoning_effort", None)
        if isinstance(_eff, str) and _eff.strip().lower() in ("none", "off", "disable"):
            _suppress_reasoning = True
    if _suppress_reasoning is None:
        _rs = getattr(request, "suppress_reasoning", None)
        if _rs is not None:
            _suppress_reasoning = bool(_rs)
    if _suppress_reasoning is None:
        try:
            from codai.models.manager import multi_model_manager as _mmm_sr
            _sr_cfg = _mmm_sr._config_for_model(getattr(request, "model", None) or "") or {}
            _suppress_reasoning = bool(_sr_cfg.get("suppress_reasoning", False))
        except Exception:
            _suppress_reasoning = False

    # DEBUG: Print force_reasoning status when debug mode is enabled
    if get_global_debug():
        # Get ggg and tools_closer_prompt from global_args
        ggg_enabled = getattr(global_args, 'grammar_guided_gen', False) if global_args else False
        tools_closer = getattr(global_args, 'tools_closer_prompt', False) if global_args else False
        
        print(f"\n{'='*60}")
        print(f"=== REASONING MODE DEBUG ===")
        print(f"{'='*60}")
        print(f"force_reasoning CLI args: {force_reasoning_args}")
        print(f"enable_thinking API param: {enable_thinking_api}")
        print(f"ggg (grammar-guided-gen) CLI flag: {ggg_enabled}")
        print(f"tools-closer-prompt CLI flag: {tools_closer}")
        # Debug stop sequences if available
        if 'raw_stop_sequences' in locals():
            print(f"stop argument for chat call: {raw_stop_sequences}")
    
    # Get model family for reasoning tokens
    from codai.models.utils import get_model_family, get_reasoning_stop_tokens, get_resolved_model_name
    model_family = get_model_family(request.model)
    
    # Check if model is qwen3 and force_reasoning is enabled
    is_qwen3 = 'qwen3' in model_family.lower() if model_family else False
    use_qwen3_penalties = is_qwen3 and force_reasoning_args

    # The reasoning channel must be separated whenever the model actually thinks —
    # which includes models that AUTO-think: their chat template pre-fills <think>
    # at generation regardless of the API `enable_thinking` flag (e.g. Qwen3, QwQ,
    # DeepSeek-R1). So activate on the explicit flag OR a thinking-model name.
    _thinking_model = bool(_re.search(
        r'qwen3|qwq|deepseek[-_]?r[12]|[-_]reasoner|[-_]thinking|glm[-_]?z1',
        (getattr(request, 'model', '') or ''), _re.IGNORECASE))
    _reasoning_active = bool(reasoning_enabled) or _thinking_model

    # System prompt addon for qwen3 with force_reasoning
    qwen3_system_addon = ""
    if use_qwen3_penalties:
        qwen3_system_addon = "\n\nCRITICAL: Do not repeat tool calls. If a tool fails with an [ERROR], do not retry the exact same parameters. Propose a different approach or ask for clarification."
        if get_global_debug():
            print(f"QWEEN3: Adding penalties and system addon for qwen3 with force_reasoning")
    
    # Handle 'chat' - enable thinking API parameter
    # Note: This only works with compatible APIs (OpenAI-like)
    # We'll set it on the request if supported
    if "chat" in force_reasoning_args or enable_thinking_api:
        if hasattr(request, 'thinking'):
            request.thinking = {"type": "enabled"}
        if get_global_debug():
            print(f"CHAT: Reasoning API param enabled")
    
    # Handle 'inject' - system prompt injection
    # Skip for 'raw' mode since it handles everything separately
    if "raw" not in force_reasoning_args and "inject" in force_reasoning_args:
        from codai.models.templates import AgenticTemplateManager
        template_manager = AgenticTemplateManager(request.model)
        
        # Use reasoning tag (]]) when prompt is also selected for consistency
        use_reasoning_tag = "prompt" in force_reasoning_args
        
        # Get the current system prompt if exists
        system_content = None
        for msg in messages:
            if msg.role == "system":
                system_content = msg.content
                break
        if system_content:
            # Inject agentic instructions
            system_content = template_manager.get_agent_system_prompt(system_content, use_reasoning_tag=use_reasoning_tag)
        else:
            system_content = template_manager.get_agent_system_prompt("You are a helpful assistant.", use_reasoning_tag=use_reasoning_tag)
        # Update or add system message
        from codai.pydantic.textrequest import ChatMessage
        system_found = False
        for i, msg in enumerate(messages):
            if msg.role == "system":
                messages[i] = ChatMessage(role="system", content=system_content)
                system_found = True
                break
        if not system_found:
            messages = [ChatMessage(role="system", content=system_content)] + list(messages)
        
        if get_global_debug():
            print(f"INJECT: System prompt injected with agentic instructions")
            print(f"\n--- INJECTED SYSTEM PROMPT ---")
            print(system_content)
            print(f"--- END SYSTEM PROMPT ---")
    
    # Handle 'prompt' - prompt seeding (ends with thought tag)
    # Note: 'prompt' and 'raw' are mutually exclusive - raw bypasses this
    if "prompt" in force_reasoning_args and "raw" not in force_reasoning_args:
        from codai.models.templates import AgenticTemplateManager
        template_manager = AgenticTemplateManager(request.model)
        
        # Convert messages to the format expected by force_reasoning_prompt
        user_message = ""
        system_prompt = "You are a helpful assistant."
        
        # Extract system and user messages
        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            elif msg.role == "user":
                user_message = msg.content
        
        # Add qwen3 system addon if applicable
        if qwen3_system_addon:
            system_prompt = system_prompt + qwen3_system_addon
        
        # Get the seeded prompt (ends with thought tag)
        seeded_prompt = template_manager.force_reasoning_prompt(system_prompt, user_message)
        
        # Replace messages with the seeded prompt (as a single user message for raw completion)
        from codai.pydantic.textrequest import ChatMessage
        messages = [ChatMessage(role="user", content=seeded_prompt)]
        
        if get_global_debug():
            print(f"PROMPT: Prompt seeding applied (ends with thought tag)")
            print(f"\n--- SEEDED PROMPT (last 80 chars) ---")
            print(f"...{seeded_prompt[-80:]}")
            print(f"--- END SEEDED PROMPT ---")
    
    # Handle 'raw' - use template_manager.format_for_raw_completion for raw completion
    # This bypasses the chat API and uses the model's native template with reasoning seed
    # The template_manager.format_for_raw_completion will be called in the block below
    
    # Prepare stop sequences
    stop_sequences = []
    if request.stop:
        if isinstance(request.stop, str):
            stop_sequences = [request.stop]
        else:
            stop_sequences = list(request.stop)
    
    # Handle 'stop' - add reasoning stop tokens (also done for 'inject' and 'prompt')
    # Skip for 'raw' mode since it handles stop tokens separately
    if "raw" not in force_reasoning_args and ("stop" in force_reasoning_args or "inject" in force_reasoning_args or "prompt" in force_reasoning_args):
        _, _, additional_stops = get_reasoning_stop_tokens(model_family)
        for stop_token in additional_stops:
            if stop_token not in stop_sequences:
                stop_sequences.append(stop_token)
        
        # When using prompt seeding, also add ]]> to force stopping after reasoning
        if "prompt" in force_reasoning_args:
            # Add common reasoning end tags based on model family
            if "</think>" not in stop_sequences:
                stop_sequences.append("</think>\n")
        
        if get_global_debug():
            print(f"STOP: Added reasoning stop tokens: {additional_stops}")
    
    # Format messages with tools if provided - BUT SKIP for raw mode
    # (raw mode handles tools separately via format_for_raw_completion)
    # Get tools_closer_prompt from global args
    tools_closer = getattr(global_args, 'tools_closer_prompt', False) if global_args else False
    if request.tools and "raw" not in force_reasoning_args:
        messages = format_tools_for_prompt(request.tools, messages, tools_closer_prompt=tools_closer)
    
    # Get the tool_parser from the current manager
    tool_parser = current_manager.tool_parser if hasattr(current_manager, 'tool_parser') else ModelParserAdapter()
    
    # Convert messages to dict format for chat completion
    messages_dict = []
    for msg in messages:
        msg_dict = {"role": msg.role}
        # Always include content key - llama_cpp template expects it
        # Convert content to string if it's a list (multipart content)
        content = msg.content
        if content is None:
            content = ""
        elif isinstance(content, list):
            _has_image = _vision_ok and any(
                isinstance(it, dict) and it.get('type') in ('image_url', 'input_image')
                for it in content)
            if _has_image:
                # Vision (mmproj) model: keep OpenAI multipart content so the
                # llama.cpp multimodal handler receives the images themselves.
                content = _normalize_vision_content(content)
            else:
                # Handle multipart content array format: [{"type": "text", "text": "..."}]
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get('type') == 'text' and 'text' in item:
                            parts.append(item['text'])
                        else:
                            parts.append(f"[{item.get('type', 'unknown')} content]")
                    else:
                        parts.append(str(item))
                content = '\n'.join(parts)
        # Ensure content is never None - convert to string (but keep multipart
        # vision content as a list so the multimodal handler can consume it).
        if isinstance(content, list):
            msg_dict["content"] = content
        else:
            msg_dict["content"] = str(content) if content is not None else ""
        # Handle tool_calls - convert to proper format if present
        if msg.tool_calls:
            # tool_calls should be a list of dicts with 'id', 'type', 'function'
            # keys; normalise function.arguments from a JSON string to a dict so
            # templates that do `arguments|items` (Qwen, …) don't raise.
            msg_dict["tool_calls"] = _normalize_tool_call_arguments(msg.tool_calls)
        if msg.name:
            msg_dict["name"] = msg.name
        if msg.tool_call_id:
            msg_dict["tool_call_id"] = msg.tool_call_id
        messages_dict.append(msg_dict)
    
    # Final safety check: ensure NO message has None content before passing to llama_cpp
    # Also ensure content key always exists (not just None check)
    for i, m in enumerate(messages_dict):
        # Handle missing content key entirely
        if "content" not in m:
            messages_dict[i]["content"] = ""
        # Handle None content
        elif m.get("content") is None:
            messages_dict[i]["content"] = ""
        # Handle content that's not a string (shouldn't happen but be safe).
        # A list is legitimate multipart vision content — leave it intact.
        elif not isinstance(m["content"], str) and not isinstance(m["content"], list):
            messages_dict[i]["content"] = str(m["content"])

    # Model-level max_tokens authority: clamp the client's requested reply length
    # to the model-level cap (client honored only when smaller). Done before
    # compaction so the window-fit trim works off the clamped value.
    _clamp_max_tokens(request)

    # Auto-compact (per-model or global, OFF by default): when the prompt nears
    # the model's context window, shrink it using the configured strategy
    # (drop_oldest | keep_head_tail | summarize). Resolve the effective settings
    # now; the streaming path applies it inside stream_chat_response so it can
    # stream progress to the client, while the non-streaming path applies it
    # inline just below. The raw two-pass path builds its prompt from only the
    # system + last user turn, so compaction there is a no-op and is skipped.
    _compact_plan = _resolve_compaction(request, current_manager)
    if _compact_plan and not request.stream:
        async for _ev in _auto_compact_events(_compact_plan, messages_dict):
            if _ev[0] == "status":
                print(f"[auto-compact] {_ev[1].strip()}", flush=True)
            else:
                _, messages_dict, _info, _cerr = _ev
                if _info:
                    print(f"[auto-compact] {getattr(request, 'model', '?')}: "
                          f"~{_info['before_tokens']}→{_info['after_tokens']} tokens "
                          f"(dropped {_info['dropped']} msgs via {_info['strategy']})",
                          flush=True)
                if _cerr:
                    raise HTTPException(status_code=400, detail=_cerr)
        # Apply any max_tokens trim decided by the layered compaction so the reply
        # reservation fits the window (the reply shares n_ctx with the prompt).
        _eff_mt = _compact_plan.get("effective_max_tokens")
        if _eff_mt is not None:
            request.max_tokens = _eff_mt


    # Convert tools to dict format if present
    tools_dict = None
    if request.tools:
        tools_dict = []
        for tool in request.tools:
            tools_dict.append({
                "type": tool.type,
                "function": {
                    "name": tool.function.name,
                    "description": tool.function.description,
                    "parameters": tool.function.parameters
                }
            })
    
    # Handle raw mode - use generate() instead of generate_chat() for raw prompt completion
    # Note: These may have been set earlier in the prompt handling section
    # Initialize only if not already set
    if 'use_raw_mode' not in locals():
        use_raw_mode = False
    if 'raw_prompt_for_generation' not in locals():
        raw_prompt_for_generation = None
    if 'raw_stop_sequences' not in locals():
        raw_stop_sequences = None
    
    # Check if we need to set up raw mode (if not already done in prompt handling)
    if "raw" in force_reasoning_args and not use_raw_mode:
        # Create template_manager if not already created
        if 'template_manager' not in locals():
            from codai.models.templates import AgenticTemplateManager
            template_manager = AgenticTemplateManager(request.model)
        
        # Use template_manager.format_for_raw_completion which handles everything
        if hasattr(template_manager, 'format_for_raw_completion'):
            # Extract system and user messages
            system_prompt = "You are a helpful assistant."
            user_message = ""
            for msg in messages:
                if msg.role == "system":
                    system_prompt = msg.content
                elif msg.role == "user":
                    user_message = msg.content
            
            raw_prompt_for_generation, raw_stop_sequences = template_manager.format_for_raw_completion(
                system_prompt=system_prompt,
                user_message=user_message,
                inject_system=True,
                force_reasoning=True,
                tools=request.tools,  # Pass tools for family-specific formatting
                tools_closer_prompt=tools_closer  # Pass tools-closer-prompt flag
            )
            use_raw_mode = True
            
            if get_global_debug():
                print(f"RAW: Using template_manager.format_for_raw_completion")
                print(f"RAW: Prompt ends with: ...{raw_prompt_for_generation[-80:]}")
        else:
            if get_global_debug():
                print(f"RAW: template_manager.format_for_raw_completion not available")
    
    response_model_name = get_resolved_model_name(requested_model, multi_model_manager)
    
    # Handle raw mode - two pass: first capture reasoning, then get final answer
    if use_raw_mode and raw_prompt_for_generation:
        if get_global_debug():
            print(f"RAW: Starting two-pass generation")
            print(f"RAW: First pass prompt: ...{raw_prompt_for_generation[-100:]}")
        
        # Build extra params for qwen3
        extra_params = {}
        if use_qwen3_penalties:
            extra_params = {
                'repeat_penalty': 1.15,
                'presence_penalty': 1.5,
                'frequency_penalty': 0.5,
            }
        
        if request.stream:
            # For streaming, we need to handle it differently
            # First pass: generate until reasoning close tag (stream it)
            async def raw_stream_generate():
                import json  # Local import for nested function
                thought_tag, close_tag, _ = get_reasoning_stop_tokens(model_family)
                reasoning_text = ""
                
                if get_global_debug():
                    print(f"DEBUG: raw_stream_generate started, stream=True")
                
                # Use the backend's async generate if available
                if hasattr(current_manager.backend, 'generate_stream'):
                    # Gate visible content so a tool call (e.g. gemma's
                    # <|tool_call>call:NAME{…}) isn't streamed as a message before it's
                    # parsed; reasoning_text still accumulates the full text for the
                    # end-of-stream tool extraction below.
                    _gate = _ToolCallStreamGate()
                    async for chunk in current_manager.backend.generate_stream(
                        prompt=raw_prompt_for_generation,
                        max_tokens=request.max_tokens or 2048,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        stop=raw_stop_sequences,
                        **extra_params,
                    ):
                        reasoning_text += chunk

                        # Debug: log first pass chunks
                        if get_global_debug():
                            print(f"DEBUG FIRST PASS: chunk length={len(chunk)}, total reasoning so far={len(reasoning_text)}")

                        _emit = _gate.feed(chunk)
                        if _emit:
                            yield f"data: {json.dumps({'choices': [{'delta': {'content': _emit}, 'finish_reason': None}]})}\n\n"

                        # Check if we hit the close tag
                        if close_tag and close_tag in reasoning_text:
                            if get_global_debug():
                                print(f"DEBUG: Close tag detected in first pass, reasoning length={len(reasoning_text)}")
                            break
                    # Release any held-back tail that turned out not to be a tool call.
                    _tail = _gate.flush()
                    if _tail:
                        yield f"data: {json.dumps({'choices': [{'delta': {'content': _tail}, 'finish_reason': None}]})}\n\n"
                else:
                    # Fallback: non-streaming
                    if get_global_debug():
                        print(f"DEBUG: Using non-streaming fallback for first pass")
                    first_pass_result = await asyncio.to_thread(
                        current_manager.generate,
                        prompt=raw_prompt_for_generation,
                        max_tokens=request.max_tokens or 2048,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        stop=raw_stop_sequences,
                        **extra_params,
                    )
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': first_pass_result}, 'finish_reason': None}]})}\n\n"
                
                # After reasoning, yield the close tag and continue with final answer
                if close_tag:
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': close_tag}, 'finish_reason': None}]})}\n\n"
                
                # Second pass: get the rest
                full_prompt = raw_prompt_for_generation + reasoning_text + (close_tag or "")
                
                if get_global_debug():
                    print(f"DEBUG: raw_stream_generate second pass, full_prompt length: {len(full_prompt)}")
                
                second_pass_result = await asyncio.to_thread(
                    current_manager.generate,
                    prompt=full_prompt,
                    max_tokens=request.max_tokens or 2048,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stop=stop_sequences,
                    **extra_params,
                )
                
                # FIX: Apply repetition filtering to both reasoning and final text
                reasoning_text = filter_repetition(reasoning_text)
                second_pass_result = filter_repetition(second_pass_result)
                
                # FIX: If reasoning contains tool call tags, split at the first tool tag
                # The tool call part should NOT be in reasoning - it should be left for tool extraction
                tool_tag_patterns = ["<tool_call>", "<tool>", "<|tool_call>", "<|tool_call|", "<function="]
                earliest_tool_idx = len(reasoning_text)
                earliest_tool_tag = None
                for tag in tool_tag_patterns:
                    idx = reasoning_text.find(tag)
                    if idx != -1 and idx < earliest_tool_idx:
                        earliest_tool_idx = idx
                        earliest_tool_tag = tag
                
                if earliest_tool_tag:
                    # Split: everything before the tool tag is reasoning, everything from the tag onwards goes to second_pass_result
                    tool_part = reasoning_text[earliest_tool_idx:]
                    reasoning_text = reasoning_text[:earliest_tool_idx].strip()
                    # Prepend the tool part to second_pass_result so it can be extracted as a tool call
                    second_pass_result = tool_part + second_pass_result
                    if get_global_debug():
                        print(f"DEBUG: Moved tool call from reasoning to second_pass_result: {tool_part[:100]}...")
                
                # In debug mode, dump the full generated text (second pass result)
                if get_global_debug():
                    print(f"\n{'='*80}")
                    print(f"=== RAW STREAM: FULL GENERATED TEXT (DEBUG) ===")
                    print(f"{'='*80}")
                    print(f"--- SECOND PASS RESULT ---")
                    print(_clip_for_log(second_pass_result))
                    print(f"--- END SECOND PASS RESULT ---")
                    print(f"{'='*80}\n")

                    # Also dump the reasoning text from first pass
                    print(f"\n{'='*80}")
                    print(f"=== RAW STREAM: REASONING TEXT (DEBUG) ===")
                    print(f"{'='*80}")
                    print(_clip_for_log(reasoning_text))
                    print(f"{'='*80}\n")
                
                # Try to extract tool calls from the second pass result ONLY
                # FIX: Do NOT fall back to reasoning text - tool calls should only come from final response
                extracted_tool_calls = None
                text_for_tool_extraction = second_pass_result
                
                # CRITICAL: Only extract from second pass, never from reasoning
                # Reasoning may contain partial/incomplete tool calls that confuse the parser
                if get_global_debug():
                    print(f"DEBUG: Tool extraction - using second_pass_result only")
                    print(f"DEBUG: Second pass result length: {len(second_pass_result) if second_pass_result else 0}")
                    print(f"DEBUG: Reasoning text length: {len(reasoning_text) if reasoning_text else 0}")
                
                if request.tools and text_for_tool_extraction:
                    # Convert tools for ModelParserAdapter
                    from codai.pydantic.textrequest import Tool, ToolFunction
                    from codai.models.parser import ModelParserAdapter
                    
                    tools_list = []
                    for t in request.tools:
                        try:
                            if isinstance(t, dict):
                                func_data = t.get("function", {})
                                tool_func = ToolFunction(
                                    name=func_data.get("name", ""),
                                    description=func_data.get("description"),
                                    parameters=func_data.get("parameters")
                                )
                            else:
                                tool_func = ToolFunction(
                                    name=t.function.name if hasattr(t.function, 'name') else str(t.function),
                                    description=t.function.description if hasattr(t.function, 'description') else None,
                                    parameters=t.function.parameters if hasattr(t.function, 'parameters') else None
                                )
                            tools_list.append(Tool(type=t.get("type", "function") if isinstance(t, dict) else t.type, function=tool_func))
                        except Exception as e:
                            logger.debug("Error converting tool in raw stream: %s", e)
                            continue
                    
                    if tools_list:
                        adapter = ModelParserAdapter(model_name=response_model_name)
                        extracted_tool_calls = adapter.extract_tool_calls(text_for_tool_extraction, tools_list)
                        
                        # FIX: Validate extracted tool calls have valid JSON
                        if extracted_tool_calls:
                            from codai.models.parser import validate_json_complete
                            validated_calls = []
                            for tc in extracted_tool_calls:
                                args = tc.get('function', {}).get('arguments', '{}')
                                if isinstance(args, str) and validate_json_complete(args):
                                    validated_calls.append(tc)
                                elif isinstance(args, dict):
                                    # Dict is already valid
                                    validated_calls.append(tc)
                            
                            if len(validated_calls) != len(extracted_tool_calls):
                                if get_global_debug():
                                    print(f"DEBUG: Filtered out {len(extracted_tool_calls) - len(validated_calls)} invalid tool calls")
                            extracted_tool_calls = validated_calls if validated_calls else None
                        
                        if global_debug and extracted_tool_calls:
                            print(f"\n{'='*80}")
                            print(f"=== RAW STREAM: EXTRACTED TOOL CALLS (DEBUG) ===")
                            print(f"{'='*80}")
                            print(_clip_for_log(json.dumps(extracted_tool_calls, indent=2)))
                            print(f"{'='*80}\n")
                        elif get_global_debug():
                            print(f"DEBUG: No tool calls found in raw stream")
                
                if extracted_tool_calls:
                    # Yield tool calls instead of content
                    yield f"data: {json.dumps({'choices': [{'delta': {'tool_calls': extracted_tool_calls}, 'finish_reason': 'tool_calls'}]})}\n\n"
                else:
                    # No tool calls, yield the content as usual
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': second_pass_result}, 'finish_reason': 'stop'}]})}\n\n"
                yield "data: [DONE]\n\n"

            async def _raw_stream_with_release():
                try:
                    async for chunk in raw_stream_generate():
                        yield chunk
                finally:
                    _release_instance()

            from fastapi.responses import StreamingResponse
            return StreamingResponse(_raw_stream_with_release(), media_type="text/event-stream")
        
        # Non-streaming path (already implemented above)
        # First pass: generate until reasoning close tag
        first_pass_result = await asyncio.to_thread(
            current_manager.generate,
            prompt=raw_prompt_for_generation,
            max_tokens=request.max_tokens or 2048,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=raw_stop_sequences,
            **extra_params,
        )
        
        if get_global_debug():
            print(f"RAW: First pass result: ...{first_pass_result[-200:]}")
        
        # Dump first pass result if --dump is enabled
        global_dump = getattr(global_args, 'dump', False) if global_args else False
        if global_dump:
            print(f"\n{'='*80}")
            print(f"=== RAW MODE: FIRST PASS RESULT (DUMP) ===")
            print(f"{'='*80}")
            print(first_pass_result)
            print(f"{'='*80}\n")
        
        # Extract reasoning (everything up to the close tag)
        thought_tag, close_tag, _ = get_reasoning_stop_tokens(model_family)
        reasoning_text = ""
        final_text = first_pass_result
        
        # Define tool tags that indicate end of reasoning
        tool_tags = ["<tool_call>", "<tool>", "<|tool_call>", "<|tool_call|>", "<|tool|>", "<function="]
        
        if close_tag and close_tag in first_pass_result:
            # Split at close tag
            parts = first_pass_result.split(close_tag, 1)
            reasoning_text = parts[0]
            final_text = parts[1] if len(parts) > 1 else ""
        else:
            # Try to find tool tags as fallback stop markers
            earliest_tool_idx = len(first_pass_result)
            earliest_tool_tag = None
            for tag in tool_tags:
                idx = first_pass_result.find(tag)
                if idx != -1 and idx < earliest_tool_idx:
                    earliest_tool_idx = idx
                    earliest_tool_tag = tag
            
            if earliest_tool_tag:
                # Split at tool tag
                if get_global_debug():
                    print(f"RAW: No close tag found, using tool tag '{earliest_tool_tag}' as fallback")
                parts = first_pass_result.split(earliest_tool_tag, 1)
                reasoning_text = parts[0]
                final_text = earliest_tool_tag + (parts[1] if len(parts) > 1 else "")
        
        if get_global_debug():
            print(f"RAW: Extracted reasoning: {reasoning_text[:100]}...")
            print(f"RAW: Final text before cleanup: {final_text[:100]}...")
        
        # Dump extraction details if --dump is enabled
        if global_dump:
            print(f"\n{'='*80}")
            print(f"=== RAW MODE: EXTRACTION (DUMP) ===")
            print(f"{'='*80}")
            print(f"Close tag used: {close_tag}")
            print(f"\n--- REASONING TEXT ---")
            print(_clip_for_log(reasoning_text))
            print(f"\n--- FINAL TEXT (before cleanup) ---")
            print(_clip_for_log(final_text))
            print(f"{'='*80}\n")
        
        # Clean up control tokens from final text
        final_text = cleanup_control_tokens(final_text)
        
        # FIX: Apply repetition filtering to reasoning and final text
        reasoning_text = filter_repetition(reasoning_text)
        final_text = filter_repetition(final_text)
        
        # FIX: If reasoning contains tool call tags, split at the first tool tag
        # The tool call part should NOT be in reasoning - it should be left for tool extraction in final_text
        tool_tag_patterns = ["<tool_call>", "<tool>", "<|tool_call>", "<|tool_call|>", "<function="]
        earliest_tool_idx = len(reasoning_text)
        earliest_tool_tag = None
        for tag in tool_tag_patterns:
            idx = reasoning_text.find(tag)
            if idx != -1 and idx < earliest_tool_idx:
                earliest_tool_idx = idx
                earliest_tool_tag = tag
        
        if earliest_tool_tag:
            # Split: everything before the tool tag is reasoning, everything from the tag onwards goes to final_text
            tool_part = reasoning_text[earliest_tool_idx:]
            reasoning_text = reasoning_text[:earliest_tool_idx].strip()
            # Prepend the tool part to final_text so it can be extracted as a tool call
            final_text = tool_part + final_text
            if get_global_debug():
                print(f"RAW: Moved tool call from reasoning to final_text: {tool_part[:100]}...")
        
        if get_global_debug():
            print(f"RAW: Final text after cleanup: {final_text[:100]}...")
        
        # If we have reasoning, continue with second pass to get more complete answer
        # Build the full prompt with reasoning included
        full_prompt = raw_prompt_for_generation + reasoning_text + (close_tag or "")
        
        # Second pass: generate the rest (or just use what we have)
        # For now, just return what we have + optionally continue
        if final_text.strip():
            # We have a complete answer after reasoning
            generated_text = reasoning_text + (close_tag or "") + final_text
        else:
            # Need second pass to get answer
            second_pass_result = await asyncio.to_thread(
                current_manager.generate,
                prompt=full_prompt,
                max_tokens=request.max_tokens or 2048,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=stop_sequences,
                **extra_params,
            )
            # Clean up the second pass result
            second_pass_result = cleanup_control_tokens(second_pass_result)
            generated_text = reasoning_text + (close_tag or "") + second_pass_result
        
        # Additional cleanup of the full generated text
        generated_text = cleanup_control_tokens(generated_text)
        
        if get_global_debug():
            print(f"RAW: Generated text after cleanup: {generated_text[:100]}...")
        
        # Pass through the formatter/parser (same as regular mode)
        # Pipeline: Model output -> Extract reasoning (if raw mode) -> ModelParserAdapter (extract tools) -> OpenAIFormatter (final format)
        from codai.models.parser import OpenAIFormatter, ModelParserAdapter
        
        # Convert request tools for ModelParserAdapter
        tools_list = None
        if request.tools:
            from codai.pydantic.textrequest import Tool, ToolFunction
            tools_list = []
            for t in request.tools:
                try:
                    # Handle both dict and pydantic model formats
                    if isinstance(t, dict):
                        func_data = t.get("function", {})
                        tool_func = ToolFunction(
                            name=func_data.get("name", ""),
                            description=func_data.get("description"),
                            parameters=func_data.get("parameters")
                        )
                    else:
                        # Pydantic model
                        tool_func = ToolFunction(
                            name=t.function.name if hasattr(t.function, 'name') else str(t.function),
                            description=t.function.description if hasattr(t.function, 'description') else None,
                            parameters=t.function.parameters if hasattr(t.function, 'parameters') else None
                        )
                    tools_list.append(Tool(type=t.get("type", "function") if isinstance(t, dict) else t.type, function=tool_func))
                except Exception as e:
                    logger.debug("Error converting tool in raw mode: %s (type: %s)", e, type(t))
                    continue
        
        # Step 1: Use ModelParserAdapter to extract tool calls from final_text (NOT generated_text which includes reasoning)
        # This fixes Bug 2 and Bug 3: reasoning was appearing in both content AND reasoning fields
        # because the parser was receiving the full generated_text including reasoning
        extracted_tool_calls = None
        clean_text = final_text  # Use final_text (after reasoning) instead of generated_text (which includes reasoning)
        if tools_list:
            adapter = ModelParserAdapter(model_name=response_model_name)
            # Extract tool calls from final_text only (after reasoning is done)
            extracted_tool_calls = adapter.extract_tool_calls(final_text, tools_list)
            
            # FIX: Validate extracted tool calls have valid JSON
            if extracted_tool_calls:
                from codai.models.parser import validate_json_complete
                validated_calls = []
                for tc in extracted_tool_calls:
                    args = tc.get('function', {}).get('arguments', '{}')
                    if isinstance(args, str) and validate_json_complete(args):
                        validated_calls.append(tc)
                    elif isinstance(args, dict):
                        # Dict is already valid
                        validated_calls.append(tc)
                
                if len(validated_calls) != len(extracted_tool_calls):
                    logger.debug("Filtered out %d invalid tool calls in non-streaming", len(extracted_tool_calls) - len(validated_calls))
                extracted_tool_calls = validated_calls if validated_calls else None
            
            if extracted_tool_calls:
                # Strip tool calls from the text
                clean_text = adapter.strip_tool_calls_from_content(final_text)
                if get_global_debug():
                    print(f"RAW: Extracted {len(extracted_tool_calls)} tool calls from final_text (after reasoning)")
        
        # Estimate token counts
        prompt_tokens = len(raw_prompt_for_generation.split())
        completion_tokens = len(clean_text.split()) if clean_text else 0
        
        # Get context size
        context_size = current_manager.get_context_size()
        
        # Step 2: Use OpenAIFormatter for final formatting
        formatter = OpenAIFormatter(response_model_name)
        try:
            formatted_response = formatter.format_full(
                text=clean_text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                tool_calls=extracted_tool_calls,
                context_size=context_size
            )
        except Exception as e:
            print(f"RAW: ERROR in formatter.format_full: {e}")
            formatted_response = None
        
        if get_global_debug():
            if formatted_response and isinstance(formatted_response, dict):
                try:
                    choices = formatted_response.get('choices', [])
                    if choices and len(choices) > 0:
                        message = choices[0].get('message', {}) if isinstance(choices[0], dict) else {}
                        content = message.get('content', '') if isinstance(message, dict) else ''
                        print(f"RAW: Passed through formatter, got: {str(content)[:100]}...")
                    else:
                        print(f"RAW: WARNING - formatter returned empty choices!")
                except Exception as e:
                    print(f"RAW: ERROR accessing formatter response: {e}")
            else:
                print(f"RAW: WARNING - formatter returned None or invalid response!")
        
        # Add mock reasoning stats if 'mock' is in force_reasoning_args
        # But only if we DON'T already have real reasoning from extraction
        has_real_reasoning = reasoning_text and len(reasoning_text.strip()) > 10
        
        if force_reasoning_args and "mock" in force_reasoning_args and formatted_response and not has_real_reasoning:
            # Add fake reasoning tokens to trigger VSCode plugin stats
            mock_reasoning_tokens = 50
            
            # Update usage
            if "usage" in formatted_response:
                formatted_response["usage"]["completion_tokens"] += mock_reasoning_tokens
                formatted_response["usage"]["total_tokens"] += mock_reasoning_tokens
                formatted_response["usage"]["completion_tokens_details"] = {
                    "reasoning_tokens": mock_reasoning_tokens
                }
            
            # Add reasoning to message if not present
            if "choices" in formatted_response and formatted_response["choices"]:
                choice = formatted_response["choices"][0]
                if "message" in choice and "reasoning" not in choice["message"]:
                    choice["message"]["reasoning"] = "Processing task in optimized mode..."
        elif has_real_reasoning and formatted_response:
            # We have real reasoning from extraction - add it to the message
            if "choices" in formatted_response and formatted_response["choices"]:
                choice = formatted_response["choices"][0]
                if "message" in choice:
                    choice["message"]["reasoning"] = reasoning_text.strip()
                    # Also update usage with actual reasoning tokens
                    if "usage" in formatted_response:
                        reasoning_tokens = len(reasoning_text.strip().split())
                        formatted_response["usage"]["completion_tokens_details"] = {
                            "reasoning_tokens": reasoning_tokens
                        }
        
        # Dump parsed output if enabled
        if global_dump:
            import json
            print(f"\n{'='*80}")
            print(f"=== RAW MODE PARSED OUTPUT (DUMP) ===")
            print(f"{'='*80}")
            print(_clip_for_log(json.dumps(formatted_response, indent=2)))
            print(f"{'='*80}\n")
        
        # Add rate limit headers
        headers = {}
        if formatted_response and 'usage' in formatted_response:
            headers = current_manager.backend.get_rate_limit_headers(
                prompt_tokens=formatted_response.get('usage', {}).get('prompt_tokens', 0),
                completion_tokens=formatted_response.get('usage', {}).get('completion_tokens', 0)
            ) if hasattr(current_manager.backend, 'get_rate_limit_headers') else {}
        
        # Ensure we have a valid response to return
        if not formatted_response:
            # Create a minimal fallback response
            formatted_response = {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": response_model_name,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": clean_text or ""
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "context_size": context_size
                }
            }
        
        from fastapi.responses import JSONResponse
        log_response_payload(formatted_response, streamed=False)
        return JSONResponse(content=formatted_response, headers=headers)

    # Compute prefix key for prompt-aggregation scheduling
    _prefix_key = prompt_cache_manager.get_prefix_key(messages_dict)

    if request.stream:
        async def _managed_stream():
            try:
                async for chunk in stream_chat_response(
                    messages_dict,
                    response_model_name,
                    request.max_tokens,
                    request.temperature,
                    request.top_p,
                    stop_sequences,
                    tools_dict,
                    current_manager,
                    tool_parser,
                    request.response_format,
                    _prefix_key,
                    enable_thinking=reasoning_enabled,
                    compact_plan=_compact_plan,
                    suppress_reasoning=_suppress_reasoning,
                    reasoning_active=_reasoning_active,
                    repeat_penalty=request.repeat_penalty,
                    presence_penalty=request.presence_penalty,
                    frequency_penalty=request.frequency_penalty,
                ):
                    yield chunk
            finally:
                _release_instance()

        from fastapi.responses import StreamingResponse
        return StreamingResponse(_managed_stream(), media_type="text/event-stream")
    else:
        try:
            return await generate_chat_response(
                messages_dict,
                response_model_name,
                request.max_tokens,
                request.temperature,
                request.top_p,
                stop_sequences,
                tools_dict,
                current_manager,
                tool_parser,
                request.response_format,
                force_reasoning_args,
                enable_thinking=reasoning_enabled,
                suppress_reasoning=_suppress_reasoning,
                reasoning_active=_reasoning_active,
                repeat_penalty=request.repeat_penalty,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
            )
        finally:
            _release_instance()

import re as _re

_TOOL_SPAN_RE = _re.compile(r'<(tool|tool_call)\b[\s\S]*?</\1\s*>', _re.IGNORECASE)
_TOOL_OPEN_RE = _re.compile(r'<(?:tool|tool_call)\b', _re.IGNORECASE)
_TOOL_OPEN_TAGS = ('<tool>', '<tool_call>', '<|tool_call>', '<|tool_call|>',
                   '<｜dsml｜tool_calls>')
# gemma/qwen native special-token tool marker `<|tool_call>` — usually a special
# token stripped on decode, but some GGUFs emit it as plain text. Treat it like a
# tool-open: withhold everything from it to the end so the raw marker (and the
# `call:NAME{…}` that follows) never leaks to the client as visible content.
_NATIVE_TOOL_OPEN_RE = _re.compile(r'<\|tool_call', _re.IGNORECASE)
# gemma-4 native tool call: `call:NAME{…}` (the <|tool_call> markers are stripped
# by skip_special_tokens). Once it starts we withhold everything to the end of the
# stream — the call is surfaced as structured tool_calls after generation.
_GEMMA_CALL_OPEN_RE = _re.compile(r'call:\s*[A-Za-z_]\w*\s*\{')
# DeepSeek V4 (ds4) native tool calls: `<｜DSML｜tool_calls>…` (｜ is U+FF5C; ASCII
# | tolerated). Withhold from the first DSML marker to the end so the raw block is
# never streamed as visible content — it's surfaced as structured tool_calls.
_DSML_OPEN_RE = _re.compile(r'<[｜|]DSML[｜|]')

# Reasoning channel: a closing think tag terminates the model's thought. Qwen-style
# chat templates PRE-FILL the opening <think> in the prompt, so the stream begins
# inside the thought and ends with a bare </think> (no opening tag in the output).
_THINK_CLOSE_RE = _re.compile(r'</(?:think|thinking|thought)\s*>', _re.IGNORECASE)
_THINK_OPEN_RE = _re.compile(r'^\s*<(?:think|thinking|thought)\s*>', _re.IGNORECASE)
_THINK_CLOSE_TAGS = ('</think>', '</thinking>', '</thought>')


def _gate_reasoning(buffer: str, final: bool = False):
    """Split a streaming buffer at the reasoning close tag.

    Returns ``(reasoning_out, content_out, new_buffer, closed)``:
      - before any close tag → the text is reasoning, but a trailing fragment that
        could be the start of a partial close tag (e.g. ``</thin``) is held back in
        ``new_buffer`` so half a tag is never emitted as reasoning;
      - when a close tag is found → text before it is reasoning, text after it is
        content, and ``closed`` is True. A leading explicit ``<think>`` is dropped.
    On ``final`` the whole remaining buffer is flushed as reasoning."""
    # Drop an explicit opening tag if the model emitted one (most don't).
    om = _THINK_OPEN_RE.match(buffer)
    if om:
        buffer = buffer[om.end():]
    m = _THINK_CLOSE_RE.search(buffer)
    if m:
        return buffer[:m.start()], buffer[m.end():], "", True
    if final:
        return buffer, "", "", False
    # Hold back a trailing fragment that may be the prefix of a close tag.
    idx = buffer.rfind('<')
    if idx != -1:
        tail = buffer[idx:].lower()
        if any(t.startswith(tail) for t in _THINK_CLOSE_TAGS):
            return buffer[:idx], "", buffer[idx:], False
    return buffer, "", "", False


def _gate_tool_content(buffer: str, final: bool = False):
    """Split accumulated stream text into (content_to_emit, held_buffer).

    During tool-enabled streaming the model emits ``<tool>{json}</tool>`` spans
    inline. Those must NOT reach the client as visible ``content`` (they're
    surfaced separately as structured ``tool_calls``); otherwise the raw tags leak
    into the chat. This withholds any complete or in-progress tool span, plus a
    trailing partial ``<`` that could still grow into a tool tag, and streams only
    the safe text around them. With ``final=True`` any leftover (possibly unclosed)
    tool span is dropped and the rest emitted.
    """
    emit = []
    # Strip complete tool spans, emitting the text around each.
    while True:
        m = _TOOL_SPAN_RE.search(buffer)
        if not m:
            break
        emit.append(buffer[:m.start()])
        buffer = buffer[m.end():]
    # An open tag with no close yet → hold from there (a call is in progress).
    m = _TOOL_OPEN_RE.search(buffer)
    if m:
        emit.append(buffer[:m.start()])
        held = '' if final else buffer[m.start():]
        return ''.join(emit), held
    # gemma/qwen native `<|tool_call>` marker — withhold from it to the end.
    nm = _NATIVE_TOOL_OPEN_RE.search(buffer)
    if nm:
        emit.append(buffer[:nm.start()])
        held = '' if final else buffer[nm.start():]
        return ''.join(emit), held
    # gemma-4 `call:NAME{…}` — withhold from the call onward (extracted at the end).
    gm = _GEMMA_CALL_OPEN_RE.search(buffer)
    if gm:
        emit.append(buffer[:gm.start()])
        held = '' if final else buffer[gm.start():]
        return ''.join(emit), held
    # DeepSeek V4 DSML tool call (<｜DSML｜…>) — withhold from the marker onward.
    dm = _DSML_OPEN_RE.search(buffer)
    if dm:
        emit.append(buffer[:dm.start()])
        held = '' if final else buffer[dm.start():]
        return ''.join(emit), held
    # Hold back a trailing '<…' that could still become a tool open tag.
    if not final:
        lt = buffer.rfind('<')
        if lt != -1:
            tail = buffer[lt:].lower()
            if any(t.startswith(tail) for t in _TOOL_OPEN_TAGS):
                emit.append(buffer[:lt])
                return ''.join(emit), buffer[lt:]
        # Hold a trailing 'call:NAME' (no '{' yet) that may grow into a gemma call.
        cm = _re.search(r'call:\s*[A-Za-z_]?\w*$', buffer)
        if cm:
            emit.append(buffer[:cm.start()])
            return ''.join(emit), buffer[cm.start():]
    emit.append(buffer)
    return ''.join(emit), ''


def _context_overflow_detail(e) -> Optional[str]:
    """If the exception is a context-window overflow (prompt + generation exceed
    the model's n_ctx), return a clear client-facing message; else None."""
    s = str(e)
    low = s.lower()
    markers = ("exceed context window", "context window of", "requested tokens",
               "exceeds n_ctx", "exceed the context", "context length",
               "n_ctx", "kv cache is full", "context shift is disabled")
    if any(m in low for m in markers) or ("token" in low and "exceed" in low):
        return ("The conversation is too long for this model's context window "
                f"({s}). Shorten the prompt or lower max_tokens, or increase the "
                "model's context size (n_ctx) in its configuration.")
    return None


def _detect_runaway_repetition(text: str) -> bool:
    """Heuristic guard against a model stuck emitting the same fragment forever
    (e.g. Qwen collapsing into a malformed parallel tool-call loop). It normalises
    away the *variable* parts — quoted strings, filesystem paths, whitespace — so a
    loop whose only difference each cycle is the path/arg still reads as periodic,
    then looks for a short structural unit repeated back-to-back many times.

    Tuned to fire only on genuine degeneration (>=5 identical structural periods),
    so ordinary prose or code — which doesn't repeat a structural unit 5x verbatim —
    won't trip it."""
    tail = text[-1600:]
    skel = _re.sub(r'"[^"]*"', '""', tail)        # collapse quoted strings/args
    skel = _re.sub(r'/[^\s"<>]+', '/', skel)        # collapse filesystem paths
    skel = _re.sub(r'\s+', ' ', skel)
    n = len(skel)
    for period in range(6, 140):
        if n < period * 5:
            break
        unit = skel[-period:]
        if not unit.strip():
            continue
        reps = 1
        i = n - 2 * period
        while i >= 0 and skel[i:i + period] == unit:
            reps += 1
            i -= period
        if reps >= 5:
            return True
    return False


async def stream_chat_response(
    messages: List[Dict],
    model_name: str,
    max_tokens: Optional[int],
    temperature: float,
    top_p: float,
    stop: List[str],
    tools: Optional[List[Dict]],
    current_manager: ModelManager,
    tool_parser: ToolCallParser,
    response_format: Optional[Dict] = None,
    prefix_key: str = "",
    enable_thinking: bool = False,
    compact_plan: Optional[Dict] = None,
    suppress_reasoning: bool = False,
    reasoning_active: bool = False,
    repeat_penalty: float = 1.0,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
) -> AsyncGenerator[str, None]:
    """Stream chat completion response with queue notifications."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    request_id = f"req-{uuid.uuid4().hex[:8]}"
    _tid = None

    generated_text = ""

    # Auto-compact an over-long history before generation, streaming progress to
    # the client as status content deltas (the same mechanism as the "Waiting for
    # model reply…" notices — visible text, not part of the saved completion).
    if compact_plan:
        try:
            async for _ev in _auto_compact_events(compact_plan, messages):
                if _ev[0] == "status":
                    _sc = {
                        "id": completion_id, "object": "chat.completion.chunk",
                        "created": created, "model": model_name,
                        "choices": [{"index": 0, "delta": {"content": _ev[1]},
                                     "finish_reason": None}],
                        "x_compaction": {"status": "compacting"},
                    }
                    yield f"data: {json.dumps(_sc)}\n\n"
                else:
                    _, messages, _cinfo, _cerr = _ev
                    if _cinfo:
                        print(f"[auto-compact] {model_name}: "
                              f"~{_cinfo['before_tokens']}→{_cinfo['after_tokens']} tokens "
                              f"(dropped {_cinfo['dropped']} msgs via {_cinfo['strategy']})",
                              flush=True)
                    if _cerr:
                        _ec = {
                            "id": completion_id, "object": "chat.completion.chunk",
                            "created": created, "model": model_name,
                            "choices": [{"index": 0, "delta": {"content": "\n⚠ " + _cerr},
                                         "finish_reason": "stop"}],
                        }
                        yield f"data: {json.dumps(_ec)}\n\n"
                        yield "data: [DONE]\n\n"
                        return
        except Exception as _ce:
            print(f"[auto-compact] streaming compaction failed: {_ce}", flush=True)
        # Apply any max_tokens trim decided by the layered compaction so the reply
        # reservation fits the window (the reply shares n_ctx with the prompt).
        _eff_mt = compact_plan.get("effective_max_tokens")
        if _eff_mt is not None:
            max_tokens = _eff_mt

    # Check if model is loaded - if not, notify waiting clients
    # The model manager exists but backend may not be loaded yet in on-demand mode
    model_loaded = False
    if current_manager is not None:
        if hasattr(current_manager, 'backend') and current_manager.backend is not None:
            # Check if backend has the model loaded
            if hasattr(current_manager.backend, 'model') and current_manager.backend.model is not None:
                model_loaded = True
        elif hasattr(current_manager, 'model') and current_manager.model is not None:
            # Alternative check for some model managers
            model_loaded = True
    
    # If model not loaded, add to queue and send waiting notifications
    if not model_loaded:
        await queue_manager.add_waiting(request_id, prefix_key=prefix_key)
        wait_interval = 2.0  # Send waiting update every 2 seconds
        last_wait_update = time.time()
        
        # Send initial waiting message
        data = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": {"content": "Waiting for model reply...\n"},
                "finish_reason": None,
            }],
            "x_queue_info": {
                "status": "waiting",
                "message": "Waiting for model reply...",
            },
        }
        yield f"data: {json.dumps(data)}\n\n"
        
        # Keep sending wait updates until model is loaded
        # In a real implementation, this would check a loading status
        # For now, we'll send a few updates then proceed
        max_wait_updates = 5
        wait_count = 0
        while wait_count < max_wait_updates:
            await asyncio.sleep(wait_interval)
            wait_time = await queue_manager.get_wait_time(request_id)
            wait_count += 1
            
            queue_pos = await queue_manager.get_queue_position(request_id)
            
            data = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {"content": f""},
                    "finish_reason": None,
                }],
                "x_queue_info": {
                    "status": "waiting",
                    "message": f"Waiting for model reply... ({int(wait_time)}s)",
                    "queue_position": queue_pos,
                    "wait_time_seconds": int(wait_time),
                },
            }
            yield f"data: {json.dumps(data)}\n\n"
    
    # Mark as starting processing
    await queue_manager.start_processing(request_id, model_name)
    _tid = task_registry.register("text", title=(model_name or "chat"),
                                  model=model_name or "", task_id=request_id)
    task_registry.start(_tid)
    
    # Send "Model starting" message
    data = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": {"content": ""},
            "finish_reason": None,
        }],
        "x_queue_info": {
            "status": "starting",
            "message": "Model starting",
        },
    }
    yield f"data: {json.dumps(data)}\n\n"
    
    try:
        chunk_count = 0
        _gen_t0 = None          # wall-clock of the first generated token (for it/s)
        # Buffer for withholding in-progress tool tags from the content stream.
        content_buffer = ""
        # Exact content deltas actually streamed to the client (post-format,
        # post tool-gating) — logged once at the end under --debug-requests so we
        # see the real reply, not just what we extracted internally.
        client_sent_content = ""
        # Reasoning channel state. When thinking is enabled the model starts INSIDE
        # its thought (Qwen pre-fills the opening <think>), so route everything up to
        # the closing </think> into the reasoning field instead of content. The gate
        # runs whenever thinking is enabled — even with suppress_reasoning, where it
        # still strips the thought from content but emits no reasoning deltas. The
        # boundary may straddle chunks, so buffer until it's resolvable.
        _reason_active = bool(reasoning_active)
        _reason_closed = False
        _reason_buf = ""
        reasoning_text = ""

        # Debug: Print what is being passed to the model
        if get_global_debug():
            print(f"\n{'='*80}")
            print(f"=== MODEL INPUT (DEBUG) ===")
            print(f"{'='*80}")
            print(f"Model: {model_name}")
            print(f"Max tokens: {max_tokens}")
            print(f"Temperature: {temperature}")
            print(f"Top P: {top_p}")
            print(f"Stop sequences: {stop}")
            print(f"Tools: {tools is not None}")
            print(f"Response format: {response_format}")
            print(f"\n--- Messages ---")
            for i, msg in enumerate(messages):
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                if content and len(content) > 500:
                    content = content[:500] + "... [truncated]"
                print(f"[{i}] {role}: {repr(content)}")
            print(f"{'='*80}\n")
        
        # Use generate_chat_stream for proper chat template handling
        async for chunk in current_manager.generate_chat_stream(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            tools=tools,
            response_format=response_format,
            enable_thinking=enable_thinking,
            repeat_penalty=repeat_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        ):
            # Cooperative cancellation: stop streaming if the task was cancelled.
            if task_registry.is_cancelled(_tid):
                break
            chunk_count += 1
            # Publish live throughput (tokens/s) onto the task for the Tasks page.
            # The streamer yields ~one token per chunk; refresh every few tokens to
            # keep the registry lock cold.
            if _gen_t0 is None:
                _gen_t0 = time.time()
            elif chunk_count % 8 == 0:
                _elapsed = time.time() - _gen_t0
                if _elapsed > 0:
                    task_registry.update(
                        _tid, step=chunk_count,
                        rate=round(chunk_count / _elapsed, 1))
            # Always filter malformed content (regex-based, works per-chunk)
            filtered_chunk = filter_malformed_content(chunk)
            
            # NOTE: filter_repetition() and strip_tool_calls_from_content() are NOT applied
            # per-chunk because they need the full accumulated text to work correctly:
            # - filter_repetition() needs enough context (6+ words) to detect n-gram repetitions
            # - strip_tool_calls_from_content() needs complete XML tags that span multiple chunks
            # Both are applied to the complete generated_text after streaming completes.
            
            # Pass through all content including whitespace - it's essential for message composition
            generated_text += filtered_chunk

            # Anti-loop safety net: if the model has collapsed into a runaway
            # repetition (e.g. a malformed parallel tool-call flood), stop pulling
            # tokens instead of burning the whole context. Check periodically once
            # there's enough text to judge; downstream finalisation still runs on
            # what we have (the parser's repetition guard keeps the first real call).
            if chunk_count % 32 == 0 and len(generated_text) > 600 \
                    and _detect_runaway_repetition(generated_text):
                if _debug_requests_enabled():
                    print(f"# <<< [anti-loop] runaway repetition detected at "
                          f"{chunk_count} tok — stopping generation", flush=True)
                logger.warning("stream_chat_response: runaway repetition detected for "
                               "model=%s at %d chunks; truncating generation",
                               model_name, chunk_count)
                break

            # Live progress under --debug-requests so a non-terminating / looping
            # generation is visible AS IT HAPPENS — the end-of-stream response logs
            # below never fire if the model never stops. The front pumps engine
            # stdout line-by-line, so emit newline-terminated snapshots (every N
            # chunks) of the accumulated tail; a loop shows up as the same text
            # repeating across snapshots.
            if _debug_requests_enabled():
                if chunk_count == 1:
                    print(f"# <<< STREAMING [live] model={model_name} "
                          f"(snapshots every 64 tokens until stop)", flush=True)
                if chunk_count % 64 == 0:
                    _tail = generated_text[-220:].replace("\n", "\\n")
                    print(f"# <<< [live @{chunk_count} tok] …{_tail}", flush=True)

            # Reasoning gate: while the model is still inside its thought, route the
            # text into the reasoning channel (delta.reasoning / reasoning_content)
            # instead of content — or drop it when suppress_reasoning is set. On the
            # closing </think> we flip to content and keep streaming normally.
            if _reason_active and not _reason_closed:
                _reason_buf += filtered_chunk
                _r_out, _c_out, _reason_buf, _closed = _gate_reasoning(_reason_buf)
                if _r_out:
                    reasoning_text += _r_out
                    if not suppress_reasoning:
                        rdata = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [{
                                "index": 0,
                                "delta": {"reasoning": _r_out, "reasoning_content": _r_out},
                                "finish_reason": None,
                            }],
                        }
                        yield f"data: {json.dumps(rdata)}\n\n"
                        await asyncio.sleep(0)
                if not _closed:
                    await asyncio.sleep(0)
                    continue
                _reason_closed = True
                filtered_chunk = _c_out          # post-</think> remainder is content
                if not filtered_chunk:
                    await asyncio.sleep(0)
                    continue

            # When tools are enabled, gate the content so in-progress <tool>…</tool>
            # spans are never streamed as visible text (they're surfaced as
            # structured tool_calls after the stream). Without tools, stream as-is.
            if tools:
                content_buffer += filtered_chunk
                filtered_chunk, content_buffer = _gate_tool_content(content_buffer)
                if not filtered_chunk:
                    await asyncio.sleep(0)
                    continue

            data = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {"content": filtered_chunk},
                    "finish_reason": None,
                }],
            }
            client_sent_content += filtered_chunk
            yield f"data: {json.dumps(data)}\n\n"
            # Explicitly flush to ensure data is sent immediately
            await asyncio.sleep(0)

        # Stream ended while still inside the thought (no closing </think>). Flush
        # whatever reasoning is held as reasoning, not content.
        if _reason_active and not _reason_closed and _reason_buf:
            _r_out, _c_out, _reason_buf, _closed = _gate_reasoning(_reason_buf, final=True)
            _reason_closed = True
            if _r_out:
                reasoning_text += _r_out
                if not suppress_reasoning:
                    rdata = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {"reasoning": _r_out, "reasoning_content": _r_out},
                            "finish_reason": None,
                        }],
                    }
                    yield f"data: {json.dumps(rdata)}\n\n"

        # The post-</think> answer text — reasoning stripped — used for final tool
        # extraction and logging so the thought never re-enters the content channel.
        answer_text = generated_text
        if _reason_active:
            _cm = _THINK_CLOSE_RE.search(generated_text)
            answer_text = generated_text[_cm.end():] if _cm else ""

        # Flush any safe trailing text held back by the tool-content gate
        # (dropping leftover/unclosed tool tags — they become tool_calls below).
        if tools and content_buffer:
            tail_content, content_buffer = _gate_tool_content(content_buffer, final=True)
            if tail_content:
                data = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": tail_content},
                        "finish_reason": None,
                    }],
                }
                client_sent_content += tail_content
                yield f"data: {json.dumps(data)}\n\n"


        # In debug mode, dump the full generated text
        if get_global_debug():
            print(f"\n{'='*80}")
            print(f"=== FULL GENERATED TEXT (DEBUG) ===")
            print(f"{'='*80}")
            # Show both raw (actual) content and escaped representation
            print(f"--- RAW CONTENT (actual newlines shown as lines) ---")
            print(_clip_for_log(generated_text))
            print(f"--- END RAW CONTENT ---")
            print(f"--- ESCAPED CONTENT (repr() - shows \\n for newlines) ---")
            print(_clip_for_log(repr(generated_text)))
            print(f"--- END ESCAPED CONTENT ---")
            print(f"{'='*80}\n")
        
        # Check for tool calls in complete output (for API response format)
        if tools:
            # Convert tools back to Tool objects for parsing
            from typing import cast
            tool_objects = []
            for t in tools:
                try:
                    # Handle both dict and pydantic model formats
                    if isinstance(t, dict):
                        func_data = t.get("function", {})
                        tool_func = ToolFunction(
                            name=func_data.get("name", ""),
                            description=func_data.get("description"),
                            parameters=func_data.get("parameters")
                        )
                    else:
                        # Pydantic model
                        tool_func = ToolFunction(
                            name=t.function.name if hasattr(t.function, 'name') else str(t.function),
                            description=t.function.description if hasattr(t.function, 'description') else None,
                            parameters=t.function.parameters if hasattr(t.function, 'parameters') else None
                        )
                    tool_objects.append(Tool(type=t.get("type", "function") if isinstance(t, dict) else t.type, function=tool_func))
                except Exception as e:
                    logger.debug("Error converting tool: %s (type: %s)", e, type(t))
                    continue
            try:
                tool_calls = tool_parser.extract_tool_calls(answer_text, tool_objects)

                # FIX: Validate extracted tool calls have valid JSON (stream_chat_response)
                if tool_calls:
                    from codai.models.parser import validate_json_complete
                    validated_calls = []
                    for tc in tool_calls:
                        args = tc.get('function', {}).get('arguments', '{}')
                        if isinstance(args, str) and validate_json_complete(args):
                            validated_calls.append(tc)
                        elif isinstance(args, dict):
                            validated_calls.append(tc)
                    if len(validated_calls) != len(tool_calls):
                        logger.debug("Filtered out %d invalid tool calls in stream_chat_response", len(tool_calls) - len(validated_calls))
                    tool_calls = validated_calls if validated_calls else None
            except Exception as e:
                logger.debug("Error extracting tool calls: %s", e)
                tool_calls = None
            if tool_calls:
                # In debug mode, dump tool calls
                if get_global_debug():
                    print(f"\n{'='*80}")
                    print(f"=== EXTRACTED TOOL CALLS (DEBUG) ===")
                    print(f"{'='*80}")
                    print(json.dumps(tool_calls, indent=2))
                    print(f"{'='*80}\n")
                # Tool calls were extracted and stripped from content during streaming
                # Just send the tool_calls chunk
                log_response_exchange(generated_text, tool_calls=tool_calls,
                                      finish_reason="tool_calls", streamed=True,
                                      stage="pre-format extracted")
                data = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {"tool_calls": tool_calls},
                        "finish_reason": "tool_calls",
                        "logprobs": None,
                        "native_finish_reason": "tool_calls",
                    }],
                }
                log_response_payload(data, streamed=True)
                yield f"data: {json.dumps(data)}\n\n"
            else:
                # Calculate token counts for usage in final chunk
                prompt_text = "\n".join([m.get("content", "") for m in messages])
                prompt_tokens = len(prompt_text.split())
                completion_tokens = len(generated_text.split()) if generated_text else 0
                
                # Get context size
                context_size = current_manager.get_context_size()
                
                # Use OpenAIFormatter for final chunk sanitization
                formatter = OpenAIFormatter(model_name)
                usage_details = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }
                log_response_exchange(generated_text, finish_reason="stop",
                                      streamed=True, stage="pre-format extracted")
                log_response_exchange(client_sent_content, finish_reason="stop",
                                      streamed=True, stage="post-format sent to client")
                final_chunk = formatter.format_litellm_chunk("", is_final=True, usage=usage_details, context_size=context_size)
                log_response_payload(final_chunk, streamed=True)
                yield f"data: {json.dumps(final_chunk)}\n\n"
        else:
            # Calculate token counts for usage in final chunk
            prompt_text = "\n".join([m.get("content", "") for m in messages])
            prompt_tokens = len(prompt_text.split())
            completion_tokens = len(generated_text.split()) if generated_text else 0

            # Read accurate usage (including cached_tokens) from the backend
            _model_key_for_cache = getattr(current_manager, 'model_name', None) or model_name
            last_usage = (current_manager.get_last_usage()
                          if hasattr(current_manager, 'get_last_usage') else {})
            if last_usage.get('prompt_tokens'):
                prompt_tokens = last_usage['prompt_tokens']
            if last_usage.get('completion_tokens'):
                completion_tokens = last_usage['completion_tokens']
            cached_tokens = last_usage.get('cached_tokens', 0)

            # Store in prompt cache manager for future prefix matching
            prompt_cache_manager.store(messages, _model_key_for_cache,
                                       prompt_tokens, cached_tokens)

            # Get context size
            context_size = current_manager.get_context_size()

            # Build complete final chunk with all OpenAI fields
            final_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "logprobs": None,
                    "native_finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "context_size": context_size,
                    "prompt_tokens_details": {
                        "cached_tokens": cached_tokens,
                        "audio_tokens": 0,
                    },
                    "completion_tokens_details": {
                        "reasoning_tokens": 0,
                        "audio_tokens": 0,
                    },
                },
                "provider": {
                    "provider_name": "coderai",
                    "provider_id": "coderai",
                },
                "system_fingerprint": None,
            }
            log_response_exchange(generated_text, finish_reason="stop",
                                  streamed=True, stage="pre-format extracted")
            log_response_exchange(client_sent_content, finish_reason="stop",
                                  streamed=True, stage="post-format sent to client")
            log_response_payload(final_chunk, streamed=True)
            yield f"data: {json.dumps(final_chunk)}\n\n"

        yield "data: [DONE]\n\n"
    except Exception as e:
        print(f"Error during streaming generation: {e}")
        # Surface errors as a STRUCTURED error event (not as assistant content) so
        # the client treats it as an error and it doesn't pollute the chat history.
        _ctx = _context_overflow_detail(e)
        if _ctx:
            err = {"error": {"message": _ctx, "type": "invalid_request_error",
                             "code": "context_length_exceeded", "param": "messages"}}
        else:
            err = {"error": {"message": str(e), "type": "internal_error",
                             "code": "generation_error"}}
        yield f"data: {json.dumps(err)}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        # Always clean up queue state
        await queue_manager.finish_processing()
        if _tid:
            task_registry.finish(
                _tid, "cancelled" if task_registry.is_cancelled(_tid) else "done")

async def generate_chat_response(
    messages: List[Dict],
    model_name: str,
    max_tokens: Optional[int],
    temperature: float,
    top_p: float,
    stop: List[str],
    tools: Optional[List[Dict]],
    current_manager: ModelManager,
    tool_parser: ToolCallParser,
    response_format: Optional[Dict] = None,
    force_reasoning_args: Optional[List[str]] = None,
    enable_thinking: bool = False,
    suppress_reasoning: bool = False,
    reasoning_active: bool = False,
    repeat_penalty: float = 1.0,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
) -> Dict:
    """Generate non-streaming chat completion response."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    
    # Debug: Print what is being passed to the model
    if get_global_debug():
        print(f"\n{'='*80}")
        print(f"=== MODEL INPUT (DEBUG) ===")
        print(f"{'='*80}")
        print(f"Model: {model_name}")
        print(f"Max tokens: {max_tokens}")
        print(f"Temperature: {temperature}")
        print(f"Top P: {top_p}")
        print(f"Stop sequences: {stop}")
        print(f"Tools: {tools is not None}")
        print(f"Response format: {response_format}")
        print(f"\n--- Messages ---")
        for i, msg in enumerate(messages):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            if content and len(content) > 500:
                content = content[:500] + "... [truncated]"
            print(f"[{i}] {role}: {repr(content)}")
        print(f"{'='*80}\n")
    
    try:
        # Use generate_chat for proper chat template handling
        generated_text = await asyncio.to_thread(
            current_manager.generate_chat,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            tools=tools,
            response_format=response_format,
            enable_thinking=enable_thinking,
            repeat_penalty=repeat_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )
        
        # Always filter out malformed content
        generated_text = filter_malformed_content(generated_text)
        
        # Apply repetition filtering to prevent infinite loops
        generated_text = filter_repetition(generated_text)
        
        # Dump raw output if enabled
        global_dump = getattr(global_args, 'dump', False) if global_args else False
        if global_dump:
            print(f"\n{'='*80}")
            print(f"=== RAW MODEL OUTPUT (DUMP) ===")
            print(f"{'='*80}")
            print(_clip_for_log(generated_text))
            print(f"{'='*80}\n")
        
        # Separate the model's thinking from the answer. Surface it as a reasoning
        # field (unless suppress_reasoning), keeping content clean — and so the
        # thought never confuses tool extraction below. Handles the bare-</think>
        # form Qwen-style pre-fill templates produce (see extract_reasoning_content).
        reasoning_text = ""
        if reasoning_active:
            from codai.models.parser import extract_reasoning_content
            reasoning_text, generated_text = extract_reasoning_content(generated_text, model_name)

        response_message = {
            "role": "assistant",
            "content": generated_text,
        }
        if reasoning_text and not suppress_reasoning:
            response_message["reasoning"] = reasoning_text
            response_message["reasoning_content"] = reasoning_text

        finish_reason = "stop"

        # Check for tool calls
        if tools:
            # Convert tools back to Tool objects for parsing
            tool_objects = []
            for t in tools:
                try:
                    # Handle both dict and pydantic model formats
                    if isinstance(t, dict):
                        func_data = t.get("function", {})
                        tool_func = ToolFunction(
                            name=func_data.get("name", ""),
                            description=func_data.get("description"),
                            parameters=func_data.get("parameters")
                        )
                    else:
                        # Pydantic model
                        tool_func = ToolFunction(
                            name=t.function.name if hasattr(t.function, 'name') else str(t.function),
                            description=t.function.description if hasattr(t.function, 'description') else None,
                            parameters=t.function.parameters if hasattr(t.function, 'parameters') else None
                        )
                    tool_objects.append(Tool(type=t.get("type", "function") if isinstance(t, dict) else t.type, function=tool_func))
                except Exception as e:
                    logger.debug("Error converting tool: %s (type: %s)", e, type(t))
                    continue
            try:
                tool_calls = tool_parser.extract_tool_calls(generated_text, tool_objects)
                
                # FIX: Validate extracted tool calls have valid JSON (generate_chat_response)
                if tool_calls:
                    from codai.models.parser import validate_json_complete
                    validated_calls = []
                    for tc in tool_calls:
                        args = tc.get('function', {}).get('arguments', '{}')
                        if isinstance(args, str) and validate_json_complete(args):
                            validated_calls.append(tc)
                        elif isinstance(args, dict):
                            validated_calls.append(tc)
                    if len(validated_calls) != len(tool_calls):
                        logger.debug("Filtered out %d invalid tool calls in generate_chat_response", len(tool_calls) - len(validated_calls))
                    tool_calls = validated_calls if validated_calls else None
            except Exception as e:
                logger.debug("Error extracting tool calls: %s", e)
                tool_calls = None
            if tool_calls:
                # Always strip tool call format from content
                clean_content = tool_parser.strip_tool_calls_from_content(generated_text)
                response_message["content"] = clean_content if clean_content.strip() else None
                response_message["tool_calls"] = tool_calls
                finish_reason = "tool_calls"
        
        # Read accurate usage (including cached_tokens) from the backend
        _model_key_for_cache = getattr(current_manager, 'model_name', None) or model_name
        last_usage = (current_manager.get_last_usage()
                      if hasattr(current_manager, 'get_last_usage') else {})
        prompt_text = "\n".join([m.get("content", "") for m in messages])
        prompt_tokens = last_usage.get('prompt_tokens') or len(prompt_text.split())
        completion_tokens = last_usage.get('completion_tokens') or (
            len(generated_text.split()) if generated_text else 0)
        cached_tokens = last_usage.get('cached_tokens', 0)

        # Store in prompt cache manager for future prefix matching
        prompt_cache_manager.store(messages, _model_key_for_cache,
                                   prompt_tokens, cached_tokens)

        # Get context size
        context_size = current_manager.get_context_size()
        
        # Use OpenAIFormatter for final sanitization
        log_response_exchange(response_message.get("content", ""),
                              tool_calls=response_message.get("tool_calls"),
                              finish_reason=finish_reason, streamed=False,
                              stage="pre-format extracted")
        formatter = OpenAIFormatter(model_name)
        formatted_response = formatter.format_litellm_full(
            text=response_message.get("content", ""),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            tool_calls=response_message.get("tool_calls"),
            context_size=context_size
        )
        # Patch in the real cached_tokens value
        if formatted_response and 'usage' in formatted_response:
            details = formatted_response['usage'].setdefault('prompt_tokens_details', {})
            details['cached_tokens'] = cached_tokens

        # Carry the extracted reasoning onto the formatted message (the formatter
        # doesn't take a reasoning arg here). Emit both field names for client
        # compatibility (`reasoning` and DeepSeek-style `reasoning_content`).
        if reasoning_text and not suppress_reasoning and formatted_response.get("choices"):
            _msg = formatted_response["choices"][0].setdefault("message", {})
            _msg["reasoning"] = reasoning_text
            _msg["reasoning_content"] = reasoning_text

        # Add mock reasoning stats if 'mock' is in force_reasoning_args
        # But only if we don't already have real reasoning in the response
        # Check if reasoning already exists in the message
        existing_reasoning = None
        if "choices" in formatted_response and formatted_response["choices"]:
            choice = formatted_response["choices"][0]
            if "message" in choice:
                existing_reasoning = choice["message"].get("reasoning")
        
        if force_reasoning_args and "mock" in force_reasoning_args and formatted_response and not existing_reasoning:
            # Add fake reasoning tokens to trigger VSCode plugin stats
            mock_reasoning_tokens = 50
            
            # Update usage
            if "usage" in formatted_response:
                formatted_response["usage"]["completion_tokens"] += mock_reasoning_tokens
                formatted_response["usage"]["total_tokens"] += mock_reasoning_tokens
                formatted_response["usage"]["completion_tokens_details"] = {
                    "reasoning_tokens": mock_reasoning_tokens
                }
            
            # Add reasoning to message if not present
            if "choices" in formatted_response and formatted_response["choices"]:
                choice = formatted_response["choices"][0]
                if "message" in choice and "reasoning" not in choice["message"]:
                    choice["message"]["reasoning"] = "Processing task in optimized mode..."
        
        # Dump parsed output if enabled
        if global_dump:
            import json
            print(f"\n{'='*80}")
            print(f"=== PARSED OUTPUT (DUMP) ===")
            print(f"{'='*80}")
            print(_clip_for_log(json.dumps(formatted_response, indent=2)))
            print(f"{'='*80}\n")

        log_response_payload(formatted_response, streamed=False)
        return formatted_response
    except Exception as e:
        print(f"Error during generation: {e}")
        _ctx = _context_overflow_detail(e)
        if _ctx:
            raise HTTPException(status_code=400, detail=_ctx)
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

# =============================================================================
# Legacy Text Completions Endpoint (/v1/completions)
# =============================================================================
# NOTE: This is a legacy endpoint for backward compatibility.
# It uses raw text completion (no chat template) instead of the modern
# /v1/chat/completions API. Consider using /v1/chat/completions instead.
# =============================================================================

from codai.pydantic.textrequest import CompletionRequest


@router.post("/v1/completions", summary="Legacy text completions")
async def completions(request: CompletionRequest):
    """Legacy text completions endpoint (for backward compatibility)."""
    # Get the model for this request
    requested_model = request.model
    
    # Use the manager to resolve the model and manage VRAM (handles ondemand unloading)
    # In a thread: request_model may block (thermal cooldown / waiting for a busy
    # model) and we must not stall the event loop.
    model_info = await asyncio.to_thread(
        multi_model_manager.request_model,
        requested_model=requested_model,
        model_type="text",
    )
    
    # Check if the model was rejected as not allowed
    if model_info.get('error'):
        raise HTTPException(status_code=404, detail=model_info['error'])
    
    # Acquire an instance (session-affinity when derivable, else least-busy;
    # increments ref-count; released on response completion).
    _model_key = model_info.get('model_key')
    _instance_idx = None
    _session_key = _conversation_session_key(request)
    _acq = multi_model_manager.acquire_model_instance(
        _model_key, session_key=_session_key) if _model_key else None
    if _acq:
        _instance_idx, mm = _acq
    else:
        mm = multi_model_manager.get_model_for_request(requested_model)

    def _release_instance():
        if _instance_idx is not None and _model_key:
            multi_model_manager.release_model_instance(_model_key, _instance_idx)

    if mm is None:
        _release_instance()
        if model_manager.backend is not None:
            current_manager = model_manager
        else:
            raise HTTPException(status_code=503, detail="Model not loaded")
    else:
        current_manager = mm

    prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]
    stop_sequences = []
    if request.stop:
        stop_sequences = [request.stop] if isinstance(request.stop, str) else request.stop

    if request.stream:
        async def _managed_completion_stream():
            try:
                async for chunk in stream_completion_response(
                    prompts[0],
                    request.model,
                    request.max_tokens,
                    request.temperature,
                    request.top_p,
                    stop_sequences,
                    current_manager,
                ):
                    yield chunk
            finally:
                _release_instance()

        from fastapi.responses import StreamingResponse
        return StreamingResponse(_managed_completion_stream(), media_type="text/event-stream")
    else:
        try:
            return await generate_completion_response(
                prompts[0],
                request.model,
                request.max_tokens,
                request.temperature,
                request.top_p,
                stop_sequences,
                current_manager,
            )
        finally:
            _release_instance()

async def stream_completion_response(
    prompt: str,
    model_name: str,
    max_tokens: Optional[int],
    temperature: float,
    top_p: float,
    stop: List[str],
    current_manager: ModelManager,
) -> AsyncGenerator[str, None]:
    """Stream legacy completion response."""
    completion_id = f"cmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    generated_text = ""
    
    try:
        async for chunk in current_manager.generate_stream(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        ):
            generated_text += chunk
            data = {
                "id": completion_id,
                "object": "text_completion",
                "created": created,
                "model": model_name,
                "choices": [{
                    "text": chunk,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(data)}\n\n"
        
        # Calculate token counts
        if current_manager.tokenizer:
            prompt_tokens = len(current_manager.tokenizer.encode(prompt))
            completion_tokens = len(current_manager.tokenizer.encode(generated_text))
        else:
            prompt_tokens = len(prompt.split())
            completion_tokens = len(generated_text.split())
        
        # Get context size
        context_size = current_manager.get_context_size()
        
        # Send final chunk with usage
        final_chunk = {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": model_name,
            "choices": [{
                "text": "",
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "context_size": context_size,
            },
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        print(f"Error during streaming completion: {e}")
        yield f"data: {json.dumps({'choices': [{'finish_reason': 'stop'}]})}\n\n"
        yield "data: [DONE]\n\n"

async def generate_completion_response(
    prompt: str,
    model_name: str,
    max_tokens: Optional[int],
    temperature: float,
    top_p: float,
    stop: List[str],
    current_manager: ModelManager,
) -> Dict:
    """Generate non-streaming legacy completion response."""
    completion_id = f"cmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    
    try:
        generated_text = await asyncio.to_thread(
            current_manager.generate,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )
        
        # Calculate token counts if tokenizer available
        if current_manager.tokenizer:
            prompt_tokens = len(current_manager.tokenizer.encode(prompt))
            completion_tokens = len(current_manager.tokenizer.encode(generated_text))
        else:
            prompt_tokens = len(prompt.split())
            completion_tokens = len(generated_text.split())
        
        # Get context size
        context_size = current_manager.get_context_size()
        
        return {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": model_name,
            "choices": [{
                "text": generated_text,
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "context_size": context_size,
            },
        }
    except Exception as e:
        print(f"Error during completion: {e}")
        _ctx = _context_overflow_detail(e)
        if _ctx:
            raise HTTPException(status_code=400, detail=_ctx)
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")
