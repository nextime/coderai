# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Kimi-K3 backend — the kimi-k3-in-c C engine, driven in-process.

Mirrors :class:`~codai.backends.colibri.ColibriBackend` but for a single family
(Kimi-K3) served by the patched ``kimi-k3-in-c`` engine. The engine tokenizes a
rendered XTML prompt string (unlike colibri's Kimi ``K3CHAT1`` wire), so this backend
renders with :func:`~codai.backends.colibri_families.render_kimi_xtml` and drives the
engine through :class:`~codai.api.colibri_worker.MuxEngine` (managed by
:mod:`codai.api.k3_worker`). Tool/think parsing is handled the usual way over the
returned text.
"""

import asyncio
import threading
from typing import AsyncGenerator, Dict, List, Optional

from codai.backends.base import ModelBackend
from codai.backends.colibri_families import render_kimi_xtml

# End-of-turn / structural markers to trim from the visible reply.
_K3_STOP_MARKERS = ("<|close|>response", "<|close|>message", "<|close|>",
                    "<|end_of_msg|>", "<|open|>")
_K3_STRIP_TOKENS = ("<|open|>", "<|close|>", "<|sep|>", "<|end_of_msg|>")


def _k3_cut_index(text: str) -> int:
    cut = len(text)
    for m in _K3_STOP_MARKERS:
        i = text.find(m)
        if i != -1 and i < cut:
            cut = i
    return cut


def clean_k3_output(text: str) -> str:
    """Trim a Kimi-K3 completion at its response-channel boundary and strip XTML tokens.

    With the backend's default non-thinking prompt the engine generates the response
    directly, so the visible stream is the answer up to ``<|close|>response`` /
    ``<|end_of_msg|>``; any residual XTML control tokens are removed defensively.
    """
    if not text:
        return text
    text = text[:_k3_cut_index(text)]
    for t in _K3_STRIP_TOKENS:
        text = text.replace(t, "")
    return text


class K3Backend(ModelBackend):
    """In-process backend that drives a managed kimi-k3-in-c engine (Kimi-K3)."""

    _inflight = 0
    _inflight_lock = threading.Lock()

    @classmethod
    def _enter_request(cls):
        with cls._inflight_lock:
            cls._inflight += 1

    @classmethod
    def _exit_request(cls):
        with cls._inflight_lock:
            cls._inflight = max(0, cls._inflight - 1)

    @classmethod
    def any_request_active(cls) -> bool:
        with cls._inflight_lock:
            return cls._inflight > 0

    def __init__(self, cfg=None):
        if cfg is None:
            from codai.config import K3Config
            cfg = K3Config()
        self._cfg = cfg
        self._model_id = getattr(cfg, "model_id", "kimi-k3") or "kimi-k3"
        self._svc_key: Optional[str] = None
        self._engine = None
        self._ctx = int(getattr(cfg, "ctx", 4096) or 4096)
        # Default to the response-only channel so the visible stream is the answer, not
        # the structural think channel (kimi-k3-in-c serve emits raw token text).
        self._enable_thinking = False
        self._last_usage: Dict = {}

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def load_model(self, model_name: str, **kwargs) -> None:
        from codai.api import k3_worker
        if model_name:
            self._model_id = model_name
        _ctx = kwargs.get("n_ctx", kwargs.get("ctx"))
        if isinstance(_ctx, (list, tuple)):
            _ctx = _ctx[0] if _ctx else None
        try:
            _ctx = int(_ctx) if _ctx else 0
        except (TypeError, ValueError):
            _ctx = 0
        if _ctx > 0:
            self._ctx = _ctx
        model_dir = self._resolve_checkpoint(model_name)
        _resolved, self._svc_key = k3_worker.resolve_service_key(self._cfg, model_dir)
        self._engine = k3_worker.ensure_engine(
            self._cfg, model_dir=model_dir, ctx=(self._ctx or None))

    def _resolve_checkpoint(self, model_name: str) -> Optional[str]:
        """Map the requested model name/alias/path to its Kimi-K3 checkpoint directory."""
        import os
        cand = os.path.expanduser(model_name or "")
        if cand and os.path.isdir(cand):
            return os.path.abspath(cand)
        mp = os.path.expanduser((getattr(self._cfg, "model_path", "") or "").strip())
        if mp and os.path.isdir(mp):
            return os.path.abspath(mp)
        # A models.json entry's path for this name/alias.
        try:
            from codai.admin.routes import config_manager
            md = getattr(config_manager, "models_data", {}) or {}
            name_l = (model_name or "").strip().lower()
            for lst in md.values():
                if not isinstance(lst, list):
                    continue
                for m in lst:
                    if not isinstance(m, dict):
                        continue
                    path = str(m.get("path") or "")
                    base = os.path.basename(path.rstrip("/")).lower()
                    cands = {path.lower(), base, str(m.get("alias") or "").lower(),
                             str(m.get("id") or "").lower()}
                    if name_l in cands and path and os.path.isdir(os.path.expanduser(path)):
                        return os.path.abspath(os.path.expanduser(path))
        except Exception:
            pass
        return None

    def get_model_name(self) -> str:
        return self._model_id

    def get_context_size(self) -> int:
        return self._ctx

    def get_last_usage(self) -> dict:
        return dict(self._last_usage)

    def cleanup(self) -> None:
        from codai.api import k3_worker
        key = getattr(self, "_svc_key", None) or getattr(self._cfg, "model_id", self._model_id)
        k3_worker.stop_service(key)
        self._engine = None

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _need_engine(self):
        if self._engine is None or not self._engine.is_alive():
            self.load_model(self._model_id)
        return self._engine

    def _store_usage(self, stats: dict) -> None:
        if stats:
            pt = int(stats.get("prompt_tokens", 0) or 0)
            ct = int(stats.get("completion_tokens", 0) or 0)
            self._last_usage = {"prompt_tokens": pt, "completion_tokens": ct,
                                "total_tokens": pt + ct}

    def format_messages(self, messages) -> str:
        return render_kimi_xtml(messages, enable_thinking=self._enable_thinking)

    # ------------------------------------------------------------------ #
    # chat-level generation
    # ------------------------------------------------------------------ #
    def generate_chat(self, messages: List[Dict], max_tokens=None, temperature=0.7,
                      top_p=1.0, stop=None, tools=None, response_format=None):
        self._enter_request()
        try:
            engine = self._need_engine()
            prompt = render_kimi_xtml(messages, enable_thinking=self._enable_thinking)
            chunks: List[str] = []

            def _hit_turn_boundary():
                joined = "".join(chunks)
                return _k3_cut_index(joined) < len(joined)

            stats = engine.run(prompt, int(max_tokens or 1024), float(temperature),
                               float(top_p), on_text=chunks.append,
                               cancelled=_hit_turn_boundary)
            self._store_usage(stats)
            return clean_k3_output("".join(chunks))
        finally:
            self._exit_request()

    async def generate_chat_stream(self, messages: List[Dict], max_tokens=None,
                                   temperature=0.7, top_p=1.0, stop=None, tools=None,
                                   response_format=None) -> AsyncGenerator[str, None]:
        self._enter_request()
        try:
            engine = self._need_engine()
            prompt = render_kimi_xtml(messages, enable_thinking=self._enable_thinking)
            async for chunk in self._stream(engine, prompt, int(max_tokens or 1024),
                                            float(temperature), float(top_p)):
                yield chunk
        finally:
            self._exit_request()

    # ------------------------------------------------------------------ #
    # plain completion (fallback path)
    # ------------------------------------------------------------------ #
    def generate(self, prompt: str, max_tokens=None, temperature: float = 0.7,
                 top_p: float = 1.0, stop=None, repeat_penalty: float = 1.0,
                 presence_penalty: float = 0.0, frequency_penalty: float = 0.0) -> str:
        return self.generate_chat([{"role": "user", "content": prompt}],
                                  max_tokens, temperature, top_p, stop)

    async def generate_stream(self, prompt: str, max_tokens=None, temperature: float = 0.7,
                              top_p: float = 1.0, stop=None, repeat_penalty: float = 1.0,
                              presence_penalty: float = 0.0,
                              frequency_penalty: float = 0.0) -> AsyncGenerator[str, None]:
        async for chunk in self.generate_chat_stream(
                [{"role": "user", "content": prompt}], max_tokens, temperature, top_p, stop):
            yield chunk

    # ------------------------------------------------------------------ #
    # SSE streaming (mirrors ColibriBackend._stream)
    # ------------------------------------------------------------------ #
    async def _stream(self, engine, prompt: str, max_tokens: int, temperature: float,
                      top_p: float) -> AsyncGenerator[str, None]:
        loop = asyncio.get_event_loop()
        out_queue: asyncio.Queue = asyncio.Queue()
        _SENTINEL = object()
        raw: List[str] = []

        def _hit_turn_boundary():
            joined = "".join(raw)
            return _k3_cut_index(joined) < len(joined)

        def _on_text(text: str):
            if text:
                raw.append(text)
                loop.call_soon_threadsafe(out_queue.put_nowait, True)

        def _worker():
            try:
                stats = engine.run(prompt, max_tokens, temperature, top_p,
                                   on_text=_on_text, cancelled=_hit_turn_boundary)
                self._store_usage(stats)
            except Exception as exc:
                loop.call_soon_threadsafe(out_queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(out_queue.put_nowait, _SENTINEL)

        threading.Thread(target=_worker, daemon=True).start()
        _HOLD = 16
        yielded = 0
        while True:
            item = await out_queue.get()
            if item is _SENTINEL:
                break
            if isinstance(item, Exception):
                raise item
            cleaned = clean_k3_output("".join(raw))
            safe = max(yielded, len(cleaned) - _HOLD)
            if safe > yielded:
                yield cleaned[yielded:safe]
                yielded = safe
        cleaned = clean_k3_output("".join(raw))
        if len(cleaned) > yielded:
            yield cleaned[yielded:]
