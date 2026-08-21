# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""vLLM proxy backend.

vLLM serves an OpenAI-compatible HTTP API with continuous batching, so this backend is a
thin proxy (mirroring :class:`~codai.backends.ktransformers.KtransformersBackend`): it
forwards chat/completion requests to the managed vLLM subprocess (lifecycle in
:mod:`codai.api.vllm_worker`, which runs vLLM in an isolated venv).
"""

import asyncio
import threading
from typing import AsyncGenerator, Dict, List, Optional

from codai.backends.base import ModelBackend


class VllmBackend(ModelBackend):
    """Thin OpenAI proxy to a managed vLLM server."""

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
            from codai.config import VllmConfig
            cfg = VllmConfig()
        self._cfg = cfg
        self._model_id = getattr(cfg, "model_id", "vllm") or "vllm"
        self._svc_key: Optional[str] = None
        self._url: Optional[str] = None
        self._ctx = int(getattr(cfg, "ctx", 32768) or 32768)
        self._served_name = self._model_id
        self._last_usage: Dict = {}

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def load_model(self, model_name: str, **kwargs) -> None:
        from codai.api import vllm_worker
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
        model_path = self._resolve_model_dir(model_name)
        _resolved, self._svc_key = vllm_worker.resolve_service_key(self._cfg, model_path)
        self._url = vllm_worker.ensure_service(self._cfg, model_path=model_path)
        self._served_name = getattr(self._cfg, "model_id", "vllm") or "vllm"

    def _resolve_model_dir(self, model_name: str) -> Optional[str]:
        """Resolve the requested model to an HF model dir / id for vLLM."""
        import os
        cand = os.path.expanduser(model_name or "")
        if cand and os.path.isdir(cand):
            return os.path.abspath(cand)
        mp = os.path.expanduser((getattr(self._cfg, "model_path", "") or "").strip())
        if mp and os.path.isdir(mp):
            return os.path.abspath(mp)
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
        # vLLM also accepts a bare HF repo id (not a local dir).
        return mp or (getattr(self._cfg, "model_path", "") or "").strip() or None

    def get_model_name(self) -> str:
        return self._model_id

    def get_context_size(self) -> int:
        return self._ctx

    def get_last_usage(self) -> dict:
        return dict(self._last_usage)

    def cleanup(self) -> None:
        from codai.api import vllm_worker
        key = getattr(self, "_svc_key", None) or getattr(self._cfg, "model_id", self._model_id)
        vllm_worker.stop_service(key)
        self._url = None

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _base(self) -> str:
        if not self._url:
            raise RuntimeError("vLLM service not started")
        return self._url

    def _store_usage(self, usage: dict) -> None:
        if usage:
            self._last_usage = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }

    def format_messages(self, messages) -> str:
        parts = []
        for m in messages:
            role = m.get("role") if isinstance(m, dict) else getattr(m, "role", "")
            content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    def _chat_payload(self, messages, max_tokens, temperature, top_p, stop, stream):
        payload = {
            "model": getattr(self, "_served_name", self._model_id),
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if stop:
            payload["stop"] = stop
        return payload

    # ------------------------------------------------------------------ #
    # chat-level generation
    # ------------------------------------------------------------------ #
    def generate_chat(self, messages: List[Dict], max_tokens=None, temperature=0.7,
                      top_p=1.0, stop=None, tools=None, response_format=None):
        import requests
        self._enter_request()
        try:
            payload = self._chat_payload(messages, max_tokens, temperature, top_p, stop, False)
            if response_format and response_format.get("type") == "json_object":
                payload["response_format"] = {"type": "json_object"}
            if tools:
                payload["tools"] = tools
            r = requests.post(self._base() + "/v1/chat/completions", json=payload, timeout=3600)
            r.raise_for_status()
            data = r.json()
            self._store_usage(data.get("usage", {}))
            return data["choices"][0]["message"].get("content") or ""
        finally:
            self._exit_request()

    async def generate_chat_stream(self, messages: List[Dict], max_tokens=None,
                                   temperature=0.7, top_p=1.0, stop=None, tools=None,
                                   response_format=None) -> AsyncGenerator[str, None]:
        self._enter_request()
        try:
            payload = self._chat_payload(messages, max_tokens, temperature, top_p, stop, True)
            if tools:
                payload["tools"] = tools
            async for chunk in self._stream(self._base() + "/v1/chat/completions", payload,
                                            delta_key="delta"):
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
    # SSE streaming
    # ------------------------------------------------------------------ #
    async def _stream(self, url: str, payload: dict, delta_key: str
                      ) -> AsyncGenerator[str, None]:
        import json
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()
        _SENTINEL = object()

        def _handle_line(line: str) -> bool:
            if not line or not line.startswith("data:"):
                return False
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                return True
            try:
                obj = json.loads(data)
            except ValueError:
                return False
            choice = (obj.get("choices") or [{}])[0]
            text = (choice.get(delta_key) or {}).get("content") or ""
            if text:
                loop.call_soon_threadsafe(queue.put_nowait, text)
            if obj.get("usage"):
                self._store_usage(obj["usage"])
            return bool(choice.get("finish_reason"))

        def _worker():
            import requests
            try:
                with requests.post(url, json=payload, stream=True, timeout=3600) as r:
                    r.raise_for_status()
                    buf = b""
                    done = False
                    for bchunk in r.iter_content(chunk_size=8192):
                        if not bchunk:
                            continue
                        buf += bchunk
                        while b"\n" in buf:
                            raw, buf = buf.split(b"\n", 1)
                            if _handle_line(raw.decode("utf-8", "replace").strip()):
                                done = True
                                break
                        if done:
                            break
                    if not done and buf.strip():
                        _handle_line(buf.decode("utf-8", "replace").strip())
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

        threading.Thread(target=_worker, daemon=True).start()
        while True:
            item = await queue.get()
            if item is _SENTINEL:
                break
            if isinstance(item, Exception):
                raise item
            yield item
