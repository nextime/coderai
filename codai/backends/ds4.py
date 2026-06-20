# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""ds4 (DeepSeek V4) proxy backend.

ds4-server already speaks the OpenAI HTTP API, so this backend is a thin proxy: it
forwards chat/completion requests to the managed ``ds4-server`` subprocess (whose
lifecycle is owned by :mod:`codai.api.ds4_worker`) and adapts the responses to the
:class:`~codai.backends.base.ModelBackend` contract the model manager expects.

Tool/think parsing is handled the same way as the other backends — by
``ModelParserAdapter`` over the returned text — so tools are not forwarded to
ds4-server; the text-level ``DeepSeekParser`` extracts ``<think>`` and tool calls.
"""

import asyncio
import threading
from typing import AsyncGenerator, Dict, List, Optional

from codai.backends.base import ModelBackend


class Ds4Backend(ModelBackend):
    """Proxy backend that routes generation to a managed ds4-server."""

    # Process-wide count of in-flight ds4 requests (across all backend instances).
    # The KV-cache janitor checks this so it never sweeps mid-request — see
    # codai.api.ds4_kv_janitor. Incremented/decremented around every generate path.
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
        """True while any ds4 request is being served (prefill/decode in flight)."""
        with cls._inflight_lock:
            return cls._inflight > 0

    def __init__(self, cfg=None):
        # cfg is a codai.config.Ds4Config. When omitted, resolve the active one.
        if cfg is None:
            from codai.config import Ds4Config
            cfg = Ds4Config()
        self._cfg = cfg
        self._model_id = getattr(cfg, "model_id", "deepseek-v4") or "deepseek-v4"
        self._svc_key: Optional[str] = None   # ds4_worker service key (file or model_id)
        self._url: Optional[str] = None
        self._ctx = int(getattr(cfg, "ctx", 100000) or 100000)
        self._last_usage: Dict = {}

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def load_model(self, model_name: str, **kwargs) -> None:
        from codai.api import ds4_worker
        if model_name:
            self._model_id = model_name
        # Honour the model's configured context window (n_ctx / ctx from its
        # models.json entry, forwarded by the manager) over the ds4 global ctx.
        _ctx = kwargs.get("n_ctx", kwargs.get("ctx"))
        if isinstance(_ctx, (list, tuple)):
            _ctx = _ctx[0] if _ctx else None
        try:
            _ctx = int(_ctx) if _ctx else 0
        except (TypeError, ValueError):
            _ctx = 0
        if _ctx > 0:
            self._ctx = _ctx
        # Resolve the requested model to a concrete .gguf so ds4-server serves THE
        # deepseek4 model that was asked for (not a fixed downloaded variant).
        model_file = self._resolve_gguf(model_name)
        # Per-model ds4 launch overrides (streaming / expert-cache reserve / extra
        # args+env) from THIS model's own config entry — these vary by quant and
        # context, so they live per-model. Unset fields inherit the global ds4
        # config, which acts as the default/template.
        overrides = self._ds4_overrides(model_name, model_file)
        if overrides:
            import dataclasses
            try:
                self._cfg = dataclasses.replace(self._cfg, **overrides)
                print(f"[ds4] per-model overrides for '{model_name}': "
                      + ", ".join(f"{k}={v!r}" for k, v in overrides.items()), flush=True)
            except Exception as exc:
                print(f"[ds4] failed to apply per-model overrides: {exc}", flush=True)
        _resolved, self._svc_key = ds4_worker.resolve_service_key(self._cfg, model_file)
        self._url = ds4_worker.ensure_service(
            self._cfg, model_file=model_file, ctx=(self._ctx or None),
            model_name=model_name)

    @staticmethod
    def _ds4_overrides(model_name: str, model_file: Optional[str]) -> Dict:
        """Per-model ds4 overrides from the model's own models.json entry.

        Looks up THIS model's config entry (by resolved path or name) and returns
        the subset of :class:`Ds4Config` fields its optional ``ds4`` block sets:
        ``ssd_streaming``, ``expert_cache_reserve_gb``, ``extra_args``,
        ``extra_env``. Unset/blank fields are omitted so the global ds4 config
        stays the default. Best-effort: any failure yields no overrides."""
        out: Dict = {}
        try:
            import os
            from codai.admin.routes import config_manager
            md = getattr(config_manager, "models_data", {}) or {}
            target = os.path.basename(os.path.expanduser(model_file or "")) or None
            name_l = (model_name or "").strip().lower()
            entry = None
            for lst in md.values():
                if not isinstance(lst, list):
                    continue
                for m in lst:
                    if not isinstance(m, dict):
                        continue
                    path = str(m.get("path") or m.get("id") or "")
                    base = os.path.basename(path)
                    stem = base[:-5] if base.lower().endswith(".gguf") else base
                    cands = {path.lower(), base.lower(), stem.lower(),
                             str(m.get("alias") or "").lower()}
                    if (target and base == target) or (name_l and name_l in cands):
                        entry = m
                        break
                if entry:
                    break
            ds4o = entry.get("ds4") if entry and isinstance(entry.get("ds4"), dict) else None
            if not ds4o:
                return out
            if ds4o.get("ssd_streaming") is not None:
                out["ssd_streaming"] = bool(ds4o["ssd_streaming"])
            rg = ds4o.get("expert_cache_reserve_gb")
            if rg not in (None, "", 0, "0"):
                try:
                    out["expert_cache_reserve_gb"] = max(0, int(rg))
                except (TypeError, ValueError):
                    pass
            for k in ("extra_args", "extra_env"):
                v = ds4o.get(k)
                if v and str(v).strip():
                    out[k] = str(v).strip()
        except Exception:
            pass
        return out

    @staticmethod
    def _resolve_gguf(model_name: str):
        """Map a requested model name/alias/path/basename to a local .gguf path.

        Delegates to the manager's config/cache-aware resolver so a bare client id
        (e.g. "Foo-ds4-Q2_K", no path/extension) resolves to the configured file.
        """
        try:
            from codai.models.manager import _resolve_local_gguf
            return _resolve_local_gguf(model_name)
        except Exception:
            import os
            cand = os.path.expanduser(model_name or "")
            return cand if cand.lower().endswith(".gguf") and os.path.isfile(cand) else None

    def get_model_name(self) -> str:
        return self._model_id

    def get_context_size(self) -> int:
        return self._ctx

    def get_last_usage(self) -> dict:
        return dict(self._last_usage)

    def cleanup(self) -> None:
        from codai.api import ds4_worker
        key = getattr(self, "_svc_key", None) or getattr(self._cfg, "model_id", self._model_id)
        ds4_worker.stop_service(key)
        self._url = None

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _base(self) -> str:
        if not self._url:
            raise RuntimeError("ds4 service not started")
        return self._url

    def _store_usage(self, usage: dict) -> None:
        if usage:
            self._last_usage = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }

    def format_messages(self, messages) -> str:
        # ds4-server applies DeepSeek V4's own chat template server-side; this is only
        # used by callers that need a flat prompt string.
        parts = []
        for m in messages:
            role = m.get("role") if isinstance(m, dict) else getattr(m, "role", "")
            content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    def _chat_payload(self, messages, max_tokens, temperature, top_p, stop, stream):
        payload = {
            "model": self._model_id,
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
    # chat-level generation (preferred by the manager)
    # ------------------------------------------------------------------ #
    def generate_chat(self, messages: List[Dict], max_tokens=None, temperature=0.7,
                      top_p=1.0, stop=None, tools=None, response_format=None):
        import requests
        self._enter_request()
        try:
            payload = self._chat_payload(messages, max_tokens, temperature, top_p, stop, False)
            if response_format and response_format.get("type") == "json_object":
                payload["response_format"] = {"type": "json_object"}
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
    # SSE streaming: iterate the blocking requests stream on a worker thread
    # and hand chunks to the event loop through an asyncio.Queue.
    # ------------------------------------------------------------------ #
    async def _stream(self, url: str, payload: dict, delta_key: str
                      ) -> AsyncGenerator[str, None]:
        import json
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()
        _SENTINEL = object()

        def _handle_line(line: str) -> bool:
            """Process one SSE line; return True when the stream should stop."""
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
                    # Parse the SSE byte stream ourselves and split on the b"\n"
                    # byte. requests' iter_lines(decode_unicode=True) decodes each
                    # network chunk independently, so a multibyte UTF-8 char split
                    # across chunks gets corrupted — which breaks that event's JSON
                    # and silently drops the token (mangled '—', truncated tails).
                    # b"\n" (0x0A) can never fall inside a UTF-8 multibyte sequence,
                    # so splitting bytes first and decoding whole lines is safe.
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
                    # Flush any final line the stream left without a trailing newline.
                    if not done and buf.strip():
                        _handle_line(buf.decode("utf-8", "replace").strip())
            except Exception as exc:  # surface to the consumer
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
