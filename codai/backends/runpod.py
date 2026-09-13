# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""RunPod remote-GPU proxy backend.

A thin OpenAI proxy (mirroring :class:`codai.backends.vllm.VllmBackend`) that
forwards chat/completion requests to a model running on RunPod — either a
serverless endpoint (RunPod autoscales) or a coderai-managed pod pool. It uses
NO local VRAM and downloads NOTHING: the remote worker holds the weights.

Lifecycle + URL resolution live in :mod:`codai.api.runpod_worker`; account
settings (API key, endpoints, caps) in :class:`codai.config.RunpodConfig`.

Phase 2: the serverless path is live. Pods mode raises a clear error until the
pod pool lands in the next phase.
"""

import asyncio
import threading
from typing import AsyncGenerator, Dict, List, Optional

from codai.backends.base import ModelBackend


class RunpodBackend(ModelBackend):
    """Thin OpenAI proxy to a model hosted on RunPod."""

    # RunPod requests run on REMOTE GPUs, so they must not count against the local
    # global concurrency gate. This class-level inflight is informational only.
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

    # RunPod models never occupy local VRAM.
    uses_local_vram = False

    def __init__(self, cfg=None):
        if cfg is None:
            from codai.config import RunpodConfig
            cfg = RunpodConfig()
        self._acct = cfg
        self._model_id = "runpod"
        self._served_name = "runpod"
        self._url: Optional[str] = None          # serverless: fixed base URL
        self._headers: Dict[str, str] = {}
        self._ctx = 32768
        self._mcfg = None
        self._mode = "serverless"
        self._pool = None                         # pods: RunpodPodPool
        self._last_usage: Dict = {}

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def load_model(self, model_name: str, **kwargs) -> None:
        """Resolve the remote endpoint for ``model_name``. Downloads nothing."""
        from codai.api import runpod_worker
        if model_name:
            self._model_id = model_name
        block = runpod_worker.model_runpod_block(model_name)
        self._mcfg = runpod_worker.parse_model_runpod(block)
        # served name: explicit served_model wins, else the model's own name.
        self._served_name = self._mcfg.served_model or model_name or "runpod"
        if self._mcfg.ctx:
            self._ctx = self._mcfg.ctx
        _ctx = kwargs.get("n_ctx", kwargs.get("ctx"))
        if isinstance(_ctx, (list, tuple)):
            _ctx = _ctx[0] if _ctx else None
        try:
            _ctx = int(_ctx) if _ctx else 0
        except (TypeError, ValueError):
            _ctx = 0
        if _ctx > 0:
            self._ctx = _ctx

        self._mode = (self._mcfg.mode or "pods").lower()
        if self._mode == "serverless":
            self._url = runpod_worker.serverless_base_url(self._acct, self._mcfg)
            self._headers = runpod_worker.auth_headers(self._acct)
            print(f"[runpod] '{model_name}' -> serverless endpoint "
                  f"{self._mcfg.endpoint_id} ({self._url})", flush=True)
        else:
            # pods / auto: coderai-managed pool of remote GPU pods. (auto currently
            # behaves as pods; serverless-vs-pods auto-arbitration is a later refinement.)
            self._pool = runpod_worker.get_pod_pool(
                model_name, self._acct, self._mcfg, self._served_name)
            # A RunPod proxy URL is reachable by anyone who learns it, so the pod
            # is launched locked to a bearer token (generated when none is
            # configured) and every request we send must carry it.
            _key = getattr(self._pool, "api_key", "")
            self._headers = {"Authorization": f"Bearer {_key}"} if _key else {}
            # Eagerly warm a pod only when min_pods >= 1; otherwise provision lazily
            # on the first request (min_pods=0 = scale-to-zero when idle).
            if self._mcfg.min_pods >= 1:
                self._pool.ensure_ready()
            print(f"[runpod] '{model_name}' -> managed pod pool "
                  f"(min={self._mcfg.min_pods}, max={self._mcfg.max_pods})", flush=True)

    def get_model_name(self) -> str:
        return self._model_id

    def get_context_size(self) -> int:
        return self._ctx

    def get_last_usage(self) -> dict:
        return dict(self._last_usage)

    def cleanup(self) -> None:
        # Serverless: nothing to tear down (RunPod owns it). Pods: the pool
        # reaper handles teardown (next phase).
        self._url = None

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _base(self) -> str:
        if not self._url:
            raise RuntimeError("RunPod endpoint not resolved")
        return self._url

    def _acquire_endpoint(self):
        """Return (base_url, release_callable) for one request. Serverless yields the
        fixed endpoint; pods pick (and provision-on-demand) a pod from the pool."""
        if self._mode == "serverless":
            return self._base(), (lambda: None)
        if not self._pool:
            raise RuntimeError("RunPod pod pool not initialised")
        pod, url = self._pool.acquire()
        # The pod runs a vLLM OpenAI server at /v1; its proxy URL has no path, so
        # add /v1 to match the serverless base (which already ends in /openai/v1).
        return url.rstrip("/") + "/v1", (lambda: self._pool.release(pod))

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
            "model": self._served_name,
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
        base, release = self._acquire_endpoint()
        try:
            payload = self._chat_payload(messages, max_tokens, temperature, top_p, stop, False)
            if response_format and response_format.get("type") == "json_object":
                payload["response_format"] = {"type": "json_object"}
            if tools:
                payload["tools"] = tools
            r = requests.post(base + "/chat/completions", json=payload,
                              headers=self._headers, timeout=3600)
            r.raise_for_status()
            data = r.json()
            self._store_usage(data.get("usage", {}))
            return data["choices"][0]["message"].get("content") or ""
        finally:
            release()
            self._exit_request()

    async def generate_chat_stream(self, messages: List[Dict], max_tokens=None,
                                   temperature=0.7, top_p=1.0, stop=None, tools=None,
                                   response_format=None) -> AsyncGenerator[str, None]:
        self._enter_request()
        # Pod acquire may provision (blocking) — do it off the event loop.
        base, release = await asyncio.to_thread(self._acquire_endpoint)
        try:
            payload = self._chat_payload(messages, max_tokens, temperature, top_p, stop, True)
            if tools:
                payload["tools"] = tools
            async for chunk in self._stream(base + "/chat/completions", payload,
                                            delta_key="delta"):
                yield chunk
        finally:
            release()
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
        headers = self._headers

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
                with requests.post(url, json=payload, headers=headers,
                                   stream=True, timeout=3600) as r:
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
