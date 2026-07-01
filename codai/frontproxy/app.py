# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""The front: a thin, always-responsive reverse proxy in front of the engines.

It imports no torch/transformers/diffusers, so its event loop is never blocked by
model work. It streams requests/responses (incl. SSE) to the engine chosen by
:mod:`codai.frontproxy.router`, and serves an aggregated, cached status so the web
UI stays live even while an engine is busy loading a model.
"""

import json
import sys
import time
from typing import Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from codai.frontproxy.registry import EngineRegistry
from codai.frontproxy.engine_supervisor import EngineSupervisor
from codai.frontproxy import router as _router
from codai.frontproxy.reqqueue import FrontQueue, QueueFull

# Hop-by-hop headers that must not be forwarded verbatim (RFC 7230 §6.1) plus
# length/host headers that the client/StreamingResponse recompute.
_HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
}
# Also strip any client-supplied internal token so a caller can't spoof/override the
# real one the front injects — only the front's httpx default header reaches engines.
_DROP_REQ = _HOP_BY_HOP | {"host", "content-length", "x-coderai-internal",
                          "x-coderai-broker-authed"}
# Also drop date/server from relayed engine responses: the front's own ASGI server
# (uvicorn/starlette) adds its own Date and Server headers, so keeping the engine's
# too produces DUPLICATE header lines — which nginx logs as a warning on every
# request, flooding the terminal. Strip them here so each appears exactly once.
_DROP_RESP = _HOP_BY_HOP | {"content-length", "date", "server"}

# Admin paths handled by the coderai-system worker (cache scan, HF downloads, cache
# management) rather than a GPU engine — so they stay responsive during generation
# and survive engine restarts. Matched as substrings of the request path.
_SYSTEM_PATHS = (
    "/admin/api/cached-models", "/admin/api/cache-stats", "/admin/api/cache",
    "/admin/api/model-download", "/admin/api/download-stream",
    "/admin/api/downloads", "/admin/api/download-cancel",
    "/admin/api/model-upload", "/admin/api/model-free-disk",
    "/admin/api/hf-search", "/admin/api/hf-files", "/admin/api/hf-model-info",
    "/admin/api/hf-model-files", "/admin/api/ds4/default-models",
    "/admin/api/model-add-known", "/admin/api/model-mark-download",
    "/admin/api/model-unmark-download",
    # Config-based reads/edits (no GPU): the system worker has config_manager +
    # models.json, so these stay fast and never touch the busy engine.
    "/admin/api/models", "/admin/api/accel-presets", "/admin/api/accel-loras",
    "/admin/api/turboquant-info", "/admin/api/model-enable",
    "/admin/api/model-disable",
)


class FrontProxy:
    def __init__(self, config, config_dir=None):
        self.config = config
        self.default_engine = getattr(config.server, "default_engine", None)
        # Per-model engine pins are read from models.json (torch-free) and refreshed
        # when the file changes, so admin edits take effect without a front restart.
        import os
        self._models_path = os.path.join(config_dir, "models.json") if config_dir else None
        self._pins: dict = {}
        self._pins_mtime: float = -1.0
        # The front owns config as the authority for reads (settings GET, status,
        # capacity). A settings SAVE still runs on the engine (it applies live
        # runtime changes — thermal, RAM monitor — that only exist there) and
        # persists config.json; the front re-reads it on mtime change so everything
        # it serves stays current. self.config is mutated in place so components
        # holding a reference (admin_data, supervisor) see the refresh.
        self._config_dir = config_dir
        self._config_path = os.path.join(config_dir, "config.json") if config_dir else None
        self._config_mtime: float = -1.0
        # Last-good /v1/models list per engine. An engine that's mid-load is
        # GIL-blocked and misses health polls; without this its models (incl. a
        # freshly-added one) would flicker out of the aggregated /v1/models while
        # it loads. Keyed by engine name.
        self._engine_models_cache: dict = {}
        # Front-managed generation queue: admission control + ordering + queue
        # position, sized per-model to max_instances so the engine never queues.
        self.reqqueue = FrontQueue()
        # Per-shared-GPU model-swap scheduler (GGUF-isolation split): serialize which
        # model owns a shared card so two forwards never contend for VRAM, batching
        # same-model requests before swapping. Keyed by the engines' shared-GPU
        # selector; created lazily for engines that actually have a co-located sibling.
        self._swap_gates = {}
        # Recent inference activity (front-tracked, since the front relays every
        # request) so the Overview dashboard's activity table is served natively
        # without asking the engine. Newest first; bounded.
        import collections as _collections
        self._recent_activity = _collections.deque(maxlen=25)
        # Last-good per-model instance detail from an engine (synced only when the
        # engine is idle), so model-loaded-status can serve loaded/running from the
        # front's own state without ever hitting a busy engine.
        self._mls_cache: dict = {}
        self.registry = EngineRegistry()
        self.supervisor: Optional[EngineSupervisor] = None
        # Per-run secret shared only with the engines (passed via env at spawn). The
        # front stamps every engine request with it and engines reject requests that
        # lack it, so nothing on localhost can talk to an engine bypassing the front.
        import secrets
        self.internal_token = secrets.token_urlsafe(32)
        _auth = {"x-coderai-internal": self.internal_token}
        # Short client for status/UI; long client (no read timeout) for generation
        # that may legitimately wait for a model load.
        self._short = httpx.AsyncClient(timeout=config.server.proxy_status_timeout,
                                        headers=_auth)
        self._long = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=None, write=None, pool=None),
            headers=_auth)
        # Last-good primary task list. The primary engine can't answer the short
        # tasks poll while it's GIL-busy generating, so without this its running
        # generation would vanish from the Tasks page on every poll under load.
        self._primary_tasks_cache: list = []
        self._primary_tasks_at: float = 0.0
        self._broker = None
        self.debug_engine = False   # --debug-engine: verbose engine lifecycle

    async def aclose(self):
        await self._short.aclose()
        await self._long.aclose()

    # ------------------------------------------------------------------ broker
    def start_broker(self):
        """Run the AISBF broker here in the front (always-responsive, one
        registration for the whole node) instead of inside a model engine. Brokered
        requests are dispatched to the right engine through the same router/proxy."""
        cfg = getattr(self.config, "broker", None)
        if cfg is None or not getattr(cfg, "enabled", False):
            print("[front] AISBF broker not started (broker.enabled is false in config)",
                  flush=True)
            return
        try:
            from codai.broker import build_broker_runtime_config, BrokerConfigError
            from codai.broker.client import BrokerClient
            from codai.broker.service import BrokerService
            from codai.broker.dispatcher import execute_broker_request
        except Exception as exc:
            print(f"[front] broker not available: {exc}", flush=True)
            return
        try:
            runtime = build_broker_runtime_config(cfg)
        except BrokerConfigError as exc:
            print(f"[front] broker disabled (invalid config): {exc}", flush=True)
            return
        if not runtime.enabled:
            return
        client = BrokerClient(runtime)

        async def _dispatch(message):
            envelope = client.message_to_envelope(message)
            return await execute_broker_request(None, envelope,
                                                executor=self.broker_execute)
        client.dispatcher = _dispatch

        # Streaming dispatcher: for stream=true inference, yield engine SSE chunks so
        # the broker client relays them token-by-token (chunk envelopes) instead of
        # buffering the whole reply.
        from codai.broker.dispatcher import (resolve_broker_request,
                                             BrokerDispatchError)

        async def _stream_dispatch(message):
            envelope = client.message_to_envelope(message)
            try:
                headers, body = resolve_broker_request(envelope)
            except BrokerDispatchError:
                yield 'data: {"error":"unsupported broker request"}\n\n'
                return
            async for chunk in self.broker_execute_stream(
                    method=envelope.method, path=envelope.path, headers=headers,
                    query=envelope.query, body=body):
                yield chunk
        client.stream_dispatcher = _stream_dispatch
        # Gate the out-of-band `pending` keepalives by the per-model / global
        # load_status_updates flag (default on). Returns True when the model is
        # unknown so the relay deadline is still protected by default.
        client.status_gate = self._load_status_enabled
        self._broker = BrokerService(client)   # app=None → keep our dispatcher
        self._broker.start()
        print("[front] AISBF broker started (front-managed, routes to engines)",
              flush=True)

    async def stop_broker(self):
        if self._broker is not None:
            await self._broker.stop()
            self._broker = None

    async def collect_models(self, headers):
        """Full node model list. Served from the coderai-system worker (config-based
        list_models, off the GPU engines) in a single fast call, so /v1/models never
        fans out to — or stalls on — a busy engine. Falls back to a per-engine union
        only if the worker is unavailable. Returns ("ok", {...}) or ("passthrough",
        httpx.Response)."""
        sysw = self._system_engine()
        if sysw is not None:
            try:
                r = await self._short.get(sysw.url + "/v1/models", headers=headers)
                if r.status_code == 200:
                    data = r.json().get("data") or []
                    self._engine_models_cache["__system__"] = data  # last-good
                    return ("ok", {"object": "list", "data": data})
            except Exception:
                pass
            # Worker briefly unreachable: serve its last-good list if we have one.
            cached = self._engine_models_cache.get("__system__")
            if cached:
                return ("ok", {"object": "list", "data": cached})

        seen, order, relay = {}, [], None
        for e in self.registry.all():
            if getattr(e, "role", "engine") == "system":
                continue
            models = None
            if e.healthy:
                try:
                    r = await self._short.get(e.url + "/v1/models", headers=headers)
                    if r.status_code == 200:
                        try:
                            models = r.json().get("data") or []
                        except Exception:
                            models = None
                    else:
                        relay = relay or r
                except Exception:
                    models = None
            if models is not None:
                self._engine_models_cache[e.name] = models   # refresh last-good
            else:
                # Unhealthy/unreachable (e.g. mid-load, GIL-blocked): fall back to
                # this engine's last-known list so its models stay listed — they're
                # still assigned to it and will serve once it's free again.
                models = self._engine_models_cache.get(e.name) or []
            for m in models:
                mid = m.get("id")
                if mid and mid not in seen:
                    seen[mid] = m
                    order.append(mid)
        if not order and relay is not None:
            return ("passthrough", relay)
        return ("ok", {"object": "list", "data": [seen[i] for i in order]})

    async def broker_execute(self, *, method, path, headers, query, body):
        _clean_path = path.split("?", 1)[0].rstrip("/")
        # Brokered capabilities must describe the WHOLE node. Routing this to a
        # single engine would report only that engine's CUDA-visible card (its
        # torch hardware summary), so a multi-GPU node looks like it has one card.
        # Build it here in the (torch-free) front, which enumerates every physical
        # GPU via nvidia-smi + sysfs.
        if method.upper() == "GET" and _clean_path == "/coderai/capabilities":
            from codai.broker.capabilities import (
                build_capabilities_document, build_hardware_summary)
            import json as _json
            doc = build_capabilities_document(hardware=build_hardware_summary())
            return {"status_code": 200,
                    "headers": {"content-type": "application/json"},
                    "body": _json.dumps(doc).encode()}
        # Brokered models.list must reflect the WHOLE node (union across engines),
        # not a single engine's assigned subset.
        if method.upper() == "GET" and _clean_path == "/v1/models":
            hdrs = {k: v for k, v in (headers or {}).items() if k.lower() not in _DROP_REQ}
            kind, val = await self.collect_models(hdrs)
            if kind == "ok":
                import json as _json
                return {"status_code": 200,
                        "headers": {"content-type": "application/json"},
                        "body": _json.dumps(val).encode()}
            return {"status_code": val.status_code, "headers": dict(val.headers),
                    "body": val.content}
        return await self._broker_execute_route(method=method, path=path,
                                                headers=headers, query=query, body=body)

    async def _broker_execute_route(self, *, method, path, headers, query, body):
        """Executor for brokered requests: route to an engine over HTTP and return
        the buffered response (the broker dispatcher base64s/relays it)."""
        import json as _json
        import asyncio as _asyncio
        model = None
        if method.upper() == "POST" and _router.is_inference_path(path):
            try:
                model = (_json.loads(body or b"{}") or {}).get("model")
            except Exception:
                model = None
        _is_infer = _router.is_inference_path(path)

        def _pick():
            return _router.pick_engine(
                self.registry, path, method, model,
                required_cap=self._required_cap(path, model),
                default_engine=self.default_engine, pinned=self._pin_for(model),
                pin_fallback=bool(self._model_info(model).get("engine_fallback")))

        # Brokered requests must not hard-fail in the startup/reload window where no
        # engine is ready yet (e.g. an OOM-triggered evict+reload in progress). Wait
        # + retry for readiness, attempt-bound, mirroring the streaming path; only
        # return the 503 once exhausted.
        engine = _pick()
        if engine is None and _is_infer:
            for _ in range(6):
                await _asyncio.sleep(5.0)
                engine = _pick()
                if engine is not None:
                    break
        if engine is None:
            print("[front] broker route: NO ENGINE for path=%s model=%r "
                  "(required_cap=%r pin=%r) — returning 503"
                  % (path, model, self._required_cap(path, model), self._pin_for(model)),
                  flush=True)
            return {"status_code": 503, "headers": {"content-type": "application/json"},
                    "body": b'{"error":"No engine is ready yet."}'}
        send_headers = {k: v for k, v in (headers or {}).items()
                        if k.lower() not in _DROP_REQ}
        # Mark this as a broker-relayed, already-authenticated request so the engine
        # skips its end-user Bearer check (the AISBF broker authenticated upstream).
        # Signed with the internal token; the engine accepts it only if it matches.
        if self.internal_token:
            send_headers["x-coderai-broker-authed"] = self.internal_token
        # Shared-GPU swap gate (all inference kinds): wait out any in-flight swap on
        # a shared card so this request doesn't contend for VRAM.
        try:
            _swap_tok = await self._swap_acquire(engine, model, path, method)
        except Exception:
            _swap_tok = None
        # Front-managed generation queue (text only) — same per-model gate as the
        # direct proxy path, so brokered and direct requests share one queue.
        _qkey = None
        if (method.upper() == "POST" and _router.is_inference_path(path)
                and self._task_kind(path) == "text"):
            _qkey = self._queue_key(model)
            try:
                await self.reqqueue.acquire(
                    _qkey, self._model_capacity(model), self._queue_max_waiting(),
                    rid=engine.name + ":" + (model or ""), model=model or "",
                    engine=engine.name)
            except QueueFull:
                self._swap_release(_swap_tok)
                return {"status_code": 503,
                        "headers": {"content-type": "application/json"},
                        "body": b'{"error":"Server busy: the generation queue is '
                                b'full, please retry shortly."}'}
        # Count brokered inference as in-flight too (with metadata) so it shows on
        # the Tasks page even when the engine is too busy to report it itself.
        _rid = engine.enter_request(
            {"model": model or "", "kind": self._task_kind(path), "path": path}
            if _router.is_inference_path(path) else None)
        import time as _t
        _started = _t.time()
        _status = 502
        # Not-ready statuses (engine up but model still loading/reloading) and
        # connection failures (engine just (re)starting) are retried rather than
        # relayed as an error — mirrors the streaming path. A 4xx is a real client
        # error and is returned as-is.
        _RETRY_STATUS = {425, 429, 500, 502, 503, 504}
        r = None
        try:
            for _attempt in range(6):
                _last = (_attempt >= 5)
                try:
                    r = await self._long.request(method, engine.url + path,
                                                 headers=send_headers,
                                                 params=query or {},
                                                 content=body or b"")
                except Exception as exc:
                    if _is_infer and not _last:
                        await _asyncio.sleep(5.0)
                        continue
                    print("[front] broker route: engine#%s (%s) unreachable for %s: %s"
                          % (engine.id, engine.name, path, exc), flush=True)
                    return {"status_code": 502,
                            "headers": {"content-type": "application/json"},
                            "body": ('{"error":"engine#%s unreachable: %s"}'
                                     % (engine.id, exc)).encode()}
                _status = r.status_code
                if _is_infer and _status in _RETRY_STATUS and not _last:
                    await _asyncio.sleep(5.0)
                    continue
                break
        finally:
            engine.exit_request(_rid)
            if _qkey is not None:
                await self.reqqueue.release(_qkey)
            self._swap_release(_swap_tok)
            if _router.is_inference_path(path):
                self._record_activity(model, self._task_kind(path), _status, _started)
        # Surface the engine's actual reply so a brokered request that "doesn't get
        # executed" (e.g. an instant small error body) is diagnosable from the log.
        print("[front] broker route: %s %s -> engine#%s(%s) status=%s bytes=%d preview=%r"
              % (method, path, engine.id, engine.name, r.status_code,
                 len(r.content), r.content[:200]), flush=True)
        return {"status_code": r.status_code, "headers": dict(r.headers),
                "body": r.content}

    async def broker_execute_stream(self, *, method, path, headers, query, body):
        """Streaming executor for brokered inference: route to the engine, open a
        streamed SSE response, and YIELD each chunk as it arrives (as text). The
        broker client wraps each yielded chunk in a ``chunk`` envelope and sends a
        terminal ``done`` — so the AISBF relay streams tokens to the client instead
        of buffering the whole reply. Shares the per-model queue + in-flight tracking
        with the buffered path."""
        import json as _json
        import asyncio as _asyncio
        model = None
        if method.upper() == "POST" and _router.is_inference_path(path):
            try:
                model = (_json.loads(body or b"{}") or {}).get("model")
            except Exception:
                model = None
        _is_infer = _router.is_inference_path(path)
        # Brokered streaming must behave like the direct/buffered path: when the
        # target model isn't ready yet (an engine still warming up, or an
        # OOM-triggered evict+reload in progress) DO NOT relay the instant
        # not-ready reply to the client — over the broker that lands as a single
        # empty SSE chunk (the "empty reply" symptom). Instead wait + retry for
        # readiness, attempt-bound, and only surface an error once exhausted. The
        # engine's own /v1/chat/completions handler already waits ~5min for a load;
        # this guards the window before it even accepts the request (no engine
        # ready, or a non-200 not-ready status).
        _MAX_TRIES = 6
        _RETRY_WAIT = 5.0
        # Load-status SSE: while we wait for an engine/model to become ready, emit
        # a NON-content chunk (empty delta.content + x_queue_info) so a watching
        # client sees "still loading" without polluting the assembled reply. Gated
        # by the per-model / global load_status_updates flag (default on); the
        # out-of-band broker `pending` keepalive is gated by the same flag in the
        # broker client. This runs in the front-proxy event loop, which stays
        # responsive even while the engine is GIL-blocked loading the model.
        _status_on = self._load_status_enabled(model)
        import time as _t0

        def _status_sse(msg):
            return "data: " + _json.dumps({
                "id": "chatcmpl-load",
                "object": "chat.completion.chunk",
                "created": int(_t0.time()),
                "model": model or "",
                "choices": [{"index": 0, "delta": {"content": ""},
                             "finish_reason": None}],
                "x_queue_info": {"status": "loading", "message": msg},
            }) + "\n\n"

        def _pick():
            return _router.pick_engine(
                self.registry, path, method, model,
                required_cap=self._required_cap(path, model),
                default_engine=self.default_engine, pinned=self._pin_for(model),
                pin_fallback=bool(self._model_info(model).get("engine_fallback")))

        engine = _pick()
        if engine is None and _is_infer:
            for _ in range(_MAX_TRIES):
                if _status_on:
                    yield _status_sse("waiting for an engine to come up")
                await _asyncio.sleep(_RETRY_WAIT)
                engine = _pick()
                if engine is not None:
                    break
        if engine is None:
            yield 'data: {"error":"No engine is ready yet."}\n\n'
            return
        send_headers = {k: v for k, v in (headers or {}).items()
                        if k.lower() not in _DROP_REQ}
        if self.internal_token:
            send_headers["x-coderai-broker-authed"] = self.internal_token
        try:
            _swap_tok = await self._swap_acquire(engine, model, path, method)
        except Exception:
            _swap_tok = None
        _qkey = None
        if (method.upper() == "POST" and _is_infer
                and self._task_kind(path) == "text"):
            _qkey = self._queue_key(model)
            try:
                await self.reqqueue.acquire(
                    _qkey, self._model_capacity(model), self._queue_max_waiting(),
                    rid=engine.name + ":" + (model or ""), model=model or "",
                    engine=engine.name)
            except QueueFull:
                self._swap_release(_swap_tok)
                yield ('data: {"error":"Server busy: the generation queue is full, '
                       'please retry shortly."}\n\n')
                return
        _rid = engine.enter_request(
            {"model": model or "", "kind": self._task_kind(path), "path": path}
            if _is_infer else None)
        import time as _t
        _started = _t.time()
        _status = 502
        # Statuses that mean "not ready / try again" rather than a real client
        # error: the engine is up but still loading/reloading the model. A 4xx is a
        # genuine error and must be relayed as-is, not retried. We do NOT inject any
        # placeholder ("thinking…") chunk — we just wait for the real response. The
        # model-load wait is held open engine-side (/v1/chat/completions waits
        # ~5min) and at the broker protocol level (periodic `pending` keepalives
        # keep the relay deadline extended).
        _RETRY_STATUS = {425, 429, 500, 502, 503, 504}
        try:
            for _attempt in range(_MAX_TRIES):
                _last = (_attempt >= _MAX_TRIES - 1)
                # Connection-level failure means the engine isn't accepting yet
                # (just (re)starting): wait + retry instead of relaying an instant
                # "unreachable" (which lands as a single empty SSE chunk).
                try:
                    rp_req = self._long.build_request(method, engine.url + path,
                                                      headers=send_headers,
                                                      params=query or {},
                                                      content=body or b"")
                    rp_resp = await self._long.send(rp_req, stream=True)
                except Exception as exc:
                    if _is_infer and not _last:
                        if _status_on:
                            yield _status_sse("engine starting up")
                        await _asyncio.sleep(_RETRY_WAIT)
                        continue
                    yield ('data: {"error":"engine#%s unreachable: %s"}\n\n'
                           % (engine.id, exc))
                    break
                _status = rp_resp.status_code
                # Not-ready (model still loading / mid OOM-reload): retry instead of
                # relaying the empty/error reply. No tokens were generated on a
                # non-200, so re-sending the body is safe (no duplicate output).
                if _is_infer and _status in _RETRY_STATUS and not _last:
                    await rp_resp.aclose()
                    if _status_on:
                        yield _status_sse("model loading")
                    await _asyncio.sleep(_RETRY_WAIT)
                    continue
                _meas = (_status == 200 and "text/event-stream"
                         in (rp_resp.headers.get("content-type") or ""))
                ntok = 0
                _gen_t0 = None      # wall-clock of the first streamed token (for tok/s)
                async for raw in rp_resp.aiter_raw():
                    if not raw:
                        continue
                    if _meas:
                        ntok += raw.count(b"data:")
                        m = (engine.active or {}).get(_rid)
                        if m is not None:
                            m["step"] = ntok
                            # Publish tokens/s too, measured from the first token so the
                            # model-load/queue wait doesn't drag the average down. The
                            # Tasks page reads m["rate"] (see _merge_engine_tasks); without
                            # this it showed token progress but a frozen 0 speed.
                            if _gen_t0 is None:
                                _gen_t0 = _t.time()
                            else:
                                _el = _t.time() - _gen_t0
                                if _el > 0:
                                    m["rate"] = round(ntok / _el, 1)
                    yield raw.decode("utf-8", "replace")
                await rp_resp.aclose()
                break
        finally:
            engine.exit_request(_rid)
            if _qkey is not None:
                await self.reqqueue.release(_qkey)
            self._swap_release(_swap_tok)
            if _is_infer:
                self._record_activity(model, self._task_kind(path), _status, _started)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _filter_headers(headers, drop) -> list:
        return [(k, v) for k, v in headers.items() if k.lower() not in drop]

    def _model_info(self, model: Optional[str]) -> dict:
        """Return {"engine": pin, "backend": backend} for a model from models.json.

        Builds a {model id / alias / short-name → info} map, refreshed on file mtime
        change. Used for per-model engine pins and capability detection (e.g. a
        ``whisper-server`` backend → the ``whisper`` capability)."""
        if not model or not self._models_path:
            return {}
        import os
        try:
            mtime = os.path.getmtime(self._models_path)
        except OSError:
            return {}
        if mtime != self._pins_mtime:
            self._pins = self._load_pins()
            self._pins_mtime = mtime
        m = model.lower()
        return self._pins.get(m) or self._pins.get(m.split("/")[-1]) or {}

    def _pin_for(self, model: Optional[str]) -> Optional[str]:
        return self._model_info(model).get("engine")

    def _load_pins(self) -> dict:
        import json as _json
        info: dict = {}
        try:
            data = _json.load(open(self._models_path))
        except Exception:
            return info
        for key, lst in data.items():
            if not isinstance(lst, list):
                continue
            for m in lst:
                if not isinstance(m, dict):
                    continue
                rec = {"engine": (m.get("engine") or "").strip() or None,
                       "backend": (m.get("backend") or "").strip() or None,
                       "path": (m.get("path") or m.get("id") or "").strip() or None,
                       # Canonical model id (NOT the alias) — what the loaded-model
                       # list should display so each model shows once, by its id.
                       "model_id": (m.get("id") or m.get("path")
                                    or m.get("alias") or "").strip() or None,
                       "engine_fallback": bool(m.get("engine_fallback")),
                       # Per-model concurrency ceiling — the front queue sizes its
                       # gate to this so it never over-subscribes the engine.
                       "max_instances": m.get("max_instances"),
                       # Per-model override for load-status signalling (None =
                       # inherit the global models.load_status_updates).
                       "load_status_updates": m.get("load_status_updates"),
                       # Per-model wait-keepalive mode (None = inherit global
                       # models.wait_status_mode): silent | invisible | visible.
                       "wait_status_mode": m.get("wait_status_mode")}
                for field_ in (m.get("path"), m.get("id"), m.get("alias")):
                    if not field_:
                        continue
                    f = str(field_).lower()
                    base = f.split("/")[-1]
                    info[f] = rec
                    info[base] = rec
                    # A gguf's automatic alias is its filename without '.gguf' — index
                    # the stem too so a bare-alias request resolves to this record
                    # (and thus its pin / gguf capability), not just the '.gguf' form.
                    if base.endswith(".gguf"):
                        info[base[:-5]] = rec
                        info[f[:-5]] = rec
        return info

    def _record_activity(self, model, kind, status, started_at):
        """Append one completed inference to the recent-activity ring (newest first),
        for the front-native Overview dashboard."""
        try:
            import time as _t
            self._recent_activity.appendleft({
                "time": started_at or _t.time(),
                "model": model or "", "type": kind or "text",
                "status": int(status or 0),
                "duration": round(_t.time() - (started_at or _t.time()), 1),
            })
        except Exception:
            pass

    def _queue_key(self, model: Optional[str]) -> str:
        """Stable gate key for a model: its canonical id (so every alias/path of
        the same model shares one gate), else the raw model string."""
        info = self._model_info(model)
        return (info.get("model_id") or model or "").lower()

    def _swap_cap(self) -> int:
        try:
            return int(getattr(self.config.server, "gpu_swap_batch", 10) or 10)
        except Exception:
            return 10

    def _swap_gate_for(self, engine):
        """Return the shared-GPU model-swap gate for `engine`, or None when it isn't
        on a shared card (nothing to serialize). Co-location is the same signal the
        supervisor uses for cross-engine VRAM release: an engine with a co-located
        sibling carries CODERAI_COSITED_URLS; engines sharing a card have the same
        CODERAI_ENGINE_GPUS selector, which keys the gate so both share one."""
        try:
            if engine is None or getattr(engine, "role", "engine") == "system":
                return None
            env = getattr(engine, "env", None) or {}
            if not env.get("CODERAI_COSITED_URLS"):
                return None  # no sibling on this card → no cross-engine contention
            gkey = env.get("CODERAI_ENGINE_GPUS") or getattr(engine, "url", "") or "shared"
            gate = self._swap_gates.get(gkey)
            if gate is None:
                from codai.frontproxy.reqqueue import GpuSwapGate
                gate = GpuSwapGate(
                    cap=self._swap_cap(),
                    log=lambda m: print("[front] " + m, flush=True))
                self._swap_gates[gkey] = gate
            return gate
        except Exception:
            return None

    def _swap_owner_key(self, engine, model: Optional[str]) -> str:
        """The model identity that determines GPU residency for the swap gate. Same
        model → same owner (runs free); different model → a swap. Falls back to the
        engine name for inference without an explicit model."""
        return self._queue_key(model) or getattr(engine, "name", "") or "?"

    async def _swap_acquire(self, engine, model, path, method):
        """Acquire this engine's shared-GPU swap slot for a GPU-inference request.
        Returns a (gate, key) token for _swap_release, or None when no gate applies
        (single-card engine or non-inference request)."""
        if str(method).upper() != "POST" or not _router.is_inference_path(path):
            return None
        gate = self._swap_gate_for(engine)
        if gate is None:
            return None
        key = self._swap_owner_key(engine, model)
        await gate.acquire(key)
        return (gate, key)

    def _swap_release(self, token) -> None:
        """Synchronous on purpose: called from `finally` blocks that may run while
        the request coroutine is being cancelled — a sync release always completes
        and frees the slot, where an awaited one could be cancelled mid-way and
        strand the slot (blocking every queued swap behind it)."""
        if token is not None:
            try:
                token[0].release(token[1])
            except Exception:
                pass

    def _model_capacity(self, model: Optional[str]) -> int:
        """Per-model concurrency = its max_instances, falling back to the global
        server default. This is the number of front queue slots for the model."""
        info = self._model_info(model)
        mi = info.get("max_instances")
        if mi:
            try:
                return max(1, int(mi))
            except (TypeError, ValueError):
                pass
        _models = getattr(self.config, "models", None)
        return max(1, int(getattr(_models, "max_model_instances", 1) or 1))

    def _load_status_enabled(self, model: Optional[str]) -> bool:
        """Whether to emit load-status signals (out-of-band broker `pending`
        keepalives + a non-content SSE status chunk) while a model is loading /
        not ready. Per-model ``load_status_updates`` in models.json wins; else the
        global ``models.load_status_updates`` (default True)."""
        ov = self._model_info(model).get("load_status_updates")
        if ov is not None:
            return bool(ov)
        _models = getattr(self.config, "models", None)
        return bool(getattr(_models, "load_status_updates", True))

    def _wait_status_mode(self, model: Optional[str]) -> str:
        """Resolve the wait-keepalive mode (silent|invisible|visible) for the DIRECT
        streaming path: per-model ``wait_status_mode`` in models.json wins, else the
        global ``models.wait_status_mode`` (default invisible). For back-compat,
        derive from load_status_updates when no mode is set anywhere."""
        ov = self._model_info(model).get("wait_status_mode")
        if not ov:
            ov = getattr(getattr(self.config, "models", None), "wait_status_mode", None)
        if not ov:
            ov = "invisible" if self._load_status_enabled(model) else "silent"
        ov = str(ov).strip().lower()
        return ov if ov in ("silent", "invisible", "visible") else "invisible"

    @staticmethod
    def _peek_stream(body_bytes) -> bool:
        try:
            import json as _j
            return bool((_j.loads(body_bytes or b"{}") or {}).get("stream"))
        except Exception:
            return False

    @staticmethod
    def _peek_thinking(body_bytes) -> bool:
        """Best-effort detection of whether reasoning/thinking is requested, so the
        wait keepalive can use the reasoning channel (no content pollution)."""
        try:
            import json as _j
            d = _j.loads(body_bytes or b"{}") or {}
        except Exception:
            return False
        if d.get("enable_thinking") is True or d.get("thinking") is True:
            return True
        if d.get("reasoning_effort") or d.get("reasoning"):
            return True
        ctk = d.get("chat_template_kwargs")
        if isinstance(ctk, dict) and ctk.get("enable_thinking"):
            return True
        return False

    def _queue_max_waiting(self) -> int:
        self._refresh_config_if_changed()
        return max(0, int(getattr(self.config.server, "queue_max_size", 6) or 0))

    def _refresh_config_if_changed(self) -> None:
        """Re-read config.json into self.config (in place) when it changes, so the
        front's config-derived endpoints reflect a settings save (which the engine
        persists) without a restart. Cheap: only acts on an mtime change."""
        if not self._config_path:
            return
        import os
        try:
            mtime = os.path.getmtime(self._config_path)
        except OSError:
            return
        if mtime == self._config_mtime:
            return
        self._config_mtime = mtime
        try:
            from codai.config import ConfigManager
            cm = ConfigManager(self._config_dir)
            cm.load()
            new = cm.config
            for f in ("server", "backend", "models", "offload", "vulkan", "image",
                      "whisper", "archive", "thermal", "jobs", "enhance", "ds4",
                      "compaction", "broker", "system_prompt", "tools_closer_prompt",
                      "grammar_guided", "parser", "tmp_dir"):
                if hasattr(new, f):
                    setattr(self.config, f, getattr(new, f))
            self.default_engine = getattr(self.config.server, "default_engine", None)
        except Exception:
            pass

    def _required_cap(self, path: str, model: Optional[str]) -> Optional[str]:
        ds4 = getattr(self.config, "ds4", None)
        info = self._model_info(model)
        cap = _router.required_capability(
            model, path=path,
            backend=info.get("backend"),
            ds4_model_id=getattr(ds4, "model_id", None) if ds4 else None,
            ds4_enabled=bool(getattr(ds4, "enabled", False)) if ds4 else False)
        # The name heuristic can't see that a bare alias (e.g. '…-q4_k_m', no
        # literal 'gguf') backs a .gguf file, so it falls through to
        # 'transformers' (CUDA-only) and the request never reaches a Vulkan/AMD
        # engine. Correct it from the model's configured path. (whisper/ds4 take
        # precedence above and are left untouched.)
        if cap == "transformers":
            mpath = (info.get("path") or "").lower()
            if mpath.endswith(".gguf"):
                cap = "gguf"
        return cap

    @staticmethod
    def _peek_model(body: bytes, content_type: str) -> Optional[str]:
        if not body or "application/json" not in (content_type or "").lower():
            return None
        try:
            return (json.loads(body) or {}).get("model")
        except Exception:
            return None

    # High-frequency dashboard pollers: serve with a short timeout and a graceful
    # fallback so a momentarily-blocked engine loop can never hang the web UI.
    _POLL_PATHS = {"/admin/api/tasks", "/admin/api/system-stats"}

    async def poll(self, request: Request) -> Response:
        prim = self.registry.primary()
        if prim is None:
            return JSONResponse({"engine": "down", "tasks": [], "queue": []})
        is_tasks = request.url.path.rstrip("/").endswith("/tasks")
        # If the primary is mid-generation, don't poke it (the call would just burn
        # the timeout). Serve the merged/synthesized task list from the front's
        # last-good cache + live in-flight tracking — no engine hit during work.
        if is_tasks and int(getattr(prim, "inflight", 0) or 0) > 0:
            # In-flight: try a SHORT poll FIRST. An image/diffusers generation
            # releases the GIL, so the engine can still answer with the REAL
            # step/total progress (otherwise the task page only shows "working…").
            # Only fall back to the synthesized list when the engine is genuinely
            # GIL-bound (text gen) and the quick poll times out — keeping the page
            # responsive either way.
            try:
                headers = self._filter_headers(request.headers, _DROP_REQ)
                r = await self._short.get(
                    prim.url + request.url.path, headers=headers,
                    params=request.query_params,
                    timeout=httpx.Timeout(connect=2.0, read=1.5, write=2.0, pool=2.0))
                if r.status_code == 200:
                    data = r.json()
                    _ptasks = data.get("tasks") or []
                    self._primary_tasks_cache = _ptasks
                    self._primary_tasks_at = time.monotonic()
                    data["tasks"] = self._merge_engine_tasks(prim, _ptasks)
                    data["cooling_engines"] = self._cooling_engines()
                    return JSONResponse(data)
            except Exception:
                pass
            ptasks = (self._primary_tasks_cache or []) \
                if (time.monotonic() - self._primary_tasks_at) < 120 else []
            return JSONResponse({"engine": "busy", "stale": True,
                                 "tasks": self._merge_engine_tasks(prim, ptasks),
                                 "cooling_engines": self._cooling_engines(),
                                 "queue": []})
        try:
            headers = self._filter_headers(request.headers, _DROP_REQ)
            r = await self._short.get(prim.url + request.url.path, headers=headers,
                                      params=request.query_params)
            if is_tasks and r.status_code == 200:
                try:
                    data = r.json()
                    _ptasks = data.get("tasks") or []
                    # Cache the primary's live task list as last-good, so when a later
                    # poll times out (engine GIL-busy generating) we can still show its
                    # running work instead of dropping it from the page.
                    self._primary_tasks_cache = _ptasks
                    self._primary_tasks_at = time.monotonic()
                    data["tasks"] = self._merge_engine_tasks(prim, _ptasks)
                    data["cooling_engines"] = self._cooling_engines()
                    return JSONResponse(data)
                except Exception:
                    pass
            return Response(content=r.content, status_code=r.status_code,
                            headers=dict(self._filter_headers(r.headers, _DROP_RESP)),
                            media_type=r.headers.get("content-type"))
        except Exception:
            # Engine busy (event loop blocked by GIL-heavy work) — don't hang the UI.
            # Fall back to the last-good primary task list (recent) so a running
            # generation stays visible while the engine can't answer the poll, plus
            # the running tasks the supervisor saw on the other engines.
            ptasks = []
            if is_tasks and (time.monotonic() - self._primary_tasks_at) < 120:
                ptasks = self._primary_tasks_cache or []
            tasks = self._merge_engine_tasks(prim, ptasks) if is_tasks else []
            return JSONResponse({"engine": "loading", "stale": True,
                                 "tasks": tasks, "queue": []})

    @staticmethod
    def _has_cred(request: Request) -> bool:
        """Light auth: a session cookie or bearer token is merely PRESENT. Used for
        low-sensitivity, front-served telemetry (GPU stats, engine status tiles) that
        must stay live even while the primary engine is busy generating — full session
        validation round-trips to that (possibly saturated) engine, which would make
        the dashboard's own status panels vanish exactly when you want to watch them."""
        return bool(request.cookies.get("session")) or \
            request.headers.get("authorization", "").lower().startswith("bearer ")

    async def is_admin(self, request: Request) -> bool:
        """Authorize a front-handled admin action by validating the caller's session
        against the primary engine (which owns sessions). 200 → authorized.

        Probes the engine's /admin/api/whoami (admin-gated). NOTE: do not point this
        at /admin/api/status — that route is front-only (removed from the engine in
        def78c1), so probing it on the engine 404s and silently fails every
        front-handled admin action (model load/unload, engine mgmt)."""
        prim = self.registry.primary()
        if prim is None:
            return False
        try:
            headers = self._filter_headers(request.headers, _DROP_REQ)
            r = await self._short.get(prim.url + "/admin/api/whoami", headers=headers)
            return r.status_code == 200
        except Exception:
            return False

    async def gpu_stats(self, request: Request) -> Response:
        """Serve per-card GPU stats (util/VRAM/temperature) FROM THE FRONT, using the
        torch-free gpu_detect enumeration (nvidia-smi + AMD sysfs). This keeps the
        dashboard's temps/stats live even when an engine is saturated generating —
        previously /admin/api/gpu-stats fell through to the generic proxy and blocked
        on the busy engine, hanging the whole UI. Run in a thread so the subprocess
        (nvidia-smi) never blocks the front's event loop. Auth is light on purpose:
        GPU telemetry is low-sensitivity, and full session validation lives on the
        (possibly busy) engine, which would defeat the point — so we only require a
        session cookie or bearer token to be present."""
        if not self._has_cred(request):
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        import asyncio
        from codai.frontproxy.gpu_detect import gpu_stats as _gpu_stats
        try:
            cards = await asyncio.to_thread(_gpu_stats)
        except Exception as exc:
            return JSONResponse({"cards": [], "error": str(exc)})
        return JSONResponse({"cards": cards})

    async def system_stats(self, request: Request) -> Response:
        """CPU/GPU/RAM/VRAM telemetry for the Tasks header, built on the FRONT
        (psutil + torch-free gpu_detect) so it stays live while an engine is busy
        generating. Run in a thread so the brief CPU sample never blocks the loop."""
        if not self._has_cred(request):
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        import asyncio
        try:
            return JSONResponse(await asyncio.to_thread(self._build_system_stats))
        except Exception as exc:
            return JSONResponse({"cpu": {}, "gpu": {}, "ram": None, "vram": None,
                                 "error": str(exc)})

    @staticmethod
    def _build_system_stats() -> dict:
        cpu = {"util": None, "temp": None, "cores": None}
        ram = None
        try:
            import psutil
            cores = psutil.cpu_count() or 1
            cpu["cores"] = cores
            # System-wide load sampled briefly, scaled to the per-core sum the tile
            # expects (0..cores*100).
            avg = psutil.cpu_percent(interval=0.15)
            cpu["util"] = round((avg or 0.0) * cores / 100.0 * 100.0, 1)
            try:
                temps = psutil.sensors_temperatures() or {}
                for key in ("k10temp", "coretemp", "zenpower", "cpu_thermal"):
                    if temps.get(key):
                        cpu["temp"] = max(t.current for t in temps[key]
                                          if t.current is not None)
                        break
            except Exception:
                pass
            vm = psutil.virtual_memory()
            ram = {"used": round(vm.used / 1e9, 2), "total": round(vm.total / 1e9, 2),
                   "percent": vm.percent}
        except Exception:
            pass
        gpu = {"util": None, "temp": None, "name": None}
        vram = None
        try:
            from codai.frontproxy.gpu_detect import gpu_stats as _gs
            cards = _gs()
            if cards:
                utils = [c.get("util") for c in cards if c.get("util") is not None]
                temps = [c.get("temp") for c in cards if c.get("temp") is not None]
                gpu["util"] = round(sum(utils) / len(utils), 1) if utils else None
                gpu["temp"] = max(temps) if temps else None
                gpu["name"] = (cards[0]["name"] if len(cards) == 1
                               else f"{len(cards)} GPUs")
                used = sum((c.get("mem_used") or 0) for c in cards)
                total = sum((c.get("mem_total") or 0) for c in cards)
                if total:
                    vram = {"used": round(used, 2), "total": round(total, 2),
                            "free": round(total - used, 2),
                            "percent": round(used / total * 100, 1),
                            "gpu": gpu["name"]}
        except Exception:
            pass
        return {"cpu": cpu, "gpu": gpu, "ram": ram, "vram": vram}

    async def batch(self, request: Request) -> Response:
        """Fan out several engine GET reads CONCURRENTLY (server-side) and return
        them in one response.

        A page that needs ~10 ``/admin/api/*`` reads otherwise fires ~10 browser
        requests; the browser caps ~6 connections per host, so during a generation
        (engine event loop GIL-busy) the stuck requests saturate that limit and the
        whole page freezes. Here the front issues all of them at once over its own
        (un-capped) connection pool, each individually bounded, so the browser makes
        ONE request and a slow/blocked sub-call returns an error marker without
        holding up the rest. Body: {"paths": ["/admin/api/models", …]}. Returns
        {"results": {path: {status, json|text|error, stale?}}}."""
        if not self._has_cred(request):
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        try:
            payload = await request.json()
            paths = payload.get("paths") or []
        except Exception:
            paths = []
        prim = self.registry.primary()
        if prim is None or not isinstance(paths, list) or not paths:
            return JSONResponse({"results": {}})
        headers = self._filter_headers(request.headers, _DROP_REQ)
        _bound = httpx.Timeout(connect=10.0, read=12.0, write=12.0, pool=12.0)

        async def _one(p):
            if (not isinstance(p, str) or "/admin/api/" not in p
                    or "stream" in p):
                return p, {"status": 400, "error": "unsupported path"}
            try:
                r = await self._long.request("GET", prim.url + p, headers=headers,
                                             timeout=_bound)
                ctype = (r.headers.get("content-type") or "").lower()
                out = {"status": r.status_code}
                if "application/json" in ctype:
                    try:
                        out["json"] = r.json()
                    except Exception:
                        out["text"] = r.text
                else:
                    out["text"] = r.text
                return p, out
            except Exception:
                return p, {"status": 503, "error": "engine busy (generating)",
                           "stale": True}

        import asyncio as _asyncio
        # Cap fan-out width so a pathological request can't spawn unbounded calls.
        pairs = await _asyncio.gather(*[_one(p) for p in paths[:24]])
        return JSONResponse({"results": dict(pairs)})

    def _canonical_loaded(self, keys) -> list:
        """Map an engine's loaded-model keys to canonical model ids, deduped.

        A model can be resident under several keys — its real id/path, the
        auto-derived gguf stem, an explicit alias, a type-prefixed key (``audio:`` …).
        Resolve each to the model's configured id so the loaded-model list shows each
        model once, by its actual id rather than an alias."""
        seen: dict = {}
        for k in sorted(keys):
            canon = (self._model_info(k).get("model_id") or k)
            seen.setdefault(canon, None)
        return list(seen.keys())

    def engines_list(self) -> list:
        out = []
        for e in self.registry.all():
            if getattr(e, "role", "engine") == "system":
                continue   # the cache/downloads worker isn't a GPU engine tile
            try:
                pid = e.proc.pid if e.proc else None
            except Exception:
                pid = None
            out.append({"id": e.id, "name": e.name, "backend": e.backend,
                        "gpu": e.gpu, "healthy": e.healthy, "primary": e.primary,
                        "vram": e.vram, "cooling": bool(e.cooling),
                        "temp": getattr(e, "therm_temp", None),
                        "thermal_paused": bool(getattr(e, "therm_paused", False)),
                        "thermal_frozen": bool(getattr(e, "therm_sigstopped", False)),
                        "loaded_models": self._canonical_loaded(e.loaded_models),
                        "inflight": int(getattr(e, "inflight", 0) or 0),
                        "processing": (int(getattr(e, "inflight", 0) or 0) > 0
                                       or bool(getattr(e, "loading", None))),
                        "pid": pid})
        return out

    def _running_models(self) -> list:
        """Canonical ids of models currently SERVING a request, from the front's
        own in-flight tracking (engine.active). Front-native, so it's accurate even
        while the engine is too busy to answer its own status."""
        running = set()
        for e in self.registry.all():
            for _rid, m in list((e.active or {}).items()):
                mid = m.get("model")
                if mid:
                    running.add(self._model_info(mid).get("model_id") or mid)
        return sorted(running)

    async def model_loaded_status(self, request: Request):
        """Loaded-model status, served from the FRONT's own state.

        The front issues every load/unload and the supervisor keeps each engine's
        resident set in the registry (refreshed by the cheap engine-state poll that
        answers even mid-generation), so ``loaded`` + ``running`` come from the front
        with no engine round-trip. Richer per-model instance detail (instances /
        configured_max) is synced from the engine ONLY when it's idle — never adding
        load during generation — and cached for use while it's busy."""
        running = self._running_models()
        loaded = set()
        for e in self.registry.all():
            if getattr(e, "role", "engine") == "system":
                continue
            loaded |= set(e.loaded_models or [])
        data = {
            "loaded": sorted(self._canonical_loaded(loaded)),
            "running": running,
            "instances": self._mls_cache.get("instances", {}),
            "configured_max": self._mls_cache.get("configured_max", {}),
        }
        prim = self.registry.primary()
        # Sync instance detail only when the primary is idle (inflight == 0).
        if prim is not None and int(getattr(prim, "inflight", 0) or 0) == 0:
            try:
                headers = self._filter_headers(request.headers, _DROP_REQ)
                r = await self._short.get(prim.url + request.url.path, headers=headers,
                                          params=request.query_params)
                if r.status_code == 200:
                    ed = r.json()
                    if isinstance(ed, dict):
                        self._mls_cache = {
                            "instances": ed.get("instances", {}) or {},
                            "configured_max": ed.get("configured_max", {}) or {}}
                        data["instances"] = self._mls_cache["instances"]
                        data["configured_max"] = self._mls_cache["configured_max"]
                        merged = loaded | set(ed.get("loaded") or [])
                        data["loaded"] = sorted(self._canonical_loaded(merged))
            except Exception:
                pass
        return JSONResponse(data)

    async def _forward_to_engine(self, request: Request, engine, body: bytes):
        """Re-issue an admin POST verbatim to a specific engine and relay its reply."""
        send_headers = self._filter_headers(request.headers, _DROP_REQ)
        try:
            r = await self._long.request(
                request.method, engine.url + request.url.path,
                headers=send_headers, params=request.query_params, content=body or b"")
        except Exception as exc:
            return JSONResponse(
                {"detail": f"engine {engine.name} unreachable: {exc}"}, status_code=502)
        return Response(content=r.content, status_code=r.status_code,
                        headers=dict(self._filter_headers(r.headers, _DROP_RESP)),
                        media_type=r.headers.get("content-type"))

    @staticmethod
    def _key_matches_path(key: str, path: str) -> bool:
        from codai.frontproxy.registry import _short_stem
        return (key == path or key.endswith(f":{path}")
                or key.endswith(path.split("/")[-1])
                or _short_stem(key) == _short_stem(path))

    def _engine_by_name(self, name: Optional[str]):
        if not name:
            return None
        for e in self.registry.all():
            if e.name == name:
                return e
        return None

    async def model_unload(self, request: Request):
        """Route an admin model-unload to the engine that actually has the model
        loaded. Unload is otherwise proxied to the primary, which doesn't hold a
        model loaded on a secondary engine and reports it as never-loaded."""
        if not await self.is_admin(request):
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        body = await request.body()
        try:
            path = (json.loads(body or b"{}") or {}).get("path", "")
        except Exception:
            path = ""
        target = None
        if path:
            for e in self.registry.all():
                if any(self._key_matches_path(k, path) for k in e.loaded_models):
                    target = e
                    break
        if target is None:
            target = self.registry.primary()
        if target is None:
            return JSONResponse({"detail": "No engine available"}, status_code=503)
        resp = await self._forward_to_engine(request, target, body)
        # The front issues the unload, so it updates its OWN resident-model list
        # immediately (the supervisor's engine-state poll later reconciles) — the
        # models page reflects it at once instead of waiting a poll cycle.
        if path and 200 <= getattr(resp, "status_code", 500) < 300:
            target.loaded_models = {k for k in target.loaded_models
                                    if not self._key_matches_path(k, path)}
        return resp

    async def model_load(self, request: Request):
        """Route an admin model-load to the model's pinned engine (or one that's
        already serving it), so loading a secondary-engine model from the UI lands
        on the right engine instead of always the primary."""
        if not await self.is_admin(request):
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        body = await request.body()
        try:
            path = (json.loads(body or b"{}") or {}).get("path", "")
        except Exception:
            path = ""
        target = None
        if path:
            # Already loaded somewhere? Reuse that engine.
            for e in self.registry.all():
                if any(self._key_matches_path(k, path) for k in e.loaded_models):
                    target = e
                    break
            # Otherwise honour the model's engine pin from models.json.
            if target is None:
                target = self._engine_by_name(self._pin_for(path))
        if target is None or not target.healthy:
            target = self.registry.primary()
        if target is None:
            return JSONResponse({"detail": "No engine available"}, status_code=503)
        resp = await self._forward_to_engine(request, target, body)
        # Front-maintained resident list: mark the model loaded on its engine as
        # soon as the load succeeds (supervisor poll reconciles the exact keys).
        if path and 200 <= getattr(resp, "status_code", 500) < 300:
            target.loaded_models = set(target.loaded_models) | {path}
        return resp

    async def task_action(self, request: Request) -> Response:
        """Per-task actions (cancel / interrupt / pause / resume / restart, and
        DELETE remove) must reach the engine that OWNS the task. A task can run on
        ANY engine (or the system worker, for downloads), so the catch-all's
        always-the-primary routing makes a task on another engine read 'Task not
        found'. Front-only synthetic Tasks-page entries have no engine task at all.

        Resolve it here: handle the front-synthetic ids, else fan the request out to
        every engine — each validates the same session cookie (HMAC), so the engine
        that owns the task acts on it and the rest 404; the first success wins."""
        if not await self.is_admin(request):
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        # The task id is the last or second-to-last path segment (…/{id} for DELETE,
        # …/{id}/{action} for POST).
        parts = [p for p in request.url.path.split("/") if p]
        task_id = parts[-1] if request.method == "DELETE" else (
            parts[-2] if len(parts) >= 2 else "")
        # Front-only synthetic entries (see _merge_engine_tasks): no engine task.
        if task_id.startswith("queued-"):
            # A front-queued (not-yet-running) request is owned by the waiting client
            # coroutine; it drops out when that client disconnects. There's nothing
            # to cancel on an engine.
            return JSONResponse(
                {"detail": "Queued request — it cancels when the client disconnects."},
                status_code=409)
        if task_id.startswith(("inflight-", "loading-")):
            return JSONResponse({"detail": "Task not found"}, status_code=404)
        # Real task id: fan out to every live engine (incl. the system worker, which
        # owns download/cache tasks). First 2xx wins; otherwise relay the last reply.
        body = await request.body()
        last = None
        for e in self.registry.all():
            if not e.is_alive():
                continue
            resp = await self._forward_to_engine(request, e, body)
            code = getattr(resp, "status_code", 502)
            if 200 <= code < 300:
                return resp
            last = resp
        return last if last is not None else JSONResponse(
            {"detail": "Task not found"}, status_code=404)

    async def _stream_with_keepalive(self, request: Request, engine, path: str,
                                     body_bytes: bytes, model, mode: str,
                                     thinking: bool):
        """Direct streaming inference with a wait-keepalive so the client doesn't
        time out while a front queue slot is acquired and the engine loads the model.

        Mirrors broker_execute_stream but returns a StreamingResponse: commit to a
        200 text/event-stream up front, emit keepalive chunks (per ``mode`` /
        ``thinking``) while waiting, then relay the engine's real SSE; end the stream
        cleanly if the engine dies mid-flight."""
        import json as _json
        import asyncio as _asyncio
        import time as _t

        def _ka(msg: str) -> bytes:
            # "silent" still keeps the connection alive — but as an SSE COMMENT
            # (a ':' line), which keeps the socket/stream open without emitting any
            # chat.completion.chunk (no event for parsers, no content, no metadata).
            if mode == "silent":
                return (": " + msg + "\n\n").encode()
            if thinking:
                delta = {"reasoning_content": msg}
            elif mode == "visible":
                delta = {"content": msg}
            else:                       # invisible
                delta = {"content": ""}
            return ("data: " + _json.dumps({
                "id": "chatcmpl-wait", "object": "chat.completion.chunk",
                "created": int(_t.time()), "model": model or "",
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                "x_queue_info": {"status": "loading", "message": msg},
            }) + "\n\n").encode()

        send_headers = self._filter_headers(request.headers, _DROP_REQ)
        _KA = 3.0            # keepalive cadence while waiting
        _MAX_TRIES = 6
        _RETRY = 5.0
        _RETRY_STATUS = {425, 429, 500, 502, 503, 504}
        is_text = self._task_kind(path) == "text"

        async def _gen():
            _qkey = None
            _acq = None
            _rid = None
            _status = 502
            _started = _t.time()
            rp_resp = None
            _swap_tok = None
            _swap_acq = None
            try:
                # 0. Shared-GPU swap gate: if a different model owns the card, wait
                #    for the swap (keepalive so the client doesn't time out).
                _gate = self._swap_gate_for(engine)
                if _gate is not None:
                    _skey = self._swap_owner_key(engine, model)
                    _swap_acq = _asyncio.ensure_future(_gate.acquire(_skey))
                    while True:
                        try:
                            await _asyncio.wait_for(_asyncio.shield(_swap_acq),
                                                    timeout=_KA)
                            _swap_tok = (_gate, _skey)
                            break
                        except _asyncio.TimeoutError:
                            yield _ka("waiting for GPU (another model is finishing)")
                # 1. Front per-model queue slot (text only) — keepalive while waiting.
                if is_text:
                    _qkey = self._queue_key(model)
                    _acq = _asyncio.ensure_future(self.reqqueue.acquire(
                        _qkey, self._model_capacity(model), self._queue_max_waiting(),
                        rid=engine.name + ":" + (model or ""), model=model or "",
                        engine=engine.name))
                    while True:
                        try:
                            await _asyncio.wait_for(_asyncio.shield(_acq), timeout=_KA)
                            break
                        except _asyncio.TimeoutError:
                            yield _ka("queued — waiting for a free slot")
                        except QueueFull:
                            yield (b'data: {"error":"Server busy: the generation '
                                   b'queue is full, please retry shortly."}\n\n')
                            return
                    _qkey = _qkey   # slot held now

                _rid = engine.enter_request(
                    {"model": model or "", "kind": self._task_kind(path), "path": path})

                # 2. Send to the engine; retry on not-ready (model still loading),
                #    keepalive between attempts.
                for _attempt in range(_MAX_TRIES):
                    _last = (_attempt >= _MAX_TRIES - 1)
                    try:
                        rp_req = self._long.build_request(
                            "POST", engine.url + path, headers=send_headers,
                            params=request.query_params, content=body_bytes)
                        rp_resp = await self._long.send(rp_req, stream=True)
                    except Exception as exc:
                        if not _last:
                            yield _ka("engine starting up")
                            await _asyncio.sleep(_RETRY)
                            continue
                        yield ('data: {"error":"engine#%s unreachable: %s"}\n\n'
                               % (engine.id, exc)).encode()
                        return
                    _status = rp_resp.status_code
                    if _status in _RETRY_STATUS and not _last:
                        try:
                            await rp_resp.aclose()
                        except Exception:
                            pass
                        yield _ka("model loading")
                        await _asyncio.sleep(_RETRY)
                        continue
                    break

                # 3. Relay the real stream, counting tokens for the Tasks page.
                if rp_resp is not None:
                    _m = (engine.active or {}).get(_rid)
                    t0 = _t.monotonic(); ntok = 0; last = 0.0
                    try:
                        async for raw in rp_resp.aiter_raw():
                            ntok += raw.count(b"data:")
                            now = _t.monotonic()
                            if now - last >= 0.5:
                                last = now; dt = now - t0
                                if _m is not None:
                                    _m["step"] = ntok
                                    _m["rate"] = round(ntok / dt, 1) if dt > 0 else 0.0
                            yield raw
                    except httpx.HTTPError as exc:
                        print(f"[front] upstream engine#{engine.id} ({engine.name}) "
                              f"closed the stream early on {path}: {exc!r}", flush=True)
            finally:
                if rp_resp is not None:
                    try:
                        await rp_resp.aclose()
                    except Exception:
                        pass
                # Cancel a still-pending queue acquire (client gave up mid-wait); the
                # queue drops the waiter on cancellation.
                if _acq is not None and not _acq.done():
                    _acq.cancel()
                if _rid is not None:
                    engine.exit_request(_rid)
                if _qkey is not None and (_acq is None or _acq.done()):
                    try:
                        await self.reqqueue.release(_qkey)
                    except Exception:
                        pass
                # Release / cancel the shared-GPU swap slot.
                if _swap_acq is not None and not _swap_acq.done():
                    _swap_acq.cancel()
                self._swap_release(_swap_tok)
                if _router.is_inference_path(path):
                    self._record_activity(model, self._task_kind(path), _status, _started)

        return StreamingResponse(_gen(), status_code=200,
                                 media_type="text/event-stream")

    def _cooling_engines(self) -> list:
        """Which engines are in thermal cooldown right now (for the Tasks banner).

        Covers both a locally-reported cooldown (engine.cooling, from the engine's
        own checkpoint loop) and a front-driven supervisor pause (engine.therm_paused)
        — the latter matters when the engine is SIGSTOPped and can't report at all."""
        out = []
        for e in self.registry.all():
            if e.cooling:
                out.append({"engine": e.name, "gpu": e.cooling.get("gpu"),
                            "cpu": e.cooling.get("cpu"),
                            "message": e.cooling.get("message")})
            elif getattr(e, "therm_paused", False):
                out.append({
                    "engine": e.name, "gpu": getattr(e, "therm_temp", None),
                    "cpu": None,
                    "message": ("thermal pause (frozen)"
                                if getattr(e, "therm_sigstopped", False)
                                else "thermal pause")})
        return out

    @staticmethod
    def _task_kind(path: str) -> str:
        """Coarse task kind from the inference path, for synthesized in-flight tasks."""
        p = (path or "").lower()
        if "/chat/completions" in p or "/completions" in p:
            return "text"
        if "/images" in p:
            return "image"
        if "/video" in p:
            return "video"
        if "/audio/speech" in p or "/tts" in p:
            return "tts"
        if "/audio" in p or "/transcri" in p:
            return "audio"
        if "/embeddings" in p:
            return "embedding"
        return "text"

    def _merge_engine_tasks(self, primary, primary_tasks: list) -> list:
        """Tasks from all engines, each tagged with the engine *name* it runs on.

        Also injects SYNTHETIC entries for requests the front itself has in flight
        (engine.active) that aren't already represented by a real task — so work
        shows on the Tasks page even when the target engine is too GIL-busy
        generating to report it. Deduped by (engine, model)."""
        merged = []
        seen = set()
        # Primary's tasks (from its authed response) — tag with the primary name.
        def _mark_cooling(t, e):
            """Surface a thermal cooldown on the task entry itself (the row reads
            t.cooling / t.cooling_message), so a paused-for-heat task shows why."""
            if e is not None and getattr(e, "cooling", None) and t.get("status") == "running":
                t["cooling"] = True
                _cm = e.cooling.get("message") if isinstance(e.cooling, dict) else None
                t.setdefault("cooling_message", _cm or "paused for thermal cooldown")
        for t in primary_tasks:
            if isinstance(t, dict):
                t = dict(t)
                t.setdefault("engine", primary.name if primary else None)
                _mark_cooling(t, primary)
                seen.add(t.get("id"))
            merged.append(t)
        # Tasks the supervisor saw on the other engines.
        for e in self.registry.all():
            if primary is not None and e.id == primary.id:
                continue
            for t in (e.tasks or []):
                if not isinstance(t, dict) or t.get("id") in seen:
                    continue
                t = dict(t)
                t["engine"] = e.name
                _mark_cooling(t, e)
                merged.append(t)
                seen.add(t.get("id"))
        # Synthetic "loading" tasks parsed from the log stream, for any engine that
        # is loading a model but whose event loop is GIL-blocked (so its real
        # loading task never reached us). Skip if a real loading task for the same
        # engine already surfaced above.
        have_loading = {(t.get("engine"), t.get("model")) for t in merged
                        if isinstance(t, dict) and t.get("kind") == "loading"}
        for e in self.registry.all():
            ld = e.loading
            if not ld or (e.name, ld.get("model")) in have_loading:
                continue
            merged.append({
                "id": f"loading-{e.name}",
                "kind": "loading",
                "title": f"Loading {ld.get('model') or 'model'}",
                "model": ld.get("model") or "",
                "status": "running",
                "step": ld.get("step", 0),
                "total": ld.get("total", 0),
                "rate": 0.0,
                "message": ld.get("message") or "Loading",
                "engine": e.name,
                "active": True,
                "cancellable": False,
                "pausable": False,
                "restartable": False,
            })
        # Overlay the front's LIVE in-flight counts onto the matching real task.
        # The primary's task list is often served from the last-good cache while the
        # engine is briefly busy, which FROZE the token count (step) even though the
        # speed — a cumulative average over growing elapsed time — kept drifting, so
        # it looked like "only the speed updates". The front is itself relaying the
        # SSE stream and counting tokens (engine.active, refreshed ~2×/s by the
        # streaming proxy), so use that as the live source of step/rate.
        live = {}            # (engine, model) → {step, rate}
        live_by_engine = {}  # engine → list of live in-flight metas
        for e in self.registry.all():
            for _rid, m in list((e.active or {}).items()):
                meta = {"step": m.get("step") or 0, "rate": m.get("rate") or 0.0}
                k = (e.name, m.get("model") or "")
                if k not in live or meta["step"] > live[k]["step"]:
                    live[k] = meta
                live_by_engine.setdefault(e.name, []).append(meta)
        for t in merged:
            if not isinstance(t, dict) or t.get("status") != "running":
                continue
            lm = live.get((t.get("engine"), t.get("model") or ""))
            # Exact (engine, model) match failed — the client's model string (alias /
            # path) can differ from the task's resolved name. Fall back to the engine's
            # sole live generation when there's exactly one (the common case).
            if lm is None and t.get("kind") in ("text", "generation"):
                eng_live = live_by_engine.get(t.get("engine")) or []
                if len(eng_live) == 1:
                    lm = eng_live[0]
            if not lm:
                continue
            if (lm["step"] or 0) > (t.get("step") or 0):
                t["step"] = lm["step"]
            if lm["rate"]:
                t["rate"] = lm["rate"]

        # Synthetic in-flight tasks for requests the front dispatched but that have
        # no real task yet (engine too busy to report). Dedup by (engine, model) so
        # we don't double-show a generation the engine already reported.
        import time as _t
        have = {(t.get("engine"), t.get("model")) for t in merged if isinstance(t, dict)}
        for e in self.registry.all():
            for rid, m in list((e.active or {}).items()):
                key = (e.name, m.get("model") or "")
                if key in have:
                    continue
                have.add(key)
                merged.append({
                    "id": f"inflight-{rid}",
                    "kind": m.get("kind") or "text",
                    "title": (m.get("model") or "generation"),
                    "model": m.get("model") or "",
                    "status": "running",
                    "step": m.get("step") or 0, "total": 0,
                    "rate": m.get("rate") or 0.0,
                    "message": "generating",
                    "started_at": m.get("started_at"),
                    "engine": e.name,
                    "active": True,
                    "cancellable": False,
                    "pausable": False,
                    "restartable": False,
                })
        # Front-queued (not yet running) generations: requests waiting on the
        # front's per-model gate. The engine hasn't seen them yet, so only the
        # front can report them — show each with its queue position.
        for q in self.reqqueue.snapshot():
            merged.append({
                "id": f"queued-{q.get('rid')}-{q.get('position')}",
                "kind": "text",
                "title": (q.get("model") or "generation"),
                "model": q.get("model") or "",
                "status": "queued",
                "position": q.get("position"),
                "step": 0, "total": 0, "rate": 0.0,
                "message": f"queued (#{q.get('position')})",
                "started_at": q.get("enqueued_at"),
                "engine": q.get("engine"),
                "active": True,
                "cancellable": True,
                "pausable": False,
                "restartable": False,
            })
        return merged

    def _system_engine(self):
        """The coderai-system worker, if it's up."""
        for e in self.registry.all():
            if getattr(e, "role", "engine") == "system" and e.is_alive():
                return e
        return None

    async def _proxy_passthrough(self, request: Request, engine) -> Response:
        """Stream a request through to a specific worker (used for the coderai-system
        worker). The worker is always responsive, so the long (no-read-timeout)
        client is safe and handles both buffered JSON and SSE (download-stream)."""
        method = request.method
        url = engine.url + request.url.path
        headers = self._filter_headers(request.headers, _DROP_REQ)
        content = (request.stream()
                   if method in ("POST", "PUT", "PATCH") else None)
        rp_req = self._long.build_request(method, url, headers=headers,
                                          params=request.query_params, content=content)
        try:
            rp_resp = await self._long.send(rp_req, stream=True)
        except Exception as exc:
            return JSONResponse(
                {"error": f"coderai-system worker unreachable: {exc}"}, status_code=502)

        async def _release():
            await rp_resp.aclose()

        return StreamingResponse(
            rp_resp.aiter_raw(), status_code=rp_resp.status_code,
            headers=dict(self._filter_headers(rp_resp.headers, _DROP_RESP)),
            media_type=rp_resp.headers.get("content-type"),
            background=BackgroundTask(_release))

    # -------------------------------------------------------------------- proxy
    async def proxy(self, request: Request) -> Response:
        path = request.url.path
        method = request.method

        # Cache scan / HF downloads / cache management → the coderai-system worker
        # (off the GPU engines, always responsive). Falls through to normal routing
        # if the worker isn't up yet.
        if any(s in path for s in _SYSTEM_PATHS):
            sysw = self._system_engine()
            if sysw is not None:
                return await self._proxy_passthrough(request, sysw)

        # Inference JSON bodies are small: buffer so we can route by `model`, then
        # forward the buffered bytes. Everything else streams through unbuffered.
        body_bytes: Optional[bytes] = None
        model = None
        if method == "POST" and _router.is_inference_path(path):
            body_bytes = await request.body()
            model = self._peek_model(body_bytes, request.headers.get("content-type", ""))

        engine = _router.pick_engine(
            self.registry, path, method, model,
            required_cap=self._required_cap(path, model),
            default_engine=self.default_engine, pinned=self._pin_for(model),
            pin_fallback=bool(self._model_info(model).get("engine_fallback")))
        if engine is None:
            return JSONResponse(
                {"error": "No engine is ready yet (still starting/loading)."},
                status_code=503)

        # Bound page-data reads with a finite timeout + graceful fallback, so a
        # busy engine can never hang a page *forever*. The catch-all otherwise uses
        # _long (no read timeout): a handful of stuck /admin/api GETs then saturate
        # the browser's ~6-connections-per-host limit and freeze the whole page
        # behind them. Excludes endpoints that are legitimately slow (HF hub
        # lookups) or streaming (…-stream, SSE), which keep the unbounded path.
        if (method == "GET" and "/admin/api/" in path and "stream" not in path
                and "/hf-" not in path
                and "text/event-stream"
                not in (request.headers.get("accept", "").lower())):
            try:
                r = await self._long.request(
                    "GET", engine.url + path,
                    headers=self._filter_headers(request.headers, _DROP_REQ),
                    params=request.query_params,
                    timeout=httpx.Timeout(connect=10.0, read=12.0, write=12.0,
                                          pool=12.0))
            except Exception:
                return JSONResponse(
                    {"error": "engine busy (generating); data temporarily "
                              "unavailable", "stale": True}, status_code=503)
            return Response(
                content=r.content, status_code=r.status_code,
                headers=dict(self._filter_headers(r.headers, _DROP_RESP)),
                media_type=r.headers.get("content-type"))

        # Streaming inference always goes through the keepalive path: the WHOLE point
        # is to hold the connection open (from the front, which stays responsive even
        # when the engine is stuck/GIL-blocked loading) while we acquire a queue slot
        # and the engine loads the model. The mode only changes the payload —
        # "silent" still keeps the socket alive (SSE comments), "invisible"/"visible"
        # add status, thinking uses the reasoning channel.
        if (method == "POST" and _router.is_inference_path(path)
                and body_bytes is not None and self._peek_stream(body_bytes)):
            return await self._stream_with_keepalive(
                request, engine, path, body_bytes, model,
                self._wait_status_mode(model), self._peek_thinking(body_bytes))

        # Front-managed generation queue (text only). Acquire a per-model slot
        # before dispatching: if all max_instances slots are busy this awaits
        # (showing as "queued" on the Tasks page) until one frees; if too many are
        # already waiting it returns 503. A client disconnect while queued cancels
        # this await and drops it from the queue. Other inference kinds (images,
        # audio, embeddings…) pass through unqueued.
        _qkey = None
        if (method == "POST" and _router.is_inference_path(path)
                and self._task_kind(path) == "text"):
            _qkey = self._queue_key(model)
            try:
                await self.reqqueue.acquire(
                    _qkey, self._model_capacity(model), self._queue_max_waiting(),
                    rid=engine.name + ":" + (model or ""), model=model or "",
                    engine=engine.name)
            except QueueFull:
                return JSONResponse(
                    {"error": "Server busy: the generation queue is full, "
                              "please retry shortly."}, status_code=503)

        url = engine.url + path
        headers = self._filter_headers(request.headers, _DROP_REQ)
        content = body_bytes if body_bytes is not None else request.stream()

        # Shared-GPU swap gate (all inference kinds, incl. image/video): wait for the
        # card if a different model currently owns it, so this forward never contends.
        try:
            _swap_tok = await self._swap_acquire(engine, model, path, method)
        except Exception:
            _swap_tok = None

        rp_req = self._long.build_request(
            method, url, headers=headers, params=request.query_params,
            content=content)
        # Count this as in-flight on the chosen engine so a restart can drain it:
        # decremented only once the response is fully streamed (or send failed).
        # Attach metadata (model/kind) for inference so the front can show a task
        # for it even when the engine is too busy to answer its own Tasks poll.
        _meta = ({"model": model or "", "kind": self._task_kind(path), "path": path}
                 if _router.is_inference_path(path) else None)
        _rid = engine.enter_request(_meta)
        import time as _t
        _started = _t.time()
        try:
            rp_resp = await self._long.send(rp_req, stream=True)
        except Exception as exc:
            engine.exit_request(_rid)
            if _qkey is not None:
                await self.reqqueue.release(_qkey)
            self._swap_release(_swap_tok)
            return JSONResponse(
                {"error": f"Engine#{engine.id} unreachable: {exc}"}, status_code=502)

        async def _release():
            try:
                await rp_resp.aclose()
            finally:
                engine.exit_request(_rid)
                if _qkey is not None:
                    await self.reqqueue.release(_qkey)
                self._swap_release(_swap_tok)
                if _meta is not None:
                    self._record_activity(model, self._task_kind(path),
                                          rp_resp.status_code, _started)

        # Measure throughput from the SSE stream the front relays, and publish it on
        # the in-flight metadata. This gives the Tasks page a live it/s for the
        # synthesized task even while the engine is too busy to report its own — each
        # SSE "data:" event is ~one token for chat/text completions.
        _meas = (_rid is not None and rp_resp.status_code == 200
                 and "text/event-stream" in (rp_resp.headers.get("content-type") or ""))

        async def _relay_iter():
            import time as _t
            t0 = _t.monotonic(); ntok = 0; last = 0.0
            try:
                async for raw in rp_resp.aiter_raw():
                    if _meas:
                        ntok += raw.count(b"data:")
                        now = _t.monotonic()
                        if now - last >= 0.5:          # refresh ~2×/s, keep it cheap
                            last = now
                            dt = now - t0
                            m = (engine.active or {}).get(_rid)
                            if m is not None:
                                m["step"] = ntok
                                m["rate"] = round(ntok / dt, 1) if dt > 0 else 0.0
                    yield raw
            except httpx.HTTPError as exc:
                # The upstream engine dropped the connection mid-stream — almost
                # always because it CRASHED (e.g. a CUDA ggml_abort → SIGABRT on a
                # VRAM-OOM during a large-context decode) or was restarted. The body
                # is already partly sent, so we can't swap in a clean JSON error; just
                # stop relaying. The supervisor respawns the engine. Without this the
                # truncated read escapes as an unhandled ASGI exception (a noisy
                # traceback) on every such crash.
                print(f"[front] upstream engine#{engine.id} ({engine.name}) closed the "
                      f"stream early on {path}: {exc!r}", flush=True)
                return

        resp_headers = self._filter_headers(rp_resp.headers, _DROP_RESP)
        return StreamingResponse(
            _relay_iter(),
            status_code=rp_resp.status_code,
            headers=dict(resp_headers),
            media_type=rp_resp.headers.get("content-type"),
            background=BackgroundTask(_release),
        )

    # ----------------------------------------------------------------- status
    async def status(self, request: Request) -> Response:
        """Aggregate /admin/api/status across engines, with a last-good cache.

        Proxies the user's authed request to the primary engine (sessions live
        there), then overlays cross-engine VRAM/loaded-model totals from the
        registry so the dashboard reflects every GPU. On engine timeout, serve the
        cache plus an ``engine: loading|down`` marker — the UI never hangs.
        """
        # Status is served ENTIRELY from the front's own state (registry, config,
        # models.json, live torch-free GPU/RAM stats, front-tracked activity). The
        # engine is never asked — it only generates; the front owns stats. This
        # keeps Overview instant and correct regardless of engine load.
        if not self._has_cred(request):
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        self._refresh_config_if_changed()
        return JSONResponse(self._native_status())

    def _enabled_models_and_aliases(self):
        """(enabled_model_ids, {alias: [ids]}) parsed from models.json. Best-effort."""
        import json as _json
        enabled, aliases, seen = [], {}, set()
        if not self._models_path:
            return enabled, aliases
        try:
            data = _json.load(open(self._models_path))
        except Exception:
            return enabled, aliases
        for _key, lst in (data.items() if isinstance(data, dict) else []):
            if not isinstance(lst, list):
                continue
            for m in lst:
                if not isinstance(m, dict) or m.get("enabled") is False:
                    continue
                mid = (m.get("id") or m.get("path") or m.get("alias") or "").strip()
                if mid and mid not in seen:
                    seen.add(mid)
                    enabled.append(mid)
                alias = (m.get("alias") or "").strip()
                if alias and mid:
                    aliases.setdefault(alias, []).append(mid)
        return enabled, aliases

    def _native_status(self) -> dict:
        """Status built entirely from the front's own state (registry, config,
        models.json, live torch-free GPU stats) — no engine round-trip — so the
        Overview dashboard stays live while the engine is GIL-busy generating.
        Fields only the engine knows (recent_activity, the RAM watcher) are carried
        over from the last good engine status when available."""
        cfg = self.config
        loaded = set()
        for e in self.registry.all():
            loaded |= set(e.loaded_models or [])
        loaded_ids = self._canonical_loaded(loaded)
        enabled, aliases = self._enabled_models_and_aliases()
        vram = None
        try:
            from codai.frontproxy.gpu_detect import gpu_stats as _gs
            cards = _gs()
            used = sum((c.get("mem_used") or 0) for c in cards)
            total = sum((c.get("mem_total") or 0) for c in cards)
            if total:
                vram = {"used": round(used, 2), "free": round(total - used, 2),
                        "total": round(total, 2),
                        "gpu": cards[0]["name"] if len(cards) == 1
                               else f"{len(cards)} GPUs"}
        except Exception:
            pass
        active = sum(int(getattr(e, "inflight", 0) or 0) for e in self.registry.all())
        # System RAM, read on the front (torch-free) via psutil when available.
        ram = None
        try:
            import psutil
            vm = psutil.virtual_memory()
            ram = {"used": round((vm.total - vm.available) / 1e9, 2),
                   "free": round(vm.available / 1e9, 2),
                   "total": round(vm.total / 1e9, 2)}
        except Exception:
            pass
        body = {
            "status": "ok",
            "engine": "ok",
            "backend": getattr(getattr(cfg, "backend", None), "type", None),
            "load_mode": getattr(getattr(cfg, "models", None),
                                 "default_load_mode", None),
            "loaded_models": loaded_ids,
            "models_loaded": len(loaded_ids),
            "enabled_models": enabled,
            "enabled_aliases": aliases,
            "vram": vram,
            "ram": ram,
            "requests": {"active": active, "total": len(self._recent_activity)},
            "recent_activity": list(self._recent_activity),
        }
        return body


class _PollNoiseFilter:
    """Hide web-UI traffic from the front's access log unless --debug-web.

    The admin dashboard constantly polls/reads (status, gpu-stats, tasks, settings,
    downloads, model-loaded-status, models, …) plus loads static assets — all noise
    for normal operation. So drop **read** requests (GET/HEAD/OPTIONS) to /admin,
    /static, /, and /favicon. Real API calls (/v1/...) and admin **mutations**
    (POST/PUT/PATCH/DELETE — model-configure, deletes, etc.) still log.
    """
    _READ = ("GET", "HEAD", "OPTIONS")
    _WEB_PREFIXES = ("/admin", "/static", "/login", "/logout")
    _WEB_EXACT = ("/", "/favicon.ico")

    def filter(self, record):
        try:
            a = record.args
            if isinstance(a, (tuple, list)) and len(a) >= 3:
                method = str(a[1]).upper()
                path = str(a[2]).split("?", 1)[0]
                if method in self._READ and (
                        path in self._WEB_EXACT
                        or any(path == p or path.startswith(p + "/") or path == p
                               for p in self._WEB_PREFIXES)):
                    return False
        except Exception:
            pass
        return True


def _front_log_config(debug_web: bool):
    """uvicorn log config that prefixes every front-process line with ``[front]``
    (so it's never confused with an engine's ``[nvidia]``/``[radeon]`` lines) and
    routes codai/broker logs through the same handler. Drops poll noise unless
    --debug-web."""
    import copy
    import uvicorn
    lc = copy.deepcopy(uvicorn.config.LOGGING_CONFIG)
    for fmt in lc.get("formatters", {}).values():
        if "fmt" in fmt and not fmt["fmt"].startswith(("[front]", "[%(asctime)s]")):
            # Prefix each line with an HH:MM:SS timestamp + the [front] tag so it
            # matches the engine log format ([HH:MM:SS][nvidia] …).
            fmt["fmt"] = "[%(asctime)s][front] " + fmt["fmt"]
            fmt["datefmt"] = "%H:%M:%S"
    # Surface codai/broker logs (the broker now runs here) via uvicorn's handler.
    lc.setdefault("loggers", {})
    lc["loggers"]["codai"] = {"handlers": ["default"], "level": "INFO", "propagate": False}
    if not debug_web:
        lc.setdefault("filters", {})["pollnoise"] = {
            "()": "codai.frontproxy.app._PollNoiseFilter"}
        lc["handlers"].get("access", {}).setdefault("filters", []).append("pollnoise")
    return lc


def build_app(config, config_dir=None) -> FastAPI:
    front = FrontProxy(config, config_dir=config_dir)
    app = FastAPI(title="CoderAI Front", docs_url=None, redoc_url=None,
                  openapi_url=None)
    app.state.front = front

    @app.on_event("startup")
    async def _startup():
        front.supervisor = EngineSupervisor(config, None, front.registry,
                                            models_path=front._models_path,
                                            internal_token=front.internal_token,
                                            debug=front.debug_engine)
        front.supervisor.start()
        front.start_broker()

    @app.on_event("shutdown")
    async def _shutdown():
        await front.stop_broker()
        if front.supervisor:
            front.supervisor.stop_all()
        await front.aclose()

    @app.get("/healthz", include_in_schema=False)
    async def _healthz():
        prim = front.registry.primary()
        return {"ok": True, "engine_ready": bool(prim and prim.healthy),
                "engines": [{"id": e.id, "gpu": e.gpu, "healthy": e.healthy}
                            for e in front.registry.all()]}

    # Status/UI poll endpoints get the cached, cross-engine-aggregated handler so a
    # busy engine can never hang the dashboard.
    @app.get("/admin/api/status", include_in_schema=False)
    async def _status(request: Request):
        return await front.status(request)

    @app.get("/admin/api/model-loaded-status", include_in_schema=False)
    async def _model_loaded_status(request: Request):
        return await front.model_loaded_status(request)

    # Load/unload must target the engine that owns (or is pinned to) the model, not
    # always the primary. Registered before the catch-all so they aren't proxied.
    @app.post("/admin/api/model-load", include_in_schema=False)
    async def _model_load(request: Request):
        return await front.model_load(request)

    @app.post("/admin/api/model-unload", include_in_schema=False)
    async def _model_unload(request: Request):
        return await front.model_unload(request)

    @app.get("/admin/api/tasks", include_in_schema=False)
    async def _tasks(request: Request):
        return await front.poll(request)

    # Per-task actions must reach the engine that OWNS the task (any engine, not
    # just the primary), else the UI reads "Task not found". Registered before the
    # catch-all so they're routed here instead of proxied to the primary.
    @app.api_route("/admin/api/tasks/{task_id}/{action}", methods=["POST"],
                   include_in_schema=False)
    async def _task_action(task_id: str, action: str, request: Request):
        return await front.task_action(request)

    @app.delete("/admin/api/tasks/{task_id}", include_in_schema=False)
    async def _task_remove(task_id: str, request: Request):
        return await front.task_action(request)

    @app.get("/admin/api/system-stats", include_in_schema=False)
    async def _system_stats(request: Request):
        return await front.system_stats(request)

    # GPU stats are served by the FRONT (torch-free gpu_detect) so temps/util stay
    # live even when an engine is busy generating — registered before the catch-all
    # so it isn't proxied to a (possibly blocked) engine.
    @app.get("/admin/api/gpu-stats", include_in_schema=False)
    async def _gpu_stats(request: Request):
        return await front.gpu_stats(request)

    # Aggregate several engine reads into ONE response (concurrent, bounded) so a
    # page makes one browser request instead of ~10 — avoids saturating the
    # browser's per-host connection limit and freezing during a generation.
    @app.post("/admin/api/batch", include_in_schema=False)
    async def _batch(request: Request):
        return await front.batch(request)

    # /v1/models is the union across engines (each engine registers only the models
    # the front assigned to it). Registered before the catch-all so it's aggregated.
    @app.get("/v1/models", include_in_schema=False)
    async def _models(request: Request):
        headers = front._filter_headers(request.headers, _DROP_REQ)
        kind, val = await front.collect_models(headers)
        if kind == "passthrough":
            return Response(content=val.content, status_code=val.status_code,
                            headers=dict(front._filter_headers(val.headers, _DROP_RESP)),
                            media_type=val.headers.get("content-type"))
        return JSONResponse(val)

    # Engine management (front-owned: it runs the supervisor). Registered before
    # the catch-all so they aren't proxied to an engine.
    @app.get("/admin/api/engines", include_in_schema=False)
    async def _engines(request: Request):
        # Light auth (cookie/bearer present) like /admin/api/gpu-stats: the data is
        # the front's OWN cached registry (engine name/health/VRAM/loaded models),
        # not the engines themselves. Validating the session against the primary
        # engine would 401 whenever it's busy generating — making the task page's
        # engine tiles disappear exactly when the operator wants to see them.
        if not front._has_cred(request):
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        return JSONResponse({"engines": front.engines_list()})

    @app.post("/admin/api/engines/{eid}/restart", include_in_schema=False)
    async def _engine_restart(eid: int, request: Request):
        if not await front.is_admin(request):
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        ok = bool(front.supervisor and front.supervisor.restart_engine(eid))
        return JSONResponse({"success": ok}, status_code=200 if ok else 404)

    # Serve the admin / Studio UI pages from the FRONT (rendered here, sessions
    # validated locally) so navigating the dashboard never waits on a GIL-busy
    # engine mid-generation. The engine handles only generation; pages live here.
    # Registered before the catch-all so page GETs aren't proxied. Login/logout/
    # change-password POST and all /admin/api/* data calls still fall through.
    try:
        from codai.frontproxy.ui_pages import register_ui_pages
        register_ui_pages(app, config_dir)
    except Exception as _exc:
        print(f"[front] could not register local UI pages ({_exc}); "
              f"pages will be proxied to the engine", flush=True)

    # Front-served admin DATA endpoints backed purely by auth.json (tokens, users)
    # — answered from disk so they stay live while the engine is busy generating.
    try:
        from codai.frontproxy.admin_data import register_admin_data
        register_admin_data(app, config_dir, config=config,
                            on_config_read=front._refresh_config_if_changed)
    except Exception as _exc:
        print(f"[front] could not register local admin-data endpoints ({_exc}); "
              f"they will be proxied to the engine", flush=True)

    # Catch-all reverse proxy for everything else (admin UI, /v1 inference, files…).
    @app.api_route("/{path:path}", include_in_schema=False,
                   methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
    async def _proxy(path: str, request: Request):
        return await front.proxy(request)

    return app


def _serve_front(app, **uvicorn_kwargs) -> None:
    """Serve the front with uvicorn, but own the SIGINT/SIGTERM handling so a
    Ctrl-C ALWAYS tears the engines down — even if uvicorn's graceful shutdown
    hangs draining an in-flight proxy stream to a stuck (e.g. mid-CUDA) engine.

    On the first signal we ask uvicorn to exit AND arm a watchdog that force-stops
    the engines (escalating to SIGKILL of their process groups) after a short
    grace, regardless of whether the drain ever completes. A second Ctrl-C stops
    them immediately. As a backstop, engines are also stopped after serve returns.
    """
    import signal
    import threading
    import uvicorn

    supervisor = getattr(app.state.front, "supervisor", None)
    server = uvicorn.Server(uvicorn.Config(app, **uvicorn_kwargs))
    server.install_signal_handlers = lambda: None   # we manage signals ourselves

    state = {"hits": 0}

    def _handle(signum, _frame):
        state["hits"] += 1
        server.should_exit = True
        if state["hits"] >= 2:
            server.force_exit = True
            if supervisor is not None:
                supervisor.stop_all(grace=0.0)
            return
        print("\n[front] shutdown requested — stopping engines "
              "(Ctrl-C again to force)…", flush=True)

        def _watchdog():
            # If the graceful drain hasn't finished promptly, force engines down
            # so a stuck upstream stream can't keep them (and us) alive.
            time.sleep(6.0)
            if supervisor is not None:
                supervisor.stop_all(grace=5.0)
            server.force_exit = True
        threading.Thread(target=_watchdog, daemon=True).start()

    for _sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(_sig, _handle)
        except Exception:
            pass

    try:
        server.run()
    finally:
        # Backstop: whatever path we exited by, make sure no engine survives us.
        if supervisor is not None:
            supervisor.stop_all(grace=5.0)


class _TimestampedStdout:
    """Wrap a text stream so every new line begins with an ``[HH:MM:SS]`` tag.

    Lines that already carry a timestamp — engine lines re-emitted by the
    supervisor and uvicorn lines, both of which start with ``[`` + a digit — are
    passed through untouched (no double timestamp), as are in-place tqdm progress
    updates (which start with a carriage return). Splits only on ``\\n`` so a
    ``\\r`` inside a progress line is treated as ordinary content, preserving the
    single-line overwrite rendering. Unknown attributes (isatty/flush/fileno/…)
    delegate to the wrapped stream so TTY detection and flushing keep working."""

    def __init__(self, stream):
        self._stream = stream
        self._at_line_start = True

    def write(self, s):
        if not s:
            return 0
        ts = f'[{time.strftime("%H:%M:%S")}]'
        parts = s.split('\n')
        buf = []
        for idx, part in enumerate(parts):
            if idx > 0:
                buf.append('\n')
                self._at_line_start = True
            if not part:
                continue
            if self._at_line_start:
                already_ts = part[:1] == '[' and part[1:2].isdigit()
                if not part.startswith('\r') and not already_ts:
                    buf.append(ts)
                self._at_line_start = False
            buf.append(part)
        return self._stream.write(''.join(buf))

    def __getattr__(self, name):
        return getattr(self._stream, name)


def run_front(config, args) -> None:
    """Build the front app, start engine supervision, and serve on the public port."""
    # Timestamp every terminal line the front emits (raw prints, uvicorn logs, and
    # re-emitted engine output) at HH:MM:SS. Installed before uvicorn binds its log
    # handlers so they write through the wrapped stream.
    if not isinstance(sys.stdout, _TimestampedStdout):
        sys.stdout = _TimestampedStdout(sys.stdout)
    config_dir = getattr(args, "config", None) if args is not None else None
    app = build_app(config, config_dir=config_dir)
    app.state.front.debug_engine = getattr(args, "debug_engine", False)
    host = config.server.host
    port = config.server.port
    print(f"\n[front] CoderAI front proxy on http://{host}:{port}")
    print(f"[front] Admin UI: http://{host}:{port}/admin")

    _log_config = _front_log_config(getattr(args, "debug_web", False))
    if config.server.https:
        import ssl
        keyfile = config.server.https_key_path
        certfile = config.server.https_cert_path
        if not (keyfile and certfile):
            print("[front] HTTPS requested but no cert/key configured; using HTTP.")
            _serve_front(app, host=host, port=port, log_config=_log_config)
            return
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(certfile, keyfile)
        # uvicorn.Server reads ssl via Config(ssl_*), so pass cert/key paths.
        _serve_front(app, host=host, port=port, log_config=_log_config,
                     ssl_keyfile=keyfile, ssl_certfile=certfile)
    else:
        _serve_front(app, host=host, port=port, log_config=_log_config)
