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
        # Last-good /v1/models list per engine. An engine that's mid-load is
        # GIL-blocked and misses health polls; without this its models (incl. a
        # freshly-added one) would flicker out of the aggregated /v1/models while
        # it loads. Keyed by engine name.
        self._engine_models_cache: dict = {}
        # Front-managed generation queue: admission control + ordering + queue
        # position, sized per-model to max_instances so the engine never queues.
        self.reqqueue = FrontQueue()
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
        self._status_cache: Optional[dict] = None
        self._status_cache_at: float = 0.0
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
        self._broker = BrokerService(client)   # app=None → keep our dispatcher
        self._broker.start()
        print("[front] AISBF broker started (front-managed, routes to engines)",
              flush=True)

    async def stop_broker(self):
        if self._broker is not None:
            await self._broker.stop()
            self._broker = None

    async def collect_models(self, headers):
        """Union of every healthy engine's /v1/models. Each engine registers only
        the models the front assigned to it, so the union is the full set with no
        duplicates. Returns ("ok", {...}) or ("passthrough", httpx.Response) when an
        auth/error response should be relayed instead."""
        seen, order, relay = {}, [], None
        for e in self.registry.all():
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
        model = None
        if method.upper() == "POST" and _router.is_inference_path(path):
            try:
                model = (_json.loads(body or b"{}") or {}).get("model")
            except Exception:
                model = None
        engine = _router.pick_engine(
            self.registry, path, method, model,
            required_cap=self._required_cap(path, model),
            default_engine=self.default_engine, pinned=self._pin_for(model),
            pin_fallback=bool(self._model_info(model).get("engine_fallback")))
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
                return {"status_code": 503,
                        "headers": {"content-type": "application/json"},
                        "body": b'{"error":"Server busy: the generation queue is '
                                b'full, please retry shortly."}'}
        # Count brokered inference as in-flight too (with metadata) so it shows on
        # the Tasks page even when the engine is too busy to report it itself.
        _rid = engine.enter_request(
            {"model": model or "", "kind": self._task_kind(path), "path": path}
            if _router.is_inference_path(path) else None)
        try:
            r = await self._long.request(method, engine.url + path,
                                         headers=send_headers, params=query or {},
                                         content=body or b"")
        except Exception as exc:
            print("[front] broker route: engine#%s (%s) unreachable for %s: %s"
                  % (engine.id, engine.name, path, exc), flush=True)
            return {"status_code": 502,
                    "headers": {"content-type": "application/json"},
                    "body": ('{"error":"engine#%s unreachable: %s"}'
                             % (engine.id, exc)).encode()}
        finally:
            engine.exit_request(_rid)
            if _qkey is not None:
                await self.reqqueue.release(_qkey)
        # Surface the engine's actual reply so a brokered request that "doesn't get
        # executed" (e.g. an instant small error body) is diagnosable from the log.
        print("[front] broker route: %s %s -> engine#%s(%s) status=%s bytes=%d preview=%r"
              % (method, path, engine.id, engine.name, r.status_code,
                 len(r.content), r.content[:200]), flush=True)
        return {"status_code": r.status_code, "headers": dict(r.headers),
                "body": r.content}

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
                       "max_instances": m.get("max_instances")}
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

    def _queue_key(self, model: Optional[str]) -> str:
        """Stable gate key for a model: its canonical id (so every alias/path of
        the same model shares one gate), else the raw model string."""
        info = self._model_info(model)
        return (info.get("model_id") or model or "").lower()

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

    def _queue_max_waiting(self) -> int:
        return max(0, int(getattr(self.config.server, "queue_max_size", 6) or 0))

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
        against the primary engine (which owns sessions). 200 → authorized."""
        prim = self.registry.primary()
        if prim is None:
            return False
        try:
            headers = self._filter_headers(request.headers, _DROP_REQ)
            r = await self._short.get(prim.url + "/admin/api/status", headers=headers)
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
            try:
                pid = e.proc.pid if e.proc else None
            except Exception:
                pid = None
            out.append({"id": e.id, "name": e.name, "backend": e.backend,
                        "gpu": e.gpu, "healthy": e.healthy, "primary": e.primary,
                        "vram": e.vram, "cooling": bool(e.cooling),
                        "loaded_models": self._canonical_loaded(e.loaded_models),
                        "pid": pid})
        return out

    async def model_loaded_status(self, request: Request):
        """Proxy /admin/api/model-loaded-status to the primary, then union in the
        models loaded on every *other* engine. Otherwise the models page only sees
        the primary engine's pool and shows models loaded on a secondary (e.g. the
        radeon engine) as idle."""
        prim = self.registry.primary()
        if prim is None:
            return JSONResponse({"loaded": [], "instances": {}, "configured_max": {}})
        try:
            headers = self._filter_headers(request.headers, _DROP_REQ)
            r = await self._short.get(prim.url + request.url.path, headers=headers,
                                      params=request.query_params)
            if r.status_code != 200:
                return Response(content=r.content, status_code=r.status_code,
                                headers=dict(self._filter_headers(r.headers, _DROP_RESP)),
                                media_type=r.headers.get("content-type"))
            data = r.json()
        except Exception:
            return JSONResponse({"loaded": [], "instances": {}, "configured_max": {}})
        if isinstance(data, dict):
            loaded = set(data.get("loaded") or [])
            for e in self.registry.all():
                if prim is not None and e.id == prim.id:
                    continue
                loaded |= e.loaded_models
            data["loaded"] = sorted(loaded)
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
        return await self._forward_to_engine(request, target, body)

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
        return await self._forward_to_engine(request, target, body)

    def _cooling_engines(self) -> list:
        """Which engines are in thermal cooldown right now (for the Tasks banner)."""
        out = []
        for e in self.registry.all():
            if e.cooling:
                out.append({"engine": e.name, "gpu": e.cooling.get("gpu"),
                            "cpu": e.cooling.get("cpu"),
                            "message": e.cooling.get("message")})
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
        for t in primary_tasks:
            if isinstance(t, dict):
                t = dict(t)
                t.setdefault("engine", primary.name if primary else None)
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

    # -------------------------------------------------------------------- proxy
    async def proxy(self, request: Request) -> Response:
        path = request.url.path
        method = request.method

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
        try:
            rp_resp = await self._long.send(rp_req, stream=True)
        except Exception as exc:
            engine.exit_request(_rid)
            if _qkey is not None:
                await self.reqqueue.release(_qkey)
            return JSONResponse(
                {"error": f"Engine#{engine.id} unreachable: {exc}"}, status_code=502)

        async def _release():
            try:
                await rp_resp.aclose()
            finally:
                engine.exit_request(_rid)
                if _qkey is not None:
                    await self.reqqueue.release(_qkey)

        # Measure throughput from the SSE stream the front relays, and publish it on
        # the in-flight metadata. This gives the Tasks page a live it/s for the
        # synthesized task even while the engine is too busy to report its own — each
        # SSE "data:" event is ~one token for chat/text completions.
        _meas = (_rid is not None and rp_resp.status_code == 200
                 and "text/event-stream" in (rp_resp.headers.get("content-type") or ""))

        async def _counting_iter():
            import time as _t
            t0 = _t.monotonic(); ntok = 0; last = 0.0
            async for raw in rp_resp.aiter_raw():
                ntok += raw.count(b"data:")
                now = _t.monotonic()
                if now - last >= 0.5:              # refresh ~2×/s, keep it cheap
                    last = now
                    dt = now - t0
                    m = (engine.active or {}).get(_rid)
                    if m is not None:
                        m["step"] = ntok
                        m["rate"] = round(ntok / dt, 1) if dt > 0 else 0.0
                yield raw

        resp_headers = self._filter_headers(rp_resp.headers, _DROP_RESP)
        return StreamingResponse(
            _counting_iter() if _meas else rp_resp.aiter_raw(),
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
        prim = self.registry.primary()
        if prim is None:
            return self._cached_status("down")
        try:
            headers = self._filter_headers(request.headers, _DROP_REQ)
            r = await self._short.get(prim.url + request.url.path, headers=headers,
                                      params=request.query_params)
            if r.status_code != 200:
                # Pass through auth redirects/errors unchanged (e.g. login needed).
                return Response(content=r.content, status_code=r.status_code,
                                headers=dict(self._filter_headers(r.headers, _DROP_RESP)),
                                media_type=r.headers.get("content-type"))
            data = r.json()
            data = self._overlay_engine_totals(data)
            self._status_cache = data
            self._status_cache_at = time.monotonic()
            return JSONResponse(data)
        except Exception:
            return self._cached_status("loading")

    def _overlay_engine_totals(self, data: dict) -> dict:
        engines = self.registry.all()
        per = []
        used = free = total = 0.0
        loaded = set()
        have_vram = False
        for e in engines:
            per.append({"id": e.id, "name": e.name, "backend": e.backend,
                        "gpu": e.gpu, "healthy": e.healthy, "vram": e.vram,
                        "capabilities": sorted(e.capabilities),
                        "loaded_models": sorted(e.loaded_models)})
            loaded |= e.loaded_models
            if e.vram:
                have_vram = True
                used += e.vram.get("used", 0.0)
                free += e.vram.get("free", 0.0)
                total += e.vram.get("total", 0.0)
        data["x_engines"] = per
        if len([e for e in engines if e.healthy]) > 1:
            if have_vram:
                data["vram"] = {"used": round(used, 2), "free": round(free, 2),
                                "total": round(total, 2),
                                "gpu": f"{len(engines)} engines"}
            if loaded:
                data["loaded_models"] = sorted(loaded)
                data["models_loaded"] = len(loaded)
        return data

    def _cached_status(self, engine_state: str) -> Response:
        body = dict(self._status_cache or {"models_loaded": 0, "loaded_models": [],
                                           "vram": None})
        body["engine"] = engine_state
        body["stale"] = True
        if self._status_cache_at:
            body["stale_age_seconds"] = round(time.monotonic() - self._status_cache_at, 1)
        return JSONResponse(body)


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

    @app.get("/admin/api/system-stats", include_in_schema=False)
    async def _system_stats(request: Request):
        return await front.poll(request)

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
