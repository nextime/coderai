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
FastAPI application module for codai API.
Contains the FastAPI app initialization, lifespan, and core endpoints.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from codai.broker.client import BrokerClient
from codai.broker.service import BrokerService

logger = logging.getLogger(__name__)

# Import from codai modules
from codai.pydantic.textrequest import ModelList
from codai.models.manager import model_manager, multi_model_manager
from codai.broker import build_capabilities_document, build_hardware_summary

# Import global state functions - re-export for backward compatibility
from codai.api.state import (
    get_load_mode,
    set_load_mode,
    get_global_args,
    set_global_args,
    set_global_file_path,
    get_global_debug,
    set_global_debug,
)

# Aliases for backward compatibility
global_debug = False
global_file_path = None


def set_global_debug_wrapper(debug: bool):
    """Set the global debug flag."""
    global global_debug
    global_debug = debug
    set_global_debug(debug)


def set_global_file_path_wrapper(path: str):
    """Set the global file path."""
    global global_file_path
    global_file_path = path
    set_global_file_path(path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    import asyncio as _asyncio

    async def _archive_cleanup_loop():
        from codai.api.archive import archive_manager
        while True:
            await _asyncio.sleep(3600)
            try:
                archive_manager.run_cleanup()
            except Exception:
                pass

    cleanup_task = _asyncio.create_task(_archive_cleanup_loop())
    broker_service = None
    broker_runtime = getattr(app.state, "broker_runtime", None)
    if broker_runtime is not None:
        broker_service = BrokerService(BrokerClient(broker_runtime), app)
        app.state.broker_service = broker_service
        broker_service.start()
    # Run initial cleanup on startup
    try:
        from codai.api.archive import archive_manager
        archive_manager.run_cleanup()
    except Exception:
        pass

    yield

    if broker_service is not None:
        await broker_service.stop()

    cleanup_task.cancel()
    try:
        await cleanup_task
    except _asyncio.CancelledError:
        pass

    # Shutdown
    multi_model_manager.cleanup()
    model_manager.cleanup()
    # Stop whisper-server if running
    if multi_model_manager.whisper_server:
        multi_model_manager.whisper_server.stop()


# Create the FastAPI app
app = FastAPI(
    title="OpenAI-Compatible API",
    description="OpenAI-compatible API supporting NVIDIA (CUDA) and Vulkan backends",
    version="2.0.0",
    lifespan=lifespan,
)


# Import routers from submodules
from codai.api.transcriptions import router as transcriptions_router
from codai.api.images import router as images_router
from codai.api.tts import router as tts_router
from codai.api.text import router as text_router
from codai.api.video import router as video_router
from codai.api.audio_gen import router as audio_gen_router
from codai.api.audio_stems import router as audio_stems_router
from codai.api.audio_clean import router as audio_clean_router
from codai.api.embeddings import router as embeddings_router
from codai.api.pipelines import router as pipelines_router
from codai.api.custom_pipelines import router as custom_pipelines_router
from codai.api.voice_clone import router as voice_clone_router
from codai.api.voice_convert import router as voice_convert_router
from codai.api.faceswap import router as faceswap_router
from codai.api.characters import router as characters_router
from codai.api.loras import router as loras_router
from codai.api.spatial import router as spatial_router
from codai.api.environments import router as environments_router
from codai.admin.routes import router as admin_router

# Import and add middleware
from codai.api.log import log_requests
from codai.api.ratelimit import RateLimitMiddleware, BearerAuthMiddleware
app.middleware("http")(log_requests)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(BearerAuthMiddleware)

# Reverse-proxy support: update ASGI scope with forwarded headers so that
# request.url, redirects, and url_for() reflect the public-facing URL.
try:
    from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
    app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")
except ImportError:
    pass


class _InternalAuthMiddleware:
    """Reject any HTTP request that doesn't carry the front's internal token.

    Active only when CODERAI_INTERNAL_TOKEN is set (i.e. this process is an engine
    spawned by the front). It binds 127.0.0.1, but this also blocks anything else on
    localhost from talking to the engine directly and bypassing the front. In
    single-process mode the token is unset and this is a no-op."""

    def __init__(self, app):
        self._app = app
        self._token = os.environ.get("CODERAI_INTERNAL_TOKEN")

    async def __call__(self, scope, receive, send):
        if self._token and scope.get("type") == "http":
            headers = dict(scope.get("headers", []))
            got = headers.get(b"x-coderai-internal", b"").decode("latin-1")
            if got != self._token:
                await send({"type": "http.response.start", "status": 403,
                            "headers": [(b"content-type", b"application/json")]})
                await send({"type": "http.response.body",
                            "body": b'{"error":"forbidden: engines are reachable only '
                                    b'through the front proxy"}'})
                return
        await self._app(scope, receive, send)


class _ForwardedPrefixMiddleware:
    """Populate ASGI root_path from X-Forwarded-Prefix / X-Script-Name headers."""

    def __init__(self, app):
        self._app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            headers = dict(scope.get("headers", []))
            prefix = (
                headers.get(b"x-forwarded-prefix", b"")
                or headers.get(b"x-script-name", b"")
            )
            if prefix:
                scope = dict(scope)
                scope["root_path"] = prefix.decode().rstrip("/")
        await self._app(scope, receive, send)


app.add_middleware(_ForwardedPrefixMiddleware)
# Added last → outermost: the internal-token gate runs before anything else, so a
# request without the front's token never reaches a route.
app.add_middleware(_InternalAuthMiddleware)

# Mount static files for admin dashboard
from fastapi.staticfiles import StaticFiles
from pathlib import Path
admin_static_dir = Path(__file__).parent.parent / "admin" / "static"
if admin_static_dir.exists():
    app.mount("/static/admin", StaticFiles(directory=str(admin_static_dir)), name="admin_static")

# Serve a favicon at the conventional /favicon.ico path so browsers stop 404-ing on it.
from fastapi.responses import FileResponse, Response as _FaviconResponse
_favicon_path = admin_static_dir / "favicon.ico"


@app.get("/healthz", include_in_schema=False)
async def healthz():
    """Cheap liveness probe that touches no torch/model state.

    The front proxy's engine supervisor polls this to distinguish a *slow* engine
    (busy loading a model — the event loop may be blocked, so this can be late but
    will eventually answer) from a *dead* one (connection refused). It must stay
    trivial and dependency-free so it returns the instant the loop is free."""
    import os as _os
    return {"ok": True, "pid": _os.getpid()}


_ENGINE_VRAM_CACHE = {"at": 0.0, "vram": None, "names": {}}
_ENGINE_VRAM_TTL = 4.0   # seconds


def _cached_engine_vram():
    """This engine's CUDA VRAM, cached with a short TTL so the frequently-polled
    health endpoint doesn't hit the CUDA context on every call (which can stall
    behind a running forward pass and make the engine look 'not responding').
    Device names are static, so cached for the process lifetime."""
    import time as _t
    now = _t.monotonic()
    if _ENGINE_VRAM_CACHE["vram"] is not None and \
            (now - _ENGINE_VRAM_CACHE["at"]) < _ENGINE_VRAM_TTL:
        return _ENGINE_VRAM_CACHE["vram"]
    vram = None
    try:
        # Query VRAM context-free (pynvml → nvidia-smi → torch-if-inited) so an
        # engine that never loads a torch model (the GGUF/llama.cpp engine) doesn't
        # pin a ~256 MiB CUDA context just to answer this health poll.
        from codai.models.gpu_query import gpu_memory
        gpus = gpu_memory()
        if gpus:
            used = free = total = 0
            devs = []
            _names = _ENGINE_VRAM_CACHE["names"]
            for d in gpus:
                i, f, t = d["index"], d["free"], d["total"]
                used += (t - f); free += f; total += t
                if i not in _names:
                    _names[i] = d["name"]
                devs.append({"index": i, "name": _names[i],
                             "free": round(f / 1e9, 2), "total": round(t / 1e9, 2)})
            n = len(devs)
            label = (_names.get(0, "CUDA") if n == 1 else f"{n}× CUDA")
            vram = {"used": round(used / 1e9, 2), "free": round(free / 1e9, 2),
                    "total": round(total / 1e9, 2), "gpu": label,
                    "devices": devs, "device_count": n}
    except Exception:
        vram = _ENGINE_VRAM_CACHE["vram"]   # keep last-good on transient error
    _ENGINE_VRAM_CACHE["vram"] = vram
    _ENGINE_VRAM_CACHE["at"] = now
    return vram


@app.get("/internal/engine-state", include_in_schema=False)
async def internal_engine_state():
    """Auth-free engine introspection for the front proxy's router/aggregator.

    Engines bind 127.0.0.1 only, so this is not publicly reachable. Returns which
    models are resident (for model→engine routing) and this engine's GPU/VRAM (for
    cross-engine status aggregation). Kept cheap so it answers even mid-generation.
    """
    import os as _os
    try:
        loaded = list(multi_model_manager.models.keys())
    except Exception:
        loaded = []
    # Whisper-server models run as their own subprocess (tracked in whisper_servers,
    # not in .models). Surface each running server under both its id and `audio:`
    # alias so the front's cross-engine aggregation shows it as loaded — including
    # when it runs on a secondary engine.
    try:
        for _wid, _wsm in multi_model_manager.whisper_servers.items():
            if _wsm.is_running():
                loaded.append(_wid)
                loaded.append(f"audio:{_wid}")
                # Also surface the backing gguf path so the GGUF-file row in the
                # models page reflects that the file is in use by a running server.
                _mp = getattr(_wsm, "_model_path", None)
                if _mp:
                    loaded.append(_mp)
    except Exception:
        pass
    # A LoRA training job loads its base pipeline OUTSIDE the model manager (and
    # unloads all manager models first), so without this the engine would report 0
    # loaded models mid-training even though the GPU is fully busy. Surface the base
    # model being trained so the engines card reflects what's actually resident.
    try:
        from codai.api.loras import active_training_model
        _tm = active_training_model()
        if _tm and _tm not in loaded:
            loaded.append(_tm)
    except Exception:
        pass
    # VRAM is CACHED with a short TTL: this endpoint is polled every couple seconds
    # by the front's health monitor, and calling torch.cuda.mem_get_info /
    # get_device_name on EVERY poll can serialize behind the running generation on
    # the CUDA context and stall the handler past the poll timeout — which is exactly
    # what flips a busy engine to "not responding". Refresh at most every few seconds
    # (device names cached permanently — they don't change), so mid-generation polls
    # return the last snapshot instantly instead of blocking.
    vram = _cached_engine_vram()
    # Running tasks so the front can show cross-engine activity without needing a
    # session on this engine (sessions live only on the primary).
    tasks = []
    try:
        from codai.tasks import task_registry
        tasks = [t for t in task_registry.list()
                 if t.get("status") in ("running", "queued", "paused")]
    except Exception:
        tasks = []
    # This engine's thermal cooldown state, so the front can show WHICH engine is
    # cooling (each engine pauses on its own GPUs; CPU pauses everything).
    cooling = None
    try:
        from codai.models import thermal
        cs = thermal.get_cooldown_state()
        if cs.get("active"):
            cooling = {"gpu": cs.get("gpu"), "cpu": cs.get("cpu"),
                       "message": cs.get("message")}
    except Exception:
        cooling = None
    # Whether the front's thermal supervisor currently holds this engine in a
    # cooperative pause (so the front can confirm the engine acknowledged it).
    paused = False
    try:
        from codai.models import thermal as _therm
        paused = _therm.external_pause_active()
    except Exception:
        paused = False
    return {"ok": True, "pid": _os.getpid(), "loaded_models": loaded,
            "vram": vram, "tasks": tasks, "cooling": cooling, "paused": paused}


@app.post("/internal/evict-vram", include_in_schema=False)
async def internal_evict_vram(request: Request):
    """A co-located sibling engine (same GPU) asks this engine to release VRAM it
    can't evict itself — the GGUF-isolation split runs two engines on one NVIDIA
    card. Evicts all idle (non-busy) models and reports GB freed. Runs in a thread
    so eviction's blocking CUDA frees don't stall the event loop."""
    import asyncio
    try:
        from codai.models.manager import multi_model_manager
        freed = await asyncio.to_thread(multi_model_manager.release_idle_vram)
        return {"ok": True, "freed_gb": float(freed)}
    except Exception as e:
        return {"ok": False, "error": str(e), "freed_gb": 0.0}


@app.post("/internal/thermal-pause", include_in_schema=False)
async def internal_thermal_pause(request: Request):
    """Front-driven thermal pause: the supervisor (which monitors temps centrally)
    asks this engine to stop generating. Honoured cooperatively at the next safe
    point — request entry / between tokens — via codai.models.thermal."""
    import json as _json
    try:
        data = _json.loads((await request.body()) or b"{}") or {}
    except Exception:
        data = {}
    try:
        from codai.models import thermal as _therm
        _therm.set_external_pause(True, reason=str(data.get("reason") or "thermal"))
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "paused": True}


@app.post("/internal/thermal-resume", include_in_schema=False)
async def internal_thermal_resume(request: Request):
    """Lift the front-driven thermal pause once the hardware has cooled."""
    try:
        from codai.models import thermal as _therm
        _therm.set_external_pause(False)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "paused": False}


@app.post("/internal/reload-config", include_in_schema=False)
async def internal_reload_config(request: Request):
    """Live config refresh pushed by the front when models.json changes: re-read
    models.json and update this engine's assigned-model set, so /v1/models and
    request validation reflect added/removed models without a restart. Weights are
    untouched — (re)loading still happens lazily on the next request."""
    import json as _json
    try:
        body = await request.body()
        data = _json.loads(body or b"{}") or {}
    except Exception:
        data = {}
    n = 0
    try:
        from codai.admin.routes import config_manager
        if config_manager is not None and getattr(config_manager, "models_path", None):
            with open(config_manager.models_path) as _f:
                config_manager.models_data = _json.load(_f)
            _cats = ("text_models", "gguf_models", "vision_models", "image_models",
                     "audio_models", "tts_models", "video_models", "audio_gen_models",
                     "embedding_models", "spatial_models")
            n = sum(len(config_manager.models_data.get(c, []) or []) for c in _cats)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    try:
        from codai.models.manager import multi_model_manager
        assigned = data.get("assigned")
        if assigned is not None:
            multi_model_manager.set_assigned_models(set(assigned))
    except Exception:
        pass
    # Invalidate the cached-models scan (it merges per-model config) so a saved
    # config edit is visible immediately when an engine serves the models page
    # (single-process / system-worker-down fallback).
    try:
        from codai.admin.routes import _invalidate_cache_scan
        _invalidate_cache_scan()
    except Exception:
        pass
    return {"ok": True, "models": n}


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    if _favicon_path.exists():
        return FileResponse(str(_favicon_path), media_type="image/x-icon")
    return _FaviconResponse(status_code=404)

# Include routers from submodules
app.include_router(transcriptions_router, tags=["Audio"])
app.include_router(images_router, tags=["Images"])
app.include_router(tts_router, tags=["Audio"])
app.include_router(text_router, tags=["Text"])
app.include_router(video_router, tags=["Video"])
app.include_router(audio_gen_router, tags=["Audio"])
app.include_router(audio_stems_router, tags=["Audio"])
app.include_router(audio_clean_router, tags=["Audio"])
app.include_router(embeddings_router, tags=["Embeddings"])
app.include_router(pipelines_router, tags=["Pipelines"])
app.include_router(custom_pipelines_router, tags=["Pipelines"])
app.include_router(voice_clone_router, tags=["Audio"])
app.include_router(voice_convert_router, tags=["Audio"])
app.include_router(faceswap_router, tags=["Images"])
app.include_router(characters_router, tags=["Characters"])
app.include_router(loras_router, tags=["LoRAs"])
app.include_router(environments_router, tags=["Environments"])
app.include_router(spatial_router, tags=["Spatial / 3D"])
app.include_router(admin_router, tags=["Admin"])


@app.exception_handler(401)
async def unauthorized_redirect(request: Request, exc: HTTPException):
    """Redirect browser clients to login page on 401; return JSON for API clients."""
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        from fastapi.responses import RedirectResponse
        from codai.api.urlutils import get_public_prefix
        prefix = get_public_prefix(request)
        return RedirectResponse(url=f"{prefix}/login", status_code=302)
    return JSONResponse(status_code=401, content={"detail": exc.detail})


from codai.tasks import TaskCancelled, task_registry


@app.exception_handler(TaskCancelled)
async def task_cancelled_handler(request: Request, exc: TaskCancelled):
    """A worker observed its task was cancelled and unwound. Finish the task
    (cancelled) and return 499 (client-closed-request style). The task id is
    carried on the exception so any generation/training worker can simply
    `raise` without bookkeeping."""
    tid = exc.args[0] if exc.args else None
    if tid:
        task_registry.finish(tid, "cancelled", "cancelled by user")
    return JSONResponse(status_code=499, content={"detail": "Task cancelled", "task_id": tid})


@app.get("/v1/models", response_model=ModelList, summary="List available models", tags=["Core"])
async def list_models():
    """List available models."""
    models = multi_model_manager.list_models()
    return ModelList(data=models)


@app.get("/coderai/capabilities", summary="Server capability document", tags=["Core"])
async def get_broker_capabilities():
    """Return broker capability metadata."""
    return build_capabilities_document(hardware=build_hardware_summary())


@app.get("/v1/files/{filename}", summary="Download a generated file", tags=["Files"])
async def get_file(filename: str):
    """Serve uploaded/generated files."""
    if not global_file_path:
        raise HTTPException(status_code=404, detail="File not found")
    # Prevent path traversal: resolve to real paths and confirm the result
    # stays inside the configured directory.
    safe_base = os.path.realpath(global_file_path)
    candidate = os.path.realpath(os.path.join(global_file_path, filename))
    if not (candidate == safe_base or candidate.startswith(safe_base + os.sep)):
        raise HTTPException(status_code=403, detail="Access denied")
    if not os.path.isfile(candidate):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(candidate)


_IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.gif'}
_VIDEO_EXTS = {'.mp4', '.webm', '.avi', '.mov'}
_AUDIO_EXTS = {'.wav', '.mp3', '.ogg', '.flac', '.aac', '.m4a'}


@app.get("/v1/archive", summary="List archived generations", tags=["Files"])
async def list_archive(request: Request):
    """List all generated files in the output directory."""
    if not global_file_path or not os.path.isdir(global_file_path):
        return {"files": []}
    from codai.api.urlutils import build_file_url
    files = []
    try:
        names = os.listdir(global_file_path)
    except OSError:
        return {"files": []}
    for fname in names:
        fpath = os.path.join(global_file_path, fname)
        if not os.path.isfile(fpath):
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext in _IMAGE_EXTS:
            ftype = 'image'
        elif ext in _VIDEO_EXTS:
            ftype = 'video'
        elif ext in _AUDIO_EXTS:
            ftype = 'audio'
        else:
            continue
        stat = os.stat(fpath)
        files.append({
            'filename': fname,
            'type': ftype,
            'size': stat.st_size,
            'created': stat.st_mtime,
            'url': build_file_url(fname, request),
        })
    files.sort(key=lambda f: f['created'], reverse=True)
    return {"files": files}


@app.delete("/v1/archive/{filename}", summary="Delete an archived file", tags=["Files"])
async def delete_archive_file(filename: str):
    """Delete a generated file from the output directory."""
    if not global_file_path:
        raise HTTPException(status_code=404, detail="File not found")
    safe_base = os.path.realpath(global_file_path)
    candidate = os.path.realpath(os.path.join(global_file_path, filename))
    if not (candidate == safe_base or candidate.startswith(safe_base + os.sep)):
        raise HTTPException(status_code=403, detail="Access denied")
    if not os.path.isfile(candidate):
        raise HTTPException(status_code=404, detail="File not found")
    os.remove(candidate)
    return {"deleted": filename}
