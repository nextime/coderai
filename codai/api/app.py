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

# Mount static files for admin dashboard
from fastapi.staticfiles import StaticFiles
from pathlib import Path
admin_static_dir = Path(__file__).parent.parent / "admin" / "static"
if admin_static_dir.exists():
    app.mount("/static/admin", StaticFiles(directory=str(admin_static_dir)), name="admin_static")

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
