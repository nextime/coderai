"""
FastAPI application module for codai API.
Contains the FastAPI app initialization, lifespan, and core endpoints.
"""

from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

# Import from codai modules
from codai.pydantic.textrequest import ModelList
from codai.models.manager import model_manager, multi_model_manager

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
    # Startup
    yield
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
from codai.admin.routes import router as admin_router

# Import and add middleware
from codai.api.log import log_requests
app.middleware("http")(log_requests)

# Mount static files for admin dashboard
from fastapi.staticfiles import StaticFiles
from pathlib import Path
admin_static_dir = Path(__file__).parent.parent / "admin" / "static"
if admin_static_dir.exists():
    app.mount("/static/admin", StaticFiles(directory=str(admin_static_dir)), name="admin_static")

# Include routers from submodules
app.include_router(transcriptions_router)
app.include_router(images_router)
app.include_router(tts_router)
app.include_router(text_router)
app.include_router(admin_router)


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    """List available models."""
    models = multi_model_manager.list_models()
    return ModelList(data=models)


@app.get("/v1/files/{filename}")
async def get_file(filename: str):
    """Serve uploaded/generated files."""
    print(f"DEBUG get_file: filename={filename}, global_file_path={global_file_path}")
    if global_file_path:
        import os
        file_path = os.path.join(global_file_path, filename)
        print(f"DEBUG get_file: full path={file_path}, exists={os.path.exists(file_path)}")
        if os.path.exists(file_path):
            return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="File not found")
