# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""The ``coderai-system`` worker app.

A lightweight process (spawned + supervised by the front like an engine) that
owns the slow, I/O-bound, non-GPU work: model cache scanning, HuggingFace
downloads and lookups, and GGUF/HF cache management. Keeping it out of the engine
means those operations never block generation and survive engine restarts; the
front routes the relevant /admin/api/* paths here instead of to an engine.

It mounts only the admin router (whose cache/download/HF handlers import no torch),
so the worker stays light — but it does NOT need to be torch-free; it simply never
does GPU work. Sessions and config are initialized from the shared config_dir, the
same way an engine does.
"""
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


def build_system_app(config, config_dir, internal_port: int = 0) -> FastAPI:
    from codai.admin.routes import router as admin_router
    from codai.admin.routes import init_session_manager, set_config_manager
    from codai.api.app import _InternalAuthMiddleware, _ForwardedPrefixMiddleware
    from codai.config import ConfigManager

    app = FastAPI(title="CoderAI System", docs_url=None, redoc_url=None,
                  openapi_url=None)

    # Sessions + config from the shared config_dir (same auth.json/config.json the
    # engine and front use), so the cookie the front forwards validates here too.
    try:
        init_session_manager(Path(config_dir), port=internal_port or 0)
    except Exception as exc:
        print(f"[system] session manager init failed: {exc}", flush=True)
    try:
        cm = ConfigManager(str(config_dir))
        cm.load()
        set_config_manager(cm)
    except Exception as exc:
        print(f"[system] config manager init failed: {exc}", flush=True)

    app.add_middleware(_ForwardedPrefixMiddleware)
    app.add_middleware(_InternalAuthMiddleware)   # only the front (internal token) may call us

    static_dir = Path(__file__).resolve().parent / "admin" / "static"
    if static_dir.exists():
        from fastapi.staticfiles import StaticFiles
        app.mount("/static/admin", StaticFiles(directory=str(static_dir)),
                  name="admin_static")

    @app.get("/healthz", include_in_schema=False)
    async def _healthz():
        return {"ok": True, "role": "system"}

    @app.get("/internal/engine-state", include_in_schema=False)
    async def _engine_state():
        # The front's supervisor health-polls this. The system worker holds no
        # models and no GPU, so report an empty, always-cheap snapshot.
        return {"ok": True, "pid": os.getpid(), "role": "system",
                "loaded_models": [], "vram": None, "tasks": [], "cooling": None}

    @app.post("/internal/reload-config", include_in_schema=False)
    async def _reload_config(request: Request):
        # Re-read config/models.json so cache scans + downloads see admin edits.
        try:
            cm2 = ConfigManager(str(config_dir))
            cm2.load()
            set_config_manager(cm2)
        except Exception as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
        return JSONResponse({"ok": True})

    app.include_router(admin_router, tags=["Admin"])
    return app
