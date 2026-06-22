# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Serve the admin / Studio UI pages directly from the front proxy.

Architecture: the engine handles only generation; the front owns the web UI.
Previously every page navigation was reverse-proxied to the primary engine, so
opening a page while that engine was mid-generation (its event loop busy) made
the whole UI appear to hang until the generation finished.

These page GETs are now rendered by the front itself, with sessions validated
LOCALLY: the cookie is a value signed with a secret stored under ``config_dir``
(shared with the engine via ``auth.json``), so the front can authenticate it
without any round-trip to a (possibly busy) engine. Mutating auth actions
(login / logout / change-password POST) and all ``/admin/api/*`` data calls
still fall through to the catch-all proxy and reach the engine.
"""
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles


def register_ui_pages(app: FastAPI, config_dir) -> bool:
    """Register local UI page + static routes on the front app.

    Must be called BEFORE the catch-all reverse-proxy route so these paths are
    served locally instead of being forwarded to an engine. Returns True when
    wired up, False if it couldn't (no config_dir / templates missing) — in which
    case the catch-all keeps proxying pages to the engine as before.
    """
    if not config_dir:
        return False
    try:
        from codai.admin.auth import SessionManager
        from codai.admin.routes import (
            templates, templates_dir, _tmpl, _default_whisper_server_path)
        from codai.api.urlutils import get_public_prefix
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[front] UI pages not served locally ({exc}); proxying to engine",
              flush=True)
        return False

    cfg_path = Path(config_dir)
    if not Path(templates_dir).exists():
        return False
    sm = SessionManager(cfg_path)

    def _user(request: Request) -> Optional[str]:
        """Locally validate whichever ``session``/``session_<port>`` cookie is
        present. Validation is by HMAC signature against the shared secret, so the
        exact (port-derived) cookie name doesn't matter to the front."""
        for k, v in request.cookies.items():
            if k != "session" and not k.startswith("session_"):
                continue
            if v.endswith(".MUST_CHANGE"):
                v = v[:-12]
            u = sm.validate_session(v)
            if u:
                return u
        return None

    def _to(request: Request, path: str):
        return RedirectResponse(url=get_public_prefix(request) + path, status_code=302)

    def _auth_or_redirect(request: Request, admin: bool = False):
        """Return (username, None) when allowed, or (None, RedirectResponse)."""
        u = _user(request)
        if not u:
            return None, _to(request, "/login")
        if admin and not sm.is_admin(u):
            return None, _to(request, "/admin")
        return u, None

    # ---------------------------------------------------------------- pages
    @app.get("/login", include_in_schema=False)
    async def _login_page(request: Request):
        if _user(request):
            return _to(request, "/admin")
        return _tmpl(request, "login.html", {"error": None})

    @app.get("/admin", include_in_schema=False)
    async def _dashboard(request: Request):
        u, redir = _auth_or_redirect(request)
        if redir:
            return redir
        return _tmpl(request, "dashboard.html",
                     {"username": u, "is_admin": sm.is_admin(u)})

    @app.get("/chat", include_in_schema=False)
    async def _chat(request: Request):
        u, redir = _auth_or_redirect(request)
        if redir:
            return redir
        return _tmpl(request, "chat.html",
                     {"username": u, "is_admin": sm.is_admin(u)})

    @app.get("/admin/change-password", include_in_schema=False)
    async def _change_password(request: Request):
        u, redir = _auth_or_redirect(request)
        if redir:
            return redir
        user = sm.get_user(u)
        return _tmpl(request, "change_password.html", {
            "username": u, "is_admin": sm.is_admin(u),
            "must_change": user.get("must_change_password", False) if user else False,
            "error": None,
        })

    @app.get("/admin/models", include_in_schema=False)
    async def _models_page(request: Request):
        u, redir = _auth_or_redirect(request, admin=True)
        if redir:
            return redir
        return _tmpl(request, "models.html", {
            "username": u, "is_admin": True,
            "default_whisper_server_path": _default_whisper_server_path(),
        })

    @app.get("/admin/tokens", include_in_schema=False)
    async def _tokens_page(request: Request):
        u, redir = _auth_or_redirect(request, admin=True)
        if redir:
            return redir
        return _tmpl(request, "tokens.html", {"username": u, "is_admin": True})

    @app.get("/admin/users", include_in_schema=False)
    async def _users_page(request: Request):
        u, redir = _auth_or_redirect(request, admin=True)
        if redir:
            return redir
        return _tmpl(request, "users.html", {
            "username": u, "is_admin": True, "users": sm.list_users()})

    @app.get("/admin/tasks", include_in_schema=False)
    async def _tasks_page(request: Request):
        u, redir = _auth_or_redirect(request, admin=True)
        if redir:
            return redir
        return _tmpl(request, "tasks.html", {"username": u, "is_admin": True})

    @app.get("/admin/settings", include_in_schema=False)
    async def _settings_page(request: Request):
        u, redir = _auth_or_redirect(request, admin=True)
        if redir:
            return redir
        return _tmpl(request, "settings.html", {"username": u, "is_admin": True})

    @app.get("/admin/archive", include_in_schema=False)
    async def _archive_page(request: Request):
        u, redir = _auth_or_redirect(request, admin=True)
        if redir:
            return redir
        return _tmpl(request, "archive.html", {"username": u, "is_admin": True})

    # ---------------------------------------------------------------- static
    static_dir = Path(__file__).resolve().parent.parent / "admin" / "static"
    if static_dir.exists():
        app.mount("/static/admin",
                  StaticFiles(directory=str(static_dir)), name="front_admin_static")

        @app.get("/favicon.ico", include_in_schema=False)
        async def _favicon():
            return FileResponse(str(static_dir / "favicon.ico"))

    print("[front] serving UI pages locally (engine handles generation only)",
          flush=True)
    return True
