# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Front-served admin DATA endpoints that need no engine.

The engine handles only generation; pages AND the data they read should come
from the front. These endpoints are backed purely by ``auth.json`` under
``config_dir`` (tokens, users), so the front answers them from disk with no
round-trip to a (possibly GIL-busy) engine — which is why the Tokens/Users pages
otherwise showed no data while a generation was running. Sessions are validated
locally against the same shared secret. Registered before the catch-all proxy so
these paths are served here instead of forwarded to the engine.
"""
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


def register_admin_data(app: FastAPI, config_dir) -> bool:
    if not config_dir:
        return False
    try:
        from pathlib import Path
        from codai.admin.auth import SessionManager
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[front] admin data endpoints not served locally ({exc})", flush=True)
        return False

    sm = SessionManager(Path(config_dir))

    def _user(request: Request) -> Optional[str]:
        for k, v in request.cookies.items():
            if k != "session" and not k.startswith("session_"):
                continue
            if v.endswith(".MUST_CHANGE"):
                v = v[:-12]
            u = sm.validate_session(v)
            if u:
                return u
        return None

    def _admin(request: Request):
        """Return (username, None) if a valid admin session, else (None, response)."""
        u = _user(request)
        if not u:
            return None, JSONResponse({"detail": "Not authenticated"}, status_code=401)
        if not sm.is_admin(u):
            return None, JSONResponse({"detail": "Admin access required"}, status_code=403)
        return u, None

    # ------------------------------------------------------------------ tokens
    @app.get("/admin/api/tokens", include_in_schema=False)
    async def _list_tokens(request: Request):
        _u, err = _admin(request)
        if err:
            return err
        auth_data = sm._load_auth_data()
        out = []
        for t in auth_data.get("tokens", []):
            out.append({"id": t["id"], "name": t["name"], "token": t["token"],
                        "provider": t["provider"], "created_at": t["created_at"],
                        "last_used": t.get("last_used")})
        return JSONResponse(out)

    @app.post("/admin/api/tokens", include_in_schema=False)
    async def _create_token(request: Request):
        _u, err = _admin(request)
        if err:
            return err
        try:
            data = await request.json()
        except Exception:
            data = {}
        name = data.get("name")
        provider = data.get("provider", "openai")
        if not name:
            return JSONResponse({"detail": "Token name is required"}, status_code=400)
        import secrets
        from datetime import datetime, timezone

        def _mut(auth_data):
            tokens = auth_data.setdefault("tokens", [])
            token_id = max([t.get("id", 0) for t in tokens], default=0) + 1
            new_token = {
                "id": token_id, "name": name,
                "token": f"sk-coderai-{secrets.token_hex(32)}",
                "provider": provider,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "last_used": None,
            }
            tokens.append(new_token)
            return True, new_token

        nt = sm.update_auth_data(_mut)
        return JSONResponse({"token": nt["token"], "id": nt["id"],
                             "name": nt["name"], "provider": nt["provider"]})

    @app.delete("/admin/api/tokens/{token_id}", include_in_schema=False)
    async def _delete_token(token_id: int, request: Request):
        _u, err = _admin(request)
        if err:
            return err

        def _mut(auth_data):
            tokens = auth_data.get("tokens", [])
            kept = [t for t in tokens if t["id"] != token_id]
            if len(kept) == len(tokens):
                return False, False
            auth_data["tokens"] = kept
            return True, True

        if not sm.update_auth_data(_mut):
            return JSONResponse({"detail": "Token not found"}, status_code=404)
        return JSONResponse({"success": True})

    # ------------------------------------------------------------------- users
    @app.get("/admin/api/users", include_in_schema=False)
    async def _list_users(request: Request):
        _u, err = _admin(request)
        if err:
            return err
        return JSONResponse(sm.list_users())

    @app.post("/admin/api/users", include_in_schema=False)
    async def _create_user(request: Request):
        _u, err = _admin(request)
        if err:
            return err
        try:
            data = await request.json()
        except Exception:
            data = {}
        new_username = data.get("username")
        password = data.get("password")
        role = data.get("role", "user")
        if not new_username or not password:
            return JSONResponse({"detail": "Username and password required"},
                                status_code=400)
        if not sm.create_user(new_username, password, role):
            return JSONResponse({"detail": "User already exists"}, status_code=400)
        return JSONResponse({"success": True})

    @app.delete("/admin/api/users/{user_id}", include_in_schema=False)
    async def _delete_user(user_id: int, request: Request):
        _u, err = _admin(request)
        if err:
            return err
        users = sm._load_auth_data().get("users", [])
        user = next((u for u in users if u["id"] == user_id), None)
        if not user:
            return JSONResponse({"detail": "User not found"}, status_code=404)
        if not sm.delete_user(user["username"]):
            return JSONResponse({"detail": "Cannot delete user"}, status_code=400)
        return JSONResponse({"success": True})

    print("[front] serving tokens/users data locally (no engine round-trip)",
          flush=True)
    return True
