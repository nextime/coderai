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
from fastapi.responses import RedirectResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


_FA_WARNED = False
# Rejected forward-auth attempts already reported, as {(peer, reason): monotonic}.
# A refusal is a 302 to the login form and nothing else, which is indistinguishable
# from "no session yet" -- so the one thing an operator needs is WHY. Bounded and
# time-windowed: a misconfigured proxy must not be able to flood the log.
_FA_REJECTS: dict = {}
_FA_REJECT_WINDOW = 60.0


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

    def _fa_cfg():
        """The forward-auth settings, or None when the feature is off.

        Read per request so toggling it in Settings takes effect without a
        restart. Env wins over the file so a container can be switched from its
        unit file without editing a mounted config:
        CODERAI_ADMIN_FORWARD_AUTH=1|0 and CODERAI_ADMIN_FORWARD_AUTH_SECRET.
        """
        import os
        cfg = None
        try:
            from codai.admin import routes as _ar
            mgr = getattr(_ar, "config_manager", None)
            cfg = getattr(getattr(mgr, "config", None), "admin", None)
            cfg = getattr(cfg, "forward_auth", None)
        except Exception:
            cfg = None
        if cfg is None:
            return None
        env_on = os.environ.get("CODERAI_ADMIN_FORWARD_AUTH", "")
        enabled = cfg.enabled
        if env_on != "":
            enabled = env_on.strip().lower() in ("1", "true", "yes", "on")
        if not enabled:
            return None
        secret = os.environ.get("CODERAI_ADMIN_FORWARD_AUTH_SECRET", "") or cfg.shared_secret
        if not secret:
            # Enabled with no secret is not a configuration, it is an open door:
            # anyone who can reach the port could send the header. Stay off and
            # say so once.
            global _FA_WARNED
            if not _FA_WARNED:
                print("[front] admin forward-auth is enabled but no shared secret is "
                      "set — staying OFF (set admin.forward_auth.shared_secret or "
                      "CODERAI_ADMIN_FORWARD_AUTH_SECRET)", flush=True)
                _FA_WARNED = True
            return None
        return cfg, secret

    def _fa_peer_allowlist(cfg) -> list:
        """The trusted-proxy list, with an env override.

        CODERAI_ADMIN_FORWARD_AUTH could already be turned on from a unit file and
        the secret supplied the same way, but the peer list could only come from
        config.json -- so a container whose config is mounted read-only (or, as on
        the Digesta host, whose config was written to a path the app does not read)
        could enable the feature by env and then reject every request from its own
        proxy, because the list in effect was still the default
        ["127.0.0.1", "::1"]. Comma- or space-separated.
        """
        import os
        raw = os.environ.get("CODERAI_ADMIN_FORWARD_AUTH_TRUSTED_PROXIES", "")
        if raw.strip():
            return [t for t in (x.strip() for x in raw.replace(",", " ").split()) if t]
        return [str(t).strip() for t in (cfg.trusted_proxies or []) if str(t).strip()]

    def _fa_peer_allowed(peer: str, trusted: list) -> bool:
        """Is this peer one of the trusted proxies?

        Compared as ADDRESSES, not strings. The internal nginx listens on both
        `8776` and `[::]:8776`, so a connection that lands on the v6 socket makes
        $remote_addr -- and therefore X-Forwarded-For, and therefore the peer --
        the IPv4-mapped form `::ffff:10.89.0.178`. A string membership test
        rejects that even when the operator has listed `10.89.0.178`, which is a
        refusal nothing in the configuration explains. CIDR entries work too, so
        a container network can be trusted as `10.89.0.0/24` rather than by
        pinning one address.

        Anything that is not an address or a network (a unix socket path, say) is
        still compared literally, so an exotic entry keeps working.
        """
        import ipaddress
        if not peer:
            return False
        if peer in trusted:          # the common case, and the cheapest
            return True
        try:
            ip = ipaddress.ip_address(peer)
        except ValueError:
            return False
        mapped = getattr(ip, "ipv4_mapped", None)
        candidates = [ip] + ([mapped] if mapped else [])
        for entry in trusted:
            if "/" in entry:
                try:
                    net = ipaddress.ip_network(entry, strict=False)
                except ValueError:
                    continue
                if any(c in net for c in candidates if c.version == net.version):
                    return True
                continue
            try:
                want = ipaddress.ip_address(entry)
            except ValueError:
                continue
            wmapped = getattr(want, "ipv4_mapped", None)
            if any(c == w for c in candidates for w in ([want] + ([wmapped] if wmapped else []))):
                return True
        return False

    def _fa_reject(peer: str, reason: str) -> bool:
        """Say once, per peer and reason, why a forward-auth attempt was refused."""
        import time
        now = time.monotonic()
        key = (peer, reason)
        last = _FA_REJECTS.get(key)
        if last is not None and (now - last) < _FA_REJECT_WINDOW:
            return False
        if len(_FA_REJECTS) > 64:
            _FA_REJECTS.clear()
        _FA_REJECTS[key] = now
        print(f"[front] admin forward-auth refused: {reason} "
              f"(peer {peer or 'unknown'!r})", flush=True)
        return False

    def _fa_trusted(request: Request, cfg, secret: str) -> bool:
        """Both checks, or nothing: the peer must be a trusted proxy AND the shared
        secret must match. Either one alone is forgeable.

        The peer is ``request.client.host``, which uvicorn's ProxyHeadersMiddleware
        has already rewritten from X-Forwarded-For -- so it is the address the proxy
        reports, not the socket peer, and it is what the access log shows. That
        matters for what belongs in trusted_proxies: on a container network the
        value is the address the proxy is SNATed to, not 127.0.0.1, even though the
        TCP connection into the front is over loopback.
        """
        import hmac
        peer = (request.client.host if request.client else "") or ""
        trusted = _fa_peer_allowlist(cfg)
        if not _fa_peer_allowed(peer, trusted):
            return _fa_reject(peer, f"peer is not in trusted_proxies {trusted}")
        sent = request.headers.get(cfg.shared_secret_header.lower(), "") or ""
        if not sent:
            return _fa_reject(peer, f"no {cfg.shared_secret_header} header")
        if not hmac.compare_digest(sent, secret):
            return _fa_reject(peer, f"{cfg.shared_secret_header} does not match")
        return True

    def _fa_identity(request: Request):
        """(username, is_admin) from a trusted proxy's headers, or (None, False).

        Returns nothing at all unless the trust check passed, so a client that
        sends X-Forwarded-User directly is simply ignored.
        """
        got = _fa_cfg()
        if not got:
            return None, False
        cfg, secret = got
        if not _fa_trusted(request, cfg, secret):
            return None, False
        user = (request.headers.get(cfg.user_header.lower(), "") or "").strip()
        if not user or "/" in user or "\\" in user or len(user) > 128:
            return None, False
        groups_raw = request.headers.get(cfg.groups_header.lower(), None)
        if groups_raw is None:
            is_admin = bool(cfg.admin_without_groups)
        else:
            groups = {g.strip().lower() for g in str(groups_raw).replace(";", ",").split(",")}
            is_admin = (cfg.admin_group or "").strip().lower() in groups
        return user, is_admin

    def _fa_login(request: Request):
        """Establish a normal CoderAI session for a forward-authed user.

        Returns the username, or None. Creates the account on first sight when
        auto_create_users is on; otherwise the user must already exist, so an
        upstream directory cannot mint CoderAI accounts by itself.
        """
        import secrets
        user, is_admin = _fa_identity(request)
        if not user:
            return None, False
        got = _fa_cfg()
        cfg = got[0] if got else None
        known = any(u.get("username") == user
                    for u in (sm._load_auth_data().get("users") or []))
        if not known:
            if not (cfg and cfg.auto_create_users):
                print(f"[front] forward-auth: refusing unknown user {user!r} "
                      f"(auto_create_users is off)", flush=True)
                return None, False
            # A random password nobody is told: this account is only ever
            # reached through the proxy.
            sm.create_user(user, secrets.token_urlsafe(32),
                           role="admin" if is_admin else "user")
            print(f"[front] forward-auth: created user {user!r} "
                  f"(admin={is_admin})", flush=True)
        return user, is_admin

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

    # ------------------------------------------------------- forward auth (SSO)
    # A session, not just a rendered page: the dashboard's own data calls go to
    # /admin/api/*, which authenticate by session cookie. So mint the normal
    # signed session, inject it into THIS request so those handlers see it, and
    # set it on the response so the browser keeps it.
    #
    # /v1/* is explicitly excluded: this feature is GUI-only and must not change
    # how the API authenticates.
    @app.middleware("http")
    async def _forward_auth_session(request: Request, call_next):
        path = request.url.path
        if path.startswith("/v1/") or path in ("/healthz", "/health"):
            return await call_next(request)
        if _user(request):                      # already has a session of ours
            return await call_next(request)
        user, _is_admin = _fa_login(request)
        if not user:
            return await call_next(request)
        cookie = sm.create_session(user)
        if cookie.endswith(".MUST_CHANGE"):     # never force a password change
            cookie = cookie[:-12]               # on an account reached via SSO
        raw = [(k, v) for k, v in request.scope.get("headers", [])
               if k.lower() != b"cookie"]
        existing = request.headers.get("cookie", "")
        merged = f"session={cookie}" + (f"; {existing}" if existing else "")
        raw.append((b"cookie", merged.encode()))
        request.scope["headers"] = raw
        response = await call_next(request)
        try:
            response.set_cookie("session", cookie, httponly=True, samesite="lax",
                                secure=request.url.scheme == "https", path="/")
        except Exception:
            pass
        return response

    # ---------------------------------------------------------------- pages
    @app.get("/login", include_in_schema=False)
    async def _login_page(request: Request):
        if _user(request):
            return _to(request, "/admin")
        got = _fa_cfg()
        if got and not got[0].keep_local_login:
            # The proxy is the only way in. Showing a password box here would
            # invite someone to look for a way around the proxy.
            return JSONResponse(
                {"detail": "This install authenticates through its reverse proxy. "
                           "Reach the GUI through it, or set "
                           "admin.forward_auth.keep_local_login to allow the "
                           "password form as a fallback."},
                status_code=403)
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

    @app.get("/admin/runpod", include_in_schema=False)
    async def _runpod_stats_page(request: Request):
        u, redir = _auth_or_redirect(request, admin=True)
        if redir:
            return redir
        return _tmpl(request, "runpod.html", {"username": u, "is_admin": True})

    @app.get("/admin/cluster", include_in_schema=False)
    async def _cluster_page(request: Request):
        u, redir = _auth_or_redirect(request, admin=True)
        if redir:
            return redir
        return _tmpl(request, "cluster.html", {"username": u, "is_admin": True})

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
