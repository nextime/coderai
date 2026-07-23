# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Simple in-process token-bucket rate limiter middleware.

Each distinct (client-IP, route-prefix) pair gets its own bucket.
Limits are configured via RateLimitConfig.  The defaults below are
intentionally generous; tighten them through the config file or CLI.

Endpoints covered:
  /v1/chat/completions      — expensive LLM inference
  /v1/images/               — image generation
  /v1/audio/                — TTS / STT / audio generation
  /v1/video/                — video generation
  /v1/embeddings            — embedding
  /v1/completions           — legacy completions
"""

import os
import time
import threading
from collections import defaultdict
from typing import Dict, Tuple

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


# Lightweight, read-only generation-progress polls. Clients (e.g. the township
# script) poll these ~once/second WHILE a generation runs, so they must be exempt
# from BOTH auth and rate limiting — otherwise the polls consume the rate budget
# and the actual generation request gets 429'd (and the polls themselves 429,
# leaving the step bar stuck).
_PROGRESS_PATHS = {
    "/v1/images/progress",
    "/v1/video/progress",
    "/v1/audio/progress",
    "/v1/loras/progress",
}


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Reject /v1/ API requests that lack a valid Bearer token or active web session."""

    _EXEMPT_PATHS = _PROGRESS_PATHS

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if not path.startswith("/v1/") or path in self._EXEMPT_PATHS:
            return await call_next(request)

        # Requests from the ASGI broker bridge are in-process and have no real
        # Bearer token. Identify them by the sentinel server tuple set in asgi_bridge.py.
        if request.scope.get("server") == ("internal", 80):
            return await call_next(request)

        # Requests the front relays on behalf of an ALREADY-authenticated caller —
        # the AISBF broker authenticates at the aisbf layer (registration token), so
        # those requests carry no end-user Bearer. The front marks them with the
        # internal shared token in x-coderai-broker-authed. It's unforgeable by
        # external clients (only the front knows the token, and the front strips any
        # client-supplied copy), so trust it and skip the end-user Bearer check.
        # NOTE: this is NOT the plain internal token, which rides on EVERY front→engine
        # request (direct included) — so direct requests still require a real Bearer.
        _int = os.environ.get("CODERAI_INTERNAL_TOKEN")
        if _int and request.headers.get("x-coderai-broker-authed", "") == _int:
            return await call_next(request)

        from codai.admin import routes as _admin_routes
        sm = _admin_routes.session_manager
        if sm is None:
            return await call_next(request)

        # Accept a valid Bearer token
        auth_header = request.headers.get("authorization", "")
        if auth_header.lower().startswith("bearer "):
            token = auth_header[7:].strip()
            if sm.verify_token(token):
                return await call_next(request)

        # Accept a valid web session cookie (logged-in browser user)
        cookie = request.cookies.get("session", "")
        if cookie.endswith(".MUST_CHANGE"):
            cookie = cookie[:-12]
        if cookie and sm.validate_session(cookie):
            return await call_next(request)

        return JSONResponse(
            status_code=401,
            content={
                "error": {
                    "message": "Invalid API key. Provide a valid Bearer token.",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key",
                }
            },
        )


# Per-route-prefix defaults: (max_requests, window_seconds)
_DEFAULT_LIMITS: Dict[str, Tuple[int, int]] = {
    "/v1/chat/completions": (60, 60),
    "/v1/completions":      (60, 60),
    "/v1/images/":          (30, 60),
    "/v1/audio/":           (60, 60),
    "/v1/video/":           (10, 60),
    # Embeddings are cheap and legitimately arrive in bulk (indexers embedding
    # whole corpora at several req/s) — keep this an abuse guard, not a throttle.
    "/v1/embeddings":       (1200, 60),
}

# API prefixes that count against the request queue
_QUEUED_PREFIXES = ("/v1/",)

# Global toggle — set to False to disable rate limiting entirely.
RATE_LIMITING_ENABLED = True


class _Bucket:
    """Fixed-window counter."""
    __slots__ = ("count", "window_start")

    def __init__(self, now: float):
        self.count = 0
        self.window_start = now



class RateLimitMiddleware(BaseHTTPMiddleware):
    """Apply per-IP, per-route-prefix rate limiting to API endpoints."""

    def __init__(self, app, limits: Dict[str, Tuple[int, int]] = None):
        super().__init__(app)
        self._limits = limits or _DEFAULT_LIMITS
        # (client_ip, prefix) → _Bucket
        self._buckets: Dict[Tuple[str, str], _Bucket] = defaultdict(lambda: _Bucket(time.monotonic()))
        self._lock = threading.Lock()

    def _get_prefix(self, path: str) -> str:
        for prefix in self._limits:
            if path.startswith(prefix):
                return prefix
        return ""

    # Lightweight polling endpoints that must never be rate-limited
    _EXEMPT_PATHS = _PROGRESS_PATHS

    async def dispatch(self, request: Request, call_next):
        if not RATE_LIMITING_ENABLED:
            return await call_next(request)

        path = request.url.path

        if path in self._EXEMPT_PATHS:
            return await call_next(request)

        # Queue-size enforcement for authenticated API requests (not for status
        # polls). PER-MODEL: the request is rejected only when its own model's
        # queue is full — other loaded models keep accepting and run in
        # parallel. (request.body() caches, so downstream handlers still read it.)
        if path not in self._EXEMPT_PATHS and any(path.startswith(p) for p in _QUEUED_PREFIXES):
            from codai.queue.manager import queue_manager
            _model = None
            if request.method == "POST" and "json" in (
                    request.headers.get("content-type") or ""):
                try:
                    import json as _json
                    _model = (_json.loads(await request.body() or b"{}")
                              or {}).get("model")
                except Exception:
                    _model = None
            if await queue_manager.is_full_for(_model or ""):
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": {
                            "message": "Server queue is full. Please retry later.",
                            "type": "rate_limit_error",
                            "code": 429,
                        }
                    },
                    headers={"Retry-After": "5"},
                )

        prefix = self._get_prefix(path)
        if not prefix:
            return await call_next(request)

        max_req, window = self._limits[prefix]
        client_ip = (
            request.headers.get("x-forwarded-for", "").split(",")[0].strip()
            or (request.client.host if request.client else "unknown")
        )
        key = (client_ip, prefix)
        now = time.monotonic()

        with self._lock:
            bucket = self._buckets[key]
            if now - bucket.window_start >= window:
                bucket.count = 0
                bucket.window_start = now
            bucket.count += 1
            count = bucket.count

        remaining = max(0, max_req - count)
        reset_at = int(time.time() + (window - (now - self._buckets[key].window_start)))

        if count > max_req:
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "message": "Rate limit exceeded. Please slow down.",
                        "type": "rate_limit_error",
                        "code": 429,
                    }
                },
                headers={
                    "X-RateLimit-Limit": str(max_req),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_at),
                    "Retry-After": str(window),
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(max_req)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_at)
        return response
