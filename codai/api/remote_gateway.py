# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Remote gateway — forward any /v1 request to a coderai running elsewhere.

The isolated-venv subsystems each grew a ``service_url`` of their own, and text
models can be pointed at any OpenAI endpoint (codai/backends/remote_openai.py).
Everything else — images, video, embeddings, rerank, OCR, TTS, STT, voice
cloning, audio generation, stems, 3D, pipelines — loads its model in-process
with no service layer to redirect. Writing thirteen more services would be the
long way round; this is the short one.

Because a coderai exposes the whole API, *another coderai* is a perfectly good
remote for any of them. This middleware sits in front of the routes and, when a
request is addressed to something configured as remote, replays it verbatim to
that endpoint and streams the answer back. The client sees no difference; the
local process never loads the model.

Two levels of routing, most specific first:

1. per model — a models.json entry with ``service_url`` (the same field the
   remote text backend uses), matched against the request's ``model``;
2. per capability — ``remotes.endpoints`` in config.json, e.g.
   ``{"images": "http://gpu-box:8000", "video": "https://pod-xyz/"}``,
   keyed by the capability the path belongs to.

Chat and completions are deliberately NOT handled here: those go through
RemoteOpenAIBackend, which keeps usage accounting, tool parsing and the model
manager's bookkeeping intact.

Cost: when nothing is configured as remote the middleware returns on its first
branch and the body is never buffered.
"""

import json
import os
import re
from typing import Optional, Tuple

#: Path prefix -> capability name. Longest prefix wins, so the specific audio
#: paths are matched before the generic ones.
_CAPABILITY_PREFIXES = (
    ("/v1/audio/transcriptions", "stt"),
    ("/v1/audio/translations", "stt"),
    ("/v1/audio/diarization", "speaker"),
    ("/v1/audio/speaker", "speaker"),
    ("/v1/audio/speakers", "speaker"),
    ("/v1/audio/speech", "tts"),
    ("/v1/audio/clone", "voice"),
    ("/v1/audio/voices", "voice"),
    ("/v1/audio/convert", "voice"),
    ("/v1/audio/watermark", "voice"),
    ("/v1/audio/generate", "audio_gen"),
    # Progress polling has to follow the job: it is the remote that is rendering.
    ("/v1/audio/progress", "audio_gen"),
    ("/v1/audio/stems", "stems"),
    ("/v1/audio/cleanup", "audio_clean"),
    ("/v1/images/to3d", "spatial"),
    ("/v1/images/from3d", "spatial"),
    ("/v1/images/faceswap", "faceswap"),
    ("/v1/images", "images"),
    ("/v1/video/to3d", "spatial"),
    ("/v1/video/from3d", "spatial"),
    ("/v1/video", "video"),
    ("/v1/3d", "spatial"),
    ("/v1/embeddings", "embeddings"),
    ("/v1/rerank", "rerank"),
    ("/v1/ocr", "ocr"),
    ("/v1/loras", "loras"),
    ("/v1/characters", "characters"),
    ("/v1/environments", "environments"),
    ("/v1/pipelines", "pipelines"),
    ("/v1/faceswap", "faceswap"),
)

#: filename -> the remote base that produced it. A generated file lives on the
#: machine that rendered it, and `response_format: "url"` hands the client a URL
#: built from THAT machine's base — unreachable from here, and a 404 if fetched
#: locally. So rewrite those URLs to point at this instance and remember where to
#: fetch them from. Bounded: this is a routing hint, not a store.
_FILE_ORIGINS: "OrderedDict[str, str]" = None
_FILE_ORIGINS_MAX = 4096


def _remember_file(filename: str, base: str) -> None:
    global _FILE_ORIGINS
    from collections import OrderedDict
    if _FILE_ORIGINS is None:
        _FILE_ORIGINS = OrderedDict()
    _FILE_ORIGINS[filename] = base
    _FILE_ORIGINS.move_to_end(filename)
    while len(_FILE_ORIGINS) > _FILE_ORIGINS_MAX:
        _FILE_ORIGINS.popitem(last=False)


def file_origin(filename: str) -> str:
    return (_FILE_ORIGINS or {}).get(filename, "")


#: Handled by RemoteOpenAIBackend instead — see the module docstring.
_SKIP_PATHS = ("/v1/chat/completions", "/v1/completions", "/v1/models")

#: `model` field in a multipart body, without paying for a full form parse.
_MULTIPART_MODEL = re.compile(
    rb'name="model"\r?\n(?:[^\r\n]*\r?\n)*?\r?\n([^\r\n]*)', re.IGNORECASE)


def capability_for(path: str) -> str:
    """The capability a /v1 path belongs to, or '' when it maps to none."""
    p = (path or "").rstrip("/") or "/"
    best = found = ""
    for prefix, cap in _CAPABILITY_PREFIXES:
        # `-` as well as `/`: /v1/audio/speaker-verify is a sibling of
        # /v1/audio/speaker, not a different capability.
        if (p == prefix or p.startswith(prefix + "/") or p.startswith(prefix + "-")) \
                and len(prefix) > len(best):
            best, found = prefix, cap
    return found if best else ""


def _remotes_config():
    try:
        from codai.admin.routes import config_manager
        return getattr(getattr(config_manager, "config", None), "remotes", None)
    except Exception:
        return None


def capability_endpoints() -> dict:
    """The configured capability -> URL map (config first, then env)."""
    cfg = _remotes_config()
    out = {}
    if cfg is not None and getattr(cfg, "enabled", True):
        eps = getattr(cfg, "endpoints", None)
        if isinstance(eps, dict):
            out.update({str(k).strip().lower(): str(v).strip()
                        for k, v in eps.items() if str(v).strip()})
    # CODERAI_REMOTE_IMAGES_URL=… etc. — handy for a container with no config edit.
    for _, cap in _CAPABILITY_PREFIXES:
        if cap in out:
            continue
        val = os.environ.get(f"CODERAI_REMOTE_{cap.upper()}_URL", "").strip()
        if val:
            out[cap] = val
    return out


def capability_pods() -> dict:
    """The configured capability -> RunPod pod block map (``remotes.pods``).

    A capability listed here is served by a coderai-managed pod: provisioned on
    the first request, health-checked, scaled to ``max_pods``, and reaped after
    ``idle_timeout_s`` — the same budgets and lifecycle a RunPod-served LLM gets,
    which is what stops a forgotten image/video pod billing all night.
    """
    cfg = _remotes_config()
    if cfg is None or not getattr(cfg, "enabled", True):
        return {}
    pods = getattr(cfg, "pods", None)
    if not isinstance(pods, dict):
        return {}
    return {str(k).strip().lower(): (v if isinstance(v, dict) else {})
            for k, v in pods.items()}


def _model_remote(model: str) -> str:
    if not model:
        return ""
    try:
        from codai.backends.remote_openai import model_remote_url
        return model_remote_url(model)
    except Exception:
        return ""


#: (expires_at, answer) — `any_remote_configured` runs per request, so don't walk
#: the whole model list every time. A few seconds stale is harmless: the worst
#: case is one request served locally right after a remote is configured.
_ANY_CACHE = [0.0, False]
_ANY_TTL = 5.0


def any_remote_configured() -> bool:
    """True when *something* could be remote. Keeps the hot path free."""
    import time
    now = time.time()
    if now < _ANY_CACHE[0]:
        return _ANY_CACHE[1]
    answer = _any_remote_configured_uncached()
    _ANY_CACHE[0], _ANY_CACHE[1] = now + _ANY_TTL, answer
    return answer


def _any_remote_configured_uncached() -> bool:
    if capability_endpoints() or capability_pods():
        return True
    try:
        from codai.admin.routes import config_manager
        md = getattr(config_manager, "models_data", None)
        if isinstance(md, dict):
            for lst in md.values():
                if isinstance(lst, list):
                    for m in lst:
                        if isinstance(m, dict) and str(m.get("service_url") or "").strip():
                            return True
    except Exception:
        pass
    return False


def _model_from_body(body: bytes, content_type: str) -> str:
    """Pull the requested model out of a JSON or multipart body."""
    if not body:
        return ""
    ctype = (content_type or "").lower()
    if "json" in ctype or body[:1] in (b"{", b"["):
        try:
            obj = json.loads(body)
        except Exception:
            return ""
        return str(obj.get("model") or "") if isinstance(obj, dict) else ""
    if "multipart/form-data" in ctype:
        m = _MULTIPART_MODEL.search(body[:262144])
        if m:
            return m.group(1).decode("utf-8", "replace").strip()
    if "application/x-www-form-urlencoded" in ctype:
        from urllib.parse import parse_qs
        vals = parse_qs(body.decode("utf-8", "replace")).get("model")
        return (vals or [""])[0]
    return ""


class Target:
    """Where a request is going: a fixed URL, or a pod pool to borrow one from."""

    def __init__(self, url: str = "", pool=None, reason: str = ""):
        self.url = (url or "").rstrip("/")
        self.pool = pool
        self.reason = reason

    def acquire(self):
        """Return (base_url, release). A pool provisions on demand and bills."""
        if self.pool is None:
            return self.url, (lambda: None)
        handle, url = self.pool.acquire()
        return url.rstrip("/"), (lambda: self.pool.release(handle))


def resolve_target(path: str, method: str, query: str, body: bytes,
                   content_type: str) -> Optional["Target"]:
    """Decide where this request should go, or None to serve it locally."""
    if not path.startswith("/v1/") or path in _SKIP_PATHS:
        return None
    # A generated file is fetched from whichever remote rendered it.
    if path.startswith("/v1/files/"):
        base = file_origin(path[len("/v1/files/"):])
        return Target(url=base, reason="generated file") if base else None
    cap = capability_for(path)
    model = _model_from_body(body, content_type)
    if not model and query:
        from urllib.parse import parse_qs
        model = (parse_qs(query).get("model") or [""])[0]
    url = _model_remote(model)
    if url:
        return Target(url=url, reason=f"model {model!r}")
    if not cap:
        return None
    url = capability_endpoints().get(cap, "")
    # "runpod" as the endpoint is shorthand for "use the pod block for this
    # capability" — so the common case needs no second config block.
    if url and url.strip().lower() != "runpod":
        return Target(url=url, reason=f"capability {cap!r}")
    pods = capability_pods()
    if url or cap in pods:
        from codai.api.runpod_worker import get_capability_pool
        return Target(pool=get_capability_pool(cap, pods.get(cap) or {}),
                      reason=f"capability {cap!r} on a RunPod pod")
    return None


class RemoteGatewayMiddleware:
    """Replay /v1 requests bound for a remote coderai and stream the answer back."""

    #: Fallback when no config is loaded yet.
    MAX_BODY_MB = float(os.environ.get("CODERAI_REMOTE_MAX_BODY_MB", "512"))

    def __init__(self, app):
        self._app = app

    @property
    def max_body_bytes(self) -> int:
        """Bodies larger than this are served locally rather than buffered in RAM."""
        cfg = _remotes_config()
        try:
            mb = float(getattr(cfg, "max_body_mb", 0) or 0) or self.MAX_BODY_MB
        except (TypeError, ValueError):
            mb = self.MAX_BODY_MB
        return int(mb * 1024 * 1024)

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            return await self._app(scope, receive, send)
        path = scope.get("path") or ""
        if not path.startswith("/v1/") or path in _SKIP_PATHS:
            return await self._app(scope, receive, send)
        if not any_remote_configured():
            return await self._app(scope, receive, send)

        headers = {k.decode("latin-1").lower(): v.decode("latin-1")
                   for k, v in scope.get("headers") or []}
        try:
            clen = int(headers.get("content-length") or 0)
        except ValueError:
            clen = 0
        cap = self.max_body_bytes
        if clen > cap:
            print(f"[remote-gateway] {path}: body {clen / 1e6:.0f} MB exceeds the "
                  f"{cap / 1e6:.0f} MB buffer cap — serving locally", flush=True)
            return await self._app(scope, receive, send)

        body, messages = await _drain(receive, cap)
        replay = _replayer(messages)
        if body is None:                       # over the cap mid-stream
            return await self._app(scope, replay, send)

        try:
            target = resolve_target(path, scope.get("method", "GET"),
                                    (scope.get("query_string") or b"").decode("latin-1"),
                                    body, headers.get("content-type", ""))
        except Exception as exc:
            # A misconfigured remote must not silently fall back to running the
            # model here — that is how you discover it by watching local VRAM.
            return await _error(send, 502, f"remote routing failed: {exc}")
        if not target:
            return await self._app(scope, replay, send)
        await self._forward(scope, headers, body, target, send)

    async def _forward(self, scope, headers, body, target, send):
        import asyncio
        try:
            base, release = await asyncio.to_thread(target.acquire)
        except Exception as exc:
            return await _error(send, 502,
                                f"could not reach a remote for this request: {exc}")
        try:
            await self._send_upstream(scope, headers, body, base, target.reason, send)
        finally:
            release()

    async def _send_upstream(self, scope, headers, body, base, reason, send):
        import asyncio
        path = scope.get("path")
        qs = (scope.get("query_string") or b"").decode("latin-1")
        dest = base.rstrip("/") + path + (f"?{qs}" if qs else "")
        fwd = {k: v for k, v in headers.items()
               if k in ("content-type", "accept", "accept-language")}
        cfg = _remotes_config()
        key = (getattr(cfg, "api_key", "") if cfg else "") \
            or os.environ.get("CODERAI_REMOTE_API_KEY", "")
        if key:
            fwd["Authorization"] = f"Bearer {key}"
        elif headers.get("authorization"):
            fwd["Authorization"] = headers["authorization"]
        print(f"[remote-gateway] {path} -> {base} ({reason})", flush=True)

        def _call():
            import requests
            return requests.request(scope.get("method", "POST"), dest, data=body or None,
                                    headers=fwd, stream=True, timeout=(30, 3600))
        try:
            resp = await asyncio.to_thread(_call)
        except Exception as exc:
            return await _error(send, 502, f"remote endpoint {base} unreachable: {exc}")

        ctype = resp.headers.get("Content-Type") or "application/json"
        out = [(b"content-type", ctype.encode("latin-1"))]
        for h in ("content-disposition", "cache-control"):
            if resp.headers.get(h):
                out.append((h.encode(), resp.headers[h].encode("latin-1")))

        # A JSON answer may carry /v1/files/… URLs built from the REMOTE's base.
        # Those are useless to the client (wrong host, and a 404 if fetched here),
        # so rewrite them to this instance and remember where the file actually
        # is. Small bodies only: everything else streams through untouched.
        if "json" in ctype.lower():
            raw = await asyncio.to_thread(lambda: resp.content)
            resp.close()
            raw = _rewrite_file_urls(raw, base, _local_base(scope, headers))
            out = [h for h in out if h[0] != b"content-length"]
            out.append((b"content-length", str(len(raw)).encode()))
            await send({"type": "http.response.start", "status": resp.status_code,
                        "headers": out})
            return await send({"type": "http.response.body", "body": raw})

        await send({"type": "http.response.start", "status": resp.status_code,
                    "headers": out})
        it = resp.iter_content(chunk_size=65536)
        while True:
            chunk = await asyncio.to_thread(lambda: next(it, None))
            if chunk is None:
                break
            await send({"type": "http.response.body", "body": chunk, "more_body": True})
        await send({"type": "http.response.body", "body": b""})
        resp.close()


#: Any absolute URL pointing at a served file.
_FILE_URL = re.compile(rb'https?://[^"\s]*?/v1/files/([^"\s/?]+)')


def _local_base(scope, headers) -> str:
    """This instance's public base URL, as the client reached it."""
    proto = (headers.get("x-forwarded-proto") or scope.get("scheme") or "http").split(",")[0]
    host = headers.get("host") or ""
    prefix = (headers.get("x-forwarded-prefix") or "").rstrip("/")
    return f"{proto}://{host}{prefix}" if host else prefix


def _rewrite_file_urls(raw: bytes, remote_base: str, local_base: str) -> bytes:
    """Point every /v1/files/ URL at us, and note which remote holds the file."""
    if b"/v1/files/" not in raw:
        return raw

    def _sub(m):
        name = m.group(1).decode("latin-1")
        _remember_file(name, remote_base)
        return (local_base.encode("latin-1") if local_base else b"") + b"/v1/files/" + m.group(1)

    return _FILE_URL.sub(_sub, raw)


async def _error(send, status: int, message: str):
    msg = json.dumps({"error": {"message": message,
                                "type": "remote_gateway_error"}}).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"),
                            (b"content-length", str(len(msg)).encode())]})
    await send({"type": "http.response.body", "body": msg})


async def _drain(receive, cap: int):
    """Buffer the request body so it can be inspected and then replayed.

    Returns ``(body, messages)``; ``body`` is None when the request outgrew the
    cap, in which case the collected messages still replay it faithfully.
    """
    body = bytearray()
    messages = []
    over = False
    while True:
        msg = await receive()
        messages.append(msg)
        if msg["type"] != "http.request":
            break
        if not over:
            body += msg.get("body", b"")
            if len(body) > cap:
                over = True
        if not msg.get("more_body"):
            break
    return (None if over else bytes(body)), messages


def _replayer(messages):
    """A `receive` that hands back the buffered messages, then waits."""
    pending = list(messages)

    async def receive():
        if pending:
            return pending.pop(0)
        return {"type": "http.disconnect"}

    return receive
