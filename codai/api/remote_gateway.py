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


#: Handled by RemoteOpenAIBackend instead — see the module docstring. The two
#: transfer endpoints are here for a different reason: an upload is addressed to
#: the instance it is sent to (that is the whole point of sending it), so
#: forwarding one would bounce the weights straight back out again.
_SKIP_PATHS = ("/v1/chat/completions", "/v1/completions", "/v1/models",
               "/v1/models/upload", "/v1/models/uploaded", "/v1/models/register",
               # The test run dispatches its own probe; forwarding the test
               # itself would test the remote's routing, not ours.
               "/v1/models/test",
               # …and this one answers "what does THIS process believe?", which
               # is only meaningful locally. Forwarded, it would report the pod's
               # view of a config the pod does not have — the exact confusion it
               # was added to end.
               "/v1/models/test/state")

#: Orchestration endpoints: a sequence of calls to other endpoints, not a model.
#: They are NEVER forwarded as a whole — the chain stays here and each step goes
#: out as its own /v1 request through the front (codai/broker/asgi_bridge.py), so
#: every step is placed by its OWN model's configuration. Forwarding the
#: orchestration instead would move the logic to a pod and place every step by
#: that pod's catalogue, which is the opposite of what these are for.
_ORCHESTRATION_PREFIXES = ("/v1/pipelines", "/v1/characters", "/v1/environments")

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


def model_placement(model: str) -> tuple:
    """Where this specific model runs: ("local"|"url"|"pod"|"", detail).

    Per-model placement beats the capability map, so two models of the SAME kind
    can differ — one video model pinned local while another runs on a pod. A
    model with nothing set returns "" and falls through to the capability
    setting.
    """
    if not model:
        return "", None
    try:
        from codai.models.manager import _model_entry_for
        entry = _model_entry_for(model) or {}
    except Exception:
        entry = {}
    # An explicit local pin opts this model OUT of a capability-wide remote.
    if str(entry.get("placement") or "").strip().lower() == "local":
        return "local", None
    url = _model_remote(model)
    if url:
        return "url", url
    backend = str(entry.get("backend") or "").strip().lower()
    block = entry.get("runpod") if isinstance(entry.get("runpod"), dict) else None
    if backend == "runpod" and block is not None:
        return "pod", (entry, block)
    return "", None


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


def _model_from_body(body: bytes, content_type: str, field: str = "model") -> str:
    """Pull the requested model out of a JSON or multipart body.

    ``field`` because not every endpoint calls it "model": /v1/ocr names its
    engine in a field called `engine`, so a per-model placement for OCR resolved
    correctly and was then refused by the gateway, which could not see which
    engine the request was for.
    """
    if not body:
        return ""
    ctype = (content_type or "").lower()
    if "json" in ctype or body[:1] in (b"{", b"["):
        try:
            obj = json.loads(body)
        except Exception:
            return ""
        return str(obj.get(field) or "") if isinstance(obj, dict) else ""
    if "multipart/form-data" in ctype:
        pat = (_MULTIPART_MODEL if field == "model"
               else re.compile(rb'name="' + field.encode() +
                               rb'"\r?\n(?:[^\r\n]*\r?\n)*?\r?\n([^\r\n]*)',
                               re.IGNORECASE))
        m = pat.search(body[:262144])
        if m:
            return m.group(1).decode("utf-8", "replace").strip()
    if "application/x-www-form-urlencoded" in ctype:
        from urllib.parse import parse_qs
        vals = parse_qs(body.decode("utf-8", "replace")).get(field)
        return (vals or [""])[0]
    return ""


#: Endpoints whose "which model" field is not called `model`.
_MODEL_FIELD = {"/v1/ocr": "engine", "/v1/ocr/batch": "engine"}


class Target:
    """Where a request is going: a fixed URL, or a pod pool to borrow one from."""

    def __init__(self, url: str = "", pool=None, reason: str = ""):
        self.url = (url or "").rstrip("/")
        self.pool = pool
        self.reason = reason
        self.api_key = ""
        #: The model entry whose weights must be pushed before serving, when the
        #: remote has no way to fetch them (`source: "upload"`).
        self.upload_entry = None
        #: The model entry whose LOCAL text adapters must be pushed before the
        #: pod loads the model. Only a coderai pod can receive them.
        self.lora_entry = None

    def acquire(self):
        """Return (base_url, release). A pool provisions on demand and bills."""
        if self.pool is None:
            return self.url, (lambda: None)
        handle, url = self.pool.acquire()
        # The pod is locked to the pool's bearer token; carry it or every
        # forwarded request comes back 401.
        self.api_key = getattr(self.pool, "api_key", "") or self.api_key
        return url.rstrip("/"), (lambda: self.pool.release(handle))


#: Header that overrides placement for ONE request: "local" keeps it here,
#: "remote" insists on the configured remote. It exists for the test run — you
#: cannot verify that a pod works by sending it a request that the configuration
#: decides to serve locally — and is never sent by normal clients.
PLACEMENT_HEADER = "x-coderai-placement"


def resolve_target(path: str, method: str, query: str, body: bytes,
                   content_type: str, force: str = "") -> Optional["Target"]:
    """Decide where this request should go, or None to serve it locally.

    ``force`` is the one-request override: "local" pins it here whatever the
    configuration says, "remote" refuses to fall back to local so a test cannot
    quietly pass by running in the wrong place.
    """
    force = (force or "").strip().lower()
    if force == "local":
        return None
    if not path.startswith("/v1/") or path in _SKIP_PATHS:
        return None
    if path.startswith(_ORCHESTRATION_PREFIXES):
        return None
    # A generated file is fetched from whichever remote rendered it.
    if path.startswith("/v1/files/"):
        base = file_origin(path[len("/v1/files/"):])
        return Target(url=base, reason="generated file") if base else None
    cap = capability_for(path)
    field = _MODEL_FIELD.get(path, "model")
    model = _model_from_body(body, content_type, field)
    if not model and query:
        from urllib.parse import parse_qs
        model = (parse_qs(query).get(field) or [""])[0]

    # Per model first, so two models of the same kind can be placed differently.
    where, detail = model_placement(model)
    if where == "local":
        return None                              # pinned here, whatever the capability says
    if where == "url":
        return Target(url=detail, reason=f"model {model!r}")
    if where == "pod":
        entry, block = detail
        from codai.api.runpod_worker import (get_model_pod_pool, parse_model_runpod,
                                             resolve_model_source)
        target = Target(pool=get_model_pod_pool(model, entry, block),
                        reason=f"model {model!r} on its own RunPod pod")
        mcfg = parse_model_runpod(block)
        kind, _ = resolve_model_source(entry, mcfg)
        if kind == "upload":
            target.upload_entry = entry
        # Local adapters only reach a coderai pod: vLLM and llama.cpp images have
        # no endpoint to receive an upload.
        try:
            from codai.api.runpod_worker import resolve_pod_engine
            from codai.models.text_loras import configured_specs, portable_specs
            if resolve_pod_engine(mcfg, model, str(entry.get("path") or ""), entry) \
                    in ("coderai", "custom"):
                _, needs_sending = portable_specs(configured_specs(entry))
                if needs_sending:
                    target.lora_entry = entry
        except Exception:
            pass
        return target
    if not cap:
        if force == "remote":
            raise RuntimeError(
                f"nothing configures {model or path!r} to run remotely — set a "
                "service_url or a RunPod pod on the model, or a remote for its "
                "capability")
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
    if force == "remote":
        raise RuntimeError(
            f"nothing configures {model or cap!r} to run remotely — set a "
            "service_url or a RunPod pod on the model, or a remote for its "
            "capability")
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
        headers_early = {k.decode("latin-1").lower(): v.decode("latin-1")
                         for k, v in scope.get("headers") or []}
        if headers_early.get(PLACEMENT_HEADER, "").strip().lower() == "local":
            return await self._app(scope, receive, send)
        if not any_remote_configured() \
                and headers_early.get(PLACEMENT_HEADER, "").strip().lower() != "remote":
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
                                    body, headers.get("content-type", ""),
                                    force=headers.get(PLACEMENT_HEADER, ""))
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
            # Weights the remote cannot fetch itself, when asked to upload them.
            if target.upload_entry:
                await asyncio.to_thread(ensure_model_uploaded, base, target.api_key,
                                        target.upload_entry)
            # A text LoRA is applied when the model LOADS, so the adapter has to
            # be on the pod before the first request — not carried by it.
            if target.lora_entry is not None:
                await asyncio.to_thread(ensure_text_loras, base, target.api_key,
                                        target.lora_entry)
                target.lora_entry = None        # sent once per pod
            # Ship any LoRA the request names, so the remote can actually load it.
            if b'"loras"' in (body or b"") and "json" in headers.get("content-type", ""):
                body = await asyncio.to_thread(
                    sync_loras, base, target.api_key, body)
            await self._send_upstream(scope, headers, body, base, target.reason, send,
                                      api_key=target.api_key)
        finally:
            release()

    async def _send_upstream(self, scope, headers, body, base, reason, send,
                             api_key: str = "", retried: bool = False):
        import asyncio
        path = scope.get("path")
        qs = (scope.get("query_string") or b"").decode("latin-1")
        dest = base.rstrip("/") + path + (f"?{qs}" if qs else "")
        fwd = {k: v for k, v in headers.items()
               if k in ("content-type", "accept", "accept-language")}
        cfg = _remotes_config()
        # A pod's own token wins: it is the one that pod was launched with.
        key = api_key or (getattr(cfg, "api_key", "") if cfg else "") \
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

            # "I do not know that model" is recoverable: teach the remote and try
            # once more. This is what makes a pod an extension of this system
            # rather than a fixed catalogue frozen at boot — any request may name
            # a model the pod has never heard of.
            if _is_unknown_model(resp.status_code, raw) and not retried:
                model = _model_from_body(body, headers.get("content-type", ""))
                if await asyncio.to_thread(teach_model, base, api_key, model):
                    return await self._send_upstream(
                        scope, headers, body, base, reason, send,
                        api_key=api_key, retried=True)

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


def sync_loras(base: str, api_key: str, body: bytes) -> bytes:
    """Make sure a remote can load the LoRAs this request names.

    A LoRA lives as a file on THIS machine — trained here, or uploaded here. A
    pod has never seen it, so a request naming `{"model": "/AI/loras/x.safetensors"}`
    would fail there. The blob store is content-addressed, so each adapter is
    hashed, checked against the remote's /v1/loras/blob/<hash>, uploaded only
    when missing, and the reference rewritten to that id. Repeat requests upload
    nothing.

    QLoRA adapters need no special handling: the quantisation lives in how the
    BASE model is loaded, while the adapter is the same safetensors file.

    Anything that cannot be resolved to a local file — an HF repo id, a URL —
    is left untouched: the remote can fetch those itself. Failure anywhere
    returns the body unchanged, so the request still goes out and fails (or
    succeeds) on the remote's own terms rather than here.
    """
    import hashlib
    import requests

    try:
        obj = json.loads(body or b"{}")
    except Exception:
        return body
    if not isinstance(obj, dict):
        return body
    specs = obj.get("loras")
    if not isinstance(specs, list) or not specs:
        return body

    from codai.api.loras import resolve_lora_ref
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    changed = False
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        if str(spec.get("id") or "").startswith("sha256:"):
            continue                       # already content-addressed
        try:
            local = resolve_lora_ref(spec)
        except Exception:
            local = None
        if not local or not os.path.isfile(local):
            continue                       # an HF id or URL — the remote resolves it
        try:
            with open(local, "rb") as fh:
                data = fh.read()
            digest = hashlib.sha256(data).hexdigest()
            have = requests.get(f"{base}/v1/loras/blob/{digest}",
                                headers=headers, timeout=30)
            if have.status_code != 200:
                up = requests.post(f"{base}/v1/loras/upload", data=data,
                                   headers={**headers,
                                            "Content-Type": "application/octet-stream"},
                                   timeout=1800)
                up.raise_for_status()
                print(f"[remote-gateway] uploaded LoRA {os.path.basename(local)} "
                      f"({len(data) / 1e6:.0f} MB) to {base}", flush=True)
        except Exception as exc:
            print(f"[remote-gateway] could not send LoRA {local} to {base}: {exc}",
                  flush=True)
            continue
        # Keep weight/name; drop the local path, which means nothing over there.
        for key in ("model", "path", "url", "file", "data"):
            spec.pop(key, None)
        spec["id"] = f"sha256:{digest}"
        changed = True
    return json.dumps(obj).encode() if changed else body


#: (base_url, model name) already pushed — a pod keeps what it was sent, so the
#: check is worth skipping on the second request. A new pod has a new URL, so it
#: pays the upload again: pod storage is disposable.
_UPLOADED: set = set()


def push_lora_file(base: str, api_key: str, path: str) -> str:
    """Put one adapter file in a remote's blob store; return its ``sha256:`` id.

    Content-addressed, so a pod that already has it is told nothing and a repeat
    costs one HEAD-shaped check. Returns '' when the transfer fails, which the
    caller reports rather than silently serving a model without its adapter.
    """
    import hashlib
    import requests

    if not path or not os.path.isfile(path):
        return ""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        with open(path, "rb") as fh:
            data = fh.read()
        digest = hashlib.sha256(data).hexdigest()
        have = requests.get(f"{base}/v1/loras/blob/{digest}", headers=headers, timeout=30)
        if have.status_code != 200:
            up = requests.post(f"{base}/v1/loras/upload", data=data,
                               headers={**headers,
                                        "Content-Type": "application/octet-stream"},
                               timeout=1800)
            up.raise_for_status()
            print(f"[remote-gateway] sent adapter {os.path.basename(path)} "
                  f"({len(data) / 1e6:.0f} MB) to {base}", flush=True)
        return f"sha256:{digest}"
    except Exception as exc:
        print(f"[remote-gateway] could not send adapter {path} to {base}: {exc}",
              flush=True)
        return ""


def ensure_text_loras(base: str, api_key: str, entry: dict) -> dict:
    """Send a text model's LOCAL adapters to a coderai pod.

    A text LoRA is part of how the model loads, not something the request
    carries, so it has to be on the pod before the model is loaded there. Only a
    coderai pod can receive one — a vLLM or llama.cpp pod has no endpoint to
    upload to, which is why those need a HuggingFace repo id instead.

    Returns the config the REMOTE should use: the same entry with local adapter
    paths replaced by the content ids it now holds. An adapter that could not be
    sent is dropped from that config, so the pod loads the model without it
    rather than failing on a path it cannot see.
    """
    from codai.models.text_loras import configured_specs, local_path, is_portable

    specs = configured_specs(entry or {})
    if not specs:
        return dict(entry or {})

    rewritten, dropped = [], []
    for spec in specs:
        source = str(spec.get("source") or "")
        if is_portable(source):
            rewritten.append({**spec, "source": source})
            continue
        path = local_path(source)
        if not path:
            dropped.append(source)
            continue
        if os.path.isdir(path):
            # A PEFT adapter directory is several files; the blob store holds
            # single files. Publishing it to HuggingFace is the way to move it.
            dropped.append(source)
            print(f"[remote-gateway] {source} is a directory — publish it to "
                  "HuggingFace to use it on a pod", flush=True)
            continue
        blob = push_lora_file(base, api_key, path)
        if blob:
            rewritten.append({**spec, "source": blob})
        else:
            dropped.append(source)

    out = {k: v for k, v in (entry or {}).items()
           if k not in ("lora_path", "lora_model_dir", "loras", "lora_scale")}
    if rewritten:
        out["loras"] = [{"path": s["source"], "weight": s.get("weight", 1.0),
                         "name": s.get("name")} for s in rewritten]
    if dropped:
        print(f"[remote-gateway] adapters not available remotely: {', '.join(dropped)}",
              flush=True)
    return out


#: (base_url, model) already taught to a remote. A pod keeps what it learns, and
#: a new pod has a new URL, so this never goes stale in a harmful way.
_REGISTERED: set = set()


def teach_model(base: str, api_key: str, model: str) -> bool:
    """Tell a remote about a model it does not know, so it can fetch and serve it.

    A pod is seeded at boot with what we knew then. Making it an EXTENSION of the
    local system means any request can name any model afterwards — so when a
    remote says "not available", we send it that model's entry (resolved to
    something it can fetch) and let it try again. Uploading is used only when the
    model exists nowhere else.

    Returns True when something was registered, i.e. a retry is worth making.
    """
    import requests

    if not model or (base, model) in _REGISTERED:
        return False
    try:
        from codai.models.manager import _model_entry_for
        entry = _model_entry_for(model) or {}
    except Exception:
        entry = {}
    if not entry:
        return False

    from codai.api.runpod_worker import (parse_model_runpod, resolve_model_source,
                                         seed_model_env)
    block = entry.get("runpod") if isinstance(entry.get("runpod"), dict) else {}
    mcfg = parse_model_runpod(block)
    kind, value = resolve_model_source(entry, mcfg)
    if kind == "upload":
        ensure_model_uploaded(base, api_key, entry)
        _REGISTERED.add((base, model))
        return True
    if kind not in ("hf", "url") or not value:
        print(f"[remote-gateway] cannot teach {model!r} to {base}: no HuggingFace id "
              "or URL it could fetch — set hf_repo/model_url, or source: upload",
              flush=True)
        return False

    payload = seed_model_env(entry, "", value)
    if not payload:
        return False
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        r = requests.post(f"{base}/v1/models/register", data=payload,
                          headers=headers, timeout=60)
        if not r.ok:
            print(f"[remote-gateway] {base} refused {model!r}: "
                  f"{r.status_code} {(r.text or '')[:160]}", flush=True)
            return False
        print(f"[remote-gateway] taught {base} about {model!r} ({value})", flush=True)
        _REGISTERED.add((base, model))
        return True
    except Exception as exc:
        print(f"[remote-gateway] could not teach {model!r} to {base}: {exc}", flush=True)
        return False


def ensure_model_uploaded(base: str, api_key: str, entry: dict) -> None:
    """Push a model's weights to a remote that cannot fetch them itself.

    Only for `source: "upload"` — coderai resolves a HuggingFace repo id or a URL
    first, because those cost nothing here and run at datacenter speed. Uploading
    is the deliberate fallback for weights that exist nowhere else: a local merge,
    a file whose host is gone, something you are not going to publish.
    """
    import requests

    path = str((entry or {}).get("path") or "")
    if not path or not os.path.exists(path):
        return
    name = os.path.basename(path.rstrip("/"))
    if (base, name) in _UPLOADED:
        return
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    is_dir = os.path.isdir(path)
    size = (os.path.getsize(path) if not is_dir else
            sum(os.path.getsize(os.path.join(r, f))
                for r, _, fs in os.walk(path) for f in fs))
    try:
        have = requests.get(f"{base}/v1/models/uploaded",
                            params={"name": name, "bytes": 0 if is_dir else size},
                            headers=headers, timeout=30)
        if have.status_code == 200:
            _UPLOADED.add((base, name))
            return
    except Exception as exc:
        print(f"[remote-gateway] upload check failed for {name}: {exc}", flush=True)
        return

    section = (entry.get("model_type") or "text_models")
    params = {"name": name, "model_type": section, "tar": 1 if is_dir else 0}
    if entry.get("alias"):
        params["alias"] = entry["alias"]
    print(f"[remote-gateway] uploading {name} ({size / 1e9:.1f} GB) to {base} — "
          "this is the cold cost of `source: upload`", flush=True)
    try:
        if is_dir:
            import subprocess
            proc = subprocess.Popen(
                ["tar", "-cf", "-", "-C", os.path.dirname(path), name],
                stdout=subprocess.PIPE)
            resp = requests.post(f"{base}/v1/models/upload", params=params,
                                 data=proc.stdout, headers=headers, timeout=None)
            proc.stdout.close()
            proc.wait()
        else:
            with open(path, "rb") as fh:
                resp = requests.post(f"{base}/v1/models/upload", params=params,
                                     data=fh, headers=headers, timeout=None)
        resp.raise_for_status()
        _UPLOADED.add((base, name))
        print(f"[remote-gateway] uploaded {name} to {base}", flush=True)
    except Exception as exc:
        # Don't cache a failure: the next request retries rather than serving
        # from a pod that has half a model.
        print(f"[remote-gateway] upload of {name} to {base} failed: {exc}", flush=True)


#: What a coderai says when a model is not in its catalogue.
_UNKNOWN_MODEL = ("is not available", "not allowed", "unknown model",
                  "model not found")


def _is_unknown_model(status: int, raw: bytes) -> bool:
    """True when the remote refused because it does not know the model."""
    if status not in (400, 404):
        return False
    try:
        text = (raw or b"").decode("utf-8", "replace").lower()
    except Exception:
        return False
    return any(marker in text for marker in _UNKNOWN_MODEL)


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
