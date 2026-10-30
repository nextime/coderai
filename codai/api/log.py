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
Request logging middleware for the codai API.
"""

import json
import re
import time
from collections import deque
from fastapi import Request

# In-memory ring buffer of recent API requests (max 50)
_activity: deque = deque(maxlen=50)

# Number of leading characters of a base64/data-URI blob to keep in debug output.
_BLOB_PREVIEW_CHARS = 48
# A string is treated as a binary blob (and truncated) when it is a data: URI, or
# when it is long and made up only of base64 characters (real prompts contain
# spaces/punctuation, so they never match this).
_B64_RE = re.compile(r'^[A-Za-z0-9+/=\s]+$')


def _is_blob(s: str) -> bool:
    if s.startswith("data:"):
        return True
    return len(s) > 256 and bool(_B64_RE.match(s[:256]))


def _redact_blobs(obj):
    """Recursively copy a JSON-ish value, truncating base64/data-URI blobs (e.g.
    init_image, image, mask, character_references) to their first few bytes so the
    debug log stays readable instead of dumping tens of KB of base64."""
    if isinstance(obj, dict):
        return {k: _redact_blobs(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redact_blobs(v) for v in obj]
    if isinstance(obj, str) and _is_blob(obj):
        return f"{obj[:_BLOB_PREVIEW_CHARS]}…[{len(obj)} chars total, truncated]"
    return obj


def _summarise(parsed) -> str:
    """One line describing a request, for the high-volume paths.

    Says what is needed to follow what the server was asked to do -- the model,
    how many items, the shape -- without reproducing the caller's content.
    """
    if not isinstance(parsed, dict):
        return "(no JSON body)"
    bits = []
    model = parsed.get("model")
    if model:
        bits.append(f"model={model}")
    inp = parsed.get("input")
    if isinstance(inp, list):
        bits.append(f"inputs={len(inp)}")
        chars = sum(len(x) for x in inp if isinstance(x, str))
        if chars:
            bits.append(f"chars={chars}")
    elif isinstance(inp, str):
        bits.append(f"inputs=1 chars={len(inp)}")
    # The `image` field, counted. Its absence here cost a real debugging
    # detour: `model=gme-… inputs=4 chars=84` was read as a text-only request,
    # so an mtmd failure on the IMAGE path looked impossible and the search went
    # looking for text being mistaken for pictures. The count is shape, not
    # content -- the same rule the rest of this line follows.
    img = parsed.get("image")
    if isinstance(img, list):
        bits.append(f"images={len(img)}")
    elif isinstance(img, str):
        bits.append("images=1")
    for key in ("dimensions", "encoding_format"):
        if parsed.get(key) is not None:
            bits.append(f"{key}={parsed[key]}")
    return " ".join(bits) or "(empty body)"


def get_recent_activity():
    return list(_activity)


# --- Durable request-debug sink -------------------------------------------
# The FULL REQUEST DEBUG below is normally emitted with print(), i.e. to this
# process's stdout, which the front pumps into the shared log. If an engine's
# stdout stream dies (a native library closing/redirecting fd 1 after model
# load has been observed on the radeon GGUF embedder), that print() output is
# silently lost even though the engine keeps serving. To make the request
# debug survive regardless of stdout, we ALSO append it — from the engine's own
# process — straight to a file on the mounted cache. Each block is written with
# a single O_APPEND write() so concurrent engines never interleave mid-block.
import os as _os

_DURABLE_FD = None
_DURABLE_INIT = False


def _durable_log_path():
    lf = _os.environ.get("CODERAI_LOG_FILE")
    d = (_os.path.dirname(lf) if lf else None) or _os.environ.get(
        "CODERAI_LOG_DIR") or "/cache/logs"
    return _os.path.join(d, "requests.log")


def _durable_fd():
    global _DURABLE_FD, _DURABLE_INIT
    if _DURABLE_INIT:
        return _DURABLE_FD
    _DURABLE_INIT = True
    try:
        path = _durable_log_path()
        _os.makedirs(_os.path.dirname(path), exist_ok=True)
        _DURABLE_FD = _os.open(path, _os.O_WRONLY | _os.O_CREAT | _os.O_APPEND, 0o644)
    except Exception:
        _DURABLE_FD = None
    return _DURABLE_FD


def _engine_tag():
    return (_os.environ.get("CODERAI_ENGINE_NAME")
            or _os.environ.get("CODERAI_ENGINE_BACKEND") or "engine")


def _durable_request_debug(method, url, parsed):
    """Append one FULL REQUEST DEBUG block to the durable requests.log, tagged
    like the pumped lines ([ts][engine]) so it reads the same. Best-effort;
    never raises into the request path."""
    fd = _durable_fd()
    if fd is None:
        return
    try:
        pre = f"[{time.strftime('%H:%M:%S')}][{_engine_tag()}] "
        body = json.dumps(_redact_blobs(parsed), indent=2)
        lines = [pre + "=" * 70,
                 pre + "=== FULL REQUEST DEBUG ===",
                 pre + f"Method: {method}  URL: {url}"]
        lines += [pre + ln for ln in body.split("\n")]
        lines.append(pre + "=" * 70)
        _os.write(fd, ("\n".join(lines) + "\n").encode("utf-8", "replace"))
    except Exception:
        pass


# NOTE: long free text is deliberately NOT truncated here. A prompt is the most
# useful thing in the log -- a 4,000-character script prompt was what identified
# a music defect -- and capping it would trade the diagnosis for bytes. The
# volume problem was one endpoint called per item rather than per user action,
# which _SUMMARY_ONLY_PATHS addresses without touching anything else.

#: Paths whose bodies are SUMMARISED rather than dumped, even under debug.
#:
#: These are called per item, not per user action, so a full dump is the log
#: rather than a part of it. One production day: 32,633 embedding requests out
#: of 33,009 total, and the radeon engine alone produced 426,682 of the log's
#: 548,687 lines -- about 78% -- which filled the 2,000-line view in roughly
#: four minutes and left no record of the model loads and evictions anyone was
#: trying to diagnose. The activity tracking below is unaffected: it is one
#: dict entry per request and is what the admin page reads.
_SUMMARY_ONLY_PATHS = {
    "/v1/embeddings",
}

_TRACKED_PATHS = {
    "/v1/chat/completions": "chat",
    "/v1/completions": "completion",
    "/v1/images/generations": "image",
    "/v1/images/edits": "image-edit",
    "/v1/images/inpaint": "image-inpaint",
    "/v1/images/upscale": "image-upscale",
    "/v1/images/deblur": "image-deblur",
    "/v1/images/unpixelate": "image-unpixelate",
    "/v1/images/outfit": "image-outfit",
    "/v1/images/faceswap": "image-faceswap",
    "/v1/images/depth": "image-depth",
    "/v1/images/segment": "image-segment",
    "/v1/video/generations": "video",
    "/v1/video/upscale": "video-upscale",
    "/v1/video/subtitle": "video-subtitle",
    "/v1/video/interpolate": "video-interpolate",
    "/v1/video/dub": "video-dub",
    "/v1/audio/speech": "tts",
    "/v1/audio/transcriptions": "transcription",
    "/v1/audio/generate": "audio-generate",
    "/v1/audio/clone": "voice-clone",
    "/v1/audio/convert": "voice-convert",
    "/v1/audio/stems": "audio-stems",
    "/v1/audio/cleanup": "audio-cleanup",
    "/v1/embeddings": "embedding",
    "/v1/pipelines/image-to-video": "pipeline-image-to-video",
    "/v1/pipelines/video-dub": "pipeline-video-dub",
    "/v1/pipelines/story": "pipeline-story",
    "/v1/pipelines/audio-dub": "pipeline-audio-dub",
    "/v1/pipelines/audio-understand": "pipeline-audio-understand",
    "/v1/pipelines/audio-music-dub": "pipeline-audio-music-dub",
    "/v1/pipelines/custom": "pipeline-custom",
    "/v1/pipelines/run": "pipeline-run",
}


async def log_requests(request: Request, call_next):
    """Log all incoming requests for debugging."""
    from codai.api.state import get_global_debug
    global_debug = get_global_debug()

    path = request.url.path
    tracked = path in _TRACKED_PATHS

    if tracked or path in ["/v1/chat/completions", "/v1/completions"]:
        body = b""
        body_str = ""
        model = "—"
        if request.method in ("POST", "PUT", "PATCH"):
            try:
                body = await request.body()
                body_str = body.decode('utf-8')
                parsed = json.loads(body_str)
                model = parsed.get("model", "—")

                if global_debug and path in _SUMMARY_ONLY_PATHS:
                    print(f"[req] {request.method} {path} {_summarise(parsed)}")
                elif global_debug:
                    print(f"\n{'='*80}")
                    print(f"=== FULL REQUEST DEBUG ===")
                    print(f"Method: {request.method}  URL: {request.url}")
                    print(json.dumps(_redact_blobs(parsed), indent=2))
                    print(f"{'='*80}\n")
                    # Durable copy, independent of this process's stdout, so the
                    # request body survives even if the engine's log stream dies.
                    _durable_request_debug(request.method, request.url, parsed)
            except Exception as e:
                if global_debug:
                    print(f"Error reading request body: {e}")

        t0 = time.time()
        response = await call_next(request)
        duration = time.time() - t0

        if tracked:
            _activity.appendleft({
                "time": int(t0),
                "model": model,
                "type": _TRACKED_PATHS[path],
                "status": response.status_code,
                "duration": round(duration, 2),
            })

        if global_debug:
            print(f"DEBUG: Response status: {response.status_code}")

        return response
    else:
        return await call_next(request)