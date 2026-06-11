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


def get_recent_activity():
    return list(_activity)


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

                if global_debug:
                    print(f"\n{'='*80}")
                    print(f"=== FULL REQUEST DEBUG ===")
                    print(f"Method: {request.method}  URL: {request.url}")
                    print(json.dumps(_redact_blobs(parsed), indent=2))
                    print(f"{'='*80}\n")
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