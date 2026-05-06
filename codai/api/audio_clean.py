import base64
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from codai.api.audio_backends import detect_audio_backends

router = APIRouter()

global_args = None
global_file_path = None


def set_global_args(args):
    global global_args
    global_args = args


def set_global_file_path(path):
    global global_file_path
    global_file_path = path


def _decode_audio(data: str) -> bytes:
    if data.startswith("data:"):
        _, enc = data.split(",", 1)
        return base64.b64decode(enc)
    return base64.b64decode(data)


def _ffmpeg_binary() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise HTTPException(status_code=501, detail="ffmpeg is required for native audio cleanup")
    return ffmpeg


def _base_url(http_request: Request) -> str:
    url_setting = getattr(global_args, "url", "auto") if global_args else "auto"
    if url_setting != "auto":
        return url_setting.rstrip("/")
    host = http_request.headers.get("host", "127.0.0.1") if http_request else "127.0.0.1"
    if ":" in host:
        parts = host.split(":")
        if len(parts) == 2 and parts[1].isdigit():
            host = parts[0]
    proto = "https" if getattr(global_args, "https", False) else "http"
    port = getattr(global_args, "port", 8000) if global_args else 8000
    return f"{proto}://{host}:{port}"


def _persist_file(path: str, suffix: str, http_request: Request) -> dict:
    data = Path(path).read_bytes()
    if global_file_path:
        os.makedirs(global_file_path, exist_ok=True)
        filename = f"{uuid.uuid4().hex}{suffix}"
        out_path = os.path.join(global_file_path, filename)
        with open(out_path, "wb") as handle:
            handle.write(data)
        return {"url": f"{_base_url(http_request)}/v1/files/{filename}"}
    return {f"b64_{suffix.lstrip('.')}": base64.b64encode(data).decode("ascii")}


def _run_ffmpeg(command):
    proc = subprocess.run(command, capture_output=True, text=True)
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or "ffmpeg command failed"
        raise HTTPException(status_code=500, detail=detail)


def restore_with_provider(audio_bytes: bytes, options: dict, workdir: str) -> dict:
    raise HTTPException(status_code=501, detail="ML audio restoration backend not installed")


def _cleanup_audio(audio_bytes: bytes, options: dict, workdir: str) -> dict:
    ffmpeg = _ffmpeg_binary()
    src = os.path.join(workdir, "input.wav")
    dst = os.path.join(workdir, "cleaned.wav")
    with open(src, "wb") as handle:
        handle.write(audio_bytes)

    filters = []
    applied = []
    if options.get("noise_reduction"):
        filters.append("afftdn=nf=-25")
        applied.append("noise_reduction")
    if options.get("remove_hum"):
        filters.append("highpass=f=60,lowpass=f=15000")
        applied.append("remove_hum")
    if options.get("repair_clicks"):
        filters.append("adeclick=t=40")
        applied.append("repair_clicks")
    if options.get("normalize"):
        filters.append("loudnorm=I=-16:TP=-1.5:LRA=11")
        applied.append("normalize")

    if not filters:
        raise HTTPException(status_code=400, detail="Select at least one cleanup operation")

    _run_ffmpeg([ffmpeg, "-y", "-i", src, "-af", ",".join(filters), dst])
    return {
        "path": dst,
        "engine": "ffmpeg-filter-chain",
        "applied": applied,
        "limitations": [
            "best-effort cleanup only",
            "not equivalent to spectral or ML restoration",
            "heavy damage may remain audible",
        ],
    }


class AudioCleanupRequest(BaseModel):
    audio: str
    noise_reduction: Optional[bool] = True
    normalize: Optional[bool] = False
    remove_hum: Optional[bool] = False
    repair_clicks: Optional[bool] = False
    response_format: Optional[str] = "url"
    fallback_mode: Optional[bool] = False
    model_config = ConfigDict(extra="allow")


@router.post("/v1/audio/cleanup")
async def cleanup_audio(request: AudioCleanupRequest, http_request: Request = None):
    try:
        audio_bytes = _decode_audio(request.audio)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid audio payload: {exc}")

    options = {
        "noise_reduction": bool(request.noise_reduction),
        "normalize": bool(request.normalize),
        "remove_hum": bool(request.remove_hum),
        "repair_clicks": bool(request.repair_clicks),
    }
    with tempfile.TemporaryDirectory(prefix="codai-clean-") as workdir:
        backend_info = detect_audio_backends()["restoration"]
        if request.fallback_mode:
            result = _cleanup_audio(audio_bytes, options, workdir)
            quality = "best-effort"
            dependency = "ffmpeg"
            model_name = None
        else:
            result = restore_with_provider(audio_bytes, options, workdir)
            quality = "ml"
            dependency = "python"
            model_name = result.get("model")
        payload = _persist_file(result["path"], ".wav", http_request)
        return {
            "created": int(time.time()),
            "backend": {
                "engine": result["engine"],
                "model": model_name,
                "quality": quality,
                "dependency": dependency,
                "ml_backend_available": backend_info["available"],
                "preferred_engine": backend_info["engine"],
            },
            "applied": result["applied"],
            "limitations": result["limitations"],
            "data": [payload],
        }
