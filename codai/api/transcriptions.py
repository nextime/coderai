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
Audio transcription endpoint for the codai API.
"""

import io
import os
import tempfile

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import PlainTextResponse
from typing import Optional

# Maximum upload size: 100 MB
_MAX_AUDIO_BYTES = 100 * 1024 * 1024

# Safe audio extensions (user-supplied extension is NOT trusted for the suffix)
_SAFE_EXTENSIONS = {'.wav', '.mp3', '.ogg', '.flac', '.m4a', '.webm', '.mp4'}

# Import from codai modules
from codai.models.manager import multi_model_manager


# Global reference to be set by coderai
global_args = None


def set_global_args(args):
    """Set global args from coderai."""
    global global_args
    global_args = args


# =============================================================================
# Response formatting helpers
# =============================================================================

def _seconds_to_srt_time(s: float) -> str:
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:06.3f}".replace('.', ',')


def _seconds_to_vtt_time(s: float) -> str:
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:06.3f}"


def _format_response(fmt: str, text: str, segments: list):
    """Format a transcription result according to the requested response_format."""
    fmt = (fmt or "json").lower()

    if fmt == "text":
        return PlainTextResponse(text)

    if fmt == "srt":
        lines = []
        for i, seg in enumerate(segments, 1):
            start = _seconds_to_srt_time(seg.get("start", 0))
            end = _seconds_to_srt_time(seg.get("end", 0))
            lines.append(f"{i}\n{start} --> {end}\n{seg['text'].strip()}\n")
        srt_body = "\n".join(lines) if lines else f"1\n00:00:00,000 --> 00:00:00,000\n{text}\n"
        return PlainTextResponse(srt_body, media_type="text/plain")

    if fmt == "vtt":
        lines = ["WEBVTT\n"]
        for seg in segments:
            start = _seconds_to_vtt_time(seg.get("start", 0))
            end = _seconds_to_vtt_time(seg.get("end", 0))
            lines.append(f"{start} --> {end}\n{seg['text'].strip()}\n")
        if not segments:
            lines.append(f"00:00:00.000 --> 00:00:00.000\n{text}\n")
        return PlainTextResponse("\n".join(lines), media_type="text/vtt")

    if fmt == "verbose_json":
        return {
            "task": "transcribe",
            "language": "unknown",
            "duration": segments[-1].get("end", 0) if segments else 0,
            "text": text,
            "segments": [
                {
                    "id": i,
                    "start": s.get("start", 0),
                    "end": s.get("end", 0),
                    "text": s.get("text", "").strip(),
                }
                for i, s in enumerate(segments)
            ],
        }

    # Default: json
    return {"text": text}


# =============================================================================
# Router and Endpoints
# =============================================================================

router = APIRouter()


@router.post("/v1/audio/transcriptions")
async def create_transcription(
    model: str = Form(...),
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(0.0),
):
    """
    Audio transcription endpoint (OpenAI-compatible).
    """
    file_content = await file.read()
    if len(file_content) > _MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio file too large (max 100 MB)")

    # Check if the requested model maps to a configured whisper-server instance first.
    # Try alias round-robin resolution before direct ID lookup.
    whisper_model_id = multi_model_manager.resolve_whisper_alias_model_id(model)
    whisper_server = (
        multi_model_manager.whisper_servers.get(whisper_model_id)
        if whisper_model_id is not None
        else multi_model_manager.whisper_servers.get(model)
    )
    if whisper_server is not None:
        multi_model_manager.request_model(requested_model=model, model_type="audio")
        if not whisper_server.is_running():
            whisper_server.start(
                getattr(whisper_server, "_model_path", None),
                gpu_device=getattr(whisper_server, "_gpu_device", 0),
            )
            if whisper_server.is_running():
                ws_key = f"audio:{whisper_model_id or model}"
                multi_model_manager.models[ws_key] = whisper_server
                multi_model_manager.active_in_vram = ws_key
                multi_model_manager.models_in_vram.add(ws_key)
        if not whisper_server.is_running():
            raise HTTPException(status_code=500, detail="whisper-server failed to start")
        result = whisper_server.transcribe(
            file_content,
            language=language,
            prompt=prompt
        )
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        return _format_response(response_format, result.get("text", ""), [])

    # Use the manager to resolve the model and manage VRAM
    model_info = multi_model_manager.request_model(
        requested_model=model,
        model_type="audio"
    )

    # Check if the model was rejected as not allowed
    if model_info.get('error'):
        raise HTTPException(status_code=404, detail=model_info['error'])

    model_name = model_info['model_name']
    model_key = model_info['model_key']
    whisper_model = model_info['model_object']

    if not model_name:
        raise HTTPException(
            status_code=400,
            detail="Audio transcription not configured. Use --audio-model or --whisper-server."
        )

    # Determine a safe file extension from the upload's content-type or filename,
    # never trusting the raw user-supplied value for arbitrary suffixes.
    raw_ext = os.path.splitext(file.filename or '')[1].lower()
    safe_ext = raw_ext if raw_ext in _SAFE_EXTENSIONS else '.wav'

    # Save to temp file (needed for some backends)
    with tempfile.NamedTemporaryFile(delete=False, suffix=safe_ext) as tmp:
        tmp.write(file_content)
        tmp_path = tmp.name
    
    try:
        # Try faster-whisper first
        try:
            from faster_whisper import WhisperModel

            if whisper_model is None:
                whisper_model = WhisperModel(
                    model_name,
                    device="cpu",
                    compute_type="int8",
                )
                multi_model_manager.add_model(model_key, whisper_model)
                multi_model_manager.current_model_key = model_key

            raw_segments, _ = whisper_model.transcribe(
                tmp_path,
                language=language,
                initial_prompt=prompt,
                temperature=temperature,
            )
            # Materialise the generator so we have all segment data
            segments = [
                {"start": s.start, "end": s.end, "text": s.text}
                for s in raw_segments
            ]
            full_text = "".join(s["text"] for s in segments)
            return _format_response(response_format, full_text.strip(), segments)

        except ImportError:
            pass

        # Try whispercpp as fallback
        try:
            import whispercpp

            if whisper_model is None:
                whisper_model = whispercpp.Whisper.from_pretrained(model_name)
                multi_model_manager.add_model(model_key, whisper_model)
                multi_model_manager.current_model_key = model_key

            result = whisper_model.transcribe(tmp_path)

            text = ""
            if hasattr(result, 'text'):
                text = result.text
            elif isinstance(result, dict):
                text = result.get('text', '')
            elif isinstance(result, list):
                for segment in result:
                    if hasattr(segment, 'text'):
                        text += segment.text
                    elif isinstance(segment, dict):
                        text += segment.get('text', '')

            # whispercpp does not expose per-segment timestamps easily
            return _format_response(response_format, text.strip(), [])

        except ImportError as e:
            raise HTTPException(
                status_code=501,
                detail="Audio transcription not available. Install faster-whisper or whispercpp."
            )
            
    except Exception as e:
        print(f"Transcription error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
