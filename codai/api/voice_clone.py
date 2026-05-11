"""
Voice cloning endpoints.

POST   /v1/audio/clone              — synthesize speech in a cloned voice
GET    /v1/audio/voices             — list saved voice profiles
POST   /v1/audio/voices             — save a named voice profile (ref audio + transcript)
POST   /v1/audio/voices/extract     — extract voice profile from audio or video file
PATCH  /v1/audio/voices/{name}      — update description / replace reference audio
DELETE /v1/audio/voices/{name}      — delete a voice profile
"""

import asyncio
import base64
import io
import json
import os
import subprocess
import tempfile
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form
from pydantic import BaseModel, ConfigDict

router = APIRouter()

global_args = None
global_file_path = None

# Directory where voice profiles are stored
_VOICES_DIR: Optional[str] = None


def set_global_args(args):
    global global_args, _VOICES_DIR
    global_args = args
    # Store voice profiles alongside output files, or in a default location
    base = getattr(args, 'file_path', None) or os.path.expanduser('~/.coderai/voices')
    _VOICES_DIR = os.path.join(base if os.path.isdir(base) else os.path.dirname(base) if base else os.path.expanduser('~/.coderai'), 'voices')
    os.makedirs(_VOICES_DIR, exist_ok=True)


def set_global_file_path(path):
    global global_file_path
    global_file_path = path


def _voices_dir() -> str:
    if _VOICES_DIR:
        return _VOICES_DIR
    d = os.path.expanduser('~/.coderai/voices')
    os.makedirs(d, exist_ok=True)
    return d


def _voice_path(name: str) -> str:
    return os.path.join(_voices_dir(), name)


def _list_voices() -> list:
    d = _voices_dir()
    voices = []
    for entry in os.scandir(d):
        if entry.is_dir():
            meta_path = os.path.join(entry.path, 'meta.json')
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                voices.append(meta)
    return sorted(voices, key=lambda v: v.get('created_at', 0))


def _save_voice(name: str, audio_bytes: bytes, audio_ext: str, transcript: str, description: str = '') -> dict:
    vdir = _voice_path(name)
    os.makedirs(vdir, exist_ok=True)
    audio_file = os.path.join(vdir, f'ref{audio_ext}')
    with open(audio_file, 'wb') as f:
        f.write(audio_bytes)
    meta = {
        'name': name,
        'description': description,
        'transcript': transcript,
        'audio_file': audio_file,
        'audio_ext': audio_ext,
        'created_at': int(time.time()),
    }
    with open(os.path.join(vdir, 'meta.json'), 'w') as f:
        json.dump(meta, f)
    return meta


def _load_voice(name: str) -> Optional[dict]:
    meta_path = os.path.join(_voice_path(name), 'meta.json')
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        return json.load(f)


def _decode_audio(data: str) -> tuple[bytes, str]:
    """Decode base64 audio data, return (bytes, ext)."""
    if data.startswith('data:'):
        mime, b64 = data.split(',', 1)
        ext = '.' + mime.split('/')[1].split(';')[0]
        return base64.b64decode(b64), ext
    return base64.b64decode(data), '.wav'


def _decode_b64_or_url(data: str) -> bytes:
    if data.startswith('data:'):
        _, b64 = data.split(',', 1)
        return base64.b64decode(b64)
    if data.startswith('http://') or data.startswith('https://'):
        import urllib.request
        with urllib.request.urlopen(data, timeout=60) as r:
            return r.read()
    return base64.b64decode(data)


def _f5tts_clone(ref_audio_path: str, ref_text: str, gen_text: str,
                  speed: float = 1.0, seed: Optional[int] = None) -> bytes:
    """Run F5-TTS voice cloning, return WAV bytes."""
    from f5_tts.api import F5TTS
    import soundfile as sf
    import numpy as np

    device = None
    if global_args:
        import torch
        if torch.cuda.is_available():
            device = 'cuda'

    tts = F5TTS(device=device)
    wav, sr, _ = tts.infer(
        ref_file=ref_audio_path,
        ref_text=ref_text,
        gen_text=gen_text,
        speed=speed,
        seed=seed,
        show_info=lambda x: None,
        progress=lambda x, **kw: x,
    )

    buf = io.BytesIO()
    sf.write(buf, wav, sr, format='WAV')
    return buf.getvalue()


def _save_audio_response(audio_bytes: bytes, http_request: Request) -> dict:
    import uuid
    filename = f"{uuid.uuid4().hex}.wav"
    if global_file_path:
        os.makedirs(global_file_path, exist_ok=True)
        fpath = os.path.join(global_file_path, filename)
        with open(fpath, 'wb') as f:
            f.write(audio_bytes)
        from codai.api.urlutils import build_file_url
        return {"url": build_file_url(filename, http_request)}
    return {"b64_wav": base64.b64encode(audio_bytes).decode()}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class VoiceExtractRequest(BaseModel):
    name: str
    description: str = ''
    audio: Optional[str] = None      # base64/URL audio file
    video: Optional[str] = None      # base64/URL video (audio track extracted)
    transcript: Optional[str] = ''   # optional; auto-transcribed if omitted
    model_config = ConfigDict(extra="allow")


class VoicePatchRequest(BaseModel):
    description: Optional[str] = None
    transcript: Optional[str] = None
    audio: Optional[str] = None      # replace reference audio (base64)
    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------------
# Voice profile management
# ---------------------------------------------------------------------------

@router.get("/v1/audio/voices")
async def list_voices():
    """List all saved voice profiles."""
    return {"voices": _list_voices()}


@router.post("/v1/audio/voices")
async def create_voice(
    name: str = Form(...),
    transcript: str = Form(...),
    description: str = Form(''),
    audio: UploadFile = File(...),
):
    """Save a named voice profile from a reference audio file + transcript."""
    if not name.replace('-', '').replace('_', '').isalnum():
        raise HTTPException(status_code=400, detail="Voice name must be alphanumeric (hyphens/underscores allowed)")

    audio_bytes = await audio.read()
    ext = os.path.splitext(audio.filename)[1] or '.wav'

    # Validate audio is readable
    try:
        import soundfile as sf, io as _io
        sf.info(_io.BytesIO(audio_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio file: {e}")

    meta = _save_voice(name, audio_bytes, ext, transcript, description)
    return {"created": True, "voice": meta}


@router.delete("/v1/audio/voices/{name}")
async def delete_voice(name: str):
    """Delete a saved voice profile."""
    import shutil
    vdir = _voice_path(name)
    if not os.path.exists(vdir):
        raise HTTPException(status_code=404, detail=f"Voice '{name}' not found")
    shutil.rmtree(vdir)
    return {"deleted": True, "name": name}


@router.patch("/v1/audio/voices/{name}")
async def patch_voice(name: str, req: VoicePatchRequest):
    """Update description, transcript, or reference audio of a saved voice profile."""
    meta = _load_voice(name)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Voice '{name}' not found")
    vdir = _voice_path(name)

    if req.description is not None:
        meta['description'] = req.description
    if req.transcript is not None:
        meta['transcript'] = req.transcript
    if req.audio:
        audio_bytes, ext = _decode_audio(req.audio)
        audio_file = os.path.join(vdir, f'ref{ext}')
        # Remove old audio file(s)
        for f in os.listdir(vdir):
            if f.startswith('ref.') or f.startswith('ref'):
                try:
                    os.unlink(os.path.join(vdir, f))
                except Exception:
                    pass
        with open(audio_file, 'wb') as f:
            f.write(audio_bytes)
        meta['audio_file'] = audio_file
        meta['audio_ext'] = ext

    with open(os.path.join(vdir, 'meta.json'), 'w') as f:
        json.dump(meta, f)
    return {"updated": True, "voice": meta}


@router.get("/v1/audio/voices/{name}")
async def get_voice(name: str):
    """Get a single voice profile metadata."""
    meta = _load_voice(name)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Voice '{name}' not found")
    return {"voice": meta}


@router.post("/v1/audio/voices/extract")
async def extract_voice(req: VoiceExtractRequest):
    """
    Extract a voice profile from a source audio or video file.

    - audio: base64/URL audio file (wav, mp3, flac, …)
    - video: base64/URL video file — audio track is extracted automatically
    - transcript: optional; auto-transcribed with Whisper when omitted
    """
    if not req.name.replace('-', '').replace('_', '').isalnum():
        raise HTTPException(status_code=400, detail="Voice name must be alphanumeric (hyphens/underscores allowed)")
    if not req.audio and not req.video:
        raise HTTPException(status_code=400, detail="Provide audio or video")

    temps = []
    try:
        audio_bytes: Optional[bytes] = None
        audio_ext = '.wav'

        if req.video:
            # Extract audio track from video with ffmpeg
            raw = _decode_b64_or_url(req.video)
            in_tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            in_tmp.write(raw)
            in_tmp.close()
            temps.append(in_tmp.name)
            out_tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            out_tmp.close()
            temps.append(out_tmp.name)
            r = subprocess.run(
                ['ffmpeg', '-y', '-i', in_tmp.name, '-vn', '-acodec', 'pcm_s16le',
                 '-ar', '22050', '-ac', '1', out_tmp.name],
                capture_output=True, timeout=120,
            )
            if r.returncode != 0:
                raise HTTPException(status_code=422,
                    detail=f"Audio extraction from video failed: {r.stderr.decode(errors='replace')[:200]}")
            with open(out_tmp.name, 'rb') as f:
                audio_bytes = f.read()
        else:
            audio_bytes, audio_ext = _decode_audio(req.audio)

        # Auto-transcribe if transcript not provided
        transcript = req.transcript or ''
        if not transcript:
            try:
                import whisper
                in_tmp2 = tempfile.NamedTemporaryFile(suffix=audio_ext, delete=False)
                in_tmp2.write(audio_bytes)
                in_tmp2.close()
                temps.append(in_tmp2.name)
                model = whisper.load_model('base')
                result = model.transcribe(in_tmp2.name)
                transcript = result.get('text', '').strip()
            except Exception:
                pass

        # Validate audio
        try:
            import soundfile as sf
            sf.info(io.BytesIO(audio_bytes))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid audio: {e}")

        meta = _save_voice(req.name, audio_bytes, audio_ext, transcript, req.description)
        return {"created": True, "voice": meta}

    finally:
        for t in temps:
            try:
                os.unlink(t)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Voice cloning TTS
# ---------------------------------------------------------------------------

class VoiceCloneRequest(BaseModel):
    text: str                               # text to synthesize
    voice_name: Optional[str] = None        # use a saved voice profile
    ref_audio: Optional[str] = None         # base64 reference audio (if not using saved voice)
    ref_text: Optional[str] = None          # transcript of ref_audio
    speed: Optional[float] = 1.0
    seed: Optional[int] = None
    response_format: Optional[str] = "url"
    model_config = ConfigDict(extra="allow")


@router.post("/v1/audio/clone")
async def clone_voice(request: VoiceCloneRequest, http_request: Request = None):
    """
    Synthesize speech in a cloned voice using F5-TTS.

    Provide either:
    - voice_name: name of a saved voice profile
    - ref_audio (base64) + ref_text: inline reference audio
    """
    # Resolve reference audio
    ref_audio_path = None
    ref_text = request.ref_text or ''
    temps = []

    try:
        if request.voice_name:
            meta = _load_voice(request.voice_name)
            if not meta:
                raise HTTPException(status_code=404, detail=f"Voice '{request.voice_name}' not found")
            ref_audio_path = meta['audio_file']
            ref_text = ref_text or meta.get('transcript', '')
        elif request.ref_audio:
            audio_bytes, ext = _decode_audio(request.ref_audio)
            tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
            tmp.write(audio_bytes)
            tmp.close()
            ref_audio_path = tmp.name
            temps.append(ref_audio_path)
        else:
            raise HTTPException(status_code=400, detail="Provide voice_name or ref_audio")

        if not ref_text:
            raise HTTPException(status_code=400, detail="ref_text (transcript of reference audio) is required for voice cloning")

        try:
            audio_bytes = await asyncio.get_event_loop().run_in_executor(
                None, _f5tts_clone,
                ref_audio_path, ref_text, request.text,
                request.speed or 1.0, request.seed,
            )
        except ImportError:
            raise HTTPException(status_code=501, detail="f5-tts not installed. Run: pip install f5-tts")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Voice cloning failed: {e}")

        result = _save_audio_response(audio_bytes, http_request)
        return {"created": int(time.time()), "data": [result]}

    finally:
        for t in temps:
            try:
                os.unlink(t)
            except Exception:
                pass
