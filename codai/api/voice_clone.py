"""
Voice cloning endpoints.

POST /v1/audio/clone          — synthesize speech in a cloned voice
GET  /v1/audio/voices         — list saved voice profiles
POST /v1/audio/voices         — save a named voice profile (ref audio + transcript)
DELETE /v1/audio/voices/{name} — delete a voice profile
"""

import asyncio
import base64
import io
import json
import os
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
        host = http_request.headers.get('host', '127.0.0.1') if http_request else '127.0.0.1'
        if ':' in host:
            parts = host.split(':')
            if len(parts) == 2 and parts[1].isdigit():
                host = parts[0]
        use_https = getattr(global_args, 'https', False) if global_args else False
        proto = 'https' if use_https else 'http'
        port = getattr(global_args, 'port', 8000) if global_args else 8000
        return {"url": f"{proto}://{host}:{port}/v1/files/{filename}"}
    return {"b64_wav": base64.b64encode(audio_bytes).decode()}


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
