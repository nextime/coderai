"""
Voice conversion endpoint — converts timbre while preserving pitch, melody and expression.
Unlike TTS-based dubbing, this works correctly for singing and music.

POST /v1/audio/convert   — convert voice timbre in audio (speech or singing)
"""

import asyncio
import base64
import io
import os
import tempfile
import time
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict

router = APIRouter()

global_args = None
global_file_path = None

_wrapper = None   # SeedVCWrapper singleton


def set_global_args(args):
    global global_args
    global_args = args


def set_global_file_path(path):
    global global_file_path
    global_file_path = path


def _get_wrapper():
    global _wrapper
    if _wrapper is None:
        from seed_vc.seed_vc_wrapper import SeedVCWrapper
        _wrapper = SeedVCWrapper()
    return _wrapper


def _decode_audio_to_file(data: str, suffix: str = '.wav') -> str:
    if data.startswith('data:'):
        _, b64 = data.split(',', 1)
        raw = base64.b64decode(b64)
    else:
        raw = base64.b64decode(data)
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(raw)
    tmp.close()
    return tmp.name


def _save_response(audio_np: np.ndarray, sr: int, http_request) -> dict:
    buf = io.BytesIO()
    sf.write(buf, audio_np, sr, format='WAV')
    wav_bytes = buf.getvalue()
    import uuid
    filename = f'{uuid.uuid4().hex}_converted.wav'
    if global_file_path:
        os.makedirs(global_file_path, exist_ok=True)
        fpath = os.path.join(global_file_path, filename)
        with open(fpath, 'wb') as f:
            f.write(wav_bytes)
        host = http_request.headers.get('host', '127.0.0.1') if http_request else '127.0.0.1'
        if ':' in host:
            parts = host.split(':')
            if len(parts) == 2 and parts[1].isdigit():
                host = parts[0]
        proto = 'https' if getattr(global_args, 'https', False) else 'http'
        port = getattr(global_args, 'port', 8000) if global_args else 8000
        return {'url': f'{proto}://{host}:{port}/v1/files/{filename}'}
    return {'b64_wav': base64.b64encode(wav_bytes).decode()}


class VoiceConvertRequest(BaseModel):
    """
    Convert the timbre of source_audio to match target_voice,
    while preserving pitch, melody, rhythm and expression.

    Use f0_condition=True for singing/music (slower but pitch-accurate).
    Use f0_condition=False for speech (faster).
    """
    source_audio: str                       # base64 audio to convert (the performance)
    target_voice: Optional[str] = None      # base64 reference audio for target timbre
    voice_name: Optional[str] = None        # saved voice profile name

    f0_condition: Optional[bool] = False    # True = singing/music mode (preserves pitch)
    pitch_shift: Optional[int] = 0         # semitones to shift after conversion
    diffusion_steps: Optional[int] = 10    # quality vs speed (10–30)
    length_adjust: Optional[float] = 1.0
    inference_cfg_rate: Optional[float] = 0.7

    response_format: Optional[str] = 'url'
    model_config = ConfigDict(extra='allow')


@router.post('/v1/audio/convert')
async def convert_voice(request: VoiceConvertRequest, http_request: Request = None):
    """
    Voice conversion: preserves pitch/melody/expression, changes only timbre.
    Set f0_condition=True for singing and music.
    """
    target_path = None
    temps = []
    try:
        if request.voice_name:
            from codai.api.voice_clone import _load_voice
            meta = _load_voice(request.voice_name)
            if not meta:
                raise HTTPException(status_code=404, detail=f"Voice '{request.voice_name}' not found")
            target_path = meta['audio_file']
        elif request.target_voice:
            target_path = _decode_audio_to_file(request.target_voice)
            temps.append(target_path)
        else:
            raise HTTPException(status_code=400, detail='Provide voice_name or target_voice')

        source_path = _decode_audio_to_file(request.source_audio)
        temps.append(source_path)

        try:
            wrapper = _get_wrapper()
        except ImportError:
            raise HTTPException(status_code=501,
                detail='seed-vc not installed. Run: pip install seed-vc')

        def _run():
            return wrapper.convert_voice(
                source=source_path,
                target=target_path,
                diffusion_steps=request.diffusion_steps or 10,
                length_adjust=request.length_adjust or 1.0,
                inference_cfg_rate=request.inference_cfg_rate or 0.7,
                f0_condition=bool(request.f0_condition),
                pitch_shift=request.pitch_shift or 0,
                stream_output=False,
            )

        try:
            audio_out = await asyncio.get_event_loop().run_in_executor(None, _run)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'Voice conversion failed: {e}')

        sr = 44100 if request.f0_condition else 22050
        if isinstance(audio_out, tuple):
            audio_out = audio_out[0]

        result = _save_response(np.array(audio_out).flatten(), sr, http_request)
        return {'created': int(time.time()), 'data': [result]}

    finally:
        for t in temps:
            try:
                os.unlink(t)
            except Exception:
                pass
