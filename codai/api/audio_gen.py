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
Audio generation endpoints for the codai API.
Supports music, sound effects, and ambient audio via MusicGen, AudioLDM2, StableAudio, etc.
POST /v1/audio/generate
"""

import asyncio
import base64
import io
import os
import time
import uuid

from fastapi import APIRouter, HTTPException, Request

from codai.models.manager import multi_model_manager
from codai.pydantic.audiogenrequest import AudioGenerationRequest, AudioGenerationResponse

router = APIRouter()

global_args = None
global_file_path = None

# =============================================================================
# Audio generation progress tracking
# =============================================================================
_aud_progress: dict = {
    "current": 0, "total": 0, "active": False,
    "started_at": 0.0, "it_per_s": 0.0, "unit": "it",
    "phase": "idle", "model": "",
}

def _aud_progress_loading(model_name: str = ""):
    _aud_progress["phase"] = "loading"
    _aud_progress["active"] = True
    _aud_progress["current"] = 0
    _aud_progress["total"] = 0
    _aud_progress["it_per_s"] = 0.0
    _aud_progress["started_at"] = time.monotonic()
    _aud_progress["model"] = model_name or ""

def _aud_progress_reset(total: int, unit: str = "it"):
    _aud_progress["current"] = 0
    _aud_progress["total"] = total
    _aud_progress["active"] = True
    _aud_progress["phase"] = "generating"
    _aud_progress["started_at"] = time.monotonic()
    _aud_progress["it_per_s"] = 0.0
    _aud_progress["unit"] = unit

def _aud_progress_done():
    _aud_progress["current"] = max(_aud_progress["current"], _aud_progress["total"])
    _aud_progress["active"] = False
    _aud_progress["phase"] = "idle"

def _aud_progress_step(step: int):
    _aud_progress["current"] = step
    elapsed = time.monotonic() - _aud_progress["started_at"]
    if elapsed > 0 and step > 0:
        _aud_progress["it_per_s"] = round(step / elapsed, 2)


def set_global_args(args):
    global global_args
    global_args = args


def set_global_file_path(path):
    global global_file_path
    global_file_path = path


def _derive_device() -> str:
    if global_args:
        d = getattr(global_args, 'vulkan_device', None)
        if d is not None:
            return f"cuda:{d}"
    return "cuda:0"


def _save_audio_response(audio_data: bytes, ext: str, http_request: Request) -> dict:
    filename = f"{uuid.uuid4().hex}.{ext}"
    if global_file_path:
        os.makedirs(global_file_path, exist_ok=True)
        fpath = os.path.join(global_file_path, filename)
        with open(fpath, 'wb') as f:
            f.write(audio_data)
        from codai.api.urlutils import build_file_url
        return {"url": build_file_url(filename, http_request)}
    else:
        b64 = base64.b64encode(audio_data).decode()
        return {f"b64_{ext}": b64}


def _load_musicgen(model_name: str, device: str):
    from audiocraft.models import MusicGen, AudioGen
    name_lower = model_name.lower()
    if 'audiogen' in name_lower:
        model = AudioGen.get_pretrained(model_name)
    else:
        model = MusicGen.get_pretrained(model_name)
    model.set_generation_params(duration=30)
    return model


def _load_audioldm(model_name: str, device: str):
    import torch
    from diffusers import AudioLDM2Pipeline
    pipe = AudioLDM2Pipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    return pipe


def _detect_audio_gen_type(model_name: str) -> str:
    n = model_name.lower()
    if 'audioldm' in n or 'stable-audio' in n:
        return 'audioldm'
    if 'audiogen' in n:
        return 'audiogen'
    return 'musicgen'


def _generate_audio(pipe, model_name: str, request: AudioGenerationRequest):
    """Run generation and return (audio_bytes, ext)."""
    import numpy as np, io as _io

    model_type = _detect_audio_gen_type(model_name)

    if model_type in ('musicgen', 'audiogen'):
        pipe.set_generation_params(
            duration=request.duration,
            top_k=request.top_k,
            top_p=request.top_p,
            temperature=request.temperature,
            cfg_coef=request.cfg_coef,
        )
        # MusicGen/AudioGen generate in one shot — track elapsed only
        _aud_progress_reset(0, unit="s")
        if request.melody and model_type == 'musicgen':
            import torchaudio, torch
            raw = _decode_b64_or_url(request.melody)
            melody_wav, sr = torchaudio.load(_io.BytesIO(raw))
            wav = pipe.generate_with_chroma([request.prompt], melody_wav.unsqueeze(0), sr)
        else:
            wav = pipe.generate([request.prompt])
        audio_np = wav[0, 0].cpu().numpy()
        sr = pipe.sample_rate

    elif model_type == 'audioldm':
        num_steps = 50
        _aud_progress_reset(num_steps, unit="it")

        def _aud_step_cb(pipe, step_index, timestep, callback_kwargs):
            _aud_progress_step(step_index + 1)
            return callback_kwargs

        result = pipe(
            request.prompt,
            num_inference_steps=num_steps,
            audio_length_in_s=request.duration,
            callback_on_step_end=_aud_step_cb,
        )
        audio_np = result.audios[0]
        sr = 16000

    # Write to wav
    import scipy.io.wavfile as wavfile
    buf = _io.BytesIO()
    audio_int16 = (audio_np * 32767).astype(np.int16)
    wavfile.write(buf, sr, audio_int16)
    return buf.getvalue(), 'wav'


def _decode_b64_or_url(data: str) -> bytes:
    if data.startswith("data:"):
        _, enc = data.split(",", 1)
        return base64.b64decode(enc)
    if data.startswith("http"):
        import urllib.request
        with urllib.request.urlopen(data, timeout=30) as r:
            return r.read()
    return base64.b64decode(data)


@router.get("/v1/audio/progress")
async def get_audio_progress():
    """Return current audio generation progress including speed."""
    elapsed = time.monotonic() - _aud_progress["started_at"] if _aud_progress["active"] else 0.0
    total = _aud_progress["total"]
    current = _aud_progress["current"]
    return {
        "current":  current,
        "total":    total,
        "active":   _aud_progress["active"],
        "phase":    _aud_progress.get("phase", "idle"),
        "model":    _aud_progress.get("model", ""),
        "pct":      int(current / total * 100) if total > 0 else 0,
        "it_per_s": _aud_progress["it_per_s"],
        "elapsed":  round(elapsed, 1),
        "unit":     _aud_progress["unit"],
    }


@router.post("/v1/audio/generate", response_model=AudioGenerationResponse)
async def audio_generate(request: AudioGenerationRequest, http_request: Request = None):
    """
    Generate music, sound effects, or ambient audio.
    Compatible models: MusicGen, AudioGen, AudioLDM2, StableAudio.
    """
    _aud_progress_loading(request.model or "audio")
    model_info = multi_model_manager.request_model(request.model, model_type="audio_gen")
    model_name = model_info.get('model_name')
    if not model_name:
        err = model_info.get('error', f"Model '{request.model}' not found")
        raise HTTPException(status_code=404, detail=err)

    model_key = model_info['model_key']
    pipe = model_info.get('model_object')

    if pipe is None:
        device = _derive_device()
        model_type = _detect_audio_gen_type(model_name)
        try:
            if model_type in ('musicgen', 'audiogen'):
                pipe = await asyncio.get_event_loop().run_in_executor(
                    None, _load_musicgen, model_name, device)
            else:
                pipe = await asyncio.get_event_loop().run_in_executor(
                    None, _load_audioldm, model_name, device)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load audio gen model: {e}")
        multi_model_manager.models[model_key] = pipe
        multi_model_manager.current_model_key = model_key

    try:
        audio_bytes, ext = await asyncio.get_event_loop().run_in_executor(
            None, _generate_audio, pipe, model_name, request)
    except Exception as e:
        _aud_progress_done()
        raise HTTPException(status_code=500, detail=f"Audio generation failed: {e}")
    finally:
        _aud_progress_done()

    result = _save_audio_response(audio_bytes, ext, http_request)

    try:
        from codai.api.archive import archive_manager
        asyncio.get_event_loop().create_task(asyncio.to_thread(
            archive_manager.save_generation,
            "audio", "/v1/audio/generate",
            model_name,
            request.prompt,
            {
                "duration": request.duration,
                "top_k": request.top_k,
                "top_p": request.top_p,
                "temperature": request.temperature,
                "cfg_coef": request.cfg_coef,
            },
            [(audio_bytes, ext)],
        ))
    except Exception:
        pass

    return AudioGenerationResponse(created=int(time.time()), data=[result])