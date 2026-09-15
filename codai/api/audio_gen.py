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
from codai.tasks import task_registry, TaskCancelled

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


class _TransformersMusicGen:
    """MusicGen through transformers, wearing audiocraft's interface.

    audiocraft is Meta's library and is optional on purpose: it pins an old
    torch, so installing it into a pod image would drag torch backwards and
    undo the CUDA build pin. transformers ships MusicGen itself, so a pod can
    serve the model without it — a rented pod failed with "No module named
    'audiocraft'" and there was no reason for it to need one.

    Only what the generation path calls is implemented: set_generation_params,
    generate, sample_rate. Melody conditioning is not — that is audiocraft's
    generate_with_chroma, and a caller asking for it gets audiocraft's own
    ImportError rather than a silently different result.
    """

    def __init__(self, model_name: str, device: str):
        from transformers import AutoProcessor, MusicgenForConditionalGeneration
        self._processor = AutoProcessor.from_pretrained(model_name)
        self._model = MusicgenForConditionalGeneration.from_pretrained(model_name)
        self._device = device
        self._model.to(device)
        self._params = {"duration": 30.0}

    def set_generation_params(self, **kwargs):
        self._params.update({k: v for k, v in kwargs.items() if v is not None})

    @property
    def sample_rate(self) -> int:
        return int(self._model.config.audio_encoder.sampling_rate)

    def generate(self, prompts):
        import torch
        inputs = self._processor(text=list(prompts), padding=True,
                                 return_tensors="pt").to(self._device)
        # transformers counts tokens, audiocraft counts seconds. 50 Hz is
        # MusicGen's frame rate, which is what makes `duration` mean anything.
        frame_rate = getattr(self._model.config.audio_encoder, "frame_rate", 50)
        max_new = max(1, int(float(self._params.get("duration", 30.0)) * frame_rate))
        gen = {"max_new_tokens": max_new, "do_sample": True}
        if self._params.get("temperature"):
            gen["temperature"] = float(self._params["temperature"])
        if self._params.get("top_k"):
            gen["top_k"] = int(self._params["top_k"])
        if self._params.get("top_p"):
            gen["top_p"] = float(self._params["top_p"])
        if self._params.get("cfg_coef"):
            gen["guidance_scale"] = float(self._params["cfg_coef"])
        with torch.no_grad():
            return self._model.generate(**inputs, **gen)   # (batch, channels, samples)


def _audio_backend(model_config: dict = None) -> str:
    """Which MusicGen implementation to use: 'audiocraft', 'transformers', 'auto'.

    Per model first, then the environment (which is how a pod is told), then
    'auto' — audiocraft when it is importable, transformers otherwise.

    The choice is worth exposing because the two are not equivalent: audiocraft
    has melody conditioning and AudioGen, transformers has neither but needs
    nothing beyond what every image already carries.
    """
    import os as _os
    chosen = str((model_config or {}).get("audio_backend") or "").strip().lower()
    if not chosen:
        chosen = str(_os.environ.get("CODERAI_AUDIO_BACKEND", "")).strip().lower()
    return chosen if chosen in ("audiocraft", "transformers") else "auto"


def _load_musicgen(model_name: str, device: str, model_config: dict = None):
    backend = _audio_backend(model_config)
    if backend == "transformers":
        print(f"[audio] {model_name}: transformers backend requested "
              "(no melody conditioning)", flush=True)
        return _TransformersMusicGen(model_name, device)
    try:
        from audiocraft.models import MusicGen, AudioGen
    except ImportError:
        if backend == "audiocraft":
            # Asked for explicitly and not here: say so rather than quietly
            # serving something with different capabilities.
            raise RuntimeError(
                "audio_backend is set to 'audiocraft' but audiocraft is not "
                "installed. Install it, or set audio_backend to 'transformers' "
                "(which cannot do melody conditioning).")
        # Next best: audiocraft in its own venv. It pins torch==2.1.0, which has
        # no build for the CUDA this server runs, so it cannot share this venv —
        # but it can live next door and be driven as a subprocess, which is what
        # keeps melody conditioning available at all.
        try:
            from codai.api import audiocraft_worker
            if audiocraft_worker.available():
                print(f"[audio] using audiocraft from {audiocraft_worker._VENV}",
                      flush=True)
                return audiocraft_worker.AudiocraftModel(model_name)
        except Exception as exc:
            print(f"[audio] isolated audiocraft venv unusable ({exc})", flush=True)
        print(f"[audio] audiocraft not installed — serving {model_name} through "
              "transformers instead (no melody conditioning)", flush=True)
        return _TransformersMusicGen(model_name, device)
    name_lower = model_name.lower()
    if 'audiogen' in name_lower:
        model = AudioGen.get_pretrained(model_name)
    else:
        model = MusicGen.get_pretrained(model_name)
    model.set_generation_params(duration=30)
    return model


def _load_audioldm(model_name: str, device: str, model_config: dict = None):
    import torch
    from diffusers import AudioLDM2Pipeline
    from codai.models.hf_loading import resolve_dtype
    dtype = resolve_dtype(model_config, default='f16')
    _xtra = {}
    # Apply 4-bit/8-bit quantization to the diffusion backbone when configured.
    _mc = model_config or {}
    if _mc.get('load_in_4bit') or _mc.get('load_in_8bit'):
        _bits = 4 if _mc.get('load_in_4bit') else 8
        try:
            from diffusers.quantizers import PipelineQuantizationConfig
            _qk = ({'load_in_4bit': True, 'bnb_4bit_compute_dtype': dtype}
                   if _mc.get('load_in_4bit') else {'load_in_8bit': True})
            _xtra['quantization_config'] = PipelineQuantizationConfig(
                quant_backend=f"bitsandbytes_{_bits}bit",
                quant_kwargs=_qk,
                components_to_quantize=["transformer", "unet"],
            )
            print(f"AudioLDM quantization: {_bits}-bit (bitsandbytes)")
        except Exception as e:
            print(f"AudioLDM quantization unavailable: {e}")
    pipe = AudioLDM2Pipeline.from_pretrained(model_name, torch_dtype=dtype, **_xtra)
    # CPU offload when configured; otherwise place on device (skip for quantized).
    _off = _mc.get('offload_strategy')
    if _off in ('cpu', 'sequential', 'model', 'disk') and hasattr(pipe, 'enable_model_cpu_offload'):
        pipe.enable_model_cpu_offload()
    elif 'quantization_config' not in _xtra:
        pipe = pipe.to(device)
    return pipe


def _detect_audio_gen_type(model_name: str) -> str:
    n = model_name.lower()
    if 'audioldm' in n or 'stable-audio' in n:
        return 'audioldm'
    if 'audiogen' in n:
        return 'audiogen'
    return 'musicgen'


def _generate_audio(pipe, model_name: str, request: AudioGenerationRequest, task_id=None):
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
            task_registry.raise_if_cancelled(task_id)
            task_registry.wait_if_paused(task_id)
            task_registry.step(task_id, step_index + 1)
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


@router.get("/v1/audio/progress", summary="Audio generation progress")
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


@router.post("/v1/audio/generate", response_model=AudioGenerationResponse, summary="Generate audio, music or SFX")
async def audio_generate(request: AudioGenerationRequest, http_request: Request = None):
    """
    Generate music, sound effects, or ambient audio.
    Compatible models: MusicGen, AudioGen, AudioLDM2, StableAudio.
    """
    _aud_progress_loading(request.model or "audio")
    model_info = await asyncio.to_thread(
        multi_model_manager.request_model, request.model, model_type="audio_gen")
    model_name = model_info.get('model_name')
    if not model_name:
        err = model_info.get('error', f"Model '{request.model}' not found")
        raise HTTPException(status_code=404, detail=err)

    model_key = model_info['model_key']
    pipe = model_info.get('model_object')

    if pipe is None:
        device = _derive_device()
        model_type = _detect_audio_gen_type(model_name)
        _ag_cfg = model_info.get('config') or {}
        from codai.tasks import loading_task
        try:
            with loading_task(model_name, model_type="audio"):
                if model_type in ('musicgen', 'audiogen'):
                    pipe = await asyncio.get_event_loop().run_in_executor(
                        None, _load_musicgen, model_name, device, _ag_cfg)
                else:
                    pipe = await asyncio.get_event_loop().run_in_executor(
                        None, _load_audioldm, model_name, device, _ag_cfg)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load audio gen model: {e}")
        multi_model_manager.models[model_key] = pipe
        multi_model_manager.current_model_key = model_key

    _tid = task_registry.register(
        "audio", title=(request.prompt or "")[:80], model=model_name or "")
    task_registry.start(_tid)
    try:
        audio_bytes, ext = await asyncio.get_event_loop().run_in_executor(
            None, _generate_audio, pipe, model_name, request, _tid)
    except TaskCancelled:
        _aud_progress_done()
        raise  # global handler finishes the task (cancelled) + returns HTTP 499
    except Exception as e:
        task_registry.finish(_tid, "error", str(e)[:200])
        _aud_progress_done()
        raise HTTPException(status_code=500, detail=f"Audio generation failed: {e}")
    finally:
        _aud_progress_done()
    task_registry.finish(_tid, "done")

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