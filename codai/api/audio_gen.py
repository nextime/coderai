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
from typing import Optional

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


def _to_mp3(wav_bytes: bytes) -> bytes:
    """Transcode with ffmpeg, which every media path here already requires."""
    import subprocess, tempfile
    with tempfile.TemporaryDirectory() as d:
        src, dst = os.path.join(d, "in.wav"), os.path.join(d, "out.mp3")
        with open(src, 'wb') as f:
            f.write(wav_bytes)
        r = subprocess.run(["ffmpeg", "-nostdin", "-y", "-i", src,
                            "-codec:a", "libmp3lame", "-q:a", "2", dst],
                           capture_output=True)
        if r.returncode != 0 or not os.path.exists(dst):
            tail = (r.stderr or b"").decode(errors="replace").strip()[-300:]
            raise RuntimeError(f"mp3 encoding failed: {tail}")
        with open(dst, 'rb') as f:
            return f.read()


def _save_audio_response(audio_data: bytes, ext: str, http_request: Request,
                         response_format: str = None) -> dict:
    """Return the generated audio the way the CALLER asked for it.

    This used to switch on whether a file directory was configured, so
    `response_format` -- a declared field documented as url | b64_wav | b64_mp3
    -- did nothing whenever one was: a client asking for base64 silently got a
    URL instead of bytes. Same bug the image endpoint had with b64_json.
    """
    want = (response_format or "url").strip().lower()
    if want in ("b64_mp3", "mp3"):
        # Asked for mp3, so send mp3. Returning base64 of a wav under a
        # b64_mp3 key would be a decoding failure at the client, not a format
        # it can use.
        return {"b64_mp3": base64.b64encode(_to_mp3(audio_data)).decode()}
    if want.startswith("b64") or want in ("base64", "bytes"):
        return {f"b64_{ext}": base64.b64encode(audio_data).decode()}

    filename = f"{uuid.uuid4().hex}.{ext}"
    if global_file_path:
        os.makedirs(global_file_path, exist_ok=True)
        fpath = os.path.join(global_file_path, filename)
        with open(fpath, 'wb') as f:
            f.write(audio_data)
        from codai.api.urlutils import build_file_url
        return {"url": build_file_url(filename, http_request)}
    # No directory to serve from: base64 is the only thing left, and it is
    # better than a URL that would 404.
    return {f"b64_{ext}": base64.b64encode(audio_data).decode()}


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
    # Stable Audio is NOT AudioLDM. It was mapped here to 'audioldm', which
    # loads AudioLDM2Pipeline -- a different architecture with a different
    # projection model and a different sample rate -- so a configured
    # stable-audio-open entry could only ever fail at load. It has its own
    # pipeline in diffusers and its own branch below.
    if 'stable-audio' in n or 'stable_audio' in n:
        return 'stable_audio'
    if 'audioldm' in n:
        return 'audioldm'
    if 'audiogen' in n:
        return 'audiogen'
    if 'ace-step' in n or 'acestep' in n:
        return 'ace_step'
    return 'musicgen'


def _load_stable_audio(model_name: str, device: str, model_config: dict = None):
    """Stable Audio Open: text (and a negative prompt) to music or an effect.

    44.1 kHz stereo, up to about 47 seconds, which is what makes it the right
    model for a loop or a sound effect rather than a whole track.
    """
    import torch
    from diffusers import StableAudioPipeline

    from codai.models.hf_loading import resolve_dtype
    dtype = resolve_dtype(model_config, default='f16')
    pipe = StableAudioPipeline.from_pretrained(model_name, torch_dtype=dtype)
    _off = (model_config or {}).get('offload_strategy')
    if _off in ('cpu', 'sequential', 'model', 'disk') and hasattr(
            pipe, 'enable_model_cpu_offload'):
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)
    return pipe


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

    elif model_type == 'ace_step':
        # The worker returns a finished wav: generation happened in another
        # interpreter, so there is no array to resample or clip here, and the
        # pipeline exposes no per-step callback to report progress from.
        _aud_progress_reset(0, unit="s")
        # Which fields the CLIENT actually sent, not which have a value: the
        # shared request defaults are MusicGen's (10s, cfg 3.0) and would make
        # every ACE-Step call a 10-second clip at a guidance it is not tuned
        # for. A field left out falls back to ACE-Step's own default; a field
        # explicitly set to 10 still means 10.
        _sent = getattr(request, 'model_fields_set', set())
        wav = pipe.generate(
            prompt=request.prompt,
            lyrics=getattr(request, 'lyrics', None) or "",
            duration=(float(request.duration) if 'duration' in _sent
                      and request.duration else 60.0),
            steps=int(getattr(request, 'steps', None) or 60),
            guidance_scale=(float(request.cfg_coef) if 'cfg_coef' in _sent
                            and request.cfg_coef is not None else 15.0),
            seed=request.seed,
        )
        task_registry.raise_if_cancelled(task_id)
        return wav, 'wav'

    elif model_type == 'stable_audio':
        num_steps = int(getattr(request, 'steps', None) or 100)
        _aud_progress_reset(num_steps, unit="it")

        def _sa_step_cb(pipe, step_index, timestep, callback_kwargs):
            task_registry.raise_if_cancelled(task_id)
            task_registry.wait_if_paused(task_id)
            task_registry.step(task_id, step_index + 1)
            _aud_progress_step(step_index + 1)
            return callback_kwargs

        import torch
        _gen = None
        if getattr(request, 'seed', None) is not None:
            _gen = torch.Generator(pipe.device).manual_seed(int(request.seed))
        result = pipe(
            request.prompt,
            negative_prompt=getattr(request, 'negative_prompt', None),
            num_inference_steps=num_steps,
            audio_end_in_s=float(request.duration or 10.0),
            num_waveforms_per_prompt=1,
            generator=_gen,
            callback_on_step_end=_sa_step_cb,
        )
        # (channels, samples) float32 at 44.1 kHz. Kept STEREO: this model's
        # point is that it produces a usable stereo bed, and folding it to mono
        # here would throw that away.
        audio_np = result.audios[0].T.float().cpu().numpy()
        sr = int(pipe.vae.config.sampling_rate)

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
    # scipy writes (samples,) as mono and (samples, channels) as multichannel.
    # Clipped before the cast: a float sample above 1.0 wraps round to a loud
    # negative one on int16 overflow, which is heard as a click.
    audio_np = np.asarray(audio_np)
    if audio_np.ndim > 1 and audio_np.shape[0] < audio_np.shape[-1]:
        audio_np = audio_np.T          # (channels, samples) -> (samples, channels)
    audio_int16 = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)
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
async def get_audio_progress(request_id: Optional[str] = None):
    """Audio generation progress — this request's own, or the global slot.

    With ``request_id`` (the id the caller passed as ``request_id`` in the
    generation body or an ``X-Request-Id`` header) this answers for THAT
    request only, so two callers generating at once can each follow their own.
    A 404 means that id is finished and pruned, or was never used.

    Without it, the single global slot below, exactly as before.
    """
    if request_id:
        from codai.api import request_progress as _rp
        _rid = _rp.sanitize(request_id)
        _view = _rp.view(_rid) if _rid else None
        if _view is None:
            raise HTTPException(
                status_code=404,
                detail=f"no generation in progress for request_id {request_id!r}")
        return _view
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

    A caller may name this request (``request_id`` in the body, or an
    ``X-Request-Id`` header) and then poll
    ``GET /v1/audio/progress?request_id=...`` for its OWN progress.
    """
    from codai.api import request_progress as _rp
    _rp.set_current(_rp.from_request(request, http_request))
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
                elif model_type == 'stable_audio':
                    pipe = await asyncio.get_event_loop().run_in_executor(
                        None, _load_stable_audio, model_name, device, _ag_cfg)
                elif model_type == 'ace_step':
                    # ACE-Step ships its own package pinning a stack this venv
                    # cannot hold, so it runs in an isolated venv behind a
                    # local HTTP service, like CrisperWhisper and Canary. The
                    # handle cached here exposes cleanup(), so a model swap
                    # evicts it and frees its VRAM like any other model.
                    from codai.api.acestep_worker import AceStep
                    pipe = await asyncio.get_event_loop().run_in_executor(
                        None, AceStep, model_name, _ag_cfg)
                else:
                    pipe = await asyncio.get_event_loop().run_in_executor(
                        None, _load_audioldm, model_name, device, _ag_cfg)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load audio gen model: {e}")
        multi_model_manager.models[model_key] = pipe
        multi_model_manager.current_model_key = model_key

    # Claims the caller's own id when they named one, so
    # /v1/audio/progress?request_id=... follows THIS request.
    _tid = task_registry.register(
        "audio", title=(request.prompt or "")[:80], model=model_name or "",
        task_id=_rp.take())
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

    result = _save_audio_response(audio_bytes, ext, http_request,
                                  getattr(request, 'response_format', None))

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