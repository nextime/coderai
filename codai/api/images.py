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
Image generation endpoints for the codai API.
"""

import asyncio
import base64
import io
import logging
import os
import time
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Request

_log = logging.getLogger(__name__)
from PIL import Image
from pydantic import BaseModel, ConfigDict

# Import from codai modules
from codai.models.manager import multi_model_manager
from codai.pydantic.imagerequest import ImageGenerationRequest
from codai.api.state import get_load_mode
from codai.tasks import task_registry, TaskCancelled


# =============================================================================
# Prompt embedding cache (diffusers)
#
# Caches text-encoder outputs keyed by (prompt, negative_prompt, model_name).
# When the same prompt is requested again the encode step is skipped and the
# cached tensors are passed directly to the pipeline, saving CLIP/T5 compute.
# sd.cpp handles encoding internally — no equivalent caching is possible there.
# =============================================================================

import hashlib as _hashlib
import threading as _threading

# Serializes all diffusers from_pretrained() calls.
# huggingface_hub acquires per-repo .lock files during from_pretrained; running
# two from_pretrained calls concurrently (or one alongside snapshot_download on
# the same repo) causes a filelock deadlock that hangs the process indefinitely.
# A single threading.Lock here ensures only one pipeline loads at a time.
_DIFFUSERS_LOAD_LOCK = _threading.Lock()


class _PromptEmbedCache:
    """Single-entry LRU cache for diffusers prompt embeddings."""

    _MAX_ENTRIES = 32
    _TTL = 600.0  # 10 minutes

    def __init__(self):
        self._store: dict = {}   # key -> (embeds_dict, timestamp)
        self._lock = _threading.Lock()

    @staticmethod
    def _key(prompt: str, negative_prompt: str, model_name: str) -> str:
        raw = f"{model_name}\x00{prompt}\x00{negative_prompt or ''}"
        return _hashlib.sha256(raw.encode()).hexdigest()[:24]

    def get(self, prompt: str, negative_prompt: str, model_name: str) -> Optional[dict]:
        k = self._key(prompt, negative_prompt, model_name)
        with self._lock:
            entry = self._store.get(k)
            if entry is None:
                return None
            embeds, ts = entry
            if time.time() - ts > self._TTL:
                del self._store[k]
                return None
            return embeds

    def put(self, prompt: str, negative_prompt: str, model_name: str,
            embeds: dict) -> None:
        k = self._key(prompt, negative_prompt, model_name)
        with self._lock:
            self._store[k] = (embeds, time.time())
            # Evict oldest if over limit
            if len(self._store) > self._MAX_ENTRIES:
                oldest = min(self._store, key=lambda x: self._store[x][1])
                del self._store[oldest]

    def invalidate_model(self, model_name: str) -> None:
        """Drop all entries for a model (e.g. on pipeline unload)."""
        suffix = _hashlib.sha256(model_name.encode()).hexdigest()[:8]
        with self._lock:
            drop = [k for k in self._store
                    if self._key("", "", model_name)[:8] == k[:8] or True
                    # safest: just rebuild key and compare
                    ]
            # Rebuild properly: iterate and check by re-computing key prefix
            # (can't reconstruct original prompts, so use model name hash marker)
            self._store = {
                k: v for k, v in self._store.items()
                if not k.startswith(_hashlib.sha256(model_name.encode()).hexdigest()[:4])
            }


_embed_cache = _PromptEmbedCache()


# Global reference to be set by coderai
global_args = None
global_file_path = None

# Model semaphores for concurrency control (provided by coderai)
model_semaphores = {}
queue_flags = {}

# =============================================================================
# Generation progress tracking
# =============================================================================
import time as _time

_gen_progress: dict = {
    "current": 0, "total": 0, "active": False,
    "started_at": 0.0, "it_per_s": 0.0,
    "phase": "idle", "model": "",
}

def _progress_loading(model_name: str = ""):
    _gen_progress["phase"] = "loading"
    _gen_progress["active"] = True
    _gen_progress["current"] = 0
    _gen_progress["total"] = 0
    _gen_progress["it_per_s"] = 0.0
    _gen_progress["started_at"] = _time.monotonic()
    _gen_progress["model"] = model_name or ""

def _progress_reset(total: int):
    _gen_progress["current"] = 0
    _gen_progress["total"] = total
    _gen_progress["active"] = True
    _gen_progress["phase"] = "generating"
    _gen_progress["started_at"] = _time.monotonic()
    _gen_progress["it_per_s"] = 0.0

def _progress_done():
    _gen_progress["current"] = _gen_progress["total"]
    _gen_progress["active"] = False
    _gen_progress["phase"] = "idle"

def _progress_step(step: int):
    _gen_progress["current"] = step
    elapsed = _time.monotonic() - _gen_progress["started_at"]
    if elapsed > 0 and step > 0:
        _gen_progress["it_per_s"] = round(step / elapsed, 2)


# =============================================================================
# Helper Functions
# =============================================================================

def get_cfg_scale():
    """Get CFG scale for image generation. Auto-detect VRAM for Vulkan."""
    global global_args
    
    cfg_scale = getattr(global_args, 'image_cfg_scale', 1.0)
    
    # If using Vulkan and CLI didn't specify cfg_scale (default 1.0), check VRAM
    if cfg_scale == 1.0:  # Only auto-detect if using default
        backend = getattr(global_args, 'backend', 'auto')
        image_backend = getattr(global_args, 'image_backend', 'auto')
        
        # Check if using Vulkan (either global or image-specific)
        use_vulkan = (backend == 'vulkan') or (image_backend == 'vulkan') or (image_backend == 'auto' and backend == 'auto')
        
        if use_vulkan:
            # Try to detect VRAM
            try:
                import subprocess
                # Try vulkaninfo first
                result = subprocess.run(['vulkaninfo', '-J'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    import json
                    data = json.loads(result.stdout)
                    # Find device memory
                    for dev in data.get('devices', []):
                        mem = dev.get('deviceMemoryHeap', [{}])
                        for heap in mem:
                            if heap.get('flags', []).get('deviceLocal', False):
                                vram_mb = heap.get('size', 0) / (1024 * 1024)
                                _log.debug("Detected VRAM: %.0f MB", vram_mb)
                                if vram_mb < 16000:  # Less than 16GB
                                    return 1.0
                                break
            except Exception as e:
                _log.debug("Could not detect VRAM: %s", e)
                return 1.0
    
    return cfg_scale


def save_image_response(img, request_format="base64", http_request=None):
    """
    Save image to file path if configured, return response dict.
    
    If --file-path is set and request_format is url (not base64), return only URL.
    If --file-path is set and request_format is base64, return both URL and base64.
    If --file-path is not set, return base64 as usual.
    """
    global global_file_path, global_args
    
    # Convert to PIL Image if needed
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    
    result = {}
    
    # Save to file path if configured
    if global_file_path:
        os.makedirs(global_file_path, exist_ok=True)
        # Generate unique filename
        filename = f"{uuid.uuid4().hex}.png"
        file_path = os.path.join(global_file_path, filename)
        img.save(file_path, format="PNG")
        # Add URL to response
        from codai.api.urlutils import build_file_url
        result["url"] = build_file_url(filename, http_request)
        
        # If client explicitly requested base64, include it
        # Otherwise, only return URL when file-path is set
        if request_format == "base64":
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_bytes = buffered.getvalue()
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            result["b64_json"] = img_base64
    else:
        # No file-path, return base64 as usual
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        result["b64_json"] = img_base64
    
    return result


def set_global_args(args):
    """Set global args from coderai."""
    global global_args
    global_args = args


def set_global_file_path(path):
    """Set global file path from coderai."""
    global global_file_path
    global_file_path = path


def set_model_semaphores(semaphores):
    """Set model semaphores from coderai for concurrency control."""
    global model_semaphores
    model_semaphores = semaphores


def set_queue_flags(flags):
    """Set queue flags from coderai."""
    global queue_flags
    queue_flags = flags


def _is_gguf_model(model_name: str) -> bool:
    """Check if a model name/path indicates a GGUF model."""
    if not model_name:
        return False
    return (model_name.endswith('.gguf') or 
            'gguf' in model_name.lower() or
            (model_name.startswith('http') and '.gguf' in model_name))


def _derive_diffusers_device(global_args) -> str:
    """Derive the CUDA device string for diffusers from global args.
    
    Checks --image-vulkan-device then --vulkan-device to determine
    which CUDA device to target. Defaults to 'cuda:0'.
    """
    if global_args:
        # Check image-specific device first
        image_device = getattr(global_args, 'image_vulkan_device', None)
        if image_device is not None:
            return f"cuda:{image_device}"
        # Fall back to general device
        device_id = getattr(global_args, 'vulkan_device', 0)
        if device_id is not None and device_id != 0:
            return f"cuda:{device_id}"
    return "cuda:0"


def _disable_safety_checker(pipe):
    """Null out every safety gate a diffusers pipeline may have.

    Works on SD 1.x/2.x (safety_checker + feature_extractor),
    SDXL/Flux/video pipelines (no safety_checker but may have safety_concept),
    and any future pipeline that gains one. Safe to call on pipelines that
    never had any of these attributes.
    """
    if hasattr(pipe, 'safety_checker') and pipe.safety_checker is not None:
        pipe.safety_checker = None
    if hasattr(pipe, 'feature_extractor') and pipe.feature_extractor is not None:
        # Keep the extractor object but disconnect it from the checker so
        # it cannot produce a blocking signal.
        try:
            pipe.feature_extractor = None
        except Exception:
            pass
    if hasattr(pipe, 'safety_concept'):
        pipe.safety_concept = None
    if hasattr(pipe, 'requires_safety_checker'):
        try:
            pipe.requires_safety_checker = False
        except Exception:
            pass
    return pipe


def _apply_image_acceleration(pipeline, model_config):
    """Fuse a configured acceleration/distillation LoRA (Lightning / Turbo / LCM /
    Hyper-SD) into a freshly loaded diffusers image pipeline. No-op when no
    acceleration is configured. Failures are caught inside apply_accel_to_pipeline."""
    try:
        from codai.models.acceleration import resolve_acceleration, apply_accel_to_pipeline
        accel = resolve_acceleration(model_config)
        if accel:
            print(f"  [image][accel] applying {accel.get('preset')} "
                  f"(steps={accel.get('steps')}, guidance={accel.get('guidance_scale')})")
            apply_accel_to_pipeline(pipeline, accel)
    except Exception as e:
        print(f"  [image][accel] skipped: {e}")
    return pipeline


def _load_diffusers_pipeline(model_name: str, global_args, model_config: dict = None):
    """
    Try to load a model using the diffusers library.

    Returns the loaded pipeline or None if diffusers can't handle this model.
    Raises Exception if loading fails for other reasons.

    Per-model configuration (model_config) is the source of truth and takes
    precedence over CLI/global args for precision, offload, quantization, etc.
    """
    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, DiffusionPipeline
    import torch

    _mc = model_config or {}

    def _cfg(key, default=None):
        """Read a value from the per-model configuration only (source of truth)."""
        v = _mc.get(key)
        return v if v is not None else default

    # All loading parameters come from the per-model configuration.
    no_ram = bool(_cfg('no_ram', False))
    precision = _cfg('precision', 'f32') or 'f32'
    precision_map = {
        'bf16': torch.bfloat16,
        'f32': torch.float32,
        'f16': torch.float16,
    }
    if hasattr(torch, 'float8_e4m3fn'):
        precision_map['f8'] = torch.float8_e4m3fn
    dtype = precision_map.get(precision, torch.float32)
    
    # --no-ram mode: override dtype to auto-select best for GPU
    if no_ram:
        dtype = torch.float16  # Use fp16 to maximize VRAM efficiency
        print(f"--no-ram mode: Using precision fp16 for maximum VRAM efficiency")
    else:
        # A quantized checkpoint — pre-quantized (bnb 4/8-bit, fp8, nf4, gptq, awq;
        # e.g. Z-Image-Turbo-unsloth-bnb-4bit) or runtime-quantized via config —
        # dequantizes to a HALF compute dtype, and modern diffusion transformers use
        # FlashAttention, which only supports fp16/bf16. Loading such a model at the
        # f32 default both wastes VRAM and crashes generation with "FlashAttention
        # only support fp16 and bf16 data type". When precision was left at f32 for a
        # quantized model, use bf16 instead.
        if dtype == torch.float32:
            _name_l = (model_name or '').lower()
            _prequant = any(t in _name_l for t in (
                'bnb-4bit', 'bnb_4bit', '-4bit', '_4bit', '4bit', '8bit', 'fp8',
                'int4', 'int8', 'nf4', 'gptq', 'awq'))
            _cfg_quant = bool(_cfg('load_in_4bit') or _cfg('load_in_8bit')
                              or _cfg('component_quantization'))
            if _prequant or _cfg_quant:
                print("  [image] quantized model with f32 precision → using bf16 "
                      "(f32 breaks FlashAttention and defeats quantization)")
                dtype = torch.bfloat16
                precision = 'bf16'
        print(f"Using precision: {precision} ({dtype})")
    
    # CPU offload comes from the per-model configuration: an explicit
    # cpu_offload flag, or an offload_strategy that implies CPU offloading.
    _offload_strategy = _mc.get('offload_strategy')
    use_sequential_offload = bool(
        _mc.get('cpu_offload')
        or (_offload_strategy in ('cpu', 'sequential', 'model', 'disk'))
    )

    # Quantization (per-model config).  Builds a diffusers quantization config
    # applied per-component so 4-bit/8-bit image models use less VRAM.  Per-model
    # 'component_quantization' overrides win; otherwise the global flag applies
    # to all heavy components (backbone + text encoders).
    from codai.models.hf_loading import (
        build_pipeline_quant_config, build_gguf_pipeline_components)
    _img_quant_config, _img_quant_desc = build_pipeline_quant_config(model_name, _mc, dtype)
    if _img_quant_config is not None:
        print(f"Image quantization: {_img_quant_desc}")
    _img_gguf_components, _img_gguf_desc = build_gguf_pipeline_components(model_name, _mc, dtype)
    if _img_gguf_components:
        print(f"Image GGUF components: {_img_gguf_desc}")
    
    # --no-ram mode: never use CPU offload
    if no_ram and use_sequential_offload:
        print("--no-ram mode: ignoring --image-cpu-offload, forcing full GPU loading")
        use_sequential_offload = False
    
    # Refuse to load a model that is currently being downloaded — the HF hub
    # file lock on the same repo would deadlock the process.
    try:
        from codai.admin.routes import get_active_download_model_ids
        active_downloads = get_active_download_model_ids()
        if model_name in active_downloads:
            raise RuntimeError(
                f"Model '{model_name}' is currently being downloaded. "
                "Wait for the download to finish before loading it."
            )
    except ImportError:
        pass

    # =====================================================================
    # --no-ram mode: load directly on GPU, no CPU RAM fallback
    # =====================================================================
    if no_ram and torch.cuda.is_available():
        cuda_device = _derive_diffusers_device(global_args)
        print(f"--no-ram mode: loading diffusers model directly on {cuda_device}")
        
        try:
            _xtra = {}
            if _img_quant_config is not None:
                _xtra['quantization_config'] = _img_quant_config
            if _img_gguf_components:
                _xtra.update(_img_gguf_components)
            with _DIFFUSERS_LOAD_LOCK:
                try:
                    pipeline = StableDiffusionXLPipeline.from_pretrained(
                        model_name,
                        torch_dtype=dtype,
                        use_safetensors=True,
                        **_xtra,
                    )
                except Exception:
                    pipeline = DiffusionPipeline.from_pretrained(
                        model_name,
                        torch_dtype=dtype,
                        use_safetensors=True,
                        **_xtra,
                    )
            try:
                pipeline = pipeline.to(cuda_device)
            except Exception:
                if _img_quant_config is None:
                    raise  # only quantized pipelines may reject .to()
            print(f"--no-ram: Diffusers model loaded on {cuda_device}")
            return _apply_image_acceleration(pipeline, _mc)
        except Exception as e:
            raise RuntimeError(
                f"--no-ram: Failed to load diffusers model entirely on GPU ({cuda_device}). "
                f"The model may be too large for available VRAM. Error: {e}"
            )
    
    # =====================================================================
    # Proactive VRAM eviction before first load attempt
    # =====================================================================
    # Evict only the minimum needed so models that fit together can coexist.
    # Prefer the model's configured/estimated VRAM need; only fall back to the
    # blunt "free < 15%" heuristic when the size is genuinely unknown.
    if torch.cuda.is_available():
        try:
            from codai.models.manager import multi_model_manager as _mmm
            if _mmm.models:
                _free, _total = torch.cuda.mem_get_info()
                _free_gb = _free / 1e9
                # Needed VRAM for this model (config used_vram_gb, with quant/offload
                # factors applied) — 0 when it can't be determined.
                _key = None
                for _k in (f"image:{model_name}", model_name):
                    if _k in _mmm.config:
                        _key = _k
                        break
                _need_gb = _mmm._get_model_used_vram_gb(_key or model_name, model_name)
                if _need_gb > 0:
                    if _free_gb < _need_gb:
                        print(f"Image model needs {_need_gb:.1f} GB, {_free_gb:.1f} GB free "
                              f"— evicting the minimum to fit (others may coexist)")
                        _mmm._evict_models_for_vram(_need_gb)
                    else:
                        print(f"Image model needs {_need_gb:.1f} GB, {_free_gb:.1f} GB free "
                              f"— no eviction needed (coexisting with loaded models)")
                elif _total > 0 and (_free / _total) < 0.15:
                    # Size unknown and VRAM nearly full — evict LRU one at a time
                    # until we clear ~25% headroom, instead of nuking everything.
                    print(f"Low VRAM ({_free_gb:.1f} GB free of {_total/1e9:.1f} GB), "
                          f"unknown model size — evicting LRU to free headroom")
                    _mmm._evict_models_for_vram(_total * 0.25 / 1e9)
        except Exception as _ee:
            print(f"  Proactive eviction skipped: {_ee}")

    # =====================================================================
    # Standard loading path (with OOM fallback)
    # =====================================================================
    # Track loading attempts for OOM handling
    pipeline = None
    load_attempt = 0
    max_attempts = 3
    
    while pipeline is None and load_attempt < max_attempts:
        try:
            load_attempt += 1
            print(f"Loading attempt {load_attempt}/{max_attempts}...")

            # Acquire the global load lock before any from_pretrained call.
            # This prevents concurrent HF hub file-lock conflicts (e.g. when
            # another pipeline or snapshot_download holds the same .lock file).
            with _DIFFUSERS_LOAD_LOCK:
                # Re-check download conflict inside the lock — a download may
                # have started between our first check and acquiring the lock.
                try:
                    from codai.admin.routes import get_active_download_model_ids
                    if model_name in get_active_download_model_ids():
                        raise RuntimeError(
                            f"Model '{model_name}' started downloading while waiting "
                            "for the load lock. Wait for the download to finish."
                        )
                except ImportError:
                    pass

                # Inject per-model quantization config when configured.
                _xtra = {}
                if _img_quant_config is not None:
                    _xtra['quantization_config'] = _img_quant_config
                if _img_gguf_components:
                    _xtra.update(_img_gguf_components)
                # Try to load as Stable Diffusion XL first, then generic DiffusionPipeline
                try:
                    pipeline = StableDiffusionXLPipeline.from_pretrained(
                        model_name,
                        torch_dtype=dtype,
                        use_safetensors=True,
                        **_xtra,
                    )
                except Exception:
                    # Try generic diffusion pipeline (supports custom pipelines like ZImagePipeline)
                    pipeline = DiffusionPipeline.from_pretrained(
                        model_name,
                        torch_dtype=dtype,
                        use_safetensors=True,
                        **_xtra,
                    )

            # Apply memory optimizations based on attempt
            if torch.cuda.is_available():
                if load_attempt >= 2:
                    # Second attempt: enable attention slicing
                    print("Enabling attention slicing for lower VRAM usage...")
                    if hasattr(pipeline, 'enable_attention_slicing'):
                        pipeline.enable_attention_slicing()

                if _img_quant_config is not None:
                    # Quantized (bitsandbytes) pipelines are already placed on GPU
                    # by from_pretrained and cannot be moved with .to(); only the
                    # non-quantized components need an explicit device move.
                    print("Quantized pipeline — placing non-quantized components on GPU")
                    try:
                        pipeline = pipeline.to("cuda")
                    except Exception:
                        pass  # bitsandbytes components stay where loaded
                elif load_attempt >= 3 or use_sequential_offload:
                    # Third attempt or offload requested: enable sequential CPU offload
                    print("Enabling sequential CPU offload for lower VRAM usage...")
                    if hasattr(pipeline, 'enable_sequential_cpu_offload'):
                        pipeline.enable_sequential_cpu_offload()
                else:
                    # First attempt: try regular GPU
                    pipeline = pipeline.to("cuda")
            else:
                pipeline = pipeline.to("cpu")
            
        except Exception as load_error:
            error_msg = str(load_error).lower()
            is_oom = any(x in error_msg for x in ['out of memory', 'oom', 'cuda error', 'cudamalloc'])
            
            if is_oom and load_attempt < max_attempts:
                print(f"OOM during model loading: {load_error}")
                # Evict other loaded models from VRAM before retrying
                from codai.models.manager import multi_model_manager as _mmm
                if _mmm.models:
                    print(f"Evicting {len(_mmm.models)} loaded model(s) to free VRAM for retry...")
                    _mmm.unload_all_models()
                else:
                    import gc as _gc
                    _gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                print(f"Retrying with more aggressive memory optimization...")
                pipeline = None  # Reset for retry
            else:
                print(f"Failed to load model (attempt {load_attempt}): {load_error}")
                if load_attempt >= max_attempts:
                    raise
                pipeline = None

    return _apply_image_acceleration(pipeline, _mc)


async def _apply_vae_override(pipeline, vae_model_id: str):
    """Swap the pipeline's VAE with an alternate model (diffusers only)."""
    try:
        import torch
        from diffusers import AutoencoderKL
        dtype = next(pipeline.parameters()).dtype if hasattr(pipeline, 'parameters') else torch.float16
        vae = AutoencoderKL.from_pretrained(vae_model_id, torch_dtype=dtype)
        vae = vae.to(pipeline.device)
        pipeline.vae = vae
        _log.info("VAE override applied: %s", vae_model_id)
    except Exception as e:
        _log.warning("Could not load VAE override %s: %s", vae_model_id, e)


def _ensure_ip_adapter_loaded(pipeline) -> bool:
    """Lazily load IP-Adapter weights matching the pipeline architecture.

    Returns True if IP-Adapter is loaded and ready (so the caller can pass
    ip_adapter_image), False if this pipeline type isn't supported.  The result
    is cached on the pipeline so repeated requests don't reload.
    """
    # Already loaded (or already known-unsupported) for this pipeline instance.
    flag = getattr(pipeline, '_coderai_ip_state', None)
    if flag == 'loaded':
        return True
    if flag == 'unsupported':
        return False
    if not hasattr(pipeline, 'load_ip_adapter'):
        try:
            pipeline._coderai_ip_state = 'unsupported'
        except Exception:
            pass
        return False

    # Detect SDXL vs SD1.5 by the presence of a second text encoder.
    is_sdxl = hasattr(pipeline, 'text_encoder_2') and getattr(pipeline, 'text_encoder_2', None) is not None
    cls_name = type(pipeline).__name__.lower()
    if 'xl' in cls_name:
        is_sdxl = True

    attempts = []
    if is_sdxl:
        attempts.append(("h94/IP-Adapter", "sdxl_models", "ip-adapter_sdxl.bin"))
    else:
        attempts.append(("h94/IP-Adapter", "models", "ip-adapter_sd15.bin"))

    for repo, subfolder, weight_name in attempts:
        try:
            pipeline.load_ip_adapter(repo, subfolder=subfolder, weight_name=weight_name)
            pipeline._coderai_ip_state = 'loaded'
            _log.info("IP-Adapter loaded: %s/%s/%s", repo, subfolder, weight_name)
            return True
        except Exception as e:
            _log.warning("IP-Adapter load failed (%s/%s): %s", repo, weight_name, e)

    try:
        pipeline._coderai_ip_state = 'unsupported'
    except Exception:
        pass
    return False


def _apply_loras(pipeline, loras):
    """Load and activate LoRA weights on a diffusers pipeline."""
    try:
        # peft dispatches AWQ for any non-bnb target when gptqmodel is installed;
        # gptqmodel 7.1.0 renamed the class peft imports, which would crash
        # load_lora_weights below. Alias it before applying any adapter.
        from codai.models.peft_compat import ensure_peft_awq_compat
        ensure_peft_awq_compat()
        # The pipeline is cached and reused across requests, so the adapters we
        # added last time linger. Re-loading the same fighter then fails with
        # "Adapter name <x> already in use", and stale adapters from a different
        # request would accumulate in VRAM. Drop the ones WE added previously
        # (tracked by name) before re-applying — leaving any acceleration/turbo
        # LoRA (a different adapter name, loaded elsewhere) untouched.
        prev = list(getattr(pipeline, "_coderai_request_adapters", None) or [])
        if prev:
            try:
                pipeline.delete_adapters(prev)
            except Exception:
                pass
            pipeline._coderai_request_adapters = []
        names = []
        weights = []
        for i, lora in enumerate(loras):
            name = lora.name or f"lora_{i}"
            # Defensive: if this exact name somehow survived, drop it first so the
            # load can't collide.
            try:
                pipeline.delete_adapters(name)
            except Exception:
                pass
            pipeline.load_lora_weights(lora.model, adapter_name=name)
            names.append(name)
            weights.append(float(lora.weight if lora.weight is not None else 1.0))
        if names:
            pipeline.set_adapters(names, weights)
            pipeline._coderai_request_adapters = list(names)
            _log.info("LoRA weights applied: %s", names)
    except Exception as e:
        _log.warning("Could not apply LoRA weights: %s", e)


async def _generate_with_diffusers(pipeline, request, global_args, http_request=None):
    """Generate images using a diffusers pipeline (with prompt-embedding cache)."""
    import torch
    import numpy as np
    import time as time_module

    if getattr(request, 'disable_safety_checker', False):
        _disable_safety_checker(pipeline)

    # Apply optional per-request VAE override
    if getattr(request, 'vae_model', None):
        await _apply_vae_override(pipeline, request.vae_model)

    # Apply optional per-request LoRA weights
    if getattr(request, 'loras', None):
        _apply_loras(pipeline, request.loras)

    # Determine size
    width, height = 1024, 1024
    if request.size:
        parts = request.size.split("x")
        if len(parts) == 2:
            try:
                width = int(parts[0])
                height = int(parts[1])
            except ValueError:
                pass

    if width != width or width == float('inf'):
        width = 512
    if height != height or height == float('inf'):
        height = 512

    # Enable memory optimizations
    try:
        if hasattr(pipeline, 'enable_attention_slicing'):
            pipeline.enable_attention_slicing(slice_size="auto")
        if hasattr(pipeline, 'vae') and hasattr(pipeline.vae, 'enable_slicing'):
            pipeline.vae.enable_slicing()
        elif hasattr(pipeline, 'enable_vae_slicing'):
            pipeline.enable_vae_slicing()
    except Exception as e:
        print(f"Warning: Could not enable memory optimizations: {e}")

    timestamp = int(time_module.time())

    seed = request.seed if request.seed is not None else getattr(global_args, 'image_seed', None)
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)

    quality = request.quality or "standard"
    # Acceleration/distillation defaults (Lightning / Turbo / LCM / Hyper-SD): when
    # the loaded pipeline has a fused distill LoRA, default to its low step-count /
    # guidance. The request always wins if it specified steps/guidance.
    _accel = getattr(pipeline, '_coderai_accel', None)
    _accel_steps = _accel.get('steps') if _accel else None
    _accel_cfg = _accel.get('guidance_scale') if _accel else None
    num_steps = request.steps if request.steps else (
        _accel_steps if _accel_steps else (30 if quality == "standard" else 50))
    cfg_scale = request.guidance_scale if request.guidance_scale else (
        _accel_cfg if _accel_cfg is not None else
        (getattr(global_args, 'image_cfg_scale', 7.5) if quality == "standard" else 9.0)
    )

    _progress_reset(num_steps)

    # Register this generation as a cancellable task (live view + cooperative
    # cancel via the step callback below).
    _tid = task_registry.register(
        "image", title=(request.prompt or "")[:80],
        model=getattr(request, 'model', '') or '', total=num_steps)
    task_registry.start(_tid)

    # ------------------------------------------------------------------
    # Prompt embedding cache
    # Try to encode the prompt once and reuse the embeddings.
    # Falls back to passing the plain text prompt if encoding fails.
    # ------------------------------------------------------------------
    # Bind the cache entry to this specific *loaded* pipeline instance, so a
    # reload (the model-swapping workflow does this constantly) is a guaranteed
    # cache miss instead of feeding stale GPU tensors into a fresh pipeline.
    _pipe_cls = str(type(pipeline).__name__)
    _model_name = getattr(pipeline, 'model_name_or_path', None) or _pipe_cls
    model_id = f"{_model_name}#{id(pipeline)}"
    neg_prompt = getattr(request, 'negative_prompt', None) or ""
    do_cfg = cfg_scale > 1.0

    # Some pipelines (Z-Image) return prompt embeddings as *per-sample lists*
    # rather than stacked tensors, and the pipeline consumes/extends those
    # lists during a run.  Reusing them on a later request corrupts the batch
    # dimension (`assert len(size) == bsz` in the Z-Image transformer's
    # unpatchify).  These are not safe to cache, so encode fresh every time.
    _embed_cacheable = "ZImage" not in _pipe_cls

    cached_embeds = _embed_cache.get(request.prompt, neg_prompt, model_id) if _embed_cacheable else None
    embed_kwargs = {}
    cache_hit = False

    if cached_embeds is not None:
        embed_kwargs = cached_embeds
        cache_hit = True
        print(f"Prompt embed cache HIT for model '{model_id}'")
    elif not _embed_cacheable:
        # Z-Image et al.: don't pre-encode at all.  Leaving embed_kwargs empty
        # makes the call below pass the raw prompt, so the pipeline runs its own
        # native encode + batching (the only path that survives reuse here).
        pass
    else:
        # Try to encode and cache
        try:
            if hasattr(pipeline, 'encode_prompt'):
                import inspect as _inspect
                _ep_params = set(_inspect.signature(pipeline.encode_prompt).parameters)
                _ep_kwargs = {"prompt": request.prompt}
                if "device" in _ep_params:
                    _ep_kwargs["device"] = pipeline.device
                if "num_images_per_prompt" in _ep_params:
                    _ep_kwargs["num_images_per_prompt"] = 1
                if "do_classifier_free_guidance" in _ep_params:
                    _ep_kwargs["do_classifier_free_guidance"] = do_cfg
                if "negative_prompt" in _ep_params:
                    _ep_kwargs["negative_prompt"] = neg_prompt or None
                enc = pipeline.encode_prompt(**_ep_kwargs)
                # enc is a tuple; length varies by pipeline type
                if len(enc) == 2:
                    # SD 1.x: (prompt_embeds, negative_prompt_embeds)
                    embed_kwargs = {
                        'prompt_embeds': enc[0],
                        'negative_prompt_embeds': enc[1],
                    }
                elif len(enc) == 4:
                    # SDXL: (prompt_embeds, negative_prompt_embeds,
                    #        pooled_prompt_embeds, negative_pooled_prompt_embeds)
                    embed_kwargs = {
                        'prompt_embeds': enc[0],
                        'negative_prompt_embeds': enc[1],
                        'pooled_prompt_embeds': enc[2],
                        'negative_pooled_prompt_embeds': enc[3],
                    }
                if embed_kwargs and _embed_cacheable:
                    _embed_cache.put(request.prompt, neg_prompt, model_id, embed_kwargs)
                    print(f"Prompt embed cache STORE for model '{model_id}'")
        except Exception as e:
            print(f"Warning: prompt encode/cache failed ({e}), using plain text prompt")
            embed_kwargs = {}

    def _step_cb(pipe, step_index, timestep, callback_kwargs):
        # Cooperative cancellation: abort at the next step boundary if cancelled.
        task_registry.raise_if_cancelled(_tid)
        # Cooperative pause: block here while the user has paused this task.
        task_registry.wait_if_paused(_tid)
        task_registry.step(_tid, step_index + 1)
        _progress_step(step_index + 1)
        # Mid-generation thermal checkpoint: pause between denoise steps if too hot.
        try:
            from codai.models.thermal import checkpoint as _thermal_checkpoint
            _thermal_checkpoint(context="image-gen")
        except Exception:
            pass
        return callback_kwargs

    # Resolve character references (saved profiles + inline images)
    char_images = []
    try:
        profiles = getattr(request, 'character_profiles', None) or []
        if profiles:
            from codai.api.characters import resolve_character_profiles
            char_images += resolve_character_profiles(profiles)
        inline = getattr(request, 'character_references', None) or []
        char_images += list(inline)
    except Exception:
        pass

    # Environment profiles feed the SAME IP-Adapter reference set, so a
    # regenerated location keyframe can match the references kept on disk.
    try:
        env_profiles = getattr(request, 'environment_profiles', None) or []
        if env_profiles:
            from codai.api.environments import resolve_environment_profiles
            char_images += resolve_environment_profiles(env_profiles)
    except Exception:
        pass

    # Build call kwargs
    if embed_kwargs:
        call_kwargs = dict(
            num_images_per_prompt=request.n,
            height=height,
            width=width,
            generator=generator,
            guidance_scale=cfg_scale,
            num_inference_steps=num_steps,
            callback_on_step_end=_step_cb,
            **embed_kwargs,
        )
    else:
        call_kwargs = dict(
            prompt=request.prompt,
            negative_prompt=neg_prompt or None,
            num_images_per_prompt=request.n,
            height=height,
            width=width,
            generator=generator,
            guidance_scale=cfg_scale,
            num_inference_steps=num_steps,
            callback_on_step_end=_step_cb,
        )

    # Inject IP-Adapter images if character references provided.  The pipeline
    # must have IP-Adapter *weights* loaded first — _ensure_ip_adapter_loaded
    # lazily downloads + loads the right checkpoint for the architecture.
    if char_images and hasattr(pipeline, 'set_ip_adapter_scale'):
        try:
            if _ensure_ip_adapter_loaded(pipeline):
                strength = getattr(request, 'character_strength', 0.6) or 0.6
                ref_imgs = []
                for ref in char_images:
                    from PIL import Image as PILImage
                    if ref.startswith('data:'):
                        _, b64 = ref.split(',', 1)
                        raw = base64.b64decode(b64)
                    else:
                        raw = base64.b64decode(ref)
                    ref_imgs.append(PILImage.open(io.BytesIO(raw)).convert('RGB'))
                pipeline.set_ip_adapter_scale(strength)
                call_kwargs['ip_adapter_image'] = ref_imgs[0] if len(ref_imgs) == 1 else ref_imgs
            else:
                print("Note: IP-Adapter weights unavailable for this pipeline — "
                      "relying on prompt/LoRA for character consistency")
        except Exception as _ip_err:
            print(f"Warning: IP-Adapter injection failed ({_ip_err}), continuing without character refs")

    # Reset the diffusers GLOBAL attention backend to the environment default
    # (native/SDPA) before generating. diffusers' Model.set_attention_backend()
    # ALSO flips a process-wide active backend, and the video path sets it to
    # flash-attn — which then leaks to image transformers that don't set their own
    # (e.g. Z-Image passes backend=None → uses the global) and crashes with
    # "`attn_mask` is not supported for flash-attn 2". Image + video share the
    # engine process, so restore the default here so masked image attention (SDPA)
    # always works. Cheap + idempotent; no-op if diffusers lacks the dispatcher.
    try:
        from diffusers.models.attention_dispatch import (
            _AttentionBackendRegistry, AttentionBackendName)
        from diffusers.utils.constants import DIFFUSERS_ATTN_BACKEND
        _AttentionBackendRegistry.set_active_backend(
            AttentionBackendName(DIFFUSERS_ATTN_BACKEND))
    except Exception:
        pass

    try:
        result = await asyncio.to_thread(pipeline, **call_kwargs)
    except TaskCancelled:
        _progress_done()
        raise  # global handler finishes the task (cancelled) + returns HTTP 499
    except TypeError:
        # Older pipeline that doesn't support callback_on_step_end
        call_kwargs.pop('callback_on_step_end', None)
        try:
            result = await asyncio.to_thread(pipeline, **call_kwargs)
        except TaskCancelled:
            _progress_done()
            raise
    except Exception as e:
        task_registry.finish(_tid, "error", str(e)[:200])
        _progress_done()
        raise
    finally:
        _progress_done()

    # Extract images
    images = []
    try:
        result_images = result.images
    except Exception as img_err:
        result_images = getattr(result, 'image', None) or getattr(result, 'output', None)
        if result_images is None:
            raise Exception(f"Could not extract images from diffusers result: {img_err}")

    _archive_artifacts = []
    for img in result_images:
        if isinstance(img, np.ndarray):
            img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
            img = np.clip(img, 0.0, 1.0)
        img_data = save_image_response(img, request.response_format, http_request)
        images.append(img_data)
        try:
            _buf = io.BytesIO()
            if isinstance(img, Image.Image):
                img.convert("RGB").save(_buf, "PNG")
            else:
                Image.fromarray(img).convert("RGB").save(_buf, "PNG")
            _archive_artifacts.append((_buf.getvalue(), "png"))
        except Exception:
            pass

    if _archive_artifacts:
        try:
            from codai.api.archive import archive_manager
            asyncio.create_task(asyncio.to_thread(
                archive_manager.save_generation,
                "image", "/v1/images/generations",
                getattr(request, 'model', None) or model_id,
                request.prompt,
                {
                    "size": request.size,
                    "n": request.n,
                    "steps": num_steps,
                    "guidance_scale": cfg_scale,
                    "seed": seed,
                    "quality": quality,
                    "negative_prompt": neg_prompt or None,
                },
                _archive_artifacts,
            ))
        except Exception:
            pass

    task_registry.finish(_tid, "done")
    return {
        "created": timestamp,
        "data": images,
        "prompt_cache_hit": cache_hit,
    }


async def _generate_with_sdcpp(sd_model, request, global_args, http_request=None,
                               model_config=None):
    """Generate images using stable-diffusion-cpp-python."""
    import time

    # Parse size
    width, height = 512, 512
    if request.size:
        parts = request.size.split("x")
        if len(parts) == 2:
            try:
                width = int(parts[0])
                height = int(parts[1])
            except ValueError:
                pass

    # Acceleration/distillation defaults (Lightning / Turbo / LCM): sd.cpp can't
    # fuse a diffusers LoRA, but honour the preset's low step-count / guidance and
    # inject the distill LoRA via the "<lora:name:weight>" prompt syntax when a
    # lora_model_dir is configured.
    from codai.models.acceleration import resolve_acceleration
    _accel = resolve_acceleration(model_config)
    _accel_steps = _accel.get('steps') if _accel else None
    _accel_cfg = _accel.get('guidance_scale') if _accel else None
    # Use default steps for fast generation
    steps = request.steps if request.steps else (_accel_steps or 4)
    cfg_scale = request.guidance_scale or _accel_cfg or get_cfg_scale()

    prompt = request.prompt
    if _accel and _accel.get('lora') and (model_config or {}).get('lora_model_dir'):
        from codai.models.acceleration import _split_lora_ref
        _repo, _wn = _split_lora_ref(_accel['lora'])
        _lname = (_wn or _repo).rsplit('/', 1)[-1]
        for _suf in ('.safetensors', '.ckpt', '.pt', '.bin'):
            if _lname.endswith(_suf):
                _lname = _lname[: -len(_suf)]
        prompt = f"{prompt} <lora:{_lname}:{_accel.get('lora_weight') or 1.0}>"

    _progress_reset(steps)

    # sd.cpp runs the whole diffusion inside one C call, so it can't be aborted
    # mid-step (raising from its progress callback won't reliably unwind the C
    # extension). We still register the task for visibility + step progress; a
    # cancel takes effect when control returns to Python.
    _tid = task_registry.register(
        "image", title=(request.prompt or "")[:80],
        model=getattr(request, 'model', '') or '', total=steps)
    task_registry.start(_tid)

    def _sdcpp_progress(step: int, total: int, elapsed: float):
        task_registry.step(_tid, step)
        _progress_step(step)

    # Use request seed if provided, otherwise use CLI default seed
    seed = request.seed if request.seed is not None else getattr(global_args, 'image_seed', None)

    try:
        result = await asyncio.to_thread(
            sd_model.generate_image,
            prompt=prompt,
            negative_prompt='',
            width=width,
            height=height,
            cfg_scale=cfg_scale,
            sample_steps=steps,
            seed=seed if seed is not None else 42,
            batch_count=request.n if request.n else 1,
            progress_callback=_sdcpp_progress,
        )
    except TypeError:
        result = await asyncio.to_thread(
            sd_model.generate_image,
            prompt=prompt,
            negative_prompt='',
            width=width,
            height=height,
            cfg_scale=cfg_scale,
            sample_steps=steps,
            seed=seed if seed is not None else 42,
            batch_count=request.n if request.n else 1,
        )
    except Exception as e:
        task_registry.finish(_tid, "error", str(e)[:200])
        _progress_done()
        raise
    finally:
        _progress_done()

    # Small delay to let Vulkan driver settle after generation
    time.sleep(0.1)
    
    # Convert results to response format
    images = []
    _archive_artifacts = []
    for img in result:
        img_data = save_image_response(img, http_request=http_request)
        images.append(img_data)
        try:
            _buf = io.BytesIO()
            if isinstance(img, Image.Image):
                img.convert("RGB").save(_buf, "PNG")
            else:
                Image.fromarray(img).convert("RGB").save(_buf, "PNG")
            _archive_artifacts.append((_buf.getvalue(), "png"))
        except Exception:
            pass

    if _archive_artifacts:
        try:
            import asyncio as _asyncio
            from codai.api.archive import archive_manager
            _asyncio.create_task(_asyncio.to_thread(
                archive_manager.save_generation,
                "image", "/v1/images/generations",
                getattr(request, 'model', None),
                request.prompt,
                {
                    "size": request.size,
                    "n": request.n,
                    "steps": steps,
                    "seed": seed,
                },
                _archive_artifacts,
            ))
        except Exception:
            pass

    task_registry.finish(_tid, "done")
    return {
        "created": int(time.time()),
        "data": images
    }


def _load_sdcpp_model(model_path: str, global_args, model_config: dict = None):
    """
    Try to load a model using stable-diffusion-cpp-python.
    
    Returns the loaded StableDiffusion model or None.
    """
    from stable_diffusion_cpp import StableDiffusion
    import stable_diffusion_cpp.stable_diffusion_cpp as sd_cpp
    import ctypes

    # Check for --no-ram mode
    no_ram = getattr(global_args, 'no_ram', False) if global_args else False

    print(f"Loading sd.cpp model from: {model_path}")

    # Intercept sd.cpp log to detect partial-init failures (e.g. unknown SD version)
    log_lines = []
    @sd_cpp.sd_log_callback
    def _log_cb(level, text, data):
        if text:
            line = text.decode('utf-8', errors='replace').rstrip()
            log_lines.append(line)
    sd_cpp.sd_set_log_callback(_log_cb, None)

    # Build sd.cpp constructor args from config
    kwargs = {
        'model_path': model_path,
        'offload_params_to_cpu': False,
        'keep_clip_on_cpu': False,
        'keep_control_net_on_cpu': False,
        'keep_vae_on_cpu': False,
    }

    # Add optional paths from CLI args
    if global_args:
        if hasattr(global_args, 'vae_path') and global_args.vae_path:
            kwargs['vae_path'] = global_args.vae_path
        if hasattr(global_args, 'llm_path') and global_args.llm_path:
            kwargs['lora_model_dir'] = global_args.llm_path

    # If backend is explicitly cpu, offload to CPU
    backend = (model_config or {}).get('backend', 'auto') if model_config else 'auto'
    if backend == 'cpu':
        kwargs['offload_params_to_cpu'] = True
        kwargs['keep_clip_on_cpu'] = True
        kwargs['keep_vae_on_cpu'] = True

    if no_ram:
        print("--no-ram mode: sd.cpp maximizing GPU usage (no CPU offload for CLIP/VAE/ControlNet)")

    try:
        sd_model = StableDiffusion(**kwargs)
    except Exception as e:
        if 'cpu' not in str(backend) and ('memory' in str(e).lower() or 'cuda' in str(e).lower() or 'out of' in str(e).lower()):
            print(f"GPU load failed ({e}), retrying with CPU offload...")
            kwargs['offload_params_to_cpu'] = True
            kwargs['keep_clip_on_cpu'] = True
            kwargs['keep_vae_on_cpu'] = True
            sd_model = StableDiffusion(**kwargs)
        else:
            raise
    finally:
        # Restore default log callback
        sd_cpp.sd_set_log_callback(None, None)

    # Check if sd.cpp failed to identify the model architecture.
    # In this case new_sd_ctx returns a non-null but broken context that
    # will segfault on free_sd_ctx — null out the pointer before raising so
    # the destructor skips the free call and doesn't kill the server process.
    failed_version = any('get sd version from file failed' in l for l in log_lines)
    if failed_version:
        try:
            if hasattr(sd_model, '_model') and hasattr(sd_model._model, 'model'):
                sd_model._model.model = None
        except Exception:
            pass
        raise ValueError(
            f"sd.cpp could not identify the model architecture in '{model_path}'. "
            "This model may require a newer version of stable-diffusion-cpp-python, "
            "or it may not be a supported Stable Diffusion GGUF format."
        )

    return sd_model


# =============================================================================
# Router and Endpoints
# =============================================================================

router = APIRouter()


@router.get("/v1/images/progress", summary="Image generation progress")
async def get_image_progress():
    """Return current image generation step progress including speed."""
    elapsed = _time.monotonic() - _gen_progress["started_at"] if _gen_progress["active"] else 0.0
    return {
        "current":  _gen_progress["current"],
        "total":    _gen_progress["total"],
        "active":   _gen_progress["active"],
        "phase":    _gen_progress.get("phase", "idle"),
        "model":    _gen_progress.get("model", ""),
        "pct":      int(_gen_progress["current"] / _gen_progress["total"] * 100)
                    if _gen_progress["total"] > 0 else 0,
        "it_per_s": _gen_progress["it_per_s"],
        "elapsed":  round(elapsed, 1),
    }


@router.post("/v1/images/generations", summary="Generate images (text-to-image)")
async def create_image_generation(request: ImageGenerationRequest, http_request: Request = None):
    """
    Image generation endpoint (OpenAI-compatible).
    
    Supports:
    - Stable Diffusion via stable-diffusion-cpp-python (sd.cpp)
    - Stable Diffusion XL (via local inference with diffusers)
    """
    global global_args, global_file_path, model_semaphores, queue_flags
    
    # Get or create semaphore for this model
    model_key = f"image:{request.model}" if request.model else "image"
    mode = get_load_mode()
    
    # Check if --image-1 is set (no queue, return 409 if busy)
    use_1_mode = queue_flags.get("image_1", False)
    
    # In loadall mode, allow 1 concurrent request per model
    # In ondemand mode, serialize all requests (use global semaphore)
    if mode == "loadall":
        if model_key not in model_semaphores:
            model_semaphores[model_key] = asyncio.Semaphore(1)
        semaphore = model_semaphores[model_key]
    else:
        # Use a global semaphore for ondemand mode
        if "global_image" not in model_semaphores:
            model_semaphores["global_image"] = asyncio.Semaphore(1)
        semaphore = model_semaphores["global_image"]
    
    # Try to acquire semaphore without blocking
    if use_1_mode:
        acquired = semaphore.locked()
        if acquired:
            raise HTTPException(
                status_code=409,
                detail="Image model is busy. Try again later."
            )
    
    async with semaphore:
        # =====================================================================
        # Step 1: Ask the manager to resolve the model and manage VRAM
        # =====================================================================
        _progress_loading(request.model or "image")
        # Reserve VRAM for any per-request LoRA adapters so eviction frees enough
        # headroom for base weights + adapters before the pipeline loads.
        _lora_extra_gb = 0.0
        if getattr(request, 'loras', None):
            # Resolve id/url/inline-file LoRA refs to local paths now (clean 400 on
            # a missing blob / unknown name) before any model work; _apply_loras
            # then reads lora.model as usual.
            from codai.api.loras import resolve_request_loras
            resolve_request_loras(request.loras)
            try:
                _lora_extra_gb = multi_model_manager._lora_vram_gb(request.loras)
            except Exception:
                _lora_extra_gb = 0.0
        model_info = await asyncio.to_thread(
            multi_model_manager.request_model,
            requested_model=request.model,
            model_type="image",
            extra_vram_gb=_lora_extra_gb,
        )
        
        # Check if the model was rejected as not allowed
        if model_info.get('error'):
            raise HTTPException(status_code=404, detail=model_info['error'])
        
        model_name = model_info['model_name']
        model_key = model_info['model_key']
        pipeline = model_info['model_object']
        
        # If no image model configured, try to use main --model as fallback
        if not model_name:
            main_model = getattr(global_args, 'model', None)
            if main_model and isinstance(main_model, list) and len(main_model) > 0:
                model_name = main_model[0]
            elif main_model:
                model_name = main_model
            
            # Check if main model is a GGUF file - can't use for image generation
            if model_name and _is_gguf_model(model_name):
                print(f"Note: Main model is a GGUF file (for text), not suitable for image generation")
                model_name = None
            
            if model_name:
                model_key = f"image:{model_name}"
        
        # If still no image model configured, return an error
        if not model_name:
            raise HTTPException(
                status_code=400,
                detail="Image generation not configured. Use --image-model to specify a model."
            )
        
        # =====================================================================
        # Step 2: Check if model is a sd.cpp StableDiffusion instance
        # =====================================================================
        is_sdcpp = False
        if pipeline is not None:
            try:
                from stable_diffusion_cpp import StableDiffusion
                if isinstance(pipeline, StableDiffusion):
                    is_sdcpp = True
            except ImportError:
                pass
        
        # =====================================================================
        # Step 3: If already loaded, generate with appropriate backend
        # =====================================================================
        if pipeline is not None:
            if is_sdcpp:
                print(f"Using cached sd.cpp model for generation")
                _sdcpp_cfg = (multi_model_manager.config.get(model_key)
                              or multi_model_manager.config.get(model_name) or {})
                return await _generate_with_sdcpp(pipeline, request, global_args,
                                                  http_request, model_config=_sdcpp_cfg)
            else:
                # Assume it's a diffusers pipeline
                print(f"Using cached diffusers pipeline for generation")
                return await _generate_with_diffusers(pipeline, request, global_args, http_request)
        
        # =====================================================================
        # Step 4: Model not loaded - try to load it
        # =====================================================================
        is_gguf = _is_gguf_model(model_name)
        diffusers_error = None
        sdcpp_error = None

        # Show the load as a (non-cancellable) Tasks-page entry spanning both
        # backend attempts; finished done on success, error only if all fail.
        from codai.tasks import task_registry as _treg
        _ltid = _treg.register("loading", title=f"Loading {model_name}",
                               model=model_key, status="running",
                               cancellable=False, pausable=False)

        # Try diffusers first (for non-GGUF models)
        if not is_gguf:
            try:
                print(f"Loading diffusers model: {model_name}")
                _diff_cfg = (multi_model_manager.config.get(model_key)
                             or multi_model_manager.config.get(model_name) or {})
                _vram_before = multi_model_manager.vram_before_load()
                pipeline = await asyncio.to_thread(
                    _load_diffusers_pipeline, model_name, global_args, _diff_cfg)

                if pipeline is not None:
                    # Cache the loaded pipeline in the manager
                    multi_model_manager.add_model(model_key, pipeline)
                    multi_model_manager.current_model_key = model_key
                    try:
                        multi_model_manager.record_vram_delta(model_key, _vram_before)
                    except Exception:
                        pass
                    print(f"Loaded diffusers model: {model_name}")

                    _treg.finish(_ltid, "done")
                    return await _generate_with_diffusers(pipeline, request, global_args, http_request)
                    
            except ImportError as e:
                diffusers_error = str(e)
                print(f"diffusers not available: {diffusers_error}")
            except Exception as e:
                import traceback
                diffusers_error = str(e)
                print(f"diffusers error: {diffusers_error}")
                print(f"Traceback: {traceback.format_exc()}")
        
        # Try stable-diffusion-cpp-python (for GGUF models or as fallback)
        try:
            # For GGUF models or URLs, resolve the model path through the cache
            resolved_path = model_name
            if is_gguf or model_name.startswith('http://') or model_name.startswith('https://'):
                resolved_path = await asyncio.to_thread(multi_model_manager.load_model, model_name)
                if not resolved_path:
                    raise Exception(f"Failed to resolve model path: {model_name}")

            # Only use sd.cpp if we have a local file path
            if resolved_path and os.path.isfile(resolved_path):
                cfg = multi_model_manager.config.get(model_key) or multi_model_manager.config.get(model_name) or {}
                _vram_before = multi_model_manager.vram_before_load()
                sd_model = await asyncio.to_thread(_load_sdcpp_model, resolved_path, global_args, model_config=cfg)

                if sd_model is not None:
                    # Cache the loaded model in the manager
                    multi_model_manager.add_model(model_key, sd_model)
                    multi_model_manager.current_model_key = model_key
                    try:
                        multi_model_manager.record_vram_delta(model_key, _vram_before)
                    except Exception:
                        pass
                    print(f"Loaded sd.cpp model: {model_name}")

                    _treg.finish(_ltid, "done")
                    return await _generate_with_sdcpp(sd_model, request, global_args,
                                                      http_request, model_config=cfg)
            else:
                sdcpp_error = f"Model '{model_name}' is not a local file, cannot use sd.cpp"
                print(sdcpp_error)
                
        except ImportError as e:
            sdcpp_error = str(e)
            print(f"stable-diffusion-cpp-python not available: {sdcpp_error}")
        except Exception as e:
            sdcpp_error = str(e)
            print(f"sd.cpp error: {sdcpp_error}")
        
        # =====================================================================
        # Step 5: Both backends failed - return error
        # =====================================================================
        error_details = []
        if diffusers_error:
            error_details.append(f"diffusers: {diffusers_error}")
        if sdcpp_error:
            error_details.append(f"sd.cpp: {sdcpp_error}")
        
        _treg.finish(_ltid, "error", "; ".join(error_details) or "no compatible backend")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to load image model '{model_name}'. Errors: {'; '.join(error_details) if error_details else 'No compatible backend found'}"
        )


# =============================================================================
# Image-to-Image Endpoint  (POST /v1/images/edits)
# OpenAI-compatible: accepts image + prompt, returns edited image
# =============================================================================

class ImageEditRequest(BaseModel):
    model: str
    prompt: str
    image: str              # base64-encoded PNG or "data:image/...;base64,..."
    mask: Optional[str] = None   # optional inpaint mask (base64 PNG)
    n: int = 1
    size: Optional[str] = "1024x1024"
    response_format: Optional[str] = "url"
    strength: Optional[float] = 0.75    # denoising strength (0=keep original, 1=ignore)
    steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    seed: Optional[int] = None
    quality: Optional[str] = "standard"
    user: Optional[str] = None

    class Config:
        extra = "allow"


def _decode_b64_image(data: str):
    """Decode base64 image string to PIL Image."""
    from PIL import Image as PILImage
    if data.startswith("data:"):
        _, encoded = data.split(",", 1)
        raw = base64.b64decode(encoded)
    else:
        raw = base64.b64decode(data)
    return PILImage.open(io.BytesIO(raw)).convert("RGB")


def _load_img2img_pipeline(model_name: str, global_args):
    """Load a diffusers img2img pipeline."""
    import torch
    from diffusers import (
        StableDiffusionImg2ImgPipeline,
        StableDiffusionXLImg2ImgPipeline,
        DiffusionPipeline,
    )

    device = _derive_diffusers_device(global_args)
    precision = getattr(global_args, 'image_precision', 'bf16') if global_args else 'bf16'
    dtype_map = {'bf16': torch.bfloat16, 'f16': torch.float16, 'f32': torch.float32}
    torch_dtype = dtype_map.get(precision, torch.bfloat16)

    name_lower = model_name.lower()
    if 'xl' in name_lower or 'sdxl' in name_lower or 'flux' in name_lower:
        PipeClass = StableDiffusionXLImg2ImgPipeline
    elif 'stable-diffusion' in name_lower or 'sd' in name_lower:
        PipeClass = StableDiffusionImg2ImgPipeline
    else:
        PipeClass = DiffusionPipeline   # generic fallback

    for attempt in range(3):
        try:
            with _DIFFUSERS_LOAD_LOCK:
                pipe = PipeClass.from_pretrained(model_name, torch_dtype=torch_dtype)
            pipe = pipe.to(device)
            if attempt >= 1:
                pipe.enable_attention_slicing()
            if attempt >= 2:
                pipe.enable_sequential_cpu_offload()
            return pipe
        except RuntimeError as e:
            if 'out of memory' in str(e).lower() and attempt < 2:
                import gc, torch
                from codai.models.manager import multi_model_manager as _mmm
                if _mmm.models:
                    print(f"OOM loading img2img — evicting {len(_mmm.models)} model(s) to free VRAM...")
                    _mmm.unload_all_models()
                else:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                continue
            raise


@router.post("/v1/images/edits", summary="Edit an image (instruction / img2img)")
async def create_image_edit(request: ImageEditRequest, http_request: Request = None):
    """
    Image-to-image editing endpoint (OpenAI-compatible).
    Accepts a base64-encoded source image and returns an edited image.
    """
    global global_args

    if not request.image:
        raise HTTPException(status_code=400, detail="image is required")

    _progress_loading(request.model or "image")
    model_info = await asyncio.to_thread(
        multi_model_manager.request_model, request.model, model_type="image")
    model_name = model_info.get('model_name')
    if not model_name:
        err = model_info.get('error', f"Model '{request.model}' not found or not registered")
        raise HTTPException(status_code=404, detail=err)

    model_key = f"img2img:{model_name}"
    pipe = multi_model_manager.models.get(model_key)

    if pipe is None:
        try:
            pipe = await asyncio.get_event_loop().run_in_executor(
                None, _load_img2img_pipeline, model_name, global_args
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load img2img model: {e}")
        multi_model_manager.models[model_key] = pipe

    try:
        import torch
        source_img = _decode_b64_image(request.image)

        width, height = source_img.size
        if request.size:
            parts = request.size.split("x")
            if len(parts) == 2:
                try:
                    width, height = int(parts[0]), int(parts[1])
                    source_img = source_img.resize((width, height))
                except ValueError:
                    pass

        seed = request.seed or getattr(global_args, 'image_seed', None)
        generator = torch.Generator(device=pipe.device).manual_seed(seed) if seed else None
        quality = request.quality or "standard"
        num_steps = request.steps or (30 if quality == "standard" else 50)
        cfg_scale = request.guidance_scale or (getattr(global_args, 'image_cfg_scale', 7.5) if quality == "standard" else 9.0)

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: pipe(
                prompt=request.prompt,
                image=source_img,
                strength=request.strength,
                num_inference_steps=num_steps,
                guidance_scale=cfg_scale,
                num_images_per_prompt=request.n,
                generator=generator,
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image editing failed: {e}")

    images = []
    for img in result.images:
        img_data = save_image_response(img, request.response_format, http_request)
        images.append(img_data)

    return {"created": int(time.time()), "data": images}




# =============================================================================
# Inpainting Endpoint  (POST /v1/images/inpaint)
# =============================================================================

class ImageInpaintRequest(BaseModel):
    model: str
    prompt: str
    image: str              # base64 source image
    mask: str               # base64 mask (white = inpaint region)
    n: int = 1
    size: Optional[str] = "1024x1024"
    response_format: Optional[str] = "url"
    steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    strength: Optional[float] = 0.99
    seed: Optional[int] = None
    quality: Optional[str] = "standard"

    class Config:
        extra = "allow"


def _load_inpaint_pipeline(model_name: str, global_args):
    import torch
    from diffusers import (
        StableDiffusionInpaintPipeline,
        StableDiffusionXLInpaintPipeline,
        DiffusionPipeline,
    )
    device = _derive_diffusers_device(global_args)
    precision = getattr(global_args, 'image_precision', 'bf16') if global_args else 'bf16'
    dtype_map = {'bf16': torch.bfloat16, 'f16': torch.float16, 'f32': torch.float32}
    torch_dtype = dtype_map.get(precision, torch.bfloat16)
    n = model_name.lower()
    if 'xl' in n or 'sdxl' in n:
        PClass = StableDiffusionXLInpaintPipeline
    elif 'stable-diffusion' in n or 'inpaint' in n:
        PClass = StableDiffusionInpaintPipeline
    else:
        PClass = DiffusionPipeline
    for attempt in range(3):
        try:
            with _DIFFUSERS_LOAD_LOCK:
                pipe = PClass.from_pretrained(model_name, torch_dtype=torch_dtype)
            pipe = pipe.to(device)
            if attempt >= 1:
                pipe.enable_attention_slicing()
            if attempt >= 2:
                pipe.enable_sequential_cpu_offload()
            return pipe
        except RuntimeError as e:
            if 'out of memory' in str(e).lower() and attempt < 2:
                import gc, torch
                from codai.models.manager import multi_model_manager as _mmm
                if _mmm.models:
                    print(f"OOM loading inpaint — evicting {len(_mmm.models)} model(s) to free VRAM...")
                    _mmm.unload_all_models()
                else:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                continue
            raise


@router.post("/v1/images/inpaint", summary="Inpaint a masked region")
async def create_image_inpaint(request: ImageInpaintRequest, http_request: Request = None):
    """Inpaint a masked region of an image (OpenAI-compatible extension)."""
    global global_args
    if not request.image or not request.mask:
        raise HTTPException(status_code=400, detail="image and mask are required")
    _progress_loading(request.model or "image")
    model_info = await asyncio.to_thread(
        multi_model_manager.request_model, request.model, model_type="image")
    model_name = model_info.get('model_name')
    if not model_name:
        raise HTTPException(status_code=404, detail=model_info.get('error', 'Model not found'))
    model_key = f"inpaint:{model_name}"
    pipe = multi_model_manager.models.get(model_key)
    if pipe is None:
        try:
            pipe = await asyncio.get_event_loop().run_in_executor(
                None, _load_inpaint_pipeline, model_name, global_args)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load inpaint model: {e}")
        multi_model_manager.models[model_key] = pipe
    try:
        import torch
        source_img = _decode_b64_image(request.image)
        mask_img = _decode_b64_image(request.mask).convert("L")  # greyscale mask
        if request.size:
            parts = request.size.split("x")
            if len(parts) == 2:
                try:
                    w, h = int(parts[0]), int(parts[1])
                    source_img = source_img.resize((w, h))
                    mask_img = mask_img.resize((w, h))
                except ValueError:
                    pass
        seed = request.seed or getattr(global_args, 'image_seed', None)
        generator = torch.Generator(device=pipe.device).manual_seed(seed) if seed else None
        quality = request.quality or "standard"
        num_steps = request.steps or (30 if quality == "standard" else 50)
        cfg_scale = request.guidance_scale or (getattr(global_args, 'image_cfg_scale', 7.5) if quality == "standard" else 9.0)
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: pipe(
                prompt=request.prompt,
                image=source_img,
                mask_image=mask_img,
                strength=request.strength,
                num_inference_steps=num_steps,
                guidance_scale=cfg_scale,
                num_images_per_prompt=request.n,
                generator=generator,
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inpainting failed: {e}")
    images = [save_image_response(img, request.response_format, http_request) for img in result.images]
    return {"created": int(time.time()), "data": images}


# =============================================================================
# Image Upscale Endpoint  (POST /v1/images/upscale)
# =============================================================================

class ImageUpscaleRequest(BaseModel):
    model: str
    image: str
    scale: Optional[int] = 4
    response_format: Optional[str] = "url"
    class Config:
        extra = "allow"


def _resolve_esrgan_weights(model_name: str):
    """Return local weights for an (Real-)ESRGAN model id. Accepts a local .pth /
    .safetensors file, a local directory containing one, or a Hugging Face repo id
    (the weights are downloaded via the HF cache). diffusers-style repos ship the
    RRDBNet weights as `diffusion_pytorch_model.safetensors` + a `config.json`, so
    we also fetch the config alongside. Returns the weights path, or None."""
    import os
    if os.path.isfile(model_name) and model_name.lower().endswith(('.pth', '.safetensors')):
        return model_name
    if os.path.isdir(model_name):
        cands = [f for f in sorted(os.listdir(model_name))
                 if f.lower().endswith(('.pth', '.safetensors'))]
        # Prefer .pth, then a full (non-fp16) safetensors.
        cands.sort(key=lambda f: (not f.lower().endswith('.pth'), '.fp16.' in f.lower(), len(f)))
        return os.path.join(model_name, cands[0]) if cands else None
    # Treat as an HF repo id → find and fetch a weight file (.pth or .safetensors).
    try:
        from huggingface_hub import list_repo_files, hf_hub_download
        files = list_repo_files(model_name)
        weights = [f for f in files if f.lower().endswith(('.pth', '.safetensors'))]
        if not weights:
            return None
        # Prefer: real .pth > full safetensors > fp16 safetensors; then a
        # general-purpose x4plus over anime/x2 specialised; then shorter name.
        def _rank(f):
            fl = f.lower()
            return (not fl.endswith('.pth'), '.fp16.' in fl,
                    'anime' in fl, 'x2' in fl, len(f))
        weights.sort(key=_rank)
        wp = hf_hub_download(model_name, weights[0])
        # Best-effort: co-locate config.json (diffusers repos carry the arch there).
        if 'config.json' in files:
            try:
                hf_hub_download(model_name, 'config.json')
            except Exception:
                pass
        return wp
    except Exception:
        return None


def _esrgan_state_dict(weights_path: str):
    """Load an (Real-)ESRGAN RRDBNet state dict from a .pth or .safetensors file,
    unwrapping the `params_ema` / `params` container used by the original .pth
    checkpoints (diffusers safetensors store the bare RRDBNet keys)."""
    if weights_path.lower().endswith('.safetensors'):
        from safetensors.torch import load_file
        return load_file(weights_path)
    import torch
    loadnet = torch.load(weights_path, map_location='cpu')
    if isinstance(loadnet, dict) and 'params_ema' in loadnet:
        return loadnet['params_ema']
    if isinstance(loadnet, dict) and 'params' in loadnet:
        return loadnet['params']
    return loadnet


def _build_realesrgan(weights_path: str, device):
    """Build a RealESRGANer from .pth or .safetensors weights. The RRDBNet
    architecture + native scale come from a sibling config.json when present
    (diffusers repos), otherwise they are inferred from the filename."""
    import os, json, hashlib, tempfile
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    num_in_ch, num_out_ch, num_feat, num_grow_ch, num_block, scale = 3, 3, 64, 32, 23, 4
    cfg_path = os.path.join(os.path.dirname(weights_path), 'config.json')
    cfg = None
    if os.path.isfile(cfg_path):
        try:
            cfg = json.load(open(cfg_path))
        except Exception:
            cfg = None
    if cfg:
        num_in_ch = int(cfg.get('num_in_ch', num_in_ch))
        num_out_ch = int(cfg.get('num_out_ch', num_out_ch))
        num_feat = int(cfg.get('num_feat', num_feat))
        num_grow_ch = int(cfg.get('num_grow_ch', num_grow_ch))
        num_block = int(cfg.get('num_block', num_block))
        scale = int(cfg.get('scale', scale))
    else:
        fn = os.path.basename(weights_path).lower()
        if 'anime' in fn or '6b' in fn:        # RealESRGAN_x4plus_anime_6B
            num_block, scale = 6, 4
        elif 'x2' in fn:                        # RealESRGAN_x2plus
            num_block, scale = 23, 2
        else:                                   # RealESRGAN_x4plus (default)
            num_block, scale = 23, 4

    model_obj = RRDBNet(num_in_ch=num_in_ch, num_out_ch=num_out_ch, num_feat=num_feat,
                        num_block=num_block, num_grow_ch=num_grow_ch, scale=scale)

    # RealESRGANer always torch.load()s model_path; a .safetensors won't load that
    # way, so convert once to a cached .pth wrapped as {'params_ema': sd}.
    load_path = weights_path
    if weights_path.lower().endswith('.safetensors'):
        import torch
        sd = _esrgan_state_dict(weights_path)
        h = hashlib.sha1(os.path.abspath(weights_path).encode()).hexdigest()[:16]
        load_path = os.path.join(tempfile.gettempdir(), f"realesrgan_{h}.pth")
        if not os.path.isfile(load_path):
            torch.save({'params_ema': sd}, load_path)

    half = 'cuda' in str(device)
    # tile>0 keeps VRAM bounded on large frames (no visible seams at this size).
    return RealESRGANer(scale=scale, model_path=load_path, model=model_obj,
                        tile=512, tile_pad=10, pre_pad=0, half=half, device=device)


# Private cache of loaded super-resolution upscalers, keyed by resolved model id.
# Deliberately NOT stored in multi_model_manager.models — that registry is keyed
# by real model ids, so a synthetic 'upscale:<id>' key there makes a later
# request_model() resolve to the bogus key and try to (re)load it.
_UPSCALER_CACHE: dict = {}


def _load_upscaler(model_name: str, global_args):
    import logging
    _log = logging.getLogger(__name__)
    device = _derive_diffusers_device(global_args)
    n = model_name.lower()
    try:
        import torch
        _dtype = torch.float16 if 'cuda' in str(device) else torch.float32
    except Exception:
        _dtype = None
    _kw = {} if _dtype is None else {"torch_dtype": _dtype}

    # 1. Real-ESRGAN / ESRGAN — GAN super-resolution from a .pth (fast, one
    #    forward pass per frame; ideal for video). Resolves local or HF weights.
    if 'esrgan' in n:
        try:
            wp = _resolve_esrgan_weights(model_name)
            if wp:
                return ('realesrgan', _build_realesrgan(wp, device))
            _log.warning("no .pth weights found for ESRGAN model '%s'", model_name)
        except Exception as e:
            _log.warning("Real-ESRGAN load failed for '%s': %s", model_name, e)

    # 2. Latent upscaler (StableDiffusionLatentUpscalePipeline, fixed ×2).
    if 'latent' in n and ('upscal' in n or 'x2' in n):
        try:
            from diffusers import StableDiffusionLatentUpscalePipeline
            pipe = StableDiffusionLatentUpscalePipeline.from_pretrained(model_name, **_kw)
            return ('diffusers_latent', pipe.to(device))
        except Exception as e:
            _log.warning("latent upscaler load failed for '%s': %s", model_name, e)

    # 3. x4 SD super-res upscaler (StableDiffusionUpscalePipeline).
    if 'upscal' in n:
        try:
            from diffusers import StableDiffusionUpscalePipeline
            pipe = StableDiffusionUpscalePipeline.from_pretrained(model_name, **_kw)
            return ('diffusers', pipe.to(device))
        except Exception:
            pass
        # Generic fallback: let diffusers pick the right pipeline class.
        try:
            from diffusers import DiffusionPipeline
            pipe = DiffusionPipeline.from_pretrained(model_name, **_kw)
            cls = type(pipe).__name__.lower()
            kind = 'diffusers_latent' if 'latent' in cls else 'diffusers'
            return (kind, pipe.to(device))
        except Exception as e:
            _log.warning("diffusers upscaler load failed for '%s': %s", model_name, e)

    # Not a recognised upscaler — callers treat 'pil' as "no real model".
    return ('pil', None)


def _run_upscale(upscaler, image_bytes: bytes, scale: int):
    from PIL import Image as PILImage
    import numpy as np, io as _io
    img = PILImage.open(_io.BytesIO(image_bytes)).convert("RGB")
    backend, model = upscaler
    if backend == 'realesrgan':
        out_arr, _ = model.enhance(np.array(img), outscale=scale)
        return PILImage.fromarray(out_arr)
    if backend == 'diffusers_latent':
        # Latent upscaler: fixed ×2; needs guidance_scale=0 for an unconditioned
        # detail-preserving upscale of an arbitrary input image.
        result = model(prompt="", image=img, num_inference_steps=20,
                       guidance_scale=0)
        return result.images[0]
    if backend == 'diffusers':
        result = model(prompt="", image=img, num_inference_steps=20)
        return result.images[0]
    # PIL fallback
    w, h = img.size
    return img.resize((w * scale, h * scale), PILImage.LANCZOS)


@router.post("/v1/images/upscale", summary="Upscale an image")
async def create_image_upscale(request: ImageUpscaleRequest, http_request: Request = None):
    """Upscale an image using Real-ESRGAN or PIL LANCZOS fallback."""
    global global_args
    _progress_loading(request.model or "image")
    model_info = await asyncio.to_thread(
        multi_model_manager.request_model, request.model, model_type="image")
    model_name = model_info.get('model_name') or request.model
    upscaler = _UPSCALER_CACHE.get(model_name)
    if upscaler is None:
        try:
            upscaler = await asyncio.get_event_loop().run_in_executor(
                None, _load_upscaler, model_name, global_args)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load upscaler: {e}")
        if upscaler[0] != 'pil':
            _UPSCALER_CACHE[model_name] = upscaler
    raw = base64.b64decode(request.image.split(',', 1)[-1] if ',' in request.image else request.image)
    try:
        out_img = await asyncio.get_event_loop().run_in_executor(
            None, _run_upscale, upscaler, raw, request.scale or 4)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upscaling failed: {e}")
    result = save_image_response(out_img, request.response_format, http_request)
    return {"created": int(time.time()), "data": [result]}


# =============================================================================
# Depth Estimation Endpoint  (POST /v1/images/depth)
# =============================================================================

class ImageDepthRequest(BaseModel):
    model: Optional[str] = None
    image: str
    response_format: Optional[str] = "url"
    class Config:
        extra = "allow"


def _load_depth_model(model_name: str, global_args, model_config: dict = None):
    device = _derive_diffusers_device(global_args)
    try:
        from transformers import pipeline as hf_pipeline
        from codai.models.hf_loading import pipeline_device_kwargs
        pk = pipeline_device_kwargs(model_config)
        # device and device_map are mutually exclusive in HF pipeline.
        if 'device_map' in pk:
            pipe = hf_pipeline("depth-estimation", model=model_name, **pk)
        else:
            pipe = hf_pipeline("depth-estimation", model=model_name, device=device, **pk)
        return ('transformers', pipe)
    except Exception:
        pass
    try:
        import torch, timm
        model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        model.eval().to(device)
        return ('midas', (model, device))
    except Exception as e:
        raise RuntimeError(f"Cannot load depth model: {e}")


def _run_depth(depth_model, image_bytes: bytes):
    from PIL import Image as PILImage
    import numpy as np, io as _io
    img = PILImage.open(_io.BytesIO(image_bytes)).convert("RGB")
    backend, model = depth_model
    if backend == 'transformers':
        result = model(img)
        depth_arr = np.array(result['depth'])
    else:
        import torch
        model_obj, device = model
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
        inp = transforms(np.array(img)).to(device)
        with torch.no_grad():
            depth_arr = model_obj(inp).squeeze().cpu().numpy()
    # Normalise to 0-255
    d_min, d_max = depth_arr.min(), depth_arr.max()
    if d_max > d_min:
        depth_arr = ((depth_arr - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    else:
        depth_arr = depth_arr.astype(np.uint8)
    return PILImage.fromarray(depth_arr)


def _resolve_spatial_model(requested: Optional[str], capability: str) -> Optional[str]:
    """Return a configured spatial model for the given capability, or the requested ID."""
    if requested:
        return requested
    try:
        from codai.admin.routes import config_manager
        if config_manager is not None:
            for entry in config_manager.models_data.get("spatial_models", []):
                if isinstance(entry, dict):
                    saved_caps = entry.get("capabilities") or []
                    mid = entry.get("path") or entry.get("id") or ""
                    if capability in saved_caps and mid:
                        return mid
        # Fall back to runtime list using name heuristic
        from codai.models.capabilities import detect_model_capabilities
        for m in multi_model_manager.spatial_models:
            caps = detect_model_capabilities(m)
            if getattr(caps, capability, False):
                return m
        if multi_model_manager.spatial_models:
            return multi_model_manager.spatial_models[0]
    except Exception:
        pass
    return None


@router.post("/v1/images/depth", summary="Estimate a depth map")
async def create_image_depth(request: ImageDepthRequest, http_request: Request = None):
    """Estimate depth map from an image."""
    global global_args
    model_name = _resolve_spatial_model(request.model, "depth_estimation")
    if not model_name:
        raise HTTPException(status_code=400, detail="No depth estimation model configured. Add one via Admin > Models.")
    model_key = f"depth:{model_name}"
    depth_model = multi_model_manager.models.get(model_key)
    if depth_model is None:
        _sp_cfg = (multi_model_manager.config.get(f"spatial:{model_name}")
                   or multi_model_manager.config.get(model_name) or {})
        try:
            depth_model = await asyncio.get_event_loop().run_in_executor(
                None, _load_depth_model, model_name, global_args, _sp_cfg)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load depth model: {e}")
        multi_model_manager.models[model_key] = depth_model
    raw = base64.b64decode(request.image.split(',', 1)[-1] if ',' in request.image else request.image)
    try:
        depth_img = await asyncio.get_event_loop().run_in_executor(
            None, _run_depth, depth_model, raw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Depth estimation failed: {e}")
    result = save_image_response(depth_img, request.response_format, http_request)
    return {"created": int(time.time()), "data": [result]}


# =============================================================================
# Segmentation Endpoint  (POST /v1/images/segment)
# =============================================================================

class ImageSegmentRequest(BaseModel):
    model: Optional[str] = None
    image: str
    points: Optional[list] = None   # [[x,y], ...] positive prompt points for SAM
    boxes: Optional[list] = None    # [[x1,y1,x2,y2], ...] box prompts
    response_format: Optional[str] = "url"
    class Config:
        extra = "allow"


def _load_segmentation_model(model_name: str, global_args, model_config: dict = None):
    device = _derive_diffusers_device(global_args)
    from codai.models.hf_loading import build_from_pretrained_kwargs, pipeline_device_kwargs
    try:
        from transformers import SamModel, SamProcessor
        import torch
        fp = build_from_pretrained_kwargs(model_config)
        model = SamModel.from_pretrained(model_name, **fp)
        # Quantized/offloaded models are already placed; only plain models move.
        if 'quantization_config' not in fp and 'device_map' not in fp:
            model = model.to(device)
        processor = SamProcessor.from_pretrained(model_name)
        return ('sam', (model, processor, device))
    except Exception:
        pass
    try:
        from transformers import pipeline as hf_pipeline
        pk = pipeline_device_kwargs(model_config)
        if 'device_map' in pk:
            pipe = hf_pipeline("image-segmentation", model=model_name, **pk)
        else:
            pipe = hf_pipeline("image-segmentation", model=model_name, device=device, **pk)
        return ('transformers', pipe)
    except Exception as e:
        raise RuntimeError(f"Cannot load segmentation model: {e}")


def _run_segmentation(seg_model, image_bytes: bytes, points, boxes):
    from PIL import Image as PILImage
    import numpy as np, io as _io
    img = PILImage.open(_io.BytesIO(image_bytes)).convert("RGB")
    backend, model_data = seg_model

    if backend == 'sam':
        import torch
        sam_model, processor, device = model_data
        input_points = [points] if points else None
        input_boxes = [boxes] if boxes else None
        inputs = processor(img, input_points=input_points,
                           input_boxes=input_boxes, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = sam_model(**inputs)
        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs['original_sizes'].cpu(),
            inputs['reshaped_input_sizes'].cpu()
        )[0]
        # Take best mask, overlay on image
        mask_np = masks[0, 0].numpy().astype(np.uint8) * 255
        overlay = np.array(img.copy())
        overlay[mask_np == 0] = overlay[mask_np == 0] // 2
        return PILImage.fromarray(overlay)

    else:  # transformers generic
        results = model_data(img)
        # Draw first segment mask
        out = np.array(img)
        if results:
            mask = np.array(results[0]['mask'])
            out[mask == 0] = out[mask == 0] // 2
        return PILImage.fromarray(out)


@router.post("/v1/images/segment", summary="Segment an image")
async def create_image_segment(request: ImageSegmentRequest, http_request: Request = None):
    """Segment objects in an image using SAM or similar models."""
    global global_args
    model_name = _resolve_spatial_model(request.model, "image_segmentation")
    if not model_name:
        raise HTTPException(status_code=400, detail="No segmentation model configured. Add one via Admin > Models.")
    model_key = f"segment:{model_name}"
    seg_model = multi_model_manager.models.get(model_key)
    if seg_model is None:
        _sp_cfg = (multi_model_manager.config.get(f"spatial:{model_name}")
                   or multi_model_manager.config.get(model_name) or {})
        try:
            seg_model = await asyncio.get_event_loop().run_in_executor(
                None, _load_segmentation_model, model_name, global_args, _sp_cfg)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load segmentation model: {e}")
        multi_model_manager.models[model_key] = seg_model
    raw = base64.b64decode(request.image.split(',', 1)[-1] if ',' in request.image else request.image)
    try:
        seg_img = await asyncio.get_event_loop().run_in_executor(
            None, _run_segmentation, seg_model, raw,
            request.points, request.boxes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {e}")
    result = save_image_response(seg_img, request.response_format, http_request)
    return {"created": int(time.time()), "data": [result]}


# =============================================================================
# Deblur Endpoint  (POST /v1/images/deblur)
# =============================================================================

class ImageDeblurRequest(BaseModel):
    image: str                              # base64 input image
    strength: Optional[float] = 0.5        # 0–1, deblur aggressiveness
    response_format: Optional[str] = "url"
    model_config = ConfigDict(extra="allow")


def _run_deblur(image_bytes: bytes, strength: float) -> "PILImage.Image":
    """Blind deblur using Wiener deconvolution + sharpening."""
    import numpy as np
    import cv2
    from scipy.signal import wiener
    from PIL import Image as PILImage

    img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0

    # Wiener filter per channel
    noise_power = max(0.001, (1.0 - strength) * 0.05)
    deblurred = np.stack([
        wiener(arr[:, :, c], mysize=5, noise=noise_power)
        for c in range(3)
    ], axis=2)
    deblurred = np.clip(deblurred, 0.0, 1.0)

    # Unsharp mask pass for edge recovery
    blur_sigma = max(0.5, (1.0 - strength) * 2.0)
    blurred = cv2.GaussianBlur(deblurred, (0, 0), blur_sigma)
    sharpened = cv2.addWeighted(deblurred, 1.0 + strength, blurred, -strength, 0)
    sharpened = np.clip(sharpened, 0.0, 1.0)

    return PILImage.fromarray((sharpened * 255).astype(np.uint8))


@router.post("/v1/images/deblur", summary="Deblur an image")
async def create_image_deblur(request: ImageDeblurRequest, http_request: Request = None):
    """Remove blur from an image using Wiener deconvolution and unsharp masking."""
    raw = base64.b64decode(request.image.split(',', 1)[-1] if ',' in request.image else request.image)
    try:
        result_img = await asyncio.get_event_loop().run_in_executor(
            None, _run_deblur, raw, request.strength or 0.5)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deblur failed: {e}")
    result = save_image_response(result_img, request.response_format, http_request)
    return {"created": int(time.time()), "data": [result]}


# =============================================================================
# Unpixelate Endpoint  (POST /v1/images/unpixelate)
# Uses Real-ESRGAN super-resolution — designed exactly for this use case.
# =============================================================================

class ImageUnpixelateRequest(BaseModel):
    image: str
    scale: Optional[int] = 4               # 2, 4, or 8
    model: Optional[str] = None            # optional custom Real-ESRGAN model path
    response_format: Optional[str] = "url"
    model_config = ConfigDict(extra="allow")


def _run_unpixelate(image_bytes: bytes, scale: int, model_path: Optional[str]) -> "PILImage.Image":
    import numpy as np
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    import torch
    from PIL import Image as PILImage

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_path and os.path.exists(model_path):
        mp = model_path
    else:
        # Download RealESRGAN_x4plus on demand
        from codai.platform_paths import default_realesrgan_model_path
        mp = str(default_realesrgan_model_path())
        if not os.path.exists(mp):
            os.makedirs(os.path.dirname(mp), exist_ok=True)
            import urllib.request
            url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
            print(f'Downloading RealESRGAN_x4plus.pth…')
            urllib.request.urlretrieve(url, mp)

    model_obj = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(scale=4, model_path=mp, model=model_obj,
                              half=device.type == 'cuda', device=device)

    img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
    out_arr, _ = upsampler.enhance(np.array(img), outscale=scale)
    return PILImage.fromarray(out_arr)


@router.post("/v1/images/unpixelate", summary="Restore a pixelated image")
async def create_image_unpixelate(request: ImageUnpixelateRequest, http_request: Request = None):
    """Remove pixelation / upscale with detail recovery using Real-ESRGAN."""
    raw = base64.b64decode(request.image.split(',', 1)[-1] if ',' in request.image else request.image)
    try:
        result_img = await asyncio.get_event_loop().run_in_executor(
            None, _run_unpixelate, raw, request.scale or 4, request.model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unpixelate failed: {e}")
    result = save_image_response(result_img, request.response_format, http_request)
    return {"created": int(time.time()), "data": [result]}


# =============================================================================
# Outfit Change Endpoint  (POST /v1/images/outfit)
# Auto-generates a clothing mask via person segmentation, then inpaints.
# =============================================================================

class ImageOutfitRequest(BaseModel):
    model: str                              # inpaint model id
    image: Optional[str] = None            # base64 source image (image mode)
    video: Optional[str] = None            # base64 source video (video mode)
    prompt: str                             # description of the new outfit
    negative_prompt: Optional[str] = None
    mask: Optional[str] = None             # optional manual mask (base64); auto-generated if absent
    steps: Optional[int] = 30
    guidance_scale: Optional[float] = 7.5
    strength: Optional[float] = 0.99
    seed: Optional[int] = None
    response_format: Optional[str] = "url"
    model_config = ConfigDict(extra="allow")


def _generate_clothing_mask(img_arr) -> "np.ndarray":
    """
    Generate a rough clothing mask using GrabCut person segmentation.
    Returns a binary mask (255 = clothing area to replace).
    """
    import numpy as np
    import cv2
    h, w = img_arr.shape[:2]
    bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)

    # GrabCut with a central rect (assumes person is roughly centered)
    mask_gc = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    margin_x, margin_y = w // 8, h // 8
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)
    cv2.grabCut(bgr, mask_gc, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    fg_mask = np.where((mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    # Exclude top 25% (head/hair) and bottom 10% (feet)
    fg_mask[:h // 4, :] = 0
    fg_mask[int(h * 0.9):, :] = 0

    # Dilate slightly so inpaint covers clothing edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)
    return fg_mask


@router.post("/v1/images/outfit", summary="Change outfit / clothing")
async def create_image_outfit(request: ImageOutfitRequest, http_request: Request = None):
    """Change the outfit/clothing in an image or video using inpainting."""
    global global_args

    if request.video:
        return await _outfit_video(request, http_request)

    raw = base64.b64decode(request.image.split(',', 1)[-1] if ',' in request.image else request.image)
    from PIL import Image as PILImage
    import numpy as np
    img = PILImage.open(io.BytesIO(raw)).convert("RGB")
    img_arr = np.array(img)

    # Generate or decode mask
    if request.mask:
        mask_raw = base64.b64decode(request.mask.split(',', 1)[-1] if ',' in request.mask else request.mask)
        mask_img = PILImage.open(io.BytesIO(mask_raw)).convert("L")
    else:
        try:
            mask_arr = await asyncio.get_event_loop().run_in_executor(
                None, _generate_clothing_mask, img_arr)
            mask_img = PILImage.fromarray(mask_arr)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Mask generation failed: {e}")

    # Load inpaint pipeline
    model_key = f"inpaint:{request.model}"
    pipeline = multi_model_manager.models.get(model_key)
    if pipeline is None:
        try:
            pipeline = await asyncio.get_event_loop().run_in_executor(
                None, _load_inpaint_pipeline, request.model, global_args)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load inpaint model: {e}")
        multi_model_manager.models[model_key] = pipeline

    # Run inpaint
    import torch
    generator = torch.Generator().manual_seed(request.seed) if request.seed is not None else None

    def _run():
        kwargs = dict(
            prompt=request.prompt,
            image=img,
            mask_image=mask_img,
            num_inference_steps=request.steps or 30,
            guidance_scale=request.guidance_scale or 7.5,
            strength=request.strength or 0.99,
        )
        if request.negative_prompt:
            kwargs['negative_prompt'] = request.negative_prompt
        if generator:
            kwargs['generator'] = generator
        if hasattr(pipeline, 'safety_checker'):
            pipeline.safety_checker = None
        return pipeline(**kwargs).images[0]

    try:
        result_img = await asyncio.get_event_loop().run_in_executor(None, _run)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Outfit change failed: {e}")

    result = save_image_response(result_img, request.response_format, http_request)
    return {"created": int(time.time()), "data": [result]}


async def _outfit_video(request: ImageOutfitRequest, http_request):
    """Process outfit change frame-by-frame on a video."""
    import subprocess
    import tempfile
    import shutil

    raw = base64.b64decode(request.video.split(',', 1)[-1] if ',' in request.video else request.video)
    temps = []
    try:
        in_path = tempfile.mktemp(suffix='.mp4')
        temps.append(in_path)
        with open(in_path, 'wb') as f:
            f.write(raw)

        frames_dir = tempfile.mkdtemp()
        temps.append(frames_dir)
        subprocess.run(['ffmpeg', '-y', '-i', in_path, f'{frames_dir}/%08d.png'],
                       capture_output=True, check=True)

        probe = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=r_frame_rate', '-of', 'default=nw=1:nk=1', in_path],
            capture_output=True, text=True)
        fps_str = probe.stdout.strip() or '25/1'
        num, den = fps_str.split('/')
        fps = float(num) / float(den)

        # Load pipeline once
        model_key = f"inpaint:{request.model}"
        pipeline = multi_model_manager.models.get(model_key)
        if pipeline is None:
            pipeline = await asyncio.get_event_loop().run_in_executor(
                None, _load_inpaint_pipeline, request.model, global_args)
            multi_model_manager.models[model_key] = pipeline

        import torch
        from PIL import Image as PILImage
        import numpy as np
        import cv2

        generator = torch.Generator().manual_seed(request.seed) if request.seed is not None else None

        def _process_frames():
            for fname in sorted(os.listdir(frames_dir)):
                fpath = os.path.join(frames_dir, fname)
                img = PILImage.open(fpath).convert("RGB")
                img_arr = np.array(img)
                if request.mask:
                    mask_raw = base64.b64decode(request.mask.split(',', 1)[-1] if ',' in request.mask else request.mask)
                    mask_img = PILImage.open(io.BytesIO(mask_raw)).convert("L")
                else:
                    mask_arr = _generate_clothing_mask(img_arr)
                    mask_img = PILImage.fromarray(mask_arr)
                kwargs = dict(
                    prompt=request.prompt,
                    image=img,
                    mask_image=mask_img,
                    num_inference_steps=request.steps or 30,
                    guidance_scale=request.guidance_scale or 7.5,
                    strength=request.strength or 0.99,
                )
                if request.negative_prompt:
                    kwargs['negative_prompt'] = request.negative_prompt
                if generator:
                    kwargs['generator'] = generator
                if hasattr(pipeline, 'safety_checker'):
                    pipeline.safety_checker = None
                result = pipeline(**kwargs).images[0]
                result.save(fpath)

        await asyncio.get_event_loop().run_in_executor(None, _process_frames)

        out_path = tempfile.mktemp(suffix='_outfit.mp4')
        temps.append(out_path)
        subprocess.run(
            ['ffmpeg', '-y', '-framerate', str(fps), '-i', f'{frames_dir}/%08d.png',
             '-i', in_path, '-map', '0:v', '-map', '1:a?',
             '-c:v', 'libx264', '-c:a', 'copy', '-shortest', out_path],
            capture_output=True, check=True)

        with open(out_path, 'rb') as f:
            out_bytes = f.read()

        if global_file_path:
            fname = f'{uuid.uuid4().hex}_outfit.mp4'
            fpath_out = os.path.join(global_file_path, fname)
            os.makedirs(global_file_path, exist_ok=True)
            with open(fpath_out, 'wb') as f:
                f.write(out_bytes)
            from codai.api.urlutils import build_file_url
            data = [{'url': build_file_url(fname, http_request)}]
        else:
            data = [{'b64_mp4': base64.b64encode(out_bytes).decode()}]

        return {'created': int(time.time()), 'data': data}

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f'ffmpeg error: {e.stderr.decode()[:200]}')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Video outfit change failed: {e}')
    finally:
        for t in temps:
            try:
                if os.path.isdir(t):
                    shutil.rmtree(t)
                else:
                    os.unlink(t)
            except Exception:
                pass
