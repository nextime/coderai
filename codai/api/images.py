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
_gen_progress: dict = {"current": 0, "total": 0, "active": False}

def _progress_reset(total: int):
    _gen_progress["current"] = 0
    _gen_progress["total"] = total
    _gen_progress["active"] = True

def _progress_done():
    _gen_progress["current"] = _gen_progress["total"]
    _gen_progress["active"] = False

def _progress_step(step: int):
    _gen_progress["current"] = step


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
        # Determine base URL based on --url argument
        url_setting = getattr(global_args, 'url', 'auto') if global_args else 'auto'
        if url_setting == 'auto':
            # Use server host from request headers (what client used to connect)
            if http_request:
                # Get the Host header - this is what the client used to reach the server
                # The Host header typically includes the port, e.g., "192.168.1.1:6745"
                client_host = http_request.headers.get('host', '')
                if not client_host:
                    # Fallback to client IP if no Host header
                    client_host = http_request.client.host if http_request.client else '127.0.0.1'
                
                # Strip port from host if present (Host header includes port like "192.168.1.1:6745")
                if ':' in client_host:
                    # Check if the part after : is a port number
                    parts = client_host.split(':')
                    if len(parts) == 2 and parts[1].isdigit():
                        # It's a port number, strip it
                        client_host = parts[0]
                    elif len(parts) > 2:
                        # IPv6 or other format, take last part as port check
                        last_part = parts[-1]
                        if last_part.isdigit():
                            client_host = ':'.join(parts[:-1])
                
                # Check if HTTPS is enabled
                use_https = getattr(global_args, 'https', False) or getattr(global_args, 'pubkey', None)
                protocol = "https" if use_https else "http"
                port = getattr(global_args, 'port', 8000)
                base_url = f"{protocol}://{client_host}:{port}"
            else:
                base_url = "http://127.0.0.1:8000"
        else:
            # Use explicitly provided URL (strip trailing slash if present)
            base_url = url_setting.rstrip('/')
        result["url"] = f"{base_url}/v1/files/{filename}"
        
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


def _load_diffusers_pipeline(model_name: str, global_args):
    """
    Try to load a model using the diffusers library.
    
    Returns the loaded pipeline or None if diffusers can't handle this model.
    Raises Exception if loading fails for other reasons.
    """
    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, DiffusionPipeline
    import torch
    
    # Check for --no-ram mode
    no_ram = getattr(global_args, 'no_ram', False) if global_args else False
    
    # Determine precision from CLI argument (--image-precision)
    precision = getattr(global_args, 'image_precision', 'f32') or 'f32'
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
        print(f"Using precision: {precision} ({dtype})")
    
    # Check if CPU offload is requested via CLI
    use_sequential_offload = getattr(global_args, 'image_cpu_offload', False)
    
    # --no-ram mode: never use CPU offload
    if no_ram and use_sequential_offload:
        print("--no-ram mode: ignoring --image-cpu-offload, forcing full GPU loading")
        use_sequential_offload = False
    
    # =====================================================================
    # --no-ram mode: load directly on GPU, no CPU RAM fallback
    # =====================================================================
    if no_ram and torch.cuda.is_available():
        cuda_device = _derive_diffusers_device(global_args)
        print(f"--no-ram mode: loading diffusers model directly on {cuda_device}")
        
        try:
            try:
                pipeline = StableDiffusionXLPipeline.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    use_safetensors=True,
                )
            except Exception:
                pipeline = DiffusionPipeline.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    use_safetensors=True,
                )
            
            pipeline = pipeline.to(cuda_device)
            print(f"--no-ram: Diffusers model loaded on {cuda_device}")
            return pipeline
        except Exception as e:
            raise RuntimeError(
                f"--no-ram: Failed to load diffusers model entirely on GPU ({cuda_device}). "
                f"The model may be too large for available VRAM. Error: {e}"
            )
    
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
            
            # Try to load as Stable Diffusion XL first, then generic DiffusionPipeline
            try:
                pipeline = StableDiffusionXLPipeline.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    use_safetensors=True,
                )
            except Exception:
                # Try generic diffusion pipeline (supports custom pipelines like ZImagePipeline)
                pipeline = DiffusionPipeline.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    use_safetensors=True,
                )
            
            # Apply memory optimizations based on attempt
            if torch.cuda.is_available():
                if load_attempt >= 2:
                    # Second attempt: enable attention slicing
                    print("Enabling attention slicing for lower VRAM usage...")
                    if hasattr(pipeline, 'enable_attention_slicing'):
                        pipeline.enable_attention_slicing()
                
                if load_attempt >= 3 or use_sequential_offload:
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
                print(f"Retrying with more aggressive memory optimization...")
                pipeline = None  # Reset for retry
            else:
                print(f"Failed to load model (attempt {load_attempt}): {load_error}")
                if load_attempt >= max_attempts:
                    raise
                pipeline = None
    
    return pipeline


async def _generate_with_diffusers(pipeline, request, global_args, http_request=None):
    """Generate images using a diffusers pipeline (with prompt-embedding cache)."""
    import torch
    import numpy as np
    import time as time_module

    if getattr(request, 'disable_safety_checker', False):
        _disable_safety_checker(pipeline)

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
        if hasattr(pipeline, 'enable_vae_slicing'):
            pipeline.enable_vae_slicing()
    except Exception as e:
        print(f"Warning: Could not enable memory optimizations: {e}")

    timestamp = int(time_module.time())

    seed = request.seed if request.seed is not None else getattr(global_args, 'image_seed', None)
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)

    quality = request.quality or "standard"
    num_steps = request.steps if request.steps else (30 if quality == "standard" else 50)
    cfg_scale = request.guidance_scale if request.guidance_scale else (
        getattr(global_args, 'image_cfg_scale', 7.5) if quality == "standard" else 9.0
    )

    _progress_reset(num_steps)

    # ------------------------------------------------------------------
    # Prompt embedding cache
    # Try to encode the prompt once and reuse the embeddings.
    # Falls back to passing the plain text prompt if encoding fails.
    # ------------------------------------------------------------------
    model_id = getattr(pipeline, 'model_name_or_path', None) or str(type(pipeline).__name__)
    neg_prompt = getattr(request, 'negative_prompt', None) or ""
    do_cfg = cfg_scale > 1.0

    cached_embeds = _embed_cache.get(request.prompt, neg_prompt, model_id)
    embed_kwargs = {}
    cache_hit = False

    if cached_embeds is not None:
        embed_kwargs = cached_embeds
        cache_hit = True
        print(f"Prompt embed cache HIT for model '{model_id}'")
    else:
        # Try to encode and cache
        try:
            if hasattr(pipeline, 'encode_prompt'):
                enc = pipeline.encode_prompt(
                    prompt=request.prompt,
                    device=pipeline.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=do_cfg,
                    negative_prompt=neg_prompt or None,
                )
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
                if embed_kwargs:
                    _embed_cache.put(request.prompt, neg_prompt, model_id, embed_kwargs)
                    print(f"Prompt embed cache STORE for model '{model_id}'")
        except Exception as e:
            print(f"Warning: prompt encode/cache failed ({e}), using plain text prompt")
            embed_kwargs = {}

    def _step_cb(pipe, step_index, timestep, callback_kwargs):
        _progress_step(step_index + 1)
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

    # Inject IP-Adapter images if character references provided
    if char_images and hasattr(pipeline, 'set_ip_adapter_scale'):
        try:
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
        except Exception as _ip_err:
            print(f"Warning: IP-Adapter injection failed ({_ip_err}), continuing without character refs")

    try:
        result = await asyncio.to_thread(pipeline, **call_kwargs)
    except TypeError:
        # Older pipeline that doesn't support callback_on_step_end
        call_kwargs.pop('callback_on_step_end', None)
        result = await asyncio.to_thread(pipeline, **call_kwargs)
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

    return {
        "created": timestamp,
        "data": images,
        "prompt_cache_hit": cache_hit,
    }


async def _generate_with_sdcpp(sd_model, request, global_args, http_request=None):
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
    
    # Use default steps for fast generation
    steps = request.steps if request.steps else 4

    _progress_reset(steps)

    def _sdcpp_progress(step: int, total: int, elapsed: float):
        _progress_step(step)

    # Use request seed if provided, otherwise use CLI default seed
    seed = request.seed if request.seed is not None else getattr(global_args, 'image_seed', None)

    try:
        result = await asyncio.to_thread(
            sd_model.generate_image,
            prompt=request.prompt,
            negative_prompt='',
            width=width,
            height=height,
            cfg_scale=get_cfg_scale(),
            sample_steps=steps,
            seed=seed if seed is not None else 42,
            batch_count=request.n if request.n else 1,
            progress_callback=_sdcpp_progress,
        )
    except TypeError:
        result = await asyncio.to_thread(
            sd_model.generate_image,
            prompt=request.prompt,
            negative_prompt='',
            width=width,
            height=height,
            cfg_scale=get_cfg_scale(),
            sample_steps=steps,
            seed=seed if seed is not None else 42,
            batch_count=request.n if request.n else 1,
        )
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


@router.get("/v1/images/progress")
async def get_image_progress():
    """Return current image generation step progress."""
    return {
        "current": _gen_progress["current"],
        "total":   _gen_progress["total"],
        "active":  _gen_progress["active"],
        "pct": int(_gen_progress["current"] / _gen_progress["total"] * 100)
               if _gen_progress["total"] > 0 else 0,
    }


@router.post("/v1/images/generations")
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
        model_info = multi_model_manager.request_model(
            requested_model=request.model,
            model_type="image"
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
                return await _generate_with_sdcpp(pipeline, request, global_args, http_request)
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
        
        # Try diffusers first (for non-GGUF models)
        if not is_gguf:
            try:
                print(f"Loading diffusers model: {model_name}")
                pipeline = _load_diffusers_pipeline(model_name, global_args)
                
                if pipeline is not None:
                    # Cache the loaded pipeline in the manager
                    multi_model_manager.add_model(model_key, pipeline)
                    multi_model_manager.current_model_key = model_key
                    print(f"Loaded diffusers model: {model_name}")
                    
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
                resolved_path = multi_model_manager.load_model(model_name)
                if not resolved_path:
                    raise Exception(f"Failed to resolve model path: {model_name}")
            
            # Only use sd.cpp if we have a local file path
            if resolved_path and os.path.isfile(resolved_path):
                cfg = multi_model_manager.config.get(model_key) or multi_model_manager.config.get(model_name) or {}
                sd_model = _load_sdcpp_model(resolved_path, global_args, model_config=cfg)
                
                if sd_model is not None:
                    # Cache the loaded model in the manager
                    multi_model_manager.add_model(model_key, sd_model)
                    multi_model_manager.current_model_key = model_key
                    print(f"Loaded sd.cpp model: {model_name}")
                    
                    return await _generate_with_sdcpp(sd_model, request, global_args, http_request)
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
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise


@router.post("/v1/images/edits")
async def create_image_edit(request: ImageEditRequest, http_request: Request = None):
    """
    Image-to-image editing endpoint (OpenAI-compatible).
    Accepts a base64-encoded source image and returns an edited image.
    """
    global global_args

    if not request.image:
        raise HTTPException(status_code=400, detail="image is required")

    model_info = multi_model_manager.request_model(request.model, model_type="image")
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
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise


@router.post("/v1/images/inpaint")
async def create_image_inpaint(request: ImageInpaintRequest, http_request: Request = None):
    """Inpaint a masked region of an image (OpenAI-compatible extension)."""
    global global_args
    if not request.image or not request.mask:
        raise HTTPException(status_code=400, detail="image and mask are required")
    model_info = multi_model_manager.request_model(request.model, model_type="image")
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


def _load_upscaler(model_name: str, global_args):
    device = _derive_diffusers_device(global_args)
    n = model_name.lower()
    try:
        from diffusers import StableDiffusionUpscalePipeline
        if 'upscal' in n or 'esrgan' in n:
            pipe = StableDiffusionUpscalePipeline.from_pretrained(model_name)
            return ('diffusers', pipe.to(device))
    except Exception:
        pass
    # Try basicsr / Real-ESRGAN
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        model_obj = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                            num_block=23, num_grow_ch=32, scale=4)
        upsampler = RealESRGANer(scale=4, model_path=model_name,
                                  model=model_obj, device=device)
        return ('realesrgan', upsampler)
    except Exception:
        pass
    # Fallback: PIL LANCZOS
    return ('pil', None)


def _run_upscale(upscaler, image_bytes: bytes, scale: int):
    from PIL import Image as PILImage
    import numpy as np, io as _io
    img = PILImage.open(_io.BytesIO(image_bytes)).convert("RGB")
    backend, model = upscaler
    if backend == 'realesrgan':
        out_arr, _ = model.enhance(np.array(img), outscale=scale)
        return PILImage.fromarray(out_arr)
    if backend == 'diffusers':
        result = model(prompt="", image=img, num_inference_steps=20)
        return result.images[0]
    # PIL fallback
    w, h = img.size
    return img.resize((w * scale, h * scale), PILImage.LANCZOS)


@router.post("/v1/images/upscale")
async def create_image_upscale(request: ImageUpscaleRequest, http_request: Request = None):
    """Upscale an image using Real-ESRGAN or PIL LANCZOS fallback."""
    global global_args
    model_info = multi_model_manager.request_model(request.model, model_type="image")
    model_name = model_info.get('model_name') or request.model
    model_key = f"upscale:{model_name}"
    upscaler = multi_model_manager.models.get(model_key)
    if upscaler is None:
        try:
            upscaler = await asyncio.get_event_loop().run_in_executor(
                None, _load_upscaler, model_name, global_args)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load upscaler: {e}")
        multi_model_manager.models[model_key] = upscaler
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
    model: str
    image: str
    response_format: Optional[str] = "url"
    class Config:
        extra = "allow"


def _load_depth_model(model_name: str, global_args):
    device = _derive_diffusers_device(global_args)
    try:
        from transformers import pipeline as hf_pipeline
        pipe = hf_pipeline("depth-estimation", model=model_name, device=device)
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


@router.post("/v1/images/depth")
async def create_image_depth(request: ImageDepthRequest, http_request: Request = None):
    """Estimate depth map from an image."""
    global global_args
    model_name = request.model
    model_key = f"depth:{model_name}"
    depth_model = multi_model_manager.models.get(model_key)
    if depth_model is None:
        try:
            depth_model = await asyncio.get_event_loop().run_in_executor(
                None, _load_depth_model, model_name, global_args)
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
    model: str
    image: str
    points: Optional[list] = None   # [[x,y], ...] positive prompt points for SAM
    boxes: Optional[list] = None    # [[x1,y1,x2,y2], ...] box prompts
    response_format: Optional[str] = "url"
    class Config:
        extra = "allow"


def _load_segmentation_model(model_name: str, global_args):
    device = _derive_diffusers_device(global_args)
    try:
        from transformers import SamModel, SamProcessor
        import torch
        model = SamModel.from_pretrained(model_name).to(device)
        processor = SamProcessor.from_pretrained(model_name)
        return ('sam', (model, processor, device))
    except Exception:
        pass
    try:
        from transformers import pipeline as hf_pipeline
        pipe = hf_pipeline("image-segmentation", model=model_name, device=device)
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


@router.post("/v1/images/segment")
async def create_image_segment(request: ImageSegmentRequest, http_request: Request = None):
    """Segment objects in an image using SAM or similar models."""
    global global_args
    model_name = request.model
    model_key = f"segment:{model_name}"
    seg_model = multi_model_manager.models.get(model_key)
    if seg_model is None:
        try:
            seg_model = await asyncio.get_event_loop().run_in_executor(
                None, _load_segmentation_model, model_name, global_args)
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


@router.post("/v1/images/deblur")
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
        mp = os.path.expanduser('~/.cache/realesrgan/RealESRGAN_x4plus.pth')
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


@router.post("/v1/images/unpixelate")
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


@router.post("/v1/images/outfit")
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
            host = http_request.headers.get('host', '127.0.0.1') if http_request else '127.0.0.1'
            if ':' in host:
                parts = host.split(':')
                if len(parts) == 2 and parts[1].isdigit():
                    host = parts[0]
            proto = 'https' if getattr(global_args, 'https', False) else 'http'
            port = getattr(global_args, 'port', 8000) if global_args else 8000
            data = [{'url': f'{proto}://{host}:{port}/v1/files/{fname}'}]
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
