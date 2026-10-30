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
import re as _re
import time
import uuid
from typing import List, Optional

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


#: Spellings that mean "give me the bytes inline". ``b64_json`` is the
#: OpenAI-API value and is what every SDK client sends; ``base64`` is this
#: codebase's own older internal name. Both are accepted because the request
#: model documents the first and this function used to test only the second --
#: so a caller asking for base64 the standard way got a URL and no bytes, with
#: nothing to say why. Measured 2026-10-27 against the live server.
_B64_FORMATS = ("base64", "b64_json", "b64")


def _wants_b64(request_format) -> bool:
    return str(request_format or "").strip().lower() in _B64_FORMATS


def save_image_response(img, request_format="base64", http_request=None):
    """
    Save image to file path if configured, return response dict.

    If --file-path is set and request_format is url, return only the URL/path.
    If --file-path is set and base64 was asked for (``b64_json`` or
    ``base64``), return the URL/path AND the bytes.
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
        # The relative path as well, as /v1/compose returns it: a caller that
        # feeds the result into another call should not have to know, or depend
        # on, this install's public hostname.
        result["path"] = f"/v1/files/{filename}"
        
        # If client explicitly requested base64, include it
        # Otherwise, only return URL when file-path is set
        if _wants_b64(request_format):
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


def _refuse_uniform_output(images, model_name: str, dtype=None) -> None:
    """Raise when a pipeline returned a FLAT image -- one single colour.

    A black PNG answered 200 OK is the worst shape a failure can take: the
    client gets a plausible file, the log shows a success, and the only way
    anyone learns is a person looking at the picture. It was reported from the
    field exactly that way: an image-edit request that answered with a black
    picture and no error.

    Its cause is almost always numeric. fp16 tops out near 65504, a
    flow-matching DiT runs past it, the latents become NaN, and a VAE decodes
    NaN to uniform black. So the message names the dtype, which is the thing
    worth changing.

    A real picture is never one colour: even a photograph of a dark room has
    sensor noise, and a diffusion sample has more. A uniform result is
    therefore evidence of arithmetic, not of content -- and the check is on
    the extremes only (0 or 255), so a deliberately flat mid-grey render is
    left alone.
    """
    from PIL import Image as _Image
    for idx, img in enumerate(images or []):
        if not isinstance(img, _Image.Image):
            continue
        try:
            extrema = img.convert("RGB").getextrema()
        except Exception:
            continue                      # cannot judge it: let it through
        flat = all(lo == hi for lo, hi in extrema)
        if not flat:
            continue
        value = tuple(lo for lo, _hi in extrema)
        if value not in ((0, 0, 0), (255, 255, 255)):
            continue                      # flat, but not an overflow signature
        where = "black" if value == (0, 0, 0) else "white"
        note = ""
        if dtype is not None and "float16" in str(dtype):
            note = (" The model was run in fp16, whose range a diffusion "
                    "transformer can exceed; set this model's precision to "
                    "bf16 (same VRAM, far wider range).")
        raise RuntimeError(
            f"{model_name!r} returned a uniformly {where} image "
            f"(image {idx + 1} of {len(images)}), which is what NaN latents "
            f"decode to rather than a picture.{note}")


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


#: Sampler name → diffusers scheduler class. The names are the ones model
#: cards and UIs use; several map to the same class because the community
#: spells them differently.
SAMPLERS = {
    "euler": "EulerDiscreteScheduler",
    "euler_a": "EulerAncestralDiscreteScheduler",
    "euler_ancestral": "EulerAncestralDiscreteScheduler",
    "heun": "HeunDiscreteScheduler",
    "dpm": "DPMSolverMultistepScheduler",
    "dpmpp_2m": "DPMSolverMultistepScheduler",
    "dpm_2m": "DPMSolverMultistepScheduler",
    "dpmpp_sde": "DPMSolverSDEScheduler",
    "dpmpp_2s": "DPMSolverSinglestepScheduler",
    "ddim": "DDIMScheduler",
    "ddpm": "DDPMScheduler",
    "pndm": "PNDMScheduler",
    "lms": "LMSDiscreteScheduler",
    "deis": "DEISMultistepScheduler",
    "unipc": "UniPCMultistepScheduler",
    "lcm": "LCMScheduler",
    "kdpm2": "KDPM2DiscreteScheduler",
    "kdpm2_a": "KDPM2AncestralDiscreteScheduler",
    # Flow-matching models (FLUX, SD3, Z-Image, LTX) have their own family;
    # handing one a DDIM would produce noise, so they are named separately and
    # the swap below refuses to cross between the families.
    "flow_euler": "FlowMatchEulerDiscreteScheduler",
    "flow_heun": "FlowMatchHeunDiscreteScheduler",
    "flow_lcm": "FlowMatchLCMScheduler",
}

#: Schedulers for flow-matching models. A pipeline whose scheduler is one of
#: these cannot be given a classic DDPM-family sampler, and vice versa.
#: Shared with the model-card path, which applies a scheduler from the
#: acceleration config and must draw the family line in the same place.
from codai.models.acceleration import FLOW_SCHEDULERS as _FLOW_SCHEDULERS


def _apply_sampler(pipeline, sampler: str) -> str:
    """Swap the pipeline's scheduler for the named sampler.

    Refuses to cross between the flow-matching and DDPM families. They
    parameterise the trajectory differently, so giving a FLUX-family pipeline
    a DDIM does not produce a worse picture -- it produces noise -- and a
    silent swap would look like the model being broken.

    Raises ``ValueError`` with the valid names; the caller decides whether
    that is a 400 (the request asked) or a warning (a model card asked).
    """
    name = str(sampler or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not name:
        return ""
    cls_name = SAMPLERS.get(name)
    if not cls_name:
        raise ValueError(
            f"unknown sampler {sampler!r}; expected one of "
            f"{', '.join(sorted(SAMPLERS))}")
    import diffusers
    cls = getattr(diffusers, cls_name, None)
    if cls is None:
        raise ValueError(f"{cls_name} is not available in this diffusers")
    current = type(getattr(pipeline, "scheduler", None)).__name__
    if (current in _FLOW_SCHEDULERS) != (cls_name in _FLOW_SCHEDULERS):
        raise ValueError(
            f"{sampler!r} ({cls_name}) is not compatible with this model, "
            f"whose scheduler is {current}: flow-matching and DDPM-family "
            f"samplers parameterise the trajectory differently and cannot be "
            f"substituted for one another")
    if current == cls_name:
        return cls_name
    pipeline.scheduler = cls.from_config(pipeline.scheduler.config)
    _log.info("sampler: %s -> %s", current, cls_name)
    return cls_name


def card_view(model_config) -> dict:
    """The model's card as one dict: its models.json entry under the runtime kwargs.

    The manager keeps runtime kwargs FLAT and the original models.json entry
    under ``_raw_cfg``, so a key that was never promoted to a top-level kwarg
    is invisible to anything reading the config directly. ``defaults`` is one
    of those, and reading it from the flat dict is how a card's steps and
    guidance were silently ignored on the image path: ``/v1/models`` published
    them (it reads the entry) while generation used the generic 30 at CFG 7.5.
    The video path had already hit this and merged by hand at its call site;
    doing it here fixes every caller, and is the same lesson as the
    ``gpu_split`` and ``sequential_text_encoder`` promotions in
    ``build_runtime_kwargs``.

    The flat kwargs win on a collision: they are what the loader was actually
    given.
    """
    cfg = model_config or {}
    raw = cfg.get('_raw_cfg') if isinstance(cfg.get('_raw_cfg'), dict) else None
    if not raw:
        return dict(cfg) if isinstance(cfg, dict) else {}
    merged = dict(raw)
    merged.update({k: v for k, v in cfg.items() if k != '_raw_cfg'})
    return merged


def model_defaults(model_config) -> dict:
    """A model card's own recommended settings, from its ``defaults`` entry.

    Distinct from ``call_kwargs`` in one way that matters: these are DEFAULTS.
    They fill in what the caller left out and never override what the caller
    sent, while ``call_kwargs`` applies on every call because the card says the
    model does not work without it.

        "defaults": {"steps": 8, "guidance_scale": 1.0, "sampler": "euler",
                     "modes": ["t2v", "i2v"], "license": "..."}

    Why it exists: a distilled model wants 8 steps at CFG 1 and a
    non-distilled sibling wants 30 at CFG 4, and there is nowhere else for
    that to live. Without it every client maintains its own table of
    per-model settings -- keyed on the model NAME, which breaks the moment a
    model is renamed or a second quantization of it is added -- and a caller
    who has not built that table gets 30 steps from an 8-step model and pays
    four times over for a worse picture.

    Published in ``/v1/models`` as ``defaults`` so a client can show them
    rather than guess them.
    """
    try:
        d = dict(card_view(model_config).get("defaults") or {})
    except Exception:
        return {}
    return d


def _model_call_kwargs(pipeline, model_config) -> dict:
    """Extra ``__call__`` arguments this model's card specifies.

    Some published models document a setting that is neither steps, guidance
    nor a scheduler option, and a client has no field to send it in: MOSS-Image
    asks for ``uncond_skip_layers=[22]`` on every call. Without a passthrough
    the only way to honour a card like that is to hardcode the model, so
    ``call_kwargs`` on the model entry carries them::

        "call_kwargs": {"uncond_skip_layers": [22], "shift": 5}

    Filtered against the pipeline's own signature, like every other kwarg here:
    a diffusers ``__call__`` ends in ``**kwargs``, so an argument it does not
    understand is accepted and silently dropped, and a setting that does
    nothing is worse than one that is refused. What is dropped is logged.
    """
    extra = {}
    try:
        # Through the card view, for the same reason as model_defaults: a key
        # that was never promoted to a top-level runtime kwarg lives only in
        # `_raw_cfg`, and `call_kwargs` is one of those -- MOSS-Image's
        # `uncond_skip_layers` would never have been applied.
        extra = dict(card_view(model_config).get("call_kwargs") or {})
    except Exception:
        return {}
    if not extra:
        return {}
    try:
        import inspect
        params = inspect.signature(pipeline.__call__).parameters
    except (TypeError, ValueError):
        params = {}
    if params and not any(p.kind is p.VAR_KEYWORD for p in params.values()):
        pass            # a strict signature: filter below is enough
    if params:
        kept = {k: v for k, v in extra.items() if k in params}
        dropped = sorted(set(extra) - set(kept))
        if dropped:
            _log.warning("%s does not take %s from this model's call_kwargs; "
                         "they were NOT applied", type(pipeline).__name__,
                         ", ".join(dropped))
        return kept
    return extra


def _apply_component_dtypes(pipeline, model_config):
    """Cast named pipeline components to their configured dtype.

    A pipeline loads at one dtype, and some models need a component at
    another: MOSS-Image's card asks for the **VAE in fp32** while the
    transformer and text encoder run bf16, because its latents overflow a half
    type in the decoder -- the same arithmetic that turns a picture black.

        "component_dtype": {"vae": "f32"}

    Applied after the load, so it costs one cast rather than a second copy of
    the weights. A component that is quantized is left alone and said so: a
    bnb 4-bit module cannot be cast, and silently not casting it would be the
    quiet kind of wrong.
    """
    import torch
    want = {}
    try:
        want = dict((model_config or {}).get("component_dtype") or {})
    except Exception:
        return pipeline
    if not want:
        return pipeline
    table = {"f32": torch.float32, "fp32": torch.float32,
             "f16": torch.float16, "fp16": torch.float16,
             "bf16": torch.bfloat16}
    for name, spec in want.items():
        dt = table.get(str(spec or "").strip().lower())
        comp = getattr(pipeline, name, None)
        if dt is None:
            _log.warning("component_dtype %r for %r is not a dtype I know",
                         spec, name)
            continue
        if comp is None:
            _log.warning("component_dtype: this pipeline has no %r", name)
            continue
        if getattr(comp, "is_quantized", False) or getattr(
                comp, "quantization_method", None):
            _log.warning("component_dtype: %r is quantized, leaving it as it "
                         "is (a quantized module cannot be cast)", name)
            continue
        try:
            comp.to(dt)
            print(f"  component '{name}' cast to {spec}")
        except Exception as e:
            _log.warning("could not cast %r to %s: %s", name, spec, e)
    return pipeline


def _finish_image_load(pipeline, model_config):
    """Apply everything the model's own config adds to a loaded pipeline.

    Acceleration first, then the per-model LoRA stack — in that order because
    the LoRA step activates the union of both adapter groups, so it needs the
    acceleration adapters to already be registered.

    A failing acceleration LoRA costs speed and is swallowed; a failing
    per-model LoRA costs identity and is raised, which fails the load rather
    than quietly serving the base checkpoint under the fine-tune's name.
    """
    pipeline = _apply_image_acceleration(pipeline, model_config)
    _apply_component_dtypes(pipeline, model_config)
    from codai.models.diffusion_loras import apply_model_loras
    apply_model_loras(pipeline, model_config)
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

    # Wait if the GPU is reserved for an exclusive op (LoRA training here or on a
    # co-located sibling) so this load doesn't contend with it.
    try:
        from codai.models import gpu_lock
        gpu_lock.wait_until_free(context=f"image load '{model_name}'")
    except Exception:
        pass

    _mc = model_config or {}

    # The id the CALLER asked for, kept before `model_name` is rewritten to the
    # checkpoint path below. Sizing needs it: two cards can share one
    # checkpoint with different quantization and offload, and the one that was
    # asked for is the one whose footprint applies.
    _requested_id = model_name

    # A multi-config sibling is registered under its alias, so `model_name` may
    # BE that alias rather than something from_pretrained can open. The config
    # carries the checkpoint it is a config of; prefer it when present. For a
    # primary entry the two are the same string.
    _cfg_path = str(_mc.get('path') or '').strip()
    if _cfg_path and _cfg_path != model_name:
        print(f"Image model '{model_name}' is a config of '{_cfg_path}'")
        model_name = _cfg_path

    def _cfg(key, default=None):
        """Read a value from the per-model configuration only (source of truth)."""
        v = _mc.get(key)
        return v if v is not None else default

    # All loading parameters come from the per-model configuration.
    no_ram = bool(_cfg('no_ram', False))
    # A card that does not state a precision gets a HALF one, not f32.
    #
    # f32 was the default here, and only --no-ram mode coerced it down, so
    # whether a model loaded at 2 bytes or 4 per parameter came down to an
    # unrelated flag. Measured on one production run: CyberRealisticPony and
    # wai-illustrious (no_ram=True) loaded at 7.49 GB, while PonyRealism and
    # pornmaster-pro -- the same SDXL architecture, no_ram=False -- measured
    # 14.98 GB, exactly double, filled the card and failed with "CUDA out of
    # memory. Tried to allocate 12.00 MiB". The inflation then spread to the
    # estimate: the f32 precision factor turned that 14.98 GB into a 30.9 GB
    # need, which triggered a round of eviction that could not possibly help
    # ("this model needs more than the whole card").
    #
    # Nothing was gained for it. These checkpoints are published in fp16/bf16,
    # so f32 doubles the weights by upcasting what was never computed at that
    # width. bf16 rather than fp16 because it keeps f32's range: a flow-matching
    # DiT run in fp16 goes past 65504, the latents go NaN, and the VAE decodes
    # NaN to uniform black -- answered 200 OK with a plausible PNG (the same
    # reasoning as the --no-ram branch below).
    #
    # An explicit 'f32' on the card is still honoured: this changes the default,
    # not the operator's choice.
    _precision_cfg = _cfg('precision')
    precision = str(_precision_cfg or '').strip() or 'bf16'
    if not _precision_cfg:
        print(f"Using precision: {precision} (card states none; f32 would "
              f"double the weights of a checkpoint published at half width)")
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
        # ONLY FROM f32. fp16 and bf16 are both two bytes per parameter, so
        # coercing a configured bf16 to fp16 saves nothing at all -- it only
        # narrows the dynamic range, and it silently overrode a precision the
        # operator had chosen for the model.
        #
        # Measured 2026-10-25: Z-Image-Turbo, configured bf16 and loaded fp16
        # by this line, returned a BLACK image for every edit. fp16 tops out
        # near 65504 and a flow-matching DiT runs past it, so the latents go
        # NaN and the VAE decodes NaN to uniform black -- answered 200 OK, with
        # a plausible-looking PNG and nothing in the log.
        if dtype is torch.float32:
            dtype = torch.float16
            print("--no-ram mode: precision was f32 -> using fp16 "
                  "(half the VRAM, and FlashAttention needs a half type)")
        else:
            print(f"--no-ram mode: keeping the configured precision "
                  f"{precision} (fp16 would be the same size and a narrower "
                  f"range)")
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
        build_pipeline_quant_config, build_gguf_pipeline_components,
        build_component_models, build_single_file_pipeline)
    _img_quant_config, _img_quant_desc = build_pipeline_quant_config(model_name, _mc, dtype)
    if _img_quant_config is not None:
        print(f"Image quantization: {_img_quant_desc}")
    _img_gguf_components, _img_gguf_desc = build_gguf_pipeline_components(model_name, _mc, dtype)
    if _img_gguf_components:
        print(f"Image GGUF components: {_img_gguf_desc}")
    # A component taken from another repo (a fine-tuned transformer served on
    # the base pipeline's VAE and text encoder). Merged after the GGUF ones so
    # an explicit override wins over a GGUF file for the same component.
    # A pipeline that ships in its own repository (MOSS, Kromoss). Resolved
    # here so a refusal happens BEFORE any weights move, and so the ordinary
    # repo load is untouched when no model asks for it.
    _img_remote = {}
    try:
        from codai.models import remote_code
        from codai.admin.routes import config_manager as _cm
        _img_remote = remote_code.kwargs_for(
            getattr(_cm, "config", None), model_name, _cfg('custom_pipeline'))
        if not _img_remote:
            # No card named a custom_pipeline — ask the repository itself. A
            # checkpoint whose model_index.json declares a class diffusers does
            # not have is telling us it ships its own, and letting that load
            # proceed produced "module diffusers has no attribute
            # MOSSImagePipeline": a true statement that names neither the cause
            # nor the fix, and reads as a broken model.
            _img_remote = remote_code.kwargs_for_declared_class(
                getattr(_cm, "config", None), model_name)
        if _img_remote:
            _declared = _img_remote.pop("_declared_class", None)
            print(f"  Custom pipeline from the repository: "
                  f"{_img_remote['custom_pipeline']} (allow-listed)"
                  + (f" — it declares {_declared}, which is not in diffusers"
                     if _declared else ""))
    except PermissionError:
        raise
    except Exception as e:
        print(f"  custom_pipeline check skipped: {e}")

    _img_comp_models, _img_comp_desc = build_component_models(model_name, _mc, dtype)
    if _img_comp_models:
        print(f"Image components from other repos: {_img_comp_desc}")
        _img_gguf_components = {**_img_gguf_components, **_img_comp_models}
    
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
            _xtra.update(_img_remote)
            with _DIFFUSERS_LOAD_LOCK:
                pipeline = build_single_file_pipeline(model_name, _mc, dtype, _xtra)
                if pipeline is None:
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
            return _finish_image_load(pipeline, _mc)
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
                # requested_id: same-path siblings (one checkpoint, two cards
                # with different settings) must be sized by the card that was
                # ASKED for, not by whichever one registered the path.
                # `request` does not exist in this function. Reading it raised
                # NameError on EVERY image load, and the except below swallowed
                # it as "Proactive eviction skipped: name 'request' is not
                # defined" -- so the whole minimum-eviction block never ran and
                # every image load fell through to the blunt eviction further
                # down, which unloads a model that would have fitted alongside.
                _need_gb = _mmm._get_model_used_vram_gb(
                    _key or model_name, model_name,
                    requested_id=_requested_id)
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
    # Host-RAM admission
    # =====================================================================
    # VRAM is handled above by eviction; host RAM has to be handled by refusal,
    # because the kernel's answer to running out of it is to kill the engine. The
    # image path is where it bit hardest: the cards that quantize only their text
    # encoder stage a bf16 transformer into host RAM, and on a full card nearly
    # all of it spills. Raised, not caught — the caller turns it into a 503 that
    # names the model, the size and the fix. See codai/models/host_ram.py.
    try:
        from codai.models.manager import multi_model_manager as _mmm
    except Exception:
        _mmm = None
    if _mmm is not None:
        _hkey = None
        for _k in (f"image:{model_name}", model_name):
            if _k in _mmm.config:
                _hkey = _k
                break
        _mmm.ensure_host_ram_for(_hkey or f"image:{model_name}", model_name,
                                 _requested_id)

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
                _xtra.update(_img_remote)
                # A single ComfyUI/A1111-style checkpoint file, when that is
                # what the model's path names; None otherwise, which leaves the
                # ordinary repo load below untouched.
                pipeline = build_single_file_pipeline(model_name, _mc, dtype, _xtra)
                # Try to load as Stable Diffusion XL first, then generic DiffusionPipeline
                if pipeline is None and _img_remote:
                    # A model whose pipeline class lives in its own repository
                    # is only loadable through DiffusionPipeline: trying SDXL
                    # first would fail on every such model and bury the real
                    # error under a class mismatch.
                    pipeline = DiffusionPipeline.from_pretrained(
                        model_name,
                        torch_dtype=dtype,
                        use_safetensors=True,
                        **_xtra,
                    )
                elif pipeline is None:
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

                if use_sequential_offload or load_attempt >= 3:
                    # Offload was asked for (or we are on the last-ditch attempt).
                    # This is checked BEFORE the quantized branch on purpose: a
                    # model can need both, and it used to be impossible to have
                    # them. Quantizing the text encoder set _img_quant_config,
                    # which took the .to("cuda") branch below and discarded
                    # offload_strategy without a word -- so a pipeline that fits
                    # the card only in pieces was loaded whole and OOMed on its
                    # activations instead. Measured: Qwen-Image-Edit-2511 with a
                    # Q4_K_M transformer and a 4-bit text encoder is 22.68 GiB
                    # resident on a 24 GB card, which leaves 1.32 GiB and dies
                    # needing 390 MiB.
                    #
                    # "model" and "sequential" are also no longer the same thing.
                    # Model offload keeps the component in USE on the GPU, so
                    # peak is the transformer plus activations and a text encoder
                    # pays one PCIe trip per request. Sequential offload walks
                    # submodules: far less VRAM, far slower. The admin UI has
                    # described them as different for a long time; the image path
                    # ran both as sequential.
                    _strategy = (_offload_strategy or '').strip().lower()
                    if _strategy == 'model' and hasattr(pipeline, 'enable_model_cpu_offload'):
                        print("Enabling MODEL CPU offload (active component on GPU)...")
                        # A fully 4-bit module has no floating-point parameters, so
                        # transformers' .dtype property raises StopIteration and diffusers
                        # reads it unguarded in to(). In a worker thread that becomes
                        # "StopIteration interacts badly with generators" and fails the
                        # load. Idempotent and cached after the first call.
                        from codai.models.peft_compat import ensure_quantized_dtype_compat
                        ensure_quantized_dtype_compat()
                        pipeline.enable_model_cpu_offload()
                    elif hasattr(pipeline, 'enable_sequential_cpu_offload'):
                        print("Enabling sequential CPU offload for lower VRAM usage...")
                        pipeline.enable_sequential_cpu_offload()
                    else:
                        print("Pipeline supports no CPU offload — loading on GPU")
                        pipeline = pipeline.to("cuda")
                elif _img_quant_config is not None:
                    # Quantized (bitsandbytes) pipelines are already placed on GPU
                    # by from_pretrained and cannot be moved with .to(); only the
                    # non-quantized components need an explicit device move.
                    print("Quantized pipeline — placing non-quantized components on GPU")
                    try:
                        pipeline = pipeline.to("cuda")
                    except Exception:
                        pass  # bitsandbytes components stay where loaded
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

    return _finish_image_load(pipeline, _mc)


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


def ip_adapter_image_arg(pipeline, images):
    """Shape the reference images the way diffusers expects for `ip_adapter_image`.

    diffusers matches the OUTER list against the number of loaded IP-Adapters, not the
    number of reference images. Passing a character profile's four references as a flat
    list therefore raised

      `ip_adapter_image` must have same length as the number of IP Adapters.
      Got 4 images and 1 IP Adapters.

    Several images conditioning ONE adapter have to be nested one level deeper.
    """
    try:
        n_adapters = len(pipeline.unet.encoder_hid_proj.image_projection_layers)
    except Exception:
        n_adapters = 1
    if n_adapters <= 1:
        return images[0] if len(images) == 1 else [images]
    # One (possibly multi-image) entry per loaded adapter.
    if len(images) == n_adapters:
        return images
    return [images] * n_adapters


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

    # Attention slicing and IP-Adapter are mutually exclusive in diffusers: the
    # adapter loader rebuilds every attention processor, and re-instantiating a
    # SlicedAttnProcessor without its required slice_size raises
    #   SlicedAttnProcessor.__init__() missing 1 required positional argument
    # which is what silently killed every IP-Adapter load (we enable slicing for
    # VRAM headroom). Turn it off for the load and leave it off while the adapter is
    # in use — re-enabling would swap the IP-Adapter processors straight back out.
    sliced = False
    try:
        if getattr(pipeline, '_attn_slicing_enabled', True) and \
                hasattr(pipeline, 'disable_attention_slicing'):
            pipeline.disable_attention_slicing()
            sliced = True
    except Exception:
        pass

    # Second blocker, on a 4-bit model: diffusers' _load_ip_adapter_weights() finishes
    # with `self.to(dtype=self.dtype)` on the denoiser. That is a no-op — it casts to
    # the dtype the module already has — but bitsandbytes refuses *any* dtype argument
    # on a quantized model ("Casting a quantized model to a new `dtype` is
    # unsupported"), so the whole load aborted. Make that specific no-op tolerable for
    # the duration of the load, honouring a device move if one was also requested.
    def _tolerate_noop_dtype_cast(module):
        original = module.to

        def _to(*args, **kwargs):
            try:
                return original(*args, **kwargs)
            except (ValueError, TypeError, NotImplementedError) as exc:
                if 'quantized model' not in str(exc).lower():
                    raise
                import torch as _t
                device = kwargs.get('device')
                if device is None:
                    for a in args:
                        if isinstance(a, (str, _t.device)):
                            device = a
                            break
                _log.debug("IP-Adapter: ignoring same-dtype cast on quantized %s",
                           type(module).__name__)
                return original(device) if device is not None else module

        module.to = _to
        return original

    patched = []
    for attr in ('unet', 'transformer'):
        mod = getattr(pipeline, attr, None)
        if mod is not None and hasattr(mod, 'to'):
            try:
                patched.append((mod, _tolerate_noop_dtype_cast(mod)))
            except Exception:
                pass

    try:
        ok = _do_load_ip_adapter(pipeline, attempts, sliced, _log)
    finally:
        for mod, original in patched:
            try:
                mod.to = original
            except Exception:
                pass
    if ok and patched:
        # The cast we just suppressed was diffusers' way of bringing the NEWLY added
        # IP-Adapter modules up to the model dtype. Skipping it wholesale left them in
        # the checkpoint's fp16 against a bf16 pipeline, which fails at the first matmul
        # ("expected mat1 and mat2 to have the same dtype: BFloat16 != Half"). So do the
        # cast ourselves, on the non-quantized modules only.
        _align_ip_adapter_dtype(pipeline, _log)
    return ok


def _align_ip_adapter_dtype(pipeline, _log) -> None:
    """Cast the IP-Adapter modules to the pipeline's real compute dtype."""
    dtype = None
    for attr in ('text_encoder', 'text_encoder_2', 'vae'):
        module = getattr(pipeline, attr, None)
        if module is None:
            continue
        try:
            dtype = next(module.parameters()).dtype
            break
        except (StopIteration, AttributeError):
            continue
    if dtype is None:
        return

    denoiser = getattr(pipeline, 'unet', None) or getattr(pipeline, 'transformer', None)
    targets = []
    if getattr(denoiser, 'encoder_hid_proj', None) is not None:
        targets.append(denoiser.encoder_hid_proj)
    if getattr(pipeline, 'image_encoder', None) is not None:
        targets.append(pipeline.image_encoder)
    try:                       # IPAdapterAttnProcessor holds to_k_ip / to_v_ip Linears
        targets.extend(p for p in denoiser.attn_processors.values() if hasattr(p, 'to'))
    except Exception:
        pass

    cast = 0
    for target in targets:
        try:
            target.to(dtype=dtype)
            cast += 1
        except Exception:
            pass
    _log.info("IP-Adapter: aligned %d module(s) to %s", cast, dtype)


def _do_load_ip_adapter(pipeline, attempts, sliced, _log) -> bool:
    for repo, subfolder, weight_name in attempts:
        try:
            pipeline.load_ip_adapter(repo, subfolder=subfolder, weight_name=weight_name)
            pipeline._coderai_ip_state = 'loaded'
            _log.info("IP-Adapter loaded: %s/%s/%s%s", repo, subfolder, weight_name,
                      " (attention slicing disabled for it)" if sliced else "")
            return True
        except Exception as e:
            _log.warning("IP-Adapter load failed (%s/%s): %s", repo, weight_name, e)
            _log.debug("IP-Adapter load traceback", exc_info=True)

    if sliced:      # restore the previous memory setting — the adapter isn't loaded
        try:
            pipeline.enable_attention_slicing()
        except Exception:
            pass

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
            if not loras:
                # This request carries no LoRA of its own and the previous
                # one's adapters are gone. Re-activate the pipeline's own
                # adapters explicitly: deleting the active set must not leave
                # the model running without the LoRAs that define it.
                from codai.models.diffusion_loras import persistent_adapters
                keep_names, keep_weights = persistent_adapters(pipeline)
                if keep_names:
                    try:
                        pipeline.set_adapters(keep_names, keep_weights)
                    except Exception as e:
                        _log.warning("could not re-activate %s: %s", keep_names, e)
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
            # set_adapters replaces the ACTIVE SET, so the pipeline's own
            # adapters have to be named again here or this request silently
            # switches them off: the acceleration LoRA (the model stops being
            # distilled, and 4 steps on an un-distilled model is mush) and the
            # per-model LoRA (the id stops being the fine-tune it is published
            # as). The video path has always re-included them; the image path
            # did not.
            from codai.models.diffusion_loras import persistent_adapters
            keep_names, keep_weights = persistent_adapters(pipeline)
            pipeline.set_adapters(names + keep_names, weights + keep_weights)
            pipeline._coderai_request_adapters = list(names)
            _log.info("LoRA weights applied: %s%s", names,
                      f" (kept {keep_names})" if keep_names else "")
    except Exception as e:
        _log.warning("Could not apply LoRA weights: %s", e)


#: Diffusers' VAE downsamples by 8, so a side that is not a multiple of 8
#: cannot be expressed as a latent: the pipeline either raises or renders a
#: ragged edge, depending on the model.
_SIZE_MULTIPLE = 8

_SIZE_RE = _re.compile(r"^\s*(\d+)\s*[x\u00d7*]\s*(\d+)\s*$", _re.I)


def _size_limits(args=None):
    """(max_side, max_pixels) this install allows, from config."""
    a = args or global_args
    side = getattr(a, 'image_max_side', None) or 4096
    pixels = getattr(a, 'image_max_pixels', None) or 4096 * 1024
    return int(side), int(pixels)


def _parse_size(size, default_w: int, default_h: int, args=None):
    """``"WIDTHxHEIGHT"`` as a usable (width, height), or HTTP 400.

    Three things used to go wrong here, all of them quietly. A size that did
    not parse fell through to the default, so ``"1024"`` or ``"1k x 1k"``
    generated a square nobody asked for and the response looked successful. A
    side that was not a multiple of 8 went to the pipeline as-is, where the
    VAE cannot represent it. And there was no ceiling at all, so ``4096x4096``
    was accepted and the request came back as a CUDA OOM -- which reads like a
    busy GPU rather than a size the caller chose.

    So: a malformed size is a 400 that says what was wrong, an oversized one
    is a 400 that names the limit, and a side off the grid is snapped to the
    nearest multiple of 8 and logged. ``"auto"`` means the default, as it does
    in the OpenAI image API.
    """
    if not size or str(size).strip().lower() == "auto":
        return default_w, default_h

    m = _SIZE_RE.match(str(size))
    if not m:
        raise HTTPException(status_code=400, detail=(
            f"size {size!r} is not understood: expected \"WIDTHxHEIGHT\" in "
            f"pixels (e.g. \"1024x1024\"), or \"auto\""))
    width, height = int(m.group(1)), int(m.group(2))
    if width == 0 or height == 0:
        # Snapping this up to the 8-pixel minimum would answer an uninitialised
        # client variable with an 8x8 image, which is not what it asked for.
        raise HTTPException(status_code=400, detail=(
            f"size {size!r} has a zero dimension"))

    max_side, max_pixels = _size_limits(args)
    if width > max_side or height > max_side:
        raise HTTPException(status_code=400, detail=(
            f"size {width}x{height} exceeds the maximum side of {max_side} px "
            f"(image.max_side)"))
    if width * height > max_pixels:
        raise HTTPException(status_code=400, detail=(
            f"size {width}x{height} is {width * height / 1e6:.1f} MP, above "
            f"this install's limit of {max_pixels / 1e6:.1f} MP "
            f"(image.max_pixels). Raise it if the card can take it."))

    snapped_w = max(_SIZE_MULTIPLE,
                    int(round(width / _SIZE_MULTIPLE)) * _SIZE_MULTIPLE)
    snapped_h = max(_SIZE_MULTIPLE,
                    int(round(height / _SIZE_MULTIPLE)) * _SIZE_MULTIPLE)
    if (snapped_w, snapped_h) != (width, height):
        _log.info("size %dx%d is not on the 8-pixel grid the VAE needs — "
                  "generating %dx%d", width, height, snapped_w, snapped_h)
    return snapped_w, snapped_h


def _resolve_clip_skip(request, model_config, args=None):
    """``clip_skip`` for this request: asked for, else the model's, else global.

    None means "whatever the pipeline does by default", which is the last CLIP
    layer on SD 1.x and the penultimate one on SDXL. The booru-tag fine-tunes
    want 2, and a client speaking the OpenAI image API has no field to say so,
    so the model's own config has to be able to.
    """
    for v in (getattr(request, 'clip_skip', None),
              (model_config or {}).get('clip_skip'),
              getattr(args or global_args, 'image_clip_skip', None)):
        if v is None:
            continue
        try:
            v = int(v)
        except (TypeError, ValueError):
            continue
        return v if v > 0 else None
    return None


def _resolve_prompt_weighting(request, model_config, args=None) -> bool:
    """Whether to parse the A1111 weight grammar. On unless something says no."""
    for v in (getattr(request, 'prompt_weighting', None),
              (model_config or {}).get('prompt_weighting'),
              getattr(args or global_args, 'image_prompt_weighting', None)):
        if v is not None:
            return bool(v)
    return True


def _apply_prefix(prefix, text: str) -> str:
    """``prefix`` in front of ``text``, comma-joined, applied at most once.

    Booru-tag prompts are a comma-separated list, so the two parts join with
    ", " and the prefix's own trailing comma is not doubled. If ``text``
    already starts with the prefix -- a caller who pasted a full prompt
    including the ladder -- it is left alone rather than repeated, which would
    only dilute the rest of the prompt.
    """
    head = (prefix or "").strip().rstrip(",").strip()
    body = (text or "").strip()
    if not head:
        return text or ""
    if body.lower().startswith(head.lower()):
        return body
    return f"{head}, {body}" if body else head


def _effective_prompts(request, model_config=None):
    """``(prompt, negative_prompt)`` with the model's standing prefixes applied.

    A model fine-tuned on a quality ladder wants it on every request, and a
    client speaking the plain OpenAI image API sends a sentence and has no
    field to put the ladder in. So the prefix belongs to the model:
    ``prompt_prefix`` and ``negative_prompt_prefix`` on its config entry.

    A request can override either one -- its own value wins, and an empty
    string switches the prefix off for that call, which is what a caller
    writing its own complete prompt needs. Nothing configured means the
    prompts are returned exactly as sent.
    """
    cfg = model_config or {}
    pos = getattr(request, 'prompt_prefix', None)
    if pos is None:
        pos = cfg.get('prompt_prefix')
    neg = getattr(request, 'negative_prompt_prefix', None)
    if neg is None:
        neg = cfg.get('negative_prompt_prefix')

    prompt = _apply_prefix(pos, getattr(request, 'prompt', '') or "")
    negative = _apply_prefix(neg, getattr(request, 'negative_prompt', None) or "")
    return prompt, negative


def _apply_embeddings(pipeline, request, model_config=None, args=None):
    """Load this request's textual inversions onto the pipeline, if any.

    Must happen before the prompt is tokenized: loading registers the trigger
    word, and until it is registered the word tokenizes as ordinary text, so
    the prompt quietly asks for something else. A name the caller gave
    explicitly and that cannot be loaded is a 400 -- rendering without it
    would look like a success.
    """
    try:
        from codai.models import textual_inversion as ti
    except Exception:
        return []
    refs = list(getattr(request, 'embeddings', None) or [])
    refs += list((model_config or {}).get('textual_inversion') or [])
    directory = getattr(args or global_args, 'image_embeddings_dir', None)
    if not refs and not directory:
        return []
    try:
        return ti.apply(pipeline, refs, directory, strict=bool(refs))
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def _weighted_embed_kwargs(pipeline, prompt, neg_prompt, clip_skip, weighting):
    """Encode prompt + negative ourselves, or None to use the pipeline's path.

    Keeps the whole prompt (as many 75-token windows as it takes) and applies
    per-token weights, which is the only way the booru-tag prompting style
    survives contact with CLIP's 77-token limit.
    """
    try:
        from codai.models.prompt_weighting import (
            encode_weighted_prompt, has_weighting_syntax, supports_weighted_embeds,
            window_count,
        )
        if not supports_weighted_embeds(pipeline):
            return None
        if not weighting and has_weighting_syntax(prompt):
            # Worth saying: the caller wrote the grammar and asked for it not to
            # be parsed, so those brackets are about to reach CLIP as text.
            _log.info("prompt_weighting is off: '(tag:1.3)' / BREAK in this "
                      "prompt are being sent to the model literally")
        kwargs = encode_weighted_prompt(
            pipeline, prompt or "", neg_prompt or "",
            clip_skip=clip_skip, plain=not weighting,
            dtype=getattr(getattr(pipeline, 'text_encoder', None), 'dtype', None),
        )
        if kwargs and _log.isEnabledFor(logging.INFO):
            n = window_count(pipeline.tokenizer, prompt or "", plain=not weighting)
            if n > 1:
                _log.info("Prompt encoded in %d windows: %d tokens of room, "
                          "none truncated (CLIP alone holds 75)", n, n * 75)
        return kwargs
    except Exception as e:
        _log.warning("Weighted prompt encode failed (%s) — falling back to the "
                     "pipeline's own encoder", e)
        return None


async def _generate_with_diffusers(pipeline, request, global_args, http_request=None,
                                   model_config=None):
    """Generate images using a diffusers pipeline (with prompt-embedding cache)."""
    import torch
    import numpy as np
    import time as time_module

    # The same weights serve both directions: a model an edit request loaded as
    # img2img can create a character here without a second copy. Unknown
    # families are left exactly as they are.
    _view = _task_view(pipeline, "text2img")
    if _view is None:
        raise HTTPException(status_code=501, detail=(
            f"{type(pipeline).__name__} cannot generate from a prompt alone: "
            f"diffusers offers no text-to-image pipeline for this family."))
    pipeline = _view

    if getattr(request, 'disable_safety_checker', False):
        _disable_safety_checker(pipeline)

    # Apply optional per-request VAE override
    if getattr(request, 'vae_model', None):
        await _apply_vae_override(pipeline, request.vae_model)

    # Apply optional per-request LoRA weights
    if getattr(request, 'loras', None):
        _apply_loras(pipeline, request.loras)

    # Determine size. A malformed or oversized size is a 400 from here, not a
    # silent fall-back to a square the caller did not ask for.
    width, height = _parse_size(request.size, 1024, 1024, global_args)

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
    # The model card's own recommendation, between the request and the generic
    # fallback: it is better information than a 30-step default (a distilled
    # model wants 8 at CFG 1) and worse information than what the caller
    # explicitly asked for. A fused distill LoRA detected on the pipeline is
    # more specific still, so it keeps precedence.
    _dflt = model_defaults(model_config)
    _d_steps = _dflt.get("steps") or _dflt.get("num_inference_steps")
    _d_cfg = _dflt.get("guidance_scale")
    if _d_cfg is None:
        _d_cfg = _dflt.get("cfg_scale")
    # A distilled checkpoint whose card says nothing. 30 steps at CFG 7.5 is
    # the wrong answer for a model distilled for 8 at CFG 1 -- four times the
    # work AND an over-guided, burnt-looking picture. Below the card's own
    # defaults and below a fused distill LoRA, because both know more; above
    # the generic fallback, because the model's own name knows more than that.
    _distilled = None
    if not _d_steps and _accel_steps is None:
        try:
            from codai.models.acceleration import distilled_defaults_for
            _distilled = distilled_defaults_for(
                str((model_config or {}).get('path') or '')
                or str(getattr(request, 'model', '') or ''))
        except Exception:
            _distilled = None
        if _distilled and not (request.steps and request.guidance_scale):
            print(f"  '{_distilled['token']}' in the model id: this checkpoint is "
                  f"distilled, so {_distilled['steps']} steps at CFG "
                  f"{_distilled['guidance_scale']} (its card carries no defaults; "
                  f"the generic fallback would be 30 at CFG 7.5)")

    num_steps = request.steps if request.steps else (
        _accel_steps if _accel_steps else
        (int(_d_steps) if _d_steps else
         (int(_distilled["steps"]) if _distilled else
          (30 if quality == "standard" else 50))))
    cfg_scale = request.guidance_scale if request.guidance_scale else (
        _accel_cfg if _accel_cfg is not None else
        (float(_d_cfg) if _d_cfg is not None else
         (float(_distilled["guidance_scale"]) if _distilled else
          (getattr(global_args, 'image_cfg_scale', 7.5) if quality == "standard" else 9.0)))
    )
    # The sampler: what the caller asked for, else what the card recommends.
    # A bad name from the CALLER is a 400 -- they can fix it -- while a bad one
    # from a model card is a warning, because refusing the generation over the
    # operator's config would strand a request that is itself fine.
    _want_sampler = getattr(request, 'sampler', None)
    if _want_sampler:
        try:
            _apply_sampler(pipeline, _want_sampler)
        except ValueError as _e:
            raise HTTPException(status_code=400, detail=str(_e))
    elif _dflt.get("sampler"):
        try:
            _apply_sampler(pipeline, _dflt["sampler"])
        except Exception as _e:
            _log.warning("this model's default sampler %r could not be applied: %s",
                         _dflt.get("sampler"), _e)

    _progress_reset(num_steps)

    # Register this generation as a cancellable task (live view + cooperative
    # cancel via the step callback below).
    # The caller's own id, when they named one, so /v1/images/progress can be
    # asked about THIS request instead of a global slot shared with everyone
    # else generating at the same time. Claimed once: the short "loading" task
    # registered earlier keeps its own id (see codai.api.request_progress).
    from codai.api import request_progress as _rp
    _tid = task_registry.register(
        "image", title=(request.prompt or "")[:80],
        model=getattr(request, 'model', '') or '', total=num_steps,
        task_id=_rp.take())
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
    # The model's standing prefixes, applied once here so the encode, the
    # cache key and the pipeline call all see the same text.
    prompt, neg_prompt = _effective_prompts(request, model_config)
    do_cfg = cfg_scale > 1.0

    # Learned prompt tokens first: they change what the prompt tokenizes TO.
    _ti_tokens = _apply_embeddings(pipeline, request, model_config, global_args)

    # clip_skip and the weight grammar both change what the embeddings ARE, so
    # they belong in the cache key: without them the first request's layer
    # choice would be served to every later one.
    _clip_skip = _resolve_clip_skip(request, model_config, global_args)
    _weighting = _resolve_prompt_weighting(request, model_config, global_args)
    model_id = (f"{_model_name}#{id(pipeline)}"
                f"|cs={_clip_skip}|w={int(_weighting)}"
                f"|ti={','.join(_ti_tokens)}")

    # Some pipelines (Z-Image) return prompt embeddings as *per-sample lists*
    # rather than stacked tensors, and the pipeline consumes/extends those
    # lists during a run.  Reusing them on a later request corrupts the batch
    # dimension (`assert len(size) == bsz` in the Z-Image transformer's
    # unpatchify).  These are not safe to cache, so encode fresh every time.
    _embed_cacheable = "ZImage" not in _pipe_cls

    cached_embeds = _embed_cache.get(prompt, neg_prompt, model_id) if _embed_cacheable else None
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
        # First choice: encode it ourselves, in as many 75-token windows as the
        # prompt needs and with per-token weights. The pipeline's own encoder
        # cannot do either — it truncates at 77 and reads the brackets as text.
        embed_kwargs = _weighted_embed_kwargs(
            pipeline, prompt, neg_prompt, _clip_skip, _weighting) or {}
        if embed_kwargs and _embed_cacheable:
            _embed_cache.put(prompt, neg_prompt, model_id, embed_kwargs)

    if not embed_kwargs and not cache_hit and _embed_cacheable:
        # Not a CLIP-pair pipeline (Flux, SD3), or the encode failed: the
        # pipeline's own path, as before.
        try:
            if hasattr(pipeline, 'encode_prompt'):
                import inspect as _inspect
                _ep_params = set(_inspect.signature(pipeline.encode_prompt).parameters)
                _ep_kwargs = {"prompt": prompt}
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
                _call_params = set(
                    _inspect.signature(pipeline.__call__).parameters)
                if len(enc) == 2 and 'prompt_embeds_mask' in _call_params:
                    # Krea 2 and the Qwen-Image family also return a 2-tuple,
                    # but it is (prompt_embeds, prompt_embeds_mask) -- not a
                    # negative. Passing the mask AS the negative is what the
                    # pipeline reports as "If `prompt_embeds` are provided,
                    # `prompt_embeds_mask` also have to be passed", which reads
                    # like a missing argument rather than a mislabelled one.
                    # These are not CLIP-pair pipelines, so weighted 75-token
                    # windows mean nothing to them: let the pipeline encode the
                    # prompt itself, the way Z-Image already does.
                    embed_kwargs = {}
                elif len(enc) == 2:
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
                    _embed_cache.put(prompt, neg_prompt, model_id, embed_kwargs)
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
    char_groups = []
    try:
        from codai.api.characters import character_groups
        char_groups = character_groups(request)
    except Exception:
        char_groups = []
    try:
        profiles = getattr(request, 'character_profiles', None) or []
        if profiles:
            from codai.api.characters import resolve_character_profiles
            char_images += resolve_character_profiles(profiles)
        inline = getattr(request, 'character_references', None) or []
        char_images += list(inline)
        # `characters` is the field that can name several people. Its images
        # join the flat IP-Adapter list too, so a single-character request
        # behaves identically however it was spelled.
        for _g in char_groups:
            for _img in _g["images"]:
                if _img not in char_images:
                    char_images.append(_img)
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
            prompt=prompt,
            negative_prompt=neg_prompt or None,
            num_images_per_prompt=request.n,
            height=height,
            width=width,
            generator=generator,
            guidance_scale=cfg_scale,
            num_inference_steps=num_steps,
            callback_on_step_end=_step_cb,
        )

    # Arguments this model's card specifies that no request field carries.
    call_kwargs.update(_model_call_kwargs(pipeline, model_config))

    # --- two or more characters in one frame ------------------------------
    # IP-Adapter conditions on the MEAN of its reference images, so appending a
    # second character's pictures to the same list asks for one person who
    # resembles both -- never two people. The only way to keep them apart is a
    # pipeline that takes several references and treats them as several
    # subjects: FLUX.2 accepts up to ten, Qwen-Image-Edit-2511 accepts
    # several. Both expose that as a LIST in `image`, which is also how
    # _edit_call_kwargs feeds a multi-reference edit, so the same convention
    # serves generation.
    #
    # One reference per character, the first: a model given four pictures of
    # Alice and four of Bob has no way to know that eight references are two
    # people, and the count is what it keys identity on.
    _multi = [g for g in char_groups if g.get("images")]
    if len(_multi) > 1:
        if _is_instruction_edit(pipeline):
            _refs = [_decode_b64_image(g["images"][0], "characters")
                     for g in _multi]
            call_kwargs["image"] = _refs
            # The references ARE the conditioning here; blending them through
            # IP-Adapter as well would undo the separation.
            char_images = []
            _names = ", ".join(str(g.get("name") or "?") for g in _multi)
            _log.info("Multi-character generation: %d references (%s) passed as "
                      "a reference list", len(_refs), _names)
        else:
            _log.warning(
                "%d characters were requested (%s) but %s takes a single "
                "reference set: IP-Adapter averages its references, so they "
                "will blend into one subject rather than appearing as %d "
                "people. Use a multi-reference model (FLUX.2, "
                "Qwen-Image-Edit-2511) for a scene with more than one of your "
                "characters.",
                len(_multi), ", ".join(str(g.get("name") or "?") for g in _multi),
                type(pipeline).__name__, len(_multi))

    # Inject IP-Adapter images if character references provided.  The pipeline
    # must have IP-Adapter *weights* loaded first — _ensure_ip_adapter_loaded
    # lazily downloads + loads the right checkpoint for the architecture.
    # A pipeline is cached and reused across requests. Once an IP-Adapter is attached,
    # the UNet config carries encoder_hid_dim_type='ip_image_proj' and every later call
    # REQUIRES image_embeds — so a plain request with no character references on the same
    # cached pipeline died with
    #   ... has the config param `encoder_hid_dim_type` set to 'ip_image_proj' which
    #   requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`
    # Detach it again when this request doesn't need it.
    if not char_images and getattr(pipeline, '_coderai_ip_state', None) == 'loaded':
        try:
            # unload_ip_adapter() restores the previous attention processors, which trips
            # the same SlicedAttnProcessor(slice_size) constructor as the load — so keep
            # slicing off across the unload too, then restore it.
            if hasattr(pipeline, 'disable_attention_slicing'):
                pipeline.disable_attention_slicing()
            pipeline.unload_ip_adapter()
            pipeline._coderai_ip_state = None
            _log.info("IP-Adapter unloaded (no character references this request)")
        except Exception as _e:
            _log.warning("Could not unload IP-Adapter: %s", _e)
        finally:
            if hasattr(pipeline, 'enable_attention_slicing'):
                try:
                    pipeline.enable_attention_slicing()   # restore the memory setting
                except Exception:
                    pass

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
                call_kwargs['ip_adapter_image'] = ip_adapter_image_arg(pipeline, ref_imgs)
            else:
                print("Note: IP-Adapter weights unavailable for this pipeline — "
                      "relying on prompt/LoRA for character consistency")
        except Exception as _ip_err:
            print(f"Warning: IP-Adapter injection failed ({_ip_err}), continuing without character refs")

    # Pin the denoising model's attention to a mask-supporting backend before
    # generating. The diffusers "active attention backend" is PROCESS-WIDE global
    # mutable state (a class attribute on _AttentionBackendRegistry) shared with the
    # video path, which sets it to flash-attn-2. A transformer whose per-module
    # backend is None reads that global at dispatch time — and flash-attn-2 REJECTS
    # attn_mask ("`attn_mask` is not supported for flash-attn 2."), so a masked image
    # transformer (e.g. Z-Image) crashes → HTTP 500. Simply resetting the global
    # isn't enough: the reset isn't atomic with the pipeline() call (which runs in a
    # to_thread worker), so a concurrent/subsequent video generation can flip the
    # shared global back to flash before image attention dispatches — a race.
    #
    # Setting the per-module backend EXPLICITLY makes image dispatch immune to the
    # shared global (the dispatcher passes processor._attention_backend, bypassing
    # get_active_backend()). native (SDPA) supports masks. This doesn't disturb the
    # video path — it sets its own explicit per-module 'flash', so it ignores the
    # global too. Image pipelines were never intentionally on flash, so pinning
    # native here is non-regressive. Cheap + idempotent; no-op if unsupported.
    #
    # This is also the floor on image generation speed, so it is worth stating
    # rather than re-investigating. Measured 2026-10-28 on
    # Z-Image-Turbo-IntoRealism, 768x1344, 10 steps, idle RTX 3090: 21.4 s warm,
    # i.e. 2.1 s/step. Of the three things worth suspecting, two were ruled out
    # -- the card's step count IS applied (the per-request progress endpoint
    # reports total=10) and there is no bnb repack (the checkpoint is fp8 upcast
    # to bf16) -- and the third was an unnecessary enable_model_cpu_offload from
    # the model page's default offload_strategy, which cost 12 s a request on a
    # model that fits the card whole (72 s -> 33 s -> 21.4 s).
    #
    # What is left is SDPA, and for THIS family that is not a choice: Z-Image
    # and Krea 2 mask their text conditioning, and neither SageAttention nor
    # flash-attn 2 accepts an attn_mask (see _ATTN_MASK_FAMILIES in
    # codai/api/video.py -- measured on LTX-2.3-10Eros, which failed on sage and
    # then identically on flash). So a few seconds per 1 Mpix still is the
    # hardware, not a misconfiguration.
    #
    # The remaining opportunity, NOT taken here: this pins every image pipeline,
    # including families that never pass a mask and could use a fast kernel. The
    # video path already makes that distinction per pipeline class
    # (_pipeline_uses_attn_mask); doing the same here would need the masking
    # image families enumerated and each one measured, since getting it wrong
    # is a 500 at the first denoising step rather than a slow render.
    for _cn in ("transformer", "unet"):
        _c = getattr(pipeline, _cn, None)
        if _c is not None and hasattr(_c, "set_attention_backend"):
            try:
                _c.set_attention_backend("native")
            except Exception:
                pass
    try:
        from diffusers.models.attention_dispatch import (
            _AttentionBackendRegistry, AttentionBackendName)
        from diffusers.utils.constants import DIFFUSERS_ATTN_BACKEND
        _AttentionBackendRegistry.set_active_backend(
            AttentionBackendName(DIFFUSERS_ATTN_BACKEND))
    except Exception:
        pass

    # Sequential text-encoder offload: encode the prompt, hand the encoder's
    # VRAM back, then denoise from the embeddings. The encoder runs once per
    # request and is dead weight for every step after, but it keeps its VRAM --
    # 7.5 GiB of Z-Image's 19.2 GiB pipeline. Opt-in per model
    # (`sequential_text_encoder`), because freeing it costs a host-RAM copy and
    # only pays on models big enough to care.
    if bool((model_config or {}).get('sequential_text_encoder')):
        try:
            from codai.models.text_encoder_offload import encode_then_free
            call_kwargs, _ = await asyncio.to_thread(
                encode_then_free, pipeline, call_kwargs)
        except Exception as _teo:
            print(f"  [text-encoder-offload] skipped ({_teo})", flush=True)

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

    # A flat image is arithmetic, not content -- see _refuse_uniform_output.
    _refuse_uniform_output(result_images,
                           getattr(request, "model", None) or _model_name,
                           getattr(pipeline, "dtype", None))

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
                prompt,
                {
                    "requested_prompt": request.prompt if prompt != request.prompt else None,
                    "negative_prompt": neg_prompt or None,
                    "size": f"{width}x{height}",
                    "requested_size": request.size,
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

    # Parse size (same validation as the diffusers path: sd.cpp's VAE is also
    # an 8x downsample, and it answers an off-grid size with a C-level abort).
    width, height = _parse_size(request.size, 512, 512, global_args)

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

    prompt, neg_prompt = _effective_prompts(request, model_config)
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

    # sd.cpp parses the weight grammar and the token windows itself, inside the
    # engine, so the prompt goes through as written. What it was never given is
    # the negative prompt: this call hardcoded '' and silently discarded
    # whatever the caller asked to avoid -- the half of the booru-tag idiom that
    # holds quality down (score_4, score_5, score_6) stopped at the API.
    _sd_kwargs = dict(
        prompt=prompt,
        negative_prompt=neg_prompt or '',
        width=width,
        height=height,
        cfg_scale=cfg_scale,
        sample_steps=steps,
        seed=seed if seed is not None else 42,
        batch_count=request.n if request.n else 1,
        progress_callback=_sdcpp_progress,
    )
    _cs = _resolve_clip_skip(request, model_config, global_args)
    if _cs:
        _sd_kwargs['clip_skip'] = _cs
    # Drop anything this build of the binding does not take, rather than losing
    # the whole call to a TypeError and retrying blind.
    try:
        import inspect as _inspect
        _accepted = set(_inspect.signature(sd_model.generate_image).parameters)
        if 'kwargs' not in _accepted:
            _sd_kwargs = {k: v for k, v in _sd_kwargs.items() if k in _accepted}
    except (TypeError, ValueError):
        pass

    try:
        result = await asyncio.to_thread(sd_model.generate_image, **_sd_kwargs)
    except TypeError:
        _sd_kwargs.pop('progress_callback', None)
        _sd_kwargs.pop('clip_skip', None)
        result = await asyncio.to_thread(sd_model.generate_image, **_sd_kwargs)
    except Exception as e:
        task_registry.finish(_tid, "error", str(e)[:200])
        _progress_done()
        raise
    finally:
        _progress_done()

    # Small delay to let Vulkan driver settle after generation. On the loop
    # this stopped every other request in the process for 100 ms per image.
    await asyncio.sleep(0.1)
    
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
                prompt,
                {
                    "requested_prompt": request.prompt if prompt != request.prompt else None,
                    "negative_prompt": neg_prompt or None,
                    "size": f"{width}x{height}",
                    "requested_size": request.size,
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
async def get_image_progress(request_id: Optional[str] = None):
    """Image generation progress — this request's own, or the global slot.

    With ``request_id`` (the id the caller passed as ``request_id`` in the
    generation body or an ``X-Request-Id`` header) this answers for THAT
    request only: `phase` is "queued" while it waits, "loading" while the model
    loads and "generating" once steps are counted, and a 404 means the id is
    finished and pruned or was never used. Two callers generating at once can
    each follow their own.

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

    A caller may name this request (``request_id`` in the body, or an
    ``X-Request-Id`` header) and then poll
    ``GET /v1/images/progress?request_id=...`` for its OWN steps rather than a
    slot shared with everyone else generating at the same time.
    """
    global global_args, global_file_path, model_semaphores, queue_flags
    from codai.api import request_progress as _rp
    _rp_token = _rp.set_current(_rp.from_request(request, http_request))
    
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
                # Resolved, not looked up raw: typed entries are keyed
                # "image:<name>", so a bare config.get(alias) misses and every
                # per-model setting is silently replaced by the global default.
                _sdcpp_cfg = (multi_model_manager._config_for_model(model_key)
                              or multi_model_manager._config_for_model(model_name)
                              or {})
                return await _generate_with_sdcpp(pipeline, request, global_args,
                                                  http_request, model_config=_sdcpp_cfg)
            else:
                # Assume it's a diffusers pipeline
                print(f"Using cached diffusers pipeline for generation")
                # See the note above: resolved, not a raw config.get.
                _diff_cfg = (multi_model_manager._config_for_model(model_key)
                             or multi_model_manager._config_for_model(model_name)
                             or {})
                return await _generate_with_diffusers(pipeline, request, global_args,
                                                      http_request, model_config=_diff_cfg)
        
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
                # See the note above: resolved, not a raw config.get.
                _diff_cfg = (multi_model_manager._config_for_model(model_key)
                             or multi_model_manager._config_for_model(model_name)
                             or {})
                _vram_before = multi_model_manager.vram_before_load()
                # Reserve the estimate for the whole load window, the way the
                # video path does. Without it an image load is INVISIBLE to the
                # eviction ladder for as long as it takes: ensure_vram_for sees
                # a card that is actually filling up as free, grants somebody
                # else's reservation, and both OOM. Measured 2026-10-28:
                # Krea-2-RealDream held 18.8 GiB of a 23.56 GiB card for 194 s
                # and every captioned scene of a concurrent compose died on
                # "CUDA out of memory. Tried to allocate 86.00 MiB".
                multi_model_manager.note_loading(model_key)
                try:
                    pipeline = await asyncio.to_thread(
                        _load_diffusers_pipeline, model_name, global_args, _diff_cfg)
                finally:
                    # Cleared here, not after add_model: there is no await
                    # between the two, so no other task can observe the gap.
                    multi_model_manager.clear_loading(model_key)

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
                    return await _generate_with_diffusers(pipeline, request, global_args,
                                                          http_request, model_config=_diff_cfg)
                    
            except HTTPException:
                # The model loaded and the request itself was rejected (a bad
                # size, an embedding that does not exist). Falling through to
                # the next backend would answer a 400 with a confused report
                # about which backends failed to load.
                raise
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
                cfg = (multi_model_manager._config_for_model(model_key)
                       or multi_model_manager._config_for_model(model_name) or {})
                _vram_before = multi_model_manager.vram_before_load()
                multi_model_manager.note_loading(model_key)
                try:
                    sd_model = await asyncio.to_thread(_load_sdcpp_model, resolved_path, global_args, model_config=cfg)
                finally:
                    multi_model_manager.clear_loading(model_key)

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
                
        except HTTPException:
            raise                   # the request was rejected, not the backend
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
    image: str              # /v1/files path, sha256: id, data URI, URL or base64
    mask: Optional[str] = None   # accepted but unused here -- see /v1/images/inpaint
    #: Up to two FURTHER images, for an instruction model that can compose them
    #: ("put the person from image 2 in image 1"). Ignored by denoising models.
    images: Optional[List[str]] = None
    #: Alias of `steps`, which is what the instruction-edit model cards call it.
    num_inference_steps: Optional[int] = None
    #: Accepted and ignored by instruction-edit models, which have no such knob
    #: -- a client cannot know which kind of model it is addressing.
    negative_prompt: Optional[str] = None   # what to avoid; CFG needs it as much as a t2i call does
    clip_skip: Optional[int] = None         # None = the model's configured default
    prompt_weighting: Optional[bool] = None # None = on; see codai/models/prompt_weighting.py
    #: Prepended to the prompts, comma-joined. None = the model's own prefix,
    #: "" = none for this call. See _effective_prompts.
    prompt_prefix: Optional[str] = None
    negative_prompt_prefix: Optional[str] = None
    n: int = 1
    #: Omitted means "keep the source's shape" (see _default_edit_size), NOT a
    #: square: defaulting to 1024x1024 silently distorted every non-square photo.
    size: Optional[str] = None
    response_format: Optional[str] = "url"
    strength: Optional[float] = 0.75    # denoising strength (0=keep original, 1=ignore)
    steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    seed: Optional[int] = None
    quality: Optional[str] = "standard"
    user: Optional[str] = None

    class Config:
        extra = "allow"


def _default_edit_size(width: int, height: int, multiple: int = 64,
                       target_pixels: int = 1024 * 1024) -> tuple:
    """(w, h) keeping the source aspect at ~``target_pixels``, on a grid.

    An omitted ``size`` used to mean 1024x1024, which stretched every photo that
    was not square -- a 3:2 holiday snap came back visibly squashed. Diffusion
    models need each side on a multiple (usually 64), so the aspect is preserved
    as closely as that grid allows rather than exactly.
    """
    width, height = max(1, int(width)), max(1, int(height))
    scale = (target_pixels / float(width * height)) ** 0.5
    w = int(round(width * scale / multiple)) * multiple
    h = int(round(height * scale / multiple)) * multiple
    return max(multiple, w), max(multiple, h)


def _image_bytes(ref: str, field: str = "image") -> bytes:
    """The bytes behind any accepted image reference.

    Accepts everything the compose endpoints accept -- a ``/v1/files/...`` path
    or a ``sha256:`` id from ``/v1/files/upload``, a ``data:`` URI, an http(s)
    URL, or raw base64 -- so a caller that uploaded the photo once can name it
    on each edit instead of re-sending it. A file this install already holds is
    read from disk, never fetched back over HTTP.
    """
    if not ref:
        raise HTTPException(status_code=400, detail=f"{field} is required")
    from codai.compose.media import MediaError, fetch_bytes
    try:
        return fetch_bytes(ref)
    except MediaError as e:
        raise HTTPException(status_code=400, detail=f"{field}: {e}")


def _decode_b64_image(data: str, field: str = "image"):
    """Any accepted image reference → an RGB PIL image."""
    from PIL import Image as PILImage
    return PILImage.open(io.BytesIO(_image_bytes(data, field))).convert("RGB")


def _prompt_call_kwargs(pipeline, request, model_config=None):
    """Prompt kwargs for an img2img / inpaint call: embeddings, or plain text.

    The weighted encoder keeps the whole prompt and the weights; if this is not
    a CLIP-pair pipeline it returns None and the text goes through as before.
    Either way the negative prompt is passed -- the edit endpoints used to have
    nowhere to put one, so an edit could only ever be steered positively.
    """
    _apply_embeddings(pipeline, request, model_config, global_args)
    prompt, neg = _effective_prompts(request, model_config)
    kw = _weighted_embed_kwargs(
        pipeline, prompt, neg,
        _resolve_clip_skip(request, model_config, global_args),
        _resolve_prompt_weighting(request, model_config, global_args))
    if kw:
        return kw
    return {"prompt": prompt, "negative_prompt": neg or None}


def _accepts_image(pipeline) -> bool:
    """Whether this pipeline's ``__call__`` takes an ``image`` at all.

    Worth checking, because the answer was no and nothing said so. Every
    diffusers ``__call__`` ends in ``**kwargs`` (it carries the deprecated
    ``callback`` pair), so a *text-to-image* pipeline handed ``image=`` and
    ``strength=`` accepts both, drops them, and renders a brand-new picture
    from the prompt at its default 1024x1024. The request answers 200 with an
    image, so it reads as a poor edit rather than as no edit at all -- and as
    "size was ignored", since 1024x1024 is the text-to-image default rather
    than the source's shape.
    """
    import inspect
    try:
        params = inspect.signature(pipeline.__call__).parameters
    except (TypeError, ValueError):
        return True          # can't introspect it; let the pipeline speak
    if "image" in params:
        return True
    # Only *args/**kwargs: there is no parameter list to judge, so do not
    # refuse on the strength of not finding a name in it.
    named = [p for p in params.values()
             if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
    return not named


#: The three things one set of image weights can be asked to do. A checkpoint
#: is a UNet, a VAE and its text encoders; which of these tasks it performs is
#: a property of the pipeline class wrapped around them, not of the weights.
TASKS = ("text2img", "img2img", "inpaint")

#: Where the per-task pipelines hang off the loaded model. Deliberately an
#: attribute of the base pipeline rather than another entry in the manager's
#: models dict: the views share the base's components, so they must die exactly
#: when it does, and a second models-dict key is what let a cache key end up in
#: a from_pretrained() call (see _TASK_NAMESPACES in codai/models/manager.py).
_VIEWS_ATTR = "_coderai_task_views"


def _task_mappings():
    from diffusers.pipelines.auto_pipeline import (
        AUTO_IMAGE2IMAGE_PIPELINES_MAPPING,
        AUTO_INPAINT_PIPELINES_MAPPING,
        AUTO_TEXT2IMAGE_PIPELINES_MAPPING,
    )
    return {
        "text2img": AUTO_TEXT2IMAGE_PIPELINES_MAPPING,
        "img2img": AUTO_IMAGE2IMAGE_PIPELINES_MAPPING,
        "inpaint": AUTO_INPAINT_PIPELINES_MAPPING,
    }


def _task_of(pipeline) -> Optional[str]:
    """Which of :data:`TASKS` this pipeline class performs, or None.

    Asked of diffusers' own task tables rather than guessed: a signature check
    would call a ControlNet text-to-image pipeline "img2img", since its
    required ``image`` is the control hint. None means a family diffusers does
    not map, which is left alone -- a smaller set than it sounds, since SD,
    SDXL, Flux and Z-Image all have entries for all three tasks.
    """
    name = type(pipeline).__name__
    for task, mapping in _task_mappings().items():
        if any(getattr(c, "__name__", None) == name for c in mapping.values()):
            return task
    return None


def _task_view(base, task: str):
    """``base`` as a pipeline for ``task``, sharing one copy of the weights.

    This is what lets one model serve both character creation and editing.
    ``AutoPipelineFor*.from_pipe`` builds the task's pipeline class around the
    components already in VRAM, so a model used for text-to-image and
    image-to-image costs what it costs once -- 18 GB for an SDXL fine-tune,
    which is the difference between both tasks being resident and an eviction
    on every alternation.

    Views are cached on the base and shared, so a LoRA or IP-Adapter applied
    through one task is visible to the others: same UNet, by design.

    Returns the base unchanged when it already performs ``task`` or when its
    family is not one diffusers maps, and None when a conversion was needed
    and failed.
    """
    # Krea 2 first: diffusers maps no img2img/inpaint class for the family, so
    # _task_of returns None and an edit request would fall through to
    # text-to-image and silently ignore the image it was given. CoderAI
    # provides the edit pipeline (codai/pipelines/krea2_edit.py), built from
    # the loaded components rather than loaded again -- a Krea 2 transformer is
    # 12B and a second copy would mean an eviction on every alternation.
    if task in ("img2img", "inpaint"):
        try:
            from codai.pipelines.krea2_edit import build_edit_view, is_krea2_pipeline
            if is_krea2_pipeline(base):
                views = getattr(base, _VIEWS_ATTR, None)
                if views is not None and task in views:
                    return views[task]
                view = build_edit_view(base)
                if view is not None:
                    if views is None:
                        views = {}
                        try:
                            setattr(base, _VIEWS_ATTR, views)
                        except Exception:
                            pass
                    views[task] = view
                    _log.info("%s also serving %s as %s (same weights, "
                              "in-context edit)", type(base).__name__, task,
                              type(view).__name__)
                    return view
                _log.warning("Krea 2 editing is unavailable on this build")
                return None
        except Exception as exc:
            _log.warning("Krea 2 edit view failed: %s", exc)

    current = _task_of(base)
    if current == task or current is None:
        return base

    views = getattr(base, _VIEWS_ATTR, None)
    if views is None:
        views = {}
        try:
            setattr(base, _VIEWS_ATTR, views)
        except Exception:
            pass        # a pipeline that refuses attributes still converts
    if task in views:
        return views[task]

    from diffusers import (AutoPipelineForImage2Image, AutoPipelineForInpainting,
                           AutoPipelineForText2Image)
    Auto = {"text2img": AutoPipelineForText2Image,
            "img2img": AutoPipelineForImage2Image,
            "inpaint": AutoPipelineForInpainting}[task]
    try:
        view = Auto.from_pipe(base)
    except Exception as exc:
        _log.warning("cannot use %s for %s: %s",
                     type(base).__name__, task, exc)
        return None
    views[task] = view
    _log.info("%s also serving %s as %s (same weights, no second copy)",
              type(base).__name__, task, type(view).__name__)
    return view


async def _pipeline_for_task(model_info: dict, task: str):
    """The pipeline for ``task``, from ONE loaded copy of the model's weights.

    A client that creates characters (text-to-image) and then edits them
    (image-to-image) uses the same checkpoint for both, so loading one per task would hold the weights
    twice -- 18 GB each for an SDXL fine-tune, and an eviction on every
    alternation between the two. The model is loaded once, under the manager's
    own key so its VRAM accounting and eviction see one model, and each task
    gets a view onto those components.

    The load goes through ``_load_diffusers_pipeline``, which is the only
    loader that honours the model's config -- precision, offload, quantization,
    a VAE override. The editing endpoints used to load their own copy with a
    hardcoded dtype and none of that, so a 4-bit model served text-to-image at
    4 bits and editing at bf16, out of two copies.
    """
    model_name = model_info.get('model_name')
    model_key = model_info.get('model_key') or f"image:{model_name}"
    base = model_info.get('model_object') or multi_model_manager.models.get(model_key)

    configured_error = None
    cfg = {}
    if base is None:
        cfg = (multi_model_manager._config_for_model(model_key)
               or multi_model_manager._config_for_model(model_name) or {})
        try:
            base = await asyncio.to_thread(
                _load_diffusers_pipeline, model_name, global_args, cfg)
        except Exception as e:
            base = None
            configured_error = e
            _log.warning("the configured loader could not open %r for %s (%s)",
                         model_name, task, e)
        if base is not None:
            multi_model_manager.add_model(model_key, base)
            multi_model_manager.current_model_key = model_key

    if base is None:
        # THE CONFIGURED LOADER'S FAILURE IS THE ONE WORTH REPORTING.
        #
        # The task-only fallback below knows nothing about the model: not its
        # path, so an alias-keyed sibling is handed to from_pretrained as if it
        # were a repo id, and not its quantization, so a model that only fits
        # when quantized is attempted whole. For a CONFIGURED model it
        # therefore fails later and worse, and its error describes a different
        # problem entirely.
        #
        # Measured in production 2026-10-25: 'Qwen-Image-Edit-Rapid-AIO' failed
        # because its fp8 component cannot be 4-bit quantized and the whole
        # pipeline then OOMed -- and the client was told
        # "Qwen-Image-Edit-Rapid-AIO is not a local folder and is not a valid
        # model identifier listed on huggingface.co/models", which sent
        # everyone looking for a missing repo. Credits were refunded for it.
        if cfg:
            raise HTTPException(status_code=500, detail=(
                f"could not load {model_name!r} for {task}: {configured_error}"))
        # No configuration at all: the task-only load is all there ever was.
        loader = _load_inpaint_pipeline if task == "inpaint" else _load_img2img_pipeline
        try:
            return await asyncio.to_thread(loader, model_name, global_args)
        except Exception as e:
            raise HTTPException(status_code=500,
                                detail=f"Failed to load {task} model: {e}")

    view = _task_view(base, task)
    if view is None or (task in ("img2img", "inpaint") and not _accepts_image(view)):
        raise HTTPException(status_code=501, detail=(
            f"'{model_name}' loaded as {type(base).__name__}, which cannot do "
            f"{task}: diffusers offers no {task} pipeline for this family."))
    return view


def _is_instruction_edit(pipeline) -> bool:
    """Whether this pipeline edits by instruction rather than by denoising.

    Decided from the signature, not a list of class names, because the rule is
    exact and generalises: an instruction-edit pipeline takes an ``image`` and
    has **no** ``strength``. Measured across this diffusers::

        QwenImageEditPlus / QwenImageEdit / FluxKontext   image ✓  strength ✗
        SDXL-Img2Img / QwenImageImg2Img / FluxImg2Img     image ✓  strength ✓

    ``strength`` is what makes img2img img2img: the fraction of the source to
    destroy before re-denoising. A model that rewrites the picture from an
    instruction has nothing to put there, which is why "add a necklace, keep
    everything else" cannot work on the denoising path at any strength.

    ``true_cfg_scale`` is NOT the discriminator -- QwenImageImg2Img and
    FluxImg2Img have it too.
    """
    import inspect
    try:
        params = inspect.signature(pipeline.__call__).parameters
    except (TypeError, ValueError):
        return False
    return "image" in params and "strength" not in params


def _edit_call_kwargs(pipeline, request, images, steps, cfg, generator, n):
    """Call kwargs for an edit, shaped to what THIS pipeline accepts.

    Filtered against the signature rather than branched on the model, so one
    code path serves denoising img2img and instruction editing. Two specifics
    the instruction models need:

    * ``strength`` is dropped where it does not exist. It must not be an error
      for a caller to send it -- a client cannot know which kind of model it is
      talking to, and these pipelines have no ``**kwargs`` to swallow it, so
      passing it through would be a hard TypeError.
    * ``guidance_scale`` becomes ``true_cfg_scale``, which is the knob those
      pipelines actually read; with a Lightning-distilled checkpoint it wants
      to be 1.0 and the step count 4.
    """
    import inspect
    try:
        params = set(inspect.signature(pipeline.__call__).parameters)
    except (TypeError, ValueError):
        params = set()

    instruction = _is_instruction_edit(pipeline)
    kwargs = {
        "image": images if (instruction and len(images) > 1) else images[0],
        "num_inference_steps": steps,
        "num_images_per_prompt": n,
        "generator": generator,
    }
    if "strength" in params:
        kwargs["strength"] = request.strength
    if instruction and "true_cfg_scale" in params:
        kwargs["true_cfg_scale"] = cfg
    else:
        kwargs["guidance_scale"] = cfg
    if params:
        dropped = [k for k in kwargs if k not in params]
        for k in dropped:
            kwargs.pop(k)
        if dropped:
            _log.debug("edit: %s does not take %s", type(pipeline).__name__, dropped)
    return kwargs


def _load_img2img_pipeline(model_name: str, global_args):
    """Load a diffusers img2img pipeline.

    The class comes from the checkpoint's own ``model_index.json``, through
    ``AutoPipelineForImage2Image``, rather than from guessing at the model's
    name. Guessing is what broke editing on the booru-tag fine-tunes:
    "nesaorg/PonyRealism-Uncensored" contains none of 'xl', 'sdxl',
    'stable-diffusion' or 'sd', so it fell through to the generic
    ``DiffusionPipeline`` -- which faithfully honours ``_class_name``, and that
    checkpoint declares ``StableDiffusionXLPipeline``, the TEXT-TO-IMAGE class.
    See :func:`_accepts_image` for why that failed silently.
    """
    import torch
    from diffusers import AutoPipelineForImage2Image

    device = _derive_diffusers_device(global_args)
    precision = getattr(global_args, 'image_precision', 'bf16') if global_args else 'bf16'
    dtype_map = {'bf16': torch.bfloat16, 'f16': torch.float16, 'f32': torch.float32}
    torch_dtype = dtype_map.get(precision, torch.bfloat16)

    PipeClass = AutoPipelineForImage2Image

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

    pipe = await _pipeline_for_task(model_info, "img2img")
    _edit_key = model_info.get('model_key') or f"image:{model_name}"

    try:
        import torch
        # Up to three images for an instruction edit ("put the person from
        # image 2 into image 1"); `image` remains the first and only one for
        # every denoising model.
        refs = [request.image] + list(getattr(request, 'images', None) or [])
        if len(refs) > 3:
            raise HTTPException(status_code=400, detail=(
                f"{len(refs)} images given; an instruction edit takes at most 3 "
                f"(`image` plus up to 2 in `images`)"))
        sources = [_decode_b64_image(r) for r in refs]
        source_img = sources[0]

        width, height = source_img.size
        if request.size and str(request.size).strip().lower() != "auto":
            # Validated, so a typo is a 400 instead of an edit at some other
            # size; snapped, because the VAE needs the 8-pixel grid.
            width, height = _parse_size(request.size, width, height, global_args)
            if (width, height) != source_img.size:
                source_img = source_img.resize((width, height))
        else:
            # No size asked for: keep the photo's own shape at ~1 MP.
            width, height = _default_edit_size(*source_img.size)
            if (width, height) != source_img.size:
                source_img = source_img.resize((width, height))

        # source_img is what the size logic above resized; put it back as the
        # first reference, or the pipeline is handed the original instead and
        # `size` silently does nothing.
        sources[0] = source_img
        if len(sources) > 1:
            # Every reference is conditioning for the same output, so they go
            # in at one shape.
            sources = [im if im.size == source_img.size else im.resize(source_img.size)
                       for im in sources]

        seed = request.seed or getattr(global_args, 'image_seed', None)
        generator = torch.Generator(device=pipe.device).manual_seed(seed) if seed else None
        quality = request.quality or "standard"
        _instruction = _is_instruction_edit(pipe)
        # A distilled instruction model is tuned for 4 steps at CFG 1.0;
        # handing it a denoising model's 30 steps and 7.5 wastes a minute per
        # image and overcooks the result.
        _steps_default = 4 if _instruction else (30 if quality == "standard" else 50)
        _cfg_default = 1.0 if _instruction else (
            getattr(global_args, 'image_cfg_scale', 7.5) if quality == "standard" else 9.0)
        num_steps = (getattr(request, 'num_inference_steps', None)
                     or request.steps or _steps_default)
        cfg_scale = request.guidance_scale or _cfg_default

        _edit_cfg = (multi_model_manager.config.get(_edit_key)
                     or multi_model_manager.config.get(model_name) or {})
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: pipe(
                **_edit_call_kwargs(pipe, request, sources, num_steps, cfg_scale,
                                    generator, request.n),
                **_prompt_call_kwargs(pipe, request, _edit_cfg),
            )
        )
    except HTTPException:
        raise                       # a 400 from here is the caller's, not ours
    except Exception as e:
        # The traceback goes to the log, not only into the response body: a
        # 500 whose cause reaches the caller alone is one nobody can diagnose
        # afterwards, and the log showed the request and then just the status.
        _log.error("Image editing failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Image editing failed: {e}")

    try:
        _refuse_uniform_output(result.images, request.model,
                              getattr(pipe, "dtype", None))
    except RuntimeError as e:
        _log.error("%s", e)
        raise HTTPException(status_code=500, detail=str(e))

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
    #: Omitted means "keep the source's shape" (see _default_edit_size), as on
    #: /v1/images/edits. It used to default to 1024x1024, which squashed every
    #: non-square photo -- and dragged the mask out of alignment with it.
    size: Optional[str] = None
    negative_prompt: Optional[str] = None
    clip_skip: Optional[int] = None
    prompt_weighting: Optional[bool] = None
    #: Prepended to the prompts, comma-joined. None = the model's own prefix,
    #: "" = none for this call. See _effective_prompts.
    prompt_prefix: Optional[str] = None
    negative_prompt_prefix: Optional[str] = None
    n: int = 1
    response_format: Optional[str] = "url"
    steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    strength: Optional[float] = 0.99
    seed: Optional[int] = None
    quality: Optional[str] = "standard"

    class Config:
        extra = "allow"


def _load_inpaint_pipeline(model_name: str, global_args):
    """Load a diffusers inpainting pipeline.

    Same rule as :func:`_load_img2img_pipeline`: the task class comes from the
    checkpoint, not from substrings in its name.
    """
    import torch
    from diffusers import AutoPipelineForInpainting
    device = _derive_diffusers_device(global_args)
    precision = getattr(global_args, 'image_precision', 'bf16') if global_args else 'bf16'
    dtype_map = {'bf16': torch.bfloat16, 'f16': torch.float16, 'f32': torch.float32}
    torch_dtype = dtype_map.get(precision, torch.bfloat16)
    PClass = AutoPipelineForInpainting
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
    pipe = await _pipeline_for_task(model_info, "inpaint")
    model_key = model_info.get('model_key') or f"image:{model_name}"
    try:
        import torch
        source_img = _decode_b64_image(request.image)
        mask_img = _decode_b64_image(request.mask).convert("L")  # greyscale mask
        if request.size and str(request.size).strip().lower() != "auto":
            w, h = _parse_size(request.size, *source_img.size, args=global_args)
            if (w, h) != source_img.size:
                source_img = source_img.resize((w, h))
                mask_img = mask_img.resize((w, h))
        else:
            # No size asked for: keep the photo's own shape on the grid, the
            # same rule the edit endpoint follows.
            w, h = _default_edit_size(*source_img.size)
            if (w, h) != source_img.size:
                source_img = source_img.resize((w, h))
                mask_img = mask_img.resize((w, h))
        seed = request.seed or getattr(global_args, 'image_seed', None)
        generator = torch.Generator(device=pipe.device).manual_seed(seed) if seed else None
        quality = request.quality or "standard"
        num_steps = request.steps or (30 if quality == "standard" else 50)
        cfg_scale = request.guidance_scale or (getattr(global_args, 'image_cfg_scale', 7.5) if quality == "standard" else 9.0)
        _inp_cfg = (multi_model_manager._config_for_model(model_key)
                    or multi_model_manager._config_for_model(model_name) or {})
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: pipe(
                image=source_img,
                mask_image=mask_img,
                strength=request.strength,
                num_inference_steps=num_steps,
                guidance_scale=cfg_scale,
                num_images_per_prompt=request.n,
                generator=generator,
                **_prompt_call_kwargs(pipe, request, _inp_cfg),
            )
        )
    except HTTPException:
        raise
    except Exception as e:
        # The traceback goes to the log, not only into the response body: a
        # 500 whose cause reaches the caller alone is one nobody can diagnose
        # afterwards, and the log showed the request and then just the status.
        _log.error("Inpainting failed: %s", e, exc_info=True)
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


def _configured_upscalers() -> list:
    """Image models this install could actually upscale with."""
    names = []
    try:
        from codai.models.capabilities import detect_model_capabilities
        for m in multi_model_manager.image_models:
            try:
                if detect_model_capabilities(m).image_upscaling:
                    names.append(m)
            except Exception:
                continue
    except Exception:
        pass
    return names


@router.post("/v1/images/upscale", summary="Upscale an image")
async def create_image_upscale(request: ImageUpscaleRequest, http_request: Request = None):
    """Upscale an image with a real super-resolution model.

    There is no PIL/LANCZOS fallback: a plain resize is not an upscale, and
    returning one silently charged for work that did not happen.
    """
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
    if upscaler[0] == 'pil':
        # _load_upscaler fell through to PIL LANCZOS, which is a plain resize:
        # answering 200 here bills the caller for an "AI upscale" that never
        # ran, and the result looks merely soft rather than wrong. Say so.
        usable = _configured_upscalers()
        raise HTTPException(
            status_code=400,
            detail=(f"'{model_name}' is not an upscaling model"
                    + (f" — configured upscalers: {', '.join(usable)}"
                       if usable else
                       " — no upscaling model is configured; add a Real-ESRGAN "
                       "/ SwinIR / diffusers upscaler on the model page")))
    raw = _image_bytes(request.image)
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
        # Stamp the LRU clock through the active_in_vram chokepoint: a raw
        # models[key] assignment leaves _last_used unset, so the key sorts
        # first in _lru_order() (default 0.0) and is evicted before genuinely
        # older models -- and the first-pass 'key == active_in_vram' guard
        # never protects it either.
        multi_model_manager.active_in_vram = model_key
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
        multi_model_manager.active_in_vram = model_key
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
    raw = _image_bytes(request.image)
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
    raw = _image_bytes(request.image)
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
    clip_skip: Optional[int] = None         # None = the model's configured default
    prompt_weighting: Optional[bool] = None # None = on
    #: Prepended to the prompts, comma-joined. None = the model's own prefix,
    #: "" = none for this call. See _effective_prompts.
    prompt_prefix: Optional[str] = None
    negative_prompt_prefix: Optional[str] = None
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
        multi_model_manager.active_in_vram = model_key

    # Run inpaint
    import torch
    generator = torch.Generator().manual_seed(request.seed) if request.seed is not None else None

    def _run():
        kwargs = dict(
            image=img,
            mask_image=mask_img,
            num_inference_steps=request.steps or 30,
            guidance_scale=request.guidance_scale or 7.5,
            strength=request.strength or 0.99,
            **_prompt_call_kwargs(pipeline, request),
        )
        if generator:
            kwargs['generator'] = generator
        if hasattr(pipeline, 'safety_checker'):
            pipeline.safety_checker = None
        return pipeline(**kwargs).images[0]

    try:
        result_img = await asyncio.get_event_loop().run_in_executor(None, _run)
    except HTTPException:
        raise
    except Exception as e:
        # The traceback goes to the log, not only into the response body: a
        # 500 whose cause reaches the caller alone is one nobody can diagnose
        # afterwards, and the log showed the request and then just the status.
        _log.error("Outfit change failed: %s", e, exc_info=True)
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
        await asyncio.to_thread(subprocess.run, ['ffmpeg', '-y', '-i', in_path, f'{frames_dir}/%08d.png'],
                       capture_output=True, check=True)

        probe = await asyncio.to_thread(subprocess.run, 
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
            multi_model_manager.active_in_vram = model_key

        import torch
        from PIL import Image as PILImage
        import numpy as np
        import cv2

        generator = torch.Generator().manual_seed(request.seed) if request.seed is not None else None

        def _process_frames():
            # Encoded once for the whole clip: the prompt does not change
            # between frames, and a text-encoder pass per frame is pure waste.
            prompt_kwargs = _prompt_call_kwargs(pipeline, request)
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
                    image=img,
                    mask_image=mask_img,
                    num_inference_steps=request.steps or 30,
                    guidance_scale=request.guidance_scale or 7.5,
                    strength=request.strength or 0.99,
                    **prompt_kwargs,
                )
                if generator:
                    kwargs['generator'] = generator
                if hasattr(pipeline, 'safety_checker'):
                    pipeline.safety_checker = None
                result = pipeline(**kwargs).images[0]
                result.save(fpath)

        await asyncio.get_event_loop().run_in_executor(None, _process_frames)

        out_path = tempfile.mktemp(suffix='_outfit.mp4')
        temps.append(out_path)
        await asyncio.to_thread(subprocess.run, 
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
