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
LoRA training endpoints.

Train a small per-character (or per-style) LoRA adapter from a handful of
reference images, then apply it to BOTH image and video diffusion pipelines for
consistent identity across models.

POST   /v1/loras/train      – train a LoRA from a saved character profile or images
GET    /v1/loras            – list trained LoRAs
GET    /v1/loras/progress   – training progress (for the active job)
GET    /v1/loras/{name}     – info about one trained LoRA
DELETE /v1/loras/{name}     – delete a trained LoRA

All model execution stays server-side.  Training runs in-process so it can share
the model manager's VRAM (it evicts resident models first) and honour the global
thermal-protection checkpoints.
"""

import base64
import io
import json
import os
import threading
import time
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from codai.platform_paths import default_loras_dir
from codai.queue.manager import queue_manager

router = APIRouter()

_LORAS_DIR: Optional[str] = None

# Single-job training progress (training is VRAM-heavy; we run one at a time).
_progress_lock = threading.Lock()
_progress = {
    "active": False,
    "name": None,
    "step": 0,
    "total": 0,
    "status": "idle",      # idle | preparing | training | saving | done | error
    "message": "",
    "started_at": 0.0,
    "path": None,
}
_train_lock = threading.Lock()


def set_global_args(args):
    global _LORAS_DIR
    base = getattr(args, 'file_path', None)
    if base and os.path.isdir(base):
        root = base
    elif base:
        root = os.path.dirname(base)
    else:
        root = None
    _LORAS_DIR = os.path.join(root, 'loras') if root else str(default_loras_dir())
    os.makedirs(_LORAS_DIR, exist_ok=True)


def _loras_dir() -> str:
    if _LORAS_DIR:
        os.makedirs(_LORAS_DIR, exist_ok=True)
        return _LORAS_DIR
    d = str(default_loras_dir())
    os.makedirs(d, exist_ok=True)
    return d


def _lora_dir(name: str) -> str:
    return os.path.join(_loras_dir(), name)


def _lora_weight_file(name: str) -> Optional[str]:
    """Return the path to the trained weights file for a LoRA, if present."""
    d = _lora_dir(name)
    for fn in ("pytorch_lora_weights.safetensors", "pytorch_lora_weights.bin"):
        p = os.path.join(d, fn)
        if os.path.isfile(p):
            return p
    return None


def _require_api_auth(request: Request) -> None:
    """Raise 401 if auth is enabled and the request carries no valid credential."""
    try:
        from codai.admin import routes as _admin_routes
        sm = _admin_routes.session_manager
    except Exception:
        return
    if sm is None:
        return
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        token = auth[7:].strip()
        if sm.verify_token(token):
            return
    cookie = request.cookies.get("session", "")
    if cookie.endswith(".MUST_CHANGE"):
        cookie = cookie[:-12]
    if cookie and sm.validate_session(cookie):
        return
    raise HTTPException(
        status_code=401,
        detail={"message": "Invalid API key. Provide a valid Bearer token.",
                "type": "invalid_request_error", "code": "invalid_api_key"},
    )


# ── Pydantic models ───────────────────────────────────────────────────────────

class LoraTrainRequest(BaseModel):
    name: str                              # output LoRA name (folder)
    base_model: str                        # image model key (models.json) or HF id / path
    # Optional separate UNet-based SD1.x/SDXL model to train the LoRA against,
    # when `base_model` (the generation model) is a transformer/DiT (Z-Image,
    # Flux, SD3) this trainer can't target.  Falls back to base_model if unset.
    train_base_model: Optional[str] = None
    # Target pipeline for the LoRA: "image" (default, SD1.x/SDXL UNet) or "video"
    # (Wan video DiT). For "video", `base_model` is the VIDEO model id/path and the
    # LoRA is trained against that exact model so it loads on the video pipeline.
    target: Optional[str] = "image"
    # Quantize the (large) video transformer to 4-bit for training (QLoRA). Lets a
    # 14B video model's LoRA fit on a consumer GPU. Ignored for image targets.
    quantize_4bit: Optional[bool] = True
    num_frames: Optional[int] = 1          # training video length (1 = stills-only)
    character: Optional[str] = None        # saved character profile to pull images from
    environment: Optional[str] = None      # OR saved environment profile to pull images from
    images: Optional[List[str]] = None     # OR explicit base64/data-uri images
    instance_prompt: Optional[str] = None  # e.g. "a photo of sks man"; auto from name if None
    steps: Optional[int] = 800             # training steps (balanced default)
    rank: Optional[int] = 16               # LoRA rank
    learning_rate: Optional[float] = 1e-4
    resolution: Optional[int] = 512
    seed: Optional[int] = 42
    model_config = ConfigDict(extra="allow")


# ── Base-model resolution ─────────────────────────────────────────────────────

def _configured_train_base(base_model: str) -> Optional[str]:
    """Read a per-model `lora_train_base_model` override from the image model's
    config (models.json entry), if any.  Lets a deployment declare once that e.g.
    a Z-Image generation model should train LoRAs against an SDXL model, so no
    client has to know.  Returns the configured model key/path or None."""
    try:
        from codai.models.manager import multi_model_manager
        for key in (f"image:{base_model}", base_model):
            cfg = multi_model_manager.config.get(key)
            if not cfg:
                continue
            for src in (cfg, cfg.get('_raw_cfg') or {}):
                v = src.get('lora_train_base_model') if isinstance(src, dict) else None
                if v and isinstance(v, str) and v.strip():
                    return v.strip()
    except Exception:
        pass
    return None


def _resolve_base_model_path(base_model: str, category: str = "image") -> str:
    """Resolve a model key (or path/HF id) to a diffusers model directory.

    `category` selects the models.json key prefix to try first ("image" or
    "video"), so a video LoRA resolves against the video model entry."""
    try:
        from codai.models.manager import multi_model_manager
        keys = ([f"{category}:{base_model}", f"image:{base_model}",
                 f"video:{base_model}", base_model])
        seen = set()
        for key in keys:
            if key in seen:
                continue
            seen.add(key)
            cfg = multi_model_manager.config.get(key)
            if cfg:
                for k in ('path', 'model_path', 'model', 'diffusers_path'):
                    v = cfg.get(k)
                    if v and isinstance(v, str):
                        return v
    except Exception:
        pass
    # Treat as a direct path or HF repo id.
    return base_model


def _decode_image(ref: str):
    from PIL import Image as PILImage
    if ref.startswith('data:'):
        ref = ref.split(',', 1)[1]
    raw = base64.b64decode(ref)
    return PILImage.open(io.BytesIO(raw)).convert('RGB')


def _gather_images(req: LoraTrainRequest):
    """Return a list of PIL images from the character profile and/or inline images."""
    imgs = []
    if req.character:
        try:
            from codai.api.characters import resolve_character_profiles
            for b64 in resolve_character_profiles([req.character]):
                try:
                    imgs.append(_decode_image(b64))
                except Exception:
                    pass
        except Exception:
            pass
    if req.environment:
        try:
            from codai.api.environments import resolve_environment_profiles
            for b64 in resolve_environment_profiles([req.environment]):
                try:
                    imgs.append(_decode_image(b64))
                except Exception:
                    pass
        except Exception:
            pass
    for ref in (req.images or []):
        try:
            imgs.append(_decode_image(ref))
        except Exception:
            pass
    return imgs


def _set_progress(**kw):
    with _progress_lock:
        _progress.update(kw)


def _lora_debug_enabled() -> bool:
    """LoRA training step logging to the terminal is gated on --debug-lora."""
    try:
        from codai.api.state import get_global_args
        return bool(getattr(get_global_args(), "debug_lora", False))
    except Exception:
        return False


def _dbg_lora(msg: str) -> None:
    if _lora_debug_enabled():
        print(f"  [lora][debug] {msg}", flush=True)


def _free_vram_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.mem_get_info()[0] / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def _free_train_vram() -> None:
    """Fully release training-base-model VRAM after a job.

    The trainer loads the SD1.x/SDXL base directly (not via the model manager),
    so eviction can't see it. peft adapter hooks create reference cycles that
    keep the modules alive until a gc pass runs, so empty_cache() alone leaves
    the memory pinned. Collect twice, then drop the allocator cache.
    """
    import gc
    try:
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _dbg_lora(f"freed training VRAM — {_free_vram_gb():.1f} GB now free")
    except Exception:
        pass


# ── Training base-model cache ────────────────────────────────────────────────
# The SD1.x/SDXL base used for LoRA training is expensive to load from disk.
# We keep its components cached *on CPU* between consecutive training jobs so a
# back-to-back job against the same base skips the reload. Holding it on CPU
# (not GPU) means it consumes no VRAM between jobs, so image/video generation
# can use the GPU freely — components are moved to the GPU only while a job is
# actually running, then moved back to CPU when it finishes.
_base_lock = threading.RLock()
_base_cache = {"path": None, "arch": None, "components": None}


def _load_base_components(base_path, arch, dtype):
    """Load all base components on CPU at the given dtype (no GPU placement)."""
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    if arch == "sdxl":
        from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
        return {
            "tokenizer_1": CLIPTokenizer.from_pretrained(base_path, subfolder="tokenizer"),
            "tokenizer_2": CLIPTokenizer.from_pretrained(base_path, subfolder="tokenizer_2"),
            "text_encoder_1": CLIPTextModel.from_pretrained(base_path, subfolder="text_encoder").to(dtype=dtype),
            "text_encoder_2": CLIPTextModelWithProjection.from_pretrained(base_path, subfolder="text_encoder_2").to(dtype=dtype),
            "vae": AutoencoderKL.from_pretrained(base_path, subfolder="vae").to(dtype=dtype),
            "unet": UNet2DConditionModel.from_pretrained(base_path, subfolder="unet").to(dtype=dtype),
            "noise_scheduler": DDPMScheduler.from_pretrained(base_path, subfolder="scheduler"),
        }
    from transformers import CLIPTextModel, CLIPTokenizer
    return {
        "tokenizer": CLIPTokenizer.from_pretrained(base_path, subfolder="tokenizer"),
        "text_encoder": CLIPTextModel.from_pretrained(base_path, subfolder="text_encoder").to(dtype=dtype),
        "vae": AutoencoderKL.from_pretrained(base_path, subfolder="vae").to(dtype=dtype),
        "unet": UNet2DConditionModel.from_pretrained(base_path, subfolder="unet").to(dtype=dtype),
        "noise_scheduler": DDPMScheduler.from_pretrained(base_path, subfolder="scheduler"),
    }


def _acquire_base(base_path, arch, dtype):
    """Return cached CPU-resident base components, loading from disk if needed.

    A change of base_path/arch drops the previous cache first. Must be called
    inside a training job (holds the train lock for the job's duration)."""
    with _base_lock:
        c = _base_cache
        if c["components"] is not None and c["path"] == base_path and c["arch"] == arch:
            _dbg_lora(f"reusing cached base model (CPU): {base_path}")
            return c["components"]
        if c["components"] is not None:
            _dbg_lora(f"base model changed ({c['path']} → {base_path}); dropping cache")
            _drop_base_cache_locked()
        _dbg_lora(f"loading base model from disk: {base_path} ({arch})")
        comps = _load_base_components(base_path, arch, dtype)
        c.update(path=base_path, arch=arch, components=comps)
        return comps


def _drop_base_cache_locked() -> None:
    _base_cache["components"] = None
    _base_cache["path"] = None
    _base_cache["arch"] = None


def _drop_base_cache() -> None:
    with _base_lock:
        had = _base_cache["components"] is not None
        _drop_base_cache_locked()
    if had:
        _free_train_vram()


def _release_base_cache(needed_gb: float = 0.0) -> float:
    """External VRAM releaser (registered with the model manager).

    Drops the cached training base on demand. Skips while a training job is
    running (the base is in use). Between jobs the base lives on CPU, so this
    mainly reclaims host RAM; it returns the VRAM delta it observed."""
    if not _train_lock.acquire(blocking=False):
        return 0.0
    try:
        with _base_lock:
            if _base_cache["components"] is None:
                return 0.0
        before = _free_vram_gb()
        _drop_base_cache()
        return max(0.0, _free_vram_gb() - before)
    finally:
        _train_lock.release()


# Let the model manager reclaim this cache when a generation needs VRAM.
try:
    from codai.models.manager import multi_model_manager as _mmm
    _mmm.register_external_vram_releaser(_release_base_cache)
except Exception:
    pass


# ── Training core ─────────────────────────────────────────────────────────────

def _train_lora_sync(req: LoraTrainRequest) -> dict:
    """Run a DreamBooth-style LoRA training in-process. Returns {name, path}."""
    import torch
    import torch.nn.functional as F
    from diffusers import (
        AutoencoderKL, DDPMScheduler, UNet2DConditionModel,
    )
    from diffusers.optimization import get_scheduler
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig as PeftLoraConfig
    from peft.utils import get_peft_model_state_dict
    from transformers import CLIPTextModel, CLIPTokenizer

    name = req.name

    # ── Video (Wan DiT) LoRA target ───────────────────────────────────────────
    # Train directly against the configured VIDEO model so the resulting LoRA
    # loads on the video pipeline. No SD UNet fallback here — the base IS the DiT.
    if (req.target or "image").lower() == "video":
        video_path = _resolve_base_model_path(req.base_model, category="video")
        steps = max(50, min(5000, int(req.steps or 800)))
        rank = max(2, min(128, int(req.rank or 16)))
        resolution = int(req.resolution or 512)
        lr = float(req.learning_rate or 1e-4)
        seed = int(req.seed if req.seed is not None else 42)
        _set_progress(active=True, name=name, step=0, total=steps,
                      status="preparing", message="loading reference images",
                      started_at=time.time(), path=None)
        images = _gather_images(req)
        if not images:
            raise HTTPException(status_code=400,
                                detail="No training images (provide `character` or `images`)")
        if req.instance_prompt:
            instance_prompt = req.instance_prompt
        elif req.environment and not req.character:
            instance_prompt = f"a video of {name} place"
        else:
            instance_prompt = f"a video of {name} person"
        try:
            from codai.models.manager import multi_model_manager
            multi_model_manager.unload_all_models()
        except Exception as e:
            print(f"  [lora] could not unload models before training: {e}")
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        return _train_wan(req, video_path, images, instance_prompt,
                          steps, rank, resolution, lr, seed, device)

    # Resolve the model to TRAIN against (the generation model may be a DiT this
    # trainer can't target). Precedence: explicit request override > per-model
    # `lora_train_base_model` from models.json config > the base_model itself.
    train_base = (req.train_base_model
                  or _configured_train_base(req.base_model)
                  or req.base_model)
    if train_base != req.base_model:
        print(f"  [lora] training '{name}' against base '{train_base}' "
              f"(generation model: '{req.base_model}')")
    base_path = _resolve_base_model_path(train_base)
    steps = max(50, min(5000, int(req.steps or 800)))
    rank = max(2, min(128, int(req.rank or 16)))
    resolution = int(req.resolution or 512)
    lr = float(req.learning_rate or 1e-4)
    seed = int(req.seed if req.seed is not None else 42)

    _set_progress(active=True, name=name, step=0, total=steps,
                  status="preparing", message="loading reference images",
                  started_at=time.time(), path=None)

    images = _gather_images(req)
    if not images:
        raise HTTPException(status_code=400,
                            detail="No training images (provide `character` or `images`)")

    if req.instance_prompt:
        instance_prompt = req.instance_prompt
    elif req.environment and not req.character:
        instance_prompt = f"a photo of {name} place"
    else:
        instance_prompt = f"a photo of {name} person"

    # Free VRAM: evict every resident model so training has the whole GPU.
    try:
        from codai.models.manager import multi_model_manager
        multi_model_manager.unload_all_models()
    except Exception as e:
        print(f"  [lora] could not unload models before training: {e}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dtype = torch.float32  # train in fp32 for stability

    _set_progress(status="preparing", message=f"loading base model: {base_path}")

    # This DreamBooth-LoRA trainer only supports UNet-based SD1.x / SDXL
    # pipelines.  Transformer/DiT models (Z-Image, Flux, SD3, …) have a
    # `transformer/` subfolder and no `unet/`; loading their non-CLIP tokenizer
    # as a CLIPTokenizer crashes deep inside transformers.  Detect that up front
    # and fail with an actionable message instead.
    import os as _os
    _has_unet = _os.path.isdir(_os.path.join(base_path, "unet"))
    _has_transformer = _os.path.isdir(_os.path.join(base_path, "transformer"))
    if _has_transformer and not _has_unet:
        raise ValueError(
            f"LoRA training base model '{base_path}' is a transformer/DiT "
            f"architecture (has 'transformer/', no 'unet/'). This trainer only "
            f"supports UNet-based SD1.x/SDXL models. Configure an SD1.x or SDXL "
            f"image model as the LoRA training base."
        )

    # Detect SDXL by attempting to load a second tokenizer.
    is_sdxl = False
    try:
        from transformers import CLIPTokenizer as _CT
        _CT.from_pretrained(base_path, subfolder="tokenizer_2")
        is_sdxl = True
    except Exception:
        is_sdxl = False

    if is_sdxl:
        result = _train_sdxl(req, base_path, images, instance_prompt,
                             steps, rank, resolution, lr, seed, device)
    else:
        result = _train_sd15(req, base_path, images, instance_prompt,
                             steps, rank, resolution, lr, seed, device)
    return result


def _make_dataset(images, tokenizers, text_encoders, instance_prompt,
                  resolution, vae, device, weight_dtype, is_sdxl):
    """Pre-encode latents + text embeddings for every reference image once."""
    import torch
    from torchvision import transforms

    tfm = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    latents_list = []
    with torch.no_grad():
        for img in images:
            px = tfm(img).unsqueeze(0).to(device, dtype=vae.dtype)
            lat = vae.encode(px).latent_dist.sample() * vae.config.scaling_factor
            latents_list.append(lat.to(weight_dtype).cpu())
    return latents_list


def _train_sd15(req, base_path, images, instance_prompt,
                steps, rank, resolution, lr, seed, device):
    import torch
    import torch.nn.functional as F
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from diffusers.utils import convert_state_dict_to_diffusers
    from diffusers import StableDiffusionPipeline
    from peft import LoraConfig as PeftLoraConfig
    from peft.utils import get_peft_model_state_dict
    from transformers import CLIPTextModel, CLIPTokenizer

    name = req.name
    g = torch.Generator(device=device).manual_seed(seed)

    # Consistent fp32 precision (see _train_sdxl) to avoid mixed-dtype crashes.
    weight_dtype = torch.float32
    # Components come from the cross-job CPU cache; move to GPU for this job.
    comps = _acquire_base(base_path, "sd15", weight_dtype)
    tokenizer = comps["tokenizer"]
    text_encoder = comps["text_encoder"].to(device, dtype=weight_dtype)
    vae = comps["vae"].to(device, dtype=weight_dtype)
    unet = comps["unet"].to(device, dtype=weight_dtype)
    noise_scheduler = comps["noise_scheduler"]

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    lora_cfg = PeftLoraConfig(
        r=rank, lora_alpha=rank, init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_cfg, adapter_name="default")
    lora_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=lr)

    # Pre-encode latents and the (single) instance-prompt embedding.
    latents_list = _make_dataset(images, [tokenizer], [text_encoder], instance_prompt,
                                 resolution, vae, device, torch.float32, is_sdxl=False)
    with torch.no_grad():
        tok = tokenizer(instance_prompt, padding="max_length",
                        max_length=tokenizer.model_max_length, truncation=True,
                        return_tensors="pt").input_ids.to(device)
        encoder_hidden_states = text_encoder(tok)[0]

    # VAE + text encoder are done; move them back to CPU (keeps them cached for
    # the next job) so only the UNet stays resident during training.
    vae.to("cpu")
    text_encoder.to("cpu")
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    _set_progress(status="training", message="training (SD1.5)")
    unet.train()
    n = len(latents_list)
    for step in range(steps):
        latents = latents_list[step % n].to(device)
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                  (bsz,), device=device).long()
        noisy = noise_scheduler.add_noise(latents, noise, timesteps)
        model_pred = unet(noisy, timesteps, encoder_hidden_states).sample

        if noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            target = noise
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        optimizer.step()
        optimizer.zero_grad()

        if step % 10 == 0 or step == steps - 1:
            _set_progress(step=step + 1, message=f"step {step+1}/{steps} loss={loss.item():.4f}")
            _dbg_lora(f"SD1.5 step {step+1}/{steps} loss={loss.item():.4f}")
        # Mid-training thermal checkpoint (pauses if CPU/GPU too hot).
        try:
            from codai.models.thermal import checkpoint as _thermal_checkpoint
            _thermal_checkpoint(context="lora-train", throttle_seconds=2.0)
        except Exception:
            pass

    _set_progress(status="saving", message="saving LoRA weights")
    save_dir = _lora_dir(name)
    os.makedirs(save_dir, exist_ok=True)
    unet_lora = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionPipeline.save_lora_weights(save_directory=save_dir,
                                              unet_lora_layers=unet_lora,
                                              safe_serialization=True)
    _write_meta(name, req, base_path, len(images), "sd15", instance_prompt)

    # Job done: drop this job's adapter + transients and move the UNet back to
    # CPU. The base stays cached on CPU (reused by the next job); no VRAM is held
    # afterwards, so the next image/video request gets the full GPU.
    try:
        unet.delete_adapters("default")
    except Exception:
        pass
    unet.to("cpu")
    try:
        del optimizer, latents_list, lora_params, encoder_hidden_states
    except Exception:
        pass
    _free_train_vram()

    path = _lora_weight_file(name) or save_dir
    _set_progress(active=False, status="done", message="done", path=path)
    return {"name": name, "path": path}


def _train_sdxl(req, base_path, images, instance_prompt,
                steps, rank, resolution, lr, seed, device):
    import torch
    import torch.nn.functional as F
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from diffusers import StableDiffusionXLPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig as PeftLoraConfig
    from peft.utils import get_peft_model_state_dict
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
    from torchvision import transforms

    name = req.name
    g = torch.Generator(device=device).manual_seed(seed)

    # Train in a single consistent precision. Checkpoints can store each
    # component in a different native dtype (e.g. fp16 text encoders + fp32
    # UNet), which otherwise crashes with "mat1 and mat2 have the same dtype"
    # (Half != float) in cross-attention. fp32 is the stable reference for
    # LoRA fine-tuning.
    weight_dtype = torch.float32

    # Components come from the cross-job CPU cache; move to GPU for this job.
    comps = _acquire_base(base_path, "sdxl", weight_dtype)
    tokenizer_1 = comps["tokenizer_1"]
    tokenizer_2 = comps["tokenizer_2"]
    text_encoder_1 = comps["text_encoder_1"].to(device, dtype=weight_dtype)
    text_encoder_2 = comps["text_encoder_2"].to(device, dtype=weight_dtype)
    vae = comps["vae"].to(device, dtype=weight_dtype)
    unet = comps["unet"].to(device, dtype=weight_dtype)
    noise_scheduler = comps["noise_scheduler"]

    for m in (vae, text_encoder_1, text_encoder_2, unet):
        m.requires_grad_(False)

    lora_cfg = PeftLoraConfig(
        r=rank, lora_alpha=rank, init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_cfg, adapter_name="default")
    lora_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=lr)

    tfm = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    # Pre-encode latents.
    latents_list = []
    with torch.no_grad():
        for img in images:
            px = tfm(img).unsqueeze(0).to(device, dtype=vae.dtype)
            lat = vae.encode(px).latent_dist.sample() * vae.config.scaling_factor
            latents_list.append(lat.float().cpu())

    # SDXL text conditioning: concat hidden states + pooled embeds from encoder 2.
    with torch.no_grad():
        ids_1 = tokenizer_1(instance_prompt, padding="max_length",
                            max_length=tokenizer_1.model_max_length, truncation=True,
                            return_tensors="pt").input_ids.to(device)
        ids_2 = tokenizer_2(instance_prompt, padding="max_length",
                            max_length=tokenizer_2.model_max_length, truncation=True,
                            return_tensors="pt").input_ids.to(device)
        enc1 = text_encoder_1(ids_1, output_hidden_states=True)
        enc2 = text_encoder_2(ids_2, output_hidden_states=True)
        # penultimate hidden states
        prompt_embeds = torch.cat([enc1.hidden_states[-2], enc2.hidden_states[-2]], dim=-1)
        pooled = enc2[0]  # text_embeds (pooled) from projection encoder

    add_time_ids = torch.tensor(
        [[resolution, resolution, 0, 0, resolution, resolution]],
        device=device, dtype=prompt_embeds.dtype,
    )

    # VAE + text encoders are no longer needed during the training loop; move
    # them back to CPU (keeps them cached for the next job) so only the UNet
    # stays resident — keeps SDXL fp32 training in VRAM budget.
    vae.to("cpu")
    text_encoder_1.to("cpu")
    text_encoder_2.to("cpu")
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    _set_progress(status="training", message="training (SDXL)")
    unet.train()
    n = len(latents_list)
    for step in range(steps):
        latents = latents_list[step % n].to(device)
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                  (bsz,), device=device).long()
        noisy = noise_scheduler.add_noise(latents, noise, timesteps)
        added = {"text_embeds": pooled, "time_ids": add_time_ids}
        model_pred = unet(noisy, timesteps, prompt_embeds, added_cond_kwargs=added).sample

        if noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            target = noise
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        optimizer.step()
        optimizer.zero_grad()

        if step % 10 == 0 or step == steps - 1:
            _set_progress(step=step + 1, message=f"step {step+1}/{steps} loss={loss.item():.4f}")
            _dbg_lora(f"SDXL step {step+1}/{steps} loss={loss.item():.4f}")
        try:
            from codai.models.thermal import checkpoint as _thermal_checkpoint
            _thermal_checkpoint(context="lora-train", throttle_seconds=2.0)
        except Exception:
            pass

    _set_progress(status="saving", message="saving LoRA weights")
    save_dir = _lora_dir(name)
    os.makedirs(save_dir, exist_ok=True)
    unet_lora = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionXLPipeline.save_lora_weights(save_directory=save_dir,
                                                unet_lora_layers=unet_lora,
                                                safe_serialization=True)
    _write_meta(name, req, base_path, len(images), "sdxl", instance_prompt)

    # Job done: drop this job's adapter + transients and move the UNet back to
    # CPU. The base stays cached on CPU for the next job; no VRAM held afterwards
    # so the next image/video request gets the full GPU (see _train_sd15 note).
    try:
        unet.delete_adapters("default")
    except Exception:
        pass
    unet.to("cpu")
    try:
        del optimizer, latents_list, lora_params
        del prompt_embeds, pooled, add_time_ids
    except Exception:
        pass
    _free_train_vram()

    path = _lora_weight_file(name) or save_dir
    _set_progress(active=False, status="done", message="done", path=path)
    return {"name": name, "path": path}


def _train_wan(req, base_path, images, instance_prompt,
               steps, rank, resolution, lr, seed, device):
    """Train a LoRA for a Wan video DiT directly, so it loads on the video
    pipeline. Stills are treated as 1-frame videos: VAE-encoded to latents, then a
    rectified-flow (flow-matching) loss trains PEFT LoRA adapters on the
    transformer attention layers. Wan2.2 A14B has two experts (transformer +
    transformer_2 routed by a noise boundary); both get adapters and are trained
    on the timestep range they own. The base is quantized to 4-bit (QLoRA) when
    requested so a 14B model's LoRA fits on a consumer GPU.
    """
    import os as _os
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig as PeftLoraConfig
    from peft.utils import get_peft_model_state_dict
    from torchvision import transforms
    try:
        from diffusers import (AutoencoderKLWan, WanTransformer3DModel,
                               FlowMatchEulerDiscreteScheduler, WanPipeline)
        from diffusers.utils import convert_state_dict_to_diffusers
        from transformers import UMT5EncoderModel, AutoTokenizer
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=("Wan video LoRA training needs a diffusers build with Wan "
                    f"support (AutoencoderKLWan/WanTransformer3DModel): {e}"))

    name = req.name
    torch.manual_seed(seed)
    compute_dtype = torch.bfloat16
    quantize = bool(getattr(req, "quantize_4bit", True))
    num_frames = max(1, int(getattr(req, "num_frames", 1) or 1))

    # ── 1. VAE (3D): encode each still as a 1-frame video latent, then offload ──
    _set_progress(status="preparing", message=f"loading Wan VAE: {base_path}")
    vae = AutoencoderKLWan.from_pretrained(base_path, subfolder="vae",
                                           torch_dtype=torch.float32).to(device)
    vae.requires_grad_(False)
    vae.eval()
    z_dim = int(vae.config.z_dim)
    lat_mean = torch.tensor(vae.config.latents_mean).view(1, z_dim, 1, 1, 1).to(device)
    lat_std = torch.tensor(vae.config.latents_std).view(1, z_dim, 1, 1, 1).to(device)
    spatial = (resolution // 16) * 16  # Wan VAE prefers multiples of 16
    tfm = transforms.Compose([
        transforms.Resize(spatial, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(spatial),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    latents_list = []
    with torch.no_grad():
        for img in images:
            px = tfm(img)                              # [3,H,W] in [-1,1]
            vid = px.unsqueeze(0).unsqueeze(2)         # [1,3,1,H,W]
            if num_frames > 1:
                vid = vid.repeat(1, 1, num_frames, 1, 1)
            vid = vid.to(device, dtype=torch.float32)
            lat = vae.encode(vid).latent_dist.sample()  # [1,z,t,h,w]
            lat = (lat - lat_mean) / lat_std
            latents_list.append(lat.to(compute_dtype).cpu())
    vae.to("cpu")
    del vae
    _free_train_vram()

    # ── 2. Text encoder (UMT5): encode the instance prompt once, then offload ──
    _set_progress(status="preparing", message="encoding prompt (UMT5)")
    tokenizer = AutoTokenizer.from_pretrained(base_path, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        base_path, subfolder="text_encoder", torch_dtype=compute_dtype).to(device)
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    with torch.no_grad():
        tok = tokenizer(instance_prompt, padding="max_length", max_length=512,
                        truncation=True, return_tensors="pt")
        ids = tok.input_ids.to(device)
        mask = tok.attention_mask.to(device)
        enc = text_encoder(ids, attention_mask=mask).last_hidden_state
        encoder_hidden_states = (enc * mask.unsqueeze(-1)).to(compute_dtype).cpu()
    text_encoder.to("cpu")
    del text_encoder
    _free_train_vram()

    # ── 3. Transformer expert(s) + LoRA ───────────────────────────────────────
    _set_progress(status="preparing",
                  message=f"loading Wan transformer{' (4-bit)' if quantize else ''}")
    q_cfg = None
    if quantize:
        try:
            from diffusers import BitsAndBytesConfig as _DiffBnb
            q_cfg = _DiffBnb(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=compute_dtype)
        except Exception as e:
            print(f"  [lora][wan] 4-bit unavailable ({e}); loading in bf16")
            q_cfg = None

    def _load_transformer(subfolder):
        kw = dict(subfolder=subfolder, torch_dtype=compute_dtype)
        if q_cfg is not None:
            kw["quantization_config"] = q_cfg
        tr = WanTransformer3DModel.from_pretrained(base_path, **kw)
        if q_cfg is None:
            tr = tr.to(device)
        return tr

    experts = [("transformer", _load_transformer("transformer"))]
    if _os.path.isdir(_os.path.join(base_path, "transformer_2")):
        experts.append(("transformer_2", _load_transformer("transformer_2")))

    boundary = getattr(experts[0][1].config, "boundary_ratio", None)
    lora_cfg = PeftLoraConfig(
        r=rank, lora_alpha=rank, init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    lora_params = []
    for _, tr in experts:
        tr.requires_grad_(False)
        tr.add_adapter(lora_cfg, adapter_name="default")
        try:
            tr.enable_gradient_checkpointing()
            # Make checkpointing track grads through a frozen/quantized base.
            tr.patch_embedding.register_forward_hook(
                lambda m, i, o: o.requires_grad_(True))
        except Exception:
            pass
        tr.train()
        lora_params += [p for p in tr.parameters() if p.requires_grad]
    if not lora_params:
        raise HTTPException(status_code=500,
                            detail="Wan LoRA: no trainable adapter params were created")
    optimizer = torch.optim.AdamW(lora_params, lr=lr)

    try:
        sched = FlowMatchEulerDiscreteScheduler.from_pretrained(base_path, subfolder="scheduler")
        shift = float(getattr(sched.config, "shift", 3.0) or 3.0)
        num_train_t = float(getattr(sched.config, "num_train_timesteps", 1000) or 1000)
    except Exception:
        shift, num_train_t = 3.0, 1000.0

    def _pick_expert(t_val):
        if len(experts) > 1 and boundary is not None:
            # High-noise expert (index 0) owns t >= boundary; low-noise is index 1.
            return experts[0][1] if t_val >= float(boundary) else experts[1][1]
        return experts[0][1]

    _set_progress(status="training", message="training (Wan video LoRA)")
    n = len(latents_list)
    for step in range(steps):
        x0 = latents_list[step % n].to(device, dtype=compute_dtype)
        noise = torch.randn_like(x0)
        # Rectified-flow timestep with Wan resolution shift applied to sigma.
        u = torch.rand(1, device=device)
        sigma = (shift * u) / (1.0 + (shift - 1.0) * u)
        s = sigma.view(-1, 1, 1, 1, 1)
        x_t = (1.0 - s) * x0 + s * noise
        target = noise - x0                              # flow-matching velocity
        timestep = (sigma * num_train_t).to(compute_dtype)
        tr = _pick_expert(float(sigma.item()))
        pred = tr(hidden_states=x_t, timestep=timestep,
                  encoder_hidden_states=encoder_hidden_states.to(device),
                  return_dict=False)[0]
        loss = F.mse_loss(pred.float(), target.float(), reduction="mean")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        optimizer.step()
        optimizer.zero_grad()

        if step % 10 == 0 or step == steps - 1:
            _set_progress(step=step + 1,
                          message=f"step {step+1}/{steps} loss={loss.item():.4f}")
            _dbg_lora(f"Wan step {step+1}/{steps} loss={loss.item():.4f}")
        try:
            from codai.models.thermal import checkpoint as _thermal_checkpoint
            _thermal_checkpoint(context="lora-train", throttle_seconds=2.0)
        except Exception:
            pass

    # ── 4. Save (transformer + optional transformer_2 LoRA layers) ─────────────
    _set_progress(status="saving", message="saving Wan LoRA weights")
    save_dir = _lora_dir(name)
    os.makedirs(save_dir, exist_ok=True)
    save_kwargs = {"transformer_lora_layers":
                   convert_state_dict_to_diffusers(get_peft_model_state_dict(experts[0][1]))}
    if len(experts) > 1:
        t2 = convert_state_dict_to_diffusers(get_peft_model_state_dict(experts[1][1]))
        try:
            WanPipeline.save_lora_weights(save_directory=save_dir,
                                          transformer_2_lora_layers=t2,
                                          safe_serialization=True, **save_kwargs)
        except TypeError:
            print("  [lora][wan] this diffusers WanPipeline.save_lora_weights has no "
                  "transformer_2 arg — saving high-noise expert LoRA only")
            WanPipeline.save_lora_weights(save_directory=save_dir,
                                          safe_serialization=True, **save_kwargs)
    else:
        WanPipeline.save_lora_weights(save_directory=save_dir,
                                      safe_serialization=True, **save_kwargs)
    _write_meta(name, req, base_path, len(images), "wan", instance_prompt)

    # ── 5. Tear down: drop adapters + free VRAM for the next request ───────────
    for _, tr in experts:
        try:
            tr.delete_adapters("default")
        except Exception:
            pass
        try:
            tr.to("cpu")
        except Exception:
            pass  # 4-bit modules can't move; just drop the refs below
    try:
        del optimizer, lora_params, latents_list, encoder_hidden_states, experts
    except Exception:
        pass
    _free_train_vram()

    path = _lora_weight_file(name) or save_dir
    _set_progress(active=False, status="done", message="done", path=path)
    return {"name": name, "path": path}


def _write_meta(name, req, base_path, n_images, arch, instance_prompt):
    meta = {
        "name": name,
        "base_model": req.base_model,
        "train_base_model": (req.train_base_model
                             or _configured_train_base(req.base_model)
                             or req.base_model),
        "base_path": base_path,
        "arch": arch,
        "instance_prompt": instance_prompt,
        "steps": req.steps,
        "rank": req.rank,
        "resolution": req.resolution,
        "num_images": n_images,
        "created_at": int(time.time()),
    }
    try:
        with open(os.path.join(_lora_dir(name), "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass


# ── Endpoints ─────────────────────────────────────────────────────────────────

# Scheduler model key for training. A constant key makes the central queue
# serialize trainings (one at a time, protecting the shared base cache) while
# still admitting them through the same scheduler as every other request.
_TRAIN_MODEL_KEY = "lora-train"


def _train_lora_blocking(req: LoraTrainRequest) -> dict:
    """Run one training job to completion (called inside a worker thread).

    Holds _train_lock for the job's duration so a second training never overlaps
    (also the signal _release_base_cache uses to know a job is in flight). The
    central queue already serializes us, so this acquire returns immediately.
    """
    _train_lock.acquire()
    try:
        return _train_lora_sync(req)
    except Exception:
        import traceback
        traceback.print_exc()
        # On error the base may be in a half-moved / inconsistent state — drop the
        # cache entirely (and reclaim its VRAM) rather than reuse it.
        try:
            _set_progress(active=False, status="error", message="training failed")
        except Exception:
            pass
        _drop_base_cache()
        raise
    finally:
        _train_lock.release()


@router.post("/v1/loras/train")
async def train_lora(req: LoraTrainRequest, _auth=Depends(_require_api_auth)):
    """Train a LoRA (blocking). Admitted through the central request scheduler,
    so concurrent training requests queue and run one after another (instead of
    being rejected) alongside all other model requests."""
    import asyncio
    import uuid
    if not req.name or '/' in req.name or '..' in req.name:
        raise HTTPException(status_code=400, detail="Invalid LoRA name")
    if not req.base_model:
        raise HTTPException(status_code=400, detail="base_model is required")

    request_id = f"lora-train-{uuid.uuid4().hex[:8]}"
    # Wait for a scheduler slot (queues behind other in-flight work; the constant
    # model key keeps trainings strictly one-at-a-time).
    lease = await queue_manager.acquire(request_id, _TRAIN_MODEL_KEY)
    try:
        result = await asyncio.to_thread(_train_lora_blocking, req)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LoRA training failed: {e}")
    finally:
        await queue_manager.release(lease)
    return {"ok": True, **result}


@router.get("/v1/loras/progress")
async def lora_progress():
    with _progress_lock:
        return dict(_progress)


@router.get("/v1/loras")
async def list_loras(_auth=Depends(_require_api_auth)):
    out = []
    d = _loras_dir()
    if os.path.isdir(d):
        for name in sorted(os.listdir(d)):
            wf = _lora_weight_file(name)
            if not wf:
                continue
            meta = {}
            mp = os.path.join(_lora_dir(name), "meta.json")
            if os.path.isfile(mp):
                try:
                    with open(mp) as f:
                        meta = json.load(f)
                except Exception:
                    pass
            out.append({"name": name, "path": wf, **meta})
    return {"loras": out}


@router.get("/v1/loras/{name}")
async def get_lora(name: str, _auth=Depends(_require_api_auth)):
    wf = _lora_weight_file(name)
    if not wf:
        raise HTTPException(status_code=404, detail=f"LoRA '{name}' not found")
    meta = {}
    mp = os.path.join(_lora_dir(name), "meta.json")
    if os.path.isfile(mp):
        try:
            with open(mp) as f:
                meta = json.load(f)
        except Exception:
            pass
    return {"name": name, "path": wf, **meta}


@router.delete("/v1/loras/{name}")
async def delete_lora(name: str, _auth=Depends(_require_api_auth)):
    d = _lora_dir(name)
    if not os.path.isdir(d):
        raise HTTPException(status_code=404, detail=f"LoRA '{name}' not found")
    import shutil
    shutil.rmtree(d)
    return {"ok": True, "name": name}
