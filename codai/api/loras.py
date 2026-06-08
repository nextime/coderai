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

def _resolve_base_model_path(base_model: str) -> str:
    """Resolve an image model key (or path/HF id) to a diffusers model directory."""
    try:
        from codai.api.state import multi_model_manager
        for key in (f"image:{base_model}", base_model):
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
    base_path = _resolve_base_model_path(req.base_model)
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
        from codai.api.state import multi_model_manager
        multi_model_manager.unload_all_models()
    except Exception as e:
        print(f"  [lora] could not unload models before training: {e}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dtype = torch.float32  # train in fp32 for stability

    _set_progress(status="preparing", message=f"loading base model: {base_path}")

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

    tokenizer = CLIPTokenizer.from_pretrained(base_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(base_path, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(base_path, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(base_path, subfolder="unet").to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    lora_cfg = PeftLoraConfig(
        r=rank, lora_alpha=rank, init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_cfg)
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

    # Release training tensors.
    del unet, vae, text_encoder, optimizer, latents_list
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

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

    tokenizer_1 = CLIPTokenizer.from_pretrained(base_path, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(base_path, subfolder="tokenizer_2")
    text_encoder_1 = CLIPTextModel.from_pretrained(base_path, subfolder="text_encoder").to(device)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(base_path, subfolder="text_encoder_2").to(device)
    vae = AutoencoderKL.from_pretrained(base_path, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(base_path, subfolder="unet").to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")

    for m in (vae, text_encoder_1, text_encoder_2, unet):
        m.requires_grad_(False)

    lora_cfg = PeftLoraConfig(
        r=rank, lora_alpha=rank, init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_cfg)
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

    del unet, vae, text_encoder_1, text_encoder_2, optimizer, latents_list
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    path = _lora_weight_file(name) or save_dir
    _set_progress(active=False, status="done", message="done", path=path)
    return {"name": name, "path": path}


def _write_meta(name, req, base_path, n_images, arch, instance_prompt):
    meta = {
        "name": name,
        "base_model": req.base_model,
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

@router.post("/v1/loras/train")
async def train_lora(req: LoraTrainRequest, _auth=Depends(_require_api_auth)):
    """Train a LoRA from a saved character profile or supplied images (blocking)."""
    import asyncio
    if not req.name or '/' in req.name or '..' in req.name:
        raise HTTPException(status_code=400, detail="Invalid LoRA name")
    if not req.base_model:
        raise HTTPException(status_code=400, detail="base_model is required")
    if not _train_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="A LoRA training job is already running")
    try:
        try:
            result = await asyncio.to_thread(_train_lora_sync, req)
        except HTTPException:
            raise
        except Exception as e:
            import traceback
            traceback.print_exc()
            _set_progress(active=False, status="error", message=str(e))
            raise HTTPException(status_code=500, detail=f"LoRA training failed: {e}")
    finally:
        _train_lock.release()
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
