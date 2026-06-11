"""Acceleration / distillation support (Lightning / Turbo / LCM / Hyper-SD).

Distillation adapters let diffusion models run in a handful of steps at low
guidance instead of the usual 25–50 steps at high CFG — a 5–10× speedup at minor
quality cost. This module is the single source of truth for:

  * the preset catalog (``ACCEL_PRESETS``) of known distill LoRAs / turbo models,
  * resolving a model's ``acceleration`` config block into a normalized dict,
  * fusing the distill LoRA into a diffusers pipeline at load time (kept separate
    from per-request character/environment LoRAs by *fusing* it in), and
  * supplying the low step-count / guidance defaults at generation time.

The per-model config block (in models.json) looks like::

    "acceleration": {
        "enabled": true,
        "preset": "wan22_lightning_4step",   # catalog key, or "custom"
        "lora": "repo/id:weight_name.safetensors",  # path or HF repo; overrides preset
        "lora_weight": 1.0,
        "steps": 4,
        "guidance_scale": 1.0,
        "flow_shift": 5.0,   # optional: Wan flow-match scheduler shift
        "scheduler": ""      # optional: scheduler class override (e.g. LCMScheduler)
    }

When ``preset`` is a catalog key, any field left unset is filled from the preset;
explicit fields always override. ``enabled: false`` or an absent block means no
acceleration is applied (fully backward compatible).
"""

from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Preset catalog
# ---------------------------------------------------------------------------
# Each entry describes a known distillation setup. Repo ids / weight names are
# best-effort sensible defaults — the admin UI lets the user override the LoRA
# path and every numeric field, so a wrong/renamed repo is never fatal.
#
# `lora`: either None (turbo full-model — no LoRA), a HF repo id, a local path,
#         or "repo_or_path:weight_name.safetensors" to pick a specific file.
# `applies_to`: which model categories the preset is offered for in the UI.
ACCEL_PRESETS: dict = {
    # --- Wan video (phased DMD / Lightx2v) ---
    "wan22_lightning_4step": {
        "label": "Wan2.2 Lightning (4-step DMD)",
        "family": "wan",
        "applies_to": ["video"],
        "lora": "lightx2v/Wan2.2-Lightning",
        "lora_weight": 1.0,
        "steps": 4,
        "guidance_scale": 1.0,
        "flow_shift": 5.0,
        "scheduler": "",
    },
    "wan21_lightx2v_4step": {
        "label": "Wan2.1 Lightx2v (4-step)",
        "family": "wan",
        "applies_to": ["video"],
        "lora": "Kijai/WanVideo_comfy:Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors",
        "lora_weight": 1.0,
        "steps": 4,
        "guidance_scale": 1.0,
        "flow_shift": 5.0,
        "scheduler": "",
    },
    # --- SDXL ---
    "sdxl_lightning_4step": {
        "label": "SDXL-Lightning (4-step)",
        "family": "sdxl",
        "applies_to": ["image"],
        "lora": "ByteDance/SDXL-Lightning:sdxl_lightning_4step_lora.safetensors",
        "lora_weight": 1.0,
        "steps": 4,
        "guidance_scale": 1.0,
        "flow_shift": None,
        "scheduler": "",
    },
    "sdxl_lightning_8step": {
        "label": "SDXL-Lightning (8-step)",
        "family": "sdxl",
        "applies_to": ["image"],
        "lora": "ByteDance/SDXL-Lightning:sdxl_lightning_8step_lora.safetensors",
        "lora_weight": 1.0,
        "steps": 8,
        "guidance_scale": 1.0,
        "flow_shift": None,
        "scheduler": "",
    },
    "sdxl_turbo": {
        "label": "SDXL-Turbo (full model, 1–4 step)",
        "family": "sdxl",
        "applies_to": ["image"],
        "lora": None,  # turbo is a full distilled model, not a LoRA
        "lora_weight": 1.0,
        "steps": 4,
        "guidance_scale": 1.0,
        "flow_shift": None,
        "scheduler": "",
    },
    "sdxl_lcm": {
        "label": "SDXL LCM-LoRA (4–8 step)",
        "family": "sdxl",
        "applies_to": ["image"],
        "lora": "latent-consistency/lcm-lora-sdxl",
        "lora_weight": 1.0,
        "steps": 6,
        "guidance_scale": 1.5,
        "flow_shift": None,
        "scheduler": "LCMScheduler",
    },
    "hyper_sdxl_8step": {
        "label": "Hyper-SD SDXL (8-step)",
        "family": "sdxl",
        "applies_to": ["image"],
        "lora": "ByteDance/Hyper-SD:Hyper-SDXL-8steps-lora.safetensors",
        "lora_weight": 1.0,
        "steps": 8,
        "guidance_scale": 1.0,
        "flow_shift": None,
        "scheduler": "",
    },
    # --- SD 1.5 ---
    "sd15_lcm": {
        "label": "SD1.5 LCM-LoRA (4–8 step)",
        "family": "sd15",
        "applies_to": ["image"],
        "lora": "latent-consistency/lcm-lora-sdv1-5",
        "lora_weight": 1.0,
        "steps": 6,
        "guidance_scale": 1.5,
        "flow_shift": None,
        "scheduler": "LCMScheduler",
    },
}


_NUMERIC_FIELDS = ("lora_weight", "steps", "guidance_scale", "flow_shift")


def resolve_acceleration(model_cfg: Optional[dict]) -> Optional[dict]:
    """Merge a model's ``acceleration`` block with its preset into a normalized dict.

    Returns ``None`` when acceleration is absent or disabled. Otherwise returns::

        {preset, lora, lora_weight, steps, guidance_scale, flow_shift, scheduler}

    Explicit fields in the config override the preset; unset fields fall back to
    the preset (when ``preset`` names a catalog entry).
    """
    if not isinstance(model_cfg, dict):
        return None
    accel = model_cfg.get("acceleration")
    if not isinstance(accel, dict) or not accel.get("enabled"):
        return None

    preset_key = (accel.get("preset") or "").strip()
    base = dict(ACCEL_PRESETS.get(preset_key, {})) if preset_key and preset_key != "custom" else {}

    def _pick(field, default=None):
        # Config value wins when present and non-empty; else preset; else default.
        v = accel.get(field)
        if v is not None and not (isinstance(v, str) and v.strip() == ""):
            return v
        if field in base and base[field] is not None:
            return base[field]
        return default

    out = {
        "preset": preset_key or "custom",
        "lora": _pick("lora"),
        "lora_weight": _pick("lora_weight", 1.0),
        "steps": _pick("steps"),
        "guidance_scale": _pick("guidance_scale"),
        "flow_shift": _pick("flow_shift"),
        "scheduler": (_pick("scheduler", "") or ""),
    }
    # Coerce numeric fields.
    for f in _NUMERIC_FIELDS:
        if out.get(f) is not None:
            try:
                out[f] = float(out[f]) if f in ("lora_weight", "guidance_scale", "flow_shift") else int(out[f])
            except (TypeError, ValueError):
                out[f] = None
    return out


def _split_lora_ref(ref: str):
    """Split "repo_or_path:weight_name.safetensors" → (repo_or_path, weight_name|None).

    A bare Windows-style drive letter or an existing local path with a colon is
    unlikely here; we only treat the LAST ':' as a weight-name separator when the
    suffix looks like a filename (ends in .safetensors / .ckpt / .pt / .bin).
    """
    if not ref or ":" not in ref:
        return ref, None
    head, _, tail = ref.rpartition(":")
    if tail.lower().endswith((".safetensors", ".ckpt", ".pt", ".bin")):
        return head, tail
    return ref, None


def apply_accel_to_pipeline(pipe, accel: Optional[dict]) -> None:
    """Fuse the distill LoRA into a diffusers pipeline and apply scheduler tweaks.

    The LoRA is *fused* (load → fuse → unload the adapter handle) so its weights
    are baked into the transformer/unet. This keeps acceleration orthogonal to
    per-request character/environment LoRAs: later ``load_lora_weights`` /
    ``unload_lora_weights`` cycles (e.g. video.py ``_sync_video_loras``) do not
    touch the fused acceleration weights.

    The resolved ``accel`` dict is stashed on ``pipe._coderai_accel`` so the
    generators can read the step/guidance defaults at call time. All failures are
    caught and logged — generation then simply proceeds un-accelerated.
    """
    if not accel:
        return
    try:
        pipe._coderai_accel = accel
    except Exception:
        pass

    # 1. Optional scheduler swap (e.g. LCM needs LCMScheduler).
    sched_name = (accel.get("scheduler") or "").strip()
    if sched_name:
        try:
            import diffusers as _diffusers
            sched_cls = getattr(_diffusers, sched_name, None)
            cur = getattr(pipe, "scheduler", None)
            if sched_cls is not None and cur is not None:
                pipe.scheduler = sched_cls.from_config(cur.config)
                log.info("[accel] scheduler set to %s", sched_name)
        except Exception as e:
            log.warning("[accel] scheduler swap to %s failed: %s", sched_name, e)

    # 2. Wan flow-match shift.
    flow_shift = accel.get("flow_shift")
    if flow_shift is not None:
        try:
            cur = getattr(pipe, "scheduler", None)
            cfg = getattr(cur, "config", None)
            if cur is not None and cfg is not None and "shift" in cfg:
                import diffusers as _diffusers
                new_cfg = dict(cfg)
                new_cfg["shift"] = float(flow_shift)
                pipe.scheduler = type(cur).from_config(new_cfg)
                log.info("[accel] flow-match shift set to %s", flow_shift)
        except Exception as e:
            log.warning("[accel] flow_shift apply failed: %s", e)

    # 3. Fuse the distill LoRA (when one is configured — turbo has none).
    lora_ref = accel.get("lora")
    if not lora_ref:
        return
    if not hasattr(pipe, "load_lora_weights"):
        log.warning("[accel] pipeline %s has no load_lora_weights — cannot fuse "
                    "acceleration LoRA", type(pipe).__name__)
        return
    repo, weight_name = _split_lora_ref(lora_ref)
    weight = float(accel.get("lora_weight") or 1.0)
    try:
        load_kwargs = {"adapter_name": "__accel__"}
        if weight_name:
            load_kwargs["weight_name"] = weight_name
        pipe.load_lora_weights(repo, **load_kwargs)
        try:
            pipe.set_adapters(["__accel__"], [weight])
        except Exception:
            pass
        # Bake it in, then drop the adapter handle so per-request LoRAs are clean.
        pipe.fuse_lora(lora_scale=weight)
        try:
            pipe.unload_lora_weights()
        except Exception:
            pass
        log.info("[accel] fused distillation LoRA %s (weight=%s) into %s",
                 repo, weight, type(pipe).__name__)
    except Exception as e:
        log.warning("[accel] failed to fuse acceleration LoRA %s: %s — generating "
                    "without acceleration", lora_ref, e)


def accel_call_defaults(accel: Optional[dict]) -> dict:
    """Return ``{num_inference_steps, guidance_scale}`` from the accel preset.

    Only includes keys whose value is set. Callers should apply these only when
    the request itself did not specify steps/guidance (request always wins).
    """
    if not accel:
        return {}
    out = {}
    if accel.get("steps") is not None:
        out["num_inference_steps"] = int(accel["steps"])
    if accel.get("guidance_scale") is not None:
        out["guidance_scale"] = float(accel["guidance_scale"])
    return out
