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
        # Wan2.2 A14B is a two-expert MoE: the distill LoRA must be fused into BOTH
        # the high-noise (transformer) and low-noise (transformer_2) experts, or the
        # clip collapses to a solid colour at 4 steps. These default to the locally
        # installed lightx2v/Wan2.2-Lightning weights (resolved from cache — not a
        # download); override per model in the Acceleration config for T2V or a
        # different rank/version. `lora` stays None because the two experts differ.
        "lora": None,
        "lora_high": "lightx2v/Wan2.2-Lightning:Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors",
        "lora_low": "lightx2v/Wan2.2-Lightning:Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors",
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
        # Wan2.2 A14B is a two-expert MoE: the distill LoRA differs for the
        # high-noise (transformer) and low-noise (transformer_2) experts. When
        # these are set they take precedence over the single `lora` per expert;
        # otherwise the single `lora` is applied to BOTH experts.
        "lora_high": _pick("lora_high"),
        "lora_low": _pick("lora_low"),
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
    #    `_coderai_accel_fused` records whether a distill LoRA actually baked in,
    #    so the generator only drops to the preset's low step count when the model
    #    is genuinely distilled (running 4 steps un-distilled collapses the video
    #    to a solid colour — exactly the Wan2.2 dual-expert failure mode).
    try:
        pipe._coderai_accel_fused = False
    except Exception:
        pass

    has_t2 = getattr(pipe, "transformer_2", None) is not None
    lora_high = accel.get("lora_high") or accel.get("lora")
    lora_low = accel.get("lora_low") or accel.get("lora")
    if not lora_high and not lora_low:
        # No LoRA (e.g. a full distilled model like SDXL-Turbo) — treat as distilled.
        try:
            pipe._coderai_accel_fused = True
        except Exception:
            pass
        return
    if not hasattr(pipe, "load_lora_weights"):
        log.warning("[accel] pipeline %s has no load_lora_weights — cannot fuse "
                    "acceleration LoRA", type(pipe).__name__)
        return

    # gptqmodel 7.1.0 renamed the AWQ class peft imports during dispatch; alias
    # it before any load_lora_weights so the import doesn't crash the fuse.
    try:
        from codai.models.peft_compat import ensure_peft_awq_compat
        ensure_peft_awq_compat()
    except Exception:
        pass

    weight = float(accel.get("lora_weight") or 1.0)

    def _load_one(ref, into_t2: bool, adapter: str) -> bool:
        repo, weight_name = _split_lora_ref(ref)
        kw = {"adapter_name": adapter}
        if weight_name:
            kw["weight_name"] = weight_name
        if into_t2:
            kw["load_into_transformer_2"] = True
        pipe.load_lora_weights(repo, **kw)
        return True

    loaded_adapters = []
    try:
        # High-noise expert (transformer) — always.
        if lora_high and _load_one(lora_high, False, "__accel__"):
            loaded_adapters.append("__accel__")
        # Low-noise expert (transformer_2) — only on dual-expert Wan2.2 models.
        if has_t2 and lora_low:
            try:
                if _load_one(lora_low, True, "__accel_2__"):
                    loaded_adapters.append("__accel_2__")
            except Exception as e2:
                log.warning("[accel] could not load distill LoRA into transformer_2 "
                            "(%s) — low-noise expert stays un-distilled", e2)
        elif has_t2 and not lora_low:
            log.warning("[accel] model has a second expert (transformer_2) but no "
                        "low-noise distill LoRA — set acceleration.lora_low")
        if not loaded_adapters:
            raise RuntimeError("no distill adapter registered on the pipeline")
        try:
            pipe.set_adapters(loaded_adapters, [weight] * len(loaded_adapters))
        except Exception as e:
            log.warning("[accel] could not activate distill adapters %s: %s",
                        loaded_adapters, e)
        # Keep the distill LoRA(s) as ACTIVE RUNTIME ADAPTERS rather than fusing.
        #
        # Fusing merges the adapter into the base weights. For a bitsandbytes 4-bit
        # model under CPU/disk offload the weights live on the CPU, so fuse_lora has
        # to dequantize → merge → requantize every Linear on the CPU — minutes-to-
        # hours per 14B expert, which looks exactly like a hang (high CPU, empty
        # VRAM, no progress). Runtime adapters instead apply at forward time on
        # whichever device the module is currently on (the GPU during the pass), at
        # negligible cost, and `set_adapters` natively covers transformer_2 — so the
        # low-noise expert is distilled too (the old fuse path needed an explicit
        # components=[...,"transformer_2"] or it collapsed to a solid colour).
        #
        # `_sync_video_loras` preserves these accel adapters across per-request LoRA
        # swaps and re-includes them in every set_adapters call.
        pipe._coderai_accel_adapters = list(loaded_adapters)
        pipe._coderai_accel_weight = weight
        try:
            pipe._coderai_accel_fused = True  # accel is effective (distilled preset applies)
        except Exception:
            pass
        log.info("[accel] activated distillation LoRA(s) %s (weight=%s, runtime adapters) "
                 "on %s%s", loaded_adapters, weight, type(pipe).__name__,
                 " (both experts)" if len(loaded_adapters) > 1 else "")
    except Exception as e:
        log.warning("[accel] failed to fuse acceleration LoRA (high=%s low=%s): %s "
                    "— generating WITHOUT acceleration (step count will fall back to "
                    "a safe default, not the preset's distilled count)",
                    lora_high, lora_low, e)


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
