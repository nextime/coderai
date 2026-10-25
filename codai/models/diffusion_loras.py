# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""LoRA adapters a DIFFUSION model's own config asks for.

Three kinds of LoRA can be active on one image/video pipeline at once, and
until now only two of them existed:

* the **acceleration** LoRA (Lightning / Turbo / LCM), from the per-model
  ``acceleration`` block — see ``codai.models.acceleration``;
* a **per-request** LoRA, sent by the caller (a character, an environment);
* a **per-model** LoRA, which is what this module adds: adapters that are part
  of what the model *is*, named in the model's own config.

The third is how one set of weights serves several published model ids. A base
checkpoint plus a style LoRA at 0.8 is a different model to its users: it gets
its own id, its own defaults and its own entry in ``/v1/models``, while sharing
the base weights in VRAM with its siblings.

Config (the same shape text models already use, so one idiom covers both)::

    "loras": [{"model": "org/some-lora", "weight": 0.8, "name": "style"},
              {"model": "org/other:file.safetensors", "weight": 1.0}]

A single ``lora_path`` (+ ``lora_scale``) is honoured too: the field has been in
the model config and the admin UI for a long time, and on the diffusers image
path nothing ever read it, so a LoRA configured there did nothing at all.

WHY A FAILURE HERE RAISES. An acceleration LoRA that will not load costs
speed — the caller gets a slower picture, which is why that path logs and
carries on. A per-model LoRA costs *identity*: an id published as a particular
fine-tune, serving the base checkpoint instead, answers 200 with a picture that
is not what was asked for and no indication anything is wrong. That is the
failure mode nobody notices, so it is the one that must be loud.
"""

import logging
from typing import List, Optional, Tuple

from codai.models.acceleration import _split_lora_ref
from codai.models.text_loras import configured_specs

log = logging.getLogger(__name__)

# Namespaced so these adapters can never collide with a per-request adapter
# (plain name) or an acceleration one ("__accel__"), and so they can be told
# apart on a pipeline that is reused across requests.
ADAPTER_PREFIX = "__model_"


def configured(model_cfg: Optional[dict]) -> List[dict]:
    """The per-model LoRA specs for a diffusion model, in config order."""
    return configured_specs(model_cfg or {}, include_dirs=False)


def describe(specs: List[dict]) -> str:
    return ", ".join(f"{s['name']}@{s['weight']:g}" for s in specs)


def persistent_adapters(pipe) -> Tuple[List[str], List[float]]:
    """The adapters that must stay active across per-request LoRA swaps.

    Both the acceleration and the per-model adapters belong to the pipeline
    rather than to any one request. ``set_adapters`` replaces the whole active
    set, so every caller that activates a request's adapters has to re-include
    these or it silently turns the model's own identity off.
    """
    names: List[str] = []
    weights: List[float] = []

    accel = list(getattr(pipe, "_coderai_accel_adapters", None) or [])
    if accel:
        w = float(getattr(pipe, "_coderai_accel_weight", 1.0) or 1.0)
        names.extend(accel)
        weights.extend([w] * len(accel))

    model = list(getattr(pipe, "_coderai_model_adapters", None) or [])
    if model:
        mw = list(getattr(pipe, "_coderai_model_weights", None) or [])
        names.extend(model)
        # A missing weight defaults to 1.0 rather than dropping the adapter.
        weights.extend([float(mw[i]) if i < len(mw) else 1.0
                        for i in range(len(model))])

    return names, weights


def apply_model_loras(pipe, model_cfg: Optional[dict]) -> List[str]:
    """Load this model's own LoRA adapters onto a freshly loaded pipeline.

    Returns the adapter names registered (empty when none are configured).
    Raises RuntimeError when a configured adapter cannot be applied.
    """
    specs = configured(model_cfg)
    if not specs:
        return []

    if not hasattr(pipe, "load_lora_weights"):
        raise RuntimeError(
            f"model is configured with LoRA(s) {describe(specs)} but pipeline "
            f"{type(pipe).__name__} has no load_lora_weights")

    # gptqmodel 7.1.0 renamed the AWQ class peft imports during dispatch; alias
    # it before any load_lora_weights so the import cannot crash the load.
    try:
        from codai.models.peft_compat import ensure_peft_awq_compat
        ensure_peft_awq_compat()
    except Exception:
        pass

    # A pipeline object can be re-finalised (a second config of the same
    # checkpoint, a reload after an eviction). Drop what we added last time so
    # the load cannot collide with "Adapter name already in use".
    prev = list(getattr(pipe, "_coderai_model_adapters", None) or [])
    if prev:
        try:
            pipe.delete_adapters(prev)
        except Exception:
            pass
    pipe._coderai_model_adapters = []
    pipe._coderai_model_weights = []

    names: List[str] = []
    weights: List[float] = []
    for spec in specs:
        repo, weight_name = _split_lora_ref(spec["source"])
        name = ADAPTER_PREFIX + spec["name"]
        kw = {"adapter_name": name}
        if weight_name:
            kw["weight_name"] = weight_name
        try:
            try:
                pipe.delete_adapters(name)
            except Exception:
                pass
            pipe.load_lora_weights(repo, **kw)
        except Exception as e:
            raise RuntimeError(
                f"could not load this model's LoRA '{spec['source']}' "
                f"(weight {spec['weight']:g}): {e}") from e
        names.append(name)
        weights.append(float(spec["weight"]))

    pipe._coderai_model_adapters = list(names)
    pipe._coderai_model_weights = list(weights)

    # Activate the model's adapters TOGETHER WITH the acceleration ones: an
    # id like "<base>-NSFW" is commonly the base plus a style LoRA plus the
    # distill LoRA, and activating only one group turns the other off.
    all_names, all_weights = persistent_adapters(pipe)
    try:
        pipe.set_adapters(all_names, all_weights)
    except Exception as e:
        raise RuntimeError(
            f"could not activate this model's LoRA(s) {all_names}: {e}") from e

    log.info("[image][loras] model LoRA(s) active: %s%s", describe(specs),
             " (with acceleration adapters)"
             if getattr(pipe, "_coderai_accel_adapters", None) else "")
    return names
