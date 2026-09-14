# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""LoRA adapters for TEXT models — resolution shared by every text path.

Image and video models have had LoRA support for a long time (diffusers'
``load_lora_weights``). Text models did not: ``lora_path`` and ``lora_model_dir``
existed in the model config, but only the VRAM estimator ever read them, so
configuring a LoRA on a text model silently did nothing.

The three text paths each apply adapters their own way, and there is no common
library between them:

* GGUF via llama-cpp-python — ``lora_path`` + ``lora_scale`` on the Llama object;
  the adapter must be a **GGUF-converted** adapter, not a PEFT safetensors.
* HuggingFace via transformers — PEFT: ``load_adapter`` on the loaded model.
* vLLM — ``--enable-lora --lora-modules name=<source>`` at launch, where the
  source is a path on the server or a HuggingFace repo id.

What they share is the question "which adapters does this model want, and where
are they", which is what this module answers. Everything downstream is the
backend's own business.

A QLoRA adapter needs nothing special here: the quantisation describes how the
BASE model is loaded (4-bit), while the adapter itself is an ordinary LoRA. So a
QLoRA trained against a 4-bit base loads like any other, provided the base is
loaded the way it was trained.
"""

import os
from typing import List


def _as_list(value) -> list:
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def configured_specs(config: dict) -> List[dict]:
    """The LoRA adapters a model's config asks for, normalised.

    Accepts every shape the config has grown:
      ``lora_path``        one path/repo id, or a list of them
      ``lora_model_dir``   a directory of adapters (each subdirectory is one)
      ``loras``            a list of {path|id|model, weight, name} dicts

    Returns ``[{"source": str, "weight": float, "name": str}, …]``.
    """
    if not isinstance(config, dict):
        return []
    out: List[dict] = []

    for raw in _as_list(config.get("loras")):
        if isinstance(raw, dict):
            src = (raw.get("path") or raw.get("source") or raw.get("model")
                   or raw.get("id") or "")
            if not src:
                continue
            out.append({"source": str(src),
                        "weight": float(raw.get("weight", 1.0) or 1.0),
                        "name": str(raw.get("name") or _default_name(str(src)))})
        elif raw:
            out.append({"source": str(raw), "weight": 1.0,
                        "name": _default_name(str(raw))})

    for raw in _as_list(config.get("lora_path")):
        if raw:
            out.append({"source": str(raw),
                        "weight": float(config.get("lora_scale", 1.0) or 1.0),
                        "name": _default_name(str(raw))})

    for d in _as_list(config.get("lora_model_dir")):
        d = os.path.expanduser(str(d or ""))
        if not d or not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            sub = os.path.join(d, name)
            # A directory of PEFT adapters, or a directory of adapter files.
            if os.path.isdir(sub) or name.endswith((".safetensors", ".bin", ".gguf")):
                out.append({"source": sub,
                            "weight": float(config.get("lora_scale", 1.0) or 1.0),
                            "name": _default_name(sub)})

    # Keep the first mention of each source: the config shapes overlap, and
    # loading the same adapter twice would double its effect.
    seen, unique = set(), []
    for spec in out:
        key = spec["source"]
        if key in seen:
            continue
        seen.add(key)
        unique.append(spec)
    return unique


def _default_name(source: str) -> str:
    base = os.path.basename(str(source).rstrip("/")) or str(source)
    for suffix in (".safetensors", ".bin", ".gguf"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return base or "lora"


def local_path(source: str) -> str:
    """The adapter as a local path, or '' when it is a repo id / not present."""
    p = os.path.expanduser(str(source or ""))
    return p if p and os.path.exists(p) else ""


def gguf_adapter(spec: dict) -> str:
    """The GGUF adapter file for llama.cpp, or ''.

    llama-cpp-python takes ONE adapter file and it must be GGUF-converted; a PEFT
    safetensors directory is not loadable there. Returning '' lets the caller say
    so clearly rather than have llama.cpp fail deep inside a load.
    """
    path = local_path(spec.get("source", ""))
    if not path:
        return ""
    if os.path.isfile(path) and path.endswith(".gguf"):
        return path
    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            if name.endswith(".gguf"):
                return os.path.join(path, name)
    return ""


def describe(specs: List[dict]) -> str:
    return ", ".join(f"{s['name']}@{s['weight']:g}" for s in specs) or "(none)"
