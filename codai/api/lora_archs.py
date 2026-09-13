# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Which architecture a LoRA is being trained for, and what that implies.

`codai/api/loras.py` grew one trainer per architecture (SD1.x, SDXL, Z-Image,
Wan). The newer targets — LTX-2, MiniMax-H3, Krea 2 — are all the same shape as
Wan: a flow-matching transformer over VAE latents, conditioned on a text
embedding. So rather than a fourth, fifth and sixth copy of that loop, each one is
described here and driven by the shared trainer.

Each entry says what to import, what the latents look like, and how strongly to
shift the rectified-flow timestep. Nothing here imports diffusers at module level:
an architecture whose classes this diffusers build doesn't have must fail with a
clear message at TRAIN time, not stop the server from starting.
"""

from typing import Optional

#: target -> description. `detect` matches a model path/id (lowercased).
ARCHS = {
    "wan": {
        "label": "Wan video DiT",
        "kind": "video",
        "detect": ("wan2.1", "wan2.2", "wan-t2v", "wan-i2v", "wanvace", "wan_"),
        "transformer": "WanTransformer3DModel",
        "vae": "AutoencoderKLWan",
        "text_encoder": ("transformers", "UMT5EncoderModel"),
        "shift": 5.0,
        "note": "Handled by the dedicated _train_wan (dual-expert aware).",
    },
    "ltx2": {
        "label": "LTX-2 video DiT",
        "kind": "video",
        "detect": ("ltx-2", "ltx2", "ltx-video-2"),
        "transformer": "LTX2VideoTransformer3DModel",
        "vae": "AutoencoderKLLTX2Video",
        "text_encoder": ("transformers", "T5EncoderModel"),
        "tokenizer": ("transformers", "AutoTokenizer"),
        "max_text_len": 256,
        "shift": 3.0,
    },
    "h3": {
        "label": "MiniMax-H3 omni DiT",
        "kind": "video",
        "detect": ("minimax-h3", "minimax_h3"),
        "transformer": "MiniMaxH3Transformer3DModel",
        "vae": "AutoencoderKLMiniMaxH3",
        "text_encoder": ("transformers", "AutoModel"),
        "tokenizer": ("transformers", "AutoTokenizer"),
        "max_text_len": 512,
        # H3 carries separate video/audio flow shifts (12 / 3); training stills we
        # only drive the video branch.
        "shift": 12.0,
        "needs": "diffusers>=0.40",
    },
    "krea": {
        "label": "Krea 2 image DiT",
        "kind": "image",
        "detect": ("krea-2", "krea2", "krea_2"),
        "transformer": "Krea2Transformer2DModel",
        "vae": "AutoencoderKL",
        "text_encoder": ("transformers", "AutoModel"),
        "tokenizer": ("transformers", "AutoTokenizer"),
        "max_text_len": 512,
        "shift": 3.0,
        "needs": "diffusers>=0.40 and access to the gated krea/Krea-2-* weights",
    },
}

#: What `target` may be on a train request.
TARGETS = ("image", "video", "wan", "ltx2", "h3", "krea")


def detect_arch(model_path: str, target: str = "") -> Optional[str]:
    """Architecture key for a base model, or None when it isn't one of the new ones.

    An explicit target wins; otherwise the model path/id is matched. `video` stays
    a synonym for Wan so existing configs keep meaning what they meant.
    """
    t = (target or "").strip().lower()
    if t in ARCHS:
        return t
    if t == "video":
        hay = (model_path or "").lower()
        for key, spec in ARCHS.items():
            if key == "wan":
                continue
            if any(d in hay for d in spec["detect"]):
                return key
        return "wan"
    hay = (model_path or "").lower()
    for key, spec in ARCHS.items():
        if any(d in hay for d in spec["detect"]):
            return key
    return None


def load_classes(arch: str):
    """Import the classes this architecture trains with.

    Raises RuntimeError naming the missing piece — these builds differ (LTX-2 is in
    diffusers 0.38, H3 and Krea 2 need 0.40), and a bare ImportError deep in a
    training job tells nobody anything.
    """
    spec = ARCHS.get(arch)
    if not spec:
        raise RuntimeError(f"Unknown LoRA architecture '{arch}'")
    import importlib
    diffusers = importlib.import_module("diffusers")
    missing = [n for n in (spec["transformer"], spec["vae"]) if not hasattr(diffusers, n)]
    if missing:
        need = spec.get("needs") or "a newer diffusers"
        raise RuntimeError(
            f"{spec['label']} LoRA training needs {', '.join(missing)}, which this "
            f"diffusers ({getattr(diffusers, '__version__', '?')}) does not provide. "
            f"Requires {need}.")
    tr_cls = getattr(diffusers, spec["transformer"])
    vae_cls = getattr(diffusers, spec["vae"])
    te_mod, te_name = spec["text_encoder"]
    tok_mod, tok_name = spec.get("tokenizer", ("transformers", "AutoTokenizer"))
    te_cls = getattr(importlib.import_module(te_mod), te_name)
    tok_cls = getattr(importlib.import_module(tok_mod), tok_name)
    return tr_cls, vae_cls, te_cls, tok_cls, spec
