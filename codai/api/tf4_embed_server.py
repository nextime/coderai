#!/usr/bin/env python3
# CoderAI - embedding server for models that need transformers 4
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
# GPLv3 (see the main project LICENSE).
"""Run an embedder in the transformers-4.49 venv, because transformers 5 cannot.

Launched by :mod:`codai.api.tf4_embedders`, which owns the venv and decides the
quantization. Two families, because the two models expose different APIs:

* ``jinaclip``  — ``encode_text(list)`` / ``encode_image(list)``
* ``mmembed``   — one ``encode([{txt|img: …}])`` returning ``{'hidden_states':…}``

Protocol: one JSON object per line on stdin, one per line on stdout -- the shape
the dinov2/SALAD servers use, extended with texts because these models embed
text and images into one space:

    {"ready": true, "device": "cuda", "dim": 1024, "transformers": "4.49.0"}

    <- {"texts": ["una fotografia", "a photograph"]}
    -> {"embeddings": [[...], [...]]}
    <- {"images": ["/tmp/a.png"]}
    -> {"embeddings": [[...]]}
    <- {"texts": ["query"], "instruction": "Retrieve a passage that answers…"}
    -> {"embeddings": [[...]]}
    -> {"error": "..."}

Images arrive as FILE PATHS, not bytes: the parent already writes PIL images to
a temp file for the other embed servers, and a path keeps this protocol
line-based.

Vectors are L2-normalised here. Both models return them normalised already, but
a caller comparing vectors from two backends must not have to know which ones
did.
"""
import json
import os
import sys


def _say(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _quantization_config(spec, torch):
    """BitsAndBytesConfig from what the PARENT decided, or None.

    The child cannot import coderai to read models.json, so the policy arrives
    as JSON -- including the module names to leave alone, which for a
    vision-language embedder is the part that matters: the tower turns pixels
    into the vector being compared and the projector is one small matrix whose
    error lands on every image.
    """
    if not spec or not (spec.get("load_in_4bit") or spec.get("load_in_8bit")):
        return None
    from transformers import BitsAndBytesConfig
    skip = list(spec.get("skip_modules") or []) or None
    dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16,
              "fp32": torch.float32}
    compute = dtypes.get(str(spec.get("compute_dtype") or "bf16"),
                         torch.bfloat16)
    extra = {"llm_int8_skip_modules": skip} if skip else {}
    if spec.get("load_in_4bit"):
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=compute,
                                  bnb_4bit_use_double_quant=True, **extra)
    return BitsAndBytesConfig(load_in_8bit=True, **extra)


def _load(model_id, family, device, quant, torch):
    """The model, on `device`, by whichever route its family needs."""
    from transformers import AutoModel

    # NEITHER `dtype=` nor `torch_dtype=` for jina-clip: its remote code
    # forwards **kwargs into its own __init__, so each name fails on the stack
    # that does not define it (5.12 rejects torch_dtype as "attribute name must
    # be string"; 4.49 rejects dtype as an unexpected keyword). The
    # checkpoint's config.json already names bfloat16, which is the dtype the
    # verified run loaded in and what its EVA-style vision tower needs -- in
    # fp16 the tower overflows and every image vector comes back NaN.
    kw = {"trust_remote_code": True}
    if quant is not None:
        kw["quantization_config"] = quant
    try:
        model = AutoModel.from_pretrained(model_id, **kw)
    except Exception as exc:
        # MM-Embed's class is reachable through its config's auto_map even when
        # AutoModel refuses to map it. Kept as a fallback rather than the
        # default: on transformers 4 the ordinary route works, and when it does
        # not the reason belongs in the error.
        from transformers import AutoConfig
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        ref = (getattr(cfg, "auto_map", None) or {}).get("AutoModel")
        if not ref:
            raise
        print(f"[tf4-embed] AutoModel refused {model_id} ({type(exc).__name__}: "
              f"{exc}); loading {ref} directly", file=sys.stderr, flush=True)
        model = get_class_from_dynamic_module(ref, model_id).from_pretrained(
            model_id, **kw)
    # A quantized model is already placed by accelerate; moving it raises.
    if quant is None:
        model = model.to(device)
    return model.eval()


def main():
    import numpy as np
    import torch
    import transformers
    from PIL import Image

    model_id = os.environ.get("TF4_MODEL") or "jinaai/jina-clip-v2"
    family = os.environ.get("TF4_FAMILY") or "jinaclip"
    force_cpu = os.environ.get("TF4_FORCE_CPU") == "1"
    try:
        quant_spec = json.loads(os.environ.get("TF4_QUANTIZATION") or "null")
    except Exception:
        quant_spec = None

    want_gpu = torch.cuda.is_available() and not force_cpu
    device = "cuda" if want_gpu else "cpu"
    quant = _quantization_config(quant_spec, torch) if want_gpu else None
    fell_back = ""
    try:
        model = _load(model_id, family, device, quant, torch)
    except Exception as exc:
        if device == "cpu":
            raise
        # GPU first, CPU as the fallback -- and without quantization, which
        # bitsandbytes cannot do on the CPU anyway. The REASON travels with the
        # fallback: a model quietly running on the CPU is 30x slower and tens
        # of GB of host RAM, and "ready on cpu" alone does not say whether that
        # was asked for or fallen into.
        import gc
        import traceback
        fell_back = f"{type(exc).__name__}: {exc}"
        print(f"[tf4-embed] {model_id} would not load on the GPU "
              f"({fell_back}); retrying on the CPU\n{traceback.format_exc()}",
              file=sys.stderr, flush=True)
        # Give the card back first. A load that failed part-way has already
        # allocated, and those allocations live in this process's CUDA context
        # for the rest of its life -- measured on zeiss 2026-10-31, where this
        # model held 9756 MiB of VRAM while running on the CPU, invisible to
        # the manager's ledger and unusable by anything else.
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
        device = "cpu"
        quant = None
        model = _load(model_id, family, device, None, torch)

    def _norm(rows):
        out = []
        for v in np.asarray(rows, dtype=np.float32):
            n = float(np.linalg.norm(v)) or 1.0
            out.append([float(x) for x in (v / n)])
        return out

    def _embed(req):
        """One request's vectors, by family."""
        texts = req.get("texts")
        images = req.get("images")
        instruction = str(req.get("instruction") or "")
        if family == "jinaclip":
            if texts is not None:
                return model.encode_text(list(texts))
            return model.encode_image([Image.open(p).convert("RGB")
                                       for p in images])
        if family == "mmembed":
            if texts is not None:
                items = [{"txt": t} for t in texts]
            else:
                items = [{"img": Image.open(p).convert("RGB")} for p in images]
            kw = {"max_length": 4096}
            if instruction:
                # The query side of an asymmetric retrieval pair. Without an
                # instruction the model embeds the item as a PASSAGE, which is
                # the right default for "embed this" and the wrong one for a
                # query.
                kw["is_query"] = True
                kw["instruction"] = instruction
            out = model.encode(items, **kw)
            rows = out["hidden_states"] if isinstance(out, dict) else out
            return rows.float().cpu()
        raise ValueError(f"unknown family {family!r}")

    # The width is announced rather than assumed: a caller that truncates
    # Matryoshka-style needs the real number, and a checkpoint swap must not
    # silently change what the parent believes.
    with torch.no_grad():
        probe = _embed({"texts": ["dimension probe"]})
    dim = int(np.asarray(probe).shape[-1])
    _say({"ready": True, "device": device, "dim": dim, "family": family,
          "quantized": bool(quant), "transformers": transformers.__version__,
          **({"gpu_error": fell_back} if fell_back else {})})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as exc:
            _say({"error": f"unparseable request: {exc}"})
            continue
        try:
            if req.get("texts") is None and req.get("images") is None:
                _say({"error": "request names neither 'texts' nor 'images'"})
                continue
            with torch.no_grad():
                vecs = _norm(_embed(req))
            # Refused here as well as in the parent. This process exists
            # because a model returned NaNs on another stack; shipping them
            # from the one place that knows why would be perverse.
            if not np.isfinite(np.asarray(vecs, dtype=np.float32)).all():
                _say({"error": "the model returned non-finite values"})
                continue
            _say({"embeddings": vecs})
        except Exception as exc:
            _say({"error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:          # a load failure the parent must see
        # With the traceback. A load that fails in another process, in another
        # venv, inside a model's own remote code, is otherwise one line of
        # "TypeError: cannot unpack non-iterable NoneType object" with no file
        # and no frame -- and the next step is always to run it by hand to find
        # out where. The parent logs it; the request gets the short form.
        import traceback
        _say({"error": f"{type(exc).__name__}: {exc}",
              "traceback": traceback.format_exc()[-4000:]})
        sys.exit(1)
