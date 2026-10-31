#!/usr/bin/env python3
# CoderAI - jina-clip-v2 embedding server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
# GPLv3 (see the main project LICENSE).
"""Run jina-clip-v2 in its OWN venv, because the server's transformers 5 makes it
return NaNs.

Launched by :mod:`codai.api.jinaclip_worker`, which owns the venv. The protocol
is one JSON object per line on stdin, one per line on stdout -- the same shape as
the dinov2/SALAD servers, extended because this model embeds text as well as
images and the two share one vector space:

    {"ready": true, "device": "cuda", "dim": 1024, "transformers": "4.49.0"}

    <- {"texts": ["una fotografia", "a photograph"]}
    -> {"embeddings": [[...], [...]]}
    <- {"images": ["/tmp/a.png"]}
    -> {"embeddings": [[...]]}
    -> {"error": "..."}

Images are passed as FILE PATHS, not bytes: the parent already writes PIL images
to a temp file for the other embed servers, and a path keeps this protocol
line-based.

Vectors are L2-normalised here. jina returns them normalised already, but a
caller comparing vectors from two backends must not have to know which ones did.
"""
import json
import os
import sys


def _say(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main():
    import numpy as np
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoModel

    model_id = os.environ.get("JINACLIP_MODEL") or "jinaai/jina-clip-v2"
    force_cpu = os.environ.get("JINACLIP_FORCE_CPU") == "1"
    want_gpu = torch.cuda.is_available() and not force_cpu
    device = "cuda" if want_gpu else "cpu"

    # NEITHER `dtype=` nor `torch_dtype=`. This model's remote code forwards
    # **kwargs into its own __init__, so each name fails on the stack that does
    # not define it -- and the two stacks disagree:
    #
    #   transformers 5.12: torch_dtype=torch.bfloat16
    #     -> TypeError: attribute name must be string, not 'torch.dtype'
    #   transformers 4.49: dtype=torch.bfloat16
    #     -> TypeError: JinaCLIPModel.__init__() got an unexpected keyword
    #        argument 'dtype'
    #
    # Passing nothing is not a workaround: the checkpoint's own config.json
    # names bfloat16, which is the dtype the verified run loaded in
    # ("loaded; dtype torch.bfloat16"), and bf16 is what its EVA-style vision
    # tower needs -- in fp16 the tower overflows and every image vector comes
    # back NaN.
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    try:
        model = model.to(device).eval()
    except Exception:
        if device != "cpu":
            device = "cpu"
            model = model.to(device).eval()
        else:
            raise

    # The width is announced rather than assumed: jina-clip-v2 is 1024-d, but a
    # caller that truncates Matryoshka-style needs the real number, and a
    # checkpoint swap must not silently change what the parent believes.
    with torch.no_grad():
        probe = model.encode_text(["dimension probe"])
    dim = int(np.asarray(probe).shape[-1])
    _say({"ready": True, "device": device, "dim": dim,
          "transformers": transformers.__version__})

    def _norm(rows):
        out = []
        for v in np.asarray(rows, dtype=np.float32):
            n = float(np.linalg.norm(v)) or 1.0
            out.append([float(x) for x in (v / n)])
        return out

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
            with torch.no_grad():
                if req.get("texts") is not None:
                    rows = model.encode_text(list(req["texts"]))
                elif req.get("images") is not None:
                    imgs = [Image.open(p).convert("RGB") for p in req["images"]]
                    rows = model.encode_image(imgs)
                else:
                    _say({"error": "request names neither 'texts' nor 'images'"})
                    continue
            vecs = _norm(rows)
            # Refused here as well as in the parent. This process exists because
            # the model returned NaNs on another stack; shipping them from the
            # one place that knows why would be perverse.
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
        _say({"error": f"{type(exc).__name__}: {exc}"})
        sys.exit(1)
