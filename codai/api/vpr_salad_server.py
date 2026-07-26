#!/usr/bin/env python3
# CoderAI - DINOv2-SALAD visual-place-recognition embedding server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
# GPLv3 (see the main project LICENSE).
"""Run DINOv2-SALAD as an ISOLATED subprocess so its heavy deps
(pytorch_lightning, pytorch_metric_learning) never enter the main engine venv —
they live only in a --system-site-packages venv this script is launched with.

Protocol (identical to the dinov2-embed server, so the embeddings backend reuses
the same reader): read one image FILE PATH per line on stdin, write one JSON line
per input on stdout —
    {"ready": true, "device": "cuda", "dim": 8448}   # once, after the model loads
    {"embedding": [ ... floats ... ]}                 # per image, L2-normalised
    {"error": "..."}                                  # per image, on failure

GPU is preferred; it falls back to CPU only when CUDA is unavailable or the GPU
load fails (set SALAD_FORCE_CPU=1 to force CPU).
"""
import json
import os
import sys


def main():
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import torchvision.transforms as T

    os.environ.setdefault(
        'TORCH_HOME', os.environ.get('CODERAI_TORCH_HOME', '/cache/torchhub'))
    force_cpu = os.environ.get('SALAD_FORCE_CPU') == '1'
    want_gpu = torch.cuda.is_available() and not force_cpu

    def _load(dev):
        return torch.hub.load('serizba/salad', 'dinov2_salad').eval().to(dev)

    device = 'cuda' if want_gpu else 'cpu'
    try:
        model = _load(device)
    except Exception:
        # GPU is preferred, CPU is the fallback when GPU load isn't possible.
        if device != 'cpu':
            device = 'cpu'
            model = _load(device)
        else:
            raise

    # DINOv2 backbone: side must be a multiple of the patch size (14). 322 = 14*23
    # is the SALAD reference eval resolution. ImageNet normalisation.
    size = int(os.environ.get('SALAD_IMAGE_SIZE', '322'))
    tfm = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Announce readiness (and the true descriptor width) so the parent can proceed.
    with torch.no_grad():
        _probe = model(torch.zeros(1, 3, size, size, device=device))
    dim = int(_probe.shape[-1])
    sys.stdout.write(json.dumps({"ready": True, "device": device, "dim": dim}) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        path = line.strip()
        if not path:
            continue
        try:
            im = Image.open(path).convert('RGB')
            x = tfm(im).unsqueeze(0).to(device)
            with torch.no_grad():
                v = F.normalize(model(x), dim=-1)[0]
            sys.stdout.write(json.dumps({"embedding": v.detach().cpu().tolist()}) + "\n")
        except Exception as e:  # noqa: BLE001 — report, keep serving
            sys.stdout.write(json.dumps({"error": str(e)}) + "\n")
        sys.stdout.flush()


if __name__ == '__main__':
    main()
