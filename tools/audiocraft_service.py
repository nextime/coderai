#!/usr/bin/env python3
# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""audiocraft in its own venv, spoken to over stdin/stdout.

audiocraft hard-pins torch==2.1.0, torchvision==0.16.0 and torchtext==0.16.0.
The server runs torch 2.11 built for a specific CUDA, and there is no version of
this argument that both sides win: installing audiocraft into the server's venv
downgrades torch five major versions and leaves the GPU unusable, because torch
2.1.0 has no build for that CUDA at all.

So it lives in a venv of its own and is driven as a subprocess, exactly like the
parler-tts and OCR stacks that have the same problem. This script is the far
side: one JSON object per line in, one per line out.

  {"op": "load", "model": "facebook/musicgen-small"}   -> {"ok": true, "sample_rate": 32000}
  {"op": "generate", "prompt": "...", "duration": 2.0} -> {"ok": true, "b64_wav": "..."}
  {"op": "ping"}                                        -> {"ok": true}

Audio comes back as base64 WAV rather than a path: the caller may be a pod with
no shared filesystem, and a few seconds of audio is small enough that a file
would only add a thing to clean up.
"""

import base64
import io
import json
import sys


_state = {"model": None, "name": ""}


def _load(name: str) -> dict:
    from audiocraft.models import MusicGen, AudioGen
    if "audiogen" in name.lower():
        model = AudioGen.get_pretrained(name)
    else:
        model = MusicGen.get_pretrained(name)
    _state["model"], _state["name"] = model, name
    return {"ok": True, "sample_rate": int(model.sample_rate)}


def _generate(req: dict) -> dict:
    model = _state["model"]
    if model is None:
        return {"ok": False, "error": "no model loaded"}
    params = {"duration": float(req.get("duration") or 10.0)}
    for key in ("top_k", "top_p", "temperature", "cfg_coef"):
        if req.get(key) is not None:
            params[key] = req[key]
    model.set_generation_params(**params)

    melody_b64 = req.get("melody_wav")
    if melody_b64:
        # Melody conditioning is the reason to keep audiocraft at all —
        # transformers' MusicGen has no equivalent.
        import torch, torchaudio
        raw = base64.b64decode(melody_b64)
        wav, sr = torchaudio.load(io.BytesIO(raw))
        out = model.generate_with_chroma([req.get("prompt") or ""],
                                         wav.unsqueeze(0), sr)
    else:
        out = model.generate([req.get("prompt") or ""])

    import torchaudio
    buf = io.BytesIO()
    torchaudio.save(buf, out[0].cpu(), int(model.sample_rate), format="wav")
    return {"ok": True, "b64_wav": base64.b64encode(buf.getvalue()).decode(),
            "sample_rate": int(model.sample_rate)}


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            op = req.get("op")
            if op == "ping":
                out = {"ok": True}
            elif op == "load":
                out = _load(str(req.get("model") or ""))
            elif op == "generate":
                out = _generate(req)
            else:
                out = {"ok": False, "error": f"unknown op {op!r}"}
        except Exception as exc:
            # Never die on one bad request: the caller keeps the process for the
            # next one, and a crash here costs a model load.
            out = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
