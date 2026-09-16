#!/usr/bin/env python3
# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""coqui XTTS in its own venv, spoken to over stdin/stdout.

coqui-tts imports a transformers 4.x symbol (isin_mps_friendly) that 5.x removed,
and the server runs 5.x — so it cannot share the main venv. This script is the
far side of the split: one JSON object per line in, one per line out, the same
shape audiocraft_service.py and pyannote_service.py use.

  {"op": "load", "model": "coqui/XTTS-v2"}                       -> {"ok": true, "speakers": [...], "sample_rate": 24000}
  {"op": "tts", "text": "...", "language": "en", "speaker": "..",
   "speaker_wav": "/path/or/absent", "speed": 1.0}                -> {"ok": true, "b64_wav": "...", "sample_rate": 24000}
  {"op": "ping"}                                                   -> {"ok": true}

Audio returns as base64 WAV: the caller may be a pod, and a few seconds of
speech is small enough that a file would only be a thing to clean up.
"""

import base64
import io
import json
import sys

_state = {"tts": None, "sr": 24000}


def _load(name: str) -> dict:
    import torch
    from TTS.api import TTS
    tts = TTS(name).to("cuda" if torch.cuda.is_available() else "cpu")
    _state["tts"] = tts
    # XTTS reports its output rate on the synthesizer; fall back to its default.
    try:
        _state["sr"] = int(tts.synthesizer.output_sample_rate)
    except Exception:
        _state["sr"] = 24000
    return {"ok": True, "speakers": list(getattr(tts, "speakers", None) or []),
            "sample_rate": _state["sr"]}


def _tts(req: dict) -> dict:
    tts = _state["tts"]
    if tts is None:
        return {"ok": False, "error": "no model loaded"}
    kwargs = {"text": req.get("text") or "", "language": (req.get("language") or "en")[:2]}
    for key in ("speaker", "speaker_wav"):
        if req.get(key):
            kwargs[key] = req[key]
    speed = req.get("speed")
    try:
        if speed:
            kwargs["speed"] = float(speed)
        wav = tts.tts(**kwargs)
    except TypeError:
        kwargs.pop("speed", None)          # some coqui models do not take speed
        wav = tts.tts(**kwargs)
    import numpy as np
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, np.asarray(wav, dtype="float32"), _state["sr"], format="WAV")
    return {"ok": True, "b64_wav": base64.b64encode(buf.getvalue()).decode(),
            "sample_rate": _state["sr"]}


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
            elif op == "tts":
                out = _tts(req)
            else:
                out = {"ok": False, "error": f"unknown op {op!r}"}
        except Exception as exc:
            # One bad request must not take the loaded model down with it.
            out = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
