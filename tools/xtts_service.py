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


#: HuggingFace repo id -> the name coqui's own catalogue uses. coqui's TTS(name)
#: splits on "/" and wants exactly four parts (type/lang/dataset/model); a HF
#: id has two, and fails with "not enough values to unpack (expected 4, got 2)".
#: The rest of coderai names models by HF id, so the translation lives here.
_HF_TO_COQUI = {
    "coqui/xtts-v2": "tts_models/multilingual/multi-dataset/xtts_v2",
    "coqui/xtts-v1.1": "tts_models/multilingual/multi-dataset/xtts_v1.1",
}


def _coqui_name(name: str) -> str:
    n = (name or "").strip()
    if n.count("/") == 3:
        return n                                  # already coqui's own naming
    mapped = _HF_TO_COQUI.get(n.lower())
    if mapped:
        return mapped
    # Match on the LAST path segment, so an org-less alias is recognised too.
    # The previous test required a "/", which is exactly the case that does not
    # apply to a card registered as "XTTS-v2": coqui then split a name with no
    # separator and reported "not enough values to unpack (expected 4, got 1)",
    # an error about its own four-part naming that says nothing about the alias
    # that caused it. The caller passes the card's path now, but the worker is
    # reached by other paths too and must not depend on that.
    tail = n.rsplit("/", 1)[-1].lower().replace("_", "-")
    if "xtts" in tail:
        if "v1.1" in tail or "v1-1" in tail:
            return "tts_models/multilingual/multi-dataset/xtts_v1.1"
        # An unknown XTTS repo: the v2 architecture is the safe assumption.
        return "tts_models/multilingual/multi-dataset/xtts_v2"
    return n


def _load(name: str) -> dict:
    import torch
    from TTS.api import TTS
    tts = TTS(_coqui_name(name)).to("cuda" if torch.cuda.is_available() else "cpu")
    _state["tts"] = tts
    # XTTS reports its output rate on the synthesizer; fall back to its default.
    try:
        _state["sr"] = int(tts.synthesizer.output_sample_rate)
    except Exception:
        _state["sr"] = 24000
    return {"ok": True, "speakers": list(getattr(tts, "speakers", None) or []),
            "sample_rate": _state["sr"]}


#: XTTS-v2's own language codes. Duplicated from codai.api.tts_backends on
#: purpose: this worker runs in its own venv with coqui's dependency set and
#: cannot import the server package. tests/test_tts_languages_are_published.py
#: asserts the two lists stay identical.
_XTTS_LANGUAGES = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl",
                   "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi"]


def _xtts_language(lang) -> str:
    """``lang`` as XTTS names it -- see the note on tts_backends.xtts_language.

    The bug this replaces: ``lang[:2]`` turned "zh-cn", the one XTTS code longer
    than two characters, into "zh", and XTTS raised ``KeyError: 'zh'``.
    """
    want = str(lang or "en").strip().lower().replace("_", "-")
    if want in _XTTS_LANGUAGES:
        return want
    base = want.split("-")[0]
    if base in _XTTS_LANGUAGES:
        return base
    for code in _XTTS_LANGUAGES:
        if code.split("-")[0] == base:
            return code
    return base


def _tts(req: dict) -> dict:
    tts = _state["tts"]
    if tts is None:
        return {"ok": False, "error": "no model loaded"}
    kwargs = {"text": req.get("text") or "", "language": _xtts_language(req.get("language"))}
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
    # stdout is the PROTOCOL: one JSON object per line. coqui prints a licence
    # prompt and progress to stdout during load, which the caller then tried to
    # parse — "unparseable worker reply". So the real stdout is kept for the
    # protocol and everything else the process prints goes to stderr, where the
    # caller's log tail can still see it.
    import os
    proto = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    sys.stdout = sys.stderr
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
        proto.write(json.dumps(out) + "\n")
        proto.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
