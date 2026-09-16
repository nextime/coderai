# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Drive coqui XTTS from a venv of its own.

coqui-tts imports isin_mps_friendly from transformers.pytorch_utils, which 5.x
removed. The server runs 5.x, and pinning 4.x there would break every other
capability — so XTTS gets its own interpreter, like parler-tts and the OCR
engines. This module speaks to tools/xtts_service.py running in that venv.

Unlike those, XTTS had never been isolated locally: _CoquiBackend did a plain
`from TTS.api import TTS` and gave up when it was missing. On a pod that meant
an image with a verified, working TTS venv answered "pip install coqui-tts" —
the venv was there and nothing knew to use it.

The venv is found the way the other workers find theirs: an explicit env var,
then the location the pod images bake it to, then a per-user directory.
"""

import base64
import io
import json
import os
import subprocess
import threading
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SERVICE = _REPO_ROOT / "tools" / "xtts_service.py"


def venv_dir() -> Path:
    explicit = os.environ.get("CODERAI_XTTS_VENV", "").strip()
    if explicit:
        return Path(explicit)
    baked = Path("/opt/coderai/venvs/TTS")
    if baked.exists():
        return baked
    return Path(os.path.expanduser("~/.coderai/xtts_venv"))


def venv_python() -> Path:
    return venv_dir() / ("Scripts" if os.name == "nt" else "bin") / (
        "python.exe" if os.name == "nt" else "python")


def available() -> bool:
    """True when there is a venv with coqui-tts in it to talk to."""
    py = venv_python()
    if not py.is_file() or not _SERVICE.is_file():
        return False
    try:
        return subprocess.run([str(py), "-c", "import TTS"],
                              capture_output=True, timeout=120).returncode == 0
    except Exception:
        return False


class XttsSubprocess:
    """A coqui TTS object living in another interpreter.

    Exposes what _CoquiBackend needs — `speakers`, `tts(**kwargs)` returning
    a float32 array, and the output sample rate — so the backend's synthesize()
    logic (speaker vs speaker_wav selection, speed fallback) runs unchanged.
    """

    def __init__(self, model_name: str):
        self._lock = threading.RLock()
        self._proc = subprocess.Popen(
            [str(venv_python()), str(_SERVICE)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"})
        reply = self._call({"op": "load", "model": model_name})
        if not reply.get("ok"):
            self.close()
            raise RuntimeError(f"XTTS venv could not load {model_name!r}: "
                               f"{reply.get('error')}")
        self.speakers = list(reply.get("speakers") or [])
        self.sample_rate = int(reply.get("sample_rate") or 24000)
        # _CoquiBackend reads `self._tts.synthesizer.output_sample_rate`, the
        # way it would off a real TTS object. Give it the same shape.
        self.synthesizer = type("_Synth", (), {})()
        self.synthesizer.output_sample_rate = self.sample_rate

    def tts(self, **kwargs):
        import numpy as np
        reply = self._call({"op": "tts", **kwargs})
        if not reply.get("ok"):
            err = str(reply.get("error") or "")
            # The in-process backend relies on TypeError to retry without
            # `speed`; reproduce that so its fallback still works across the wire.
            if err.startswith("TypeError"):
                raise TypeError(err)
            raise RuntimeError(f"XTTS synthesis failed: {err}")
        import soundfile as sf
        wav, sr = sf.read(io.BytesIO(base64.b64decode(reply["b64_wav"])), dtype="float32")
        self.sample_rate = int(sr)
        self.synthesizer.output_sample_rate = self.sample_rate
        return np.asarray(wav, dtype=np.float32)

    def _call(self, request: dict) -> dict:
        with self._lock:
            if self._proc.poll() is not None:
                return {"ok": False, "error": "the XTTS worker has exited"}
            self._proc.stdin.write(json.dumps(request) + "\n")
            self._proc.stdin.flush()
            line = self._proc.stdout.readline()
        if not line:
            return {"ok": False, "error": "the XTTS worker returned nothing"}
        try:
            return json.loads(line)
        except Exception as exc:
            return {"ok": False, "error": f"unparseable worker reply: {exc}"}

    def close(self) -> None:
        try:
            if self._proc.poll() is None:
                self._proc.terminate()
                self._proc.wait(timeout=10)
        except Exception:
            pass

    def __del__(self):
        self.close()
