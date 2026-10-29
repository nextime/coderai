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
import sys
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


_bootstrap_lock = threading.Lock()
_bootstrapped = False


def _venv_runs(py: Path) -> bool:
    """Whether this venv's interpreter actually RUNS (see parler_worker)."""
    try:
        return subprocess.run([str(py), "-c", "import sys"],
                              capture_output=True, timeout=120).returncode == 0
    except Exception:
        return False


def bootstrap(timeout: int = 3600) -> bool:
    """Create the venv and install coqui-tts into it (idempotent).

    parler-tts builds its own venv on first use and XTTS did not, so the same
    "needs an isolated venv" situation produced a self-healing engine in one
    case and a 501 telling the operator to go and build one in the other. This
    closes that gap.

    Fully isolated for the reason in the module docstring: coqui-tts needs a
    transformers the server cannot have, so sharing site-packages would either
    break coqui or break the server.

    Returns True when coqui-tts is importable afterwards. Never raises -- the
    caller's fallback is the same 501 it would have raised anyway, and a
    failure here must not take down a synthesis request for another engine.
    """
    global _bootstrapped
    if available():
        _bootstrapped = True
        return True
    with _bootstrap_lock:
        if available():                      # another thread got there first
            _bootstrapped = True
            return True
        d = venv_dir()
        try:
            py = venv_python()
            # A venv built by a different interpreter leaves a dangling
            # bin/python: is_file() says no, and `python -m venv` over the top
            # then dies on that same dangling link, so the engine is stuck for
            # good with a populated venv sitting right there. Clear it first.
            if d.exists() and not _venv_runs(py):
                import shutil
                print(f"[xtts] {d} was built by a different interpreter "
                      f"(its python does not run here); rebuilding …",
                      flush=True)
                shutil.rmtree(d, ignore_errors=True)
            if not py.is_file():
                print(f"[xtts] creating isolated venv at {d} …", flush=True)
                d.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run([sys.executable, "-m", "venv", str(d)],
                               check=True, timeout=600)
            py = venv_python()
            print("[xtts] installing coqui-tts into the isolated venv "
                  "(first run, downloads several GB) …", flush=True)
            subprocess.run([str(py), "-m", "pip", "install", "--no-input",
                            "coqui-tts", "soundfile", "requests"],
                           check=True, timeout=timeout)
        except Exception as exc:
            # A partially built venv is worse than none: `available()` stays
            # False, and the next attempt resumes rather than reporting a venv
            # that cannot import. Measured on a sibling worker: an install
            # interrupted by a full disk left 42 packages and no torch, and
            # the engine reported "venv can't be created" ever after.
            print(f"[xtts] bootstrap failed ({type(exc).__name__}: "
                  f"{str(exc)[:200]}); XTTS stays unavailable", flush=True)
            return False
        ok = available()
        _bootstrapped = ok
        if not ok:
            print("[xtts] coqui-tts installed but still not importable",
                  flush=True)
        return ok


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
            env={**os.environ, "PYTHONUNBUFFERED": "1",
                 # CPML acceptance: configuring an XTTS model IS the decision,
                 # and the alternative is an interactive prompt on a pipe.
                 "COQUI_TOS_AGREED": os.environ.get("COQUI_TOS_AGREED", "1")})
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
