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

from codai.api import venv_env

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SERVICE = _REPO_ROOT / "tools" / "xtts_service.py"
_REQUIREMENTS = _REPO_ROOT / "requirements-xtts.txt"

#: The same pins as requirements-xtts.txt, for an install where that file is
#: not present (the container mounts only codai/, tools/ and third_party/ over
#: the image). A test keeps the two in step.
_PINS = ("--extra-index-url", "https://download.pytorch.org/whl/cu124",
         "torch==2.6.0+cu124",
         "torchaudio==2.6.0+cu124",
         "coqui-tts",
         "soundfile>=0.12.0",
         "requests>=2.28.0")


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


def health() -> tuple:
    """``(ok, detail)`` for the venv behind XTTS.

    Two questions, because either one failing means no speech: are the
    packages importable, and can the stack run a convolution on the GPU (a
    vocoder is convolutional, and a venv with a mismatched CUDA stack imports
    perfectly and dies on the first one). The detail is carried out so the
    caller can say WHY -- "coqui-tts installed but still not importable" is
    what this used to report, and it named neither the module nor the reason.

    Both run in a clean environment: the server exports PYTHONHOME, and under
    it the probe imports the SERVER's site-packages, so a populated venv
    answered "coqui-tts is not installed" for ever (see venv_env).
    """
    py = venv_python()
    if not py.is_file():
        return False, f"there is no interpreter at {py}"
    if not _SERVICE.is_file():
        return False, f"the worker script is missing: {_SERVICE}"
    ok, why = venv_env.import_ok(py, ("TTS",))
    if not ok:
        return False, why
    return venv_env.gpu_conv_ok(py)


def available() -> bool:
    """True when there is a venv with a working coqui-tts in it to talk to."""
    return health()[0]


_bootstrap_lock = threading.Lock()
_bootstrapped = False


def _venv_ok() -> bool:
    """Whether the venv on disk is one this process can use at all.

    Not just "its python runs": a venv built by another 3.13 whose packages
    live in a different minor version's directory runs and imports nothing.
    See :func:`codai.api.venv_env.venv_usable`.
    """
    ok, _why = venv_env.venv_usable(venv_dir())
    return ok


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
            # create_venv rebuilds what cannot be used -- a dangling
            # bin/python, a minor-version mismatch against its own
            # site-packages, or a --system-site-packages build -- because
            # `python -m venv` over any of those dies on the thing that broke
            # it, leaving the engine stuck for good with a populated venv
            # sitting right there.
            if not venv_python().is_file() or not _venv_ok():
                print(f"[xtts] preparing the isolated venv at {d} …", flush=True)
            py = venv_env.create_venv(d)
            print("[xtts] installing coqui-tts + a pinned torch into the "
                  "isolated venv (first run, downloads several GB) …", flush=True)
            # From the pinned file, not `pip install coqui-tts`: that installed
            # coqui-tts and its ninety dependencies with NO torch at all, so
            # the install reported success and `import TTS` failed.
            venv_env.run([str(py), "-m", "pip", "install", "--no-input"]
                         + venv_env.pip_install_args(_REQUIREMENTS, _PINS),
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
        ok, why = health()
        _bootstrapped = ok
        if not ok:
            print(f"[xtts] the install finished but the venv is not usable: "
                  f"{why}", flush=True)
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
            env=venv_env.env_for_python(venv_python(), {
                "PYTHONUNBUFFERED": "1",
                # CPML acceptance: configuring an XTTS model IS the decision,
                # and the alternative is an interactive prompt on a pipe.
                "COQUI_TOS_AGREED": os.environ.get("COQUI_TOS_AGREED", "1")}))
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
