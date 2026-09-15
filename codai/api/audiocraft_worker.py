# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Drive audiocraft in its own venv, and look like a MusicGen model doing it.

audiocraft pins torch==2.1.0. The server runs torch 2.11 built for a specific
CUDA, and torch 2.1.0 has no build for that CUDA — so installing audiocraft
beside the server does not merely risk a conflict, it leaves the GPU unusable.
The same problem parler-tts and the OCR engines have, solved the same way: its
own venv, driven as a subprocess.

What this exposes is deliberately the shape audiocraft's own MusicGen has
(``set_generation_params``, ``generate``, ``sample_rate``), because the audio
generation path already speaks that. The caller cannot tell whether audiocraft
is in this process, in a venv next door, or absent — in which case the path
falls back to transformers' own MusicGen, which needs no venv but has no melody
conditioning.
"""

import base64
import io
import json
import os
import subprocess
import threading
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SERVICE = _REPO_ROOT / "tools" / "audiocraft_service.py"

#: Where the venv lives. The pod images bake it at /opt/coderai/venvs/audiocraft
#: — a pod is disposable and building it on first use would pay minutes of GPU
#: rental to install a library. A local install defaults under ~/.coderai.
_VENV = Path(os.environ.get("CODERAI_AUDIOCRAFT_VENV")
             or ("/opt/coderai/venvs/audiocraft"
                 if Path("/opt/coderai/venvs/audiocraft").exists()
                 else os.path.expanduser("~/.coderai/audiocraft_venv")))

_lock = threading.RLock()


def venv_python() -> Path:
    return _VENV / ("Scripts" if os.name == "nt" else "bin") / (
        "python.exe" if os.name == "nt" else "python")


def available() -> bool:
    """True when there is a venv with audiocraft in it to talk to."""
    py = venv_python()
    if not py.is_file() or not _SERVICE.is_file():
        return False
    try:
        return subprocess.run([str(py), "-c", "import audiocraft"],
                              capture_output=True, timeout=120).returncode == 0
    except Exception:
        return False


class AudiocraftModel:
    """An audiocraft model living in another interpreter."""

    def __init__(self, model_name: str):
        self._proc = subprocess.Popen(
            [str(venv_python()), str(_SERVICE)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"})
        self._params: dict = {"duration": 30.0}
        reply = self._call({"op": "load", "model": model_name})
        if not reply.get("ok"):
            self.close()
            raise RuntimeError(f"audiocraft could not load {model_name!r}: "
                               f"{reply.get('error')}")
        self._sample_rate = int(reply.get("sample_rate") or 32000)

    # -- the audiocraft-shaped surface ------------------------------------ #
    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def set_generation_params(self, **kwargs):
        self._params.update({k: v for k, v in kwargs.items() if v is not None})

    def generate(self, prompts):
        return self._audio({"op": "generate", "prompt": list(prompts)[0],
                            **self._params})

    def generate_with_chroma(self, prompts, melody_wav, sr):
        """Melody conditioning — the capability transformers does not have."""
        import torchaudio
        buf = io.BytesIO()
        wav = melody_wav[0] if melody_wav.dim() == 3 else melody_wav
        torchaudio.save(buf, wav.cpu(), int(sr), format="wav")
        return self._audio({"op": "generate", "prompt": list(prompts)[0],
                            "melody_wav": base64.b64encode(buf.getvalue()).decode(),
                            **self._params})

    # -- transport --------------------------------------------------------- #
    def _audio(self, request: dict):
        reply = self._call(request)
        if not reply.get("ok"):
            raise RuntimeError(f"audiocraft generation failed: {reply.get('error')}")
        import torch, torchaudio
        wav, _sr = torchaudio.load(io.BytesIO(base64.b64decode(reply["b64_wav"])))
        # The caller indexes [0, 0] like audiocraft's own (batch, channels, samples).
        return wav.unsqueeze(0) if wav.dim() == 2 else wav

    def _call(self, request: dict) -> dict:
        with _lock:
            if self._proc.poll() is not None:
                return {"ok": False, "error": "the audiocraft worker has exited"}
            self._proc.stdin.write(json.dumps(request) + "\n")
            self._proc.stdin.flush()
            line = self._proc.stdout.readline()
        if not line:
            return {"ok": False, "error": "the audiocraft worker returned nothing"}
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
