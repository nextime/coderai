# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Fully-managed pyannote.audio worker (speaker diarization).

pyannote.audio pins torch + pytorch-lightning + speechbrain, which conflict with
the coderai server's transformers 5.x stack — so, like NeMo/Canary, it runs in an
isolated venv. coderai owns the lifecycle: it uses a pyannote venv (baked into the
image at ``/opt/coderai/pyannote_venv`` when present, else built on demand from
``requirements-pyannote.txt``), launches ``tools/pyannote_service.py`` in it as a
local HTTP service, health-checks it, and hands back the URL.

The gated ``pyannote/speaker-diarization-3.1`` needs an HF token (accept its
conditions + ``pyannote/segmentation-3.0`` first). The worker passes ``HF_TOKEN``
/ ``HUGGINGFACE_TOKEN`` from the environment through to the service; the default
model can be pointed at an ungated mirror via ``CODERAI_DIARIZATION_MODEL`` so no
token is needed.
"""

import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SERVICE_SCRIPT = _REPO_ROOT / "tools" / "pyannote_service.py"
_REQUIREMENTS = _REPO_ROOT / "requirements-pyannote.txt"

_BAKED_VENV = Path("/opt/coderai/pyannote_venv")

_lock = threading.RLock()
_services: dict = {}   # model_name -> {"proc","port","url"}
_bootstrapped = False


def resolve_venv_dir() -> Path:
    explicit = os.environ.get("CODERAI_PYANNOTE_VENV")
    if explicit:
        return Path(explicit)
    if (_BAKED_VENV / ("Scripts" if os.name == "nt" else "bin")).exists():
        return _BAKED_VENV
    return Path(os.path.expanduser("~/.coderai/pyannote_venv"))


def _venv_python(venv: Path) -> Path:
    return venv / ("Scripts" if os.name == "nt" else "bin") / (
        "python.exe" if os.name == "nt" else "python")


def _pip_ok(py: Path) -> bool:
    try:
        return subprocess.run(
            [str(py), "-c", "import pyannote.audio"],
            capture_output=True).returncode == 0
    except Exception:
        return False


def _bootstrap_venv() -> Path:
    global _bootstrapped
    venv = resolve_venv_dir()
    py = _venv_python(venv)
    if _bootstrapped and py.exists():
        return py
    if py.exists() and _pip_ok(py):
        _bootstrapped = True
        return py
    if not py.exists():
        print(f"[pyannote] creating isolated venv at {venv} …", flush=True)
        venv.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    py = _venv_python(venv)
    if not _pip_ok(py):
        if not _REQUIREMENTS.exists():
            raise RuntimeError(f"pyannote requirements file missing: {_REQUIREMENTS}")
        print("[pyannote] installing pyannote.audio into the isolated venv "
              "(first run downloads several GB, this can take a while) …", flush=True)
        subprocess.run([str(py), "-m", "pip", "install", "-U", "pip"], check=True)
        subprocess.run([str(py), "-m", "pip", "install", "-r", str(_REQUIREMENTS)],
                       check=True)
        if not _pip_ok(py):
            raise RuntimeError("pyannote install did not yield an importable pyannote.audio")
    _bootstrapped = True
    return py


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _pump_logs(proc: subprocess.Popen, tail):
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            tail.append(line)
            print(f"[pyannote] {line}", flush=True)


def _health_ok(url: str) -> bool:
    import requests
    try:
        r = requests.get(url + "/health", timeout=3)
        return r.ok and bool(r.json().get("ok"))
    except Exception:
        return False


def _default_model() -> str:
    # Official pyannote pipeline (canonical, best-supported). It's gated: needs an
    # HF token (see _resolve_hf_token) + accepting the pyannote/speaker-diarization-3.1
    # and pyannote/segmentation-3.0 conditions. Override with CODERAI_DIARIZATION_MODEL.
    return os.environ.get("CODERAI_DIARIZATION_MODEL") or "pyannote/speaker-diarization-3.1"


def _resolve_hf_token(config: dict) -> str:
    """Find an HF token: request/config, then env, then a protected file in the
    coderai config dir (~/.coderai/hf_token or /config/coderai/hf_token). Lets the
    token be configured once without editing the container's launch command."""
    tok = (config or {}).get("hf_token")
    if tok:
        return tok
    for var in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(var):
            return os.environ[var]
    for path in (os.environ.get("CODERAI_HF_TOKEN_FILE"),
                 os.path.expanduser("~/.coderai/hf_token"),
                 "/config/coderai/hf_token"):
        if path and os.path.isfile(path):
            try:
                t = open(path).read().strip()
                if t:
                    return t
            except OSError:
                pass
    return ""


def ensure_service(model_name: str = None, config: dict = None,
                   ready_timeout: float = 1800.0) -> str:
    """Start (or reuse) the pyannote worker and return its base URL."""
    config = config or {}
    model_name = model_name or config.get("model_path") or _default_model()
    with _lock:
        svc = _services.get(model_name)
        if svc and svc["proc"].poll() is None and _health_ok(svc["url"]):
            return svc["url"]
        if svc and svc["proc"].poll() is not None:
            _services.pop(model_name, None)

        py = _bootstrap_venv()
        port = _free_port()
        url = f"http://127.0.0.1:{port}"
        env = dict(os.environ)
        try:
            from codai.models.cache import get_hf_hub_cache_dir
            hub = get_hf_hub_cache_dir()
            env["HF_HUB_CACHE"] = hub
            env["HUGGINGFACE_HUB_CACHE"] = hub
        except Exception:
            pass
        # Pass an HF token through for the gated model (harmless for ungated).
        tok = _resolve_hf_token(config)
        if tok:
            env["HF_TOKEN"] = tok
        gpu = config.get("gpu_device")
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        proc = subprocess.Popen(
            [str(py), str(_SERVICE_SCRIPT), "--model", model_name,
             "--host", "127.0.0.1", "--port", str(port)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            bufsize=1, env=env, cwd=str(_REPO_ROOT))
        import collections
        tail = collections.deque(maxlen=15)
        threading.Thread(target=_pump_logs, args=(proc, tail), daemon=True).start()
        _services[model_name] = {"proc": proc, "port": port, "url": url}

    def _tail_msg():
        joined = " | ".join(list(tail)[-5:]).strip()
        return f". Last output: {joined}" if joined else ""

    deadline = time.time() + ready_timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"pyannote worker exited (code {proc.returncode}) before becoming ready"
                + _tail_msg())
        if _health_ok(url):
            print(f"[pyannote] service ready for {model_name} at {url}", flush=True)
            return url
        time.sleep(2)
    stop_service(model_name)
    raise RuntimeError(f"pyannote worker for {model_name} did not become ready in time"
                       + _tail_msg())


def is_running(model_name: str = None) -> bool:
    model_name = model_name or _default_model()
    with _lock:
        svc = _services.get(model_name)
        return bool(svc and svc["proc"].poll() is None)


def stop_service(model_name: str = None) -> None:
    model_name = model_name or _default_model()
    with _lock:
        svc = _services.pop(model_name, None)
    if not svc:
        return
    proc = svc["proc"]
    if proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:
            pass
    if proc.poll() is None:
        try:
            proc.kill()
        except Exception:
            pass
    print(f"[pyannote] service for {model_name} stopped", flush=True)


def stop_all() -> None:
    for name in list(_services.keys()):
        stop_service(name)


import atexit as _atexit
_atexit.register(stop_all)
