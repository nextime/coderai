# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Fully-managed MeloTTS worker.

MeloTTS is the base speaker model OpenVoice V2 synthesises with before its
tone-colour converter re-paints the timbre. coderai uses it the same way: as the
base voice of the `chain` cloning engine (MeloTTS speaks the line, seed-vc moves
the timbre over), and as a plain multilingual TTS engine in its own right —
English, Spanish, French, Chinese, Japanese and Korean.

Its pins (an older transformers/tokenizers line, unidic, a Japanese tokenizer
chain) conflict with this server's stack exactly the way parler-tts's do, so it
runs in an ISOLATED venv and is reached over HTTP. This module owns the venv and
the process; the synthesis lives in tools/melotts_service.py.
"""


import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SERVICE_SCRIPT = _REPO_ROOT / "tools" / "melotts_service.py"

# Dedicated venv for the (incompatible) melotts stack. Created with access to
# the base interpreter's packages so torch/numpy aren't re-downloaded; melotts's
# pinned transformers installs into the venv and shadows the system one.
_VENV_DIR = Path(os.environ.get("CODERAI_MELOTTS_VENV")
                 or os.path.expanduser("~/.coderai/melotts_venv"))

_lock = threading.RLock()
_services: dict[str, dict] = {}   # model_name -> {"proc","port","url"}
_bootstrapped = False


def _venv_python() -> Path:
    return _VENV_DIR / ("Scripts" if os.name == "nt" else "bin") / (
        "python.exe" if os.name == "nt" else "python")


def _pip_ok(py: Path) -> bool:
    try:
        return subprocess.run([str(py), "-c", "import melo, soundfile"],
                              capture_output=True).returncode == 0
    except Exception:
        return False


def _venv_is_system_site() -> bool:
    """True if the venv was built with --system-site-packages (can't isolate)."""
    try:
        return "include-system-site-packages = true" in \
            (_VENV_DIR / "pyvenv.cfg").read_text().lower()
    except Exception:
        return False


def _bootstrap_venv() -> Path:
    """Create a fully-isolated venv and install melotts (idempotent).

    Isolation is the whole point: melotts pins an old transformers/tokenizers
    that must NOT be shared with — or shadowed by — the server's stack, so the
    venv gets its own copy of everything (torch included). Returns its python."""
    global _bootstrapped
    py = _venv_python()
    if _bootstrapped and py.exists():
        return py
    # A previously-created shared-site venv leaks the server's transformers in;
    # rebuild it isolated.
    if py.exists() and _venv_is_system_site():
        import shutil
        print("[melotts] rebuilding venv as fully isolated …", flush=True)
        shutil.rmtree(_VENV_DIR, ignore_errors=True)
    if not _venv_python().exists():
        print(f"[melotts] creating isolated venv at {_VENV_DIR} …", flush=True)
        _VENV_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-m", "venv", str(_VENV_DIR)], check=True)
    py = _venv_python()
    if not _pip_ok(py):
        print("[melotts] installing melotts + torch into the isolated venv "
              "(first run, downloads several GB, this can take a while) …", flush=True)
        subprocess.run([str(py), "-m", "pip", "install",
                        "git+https://github.com/huggingface/melotts.git",
                        "soundfile"], check=True)
        if not _pip_ok(py):
            raise RuntimeError("melotts install did not yield an importable package")
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
            print(f"[melotts] {line}", flush=True)


def _health_ok(url: str) -> bool:
    import requests
    try:
        r = requests.get(url + "/health", timeout=3)
        return r.ok and bool(r.json().get("ok"))
    except Exception:
        return False


def ensure_service(model_name: str, ready_timeout: float = 1800.0) -> str:
    """Start (or reuse) the worker for ``model_name`` and return its base URL.

    First call bootstraps the venv and downloads the model, so the timeout is
    generous. Raises RuntimeError if the service never comes up."""
    with _lock:
        svc = _services.get(model_name)
        if svc and svc["proc"].poll() is None and _health_ok(svc["url"]):
            return svc["url"]
        if svc and svc["proc"].poll() is not None:
            _services.pop(model_name, None)   # died — restart below

        py = _bootstrap_venv()
        port = _free_port()
        url = f"http://127.0.0.1:{port}"
        env = dict(os.environ)
        # The worker must use the model already pulled via coderai's HF download
        # interface — it never downloads anything itself. Point it at coderai's
        # cache and force offline mode, so a missing model fails fast instead of
        # silently fetching.
        try:
            from codai.models.cache import get_hf_hub_cache_dir
            hub = get_hf_hub_cache_dir()
            env["HF_HUB_CACHE"] = hub
            env["HUGGINGFACE_HUB_CACHE"] = hub
        except Exception:
            pass
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        proc = subprocess.Popen(
            [str(py), str(_SERVICE_SCRIPT), "--model", model_name,
             "--host", "127.0.0.1", "--port", str(port)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            bufsize=1, env=env, cwd=str(_REPO_ROOT),
        )
        import collections
        tail = collections.deque(maxlen=15)
        threading.Thread(target=_pump_logs, args=(proc, tail), daemon=True).start()
        _services[model_name] = {"proc": proc, "port": port, "url": url}

    def _tail_msg():
        joined = " | ".join(list(tail)[-5:]).strip()
        if "offline" in joined.lower() or "not" in joined.lower() and "found" in joined.lower():
            return (f". The model isn't in coderai's cache — download "
                    f"'{model_name}' from the model interface first. ({joined})")
        return f". Last output: {joined}" if joined else ""

    # Wait (outside the lock) for the service to load the model and answer.
    deadline = time.time() + ready_timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"MeloTTS worker exited (code {proc.returncode}) before becoming ready"
                + _tail_msg())
        if _health_ok(url):
            print(f"[melotts] service ready for {model_name} at {url}", flush=True)
            return url
        time.sleep(2)
    stop_service(model_name)
    raise RuntimeError(f"MeloTTS worker for {model_name} did not become ready in time"
                       + _tail_msg())


def stop_service(model_name: str) -> None:
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
    print(f"[melotts] service for {model_name} stopped", flush=True)


def stop_all() -> None:
    for name in list(_services.keys()):
        stop_service(name)


import atexit as _atexit
_atexit.register(stop_all)
