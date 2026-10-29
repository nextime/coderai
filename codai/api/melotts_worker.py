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
import threading
import time
from pathlib import Path

from codai.api import venv_env

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
        return venv_env.run([str(py), "-c", "import melo, soundfile"],
                            capture_output=True).returncode == 0
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
    # One rebuild decision for every way a venv can be unusable: a shared-site
    # build that leaks the server's transformers back in, a dangling
    # bin/python from another interpreter, or packages installed for a
    # different minor version. See venv_env.venv_usable.
    py = venv_env.create_venv(_VENV_DIR)
    if not _pip_ok(py):
        print("[melotts] installing melotts + torch into the isolated venv "
              "(first run, downloads several GB, this can take a while) …", flush=True)
        venv_env.run([str(py), "-m", "pip", "install",
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


def ensure_service(model_name: str, ready_timeout: float = 1800.0,
                   config: dict = None) -> str:
    """Start (or reuse) the worker for ``model_name`` and return its base URL.

    First call bootstraps the venv and downloads the model, so the timeout is
    generous. Raises RuntimeError if the service never comes up."""
    # An explicit `service_url` points at a service running somewhere else —
    # another host, a container, a rented RunPod pod — so nothing is spawned or
    # downloaded here and coderai just proxies to it. Same contract the TTS
    # backends have used for parler/melo, and the hook that makes a per-engine
    # remote pod possible without changing anything downstream: this function
    # already returns a URL, so its callers cannot tell the difference.
    _remote = ((config or {}).get("service_url")
               or os.environ.get("CODERAI_MELOTTS_SERVICE_URL") or "").strip()
    if _remote:
        _remote = _remote.rstrip("/")
        if not _health_ok(_remote):
            raise RuntimeError(
                f"configured service_url {_remote} is not answering its health check")
        print(f"[melotts] using the configured remote service at {_remote}", flush=True)
        return _remote

    with _lock:
        svc = _services.get(model_name)
        if svc and svc["proc"].poll() is None and _health_ok(svc["url"]):
            return svc["url"]
        if svc and svc["proc"].poll() is not None:
            _services.pop(model_name, None)   # died — restart below

        py = _bootstrap_venv()
        port = _free_port()
        url = f"http://127.0.0.1:{port}"
        # One call for all three halves: coderai's HF cache variables (this
        # worker never imports codai, so without them it downloads a second
        # copy of every model); a clean interpreter environment -- the engine
        # exports PYTHONHOME, under which this venv's python would take its
        # stdlib and site-packages from the SERVER's installation; and the
        # library path this particular venv needs, which depends on whose
        # interpreter it holds (see venv_env.env_for_venv).
        env = venv_env.env_for_python(py)
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
        joined = " | ".join(l.strip()[:300] for l in list(tail) if l.strip()).strip()
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
