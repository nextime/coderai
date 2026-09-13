# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Fully-managed CrisperWhisper worker (isolated venv).

CrisperWhisper needs its verbatim word-timestamp generation, which breaks on the
server's transformers 5.x. So it runs in its own venv (pinned transformers 4.x)
behind a small HTTP service (tools/crisperwhisper_service.py); coderai talks to it
as a remote STT backend. Mirrors codai.api.canary_worker / pyannote_worker.
"""

import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SERVICE_SCRIPT = _REPO_ROOT / "tools" / "crisperwhisper_service.py"
_REQUIREMENTS = _REPO_ROOT / "requirements-crisperwhisper.txt"
_BAKED_VENV = Path("/opt/coderai/crisperwhisper_venv")

_lock = threading.RLock()
_services: dict = {}   # model_name -> {"proc","port","url"}
_bootstrapped = False


def resolve_venv_dir() -> Path:
    explicit = os.environ.get("CODERAI_CRISPERWHISPER_VENV")
    if explicit:
        return Path(explicit)
    if (_BAKED_VENV / ("Scripts" if os.name == "nt" else "bin")).exists():
        return _BAKED_VENV
    return Path(os.path.expanduser("~/.coderai/crisperwhisper_venv"))


def _venv_python(venv: Path) -> Path:
    return venv / ("Scripts" if os.name == "nt" else "bin") / (
        "python.exe" if os.name == "nt" else "python")


def _pip_ok(py: Path) -> bool:
    try:
        return subprocess.run(
            [str(py), "-c", "import transformers, torch, soundfile"],
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
        print(f"[crisperwhisper] creating isolated venv at {venv} …", flush=True)
        venv.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    py = _venv_python(venv)
    if not _pip_ok(py):
        if not _REQUIREMENTS.exists():
            raise RuntimeError(f"CrisperWhisper requirements missing: {_REQUIREMENTS}")
        print("[crisperwhisper] installing pinned transformers+torch into the isolated "
              "venv (first run downloads several GB) …", flush=True)
        subprocess.run([str(py), "-m", "pip", "install", "-U", "pip"], check=True)
        subprocess.run([str(py), "-m", "pip", "install", "-r", str(_REQUIREMENTS)],
                       check=True)
        if not _pip_ok(py):
            raise RuntimeError("CrisperWhisper venv install did not yield importable deps")
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
            print(f"[crisperwhisper] {line}", flush=True)


def _health_ok(url: str) -> bool:
    import requests
    try:
        r = requests.get(url + "/health", timeout=3)
        return r.ok and bool(r.json().get("ok"))
    except Exception:
        return False


def ensure_service(model_name: str, config: dict = None,
                   ready_timeout: float = 1800.0) -> str:
    config = config or {}
    # An explicit `service_url` points at a CrisperWhisper worker running somewhere else — another
    # host, a container, a rented pod — so nothing is built, spawned or downloaded
    # here and coderai just proxies to it. This function already returns a URL, so
    # its callers cannot tell the difference.
    _remote = ((config.get("service_url") or "") or os.environ.get("CODERAI_CRISPERWHISPER_SERVICE_URL") or "").strip()
    if _remote:
        _remote = str(_remote).rstrip("/")
        if not _health_ok(_remote):
            raise RuntimeError(
                f"configured service_url {_remote} is not answering its health check")
        print(f"[crisperwhisper] using the configured remote service at {_remote}", flush=True)
        return _remote

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
                f"CrisperWhisper worker exited (code {proc.returncode}) before ready"
                + _tail_msg())
        if _health_ok(url):
            print(f"[crisperwhisper] service ready for {model_name} at {url}", flush=True)
            return url
        time.sleep(2)
    stop_service(model_name)
    raise RuntimeError(f"CrisperWhisper worker for {model_name} did not become ready"
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
    print(f"[crisperwhisper] service for {model_name} stopped", flush=True)


def stop_all() -> None:
    for name in list(_services.keys()):
        stop_service(name)


import atexit as _atexit
_atexit.register(stop_all)
