# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Fully-managed Parler-TTS worker.

parler-tts pins an old transformers/tokenizers/huggingface-hub that conflict with
the coderai server's stack, so it can't share this venv. Instead coderai owns the
whole lifecycle here: on first use it bootstraps a dedicated venv (installing
parler-tts), launches ``tools/parler_tts_service.py`` in it as a local HTTP
service, health-checks it, and hands back the URL. The matching
``_RemoteParlerBackend.cleanup()`` calls :func:`stop_service`, so the model
manager's normal eviction tears the process down — no manual setup or config.
"""

import os
import socket
import subprocess
import threading
import time
from pathlib import Path

from codai.api import venv_env

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SERVICE_SCRIPT = _REPO_ROOT / "tools" / "parler_tts_service.py"
_REQUIREMENTS = _REPO_ROOT / "requirements-parler.txt"

#: The same pins as requirements-parler.txt, for an install where that file is
#: not present (the container mounts only codai/, tools/ and third_party/ over
#: the image, so a repo-root file added since the image was built is absent
#: until the container is recreated). A test keeps the two in step.
_PINS = ("--extra-index-url", "https://download.pytorch.org/whl/cu124",
         "torch==2.6.0+cu124",
         "torchaudio==2.6.0+cu124",
         "git+https://github.com/huggingface/parler-tts.git",
         "soundfile>=0.12.0")

# Dedicated venv for the (incompatible) parler-tts stack. Created with access to
# the base interpreter's packages so torch/numpy aren't re-downloaded; parler's
# pinned transformers installs into the venv and shadows the system one.
_VENV_DIR = Path(os.environ.get("CODERAI_PARLER_VENV")
                 or os.path.expanduser("~/.coderai/parler_venv"))

_lock = threading.RLock()
_services: dict[str, dict] = {}   # model_name -> {"proc","port","url"}
_bootstrapped = False


def _venv_python() -> Path:
    return _VENV_DIR / ("Scripts" if os.name == "nt" else "bin") / (
        "python.exe" if os.name == "nt" else "python")


def _import_check(py: Path) -> tuple:
    """``(ok, detail)`` for "are this venv's packages importable".

    The detail matters. "install did not yield an importable package" is what
    this used to say, which names neither the package nor the reason --
    measured on zeiss, 2026-10-29, where the install had in fact succeeded and
    the import was failing on a torchaudio that did not match the pinned
    torch.
    """
    try:
        r = venv_env.run([str(py), "-c", "import parler_tts, soundfile"],
                         capture_output=True, text=True, timeout=600)
    except Exception as exc:
        return False, f"the import could not be run ({exc})"
    if r.returncode == 0:
        return True, ""
    tail = ((r.stderr or "") or (r.stdout or "")).strip().splitlines()
    return False, (tail[-1][:300] if tail else f"exit {r.returncode}")


def _pip_ok(py: Path) -> bool:
    return _import_check(py)[0]


#: The op that failed. Parler's audio decoder (DAC) is convolutional, so a
#: venv whose CUDA stack cannot run one convolution cannot synthesize anything
#: -- however importable its packages are.
_GPU_PROBE = (
    "import torch, torch.nn.functional as F\n"
    "if not torch.cuda.is_available():\n"
    "    raise SystemExit(0)\n"
    "x = torch.zeros(1, 1, 16, device='cuda')\n"
    "w = torch.zeros(1, 1, 3, device='cuda')\n"
    "F.conv1d(x, w).sum().item()\n")


def _gpu_ok(py: Path) -> tuple:
    """``(ok, detail)`` — whether this venv can actually run on the GPU.

    "Are the packages importable" is not the same question, and the difference
    cost a whole bring-up. Measured on zeiss, 2026-10-29: the venv imported
    parler_tts and soundfile perfectly, the service started, the model loaded,
    and the first synthesis died in the decoder's conv1d with
    ``CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED`` — a CUDA-13 torch that pip had
    resolved on its own. ``_pip_ok`` answered True for that venv for ever, so
    nothing ever rebuilt it.

    A CPU-only host passes: it exits before the convolution, because there is
    no GPU stack to be broken.
    """
    try:
        r = venv_env.run([str(py), "-c", _GPU_PROBE],
                         capture_output=True, text=True, timeout=600)
    except Exception as exc:
        return False, f"the probe could not be run ({exc})"
    if r.returncode == 0:
        return True, ""
    tail = ((r.stderr or "") or (r.stdout or "")).strip().splitlines()
    return False, (tail[-1][:300] if tail else f"exit {r.returncode}")


def _bootstrap_venv() -> Path:
    """Create a fully-isolated venv and install parler-tts (idempotent).

    Isolation is the whole point: parler-tts pins an old transformers/tokenizers
    that must NOT be shared with — or shadowed by — the server's stack, so the
    venv gets its own copy of everything (torch included). Returns its python."""
    global _bootstrapped
    py = _venv_python()
    if _bootstrapped and py.exists():
        return py
    # Everything that makes a venv unusable -- a shared-site build that leaks
    # the server's transformers in, an interpreter that no longer runs, a
    # minor-version mismatch against its own site-packages -- is one rebuild
    # decision, taken in venv_env.create_venv. None of them can be repaired
    # in place: `python -m venv` over the top dies on the thing that broke it.
    py = venv_env.create_venv(_VENV_DIR)
    _ok = _pip_ok(py)
    if _ok:
        # Importable is not enough: a venv whose CUDA stack cannot run a
        # convolution synthesizes nothing, and the old check called it healthy
        # for ever. Rebuilt from the pinned requirements, not from whatever
        # pip resolves today -- that resolution is what broke it.
        _ok, _why = _gpu_ok(py)
        if not _ok:
            import shutil
            print(f"[parler] the venv imports but cannot run on the GPU "
                  f"({_why}); rebuilding it from {_REQUIREMENTS.name}", flush=True)
            shutil.rmtree(_VENV_DIR, ignore_errors=True)
            py = venv_env.create_venv(_VENV_DIR)
    if not _ok:
        print("[parler] installing parler-tts + a pinned torch into the isolated "
              "venv (first run, downloads several GB, this can take a while) …",
              flush=True)
        venv_env.run([str(py), "-m", "pip", "install", "-U", "pip"], check=True)
        venv_env.run([str(py), "-m", "pip", "install"]
                     + venv_env.pip_install_args(_REQUIREMENTS, _PINS),
                     check=True)
        _imp, _why = _import_check(py)
        if not _imp:
            raise RuntimeError(
                f"the parler-tts install finished but its packages do not "
                f"import: {_why}")
        _gpu, _why = _gpu_ok(py)
        if not _gpu:
            # Loud, and not fatal: a synthesis will fail with the real error
            # from the real op, which is more useful than a bootstrap that
            # refuses to finish on a machine whose CUDA this cannot judge.
            print(f"[parler] warning: the rebuilt venv still cannot run a "
                  f"convolution on the GPU ({_why})", flush=True)
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
            print(f"[parler] {line}", flush=True)


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
    # An explicit service URL points at a parler worker running somewhere else —
    # another host, a container, a rented pod — so nothing is built, spawned or
    # downloaded here. The per-model `service_url` is handled one level up in
    # tts_backends; this env var covers direct callers too.
    _remote = (os.environ.get("CODERAI_PARLER_SERVICE_URL") or "").strip()
    if _remote:
        _remote = _remote.rstrip("/")
        if not _health_ok(_remote):
            raise RuntimeError(
                f"configured service_url {_remote} is not answering its health check")
        print(f"[parler] using the configured remote service at {_remote}", flush=True)
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
        # The worker must use the model already pulled via coderai's HF download
        # interface — it never downloads anything itself. Point it at coderai's
        # cache (all four HF variables, since it never imports codai) and force
        # offline mode, so a missing model fails fast instead of silently
        # fetching. env_for_python also drops PYTHONHOME -- under it this
        # venv's python took its stdlib and site-packages from the SERVER's
        # installation, which is how a populated venv reported itself broken --
        # and decides the library path from whose interpreter the venv holds:
        # a foreign python loading this one's extension modules is the
        # CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED shape.
        env = venv_env.env_for_python(py, {"HF_HUB_OFFLINE": "1",
                                           "TRANSFORMERS_OFFLINE": "1"})
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
                f"Parler worker exited (code {proc.returncode}) before becoming ready"
                + _tail_msg())
        if _health_ok(url):
            print(f"[parler] service ready for {model_name} at {url}", flush=True)
            return url
        time.sleep(2)
    stop_service(model_name)
    raise RuntimeError(f"Parler worker for {model_name} did not become ready in time"
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
    print(f"[parler] service for {model_name} stopped", flush=True)


def stop_all() -> None:
    for name in list(_services.keys()):
        stop_service(name)


import atexit as _atexit
_atexit.register(stop_all)
