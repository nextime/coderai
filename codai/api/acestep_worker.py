# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Fully-managed ACE-Step worker (isolated venv).

ACE-Step ships its own ``acestep`` package pinning transformers 4.50 and
spacy 3.8.4; the server runs transformers 5.x, and spacy 3.8.4 has no cp313
wheel at all, so this venv is **python 3.12** while the server is 3.13. It
therefore runs behind a small HTTP service (tools/acestep_service.py) and
coderai talks to it the way it talks to CrisperWhisper, Canary and pyannote.
Mirrors codai.api.crisperwhisper_worker.

The venv is pre-baked in the images that carry it (/opt/coderai/acestep_venv)
and otherwise built on first use, so a model that is configured but never
called costs nothing. ``AceStep`` is the handle the audio-gen path stores in
the model cache: it exposes ``cleanup()``, which is what the manager's
eviction calls, so the worker's VRAM is freed on a model swap like any other
model's.
"""

import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SERVICE_SCRIPT = _REPO_ROOT / "tools" / "acestep_service.py"
_REQUIREMENTS = _REPO_ROOT / "requirements-acestep.txt"
#: Both baked locations, because the two image families disagree: the local
#: image puts worker venvs at /opt/coderai/<name>_venv, the pod images at
#: /opt/coderai/venvs/<name> (and pass CODERAI_ACESTEP_VENV anyway).
_BAKED_VENVS = (Path("/opt/coderai/acestep_venv"),
                Path("/opt/coderai/venvs/acestep"))

#: The interpreter this venv needs. Not the server's: spacy==3.8.4, which
#: ACE-Step pins exactly, publishes no cp313 wheel.
_PY_VERSION = "3.12"

_lock = threading.RLock()
_services: dict = {}   # model_name -> {"proc","port","url"}
_bootstrapped = False


def resolve_venv_dir() -> Path:
    explicit = os.environ.get("CODERAI_ACESTEP_VENV")
    if explicit:
        return Path(explicit)
    for baked in _BAKED_VENVS:
        if (baked / ("Scripts" if os.name == "nt" else "bin")).exists():
            return baked
    return Path(os.path.expanduser("~/.coderai/acestep_venv"))


def _venv_python(venv: Path) -> Path:
    return venv / ("Scripts" if os.name == "nt" else "bin") / (
        "python.exe" if os.name == "nt" else "python")


def _pip_ok(py: Path) -> bool:
    try:
        return subprocess.run(
            [str(py), "-c", "import acestep, torch, soundfile"],
            capture_output=True).returncode == 0
    except Exception:
        return False


def _uv() -> str | None:
    from shutil import which
    return which("uv") or next(
        (p for p in (os.path.expanduser("~/.local/bin/uv"), "/usr/local/bin/uv")
         if os.path.isfile(p)), None)


def _create_venv(venv: Path) -> None:
    """Make a python 3.12 venv, by whichever route this host actually has.

    ``python3.12 -m venv`` is tried first, but on a Debian host the 3.12
    interpreter is frequently present without its ``python3.12-venv`` package,
    so ensurepip is missing and that route fails. uv builds the venv (and will
    fetch a 3.12 if the host has none), which is why it is the fallback rather
    than an error telling the user to apt-install something.
    """
    venv.parent.mkdir(parents=True, exist_ok=True)
    from shutil import which
    base = which(f"python{_PY_VERSION}")
    errors = []
    if base:
        r = subprocess.run([base, "-m", "venv", str(venv)],
                           capture_output=True, text=True)
        if r.returncode == 0:
            return
        errors.append(f"python{_PY_VERSION} -m venv: "
                      f"{(r.stderr or r.stdout).strip()[:300]}")
    else:
        errors.append(f"no python{_PY_VERSION} on PATH")
    uv = _uv()
    if uv:
        r = subprocess.run([uv, "venv", "--python", _PY_VERSION, str(venv)],
                           capture_output=True, text=True)
        if r.returncode == 0:
            return
        errors.append(f"uv venv: {(r.stderr or r.stdout).strip()[:300]}")
    else:
        errors.append("uv not found")
    raise RuntimeError(
        f"could not create a python {_PY_VERSION} venv at {venv}: "
        + "; ".join(errors))


def _install(py: Path) -> None:
    """Install requirements-acestep.txt into the venv.

    uv is preferred and then **must** be given ``--index-strategy
    unsafe-best-match``: its default is first-index-wins, so with the PyTorch
    index listed it resolves every shared name from there and pins ancient
    versions of unrelated packages (requests 2.28.1, which then conflicts with
    everything). pip already searches all indexes, so it needs no equivalent.
    """
    if not _REQUIREMENTS.exists():
        raise RuntimeError(f"ACE-Step requirements missing: {_REQUIREMENTS}")
    print("[acestep] installing ACE-Step into the isolated venv "
          "(first run downloads several GB) …", flush=True)
    uv = _uv()
    if uv:
        r = subprocess.run(
            [uv, "pip", "install", "--python", str(py),
             "--index-strategy", "unsafe-best-match",
             "-r", str(_REQUIREMENTS)],
            capture_output=True, text=True)
        if r.returncode == 0:
            return
        print(f"[acestep] uv pip install failed, falling back to pip: "
              f"{(r.stderr or r.stdout).strip()[-600:]}", flush=True)
    subprocess.run([str(py), "-m", "pip", "install", "-U", "pip"], check=True)
    subprocess.run([str(py), "-m", "pip", "install", "-r", str(_REQUIREMENTS)],
                   check=True)


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
        print(f"[acestep] creating isolated python {_PY_VERSION} venv at {venv} …",
              flush=True)
        _create_venv(venv)
        py = _venv_python(venv)
    if not _pip_ok(py):
        _install(py)
        if not _pip_ok(py):
            raise RuntimeError("ACE-Step venv install did not yield importable deps")
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
            print(f"[acestep] {line}", flush=True)


def _health_ok(url: str) -> bool:
    import requests
    try:
        r = requests.get(url + "/health", timeout=3)
        return r.ok and bool(r.json().get("ok"))
    except Exception:
        return False


def _dtype_name(config: dict) -> str:
    """The model's `precision` as the string ACE-Step's pipeline takes.

    It wants the spelled-out torch names, not coderai's short ones, and an
    unrecognised one costs twice the VRAM rather than erroring -- so the map is
    explicit and the service validates the result against torch.
    """
    precision = ((config or {}).get("precision") or "bf16").lower()
    return {"bf16": "bfloat16", "bfloat16": "bfloat16",
            "f16": "float16", "fp16": "float16", "float16": "float16",
            "f32": "float32", "fp32": "float32",
            "float32": "float32"}.get(precision, "bfloat16")


def _checkpoint_dir(model_name: str, config: dict) -> str | None:
    """Where the weights are, as a local directory the worker can read.

    ACE-Step's pipeline takes a ``checkpoint_dir`` holding four subdirectories
    (ace_step_transformer, music_dcae_f8c8, music_vocoder, umt5-base) and
    downloads them itself if given None. An HF snapshot of
    ACE-Step/ACE-Step-v1-3.5B has exactly that layout, so pointing at the
    snapshot reuses the one coderai already has instead of fetching a second
    copy into the pipeline's own directory.
    """
    for key in ("checkpoint_dir", "path", "model_path"):
        v = (config or {}).get(key)
        if v and os.path.isdir(str(v)):
            return str(v)
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(model_name, local_files_only=True)
    except Exception:
        return None       # let the pipeline download to its own location


def ensure_service(model_name: str, config: dict = None,
                   ready_timeout: float = 3600.0) -> str:
    config = config or {}
    # An explicit `service_url` points at an ACE-Step worker running somewhere
    # else -- another host, a container, a rented pod -- so nothing is built,
    # spawned or downloaded here and coderai just proxies to it.
    _remote = ((config.get("service_url") or "")
               or os.environ.get("CODERAI_ACESTEP_SERVICE_URL") or "").strip()
    if _remote:
        _remote = str(_remote).rstrip("/")
        if not _health_ok(_remote):
            raise RuntimeError(
                f"configured service_url {_remote} is not answering its health check")
        print(f"[acestep] using the configured remote service at {_remote}", flush=True)
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
            # All four HF variables: this worker never imports codai, so
            # HF_HOME and TRANSFORMERS_CACHE would otherwise stay as the image
            # set them and the weights would be fetched a second time.
            from codai.hf_cache import child_env as _hf_child_env
            env = _hf_child_env(env)
        except Exception:
            pass
        gpu = config.get("gpu_device")
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        cmd = [str(py), str(_SERVICE_SCRIPT),
               "--host", "127.0.0.1", "--port", str(port)]
        ckpt = _checkpoint_dir(model_name, config)
        if ckpt:
            cmd += ["--checkpoint-dir", ckpt]
        cmd += ["--dtype", _dtype_name(config)]
        if (config.get("offload_strategy") or "") in ("cpu", "sequential", "model", "disk"):
            cmd.append("--cpu-offload")
        # ACE-Step's own int8 path, driven by the same model-page switch every
        # other model uses -- not a separate ACE-Step-only setting.
        if config.get("load_in_8bit") or config.get("load_in_4bit"):
            cmd.append("--quantized")
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            bufsize=1, env=env, cwd=str(_REPO_ROOT))
        import collections
        tail = collections.deque(maxlen=15)
        threading.Thread(target=_pump_logs, args=(proc, tail), daemon=True).start()
        _services[model_name] = {"proc": proc, "port": port, "url": url}

    def _tail_msg():
        joined = " | ".join(l.strip()[:300] for l in list(tail) if l.strip()).strip()
        return f". Last output: {joined}" if joined else ""

    deadline = time.time() + ready_timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"ACE-Step worker exited (code {proc.returncode}) before ready"
                + _tail_msg())
        if _health_ok(url):
            print(f"[acestep] service ready for {model_name} at {url}", flush=True)
            return url
        time.sleep(2)
    stop_service(model_name)
    raise RuntimeError(f"ACE-Step worker for {model_name} did not become ready"
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
    print(f"[acestep] service for {model_name} stopped", flush=True)


def stop_all() -> None:
    for name in list(_services.keys()):
        stop_service(name)


class AceStep:
    """The ACE-Step handle the audio-gen path caches as a 'model'.

    Deliberately not a pipeline: generation happens in another interpreter and
    the result arrives as finished wav bytes. What matters to the rest of the
    server is that ``cleanup()`` exists, because that is what the manager calls
    when it evicts a model to make VRAM for the next one -- and here that
    tears the worker process down, which is the only thing that frees it.
    """

    def __init__(self, model_name: str, config: dict = None):
        self.model_name = model_name
        self.config = config or {}
        self.url = ensure_service(model_name, self.config)

    def generate(self, *, prompt: str, lyrics: str = "", duration: float = 60.0,
                 steps: int = 60, guidance_scale: float = 15.0,
                 scheduler_type: str = "euler", seed=None,
                 timeout: float = 3600.0) -> bytes:
        import requests
        body = {"prompt": prompt, "lyrics": lyrics or "",
                "duration": float(duration), "steps": int(steps),
                "guidance_scale": float(guidance_scale),
                "scheduler_type": scheduler_type}
        if seed is not None:
            body["seed"] = int(seed)
        try:
            resp = requests.post(self.url + "/generate", json=body, timeout=timeout)
        except requests.RequestException:
            # The worker may have been stopped by an eviction between requests;
            # one restart is cheaper than failing a generation.
            self.url = ensure_service(self.model_name, self.config)
            resp = requests.post(self.url + "/generate", json=body, timeout=timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"ACE-Step worker error {resp.status_code}: "
                               f"{resp.text[:300]}")
        if not resp.content:
            raise RuntimeError("ACE-Step worker returned an empty body")
        return resp.content

    def cleanup(self):
        try:
            stop_service(self.model_name)
        except Exception:
            pass


import atexit as _atexit
_atexit.register(stop_all)
