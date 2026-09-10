# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Fully-managed MiniMax-H3 worker (video + native audio generation).

H3 is a 33B flow-matching DiT that denoises video and audio in ONE packed sequence.
diffusers ships it as Modular blocks only (``MiniMaxH3ModularPipeline``, 0.40+), so
it fits neither ``_detect_pipeline_class()`` nor the main venv's diffusers 0.38 —
it runs in an isolated venv, exactly like pyannote / NeMo-Canary / vLLM.

This module owns the venv + process lifecycle and the HTTP call; the generation
itself lives in ``tools/h3_service.py``. The service is registered with the model
manager as an evictable, VRAM-tracked model, so a big LLM can push H3 off the card
(eviction calls the handle's ``cleanup()``, which stops the subprocess).

Config keys (models.json entry for the H3 model, all optional)::

    "backend": "h3",              # or an id/path containing "minimax-h3"
    "h3_venv": "/opt/coderai/h3_venv",
    "dtype": "bfloat16",
    "offload_strategy": "group",  # group | leaf | ''  (group is the default: 33B)
    "device_map": "",             # per-component device_map instead of offload
    "used_vram_gb": 22.0,         # what to reserve/evict for before starting
    "gpu_device": 0
"""

import collections
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SERVICE_SCRIPT = _REPO_ROOT / "tools" / "h3_service.py"
_REQUIREMENTS = _REPO_ROOT / "requirements-h3.txt"
_BAKED_VENV = Path("/opt/coderai/h3_venv")

# Default reservation before starting the service. H3 is 33B; with group offload
# only the executing group is resident, but the VAEs, the text encoder and the
# activations still want room. Override per model with `used_vram_gb`.
DEFAULT_VRAM_GB = 22.0

_lock = threading.RLock()
_services: dict = {}          # model -> {"proc","port","url"}
_bootstrapped = False


# ── shared rules + in-engine capability ──────────────────────────────────────

_rules = None


def rules():
    """The H3 checkpoint contracts, shared with the isolated worker.

    Loaded by path from tools/h3_common.py: the isolated venv imports it as a
    sibling module, and the engine can't reach it through the `codai.api` package
    (importing that pulls the whole FastAPI app into the worker's venv).
    """
    global _rules
    if _rules is None:
        import importlib.util
        path = _REPO_ROOT / "tools" / "h3_common.py"
        spec = importlib.util.spec_from_file_location("h3_common", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _rules = module
    return _rules


def engine_supports_h3() -> bool:
    """True when THIS process's diffusers carries the H3 modular pipeline.

    H3 needs diffusers >= 0.40. When the engine has it, H3 loads in-engine like
    every other video model (normal eviction, offload and VRAM accounting); when
    it doesn't, the isolated worker is used instead. Auto-detected so the same
    build works either side of a diffusers upgrade.
    """
    try:
        from diffusers import MiniMaxH3ModularPipeline  # noqa: F401
        return True
    except Exception:
        return False


def use_worker(config: dict = None) -> bool:
    """Whether this model should go through the isolated worker.

    ``in_process`` in the model config forces the choice either way; otherwise it
    follows what the engine's diffusers can actually do.
    """
    cfg = config or {}
    if "in_process" in cfg:
        return not bool(cfg.get("in_process"))
    if str(cfg.get("backend") or "").lower() in ("h3-worker", "h3_worker"):
        return True
    return not engine_supports_h3()


# ── detection ─────────────────────────────────────────────────────────────────

def is_h3_model(model_name: str = "", config: dict = None) -> bool:
    """True when this model must go through the H3 worker.

    Either an explicit ``"backend": "h3"`` in models.json, or a name/path that
    carries the checkpoint's identity (MiniMax-H3, minimax_h3, minimax-h3-…).
    """
    cfg = config or {}
    backend = str(cfg.get("backend") or "").strip().lower()
    if backend in ("h3", "minimax-h3", "minimax_h3"):
        return True
    if backend and backend not in ("diffusers", "video", ""):
        return False
    haystack = " ".join(str(x or "") for x in (
        model_name, cfg.get("model_path"), cfg.get("model_id"), cfg.get("path")))
    h = haystack.lower().replace("_", "-")
    return "minimax-h3" in h


def resolve_model_path(model_name: str, config: dict = None) -> str:
    cfg = config or {}
    for key in ("model_path", "path", "model_id", "repo_id"):
        value = (cfg.get(key) or "").strip() if isinstance(cfg.get(key), str) else ""
        if value:
            return os.path.expanduser(value)
    return model_name or "MiniMaxAI/MiniMax-H3"


# ── venv ──────────────────────────────────────────────────────────────────────

def resolve_venv_dir(config: dict = None) -> Path:
    configured = ((config or {}).get("h3_venv") or "").strip()
    if configured:
        return Path(os.path.expanduser(configured))
    explicit = os.environ.get("CODERAI_H3_VENV")
    if explicit:
        return Path(os.path.expanduser(explicit))
    if (_BAKED_VENV / ("Scripts" if os.name == "nt" else "bin")).exists():
        return _BAKED_VENV
    cache = os.environ.get("CODERAI_CACHE_DIR") or ("/cache" if os.path.isdir("/cache") else "")
    if cache and os.path.isdir(cache):
        return Path(cache) / "h3_venv"
    return Path(os.path.expanduser("~/.coderai/h3_venv"))


def _venv_python(venv: Path) -> Path:
    return venv / ("Scripts" if os.name == "nt" else "bin") / (
        "python.exe" if os.name == "nt" else "python")


def _diffusers_ok(py: Path) -> bool:
    """The venv is usable only if its diffusers actually carries the H3 blocks."""
    try:
        return subprocess.run(
            [str(py), "-c", "from diffusers import MiniMaxH3ModularPipeline"],
            capture_output=True).returncode == 0
    except Exception:
        return False


def _bootstrap_venv(config: dict = None) -> Path:
    global _bootstrapped
    venv = resolve_venv_dir(config)
    py = _venv_python(venv)
    if _bootstrapped and py.exists():
        return py
    if py.exists() and _diffusers_ok(py):
        _bootstrapped = True
        return py
    if not py.exists():
        print(f"[h3] creating isolated venv at {venv} …", flush=True)
        venv.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
        py = _venv_python(venv)
    if not _diffusers_ok(py):
        if not _REQUIREMENTS.exists():
            raise RuntimeError(f"H3 requirements file missing: {_REQUIREMENTS}")
        print("[h3] installing diffusers>=0.40 + torch into the isolated venv "
              "(first run downloads several GB, this takes a while) …", flush=True)
        subprocess.run([str(py), "-m", "pip", "install", "-U", "pip"], check=True)
        subprocess.run([str(py), "-m", "pip", "install", "-r", str(_REQUIREMENTS)],
                       check=True)
        if not _diffusers_ok(py):
            raise RuntimeError(
                "H3 venv built but 'from diffusers import MiniMaxH3ModularPipeline' "
                "still fails — MiniMax-H3 needs diffusers >= 0.40.0")
    _bootstrapped = True
    return py


# ── service lifecycle ─────────────────────────────────────────────────────────

def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _pump_logs(proc, tail):
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            tail.append(line)
            print(f"[h3] {line}", flush=True)


def _health_ok(url: str) -> bool:
    import requests
    try:
        r = requests.get(url + "/health", timeout=5)
        return r.ok and bool(r.json().get("ok"))
    except Exception:
        return False


def ensure_service(model_path: str, config: dict = None,
                   ready_timeout: float = 1800.0) -> str:
    """Start (or reuse) the H3 service for one checkpoint; return its base URL."""
    config = config or {}
    with _lock:
        svc = _services.get(model_path)
        if svc and svc["proc"].poll() is None and _health_ok(svc["url"]):
            return svc["url"]
        if svc and svc["proc"].poll() is not None:
            _services.pop(model_path, None)

        py = _bootstrap_venv(config)
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

        offload = str(config.get("offload_strategy") or "group").lower()
        if offload not in ("group", "leaf", ""):
            offload = "group"          # 'model'/'sequential'/'auto' have no modular equivalent
        cmd = [str(py), str(_SERVICE_SCRIPT),
               "--model", model_path,
               "--host", "127.0.0.1", "--port", str(port),
               "--dtype", str(config.get("dtype") or "bfloat16"),
               "--offload", offload]
        if config.get("device_map"):
            cmd += ["--device-map", str(config["device_map"])]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, env=env, cwd=str(_REPO_ROOT))
        tail = collections.deque(maxlen=15)
        threading.Thread(target=_pump_logs, args=(proc, tail), daemon=True).start()
        _services[model_path] = {"proc": proc, "port": port, "url": url}

    def _tail_msg():
        joined = " | ".join(list(tail)[-5:]).strip()
        return f". Last output: {joined}" if joined else ""

    deadline = time.time() + ready_timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            _services.pop(model_path, None)
            raise RuntimeError(
                f"H3 worker exited (code {proc.returncode}) before becoming ready"
                + _tail_msg())
        if _health_ok(url):
            print(f"[h3] service ready for {model_path} at {url}", flush=True)
            return url
        time.sleep(2)
    stop_service(model_path)
    raise RuntimeError(f"H3 worker for {model_path} did not become ready in time"
                       + _tail_msg())


def is_running(model_path: str) -> bool:
    with _lock:
        svc = _services.get(model_path)
        return bool(svc and svc["proc"].poll() is None)


def stop_service(model_path: str) -> None:
    with _lock:
        svc = _services.pop(model_path, None)
    if not svc:
        return
    proc = svc["proc"]
    if proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=15)
        except Exception:
            pass
    if proc.poll() is None:
        try:
            proc.kill()
        except Exception:
            pass
    print(f"[h3] service for {model_path} stopped", flush=True)


def stop_all() -> None:
    for name in list(_services.keys()):
        stop_service(name)


# ── model-manager integration ─────────────────────────────────────────────────

class H3Handle:
    """Evictable handle registered with the model manager.

    ``cleanup()`` is what the manager calls to reclaim VRAM: it stops the whole
    subprocess, which is the only way to be sure the 33B transformer's pages are
    gone (dropping the pipeline inside the worker would leave the process's CUDA
    context and allocator arenas behind).
    """

    def __init__(self, model_path: str, url: str):
        self.model_path = model_path
        self.url = url

    def cleanup(self):
        try:
            stop_service(self.model_path)
        except Exception:
            pass

    # The manager's RAM/VRAM accounting probes these on loaded models.
    def to(self, *args, **kwargs):
        return self


def acquire(model_name: str, config: dict = None) -> H3Handle:
    """Ensure the worker runs and is registered as an evictable, VRAM-tracked model."""
    from codai.models.manager import multi_model_manager
    cfg = config or {}
    model_path = resolve_model_path(model_name, cfg)
    key = f"h3:{model_path}"

    # Idempotent: reuses a healthy worker, re-spawns a dead one (new port).
    url = ensure_service(model_path, cfg)

    def _loader():
        return H3Handle(model_path, url)

    needed = float(cfg.get("used_vram_gb") or DEFAULT_VRAM_GB)
    # acquire_stt_backend is the manager's generic "register an out-of-band backend
    # as an evictable, VRAM-tracked model" entry point (STT is just where it was
    # first needed) — same call the diarization worker uses.
    handle = multi_model_manager.acquire_stt_backend(key, needed, True, _loader,
                                                     keep_resident=False)
    handle.url = url               # a re-spawned worker may sit on a new port
    return handle


def generate(model_name: str, payload: dict, config: dict = None,
             timeout: float = 7200.0) -> dict:
    """Run one generation on the worker. Returns the service's JSON response."""
    import requests
    handle = acquire(model_name, config)
    resp = requests.post(handle.url + "/generate", json=payload, timeout=timeout)
    if not resp.ok:
        detail = resp.text[:800]
        raise RuntimeError(f"H3 generation failed ({resp.status_code}): {detail}")
    data = resp.json()
    if data.get("error"):
        raise RuntimeError(f"H3 generation failed: {data['error']}")
    return data


import atexit as _atexit
_atexit.register(stop_all)
