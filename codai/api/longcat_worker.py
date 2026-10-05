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

"""Fully-managed LongCat-Video worker (Meituan, 13.6B dense, MIT).

LongCat does text-to-video, image-to-video, video-continuation, long-video, interactive
and audio-driven avatar generation in three coarse-to-fine stages. There is no diffusers
pipeline for it — diffusers merged LongCat-*Image*, not the video model — so coderai
drives the upstream repo's own ``LongCatVideoPipeline``.

That cannot happen in this process. LongCat pins Python 3.10, torch 2.6+cu124,
transformers 4.41 and numpy 1.26 against the main venv's 3.13 / 2.11+cu130 / 5.x / 2.4:
four simultaneous conflicts, so it runs in an ISOLATED venv behind a managed subprocess
(``tools/longcat_service.py``), the same shape as vLLM and the OCR Paddle/Surya engines.

This module owns the venv, the process and the HTTP call. The service is registered with
the model manager as an evictable, VRAM-tracked model, so another model can push LongCat
off the card — eviction calls the handle's ``cleanup()``, which stops the subprocess.
Dropping the pipeline inside the worker would leave its CUDA context and allocator arenas
behind, which for a 13.6B DiT is most of the card.

Unlike every other worker here, coderai CANNOT build this venv: each bootstrap in
``codai/`` uses ``sys.executable -m venv``, which is 3.13. The interpreter is the
standalone 3.10 bundled for the lip-sync tools (``/opt/coderai/py310``), and the repo
source is bundled too, because ``longcat_video`` is not on PyPI.

Per-model config keys (models.json entry, all optional)::

    "backend": "longcat",          # or an alias/path naming the checkpoint
    "longcat_source": "",          # repo checkout; blank = bundled/config
    "variant": "bf16",             # bf16 | fp8 | int8
    "quality": "fast",             # draft | fast | best
    "offload_strategy": "",        # '' | model | sequential
    "used_vram_gb": 0,             # what to reserve before starting; 0 = measured
    "gpu_device": 0
"""

import collections
import importlib.util
import os
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SERVICE_SCRIPT = _REPO_ROOT / "tools" / "longcat_service.py"
_REQUIREMENTS = _REPO_ROOT / "requirements-longcat.txt"
_BAKED_VENV = Path("/opt/coderai/longcat_venv")
_BAKED_SRC = Path("/opt/coderai/LongCat-Video")
_BAKED_PY310 = Path("/opt/coderai/py310/bin/python3.10")

# What to reserve before starting, when nothing better is known. The 13.6B DiT is ~27 GB
# at bf16 and the reported peak for a full profile is ~41.6 GB — but no official figure
# exists, and it changes with the variant, the stage and the offload mode. So this is a
# STARTING POINT only: the real number is measured on the first load and written back,
# exactly as the vLLM share is.
DEFAULT_VRAM_GB = 24.0

_lock = threading.RLock()
_services: dict = {}          # key -> {"proc","port","url","meta"}

_common = None


def common():
    """The checkpoint/segment contracts, shared with the isolated service.

    Loaded by path from tools/longcat_common.py: the 3.10 venv cannot import ``codai.*``
    (that would pull FastAPI into it), and this process must not import the venv's torch,
    so anything both sides agree on lives in a stdlib-only module."""
    global _common
    if _common is None:
        spec = importlib.util.spec_from_file_location(
            "longcat_common", _REPO_ROOT / "tools" / "longcat_common.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _common = mod
    return _common


# ── detection ─────────────────────────────────────────────────────────────────

def is_longcat_model(model_name: str = "", config: dict = None) -> bool:
    """True when this model must go through the LongCat worker.

    The backend pin wins. Then the ALIAS and the served name, before the path: an entry
    is routable by ``alias or path or id`` throughout the manager, and aliases are how one
    model carries several configs and how several models share an endpoint — keying on the
    path alone misses an entry aliased ``longcat`` over an HF repo id."""
    cfg = config or {}
    backend = str(cfg.get("backend") or "").strip().lower()
    if backend in ("longcat", "longcat-video", "longcat_video"):
        return True
    if backend and backend not in ("diffusers", "video", "auto", ""):
        return False
    haystack = " ".join(str(x or "") for x in (
        model_name, cfg.get("alias"), cfg.get("model_id"), cfg.get("model_path"),
        cfg.get("path")))
    return "longcat" in haystack.lower().replace("_", "-")


def resolve_model_path(model_name: str, config: dict = None) -> str:
    cfg = config or {}
    for key in ("model_path", "path", "model_id", "repo_id"):
        value = cfg.get(key)
        if isinstance(value, str) and value.strip():
            return os.path.expanduser(value.strip())
    return model_name or "meituan-longcat/LongCat-Video"


def service_key(model_path: str, config: dict = None) -> str:
    """One service per checkpoint AND per config.

    Sibling configs on one path (``config_id``) are DIFFERENT residencies — a bf16 entry
    and an fp8 entry of the same weights have different footprints and cannot share a
    loaded pipeline — so the config id and variant are part of the identity."""
    cfg = config or {}
    cid = str(cfg.get("config_id") or "").strip()
    variant = str(cfg.get("variant") or "bf16").strip().lower()
    suffix = f"#{cid}" if cid else f"#{variant}"
    return f"longcat:{model_path}{suffix}"


# ── venv + source ─────────────────────────────────────────────────────────────

def _cfg_section():
    """This install's ``longcat`` config section, or None."""
    try:
        from codai.admin.routes import config_manager
        return getattr(getattr(config_manager, "config", None), "longcat", None)
    except Exception:
        return None


def resolve_venv_dir(config: dict = None) -> Path:
    configured = ((config or {}).get("longcat_venv") or "").strip()
    if configured:
        return Path(os.path.expanduser(configured))
    sec = _cfg_section()
    if sec is not None and str(getattr(sec, "venv", "") or "").strip():
        return Path(os.path.expanduser(str(sec.venv).strip()))
    explicit = os.environ.get("CODERAI_LONGCAT_VENV")
    if explicit:
        return Path(os.path.expanduser(explicit))
    if (_BAKED_VENV / "bin").exists():
        return _BAKED_VENV
    cache = os.environ.get("CODERAI_CACHE_DIR") or ("/cache" if os.path.isdir("/cache") else "")
    if cache and os.path.isdir(cache):
        return Path(cache) / "longcat_venv"
    return Path(os.path.expanduser("~/.coderai/longcat_venv"))


def resolve_source_dir(config: dict = None) -> Path:
    """Where the upstream repo checkout is. ``longcat_video`` is not on PyPI."""
    configured = ((config or {}).get("longcat_source") or "").strip()
    if configured:
        return Path(os.path.expanduser(configured))
    explicit = os.environ.get("CODERAI_LONGCAT_SRC")
    if explicit:
        return Path(os.path.expanduser(explicit))
    if (_BAKED_SRC / "longcat_video").is_dir():
        return _BAKED_SRC
    return Path(os.path.expanduser("~/.coderai/LongCat-Video"))


def _venv_python(venv: Path) -> Path:
    return venv / "bin" / "python"


def _venv_ok(py: Path) -> bool:
    """Usable only if it is 3.10 with LongCat's torch and transformers.

    A bare ``import torch`` is not enough: a venv re-pointed at the wrong interpreter (the
    image rewrites ``pyvenv.cfg``) imports the MAIN venv's torch 2.11 and would fail deep
    inside the pipeline instead of here."""
    probe = ("import sys,torch,transformers;"
             "assert sys.version_info[:2]==(3,10);"
             "assert torch.__version__.startswith('2.6');"
             "assert transformers.__version__.startswith('4.41')")
    try:
        return subprocess.run([str(py), "-c", probe],
                              capture_output=True, timeout=120).returncode == 0
    except Exception:
        return False


def _find_python310(config: dict = None) -> Optional[str]:
    """A Python 3.10 interpreter to build the venv with.

    ``sys.executable`` is 3.13 and cannot be used — that is the whole reason this venv is
    provisioned out of band."""
    sec = _cfg_section()
    for cand in (((config or {}).get("longcat_python") or "").strip(),
                 str(getattr(sec, "python", "") or "").strip() if sec is not None else "",
                 os.environ.get("CODERAI_LONGCAT_PYTHON") or "",
                 str(_BAKED_PY310)):
        cand = (cand or "").strip()
        if cand and os.path.isfile(os.path.expanduser(cand)):
            return os.path.expanduser(cand)
    return shutil.which("python3.10")


def ensure_built(config: dict = None) -> Path:
    """The interpreter the service runs under. Builds the venv only if allowed to."""
    venv = resolve_venv_dir(config)
    py = _venv_python(venv)
    if py.exists() and _venv_ok(py):
        return py

    sec = _cfg_section()
    auto = bool(getattr(sec, "auto_build", False)) if sec is not None else False
    if not auto:
        raise RuntimeError(
            f"LongCat's venv is not usable at {venv}. It needs Python 3.10 with "
            f"torch 2.6+cu124 and transformers 4.41 — four pins that cannot go in the "
            f"main venv. Build it:\n"
            f"  <python3.10> -m venv {venv}\n"
            f"  {py} -m pip install -r {_REQUIREMENTS}\n"
            f"…or set longcat.auto_build = true (it downloads several GB).")

    py310 = _find_python310(config)
    if not py310:
        raise RuntimeError(
            "longcat.auto_build is on but no Python 3.10 interpreter was found. coderai "
            "cannot create one (its own is 3.13): install python3.10, or set "
            "longcat.python to one, or use the image, which bundles it at "
            f"{_BAKED_PY310}.")
    if not _REQUIREMENTS.exists():
        raise RuntimeError(f"LongCat requirements file missing: {_REQUIREMENTS}")
    if not py.exists():
        print(f"[longcat] creating the venv at {venv} with {py310} …", flush=True)
        venv.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([py310, "-m", "venv", str(venv)], check=True)
        py = _venv_python(venv)
    print("[longcat] installing torch 2.6+cu124 and the rest (several GB, this takes "
          "a while) …", flush=True)
    # Always `python -m pip`, never the venv's pip script: a copied or re-pointed venv
    # carries a stale shebang and would install into the ORIGINAL interpreter.
    subprocess.run([str(py), "-m", "pip", "install", "-U", "pip"], check=True)
    subprocess.run([str(py), "-m", "pip", "install", "-r", str(_REQUIREMENTS)], check=True)
    if not _venv_ok(py):
        raise RuntimeError(
            f"the venv at {venv} was built but still does not report Python 3.10 with "
            f"torch 2.6 / transformers 4.41")
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
            print(f"[longcat] {line}", flush=True)


def _health_ok(url: str) -> bool:
    import requests
    try:
        r = requests.get(url + "/health", timeout=5)
        return r.ok and bool(r.json().get("ok"))
    except Exception:
        return False


def ensure_service(model_path: str, config: dict = None,
                   ready_timeout: Optional[float] = None) -> str:
    """Start (or reuse) the service for one checkpoint+config; return its base URL."""
    config = config or {}
    sec = _cfg_section()

    # A configured service_url points at a LongCat already running somewhere else —
    # another host, a container, a rented pod — so nothing is spawned or downloaded and
    # coderai just proxies. This function returns a URL either way, so its callers cannot
    # tell the difference; that is what makes the engine remotizable for free.
    _remote = (str(config.get("service_url") or "").strip()
               or (str(getattr(sec, "service_url", "") or "").strip() if sec is not None else "")
               or os.environ.get("CODERAI_LONGCAT_SERVICE_URL") or "").strip()
    if _remote:
        _remote = _remote.rstrip("/")
        if not _health_ok(_remote):
            raise RuntimeError(
                f"configured service_url {_remote} is not answering its health check")
        print(f"[longcat] using the configured remote service at {_remote}", flush=True)
        return _remote

    key = service_key(model_path, config)
    with _lock:
        svc = _services.get(key)
        if svc and svc["proc"].poll() is None and _health_ok(svc["url"]):
            return svc["url"]
        if svc and svc["proc"].poll() is not None:
            _services.pop(key, None)

        py = ensure_built(config)
        source = resolve_source_dir(config)
        problems = common().source_problems(str(source))
        if problems:
            raise RuntimeError(
                "; ".join(problems) + f" — clone https://github.com/meituan-longcat/"
                f"LongCat-Video to {source} or set longcat_source on the model")

        port = int(getattr(sec, "port", 0) or 0) or _free_port()
        host = str(getattr(sec, "host", "127.0.0.1") or "127.0.0.1") if sec is not None \
            else "127.0.0.1"
        url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{port}"

        env = dict(os.environ)
        try:
            from codai.models.cache import get_hf_hub_cache_dir
            hub = get_hf_hub_cache_dir()
            env["HF_HUB_CACHE"] = hub
            env["HUGGINGFACE_HUB_CACHE"] = hub
        except Exception:
            pass
        gpu = config.get("gpu_device")
        if gpu is None and sec is not None:
            gpu = str(getattr(sec, "gpu", "") or "") or None
        if gpu is not None and str(gpu) != "":
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        if sec is not None and str(getattr(sec, "extra_env", "") or "").strip():
            import shlex
            for tok in shlex.split(str(sec.extra_env)):
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    if k.strip():
                        env[k.strip()] = v

        offload = str(config.get("offload_strategy") or "").strip().lower()
        if offload in ("auto", "balanced", "disk", "group", "leaf"):
            # The upstream pipeline exposes only the diffusers CPU-offload hooks; the
            # manager's other strategies have no equivalent here.
            offload = "model"
        cmd = [str(py), str(_SERVICE_SCRIPT),
               "--model", model_path,
               "--source", str(source),
               "--host", "127.0.0.1", "--port", str(port),
               "--dtype", str(config.get("dtype") or "bfloat16")]
        if offload:
            cmd += ["--offload", offload]
        if sec is not None and str(getattr(sec, "attention", "") or "").strip():
            cmd += ["--attention", str(sec.attention).strip()]
        if sec is not None and str(getattr(sec, "extra_args", "") or "").strip():
            import shlex
            cmd += shlex.split(str(sec.extra_args))

        print(f"[longcat] launching: {' '.join(cmd)}", flush=True)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, env=env, cwd=str(_REPO_ROOT))
        tail = collections.deque(maxlen=20)
        threading.Thread(target=_pump_logs, args=(proc, tail), daemon=True).start()
        _services[key] = {"proc": proc, "port": port, "url": url, "meta": {}}

    def _tail_msg():
        joined = " | ".join(l.strip()[:300] for l in list(tail) if l.strip()).strip()
        return f". Last output: {joined}" if joined else ""

    budget = float(ready_timeout if ready_timeout is not None
                   else (getattr(sec, "ready_timeout", 1800.0) if sec is not None else 1800.0))
    deadline = time.time() + budget
    while time.time() < deadline:
        if proc.poll() is not None:
            with _lock:
                _services.pop(key, None)
            raise RuntimeError(
                f"LongCat worker exited (code {proc.returncode}) before becoming ready"
                + _tail_msg())
        if _health_ok(url):
            print(f"[longcat] service ready for {key} at {url}", flush=True)
            return url
        time.sleep(2)
    stop_service(key)
    raise RuntimeError(f"LongCat worker for {key} did not become ready in time"
                       + _tail_msg())


def is_running(model_path: str, config: dict = None) -> bool:
    with _lock:
        svc = _services.get(service_key(model_path, config))
        return bool(svc and svc["proc"].poll() is None)


def stop_service(key: str) -> None:
    with _lock:
        svc = _services.pop(key, None)
    if not svc:
        return
    proc = svc["proc"]
    if proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=20)
        except Exception:
            pass
    if proc.poll() is None:
        try:
            proc.kill()
        except Exception:
            pass
    print(f"[longcat] service {key} stopped", flush=True)


def stop_all() -> None:
    for key in list(_services.keys()):
        stop_service(key)


def progress(model_path: str, config: dict = None) -> dict:
    """What the service reports it is doing, or {}. Polled for the progress bar."""
    import requests
    with _lock:
        svc = _services.get(service_key(model_path, config))
        url = svc["url"] if svc else ""
    if not url:
        return {}
    try:
        r = requests.get(url + "/progress", timeout=3)
        return r.json() if r.ok else {}
    except Exception:
        return {}


# ── model-manager integration ─────────────────────────────────────────────────

class LongcatHandle:
    """Evictable handle registered with the model manager.

    ``cleanup()`` stops the whole subprocess, which is the only way to be sure a 13.6B
    DiT's pages are gone: dropping the pipeline inside the worker would leave the CUDA
    context and the allocator arenas behind."""

    def __init__(self, key: str, url: str):
        self.key = key
        self.url = url

    def cleanup(self):
        try:
            stop_service(self.key)
        except Exception:
            pass

    def to(self, *args, **kwargs):      # the manager probes this on loaded models
        return self


def acquire(model_name: str, config: dict = None) -> LongcatHandle:
    """Ensure the service runs and is registered as an evictable, VRAM-tracked model."""
    from codai.models.manager import multi_model_manager
    cfg = config or {}
    model_path = resolve_model_path(model_name, cfg)
    key = service_key(model_path, cfg)

    url = ensure_service(model_path, cfg)        # idempotent; re-spawns on a new port

    def _loader():
        return LongcatHandle(key, url)

    needed = float(cfg.get("used_vram_gb") or 0.0) or DEFAULT_VRAM_GB
    # The manager's generic "register an out-of-band backend as an evictable,
    # VRAM-tracked model" entry point (STT is merely where it was first needed).
    handle = multi_model_manager.acquire_stt_backend(key, needed, True, _loader,
                                                     keep_resident=False)
    handle.url = url                             # a re-spawned worker may have moved
    return handle


def generate(model_name: str, payload: dict, config: dict = None,
             timeout: float = 14400.0) -> dict:
    """Run one generation on the service. Returns its JSON response.

    The default timeout is four hours: a 720p refinement over many segments is measured in
    minutes per segment, and a timeout that fires mid-generation wastes all of it."""
    import requests
    handle = acquire(model_name, config)
    resp = requests.post(handle.url + "/generate", json=payload, timeout=timeout)
    if not resp.ok:
        try:
            detail = resp.json().get("error") or resp.text[:800]
        except Exception:
            detail = resp.text[:800]
        raise RuntimeError(f"LongCat generation failed ({resp.status_code}): {detail}")
    data = resp.json()
    if data.get("error"):
        raise RuntimeError(f"LongCat generation failed: {data['error']}")
    return data


import atexit as _atexit
_atexit.register(stop_all)
