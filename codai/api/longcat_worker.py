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
    "bsa": "off",                  # block-sparse attention: on | off | auto
    "cp_split_hw": "",             # context-parallel tile, e.g. "1x2"
    "used_vram_gb": 0,             # what to reserve before starting; 0 = measured
    "gpu_device": 0
"""

import collections
import importlib.util
import os
import shutil
import socket
import subprocess
import sys
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
# In-flight generations per service, and an event that is SET while nothing is running.
# The model manager's own busy signal cannot see this work (acquire_stt_backend pops the
# model pool, so _is_key_busy is always False for us), so the releaser does its own drain.
_inflight: dict = {}
_idle = threading.Event()
_idle.set()
_releaser_registered = False

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
    # Vendored with coderai: third_party/longcat_video, MIT, pinned to a commit we
    # tested against. Preferred over every fallback below so a fresh install — an
    # image someone else pulls, a container, a pod — has the package already and
    # never needs to clone an upstream repo. See third_party/README.md.
    _vendored = Path(__file__).resolve().parents[2] / "third_party"
    if (_vendored / "longcat_video").is_dir():
        return _vendored
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
             "assert transformers.__version__.startswith('4.41');"
             "print(sys.version_info[0],sys.version_info[1],"
             "torch.__version__,transformers.__version__)")
    try:
        # Importing torch 2.6 from a 6 GB venv takes ~106s on an IDLE machine and
        # longer while the GPU box is busy, which is exactly when a generation is
        # requested. The old 120s limit therefore expired under load, the timeout
        # was caught as a failure, and a perfectly good venv was declared broken.
        done = subprocess.run([str(py), "-c", probe], capture_output=True,
                              text=True, timeout=900, env=_clean_py_env())
    except subprocess.TimeoutExpired:
        # NOT the same as a failed check: we could not finish asking. Saying the
        # venv is wrong here is a lie, and it is the lie that sent three
        # generations into a rebuild loop.
        print(f"[longcat] the venv check at {py} did not finish in 900s — the box "
              f"is probably loaded; treating it as unverified, not broken",
              flush=True)
        return None
    except Exception as exc:                                   # noqa: BLE001
        print(f"[longcat] could not run the venv check: {exc}", flush=True)
        return False
    if done.returncode == 0:
        return True
    detail = (done.stderr or done.stdout or "").strip().splitlines()
    print(f"[longcat] the venv at {py.parent.parent} is not usable: "
          f"{detail[-1] if detail else 'the probe failed with no output'}", flush=True)
    return False


def _clean_py_env() -> dict:
    """Environment for spawning ANOTHER Python than this process's.

    Anything that points a Python at ANOTHER interpreter's files makes the child
    die at startup with "No module named 'encodings'" — PYTHONHOME is the usual
    one, but PYTHONPATH, VIRTUAL_ENV and friends all leak the 3.13 runtime into a
    3.10 process. coderai does not set them today; it is stripped anyway because
    the failure mode is fatal, silent in the parent, and costs nothing to prevent.
    """
    env = os.environ.copy()
    for var in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONEXECUTABLE",
                "VIRTUAL_ENV", "PYTHONNOUSERSITE", "PYTHONUSERBASE",
                "__PYVENV_LAUNCHER__"):
        env.pop(var, None)
    return env


def _run_py310(cmd: list, what: str) -> None:
    """Run a step of the venv build, and RAISE WITH WHAT THE CHILD SAID.

    subprocess.run(check=True) reports an exit status and nothing else, so the
    first real failure here was a bare "returned non-zero exit status 1" while the
    reason — a fatal interpreter error — only reached the log by accident.
    """
    proc = subprocess.run(cmd, env=_clean_py_env(), capture_output=True, text=True)
    if proc.returncode == 0:
        return
    detail = ((proc.stderr or "") + ("\n" + proc.stdout if proc.stdout else "")).strip()
    if len(detail) > 2000:
        detail = detail[:1000] + "\n  …\n" + detail[-900:]
    raise RuntimeError(
        f"{what} failed (exit {proc.returncode}): {' '.join(str(c) for c in cmd)}\n"
        f"{detail or '(the command produced no output)'}")


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
    # `is not False` on purpose: the check is tri-state. True is verified, False
    # is a real mismatch, None means the probe could not finish (a loaded box —
    # importing torch 2.6 takes ~106s idle). An unverified venv is used; only a
    # verified-bad one is rebuilt.
    if py.exists() and _venv_ok(py) is not False:
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
        _run_py310([py310, "-m", "venv", str(venv)], "creating the LongCat venv")
        py = _venv_python(venv)
    print("[longcat] installing torch 2.6+cu124 and the rest (several GB, this takes "
          "a while) …", flush=True)
    # Always `python -m pip`, never the venv's pip script: a copied or re-pointed venv
    # carries a stale shebang and would install into the ORIGINAL interpreter.
    _run_py310([str(py), "-m", "pip", "install", "-U", "pip"], "upgrading pip")
    _run_py310([str(py), "-m", "pip", "install", "-r", str(_REQUIREMENTS)],
               "installing requirements-longcat.txt")
    if _venv_ok(py) is False:
        raise RuntimeError(
            f"the venv at {venv} was built but still does not report Python 3.10 with "
            f"torch 2.6 / transformers 4.41 — see the [longcat] line above for what "
            f"it did report")
    return py


# ── service lifecycle ─────────────────────────────────────────────────────────

def _visible_gpu_count(env: dict = None) -> int:
    """How many CUDA devices this service will see, or 0 when it cannot be told.

    Asking for more context-parallel ranks than there are GPUs fails inside NCCL with a
    message about nothing in particular, so it is worth catching here."""
    sel = str((env or os.environ).get("CUDA_VISIBLE_DEVICES") or "").strip()
    if sel:
        return len([x for x in sel.split(",") if x.strip() != ""])
    try:
        from codai.models.gpu_query import visible_gpu_memory
        return len(visible_gpu_memory() or [])
    except Exception:
        return 0


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

        # Same sanitising as the venv build: this launches the 3.10 interpreter,
        # and a leaked PYTHONHOME/PYTHONPATH kills it before it prints anything
        # useful.
        env = _clean_py_env()
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
        # Context parallelism: upstream splits the DiT's spatial dims across GPUs under
        # torchrun, so cp_size > 1 means N processes, not N threads. Only rank 0 binds
        # the port; the others take their jobs over a broadcast (see the service).
        cp = 0
        try:
            cp = int(config.get("cp_size") or 0)
        except (TypeError, ValueError):
            cp = 0
        if cp > 1:
            visible = _visible_gpu_count(env)
            if visible and cp > visible:
                raise RuntimeError(
                    f"cp_size is {cp} but only {visible} GPU(s) are visible to this "
                    f"service — lower it, or widen longcat.gpu / gpu_device")
            launcher = [str(py), "-m", "torch.distributed.run",
                        f"--nproc_per_node={cp}", "--nnodes=1",
                        "--rdzv-backend=c10d", "--rdzv-endpoint=127.0.0.1:0"]
        else:
            launcher = [str(py)]
        cmd = launcher + [str(_SERVICE_SCRIPT),
               "--model", model_path,
               "--source", str(source),
               "--host", "127.0.0.1", "--port", str(port),
               "--dtype", str(config.get("dtype") or "bfloat16")]
        if cp > 1:
            cmd += ["--context-parallel-size", str(cp)]
        if offload:
            cmd += ["--offload", offload]
        if sec is not None and str(getattr(sec, "attention", "") or "").strip():
            cmd += ["--attention", str(sec.attention).strip()]
        # Per-model first, then the server-wide longcat section: a checkpoint that is
        # too lossy under BSA has to be able to opt out on its own.
        bsa = str(config.get("bsa") or
                  (getattr(sec, "bsa", "") if sec is not None else "") or "").strip()
        if bsa:
            cmd += ["--bsa", bsa]
        split = str(config.get("cp_split_hw") or "").strip()
        if split:
            cmd += ["--cp-split-hw", split]
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


# ── LoRA / QLoRA training ────────────────────────────────────────────────────

_TRAIN_SCRIPT = _REPO_ROOT / "tools" / "longcat_train.py"
_QUANTIZE_SCRIPT = _REPO_ROOT / "tools" / "longcat_quantize.py"
_TRAIN_REQUIREMENTS = _REPO_ROOT / "requirements-longcat-train.txt"


def resolve_train_venv(config: dict = None) -> Path:
    """Where SimpleTuner lives. A venv of its OWN, not the inference one.

    SimpleTuner pins its own torch; sharing the inference venv would have the two fight
    over it and leave neither working."""
    sec = _cfg_section()
    configured = ((config or {}).get("longcat_train_venv") or "").strip()
    if configured:
        return Path(os.path.expanduser(configured))
    if sec is not None and str(getattr(sec, "train_venv", "") or "").strip():
        return Path(os.path.expanduser(str(sec.train_venv).strip()))
    explicit = os.environ.get("CODERAI_LONGCAT_TRAIN_VENV")
    if explicit:
        return Path(os.path.expanduser(explicit))
    # Beside the inference venv, clearly named.
    inference = resolve_venv_dir(config)
    return inference.parent / (inference.name + "-train")


def _train_venv_ok(py: Path):
    """True / False / None, the same tri-state as _venv_ok and for the same reason:
    importing simpletuner pulls its own torch, which is slow on a busy box, and a
    timeout is not evidence that the venv is wrong."""
    try:
        done = subprocess.run([str(py), "-c", "import simpletuner"],
                              capture_output=True, text=True, timeout=900,
                              env=_clean_py_env())
    except subprocess.TimeoutExpired:
        print(f"[longcat] the training venv check at {py} did not finish in 900s — "
              f"treating it as unverified, not broken", flush=True)
        return None
    except Exception as exc:                                   # noqa: BLE001
        print(f"[longcat] could not run the training venv check: {exc}", flush=True)
        return False
    if done.returncode == 0:
        return True
    detail = (done.stderr or "").strip().splitlines()
    print(f"[longcat] `import simpletuner` fails: "
          f"{detail[-1] if detail else 'no output'}", flush=True)
    return False


def _venv_python_version(py: Path) -> tuple:
    """(major, minor) of a venv's interpreter, or () if it cannot be determined.

    Read from pyvenv.cfg rather than by running it: the venv may have been built by
    an interpreter that is no longer there, and a failed exec would be
    indistinguishable from a venv that is merely incomplete.
    """
    cfg = Path(py).parent.parent / "pyvenv.cfg"
    try:
        for line in cfg.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            if key.strip() in ("version", "version_info"):
                parts = value.strip().split(".")
                return (int(parts[0]), int(parts[1]))
    except (OSError, ValueError, IndexError):
        pass
    return ()


def _find_train_python(config: dict = None) -> Optional[str]:
    """An interpreter SimpleTuner can be installed into.

    NOT the 3.10 the inference venv uses. That pin exists for LongCat itself
    (torch 2.6+cu124, transformers 4.41); SimpleTuner requires >=3.12, so asking
    pip for it under 3.10 yields "No matching distribution found for simpletuner"
    — every release filtered out by Requires-Python, which reads like the package
    does not exist.

    coderai's own interpreter is 3.13 and satisfies it, which is why the training
    venv — unlike the inference one — can be built without anything bundled.
    """
    sec = _cfg_section()
    configured = (((config or {}).get("longcat_train_python") or "").strip()
                  or (str(getattr(sec, "train_python", "") or "").strip()
                      if sec is not None else "")
                  or os.environ.get("CODERAI_LONGCAT_TRAIN_PYTHON") or "")
    if configured and os.path.isfile(os.path.expanduser(configured)):
        return os.path.expanduser(configured)
    if (3, 12) <= sys.version_info[:2] < (3, 15):
        return sys.executable
    for name in ("python3.13", "python3.12", "python3.14"):
        found = shutil.which(name)
        if found:
            return found
    return None


def ensure_train_built(config: dict = None) -> Path:
    """The interpreter SimpleTuner runs under. Builds it only if allowed to."""
    venv = resolve_train_venv(config)
    py = _venv_python(venv)
    if py.exists() and _train_venv_ok(py) is not False:
        return py
    sec = _cfg_section()
    auto = bool(getattr(sec, "train_auto_build", False)) if sec is not None else False
    if not auto:
        raise RuntimeError(
            f"LongCat LoRA training needs SimpleTuner, which is not in {venv}. "
            f"coderai does not implement this training loop: LongCat's venv is "
            f"standalone so it cannot reuse the shared trainer, and the upstream repo "
            f"has no way to create trainable LoRA layers (only load_lora from a file, "
            f"in its own key layout). Build it:\n"
            f"  <python3.12+> -m venv {venv}\n"
            f"  {py} -m pip install -r {_TRAIN_REQUIREMENTS}\n"
            f"…or set longcat.train_auto_build = true.")
    train_py = _find_train_python(config)
    if not train_py:
        raise RuntimeError(
            f"longcat.train_auto_build is on but no Python 3.12-3.14 was found for "
            f"SimpleTuner (this process is {sys.version_info.major}."
            f"{sys.version_info.minor}) — set longcat.train_python to one.")
    # A venv left by an interpreter SimpleTuner cannot use is worse than no venv:
    # creation is skipped because the directory exists, pip installs into the wrong
    # Python, and it fails identically forever. Not hypothetical — the 3.10 attempt
    # that produced "No matching distribution found for simpletuner" left exactly
    # that behind.
    existing = _venv_python_version(py) if venv.exists() else ()
    if existing and not ((3, 12) <= existing < (3, 15)):
        import shutil as _shutil
        print(f"[longcat] the training venv at {venv} is Python "
              f"{existing[0]}.{existing[1]}, which SimpleTuner cannot use — "
              f"rebuilding it", flush=True)
        _shutil.rmtree(venv, ignore_errors=True)
        py = _venv_python(venv)

    if not py.exists():
        print(f"[longcat] creating the training venv at {venv} …", flush=True)
        venv.parent.mkdir(parents=True, exist_ok=True)
        _run_py310([train_py, "-m", "venv", str(venv)], "creating the LongCat training venv")
        py = _venv_python(venv)
    _run_py310([str(py), "-m", "pip", "install", "-U", "pip"], "upgrading pip")
    _run_py310([str(py), "-m", "pip", "install", "-r", str(_TRAIN_REQUIREMENTS)],
               "installing requirements-longcat-train.txt")
    if _train_venv_ok(py) is False:
        raise RuntimeError(f"the training venv at {venv} was built but `import "
                           f"simpletuner` still fails — see the [longcat] line above")
    return py


def train_lora(job: dict, workdir: str, on_progress=None) -> dict:
    """Run one LoRA/QLoRA job. Returns SimpleTuner's result dict.

    ``job`` is written to disk and the trainer appends JSON lines beside it, the same
    protocol tools/lora_train_worker.py uses — so the existing job records, the progress
    endpoint and the Tasks page need no changes.

    GPU exclusivity is the CALLER's business: training wants the whole card, and the
    model manager's releasers (including this module's) are what clear it."""
    import json as _json

    sec = _cfg_section()
    py = ensure_train_built(job.get("request") or {})
    work = Path(os.path.expanduser(workdir))
    work.mkdir(parents=True, exist_ok=True)
    job.setdefault("base_precision",
                   str(getattr(sec, "train_base_precision", "") or "")
                   if sec is not None else "")
    job.setdefault("rank", int(getattr(sec, "train_lora_rank", 8) or 8)
                   if sec is not None else 8)
    job.setdefault("gradient_checkpointing",
                   bool(getattr(sec, "train_gradient_checkpointing", True))
                   if sec is not None else True)

    job_path = work / "job.json"
    job_path.write_text(_json.dumps(job, indent=2, default=str))
    progress_path = str(job_path) + ".progress"
    result_path = str(job_path) + ".result"

    cmd = [str(py), str(_TRAIN_SCRIPT), "--job", str(job_path)]
    print(f"[longcat] training: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, cwd=str(_REPO_ROOT),
                            env=_clean_py_env())
    threading.Thread(target=_pump_logs, args=(proc, collections.deque(maxlen=20)),
                     daemon=True).start()

    seen = 0
    while proc.poll() is None:
        seen = _drain_progress(progress_path, seen, on_progress)
        time.sleep(2)
    _drain_progress(progress_path, seen, on_progress)

    if not os.path.isfile(result_path):
        raise RuntimeError(f"the trainer exited {proc.returncode} without writing a "
                           f"result — see the [longcat] log above")
    data = _json.loads(open(result_path).read() or "{}")
    if not data.get("ok"):
        raise RuntimeError(data.get("error") or "LongCat LoRA training failed")
    return data.get("result") or {}


def quantize_int8(checkpoint_dir: str, workdir: str, on_progress=None,
                  subfolder: str = "dit", overwrite: bool = False,
                  config: dict = None) -> dict:
    """Write an INT8 copy of a checkpoint's DiT beside the bf16 one.

    Runs in the INFERENCE venv (it needs the upstream quantisation helpers and
    torch 2.6), not the training one. Never automatic and never destructive: the
    bf16 weights stay exactly where they were, and an existing base_model_int8/ is
    left alone unless `overwrite` says otherwise.

    Returns the worker's result dict: {"path": ..., "bytes": ...}.
    """
    import json as _json

    py = ensure_built(config)
    work = Path(os.path.expanduser(workdir))
    work.mkdir(parents=True, exist_ok=True)
    job = {
        "checkpoint_dir": str(checkpoint_dir),
        "source_dir": str(resolve_source_dir(config)),
        "subfolder": subfolder,
        "out_subfolder": "base_model_int8",
        "overwrite": bool(overwrite),
    }
    job_path = work / "job.json"
    job_path.write_text(_json.dumps(job, indent=2, default=str))
    progress_path = str(job_path) + ".progress"
    result_path = str(job_path) + ".result"

    cmd = [str(py), str(_QUANTIZE_SCRIPT), "--job", str(job_path)]
    print(f"[longcat] quantising: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, cwd=str(_REPO_ROOT),
                            env=_clean_py_env())
    threading.Thread(target=_pump_logs, args=(proc, collections.deque(maxlen=20)),
                     daemon=True).start()

    seen = 0
    while proc.poll() is None:
        seen = _drain_progress(progress_path, seen, on_progress)
        time.sleep(2)
    _drain_progress(progress_path, seen, on_progress)

    if not os.path.isfile(result_path):
        raise RuntimeError(f"the quantiser exited {proc.returncode} without writing a "
                           f"result — see the [longcat] log above")
    data = _json.loads(open(result_path).read() or "{}")
    if not data.get("ok"):
        raise RuntimeError(data.get("error") or "LongCat INT8 quantisation failed")
    return data


def _drain_progress(path: str, seen: int, on_progress) -> int:
    """Forward new JSON lines to the caller. Returns how many have been consumed."""
    if not on_progress or not os.path.isfile(path):
        return seen
    import json as _json
    try:
        lines = open(path).read().splitlines()
    except Exception:
        return seen
    for line in lines[seen:]:
        try:
            on_progress(**_json.loads(line))
        except Exception:
            pass
    return len(lines)


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

    register_releaser()                          # both directions, from the first use
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


def _record_measurement(key: str, vram_gb: float) -> None:
    """Write a measured footprint back, so the next reservation is not an estimate.

    The 13.6B DiT is ~27 GB at bf16 and the reported peak for a full profile is ~41.6 GB,
    but no official figure exists and it moves with the variant, the stage and the offload
    mode. So the number is measured per service key — which includes config_id/variant,
    because sibling configs are different residencies and an fp8 entry must not inherit a
    bf16 measurement."""
    if not vram_gb or vram_gb <= 0:
        return
    try:
        from codai.models.manager import multi_model_manager
        prev = float(multi_model_manager._measured_vram_gb.get(key) or 0.0)
        # Keep the HIGH-WATER mark: a draft-preset run touches far less of the card than
        # a 720p refinement, and reserving the smaller figure would under-evict for the
        # bigger one later.
        if vram_gb > prev + 0.05:
            multi_model_manager._measured_vram_gb[key] = float(vram_gb)
            print(f"[longcat] measured {vram_gb:.2f} GB for {key}"
                  + (f" (was {prev:.2f})" if prev else ""), flush=True)
    except Exception as e:
        print(f"[longcat] could not record the measurement: {e}", flush=True)


def generate(model_name: str, payload: dict, config: dict = None,
             timeout: float = 14400.0) -> dict:
    """Run one generation on the service. Returns its JSON response.

    The default timeout is four hours: a 720p refinement over many segments is measured in
    minutes per segment, and a timeout that fires mid-generation wastes all of it."""
    import requests
    cfg = config or {}
    handle = acquire(model_name, cfg)
    key = handle.key
    with _lock:
        _inflight[key] = _inflight.get(key, 0) + 1
    try:
        resp = requests.post(handle.url + "/generate", json=payload, timeout=timeout)
    finally:
        with _lock:
            _inflight[key] = max(0, _inflight.get(key, 1) - 1)
            if _inflight[key] == 0:
                _idle.set()
    if not resp.ok:
        try:
            detail = resp.json().get("error") or resp.text[:800]
        except Exception:
            detail = resp.text[:800]
        raise RuntimeError(f"LongCat generation failed ({resp.status_code}): {detail}")
    data = resp.json()
    if data.get("error"):
        raise RuntimeError(f"LongCat generation failed: {data['error']}")
    _record_measurement(key, float(data.get("vram_gb") or 0.0))
    return data


# ── eviction: giving the card back ───────────────────────────────────────────

def _request_yield(key: str) -> bool:
    """Ask a service to stop at its next segment boundary. Best-effort."""
    import requests
    with _lock:
        svc = _services.get(key)
        url = svc["url"] if svc else ""
    if not url:
        return False
    try:
        return bool(requests.post(url + "/yield", json={}, timeout=5).ok)
    except Exception:
        return False


def _drain_timeout_s() -> float:
    sec = _cfg_section()
    try:
        return max(0.0, float(getattr(sec, "evict_drain_timeout_s", 0) or 0)) or 300.0
    except Exception:
        return 300.0


def release_vram(needed_gb: float = 999.0) -> float:
    """External VRAM releaser: stop the services, letting in-flight work finish first.

    Registered with the model manager so another model can take the card — the reverse
    direction of acquire()'s eviction. The contract is ``fn(needed_gb) -> float``: safe to
    call with nothing to free, and it must not raise (the manager swallows exceptions into
    a warning, which is how voice_clone's zero-argument releaser has been silently failing).

    The wait matters more here than anywhere else. A segment is ~80 new frames and minutes
    of work, and the manager's own busy signal cannot see any of it: acquire_stt_backend
    pops the model pool, so ``_is_key_busy`` is permanently False for this model and Pass 1
    of release_idle_vram would tear the subprocess down mid-generation, losing the whole
    request. Upstream already calls torch_gc() between segments, so a boundary exists;
    until the segment loop lands, draining means waiting for the in-flight request."""
    with _lock:
        keys = list(_services.keys())
        busy = sum(_inflight.get(k, 0) for k in keys)
    if not keys:
        return 0.0
    if busy:
        budget = _drain_timeout_s()
        # Ask the services to stop at their next SEGMENT boundary. Upstream already calls
        # torch_gc() there, so it is a genuinely safe point, and it turns the wait from
        # "however long the whole request takes" into "one more segment" — the in-flight
        # request still returns, shorter, with a warning, instead of being lost.
        for key in keys:
            _request_yield(key)
        print(f"[longcat] {busy} generation(s) in flight — asked them to yield at the "
              f"next segment boundary, waiting up to {budget:.0f}s", flush=True)
        _idle.clear()
        if not _idle.wait(budget):
            print(f"[longcat] still busy after {budget:.0f}s — releasing anyway; the "
                  f"in-flight request will fail", flush=True)

    freed = 0.0
    try:
        from codai.models.manager import multi_model_manager
        for key in keys:
            freed += float(multi_model_manager._measured_vram_gb.get(key) or 0.0) \
                or DEFAULT_VRAM_GB
    except Exception:
        freed = float(len(keys)) * DEFAULT_VRAM_GB
    for key in keys:
        stop_service(key)
    if freed:
        print(f"[longcat] released ~{freed:.1f} GB (services stopped for VRAM "
              f"eviction); they restart on the next request", flush=True)
    return freed


def register_releaser() -> None:
    """Make LongCat's VRAM reclaimable by the model manager. Idempotent."""
    global _releaser_registered
    if _releaser_registered:
        return
    try:
        from codai.models.manager import multi_model_manager
        multi_model_manager.register_external_vram_releaser(release_vram)
        _releaser_registered = True
    except Exception as e:
        print(f"[longcat] could not register the VRAM releaser: {e}", flush=True)


import atexit as _atexit
_atexit.register(stop_all)
