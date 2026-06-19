# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Fully-managed ds4 (DeepSeek V4) worker.

ds4 / DwarfStar (https://github.com/antirez/ds4) is a native inference engine for
DeepSeek V4 that ships its own OpenAI-compatible HTTP server (``ds4-server``). It is
not a Python package — it's a C/CUDA/Metal binary built with ``make`` — so coderai
owns the whole lifecycle here, mirroring :mod:`codai.api.parler_worker`:

* :func:`ensure_built` clones the repo and runs the right ``make`` target so the
  ``ds4-server`` binary exists (idempotent).
* :func:`ensure_model` runs the project's ``download_model.sh`` for the configured
  weight variant if the model isn't already present.
* :func:`ensure_service` launches ``ds4-server`` on a free port, health-checks its
  ``/v1/models`` endpoint, and returns the base URL.

The matching ``Ds4Backend.cleanup()`` calls :func:`stop_service`, so the model
manager's normal eviction tears the process down.
"""

import os
import platform
import shutil
import socket
import subprocess
import sys
import threading
import time
import collections
from pathlib import Path
from typing import Optional
from typing import Optional

_lock = threading.RLock()
# Single managed server (ds4 serves one DeepSeek V4 model). Keyed by model_id so a
# config change to a different id restarts cleanly.
_services: dict[str, dict] = {}   # model_id -> {"proc","port","url"}
_built = False


def default_install_dir() -> Path:
    return Path(os.environ.get("CODERAI_DS4_DIR")
                or os.path.expanduser("~/.coderai/ds4"))


def _install_dir(cfg) -> Path:
    return Path(cfg.install_dir).expanduser() if getattr(cfg, "install_dir", None) \
        else default_install_dir()


def _server_bin(install_dir: Path) -> Path:
    return install_dir / "ds4-server"


def _detect_build_target() -> str:
    """Pick a ``make`` target from the host: CUDA → cuda-generic, macOS → metal."""
    if platform.system() == "Darwin":
        return "metal"
    if shutil.which("nvcc") or os.path.isdir("/usr/local/cuda"):
        return "cuda-generic"
    return "cpu"


def _resolve_target(cfg) -> str:
    target = (getattr(cfg, "build_target", "auto") or "auto").strip()
    if target in ("", "auto"):
        return _detect_build_target()
    # ds4's macOS Metal target is the bare ``make`` (no suffix).
    return "" if target == "metal" else target


def _run_logged(cmd, cwd, label, tail, **kw):
    """Run a subprocess, streaming its output with a ``[ds4]`` prefix into ``tail``."""
    print(f"[ds4] $ {' '.join(str(c) for c in cmd)}", flush=True)
    proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1, **kw)
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            tail.append(line)
            print(f"[ds4] {line}", flush=True)
    proc.wait()
    if proc.returncode != 0:
        joined = " | ".join(list(tail)[-5:])
        raise RuntimeError(f"{label} failed (exit {proc.returncode}). {joined}")


def ensure_built(cfg) -> Path:
    """Clone + build ds4 if the ``ds4-server`` binary is missing. Returns its path."""
    global _built
    install_dir = _install_dir(cfg)
    binary = _server_bin(install_dir)
    if binary.exists():
        _built = True
        return binary
    if not getattr(cfg, "auto_build", True):
        raise RuntimeError(
            f"ds4-server not found at {binary} and auto_build is disabled. Build ds4 "
            f"manually (git clone {cfg.repo_url}; make <target>) or enable auto_build.")

    tail = collections.deque(maxlen=40)
    install_dir.parent.mkdir(parents=True, exist_ok=True)
    if not (install_dir / ".git").exists() and not (install_dir / "Makefile").exists():
        print(f"[ds4] cloning {cfg.repo_url} → {install_dir} …", flush=True)
        _run_logged(["git", "clone", "--depth", "1", cfg.repo_url, str(install_dir)],
                    cwd=install_dir.parent, label="git clone", tail=tail)

    target = _resolve_target(cfg)
    make_cmd = ["make"] + ([target] if target else [])
    print(f"[ds4] building ds4 (make {target or 'metal'}) — this can take a while …",
          flush=True)
    _run_logged(make_cmd, cwd=install_dir, label="make", tail=tail)

    if not binary.exists():
        raise RuntimeError(
            f"ds4 build completed but {binary} is missing. Last output: "
            + " | ".join(list(tail)[-5:]))
    _built = True
    print(f"[ds4] built {binary}", flush=True)
    return binary


def ensure_model(cfg) -> None:
    """Download the configured GGUF weight variant if it isn't present.

    ds4's ``download_model.sh`` writes into ``<install_dir>/gguf/`` and updates the
    ``ds4flash.gguf`` symlink that ``ds4-server`` loads by default.
    """
    install_dir = _install_dir(cfg)
    # If the default-loaded model file already resolves, nothing to do.
    default_model = install_dir / "ds4flash.gguf"
    if default_model.exists():
        return
    script = install_dir / "download_model.sh"
    if not script.exists():
        # The binary may have been bundled (e.g. into a Docker image) without the
        # repo scripts. Shallow-clone the repo to get download_model.sh — this does
        # not rebuild the already-present binary.
        tail0 = collections.deque(maxlen=20)
        if (install_dir / ".git").exists() or (install_dir / "Makefile").exists():
            _run_logged(["git", "-C", str(install_dir), "pull", "--ff-only"],
                        cwd=install_dir, label="git pull", tail=tail0)
        else:
            tmp = install_dir.parent / (install_dir.name + ".repo")
            shutil.rmtree(tmp, ignore_errors=True)
            _run_logged(["git", "clone", "--depth", "1", cfg.repo_url, str(tmp)],
                        cwd=install_dir.parent, label="git clone (scripts)", tail=tail0)
            # Copy repo files into install_dir without clobbering the built binary.
            for item in tmp.iterdir():
                dest = install_dir / item.name
                if dest.exists():
                    continue
                if item.is_dir():
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)
            shutil.rmtree(tmp, ignore_errors=True)
    if not script.exists():
        raise RuntimeError(f"ds4 download script not found at {script}")
    variant = (getattr(cfg, "model_variant", "") or "q4-imatrix").strip()
    tail = collections.deque(maxlen=40)
    print(f"[ds4] downloading DeepSeek V4 weights (variant '{variant}', multi-GB, "
          f"resumable) …", flush=True)
    _run_logged(["bash", str(script), variant], cwd=install_dir,
                label="download_model.sh", tail=tail)


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
            print(f"[ds4] {line}", flush=True)


def _health_ok(url: str) -> bool:
    import requests
    try:
        r = requests.get(url + "/v1/models", timeout=3)
        return r.ok
    except Exception:
        return False


def _coderai_offload_dir() -> str:
    """coderai's configured disk-offload directory (config.offload.directory), or ''."""
    try:
        from codai.admin.routes import config_manager
        if config_manager is not None and config_manager.config is not None:
            d = (getattr(config_manager.config.offload, "directory", "") or "").strip()
            return os.path.expanduser(d) if d else ""
    except Exception:
        pass
    return ""


def resolve_service_key(cfg, model_file: Optional[str] = None):
    """Decide which GGUF ds4-server should serve and the key to cache it under.

    Preference: the requested model's own ``.gguf`` path → an explicit
    ``cfg.model_path`` override → '' (download the variant as a last resort).
    Returns ``(resolved_gguf_or_'', svc_key)``; the key is the file when we have
    one (so different deepseek4 models get their own server), else ``model_id``.
    """
    resolved = ""
    for cand in (model_file, getattr(cfg, "model_path", "") or ""):
        cand = os.path.expanduser((cand or "").strip())
        if cand and cand.lower().endswith(".gguf") and os.path.isfile(cand):
            resolved = cand
            break
    svc_key = resolved or (getattr(cfg, "model_id", "deepseek-v4") or "deepseek-v4")
    return resolved, svc_key


def ensure_service(cfg, model_file: Optional[str] = None,
                   ctx: Optional[int] = None,
                   ready_timeout: float = 3600.0) -> str:
    """Build (as needed), then start (or reuse) ds4-server serving the right GGUF.

    ``model_file`` is the requested model's path; when it resolves to a local
    ``.gguf`` (or ``cfg.model_path`` is set) ds4-server loads THAT via ``-m`` and
    no weights are downloaded. Only when neither resolves does it fall back to
    downloading ``cfg.model_variant``. ``ctx`` overrides the ds4 global context
    (so the per-model n_ctx configuration wins). Returns the base URL.
    """
    resolved, svc_key = resolve_service_key(cfg, model_file)
    with _lock:
        svc = _services.get(svc_key)
        if svc and svc["proc"].poll() is None and _health_ok(svc["url"]):
            return svc["url"]
        if svc and svc["proc"].poll() is not None:
            _services.pop(svc_key, None)   # died — restart below

        binary = ensure_built(cfg)
        if not resolved:
            # No local deepseek4 GGUF resolved. Downloading ds4's own variant is
            # OPT-IN (auto_download, off by default) — otherwise fail with a clear
            # message instead of silently pulling tens of GB.
            if bool(getattr(cfg, "auto_download", False)):
                ensure_model(cfg)
            else:
                raise RuntimeError(
                    "ds4: no local deepseek4 GGUF resolved for this request and "
                    "auto-download is disabled. Point the model at a deepseek4 "
                    ".gguf (or set ds4.model_path), or enable ds4.auto_download to "
                    "fetch the configured weight variant.")

        install_dir = _install_dir(cfg)
        host = getattr(cfg, "host", "127.0.0.1") or "127.0.0.1"
        port = int(getattr(cfg, "port", 0) or 0) or _free_port()
        # ds4-server binds the requested host; build the URL from a loopback address
        # for our own health checks / proxying when it's bound to 0.0.0.0.
        connect_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
        url = f"http://{connect_host}:{port}"

        # Per-model n_ctx (passed in) wins over the ds4 global ctx setting.
        try:
            ctx_val = int(ctx) if ctx else 0
        except (TypeError, ValueError):
            ctx_val = 0
        if ctx_val <= 0:
            ctx_val = int(getattr(cfg, "ctx", 100000) or 100000)

        cmd = [str(binary), "--host", host, "--port", str(port),
               "--ctx", str(ctx_val),
               "--chdir", str(install_dir)]
        if resolved:
            cmd += ["-m", resolved]
        if bool(getattr(cfg, "ssd_streaming", False)):
            # Stream MoE experts from SSD/disk instead of full residency — lets a
            # 100GB+ model run on a small GPU + modest RAM (slow but works).
            cmd += ["--ssd-streaming"]
        extra = (getattr(cfg, "extra_args", "") or "").strip()
        # Route ds4's on-disk KV checkpoints into coderai's offload directory by
        # default (unless the user set --kv-disk-dir themselves in extra_args).
        if "--kv-disk-dir" not in extra:
            off = _coderai_offload_dir()
            if off:
                kvdir = os.path.join(off, "ds4-kv")
                try:
                    os.makedirs(kvdir, exist_ok=True)
                    cmd += ["--kv-disk-dir", kvdir]
                except OSError:
                    pass
        if extra:
            import shlex
            cmd += shlex.split(extra)

        print(f"[ds4] launching ds4-server: {' '.join(cmd)}", flush=True)
        proc = subprocess.Popen(
            cmd, cwd=str(install_dir), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        tail = collections.deque(maxlen=15)
        threading.Thread(target=_pump_logs, args=(proc, tail), daemon=True).start()
        _services[svc_key] = {"proc": proc, "port": port, "url": url}

    def _tail_msg():
        joined = " | ".join(list(tail)[-5:]).strip()
        return f". Last output: {joined}" if joined else ""

    deadline = time.time() + ready_timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"ds4-server exited (code {proc.returncode}) before becoming ready"
                + _tail_msg())
        if _health_ok(url):
            print(f"[ds4] service ready for {svc_key} at {url}", flush=True)
            return url
        time.sleep(2)
    stop_service(svc_key)
    raise RuntimeError(f"ds4-server for {svc_key} did not become ready in time"
                       + _tail_msg())


def stop_service(model_id: str) -> None:
    with _lock:
        svc = _services.pop(model_id, None)
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
    print(f"[ds4] service for {model_id} stopped", flush=True)


def stop_all() -> None:
    for mid in list(_services.keys()):
        stop_service(mid)


import atexit as _atexit
_atexit.register(stop_all)
