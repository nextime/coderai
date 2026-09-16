# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Fully-managed Kimi-K3 worker — the kimi-k3-in-c C engine, driven directly.

kimi-k3-in-c (https://github.com/FareedKhan-dev/kimi-k3-in-c) is a portable-C CPU
engine for Kimi-K3 that streams the dense trunk + routed experts from disk. Upstream it
is a one-shot batch CLI; coderai applies ``packaging/patch-k3.py`` to add a resident
serve loop that speaks the SAME mux stdin/stdout protocol as colibri, so we reuse
:class:`~codai.api.colibri_worker.MuxEngine` as the protocol client. This module owns the
build + process lifecycle; the Kimi-K3 chat template lives in :mod:`codai.backends.k3`.

Lifecycle (mirrors the other managed engines):

* :func:`ensure_built` clones the repo, applies the serve-loop patch, and runs ``make``
  so the ``bin/k3`` binary exists (idempotent).
* :func:`ensure_engine` launches the engine on the configured checkpoint in serve mode
  (``SERVE=1`` + the model dir / ``--trunk`` / ``--tok`` / budgets as argv), completes
  the ``READY`` handshake, and returns a live :class:`MuxEngine`.
"""

import collections
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

from codai.api.colibri_worker import MuxEngine, _run_logged

_lock = threading.RLock()
_services: dict[str, MuxEngine] = {}


def default_install_dir() -> Path:
    return Path(os.environ.get("CODERAI_K3_DIR")
               or os.path.expanduser("~/.coderai/kimi-k3-in-c"))


def _install_dir(cfg) -> Path:
    return Path(cfg.install_dir).expanduser() if getattr(cfg, "install_dir", None) \
        else default_install_dir()


def _engine_bin(install_dir: Path) -> Path:
    """The k3 engine binary (``bin/k3`` in the repo root)."""
    for name in ("k3", "k3.exe"):
        cand = install_dir / "bin" / name
        if cand.exists():
            return cand
    return install_dir / "bin" / "k3"


def _patch_script() -> Optional[str]:
    """Path to coderai's serve-loop patch (packaging/patch-k3.py), or None."""
    here = Path(__file__).resolve()
    for base in (here.parents[2], here.parents[1]):   # repo root, then codai/
        cand = base / "packaging" / "patch-k3.py"
        if cand.exists():
            return str(cand)
    return None


def ensure_built(cfg) -> Path:
    """Clone + patch + build kimi-k3-in-c if the ``bin/k3`` binary is missing."""
    install_dir = _install_dir(cfg)
    binary = _engine_bin(install_dir)
    if binary.exists():
        return binary
    if not getattr(cfg, "auto_build", True):
        raise RuntimeError(
            f"k3 engine not found at {binary} and auto_build is disabled. Build it "
            f"manually (git clone {cfg.repo_url}; python3 packaging/patch-k3.py "
            f"src/cli/k3_run.c; make) or enable auto_build.")
    tail = collections.deque(maxlen=40)
    install_dir.parent.mkdir(parents=True, exist_ok=True)
    if not (install_dir / ".git").exists() and not (install_dir / "Makefile").exists():
        print(f"[k3] cloning {cfg.repo_url} → {install_dir} …", flush=True)
        _run_logged(["git", "clone", "--depth", "1", cfg.repo_url, str(install_dir)],
                    cwd=install_dir.parent, label="git clone", tail=tail)
    # Apply the resident serve-loop patch (idempotent) so the engine can stay warm.
    patch = _patch_script()
    k3_run = install_dir / "src" / "cli" / "k3_run.c"
    if patch and k3_run.exists():
        try:
            _run_logged(["python3", patch, str(k3_run)],
                        cwd=install_dir, label="patch-k3", tail=tail)
        except Exception as exc:
            print(f"[k3] warning: serve-loop patch failed ({exc}); building unpatched "
                  "(serve mode will not work)", flush=True)
    print("[k3] building engine (make) — CPU/AVX2, this can take a moment …", flush=True)
    _run_logged(["make", "-j"], cwd=install_dir, label="make", tail=tail)
    binary = _engine_bin(install_dir)
    if not binary.exists():
        raise RuntimeError("k3 build completed but bin/k3 is missing. Last output: "
                           + " | ".join(l.strip()[:300] for l in list(tail) if l.strip()))
    print(f"[k3] built {binary}", flush=True)
    return binary


def _resolve_checkpoint(cfg, model_dir: Optional[str]) -> str:
    """The Kimi-K3 checkpoint directory (config.json + tokenizer + shards)."""
    for cand in (model_dir, getattr(cfg, "model_path", "") or ""):
        cand = os.path.expanduser((cand or "").strip())
        if cand and os.path.isdir(cand):
            return os.path.abspath(cand)
    return ""


def resolve_service_key(cfg, model_dir: Optional[str] = None):
    ckpt = _resolve_checkpoint(cfg, model_dir)
    return ckpt, (ckpt or (getattr(cfg, "model_id", "kimi-k3") or "kimi-k3"))


def _build_argv(cfg, binary: Path, ckpt: str) -> list:
    """The k3 serve command line: model dir + trunk/tok/budget flags."""
    argv = [str(binary), ckpt]
    trunk = os.path.expanduser((getattr(cfg, "trunk_dir", "") or "").strip())
    if trunk:
        argv += ["--trunk", trunk]
    tok = os.path.expanduser((getattr(cfg, "tok_dir", "") or "").strip()) or ckpt
    argv += ["--tok", tok]
    preset = (getattr(cfg, "preset", "") or "").strip()
    if preset:
        argv += ["--preset", preset]
    else:
        argv += ["--trunk-gb", str(getattr(cfg, "trunk_gb", 16.0) or 16.0),
                 "--cache-gb", str(getattr(cfg, "cache_gb", 64.0) or 64.0)]
    extra = (getattr(cfg, "extra_args", "") or "").strip()
    if extra:
        import shlex
        argv += shlex.split(extra)
    return argv


def _build_env(cfg, ctx: int) -> dict:
    env = os.environ.copy()
    env["SERVE"] = "1"
    env["K3_MAXT"] = str(int(ctx) if ctx else int(getattr(cfg, "ctx", 4096) or 4096))
    extra_env = (getattr(cfg, "extra_env", "") or "").strip()
    if extra_env:
        import shlex
        for tok in shlex.split(extra_env):
            if "=" in tok:
                k, v = tok.split("=", 1)
                if k.strip():
                    env[k.strip()] = v
    return env


def ensure_engine(cfg, model_dir: Optional[str] = None, ctx: Optional[int] = None,
                  ready_timeout: float = 3600.0) -> MuxEngine:
    """Build (as needed), then start (or reuse) the k3 engine for a checkpoint."""
    # An explicit `service_url` points at a k3 engine hosted by
    # tools/mux_service.py somewhere else — another host, a rented pod — so
    # nothing is built or launched here. The client speaks the same surface the
    # backends use, so callers cannot tell it from a local MuxEngine.
    _remote = (str(getattr(cfg, "service_url", "") or "")
               or os.environ.get("CODERAI_K3_SERVICE_URL") or "").strip()
    if _remote:
        from codai.api.mux_remote import remote_engine_for
        eng = remote_engine_for("k3", _remote, family="kimi_k3")
        print(f"[k3] using the configured remote engine at {eng.url}", flush=True)
        return eng

    ckpt, svc_key = resolve_service_key(cfg, model_dir)
    with _lock:
        eng = _services.get(svc_key)
        if eng and eng.is_alive():
            return eng
        if eng and not eng.is_alive():
            eng.close()
            _services.pop(svc_key, None)

        binary = ensure_built(cfg)
        if not ckpt:
            raise RuntimeError(
                "k3: no Kimi-K3 checkpoint resolved for this request. Point the model at "
                "the checkpoint directory (or set k3.model_path). There is no auto-"
                "download of the ~1.56 TB checkpoint.")
        try:
            maxt = int(ctx) if ctx else 0
        except (TypeError, ValueError):
            maxt = 0
        if maxt <= 0:
            maxt = int(getattr(cfg, "ctx", 4096) or 4096)

        argv = _build_argv(cfg, binary, ckpt)
        env = _build_env(cfg, maxt)
        print(f"[k3] launching engine {' '.join(argv)} (K3_MAXT={maxt})", flush=True)
        eng = MuxEngine(binary, ckpt, max_tokens=maxt, kv_slots=1, env=env,
                        force_mux=True, family="kimi_k3", argv=argv,
                        inject_colibri_env=False)
        _services[svc_key] = eng

    deadline = time.time() + ready_timeout
    while time.time() < deadline:
        if not eng.is_alive():
            tail = eng.log_tail()
            stop_service(svc_key)
            raise RuntimeError("k3 engine exited before serving"
                               + (f". Last output: {tail}" if tail else ""))
        print(f"[k3] engine ready for {svc_key}", flush=True)
        return eng
    stop_service(svc_key)
    raise RuntimeError(f"k3 engine for {svc_key} did not become ready in time")


def stop_service(model_id: str) -> None:
    with _lock:
        eng = _services.pop(model_id, None)
    if not eng:
        return
    try:
        eng.close()
    except Exception:
        pass
    print(f"[k3] engine for {model_id} stopped", flush=True)


def stop_all() -> None:
    for mid in list(_services.keys()):
        stop_service(mid)


import atexit as _atexit
_atexit.register(stop_all)
