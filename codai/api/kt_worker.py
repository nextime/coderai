# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Fully-managed ktransformers worker — the SGLang OpenAI server, driven as a subprocess.

ktransformers (https://github.com/kvcache-ai/ktransformers) serves through SGLang's
OpenAI-compatible HTTP server, so — like :mod:`codai.api.ds4_worker` — coderai launches
``python -m sglang.launch_server`` as a managed subprocess, health-checks its
``/v1/models`` endpoint, and the backend (:mod:`codai.backends.ktransformers`) proxies
chat/completions to it. This module owns the process lifecycle only.

Heavy dependencies (SGLang + kt-kernel) are expected to be installed out of band; with
``auto_build`` a best-effort ``pip install`` is attempted, else a clear error is raised.
"""

import collections
import os
import shlex
import socket
import subprocess
import sys
import threading
import time
from typing import Optional

_lock = threading.RLock()
_services: dict[str, dict] = {}   # svc_key -> {"proc","port","url"}


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
            print(f"[kt] {line}", flush=True)


def _health_ok(url: str) -> bool:
    import requests
    try:
        r = requests.get(url + "/v1/models", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _venv_python(cfg=None) -> str:
    """The interpreter that has SGLang + kt-kernel.

    They pin their own torch (2.9.1 for kt-kernel 0.7) and cannot share the
    main venv, so a pod image bakes them into a venv of their own and says
    where with CODERAI_KT_VENV (a local install may set ktransformers.venv).
    Blank means the main interpreter, which is where a hand install lands.
    """
    venv = (str(getattr(cfg, "venv", "") or "") if cfg is not None else "") \
        or os.environ.get("CODERAI_KT_VENV", "")
    venv = os.path.expanduser(venv.strip())
    if venv:
        cand = os.path.join(venv, "bin", "python")
        if os.path.isfile(cand):
            return cand
        print(f"[kt] CODERAI_KT_VENV={venv} has no bin/python — using {sys.executable}",
              flush=True)
    return sys.executable


def _sglang_available(cfg=None) -> bool:
    py = _venv_python(cfg)
    if py == sys.executable:
        import importlib.util
        return importlib.util.find_spec("sglang") is not None
    try:
        return subprocess.run([py, "-c", "import sglang"], capture_output=True,
                              timeout=120).returncode == 0
    except Exception:
        return False


def ensure_built(cfg) -> None:
    """Ensure SGLang (the ktransformers serving frontend) is importable.

    ktransformers' kt-kernel is a native build; coderai does not compile it per-request.
    With ``auto_build`` we attempt a best-effort ``pip install sglang``, but the kt-kernel
    wheel/build must be provided out of band — otherwise raise a clear, actionable error.
    """
    if _sglang_available(cfg):
        return
    if not getattr(cfg, "auto_build", False):
        raise RuntimeError(
            "ktransformers needs SGLang (python -m sglang.launch_server) which is not "
            "installed, and auto_build is disabled. Install SGLang + kt-kernel out of "
            "band (see the ktransformers docs), or enable ktransformers.auto_build.")
    print("[kt] SGLang not found; attempting pip install (heavy) …", flush=True)
    tail = collections.deque(maxlen=40)
    try:
        proc = subprocess.Popen([sys.executable, "-m", "pip", "install", "sglang"],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                bufsize=1)
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                tail.append(line)
                print(f"[kt] {line}", flush=True)
        proc.wait()
    except Exception as exc:
        raise RuntimeError(f"ktransformers: pip install sglang failed: {exc}")
    if not _sglang_available():
        raise RuntimeError(
            "ktransformers: SGLang still not importable after pip install. Install "
            "SGLang + kt-kernel manually per the ktransformers docs. Last output: "
            + " | ".join(l.strip()[:300] for l in list(tail) if l.strip()))


def resolve_service_key(cfg, model_path: Optional[str] = None):
    mp = os.path.expanduser((model_path or getattr(cfg, "model_path", "") or "").strip())
    key = mp or (getattr(cfg, "model_id", "ktransformers") or "ktransformers")
    return mp, key


def _launch_cmd(cfg, host: str, port: int, model_path: str) -> list:
    mid = (getattr(cfg, "model_id", "ktransformers") or "ktransformers")
    cmd = [_venv_python(cfg), "-m", "sglang.launch_server",
           "--host", host, "--port", str(port),
           "--model", model_path,
           "--served-model-name", mid]
    ktw = os.path.expanduser((getattr(cfg, "kt_weight_path", "") or "").strip())
    if ktw:
        cmd += ["--kt-weight-path", ktw]
    ctx = int(getattr(cfg, "ctx", 0) or 0)
    if ctx > 0:
        cmd += ["--context-length", str(ctx)]
    extra = (getattr(cfg, "extra_args", "") or "").strip()
    if extra:
        cmd += shlex.split(extra)
    return cmd


def ensure_service(cfg, model_path: Optional[str] = None,
                   ready_timeout: float = 3600.0) -> str:
    """Launch (or reuse) the SGLang server for a ktransformers model; return its base URL."""
    # An explicit `service_url` points at a SGLang server running somewhere else — another
    # host, a container, a rented pod — so nothing is built, spawned or downloaded
    # here and coderai just proxies to it. This function already returns a URL, so
    # its callers cannot tell the difference.
    _remote = (str(getattr(cfg, "service_url", "") or "") or os.environ.get("CODERAI_KT_SERVICE_URL") or "").strip()
    if _remote:
        _remote = str(_remote).rstrip("/")
        if not _health_ok(_remote):
            raise RuntimeError(
                f"configured service_url {_remote} is not answering its health check")
        print(f"[kt] using the configured remote service at {_remote}", flush=True)
        return _remote

    resolved, svc_key = resolve_service_key(cfg, model_path)
    with _lock:
        svc = _services.get(svc_key)
        if svc and svc["proc"].poll() is None and _health_ok(svc["url"]):
            return svc["url"]
        if svc:
            _services.pop(svc_key, None)   # died — restart below

        ensure_built(cfg)
        # A directory, or a HuggingFace repo id — SGLang resolves the latter
        # itself and downloads into HF_HOME, which on a pod is the network
        # volume. A bare name that is neither is the error it always was.
        looks_like_repo = bool(resolved) and resolved.count("/") == 1 \
            and not resolved.startswith(("/", "~", "."))
        if not resolved or not (os.path.isdir(resolved) or looks_like_repo):
            raise RuntimeError(
                "ktransformers: no model resolved for this request. Point the model "
                "at the HF model dir or repo id (or set ktransformers.model_path); "
                "also set ktransformers.kt_weight_path for pre-quantized experts.")

        host = (getattr(cfg, "host", "127.0.0.1") or "127.0.0.1").strip()
        port = int(getattr(cfg, "port", 0) or 0) or _free_port()
        # Health/proxy over loopback even when SGLang is bound to 0.0.0.0.
        url_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
        url = f"http://{url_host}:{port}"
        cmd = _launch_cmd(cfg, host, port, resolved)

        env = os.environ.copy()
        applied_env = {}
        extra_env = (getattr(cfg, "extra_env", "") or "").strip()
        if extra_env:
            for tok in shlex.split(extra_env):
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    if k.strip():
                        env[k.strip()] = v
                        applied_env[k.strip()] = v
        env_note = ("  (" + " ".join(f"{k}={v}" for k, v in applied_env.items()) + ")"
                    if applied_env else "")
        print(f"[kt] launching SGLang: {' '.join(cmd)}{env_note}", flush=True)
        tail = collections.deque(maxlen=60)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, env=env)
        threading.Thread(target=_pump_logs, args=(proc, tail), daemon=True).start()
        _services[svc_key] = {"proc": proc, "port": port, "url": url}

    def _tail_msg():
        joined = " | ".join(l.strip()[:300] for l in list(tail) if l.strip()).strip()
        return f". Last output: {joined}" if joined else ""

    deadline = time.time() + ready_timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            stop_service(svc_key)
            raise RuntimeError(
                f"SGLang exited (code {proc.returncode}) before becoming ready" + _tail_msg())
        if _health_ok(url):
            print(f"[kt] service ready for {svc_key} at {url}", flush=True)
            return url
        time.sleep(2)
    stop_service(svc_key)
    raise RuntimeError(f"SGLang for {svc_key} did not become ready in time" + _tail_msg())


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
    print(f"[kt] service for {model_id} stopped", flush=True)


def stop_all() -> None:
    for mid in list(_services.keys()):
        stop_service(mid)


import atexit as _atexit
_atexit.register(stop_all)
