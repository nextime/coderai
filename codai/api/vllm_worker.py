# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Fully-managed vLLM worker — the vLLM OpenAI server, driven as a subprocess.

vLLM (https://github.com/vllm-project/vllm) exposes an OpenAI-compatible HTTP server
(``python -m vllm.entrypoints.openai.api_server``) with continuous batching + paged KV,
so — like :mod:`codai.api.kt_worker` — coderai launches it as a managed subprocess,
health-checks ``/v1/models``, and :mod:`codai.backends.vllm` proxies to it.

vLLM pins its own torch/CUDA (e.g. torch 2.13 / cu13), which conflicts with the main
coderai venv, so it runs in an ISOLATED venv (``vllm.venv``) — same pattern as the OCR
Paddle/Surya engines. This module owns the venv + process lifecycle only. Also reused by
the OCR subsystem to serve Surya2 (a VLM) for :mod:`codai.ocr.surya`.
"""

import collections
import os
import shlex
import socket
import subprocess
import threading
import time
from typing import Optional

_lock = threading.RLock()
_services: dict[str, dict] = {}   # svc_key -> {"proc","port","url"}

# Repo-root requirements for the isolated vLLM venv (auto-build).
_REQ = os.path.join(os.path.dirname(__file__), "..", "..", "requirements-vllm.txt")


def resolve_venv_dir(cfg) -> str:
    """Isolated vLLM venv: config > baked /opt/coderai/vllm_venv > /cache mount > ~/.coderai."""
    configured = (getattr(cfg, "venv", "") or "").strip()
    if configured:
        return configured
    baked = "/opt/coderai/vllm_venv"
    if os.path.isdir(baked):
        return baked
    cache = os.environ.get("CODERAI_CACHE_DIR") or ("/cache" if os.path.isdir("/cache") else "")
    if cache and os.path.isdir(cache):
        return os.path.join(cache, "vllm_venv")
    return os.path.expanduser("~/.coderai/vllm_venv")


def _venv_python(cfg) -> str:
    return os.path.join(os.path.expanduser(resolve_venv_dir(cfg)), "bin", "python")


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
            print(f"[vllm] {line}", flush=True)


def _health_ok(url: str) -> bool:
    import requests
    try:
        r = requests.get(url + "/v1/models", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def ensure_built(cfg) -> str:
    """Ensure the isolated vLLM venv exists; return its python. Builds it if auto_build."""
    py = _venv_python(cfg)
    if os.path.isfile(py):
        return py
    if not getattr(cfg, "auto_build", False):
        raise RuntimeError(
            f"vLLM isolated venv not found at {os.path.dirname(os.path.dirname(py))}. "
            f"Build it (python3 -m venv <dir> && <dir>/bin/pip install -r "
            f"requirements-vllm.txt) or enable vllm.auto_build.")
    venv_dir = os.path.expanduser(resolve_venv_dir(cfg))
    req = os.path.abspath(_REQ)
    print(f"[vllm] creating isolated venv at {venv_dir} …", flush=True)
    import sys as _sys
    try:
        os.makedirs(os.path.dirname(venv_dir) or ".", exist_ok=True)
        subprocess.run([_sys.executable, "-m", "venv", venv_dir], check=True)
        subprocess.run([py, "-m", "pip", "install", "-U", "pip"], check=True)
        if os.path.isfile(req):
            subprocess.run([py, "-m", "pip", "install", "-r", req], check=True)
        else:
            subprocess.run([py, "-m", "pip", "install", "vllm"], check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"vLLM: venv build failed: {exc}")
    if not os.path.isfile(py):
        raise RuntimeError("vLLM: venv python missing after build")
    return py


def resolve_service_key(cfg, model_path: Optional[str] = None):
    mp = os.path.expanduser((model_path or getattr(cfg, "model_path", "") or "").strip())
    key = mp or (getattr(cfg, "model_id", "vllm") or "vllm")
    return mp, key


def _launch_cmd(py, cfg, host: str, port: int, model_path: str,
                served_name: Optional[str] = None, model_config: dict = None) -> list:
    mid = served_name or (getattr(cfg, "model_id", "vllm") or "vllm")
    cmd = [py, "-m", "vllm.entrypoints.openai.api_server",
           "--host", host, "--port", str(port),
           "--model", model_path,
           "--served-model-name", mid]
    ctx = int(getattr(cfg, "ctx", 0) or 0)
    if ctx > 0:
        cmd += ["--max-model-len", str(ctx)]
    gmu = float(getattr(cfg, "gpu_memory_utilization", 0) or 0)
    if gmu > 0:
        cmd += ["--gpu-memory-utilization", str(gmu)]
    tp = int(getattr(cfg, "tensor_parallel_size", 0) or 0)
    if tp > 0:
        cmd += ["--tensor-parallel-size", str(tp)]
    mns = int(getattr(cfg, "max_num_seqs", 0) or 0)
    if mns > 0:
        cmd += ["--max-num-seqs", str(mns)]
    dtype = (getattr(cfg, "dtype", "") or "").strip()
    if dtype:
        cmd += ["--dtype", dtype]
    quant = (getattr(cfg, "quantization", "") or "").strip()
    if quant:
        cmd += ["--quantization", quant]
    # LoRA adapters configured on the MODEL (not on the vLLM backend): vLLM takes
    # them at launch as name=source pairs, where source is a path on this host or
    # a HuggingFace repo id. A QLoRA adapter is an ordinary LoRA here — the
    # quantisation is how the base model loads, which --quantization covers.
    cmd += _lora_args(model_config)

    extra = (getattr(cfg, "extra_args", "") or "").strip()
    if extra:
        cmd += shlex.split(extra)
    return cmd


def _lora_args(model_config: dict) -> list:
    """`--enable-lora --lora-modules …` for the adapters a model config asks for."""
    try:
        from codai.models.text_loras import configured_specs, describe
        specs = configured_specs(model_config or {})
    except Exception:
        return []
    if not specs:
        return []
    args = ["--enable-lora"]
    mods = []
    for spec in specs:
        mods.append(f"{spec['name']}={spec['source']}")
    args += ["--lora-modules"] + mods
    # vLLM rejects an adapter whose rank exceeds --max-lora-rank (default 16),
    # and the message points at the flag rather than the adapter, so raise the
    # ceiling to what vLLM supports rather than have a trained LoRA refused.
    args += ["--max-lora-rank", "64"]
    print(f"[lora] vLLM will serve adapters: {describe(specs)}", flush=True)
    return args


def _model_config_for(model_name: str) -> dict:
    """The model's models.json entry — vLLM serves models from the model list, so
    its LoRA settings live on the model, not on the vLLM backend config."""
    try:
        from codai.models.manager import _model_entry_for
        return _model_entry_for(model_name) or {}
    except Exception:
        return {}


def _remote_serves(url: str, name: str) -> bool:
    """True when a remote vLLM's /v1/models lists ``name`` (or a suffix match)."""
    import requests
    try:
        r = requests.get(url + "/v1/models", timeout=5)
        ids = [str(m.get("id", "")) for m in (r.json().get("data") or [])]
    except Exception:
        return False
    tail = name.rstrip("/").split("/")[-1]
    return any(i == name or i.rstrip("/").split("/")[-1] == tail for i in ids)


def ensure_service(cfg, model_path: Optional[str] = None,
                   served_name: Optional[str] = None,
                   ready_timeout: float = 3600.0) -> str:
    """Launch (or reuse) a vLLM OpenAI server for a model; return its base URL.

    ``model_path``/``served_name`` override the config (used by the OCR subsystem to serve
    surya-2 on its own vLLM instance alongside any LLM instance)."""
    # An explicit `service_url` points at a vLLM running somewhere else — another
    # host, a container, a rented RunPod pod (where vLLM's own image is the ready-
    # made option) — so nothing is spawned or downloaded here. This function
    # already returns a URL, so its callers cannot tell the difference.
    _remote = (str(getattr(cfg, "service_url", "") or "")
               or os.environ.get("CODERAI_VLLM_SERVICE_URL") or "").strip()
    if _remote:
        _remote = _remote.rstrip("/")
        if not _health_ok(_remote):
            raise RuntimeError(
                f"configured service_url {_remote} is not answering its health check")
        # A remote vLLM serves whatever it was started with. When the caller asks
        # for a specific model (the OCR path does, for surya-2), make sure that
        # model is actually there rather than silently proxying to the wrong one.
        _want = (served_name or model_path or "").strip()
        if _want and not _remote_serves(_remote, _want):
            raise RuntimeError(
                f"configured service_url {_remote} does not serve {_want}")
        print(f"[vllm] using the configured remote service at {_remote}", flush=True)
        return _remote

    resolved, svc_key = resolve_service_key(cfg, model_path)
    if served_name:
        svc_key = f"{svc_key}|{served_name}"
    with _lock:
        svc = _services.get(svc_key)
        if svc and svc["proc"].poll() is None and _health_ok(svc["url"]):
            return svc["url"]
        if svc:
            _services.pop(svc_key, None)

        py = ensure_built(cfg)
        model = resolved or (getattr(cfg, "model_path", "") or "").strip()
        if not model:
            raise RuntimeError(
                "vLLM: no model resolved. Set vllm.model_path (an HF model dir or id).")

        host = (getattr(cfg, "host", "127.0.0.1") or "127.0.0.1").strip()
        port = int(getattr(cfg, "port", 0) or 0) or _free_port()
        url_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
        url = f"http://{url_host}:{port}"
        cmd = _launch_cmd(py, cfg, host, port, model, served_name,
                          model_config=_model_config_for(resolved or model))

        env = os.environ.copy()
        # flashinfer JIT-compiles CUDA kernels with ninja at runtime, which fails on
        # hosts without a full build toolchain (nvcc/gcc wired for it). Default it OFF so
        # vLLM uses FLASH_ATTN + a native sampler out of the box; the user can re-enable
        # via extra_env. (Set before extra_env so an explicit override wins.)
        env.setdefault("VLLM_USE_FLASHINFER", "0")
        env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        # Pin this vLLM instance to specific NVIDIA GPU(s) if configured (CUDA-only).
        gpu = (getattr(cfg, "gpu", "") or "").strip()
        if gpu:
            env["CUDA_VISIBLE_DEVICES"] = gpu
        extra_env = (getattr(cfg, "extra_env", "") or "").strip()
        applied = {}
        if extra_env:
            for tok in shlex.split(extra_env):
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    if k.strip():
                        env[k.strip()] = v; applied[k.strip()] = v
        print(f"[vllm] launching: {' '.join(cmd)}"
              + (f"  ({' '.join(f'{k}={v}' for k, v in applied.items())})" if applied else ""),
              flush=True)
        tail = collections.deque(maxlen=80)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, env=env)
        threading.Thread(target=_pump_logs, args=(proc, tail), daemon=True).start()
        _services[svc_key] = {"proc": proc, "port": port, "url": url}

    def _tail_msg():
        joined = " | ".join(list(tail)[-6:]).strip()
        return f". Last output: {joined}" if joined else ""

    deadline = time.time() + ready_timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            stop_service(svc_key)
            raise RuntimeError(
                f"vLLM exited (code {proc.returncode}) before becoming ready" + _tail_msg())
        if _health_ok(url):
            print(f"[vllm] service ready for {svc_key} at {url}", flush=True)
            return url
        time.sleep(2)
    stop_service(svc_key)
    raise RuntimeError(f"vLLM for {svc_key} did not become ready in time" + _tail_msg())


def stop_service(svc_key: str) -> None:
    with _lock:
        svc = _services.pop(svc_key, None)
    if not svc:
        return
    proc = svc["proc"]
    if proc.poll() is None:
        try:
            proc.terminate(); proc.wait(timeout=10)
        except Exception:
            pass
    if proc.poll() is None:
        try:
            proc.kill()
        except Exception:
            pass
    print(f"[vllm] service for {svc_key} stopped", flush=True)


def stop_all() -> None:
    for k in list(_services.keys()):
        stop_service(k)


def is_running(cfg, model_path: Optional[str] = None, served_name: Optional[str] = None) -> bool:
    """True if a vLLM service for this (model_path, served_name) is currently up."""
    _resolved, svc_key = resolve_service_key(cfg, model_path)
    if served_name:
        svc_key = f"{svc_key}|{served_name}"
    with _lock:
        svc = _services.get(svc_key)
        return bool(svc and svc["proc"].poll() is None)


def stop_service_for(cfg, model_path: Optional[str] = None,
                     served_name: Optional[str] = None) -> float:
    """Stop the vLLM service for ONE specific (model_path, served_name), if running.

    Used by the OCR subsystem as a VRAM releaser: the managed Surya-2 vLLM subprocess
    isn't a manager-tracked model, so this is what lets on-request eviction reclaim its
    VRAM. Returns an estimate of the GB it was holding (gpu_memory_utilization × total
    VRAM), 0.0 if nothing was running."""
    _resolved, svc_key = resolve_service_key(cfg, model_path)
    if served_name:
        svc_key = f"{svc_key}|{served_name}"
    with _lock:
        svc = _services.get(svc_key)
        running = bool(svc and svc["proc"].poll() is None)
    if not running:
        return 0.0
    gmu = float(getattr(cfg, "gpu_memory_utilization", 0) or 0)
    stop_service(svc_key)
    if gmu > 0:
        try:
            from codai.models.manager import multi_model_manager
            total = multi_model_manager._total_vram_gb()
        except Exception:
            total = 24.0
        return max(0.5, gmu * total)
    return 4.6   # small VLM fallback estimate


import atexit as _atexit
_atexit.register(stop_all)
