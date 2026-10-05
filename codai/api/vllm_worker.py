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
import json
import os
import re
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
    configured = (getattr(cfg, "venv", "") or "").strip() \
        or (os.environ.get("CODERAI_VLLM_VENV") or "").strip()
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


def _pump_logs(proc, tail, meta: dict = None):
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            tail.append(line)
            # vLLM states how much of the budget was left for the cache. That figure is
            # the only direct measurement of an engine's real footprint we ever get, so
            # it is captured HERE as it streams — by the time the server is ready it has
            # scrolled out of the 80-line tail behind vLLM's route listing.
            if meta is not None and "kv_gib" not in meta:
                m = _KV_REPORT.search(line)
                if m:
                    try:
                        meta["kv_gib"] = float(m.group(1))
                    except Exception:
                        pass
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
                served_name: Optional[str] = None, model_config: dict = None,
                gpu_memory_utilization: Optional[float] = None,
                max_model_len: Optional[int] = None,
                max_num_batched_tokens: Optional[int] = None,
                max_num_seqs: Optional[int] = None) -> list:
    mid = served_name or (getattr(cfg, "model_id", "vllm") or "vllm")
    cmd = [py, "-m", "vllm.entrypoints.openai.api_server",
           "--host", host, "--port", str(port),
           "--model", model_path,
           "--served-model-name", mid]
    # A side job serving a small model (the OCR VLM engines) overrides the context the
    # LLM backend is configured with: it inherited ctx=18432, and the activation peak
    # that size profiles for was 9.4 GiB of the 11.1 GiB that made surya-ocr-2 unable to
    # start. A page is not an 18k-token conversation.
    ctx = int(max_model_len or 0) or int(getattr(cfg, "ctx", 0) or 0)
    if ctx > 0:
        cmd += ["--max-model-len", str(ctx)]
    # Already absolute (resolve_gmu translated a side job's own-share if there was one).
    gmu = float(gpu_memory_utilization or 0) or float(getattr(cfg, "gpu_memory_utilization", 0) or 0)
    if gmu > 0:
        cmd += ["--gpu-memory-utilization", str(gmu)]
    # The profiling peak that decides whether the engine can start is driven by how many
    # tokens may be in one batch, NOT by max_model_len (chunked prefill is on), so this
    # is the lever that shrinks the footprint without shrinking the usable context.
    mnbt = int(max_num_batched_tokens or 0)
    if mnbt > 0:
        cmd += ["--max-num-batched-tokens", str(mnbt)]
    tp = int(getattr(cfg, "tensor_parallel_size", 0) or 0)
    if tp > 0:
        cmd += ["--tensor-parallel-size", str(tp)]
    pp = int(getattr(cfg, "pipeline_parallel_size", 0) or 0)
    if pp > 1:
        cmd += ["--pipeline-parallel-size", str(pp)]
    executor = (getattr(cfg, "distributed_executor_backend", "") or "").strip()
    if not executor and needs_ray(cfg):
        executor = "ray"
    if executor:
        cmd += ["--distributed-executor-backend", executor]
    mns = int(max_num_seqs or 0) or int(getattr(cfg, "max_num_seqs", 0) or 0)
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


def needs_ray(cfg) -> bool:
    """Multi-node: other machines' GPUs join through a ray cluster."""
    from codai.cluster.multinode import parse_nodes
    if (getattr(cfg, "distributed_executor_backend", "") or "").strip().lower() == "ray":
        return True
    if int(getattr(cfg, "pipeline_parallel_size", 1) or 1) > 1:
        return True
    if (getattr(cfg, "ray_address", "") or "").strip():
        return True
    return bool(parse_nodes(getattr(cfg, "nodes", None)))


def _local_gpu_count(env: dict) -> int:
    vis = (env.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if vis:
        return len([x for x in vis.split(",") if x.strip()])
    try:
        from codai.backends.gpu_probe import total_memory_bytes
        return max(1, len(total_memory_bytes()))
    except Exception:
        return 1


def _start_ray(cfg, py: str, env: dict):
    """Bring the ray cluster up for this launch; returns (RayCluster, address)."""
    from codai.cluster.multinode import RayCluster
    from codai.cluster.rpc import advertise_host
    adv = ""
    try:
        from codai.admin.routes import config_manager
        adv = getattr(getattr(config_manager.config, "cluster", None), "advertise_host", "") or ""
    except Exception:
        pass
    adv = advertise_host(adv)
    tp = max(1, int(getattr(cfg, "tensor_parallel_size", 1) or 1))
    pp = max(1, int(getattr(cfg, "pipeline_parallel_size", 1) or 1))
    cluster = RayCluster(py, adv, port=int(getattr(cfg, "ray_port", 6379) or 6379),
                         address=getattr(cfg, "ray_address", "") or "",
                         nodes=getattr(cfg, "nodes", None),
                         ready_timeout_s=float(getattr(cfg, "nodes_ready_timeout_s", 600) or 600))
    address = cluster.start(gpus_needed=tp * pp, local_gpus=_local_gpu_count(env))
    env.setdefault("VLLM_HOST_IP", adv)
    env["RAY_ADDRESS"] = address
    return cluster, address


def _ensure_service_once(cfg, model_path: Optional[str] = None,
                         served_name: Optional[str] = None,
                         ready_timeout: float = 3600.0,
                         gmu_absolute: float = 0.0,
                         max_model_len: Optional[int] = None,
                         max_num_batched_tokens: Optional[int] = None,
                         max_num_seqs: Optional[int] = None) -> str:
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
        # The share is resolved by the caller (ensure_service, which may be escalating
        # after a too-small budget) and remembered here: what this instance actually
        # reserved is what stopping it frees, and the eviction path has to be told the
        # truth or it over- or under-evicts for the next model.
        gmu = float(gmu_absolute or 0.0)
        cmd = _launch_cmd(py, cfg, host, port, model, served_name,
                          model_config=_model_config_for(resolved or model),
                          gpu_memory_utilization=gmu,
                          max_model_len=max_model_len,
                          max_num_batched_tokens=max_num_batched_tokens,
                          max_num_seqs=max_num_seqs)

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
        ray = None
        if needs_ray(cfg):
            ray, _addr = _start_ray(cfg, py, env)
        print(f"[vllm] launching: {' '.join(cmd)}"
              + (f"  ({' '.join(f'{k}={v}' for k, v in applied.items())})" if applied else ""),
              flush=True)
        tail = collections.deque(maxlen=80)
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1, env=env)
        except Exception:
            if ray is not None:
                ray.stop()
            raise
        meta = {}
        threading.Thread(target=_pump_logs, args=(proc, tail, meta), daemon=True).start()
        _services[svc_key] = {"proc": proc, "port": port, "url": url, "ray": ray,
                              "gmu": gmu, "meta": meta}

    def _tail_msg():
        # The last six lines of a vLLM crash are the wrapper's own traceback
        # ("Engine core initialization failed. See root cause above.") — the
        # root cause IS above, and on a pod this message is all anyone gets.
        # Prefer the lines that name an error; fall back to a longer tail.
        lines = [l.strip() for l in tail if l.strip()]
        keyed = [l for l in lines if any(k in l for k in ("Error", "error:", "not found",
                                                          "No such", "CUDA", "Traceback"))]
        chosen = (keyed[-8:] if keyed else []) + lines[-6:]
        seen, out = set(), []
        for l in chosen:
            if l not in seen:
                seen.add(l); out.append(l[:300])
        joined = " | ".join(out)
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


# A budget too small for the engine's own footprint. vLLM says this whatever the model,
# card or context is, which is why the response is to measure rather than to guess.
_KV_EXHAUSTED = re.compile(
    r"No available memory for the cache blocks"
    r"|Available KV cache memory: *-"
    r"|less than desired GPU memory utilization",
    re.IGNORECASE)

# How much to add per attempt, and the most to try. Each attempt costs a real boot
# (minutes for a VLM), so the steps are coarse.
_GMU_ESCALATION_STEP = 0.12
_GMU_MAX_ATTEMPTS = 4

# vLLM reports what was left for the cache; this is how that figure is read back.
_KV_REPORT = re.compile(r"Available KV cache memory:\s*(-?[\d.]+)\s*GiB", re.IGNORECASE)
# How much KV cache an instance should end up with, how much spare has to be on the table
# before shrinking is worth a move, and the margin kept when proposing a smaller share.
_TARGET_KV_GB = 2.0
_SHRINK_MIN_SLACK_GB = 1.0
_SHRINK_MARGIN = 0.03


def _learned_gmu_path():
    try:
        from codai.platform_paths import legacy_style_config_dir
        return legacy_style_config_dir() / "vllm_gmu_learned.json"
    except Exception:
        return None


def _learned_gmu_key(cfg, model: str, ctx, mnbt) -> str:
    """What a learned share is valid FOR.

    The footprint depends on the model, how much memory the card has, and the limits it
    was profiled under — so a value learned for surya-ocr-2 on a 24 GB card at ctx 18432
    says nothing about olmOCR-2, or about the same model on a 48 GB card. Keying on all
    of it is what keeps this generic instead of a number that happened to fit once."""
    try:
        from codai.models.manager import multi_model_manager
        total = round(float(multi_model_manager._total_vram_gb() or 0.0), 1)
    except Exception:
        total = 0.0
    return f"{model}|{total}|{int(ctx or 0)}|{int(mnbt or 0)}"


def _load_learned(key: str) -> dict:
    """``{"ok": share that booted, "failed": highest share that could not hold the cache}``

    Both halves matter: ``ok`` lets a later start skip rediscovery (and lets a measured
    share come DOWN below the configured bid), while ``failed`` is a floor that stops the
    downward half ever proposing a budget already proven dead — which is what would
    otherwise oscillate."""
    p = _learned_gmu_path()
    if not p:
        return {}
    try:
        with open(p) as f:
            rec = (json.load(f) or {}).get(key)
    except Exception:
        return {}
    if isinstance(rec, (int, float)):          # the flat format this used to write
        return {"ok": float(rec)}
    return rec if isinstance(rec, dict) else {}


def _save_learned(key: str, ok: float = None, failed: float = None) -> None:
    """Record what booted and/or what was too small. Best-effort: a read-only config dir
    costs a few minutes of re-discovery, not a failure."""
    p = _learned_gmu_path()
    if not p:
        return
    try:
        data = {}
        try:
            with open(p) as f:
                data = json.load(f) or {}
        except Exception:
            data = {}
        rec = data.get(key)
        if isinstance(rec, (int, float)):
            rec = {"ok": float(rec)}
        if not isinstance(rec, dict):
            rec = {}
        before = dict(rec)
        if ok is not None and ok > 0:
            rec["ok"] = round(float(ok), 3)
        if failed is not None and failed > 0:
            rec["failed"] = round(max(float(failed), float(rec.get("failed") or 0.0)), 3)
        if rec == before:
            return
        data[key] = rec
        tmp = str(p) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        os.replace(tmp, p)
        print(f"[vllm] learned {rec} for {key}", flush=True)
    except Exception as e:
        print(f"[vllm] could not persist the learned share: {e}", flush=True)


def _starting_gmu(configured: float, rec: dict) -> float:
    """Where to begin: what booted before if we know it, else the configured bid — never
    at or below a share that has already failed."""
    start = float(rec.get("ok") or 0.0) or float(configured or 0.0)
    failed = float(rec.get("failed") or 0.0)
    if failed > 0:
        start = max(start, min(failed + _GMU_ESCALATION_STEP, GMU_CEILING))
    return min(max(start, 0.0), GMU_CEILING)


def _shrink_candidate(gmu: float, kv_gib: float, target_kv_gb: float,
                      rec: dict) -> Optional[float]:
    """A smaller share that would still leave ``target_kv_gb`` of cache, or None.

    Escalation alone only ever ratchets upward, so an opening bid that is too generous
    would have the engine sit on memory it does not use — 0.60 of this card when it needs
    0.25 is 8 GB another model could have had. The successful boot reports the cache it
    got, which gives the footprint directly: ``footprint = budget - kv``. Kept above any
    share known to fail, and with a margin, so shrinking cannot trade a working budget
    for a dead one."""
    try:
        from codai.models.manager import multi_model_manager
        total = float(multi_model_manager._total_vram_gb() or 0.0)
    except Exception:
        return None
    if total <= 0 or kv_gib is None:
        return None
    slack = float(kv_gib) - float(target_kv_gb)
    if slack <= _SHRINK_MIN_SLACK_GB:          # not enough spare cache to be worth a move
        return None
    footprint = (gmu * total) - float(kv_gib)
    if footprint <= 0:
        return None
    cand = ((footprint + float(target_kv_gb)) / total) + _SHRINK_MARGIN
    floor = float(rec.get("failed") or 0.0)
    if floor > 0:
        cand = max(cand, min(floor + _GMU_ESCALATION_STEP, GMU_CEILING))
    cand = min(max(cand, 0.05), GMU_CEILING)
    if cand >= gmu - _SHRINK_MARGIN:           # not meaningfully smaller
        return None
    return round(cand, 3)


def _service_meta(cfg, model_path=None, served_name=None) -> dict:
    _resolved, svc_key = resolve_service_key(cfg, model_path)
    if served_name:
        svc_key = f"{svc_key}|{served_name}"
    with _lock:
        svc = _services.get(svc_key) or {}
    return svc.get("meta") or {}


def ensure_service(cfg, model_path: Optional[str] = None,
                   served_name: Optional[str] = None,
                   ready_timeout: float = 3600.0,
                   gpu_memory_utilization: Optional[float] = None,
                   max_model_len: Optional[int] = None,
                   max_num_batched_tokens: Optional[int] = None,
                   max_num_seqs: Optional[int] = None,
                   target_kv_gb: Optional[float] = None) -> str:
    """Launch (or reuse) a vLLM OpenAI server for a model; return its base URL.

    The configured share is a STARTING POINT, not a verdict. An engine's real footprint
    is not knowable up front — it depends on the model, the card, the context and the
    multimodal limits it profiles for (surya-ocr-2 on a 24 GB 3090 needs ~10.9 GiB before
    a single KV block, almost all of it the vision encoder's profiling peak; another model
    or card is another number). So when vLLM reports the budget cannot hold the cache,
    this escalates the share and tries again, asks the evictor for the larger figure
    first, and REMEMBERS what worked, keyed by model+card+limits. That way any model,
    on any card, serving any kind of request, converges on a share that fits instead of
    depending on a constant somebody measured once.

    ``model_path``/``served_name`` override the config (used by the OCR subsystem to serve
    surya-2 on its own vLLM instance alongside any LLM instance)."""
    configured = resolve_gmu(cfg, gpu_memory_utilization)
    ctx = int(max_model_len or 0) or int(getattr(cfg, "ctx", 0) or 0)
    key = _learned_gmu_key(cfg, (served_name or model_path or "vllm"), ctx,
                           max_num_batched_tokens)
    rec = _load_learned(key)
    target_kv = float(target_kv_gb if target_kv_gb is not None else _TARGET_KV_GB)

    if configured <= 0 and not rec:
        # No share configured anywhere: nothing to size, let vLLM use its own default.
        return _ensure_service_once(
            cfg, model_path, served_name, ready_timeout, 0.0,
            max_model_len, max_num_batched_tokens, max_num_seqs)

    gmu = _starting_gmu(configured, rec)
    if rec and abs(gmu - configured) > 0.005:
        print(f"[vllm] starting at {gmu:.3f} rather than the configured {configured:.3f} "
              f"— measured previously for {key}: {rec}", flush=True)
    last = None
    for attempt in range(1, _GMU_MAX_ATTEMPTS + 1):
        try:
            url = _ensure_service_once(
                cfg, model_path, served_name, ready_timeout, gmu,
                max_model_len, max_num_batched_tokens, max_num_seqs)
        except RuntimeError as exc:
            last = exc
            if not _KV_EXHAUSTED.search(str(exc)):
                raise
            _save_learned(key, failed=gmu)
            nxt = min(gmu + _GMU_ESCALATION_STEP, GMU_CEILING)
            if gmu >= GMU_CEILING - 1e-6 or nxt <= gmu + 1e-6:
                raise
            print(f"[vllm] {served_name or model_path}: a {gmu:.3f} budget cannot hold "
                  f"the KV cache (attempt {attempt}/{_GMU_MAX_ATTEMPTS}); retrying at "
                  f"{nxt:.3f} — the engine's footprint is larger than that share, which "
                  f"is what this measures", flush=True)
            # The bigger budget needs the room to exist before vLLM asks for it.
            try:
                from codai.models.manager import multi_model_manager
                multi_model_manager._evict_models_for_vram(
                    float(prelaunch_free_gb(cfg, nxt)))
            except Exception as e:
                print(f"[vllm] evict-before-retry skipped: {e}", flush=True)
            gmu = nxt
            continue
        # Booted. Record it, and see whether the cache it got says a smaller share would
        # have done — escalation alone never gives memory back.
        _save_learned(key, ok=gmu)
        try:
            kv = (_service_meta(cfg, model_path, served_name) or {}).get("kv_gib")
            cand = _shrink_candidate(gmu, kv, target_kv, _load_learned(key))
            if cand is not None:
                print(f"[vllm] {served_name or model_path}: {kv:.2f} GiB of cache on a "
                      f"{gmu:.3f} share leaves room to spare; {cand:.3f} would still "
                      f"give ~{target_kv:.1f} GiB — using it from the next start",
                      flush=True)
                _save_learned(key, ok=cand)
        except Exception as e:
            print(f"[vllm] could not size down from the reported cache: {e}", flush=True)
        return url
    raise last if last is not None else RuntimeError("vLLM did not start")


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
    ray = svc.get("ray")
    if ray is not None:
        try:
            ray.stop()
        except Exception as exc:
            print(f"[vllm] ray teardown for {svc_key} failed: {exc}", flush=True)
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


# vLLM must not be told to budget the whole card: the driver context, fragmentation and
# any allocation made between our measurement and its own need somewhere to live.
GMU_CEILING = 0.92


def _used_by_others_gb() -> float:
    """VRAM (GB) currently held on this box by anything that is NOT the vLLM we are
    about to launch — resident embedders, an LLM, another process' context."""
    try:
        from codai.models.manager import multi_model_manager
        total = float(multi_model_manager._total_vram_gb() or 0.0)
        free = float(multi_model_manager._get_free_vram_gb() or 0.0)
        if total <= 0 or free >= 999.0:   # 999 = "unknown, assume enough"
            return 0.0
        return max(0.0, total - free)
    except Exception:
        return 0.0


def resolve_gmu(cfg, gpu_memory_utilization: Optional[float] = None) -> float:
    """The absolute ``--gpu-memory-utilization`` to launch with.

    A side job (an OCR VLM engine) passes the share it wants for ITSELF; the backend's
    own configured value is used as-is. Both are fractions of the card's TOTAL memory,
    which is what the flag means — see :func:`effective_gmu` for why nothing is added
    back for other residents any more."""
    share = float(gpu_memory_utilization or 0)
    if share > 0:
        return effective_gmu(share)
    return float(getattr(cfg, "gpu_memory_utilization", 0) or 0)


def effective_gmu(self_share: float) -> float:
    """The share this instance may use, clamped to :data:`GMU_CEILING`, plus a log line
    naming what else is on the card.

    Measured on a 24 GB 3090 (2026-10-05, two live boots of surya-ocr-2): budget
    9.57 GiB gave ``Available KV cache memory: -1.54 GiB``, budget 14.31 GiB gave
    ``+3.18 GiB`` — a fixed footprint of 11.11 and 11.13 GiB respectively. Two things
    follow. First, vLLM charges only its OWN allocations against ``total × gmu``; if
    other residents counted too, those two boots would imply wildly different activation
    peaks (8.3 vs 3.6 GiB) instead of the same 11.1. So this function does NOT add back
    what others hold: doing so made the budget depend on unrelated residents and asked
    for ``share×total + used`` for this instance while ``used`` stayed held by someone
    else. The card is cleared by the caller's evict-first pass
    (:func:`prelaunch_free_gb`) instead, which is the honest mechanism.

    Second, the share has to actually cover the footprint: 0.35 of this card is 8.4 GiB
    against an 11.1 GiB floor, so it could never start, which is why ``0.35`` crash-looped
    for hours and then appeared to "work" only when 6.1 GB of other residents inflated
    the old add-back to 0.604. Hence the 0.60 default and the batched-token cap that
    brings the 9.4 GiB activation peak down."""
    try:
        share = max(0.0, float(self_share or 0.0))
    except Exception:
        return 0.0
    if share <= 0:
        return 0.0
    eff = min(share, GMU_CEILING)
    try:
        from codai.models.manager import multi_model_manager
        total = float(multi_model_manager._total_vram_gb() or 0.0)
    except Exception:
        total = 0.0
    if total > 0:
        used = _used_by_others_gb()
        print(f"[vllm] gpu-memory-utilization {eff:.3f} = {eff * total:.1f} GB of a "
              f"{total:.1f} GB card for this instance; {used:.1f} GB currently held by "
              f"others (the evict-first pass should have cleared what it needed)",
              flush=True)
    return eff


def prelaunch_free_gb(cfg, gpu_memory_utilization: Optional[float] = None) -> float:
    """How much VRAM must be FREE before launching this vLLM for the side-job share to
    fit under :data:`GMU_CEILING` — i.e. what to ask the evictor for.

    It is the engine's own need plus the headroom the ceiling reserves, so a crowded
    card gets models evicted instead of :func:`effective_gmu` clamping and vLLM dying
    on KV cache again."""
    gmu = float(gpu_memory_utilization or 0) or float(getattr(cfg, "gpu_memory_utilization", 0) or 0)
    if gmu <= 0:
        return planned_vram_gb(cfg, gpu_memory_utilization)
    try:
        from codai.models.manager import multi_model_manager
        total = float(multi_model_manager._total_vram_gb() or 24.0)
    except Exception:
        total = 24.0
    total = total or 24.0
    return max(0.5, (min(gmu, GMU_CEILING) + (1.0 - GMU_CEILING)) * total)


def planned_vram_gb(cfg, gpu_memory_utilization: Optional[float] = None) -> float:
    """How much VRAM a vLLM instance started from ``cfg`` will claim on this box.

    vLLM allocates ``gpu_memory_utilization`` × the card's TOTAL memory up front and
    refuses to start when that much is not free — so this is also what must be freed
    BEFORE launching it. Callers that boot vLLM for a side job (the OCR VLM engines) use
    it to ask the model manager for room first; without that, Surya-2 crash-looped on a
    3090 with 11 GB free against a 14 GB demand."""
    gmu = float(gpu_memory_utilization or 0) or float(getattr(cfg, "gpu_memory_utilization", 0) or 0)
    if gmu <= 0:
        return 4.6   # small VLM fallback estimate
    try:
        from codai.models.manager import multi_model_manager
        total = multi_model_manager._total_vram_gb()
    except Exception:
        total = 24.0
    return max(0.5, gmu * float(total or 24.0))


def stop_service_for(cfg, model_path: Optional[str] = None,
                     served_name: Optional[str] = None,
                     gpu_memory_utilization: Optional[float] = None) -> float:
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
        launched_gmu = float((svc or {}).get("gmu") or 0.0)
    if not running:
        return 0.0
    stop_service(svc_key)
    # What it reserved is what we just freed. The share the caller passes is the engine's
    # OWN share; the instance was launched with that plus whatever else held the card
    # (effective_gmu), so reporting the caller's figure would under-count the release and
    # send the eviction path looking for memory that is already free.
    if launched_gmu > 0:
        try:
            from codai.models.manager import multi_model_manager
            total = float(multi_model_manager._total_vram_gb() or 24.0)
        except Exception:
            total = 24.0
        return max(0.5, launched_gmu * (total or 24.0))
    return planned_vram_gb(cfg, gpu_memory_utilization)


import atexit as _atexit
_atexit.register(stop_all)
