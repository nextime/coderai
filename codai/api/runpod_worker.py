# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""RunPod worker — per-model config parsing + endpoint resolution.

This module owns the RunPod-specific model config (parsed from the model's
models.json ``runpod`` block) and turns it into a live OpenAI base URL that
:mod:`codai.backends.runpod` proxies to.

Two modes (per-model ``mode``):
  * ``serverless`` — RunPod autoscales an endpoint; coderai just proxies to
    ``<serverless_base>/<endpoint_id>/openai/v1`` with the account API key.
    RunPod owns the worker lifecycle; there is nothing for coderai to provision.
  * ``pods`` — coderai provisions, load-balances and scales a POOL of GPU pods
    itself (lifecycle in this module; see RunpodPodPool). [built in a later phase]
  * ``auto`` — pick pods-vs-serverless by the selection criteria. [later phase]

The account settings (API key, endpoints, global caps) live in
:class:`codai.config.RunpodConfig`; this module reads the per-model block.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


# --------------------------------------------------------------------------- #
# Per-model config
# --------------------------------------------------------------------------- #
@dataclass
class RunpodModelConfig:
    """Parsed per-model ``runpod`` block. All fields optional with sane defaults."""
    mode: str = "pods"                       # pods | serverless | auto
    # --- selection (pods) ---
    cloud_types: list = field(default_factory=lambda: ["SECURE"])   # allowed pools
    selection_criteria: str = "cheaper"      # cheaper | faster
    gpu_type: str = ""                       # explicit RunPod gpuTypeId (optional)
    min_vram_gb: float = 0.0
    max_hourly_usd: float = 0.0              # per-model $/hr GPU ceiling (0 = none)
    allow_spot: bool = False
    # --- serving (pods) ---
    image: str = ""                          # blank = image picked from `engine`
    # Which server the pod runs. "auto" reads the model: an HF repo goes to vLLM,
    # a GGUF goes to llama.cpp (vLLM's GGUF support is experimental — single-file
    # only, a limited architecture list, and it still wants the original repo's
    # tokenizer — so it is not a drop-in for the llama.cpp catalogue).
    engine: str = "auto"                     # auto | vllm | llamacpp
    served_model: str = ""                   # HF id the pod/endpoint serves
    # GGUF to pull on a llama.cpp pod, in llama.cpp's own `-hf` form:
    # "user/repo:Q4_K_M" or "user/repo:file.gguf". Needed because a local
    # /AI/…/foo.gguf path means nothing on a rented machine.
    hf_gguf: str = ""
    container_disk_gb: int = 40
    volume_gb: int = 0
    port: int = 8000
    ctx: int = 0
    env: dict = field(default_factory=dict)  # extra pod env (HF_TOKEN, etc.)
    # --- scaling (pods) ---
    min_pods: int = 0
    max_pods: int = 1
    scale_up_inflight_per_pod: int = 4
    idle_timeout_s: int = 300                # destroy a pod this long after its last request
    # Boot budgets — bigger models need longer (image pull + weight download).
    boot_timeout_s: int = 300                # until the pod exposes its port
    load_timeout_s: int = 600                # until vLLM answers /v1/models
    # --- serverless ---
    endpoint_id: str = ""                    # reference an existing serverless endpoint
    min_workers: int = 0
    max_workers: int = 1
    # --- cost budget (cumulative spend, distinct from the $/hr rate cap) ---
    cost_limit_usd: float = 0.0              # 0 = unlimited
    cost_period: str = "unlimited"           # hour | day | week | month | unlimited

    @property
    def is_serverless(self) -> bool:
        return (self.mode or "").lower() == "serverless"


# Default OpenAI-compatible pod image (vLLM's official OpenAI server). The pod
# pulls the model named by ``served_model`` (needs HF_TOKEN in env for gated repos).
DEFAULT_POD_IMAGE = "vllm/vllm-openai:latest"

# llama.cpp's own server image. Its OpenAI surface (/v1/models, /v1/chat/completions)
# matches vLLM's closely enough that the readiness probe and the proxy are unchanged.
# This is what serves the GGUF catalogue remotely: those models have no safetensors
# repo to hand vLLM, and vLLM's GGUF loader is experimental (single-file only, a
# limited architecture list, and it needs the original repo for the tokenizer).
LLAMACPP_POD_IMAGE = "ghcr.io/ggml-org/llama.cpp:server-cuda"


def _looks_like_gguf(name: str) -> bool:
    return (name or "").strip().lower().endswith(".gguf")


def resolve_pod_engine(mcfg: "RunpodModelConfig", model_key: str = "",
                       model_path: str = "") -> str:
    """Decide which server a pod runs for this model: "vllm" or "llamacpp".

    An explicit ``engine`` wins. Otherwise: a GGUF (by `hf_gguf`, by the model's
    own path, or by the served name) goes to llama.cpp; anything else — an HF
    repo id — goes to vLLM.
    """
    want = (mcfg.engine or "auto").strip().lower()
    if want in ("vllm", "llamacpp"):
        return want
    if mcfg.hf_gguf:
        return "llamacpp"
    if _looks_like_gguf(mcfg.served_model) or _looks_like_gguf(model_path) \
            or _looks_like_gguf(model_key):
        return "llamacpp"
    return "vllm"


def _llamacpp_docker_args(mcfg: "RunpodModelConfig", served: str) -> str:
    """Args for the llama.cpp server image so it downloads and serves the GGUF.

    llama.cpp pulls weights itself with ``-hf user/repo:QUANT``; a local path is
    meaningless on a rented machine, so ``hf_gguf`` (or an HF-shaped
    ``served_model``) is required.
    """
    src = (mcfg.hf_gguf or "").strip()
    if not src:
        cand = (served or "").strip()
        if cand and "/" in cand and not cand.startswith("/"):
            src = cand
    if not src:
        raise RuntimeError(
            "RunPod llama.cpp pod: set `hf_gguf` on the model's runpod block to the "
            "GGUF to serve (\"user/repo:Q4_K_M\"). A local .gguf path does not exist "
            "on a rented pod, and coderai does not upload multi-GB weights.")
    args = ["--host", "0.0.0.0", "--port", str(mcfg.port or 8000),
            "-hf", src, "--alias", served or src, "-ngl", "999"]
    if mcfg.ctx and mcfg.ctx > 0:
        args += ["-c", str(mcfg.ctx)]
    return " ".join(args)


def _as_bool(v, default=False):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return default


def _as_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _as_int(v, default=0):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def parse_model_runpod(block: Optional[dict]) -> RunpodModelConfig:
    """Build a RunpodModelConfig from a model entry's ``runpod`` dict (or {})."""
    b = block or {}
    cloud = b.get("cloud_types")
    if isinstance(cloud, str):
        cloud = [c.strip().upper() for c in cloud.split(",") if c.strip()]
    elif isinstance(cloud, list):
        cloud = [str(c).strip().upper() for c in cloud if str(c).strip()]
    else:
        cloud = None
    env = b.get("env")
    if not isinstance(env, dict):
        env = {}
    cfg = RunpodModelConfig()
    cfg.mode = (b.get("mode") or cfg.mode).strip().lower()
    if cloud:
        cfg.cloud_types = cloud
    cfg.selection_criteria = (b.get("selection_criteria") or cfg.selection_criteria).strip().lower()
    cfg.gpu_type = (b.get("gpu_type") or "").strip()
    cfg.min_vram_gb = _as_float(b.get("min_vram_gb"), cfg.min_vram_gb)
    cfg.max_hourly_usd = _as_float(b.get("max_hourly_usd"), cfg.max_hourly_usd)
    cfg.allow_spot = _as_bool(b.get("allow_spot"), cfg.allow_spot)
    cfg.image = (b.get("image") or "").strip()
    cfg.engine = (b.get("engine") or cfg.engine).strip().lower() or "auto"
    cfg.served_model = (b.get("served_model") or "").strip()
    cfg.hf_gguf = (b.get("hf_gguf") or "").strip()
    cfg.container_disk_gb = _as_int(b.get("container_disk_gb"), cfg.container_disk_gb)
    cfg.volume_gb = _as_int(b.get("volume_gb"), cfg.volume_gb)
    cfg.port = _as_int(b.get("port"), cfg.port) or 8000
    cfg.ctx = _as_int(b.get("ctx"), cfg.ctx)
    cfg.env = {str(k): str(v) for k, v in env.items()}
    cfg.min_pods = max(0, _as_int(b.get("min_pods"), cfg.min_pods))
    cfg.max_pods = max(1, _as_int(b.get("max_pods"), cfg.max_pods))
    cfg.scale_up_inflight_per_pod = max(1, _as_int(b.get("scale_up_inflight_per_pod"),
                                                   cfg.scale_up_inflight_per_pod))
    cfg.idle_timeout_s = max(0, _as_int(b.get("idle_timeout_s"), cfg.idle_timeout_s))
    cfg.boot_timeout_s = max(30, _as_int(b.get("boot_timeout_s"), cfg.boot_timeout_s))
    cfg.load_timeout_s = max(30, _as_int(b.get("load_timeout_s"), cfg.load_timeout_s))
    cfg.endpoint_id = (b.get("endpoint_id") or "").strip()
    cfg.min_workers = max(0, _as_int(b.get("min_workers"), cfg.min_workers))
    cfg.max_workers = max(1, _as_int(b.get("max_workers"), cfg.max_workers))
    cfg.cost_limit_usd = _as_float(b.get("cost_limit_usd"), cfg.cost_limit_usd)
    cfg.cost_period = (b.get("cost_period") or cfg.cost_period).strip().lower()
    return cfg


def _model_path_for(model_name: str) -> str:
    """The model's local path from models.json (used to spot a GGUF), or ''."""
    try:
        from codai.models.manager import _model_entry_for
        entry = _model_entry_for(model_name) or {}
        return str(entry.get("path") or "")
    except Exception:
        return ""


def model_runpod_block(model_name: str) -> dict:
    """Fetch the ``runpod`` block from the model's models.json entry, or {}."""
    try:
        from codai.models.manager import _model_entry_for
        entry = _model_entry_for(model_name)
        if entry and isinstance(entry.get("runpod"), dict):
            return entry["runpod"]
    except Exception:
        pass
    return {}


# --------------------------------------------------------------------------- #
# Serverless resolution (RunPod owns the lifecycle — we only build the URL)
# --------------------------------------------------------------------------- #
def serverless_base_url(account_cfg, mcfg: RunpodModelConfig) -> str:
    """Return the OpenAI base URL for a serverless endpoint, e.g.
    ``https://api.runpod.ai/v2/<endpoint_id>/openai/v1``."""
    eid = (mcfg.endpoint_id or "").strip()
    if not eid:
        raise RuntimeError(
            "RunPod serverless: no endpoint_id set on the model's runpod config. "
            "Create a serverless endpoint in RunPod (a vLLM worker) and paste its id.")
    base = (getattr(account_cfg, "serverless_base", "") or "https://api.runpod.ai/v2").rstrip("/")
    return f"{base}/{eid}/openai/v1"


def auth_headers(account_cfg) -> dict:
    """Authorization header for RunPod serverless (Bearer API key)."""
    key = (getattr(account_cfg, "api_key", "") or "").strip()
    return {"Authorization": f"Bearer {key}"} if key else {}


# --------------------------------------------------------------------------- #
# Pods mode — coderai-managed pool of remote GPU pods
# --------------------------------------------------------------------------- #
import threading
import time
import uuid
from dataclasses import dataclass as _dc_pod


def _pod_health_ok(url: str, timeout: float = 4.0) -> bool:
    """True when the pod's OpenAI server answers /v1/models (vLLM is up)."""
    import requests
    try:
        r = requests.get(url.rstrip("/") + "/v1/models", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def _rank_gpus(client, mcfg: "RunpodModelConfig", account_cfg) -> list:
    """Ranked list of GPU options across the model's allowed pools per its selection
    criteria (best first). Each item: {gpu_type_id, display_name, memory_gb,
    cloud_type, price, is_spot, bid}. Raises RunpodError if nothing fits the
    constraints. _provision_one tries them in order so a capacity miss on the top
    pick falls through to the next."""
    from codai.api.runpod_client import RunpodError
    catalog = client.list_gpu_types()
    allowed = [c.upper() for c in (mcfg.cloud_types or ["SECURE"])] or ["SECURE"]
    ceiling = mcfg.max_hourly_usd or 0.0
    explicit = (mcfg.gpu_type or getattr(account_cfg, "default_gpu_type", "") or "").strip()

    cands = []   # (price, is_spot, cloud_type, gpu, on_demand_price)
    for g in catalog:
        if explicit and g["id"] != explicit and g["display_name"] != explicit:
            continue
        mem = g.get("memory_gb") or 0
        if mcfg.min_vram_gb and mem < mcfg.min_vram_gb:
            continue
        for ct in allowed:
            on_demand = g.get("secure_price") if ct == "SECURE" else g.get("community_price")
            spot = g.get("spot_price")
            options = [(on_demand, False)]
            if mcfg.allow_spot and spot:
                options.append((spot, True))
            for price, is_spot in options:
                if price is None or price <= 0:
                    continue
                if ceiling > 0 and price > ceiling:
                    continue
                cands.append((price, is_spot, ct, g, on_demand or price))
    if not cands:
        raise RunpodError(
            f"RunPod: no GPU in pools {allowed} with ≥{mcfg.min_vram_gb}GB under "
            f"${ceiling}/hr (spot={'on' if mcfg.allow_spot else 'off'}"
            + (f", gpu={explicit}" if explicit else "") + ").")

    if (mcfg.selection_criteria or "cheaper").lower() == "faster":
        # Reliability + throughput: SECURE before COMMUNITY, on-demand before spot,
        # more VRAM (throughput proxy) first, then cheapest.
        ct_rank = {"SECURE": 0, "COMMUNITY": 1}
        cands.sort(key=lambda c: (ct_rank.get(c[2], 9), c[1] is True,
                                  -(c[3].get("memory_gb") or 0), c[0]))
    else:  # cheaper
        cands.sort(key=lambda c: (c[0], c[1] is False))
    return [{"gpu_type_id": g["id"], "display_name": g["display_name"],
             "memory_gb": g.get("memory_gb"), "cloud_type": ct, "price": price,
             "is_spot": is_spot, "bid": on_demand if is_spot else 0.0}
            for (price, is_spot, ct, g, on_demand) in cands]


def _select_gpu(client, mcfg: "RunpodModelConfig", account_cfg) -> dict:
    """The single best GPU option (top of _rank_gpus)."""
    return _rank_gpus(client, mcfg, account_cfg)[0]


@_dc_pod
class PodHandle:
    pod_id: str
    url: str
    hourly_usd: float
    started_at: float
    gpu: str = ""
    is_spot: bool = False
    healthy: bool = True
    inflight: int = 0
    last_used: float = 0.0
    console_url: str = ""


def _deploy_tag(account_cfg) -> str:
    """Stable, filesystem-safe deployment tag baked into pod names."""
    t = (getattr(account_cfg, "deployment_id", "") or "default").strip() or "default"
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in t)[:24]


# Pod ids currently mid-provision (created on RunPod but not yet in a pool). The
# reaper must NOT kill these. Guarded by _pools_lock.
_provisioning_ids: set = set()
# Live info for booting pods (shown on the stats page as "provisioning").
_provisioning_info: dict = {}


def _vllm_docker_args(mcfg: "RunpodModelConfig", served: str) -> str:
    """Args appended to the vLLM OpenAI image entrypoint so it serves ``served``
    on the pod's port, reachable through the RunPod proxy."""
    args = ["--host", "0.0.0.0", "--port", str(mcfg.port or 8000),
            "--model", served, "--served-model-name", served]
    if mcfg.ctx and mcfg.ctx > 0:
        args += ["--max-model-len", str(mcfg.ctx)]
    return " ".join(args)


class RunpodPodPool:
    """A pool of coderai-managed RunPod pods for one model.

    * ``acquire()`` returns a ready pod URL (provisioning one on demand, up to
      ``max_pods``, subject to the cost caps) and bumps its in-flight count.
    * ``release()`` drops the in-flight count and stamps last-used.
    * the shared scaler thread (``_scaler_loop``) tears down idle pods to
      ``min_pods`` after ``idle_timeout_s``, re-checks health, reaps dead pods,
      and keeps ``min_pods`` warm.
    """

    def __init__(self, model_key, account_cfg, mcfg: "RunpodModelConfig", served_name):
        self.model_key = model_key
        self.account = account_cfg
        self.mcfg = mcfg
        self.served = served_name
        self.pods: list = []
        self._cv = threading.Condition(threading.RLock())
        self._provisioning = False
        self._closed = False

    # -- cost ------------------------------------------------------------- #
    def live_cost_usd(self) -> float:
        """Cost accrued by pods still running (not yet in the ledger)."""
        now = time.time()
        with self._cv:
            return sum((now - p.started_at) / 3600.0 * p.hourly_usd for p in self.pods)

    def hourly_rate(self) -> float:
        with self._cv:
            return sum(p.hourly_usd for p in self.pods)

    def _budget_blocks(self, new_rate: float) -> str:
        """Return a reason string if provisioning a pod at ``new_rate`` $/hr would
        breach a rate or spend cap, else ''."""
        acct = self.account
        # Global instantaneous rate cap.
        gmax = float(getattr(acct, "global_max_hourly_usd", 0) or 0)
        if gmax > 0 and (_all_pools_hourly_rate() + new_rate) > gmax + 1e-9:
            return f"global ${gmax}/hr rate cap"
        # Per-model cumulative spend cap (rolling period).
        from codai.api import runpod_ledger
        m_lim = float(self.mcfg.cost_limit_usd or 0)
        m_period = self.mcfg.cost_period or "unlimited"
        if m_lim > 0 and m_period != "unlimited":
            spent = runpod_ledger.spend_in_period(self.model_key, m_period) + self.live_cost_usd()
            if spent >= m_lim:
                return f"model spend ${spent:.2f} ≥ ${m_lim} / {m_period}"
        # Global cumulative spend cap.
        g_lim = float(getattr(acct, "global_cost_limit_usd", 0) or 0)
        g_period = getattr(acct, "global_cost_period", "unlimited") or "unlimited"
        if g_lim > 0 and g_period != "unlimited":
            g_spent = runpod_ledger.global_spend_in_period(g_period) + _all_pools_live_cost()
            if g_spent >= g_lim:
                return f"global spend ${g_spent:.2f} ≥ ${g_lim} / {g_period}"
        return ""

    # -- provisioning ----------------------------------------------------- #
    def _create_with_fallback(self, client, ranked, start=0):
        """Try ranked GPU candidates from ``start``; skip capacity misses. Returns
        (pod_id, sel, next_index). Raises if all fail."""
        from codai.api.runpod_client import RunpodError
        port = self.mcfg.port or 8000
        engine = resolve_pod_engine(self.mcfg, str(self.model_key),
                                    _model_path_for(self.model_key))
        if engine == "llamacpp":
            image = self.mcfg.image or LLAMACPP_POD_IMAGE
            args = _llamacpp_docker_args(self.mcfg, self.served)
        else:
            image = self.mcfg.image or DEFAULT_POD_IMAGE
            args = _vllm_docker_args(self.mcfg, self.served)
        env = dict(self.mcfg.env or {})
        dc = getattr(self.account, "data_center", "") or ""
        last = None
        for i in range(start, len(ranked)):
            sel = ranked[i]
            block = self._budget_blocks(sel["price"])
            if block:
                raise RunpodError(f"RunPod budget cap hit — not provisioning ({block}).")
            name = (f"coderai-{_deploy_tag(self.account)}-"
                    + str(self.model_key)[:20].replace("/", "_").replace(" ", "_")
                    + "-" + uuid.uuid4().hex[:6])
            print(f"[runpod] provisioning pod for {self.model_key!r}: {sel['display_name']} "
                  f"({sel['cloud_type']}{'/spot' if sel['is_spot'] else ''}) ${sel['price']}/hr",
                  flush=True)
            try:
                pod_id = client.create_pod(
                    name=name, image=image, gpu_type_id=sel["gpu_type_id"], port=port,
                    cloud_type=sel["cloud_type"], container_disk_gb=self.mcfg.container_disk_gb,
                    volume_gb=self.mcfg.volume_gb, env=env, docker_args=args,
                    is_spot=sel["is_spot"], bid_per_gpu=sel["bid"], data_center_id=dc)
                return pod_id, sel, i + 1
            except RunpodError as exc:
                msg = str(exc).lower()
                if "no longer any instances" in msg or "no instances" in msg or "capacity" in msg:
                    print(f"[runpod] {sel['display_name']} ({sel['cloud_type']}) unavailable — "
                          "trying next candidate", flush=True)
                    last = exc
                    continue
                raise
        raise last or RunpodError("RunPod: no candidate GPU could be deployed (capacity).")

    # Port should appear once the pod's container is running; a much longer wait
    # only bills for a pod that failed to start. The OpenAI server (image pull +
    # model load) gets a longer, separate budget.
    PORT_TIMEOUT_S = 300.0
    HEALTH_TIMEOUT_S = 600.0

    def _dump_pod_logs(self, client, pod_id, why):
        """Fetch + log the pod's container/vLLM output so a failed boot is
        self-diagnosing (the user's #1 cause: vLLM OOM / launch error)."""
        try:
            info = client.get_pod_logs(pod_id, tail=120)
        except Exception as exc:
            info = {"error": str(exc)}
        logs = (info.get("logs") or "").strip()
        console = info.get("console_url") or ""
        if logs:
            tail = "\n".join(logs.splitlines()[-40:])
            print(f"[runpod] pod {pod_id} vLLM/container log ({why}):\n{tail}", flush=True)
        else:
            print(f"[runpod] pod {pod_id} logs unavailable via API ({info.get('error','')}); "
                  f"check the console: {console}", flush=True)

    # How many different machines to try before giving up. A pod that RunPod accepts
    # but never actually starts (stuck pending — seen in practice) is retried on the
    # NEXT candidate rather than failing the request.
    MAX_PROVISION_ATTEMPTS = 3

    def _provision_one(self):
        """Create + boot one pod; append it healthy. Blocking (minutes).

        Retries on a different machine when a pod is accepted but never boots."""
        from codai.api.runpod_client import RunpodClient, RunpodError, pod_console_url
        client = RunpodClient(self.account)
        ranked = _rank_gpus(client, self.mcfg, self.account)
        port = self.mcfg.port or 8000
        idx, attempts, last_exc = 0, 0, None

        while idx < len(ranked) and attempts < self.MAX_PROVISION_ATTEMPTS:
            pod_id, sel, idx = self._create_with_fallback(client, ranked, idx)
            attempts += 1
            console = pod_console_url(pod_id)
            with _pools_lock:
                _provisioning_ids.add(pod_id)   # shield from the reaper during boot
                _provisioning_info[pod_id] = {
                    "model": self.model_key, "gpu": sel["display_name"],
                    "is_spot": sel["is_spot"], "hourly_usd": sel["price"],
                    "console_url": console, "started_at": time.time()}
            print(f"[runpod] pod {pod_id} created for {self.model_key!r} "
                  f"({sel['display_name']} {sel['cloud_type']}) — booting; logs: {console}",
                  flush=True)
            boot_to = float(self.mcfg.boot_timeout_s or self.PORT_TIMEOUT_S)
            load_to = float(self.mcfg.load_timeout_s or self.HEALTH_TIMEOUT_S)
            try:
                url = client.wait_ready(pod_id, port, ready_timeout=boot_to)
                # Wait for the OpenAI server inside the pod (image pull + model load).
                deadline = time.time() + load_to
                while time.time() < deadline:
                    if _pod_health_ok(url):
                        break
                    time.sleep(5)
                else:
                    raise RunpodError(f"pod {pod_id} OpenAI server did not answer "
                                      f"/v1/models within {int(load_to)}s")
            except Exception as exc:
                # Self-diagnose: pull the vLLM/container log before tearing down, so
                # the cause (OOM / bad args / model gate / stuck machine) is in the log.
                last_exc = exc
                print(f"[runpod] pod {pod_id} boot FAILED for {self.model_key!r}: {exc}",
                      flush=True)
                self._dump_pod_logs(client, pod_id, "boot failed")
                try:
                    client.terminate_pod(pod_id)
                except Exception:
                    pass
                with _pools_lock:
                    _provisioning_ids.discard(pod_id)
                    _provisioning_info.pop(pod_id, None)
                if attempts < self.MAX_PROVISION_ATTEMPTS and idx < len(ranked):
                    print(f"[runpod] retrying on the next machine "
                          f"(attempt {attempts + 1}/{self.MAX_PROVISION_ATTEMPTS})", flush=True)
                    continue
                raise
            h = PodHandle(pod_id=pod_id, url=url, hourly_usd=sel["price"],
                          started_at=time.time(), gpu=sel["display_name"],
                          is_spot=sel["is_spot"], healthy=True, last_used=time.time(),
                          console_url=console)
            with self._cv:
                self.pods.append(h)
                self._cv.notify_all()
            with _pools_lock:
                _provisioning_ids.discard(pod_id)
                _provisioning_info.pop(pod_id, None)
            print(f"[runpod] pod {pod_id} ready for {self.model_key!r} at {url}", flush=True)
            return h
        raise last_exc or RunpodError("RunPod: could not provision a pod.")

    def ensure_ready(self):
        """Provision up to max(min_pods, 1) pods and block until ≥1 is healthy.
        Called at model load so the first request has a pod."""
        want = max(self.mcfg.min_pods, 1)
        with self._cv:
            if any(p.healthy for p in self.pods):
                return
        for _ in range(want):
            try:
                self._provision_one()
            except Exception as exc:
                print(f"[runpod] ensure_ready: {exc}", flush=True)
                raise

    def acquire(self, timeout: float = 1200.0):
        """Return (PodHandle, url) for a ready pod, provisioning on demand. Bumps
        in-flight. Caller MUST call release(pod). Raises on failure/timeout."""
        from codai.api.runpod_client import RunpodError
        deadline = time.time() + timeout
        while True:
            with self._cv:
                healthy = [p for p in self.pods if p.healthy]
                if healthy:
                    p = min(healthy, key=lambda x: x.inflight)
                    p.inflight += 1
                    p.last_used = time.time()
                    return p, p.url
                can_grow = (len(self.pods) < self.mcfg.max_pods) and not self._provisioning
                if can_grow:
                    self._provisioning = True
            if can_grow:
                try:
                    self._provision_one()
                finally:
                    with self._cv:
                        self._provisioning = False
                        self._cv.notify_all()
                continue
            # Someone else is provisioning, or we're at max — wait for a free pod.
            with self._cv:
                if not any(p.healthy for p in self.pods):
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        raise RunpodError(
                            f"RunPod: no pod available for {self.model_key!r} "
                            f"(max_pods={self.mcfg.max_pods}) within {timeout}s.")
                    self._cv.wait(min(remaining, 10.0))

    def release(self, pod: PodHandle):
        with self._cv:
            pod.inflight = max(0, pod.inflight - 1)
            pod.last_used = time.time()
            self._cv.notify_all()

    def _terminate(self, pod: PodHandle, reason: str):
        from codai.api.runpod_client import RunpodClient
        from codai.api import runpod_ledger
        cost = (time.time() - pod.started_at) / 3600.0 * pod.hourly_usd
        try:
            RunpodClient(self.account).terminate_pod(pod.pod_id)
        except Exception as exc:
            print(f"[runpod] terminate {pod.pod_id} failed: {exc}", flush=True)
        runpod_ledger.record_spend(self.model_key, cost,
                                   {"pod_id": pod.pod_id, "gpu": pod.gpu, "is_spot": pod.is_spot})
        print(f"[runpod] pod {pod.pod_id} for {self.model_key!r} terminated ({reason}); "
              f"billed ~${cost:.3f}", flush=True)

    def maintain(self):
        """One scaler pass: reap idle/dead pods to min_pods, keep min warm."""
        now = time.time()
        with self._cv:
            pods = list(self.pods)
        # Idle teardown (down to min_pods), only pods with no in-flight work.
        idle_to = self.mcfg.idle_timeout_s
        keep_min = self.mcfg.min_pods
        with self._cv:
            alive = [p for p in self.pods if p.healthy]
            for p in sorted(alive, key=lambda x: x.last_used):
                if len([q for q in self.pods if q.healthy]) <= keep_min:
                    break
                if p.inflight == 0 and idle_to > 0 and (now - p.last_used) > idle_to:
                    p.healthy = False
                    self.pods.remove(p)
                    to_kill = p
                    # terminate outside the lock
                    threading.Thread(target=self._terminate, args=(to_kill, "idle"),
                                     daemon=True).start()
        # Keep min_pods warm.
        with self._cv:
            healthy_n = len([p for p in self.pods if p.healthy])
        if healthy_n < keep_min and not self._closed:
            try:
                self._provision_one()
            except Exception as exc:
                print(f"[runpod] maintain: warm provision failed: {exc}", flush=True)

    def close(self):
        self._closed = True
        with self._cv:
            pods = list(self.pods)
            self.pods.clear()
        for p in pods:
            self._terminate(p, "shutdown")


# module-level pool registry + scaler ------------------------------------- #
_pools_lock = threading.RLock()
_pools: dict = {}          # model_key -> RunpodPodPool
_scaler_started = False


def get_pod_pool(model_key, account_cfg, mcfg: "RunpodModelConfig", served_name) -> RunpodPodPool:
    with _pools_lock:
        pool = _pools.get(model_key)
        if pool is None:
            pool = RunpodPodPool(model_key, account_cfg, mcfg, served_name)
            _pools[model_key] = pool
        else:
            # refresh config each load so edits take effect
            pool.account, pool.mcfg, pool.served = account_cfg, mcfg, served_name
    _ensure_scaler()
    return pool


def _all_pools_hourly_rate() -> float:
    with _pools_lock:
        return sum(p.hourly_rate() for p in _pools.values())


def _all_pools_live_cost() -> float:
    with _pools_lock:
        return sum(p.live_cost_usd() for p in _pools.values())


def pods_status() -> list:
    """Snapshot of all pods across pools for the stats/tasks pages."""
    now = time.time()
    out = []
    with _pools_lock:
        pools = list(_pools.items())
    for key, pool in pools:
        with pool._cv:
            for p in pool.pods:
                out.append({
                    "model": key, "pod_id": p.pod_id, "gpu": p.gpu,
                    "is_spot": p.is_spot, "healthy": p.healthy, "inflight": p.inflight,
                    "hourly_usd": p.hourly_usd, "uptime_s": int(now - p.started_at),
                    "live_cost_usd": round((now - p.started_at) / 3600.0 * p.hourly_usd, 4),
                    "console_url": p.console_url, "state": "ready",
                })
    # Booting pods (created, not yet serving) — visible so a slow/failing boot is
    # obvious and its vLLM log is one click away.
    with _pools_lock:
        prov = list(_provisioning_info.items())
    for pid, info in prov:
        out.append({
            "model": info.get("model"), "pod_id": pid, "gpu": info.get("gpu"),
            "is_spot": info.get("is_spot"), "healthy": False, "inflight": 0,
            "hourly_usd": info.get("hourly_usd"),
            "uptime_s": int(now - (info.get("started_at") or now)),
            "live_cost_usd": round((now - (info.get("started_at") or now)) / 3600.0
                                   * (info.get("hourly_usd") or 0), 4),
            "console_url": info.get("console_url"), "state": "provisioning",
        })
    return out


def _known_pod_ids() -> set:
    """All pod ids this process is responsible for: live pool pods + mid-provision."""
    ids = set()
    with _pools_lock:
        for pool in _pools.values():
            with pool._cv:
                ids.update(p.pod_id for p in pool.pods)
        ids.update(_provisioning_ids)
    return ids


def reap_orphans() -> int:
    """Terminate RunPod pods tagged as OURS that no live pool is tracking — stale
    pods from a crash/restart that would otherwise bill silently. Only touches pods
    named ``coderai-<our deployment_id>-*`` (never another deployment's), and never
    a pod currently mid-provision. Returns the number reaped."""
    try:
        from codai.models.manager import get_active_runpod_config
        acct = get_active_runpod_config()
    except Exception:
        acct = None
    if acct is None or not getattr(acct, "enabled", False) or not getattr(acct, "api_key", ""):
        return 0
    from codai.api.runpod_client import RunpodClient, RunpodError
    prefix = f"coderai-{_deploy_tag(acct)}-"
    try:
        client = RunpodClient(acct)
        pods = client.list_pods()
    except RunpodError as exc:
        print(f"[runpod-reaper] list failed: {exc}", flush=True)
        return 0
    known = _known_pod_ids()
    reaped = 0
    for p in pods:
        pid, name = p.get("id"), (p.get("name") or "")
        status = str(p.get("status") or "").upper()
        if not pid or not name.startswith(prefix):
            continue                      # not ours (or another deployment's)
        if pid in known:
            continue                      # tracked by a live pool / provisioning
        if status in ("TERMINATED", "EXITED"):
            continue
        try:
            client.terminate_pod(pid)
            reaped += 1
            print(f"[runpod-reaper] terminated STALE pod {pid} ({name}) — "
                  "not tracked by any pool", flush=True)
        except Exception as exc:
            print(f"[runpod-reaper] failed to terminate {pid}: {exc}", flush=True)
    return reaped


def _scaler_loop():
    tick = 0
    # Reap once promptly on start to catch orphans left by a crash/restart.
    try:
        reap_orphans()
    except Exception as exc:
        print(f"[runpod-reaper] startup: {exc}", flush=True)
    while True:
        time.sleep(15)
        tick += 1
        try:
            with _pools_lock:
                pools = list(_pools.values())
            for pool in pools:
                try:
                    pool.maintain()
                except Exception as exc:
                    print(f"[runpod] scaler: {exc}", flush=True)
        except Exception:
            pass
        # Independent stale-pod safety sweep every ~30s, even with no local pools.
        if tick % 2 == 0:
            try:
                reap_orphans()
            except Exception as exc:
                print(f"[runpod-reaper] {exc}", flush=True)


def _ensure_scaler():
    global _scaler_started
    with _pools_lock:
        if _scaler_started:
            return
        _scaler_started = True
    threading.Thread(target=_scaler_loop, daemon=True, name="runpod-scaler").start()


def start_runpod_maintenance():
    """Start the scaler + stale-pod reaper independently of any loaded model — call
    at engine startup when RunPod is enabled so orphaned pods from a previous run are
    reaped even before the first request."""
    _ensure_scaler()


def stop_all_pods():
    with _pools_lock:
        pools = list(_pools.values())
        _pools.clear()
    for p in pools:
        try:
            p.close()
        except Exception:
            pass


import atexit as _atexit
_atexit.register(stop_all_pods)
