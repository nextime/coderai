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
    image: str = ""                          # blank = default vLLM-OpenAI image
    served_model: str = ""                   # HF id the pod/endpoint serves
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
    cfg.served_model = (b.get("served_model") or "").strip()
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
    cfg.endpoint_id = (b.get("endpoint_id") or "").strip()
    cfg.min_workers = max(0, _as_int(b.get("min_workers"), cfg.min_workers))
    cfg.max_workers = max(1, _as_int(b.get("max_workers"), cfg.max_workers))
    cfg.cost_limit_usd = _as_float(b.get("cost_limit_usd"), cfg.cost_limit_usd)
    cfg.cost_period = (b.get("cost_period") or cfg.cost_period).strip().lower()
    return cfg


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


def _select_gpu(client, mcfg: "RunpodModelConfig", account_cfg) -> dict:
    """Choose a GPU across the model's allowed pools per its selection criteria.

    Returns {gpu_type_id, display_name, memory_gb, cloud_type, price, is_spot,
    bid} where ``price`` is the effective hourly cost we'll be billed and ``bid``
    is the max bid to place for a spot pod. Raises RunpodError if nothing fits.
    """
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
    price, is_spot, ct, g, on_demand = cands[0]
    return {"gpu_type_id": g["id"], "display_name": g["display_name"],
            "memory_gb": g.get("memory_gb"), "cloud_type": ct, "price": price,
            "is_spot": is_spot, "bid": on_demand if is_spot else 0.0}


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
    def _provision_one(self):
        """Create + boot one pod; append it healthy. Blocking (minutes)."""
        from codai.api.runpod_client import RunpodClient, RunpodError
        client = RunpodClient(self.account)
        sel = _select_gpu(client, self.mcfg, self.account)
        block = self._budget_blocks(sel["price"])
        if block:
            raise RunpodError(f"RunPod budget cap hit — not provisioning ({block}).")
        name = ("coderai-" + str(self.model_key)[:24].replace("/", "_").replace(" ", "_")
                + "-" + uuid.uuid4().hex[:6])
        env = dict(self.mcfg.env or {})
        image = self.mcfg.image or DEFAULT_POD_IMAGE
        print(f"[runpod] provisioning pod for {self.model_key!r}: {sel['display_name']} "
              f"({sel['cloud_type']}{'/spot' if sel['is_spot'] else ''}) ${sel['price']}/hr",
              flush=True)
        pod_id = client.create_pod(
            name=name, image=image, gpu_type_id=sel["gpu_type_id"], port=self.mcfg.port or 8000,
            cloud_type=sel["cloud_type"], container_disk_gb=self.mcfg.container_disk_gb,
            volume_gb=self.mcfg.volume_gb, env=env,
            docker_args=_vllm_docker_args(self.mcfg, self.served),
            is_spot=sel["is_spot"], bid_per_gpu=sel["bid"],
            data_center_id=getattr(self.account, "data_center", "") or "")
        try:
            url = client.wait_ready(pod_id, self.mcfg.port or 8000, ready_timeout=900.0)
            # Now wait for the OpenAI server inside the pod (image pull + model load).
            deadline = time.time() + 900.0
            while time.time() < deadline:
                if _pod_health_ok(url):
                    break
                time.sleep(5)
            else:
                raise RunpodError(f"pod {pod_id} OpenAI server not ready in time")
        except Exception:
            try:
                client.terminate_pod(pod_id)
            except Exception:
                pass
            raise
        h = PodHandle(pod_id=pod_id, url=url, hourly_usd=sel["price"], started_at=time.time(),
                      gpu=sel["display_name"], is_spot=sel["is_spot"], healthy=True,
                      last_used=time.time())
        with self._cv:
            self.pods.append(h)
            self._cv.notify_all()
        print(f"[runpod] pod {pod_id} ready for {self.model_key!r} at {url}", flush=True)
        return h

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
                })
    return out


def _scaler_loop():
    while True:
        time.sleep(15)
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


def _ensure_scaler():
    global _scaler_started
    with _pools_lock:
        if _scaler_started:
            return
        _scaler_started = True
    threading.Thread(target=_scaler_loop, daemon=True, name="runpod-scaler").start()


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
