# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""How close this node is to its two real limits: host memory, and KV cache.

Reserved VRAM is not one of them. vLLM is launched with a
``gpu_memory_utilization`` share and claims that fraction of the card up front,
filling whatever is left after the weights with KV cache — so a pod reads ~90%
VRAM from the moment it boots, whether it is serving one request or sixty-four.
A bar pinned at 90% forever says nothing about load.

What does move with load:

* **host RAM** — the measured ceiling. On an A40 pod serving OCR pages,
  2026-10-18: 43 GB free at 32 in-flight, 12.95 GB at 64, and at 128 the engine
  was OOM-killed outright. The GPU was never the limit; memory was.
* **KV cache usage** — how much of the pre-allocated cache the live requests
  occupy. This is the GPU-side headroom, and it is what decides how many more
  requests can be admitted.
* **preemptions** — the proof the cache ran out. vLLM evicts a running request
  and recomputes it later, so throughput collapses while the counters still look
  busy. A rising count is the clearest "we are past the ceiling" signal there
  is, and it is invisible in utilisation or VRAM.
* **waiting** — requests admitted but not yet running: queueing, i.e. latency
  the client is already paying.

Read from the serving engine's own Prometheus endpoint and from cgroup memory
accounting, both cheap, both cached briefly, and every failure degrades to
"unknown" rather than to a number that looks like a measurement.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, Optional

#: Gauges worth carrying, mapped to the names used in the API.
_WANTED = {
    "vllm:kv_cache_usage_perc": "kv_cache_pct",
    "vllm:num_requests_running": "running",
    "vllm:num_requests_waiting": "waiting",
    "vllm:num_preemptions": "preemptions",
    "vllm:prefix_cache_hits": "prefix_hits",
    "vllm:prefix_cache_queries": "prefix_queries",
}

TTL_S = 2.0
TIMEOUT_S = 1.5

_CACHE: Dict[str, tuple] = {}
_LOCK = threading.Lock()


def parse_prometheus(text: str) -> dict:
    """The gauges in :data:`_WANTED` out of a Prometheus exposition body.

    Counters are summed across label sets (several engines, or a counter split
    by reason), while a usage FRACTION is taken at its maximum: adding the
    fractions of two engines would produce a meaningless >100%.
    """
    sums: Dict[str, float] = {}
    maxes: Dict[str, float] = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, rest = line.partition("{")
        if rest:
            metric = name
            _, _, value = rest.rpartition("}")
        else:
            metric, _, value = line.partition(" ")
        key = _WANTED.get(metric.strip())
        if not key:
            continue
        try:
            v = float(value.strip())
        except (TypeError, ValueError):
            continue
        if key == "kv_cache_pct":
            maxes[key] = max(maxes.get(key, 0.0), v)
        else:
            sums[key] = sums.get(key, 0.0) + v
    out: Dict[str, float] = {}
    # vLLM reports the cache fraction as 0..1; the API reports a percentage,
    # because every other figure beside it on the page is one.
    if "kv_cache_pct" in maxes:
        out["kv_cache_pct"] = round(maxes["kv_cache_pct"] * 100.0, 1)
    for k, v in sums.items():
        out[k] = int(v) if float(v).is_integer() else round(v, 2)
    if out.get("prefix_queries"):
        out["prefix_hit_pct"] = round(
            100.0 * out.get("prefix_hits", 0) / out["prefix_queries"], 1)
    return out


def engine_load(base_url: str) -> dict:
    """Serving load from one engine's ``/metrics``, or {} if unavailable."""
    base = str(base_url or "").strip().rstrip("/")
    if not base:
        return {}
    with _LOCK:
        got = _CACHE.get(base)
    if got and (time.time() - got[0]) < TTL_S:
        return dict(got[1])
    try:
        import requests
        r = requests.get(base + "/metrics", timeout=TIMEOUT_S)
        data = parse_prometheus(r.text) if r.status_code == 200 else {}
    except Exception:
        data = {}
    with _LOCK:
        _CACHE[base] = (time.time(), dict(data))
    return data


def _read_int(path: str) -> Optional[int]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return int(fh.read().strip())
    except Exception:
        return None


def host_memory() -> dict:
    """``{ram_used_gb, ram_total_gb, ram_free_gb}`` for THIS container.

    cgroup accounting first, because that is the number that gets a process
    OOM-killed: in a container /proc/meminfo reports the HOST's memory, so a pod
    with a 60 GB limit on a 1 TB machine looks like it has hundreds of gigabytes
    free right up to the moment the kernel kills it. /proc/meminfo is the
    fallback for a process that is not in a memory cgroup.
    """
    cur = _read_int("/sys/fs/cgroup/memory.current")
    lim_raw = None
    try:
        with open("/sys/fs/cgroup/memory.max", "r", encoding="utf-8") as fh:
            lim_raw = fh.read().strip()
    except Exception:
        lim_raw = None
    limit = None
    if lim_raw and lim_raw != "max":
        try:
            limit = int(lim_raw)
        except ValueError:
            limit = None
    if cur is not None and limit:
        return {"ram_used_gb": round(cur / 1e9, 2),
                "ram_total_gb": round(limit / 1e9, 2),
                "ram_free_gb": round(max(0, limit - cur) / 1e9, 2),
                "ram_source": "cgroup"}
    try:
        info = {}
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                k, _, v = line.partition(":")
                info[k.strip()] = v.strip()

        def _kb(key):
            raw = info.get(key, "").split()
            return int(raw[0]) * 1024 if raw else None
        total, avail = _kb("MemTotal"), _kb("MemAvailable")
        if total and avail is not None:
            return {"ram_used_gb": round((total - avail) / 1e9, 2),
                    "ram_total_gb": round(total / 1e9, 2),
                    "ram_free_gb": round(avail / 1e9, 2),
                    "ram_source": "meminfo"}
    except Exception:
        pass
    return {}


def snapshot(base_url: str = "") -> dict:
    """Host memory plus this engine's serving load, for ``/internal/engine-state``."""
    out = dict(host_memory())
    out.update(engine_load(base_url))
    return out


def forget() -> None:
    with _LOCK:
        _CACHE.clear()
