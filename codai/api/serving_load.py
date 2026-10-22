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
* **requests served, and the tokens in and out** — the work this engine has
  actually done since it started. Not a limit, but the figure that says whether
  a pod being paid for by the hour is earning it: a rented pod at 0 served
  requests is pure cost, and that was invisible on a page showing only
  instantaneous load. Carried across an engine restart by :func:`_accumulate`,
  since vLLM's own counters rewind to zero with the process.

``vllm:prefix_cache_hits`` is deliberately absent: in vLLM 0.27.1 the gauge is
registered from the multimodal-cache counter (`counter_prefix_cache_hits`
assigned `counter_mm_cache_hits`), so a hit rate computed from it is not the
prefix hit rate. A number that is wrong is worse than a dash.

Read from the serving engine's own Prometheus endpoint and from cgroup memory
accounting, both cheap, both cached briefly, and every failure degrades to
"unknown" rather than to a number that looks like a measurement.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, Optional

#: Metrics worth carrying, mapped to the names used in the API. Names are the
#: ones vLLM DECLARES; a counter is exposed by prometheus_client with a
#: ``_total`` suffix, which :func:`parse_prometheus` accepts for either form.
_WANTED = {
    "vllm:kv_cache_usage_perc": "kv_cache_pct",
    "vllm:num_requests_running": "running",
    "vllm:num_requests_waiting": "waiting",
    "vllm:num_preemptions": "preemptions",
    "vllm:request_success": "requests_done",
    "vllm:prompt_tokens": "prompt_tokens",
    "vllm:generation_tokens": "generation_tokens",
}

#: Figures that only ever go up, and so can be carried across an engine
#: restart. Everything else describes this instant and must not be.
_CUMULATIVE = ("requests_done", "prompt_tokens", "generation_tokens",
               "preemptions")

TTL_S = 2.0
TIMEOUT_S = 1.5

_CACHE: Dict[str, tuple] = {}

#: ``{base_url: {key: (last_raw, carried)}}`` -- see :func:`_accumulate`.
_CARRY: Dict[str, Dict[str, tuple]] = {}
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
        metric = metric.strip()
        key = _WANTED.get(metric)
        if key is None and metric.endswith("_total"):
            # prometheus_client renders a Counter named "vllm:prompt_tokens" as
            # the sample "vllm:prompt_tokens_total" (plus a "_created"
            # timestamp, which must never be read as a count). Matching only
            # the declared name silently dropped every counter on this page.
            key = _WANTED.get(metric[:-len("_total")])
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
    return out


def _accumulate(base: str, data: dict) -> dict:
    """Keep the cumulative figures cumulative across an engine restart.

    vLLM's counters start again from zero every time the engine process is
    replaced -- a model swap, a crash, an OOM kill -- so a pod that has served
    four thousand requests reports twelve shortly after one. An operator reads
    the column as "what this pod has done", and a counter that silently rewinds
    is worse than no counter: it hides exactly the restart worth noticing.

    So each drop is treated as a restart and the previous reading is carried
    forward. The result is monotonic for as long as this process lives, which
    is the pod's own lifetime.
    """
    if not data:
        return data
    with _LOCK:
        seen = _CARRY.setdefault(base, {})
        for key in _CUMULATIVE:
            raw = data.get(key)
            if not isinstance(raw, (int, float)):
                continue
            last, carried = seen.get(key, (0.0, 0.0))
            if raw < last:                       # the engine was replaced
                carried += last
            seen[key] = (float(raw), carried)
            total = carried + raw
            data[key] = int(total) if float(total).is_integer() else round(total, 2)
    return data


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
        data = _accumulate(base, data)
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
    """Drop the cached readings AND the restart carry-over."""
    with _LOCK:
        _CACHE.clear()
        _CARRY.clear()
