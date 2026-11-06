# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Live GPU/VRAM telemetry for rented pods, read FROM THE PODS.

A pod serves `/internal/engine-state`, which already carries a `vram` block —
used, free and total, per card, with the GPU's name — built context-free and
cached there for 4s. (It is the endpoint the front health-polls, so it exists on
every pod and costs the pod nothing extra.) Asking the pod is cheaper and
fresher than the alternatives:

  * RunPod's own API reports utilisation too, but every read spends quota from
    the same budget renting and boot-polling spend (a 429 there stops the fleet
    scaling — see codai.api.runpod_client), and its numbers lag by tens of
    seconds. Telemetry must never be able to cost a fleet its ability to rent.
  * the orchestrator's own GPU is not the pod's. It usually has no card at all.

Bounded on every axis, because this feeds a page an operator opens WHEN
something is wrong: one short timeout per pod, all pods concurrently, a TTL so
several open tabs collapse onto one round of requests, and a pod that does not
answer contributes nothing rather than failing the page. The telemetry is
decoration; the spend figures next to it are not.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Dict, Iterable, List, Optional, Tuple

#: How long a pod's answer is reused. The pod's own cache is 1.5s, so a shorter
#: TTL here would only re-ask for the same numbers.
TTL_S = 4.0

#: Per-pod request timeout. A pod mid-boot or saturated simply does not answer,
#: and waiting longer makes the page slower without making it truer.
TIMEOUT_S = 2.5

_CACHE: Dict[str, Tuple[float, dict]] = {}
_LOCK = threading.Lock()

#: How recently the preemption counter must have MOVED for a pod to count as
#: being past its cache ceiling now.
#:
#: ``preemptions`` is cumulative (see codai.api.serving_load._CUMULATIVE) and is
#: deliberately carried across an engine restart, so it only ever goes up. A
#: pod that preempted sixteen requests during one overload an hour ago reports
#: sixteen for the rest of its life -- and a reader shown that total has no way
#: to tell it from a pod preempting sixteen requests a minute. serving_load's
#: own note says it plainly: "a RISING count is the clearest 'we are past the
#: ceiling' signal". The total is history; the delta is the signal.
#:
#: A window rather than a bare per-poll delta because the page polls every few
#: seconds: a delta alone would show the alarm on exactly the one poll the
#: counter moved and hide it on the next, which reads as a glitch instead of a
#: condition.
PREEMPT_FRESH_S = 120.0

#: pod_id -> (highest total seen, monotonic time it last increased).
_PREEMPT: Dict[str, Tuple[float, float]] = {}


def _root(url: str) -> str:
    """The pod's root URL, with the OpenAI `/v1` suffix removed if present.

    Pool handles carry the root; some callers append `/v1` before passing one
    on, and the admin endpoints live above it either way.
    """
    u = str(url or "").strip().rstrip("/")
    return u[:-3].rstrip("/") if u.endswith("/v1") else u


def summarise(cards: Iterable[dict]) -> dict:
    """One row's worth of numbers from a pod's per-card list.

    Utilisation and temperature are the MAXIMUM across cards, not the mean: on
    a multi-card pod the hot, saturated card is the one that decides whether
    the pod can take more work, and averaging it with an idle sibling hides
    exactly the condition worth seeing. VRAM is summed, since that is the pool
    a model is actually loaded into.
    """
    util = mem_used = mem_total = temp = None
    n = 0
    for c in cards or []:
        if not isinstance(c, dict):
            continue
        n += 1
        for key, src in (("util", "util"), ("temp", "temp")):
            v = c.get(src)
            if isinstance(v, (int, float)):
                cur = util if key == "util" else temp
                best = v if cur is None else max(cur, v)
                if key == "util":
                    util = best
                else:
                    temp = best
        for src, acc in (("mem_used", "used"), ("mem_total", "total")):
            v = c.get(src)
            if isinstance(v, (int, float)):
                if acc == "used":
                    mem_used = v if mem_used is None else mem_used + v
                else:
                    mem_total = v if mem_total is None else mem_total + v
    if not n:
        return {}
    out = {"gpu_cards": n}
    if util is not None:
        out["gpu_util"] = round(float(util), 1)
    if temp is not None:
        out["gpu_temp_c"] = round(float(temp), 1)
    if mem_used is not None:
        out["vram_used_gb"] = round(float(mem_used), 2)
    if mem_total is not None:
        out["vram_total_gb"] = round(float(mem_total), 2)
    return out


def cards_from_engine_state(state: dict) -> list:
    """The `vram` block of `/internal/engine-state`, as :func:`summarise` cards.

    The pod reports FREE and total per card (that is what a context-free query
    returns); used is the difference. `util`/`temp` are present only on a pod
    new enough to collect them — older pods report VRAM alone, and the page
    shows a dash for what it was not told rather than inventing a zero.
    """
    vram = (state or {}).get("vram") or {}
    devs = vram.get("devices")
    if not isinstance(devs, list) or not devs:
        # A single-card pod that reports only the totals is still worth showing.
        if isinstance(vram.get("total"), (int, float)):
            devs = [{"free": vram.get("free"), "total": vram.get("total"),
                     "util": vram.get("util"), "temp": vram.get("temp")}]
        else:
            return []
    out = []
    for d in devs:
        if not isinstance(d, dict):
            continue
        total, free = d.get("total"), d.get("free")
        card = {}
        if isinstance(total, (int, float)):
            card["mem_total"] = float(total)
            if isinstance(free, (int, float)):
                card["mem_used"] = max(0.0, float(total) - float(free))
        for key in ("util", "temp"):
            if isinstance(d.get(key), (int, float)):
                card[key] = float(d[key])
        if card:
            out.append(card)
    return out


def _fetch(pod_id: str, url: str, key: str) -> dict:
    """Ask ONE pod. Blocking; always called off the event loop."""
    from codai.api import pod_http
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    try:
        r = pod_http.get(_root(url) + "/internal/engine-state",
                         headers=headers, timeout=TIMEOUT_S)
        if r.status_code != 200:
            return {}
        state = r.json() or {}
        out = summarise(cards_from_engine_state(state))
        # Host memory, host CPU, serving load, and what the pod has served
        # since it booted. See codai.api.serving_load for why reserved VRAM is
        # not one of the limits, why CPU is reported separately (a CPU-bound pod
        # is indistinguishable from an idle one on every GPU-side number), and
        # why the served counters survive an engine restart.
        load = state.get("load")
        if isinstance(load, dict):
            for k in ("ram_used_gb", "ram_total_gb", "ram_free_gb",
                      "cpu_util", "cpu_cores",
                      "kv_cache_pct", "running", "waiting", "preemptions",
                      "requests_done", "prompt_tokens", "generation_tokens"):
                if isinstance(load.get(k), (int, float)):
                    out[k] = load[k]
        # What the pod's own admission controller is doing, and the cache size it
        # is doing it against. Reported, never acted on from here: the gate is on
        # the pod because that is where the measurement is.
        #
        # Worth surfacing because without it a throttling pod and a saturated one
        # look identical from outside -- both answer slowly with requests
        # outstanding -- and the first question asked of a fleet that is growing
        # is which of the two it is. `kv_tokens` is the absolute the page's
        # `kv_cache_pct` is a percentage OF, and the only honest basis for
        # comparing one card's cache with another's.
        adm = state.get("admission")
        if isinstance(adm, dict):
            if isinstance(adm.get("limit"), (int, float)):
                out["admission_limit"] = adm["limit"]
            if isinstance(adm.get("ceiling"), (int, float)):
                out["admission_ceiling"] = adm["ceiling"]
            if isinstance(adm.get("reason"), str) and adm["reason"]:
                out["admission_reason"] = adm["reason"][:160]
            for k in ("kv_tokens", "kv_gib", "max_concurrency"):
                if isinstance(adm.get(k), (int, float)):
                    out[k] = adm[k]
        return out
    except Exception:
        # A pod that cannot be reached has no telemetry, which is not an error
        # worth surfacing: the pod's own health is reported beside it already.
        return {}


def cached(pod_id: str) -> dict:
    """The last answer for ``pod_id`` if it is still fresh, else {}."""
    with _LOCK:
        got = _CACHE.get(pod_id)
    if not got:
        return {}
    at, data = got
    return dict(data) if (time.time() - at) < TTL_S else {}


def _note_preemptions(pod_id: str, data: dict) -> None:
    """Turn the cumulative preemption counter into a usable signal.

    Adds, when the pod reported a count at all:

    ``preemptions_recent``
        How many more than the last time this pod was asked. 0 means the cache
        has not overflowed since, whatever the total says.
    ``preemptions_age_s``
        Seconds since the count last rose, or None if it has not risen while
        this process has been watching. The caller decides what counts as
        recent; :data:`PREEMPT_FRESH_S` is the default answer.

    Caller must NOT hold ``_LOCK``.
    """
    total = data.get("preemptions")
    if not isinstance(total, (int, float)):
        return
    total = float(total)
    now = time.monotonic()
    with _LOCK:
        seen, rose_at = _PREEMPT.get(pod_id, (None, None))
        if seen is None:
            # First sighting. The total is whatever happened before we were
            # watching, so it is history by definition: claiming it as "now"
            # is the very bug this exists to fix.
            delta = 0.0
        elif total < seen:
            # Lower than before, so this is a counter that restarted rather
            # than work that un-happened. serving_load carries these across an
            # engine restart, but a pod whose accumulator was lost would
            # otherwise report a huge negative here. Re-baseline to the NEW
            # value -- keeping the old high-water mark instead would swallow
            # every real preemption until the fresh counter climbed past it.
            delta = 0.0
        else:
            delta = total - seen
            if delta > 0:
                rose_at = now
        _PREEMPT[pod_id] = (total, rose_at)
        if len(_PREEMPT) > 512:
            for k in list(_PREEMPT)[: len(_PREEMPT) - 384]:
                _PREEMPT.pop(k, None)
    data["preemptions_recent"] = int(delta)
    data["preemptions_age_s"] = (None if rose_at is None
                                 else round(max(0.0, now - rose_at), 1))


def _store(pod_id: str, data: dict) -> None:
    _note_preemptions(pod_id, data)
    with _LOCK:
        _CACHE[pod_id] = (time.time(), dict(data))
        if len(_CACHE) > 512:                 # a fleet is bounded; this is not
            for k in sorted(_CACHE, key=lambda k: _CACHE[k][0])[:128]:
                _CACHE.pop(k, None)


def forget(pod_id: Optional[str] = None) -> None:
    """Drop cached telemetry — one pod's, or all of it."""
    with _LOCK:
        if pod_id is None:
            _CACHE.clear()
            _PREEMPT.clear()
        else:
            _CACHE.pop(pod_id, None)
            # A terminated pod's id can be reused by nothing, but its baseline
            # must go with it: a FRESH pod inheriting a stale high-water mark
            # would report a negative delta and then re-baseline silently.
            _PREEMPT.pop(pod_id, None)


async def collect(endpoints: List[Tuple[str, str, str]],
                  overall_timeout: float = 3.0) -> Dict[str, dict]:
    """``{pod_id: {gpu_util, vram_*, gpu_temp_c, ram_*, requests_done, ...}}``.

    ``endpoints`` is ``[(pod_id, url, bearer)]``. Fresh cache entries are
    answered without a request; the rest are fetched concurrently off the loop.
    The whole round is bounded: on timeout the pods that did answer are
    returned and the others are simply absent.
    """
    out: Dict[str, dict] = {}
    todo: List[Tuple[str, str, str]] = []
    for pod_id, url, key in endpoints or []:
        hit = cached(pod_id)
        if hit:
            out[pod_id] = hit
        elif url:
            todo.append((pod_id, url, key))
    if not todo:
        return out

    async def _one(pod_id, url, key):
        data = await asyncio.to_thread(_fetch, pod_id, url, key)
        if data:
            _store(pod_id, data)
            out[pod_id] = data

    try:
        await asyncio.wait_for(
            asyncio.gather(*(_one(*t) for t in todo), return_exceptions=True),
            timeout=overall_timeout)
    except (asyncio.TimeoutError, Exception):   # noqa: BLE001 - partial is fine
        pass
    return out
