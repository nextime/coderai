# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Sharing one RunPod fleet between a controller and N OCR worker processes.

WHY THIS EXISTS. Every OCR page used to pass through one Python process (the
front) and then another (the engine that forwards it to a pod), and each of
those can use one core: Python runs one thread at a time per process. Measured
2026-10-19 on a 35-pod fleet: front 99% CPU, engine 101%, the box's other 14
cores idle, pod GPUs at 0-7% KV cache, ~16 pages/s whatever the fleet size.
128 pods at ~0.65 pages/s each needs ~85.

So the page path is split across processes:

* ONE CONTROLLER -- the ordinary engine -- owns the fleet: it rents, scales,
  leases, quarantines, adopts and terminates. Nothing else ever does.
* N WORKERS (``CODERAI_OCR_WORKER=i/N``) only forward pages. nginx sends
  /v1/ocr requests straight to them, so neither Python front nor controller is
  in the page path at all.

A worker FOLLOWS the fleet through the pod registry the controller already
writes (each pod's address and token), and serves only its own slice of it.
Each pod is assigned once, to the worker holding the fewest, and the choice is
recorded (``assign``), so a pod never moves between workers while pages are on
it -- each pod's in-flight count lives in exactly one process and needs no
cross-process locking.

The two directions of communication are plain files beside the registry,
written under an exclusive lock:

* REPORTS (worker -> controller): per pod, how many pages are on it, when it
  was last used, and whether the worker found it unreachable. Without these the
  controller sees every pod as idle -- its own in-flight is always 0 now -- and
  would scale down or idle-reap pods that are full of work.
* FLAGS (controller -> worker): a pod being drained for scale-down. A worker
  stops giving it new pages, and the controller terminates it only once the
  worker reports it empty.

Separate files rather than fields in the registry because ``register_pod``
rewrites a pod's whole entry: a worker's report would be erased by the next
re-registration, and a worker writing the registry could erase the controller's.
"""

import hashlib
import json
import os
import time
from typing import Dict, Optional, Tuple

#: A worker's report older than this is ignored: the worker is gone, or stuck,
#: and its last word is no longer evidence of anything.
REPORT_MAX_AGE_S = 60.0


def worker_slot() -> Optional[Tuple[int, int]]:
    """(index, count) when THIS process is an OCR worker, else None.

    From ``CODERAI_OCR_WORKER=i/N``. Anything malformed is treated as "not a
    worker", which is the safe direction: a process that wrongly believes it is
    the controller is a process that may rent pods, so the parse is strict and
    the launcher is the only thing that sets it.
    """
    raw = str(os.environ.get("CODERAI_OCR_WORKER", "") or "").strip()
    if not raw or "/" not in raw:
        return None
    try:
        i, n = (int(x) for x in raw.split("/", 1))
    except ValueError:
        return None
    if n < 1 or not 0 <= i < n:
        return None
    return i, n


def is_follower() -> bool:
    """True in an OCR worker: this process must never rent, scale or kill."""
    return worker_slot() is not None


def owns(pod_id: str, slot: Optional[Tuple[int, int]] = None) -> bool:
    """Whether the worker in ``slot`` serves ``pod_id``.

    A stable hash, not "the i-th pod of a sorted list": a list assignment moves
    pods between workers whenever one is added or removed, and a pod that moves
    while it still has pages on it is given a second worker's 64 pages on top of
    the first's -- twice what it can take. A hash never moves a pod.
    """
    slot = slot or worker_slot()
    if slot is None:
        return True
    i, n = slot
    if n <= 1:
        return True
    h = int(hashlib.sha1(str(pod_id).encode()).hexdigest()[:12], 16)
    return h % n == i


ASSIGN = "runpod-worker-assign.json"


def assign(pool_key: str, pod_ids, n: int) -> Dict[str, int]:
    """{pod_id: worker index} for ``pod_ids``, balanced and STABLE.

    Why not a hash: measured on 128 pods across 8 workers, a hash split them
    8 to 29 -- the busiest worker would carry ~1,860 pages and saturate exactly
    as the single process did. Here each new pod goes to whichever worker
    currently has FEWEST, and the choice is recorded, so it never moves while
    pages are on it. Any worker may run this: it is deterministic under the
    lock (sorted pods, lowest index wins a tie), so N workers calling it agree.

    A pod that left is forgotten, freeing its worker's slot for the next one.
    An assignment to a worker index that no longer exists (N was lowered) is
    redone.
    """
    key = str(pool_key)
    ids = sorted(str(x) for x in pod_ids)
    n = max(1, int(n))
    out: Dict[str, int] = {}

    def _w(d):
        cur = {k: int(v) for k, v in (d.get(key) or {}).items()
               if k in ids and isinstance(v, int) and 0 <= v < n}
        counts = [0] * n
        for w in cur.values():
            counts[w] += 1
        for pid in ids:
            if pid in cur:
                continue
            w = min(range(n), key=lambda i: (counts[i], i))
            cur[pid] = w
            counts[w] += 1
        changed = d.get(key) != cur
        d[key] = cur
        out.update(cur)
        return changed

    _locked_update(ASSIGN, _w)
    return out


def _path(name: str) -> str:
    from codai.api.runpod_worker import _pod_registry_path
    return os.path.join(os.path.dirname(_pod_registry_path()), name)


def _locked_update(name: str, fn) -> dict:
    """Read-modify-write ``name`` under an exclusive lock. {} on any failure."""
    import fcntl
    path = _path(name)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a+") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            fh.seek(0)
            raw = fh.read().strip()
            try:
                data = json.loads(raw) if raw else {}
            except Exception:
                data = {}
            if not isinstance(data, dict):
                data = {}
            if fn(data):
                fh.seek(0)
                fh.truncate()
                json.dump(data, fh)
                fh.flush()
            return data
    except Exception:
        return {}


REPORTS = "runpod-worker-reports.json"
FLAGS = "runpod-pod-flags.json"
DEMAND = "runpod-worker-demand.json"


def write_reports(worker: str, pods: Dict[str, dict]) -> None:
    """Record what this worker sees on each of its pods, in one locked write.

    ``pods``: {pod_id: {"inflight": int, "used": ts, "suspect": bool}}. Entries
    this worker no longer serves are removed, so a pod that left its slice does
    not keep claiming the worker's last count.
    """
    now = time.time()

    def _w(d):
        for pid in [k for k, v in d.items()
                    if isinstance(v, dict) and v.get("worker") == worker
                    and k not in pods]:
            d.pop(pid, None)
        for pid, rep in pods.items():
            d[str(pid)] = {"worker": worker, "at": now,
                           "inflight": int(rep.get("inflight") or 0),
                           "used": float(rep.get("used") or 0.0),
                           "suspect": bool(rep.get("suspect"))}
        return True

    _locked_update(REPORTS, _w)


def write_demand(worker: str, pool_key: str, waiting: int) -> None:
    """Worker -> controller: pages that want a pod for this pool and have none.

    Reports (above) are keyed by POD, which is the one thing a cold fleet has
    none of: a worker holding pages with zero pods wrote `{}` and the controller
    read it as an idle install. Since a worker never rents and the controller
    only scales up inside its own `acquire` -- which it never calls while the
    workers serve the pages -- nothing could start the first pod. A page waited
    out `ocr.pod_pipeline_wait_s` and 503'd, and with `min_pods: 0` that was
    every page. Demand therefore travels on its own, keyed by POOL.
    """
    now = time.time()

    def _w(d):
        key = f"{worker}|{pool_key}"
        if waiting <= 0:
            return bool(d.pop(key, None))
        d[key] = {"worker": worker, "pool": str(pool_key),
                  "waiting": int(waiting), "at": now}
        return True

    _locked_update(DEMAND, _w)


def read_demand(pool_key: str, max_age: float = REPORT_MAX_AGE_S) -> int:
    """CONTROLLER: how many pages the workers have waiting for this pool.

    The SUM across workers: each one is holding different pages. A stale entry
    is ignored on age, so a worker that died holding pages cannot keep the
    fleet rented.
    """
    now = time.time()
    data = _locked_update(DEMAND, lambda d: False)
    total = 0
    for v in (data or {}).values():
        if not isinstance(v, dict) or str(v.get("pool")) != str(pool_key):
            continue
        if now - float(v.get("at") or 0) > max_age:
            continue
        total += max(0, int(v.get("waiting") or 0))
    return total


def read_reports(max_age: float = REPORT_MAX_AGE_S) -> Dict[str, dict]:
    """{pod_id: report} for every FRESH worker report."""
    now = time.time()
    data = _locked_update(REPORTS, lambda d: False)
    return {k: v for k, v in data.items()
            if isinstance(v, dict) and now - float(v.get("at") or 0) <= max_age}


def set_flag(pod_id: str, **flags) -> None:
    """Controller -> workers. ``None`` for a value removes that flag."""
    def _w(d):
        cur = dict(d.get(str(pod_id)) or {})
        for k, v in flags.items():
            if v is None:
                cur.pop(k, None)
            else:
                cur[k] = v
        if cur:
            d[str(pod_id)] = cur
        else:
            d.pop(str(pod_id), None)
        return True

    _locked_update(FLAGS, _w)


def read_flags() -> Dict[str, dict]:
    return {k: v for k, v in _locked_update(FLAGS, lambda d: False).items()
            if isinstance(v, dict)}


def clear_pod(pod_id: str) -> None:
    """Forget a pod everywhere once it is gone."""
    _locked_update(REPORTS, lambda d: d.pop(str(pod_id), None) is not None)
    _locked_update(FLAGS, lambda d: d.pop(str(pod_id), None) is not None)

    def _a(d):
        hit = False
        for v in d.values():
            if isinstance(v, dict) and v.pop(str(pod_id), None) is not None:
                hit = True
        return hit
    _locked_update(ASSIGN, _a)
