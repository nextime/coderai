# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""RunPod spend ledger — persistent per-model + global cost accounting.

Records FINAL pod costs (uptime x hourly rate) as they're torn down. Spend for a
period is a trailing rolling window ("$100 per week" = the last 7 days), which
avoids calendar-boundary edge cases. Live in-flight cost (pods still running) is
NOT stored here — it's computed from the pool and added on top for enforcement
and stats, so nothing is double-counted.

Enforcement (in the pool) refuses to provision a pod when the model's or the
account's period spend would be exceeded. cost_period "unlimited" (or a
cost_limit of 0) means no cap.
"""

import json
import os
import threading
import time

_lock = threading.RLock()

_PERIOD_SECONDS = {
    "hour": 3600.0,
    "day": 86400.0,
    "week": 7 * 86400.0,
    "month": 30 * 86400.0,
}
_PRUNE_SECONDS = 31 * 86400.0   # keep at most ~1 month of history

_GLOBAL_KEY = "__global__"


def _store_path() -> str:
    p = os.environ.get("CODERAI_RUNPOD_LEDGER")
    if p:
        return p
    for d in ("/config/coderai", os.path.expanduser("~/.coderai")):
        if os.path.isdir(d):
            return os.path.join(d, "runpod_ledger.json")
    return os.path.join(os.path.expanduser("~/.coderai"), "runpod_ledger.json")


def _load() -> dict:
    try:
        with open(_store_path()) as f:
            d = json.load(f)
            return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _save(data: dict) -> None:
    path = _store_path()
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)
    except Exception as exc:
        print(f"[runpod-ledger] save failed: {exc}", flush=True)


def _prune(entries: list, now: float) -> list:
    cutoff = now - _PRUNE_SECONDS
    return [e for e in entries if (e.get("ts") or 0) >= cutoff]


def record_spend(model_key: str, usd: float, meta: dict = None) -> None:
    """Append a final spend record for a torn-down pod (and to the global tally)."""
    if not usd or usd <= 0:
        return
    now = time.time()
    rec = {"ts": now, "usd": float(usd)}
    if meta:
        rec.update({k: meta[k] for k in ("pod_id", "gpu", "is_spot") if k in meta})
    with _lock:
        data = _load()
        for key in (model_key or "?", _GLOBAL_KEY):
            lst = data.get(key) or []
            lst.append(dict(rec))
            data[key] = _prune(lst, now)
        _save(data)


def _sum_in_window(entries: list, seconds: float, now: float) -> float:
    cutoff = now - seconds
    return sum((e.get("usd") or 0.0) for e in entries if (e.get("ts") or 0) >= cutoff)


def spend_in_period(model_key: str, period: str) -> float:
    """Trailing spend for one model over the given rolling period (0 if unlimited)."""
    secs = _PERIOD_SECONDS.get((period or "unlimited").lower())
    if not secs:
        return 0.0
    now = time.time()
    with _lock:
        return _sum_in_window(_load().get(model_key or "?") or [], secs, now)


def global_spend_in_period(period: str) -> float:
    """Trailing spend across ALL RunPod models over the given rolling period."""
    secs = _PERIOD_SECONDS.get((period or "unlimited").lower())
    if not secs:
        return 0.0
    now = time.time()
    with _lock:
        return _sum_in_window(_load().get(_GLOBAL_KEY) or [], secs, now)


def totals() -> dict:
    """Snapshot for the stats page: trailing spend per model + global, all periods."""
    now = time.time()
    out = {"per_model": {}, "global": {}}
    with _lock:
        data = _load()
    for period, secs in _PERIOD_SECONDS.items():
        out["global"][period] = round(_sum_in_window(data.get(_GLOBAL_KEY) or [], secs, now), 4)
    for key, entries in data.items():
        if key == _GLOBAL_KEY:
            continue
        out["per_model"][key] = {
            period: round(_sum_in_window(entries, secs, now), 4)
            for period, secs in _PERIOD_SECONDS.items()
        }
    return out
