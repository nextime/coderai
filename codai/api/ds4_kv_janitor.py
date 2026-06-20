"""Periodic cleanup of the ds4 on-disk KV-checkpoint cache.

ds4-server writes prompt KV checkpoints to its ``--kv-disk-dir`` (coderai defaults
this to ``<offload>/ds4-kv``) so a prefix can be reused across requests without
recomputing it. Those checkpoints are never self-pruned, so abandoned sessions
accumulate on disk indefinitely (a streamed MoE's KV files are large).

This background janitor age-prunes that directory: every
``interval_minutes`` it deletes top-level entries whose most-recent mtime is older
than ``max_age_hours``. Age is taken from the *newest* file in each entry, so an
actively-reused checkpoint (touched recently) is spared while stale ones are
reclaimed.

It reuses the proven sweep primitives from ``codai.models.tmp_janitor`` and keeps
its own module-level state (a separate, independently-configured janitor). Started
once from ``codai.main`` in the engine process that owns the ds4 worker.
"""
import os
import threading
import time
import logging
from typing import Optional, Dict, Any

from codai.models.tmp_janitor import _sweep, _FORBIDDEN

_log = logging.getLogger(__name__)

_state_lock = threading.Lock()
_state: Dict[str, Any] = {
    "enabled": False,
    "kv_dir": None,
    "max_age_hours": None,
    "interval_minutes": None,
    "last_run_ts": 0.0,
    "last_removed": 0,
    "total_removed": 0,
    "last_freed_bytes": 0,
    "runs": 0,
}
_thread: Optional[threading.Thread] = None
_started = False


def get_status() -> Dict[str, Any]:
    """Snapshot for the admin status endpoint / dashboard."""
    with _state_lock:
        return dict(_state)


def _run(kv_dir: str, max_age_hours: float, interval_minutes: float) -> None:
    max_age_seconds = max(0.0, max_age_hours) * 3600.0
    interval = max(60.0, interval_minutes * 60.0)
    while True:
        try:
            # The dir may not exist until ds4-server first starts — tolerate that.
            if os.path.isdir(kv_dir):
                removed, freed = _sweep(kv_dir, max_age_seconds)
                with _state_lock:
                    _state["last_run_ts"] = time.time()
                    _state["last_removed"] = removed
                    _state["total_removed"] += removed
                    _state["last_freed_bytes"] = freed
                    _state["runs"] += 1
                if removed:
                    _log.info("ds4 kv janitor: removed %d stale entr%s (%.1f MB) from %s",
                              removed, "y" if removed == 1 else "ies",
                              freed / (1024 * 1024), kv_dir)
        except Exception as e:  # never let the janitor die
            _log.warning("ds4 kv janitor sweep failed: %s", e)
        time.sleep(interval)


def start(kv_dir: Optional[str], enabled: bool = False,
          max_age_hours: float = 168.0, interval_minutes: float = 360.0) -> bool:
    """Start the janitor for the ds4 KV-disk dir. No-op (returns False) when
    disabled, when no dir is given, or when the path is a shared system dir."""
    global _thread, _started
    if _started:
        return True
    if not enabled or not kv_dir:
        return False
    real = os.path.abspath(os.path.expanduser(kv_dir)).rstrip("/") or "/"
    if real in _FORBIDDEN:
        _log.info("ds4 kv janitor: refusing to prune shared dir %s", real)
        return False
    with _state_lock:
        _state.update({
            "enabled": True, "kv_dir": real,
            "max_age_hours": max_age_hours, "interval_minutes": interval_minutes,
        })
    _thread = threading.Thread(target=_run, args=(real, max_age_hours, interval_minutes),
                               name="ds4-kv-janitor", daemon=True)
    _thread.start()
    _started = True
    _log.info("ds4 kv janitor: pruning %s every %.0f min (entries older than %.1f h)",
              real, interval_minutes, max_age_hours)
    return True


def sweep_once(kv_dir: str, max_age_hours: float = 168.0) -> tuple[int, int]:
    """Run a single prune pass and return (removed, freed_bytes). For cron use."""
    real = os.path.abspath(os.path.expanduser(kv_dir)).rstrip("/") or "/"
    if real in _FORBIDDEN or not os.path.isdir(real):
        raise SystemExit(f"refusing to prune {real!r} (not a valid ds4 kv dir)")
    return _sweep(real, max(0.0, max_age_hours) * 3600.0)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Prune the ds4 on-disk KV cache.")
    p.add_argument("--kv-dir", required=True, help="the ds4 --kv-disk-dir to prune")
    p.add_argument("--max-age-hours", type=float, default=168.0,
                   help="delete entries whose newest file is older than this")
    a = p.parse_args()
    n, b = sweep_once(a.kv_dir, a.max_age_hours)
    print(f"ds4 kv janitor: removed {n} entr{'y' if n == 1 else 'ies'} "
          f"({b / (1024 * 1024):.1f} MB) from {a.kv_dir}")
