"""Periodic cleanup of the ds4 on-disk KV-checkpoint cache.

ds4-server writes prompt KV checkpoints to its ``--kv-disk-dir`` (coderai defaults
this to ``<offload>/ds4-kv``) so a prefix can be reused across requests without
recomputing it. Those checkpoints are never self-pruned, so abandoned sessions
accumulate on disk indefinitely (a streamed MoE's KV files are large).

This background janitor age-prunes that directory: every
``interval_minutes`` it deletes top-level entries whose most-recent mtime is older
than ``max_age_hours``. Age is taken from the *newest* file in each entry, so a
recently-rewritten checkpoint is spared.

A checkpoint is removed ONLY when BOTH hold:
  1. ds4 has not USED it for ``max_age_hours`` — age is measured by the newest of
     mtime AND **atime**, so a checkpoint ds4 merely *reads* (a cache hit, which
     bumps atime but not mtime) counts as recently used and is kept; AND
  2. ds4 is not currently using it — no live ``ds4-server`` process holds it open
     via a file descriptor or an mmap (a belt-and-suspenders guard against deleting
     a checkpoint mid-load, which would corrupt a running generation).
Any file ds4 has read/written recently, or has open right now, is always kept.

Caveat: atime relies on the filesystem not being mounted ``noatime``; with the
common ``relatime`` default atime lags by up to ~24 h, which is negligible against
the multi-hour/day cleanup threshold.

It reuses the age/size primitives from ``codai.models.tmp_janitor`` and keeps its
own module-level state (a separate, independently-configured janitor). Started once
from ``codai.main`` in the engine process that owns the ds4 worker.
"""
import os
import shutil
import threading
import time
import logging
from typing import Optional, Dict, Any, Set

from codai.models.tmp_janitor import _dir_size, _FORBIDDEN

_log = logging.getLogger(__name__)


def _entry_newest_activity(path: str) -> float:
    """Most-recent activity time under ``path`` = max(mtime, atime) of the entry,
    or of the newest file in a directory tree. Using atime as well as mtime means a
    checkpoint ds4 only *reads* (a cache hit — bumps atime, not mtime) still counts
    as recently used and is spared."""
    def _t(p: str) -> float:
        try:
            st = os.lstat(p)
            return max(st.st_mtime, st.st_atime)
        except OSError:
            return 0.0
    newest = _t(path)
    if os.path.isdir(path) and not os.path.islink(path):
        for root, _dirs, files in os.walk(path):
            for name in files:
                m = _t(os.path.join(root, name))
                if m > newest:
                    newest = m
    return newest

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


def _ds4_request_active() -> bool:
    """True while ds4 is serving a request. Best-effort: on any error assume idle
    (False) so the janitor still runs — the per-file in-use/age guards keep deletes
    safe regardless of this coarse gate."""
    try:
        from codai.backends.ds4 import Ds4Backend
        return Ds4Backend.any_request_active()
    except Exception:
        return False


def _ds4_open_paths(root: str) -> Set[str]:
    """Real paths under ``root`` currently held open — via an fd OR an mmap — by
    any live ``ds4-server`` process. Linux ``/proc`` scan, best-effort: on any
    error it returns what it has so the janitor errs toward KEEPING files."""
    root_pref = root.rstrip("/") + "/"
    open_paths: Set[str] = set()
    try:
        pids = [d for d in os.listdir("/proc") if d.isdigit()]
    except OSError:
        return open_paths
    for pid in pids:
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                if b"ds4-server" not in f.read():
                    continue
        except OSError:
            continue
        # Open file descriptors.
        fddir = f"/proc/{pid}/fd"
        try:
            for fd in os.listdir(fddir):
                try:
                    tgt = os.readlink(os.path.join(fddir, fd))
                except OSError:
                    continue
                if tgt.startswith(root_pref):
                    open_paths.add(os.path.realpath(tgt))
        except OSError:
            pass
        # Memory-mapped files (ds4 mmaps KV checkpoints; these don't show as fds).
        try:
            with open(f"/proc/{pid}/maps", "r") as f:
                for line in f:
                    parts = line.rstrip("\n").split(None, 5)
                    if len(parts) == 6 and parts[5].startswith(root_pref):
                        open_paths.add(os.path.realpath(parts[5]))
        except OSError:
            pass
    return open_paths


def _sweep(kv_dir: str, max_age_seconds: float) -> tuple[int, int]:
    """Remove top-level entries older than the cutoff, but NEVER one ds4 currently
    has open (fd/mmap). Returns (removed, freed_bytes)."""
    now = time.time()
    removed = 0
    freed = 0
    in_use = _ds4_open_paths(kv_dir)
    try:
        entries = os.listdir(kv_dir)
    except OSError as e:
        _log.debug("ds4 kv janitor: cannot list %s: %s", kv_dir, e)
        return (0, 0)
    for name in entries:
        path = os.path.join(kv_dir, name)
        rp = os.path.realpath(path)
        # Skip if this entry (a file, or any file under a dir entry) is in use.
        if rp in in_use or any(p == rp or p.startswith(rp + "/") for p in in_use):
            continue
        try:
            if now - _entry_newest_activity(path) < max_age_seconds:
                continue
            size = _dir_size(path)
            if os.path.isdir(path) and not os.path.islink(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                os.remove(path)
            removed += 1
            freed += size
        except OSError as e:
            _log.debug("ds4 kv janitor: could not remove %s: %s", path, e)
    return (removed, freed)


def _run(kv_dir: str, max_age_hours: float, interval_minutes: float) -> None:
    max_age_seconds = max(0.0, max_age_hours) * 3600.0
    interval = max(60.0, interval_minutes * 60.0)
    while True:
        try:
            # The dir may not exist until ds4-server first starts — tolerate that.
            # Skip the whole sweep while ds4 is serving a request: deleting/scanning
            # KV files mid-inference would contend for disk with the live expert
            # streaming and KV load/store. Wait for the next interval instead.
            if os.path.isdir(kv_dir) and not _ds4_request_active():
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
