"""Periodic cleanup of the temporary-working directory.

Several pipelines write scratch files with ``tempfile.NamedTemporaryFile(delete=
False)`` / ``mkdtemp()`` (frame extraction, upscaling, interpolation, dubbing,
voice cloning…). When a generation is interrupted those temp entries are never
removed, so a dedicated ``tmp_dir`` slowly fills up (it had grown to tens of GB).

This background janitor age-prunes that directory: every
``interval_minutes`` it deletes top-level entries whose most-recent mtime is older
than ``max_age_hours``. Age-based pruning means in-flight work (touched recently)
is left alone while abandoned scratch is reclaimed.

Safety: it only ever operates on the *configured* ``tmp_dir`` (a dedicated path).
It refuses to run against a bare system temp dir (/tmp, /var/tmp, …) so it can
never delete other processes' files. Mirrors ``codai.models.ram_monitor`` in
shape: module-level state + ``get_status()``, started once from ``codai.main``.
"""
import os
import shutil
import threading
import time
import logging
from typing import Optional, Dict, Any

_log = logging.getLogger(__name__)

# Paths we must never treat as a prunable dedicated tmp dir.
_FORBIDDEN = {"/", "/tmp", "/var/tmp", "/usr/tmp", "/dev/shm"}

_state_lock = threading.Lock()
_state: Dict[str, Any] = {
    "enabled": False,
    "tmp_dir": None,
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


def _entry_newest_mtime(path: str) -> float:
    """Most-recent mtime under ``path`` (the entry itself, or the newest file in a
    directory tree). Using the newest mtime avoids deleting a directory whose top
    folder is old but which still has freshly written files inside."""
    try:
        newest = os.lstat(path).st_mtime
    except OSError:
        return 0.0
    if os.path.isdir(path) and not os.path.islink(path):
        for root, _dirs, files in os.walk(path):
            for name in files:
                try:
                    m = os.lstat(os.path.join(root, name)).st_mtime
                    if m > newest:
                        newest = m
                except OSError:
                    continue
    return newest


def _dir_size(path: str) -> int:
    total = 0
    if os.path.isdir(path) and not os.path.islink(path):
        for root, _dirs, files in os.walk(path):
            for name in files:
                try:
                    total += os.lstat(os.path.join(root, name)).st_size
                except OSError:
                    continue
    else:
        try:
            total = os.lstat(path).st_size
        except OSError:
            total = 0
    return total


def _sweep(tmp_dir: str, max_age_seconds: float) -> tuple[int, int]:
    """Remove top-level entries older than the cutoff. Returns (removed, freed)."""
    now = time.time()
    removed = 0
    freed = 0
    try:
        entries = os.listdir(tmp_dir)
    except OSError as e:
        _log.debug("tmp janitor: cannot list %s: %s", tmp_dir, e)
        return (0, 0)
    for name in entries:
        path = os.path.join(tmp_dir, name)
        try:
            if now - _entry_newest_mtime(path) < max_age_seconds:
                continue
            size = _dir_size(path)
            if os.path.isdir(path) and not os.path.islink(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                os.remove(path)
            removed += 1
            freed += size
        except OSError as e:
            _log.debug("tmp janitor: could not remove %s: %s", path, e)
    return (removed, freed)


def _run(tmp_dir: str, max_age_hours: float, interval_minutes: float) -> None:
    max_age_seconds = max(0.0, max_age_hours) * 3600.0
    interval = max(60.0, interval_minutes * 60.0)
    while True:
        try:
            removed, freed = _sweep(tmp_dir, max_age_seconds)
            with _state_lock:
                _state["last_run_ts"] = time.time()
                _state["last_removed"] = removed
                _state["total_removed"] += removed
                _state["last_freed_bytes"] = freed
                _state["runs"] += 1
            if removed:
                _log.info("tmp janitor: removed %d stale entr%s (%.1f MB) from %s",
                          removed, "y" if removed == 1 else "ies",
                          freed / (1024 * 1024), tmp_dir)
        except Exception as e:  # never let the janitor die
            _log.warning("tmp janitor sweep failed: %s", e)
        time.sleep(interval)


def start(tmp_dir: Optional[str], enabled: bool = True,
          max_age_hours: float = 24.0, interval_minutes: float = 60.0) -> bool:
    """Start the janitor for ``tmp_dir``. No-op (returns False) when disabled, when
    no dedicated tmp_dir is configured, or when tmp_dir is a shared system dir."""
    global _thread, _started
    if _started:
        return True
    if not enabled or not tmp_dir:
        return False
    real = os.path.abspath(os.path.expanduser(tmp_dir)).rstrip("/") or "/"
    if real in _FORBIDDEN:
        _log.info("tmp janitor: refusing to prune shared temp dir %s (set a dedicated tmp_dir)", real)
        return False
    if not os.path.isdir(real):
        try:
            os.makedirs(real, exist_ok=True)
        except OSError:
            return False
    with _state_lock:
        _state.update({
            "enabled": True, "tmp_dir": real,
            "max_age_hours": max_age_hours, "interval_minutes": interval_minutes,
        })
    _thread = threading.Thread(target=_run, args=(real, max_age_hours, interval_minutes),
                               name="tmp-janitor", daemon=True)
    _thread.start()
    _started = True
    _log.info("tmp janitor: pruning %s every %.0f min (entries older than %.1f h)",
              real, interval_minutes, max_age_hours)
    return True


def sweep_once(tmp_dir: str, max_age_hours: float = 24.0) -> tuple[int, int]:
    """Run a single prune pass and return (removed, freed_bytes). For cron use."""
    real = os.path.abspath(os.path.expanduser(tmp_dir)).rstrip("/") or "/"
    if real in _FORBIDDEN or not os.path.isdir(real):
        raise SystemExit(f"refusing to prune {real!r} (not a dedicated tmp dir)")
    return _sweep(real, max(0.0, max_age_hours) * 3600.0)


if __name__ == "__main__":
    # One-shot CLI for cron/systemd-timer use, e.g.:
    #   */30 * * * * /path/venv/bin/python -m codai.models.tmp_janitor \
    #                  --tmp /storage/coderai/tmp --max-age-hours 24
    import argparse
    p = argparse.ArgumentParser(description="Prune a dedicated CoderAI temp dir.")
    p.add_argument("--tmp", required=True, help="the dedicated tmp_dir to prune")
    p.add_argument("--max-age-hours", type=float, default=24.0,
                   help="delete entries whose newest file is older than this")
    a = p.parse_args()
    n, b = sweep_once(a.tmp, a.max_age_hours)
    print(f"tmp janitor: removed {n} entr{'y' if n == 1 else 'ies'} "
          f"({b / (1024 * 1024):.1f} MB) from {a.tmp}")
