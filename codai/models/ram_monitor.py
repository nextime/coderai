"""Global host-RAM watcher with leak detection and auto-mitigation.

Coderai caps host RAM via ``offload.max_ram_gb`` (see ``OffloadConfig``). The
per-load CPU budget in ``hf_loading`` keeps individual loads under the cap by
spilling to disk, and ``MultiModelManager`` evicts idle models before a new load.
This module adds the *continuous* side: a background thread samples process-tree
RSS, flags a suspected leak when RSS keeps climbing **while the scheduler is idle**
(so growth isn't just an in-flight generation), and runs a mitigation ladder when
RSS crosses a soft threshold — gc → CUDA empty_cache → malloc_trim → drop the
upscaler cache → evict idle models.

Mirrors ``codai.models.thermal`` in shape: module-level state + ``get_status()``
for the admin dashboard, started once from ``codai.main``.
"""
import threading
import time
import logging
from typing import Optional, Dict, Any

_log = logging.getLogger(__name__)

# How often to sample RSS, and how many consecutive rising idle samples count as a
# suspected leak. SOFT_FRACTION of the cap is where the mitigation ladder engages.
_POLL_SECONDS = 15.0
_LEAK_SAMPLES = 4          # consecutive idle increases before flagging a leak
_LEAK_MIN_GROWTH_GB = 0.3  # ignore sub-0.3 GB jitter between samples
_SOFT_FRACTION = 0.90      # mitigate at/above this fraction of the cap
_EVICT_TARGET_FRACTION = 0.85  # evict idle models down to this fraction of the cap

_state_lock = threading.Lock()
_state: Dict[str, Any] = {
    "rss_gb": 0.0,
    "cap_gb": None,
    "percent": None,
    "leak_suspected": False,
    "last_action": "",
    "last_action_ts": 0.0,
    "samples": 0,
}
_recent: list = []           # recent idle RSS samples (for trend detection)
_thread: Optional[threading.Thread] = None
_started = False


def get_status() -> Dict[str, Any]:
    """Snapshot for the admin status endpoint / dashboard."""
    with _state_lock:
        return dict(_state)


def _cap_gb() -> Optional[float]:
    try:
        from codai.api.state import get_global_args
        ga = get_global_args()
        cap = getattr(ga, "max_ram_gb", None) if ga else None
        return float(cap) if cap else None
    except Exception:
        return None


def _watch_enabled() -> bool:
    try:
        from codai.api.state import get_global_args
        ga = get_global_args()
        return bool(getattr(ga, "ram_leak_watch", True)) if ga else True
    except Exception:
        return True


def _scheduler_idle() -> bool:
    """True when no request is being served (so RSS growth isn't a live job)."""
    try:
        from codai.queue.manager import queue_manager
        return not queue_manager.active_leases
    except Exception:
        return True


def _process_ram_gb() -> float:
    try:
        from codai.models.manager import multi_model_manager
        return multi_model_manager._get_process_ram_gb()
    except Exception:
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1e9
        except Exception:
            return 0.0


def _mitigate(rss_gb: float, cap_gb: float, leak: bool) -> str:
    """Run the mitigation ladder; return a short description of what was done."""
    import gc
    actions = []
    for _ in range(3):
        gc.collect()
    actions.append("gc")
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            actions.append("empty_cache")
    except Exception:
        pass
    try:
        from codai.models.manager import _trim_cpu_ram
        _trim_cpu_ram()
        actions.append("trim")
    except Exception:
        pass
    # Drop the in-memory super-resolution upscaler cache — it can hold a multi-GB
    # ESRGAN/diffusers model alive long after an enhance job finished.
    try:
        from codai.api import images as _img
        if getattr(_img, "_UPSCALER_CACHE", None):
            _img._UPSCALER_CACHE.clear()
            actions.append("drop_upscalers")
    except Exception:
        pass
    # Still over and eviction is enabled → unload idle LRU models.
    try:
        from codai.models.manager import multi_model_manager as _mm
        if (_mm._get_process_ram_gb() > cap_gb * _SOFT_FRACTION
                and _mm._evict_idle_on_ram_enabled()):
            _mm._evict_models_for_ram(cap_gb * _EVICT_TARGET_FRACTION)
            actions.append("evict_idle")
    except Exception as e:
        _log.warning("RAM mitigation eviction failed: %s", e)
    desc = ("leak-suspected; " if leak else "") + "+".join(actions)
    return desc


def _loop():
    global _recent
    while True:
        try:
            time.sleep(_POLL_SECONDS)
            if not _watch_enabled():
                continue
            cap = _cap_gb()
            rss = _process_ram_gb()
            idle = _scheduler_idle()

            # Leak heuristic: only trust growth measured while idle (a live job
            # legitimately inflates RSS). Keep a short rolling window of idle samples.
            leak = False
            if idle:
                _recent.append(rss)
                _recent = _recent[-(_LEAK_SAMPLES + 1):]
                if len(_recent) > _LEAK_SAMPLES:
                    rising = all(
                        _recent[i + 1] - _recent[i] >= _LEAK_MIN_GROWTH_GB
                        for i in range(len(_recent) - 1)
                    )
                    leak = rising
            else:
                _recent = []  # reset trend while a job runs

            with _state_lock:
                _state["rss_gb"] = round(rss, 2)
                _state["cap_gb"] = cap
                _state["percent"] = round(100.0 * rss / cap, 1) if cap else None
                _state["leak_suspected"] = leak
                _state["samples"] += 1

            # Engage the ladder when over the soft threshold or a leak is suspected.
            if cap and (rss >= cap * _SOFT_FRACTION or leak):
                desc = _mitigate(rss, cap, leak)
                new_rss = _process_ram_gb()
                _log.warning(
                    "RAM watch: RSS %.1f/%.1f GB (%.0f%%)%s — mitigation [%s] → %.1f GB",
                    rss, cap, 100.0 * rss / cap,
                    " LEAK SUSPECTED" if leak else "", desc, new_rss,
                )
                with _state_lock:
                    _state["last_action"] = desc
                    _state["last_action_ts"] = time.time()
                    _state["rss_gb"] = round(new_rss, 2)
                _recent = []  # avoid re-triggering on the same growth
        except Exception as e:
            _log.debug("RAM watch loop error: %s", e)


def start() -> None:
    """Start the background watcher once. Safe to call when no cap is configured —
    it idles cheaply and begins acting as soon as a cap is set live."""
    global _thread, _started
    if _started:
        return
    _started = True
    _thread = threading.Thread(target=_loop, name="ram-monitor", daemon=True)
    _thread.start()
    _log.info("RAM monitor started (poll %.0fs)", _POLL_SECONDS)
