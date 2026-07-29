# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Thermal protection.

Before running a request against a loaded model, wait until CPU and GPU
temperatures are within safe limits.  This prevents a long sequence of heavy
generations (e.g. batched video) from driving the hardware hot enough that the
machine's own protection trips and powers off.

The guard is model-agnostic and backend-agnostic: it only reads temperatures
(NVIDIA via ``nvidia-smi``, AMD via ``rocm-smi``, CPU via ``psutil`` /
``/sys`` / ``sensors``) and sleeps.  All thresholds are configurable and the
feature can be enabled/disabled independently for CPU and GPU.

Semantics (per sensor, when enabled):
  * If the temperature is at or above ``high`` °C when a request is about to
    run, block until it drops to ``resume`` °C or below, then proceed.
  * If a temperature can't be read, that sensor is treated as safe (we never
    block on missing data).
"""
import os
import shutil
import subprocess
import threading
import time
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Cooldown state (published for the admin Tasks view)
# ---------------------------------------------------------------------------
# A thermal pause is a *global* hardware event: every worker that reaches a
# checkpoint blocks until temps recover. We publish a single process-wide state
# so the Tasks page can show that running work is paused for cooldown. A waiter
# counter (not a bool) keeps the state correct when several workers pause at
# once — the state is "active" while any worker is still cooling.
_cooldown_lock = threading.Lock()
_cooldown_waiters = 0
_cooldown_state: dict = {
    "active": False, "since": 0.0, "waited": 0.0,
    "gpu": None, "cpu": None, "message": "",
}


def get_cooldown_state() -> dict:
    """Snapshot of the current thermal cooldown (see module note). ``active`` is
    True while at least one worker is paused waiting for the hardware to cool."""
    with _cooldown_lock:
        return dict(_cooldown_state)


def soft_throttle_status() -> dict:
    """Current proactive CPU soft-throttle status, computed LIVE from the CPU temp
    and settings (so the Tasks page reflects reality regardless of when the last
    per-step checkpoint fired — video steps can be tens of seconds apart).

    ``active`` is True when soft-throttle is enabled and the CPU sits in the warm
    band [soft_temp, cpu_high). The caller (Tasks API) additionally gates this on
    there being a running generation, since throttling only happens during work.
    CPU-only by design."""
    s = _settings_from_global_args()
    if not s.soft_enabled or not s.cpu_enabled or s.soft_max_sleep <= 0:
        return {"active": False}
    cpu_t = read_cpu_temp()
    if cpu_t is None or cpu_t < s.soft_temp or cpu_t >= s.cpu_high:
        return {"active": False}
    return {"active": True, "cpu": cpu_t, "sleep": _soft_throttle_sleep(s, cpu_t),
            "engage": s.soft_temp, "cpu_high": s.cpu_high}


def _cooldown_active() -> bool:
    """True while at least one worker is in the cooldown wait loop. Used so that
    other parallel workers join the pause (cross-worker hysteresis) instead of
    racing ahead the instant their own single read dips below the high trigger."""
    with _cooldown_lock:
        return _cooldown_waiters > 0


def _cooldown_enter() -> None:
    global _cooldown_waiters
    with _cooldown_lock:
        _cooldown_waiters += 1
        _cooldown_state["active"] = True
        if not _cooldown_state.get("since"):
            _cooldown_state["since"] = time.time()


def _cooldown_update(gpu, cpu, waited, message) -> None:
    with _cooldown_lock:
        _cooldown_state["gpu"] = gpu
        _cooldown_state["cpu"] = cpu
        _cooldown_state["waited"] = waited
        _cooldown_state["message"] = message


def _cooldown_exit() -> None:
    global _cooldown_waiters
    with _cooldown_lock:
        _cooldown_waiters = max(0, _cooldown_waiters - 1)
        if _cooldown_waiters == 0:
            _cooldown_state.update({
                "active": False, "since": 0.0, "waited": 0.0,
                "gpu": None, "cpu": None, "message": "",
            })


# ---------------------------------------------------------------------------
# External (front-driven) pause
# ---------------------------------------------------------------------------
# The front proxy runs the AUTHORITATIVE thermal monitor: it reads temperatures
# centrally and stays responsive even while an engine is GIL-blocked inside a
# long native call (a single llama.cpp prefill/image-encode that the engine's own
# between-token checkpoint can't interrupt). When the front decides an engine must
# stop, it POSTs /internal/thermal-pause, which sets this flag; the engine honours
# it COOPERATIVELY at the next safe point (request entry / between tokens) — the
# same mechanism as a local temperature cooldown, just triggered remotely. The
# front clears it (/internal/thermal-resume) once the hardware has cooled. If an
# engine ignores the pause (stuck in a native call) the front escalates to an
# OS-level SIGSTOP, which doesn't need the engine to cooperate.
_external_lock = threading.Lock()
_external_pause = {"active": False, "reason": "", "since": 0.0}


def set_external_pause(active: bool, reason: str = "") -> None:
    """Set/clear the front-driven pause. Called from the /internal/thermal-* routes."""
    with _external_lock:
        active = bool(active)
        if active and not _external_pause["active"]:
            _external_pause["since"] = time.time()
        if not active:
            _external_pause["since"] = 0.0
        _external_pause["active"] = active
        _external_pause["reason"] = reason if active else ""


def external_pause_active() -> bool:
    with _external_lock:
        return _external_pause["active"]


def external_pause_state() -> dict:
    with _external_lock:
        return dict(_external_pause)


def _wait_external_pause(settings: "ThermalSettings", context: str = "") -> None:
    """Block while the front holds an external thermal pause, publishing cooldown
    state so the Tasks page shows the engine is paused for cooldown."""
    desc = f" ({context})" if context else ""
    st = external_pause_state()
    print(f"[thermal] Paused by supervisor{desc}: {st.get('reason') or 'cooldown'} "
          f"— holding until resume")
    waited = 0.0
    poll = min(2.0, max(0.5, settings.poll_seconds))
    _cooldown_enter()
    try:
        while external_pause_active():
            _cooldown_update(read_gpu_temp(), read_cpu_temp(), waited,
                             external_pause_state().get("reason")
                             or "paused by thermal supervisor")
            time.sleep(poll)
            waited += poll
    finally:
        _cooldown_exit()
    print(f"[thermal] Supervisor lifted pause{desc} after {int(waited)}s — resuming")


# ---------------------------------------------------------------------------
# Temperature readers
# ---------------------------------------------------------------------------

# Short cache so a tight wait-loop doesn't spawn nvidia-smi dozens of times.
_CACHE_TTL = 2.0
_gpu_cache: Tuple[float, Optional[float]] = (0.0, None)
_cpu_cache: Tuple[float, Optional[float]] = (0.0, None)

_NVIDIA_SMI = shutil.which("nvidia-smi")
_ROCM_SMI = shutil.which("rocm-smi")


def _run(cmd, timeout=4.0) -> Optional[str]:
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        if out.returncode == 0:
            return out.stdout
    except Exception:
        pass
    return None


# Sensor sanity ceiling: no GPU reports a REAL temperature this high — silicon
# dies long before. Readings at/above it are a broken/stuck sensor (observed:
# amdgpu SMU returning a constant 511°C = 0x1FF invalid-ADC after a GPU reset),
# and believing one would freeze the engine forever on a healthy card.
_TEMP_SANE_MAX = 150.0


def _sane_temps(temps):
    good = [t for t in temps if t is not None and 0 < t < _TEMP_SANE_MAX]
    bad = [t for t in temps if t is not None and t >= _TEMP_SANE_MAX]
    if bad:
        global _WARNED_BAD_SENSOR
        if not _WARNED_BAD_SENSOR:
            _WARNED_BAD_SENSOR = True
            print(f"[thermal] IGNORING physically impossible GPU temperature "
                  f"reading(s) {bad} °C — broken/stuck sensor (e.g. amdgpu SMU "
                  f"after a GPU reset). Power-cycle the machine to restore the "
                  f"sensor; thermal protection for that card is DISABLED until "
                  f"it reads sane values again.")
    return good


_WARNED_BAD_SENSOR = False


def _read_gpu_temp_uncached() -> Optional[float]:
    """Hottest GPU temperature in °C across ALL installed cards, or None.

    Spans every vendor (NVIDIA via nvidia-smi, AMD via sysfs) and is scoped to the
    cards THIS engine owns (``CODERAI_ENGINE_GPUS``) — so a hot GPU pauses only the
    engine using it, while a hot CPU (read globally) pauses everything. In
    single-process mode it covers all cards. Falls back to the per-vendor probes
    below if the unified reader fails."""
    try:
        from codai.frontproxy.gpu_detect import engine_gpu_stats
        temps = _sane_temps(
            [c["temp"] for c in engine_gpu_stats() if c.get("temp") is not None])
        if temps:
            return max(temps)
    except Exception:
        pass
    # NVIDIA — the inference GPU on CUDA backends.
    if _NVIDIA_SMI:
        out = _run([
            _NVIDIA_SMI,
            "--query-gpu=temperature.gpu",
            "--format=csv,noheader,nounits",
        ])
        if out:
            temps = []
            for line in out.splitlines():
                line = line.strip()
                if line:
                    try:
                        temps.append(float(line))
                    except ValueError:
                        pass
            if temps:
                return max(temps)
    # AMD ROCm GPUs.
    if _ROCM_SMI:
        out = _run([_ROCM_SMI, "--showtemp"])
        if out:
            temps = []
            for line in out.splitlines():
                if "temperature" in line.lower() and ":" in line:
                    tail = line.rsplit(":", 1)[-1]
                    tail = tail.replace("C", "").replace("c", "").strip()
                    try:
                        temps.append(float(tail))
                    except ValueError:
                        pass
            if temps:
                return max(temps)
    # psutil amdgpu fallback.
    temp = _psutil_temp(("amdgpu", "radeon"))
    return temp


def _psutil_temp(prefer_keys) -> Optional[float]:
    """Max current temp among the preferred psutil sensor groups."""
    try:
        import psutil
    except Exception:
        return None
    try:
        sensors = psutil.sensors_temperatures()
    except Exception:
        return None
    if not sensors:
        return None
    best = None
    for key in prefer_keys:
        if key in sensors:
            for entry in sensors[key]:
                cur = getattr(entry, "current", None)
                if cur is not None:
                    best = cur if best is None else max(best, cur)
    return best


def _read_cpu_temp_uncached() -> Optional[float]:
    """CPU package temperature in °C, or None if unreadable."""
    # psutil covers AMD (k10temp: Tctl/Tdie) and Intel (coretemp: Package).
    temp = _psutil_temp(("k10temp", "coretemp", "zenpower", "cpu_thermal", "k8temp"))
    if temp is not None:
        return temp
    # /sys thermal zones — pick an x86_pkg / cpu zone if present.
    try:
        base = "/sys/class/thermal"
        best = None
        for name in os.listdir(base):
            if not name.startswith("thermal_zone"):
                continue
            zpath = os.path.join(base, name)
            try:
                with open(os.path.join(zpath, "type")) as f:
                    ztype = f.read().strip().lower()
                with open(os.path.join(zpath, "temp")) as f:
                    raw = int(f.read().strip())
            except Exception:
                continue
            if any(k in ztype for k in ("x86_pkg", "cpu", "core", "tctl", "tdie")):
                val = raw / 1000.0
                best = val if best is None else max(best, val)
        if best is not None:
            return best
    except Exception:
        pass
    # `sensors` text parse as a last resort.
    smi = shutil.which("sensors")
    if smi:
        out = _run([smi])
        if out:
            best = None
            for line in out.splitlines():
                low = line.lower()
                if any(k in low for k in ("tctl", "tdie", "package id", "cpu temp")):
                    # e.g. "Tctl:         +55.0°C"
                    for tok in line.replace("+", " ").split():
                        tok = tok.replace("°C", "").replace("C", "").strip()
                        try:
                            v = float(tok)
                        except ValueError:
                            continue
                        best = v if best is None else max(best, v)
                        break
            if best is not None:
                return best
    return None


def read_gpu_temp() -> Optional[float]:
    global _gpu_cache
    now = time.monotonic()
    ts, val = _gpu_cache
    if now - ts < _CACHE_TTL:
        return val
    val = _read_gpu_temp_uncached()
    # FINAL broken-sensor chokepoint: _read_gpu_temp_uncached has several probe
    # fallbacks (rocm-smi, psutil amdgpu) that _sane_temps does NOT cover — an
    # invalid reading (e.g. amdgpu SMU stuck at 511°C = 0x1FF after a GPU reset)
    # slips through the fallback path. Drop any physically impossible value HERE
    # so no reader can act on it, then treat the card as temp-unknown (None).
    if val is not None and val >= _TEMP_SANE_MAX:
        global _WARNED_BAD_SENSOR
        if not _WARNED_BAD_SENSOR:
            _WARNED_BAD_SENSOR = True
            print(f"[thermal] IGNORING physically impossible GPU temperature "
                  f"{val:.0f}°C — broken/stuck sensor (e.g. amdgpu SMU after a "
                  f"GPU reset; power-cycle to restore it). Thermal protection "
                  f"for that card is DISABLED until it reads sane values.",
                  flush=True)
        val = None
    _gpu_cache = (now, val)
    return val


def read_cpu_temp() -> Optional[float]:
    global _cpu_cache
    now = time.monotonic()
    ts, val = _cpu_cache
    if now - ts < _CACHE_TTL:
        return val
    val = _read_cpu_temp_uncached()
    _cpu_cache = (now, val)
    return val


_gpu_util_cache: Tuple[float, Optional[float]] = (0.0, None)


def _read_gpu_util_uncached() -> Optional[float]:
    """Busiest GPU utilization in % across ALL installed cards, or None."""
    try:
        from codai.frontproxy.gpu_detect import engine_gpu_stats
        utils = [c["util"] for c in engine_gpu_stats() if c.get("util") is not None]
        if utils:
            return max(utils)
    except Exception:
        pass
    if _NVIDIA_SMI:
        out = _run([
            _NVIDIA_SMI,
            "--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits",
        ])
        if out:
            vals = []
            for line in out.splitlines():
                line = line.strip()
                if line:
                    try:
                        vals.append(float(line))
                    except ValueError:
                        pass
            if vals:
                return max(vals)
    if _ROCM_SMI:
        out = _run([_ROCM_SMI, "--showuse"])
        if out:
            vals = []
            for line in out.splitlines():
                low = line.lower()
                if "gpu use" in low and "%" in line:
                    for tok in line.replace("%", " ").split():
                        try:
                            vals.append(float(tok))
                        except ValueError:
                            continue
            if vals:
                return max(vals)
    return None


def read_gpu_util() -> Optional[float]:
    """GPU utilization % (cached ~2s), or None if unreadable."""
    global _gpu_util_cache
    now = time.monotonic()
    ts, val = _gpu_util_cache
    if now - ts < _CACHE_TTL:
        return val
    val = _read_gpu_util_uncached()
    _gpu_util_cache = (now, val)
    return val


# Persistent psutil.Process handles, so cpu_percent() can report usage *since the
# previous call* without blocking. Keyed by pid.
_proc_cpu_cache: dict = {}


def read_process_tree_cpu() -> Optional[float]:
    """CPU% of the coderai process tree (this process + all children).

    Scale is 100% PER CORE: a single fully-used core is 100%, so the value ranges
    0 .. 100*cpu_count (e.g. 24 cores → up to 2400%). Non-blocking — it measures
    usage since the previous call (the Tasks page polls every ~2 s), so the very
    first reading after start is ~0 and corrects on the next poll. Torch runs its
    compute on threads inside THIS process, so the main process already accounts
    for generation load; children cover ffmpeg/subprocess work.
    """
    try:
        import psutil
    except Exception:
        return None
    try:
        root = psutil.Process()
        procs = [root] + root.children(recursive=True)
    except Exception:
        return None
    live: dict = {}
    total = 0.0
    for p in procs:
        try:
            pid = p.pid
            cached = _proc_cpu_cache.get(pid)
            if cached is None:
                p.cpu_percent(None)        # prime; contributes ~0 this round
                live[pid] = p
            else:
                total += cached.cpu_percent(None)   # usage since last call
                live[pid] = cached
        except Exception:
            pass
    _proc_cpu_cache.clear()
    _proc_cpu_cache.update(live)
    return round(total, 1)


def read_cpu_temp_avg(samples: int = 3, max_seconds: float = 3.0) -> Optional[float]:
    """Averaged CPU temperature for stable resume/cooldown decisions.

    CPU sensors are noisy — two consecutive reads can swing ±10°C — so a single
    sample is a poor basis for deciding the hardware has cooled down. Averaging a
    few *uncached* samples gives a representative value. Only used for the resume
    side; the pause decision deliberately keeps a single read so it reacts
    immediately to a spike and never lets the CPU overheat.

    Bounded to ``max_seconds`` total (default 3s): the samples are spread across
    the budget (≈1s apart for 3 samples) so they actually capture the second-to-
    second swing, and the loop stops at the deadline. This runs during the
    cooldown wait — which already sleeps ``poll_seconds`` (e.g. 10s) between
    checks — so spending up to 3s of that gathering samples adds no real delay.
    """
    global _cpu_cache
    samples = max(1, samples)
    # Spread reads evenly across the budget (e.g. 3 samples over 3s → ~1s apart).
    gap = (max_seconds / samples) if samples > 1 else 0.0
    deadline = time.monotonic() + max_seconds
    vals = []
    for i in range(samples):
        v = _read_cpu_temp_uncached()
        if v is not None:
            vals.append(v)
        if i >= samples - 1:
            break
        # Stop early rather than sleep past the 3s budget.
        if time.monotonic() + gap >= deadline:
            break
        time.sleep(gap)
    if not vals:
        return None
    avg = sum(vals) / len(vals)
    # Refresh the cache so nearby cached reads see the averaged figure.
    _cpu_cache = (time.monotonic(), avg)
    return avg


def _debug_enabled() -> bool:
    """Thermal debug output is gated on the dedicated --debug-thermal flag
    (not the global --debug, which is too noisy for sensor polling)."""
    try:
        from codai.api.state import get_global_args
        ga = get_global_args()
        return bool(getattr(ga, "debug_thermal", False))
    except Exception:
        return False


def _dbg(msg: str) -> None:
    if _debug_enabled():
        print(f"[thermal][debug] {msg}")


def _fmt(temp: Optional[float]) -> str:
    return f"{temp:.1f}°C" if temp is not None else "n/a"


# ---------------------------------------------------------------------------
# Settings + guard
# ---------------------------------------------------------------------------

class ThermalSettings:
    """Resolved thermal-protection settings (with sane defaults)."""

    __slots__ = (
        "cpu_enabled", "gpu_enabled",
        "cpu_high", "cpu_resume", "gpu_high", "gpu_resume",
        "gpu_overrides",
        "poll_seconds",
        "soft_enabled", "soft_temp", "soft_max_sleep",
    )

    def __init__(self, cpu_enabled=True, gpu_enabled=True,
                 cpu_high=90.0, cpu_resume=87.0,
                 gpu_high=90.0, gpu_resume=87.0,
                 poll_seconds=5.0,
                 soft_enabled=False, soft_temp=80.0, soft_max_sleep=3.0,
                 gpu_overrides=None):
        self.cpu_enabled = bool(cpu_enabled)
        self.gpu_enabled = bool(gpu_enabled)
        self.cpu_high = float(cpu_high)
        self.cpu_resume = float(cpu_resume)
        self.gpu_high = float(gpu_high)
        self.gpu_resume = float(gpu_resume)
        self.gpu_overrides = dict(gpu_overrides or {})
        self.poll_seconds = max(1.0, float(poll_seconds))
        self.soft_enabled = bool(soft_enabled)
        self.soft_temp = float(soft_temp)
        self.soft_max_sleep = max(0.0, float(soft_max_sleep))

    def gpu_thresholds(self, vendor):
        """(high, resume) for a card of ``vendor``, honouring per-vendor overrides."""
        ov = (self.gpu_overrides or {}).get((vendor or "").lower())
        if isinstance(ov, dict):
            return (float(ov.get("high", self.gpu_high)),
                    float(ov.get("resume", self.gpu_resume)))
        return self.gpu_high, self.gpu_resume


def _settings_from_global_args() -> ThermalSettings:
    """Build settings from the live global_args, falling back to defaults."""
    try:
        from codai.api.state import get_global_args
        ga = get_global_args()
    except Exception:
        ga = None
    if ga is None:
        return ThermalSettings()
    g = lambda name, default: getattr(ga, name, default)
    return ThermalSettings(
        cpu_enabled=g("thermal_cpu_enabled", True),
        gpu_enabled=g("thermal_gpu_enabled", True),
        cpu_high=g("thermal_cpu_high", 90.0),
        cpu_resume=g("thermal_cpu_resume", 87.0),
        gpu_high=g("thermal_gpu_high", 90.0),
        gpu_resume=g("thermal_gpu_resume", 87.0),
        gpu_overrides=g("thermal_gpu_overrides", None),
        poll_seconds=g("thermal_poll_seconds", 5.0),
        soft_enabled=g("thermal_soft_throttle_enabled", False),
        soft_temp=g("thermal_soft_throttle_temp", 80.0),
        soft_max_sleep=g("thermal_soft_throttle_max_sleep", 3.0),
    )


def _soft_throttle_sleep(settings: "ThermalSettings", cpu_t) -> float:
    """Seconds to sleep at this checkpoint for proactive CPU soft-throttling.

    0 unless the CPU is in its warm band [soft_temp, cpu_high). Scales linearly
    from 0 at soft_temp to soft_max_sleep just below the CPU pause threshold.
    CPU-ONLY by design — GPU heat is left entirely to the hard cooldown."""
    if (not settings.soft_enabled or settings.soft_max_sleep <= 0
            or not settings.cpu_enabled or cpu_t is None):
        return 0.0
    t0 = settings.soft_temp
    if cpu_t < t0:
        return 0.0
    span = max(1.0, settings.cpu_high - t0)
    frac = min(1.0, max(0.0, (cpu_t - t0) / span))
    return settings.soft_max_sleep * frac


_last_checkpoint: dict = {}


def checkpoint(context: str = "", throttle_seconds: float = 0.0) -> None:
    """Mid-generation thermal checkpoint — call between denoise steps / tokens.

    Same semantics as ``wait_until_safe`` (pause while too hot, resume on
    cooldown), but cheap to call in a hot loop: when ``throttle_seconds`` > 0 it
    only re-checks once that much wall-time has elapsed for this ``context``, so
    a token-by-token text loop doesn't read sensors on every token. Pass 0 for
    low-frequency callers (e.g. per-step diffusion callbacks).
    """
    if throttle_seconds and throttle_seconds > 0:
        now = time.monotonic()
        last = _last_checkpoint.get(context, 0.0)
        if (now - last) < throttle_seconds:
            return
        _last_checkpoint[context] = now
    wait_until_safe(context=context)


def gpu_eval(settings: ThermalSettings):
    """Per-card GPU thermal check, scoped to THIS engine's cards.

    Returns ``(over_high, over_resume, worst)`` where ``worst`` is
    ``{name,temp,high,resume,vendor}`` for the card most over its OWN high threshold
    (or hottest vs its resume when none are over high), or ``None`` if no card temp
    is readable. Honours per-vendor overrides, so e.g. a Radeon limit can differ
    from an NVIDIA one and each card is judged against its own threshold."""
    try:
        from codai.frontproxy.gpu_detect import engine_gpu_stats
        cards = engine_gpu_stats()
    except Exception:
        cards = []
    over_high = over_resume = False
    worst = None
    worst_margin = None
    for c in cards:
        t = c.get("temp")
        if t is None:
            continue
        high, resume = settings.gpu_thresholds(c.get("vendor"))
        if t >= high:
            over_high = True
        if t > resume:
            over_resume = True
        margin = t - high
        if worst is None or margin > worst_margin:
            worst_margin = margin
            worst = {"name": c.get("name"), "temp": t, "high": high,
                     "resume": resume, "vendor": c.get("vendor")}
    return over_high, over_resume, worst


def wait_until_safe(settings: Optional[ThermalSettings] = None,
                    debug: bool = False,
                    context: str = "") -> None:
    """Block until CPU and GPU temperatures are within safe limits.

    Returns immediately when protection is disabled or temperatures are below
    their trigger thresholds.  Designed to be called from a worker thread (it
    uses a blocking ``time.sleep``); the heavy request paths already run
    ``request_model`` inside ``asyncio.to_thread``.
    """
    if settings is None:
        settings = _settings_from_global_args()

    # Front-driven pause takes precedence: the central supervisor decided this
    # engine must stop. Block here — a safe point — until it lifts the pause after
    # cooldown. Honoured even when local CPU/GPU protection is disabled, since the
    # front owns this decision (it sees temps the GIL-blocked engine can't).
    if external_pause_active():
        _wait_external_pause(settings, context)

    if not settings.cpu_enabled and not settings.gpu_enabled:
        _dbg(f"protection disabled (cpu/gpu both off) — proceeding [{context}]")
        return

    desc0 = f" [{context}]" if context else ""

    # Read current temps once (cached) and log the full picture in debug mode.
    # GPU is evaluated per-card (each card vs its own vendor threshold); gpu_t is
    # the worst offender's temperature, used for messaging/soft-throttle/debug.
    gpu_over, gpu_over_resume, gpu_worst = (
        gpu_eval(settings) if settings.gpu_enabled else (False, False, None))
    gpu_t = gpu_worst["temp"] if gpu_worst else None
    cpu_t = read_cpu_temp() if settings.cpu_enabled else None
    _dbg(
        f"check{desc0}: "
        f"GPU {_fmt(gpu_t)} (enabled={settings.gpu_enabled}, "
        f"over_high={gpu_over} over_resume={gpu_over_resume}) | "
        f"CPU {_fmt(cpu_t)} (enabled={settings.cpu_enabled}, "
        f"pause>={settings.cpu_high:.0f} resume<={settings.cpu_resume:.0f})"
    )

    # CPU is shared hardware: when it's over the limit EVERY engine pauses (a global
    # event), so the engine actually heating it (colibri) stops too and it can cool.
    cpu_active = settings.cpu_enabled

    hot = []
    if settings.gpu_enabled and gpu_over:
        hot.append(("GPU", gpu_worst["temp"], gpu_worst["resume"]))
    if cpu_active and cpu_t is not None and cpu_t >= settings.cpu_high:
        hot.append(("CPU", cpu_t, settings.cpu_resume))

    # Cross-worker hysteresis: a thermal pause is a GLOBAL hardware event. When a
    # parallel worker is already cooling down, every OTHER running generation must
    # back off too — otherwise the others keep the box hot and the first worker
    # can never reach the (lower) resume threshold. So even when our own single
    # read is below the high trigger, join the pause while temps are still above
    # the resume line and a cooldown is already in progress.
    joined = False
    if not hot and _cooldown_active():
        if (settings.gpu_enabled and gpu_over_resume) or \
           (cpu_active and cpu_t is not None and cpu_t > settings.cpu_resume):
            joined = True
    if not hot and not joined:
        # Proactive soft-throttle (CPU only): not hot enough to hard-pause, but if
        # the CPU is in the warm band [soft_temp, cpu_high) sleep a little (scaled
        # by how close to the pause threshold) so its temperature climbs slower and
        # we rarely hit the full cooldown. Caps the heat-rate of a single pegged
        # core. GPU heat is handled solely by the hard cooldown.
        _sleep = _soft_throttle_sleep(settings, cpu_t if cpu_active else None)
        if _sleep > 0:
            _dbg(f"soft-throttle{desc0}: sleeping {_sleep:.2f}s "
                 f"(GPU {_fmt(gpu_t)} CPU {_fmt(cpu_t)}, engage>={settings.soft_temp:.0f})")
            time.sleep(_sleep)
        else:
            _dbg(f"within safe limits — serving immediately{desc0}")
        return

    # Enter cooldown: wait until *every* triggered sensor is at/below resume.
    desc = f" ({context})" if context else ""
    if hot:
        # Each triggered sensor carries its own resume threshold (per-card for GPU).
        trig = ", ".join(f"{lbl} {t:.0f}°C (resume<={r:.0f}°C)" for lbl, t, r in hot)
        gpu_note = (f" [{gpu_worst['name']}]" if gpu_worst and any(h[0] == 'GPU' for h in hot)
                    else "")
        print(f"[thermal] Hardware too hot{desc}: {trig}{gpu_note} — pausing requests "
              f"until cooldown")
    else:
        # Joined an already-active cooldown started by another parallel worker.
        print(f"[thermal] Joining active cooldown{desc} — another generation is "
              f"paused; backing off until temps reach resume "
              f"(GPU<={settings.gpu_resume:.0f}°C / CPU<={settings.cpu_resume:.0f}°C)")
    waited = 0.0
    _cooldown_enter()
    try:
        while True:
            # Re-evaluate against resume thresholds (lower than trigger → hysteresis).
            # CPU temps are noisy, so average a few samples for the resume decision
            # (the pause check above stays single-read to react fast to spikes).
            _, gpu_still, gpu_w2 = (gpu_eval(settings) if settings.gpu_enabled
                                    else (False, False, None))
            ct = read_cpu_temp_avg() if settings.cpu_enabled else None
            still = []
            if settings.gpu_enabled and gpu_still:
                still.append(("GPU", gpu_w2["temp"], gpu_w2["resume"]))
            if cpu_active and ct is not None and ct > settings.cpu_resume:
                still.append(("CPU", ct, settings.cpu_resume))
            gt = gpu_w2["temp"] if gpu_w2 else None
            _dbg(f"cooldown{desc} {int(waited)}s: GPU {_fmt(gt)} CPU {_fmt(ct)} (avg-3) "
                 f"(still hot: {[s[0] for s in still] or 'none'})")
            if not still:
                break
            msg = ", ".join(f"{lbl} {t:.0f}°C>{r:.0f}°C" for lbl, t, r in still)
            _cooldown_update(gt, ct, waited, msg)
            print(f"[thermal] Cooling{desc}: {msg} — waiting "
                  f"({int(waited)}s elapsed)")
            time.sleep(settings.poll_seconds)
            waited += settings.poll_seconds
    finally:
        _cooldown_exit()
    print(f"[thermal] Temperatures back within safe limits{desc} — resuming "
          f"after {int(waited)}s")
