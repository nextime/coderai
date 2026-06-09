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
import time
from typing import Optional, Tuple


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


def _read_gpu_temp_uncached() -> Optional[float]:
    """Hottest GPU temperature in °C, or None if unreadable."""
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
        "poll_seconds",
    )

    def __init__(self, cpu_enabled=True, gpu_enabled=True,
                 cpu_high=90.0, cpu_resume=87.0,
                 gpu_high=90.0, gpu_resume=87.0,
                 poll_seconds=5.0):
        self.cpu_enabled = bool(cpu_enabled)
        self.gpu_enabled = bool(gpu_enabled)
        self.cpu_high = float(cpu_high)
        self.cpu_resume = float(cpu_resume)
        self.gpu_high = float(gpu_high)
        self.gpu_resume = float(gpu_resume)
        self.poll_seconds = max(1.0, float(poll_seconds))


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
        poll_seconds=g("thermal_poll_seconds", 5.0),
    )


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
    if not settings.cpu_enabled and not settings.gpu_enabled:
        _dbg(f"protection disabled (cpu/gpu both off) — proceeding [{context}]")
        return

    desc0 = f" [{context}]" if context else ""

    # Read current temps once (cached) and log the full picture in debug mode.
    gpu_t = read_gpu_temp() if settings.gpu_enabled else None
    cpu_t = read_cpu_temp() if settings.cpu_enabled else None
    _dbg(
        f"check{desc0}: "
        f"GPU {_fmt(gpu_t)} (enabled={settings.gpu_enabled}, "
        f"pause>={settings.gpu_high:.0f} resume<={settings.gpu_resume:.0f}) | "
        f"CPU {_fmt(cpu_t)} (enabled={settings.cpu_enabled}, "
        f"pause>={settings.cpu_high:.0f} resume<={settings.cpu_resume:.0f})"
    )

    hot = []
    if settings.gpu_enabled and gpu_t is not None and gpu_t >= settings.gpu_high:
        hot.append(("GPU", gpu_t, settings.gpu_resume))
    if settings.cpu_enabled and cpu_t is not None and cpu_t >= settings.cpu_high:
        hot.append(("CPU", cpu_t, settings.cpu_resume))
    if not hot:
        _dbg(f"within safe limits — serving immediately{desc0}")
        return

    # Enter cooldown: wait until *every* triggered sensor is at/below resume.
    desc = f" ({context})" if context else ""
    trig = ", ".join(f"{lbl} {t:.0f}°C>={settings.gpu_high if lbl=='GPU' else settings.cpu_high:.0f}°C"
                     for lbl, t, _ in hot)
    print(f"[thermal] Hardware too hot{desc}: {trig} — pausing requests "
          f"until cooldown (GPU<={settings.gpu_resume:.0f}°C / "
          f"CPU<={settings.cpu_resume:.0f}°C)")
    waited = 0.0
    while True:
        # Re-evaluate against resume thresholds (lower than trigger → hysteresis).
        # CPU temps are noisy, so average a few samples for the resume decision
        # (the pause check above stays single-read to react fast to spikes).
        gt = read_gpu_temp() if settings.gpu_enabled else None
        ct = read_cpu_temp_avg() if settings.cpu_enabled else None
        still = []
        if gt is not None and gt > settings.gpu_resume:
            still.append(("GPU", gt, settings.gpu_resume))
        if ct is not None and ct > settings.cpu_resume:
            still.append(("CPU", ct, settings.cpu_resume))
        _dbg(f"cooldown{desc} {int(waited)}s: GPU {_fmt(gt)} CPU {_fmt(ct)} (avg-3) "
             f"(still hot: {[s[0] for s in still] or 'none'})")
        if not still:
            break
        msg = ", ".join(f"{lbl} {t:.0f}°C>{r:.0f}°C" for lbl, t, r in still)
        print(f"[thermal] Cooling{desc}: {msg} — waiting "
              f"({int(waited)}s elapsed)")
        time.sleep(settings.poll_seconds)
        waited += settings.poll_seconds
    print(f"[thermal] Temperatures back within safe limits{desc} — resuming "
          f"after {int(waited)}s")
