# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Torch-free GPU detection for the front proxy.

Turns a *vendor keyword* (``"nvidia"`` / ``"radeon"`` / ``"intel"``) into the env
that pins an engine to **all** of that vendor's cards on the local machine — so the
same ``engine_specs`` work on a 1-, 2-, or 10-card box without hand-writing UUIDs or
device indices:

* **CUDA** is pinned by UUID (stable across reboots/reordering), listing every
  NVIDIA card for the NVIDIA engine and ``""`` (hidden) for the others.
* **Vulkan** is isolated by pointing ``VK_ICD_FILENAMES`` at only that vendor's ICD,
  so the engine sees exactly that vendor's cards (no index fragility, no llvmpipe /
  cross-vendor interference).

Everything shells out to ``nvidia-smi`` / ``vulkaninfo`` and reads the Vulkan ICD
directory; nothing imports torch.
"""

import glob
import os
import re
import shutil
import subprocess

# Vulkan ICD search dirs (loader defaults) and per-vendor filename patterns. Names
# vary by distro/driver, so we match several and skip disabled ones.
_ICD_DIRS = ("/usr/share/vulkan/icd.d", "/etc/vulkan/icd.d",
             "/usr/local/share/vulkan/icd.d")
_ICD_PATTERNS = {
    "nvidia": ("nvidia_icd*.json",),
    "amd": ("radeon_icd*.json", "amd_icd*.json"),      # RADV (mesa) or AMDVLK
    "intel": ("intel_icd*.json", "intel_hasvk_icd*.json"),
}
# Vulkan vendor IDs (PCI) for reporting.
VENDOR_IDS = {0x10de: "nvidia", 0x1002: "amd", 0x8086: "intel"}
_ALIASES = {"radeon": "amd", "amd": "amd", "nvidia": "nvidia", "nv": "nvidia",
            "intel": "intel"}


def _norm_vendor(vendor: str) -> str:
    return _ALIASES.get((vendor or "").strip().lower(), (vendor or "").strip().lower())


def nvidia_gpus() -> list:
    """Return [{'uuid','name','pci'}] for each NVIDIA GPU (empty if none)."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return []
    try:
        out = subprocess.run(
            [smi, "--query-gpu=uuid,name,pci.bus_id", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return []
        gpus = []
        for line in out.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if parts and parts[0]:
                gpus.append({"uuid": parts[0],
                             "name": parts[1] if len(parts) > 1 else "",
                             "pci": parts[2] if len(parts) > 2 else ""})
        return gpus
    except Exception:
        return []


def _call_with_timeout(fn, timeout, default):
    """Run ``fn()`` in a daemon thread; return its result, or ``default`` if it
    doesn't finish within ``timeout`` seconds.

    Unlike subprocess timeouts, this survives a callee stuck UNINTERRUPTIBLY
    (D-state on a wedged GPU) — we simply STOP WAITING and leave the daemon
    thread orphaned, so a hung ``vulkaninfo`` can never block the front's
    startup. The orphan clears when the GPU recovers / on reboot."""
    import threading
    box = {}
    def _runner():
        try:
            box["v"] = fn()
        except Exception:
            box["v"] = default
    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        print(f"[gpu-detect] probe exceeded {timeout}s (GPU may be wedged) — "
              f"proceeding without it", flush=True)
        return default
    return box.get("v", default)


def vulkan_devices() -> list:
    """Return [{'vendor','vendor_id','name'}] from ``vulkaninfo --summary``.

    Order matches Vulkan device indexing. Best-effort: empty if vulkaninfo is
    missing/unparseable, or if it HANGS on a wedged GPU (bounded by a wall-clock
    timeout that survives an un-killable D-state child)."""
    return _call_with_timeout(_vulkan_devices_impl, 12, [])


def _vulkan_devices_impl() -> list:
    vk = shutil.which("vulkaninfo")
    if not vk:
        return []
    try:
        out = subprocess.run([vk, "--summary"], capture_output=True, text=True,
                             timeout=10)
        text = out.stdout
    except Exception:
        return []
    devices = []
    cur = {}
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("GPU") and line.endswith(":"):
            if cur:
                devices.append(cur)
            cur = {}
        elif "=" in line:
            k, _, v = line.partition("=")
            k = k.strip().lower(); v = v.strip()
            if k == "vendorid":
                try:
                    vid = int(v, 16) if v.lower().startswith("0x") else int(v)
                except ValueError:
                    vid = None
                cur["vendor_id"] = vid
                cur["vendor"] = VENDOR_IDS.get(vid, "other")
            elif k == "devicename":
                cur["name"] = v
    if cur:
        devices.append(cur)
    return devices


_PCI_VENDOR = {"0x10de": "nvidia", "0x1002": "amd", "0x8086": "intel"}


def sysfs_gpu_vendors() -> set:
    """GPU vendors present per sysfs (``/sys/class/drm/card*/device/vendor``).

    A driver-independent fallback for AMD/Intel detection when ``vulkaninfo`` isn't
    installed. Returns vendor keywords ({"amd","nvidia","intel"})."""
    import re
    vendors = set()
    for card in glob.glob("/sys/class/drm/card*"):
        base = os.path.basename(card)
        if not re.match(r"^card\d+$", base):
            continue
        try:
            with open(os.path.join(card, "device", "vendor")) as f:
                vid = f.read().strip().lower()
        except OSError:
            continue
        v = _PCI_VENDOR.get(vid)
        if v:
            vendors.add(v)
    return vendors


def gpu_vendors() -> set:
    """All GPU vendors present, combining Vulkan enumeration (vulkaninfo) with the
    sysfs PCI-vendor fallback, so detection doesn't depend on vulkaninfo alone."""
    vendors = {d.get("vendor") for d in vulkan_devices() if d.get("vendor")}
    vendors |= sysfs_gpu_vendors()
    vendors.discard("other")   # llvmpipe / software rasterizers
    return vendors


def find_vulkan_icd(vendor: str) -> str:
    """Return the path to a Vulkan ICD JSON for ``vendor``, or '' if not found."""
    vendor = _norm_vendor(vendor)
    patterns = _ICD_PATTERNS.get(vendor, ())
    for d in _ICD_DIRS:
        for pat in patterns:
            for path in sorted(glob.glob(os.path.join(d, pat))):
                if path.endswith(".disabled") or ".disabled" in os.path.basename(path):
                    continue
                if os.path.isfile(path):
                    return path
    return ""


def vendor_env(vendor: str, allow_cross: bool = False) -> dict:
    """Env that pins an engine to **all** of ``vendor``'s cards on this machine.

    NVIDIA: CUDA visible = all NVIDIA UUIDs (+ PCI_BUS_ID order), Vulkan ICD = nvidia.
    AMD/Intel: CUDA hidden (""), Vulkan ICD = that vendor's, so it sees only those
    cards. Missing tools degrade gracefully (the key is simply omitted).

    ``allow_cross`` (opt-in) lets this engine ALSO use other vendors' Vulkan cards
    — e.g. an NVIDIA engine pooling a model across the 3090 (CUDA) and a Radeon
    (Vulkan) for more total VRAM. Off by default: each engine stays on its own
    backend's GPUs (and still splits across multiple same-backend cards)."""
    vendor = _norm_vendor(vendor)
    env = {}
    if vendor == "nvidia":
        uuids = [g["uuid"] for g in nvidia_gpus() if g.get("uuid")]
        if uuids:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(uuids)
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    else:
        # Non-NVIDIA engine: hide all CUDA cards so torch/llama-CUDA can't grab them.
        env["CUDA_VISIBLE_DEVICES"] = ""
    # Likewise hide AMD/Radeon cards from any engine that isn't the AMD one, so a
    # non-AMD engine can't pick up a Radeon (mirrors CUDA hiding for non-NVIDIA).
    if vendor != "amd":
        env["RADEON_VISIBLE_DEVICES"] = ""
    # Vulkan isolation. Detect each Vulkan device's vendor (vulkan_devices() reads
    # vendorid and preserves Vulkan index order) and pin this engine to ONLY its
    # vendor's cards — by their REAL indices, never an assumed 0..n-1 order.
    icd = find_vulkan_icd(vendor)
    vk_devs = vulkan_devices()
    vendor_idx = [i for i, d in enumerate(vk_devs) if d.get("vendor") == vendor]
    if allow_cross:
        # Opt-in cross-backend pooling: expose every REAL GPU Vulkan device (any
        # hardware vendor) so llama.cpp can split a model across backends. EXCLUDE
        # software rasterizers (llvmpipe/lavapipe/virtio → vendor "other"): they are
        # CPU-backed, have no real VRAM, and are far slower than llama.cpp's native
        # CPU offload — and including them both skews the device list (so the
        # tensor_split ratio no longer lines up with the real cards) and lets layers
        # land on a fake "GPU". Don't pin the ICD (we want all hardware vendors).
        real_idx = [i for i, d in enumerate(vk_devs)
                    if d.get("vendor") in ("nvidia", "amd", "intel")]
        if real_idx:
            env["GGML_VK_VISIBLE_DEVICES"] = ",".join(str(i) for i in real_idx)
        elif vk_devs:
            env["GGML_VK_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(len(vk_devs)))
    elif icd:
        # The vendor ICD exists: point the loader at it so ONLY this vendor's cards
        # are visible, then they are reindexed 0..n-1.
        env["VK_ICD_FILENAMES"] = icd
        n = len(vendor_idx)
        if n == 0 and vendor in gpu_vendors():
            n = 1  # vulkaninfo couldn't enumerate (e.g. missing) but the card exists
        env["GGML_VK_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(n)) if n else ""
    elif vendor_idx:
        # No vendor ICD to filter with, but this vendor DOES have Vulkan device(s):
        # select their ACTUAL indices in the unfiltered enumeration (detected, not
        # assumed), so we never grab a different vendor's card sitting at index 0.
        env["GGML_VK_VISIBLE_DEVICES"] = ",".join(str(i) for i in vendor_idx)
    else:
        # No vendor ICD AND no Vulkan device for this vendor — e.g. an NVIDIA engine
        # in a container that lacks nvidia_icd.json (the toolkit only injects it with
        # the graphics capability). Previously VK_ICD_FILENAMES was left unset, so the
        # loader fell back to ALL ICDs and the engine grabbed another vendor's card
        # (the RX 580 via RADV) — splitting the model across the wrong GPU. Give this
        # engine ZERO Vulkan devices instead (it runs on CUDA only); point the loader
        # at a non-existent ICD so no Vulkan driver loads. Per-engine, so the AMD
        # engine still keeps its Vulkan.
        env["VK_ICD_FILENAMES"] = os.path.join(_ICD_DIRS[0], "_coderai_no_vulkan.json")
        env["GGML_VK_VISIBLE_DEVICES"] = ""
    return env


def _nvidia_stats() -> list:
    """Per-GPU live stats from nvidia-smi (reports ALL cards regardless of
    CUDA_VISIBLE_DEVICES). Memory in GB."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return []
    try:
        out = subprocess.run(
            [smi, "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,"
                  "temperature.gpu,uuid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return []
    except Exception:
        return []
    cards = []
    for line in out.stdout.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) < 6 or not p[0].isdigit():
            continue
        def _f(v):
            try:
                return float(v)
            except ValueError:
                return None
        cards.append({"vendor": "nvidia", "index": int(p[0]), "name": p[1],
                      "util": _f(p[2]),
                      "mem_used": round((_f(p[3]) or 0) / 1024, 2),
                      "mem_total": round((_f(p[4]) or 0) / 1024, 2),
                      "temp": _f(p[5]),
                      "uuid": p[6] if len(p) > 6 else None})
    return cards


def _amd_gpu_name(dev: str, base: str) -> str:
    """Best-effort marketing name for an AMD card given its sysfs ``device`` dir.

    Tries, in order: amdgpu's ``product_name`` sysfs attr (e.g. "Radeon RX 7900
    XTX"), ``lspci`` for the card's exact PCI address, and the matching
    ``vulkaninfo`` device name. Falls back to ``AMD GPU (cardN)``."""
    # 1) amdgpu sysfs product_name (newer kernels expose a clean marketing name).
    try:
        with open(os.path.join(dev, "product_name")) as f:
            name = f.read().strip()
        if name:
            return name
    except OSError:
        pass
    # 2) lspci by the card's exact PCI bus address (realpath of the device dir).
    #    The generic chip name (field 2) is shared across rebrands — e.g.
    #    "[Radeon RX 470/480/570/580/590]" — but the subsystem/board name (field 4)
    #    pins the actual model ("Radeon RX 580"), so prefer it when meaningful.
    pci_addr = os.path.basename(os.path.realpath(dev))
    if re.match(r"^[0-9a-fA-F]{4}:", pci_addr):
        lspci = shutil.which("lspci")
        if lspci:
            try:
                out = subprocess.run([lspci, "-mm", "-s", pci_addr],
                                     capture_output=True, text=True, timeout=5)
                # -mm quotes each name: "class" "vendor" "device" "subvendor" "subdevice"
                fields = re.findall(r'"([^"]*)"', out.stdout)
                board = fields[4] if len(fields) >= 5 else ""
                # Use the board name unless it's a placeholder ("Device 1234").
                if board and not re.match(r"^Device\b", board):
                    return board
                if len(fields) >= 3 and fields[2]:
                    # Fall back to the chip name, preferring its [bracketed] part.
                    m = re.search(r"\[([^\]]+)\]", fields[2])
                    return m.group(1) if m else fields[2]
            except Exception:
                pass
    # 3) vulkaninfo device name (RADV/AMDVLK expose the marketing name).
    for d in vulkan_devices():
        if d.get("vendor") == "amd" and d.get("name"):
            return d["name"]
    return f"AMD GPU ({base})"


def _amd_stats() -> list:
    """Per-GPU live stats for AMD cards from sysfs (amdgpu). Memory in GB."""
    import re
    cards = []
    for card in sorted(glob.glob("/sys/class/drm/card*")):
        base = os.path.basename(card)
        if not re.match(r"^card\d+$", base):
            continue
        dev = os.path.join(card, "device")

        def _read(rel):
            try:
                with open(os.path.join(dev, rel)) as f:
                    return f.read().strip()
            except OSError:
                return None
        vendor = (_read("vendor") or "").lower()
        if vendor != "0x1002":          # AMD only (NVIDIA handled via nvidia-smi)
            continue
        busy = _read("gpu_busy_percent")
        used = _read("mem_info_vram_used")
        total = _read("mem_info_vram_total")
        temp = None
        for hw in glob.glob(os.path.join(dev, "hwmon", "hwmon*", "temp1_input")):
            t = None
            try:
                with open(hw) as f:
                    t = int(f.read().strip())
            except OSError:
                t = None
            if t is not None:
                _c = t / 1000.0
                # Broken-sensor guard at the SOURCE: amdgpu's SMU can return a
                # stuck invalid reading (observed 511°C = 0x1FF invalid-ADC
                # after a GPU reset). No real GPU is ≥150°C; report the card as
                # temp-unknown (None) rather than let a bogus value reach the
                # thermal supervisor, which would pause/SIGSTOP a healthy card.
                if _c < 150.0:
                    temp = _c
                    break
        cards.append({
            "vendor": "amd", "index": int(base[4:]),
            "name": _amd_gpu_name(dev, base),
            "util": float(busy) if busy and busy.isdigit() else None,
            # GiB (binary) to MATCH _nvidia_stats (which divides nvidia-smi MiB by
            # 1024). Mixing GiB and decimal-GB made the two cards' VRAM ~7% out of
            # step in the dashboard and any cross-card total.
            "mem_used": round(int(used) / (1024 ** 3), 2) if used and used.isdigit() else None,
            "mem_total": round(int(total) / (1024 ** 3), 2) if total and total.isdigit() else None,
            "temp": temp})
    return cards


_gpu_stats_cache = {"data": None, "at": 0.0}
_GPU_STATS_TTL = 1.5


def gpu_stats() -> list:
    """Live per-card stats for EVERY physical GPU installed (vendor-agnostic):
    ``[{vendor, index, name, util%, mem_used GB, mem_total GB, temp °C, uuid}]``.

    Independent of engine ownership — nvidia-smi and sysfs report all cards
    regardless of CUDA_VISIBLE_DEVICES — so this shows the whole machine.

    Cached for a short TTL: the dashboard polls this ~every second and nvidia-smi
    can be slow on a saturated GPU, so collapse rapid repeats onto one query."""
    import time as _t
    now = _t.time()
    c = _gpu_stats_cache
    if c["data"] is not None and now - c["at"] < _GPU_STATS_TTL:
        return c["data"]
    # _amd_stats reads live GPU sysfs (mem_info/temp) which can BLOCK
    # uninterruptibly on a wedged card — bound it so a hung Radeon can't freeze
    # the health/thermal poll threads (which would stall the HEALTHY card's
    # monitoring too). On timeout, reuse the last-good AMD rows from the cache.
    _prev = c["data"] if (c := _gpu_stats_cache) and c.get("data") else []
    _prev_amd = [r for r in _prev if isinstance(r, dict) and r.get("vendor") == "amd"]
    data = _nvidia_stats() + _call_with_timeout(_amd_stats, 6, _prev_amd)
    c["data"] = data
    c["at"] = now
    return data


def engine_gpu_stats() -> list:
    """Like :func:`gpu_stats` but scoped to the cards THIS engine owns, per the
    ``CODERAI_ENGINE_GPUS`` env the front sets (comma-separated vendor keywords
    and/or NVIDIA UUIDs). Unset → all cards (single-process / legacy mode).

    Used by thermal protection so a hot GPU only pauses the engine(s) using it,
    while a hot CPU (read globally) still pauses everything."""
    cards = gpu_stats()
    raw = (os.environ.get("CODERAI_ENGINE_GPUS") or "").strip()
    if not raw:
        return cards
    sels = {s.strip() for s in raw.split(",") if s.strip()}
    return [c for c in cards
            if c.get("vendor") in sels or (c.get("uuid") and c["uuid"] in sels)]


_VALID_DPM_LEVELS = ("auto", "low", "high", "manual",
                     "profile_standard", "profile_min_sclk", "profile_min_mclk",
                     "profile_peak")


def apply_amd_dpm_force_level(level: str) -> None:
    """Write ``level`` to every amdgpu card's power_dpm_force_performance_level.

    Locking a Polaris/GCN card to 'high' (fixed top clocks) avoids the DPM
    power-state transitions that hang these cards under sustained Vulkan
    compute. Best-effort: an unprivileged container can't write root-owned
    sysfs, so on PermissionError we log the exact host command to run instead."""
    import glob
    level = (level or "").strip()
    if level not in _VALID_DPM_LEVELS:
        print(f"[dpm] ignoring invalid level '{level}' "
              f"(valid: {', '.join(_VALID_DPM_LEVELS)})", flush=True)
        return
    wrote, denied = [], []
    for card in sorted(glob.glob("/sys/class/drm/card*")):
        dev = os.path.join(card, "device")
        try:
            with open(os.path.join(dev, "vendor")) as f:
                if f.read().strip().lower() != "0x1002":   # AMD PCI vendor id
                    continue
        except OSError:
            continue
        node = os.path.join(dev, "power_dpm_force_performance_level")
        if not os.path.exists(node):
            continue
        try:
            with open(node, "w") as f:
                f.write(level + "\n")
            wrote.append(os.path.basename(card))
        except PermissionError:
            denied.append(node)
        except OSError as e:
            print(f"[dpm] failed to set {node}: {e}", flush=True)
    if wrote:
        print(f"[dpm] locked AMD power level '{level}' on {', '.join(wrote)}",
              flush=True)
    for node in denied:
        print(f"[dpm] cannot set power level (unprivileged container). Run once "
              f"on the HOST: echo {level} | sudo tee {node}", flush=True)


def card_key(vendor: str, uuid: str = "", pci: str = "") -> str:
    """Stable identifier for one physical GPU, computable identically by the front
    (which sees every card) and by an engine (which sees only its own, in llama.cpp
    device order). NVIDIA → its UUID (stable across reboots); AMD/others → PCI bus
    address. Used to key per-card VRAM caps so a cap follows the actual card."""
    vendor = _norm_vendor(vendor)
    if vendor == "nvidia" and uuid:
        u = uuid.strip()
        if u.upper().startswith("GPU-"):
            u = u[4:]
        return f"nvidia:{u}"
    if pci:
        return f"{vendor}:{pci.strip()}"
    return f"{vendor}:?"


def gpu_cards() -> list:
    """Every physical GPU on the machine as ``[{key, vendor, name}]`` for a per-card
    settings UI. Keys are stable (see :func:`card_key`) so a saved cap maps back to
    the same card on the engine side. Software rasterizers are excluded."""
    cards = []
    for g in nvidia_gpus():
        cards.append({"key": card_key("nvidia", uuid=g.get("uuid", "")),
                      "vendor": "nvidia",
                      "name": g.get("name") or "NVIDIA GPU"})
    # AMD (and any non-NVIDIA) via sysfs DRM, keyed by PCI bus address.
    for card in sorted(glob.glob("/sys/class/drm/card*")):
        base = os.path.basename(card)
        if not re.match(r"^card\d+$", base):
            continue
        dev = os.path.join(card, "device")
        try:
            with open(os.path.join(dev, "vendor")) as f:
                vid = f.read().strip().lower()
        except OSError:
            continue
        vendor = _PCI_VENDOR.get(vid)
        if not vendor or vendor == "nvidia":   # nvidia handled above via nvidia-smi
            continue
        pci = os.path.basename(os.path.realpath(dev))
        name = _amd_gpu_name(dev, base) if vendor == "amd" else f"{vendor} GPU ({base})"
        cards.append({"key": card_key(vendor, pci=pci), "vendor": vendor, "name": name})
    return cards


def summary() -> dict:
    """Detected hardware, for a 'detect engines' UI / debugging."""
    return {"nvidia": nvidia_gpus(), "vulkan": vulkan_devices(),
            "icd": {v: find_vulkan_icd(v) for v in ("nvidia", "amd", "intel")}}
