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

"""Context-free GPU memory query.

Reporting VRAM (for /admin/api/status, the capability document, eviction
decisions, …) must NOT create a CUDA primary context. ``torch.cuda.mem_get_info``
lazily initialises the device's primary context (~256 MiB on an RTX 3090) the
first time it's called — so an engine that never loads a torch model (the
GGUF/llama.cpp engine) would still pin ~256 MiB just to answer a health poll.
That stray context can be the few hundred MB that tips a borderline torch-engine
video load (e.g. Wan2.2 A14B at 4-bit) into OOM.

NVML (pynvml / nvidia-ml-py) and ``nvidia-smi`` both read memory straight from
the driver WITHOUT a context. Query order:

  1. pynvml      — fast in-process NVML binding (no context, no fork)
  2. nvidia-smi  — subprocess fallback when the binding isn't installed
  3. torch       — ONLY if a CUDA context already exists in this process
                   (``torch.cuda.is_initialized()``), so we never create one here

The GGUF engine therefore sits at 0 MiB while idle; its real context is created
by llama.cpp when a model loads and released on unload — independent of this.
"""

import os
import shutil
import subprocess
from typing import List, Optional, Dict, Any, Set

_pynvml = None            # the imported+initialised module, or None
_pynvml_tried = False     # have we attempted import/init yet?


def _pynvml_module():
    global _pynvml, _pynvml_tried
    if _pynvml_tried:
        return _pynvml
    _pynvml_tried = True
    try:
        import pynvml  # provided by the `nvidia-ml-py` package
        pynvml.nvmlInit()
        _pynvml = pynvml
    except Exception:
        _pynvml = None
    return _pynvml


def _via_pynvml() -> Optional[List[Dict[str, Any]]]:
    p = _pynvml_module()
    if p is None:
        return None
    try:
        out: List[Dict[str, Any]] = []
        for i in range(p.nvmlDeviceGetCount()):
            h = p.nvmlDeviceGetHandleByIndex(i)
            m = p.nvmlDeviceGetMemoryInfo(h)
            name = p.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode("utf-8", "replace")
            try:
                uuid = p.nvmlDeviceGetUUID(h)
                if isinstance(uuid, bytes):
                    uuid = uuid.decode("utf-8", "replace")
            except Exception:
                uuid = None
            out.append({"index": i, "name": str(name), "uuid": uuid,
                        "free": int(m.free), "total": int(m.total)})
        return out or None
    except Exception:
        return None


def _via_nvidia_smi() -> Optional[List[Dict[str, Any]]]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        r = subprocess.run(
            [exe, "--query-gpu=index,name,uuid,memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if r.returncode != 0:
            return None
        out: List[Dict[str, Any]] = []
        for line in r.stdout.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 5:
                continue
            try:
                out.append({"index": int(parts[0]), "name": parts[1],
                            "uuid": parts[2] or None,
                            "free": int(float(parts[3])) * 1024 * 1024,
                            "total": int(float(parts[4])) * 1024 * 1024})
            except (TypeError, ValueError):
                continue
        return out or None
    except Exception:
        return None


def _via_torch_if_inited() -> Optional[List[Dict[str, Any]]]:
    """Last resort — only when a CUDA context ALREADY exists in this process, so
    we never create one here."""
    import sys
    if "torch" not in sys.modules:
        return None
    try:
        import torch
        if not (torch.cuda.is_available() and torch.cuda.is_initialized()):
            return None
        out: List[Dict[str, Any]] = []
        for i in range(torch.cuda.device_count()):
            f, t = torch.cuda.mem_get_info(i)
            try:
                uuid = "GPU-" + str(torch.cuda.get_device_properties(i).uuid)
            except Exception:
                uuid = None
            out.append({"index": i, "name": torch.cuda.get_device_name(i),
                        "uuid": uuid, "free": int(f), "total": int(t)})
        return out or None
    except Exception:
        return None


def gpu_memory() -> Optional[List[Dict[str, Any]]]:
    """Per-device VRAM for ALL physical NVIDIA cards, without creating a CUDA
    context.

    Returns a list of ``{"index", "name", "uuid", "free", "total"}`` (bytes)
    ordered by device index, or ``None`` if no NVIDIA GPU could be queried (e.g.
    non-NVIDIA host — callers fall back to their own vendor-agnostic detection).

    NOTE: NVML/nvidia-smi ignore ``CUDA_VISIBLE_DEVICES`` and report every card on
    the node. For an engine-scoped view use ``visible_gpu_memory()``."""
    return _via_pynvml() or _via_nvidia_smi() or _via_torch_if_inited()


def amd_gpu_memory() -> Optional[List[Dict[str, Any]]]:
    """Per-device VRAM for amdgpu cards via sysfs — driver-free, no context.

    Reads ``/sys/class/drm/card*/device/mem_info_vram_{total,used}`` (present only
    for amdgpu). Returns ``{index, name, uuid, free, total}`` (bytes) keyed by the
    sysfs card index, or ``None`` when no amdgpu card is present. Used by the
    Vulkan/Radeon engine (which has ``CUDA_VISIBLE_DEVICES=""``, so the NVML/CUDA
    queries report nothing for it)."""
    import glob
    import re
    out: List[Dict[str, Any]] = []
    for total_path in sorted(glob.glob('/sys/class/drm/card*/device/mem_info_vram_total')):
        used_path = total_path.replace('mem_info_vram_total', 'mem_info_vram_used')
        try:
            with open(total_path) as f:
                total = int(f.read().strip())
        except Exception:
            continue
        try:
            with open(used_path) as f:
                used = int(f.read().strip())
        except Exception:
            used = 0
        m = re.search(r'card(\d+)', total_path)
        idx = int(m.group(1)) if m else len(out)
        out.append({"index": idx, "name": "AMD GPU", "uuid": None,
                    "free": max(0, total - used), "total": total})
    return out or None


def _visible_tokens() -> Optional[Set[str]]:
    """Parse ``CUDA_VISIBLE_DEVICES`` into a set of selector tokens (physical
    index strings and/or ``GPU-<uuid>`` strings).

      - unset            → None  (no scoping; all cards visible)
      - "" (empty)       → set() (NO CUDA cards visible — e.g. a Vulkan/Radeon
                                   engine; the front sets this explicitly)
      - "0,1" / UUIDs    → {"0", "1"} / {"GPU-…", …}
    """
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    return {tok.strip() for tok in raw.split(",") if tok.strip() != ""}


def visible_gpu_memory() -> Optional[List[Dict[str, Any]]]:
    """Like :func:`gpu_memory` but restricted to the cards THIS process may use,
    per ``CUDA_VISIBLE_DEVICES`` (which NVML/nvidia-smi otherwise ignore).

    Matches each card by physical index OR UUID, so it works whether the front
    pinned the engine by index ("0,1") or by UUID ("GPU-…"). An explicit empty
    ``CUDA_VISIBLE_DEVICES`` yields ``[]`` (a Radeon/Vulkan engine sees no CUDA
    card); an unset variable yields all cards."""
    gpus = gpu_memory()
    if not gpus:
        return gpus
    want = _visible_tokens()
    if want is None:
        return gpus
    if not want:
        return []
    return [g for g in gpus
            if str(g.get("index")) in want or (g.get("uuid") and g["uuid"] in want)]
