# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""What the CUDA GGUF backend needs to know about the cards, with or without torch.

llama-cpp-python drives the GPU itself; the only thing the backend asked
torch for was "is there a CUDA device, how much memory has it, how much is
free". That made torch — 5 GB with its CUDA libraries — a hard requirement
of a pod image whose model never touches it, and put the llama pod image
over RunPod's size line. NVML answers the same three questions from a
1 MB package. torch is still preferred when present: it is what the local
install has, and its answers are the ones every number here was tuned on.
"""

from typing import List, Optional


def _torch():
    try:
        import torch
        if torch.cuda.is_available():
            return torch
    except Exception:
        pass
    return None


def _nvml():
    try:
        import pynvml
        pynvml.nvmlInit()
        return pynvml
    except Exception:
        return None


def cuda_available() -> bool:
    if _torch() is not None:
        return True
    nv = _nvml()
    try:
        return bool(nv and nv.nvmlDeviceGetCount() > 0)
    except Exception:
        return False


def runtime_label() -> str:
    """'CUDA 12.8' / 'ROCm 6.2' / 'CUDA (NVML, driver 570.x)' / '' — for the log line."""
    t = _torch()
    if t is not None:
        hip = getattr(t.version, "hip", None)
        return f"ROCm/HIP {hip}" if hip else f"CUDA {t.version.cuda}"
    nv = _nvml()
    if nv is None:
        return ""
    try:
        drv = nv.nvmlSystemGetDriverVersion()
        drv = drv.decode() if isinstance(drv, bytes) else str(drv)
        return f"CUDA (NVML, driver {drv})"
    except Exception:
        return "CUDA (NVML)"


def total_memory_bytes() -> List[int]:
    """Total memory of each card, in order."""
    t = _torch()
    if t is not None:
        return [t.cuda.get_device_properties(i).total_memory
                for i in range(t.cuda.device_count())]
    nv = _nvml()
    if nv is None:
        return []
    out = []
    try:
        for i in range(nv.nvmlDeviceGetCount()):
            h = nv.nvmlDeviceGetHandleByIndex(i)
            out.append(int(nv.nvmlDeviceGetMemoryInfo(h).total))
    except Exception:
        pass
    return out


def free_memory_bytes(index: int = 0) -> Optional[int]:
    """Free memory on one card, or None when nothing can say."""
    t = _torch()
    if t is not None:
        try:
            free, _total = t.cuda.mem_get_info(index)
            return int(free)
        except Exception:
            return None
    nv = _nvml()
    if nv is None:
        return None
    try:
        h = nv.nvmlDeviceGetHandleByIndex(index)
        return int(nv.nvmlDeviceGetMemoryInfo(h).free)
    except Exception:
        return None
