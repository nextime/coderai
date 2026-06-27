"""Broker capability documents and registration payloads."""

import glob
import os
import platform
import socket
from typing import Any, Dict, Sequence

from codai.broker.config import BrokerRuntimeConfig

DEFAULT_STUDIO_ENDPOINTS = [
    "v1/audio/speech",
    "v1/audio/transcriptions",
    "v1/audio/progress",
    "v1/audio/generate",
    "v1/audio/stems",
    "v1/audio/cleanup",
    "v1/audio/voices",
    "v1/audio/voices/{name}",
    "v1/audio/voices/extract",
    "v1/audio/clone",
    "v1/images/progress",
    "v1/images/generations",
    "v1/images/edits",
    "v1/images/inpaint",
    "v1/images/upscale",
    "v1/images/depth",
    "v1/images/segment",
    "v1/images/deblur",
    "v1/images/unpixelate",
    "v1/images/outfit",
    "v1/images/to3d",
    "v1/images/from3d",
    "v1/video/progress",
    "v1/video/generations",
    "v1/video/upscale",
    "v1/video/subtitle",
    "v1/video/interpolate",
    "v1/video/dub",
    "v1/video/to3d",
    "v1/video/from3d",
]


def build_hardware_summary() -> Dict[str, Any]:
    """Build a conservative hardware summary with VRAM when available."""

    gpus = []
    total_vram_mb = 0
    available_vram_mb = 0

    # In an ENGINE (torch already loaded), report THIS engine's cards via a
    # context-free, CUDA_VISIBLE_DEVICES-scoped query — so the probe never pins a
    # ~256 MiB CUDA context (the old torch.cuda.mem_get_info path did). The nvidia
    # engine yields its NVIDIA card; the Vulkan/Radeon engine (CUDA_VISIBLE_DEVICES
    # ="") yields nothing and falls through to the whole-node path below, as before.
    # The torch-free FRONT also falls through to the whole-node path (it must stay
    # torch-free and report every physical card).
    import sys as _sys
    try:
        if "torch" in _sys.modules:
            from codai.models.gpu_query import visible_gpu_memory
            _gpus = visible_gpu_memory()
            if _gpus:
                for d in _gpus:
                    device_total_mb = int(d["total"] / (1024 * 1024))
                    gpus.append({
                        "index": d["index"],
                        "name": d["name"],
                        "total_vram_mb": device_total_mb,
                    })
                    total_vram_mb += device_total_mb
                    available_vram_mb += int(d["free"] / (1024 * 1024))
    except Exception:
        pass

    # Torch-free path (e.g. the front, which imports no torch): enumerate every
    # physical card via nvidia-smi + sysfs so VRAM is reported for the whole node.
    if not gpus:
        try:
            from codai.frontproxy.gpu_detect import gpu_stats
            for c in gpu_stats():
                total_mb = int(round((c.get("mem_total") or 0) * 1024))
                used_mb = int(round((c.get("mem_used") or 0) * 1024))
                if total_mb <= 0:
                    continue
                gpus.append({"name": c.get("name") or c.get("vendor"),
                             "total_vram_mb": total_mb})
                total_vram_mb += total_mb
                available_vram_mb += max(0, total_mb - used_mb)
        except Exception:
            pass

    if not gpus:
        for total_path in sorted(glob.glob("/sys/class/drm/card*/device/mem_info_vram_total")):
            used_path = total_path.replace("vram_total", "vram_used")
            if not os.path.exists(used_path):
                continue
            try:
                device_total_mb = int(int(open(total_path).read()) / (1024 * 1024))
                device_used_mb = int(int(open(used_path).read()) / (1024 * 1024))
                device_available_mb = max(0, device_total_mb - device_used_mb)
                card_name = os.path.basename(os.path.dirname(os.path.dirname(total_path)))
                gpus.append(
                    {
                        "name": card_name,
                        "total_vram_mb": device_total_mb,
                    }
                )
                total_vram_mb += device_total_mb
                available_vram_mb += device_available_mb
            except Exception:
                continue

    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "gpus": gpus,
        "gpu_count": len(gpus),
        "total_vram_mb": total_vram_mb,
        "available_vram_mb": available_vram_mb,
    }


def build_capabilities_document(
    version: str = "2.0.0",
    studio_endpoints: Sequence[str] | None = None,
    hardware: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build broker capability metadata for registration."""

    document = {
        "server": "codai",
        "version": version,
        "transports": ["websocket"],
        "tunnel_only": True,
        "openai_compat": {
            "chat_completions": True,
            "responses": False,
            "models": True,
        },
        "studio": {
            "supported": True,
            "endpoints": list(studio_endpoints or DEFAULT_STUDIO_ENDPOINTS),
        },
    }
    if hardware is not None:
        document["hardware"] = hardware
    return document


def build_register_message(
    runtime: BrokerRuntimeConfig,
    request_id: str,
    hardware: Dict[str, Any] | None,
    capabilities: Dict[str, Any],
    studio_endpoints: Sequence[str] | None,
) -> Dict[str, Any]:
    """Build broker registration frame."""

    registration_token = runtime.registration_token

    return {
        "v": 1,
        "op": "register",
        "request_id": request_id,
        "registration_token": registration_token,
        "capabilities": capabilities,
        "payload": {
            "endpoint": runtime.advertised_endpoint,
            "transport": runtime.transport,
            "registration_token": registration_token,
            "hardware": hardware,
            "gpus": (hardware or {}).get("gpus", []),
            "gpu_count": (hardware or {}).get("gpu_count", 0),
            "total_vram_mb": (hardware or {}).get("total_vram_mb", 0),
            "available_vram_mb": (hardware or {}).get("available_vram_mb", 0),
            "studio_endpoints": list(studio_endpoints or DEFAULT_STUDIO_ENDPOINTS),
            "capabilities": capabilities,
        },
    }
