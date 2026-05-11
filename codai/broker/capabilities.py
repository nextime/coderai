"""Broker capability documents and registration payloads."""

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
    """Build a conservative default hardware summary."""

    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "gpus": [],
        "gpu_count": 0,
        "total_vram_mb": 0,
        "available_vram_mb": 0,
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

    return {
        "v": 1,
        "op": "register",
        "request_id": request_id,
        "payload": {
            "endpoint": runtime.advertised_endpoint,
            "transport": runtime.transport,
            "registration_token": runtime.headers.get("Authorization", "").removeprefix("Bearer "),
            "hardware": hardware,
            "studio_endpoints": list(studio_endpoints or DEFAULT_STUDIO_ENDPOINTS),
            "capabilities": capabilities,
        },
    }
