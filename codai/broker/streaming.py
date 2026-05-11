"""Broker streaming response helpers."""

from __future__ import annotations

from typing import Any

from codai.broker.models import success_envelope


def stream_chunk_envelope(request_id: str, sequence: int, data: Any) -> dict[str, Any]:
    """Build a normalized streaming chunk envelope."""

    return success_envelope(
        request_id,
        event="stream",
        payload={"sequence": sequence, "data": data},
    )


def finalize_stream(request_id: str, total_chunks: int, elapsed_ms: float) -> dict[str, Any]:
    """Build the terminal streaming metadata envelope."""

    return success_envelope(
        request_id,
        event="stream_end",
        payload={"total_chunks": total_chunks},
        metrics={"elapsed_ms": elapsed_ms},
    )
