"""Broker streaming response helpers."""

from __future__ import annotations

from typing import Any

from codai.broker.models import success_envelope


def stream_chunk_envelope(request_id: str, sequence: int, data: Any) -> dict[str, Any]:
    """Build a streaming chunk envelope.

    Uses ``event="chunk"`` with the SSE text under ``payload.chunk`` — the shape the
    AISBF broker relay consumes (``_iter_broker_stream_chunks``: it yields
    ``payload.chunk`` for ``event in {chunk,progress,output,log,data}``)."""
    return success_envelope(
        request_id,
        event="chunk",
        payload={"sequence": sequence, "chunk": data},
    )


def finalize_stream(request_id: str, total_chunks: int, elapsed_ms: float) -> dict[str, Any]:
    """Terminal streaming envelope. ``event="done"`` ends the relay's stream loop."""
    return success_envelope(
        request_id,
        event="done",
        payload={"total_chunks": total_chunks},
        metrics={"elapsed_ms": elapsed_ms},
    )
