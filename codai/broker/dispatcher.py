"""Broker request dispatcher helpers."""

from __future__ import annotations

import json
from base64 import b64encode
from time import perf_counter
from typing import Any

from codai.broker.asgi_bridge import execute_internal_request
from codai.broker.models import error_envelope, success_envelope

SUPPORTED_PREFIXES = (
    "/v1/models",
    "/v1/chat/completions",
    "/v1/images",
    "/v1/audio",
    "/v1/video",
    "/v1/pipelines",
    "/coderai/capabilities",
)

TEXT_CONTENT_TYPES = (
    "application/json",
    "application/ld+json",
    "application/problem+json",
    "application/xml",
    "application/x-www-form-urlencoded",
    "text/",
)


def is_supported_path(path: str) -> bool:
    """Return whether a broker path is supported."""

    return any(path == prefix or path.startswith(f"{prefix}/") for prefix in SUPPORTED_PREFIXES)


def _is_text_response(content_type: str | None) -> bool:
    if not content_type:
        return True

    normalized = content_type.split(";", 1)[0].strip().lower()
    return any(
        normalized == candidate or normalized.startswith(candidate)
        for candidate in TEXT_CONTENT_TYPES
    )


async def execute_broker_request(app, envelope):
    """Validate and execute a broker request envelope."""

    envelope.validate()

    if not is_supported_path(envelope.path):
        return error_envelope(
            envelope.request_id,
            code="unsupported_endpoint",
            message=f"Unsupported endpoint: {envelope.path}",
        )

    body: bytes
    if isinstance(envelope.payload, (dict, list)):
        body = json.dumps(envelope.payload, separators=(",", ":")).encode("utf-8")
    elif isinstance(envelope.payload, str):
        body = envelope.payload.encode("utf-8")
    elif isinstance(envelope.payload, bytes):
        body = envelope.payload
    elif envelope.payload is None:
        body = b""
    else:
        body = json.dumps(envelope.payload, separators=(",", ":")).encode("utf-8")

    headers = dict(envelope.headers)
    if body and "content-type" not in {key.lower() for key in headers}:
        headers["content-type"] = envelope.content_type

    started_at = perf_counter()
    response = await execute_internal_request(
        app,
        method=envelope.method,
        path=envelope.path,
        headers=headers,
        query=envelope.query,
        body=body,
    )
    elapsed_ms = round((perf_counter() - started_at) * 1000, 3)

    response_headers = response["headers"]
    if envelope.stream:
        response_headers = {
            key: value
            for key, value in response_headers.items()
            if key.lower() != "content-length"
        }

    payload: dict[str, Any] = {
        "status_code": response["status_code"],
        "headers": response_headers,
    }
    content_type = response_headers.get("content-type")
    if content_type:
        payload["content_type"] = content_type

    if _is_text_response(content_type):
        payload["body"] = response["body"].decode("utf-8")
    else:
        payload["body_base64"] = b64encode(response["body"]).decode("ascii")
        filename = response_headers.get("x-filename")
        if filename:
            payload["filename"] = filename

    if envelope.stream:
        payload["stream"] = True
    return success_envelope(
        envelope.request_id,
        payload=payload,
        metrics={"elapsed_ms": elapsed_ms},
    )
