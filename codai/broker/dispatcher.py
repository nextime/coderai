"""Broker request dispatcher helpers."""

from __future__ import annotations

import json
import logging
from base64 import b64encode
from base64 import b64decode
from time import perf_counter
from typing import Any

from codai.broker.asgi_bridge import execute_internal_request
from codai.broker.models import error_envelope, success_envelope

logger = logging.getLogger(__name__)

SUPPORTED_PREFIXES = (
    "/v1/models",
    "/v1/chat/completions",
    "/v1/images",
    "/v1/audio",
    "/v1/video",
    "/v1/pipelines",
    "/v1/files",
    "/coderai/capabilities",
    "/admin",
    "/static",
)

OP_ROUTE_MAP = {
    "models.list": ("GET", "/v1/models"),
    "chat.completions": ("POST", "/v1/chat/completions"),
    "capabilities": ("GET", "/coderai/capabilities"),
}

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


async def execute_broker_request(app, envelope, executor=None):
    """Validate and execute a broker request envelope.

    ``executor`` is an ``async (method, path, headers, query, body) -> {status_code,
    headers, body}`` callable. When omitted the request is run in-process against
    ``app`` via the ASGI bridge (engine / single-process mode). The front passes its
    own executor that proxies to the right engine over HTTP."""

    logger.debug(
        "broker dispatch → op=%s request_id=%s path=%r method=%r stream=%s",
        envelope.op, envelope.request_id, envelope.path, envelope.method, envelope.stream,
    )

    if envelope.op == "proxy":
        proxy_payload = envelope.payload or {}
        endpoint_path = str(proxy_payload.get("endpoint_path") or envelope.path or "").strip()
        if endpoint_path and not endpoint_path.startswith("/"):
            endpoint_path = f"/{endpoint_path}"
        envelope.path = endpoint_path
        envelope.method = str(proxy_payload.get("method") or envelope.method or "GET").upper()
        envelope.headers = dict(proxy_payload.get("headers") or envelope.headers)
        envelope.query = dict(proxy_payload.get("query_params") or envelope.query)
        envelope.stream = bool(proxy_payload.get("stream", envelope.stream))
        if "body" in proxy_payload:
            envelope.payload = proxy_payload.get("body")
        elif proxy_payload.get("body_base64") is not None:
            envelope.payload = b64decode(proxy_payload.get("body_base64") or "")
        elif proxy_payload.get("multipart") is not None:
            envelope.payload = {"_broker_multipart": proxy_payload.get("multipart")}
        logger.debug("broker dispatch proxy resolved → %s %s", envelope.method, envelope.path)
    elif envelope.op in OP_ROUTE_MAP:
        envelope.method, envelope.path = OP_ROUTE_MAP[envelope.op]
        logger.debug("broker dispatch op mapped → %s %s", envelope.method, envelope.path)
    elif not envelope.path:
        logger.warning("broker dispatch unsupported op=%s request_id=%s", envelope.op, envelope.request_id)
        return error_envelope(
            envelope.request_id,
            code="unsupported_operation",
            message=f"Unsupported broker op: {envelope.op}",
        )

    envelope.validate()

    if not is_supported_path(envelope.path):
        logger.warning(
            "broker dispatch unsupported path=%r op=%s request_id=%s",
            envelope.path, envelope.op, envelope.request_id,
        )
        return error_envelope(
            envelope.request_id,
            code="unsupported_endpoint",
            message=f"Unsupported endpoint: {envelope.path}",
        )

    body: bytes
    if isinstance(envelope.payload, dict) and "_broker_multipart" in envelope.payload:
        from codai.broker.asgi_bridge import _build_multipart_body

        body, multipart_content_type = _build_multipart_body(envelope.payload["_broker_multipart"] or {})
        headers = dict(envelope.headers)
        headers["content-type"] = multipart_content_type
    elif isinstance(envelope.payload, (dict, list)):
        body = json.dumps(envelope.payload, separators=(",", ":")).encode("utf-8")
        headers = dict(envelope.headers)
    elif isinstance(envelope.payload, str):
        body = envelope.payload.encode("utf-8")
        headers = dict(envelope.headers)
    elif isinstance(envelope.payload, bytes):
        body = envelope.payload
        headers = dict(envelope.headers)
    elif envelope.payload is None:
        body = b""
        headers = dict(envelope.headers)
    else:
        body = json.dumps(envelope.payload, separators=(",", ":")).encode("utf-8")
        headers = dict(envelope.headers)

    if body and "content-type" not in {key.lower() for key in headers}:
        headers["content-type"] = envelope.content_type

    started_at = perf_counter()
    if executor is not None:
        response = await executor(
            method=envelope.method, path=envelope.path, headers=headers,
            query=envelope.query, body=body,
        )
    else:
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
        body_text = response["body"].decode("utf-8")
        payload["body"] = body_text
        logger.debug(
            "broker dispatch ← op=%s request_id=%s status=%d elapsed_ms=%s body_bytes=%d body_preview=%r",
            envelope.op, envelope.request_id,
            response["status_code"], elapsed_ms,
            len(response["body"]),
            body_text[:300],
        )
    else:
        payload["body_base64"] = b64encode(response["body"]).decode("ascii")
        filename = response_headers.get("x-filename")
        if filename:
            payload["filename"] = filename
        logger.debug(
            "broker dispatch ← op=%s request_id=%s status=%d elapsed_ms=%s body_bytes=%d (binary)",
            envelope.op, envelope.request_id,
            response["status_code"], elapsed_ms,
            len(response["body"]),
        )

    if envelope.stream:
        payload["stream"] = True
    result = success_envelope(
        envelope.request_id,
        payload=payload,
        metrics={"elapsed_ms": elapsed_ms},
    )
    logger.debug(
        "broker dispatch envelope_bytes=%d op=%s request_id=%s",
        len(json.dumps(result)), envelope.op, envelope.request_id,
    )
    return result
