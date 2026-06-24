"""Helpers for executing in-process ASGI HTTP requests."""

import base64
import logging
import os
import uuid
from urllib.parse import urlencode

logger = logging.getLogger(__name__)


async def execute_api_request(request, *, method, path, headers=None, body=b""):
    """Issue a sub-request to another coderai API endpoint from inside a handler.

    A feature endpoint that builds on another (e.g. /v1/characters/generate and
    /v1/environments/generate producing reference images via /v1/images/generations)
    must NOT assume the target model lives on the same engine that's handling it: in
    a multi-engine deployment the image model may be assigned to — or pinned on — a
    different engine. So:

    * Behind a front (this process is an engine — CODERAI_INTERNAL_TOKEN is set):
      route the sub-request THROUGH THE FRONT (the single public API), which picks
      the engine that owns the target model. The caller's Authorization is forwarded
      so that engine authorises it exactly as it would a direct client call — the
      handler adds no internal headers of its own.
    * Single-process (no front): dispatch in-process via execute_internal_request
      (the internal-auth middleware is a no-op when no token is configured).

    Returns the same {"status_code", "headers", "body"} dict shape either way.
    """
    tok = os.environ.get("CODERAI_INTERNAL_TOKEN")
    front_port = 0
    if tok:
        try:
            from codai.api.state import get_global_args
            front_port = int(getattr(get_global_args(), "port", 0) or 0)
        except Exception:
            front_port = 0
    if not tok or front_port <= 0:
        # Single-process (or front port unknown): serve it in-process.
        return await execute_internal_request(
            request.app, method=method, path=path, headers=headers, body=body)

    import httpx
    fwd = {k: v for k, v in (headers or {}).items()}
    # Carry the caller's credentials so the destination engine authorises the
    # sub-request the same way it would the original client request.
    auth = request.headers.get("authorization")
    if auth and not any(k.lower() == "authorization" for k in fwd):
        fwd["Authorization"] = auth
    url = f"http://127.0.0.1:{front_port}{path}"
    logger.debug("API sub-request → front %s %s (body_bytes=%d)",
                 method.upper(), url, len(body or b""))
    try:
        async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=10.0, read=None, write=None,
                                      pool=None)) as client:
            r = await client.request(method.upper(), url, headers=fwd,
                                     content=body or b"")
        return {"status_code": r.status_code, "headers": dict(r.headers),
                "body": r.content}
    except Exception as exc:
        logger.warning("API sub-request to front failed (%s); falling back "
                       "to in-process dispatch", exc)
        return await execute_internal_request(
            request.app, method=method, path=path, headers=headers, body=body)


def _build_multipart_body(multipart):
    boundary = f"coderai-broker-{uuid.uuid4().hex}"
    chunks = []

    for field in multipart.get("fields") or []:
        name = str(field.get("name") or "")
        value = str(field.get("value") or "")
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        chunks.append(value.encode("utf-8"))
        chunks.append(b"\r\n")

    for file_entry in multipart.get("files") or []:
        name = str(file_entry.get("name") or "file")
        filename = str(file_entry.get("filename") or "upload.bin")
        content_type = str(file_entry.get("content_type") or "application/octet-stream")
        data_base64 = file_entry.get("data_base64") or ""
        file_bytes = base64.b64decode(data_base64) if data_base64 else b""
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode("utf-8")
        )
        chunks.append(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
        chunks.append(file_bytes)
        chunks.append(b"\r\n")

    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


async def execute_internal_request(app, *, method, path, headers=None, query=None, body=b""):
    logger.debug(
        "ASGI bridge → %s %s query=%s body_bytes=%d",
        method.upper(), path, query or {}, len(body),
    )

    request_headers = []
    for key, value in (headers or {}).items():
        request_headers.append((key.lower().encode("latin-1"), str(value).encode("latin-1")))

    query_string = urlencode(query or {}, doseq=True).encode("ascii")

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method.upper(),
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": query_string,
        "headers": request_headers,
        "client": ("127.0.0.1", 0),
        "server": ("internal", 80),
    }

    response = {"status_code": 500, "headers": {}, "body": b""}
    request_sent = False

    async def receive():
        nonlocal request_sent
        if request_sent:
            return {"type": "http.disconnect"}
        request_sent = True
        return {
            "type": "http.request",
            "body": body,
            "more_body": False,
        }

    async def send(message):
        if message["type"] == "http.response.start":
            response["status_code"] = message["status"]
            response["headers"] = {
                key.decode("latin-1").lower(): value.decode("latin-1")
                for key, value in message.get("headers", [])
            }
        elif message["type"] == "http.response.body":
            response["body"] += message.get("body", b"")

    await app(scope, receive, send)

    body_preview = response["body"][:200].decode("utf-8", errors="replace") if response["body"] else ""
    logger.debug(
        "ASGI bridge ← %s %s status=%d content-type=%s body_bytes=%d body_preview=%r",
        method.upper(), path,
        response["status_code"],
        response["headers"].get("content-type", ""),
        len(response["body"]),
        body_preview,
    )
    return response
