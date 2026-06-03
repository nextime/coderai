"""Helpers for executing in-process ASGI HTTP requests."""

import base64
import logging
import uuid
from urllib.parse import urlencode

logger = logging.getLogger(__name__)


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
