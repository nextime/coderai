"""Helpers for executing in-process ASGI HTTP requests."""

from urllib.parse import urlencode


async def execute_internal_request(app, *, method, path, headers=None, query=None, body=b""):
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
    return response
