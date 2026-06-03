"""Broker protocol models."""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class BrokerRequestEnvelope:
    """Normalized broker request payload."""

    request_id: str
    op: str
    method: str = "GET"
    path: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    query: Dict[str, Any] = field(default_factory=dict)
    payload: Any = None
    stream: bool = False
    content_type: str = "application/json"

    def validate(self) -> None:
        """Validate required request envelope fields."""

        if not self.request_id or not isinstance(self.request_id, str):
            raise ValueError("request_id is required")
        if not self.op or not isinstance(self.op, str):
            raise ValueError("op is required")
        if self.op != "proxy" and not self.path:
            raise ValueError("path is required")
        if not self.method or not isinstance(self.method, str):
            raise ValueError("method is required")
        if self.path and not isinstance(self.path, str):
            raise ValueError("path is required")


def success_envelope(request_id: str, payload: Any, event: str | None = None, metrics: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Build a success response envelope."""

    envelope = {
        "v": 1,
        "request_id": request_id,
        "status": "ok",
        "payload": payload,
    }
    if event is not None:
        envelope["event"] = event
    if metrics is not None:
        envelope["metrics"] = metrics
    return envelope


def error_envelope(request_id: str, code: str, message: str, details: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Build an error response envelope."""

    error = {
        "code": code,
        "message": message,
    }
    if details is not None:
        error["details"] = details
    return {
        "v": 1,
        "request_id": request_id,
        "status": "error",
        "error": error,
    }
