"""Broker protocol models."""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class BrokerRequestEnvelope:
    """Normalized broker request payload."""

    request_id: str
    method: str
    path: str
    headers: Dict[str, str] = field(default_factory=dict)
    query: Dict[str, Any] = field(default_factory=dict)
    payload: Any = None
    stream: bool = False
    content_type: str = "application/json"

    def validate(self) -> None:
        """Validate required request envelope fields."""

        if not self.request_id or not isinstance(self.request_id, str):
            raise ValueError("request_id is required")
        if not self.method or not isinstance(self.method, str):
            raise ValueError("method is required")
        if not self.path or not isinstance(self.path, str):
            raise ValueError("path is required")


def success_envelope(request_id: str, payload: Any, event: str | None = None, metrics: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Build a success response envelope."""

    envelope = {
        "request_id": request_id,
        "ok": True,
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
        "request_id": request_id,
        "ok": False,
        "error": error,
    }
