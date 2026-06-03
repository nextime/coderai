"""Broker configuration and runtime validation."""

from dataclasses import dataclass, field
from typing import Dict
from urllib.parse import quote, urlencode, urlsplit, urlunsplit


class BrokerConfigError(ValueError):
    """Raised when broker configuration is invalid."""


@dataclass
class BrokerConfig:
    """Persistent broker configuration."""

    enabled: bool = False
    base_url: str = ""
    scope: str = "user"
    username: str = ""
    provider_id: str = ""
    client_id: str = ""
    registration_token: str = ""
    advertised_endpoint: str = ""
    websocket_path: str = ""
    transport: str = "websocket"
    heartbeat_interval_seconds: int = 30
    connect_timeout_seconds: int = 10
    request_timeout_seconds: int = 30
    reconnect_initial_delay_seconds: int = 1
    reconnect_max_delay_seconds: int = 60
    websocket_ping_interval: int = 20


@dataclass
class BrokerRuntimeConfig:
    """Validated broker runtime configuration."""

    enabled: bool
    websocket_url: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    provider_id: str = ""
    client_id: str = ""
    username: str = ""
    registration_token: str = ""
    advertised_endpoint: str = ""
    websocket_path: str = ""
    transport: str = "websocket"
    heartbeat_interval_seconds: int = 30
    connect_timeout_seconds: int = 10
    request_timeout_seconds: int = 30
    reconnect_initial_delay_seconds: int = 1
    reconnect_max_delay_seconds: int = 60
    websocket_ping_interval: int = 20


def _join_broker_path(base_path: str, suffix: str) -> str:
    normalized_base = (base_path or "").rstrip("/")
    normalized_suffix = suffix if suffix.startswith("/") else f"/{suffix}"

    if normalized_base.endswith("/api") and normalized_suffix.startswith("/api/"):
        return f"{normalized_base}{normalized_suffix[4:]}"
    return f"{normalized_base}{normalized_suffix}" if normalized_base else normalized_suffix


def build_broker_runtime_config(config: BrokerConfig) -> BrokerRuntimeConfig:
    """Validate broker configuration and build runtime websocket settings."""

    runtime = BrokerRuntimeConfig(
        enabled=config.enabled,
        provider_id=config.provider_id,
        client_id=config.client_id,
        username=config.username,
        registration_token=config.registration_token,
        advertised_endpoint=config.advertised_endpoint,
        websocket_path=config.websocket_path,
        transport=config.transport,
        heartbeat_interval_seconds=config.heartbeat_interval_seconds,
        connect_timeout_seconds=config.connect_timeout_seconds,
        request_timeout_seconds=config.request_timeout_seconds,
        reconnect_initial_delay_seconds=config.reconnect_initial_delay_seconds,
        reconnect_max_delay_seconds=config.reconnect_max_delay_seconds,
        websocket_ping_interval=config.websocket_ping_interval,
    )
    if not config.enabled:
        return runtime

    custom_websocket_path = (config.websocket_path or "").strip()
    if custom_websocket_path:
        suffix = custom_websocket_path
        if not suffix.startswith("/"):
            suffix = f"/{suffix}"
    elif config.scope == "global":
        if config.username != "global":
            raise BrokerConfigError("global broker scope requires username 'global'")
        suffix = "/api/coderai/wss"
    elif config.scope == "user":
        if not config.username or config.username == "global":
            raise BrokerConfigError("user broker scope requires a non-global username")
        suffix = f"/api/u/{quote(config.username, safe='')}/coderai/wss"
    else:
        raise BrokerConfigError(f"unsupported broker scope: {config.scope}")

    for field_name, value in (
        ("provider_id", config.provider_id),
        ("client_id", config.client_id),
        ("registration_token", config.registration_token),
    ):
        if not value:
            raise BrokerConfigError(f"enabled broker config requires {field_name}")

    split_url = urlsplit(config.base_url)
    if split_url.scheme not in {"http", "https", "ws", "wss"}:
        raise BrokerConfigError("broker base_url must use http, https, ws, or wss")
    if not split_url.netloc:
        raise BrokerConfigError("broker base_url must include a host")

    scheme = {"http": "ws", "https": "wss"}.get(split_url.scheme, split_url.scheme)
    base_path = split_url.path.rstrip("/")
    path = _join_broker_path(base_path, suffix)
    query = urlencode(
        {
            "provider_id": config.provider_id,
            "client_id": config.client_id,
            "username": config.username,
            "registration_token": config.registration_token,
        }
    )
    runtime.websocket_url = urlunsplit((scheme, split_url.netloc, path, query, ""))
    runtime.headers = {
        "Authorization": f"Bearer {config.registration_token}",
        "x-coderai-provider-id": config.provider_id,
        "x-coderai-client-id": config.client_id,
        "x-coderai-username": config.username,
        "x-coderai-registration-token": config.registration_token,
        "x-coderai-advertised-endpoint": config.advertised_endpoint,
    }
    return runtime
