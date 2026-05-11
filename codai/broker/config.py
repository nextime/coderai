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
    transport: str = "websocket"
    heartbeat_interval_seconds: int = 30
    connect_timeout_seconds: int = 10
    request_timeout_seconds: int = 30
    reconnect_initial_delay_seconds: int = 1
    reconnect_max_delay_seconds: int = 60


@dataclass
class BrokerRuntimeConfig:
    """Validated broker runtime configuration."""

    enabled: bool
    websocket_url: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    advertised_endpoint: str = ""
    transport: str = "websocket"
    heartbeat_interval_seconds: int = 30
    connect_timeout_seconds: int = 10
    request_timeout_seconds: int = 30
    reconnect_initial_delay_seconds: int = 1
    reconnect_max_delay_seconds: int = 60


def build_broker_runtime_config(config: BrokerConfig) -> BrokerRuntimeConfig:
    """Validate broker configuration and build runtime websocket settings."""

    runtime = BrokerRuntimeConfig(
        enabled=config.enabled,
        advertised_endpoint=config.advertised_endpoint,
        transport=config.transport,
        heartbeat_interval_seconds=config.heartbeat_interval_seconds,
        connect_timeout_seconds=config.connect_timeout_seconds,
        request_timeout_seconds=config.request_timeout_seconds,
        reconnect_initial_delay_seconds=config.reconnect_initial_delay_seconds,
        reconnect_max_delay_seconds=config.reconnect_max_delay_seconds,
    )
    if not config.enabled:
        return runtime

    if config.scope == "global":
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
    path = f"{base_path}{suffix}" if base_path else suffix
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
        "x-coderai-advertised-endpoint": config.advertised_endpoint,
    }
    return runtime
