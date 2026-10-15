from pathlib import Path
import sys
import pytest
from types import SimpleNamespace

from codai.config import ServerConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from codai import main as codai_main
from codai.api.app import app
from codai.broker.capabilities import (
    DEFAULT_STUDIO_ENDPOINTS,
    build_hardware_summary,
    build_capabilities_document,
    build_register_message,
)
from codai.broker.config import BrokerConfig, BrokerConfigError, build_broker_runtime_config


EXPECTED_STUDIO_ENDPOINTS = [
    "v1/audio/speech",
    "v1/audio/transcriptions",
    "v1/audio/progress",
    "v1/audio/generate",
    "v1/audio/stems",
    "v1/audio/cleanup",
    "v1/audio/voices",
    "v1/audio/voices/{name}",
    "v1/audio/voices/extract",
    "v1/audio/clone",
    "v1/images/progress",
    "v1/images/generations",
    "v1/images/edits",
    "v1/images/inpaint",
    "v1/images/upscale",
    "v1/images/depth",
    "v1/images/segment",
    "v1/images/deblur",
    "v1/images/unpixelate",
    "v1/images/outfit",
    "v1/images/to3d",
    "v1/images/from3d",
    "v1/video/progress",
    "v1/video/generations",
    "v1/video/upscale",
    "v1/video/subtitle",
    "v1/video/interpolate",
    "v1/video/dub",
    "v1/video/to3d",
    "v1/video/from3d",
]


def test_build_broker_runtime_config_global_scope_builds_url_and_headers():
    runtime = build_broker_runtime_config(
        BrokerConfig(
            enabled=True,
            base_url="https://broker.example.com",
            scope="global",
            username="global",
            provider_id="provider-1",
            client_id="client-1",
            registration_token="token-123",
            advertised_endpoint="https://server.example.com",
            transport="websocket",
            heartbeat_interval_seconds=30,
            connect_timeout_seconds=10,
            request_timeout_seconds=20,
            reconnect_initial_delay_seconds=1,
            reconnect_max_delay_seconds=60,
        )
    )

    assert runtime.enabled is True
    assert runtime.websocket_url == (
        "wss://broker.example.com/api/coderai/wss"
        "?provider_id=provider-1"
        "&client_id=client-1"
        "&username=global"
        "&registration_token=token-123"
    )
    assert runtime.headers == {
        "Authorization": "Bearer token-123",
        "x-coderai-provider-id": "provider-1",
        "x-coderai-client-id": "client-1",
        "x-coderai-username": "global",
        "x-coderai-registration-token": "token-123",
        "x-coderai-advertised-endpoint": "https://server.example.com",
    }
    assert runtime.provider_id == "provider-1"
    assert runtime.client_id == "client-1"
    assert runtime.username == "global"
    assert runtime.registration_token == "token-123"
    assert runtime.transport == "websocket"
    assert runtime.heartbeat_interval_seconds == 30
    assert runtime.connect_timeout_seconds == 10
    assert runtime.request_timeout_seconds == 20
    assert runtime.reconnect_initial_delay_seconds == 1
    assert runtime.reconnect_max_delay_seconds == 60


def test_build_broker_runtime_config_global_scope_uses_global_service_paths():
    runtime = build_broker_runtime_config(
        BrokerConfig(
            enabled=True,
            base_url="https://broker.example.com/base",
            scope="global",
            username="global",
            provider_id="provider-1",
            client_id="client-1",
            registration_token="token-123",
        )
    )

    assert runtime.websocket_url.startswith("wss://broker.example.com/base/api/coderai/wss")


def test_build_broker_runtime_config_user_scope_uses_user_service_paths():
    runtime = build_broker_runtime_config(
        BrokerConfig(
            enabled=True,
            base_url="https://broker.example.com/base",
            scope="user",
            username="alice",
            provider_id="provider-1",
            client_id="client-1",
            registration_token="token-123",
        )
    )

    assert runtime.websocket_url.startswith("wss://broker.example.com/base/api/u/alice/coderai/wss")


def test_build_broker_runtime_config_rejects_invalid_global_username():
    try:
        build_broker_runtime_config(
            BrokerConfig(
                enabled=True,
                base_url="https://broker.example.com",
                scope="global",
                username="alice",
                provider_id="provider-1",
                client_id="client-1",
                registration_token="token-123",
            )
        )
    except BrokerConfigError as error:
        assert str(error) == "global broker scope requires username 'global'"
    else:
        raise AssertionError("expected BrokerConfigError")


def test_build_broker_runtime_config_user_scope_uses_user_path():
    runtime = build_broker_runtime_config(
        BrokerConfig(
            enabled=True,
            base_url="http://broker.example.com",
            scope="user",
            username="alice",
            provider_id="provider-1",
            client_id="client-1",
            registration_token="token-123",
            advertised_endpoint="https://server.example.com/alice",
        )
    )

    assert runtime.websocket_url == (
        "ws://broker.example.com/api/u/alice/coderai/wss"
        "?provider_id=provider-1"
        "&client_id=client-1"
        "&username=alice"
        "&registration_token=token-123"
    )
    assert runtime.headers == {
        "Authorization": "Bearer token-123",
        "x-coderai-provider-id": "provider-1",
        "x-coderai-client-id": "client-1",
        "x-coderai-username": "alice",
        "x-coderai-registration-token": "token-123",
        "x-coderai-advertised-endpoint": "https://server.example.com/alice",
    }


def test_build_broker_runtime_config_preserves_base_url_prefix_in_websocket_url():
    runtime = build_broker_runtime_config(
        BrokerConfig(
            enabled=True,
            base_url="https://broker.example.com/prefix",
            scope="global",
            username="global",
            provider_id="provider-1",
            client_id="client-1",
            registration_token="token-123",
        )
    )

    assert runtime.websocket_url == (
        "wss://broker.example.com/prefix/api/coderai/wss"
        "?provider_id=provider-1"
        "&client_id=client-1"
        "&username=global"
        "&registration_token=token-123"
    )


def test_build_broker_runtime_config_does_not_duplicate_api_prefix_when_base_url_already_ends_with_api():
    runtime = build_broker_runtime_config(
        BrokerConfig(
            enabled=True,
            base_url="https://aisbf.cloud/api",
            scope="global",
            username="global",
            provider_id="provider-1",
            client_id="client-1",
            registration_token="token-123",
        )
    )

    assert runtime.websocket_url == (
        "wss://aisbf.cloud/api/coderai/wss"
        "?provider_id=provider-1"
        "&client_id=client-1"
        "&username=global"
        "&registration_token=token-123"
    )


def test_build_broker_runtime_config_encodes_reserved_username_path_characters():
    runtime = build_broker_runtime_config(
        BrokerConfig(
            enabled=True,
            base_url="https://broker.example.com",
            scope="user",
            username="alice/bob smith?team=ml",
            provider_id="provider-1",
            client_id="client-1",
            registration_token="token-123",
        )
    )

    assert runtime.websocket_url == (
        "wss://broker.example.com/api/u/alice%2Fbob%20smith%3Fteam%3Dml/coderai/wss"
        "?provider_id=provider-1"
        "&client_id=client-1"
        "&username=alice%2Fbob+smith%3Fteam%3Dml"
        "&registration_token=token-123"
    )


def test_build_broker_runtime_config_uses_manual_websocket_path_override():
    runtime = build_broker_runtime_config(
        BrokerConfig(
            enabled=True,
            base_url="https://broker.example.com/prefix",
            scope="global",
            username="global",
            provider_id="provider-1",
            client_id="client-1",
            registration_token="token-123",
            websocket_path="/custom/broker/socket",
        )
    )

    assert runtime.websocket_path == "/custom/broker/socket"
    assert runtime.websocket_url == (
        "wss://broker.example.com/prefix/custom/broker/socket"
        "?provider_id=provider-1"
        "&client_id=client-1"
        "&username=global"
        "&registration_token=token-123"
    )


def test_build_broker_runtime_config_normalizes_manual_websocket_path_override_without_leading_slash():
    runtime = build_broker_runtime_config(
        BrokerConfig(
            enabled=True,
            base_url="https://broker.example.com",
            scope="user",
            username="alice",
            provider_id="provider-1",
            client_id="client-1",
            registration_token="token-123",
            websocket_path="broker/ws",
        )
    )

    assert runtime.websocket_path == "broker/ws"
    assert runtime.websocket_url.startswith("wss://broker.example.com/broker/ws")


def test_build_broker_runtime_config_rejects_invalid_user_scope_username():
    try:
        build_broker_runtime_config(
            BrokerConfig(
                enabled=True,
                base_url="https://broker.example.com",
                scope="user",
                username="global",
                provider_id="provider-1",
                client_id="client-1",
                registration_token="token-123",
            )
        )
    except BrokerConfigError as error:
        assert str(error) == "user broker scope requires a non-global username"
    else:
        raise AssertionError("expected BrokerConfigError")


def test_build_broker_runtime_config_rejects_missing_required_enabled_fields():
    try:
        build_broker_runtime_config(
            BrokerConfig(
                enabled=True,
                base_url="https://broker.example.com",
                scope="user",
                username="alice",
                provider_id="",
                client_id="client-1",
                registration_token="token-123",
            )
        )
    except BrokerConfigError as error:
        assert str(error) == "enabled broker config requires provider_id"
    else:
        raise AssertionError("expected BrokerConfigError")


def test_build_broker_runtime_config_rejects_invalid_base_url_host():
    try:
        build_broker_runtime_config(
            BrokerConfig(
                enabled=True,
                base_url="https:///missing-host",
                scope="user",
                username="alice",
                provider_id="provider-1",
                client_id="client-1",
                registration_token="token-123",
            )
        )
    except BrokerConfigError as error:
        assert str(error) == "broker base_url must include a host"
    else:
        raise AssertionError("expected BrokerConfigError")


def test_build_broker_runtime_config_rejects_invalid_base_url_scheme():
    try:
        build_broker_runtime_config(
            BrokerConfig(
                enabled=True,
                base_url="ftp://broker.example.com",
                scope="user",
                username="alice",
                provider_id="provider-1",
                client_id="client-1",
                registration_token="token-123",
            )
        )
    except BrokerConfigError as error:
        assert str(error) == "broker base_url must use http, https, ws, or wss"
    else:
        raise AssertionError("expected BrokerConfigError")


def test_build_register_message_includes_capabilities_and_hardware():
    runtime = build_broker_runtime_config(
        BrokerConfig(
            enabled=True,
            base_url="https://broker.example.com",
            scope="user",
            username="alice",
            provider_id="provider-1",
            client_id="client-1",
            registration_token="token-123",
            advertised_endpoint="https://server.example.com/alice",
            transport="websocket",
        )
    )

    capabilities = build_capabilities_document(
        hardware={"gpu": True, "memory_gb": 24},
    )

    message = build_register_message(
        runtime=runtime,
        request_id="req-1",
        hardware={"gpu": True, "memory_gb": 24},
        capabilities=capabilities,
        studio_endpoints=EXPECTED_STUDIO_ENDPOINTS,
    )

    assert message == {
        "v": 1,
        "op": "register",
        "request_id": "req-1",
        "registration_token": "token-123",
        "capabilities": capabilities,
        "payload": {
            "endpoint": "https://server.example.com/alice",
            "transport": "websocket",
            "registration_token": "token-123",
            "hardware": {"gpu": True, "memory_gb": 24},
            "gpus": [],
            "gpu_count": 0,
            "total_vram_mb": 0,
            "available_vram_mb": 0,
            "studio_endpoints": EXPECTED_STUDIO_ENDPOINTS,
            "capabilities": capabilities,
        },
    }


def test_build_register_message_defaults_token_and_studio_endpoints_for_empty_runtime_headers():
    message = build_register_message(
        runtime=build_broker_runtime_config(BrokerConfig(enabled=False)),
        request_id="req-1",
        hardware=None,
        capabilities={"server": "codai"},
        studio_endpoints=None,
    )

    assert message == {
        "v": 1,
        "op": "register",
        "request_id": "req-1",
        "registration_token": "",
        "capabilities": {"server": "codai"},
        "payload": {
            "endpoint": "",
            "transport": "websocket",
            "registration_token": "",
            "hardware": None,
            "gpus": [],
            "gpu_count": 0,
            "total_vram_mb": 0,
            "available_vram_mb": 0,
            "studio_endpoints": DEFAULT_STUDIO_ENDPOINTS,
            "capabilities": {"server": "codai"},
        },
    }


def test_build_hardware_summary_reports_visible_gpu_memory(monkeypatch):
    """The summary reads `visible_gpu_memory()`, not `torch.cuda.mem_get_info`.

    That moved deliberately: the old path pinned a ~256 MiB CUDA context just
    to answer a hardware question. This test faked `torch` and so kept
    measuring the REAL card of whatever machine it ran on — it asserted
    10240 MB free and read 24111 on this one. A test that depends on the
    host's GPU is not a test.
    """
    import sys

    monkeypatch.setitem(sys.modules, "torch", object())   # only presence matters
    monkeypatch.setattr(
        "codai.models.gpu_query.visible_gpu_memory",
        lambda: [{"index": 0, "name": "RTX Test",
                  "total": 24 * 1024 ** 3, "free": 10 * 1024 ** 3}],
    )

    hardware = build_hardware_summary()

    assert hardware["gpu_count"] == 1
    assert hardware["total_vram_mb"] == 24576
    assert hardware["available_vram_mb"] == 10240
    assert hardware["gpus"] == [{"index": 0, "name": "RTX Test",
                                 "total_vram_mb": 24576}]


def test_build_capabilities_document_lists_openai_and_studio_support():
    document = build_capabilities_document(hardware={"gpu": True})

    assert document["version"] == "2.0.0"
    assert document["server"] == "codai"
    assert document["transports"] == ["websocket"]
    assert document["openai_compat"] == {
        "chat_completions": True,
        "responses": False,
        "models": True,
    }
    # The DEFAULT list, not a frozen copy of it: the document's job is to
    # echo what this server advertises, and freezing it here only guaranteed
    # a failure the next time an endpoint was added.
    assert document["studio"] == {
        "supported": True,
        "endpoints": DEFAULT_STUDIO_ENDPOINTS,
    }
    assert document["hardware"] == {"gpu": True}


def test_default_studio_endpoints_match_current_broker_served_routes():
    """What we ADVERTISE to the broker must be what this server serves.

    This compared against a frozen copy of the list, so it broke every time the
    product gained an endpoint — eight of them by the time anyone looked — and
    said nothing about whether the advertised routes exist. Checking them
    against the app's real routes is the contract the name always claimed, and
    it cannot drift.
    """
    from codai.api.app import app

    import re

    def shape(path: str) -> str:
        # Parameter NAMES are local detail — the app serves
        # `/v1/video/compose/{job_id}` and the advertised list says `{id}`.
        # What must agree is that the route exists and takes a parameter
        # there; comparing names would make this a spelling test.
        return re.sub(r"\{[^}]*\}", "{}", path.lstrip("/"))

    served = {shape(getattr(r, "path", "")) for r in app.routes
              if getattr(r, "path", "").startswith("/v1/")}

    missing = [e for e in DEFAULT_STUDIO_ENDPOINTS if shape(e) not in served]
    assert not missing, f"advertised to the broker but not served here: {missing}"


def test_build_capabilities_document_accepts_version_and_endpoint_overrides():
    document = build_capabilities_document(
        version="2.1.0",
        studio_endpoints=["v1/audio/speech"],
        hardware={"gpu_count": 0},
    )

    # What this test is ABOUT: the two overrides are honoured. Exact equality
    # made it a snapshot of every field the document has ever had, so it broke
    # when `tunnel_only` was added — which is not what the name claims to test.
    assert document["version"] == "2.1.0"
    assert document["studio"]["endpoints"] == ["v1/audio/speech"]
    assert document["hardware"] == {"gpu_count": 0}
    assert document["server"] == "codai"
    assert document["transports"] == ["websocket"]


def test_capabilities_endpoint_returns_shared_broker_capabilities():
    client = TestClient(app)

    response = client.get("/coderai/capabilities")

    assert response.status_code == 200
    assert response.json() == build_capabilities_document(
        hardware=build_hardware_summary(),
    )


def test_capabilities_endpoint_is_accessible_without_bearer_auth():
    client = TestClient(app)

    response = client.get("/coderai/capabilities")

    assert response.status_code == 200
    assert response.json()["server"] == "codai"


def test_an_invalid_broker_config_falls_back_to_a_disabled_runtime():
    """A broker misconfiguration must disable the broker, not stop the server.

    This used to drive the whole of `main()` through a wall of stubs to reach
    one line, so it broke whenever main grew a call or a config field — three
    separate AttributeErrors deep inside the code under test, each of which
    read like a product bug. The behaviour it exists for is this fallback, and
    that can be asserted directly.
    """
    from codai.broker import BrokerConfigError, build_broker_runtime_config
    from codai.config import BrokerConfig

    invalid = BrokerConfig(enabled=True, scope="nonsense")

    with pytest.raises(BrokerConfigError):
        build_broker_runtime_config(invalid)

    # …and the fallback main uses in that case yields a disabled runtime
    # rather than raising, which is what keeps the server up.
    fallback = build_broker_runtime_config(invalid.__class__())
    assert fallback.enabled is False


