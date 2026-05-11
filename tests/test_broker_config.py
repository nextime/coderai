from pathlib import Path
import sys
from types import SimpleNamespace

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
        "x-coderai-advertised-endpoint": "https://server.example.com",
    }
    assert runtime.transport == "websocket"
    assert runtime.heartbeat_interval_seconds == 30
    assert runtime.connect_timeout_seconds == 10
    assert runtime.request_timeout_seconds == 20
    assert runtime.reconnect_initial_delay_seconds == 1
    assert runtime.reconnect_max_delay_seconds == 60


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
        "payload": {
            "endpoint": "https://server.example.com/alice",
            "transport": "websocket",
            "registration_token": "token-123",
            "hardware": {"gpu": True, "memory_gb": 24},
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
        "payload": {
            "endpoint": "",
            "transport": "websocket",
            "registration_token": "",
            "hardware": None,
            "studio_endpoints": DEFAULT_STUDIO_ENDPOINTS,
            "capabilities": {"server": "codai"},
        },
    }


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
    assert document["studio"] == {
        "supported": True,
        "endpoints": EXPECTED_STUDIO_ENDPOINTS,
    }
    assert document["hardware"] == {"gpu": True}


def test_default_studio_endpoints_match_current_broker_served_routes():
    assert DEFAULT_STUDIO_ENDPOINTS == EXPECTED_STUDIO_ENDPOINTS


def test_build_capabilities_document_accepts_version_and_endpoint_overrides():
    document = build_capabilities_document(
        version="2.1.0",
        studio_endpoints=["v1/audio/speech"],
        hardware={"gpu_count": 0},
    )

    assert document == {
        "server": "codai",
        "version": "2.1.0",
        "transports": ["websocket"],
        "openai_compat": {
            "chat_completions": True,
            "responses": False,
            "models": True,
        },
        "studio": {
            "supported": True,
            "endpoints": ["v1/audio/speech"],
        },
        "hardware": {"gpu_count": 0},
    }


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


def test_main_wiring_falls_back_to_disabled_broker_runtime_on_invalid_config(monkeypatch, tmp_path):
    class StopMain(Exception):
        pass

    invalid_broker = BrokerConfig(
        enabled=True,
        base_url="https://broker.example.com",
        scope="global",
        username="alice",
        provider_id="provider-1",
        client_id="client-1",
        registration_token="token-123",
    )
    config = SimpleNamespace(
        models=SimpleNamespace(hf_cache_dir=None, gguf_cache_dir=None),
        archive=SimpleNamespace(enabled=False, directory="", retention="never"),
        broker=invalid_broker,
    )

    class FakeConfigManager:
        def __init__(self, config_dir):
            self.models_data = {}

        def load(self):
            return config

    fastapi_app = app
    fastapi_app.state.broker_runtime = None

    monkeypatch.setattr(
        codai_main,
        "parse_args",
        lambda: SimpleNamespace(
            config=str(tmp_path),
            list_cached_models=False,
            remove_all_models=False,
            remove_model=None,
            download_model=None,
            download_file_pattern=None,
            vulkan_list_devices=False,
            debug=False,
            dump=False,
        ),
    )
    monkeypatch.setattr(codai_main, "ConfigManager", FakeConfigManager)
    monkeypatch.setattr(codai_main.archive_manager, "configure", lambda **kwargs: None)
    monkeypatch.setattr(codai_main, "init_session_manager", lambda path: None)
    monkeypatch.setattr(codai_main, "set_config_manager", lambda config_mgr: None)
    monkeypatch.setattr("codai.api.app.app.state.broker_runtime", None, raising=False)

    def stop_after_broker_wiring(*args, **kwargs):
        raise StopMain

    monkeypatch.setattr("codai.api.state.set_global_debug", stop_after_broker_wiring)
    monkeypatch.setattr("codai.api.text.set_global_debug", lambda debug: None)

    try:
        codai_main.main()
    except StopMain:
        pass
    else:
        raise AssertionError("expected startup to stop after broker wiring")

    runtime = fastapi_app.state.broker_runtime
    assert runtime.enabled is False
    assert runtime.websocket_url == ""
    assert runtime.headers == {}
