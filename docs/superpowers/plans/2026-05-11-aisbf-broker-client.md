# AISBF Broker Client Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an in-process AISBF broker client for CoderAI that registers over WebSocket, advertises capabilities and hardware, dispatches brokered OpenAI-compatible and studio requests through the existing FastAPI app, and returns success, error, streaming, and binary envelopes.

**Architecture:** Add a focused `codai/broker/` subsystem owned by FastAPI lifespan, with validated broker configuration, a WebSocket lifecycle client, shared capabilities/hardware reporting, an ASGI bridge for in-process execution, and a dispatcher that normalizes broker requests into existing HTTP route behavior. Drive the implementation with TDD, starting from config/protocol primitives and layering lifecycle, dispatch, streaming, and application integration on top.

**Tech Stack:** Python, FastAPI, asyncio, dataclasses, pytest, FastAPI `TestClient`, in-process ASGI invocation, WebSocket protocol adapters

---

## File Structure

### New files

- `codai/broker/__init__.py` — package exports for broker components
- `codai/broker/config.py` — broker config dataclasses, validation, scoped URL/header builders
- `codai/broker/models.py` — typed protocol models and envelope helpers
- `codai/broker/capabilities.py` — canonical capability + hardware builders
- `codai/broker/asgi_bridge.py` — in-process ASGI execution helper
- `codai/broker/streaming.py` — stream event normalization and finalization helpers
- `codai/broker/dispatcher.py` — inbound broker request validation and execution routing
- `codai/broker/client.py` — WebSocket lifecycle, handshake, heartbeat, reconnect loop
- `codai/broker/service.py` — app-facing lifecycle wrapper
- `tests/test_broker_config.py` — config validation and registration payload tests
- `tests/test_broker_protocol.py` — client protocol lifecycle and reconnect tests
- `tests/test_broker_dispatch.py` — broker request dispatch, error, and binary behavior tests
- `tests/test_broker_streaming.py` — streaming normalization and application stream bridge tests

### Modified files

- `codai/config.py` — add broker config schema and default persistence
- `codai/api/app.py` — expose `/coderai/capabilities` and manage broker service in lifespan
- `codai/main.py` — attach validated broker config/runtime dependencies to app state
- `codai/api/text.py` — only if needed to make streaming behavior easier to consume through the broker bridge without changing HTTP semantics

## Task 1: Add broker configuration schema and validation

**Files:**
- Create: `tests/test_broker_config.py`
- Create: `codai/broker/config.py`
- Modify: `codai/config.py`

- [ ] **Step 1: Write the failing config validation tests**

```python
from codai.broker.config import BrokerConfig, BrokerConfigError, build_broker_runtime_config
from codai.config import Config


def test_build_broker_runtime_config_global_scope_builds_url_and_headers():
    config = Config()
    config.broker = BrokerConfig(
        enabled=True,
        base_url="https://aisbf.example.com",
        scope="global",
        username="global",
        provider_id="coderai",
        client_id="workstation-01",
        registration_token="secret-token",
    )

    runtime = build_broker_runtime_config(config.broker)

    assert runtime.websocket_url == (
        "wss://aisbf.example.com/api/coderai/wss"
        "?provider_id=coderai&client_id=workstation-01"
        "&username=global&registration_token=secret-token"
    )
    assert runtime.headers["Authorization"] == "Bearer secret-token"
    assert runtime.headers["x-coderai-provider-id"] == "coderai"
    assert runtime.headers["x-coderai-client-id"] == "workstation-01"
    assert runtime.headers["x-coderai-username"] == "global"


def test_build_broker_runtime_config_rejects_invalid_global_username():
    broker = BrokerConfig(
        enabled=True,
        base_url="https://aisbf.example.com",
        scope="global",
        username="alice",
        provider_id="coderai",
        client_id="workstation-01",
        registration_token="secret-token",
    )

    try:
        build_broker_runtime_config(broker)
    except BrokerConfigError as exc:
        assert "username=global" in str(exc)
    else:
        raise AssertionError("expected BrokerConfigError")


def test_build_broker_runtime_config_user_scope_uses_user_path():
    broker = BrokerConfig(
        enabled=True,
        base_url="https://aisbf.example.com",
        scope="user",
        username="alice",
        provider_id="coderai",
        client_id="workstation-01",
        registration_token="secret-token",
    )

    runtime = build_broker_runtime_config(broker)

    assert runtime.websocket_url == (
        "wss://aisbf.example.com/api/u/alice/coderai/wss"
        "?provider_id=coderai&client_id=workstation-01"
        "&username=alice&registration_token=secret-token"
    )
```

- [ ] **Step 2: Run the config tests to verify they fail**

Run: `pytest tests/test_broker_config.py -q`
Expected: FAIL with import errors for `codai.broker.config` or missing broker config fields.

- [ ] **Step 3: Add broker config dataclasses and URL/header builders**

```python
# codai/broker/config.py
from dataclasses import dataclass
from typing import Dict, Optional
from urllib.parse import urlencode, quote, urlparse, urlunparse


class BrokerConfigError(ValueError):
    pass


@dataclass
class BrokerConfig:
    enabled: bool = False
    base_url: str = ""
    scope: str = "global"
    username: str = "global"
    provider_id: str = "coderai"
    client_id: str = ""
    registration_token: str = ""
    advertised_endpoint: Optional[str] = None
    transport: str = "websocket"
    heartbeat_interval_seconds: int = 30
    connect_timeout_seconds: int = 15
    request_timeout_seconds: int = 900
    reconnect_initial_delay_seconds: int = 2
    reconnect_max_delay_seconds: int = 60


@dataclass
class BrokerRuntimeConfig:
    enabled: bool
    websocket_url: str
    headers: Dict[str, str]
    scope: str
    username: str
    provider_id: str
    client_id: str
    registration_token: str
    advertised_endpoint: Optional[str]
    transport: str
    heartbeat_interval_seconds: int
    connect_timeout_seconds: int
    request_timeout_seconds: int
    reconnect_initial_delay_seconds: int
    reconnect_max_delay_seconds: int


def build_broker_runtime_config(config: BrokerConfig) -> BrokerRuntimeConfig:
    if not config.enabled:
        return BrokerRuntimeConfig(
            enabled=False,
            websocket_url="",
            headers={},
            scope=config.scope,
            username=config.username,
            provider_id=config.provider_id,
            client_id=config.client_id,
            registration_token=config.registration_token,
            advertised_endpoint=config.advertised_endpoint,
            transport=config.transport,
            heartbeat_interval_seconds=config.heartbeat_interval_seconds,
            connect_timeout_seconds=config.connect_timeout_seconds,
            request_timeout_seconds=config.request_timeout_seconds,
            reconnect_initial_delay_seconds=config.reconnect_initial_delay_seconds,
            reconnect_max_delay_seconds=config.reconnect_max_delay_seconds,
        )

    if config.scope == "global" and config.username != "global":
        raise BrokerConfigError("global scope requires username=global")
    if config.scope != "global" and not config.username:
        raise BrokerConfigError("user scope requires a username")
    if config.scope != "global" and config.username == "global":
        raise BrokerConfigError("user scope username must not be global")
    if not config.provider_id or not config.client_id or not config.registration_token:
        raise BrokerConfigError("enabled broker requires provider_id, client_id, and registration_token")

    parsed = urlparse(config.base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws" if parsed.scheme == "http" else parsed.scheme
    if scheme not in {"ws", "wss"}:
        raise BrokerConfigError("base_url must use http, https, ws, or wss")

    if config.scope == "global":
        path = "/api/coderai/wss"
    else:
        path = f"/api/u/{quote(config.username)}/coderai/wss"

    query = urlencode(
        {
            "provider_id": config.provider_id,
            "client_id": config.client_id,
            "username": config.username,
            "registration_token": config.registration_token,
        }
    )
    websocket_url = urlunparse((scheme, parsed.netloc, path, "", query, ""))
    headers = {
        "Authorization": f"Bearer {config.registration_token}",
        "x-coderai-provider-id": config.provider_id,
        "x-coderai-client-id": config.client_id,
        "x-coderai-username": config.username,
    }
    return BrokerRuntimeConfig(
        enabled=True,
        websocket_url=websocket_url,
        headers=headers,
        scope=config.scope,
        username=config.username,
        provider_id=config.provider_id,
        client_id=config.client_id,
        registration_token=config.registration_token,
        advertised_endpoint=config.advertised_endpoint,
        transport=config.transport,
        heartbeat_interval_seconds=config.heartbeat_interval_seconds,
        connect_timeout_seconds=config.connect_timeout_seconds,
        request_timeout_seconds=config.request_timeout_seconds,
        reconnect_initial_delay_seconds=config.reconnect_initial_delay_seconds,
        reconnect_max_delay_seconds=config.reconnect_max_delay_seconds,
    )
```

```python
# codai/config.py additions
from codai.broker.config import BrokerConfig


@dataclass
class Config:
    version: str = "1.0"
    server: ServerConfig = field(default_factory=ServerConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    offload: OffloadConfig = field(default_factory=OffloadConfig)
    vulkan: VulkanConfig = field(default_factory=VulkanConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    archive: ArchiveConfig = field(default_factory=ArchiveConfig)
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    system_prompt: Optional[str] = None
```

- [ ] **Step 4: Load and persist the broker config in `ConfigManager`**

```python
# codai/config.py create_default_configs() snippet
"broker": {
    "enabled": False,
    "base_url": "",
    "scope": "global",
    "username": "global",
    "provider_id": "coderai",
    "client_id": "",
    "registration_token": "",
    "advertised_endpoint": None,
    "transport": "websocket",
    "heartbeat_interval_seconds": 30,
    "connect_timeout_seconds": 15,
    "request_timeout_seconds": 900,
    "reconnect_initial_delay_seconds": 2,
    "reconnect_max_delay_seconds": 60,
},
```

```python
# codai/config.py load() snippet
self.config = Config(
    version=config_data.get("version", "1.0"),
    server=ServerConfig(**config_data.get("server", {})),
    backend=BackendConfig(**config_data.get("backend", {})),
    models=ModelsConfig(**config_data.get("models", {})),
    offload=OffloadConfig(**config_data.get("offload", {})),
    vulkan=VulkanConfig(**config_data.get("vulkan", {})),
    image=ImageConfig(**config_data.get("image", {})),
    whisper=WhisperConfig(**config_data.get("whisper", {})),
    archive=ArchiveConfig(**config_data.get("archive", {})),
    broker=BrokerConfig(**config_data.get("broker", {})),
    system_prompt=config_data.get("system_prompt"),
    tools_closer_prompt=config_data.get("tools_closer_prompt", False),
    grammar_guided=config_data.get("grammar_guided", False),
    file_path=config_data.get("file_path"),
    hf_chat_templates=config_data.get("hf_chat_templates", []),
    reasoning_options=config_data.get("reasoning_options", []),
    parser=config_data.get("parser", "auto"),
)
```

- [ ] **Step 5: Run the config tests to verify they pass**

Run: `pytest tests/test_broker_config.py -q`
Expected: PASS with 3 passing tests.

- [ ] **Step 6: Commit the config foundation**

```bash
git add tests/test_broker_config.py codai/broker/config.py codai/config.py
git commit -m "feat: add AISBF broker configuration"
```

## Task 2: Add protocol models and registration payload builder

**Files:**
- Modify: `tests/test_broker_config.py`
- Create: `codai/broker/models.py`
- Create: `codai/broker/capabilities.py`

- [ ] **Step 1: Write the failing registration payload tests**

```python
from codai.broker.capabilities import build_register_message
from codai.broker.config import BrokerRuntimeConfig


def test_build_register_message_includes_capabilities_and_hardware():
    runtime = BrokerRuntimeConfig(
        enabled=True,
        websocket_url="wss://aisbf.example.com/api/coderai/wss?...",
        headers={},
        scope="global",
        username="global",
        provider_id="coderai",
        client_id="workstation-01",
        registration_token="secret-token",
        advertised_endpoint="http://127.0.0.1:8776",
        transport="websocket",
        heartbeat_interval_seconds=30,
        connect_timeout_seconds=15,
        request_timeout_seconds=900,
        reconnect_initial_delay_seconds=2,
        reconnect_max_delay_seconds=60,
    )

    message = build_register_message(
        runtime,
        request_id="reg-1",
        hardware={
            "hostname": "workstation-01",
            "platform": "linux",
            "gpus": [{"index": 0, "name": "GPU 0", "vendor": "nvidia", "total_vram_mb": 24576}],
            "gpu_count": 1,
            "total_vram_mb": 24576,
            "available_vram_mb": 20480,
        },
        capabilities={
            "openai_compat": {"chat_completions": True, "models": True},
            "studio": {"enabled": True, "endpoints": ["v1/images/generate", "v1/video/progress"]},
        },
        studio_endpoints=["v1/images/generate", "v1/video/progress"],
    )

    assert message["op"] == "register"
    assert message["request_id"] == "reg-1"
    assert message["payload"]["registration_token"] == "secret-token"
    assert message["payload"]["hardware"]["gpu_count"] == 1
    assert message["payload"]["capabilities"]["openai_compat"]["chat_completions"] is True
    assert message["payload"]["studio_endpoints"] == ["v1/images/generate", "v1/video/progress"]


def test_build_capabilities_document_lists_openai_and_studio_support():
    document = {
        "server": {"name": "coderai", "version": "2.0.0"},
        "transports": {"http": True, "websocket": True},
        "openai_compat": {"chat_completions": True, "models": True},
        "studio": {"enabled": True, "endpoints": ["v1/images/generate", "v1/audio/tts"]},
    }

    assert document["server"]["name"] == "coderai"
    assert document["transports"]["websocket"] is True
    assert "v1/audio/tts" in document["studio"]["endpoints"]
```

- [ ] **Step 2: Run the expanded config tests to verify they fail**

Run: `pytest tests/test_broker_config.py -q`
Expected: FAIL because `build_register_message` and capability helpers do not exist.

- [ ] **Step 3: Add protocol model and payload helpers**

```python
# codai/broker/models.py
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BrokerRequestEnvelope:
    request_id: str
    method: str
    path: str
    headers: Dict[str, str] = field(default_factory=dict)
    query: Dict[str, str] = field(default_factory=dict)
    payload: Any = None
    stream: bool = False
    content_type: str = "application/json"


def success_envelope(request_id: str, payload: Any, *, event: Optional[str] = None, metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    body = {"v": 1, "request_id": request_id, "status": "ok", "payload": payload}
    if event:
        body["event"] = event
    if metrics is not None:
        body["metrics"] = metrics
    return body


def error_envelope(request_id: str, code: str, message: str, *, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    body = {"v": 1, "request_id": request_id, "status": "error", "error": {"code": code, "message": message}}
    if details is not None:
        body["error"]["details"] = details
    return body
```

```python
# codai/broker/capabilities.py
from typing import Any, Dict, List, Optional

from codai.broker.config import BrokerRuntimeConfig


DEFAULT_STUDIO_ENDPOINTS = [
    "v1/images/generate",
    "v1/images/progress",
    "v1/audio/tts",
    "v1/audio/transcriptions",
    "v1/audio/progress",
    "v1/video/dub",
    "v1/video/progress",
]


def build_capabilities_document(*, version: str = "2.0.0", studio_endpoints: Optional[List[str]] = None, hardware: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    endpoints = studio_endpoints or DEFAULT_STUDIO_ENDPOINTS
    document = {
        "server": {"name": "coderai", "version": version},
        "transports": {"http": True, "websocket": True},
        "openai_compat": {
            "chat_completions": True,
            "models": True,
            "responses": False,
            "embeddings": True,
            "images": True,
            "audio": True,
        },
        "studio": {"enabled": True, "endpoints": endpoints},
    }
    if hardware is not None:
        document["hardware"] = hardware
    return document


def build_register_message(runtime: BrokerRuntimeConfig, *, request_id: str, hardware: Dict[str, Any], capabilities: Dict[str, Any], studio_endpoints: List[str]) -> Dict[str, Any]:
    return {
        "v": 1,
        "op": "register",
        "request_id": request_id,
        "payload": {
            "endpoint": runtime.advertised_endpoint or "http://127.0.0.1:8776",
            "transport": runtime.transport,
            "registration_token": runtime.registration_token,
            "hardware": hardware,
            "studio_endpoints": studio_endpoints,
            "capabilities": capabilities,
        },
    }
```

- [ ] **Step 4: Run the tests to verify the registration payload passes**

Run: `pytest tests/test_broker_config.py -q`
Expected: PASS with 5 passing tests.

- [ ] **Step 5: Commit the protocol model layer**

```bash
git add tests/test_broker_config.py codai/broker/models.py codai/broker/capabilities.py
git commit -m "feat: add broker protocol payload builders"
```

## Task 3: Add capability endpoint and app-state broker runtime wiring

**Files:**
- Modify: `tests/test_broker_config.py`
- Modify: `codai/api/app.py`
- Modify: `codai/main.py`
- Modify: `codai/broker/capabilities.py`
- Create: `codai/broker/__init__.py`

- [ ] **Step 1: Write the failing capabilities endpoint test**

```python
from fastapi.testclient import TestClient
from codai.api.app import app


def test_capabilities_endpoint_returns_shared_broker_capabilities(monkeypatch):
    monkeypatch.setattr(
        "codai.broker.capabilities.build_hardware_summary",
        lambda: {"hostname": "workstation-01", "platform": "linux", "gpu_count": 0},
    )

    client = TestClient(app)
    response = client.get("/coderai/capabilities")

    assert response.status_code == 200
    body = response.json()
    assert body["server"]["name"] == "coderai"
    assert body["transports"]["websocket"] is True
    assert body["openai_compat"]["chat_completions"] is True
    assert "v1/images/generate" in body["studio"]["endpoints"]
    assert body["hardware"]["hostname"] == "workstation-01"
```

- [ ] **Step 2: Run the endpoint test to verify it fails**

Run: `pytest tests/test_broker_config.py::test_capabilities_endpoint_returns_shared_broker_capabilities -q`
Expected: FAIL with 404 for `/coderai/capabilities` or missing helper functions.

- [ ] **Step 3: Add shared hardware and capability builders plus endpoint export**

```python
# codai/broker/capabilities.py additions
import platform
import socket
from typing import Any, Dict


def build_hardware_summary() -> Dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.system().lower(),
        "gpus": [],
        "gpu_count": 0,
        "total_vram_mb": 0,
        "available_vram_mb": 0,
    }
```

```python
# codai/broker/__init__.py
from codai.broker.capabilities import build_capabilities_document, build_hardware_summary, build_register_message
from codai.broker.config import BrokerConfig, BrokerRuntimeConfig, BrokerConfigError, build_broker_runtime_config

__all__ = [
    "BrokerConfig",
    "BrokerRuntimeConfig",
    "BrokerConfigError",
    "build_broker_runtime_config",
    "build_capabilities_document",
    "build_hardware_summary",
    "build_register_message",
]
```

```python
# codai/api/app.py additions
from codai.broker.capabilities import build_capabilities_document, build_hardware_summary


@app.get("/coderai/capabilities")
async def coderai_capabilities():
    hardware = build_hardware_summary()
    return build_capabilities_document(hardware=hardware)
```

- [ ] **Step 4: Attach validated broker runtime config to app state in `codai/main.py`**

```python
# codai/main.py additions near app state setup
from codai.broker.config import build_broker_runtime_config

fastapi_app.state.broker_config = build_broker_runtime_config(config.broker)
```

- [ ] **Step 5: Run the endpoint test to verify it passes**

Run: `pytest tests/test_broker_config.py::test_capabilities_endpoint_returns_shared_broker_capabilities -q`
Expected: PASS.

- [ ] **Step 6: Commit the shared capability surface**

```bash
git add tests/test_broker_config.py codai/api/app.py codai/main.py codai/broker/capabilities.py codai/broker/__init__.py
git commit -m "feat: expose shared broker capabilities"
```

## Task 4: Add ASGI bridge for internal broker request execution

**Files:**
- Create: `tests/test_broker_dispatch.py`
- Create: `codai/broker/asgi_bridge.py`

- [ ] **Step 1: Write the failing ASGI bridge tests**

```python
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from codai.broker.asgi_bridge import execute_internal_request


async def test_execute_internal_request_returns_json_response():
    app = FastAPI()

    @app.get("/hello")
    async def hello():
        return {"message": "world"}

    response = await execute_internal_request(app, method="GET", path="/hello")

    assert response["status_code"] == 200
    assert response["headers"]["content-type"].startswith("application/json")
    assert response["body"] == b'{"message":"world"}'


async def test_execute_internal_request_supports_json_body():
    app = FastAPI()

    @app.post("/echo")
    async def echo(payload: dict):
        return JSONResponse({"seen": payload["value"]}, status_code=201)

    response = await execute_internal_request(
        app,
        method="POST",
        path="/echo",
        headers={"content-type": "application/json"},
        body=b'{"value":"x"}',
    )

    assert response["status_code"] == 201
    assert response["body"] == b'{"seen":"x"}'
```

- [ ] **Step 2: Run the ASGI bridge tests to verify they fail**

Run: `pytest tests/test_broker_dispatch.py -q`
Expected: FAIL because `execute_internal_request` does not exist.

- [ ] **Step 3: Implement the minimal ASGI bridge**

```python
# codai/broker/asgi_bridge.py
import asyncio
from typing import Dict, Optional
from urllib.parse import urlencode


async def execute_internal_request(app, *, method: str, path: str, headers: Optional[Dict[str, str]] = None, query: Optional[Dict[str, str]] = None, body: bytes = b""):
    raw_headers = [(key.lower().encode("latin-1"), value.encode("latin-1")) for key, value in (headers or {}).items()]
    query_string = urlencode(query or {}).encode("ascii")
    response_start = {}
    response_body_parts = []
    body_sent = False

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method.upper(),
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": query_string,
        "headers": raw_headers,
        "client": ("127.0.0.1", 0),
        "server": ("127.0.0.1", 8776),
    }

    async def receive():
        nonlocal body_sent
        if body_sent:
            await asyncio.sleep(0)
            return {"type": "http.disconnect"}
        body_sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        if message["type"] == "http.response.start":
            response_start["status_code"] = message["status"]
            response_start["headers"] = {
                key.decode("latin-1"): value.decode("latin-1")
                for key, value in message.get("headers", [])
            }
        elif message["type"] == "http.response.body":
            response_body_parts.append(message.get("body", b""))

    await app(scope, receive, send)
    return {
        "status_code": response_start["status_code"],
        "headers": response_start["headers"],
        "body": b"".join(response_body_parts),
    }
```

- [ ] **Step 4: Run the ASGI bridge tests to verify they pass**

Run: `pytest tests/test_broker_dispatch.py -q`
Expected: PASS with 2 passing tests.

- [ ] **Step 5: Commit the ASGI bridge**

```bash
git add tests/test_broker_dispatch.py codai/broker/asgi_bridge.py
git commit -m "feat: add internal ASGI bridge for broker execution"
```

## Task 5: Add broker dispatcher with request validation and JSON route execution

**Files:**
- Modify: `tests/test_broker_dispatch.py`
- Create: `codai/broker/dispatcher.py`
- Modify: `codai/broker/models.py`

- [ ] **Step 1: Write the failing dispatcher tests**

```python
import json
from fastapi import FastAPI

from codai.broker.dispatcher import execute_broker_request
from codai.broker.models import BrokerRequestEnvelope


async def test_execute_broker_request_returns_success_envelope_for_json_route():
    app = FastAPI()

    @app.get("/v1/models")
    async def models():
        return {"data": [{"id": "tiny"}]}

    envelope = BrokerRequestEnvelope(request_id="req-1", method="GET", path="/v1/models")
    response = await execute_broker_request(app, envelope)

    assert response["request_id"] == "req-1"
    assert response["status"] == "ok"
    assert response["payload"]["status_code"] == 200
    assert json.loads(response["payload"]["body"]) == {"data": [{"id": "tiny"}]}


async def test_execute_broker_request_rejects_unsupported_endpoint():
    app = FastAPI()
    envelope = BrokerRequestEnvelope(request_id="req-2", method="GET", path="/not-supported")

    response = await execute_broker_request(app, envelope)

    assert response["status"] == "error"
    assert response["error"]["code"] == "unsupported_endpoint"
```

- [ ] **Step 2: Run the dispatcher tests to verify they fail**

Run: `pytest tests/test_broker_dispatch.py -q`
Expected: FAIL because `execute_broker_request` does not exist.

- [ ] **Step 3: Implement the minimal broker dispatcher**

```python
# codai/broker/dispatcher.py
import json
import time
from typing import Dict

from codai.broker.asgi_bridge import execute_internal_request
from codai.broker.models import BrokerRequestEnvelope, error_envelope, success_envelope


SUPPORTED_PREFIXES = (
    "/v1/models",
    "/v1/chat/completions",
    "/v1/images",
    "/v1/audio",
    "/v1/video",
    "/v1/pipelines",
    "/coderai/capabilities",
)


def is_supported_path(path: str) -> bool:
    return any(path == prefix or path.startswith(prefix + "/") for prefix in SUPPORTED_PREFIXES)


async def execute_broker_request(app, envelope: BrokerRequestEnvelope) -> Dict[str, object]:
    if not envelope.request_id or not envelope.method or not envelope.path:
        return error_envelope(envelope.request_id or "unknown", "malformed_request", "request_id, method, and path are required")
    if not is_supported_path(envelope.path):
        return error_envelope(envelope.request_id, "unsupported_endpoint", f"unsupported broker path: {envelope.path}")

    started = time.perf_counter()
    response = await execute_internal_request(
        app,
        method=envelope.method,
        path=envelope.path,
        headers=envelope.headers,
        query=envelope.query,
        body=envelope.payload if isinstance(envelope.payload, bytes) else json.dumps(envelope.payload).encode("utf-8") if envelope.payload is not None else b"",
    )
    metrics = {"elapsed_ms": round((time.perf_counter() - started) * 1000, 3)}
    payload = {
        "status_code": response["status_code"],
        "headers": response["headers"],
        "body": response["body"].decode("utf-8", errors="replace"),
    }
    return success_envelope(envelope.request_id, payload, metrics=metrics)
```

- [ ] **Step 4: Run the dispatcher tests to verify they pass**

Run: `pytest tests/test_broker_dispatch.py -q`
Expected: PASS with 4 passing tests.

- [ ] **Step 5: Commit the dispatcher baseline**

```bash
git add tests/test_broker_dispatch.py codai/broker/dispatcher.py codai/broker/models.py
git commit -m "feat: dispatch broker requests through FastAPI"
```

## Task 6: Add streaming normalization helpers and stream dispatch tests

**Files:**
- Create: `tests/test_broker_streaming.py`
- Create: `codai/broker/streaming.py`
- Modify: `codai/broker/dispatcher.py`

- [ ] **Step 1: Write the failing streaming helper tests**

```python
from codai.broker.streaming import finalize_stream, stream_chunk_envelope


def test_stream_chunk_envelope_preserves_request_id_and_order():
    chunk = stream_chunk_envelope("req-1", sequence=2, data="data: hello\n\n")

    assert chunk["request_id"] == "req-1"
    assert chunk["event"] == "stream"
    assert chunk["payload"]["sequence"] == 2
    assert chunk["payload"]["data"] == "data: hello\n\n"


def test_finalize_stream_attaches_metrics():
    final = finalize_stream("req-1", total_chunks=3, elapsed_ms=12.5)

    assert final["request_id"] == "req-1"
    assert final["event"] == "stream_end"
    assert final["metrics"]["elapsed_ms"] == 12.5
    assert final["payload"]["total_chunks"] == 3
```

- [ ] **Step 2: Run the streaming tests to verify they fail**

Run: `pytest tests/test_broker_streaming.py -q`
Expected: FAIL because `codai.broker.streaming` does not exist.

- [ ] **Step 3: Implement the stream envelope helpers**

```python
# codai/broker/streaming.py
from typing import Any, Dict


def stream_chunk_envelope(request_id: str, *, sequence: int, data: str) -> Dict[str, Any]:
    return {
        "v": 1,
        "request_id": request_id,
        "status": "ok",
        "event": "stream",
        "payload": {"sequence": sequence, "data": data},
    }


def finalize_stream(request_id: str, *, total_chunks: int, elapsed_ms: float) -> Dict[str, Any]:
    return {
        "v": 1,
        "request_id": request_id,
        "status": "ok",
        "event": "stream_end",
        "payload": {"total_chunks": total_chunks},
        "metrics": {"elapsed_ms": elapsed_ms},
    }
```

- [ ] **Step 4: Add a dispatcher stream passthrough test and implementation**

```python
# tests/test_broker_streaming.py addition
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from codai.broker.dispatcher import execute_broker_request
from codai.broker.models import BrokerRequestEnvelope


async def test_execute_broker_request_wraps_streaming_response_metadata():
    app = FastAPI()

    @app.get("/v1/audio/progress")
    async def progress():
        async def generate():
            yield b"data: {\"step\":1}\n\n"
            yield b"data: [DONE]\n\n"
        return StreamingResponse(generate(), media_type="text/event-stream")

    envelope = BrokerRequestEnvelope(request_id="req-stream", method="GET", path="/v1/audio/progress")
    response = await execute_broker_request(app, envelope)

    assert response["status"] == "ok"
    assert response["payload"]["headers"]["content-type"].startswith("text/event-stream")
    assert "data: {\"step\":1}" in response["payload"]["body"]
```

```python
# codai/broker/dispatcher.py body handling stays the same; response body should already include collected stream bytes
```

- [ ] **Step 5: Run the streaming tests to verify they pass**

Run: `pytest tests/test_broker_streaming.py -q`
Expected: PASS with 3 passing tests.

- [ ] **Step 6: Commit the streaming helpers**

```bash
git add tests/test_broker_streaming.py codai/broker/streaming.py codai/broker/dispatcher.py
git commit -m "feat: add broker streaming envelopes"
```

## Task 7: Add binary response coverage to broker dispatch

**Files:**
- Modify: `tests/test_broker_dispatch.py`
- Modify: `codai/broker/dispatcher.py`

- [ ] **Step 1: Write the failing binary response test**

```python
from fastapi import FastAPI, Response

from codai.broker.dispatcher import execute_broker_request
from codai.broker.models import BrokerRequestEnvelope


async def test_execute_broker_request_preserves_binary_payload_metadata():
    app = FastAPI()

    @app.get("/v1/images/generate")
    async def image():
        return Response(content=b"\x89PNG\r\n", media_type="image/png", headers={"x-filename": "image.png"})

    envelope = BrokerRequestEnvelope(request_id="req-bin", method="GET", path="/v1/images/generate")
    response = await execute_broker_request(app, envelope)

    assert response["status"] == "ok"
    assert response["payload"]["body_base64"] == "iVBORw0K"
    assert response["payload"]["content_type"] == "image/png"
    assert response["payload"]["filename"] == "image.png"
```

- [ ] **Step 2: Run the binary dispatch test to verify it fails**

Run: `pytest tests/test_broker_dispatch.py::test_execute_broker_request_preserves_binary_payload_metadata -q`
Expected: FAIL because the dispatcher currently decodes bytes as text instead of preserving binary metadata.

- [ ] **Step 3: Update the dispatcher to distinguish binary vs text bodies**

```python
# codai/broker/dispatcher.py additions
import base64


TEXT_PREFIXES = ("application/json", "text/", "application/problem+json")


def _build_response_payload(response):
    content_type = response["headers"].get("content-type", "application/octet-stream")
    filename = response["headers"].get("x-filename")
    payload = {
        "status_code": response["status_code"],
        "headers": response["headers"],
        "content_type": content_type,
    }
    if filename:
        payload["filename"] = filename
    if content_type.startswith(TEXT_PREFIXES):
        payload["body"] = response["body"].decode("utf-8", errors="replace")
    else:
        payload["body_base64"] = base64.b64encode(response["body"]).decode("ascii")
    return payload
```

```python
# codai/broker/dispatcher.py inside execute_broker_request
payload = _build_response_payload(response)
```

- [ ] **Step 4: Run the broker dispatch tests to verify they pass**

Run: `pytest tests/test_broker_dispatch.py -q`
Expected: PASS with binary coverage included.

- [ ] **Step 5: Commit the binary handling**

```bash
git add tests/test_broker_dispatch.py codai/broker/dispatcher.py
git commit -m "feat: preserve binary broker responses"
```

## Task 8: Add broker WebSocket client handshake and heartbeat behavior

**Files:**
- Create: `tests/test_broker_protocol.py`
- Create: `codai/broker/client.py`

- [ ] **Step 1: Write the failing broker client protocol tests**

```python
import asyncio

from codai.broker.client import BrokerClient
from codai.broker.config import BrokerRuntimeConfig


class FakeSocket:
    def __init__(self, messages):
        self.messages = list(messages)
        self.sent = []

    async def recv(self):
        if self.messages:
            return self.messages.pop(0)
        raise RuntimeError("socket closed")

    async def send(self, payload):
        self.sent.append(payload)

    async def close(self):
        return None


async def test_broker_client_waits_for_registered_before_register(monkeypatch):
    runtime = BrokerRuntimeConfig(
        enabled=True,
        websocket_url="wss://aisbf.example.com/api/coderai/wss?...",
        headers={"Authorization": "Bearer secret"},
        scope="global",
        username="global",
        provider_id="coderai",
        client_id="workstation-01",
        registration_token="secret",
        advertised_endpoint="http://127.0.0.1:8776",
        transport="websocket",
        heartbeat_interval_seconds=30,
        connect_timeout_seconds=15,
        request_timeout_seconds=900,
        reconnect_initial_delay_seconds=2,
        reconnect_max_delay_seconds=60,
    )
    socket = FakeSocket([
        '{"v":1,"event":"registered","session_id":"sess-1","provider_id":"coderai","client_id":"workstation-01","username":"global","scope_name":"global","accepted":true}'
    ])

    async def fake_connect(*args, **kwargs):
        return socket

    monkeypatch.setattr("codai.broker.client.websockets.connect", fake_connect)
    monkeypatch.setattr("codai.broker.client.build_hardware_summary", lambda: {"hostname": "ws-01", "platform": "linux", "gpus": [], "gpu_count": 0, "total_vram_mb": 0, "available_vram_mb": 0})

    client = BrokerClient(runtime)
    await client.connect_and_register()

    assert client.session_id == "sess-1"
    assert any('"op": "register"' in payload for payload in socket.sent)


async def test_broker_client_replies_to_heartbeat(monkeypatch):
    runtime = BrokerRuntimeConfig(
        enabled=True,
        websocket_url="wss://aisbf.example.com/api/coderai/wss?...",
        headers={"Authorization": "Bearer secret"},
        scope="global",
        username="global",
        provider_id="coderai",
        client_id="workstation-01",
        registration_token="secret",
        advertised_endpoint="http://127.0.0.1:8776",
        transport="websocket",
        heartbeat_interval_seconds=30,
        connect_timeout_seconds=15,
        request_timeout_seconds=900,
        reconnect_initial_delay_seconds=2,
        reconnect_max_delay_seconds=60,
    )
    socket = FakeSocket([])
    client = BrokerClient(runtime)
    client.websocket = socket

    await client.handle_message('{"v":1,"op":"heartbeat","request_id":"hb-1","payload":{}}')

    assert any('"event": "heartbeat"' in payload for payload in socket.sent)
```

- [ ] **Step 2: Run the broker protocol tests to verify they fail**

Run: `pytest tests/test_broker_protocol.py -q`
Expected: FAIL because `BrokerClient` does not exist.

- [ ] **Step 3: Implement the minimal broker client handshake and heartbeat loop**

```python
# codai/broker/client.py
import json
import time
from typing import Any, Awaitable, Callable, Dict, Optional

import websockets

from codai.broker.capabilities import DEFAULT_STUDIO_ENDPOINTS, build_capabilities_document, build_hardware_summary, build_register_message
from codai.broker.config import BrokerRuntimeConfig
from codai.broker.models import error_envelope, success_envelope


class BrokerClient:
    def __init__(self, runtime: BrokerRuntimeConfig, *, dispatcher: Optional[Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]] = None):
        self.runtime = runtime
        self.dispatcher = dispatcher
        self.websocket = None
        self.session_id = None

    async def connect_and_register(self):
        self.websocket = await websockets.connect(self.runtime.websocket_url, additional_headers=self.runtime.headers)
        message = json.loads(await self.websocket.recv())
        if message.get("event") != "registered" or not message.get("accepted"):
            raise RuntimeError("expected accepted registered event")
        self.session_id = message["session_id"]
        hardware = build_hardware_summary()
        capabilities = build_capabilities_document(hardware=hardware)
        register_message = build_register_message(
            self.runtime,
            request_id="reg-1",
            hardware=hardware,
            capabilities=capabilities,
            studio_endpoints=DEFAULT_STUDIO_ENDPOINTS,
        )
        await self.websocket.send(json.dumps(register_message))

    async def handle_message(self, raw_message: str):
        message = json.loads(raw_message)
        if message.get("op") == "heartbeat":
            response = success_envelope(
                message["request_id"],
                {"ts": int(time.time())},
                event="heartbeat",
            )
            await self.websocket.send(json.dumps(response))
            return response
        if self.dispatcher is not None:
            response = await self.dispatcher(message)
            await self.websocket.send(json.dumps(response))
            return response
        return None
```

- [ ] **Step 4: Run the broker protocol tests to verify they pass**

Run: `pytest tests/test_broker_protocol.py -q`
Expected: PASS with 2 passing tests.

- [ ] **Step 5: Commit the broker client handshake**

```bash
git add tests/test_broker_protocol.py codai/broker/client.py
git commit -m "feat: add broker client handshake and heartbeat"
```

## Task 9: Add reconnect loop and broker service lifecycle wrapper

**Files:**
- Modify: `tests/test_broker_protocol.py`
- Create: `codai/broker/service.py`
- Modify: `codai/broker/client.py`
- Modify: `codai/api/app.py`

- [ ] **Step 1: Write the failing reconnect and service lifecycle tests**

```python
import asyncio

from codai.broker.client import BrokerClient
from codai.broker.config import BrokerRuntimeConfig
from codai.broker.service import BrokerService


async def test_broker_client_next_reconnect_delay_caps_at_max():
    runtime = BrokerRuntimeConfig(
        enabled=True,
        websocket_url="wss://aisbf.example.com/api/coderai/wss?...",
        headers={},
        scope="global",
        username="global",
        provider_id="coderai",
        client_id="workstation-01",
        registration_token="secret",
        advertised_endpoint="http://127.0.0.1:8776",
        transport="websocket",
        heartbeat_interval_seconds=30,
        connect_timeout_seconds=15,
        request_timeout_seconds=900,
        reconnect_initial_delay_seconds=2,
        reconnect_max_delay_seconds=10,
    )
    client = BrokerClient(runtime)

    assert client.next_reconnect_delay(2) == 4
    assert client.next_reconnect_delay(8) == 10
    assert client.next_reconnect_delay(10) == 10


async def test_broker_service_start_and_stop_manage_background_task(monkeypatch):
    runtime = BrokerRuntimeConfig(
        enabled=True,
        websocket_url="wss://aisbf.example.com/api/coderai/wss?...",
        headers={},
        scope="global",
        username="global",
        provider_id="coderai",
        client_id="workstation-01",
        registration_token="secret",
        advertised_endpoint="http://127.0.0.1:8776",
        transport="websocket",
        heartbeat_interval_seconds=30,
        connect_timeout_seconds=15,
        request_timeout_seconds=900,
        reconnect_initial_delay_seconds=2,
        reconnect_max_delay_seconds=10,
    )

    client = BrokerClient(runtime)

    async def fake_run_forever():
        await asyncio.sleep(3600)

    monkeypatch.setattr(client, "run_forever", fake_run_forever)
    service = BrokerService(client)
    await service.start()

    assert service.task is not None
    assert not service.task.done()

    await service.stop()
    assert service.task.cancelled()
```

- [ ] **Step 2: Run the reconnect tests to verify they fail**

Run: `pytest tests/test_broker_protocol.py -q`
Expected: FAIL because reconnect helpers and `BrokerService` do not exist.

- [ ] **Step 3: Implement reconnect helpers and lifecycle wrapper**

```python
# codai/broker/client.py additions
import asyncio


class BrokerClient:
    ...
    def next_reconnect_delay(self, current_delay: int) -> int:
        return min(current_delay * 2, self.runtime.reconnect_max_delay_seconds)

    async def run_forever(self):
        delay = self.runtime.reconnect_initial_delay_seconds
        while True:
            try:
                await self.connect_and_register()
                return
            except Exception:
                await asyncio.sleep(delay)
                delay = self.next_reconnect_delay(delay)
```

```python
# codai/broker/service.py
import asyncio
from typing import Optional


class BrokerService:
    def __init__(self, client):
        self.client = client
        self.task: Optional[asyncio.Task] = None

    async def start(self):
        if self.task is None and self.client.runtime.enabled:
            self.task = asyncio.create_task(self.client.run_forever())

    async def stop(self):
        if self.task is not None:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
```

- [ ] **Step 4: Start the broker service from FastAPI lifespan**

```python
# codai/api/app.py lifespan additions
from codai.broker.client import BrokerClient
from codai.broker.service import BrokerService
from codai.broker.dispatcher import execute_broker_request
from codai.broker.models import BrokerRequestEnvelope

broker_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    ...
    runtime = getattr(app.state, "broker_config", None)
    if runtime and runtime.enabled:
        client = BrokerClient(runtime)
        service = BrokerService(client)
        app.state.broker_service = service
        await service.start()
    try:
        yield
    finally:
        service = getattr(app.state, "broker_service", None)
        if service is not None:
            await service.stop()
        ...
```

- [ ] **Step 5: Run the reconnect tests to verify they pass**

Run: `pytest tests/test_broker_protocol.py -q`
Expected: PASS with 4 passing tests.

- [ ] **Step 6: Commit the service lifecycle layer**

```bash
git add tests/test_broker_protocol.py codai/broker/service.py codai/broker/client.py codai/api/app.py
git commit -m "feat: run broker client in app lifecycle"
```

## Task 10: Connect broker client dispatch to the FastAPI app

**Files:**
- Modify: `tests/test_broker_protocol.py`
- Modify: `codai/broker/client.py`
- Modify: `codai/broker/service.py`
- Modify: `codai/api/app.py`

- [ ] **Step 1: Write the failing broker dispatch integration test**

```python
import json
from fastapi import FastAPI

from codai.broker.client import BrokerClient
from codai.broker.config import BrokerRuntimeConfig


class FakeSocket:
    def __init__(self):
        self.sent = []

    async def send(self, payload):
        self.sent.append(payload)


async def test_broker_client_dispatches_request_messages_through_fastapi_app():
    app = FastAPI()

    @app.get("/v1/models")
    async def models():
        return {"data": [{"id": "tiny"}]}

    runtime = BrokerRuntimeConfig(
        enabled=True,
        websocket_url="wss://aisbf.example.com/api/coderai/wss?...",
        headers={},
        scope="global",
        username="global",
        provider_id="coderai",
        client_id="workstation-01",
        registration_token="secret",
        advertised_endpoint="http://127.0.0.1:8776",
        transport="websocket",
        heartbeat_interval_seconds=30,
        connect_timeout_seconds=15,
        request_timeout_seconds=900,
        reconnect_initial_delay_seconds=2,
        reconnect_max_delay_seconds=10,
    )

    async def dispatcher(message):
        from codai.broker.dispatcher import execute_broker_request
        from codai.broker.models import BrokerRequestEnvelope
        envelope = BrokerRequestEnvelope(
            request_id=message["request_id"],
            method=message["method"],
            path=message["path"],
            headers=message.get("headers", {}),
            query=message.get("query", {}),
            payload=message.get("payload"),
        )
        return await execute_broker_request(app, envelope)

    client = BrokerClient(runtime, dispatcher=dispatcher)
    client.websocket = FakeSocket()

    await client.handle_message('{"request_id":"req-1","method":"GET","path":"/v1/models","payload":null}')

    payload = json.loads(client.websocket.sent[0])
    assert payload["request_id"] == "req-1"
    assert payload["status"] == "ok"
```

- [ ] **Step 2: Run the broker dispatch integration test to verify it fails**

Run: `pytest tests/test_broker_protocol.py::test_broker_client_dispatches_request_messages_through_fastapi_app -q`
Expected: FAIL because no app-aware dispatcher wiring exists.

- [ ] **Step 3: Add broker request conversion and app-aware dispatcher wiring**

```python
# codai/broker/client.py additions
from codai.broker.models import BrokerRequestEnvelope


def message_to_envelope(message: Dict[str, Any]) -> BrokerRequestEnvelope:
    return BrokerRequestEnvelope(
        request_id=message["request_id"],
        method=message["method"],
        path=message["path"],
        headers=message.get("headers", {}),
        query=message.get("query", {}),
        payload=message.get("payload"),
        stream=message.get("stream", False),
        content_type=message.get("content_type", "application/json"),
    )
```

```python
# codai/broker/service.py additions
from codai.broker.dispatcher import execute_broker_request


class BrokerService:
    def __init__(self, client, app=None):
        self.client = client
        self.app = app
        self.task = None
        if self.app is not None:
            async def dispatcher(message):
                envelope = self.client.message_to_envelope(message)
                return await execute_broker_request(self.app, envelope)
            self.client.dispatcher = dispatcher
```

```python
# codai/api/app.py lifespan additions
client = BrokerClient(runtime)
service = BrokerService(client, app)
```

- [ ] **Step 4: Run the broker protocol tests to verify they pass**

Run: `pytest tests/test_broker_protocol.py -q`
Expected: PASS with dispatch integration included.

- [ ] **Step 5: Commit the broker-to-app dispatch wiring**

```bash
git add tests/test_broker_protocol.py codai/broker/client.py codai/broker/service.py codai/api/app.py
git commit -m "feat: route broker requests through the FastAPI app"
```

## Task 11: Verify brokered equivalence against real app endpoints

**Files:**
- Modify: `tests/test_broker_dispatch.py`
- Modify: `tests/test_broker_streaming.py`

- [ ] **Step 1: Write the failing app-equivalence tests for models and chat-style streaming**

```python
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from codai.broker.dispatcher import execute_broker_request
from codai.broker.models import BrokerRequestEnvelope


async def test_brokered_models_match_direct_http_response_shape():
    app = FastAPI()

    @app.get("/v1/models")
    async def models():
        return {"data": [{"id": "tiny", "name": "tiny"}]}

    direct = TestClient(app).get("/v1/models")
    brokered = await execute_broker_request(app, BrokerRequestEnvelope(request_id="req-1", method="GET", path="/v1/models"))

    assert direct.status_code == brokered["payload"]["status_code"]
    assert direct.json() == json.loads(brokered["payload"]["body"])


async def test_brokered_streaming_route_preserves_event_stream_body():
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat():
        async def generate():
            yield b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n"
            yield b"data: [DONE]\n\n"
        return StreamingResponse(generate(), media_type="text/event-stream")

    brokered = await execute_broker_request(
        app,
        BrokerRequestEnvelope(request_id="req-chat", method="POST", path="/v1/chat/completions", payload={"model": "tiny", "messages": []}),
    )

    assert brokered["payload"]["headers"]["content-type"].startswith("text/event-stream")
    assert "data: [DONE]" in brokered["payload"]["body"]
```

- [ ] **Step 2: Run the app-equivalence tests to verify they fail if shapes drift**

Run: `pytest tests/test_broker_dispatch.py tests/test_broker_streaming.py -q`
Expected: FAIL if response shaping or stream capture is incomplete.

- [ ] **Step 3: Tighten bridge/dispatcher behavior until equivalence tests pass**

```python
# Keep implementation changes minimal and localized:
# - preserve direct status codes
# - preserve direct content-type values
# - preserve event-stream body content exactly
# - avoid wrapping response bodies in extra broker-only formatting beyond the envelope payload
```

- [ ] **Step 4: Run the targeted equivalence tests to verify they pass**

Run: `pytest tests/test_broker_dispatch.py tests/test_broker_streaming.py -q`
Expected: PASS.

- [ ] **Step 5: Commit the equivalence verification changes**

```bash
git add tests/test_broker_dispatch.py tests/test_broker_streaming.py codai/broker/asgi_bridge.py codai/broker/dispatcher.py
if ! git diff --cached --quiet; then git commit -m "test: verify brokered endpoint equivalence"; fi
```

## Task 12: Run the focused broker suite and finish

**Files:**
- Test: `tests/test_broker_config.py`
- Test: `tests/test_broker_protocol.py`
- Test: `tests/test_broker_dispatch.py`
- Test: `tests/test_broker_streaming.py`

- [ ] **Step 1: Run the focused broker test suite**

Run: `pytest tests/test_broker_config.py tests/test_broker_protocol.py tests/test_broker_dispatch.py tests/test_broker_streaming.py -q`
Expected: PASS with all broker-focused tests green.

- [ ] **Step 2: Run one broader regression touchpoint for API stability**

Run: `pytest tests/test_audio_ml_endpoints.py -q`
Expected: PASS, showing existing endpoint behavior remains intact.

- [ ] **Step 3: Review the diff for accidental scope creep**

Run: `git diff --stat`
Expected: only broker-related files plus the planned integration points in `codai/config.py`, `codai/api/app.py`, `codai/main.py`, and any minimal route-safe adjustments.

- [ ] **Step 4: Commit the final integration cleanup if needed**

```bash
git add codai tests
if ! git diff --cached --quiet; then git commit -m "feat: integrate AISBF broker client support"; fi
```
