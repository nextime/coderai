from pathlib import Path
import asyncio
import json
import sys
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from codai.broker.client import BrokerClient
from codai.broker.capabilities import DEFAULT_STUDIO_ENDPOINTS
from codai.broker.config import BrokerRuntimeConfig
from codai.broker.dispatcher import execute_broker_request
from codai.broker.models import BrokerRequestEnvelope
from codai.broker.service import BrokerService


class Recorder:
    def __init__(self):
        self.events = []

    def add(self, event):
        self.events.append(event)


class FakeWebSocket:
    def __init__(self, messages):
        self._messages = list(messages)
        self.sent_messages = []
        self.closed = False

    async def recv(self):
        return self._messages.pop(0)

    async def send(self, message):
        self.sent_messages.append(message)

    async def close(self):
        self.closed = True
        return None


@pytest.mark.anyio("asyncio")
async def test_broker_client_waits_for_registered_before_register():
    websocket = FakeWebSocket(
        [
            json.dumps(
                {
                    "event": "registered",
                    "accepted": True,
                    "session_id": "session-123",
                }
            ),
            json.dumps({"request_id": "session-123", "status": "ok"}),
        ]
    )
    runtime = BrokerRuntimeConfig(
        enabled=True,
        websocket_url="wss://broker.example/ws",
        headers={"Authorization": "Bearer token"},
        registration_token="token",
        advertised_endpoint="http://localhost:8000",
    )

    with patch("codai.broker.client.websockets.connect", new=AsyncMock(return_value=websocket)) as connect_mock:
        client = BrokerClient(runtime)
        with patch(
            "codai.broker.client.build_hardware_summary",
            return_value={
                "hostname": "test-host",
                "platform": "linux",
                "gpus": [],
                "gpu_count": 0,
                "total_vram_mb": 0,
                "available_vram_mb": 0,
            },
        ):
            await client.connect_and_register()

    connect_mock.assert_awaited_once_with(
        runtime.websocket_url,
        additional_headers=runtime.headers,
        open_timeout=runtime.connect_timeout_seconds,
    )
    assert client.websocket is websocket
    assert client.session_id == "session-123"
    assert len(websocket.sent_messages) == 1

    register_message = json.loads(websocket.sent_messages[0])
    assert register_message["op"] == "register"
    assert register_message["request_id"] == "session-123"
    assert register_message["capabilities"] == register_message["payload"]["capabilities"]
    assert register_message["payload"]["registration_token"] == "token"
    assert register_message["payload"]["hardware"]["gpu_count"] == 0
    assert register_message["payload"]["gpus"] == []
    assert register_message["payload"]["capabilities"]["transports"] == ["websocket"]
    assert register_message["payload"]["studio_endpoints"] == DEFAULT_STUDIO_ENDPOINTS


@pytest.mark.anyio("asyncio")
async def test_broker_client_rejects_registered_ack_without_session_id():
    websocket = FakeWebSocket(
        [json.dumps({"event": "registered", "accepted": True, "session_id": ""})]
    )
    runtime = BrokerRuntimeConfig(
        enabled=True,
        websocket_url="wss://broker.example/ws",
        headers={"Authorization": "Bearer token"},
    )

    with patch("codai.broker.client.websockets.connect", new=AsyncMock(return_value=websocket)):
        client = BrokerClient(runtime)
        with pytest.raises(ValueError, match="broker did not accept registration"):
            await client.connect_and_register()

    assert websocket.sent_messages == []


@pytest.mark.anyio("asyncio")
async def test_broker_client_passes_connect_timeout_to_websocket_connection():
    websocket = FakeWebSocket(
        [
            json.dumps({"event": "registered", "accepted": True, "session_id": "session-123"}),
            json.dumps({"request_id": "session-123", "status": "ok"}),
        ]
    )
    runtime = BrokerRuntimeConfig(
        enabled=True,
        websocket_url="wss://broker.example/ws",
        headers={"Authorization": "Bearer token"},
        connect_timeout_seconds=17,
    )

    with patch("codai.broker.client.websockets.connect", new=AsyncMock(return_value=websocket)) as connect_mock:
        client = BrokerClient(runtime)
        await client.connect_and_register()

    connect_mock.assert_awaited_once_with(
        runtime.websocket_url,
        additional_headers=runtime.headers,
        open_timeout=runtime.connect_timeout_seconds,
    )


@pytest.mark.anyio("asyncio")
async def test_broker_client_applies_request_timeout_to_socket_reads():
    websocket = FakeWebSocket(
        [
            json.dumps({"event": "registered", "accepted": True, "session_id": "session-123"}),
            json.dumps({"request_id": "session-123", "status": "ok"}),
        ]
    )
    runtime = BrokerRuntimeConfig(enabled=True, request_timeout_seconds=23)
    timeout_calls = []

    async def fake_wait_for(awaitable, timeout):
        timeout_calls.append(timeout)
        return await awaitable

    with patch("codai.broker.client.websockets.connect", new=AsyncMock(return_value=websocket)), patch(
        "codai.broker.client.asyncio.wait_for", new=AsyncMock(side_effect=fake_wait_for)
    ):
        client = BrokerClient(runtime)
        await client.connect_and_register()

    assert timeout_calls == [runtime.request_timeout_seconds, runtime.request_timeout_seconds]


@pytest.mark.anyio("asyncio")
async def test_broker_client_rejects_failed_register_ack():
    websocket = FakeWebSocket(
        [
            json.dumps({"event": "registered", "accepted": True, "session_id": "session-123"}),
            json.dumps({"request_id": "session-123", "status": "error"}),
        ]
    )
    runtime = BrokerRuntimeConfig(enabled=True, headers={"Authorization": "Bearer token"})

    with patch("codai.broker.client.websockets.connect", new=AsyncMock(return_value=websocket)):
        client = BrokerClient(runtime)
        with pytest.raises(ValueError, match="broker did not acknowledge register message"):
            await client.connect_and_register()


@pytest.mark.anyio("asyncio")
async def test_broker_client_stores_registered_session_metadata():
    websocket = FakeWebSocket(
        [
            json.dumps(
                {
                    "event": "registered",
                    "accepted": True,
                    "session_id": "session-123",
                    "provider_id": "provider-1",
                    "client_id": "client-1",
                    "username": "alice",
                    "scope_name": "alice",
                    "owner_user_id": 42,
                    "expires_at": "2026-05-13T00:00:00Z",
                }
            ),
            json.dumps({"request_id": "session-123", "status": "ok"}),
        ]
    )
    runtime = BrokerRuntimeConfig(enabled=True, headers={"Authorization": "Bearer token"})

    with patch("codai.broker.client.websockets.connect", new=AsyncMock(return_value=websocket)):
        client = BrokerClient(runtime)
        await client.connect_and_register()

    assert client.session_metadata == {
        "session_id": "session-123",
        "provider_id": "provider-1",
        "client_id": "client-1",
        "username": "alice",
        "scope_name": "alice",
        "owner_user_id": 42,
        "expires_at": "2026-05-13T00:00:00Z",
    }


@pytest.mark.anyio("asyncio")
async def test_broker_client_run_forever_closes_registered_socket_before_reconnect():
    runtime = BrokerRuntimeConfig(
        enabled=True,
        reconnect_initial_delay_seconds=1,
        reconnect_max_delay_seconds=4,
    )
    client = BrokerClient(runtime)

    attempts = []
    sockets = []
    reconnected = asyncio.Event()

    class DisconnectingWebSocket(FakeWebSocket):
        def __init__(self):
            super().__init__([])

        async def recv(self):
            raise RuntimeError("disconnect")

    async def fake_connect_and_register():
        attempts.append("connect")
        websocket = DisconnectingWebSocket()
        sockets.append(websocket)
        client.websocket = websocket
        if len(attempts) >= 2:
            reconnected.set()

    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        await real_sleep(0)

    client.connect_and_register = AsyncMock(side_effect=fake_connect_and_register)

    with patch("codai.broker.client.asyncio.sleep", new=AsyncMock(side_effect=fake_sleep)):
        task = asyncio.create_task(client.run_forever())
        await asyncio.wait_for(reconnected.wait(), timeout=0.2)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert attempts[:2] == ["connect", "connect"]
    assert sockets[0].closed is True


@pytest.mark.anyio("asyncio")
async def test_broker_client_replies_to_heartbeat():
    websocket = FakeWebSocket([])
    runtime = BrokerRuntimeConfig(enabled=True)
    dispatcher = AsyncMock()
    client = BrokerClient(runtime, dispatcher=dispatcher)
    client.websocket = websocket

    with patch("codai.broker.client.time.time", return_value=1715443200):
        response = await client.handle_message(
            json.dumps(
                {
                    "op": "heartbeat",
                    "request_id": "req-heartbeat",
                }
            )
        )

    assert response == {
        "v": 1,
        "request_id": "req-heartbeat",
        "status": "ok",
        "event": "heartbeat",
        "payload": {"ts": 1715443200},
    }
    assert websocket.sent_messages == [json.dumps(response)]
    dispatcher.assert_not_awaited()


@pytest.mark.anyio("asyncio")
async def test_broker_client_dispatches_non_heartbeat_messages():
    websocket = FakeWebSocket([])
    runtime = BrokerRuntimeConfig(enabled=True)
    dispatcher = AsyncMock(
        return_value={
            "request_id": "req-dispatch",
            "status": "ok",
            "payload": {"status": "handled"},
        }
    )
    client = BrokerClient(runtime, dispatcher=dispatcher)
    client.websocket = websocket
    raw_message = json.dumps(
        {
            "op": "request",
            "request_id": "req-dispatch",
            "payload": {"path": "/v1/models"},
        }
    )

    response = await client.handle_message(raw_message)

    dispatcher.assert_awaited_once_with(json.loads(raw_message))
    assert response == {
        "request_id": "req-dispatch",
        "status": "ok",
        "payload": {"status": "handled"},
    }
    assert websocket.sent_messages == [json.dumps(response)]


@pytest.mark.anyio("asyncio")
async def test_broker_client_dispatches_request_messages_through_fastapi_app():
    from fastapi import FastAPI

    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat_route(payload: dict):
        return {"received": payload, "route": "chat"}

    runtime = BrokerRuntimeConfig(enabled=True)
    client = BrokerClient(runtime)
    service = BrokerService(client, app)
    client.websocket = FakeWebSocket([])
    message = {
        "op": "chat.completions",
        "request_id": "req-app",
        "payload": {"message": "hello"},
    }

    envelope = client.message_to_envelope(message)
    response = await client.handle_message(json.dumps(message))

    assert envelope == BrokerRequestEnvelope(
        request_id="req-app",
        op="chat.completions",
        method="POST",
        path="/v1/chat/completions",
        headers={},
        query={},
        payload={"message": "hello"},
        stream=False,
        content_type="application/json",
    )
    expected_response = await execute_broker_request(app, envelope)
    assert response["request_id"] == "req-app"
    assert response["status"] == "ok"
    assert response["payload"] == expected_response["payload"]
    assert response["payload"]["status_code"] == 200
    assert response["payload"]["content_type"] == "application/json"
    assert response["payload"]["body"] == '{"received":{"message":"hello"},"route":"chat"}'
    assert client.websocket.sent_messages == [json.dumps(response)]


@pytest.mark.anyio("asyncio")
async def test_broker_client_maps_proxy_op_to_endpoint_envelope():
    client = BrokerClient(BrokerRuntimeConfig(enabled=True))
    message = {
        "op": "proxy",
        "request_id": "req-proxy",
        "payload": {
            "endpoint_path": "v1/video/dub",
            "method": "POST",
            "headers": {"accept": "application/json"},
            "query_params": {"job_id": "123"},
            "body": {"prompt": "hello"},
            "stream": True,
        },
    }

    envelope = client.message_to_envelope(message)

    assert envelope == BrokerRequestEnvelope(
        request_id="req-proxy",
        op="proxy",
        method="POST",
        path="v1/video/dub",
        headers={"accept": "application/json"},
        query={"job_id": "123"},
        payload=message["payload"],
        stream=True,
        content_type="application/json",
    )


@pytest.mark.anyio("asyncio")
async def test_broker_client_maps_models_list_operation_to_internal_route():
    client = BrokerClient(BrokerRuntimeConfig(enabled=True))

    envelope = client.message_to_envelope(
        {
            "op": "models.list",
            "request_id": "req-models",
            "payload": {},
        }
    )

    assert envelope == BrokerRequestEnvelope(
        request_id="req-models",
        op="models.list",
        method="GET",
        path="/v1/models",
        headers={},
        query={},
        payload={},
        stream=False,
        content_type="application/json",
    )


@pytest.mark.anyio("asyncio")
async def test_broker_client_next_reconnect_delay_caps_at_max():
    runtime = BrokerRuntimeConfig(
        enabled=True,
        reconnect_initial_delay_seconds=2,
        reconnect_max_delay_seconds=5,
    )

    client = BrokerClient(runtime)

    assert client.next_reconnect_delay(2) == 4
    assert client.next_reconnect_delay(4) == 5
    assert client.next_reconnect_delay(5) == 5


@pytest.mark.anyio("asyncio")
async def test_broker_client_run_forever_stays_running_after_successful_connect():
    runtime = BrokerRuntimeConfig(
        enabled=True,
        reconnect_initial_delay_seconds=1,
        reconnect_max_delay_seconds=4,
    )
    client = BrokerClient(runtime)

    connected = asyncio.Event()

    async def fake_connect_and_register():
        connected.set()

    client.connect_and_register = AsyncMock(side_effect=fake_connect_and_register)

    task = asyncio.create_task(client.run_forever())
    await connected.wait()
    await asyncio.sleep(0)

    assert client.connect_and_register.await_count == 1
    assert not task.done()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.anyio("asyncio")
async def test_broker_client_run_forever_reconnects_after_disconnect():
    runtime = BrokerRuntimeConfig(
        enabled=True,
        reconnect_initial_delay_seconds=1,
        reconnect_max_delay_seconds=4,
    )
    client = BrokerClient(runtime)

    attempts = []
    sleep_calls = []
    reconnected = asyncio.Event()

    class DisconnectingWebSocket:
        def __init__(self):
            self.calls = 0

        async def recv(self):
            self.calls += 1
            raise RuntimeError("disconnect")

    async def fake_connect_and_register():
        attempts.append("connect")
        client.websocket = DisconnectingWebSocket()
        if len(attempts) >= 2:
            reconnected.set()

    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        sleep_calls.append(delay)
        await real_sleep(0)

    client.connect_and_register = AsyncMock(side_effect=fake_connect_and_register)

    with patch("codai.broker.client.asyncio.sleep", new=AsyncMock(side_effect=fake_sleep)):
        task = asyncio.create_task(client.run_forever())
        await asyncio.wait_for(reconnected.wait(), timeout=0.2)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert attempts[:2] == ["connect", "connect"]
    assert sleep_calls[0] == 1


@pytest.mark.anyio("asyncio")
async def test_broker_client_run_forever_processes_requests_concurrently():
    runtime = BrokerRuntimeConfig(enabled=True, request_timeout_seconds=1)
    client = BrokerClient(runtime)

    first_started = asyncio.Event()
    release_first = asyncio.Event()
    second_finished = asyncio.Event()

    class MessageWebSocket:
        def __init__(self):
            self.sent_messages = []
            self._messages = [
                json.dumps({"op": "chat.completions", "request_id": "req-1", "payload": {"message": "one"}}),
                json.dumps({"op": "capabilities", "request_id": "req-2", "payload": {}}),
            ]

        async def recv(self):
            if self._messages:
                return self._messages.pop(0)
            await asyncio.sleep(0)
            raise RuntimeError("disconnect")

        async def send(self, message):
            self.sent_messages.append(json.loads(message))

        async def close(self):
            return None

    websocket = MessageWebSocket()

    async def fake_connect_and_register():
        client.websocket = websocket

    async def dispatcher(message):
        if message["request_id"] == "req-1":
            first_started.set()
            await release_first.wait()
            return {"v": 1, "request_id": "req-1", "status": "ok", "payload": {"done": 1}}
        second_finished.set()
        return {"v": 1, "request_id": "req-2", "status": "ok", "payload": {"done": 2}}

    client.connect_and_register = AsyncMock(side_effect=fake_connect_and_register)
    client.dispatcher = dispatcher

    task = asyncio.create_task(client.run_forever())
    await asyncio.wait_for(first_started.wait(), timeout=0.2)
    await asyncio.wait_for(second_finished.wait(), timeout=0.2)
    release_first.set()
    await asyncio.sleep(0.05)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    sent_ids = [message["request_id"] for message in websocket.sent_messages]
    assert "req-2" in sent_ids


@pytest.mark.anyio("asyncio")
async def test_broker_client_run_forever_handles_heartbeat_before_reconnect():
    runtime = BrokerRuntimeConfig(
        enabled=True,
        reconnect_initial_delay_seconds=1,
        reconnect_max_delay_seconds=4,
    )
    client = BrokerClient(runtime)

    heartbeat = json.dumps({"op": "heartbeat", "request_id": "req-heartbeat"})
    attempts = []
    sleep_calls = []
    reconnected = asyncio.Event()

    class HeartbeatThenDisconnectWebSocket:
        def __init__(self):
            self.sent_messages = []
            self._messages = [heartbeat, RuntimeError("disconnect")]

        async def recv(self):
            message = self._messages.pop(0)
            if isinstance(message, Exception):
                raise message
            return message

        async def send(self, message):
            self.sent_messages.append(message)

    sockets = []

    async def fake_connect_and_register():
        attempts.append("connect")
        websocket = HeartbeatThenDisconnectWebSocket()
        client.websocket = websocket
        sockets.append(websocket)
        if len(attempts) >= 2:
            reconnected.set()

    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        sleep_calls.append(delay)
        await real_sleep(0)

    client.connect_and_register = AsyncMock(side_effect=fake_connect_and_register)

    with patch("codai.broker.client.time.time", return_value=1715443200), patch(
        "codai.broker.client.asyncio.sleep", new=AsyncMock(side_effect=fake_sleep)
    ):
        task = asyncio.create_task(client.run_forever())
        await asyncio.wait_for(reconnected.wait(), timeout=0.2)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert attempts[:2] == ["connect", "connect"]
    assert sleep_calls[0] == 1
    assert json.loads(sockets[0].sent_messages[0]) == {
        "v": 1,
        "request_id": "req-heartbeat",
        "status": "ok",
        "event": "heartbeat",
        "payload": {"ts": 1715443200},
    }


@pytest.mark.anyio("asyncio")
async def test_broker_service_start_and_stop_manage_background_task():
    runtime = BrokerRuntimeConfig(enabled=True)
    client = BrokerClient(runtime)

    started = asyncio.Event()

    async def fake_run_forever():
        started.set()
        await asyncio.Future()

    client.run_forever = AsyncMock(side_effect=fake_run_forever)
    service = BrokerService(client)

    service.start()
    await started.wait()

    task = service.task
    assert task is not None
    assert not task.done()

    await service.stop()

    client.run_forever.assert_awaited_once_with()
    assert task.cancelled()
    assert service.task is None


@pytest.mark.anyio("asyncio")
async def test_broker_service_start_is_idempotent_while_running():
    runtime = BrokerRuntimeConfig(enabled=True)
    client = BrokerClient(runtime)

    started = asyncio.Event()

    async def fake_run_forever():
        started.set()
        await asyncio.Future()

    client.run_forever = AsyncMock(side_effect=fake_run_forever)
    service = BrokerService(client)

    service.start()
    service.start()
    await started.wait()

    task = service.task
    assert task is not None

    await service.stop()

    client.run_forever.assert_awaited_once_with()


@pytest.mark.anyio("asyncio")
async def test_fastapi_lifespan_stops_broker_service_before_model_cleanup():
    from fastapi import FastAPI

    from codai.api.app import lifespan

    runtime = BrokerRuntimeConfig(enabled=True)
    recorder = Recorder()
    app = FastAPI(lifespan=lifespan)
    app.state.broker_runtime = runtime

    class StubBrokerService:
        def __init__(self, client, fastapi_app):
            recorder.add("broker-init")

        def start(self):
            recorder.add("broker-start")

        async def stop(self):
            recorder.add("broker-stop")

    def cleanup_multi_model_manager():
        recorder.add("multi-model-cleanup")

    def cleanup_model_manager():
        recorder.add("model-cleanup")

    with patch("codai.api.app.BrokerService", StubBrokerService), patch(
        "codai.api.app.multi_model_manager.cleanup", new=cleanup_multi_model_manager
    ), patch("codai.api.app.model_manager.cleanup", new=cleanup_model_manager), patch(
        "codai.api.app.multi_model_manager.whisper_server", new=None
    ), patch("codai.api.archive.archive_manager.run_cleanup", new=lambda: None):
        async with lifespan(app):
            recorder.add("inside-lifespan")

    assert recorder.events == [
        "broker-init",
        "broker-start",
        "inside-lifespan",
        "broker-stop",
        "multi-model-cleanup",
        "model-cleanup",
    ]
