"""Minimal broker websocket client for registration and heartbeats."""

from __future__ import annotations

import json
import asyncio
import time
import uuid
from typing import Any, Awaitable, Callable

import websockets

from codai.broker.capabilities import (
    DEFAULT_STUDIO_ENDPOINTS,
    build_capabilities_document,
    build_hardware_summary,
    build_register_message,
)
from codai.broker.models import BrokerRequestEnvelope, success_envelope

Dispatcher = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


class BrokerClient:
    def __init__(self, runtime, dispatcher: Dispatcher | None = None):
        self.runtime = runtime
        self.dispatcher = dispatcher
        self.websocket = None
        self.session_id = None

    async def connect_and_register(self):
        self.websocket = await websockets.connect(
            self.runtime.websocket_url,
            additional_headers=self.runtime.headers,
            open_timeout=self.runtime.connect_timeout_seconds,
        )

        registered_message = json.loads(
            await asyncio.wait_for(
                self.websocket.recv(),
                timeout=self.runtime.request_timeout_seconds,
            )
        )
        session_id = registered_message.get("session_id")
        if (
            registered_message.get("event") != "registered"
            or registered_message.get("accepted") is not True
            or not session_id
        ):
            raise ValueError("broker did not accept registration")

        self.session_id = session_id
        hardware = build_hardware_summary()
        capabilities = build_capabilities_document(
            studio_endpoints=DEFAULT_STUDIO_ENDPOINTS,
            hardware=hardware,
        )
        register_message = build_register_message(
            runtime=self.runtime,
            request_id=str(uuid.uuid4()),
            hardware=hardware,
            capabilities=capabilities,
            studio_endpoints=DEFAULT_STUDIO_ENDPOINTS,
        )
        await self.websocket.send(json.dumps(register_message))
        return register_message

    def message_to_envelope(self, message: dict[str, Any]) -> BrokerRequestEnvelope:
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

    def next_reconnect_delay(self, current_delay):
        return min(current_delay * 2, self.runtime.reconnect_max_delay_seconds)

    async def run_forever(self):
        delay = self.runtime.reconnect_initial_delay_seconds
        while True:
            try:
                await self.connect_and_register()
                delay = self.runtime.reconnect_initial_delay_seconds
                while True:
                    message = await asyncio.wait_for(
                        self.websocket.recv(),
                        timeout=self.runtime.request_timeout_seconds,
                    )
                    await self.handle_message(message)
            except asyncio.CancelledError:
                raise
            except Exception:
                await asyncio.sleep(delay)
                delay = self.next_reconnect_delay(delay)

    async def handle_message(self, raw_message):
        message = json.loads(raw_message)
        if message.get("op") == "heartbeat":
            response = success_envelope(
                message["request_id"],
                payload={"ts": int(time.time())},
                event="heartbeat",
            )
            await self.websocket.send(json.dumps(response))
            return response

        if self.dispatcher is not None:
            response = await self.dispatcher(message)
            await self.websocket.send(json.dumps(response))
            return response

        return None
