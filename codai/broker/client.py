"""Minimal broker websocket client for registration and heartbeats."""

from __future__ import annotations

import json
import asyncio
import logging
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
from codai.broker.dispatcher import OP_ROUTE_MAP
from codai.broker.models import BrokerRequestEnvelope, success_envelope, error_envelope
from codai.broker.streaming import stream_chunk_envelope, finalize_stream

Dispatcher = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]

logger = logging.getLogger(__name__)


class BrokerClient:
    def __init__(self, runtime, dispatcher: Dispatcher | None = None):
        self.runtime = runtime
        self.dispatcher = dispatcher
        # Optional async-generator dispatcher for streaming requests: yields SSE
        # chunks which we relay as ``chunk`` envelopes + a terminal ``done``.
        self.stream_dispatcher = None
        self.websocket = None
        self.session_id = None
        self.session_metadata: dict[str, Any] = {}
        self._heartbeat_task: asyncio.Task | None = None
        self._inflight_tasks: set[asyncio.Task] = set()

    async def connect_and_register(self):
        logger.info(
            "CoderAI broker connecting url=%s provider=%s client=%s username=%s",
            self.runtime.websocket_url,
            self.runtime.provider_id,
            self.runtime.client_id,
            self.runtime.username,
        )
        ping_interval = self.runtime.websocket_ping_interval or 20
        self.websocket = await websockets.connect(
            self.runtime.websocket_url,
            additional_headers=self.runtime.headers,
            open_timeout=self.runtime.connect_timeout_seconds,
            ping_interval=ping_interval,
            ping_timeout=max(ping_interval * 2, 60),
        )

        # Client speaks first: send op=register before waiting for any server message.
        hardware = build_hardware_summary()
        capabilities = build_capabilities_document(
            studio_endpoints=DEFAULT_STUDIO_ENDPOINTS,
            hardware=hardware,
        )
        request_id = f"reg-{uuid.uuid4().hex}"
        register_message = build_register_message(
            runtime=self.runtime,
            request_id=request_id,
            hardware=hardware,
            capabilities=capabilities,
            studio_endpoints=DEFAULT_STUDIO_ENDPOINTS,
        )
        logger.debug("broker → sending op=register request_id=%s", request_id)
        await self.websocket.send(json.dumps(register_message))

        # Server responds with event=registered containing the assigned session_id.
        raw_registered = await asyncio.wait_for(
            self.websocket.recv(),
            timeout=self.runtime.request_timeout_seconds,
        )
        registered_message = json.loads(raw_registered)
        logger.debug(
            "broker ← received event=%s status=%s session_id=%s bytes=%d",
            registered_message.get("event"),
            registered_message.get("status"),
            registered_message.get("session_id"),
            len(raw_registered),
        )
        session_id = registered_message.get("session_id")
        if (
            registered_message.get("event") != "registered"
            or registered_message.get("status") != "ok"
            or not session_id
        ):
            logger.error(
                "broker registration rejected: event=%s status=%s message=%r",
                registered_message.get("event"),
                registered_message.get("status"),
                raw_registered[:200],
            )
            raise ValueError("broker did not accept registration")

        self.session_id = session_id
        self.session_metadata = {
            key: registered_message.get(key)
            for key in (
                "session_id",
                "provider_id",
                "client_id",
                "username",
                "scope_name",
                "owner_user_id",
                "expires_at",
            )
        }
        _gpu_names = ", ".join(
            g.get("name", "?") for g in (hardware.get("gpus") or [])
        ) or "none"
        logger.info(
            "CoderAI broker registered provider=%s client=%s session_id=%s scope=%s"
            " gpu_count=%s gpus=[%s] total_vram_mb=%s available_vram_mb=%s",
            self.session_metadata.get("provider_id"),
            self.session_metadata.get("client_id"),
            session_id,
            self.session_metadata.get("scope_name"),
            hardware.get("gpu_count"),
            _gpu_names,
            hardware.get("total_vram_mb"),
            hardware.get("available_vram_mb"),
        )
        return register_message

    def message_to_envelope(self, message: dict[str, Any]) -> BrokerRequestEnvelope:
        payload = message.get("payload") or {}
        op = message.get("op") or "proxy"
        method = message.get("method") or "GET"
        path = message.get("path") or ""
        headers = message.get("headers", {})
        query = message.get("query", {})
        stream = message.get("stream", False)
        content_type = message.get("content_type", "application/json")

        if op == "proxy":
            method = payload.get("method", method)
            path = payload.get("endpoint_path", path)
            headers = payload.get("headers", headers)
            query = payload.get("query_params", query)
            stream = payload.get("stream", stream)
            content_type = payload.get("content_type", content_type)
        elif op in OP_ROUTE_MAP:
            method, path = OP_ROUTE_MAP[op]

        return BrokerRequestEnvelope(
            request_id=message["request_id"],
            op=op,
            method=str(method).upper(),
            path=str(path),
            headers=headers,
            query=query,
            payload=payload,
            stream=bool(stream),
            content_type=content_type,
        )

    def next_reconnect_delay(self, current_delay):
        return min(current_delay * 2, self.runtime.reconnect_max_delay_seconds)

    async def run_forever(self):
        delay = self.runtime.reconnect_initial_delay_seconds
        while True:
            try:
                await self.connect_and_register()
                self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
                delay = self.runtime.reconnect_initial_delay_seconds
                # Watchdog must outlast one full heartbeat cycle so a quiet connection
                # (no inflight requests, only periodic heartbeats) never times out.
                recv_timeout = max(
                    self.runtime.request_timeout_seconds,
                    self.runtime.heartbeat_interval_seconds * 3,
                )
                while True:
                    message = await asyncio.wait_for(
                        self.websocket.recv(),
                        timeout=recv_timeout,
                    )
                    parsed = json.loads(message)
                    op = parsed.get("op")
                    event = parsed.get("event")
                    request_id = parsed.get("request_id")
                    logger.debug(
                        "broker ← recv op=%s event=%s request_id=%s bytes=%d",
                        op, event, request_id, len(message),
                    )
                    # Server-initiated heartbeat (op=heartbeat): respond in-line.
                    if op == "heartbeat":
                        logger.debug("broker heartbeat ping request_id=%s", request_id)
                        await self.handle_message(parsed)
                        continue
                    # Heartbeat reply from server (status=ok, event=heartbeat, no op):
                    # just discard — the client initiated it, no response needed.
                    if event == "heartbeat" and not op:
                        logger.debug("broker heartbeat pong discarded request_id=%s", request_id)
                        continue
                    # Re-register ACK (event=registered, no op): discard — sent in
                    # response to our notify_models_updated() re-register push.
                    if event == "registered" and not op:
                        logger.debug("broker re-register ack discarded request_id=%s", request_id)
                        continue
                    logger.debug(
                        "broker dispatching op=%s request_id=%s as async task (inflight=%d)",
                        op, request_id, len(self._inflight_tasks),
                    )
                    task = asyncio.create_task(self.handle_message(parsed))
                    self._inflight_tasks.add(task)
                    task.add_done_callback(self._inflight_tasks.discard)
            except asyncio.CancelledError:
                await self._stop_heartbeat_task()
                await self._cancel_inflight_tasks()
                await self._close_websocket()
                raise
            except Exception as error:
                logger.warning(
                    "CoderAI broker connection lost provider=%s client=%s url=%s reconnect_in=%ss error=%s",
                    self.runtime.provider_id,
                    self.runtime.client_id,
                    self.runtime.websocket_url,
                    delay,
                    error,
                )
                await self._stop_heartbeat_task()
                await self._cancel_inflight_tasks()
                await self._close_websocket()
                await asyncio.sleep(delay)
                delay = self.next_reconnect_delay(delay)
            finally:
                await self._stop_heartbeat_task()

    async def _close_websocket(self):
        websocket = self.websocket
        self.websocket = None
        if websocket is None:
            return
        try:
            await websocket.close()
        except Exception:
            pass

    async def _stop_heartbeat_task(self):
        if self._heartbeat_task is None:
            return
        task = self._heartbeat_task
        self._heartbeat_task = None
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _cancel_inflight_tasks(self):
        if not self._inflight_tasks:
            return
        tasks = list(self._inflight_tasks)
        self._inflight_tasks.clear()
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

    async def _send_keepalives(
        self, request_id: str, interval: float = 30.0, estimated_timeout: float = 300.0
    ):
        """Send periodic pending keepalive messages for a long-running request.
        Cancelled by the caller when the request finishes."""
        try:
            while True:
                await asyncio.sleep(interval)
                if self.websocket is None:
                    break
                try:
                    await self.websocket.send(json.dumps({
                        "v": 1,
                        "event": "pending",
                        "status": "pending",
                        "request_id": request_id,
                        "payload": {"message": "processing", "estimated_timeout": estimated_timeout},
                    }))
                    logger.debug("broker → pending keepalive request_id=%s", request_id)
                except Exception:
                    break
        except asyncio.CancelledError:
            pass

    async def _heartbeat_loop(self):
        started_at = time.monotonic()
        while True:
            await asyncio.sleep(self.runtime.heartbeat_interval_seconds)
            if self.websocket is None:
                continue
            uptime = int(time.monotonic() - started_at)
            available_vram_mb = build_hardware_summary().get("available_vram_mb", 0)
            heartbeat = {
                "v": 1,
                "op": "heartbeat",
                "request_id": f"hb-{uuid.uuid4()}",
                "payload": {
                    "uptime": uptime,
                    "hardware": {
                        "available_vram_mb": available_vram_mb,
                    },
                },
            }
            logger.debug(
                "broker → heartbeat uptime=%ds available_vram_mb=%s",
                uptime, available_vram_mb,
            )
            await self.websocket.send(json.dumps(heartbeat))

    async def notify_models_updated(self) -> bool:
        """Send a re-register to AISBF so it refreshes its model cache.

        Returns True if the message was sent, False if not connected.
        AISBF handles op=register on a live session by calling _broker_refresh_models(),
        which re-fetches /v1/models from this coderai instance.
        """
        if self.websocket is None:
            return False
        try:
            hardware = build_hardware_summary()
            capabilities = build_capabilities_document(
                studio_endpoints=DEFAULT_STUDIO_ENDPOINTS,
                hardware=hardware,
            )
            register_message = build_register_message(
                runtime=self.runtime,
                request_id=f"rereg-{uuid.uuid4().hex}",
                hardware=hardware,
                capabilities=capabilities,
                studio_endpoints=DEFAULT_STUDIO_ENDPOINTS,
            )
            await self.websocket.send(json.dumps(register_message))
            logger.info(
                "broker → re-register (models updated) provider=%s client=%s",
                self.runtime.provider_id,
                self.runtime.client_id,
            )
            return True
        except Exception as exc:
            logger.warning("broker notify_models_updated failed: %s", exc)
            return False

    async def handle_message(self, raw_message):
        message = json.loads(raw_message) if isinstance(raw_message, str) else raw_message
        if message.get("op") == "heartbeat":
            response = success_envelope(
                message["request_id"],
                payload={"ts": int(time.time())},
                event="heartbeat",
            )
            await self.websocket.send(json.dumps(response))
            return response

        if self.dispatcher is not None:
            op = message.get("op")
            request_id = message.get("request_id")
            payload = message.get("payload") or {}
            endpoint_path = payload.get("endpoint_path") or message.get("path") or OP_ROUTE_MAP.get(op, (None, None))[1]
            method = payload.get("method") or message.get("method") or OP_ROUTE_MAP.get(op, ("GET", None))[0]
            stream = bool(payload.get("stream") or message.get("stream"))
            logger.info(
                "CoderAI broker received request op=%s request_id=%s provider=%s client=%s endpoint=%s method=%s stream=%s",
                op, request_id,
                message.get("provider_id") or self.session_metadata.get("provider_id"),
                message.get("client_id") or self.session_metadata.get("client_id"),
                endpoint_path, method, stream,
            )
            # Diagnostic: surface the inbound max_tokens so we can tell whether a
            # remote client's value (e.g. Hermes) survives the aisbf relay into our
            # payload, or is dropped before it reaches us. Handles the chat body
            # being the payload directly or nested under body / body_base64.
            try:
                _b = payload
                if isinstance(payload, dict):
                    if isinstance(payload.get("body"), dict):
                        _b = payload["body"]
                    elif isinstance(payload.get("body"), str):
                        try:
                            _b = json.loads(payload["body"])
                        except Exception:
                            _b = {}
                    elif payload.get("body_base64"):
                        import base64 as _b64
                        _b = json.loads(_b64.b64decode(
                            payload["body_base64"]).decode("utf-8", "replace") or "{}")
                _mt = _b.get("max_tokens") if isinstance(_b, dict) else None
                _mct = _b.get("max_completion_tokens") if isinstance(_b, dict) else None
                logger.info(
                    "CoderAI broker inbound request_id=%s max_tokens=%s "
                    "max_completion_tokens=%s body_keys=%s",
                    request_id, _mt, _mct,
                    sorted(_b.keys()) if isinstance(_b, dict) else type(_b).__name__,
                )
            except Exception as _probe_exc:
                logger.info("CoderAI broker inbound max_tokens probe failed: %s",
                            _probe_exc)
            # Send an immediate "pending" acknowledgment so the broker side extends
            # its deadline, then keep sending periodic keepalives while the request
            # runs (e.g. during model download).  The keepalive task is cancelled
            # as soon as we have a final response to send.
            keepalive_task: asyncio.Task | None = None
            if request_id:
                try:
                    await self.websocket.send(json.dumps({
                        "v": 1,
                        "event": "pending",
                        "status": "pending",
                        "request_id": request_id,
                        "payload": {"message": "processing", "estimated_timeout": 300},
                    }))
                except Exception:
                    pass
                keepalive_task = asyncio.create_task(
                    self._send_keepalives(request_id, interval=30.0, estimated_timeout=300.0)
                )

            # Streaming path: relay engine SSE chunks as `chunk` envelopes + a
            # terminal `done`, so the broker side streams tokens to the client
            # instead of buffering the whole reply.
            if stream and self.stream_dispatcher is not None and request_id:
                seq = 0
                _t_start = time.monotonic()
                try:
                    async for chunk in self.stream_dispatcher(message):
                        if not chunk:
                            continue
                        seq += 1
                        await self.websocket.send(json.dumps(
                            stream_chunk_envelope(request_id, seq, chunk)))
                    await self.websocket.send(json.dumps(finalize_stream(
                        request_id, seq,
                        round((time.monotonic() - _t_start) * 1000, 3))))
                    logger.info(
                        "CoderAI broker streamed request_id=%s chunks=%d", request_id, seq)
                except Exception as exc:
                    logger.error("CoderAI broker stream error request_id=%s: %s",
                                 request_id, exc, exc_info=True)
                    try:
                        await self.websocket.send(json.dumps(error_envelope(
                            request_id, code="stream_error", message=str(exc))))
                    except Exception:
                        pass
                finally:
                    if keepalive_task is not None:
                        keepalive_task.cancel()
                        try:
                            await keepalive_task
                        except asyncio.CancelledError:
                            pass
                return None

            try:
                response = await self.dispatcher(message)
            except Exception as exc:
                logger.error(
                    "CoderAI broker dispatcher error op=%s request_id=%s: %s",
                    op, request_id, exc, exc_info=True,
                )
                raise
            finally:
                if keepalive_task is not None:
                    keepalive_task.cancel()
                    try:
                        await keepalive_task
                    except asyncio.CancelledError:
                        pass
            reply_bytes = json.dumps(response)
            logger.info(
                "CoderAI broker replied request_id=%s status=%s event=%s reply_bytes=%d",
                response.get("request_id"),
                response.get("status"),
                response.get("event"),
                len(reply_bytes),
            )
            resp_payload = response.get("payload") or {}
            if isinstance(resp_payload, dict):
                logger.debug(
                    "broker reply payload status_code=%s content_type=%s body_bytes=%s",
                    resp_payload.get("status_code"),
                    resp_payload.get("content_type"),
                    len((resp_payload.get("body") or "").encode()) if isinstance(resp_payload.get("body"), str) else (
                        len(resp_payload.get("body", b"")) if isinstance(resp_payload.get("body"), bytes) else "n/a"
                    ),
                )
            await self.websocket.send(reply_bytes)
            return response

        return None
