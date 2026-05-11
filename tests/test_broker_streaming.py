from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from codai.api.app import app as real_app
from codai.broker.dispatcher import execute_broker_request
from codai.broker.models import BrokerRequestEnvelope
from codai.broker.streaming import finalize_stream, stream_chunk_envelope


@pytest.mark.anyio("asyncio")
async def test_stream_chunk_envelope_preserves_request_id_and_order():
    chunk_one = stream_chunk_envelope(
        request_id="req-stream",
        sequence=0,
        data={"delta": "hel"},
    )
    chunk_two = stream_chunk_envelope(
        request_id="req-stream",
        sequence=1,
        data={"delta": "lo"},
    )

    assert chunk_one == {
        "request_id": "req-stream",
        "ok": True,
        "event": "stream",
        "payload": {
            "sequence": 0,
            "data": {"delta": "hel"},
        },
    }
    assert chunk_two == {
        "request_id": "req-stream",
        "ok": True,
        "event": "stream",
        "payload": {
            "sequence": 1,
            "data": {"delta": "lo"},
        },
    }


@pytest.mark.anyio("asyncio")
async def test_finalize_stream_attaches_metrics():
    response = finalize_stream(
        request_id="req-stream",
        total_chunks=2,
        elapsed_ms=15.25,
    )

    assert response == {
        "request_id": "req-stream",
        "ok": True,
        "event": "stream_end",
        "payload": {
            "total_chunks": 2,
        },
        "metrics": {
            "elapsed_ms": 15.25,
        },
    }


@pytest.mark.anyio("asyncio")
async def test_execute_broker_request_wraps_streaming_response_metadata():
    app = FastAPI()

    @app.get("/v1/chat/completions")
    async def stream_route():
        async def generate():
            yield b"data: first\n\n"
            yield b"data: second\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    envelope = BrokerRequestEnvelope(
        request_id="req-stream-route",
        method="GET",
        path="/v1/chat/completions",
        stream=True,
    )

    response = await execute_broker_request(app, envelope)

    assert set(response) == {"request_id", "ok", "payload", "metrics"}
    assert response["request_id"] == "req-stream-route"
    assert response["ok"] is True
    assert response["payload"] == {
        "status_code": 200,
        "headers": {
            "content-type": "text/event-stream; charset=utf-8",
        },
        "content_type": "text/event-stream; charset=utf-8",
        "body": "data: first\n\ndata: second\n\n",
        "stream": True,
    }
    assert isinstance(response["metrics"], dict)
    assert set(response["metrics"]) == {"elapsed_ms"}
    assert isinstance(response["metrics"]["elapsed_ms"], int | float)
    assert response["metrics"]["elapsed_ms"] >= 0


@pytest.mark.anyio("asyncio")
async def test_brokered_streaming_route_preserves_event_stream_body():
    app = FastAPI()

    @app.get("/v1/chat/completions")
    async def stream_route():
        async def generate():
            yield b'data: {"chunk":1}\n\n'
            yield b"data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    with TestClient(app) as client:
        with client.stream("GET", "/v1/chat/completions") as direct_response:
            direct_body = b"".join(direct_response.iter_bytes())
            direct_content_type = direct_response.headers["content-type"]
            direct_status_code = direct_response.status_code

    envelope = BrokerRequestEnvelope(
        request_id="req-stream-equivalence",
        method="GET",
        path="/v1/chat/completions",
        headers={"accept": "text/event-stream"},
        stream=True,
    )

    brokered_response = await execute_broker_request(app, envelope)

    assert brokered_response["ok"] is True
    assert brokered_response["payload"]["status_code"] == direct_status_code
    assert brokered_response["payload"]["content_type"] == direct_content_type
    assert brokered_response["payload"]["headers"]["content-type"] == direct_content_type
    assert brokered_response["payload"]["stream"] is True
    assert brokered_response["payload"]["body"].encode("utf-8") == direct_body
    assert "data: [DONE]" in brokered_response["payload"]["body"]
