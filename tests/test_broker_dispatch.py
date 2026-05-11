import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from starlette.responses import Response

from codai.api.app import app as real_app
from codai.broker.asgi_bridge import execute_internal_request
from codai.broker.dispatcher import execute_broker_request
from codai.broker.models import BrokerRequestEnvelope


@pytest.mark.anyio("asyncio")
async def test_execute_internal_request_returns_json_response():
    app = FastAPI()

    @app.get("/internal")
    async def internal_route():
        return JSONResponse(
            status_code=202,
            content={"message": "ok"},
            headers={"x-broker": "ready"},
        )

    response = await execute_internal_request(
        app,
        method="GET",
        path="/internal",
        headers={"accept": "application/json"},
        query={"source": "broker", "limit": "1"},
    )

    assert response == {
        "status_code": 202,
        "headers": {
            "content-length": "16",
            "content-type": "application/json",
            "x-broker": "ready",
        },
        "body": b'{"message":"ok"}',
    }


@pytest.mark.anyio("asyncio")
async def test_execute_internal_request_supports_json_body():
    app = FastAPI()

    @app.post("/internal")
    async def internal_route(payload: dict):
        return {"received": payload}

    response = await execute_internal_request(
        app,
        method="POST",
        path="/internal",
        headers={"content-type": "application/json"},
        body=b'{"message":"hello","count":2}',
    )

    assert response == {
        "status_code": 200,
        "headers": {
            "content-length": "42",
            "content-type": "application/json",
        },
        "body": b'{"received":{"message":"hello","count":2}}',
    }


@pytest.mark.anyio("asyncio")
async def test_execute_broker_request_returns_success_envelope_for_json_route():
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat_route(payload: dict):
        return JSONResponse(
            status_code=201,
            content={"received": payload, "route": "chat"},
            headers={"x-broker": "executed"},
        )

    envelope = BrokerRequestEnvelope(
        request_id="req-123",
        method="POST",
        path="/v1/chat/completions",
        headers={"accept": "application/json"},
        payload={"message": "hello", "count": 2},
    )

    response = await execute_broker_request(app, envelope)

    assert response["request_id"] == "req-123"
    assert response["ok"] is True
    assert response["payload"] == {
        "status_code": 201,
        "headers": {
            "content-length": "57",
            "content-type": "application/json",
            "x-broker": "executed",
        },
        "content_type": "application/json",
        "body": '{"received":{"message":"hello","count":2},"route":"chat"}',
    }
    assert isinstance(response["metrics"]["elapsed_ms"], int | float)
    assert response["metrics"]["elapsed_ms"] >= 0


@pytest.mark.anyio("asyncio")
async def test_execute_broker_request_preserves_binary_payload_metadata():
    app = FastAPI()

    @app.get("/v1/images/render")
    async def image_route():
        return Response(
            content=b"\x89PNG\r\n",
            media_type="image/png",
            headers={"x-filename": "image.png"},
        )

    envelope = BrokerRequestEnvelope(
        request_id="req-binary",
        method="GET",
        path="/v1/images/render",
    )

    response = await execute_broker_request(app, envelope)

    assert response["request_id"] == "req-binary"
    assert response["ok"] is True
    assert response["payload"] == {
        "status_code": 200,
        "headers": {
            "content-length": "6",
            "content-type": "image/png",
            "x-filename": "image.png",
        },
        "content_type": "image/png",
        "filename": "image.png",
        "body_base64": "iVBORw0K",
    }
    assert isinstance(response["metrics"]["elapsed_ms"], int | float)
    assert response["metrics"]["elapsed_ms"] >= 0


@pytest.mark.anyio("asyncio")
async def test_brokered_models_match_direct_http_response_shape():
    direct_response = TestClient(real_app).get("/v1/models")

    envelope = BrokerRequestEnvelope(
        request_id="req-models-shape",
        method="GET",
        path="/v1/models",
        headers={"accept": "application/json"},
    )

    brokered_response = await execute_broker_request(real_app, envelope)
    brokered_body = json.loads(brokered_response["payload"]["body"])
    direct_body = direct_response.json()

    assert brokered_response["ok"] is True
    assert brokered_response["payload"]["status_code"] == direct_response.status_code
    assert brokered_response["payload"]["content_type"] == direct_response.headers["content-type"]
    assert brokered_response["payload"]["headers"]["content-type"] == direct_response.headers["content-type"]
    assert brokered_response["payload"]["headers"]["content-length"] == direct_response.headers["content-length"]
    assert isinstance(brokered_body, dict)
    assert isinstance(direct_body, dict)
    assert set(brokered_body) == {"data", "object"}
    assert set(direct_body) == {"data", "object"}
    assert isinstance(brokered_body["object"], str)
    assert isinstance(direct_body["object"], str)
    assert isinstance(brokered_body["data"], list)
    assert isinstance(direct_body["data"], list)
    for brokered_model, direct_model in zip(brokered_body["data"], direct_body["data"], strict=True):
        assert isinstance(brokered_model, dict)
        assert isinstance(direct_model, dict)
        assert brokered_model.keys() == direct_model.keys()
        assert {"id", "object", "owned_by"}.issubset(brokered_model)
        assert isinstance(brokered_model["id"], str)
        assert isinstance(brokered_model["object"], str)
        assert isinstance(brokered_model["owned_by"], str)
    assert brokered_body == direct_body


@pytest.mark.anyio("asyncio")
async def test_execute_broker_request_rejects_unsupported_endpoint():
    app = FastAPI()
    envelope = BrokerRequestEnvelope(
        request_id="req-unsupported",
        method="GET",
        path="/internal",
    )

    response = await execute_broker_request(app, envelope)

    assert response == {
        "request_id": "req-unsupported",
        "ok": False,
        "error": {
            "code": "unsupported_endpoint",
            "message": "Unsupported endpoint: /internal",
        },
    }
