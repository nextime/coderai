# CoderAI Broker Implementation Reference

## Purpose

This document is the single source of truth for implementing the CoderAI side of the AISBF broker and bridge integration.

The target audience is another LLM or engineer implementing CoderAI, not AISBF.

This document is mirrored in `docs/coderai-broker-implementation-reference.md` and should be kept identical in purpose and protocol coverage.

## AISBF broker mode

AISBF now includes a public broker-side WebSocket endpoint for outbound-only NAT traversal.

- Broker WebSocket endpoint: `/api/coderai/broker/ws`
- Broker WebSocket endpoints:
  - global scope: `/api/coderai/wss`
  - user scope: `/api/u/{username}/coderai/wss`
- Broker session status endpoint: `/api/coderai/broker/providers/{provider_id}/status`
- Broker session listing endpoint: `/api/coderai/broker/sessions`

Each CoderAI provider is owned either by:

- the global config admin (`user_id = null`), or
- a specific AISBF user (`user_id = <id>`)

Registration tokens are resolved from the owning provider configuration. This means:

- the global admin configures the token for globally configured `coderai` providers
- each user configures the token for their own user-scoped `coderai` providers
- a broker session is only usable by requests belonging to the same owner principal

Broker registration is now scope-aware:

- global providers register with `username=global`
- user-owned providers register with `username=<aisbf_username>`
- the same scoped path must be used by the CoderAI client when connecting over WebSocket
- deployments behind TLS termination or reverse proxies must connect with the externally visible `wss://...` URL and preserve proxy headers so AISBF can remain scheme-aware

The AISBF dashboard now exposes this token directly inside each `coderai` provider configuration:

- token input is stored in `coderai_config.registration_token`
- global admins edit global provider tokens in the admin providers page
- users edit their own provider tokens in the user providers page
- token rotation is available inline and returns a newly generated provider-scoped secret
- broker session status is shown directly in the provider editor, including owner, client id, transport, last seen, and advertised Studio endpoints

CoderAI can keep a persistent outbound connection open to AISBF, register itself, and then receive routed provider operations over that same socket.

## What AISBF now expects

### Provider type

Use provider type:

```json
{
  "type": "coderai"
}
```

### Provider config shape

```json
{
  "id": "coderai",
  "name": "CoderAI Local Bridge",
  "endpoint": "http://127.0.0.1:11437",
  "type": "coderai",
  "api_key_required": false,
  "coderai_config": {
    "transport": "http",
    "http_enabled": true,
    "websocket_enabled": true,
    "broker_enabled": true,
    "broker_mode": false,
    "broker_preferred": true,
    "discovery_enabled": true,
    "client_id": "aisbf-default",
    "bridge_path": "/coderai/ws",
    "registration_path": "/coderai/register",
    "registration_token": "optional-shared-secret",
    "bridge_token": "optional-bridge-secret",
    "request_timeout": 300,
    "model_timeout": 30
  }
}
```

### AISBF behaviors

- For `transport=http`, AISBF uses the OpenAI Python client against `endpoint + /v1`.
- For `transport=websocket`, AISBF uses a WebSocket bridge and sends framed JSON envelopes.
- AISBF uses `models.list`, `chat.completions`, `capabilities`, `register`, and `proxy` bridge operations.
- `proxy` now supports arbitrary forwarded request headers, query params, multipart form payloads, binary/base64 bodies, progress polling endpoints, and non-chat streaming event envelopes for long-running jobs.
- AISBF treats `coderai` like an OpenAI-style Studio adapter family.
- AISBF can also forward arbitrary Studio-native endpoints through `proxy` when the provider transport is WebSocket.
- AISBF validates that broker-enabled `coderai` providers have a non-empty `registration_token`.
- AISBF persists broker session metadata to `~/.aisbf/coderai_broker_sessions.json` so the dashboard can still show the last known broker session after restart, even while disconnected.

## Required CoderAI HTTP endpoints

### 1. OpenAI-compatible endpoints

CoderAI should already expose these when HTTP mode is enabled:

- `GET /v1/models`
- `POST /v1/chat/completions`
- optional additional OpenAI-compatible endpoints that Studio may use directly via generic proxy

The `/v1/models` response should preferably include as much metadata as possible:

```json
{
  "data": [
    {
      "id": "llama3.1:8b",
      "name": "llama3.1:8b",
      "description": "Local general-purpose chat model",
      "context_length": 131072,
      "architecture": {
        "input_modalities": ["text"],
        "output_modalities": ["text"]
      },
      "supported_parameters": ["temperature", "top_p", "max_tokens"],
      "default_parameters": {
        "temperature": 0.7
      },
      "pricing": null,
      "studio_capabilities": ["chat", "tool_use", "code_generation"]
    }
  ]
}
```

### 2. Capabilities endpoint

Expose:

- `GET /coderai/capabilities`

Recommended response:

```json
{
  "server": {
    "name": "coderai",
    "version": "0.1.0"
  },
  "transports": {
    "http": true,
    "websocket": true
  },
  "openai_compat": {
    "chat_completions": true,
    "models": true,
    "responses": false,
    "embeddings": true,
    "images": true,
    "audio": true
  },
  "studio": {
    "enabled": true,
    "endpoints": [
      "v1/images/generate",
      "v1/audio/tts",
      "v1/audio/transcriptions",
      "v1/video/dub"
    ]
  },
  "models": [
    {
      "id": "llama3.1:8b",
      "studio_capabilities": ["chat", "tool_use", "code_generation"]
    }
  ]
}
```

## Required WebSocket bridge

### Connection URL

CoderAI should accept WebSocket clients on:

- `/coderai/ws`

or another configured path mirrored in `coderai_config.bridge_path`.

### Headers AISBF sends

- `Authorization: Bearer <bridge_token_or_registration_token_or_api_key>` if available
- `x-coderai-client-id: <client_id>`
- `x-coderai-provider-id: <provider_id>`

### Broker connection query params

When CoderAI dials AISBF broker directly, it should connect using:

- `provider_id=<provider_id>`
- `client_id=<client_id>`
- `username=<username-or-global>`
- `registration_token=<owner-configured-token>`

AISBF broker admission currently resolves connection data in this order:

- `provider_id`: query param, then `x-coderai-provider-id`, then default `coderai`
- `client_id`: query param, then `x-coderai-client-id`, then generated fallback `anon-<timestamp>`
- `username`: query param, then `x-coderai-username`, then the scoped path value
- `registration_token`: query param, then `x-coderai-registration-token`

Important:

- the WebSocket broker flow starts as an HTTP `GET` upgrade request; this is expected
- the broker currently validates `registration_token`, not `Authorization: Bearer ...`, during WebSocket admission
- `client_id` must stay stable and match the `coderai_config.client_id` AISBF uses for that provider, or requests like `models.list` may fail even if the dashboard shows the session as connected

Example:

```text
wss://your-aisbf.example/api/coderai/wss?provider_id=coderai&client_id=workstation-01&username=global&registration_token=<owner-configured-token>
```

User-scoped example:

```text
wss://your-aisbf.example/api/u/alice/coderai/wss?provider_id=my-coderai&client_id=workstation-01&username=alice&registration_token=<owner-configured-token>
```

Recommended duplicate headers for robustness:

```text
x-coderai-provider-id: <provider_id>
x-coderai-client-id: <client_id>
x-coderai-username: <username>
x-coderai-registration-token: <registration_token>
```

### Broker registration flow

1. Open the WebSocket to the correct scoped AISBF broker URL.
2. Wait for AISBF to send an initial `registered` event.
3. Immediately send `op="register"` with hardware, capability, and endpoint metadata.
4. Wait for the `status="ok"` registration ack with the same `request_id`.
5. Enter the long-lived async broker loop for heartbeats, inbound requests, and outbound responses.

Initial server event example:

```json
{
  "v": 1,
  "event": "registered",
  "session_id": "coderai_abc123",
  "provider_id": "coderai",
  "client_id": "workstation-01",
  "username": "global",
  "scope_name": "global",
  "accepted": true,
  "owner_user_id": null,
  "expires_at": 1747179540
}
```

Required client follow-up:

```json
{
  "v": 1,
  "op": "register",
  "request_id": "reg-1",
  "payload": {
    "endpoint": "wss://client-or-descriptive-local-endpoint",
    "transport": "websocket",
    "registration_token": "<same_registration_token>",
    "hardware": {
      "gpus": [],
      "gpu_count": 0,
      "total_vram_mb": 0,
      "available_vram_mb": 0
    },
    "studio_endpoints": [],
    "capabilities": {}
  }
}
```

AISBF reads these `register` fields today:

- top-level: `v`, `op`, `request_id`, optional `registration_token`, optional `capabilities`
- from `payload`: `endpoint`, `transport`, `registration_token`, `studio_endpoints`, `hardware`, `gpus`, `gpu_count`, `total_vram_mb`, `available_vram_mb`, `capabilities`

If `payload.registration_token` or top-level `registration_token` is present and does not match the handshake token, AISBF replies with an error envelope.

### Envelope format

AISBF sends one JSON request envelope per operation:

```json
{
  "v": 1,
  "op": "chat.completions",
  "request_id": "coderai-1746960000000",
  "provider_id": "coderai",
  "client_id": "aisbf-default",
  "registration_token": "optional-shared-secret",
  "payload": {
    "model": "llama3.1:8b",
    "messages": [
      {"role": "user", "content": "hello"}
    ],
    "stream": false
  }
}
```

### Non-streaming response envelope

```json
{
  "v": 1,
  "request_id": "coderai-1746960000000",
  "status": "ok",
  "payload": {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1746960000,
    "model": "llama3.1:8b",
    "choices": [
      {
        "index": 0,
        "message": {"role": "assistant", "content": "hello"},
        "finish_reason": "stop"
      }
    ],
    "usage": {
      "prompt_tokens": 10,
      "completion_tokens": 5,
      "total_tokens": 15
    }
  }
}
```

### Error response envelope

```json
{
  "v": 1,
  "request_id": "coderai-1746960000000",
  "status": "error",
  "error": "Model not available",
  "code": "model_not_found",
  "details": {
    "model": "missing-model"
  }
}
```

### Streaming response envelopes

For `chat.completions` with `stream=true`, send multiple envelopes.

Each chunk envelope:

```json
{
  "v": 1,
  "request_id": "coderai-1746960000000",
  "status": "ok",
  "event": "chunk",
  "payload": {
    "chunk": "data: {\"id\":\"chatcmpl-123\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"delta\":{\"content\":\"hel\"},\"index\":0,\"finish_reason\":null}]}\n\n"
  }
}
```

Final envelope:

```json
{
  "v": 1,
  "request_id": "coderai-1746960000000",
  "status": "ok",
  "event": "done",
  "payload": {}
}
```

Important:

- `payload.chunk` should be a full SSE fragment already formatted exactly as AISBF should relay it.
- This keeps AISBF transport-simple and lets CoderAI own protocol correctness.
- Include `data: [DONE]\n\n` as one of the streamed chunks when the upstream semantics require it.

## Async broker behavior requirements

The broker client must be fully asynchronous.

- do not block the WebSocket receive loop on hardware probing, model loading, inference, file I/O, or reconnect bookkeeping
- handle inbound requests concurrently and correlate replies by `request_id`
- keep heartbeat handling lightweight
- preserve responsiveness to ping/pong traffic while local work is running

AISBF now independently drains queued outbound broker requests while also reading inbound WebSocket messages, so client implementations should not assume a strict request/response lockstep over the socket.

## Broker session visibility, persistence, and multi-node routing

AISBF now tracks two broker states:

- live connected sessions held in memory for active request routing
- persisted session metadata snapshots stored in `~/.aisbf/coderai_broker_sessions.json`

Persisted metadata is dashboard-facing only. It is used to show the last known session details after restart, but it is not treated as an active transport path until CoderAI reconnects.

For multi-node AISBF deployments behind a reverse proxy / load balancer:

- session status and ownership metadata are stored in the configured AISBF cache backend
- requests are enqueued into cache-backed broker queues keyed by broker session id
- the AISBF node holding the live WebSocket consumes queued requests and forwards them to CoderAI
- replies are written back through cache-backed reply keys so the AISBF node that originated the request can receive the result

Redis is the preferred backend for this distributed mode. SQLite/MySQL can operate as polling-based fallbacks. Memory/file cache backends are not suitable for cross-node broker routing.

Expected behavior:

- after reconnect, the persisted snapshot is refreshed with the new live session details
- after disconnect or AISBF restart, the dashboard may still show the last known client id / endpoint / last seen, but `connected` remains false until a new WebSocket is established

## Bridge operations CoderAI must implement

### `op = "models.list"`

Request:

```json
{
  "v": 1,
  "op": "models.list",
  "request_id": "...",
  "provider_id": "coderai",
  "client_id": "aisbf-default",
  "payload": {}
}
```

Response payload should be equivalent to `GET /v1/models`.

### `op = "chat.completions"`

Payload is equivalent to OpenAI `POST /v1/chat/completions` request body.

### `op = "capabilities"`

Response payload should be equivalent to `GET /coderai/capabilities`.

### `op = "register"`

Purpose:

- allow an outbound-only CoderAI agent to announce itself
- report its reachable transports
- report enabled Studio-native endpoints
- report model inventory
- attach metadata to the live AISBF broker session

Request payload from AISBF:

```json
{
  "provider_id": "coderai",
  "client_id": "aisbf-default",
  "transport": "websocket",
  "endpoint": "wss://broker.example/coderai/ws"
}
```

Recommended response payload:

```json
{
  "accepted": true,
  "client_id": "aisbf-default",
  "session_id": "sess_123",
  "expires_at": 1746963600,
  "transports": {
    "http": false,
    "websocket": true
  },
  "models": [
    {"id": "llama3.1:8b", "studio_capabilities": ["chat", "tool_use"]}
  ],
  "studio_endpoints": [
    "v1/video/dub",
    "v1/audio/tts"
  ]
}
```

### `op = "proxy"`

Purpose:

- tunnel arbitrary Studio-native endpoint requests over WebSocket when AISBF cannot directly reach CoderAI over HTTP.

Request payload:

```json
{
  "endpoint_path": "v1/video/dub",
  "method": "POST",
  "headers": {
    "x-request-id": "studio-job-123",
    "accept": "text/event-stream"
  },
  "query_params": {
    "job_id": "dub_123"
  },
  "body": {
    "model": "local-video-model",
    "input": "Dub this clip to Italian"
  },
  "multipart": {
    "fields": [{"name": "model", "value": "whisper-large"}],
    "files": [{"name": "file", "filename": "sample.wav", "content_type": "audio/wav", "data_base64": "<base64>"}]
  },
  "stream": true
}
```

Response payload:

```json
{
  "status_code": 200,
  "headers": {
    "content-type": "application/json"
  },
  "body": {
    "job_id": "dub_123",
    "status": "queued"
  }
}
```

Binary response payloads may instead use:

```json
{
  "status_code": 200,
  "content_type": "audio/mpeg",
  "body_base64": "<base64>",
  "headers": {
    "content-disposition": "attachment; filename=preview.mp3"
  }
}
```

Streaming and progress responses may emit multiple envelopes with `event` values like `progress`, `output`, `log`, `data`, `chunk`, and finally `done` or `completed`.

Recommended progress chunk payload:

```json
{
  "v": 1,
  "request_id": "coderai-1746960000000",
  "status": "ok",
  "event": "progress",
  "payload": {
    "chunk": "event: progress\ndata: {\"active\":true,\"current\":5,\"total\":20,\"pct\":25,\"elapsed\":12}\n\n"
  }
}
```

Capability advertisements should include endpoint metadata for custom pipelines, including supported methods, streaming mode, expected input/output modalities, and whether multipart or binary transport is required.

## Recommended CoderAI architecture

### Server components

1. **OpenAI compatibility router**
   - exposes `/v1/models`, `/v1/chat/completions`, and any other supported OpenAI endpoints

2. **Studio-native router**
   - exposes endpoints such as `v1/video/dub`, `v1/audio/tts`, `v1/images/generate`, etc.

3. **Capabilities registry**
   - enumerates enabled endpoints
   - enumerates loaded models
   - computes normalized `studio_capabilities`

4. **WebSocket bridge server**
   - accepts AISBF envelopes
   - dispatches by `op`
   - for `proxy`, internally calls the same handler used by HTTP routes
   - for `chat.completions`, either:
     - returns a full JSON result, or
     - emits `chunk` envelopes carrying ready-made SSE fragments

5. **Optional outbound broker client**
   - when behind NAT, CoderAI can establish and maintain an outbound WebSocket connection to an AISBF-reachable broker endpoint
   - that broker can multiplex messages by `client_id`

## NAT-friendly model

There are two viable patterns.

### Pattern A: AISBF directly opens WebSocket to CoderAI

- simplest
- works when CoderAI is reachable by `ws://` or `wss://`
- no NAT punching support

### Pattern B: CoderAI dials outward and stays connected

- best for NAT/private LAN
- CoderAI opens a persistent outbound WebSocket to a public AISBF-side broker
- broker stores the live session keyed by `client_id`
- AISBF routes provider operations to that session

If you implement Pattern B, keep the same envelope contract. Only the connection initiator changes.

### Recommended outbound broker flow

1. CoderAI opens persistent WebSocket to AISBF broker endpoint.
2. AISBF immediately acknowledges with `event=registered` and a `session_id`.
3. CoderAI sends `op=register` with endpoint, transports, capabilities, models, and Studio endpoints.
4. AISBF stores that live session under `provider_id + client_id`.
5. All AISBF provider operations can now be delivered to that live outbound socket.
6. If the socket drops, AISBF marks the session offline and fails in-flight requests.

## Strong recommendations for metadata

For every model, provide:

- `id`
- `name`
- `description`
- `context_length`
- `architecture.input_modalities`
- `architecture.output_modalities`
- `supported_parameters`
- `default_parameters`
- `studio_capabilities`

For server capabilities, provide:

- transport availability
- OpenAI-compatible endpoint availability
- Studio-native endpoint availability
- current server version
- optional hardware metadata (`gpu`, `memory_gb`, `quantization`, `throughput_hint`)

## Minimal Python implementation sketch for CoderAI

```python
from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
import json

app = FastAPI()


@app.get("/v1/models")
async def list_models():
    return {
        "data": [
            {
                "id": "llama3.1:8b",
                "name": "llama3.1:8b",
                "context_length": 131072,
                "studio_capabilities": ["chat", "tool_use", "code_generation"],
            }
        ]
    }


@app.get("/coderai/capabilities")
async def capabilities():
    return {
        "server": {"name": "coderai", "version": "0.1.0"},
        "transports": {"http": True, "websocket": True},
        "openai_compat": {"chat_completions": True, "models": True},
        "studio": {"enabled": True, "endpoints": ["v1/video/dub"]},
    }


@app.websocket("/coderai/ws")
async def coderai_ws(ws: WebSocket):
    await ws.accept()
    while True:
        message = await ws.receive_text()
        envelope = json.loads(message)
        op = envelope["op"]
        request_id = envelope["request_id"]

        if op == "models.list":
            await ws.send_text(json.dumps({
                "v": 1,
                "request_id": request_id,
                "status": "ok",
                "payload": await list_models(),
            }))
        elif op == "capabilities":
            await ws.send_text(json.dumps({
                "v": 1,
                "request_id": request_id,
                "status": "ok",
                "payload": await capabilities(),
            }))
        else:
            await ws.send_text(json.dumps({
                "v": 1,
                "request_id": request_id,
                "status": "error",
                "error": f"Unsupported op: {op}",
            }))
```

## Implementation checklist for the CoderAI-side LLM session

- add `/coderai/capabilities`
- add `/coderai/register`
- add `/coderai/ws`
- expose model metadata with `studio_capabilities`
- support `models.list`
- support `chat.completions`
- support streaming `chunk` and `done` events
- support `proxy` for Studio-native endpoints
- optionally support persistent outbound broker mode for NAT traversal
- protect bridge/register endpoints with a shared secret or signed token

## Compatibility notes

- AISBF currently assumes WebSocket streamed chunks arrive already formatted as SSE fragments.
- AISBF currently expects WebSocket non-streaming responses to carry the raw OpenAI-compatible response under `payload`.
- AISBF can consume either direct HTTP OpenAI compatibility or the WebSocket bridge for chat/model listing.
- AISBF generic Studio proxy now uses the provider bridge for `coderai`, making NAT traversal possible for non-chat endpoints too.
