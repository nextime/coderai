# AISBF Broker Client Design

## Goal

Implement the CoderAI side of the AISBF broker protocol so a running CoderAI instance can register itself with either a global or user-owned AISBF `coderai` provider, stay connected over an outbound WebSocket, receive brokered OpenAI-compatible and studio requests, execute them locally, and return success, error, streaming, and binary envelopes using the same `request_id`.

This design targets full first-pass coverage for both:

- OpenAI-compatible brokered operations
- Studio brokered operations

## Scope

The implementation covers:

- outbound broker WebSocket connection to AISBF using scoped URLs
- registration using `provider_id`, `client_id`, `username`, and `registration_token`
- post-connect `register` payload emission with capabilities and hardware metadata
- heartbeat handling and optional proactive heartbeat support
- automatic reconnect with backoff and re-registration
- inbound request dispatch for supported OpenAI-compatible and studio endpoints
- local execution using the same application behavior as direct HTTP access
- response envelope emission for JSON, errors, streams, and binary outputs
- performance metrics on completed requests when available
- broker-aware tests for config, protocol lifecycle, dispatch, streaming, and reconnect

The implementation does not include:

- a separate sidecar worker process
- refactoring endpoint business logic out of existing FastAPI routes unless required by broker reuse
- cross-owner broker session reuse
- speculative protocol extensions not described in `coderai-broker-implementation-reference.md`

## Chosen Approach

Use an in-process broker subsystem owned by the existing CoderAI runtime.

This subsystem will live under `codai/broker/` and run as a background async service started and stopped by the existing application lifecycle. Brokered requests will be translated into in-process ASGI requests against the existing FastAPI app where practical so direct HTTP traffic and brokered AISBF traffic share the same handlers, models, validation, and response behavior.

This was chosen over a sidecar worker because the current codebase is already organized around one FastAPI process with centralized runtime state in `codai/main.py` and `codai/api/app.py`. It also avoids duplicating endpoint behavior in broker-only code.

## Architecture

### Runtime Ownership

The broker client is a long-lived runtime service.

Startup flow:

1. Load broker config from `ConfigManager`
2. Validate scope, endpoint URL inputs, and required credentials
3. Store broker config on `fastapi_app.state`
4. Start a broker service task during FastAPI lifespan startup when enabled
5. Begin connect/register/receive loop

Shutdown flow:

1. Cancel broker receive and heartbeat tasks
2. Close the WebSocket cleanly
3. Stop reconnect attempts
4. Await shutdown completion before lifespan exits

The FastAPI lifespan remains the single owner of broker lifecycle so cleanup is deterministic and consistent with existing archive/model cleanup behavior.

### Module Layout

#### `codai/broker/config.py`

Responsibilities:

- map `Config` broker fields into a validated runtime object
- enforce scope rules for `global` vs user-owned providers
- construct the correct AISBF WebSocket URL
- expose connect/reconnect/heartbeat timeouts and intervals
- redact secrets for logs

#### `codai/broker/models.py`

Responsibilities:

- define typed broker protocol models
- represent AISBF `registered` events and heartbeat requests
- represent the outgoing `register` payload
- represent inbound brokered request envelopes
- represent outbound success, error, stream, and binary envelopes
- represent optional metrics payloads

#### `codai/broker/client.py`

Responsibilities:

- open the outbound WebSocket using the scoped URL and headers
- await the initial AISBF `registered` event
- store runtime session metadata such as `session_id`
- send the required `register` operation
- run the long-lived receive loop
- answer heartbeat requests immediately
- trigger reconnect with backoff after disconnects or fatal socket failures

#### `codai/broker/capabilities.py`

Responsibilities:

- build the canonical capability document for both HTTP and broker registration
- advertise OpenAI-compatible and studio endpoints
- derive available endpoints from current application support
- collect hostname, platform, GPU count, VRAM totals, and per-GPU metadata where detectable
- fall back conservatively when exact GPU telemetry is unavailable

#### `codai/broker/dispatcher.py`

Responsibilities:

- validate inbound brokered request envelopes
- normalize inbound requests into an execution descriptor
- map request method/path/query/headers/body into internal execution calls
- reject unsupported routes with structured broker errors
- isolate each inbound request in its own async task

#### `codai/broker/asgi_bridge.py`

Responsibilities:

- construct synthetic in-process ASGI requests against the FastAPI app
- support JSON bodies, query strings, headers, multipart data, and raw bytes
- collect response status, headers, body, and streaming events
- provide a single reuse layer so brokered requests and direct HTTP traffic stay behaviorally aligned

#### `codai/broker/streaming.py`

Responsibilities:

- translate streaming application responses into broker stream envelopes
- preserve chunk ordering per `request_id`
- send a final completion envelope with aggregate metrics
- normalize progress-style studio responses and chat streaming responses

#### `codai/broker/service.py`

Responsibilities:

- expose an app-facing service wrapper with `start()` and `stop()`
- own the broker client instance and background tasks
- present a small lifecycle API to FastAPI startup/shutdown code

## Configuration Design

Add a `broker` section to `Config` and `config.json`.

Recommended fields:

```json
{
  "broker": {
    "enabled": false,
    "base_url": "wss://aisbf.example.com",
    "scope": "global",
    "username": "global",
    "provider_id": "coderai",
    "client_id": "workstation-01",
    "registration_token": "<secret>",
    "advertised_endpoint": "http://127.0.0.1:8776",
    "transport": "websocket",
    "heartbeat_interval_seconds": 30,
    "connect_timeout_seconds": 15,
    "request_timeout_seconds": 900,
    "reconnect_initial_delay_seconds": 2,
    "reconnect_max_delay_seconds": 60
  }
}
```

Validation rules:

- `enabled=false` means broker code does not start
- `scope=global` requires `username=global`
- user scope requires non-empty `username` that is not `global`
- `provider_id`, `client_id`, and `registration_token` are required when enabled
- `base_url` must resolve to `ws://` or `wss://` when converted to broker transport
- one config instance maps to exactly one owner scope and must not be reused across unrelated principals

URL construction rules:

- global scope: `/api/coderai/wss?provider_id=...&client_id=...&username=global&registration_token=...`
- user scope: `/api/u/<username>/coderai/wss?provider_id=...&client_id=...&username=<username>&registration_token=...`

Headers should also include:

- `Authorization: Bearer <registration_token>`
- `x-coderai-provider-id`
- `x-coderai-client-id`
- `x-coderai-username`

## Registration Design

After connect, the client waits for an AISBF `registered` event. Only after receiving that event does it send the explicit `register` message.

The `register` payload includes:

- advertised endpoint and transport
- registration token echo as required by reference
- hardware inventory
- studio endpoint list
- capability document

The capability document comes from one shared builder so:

- broker `register` payloads
- local `GET /coderai/capabilities` responses

are produced from the same source and cannot drift.

## Hardware Reporting Design

Hardware reporting should use the best information already available to the process.

Preferred fields:

- hostname
- platform
- `gpus`
- `gpu_count`
- `total_vram_mb`
- `available_vram_mb`

Per-GPU preferred fields:

- `index`
- `name`
- `vendor`
- `total_vram_mb`
- `available_vram_mb`
- `used_vram_mb`

If exact values are unavailable:

- omit only fields that truly cannot be determined
- estimate conservatively where the codebase already has enough information to do so safely
- include an `estimated` marker where estimation is used

The design does not require introducing heavyweight telemetry dependencies just to populate optional fields.

## Local Capability Surface

### Local HTTP endpoint

Expose:

- `GET /coderai/capabilities`

This endpoint returns:

- server identity and version
- transport support flags
- OpenAI-compatible support map
- studio enablement and endpoint list
- optional hardware summary

### Advertised broker support

First-pass broker coverage includes:

- `GET /v1/models`
- `POST /v1/chat/completions`
- representative OpenAI-compatible routes already supported by the server where AISBF may invoke them
- studio routes already exposed by the application, especially:
  - `v1/images/generate`
  - `v1/images/progress`
  - `v1/audio/tts`
  - `v1/audio/transcriptions`
  - `v1/audio/progress`
  - `v1/video/dub`
  - `v1/video/progress`

The dispatcher should be extensible so additional routes can be added by mapping them into the same ASGI execution path without redesign.

## Request Execution Design

### Inbound request normalization

Each brokered AISBF request is normalized into an internal execution object containing at least:

- `request_id`
- HTTP method
- path
- query parameters
- headers
- body or structured payload
- content type
- streaming flag
- binary expectations if present

Malformed or incomplete requests are rejected before execution with a structured broker error envelope.

### Execution path

The preferred execution path is ASGI bridging into the existing FastAPI app.

Why:

- preserves current request validation behavior
- reuses all existing route handlers
- keeps broker logic focused on protocol translation
- avoids divergent behavior between direct and brokered clients

Broker-specific code should not reimplement chat, models, image, audio, or video endpoint logic unless an endpoint cannot be represented cleanly through the bridge.

### Concurrency

Each inbound broker request runs in its own async task so one long-running generation does not block the socket receive loop.

The broker service must continue to:

- accept heartbeat traffic
- process other inbound events
- surface task completion or failure independently per `request_id`

Timeout handling should use the configured request timeout but avoid prematurely aborting long-running studio jobs if those jobs are intentionally modeled as progress-based asynchronous flows.

## Response Envelope Design

### Success responses

Successful request completion returns an envelope containing:

- `v`
- original `request_id`
- `status: ok`
- response metadata such as HTTP status and headers when useful
- JSON payload body or binary metadata/body
- optional metrics

### Error responses

Errors return an envelope containing:

- original `request_id`
- `status: error`
- stable machine-readable error code
- human-readable message
- optional details payload

Error classes include:

- malformed broker request
- unsupported endpoint
- local validation failure
- execution failure
- timeout
- internal broker transport failure

### Streaming responses

Streaming support is required for chat completions and progress-style endpoints where the local endpoint already supports streaming or incremental progress.

Streaming design:

- emit ordered chunk envelopes using the same `request_id`
- include minimal chunk metadata needed for AISBF to reconstruct the stream
- send a final completion envelope when the stream ends
- include aggregate metrics on the final envelope when available

### Binary responses

Binary-producing endpoints such as image, audio, or video operations must return broker envelopes that preserve:

- content type
- filename or artifact naming when relevant
- byte content or the reference-compliant encoded form
- size metadata where useful

Binary encoding/normalization must be centralized in broker code, not duplicated across individual endpoints.

## Heartbeat and Reconnect Design

### Heartbeat handling

If AISBF sends:

```json
{
  "v": 1,
  "op": "heartbeat",
  "request_id": "hb-123",
  "payload": {}
}
```

CoderAI responds immediately with the same `request_id` and current timestamp.

Optional proactive heartbeat may also be sent on a configured interval. If implemented, it may include updated hardware availability such as free VRAM.

### Reconnect handling

If the WebSocket disconnects:

1. mark the session inactive
2. cancel or fail in-flight connection-scoped listeners cleanly
3. wait using bounded exponential backoff
4. reconnect using the same stable `client_id`
5. wait for a new `registered` event
6. send `register` again
7. resume the receive loop

No reconnect attempt may silently change owner scope or client identity.

## Security and Logging Design

- never log `registration_token`
- log connect target, scope, provider id, client id, and reconnect attempt counts
- log protocol failures with enough structure to debug malformed envelopes
- reject impossible scope combinations during startup rather than attempting a best-effort connection
- keep bearer token use aligned with the reference by sending it in both query params and headers for robustness

## Testing Design

This implementation should follow TDD during execution.

### Unit tests

Add focused tests for:

- broker config validation
- scoped URL construction
- header construction
- register payload generation
- capability serialization
- hardware serialization with partial data
- heartbeat request handling
- reconnect backoff progression
- inbound envelope validation

### Protocol integration tests

Use a fake AISBF WebSocket server to verify:

- successful initial connect
- waiting for `registered` before sending `register`
- correct `register` payload contents
- heartbeat request/response behavior
- reconnect and re-registration after disconnect
- error handling for malformed AISBF events

### Application integration tests

Verify equivalent behavior between direct FastAPI access and brokered execution for:

- `GET /v1/models`
- `POST /v1/chat/completions`
- at least one representative streaming endpoint
- at least one representative binary-producing studio endpoint
- at least one representative progress endpoint

### Streaming and binary tests

Add explicit tests for:

- ordered stream chunk emission
- final stream completion envelope
- metrics attachment on completion
- binary payload metadata
- multipart input translation for upload-style studio endpoints

## File Plan

Expected new files:

- `codai/broker/__init__.py`
- `codai/broker/config.py`
- `codai/broker/models.py`
- `codai/broker/client.py`
- `codai/broker/capabilities.py`
- `codai/broker/dispatcher.py`
- `codai/broker/asgi_bridge.py`
- `codai/broker/streaming.py`
- `codai/broker/service.py`
- `tests/test_broker_config.py`
- `tests/test_broker_protocol.py`
- `tests/test_broker_dispatch.py`
- `tests/test_broker_streaming.py`

Expected modified files:

- `codai/config.py`
- `codai/api/app.py`
- `codai/main.py`
- router modules only if a route needs small adjustments to be bridge-safe or capability-aware

## Risks and Mitigations

### Risk: streaming behavior differs between direct HTTP and broker execution

Mitigation:

- isolate streaming adaptation in `codai/broker/streaming.py`
- add equivalence-oriented tests against representative endpoints

### Risk: some studio routes are not easy to invoke through a synthetic ASGI request

Mitigation:

- keep ASGI bridging as the primary path
- allow a narrow fallback adapter only for endpoints whose input/output shape cannot be bridged cleanly
- keep those fallbacks inside broker modules, not scattered through endpoint code

### Risk: hardware telemetry is incomplete across backends

Mitigation:

- support partial hardware payloads
- estimate conservatively only where justified
- avoid blocking broker startup on optional telemetry

### Risk: reconnect churn causes duplicate registration confusion

Mitigation:

- treat each socket as a fresh session after `registered`
- replace stored session metadata atomically on reconnect
- reuse stable `client_id` but not stale `session_id`

## Acceptance Criteria

The design is satisfied when a configured CoderAI instance can:

- connect to either global or user-scoped AISBF broker endpoints
- receive `registered` and send a valid `register` payload
- remain connected with heartbeat support
- reconnect automatically after disconnect and re-register successfully
- serve brokered OpenAI-compatible and studio requests using current local handlers
- return structured success, error, stream, and binary responses tied to the original `request_id`
- surface capabilities and hardware metadata consistently across HTTP and broker registration
- pass the new broker-focused automated tests
