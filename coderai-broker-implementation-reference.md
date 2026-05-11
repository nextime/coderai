# CoderAI Broker Implementation Reference

## Purpose

This document is the single source of truth for implementing the CoderAI side of the AISBF broker and bridge integration.

The target audience is another LLM or engineer implementing CoderAI, not AISBF.

The implementation must let a CoderAI instance:

- expose direct HTTP and optional direct WebSocket bridge endpoints for AISBF
- connect outward to AISBF over WebSocket when CoderAI is behind NAT
- register against either a global or user-owned AISBF `coderai` provider
- receive brokered requests from AISBF
- execute those requests locally inside CoderAI
- send responses back to AISBF using the envelope protocol described here

This reference supersedes separate fragmented notes. Treat this file as the canonical contract.

## High-Level Goal

Implement a persistent outbound broker client in CoderAI plus the local handlers needed to serve AISBF requests.

The broker client should:

1. Dial AISBF over `ws://` or `wss://`
2. Authenticate using the provider-scoped `registration_token`
3. Register metadata, hardware inventory, and capabilities after connect
4. Stay connected with heartbeat support
5. Receive queued or direct broker requests from AISBF
6. Execute supported operations locally
7. Send back success, error, binary, and streaming envelopes with the same `request_id`
8. Include performance metrics for completed requests whenever available
9. Automatically reconnect if the connection drops

## AISBF Concepts You Must Match

AISBF supports `coderai` as a first-class provider.

Each `coderai` provider belongs to exactly one owner:

- global admin scope
- user scope

The broker session must register into the correct scope.

### Scope Rules

- Global provider connections use the global broker path and `username=global`
- User-owned provider connections use the user-scoped broker path and `username=<aisbf_username>`
- The registration token belongs to that exact provider configuration and owner scope
- One broker session must not be reused across unrelated owners
- AISBF rejects broker request use if the owner principal does not match the connected session owner

## Broker Endpoints

### Global Scope

```text
wss://<aisbf-host>/api/coderai/wss?provider_id=<provider_id>&client_id=<client_id>&username=global&registration_token=<token>
```

### User Scope

```text
wss://<aisbf-host>/api/u/<username>/coderai/wss?provider_id=<provider_id>&client_id=<client_id>&username=<username>&registration_token=<token>
```

### Notes

- Use `wss://` whenever AISBF is exposed through HTTPS or TLS termination
- Use the externally visible URL, not necessarily AISBF's internal bind address
- AISBF may sit behind a reverse proxy that terminates TLS
- The CoderAI client must work with both direct TLS and reverse-proxy-managed TLS

## Required Connection Parameters

The outbound WebSocket connection must include:

- `provider_id`
- `client_id`
- `username`
- `registration_token`

### Meaning

- `provider_id`: AISBF provider id such as `coderai` or `my-coderai`
- `client_id`: stable machine or session identifier chosen in provider config, such as `workstation-01`
- `username`: either `global` or the AISBF username for user-owned providers
- `registration_token`: provider-scoped secret from AISBF provider configuration

## Optional Headers

AISBF also accepts or may expect these headers:

- `Authorization: Bearer <registration_token or bridge token>`
- `x-coderai-provider-id: <provider_id>`
- `x-coderai-client-id: <client_id>`
- `x-coderai-username: <username>`

Recommended behavior:

- include both query params and headers for robustness
- use the registration token as bearer auth if no separate bridge token exists

## Broker Session Lifecycle

### 1. Connect

Open the outbound WebSocket to the correct scoped AISBF endpoint.

### 2. Wait for `registered` event

AISBF immediately sends a registration acknowledgment event on successful admission.

Example:

```json
{
  "v": 1,
  "event": "registered",
  "session_id": "coderai_abc123",
  "provider_id": "coderai",
  "client_id": "workstation-01",
  "username": "global",
  "scope_name": "global",
  "accepted": true
}
```

Store:

- `session_id`
- `provider_id`
- `client_id`
- `username`
- `scope_name`

### 3. Send explicit `register` operation

After the `registered` event, CoderAI must send a `register` message describing its capabilities, hardware inventory, and advertised endpoints.

### 4. Enter long-lived receive loop

Then keep listening for incoming broker requests from AISBF.

### 5. Heartbeat and reconnect

If the socket drops:

- reconnect with backoff
- re-register after reconnect
- preserve the same stable `client_id`

## Required `register` Message

CoderAI should send this after receiving the initial AISBF `registered` event.

```json
{
  "v": 1,
  "op": "register",
  "request_id": "reg-1",
  "payload": {
    "endpoint": "ws://local-coderai-or-descriptive-endpoint",
    "transport": "websocket",
    "registration_token": "<same_registration_token>",
    "hardware": {
      "hostname": "workstation-01",
      "platform": "linux",
      "gpus": [
        {
          "index": 0,
          "name": "NVIDIA RTX 4090",
          "vendor": "nvidia",
          "total_vram_mb": 24576,
          "available_vram_mb": 20480,
          "used_vram_mb": 4096
        }
      ],
      "gpu_count": 1,
      "total_vram_mb": 24576,
      "available_vram_mb": 20480
    },
    "studio_endpoints": [
      "v1/images/generate",
      "v1/audio/tts",
      "v1/audio/transcriptions",
      "v1/audio/progress",
      "v1/video/dub",
      "v1/video/progress"
    ],
    "capabilities": {
      "studio": {
        "enabled": true,
        "endpoints": [
          "v1/images/generate",
          "v1/images/progress",
          "v1/audio/tts",
          "v1/audio/progress",
          "v1/video/dub",
          "v1/video/progress"
        ],
        "endpoint_capabilities": {
          "v1/video/dub": {
            "methods": ["POST"],
            "input_modalities": ["text", "video", "audio"],
            "output_modalities": ["video"],
            "supports_stream": true,
            "supports_multipart": true,
            "supports_binary": true
          },
          "v1/video/progress": {
            "methods": ["GET"],
            "input_modalities": [],
            "output_modalities": ["progress"],
            "supports_stream": true,
            "supports_binary": false
          }
        }
      },
      "openai_compat": {
        "chat_completions": true,
        "models": true,
        "embeddings": true,
        "images": true,
        "audio": true
      }
    }
  }
}
```

AISBF replies with a success envelope.

### Hardware Reporting Requirements

The `register` payload should include the best hardware view available to the running CoderAI process.

Required if detectable:

- `hardware.gpus`: array of GPU objects visible to the process
- `hardware.gpu_count`: integer count
- `hardware.total_vram_mb`: total usable VRAM across advertised GPUs
- `hardware.available_vram_mb`: currently free or available VRAM across advertised GPUs

Recommended per GPU fields:

- `index`
- `name`
- `vendor`
- `total_vram_mb`
- `available_vram_mb`
- `used_vram_mb`
- backend-specific extras if trivial to expose

If exact values are unavailable, estimate conservatively and include any supporting marker such as `estimated: true`.

AISBF stores this in broker session metadata so dashboards and future routing logic can reason about available hardware.

## Required Heartbeat Support

AISBF may send heartbeat requests, and CoderAI may also proactively keep the socket alive.

### If AISBF sends heartbeat

Request example:

```json
{
  "v": 1,
  "op": "heartbeat",
  "request_id": "hb-123",
  "payload": {}
}
```

Reply example:

```json
{
  "v": 1,
  "request_id": "hb-123",
  "status": "ok",
  "event": "heartbeat",
  "payload": {
    "ts": 1746960000
  }
}
```

### Optional proactive heartbeat

CoderAI may also periodically send:

```json
{
  "v": 1,
  "op": "heartbeat",
  "request_id": "hb-self-1",
  "payload": {
    "uptime": 1234
  }
}
```

Heartbeat payloads may also refresh dynamic hardware state such as changing free VRAM:

```json
{
  "v": 1,
  "op": "heartbeat",
  "request_id": "hb-self-2",
  "payload": {
    "hardware": {
      "available_vram_mb": 18432,
      "gpus": [
        {
          "index": 0,
          "available_vram_mb": 18432,
          "used_vram_mb": 6144
        }
      ]
    }
  }
}
```

AISBF merges those updates into the broker session metadata.

## Local HTTP Endpoints CoderAI Should Expose

### OpenAI-compatible endpoints

At minimum:

- `GET /v1/models`
- `POST /v1/chat/completions`

Optional additional OpenAI-compatible endpoints may also be exposed if AISBF will use them.

Preferred `/v1/models` response:

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

### Capabilities endpoint

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
      "v1/images/progress",
      "v1/audio/tts",
      "v1/audio/transcriptions",
      "v1/audio/progress",
      "v1/video/dub",
      "v1/video/progress"
    ],
    "endpoint_capabilities": {
      "v1/images/generate": {
        "methods": ["POST"],
        "input_modalities": ["text", "image"],
        "output_modalities": ["image"],
        "supports_stream": false,
        "supports_multipart": true,
        "supports_binary": true
      },
      "v1/images/progress": {
        "methods": ["GET"],
        "input_modalities": [],
        "output_modalities": ["progress"],
        "supports_stream": true,
        "supports_binary": false
      }
    }
  },
  "models": [
    {
      "id": "llama3.1:8b",
      "studio_capabilities": ["chat", "tool_use", "code_generation"]
    }
  ]
}
```

## Direct WebSocket Bridge

### Path

CoderAI should accept WebSocket clients on:

- `/coderai/ws`

or another configured path mirrored in `coderai_config.bridge_path`.

### Headers AISBF sends

- `Authorization: Bearer <bridge_token_or_registration_token_or_api_key>` if available
- `x-coderai-client-id: <client_id>`
- `x-coderai-provider-id: <provider_id>`
- optionally `x-coderai-username: <username>`

## Incoming Request Envelope Format

AISBF sends one JSON envelope per operation.

Example:

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

## Supported Operations

CoderAI must implement these operations:

- `models.list`
- `chat.completions`
- `capabilities`
- `register`
- `proxy`
- `heartbeat`

### `op = "models.list"`

Request payload:

```json
{}
```

Response payload should match `GET /v1/models`.

### `op = "chat.completions"`

Request payload matches OpenAI `POST /v1/chat/completions` body.

Completed responses should include performance metrics whenever practical:

- `latency_ms`
- `prompt_tokens`
- `completion_tokens`
- `total_tokens`
- `tokens_per_second`

If exact values are unavailable, AISBF estimates latency from broker timing and may estimate throughput from tokens plus latency.

### `op = "capabilities"`

Response payload should match `GET /coderai/capabilities`.

### `op = "register"`

Used for outbound-only broker registration and metadata refresh.

### `op = "proxy"`

Used to tunnel arbitrary Studio-native and compatible endpoints over broker or direct WebSocket transport.

Request payload may include:

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
    "fields": [
      {"name": "model", "value": "whisper-large"}
    ],
    "files": [
      {
        "name": "file",
        "filename": "sample.wav",
        "content_type": "audio/wav",
        "data_base64": "<base64>"
      }
    ]
  },
  "content_type": "multipart/form-data",
  "stream": true
}
```

Semantics:

- `headers`: forward arbitrary request headers when safe
- `query_params`: forward arbitrary query string values
- `body`: JSON body for non-multipart requests
- `multipart.fields`: repeated form fields
- `multipart.files`: uploaded files encoded in base64 with metadata
- `content_type`: original inbound content type if relevant
- `stream: true`: caller expects incremental response events instead of only a one-shot JSON body

## Response Envelope Types

### Non-streaming success

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
    "latency_ms": 842,
    "tokens_per_second": 17.8,
    "usage": {
      "prompt_tokens": 10,
      "completion_tokens": 5,
      "total_tokens": 15
    }
  }
}
```

AISBF keeps a rolling performance window for the latest 100 completed broker requests per connected session.

When exact metrics are present, AISBF stores them directly. Otherwise it estimates:

- latency from broker request start to final reply time
- throughput from `total_tokens / latency_seconds` when token counts are present

The resulting session snapshot tracks:

- average latency
- average throughput
- average total tokens
- success rate

CoderAI should prefer sending exact `latency_ms` and `tokens_per_second` whenever it can measure them internally. AISBF estimation is only a fallback for implementations that cannot yet provide exact metrics.

### Error

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

### Proxy success with JSON body

```json
{
  "v": 1,
  "request_id": "coderai-1746960000000",
  "status": "ok",
  "payload": {
    "status_code": 200,
    "headers": {
      "content-type": "application/json"
    },
    "body": {
      "job_id": "dub_123",
      "status": "queued"
    }
  }
}
```

### Proxy success with binary body

```json
{
  "v": 1,
  "request_id": "coderai-1746960000000",
  "status": "ok",
  "payload": {
    "status_code": 200,
    "content_type": "audio/mpeg",
    "headers": {
      "content-disposition": "attachment; filename=preview.mp3"
    },
    "body_base64": "<base64>"
  }
}
```

## Streaming Event Protocol

For long-running audio, image, video, or pipeline jobs, send multiple envelopes with the same `request_id`.

Supported event types include:

- `chunk`
- `progress`
- `output`
- `log`
- `data`
- final `done`
- final `completed`

### Chat streaming chunk example

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

### Progress event example

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

### Binary stream chunk example

```json
{
  "v": 1,
  "request_id": "coderai-1746960000000",
  "status": "ok",
  "event": "output",
  "payload": {
    "chunk": {
      "data_base64": "<base64>"
    }
  }
}
```

### Final event example

```json
{
  "v": 1,
  "request_id": "coderai-1746960000000",
  "status": "ok",
  "event": "done",
  "payload": {}
}
```

Important rules:

- For SSE-style consumers, `payload.chunk` should usually already be a complete SSE fragment formatted exactly as AISBF should relay it
- Include `data: [DONE]\n\n` when upstream semantics require it
- For binary chunks, send `payload.chunk.data_base64`
- Keep all events on the same `request_id`
- End the stream with `done` or `completed`

## Progress Endpoints Used by Studio Dashboard

The AISBF Studio dashboard may proxy these progress endpoints through the broker or direct WebSocket path:

- `GET /v1/audio/progress`
- `GET /v1/video/progress`
- `GET /v1/images/progress`

You should support them when the corresponding long-running media jobs exist.

Recommended JSON shape when called over HTTP:

```json
{
  "active": true,
  "current": 5,
  "total": 20,
  "pct": 25,
  "elapsed": 12,
  "it_per_s": 1.4,
  "unit": "steps"
}
```

Recommended streamed shape when called through broker streaming mode:

```json
{
  "v": 1,
  "request_id": "req-123",
  "status": "ok",
  "event": "progress",
  "payload": {
    "chunk": "event: progress\ndata: {\"active\":true,\"current\":5,\"total\":20,\"pct\":25,\"elapsed\":12,\"it_per_s\":1.4,\"unit\":\"steps\"}\n\n"
  }
}
```

## Capability Advertisement Requirements

Advertise endpoint metadata clearly so AISBF can reason about custom pipelines.

For each custom endpoint, provide as many of these as possible:

- `methods`
- `input_modalities`
- `output_modalities`
- `supports_stream`
- `supports_multipart`
- `supports_binary`
- optional model restrictions
- optional job semantics such as `returns_job_id` and `requires_progress_polling`

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
- endpoint capability metadata
- current server version
- optional hardware metadata such as `gpu`, `memory_gb`, `quantization`, `throughput_hint`

## Recommended CoderAI Architecture

1. OpenAI compatibility router
   - exposes `/v1/models`, `/v1/chat/completions`, and any other supported OpenAI endpoints
2. Studio-native router
   - exposes endpoints such as `v1/video/dub`, `v1/audio/tts`, `v1/images/generate`, progress endpoints, and other pipelines
3. Capabilities registry
   - enumerates enabled endpoints and loaded models
   - computes normalized `studio_capabilities`
   - exposes endpoint capability metadata
4. WebSocket bridge server
   - accepts AISBF envelopes
   - dispatches by `op`
   - handles `proxy` by internally calling the same handlers used by HTTP routes
   - handles chat and non-chat streaming events
5. Optional outbound broker client
   - maintains a persistent outbound WebSocket to AISBF-reachable broker endpoints

## Minimal Implementation Checklist

- add `/coderai/capabilities`
- add `/coderai/register` if you expose direct registration over HTTP
- add `/coderai/ws`
- expose model metadata with `studio_capabilities`
- support `models.list`
- support `chat.completions`
- support chat streaming `chunk` and `done`
- support `proxy` for Studio-native endpoints
- support arbitrary forwarded headers and query params in `proxy`
- support multipart uploads in `proxy`
- support base64 binary input and output in `proxy`
- support progress endpoints used by the AISBF Studio dashboard
- support non-chat streaming events for long-running media or pipeline jobs
- optionally support persistent outbound broker mode for NAT traversal
- protect bridge and register endpoints with a shared secret or signed token

## Compatibility Notes

- AISBF expects streamed chat chunks to already be formatted as SSE fragments when using chunk-style relay
- AISBF accepts binary stream chunks encoded with `data_base64`
- AISBF generic Studio proxy now uses the `coderai` bridge for non-chat endpoints, making NAT traversal possible for files, images, audio, video, and progress polling
- Owner isolation is enforced on the AISBF side, so correct scoped registration is mandatory
