# CoderAI API Documentation

This document describes the full HTTP API exposed by CoderAI, including OpenAI-compatible endpoints, native multimodal endpoints, profile/LoRA APIs, pipelines, admin APIs, examples, and end-to-end workflows.

The API is implemented with FastAPI in `codai/api/app.py` and routers under `codai/api/`,
with admin routes under `codai/admin/routes.py` and the front proxy's own routes under
`codai/frontproxy/`. Which process answers matters when you are debugging: a request to a
`/v1` generation route is accepted by the torch-free **front** and proxied to an **engine**
subprocess, while the admin pages, `/metrics` and the cluster mesh are answered by the front
itself — see [`docs/frontend-engine-split.md`](docs/frontend-engine-split.md). The
[Endpoint Index](#endpoint-index) at the end names the owning module for every route.

Interactive, always-up-to-date OpenAPI docs are served by the running server at
`/docs` (Swagger UI, linked as **API Docs** in the admin nav) and `/redoc`, with the
raw schema at `/openapi.json`. Every endpoint there carries a tag, summary, and
per-field descriptions generated from the code. Routes the code registers with
`include_in_schema=False` — liveness probes, `/boot`, `/logs`, `/metrics`, the cluster and
internal control planes — are **not** in that schema; they are documented here instead.

The subsystems with a design of their own have a deep-dive guide in
[`docs/`](docs/README.md): OCR, chunking, moderation, video composition, the engines,
RunPod, the cluster, isolated worker environments. This document is the HTTP surface; those
explain the behaviour behind it.

## Base URL

Default local server:

```text
http://127.0.0.1:8776
```

Most client calls use the `/v1` prefix:

```text
http://127.0.0.1:8776/v1
```

## Authentication

CoderAI supports web sessions and API bearer tokens.

For `/v1/*` routes, send:

```http
Authorization: Bearer <api-token>
```

Token management is available in the admin UI and admin API:

- `GET /admin/tokens`
- `GET /admin/api/tokens`
- `POST /admin/api/tokens`

Notes:

- `/v1/images/progress` is explicitly exempt from bearer auth in middleware.
- If the admin/session manager is not initialized, API auth can be bypassed by the server.
- Admin HTML/API routes use signed session cookies; many admin API routes require an admin role.
- Some profile routes also enforce local API auth internally.
- The telemetry and dashboard reads (`/admin/api/engines`, `/admin/api/gpu-stats`,
  `/admin/api/system-stats`, `/admin/api/usage`, `/metrics`) take a *light* credential —
  a session cookie or bearer that is present and well-formed — deliberately, so the
  operator's view survives an engine that is busy generating.
- `/metrics` is open to anyone when `server.metrics_public` is set.
- `/cluster/*` is authenticated with the **shared cluster token**, not a user session.
- `/internal/*` is the front↔engine control plane and is not meant for clients.
- On a rented pod the bearer token (`CODERAI_API_TOKEN`) is required for `/boot` and
  `/logs` too, so neither is a way to read an install's log without one.

Example reusable shell variables:

```bash
export CODERAI_URL="http://127.0.0.1:8776"
export CODERAI_TOKEN="your-api-token"
```

Example JSON request:

```bash
curl -s "$CODERAI_URL/v1/models" \
  -H "Authorization: Bearer $CODERAI_TOKEN"
```

## Common Data Conventions

### Media Inputs

Media fields usually accept either:

- A URL: `http://...`, `https://...`, or a CoderAI file URL such as `/v1/files/output.png`
- Raw base64 without a data URL prefix
- Data URLs such as `data:image/png;base64,...`, `data:video/mp4;base64,...`, `data:audio/wav;base64,...`

### Media Outputs

Generation endpoints typically return:

```json
{
  "created": 1781090000,
  "data": [
    {
      "url": "/v1/files/generated.png"
    }
  ]
}
```

If `response_format` requests base64, the first data item uses a media-specific key:

- Images: `b64_json`
- Video: `b64_mp4`
- Audio: `b64_wav` or `b64_mp3`

### Progress Polling

Long-running image, video, audio, and LoRA jobs expose polling endpoints. Typical progress response:

```json
{
  "current": 12,
  "total": 30,
  "active": true,
  "phase": "generating",
  "model": "model-id",
  "pct": 40.0,
  "it_per_s": 1.3,
  "elapsed": 8.9
}
```

### Extra Fields

Most request models allow extra JSON fields (`extra="allow"`). This makes the API tolerant of OpenAI-compatible or Studio-style client parameters even when a specific route ignores them.

## Core Endpoints

### List Models

`GET /v1/models`

Returns configured models and metadata.

Response shape:

```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen/Qwen3-8B",
      "object": "model",
      "created": 1781090000,
      "owned_by": "huggingface",
      "type": "text",
      "capabilities": ["text_generation"],
      "backend": "cuda",
      "model_path": "Qwen/Qwen3-8B",
      "alias": "qwen3"
    }
  ]
}
```

A **video** model carries one extra object, `video`, describing what it can actually
emit — so a client can plan around the model instead of guessing from its name:

```json
{
  "id": "longcat",
  "type": "video",
  "capabilities": ["video_generation"],
  "video": {
    "family": "longcat",
    "native_fps": 15,
    "frame_base": 93,
    "frame_step": 80,
    "min_frames": 93,
    "max_frames": null,
    "max_frames_per_render": 93,
    "segment_frames": 93,
    "cond_frames": 13,
    "side_multiple": 64,
    "continuation": "native"
  }
}
```

| Field | Meaning |
|---|---|
| `native_fps` | The rate the model was trained to produce |
| `frame_base`, `frame_step` | Legal TOTAL frame counts are `frame_base + frame_step * k`. Wan is `1 + 4k` (its VAE's 4n+1), LongCat `93 + 80k` (one 93-frame segment, each continuation re-rendering 13 of them and so adding 80), MiniMax-H3 `5 + 17k` |
| `min_frames`, `max_frames` | Limits on the total; `null` = none (LongCat's segment loop runs as long as asked) |
| `max_frames_per_render` | How much ONE model call holds. Wan is ~81; past that the frames visibly jump, so longer clips must be chained |
| `segment_frames`, `cond_frames` | For a natively-continuing model: the segment length and how much of it re-renders the previous tail |
| `side_multiple` | Width and height should be multiples of this, or the model falls back to a slower attention path (LongCat at 480 silently loses block-sparse attention) |
| `continuation` | `native` (the server generates a long shot in segments — ask for the whole length in one request), `chained` (the client extends from a frame tail), `none` |

A count off the grid is **rounded by the server**, so a request for 250 LongCat frames
comes back a different length than it asked for. Snap client-side and the plan and the
output agree. Per-model overrides in `models.json`
(`video_native_fps`, `video_frame_base`, `video_frame_step`, `video_min_frames`,
`video_max_frames`, `video_max_frames_per_render`, `video_side_multiple`,
`video_segment_frames`, `video_cond_frames`) win over the built-in family defaults, so a
checkpoint the server has never seen before is configurable rather than mis-described.

Example:

```bash
curl -s "$CODERAI_URL/v1/models" \
  -H "Authorization: Bearer $CODERAI_TOKEN" | jq
```

### Capabilities Document

`GET /coderai/capabilities`

Returns CoderAI broker/studio capability metadata and hardware summary. This endpoint is used by AISBF and discovery integrations.

Example:

```bash
curl -s "$CODERAI_URL/coderai/capabilities" | jq
```

### Serve Generated Files

`GET /v1/files/{filename}`

Returns a generated or uploaded file from the configured output directory. Path traversal is rejected.

Example:

```bash
curl -L "$CODERAI_URL/v1/files/generated.png" \
  -H "Authorization: Bearer $CODERAI_TOKEN" \
  -o generated.png
```

### File Archive

`GET /v1/archive`

Lists generated media in the output/archive directory.

```json
{
  "files": [
    {
      "filename": "image_001.png",
      "type": "image",
      "size": 123456,
      "created": 1781090000,
      "url": "/v1/files/image_001.png"
    }
  ]
}
```

`DELETE /v1/archive/{filename}` deletes an archived file.

```bash
curl -X DELETE "$CODERAI_URL/v1/archive/image_001.png" \
  -H "Authorization: Bearer $CODERAI_TOKEN"
```

## Text Generation

CoderAI exposes OpenAI-compatible chat and legacy completion APIs.

### Chat Completions

`POST /v1/chat/completions`

Request fields:

| Field | Type | Default | Description |
|---|---:|---:|---|
| `model` | string | required | Model id from `/v1/models` |
| `messages` | array | required | Chat messages with `role` and `content` |
| `temperature` | number | `0.7` | Sampling temperature |
| `top_p` | number | `1.0` | Nucleus sampling |
| `n` | integer | `1` | Number of completions |
| `max_tokens` | integer/null | `null` | Max generated tokens |
| `stream` | boolean | `false` | Return SSE chunks |
| `stop` | string/array/null | `null` | Stop sequence(s) |
| `presence_penalty` | number | `0.0` | OpenAI-compatible field |
| `frequency_penalty` | number | `0.0` | OpenAI-compatible field |
| `repeat_penalty` | number | `1.0` | Repetition penalty |
| `tools` | array/null | `null` | Function/tool definitions |
| `tool_choice` | string/object/null | `auto` | Tool selection control |
| `enable_thinking` | boolean | `false` | Enables reasoning/thinking templates where supported |
| `response_format` | object/null | `null` | Accepted for compatibility |

Basic request:

```bash
curl -s "$CODERAI_URL/v1/chat/completions" \
  -H "Authorization: Bearer $CODERAI_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B",
    "messages": [
      {"role": "system", "content": "You are concise."},
      {"role": "user", "content": "Explain VRAM offloading in one paragraph."}
    ],
    "temperature": 0.4,
    "max_tokens": 300
  }' | jq
```

Response shape:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1781090000,
  "model": "Qwen/Qwen3-8B",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "VRAM offloading..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 42,
    "completion_tokens": 80,
    "total_tokens": 122
  }
}
```

Streaming request:

```bash
curl -N "$CODERAI_URL/v1/chat/completions" \
  -H "Authorization: Bearer $CODERAI_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B",
    "messages": [{"role": "user", "content": "Write a haiku about GPUs."}],
    "stream": true
  }'
```

Streaming responses use server-sent event style lines:

```text
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[...]}

data: [DONE]
```

Tool calling example:

```bash
curl -s "$CODERAI_URL/v1/chat/completions" \
  -H "Authorization: Bearer $CODERAI_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B",
    "messages": [{"role": "user", "content": "What is the weather in Rome?"}],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get current weather for a city",
          "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
          }
        }
      }
    ],
    "tool_choice": "auto"
  }'
```

### Legacy Completions

`POST /v1/completions`

Request fields are similar to OpenAI legacy completions:

| Field | Type | Default |
|---|---:|---:|
| `model` | string | required |
| `prompt` | string or string[] | required |
| `temperature` | number | `0.7` |
| `top_p` | number | `1.0` |
| `n` | integer | `1` |
| `max_tokens` | integer/null | `null` |
| `stream` | boolean | `false` |
| `stop` | string/array/null | `null` |
| `repeat_penalty` | number | `1.0` |

Example:

```bash
curl -s "$CODERAI_URL/v1/completions" \
  -H "Authorization: Bearer $CODERAI_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B",
    "prompt": "The fastest way to reduce inference memory is",
    "max_tokens": 120
  }' | jq
```

## Images

### Image Progress

`GET /v1/images/progress`

Returns the current image-generation progress. This route is exempt from bearer auth in middleware.

```bash
curl -s "$CODERAI_URL/v1/images/progress" | jq
```

### Generate Images

`POST /v1/images/generations`

Request fields:

| Field | Type | Default | Description |
|---|---:|---:|---|
| `model` | string | required | Image model id |
| `prompt` | string | required | Positive prompt |
| `n` | integer | `1` | Number of images |
| `size` | string | `1024x1024` | Output size |
| `steps` | integer/null | model default | Inference steps |
| `guidance_scale` | number/null | model default | CFG/guidance |
| `quality` | string | `standard` | Compatibility field |
| `style` | string/null | `null` | Compatibility/style field |
| `response_format` | string | `url` | `url` or `b64_json` |
| `seed` | integer/null | random | Deterministic seed |
| `negative_prompt` | string/null | `null` | Negative prompt |
| `disable_safety_checker` | boolean | `false` | Null the diffusers safety checker (only affects SD 1.x/2.x; SDXL/Flux ship none) |
| `vae_model` | string/null | `null` | Per-request VAE override |
| `loras` | array/null | `null` | LoRA adapters — see [LoRA references](#lora-references-in-requests) for all supported fields |
| `character_profiles` | string[]/null | `null` | Saved character profile names |
| `character_references` | string[]/null | `null` | Inline reference images |
| `character_strength` | number | `0.6` | IP-Adapter/reference strength |
| `environment_profiles` | string[]/null | `null` | Saved environment profile names |

Example:

```bash
curl -s "$CODERAI_URL/v1/images/generations" \
  -H "Authorization: Bearer $CODERAI_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "stabilityai/stable-diffusion-xl-base-1.0",
    "prompt": "cinematic photo of a brass robot botanist in a glass greenhouse, morning mist",
    "negative_prompt": "blurry, low quality, distorted hands",
    "size": "1024x1024",
    "steps": 30,
    "guidance_scale": 7.0,
    "seed": 12345,
    "response_format": "url"
  }' | jq
```

LoRA example (the weights can also be sent inline or by registry id — see
[LoRA references](#lora-references-in-requests)):

```json
{
  "model": "image-model",
  "prompt": "portrait of <character-token> as a space pilot",
  "loras": [
    {"id": "name:space_uniform", "weight": 0.8, "name": "uniform"}
  ]
}
```

Character/environment consistency example:

```json
{
  "model": "image-model",
  "prompt": "Alice explores the old library at sunset",
  "character_profiles": ["Alice"],
  "environment_profiles": ["OldLibrary"],
  "character_strength": 0.75,
  "size": "1024x1024"
}
```

### Edit Image

`POST /v1/images/edits`

Fields:

- `model` required
- `prompt` required
- `image` required, base64/URL source image
- `mask` optional
- `n`, `size`, `response_format`, `strength`, `steps`, `guidance_scale`, `seed`, `quality`

```bash
curl -s "$CODERAI_URL/v1/images/edits" \
  -H "Authorization: Bearer $CODERAI_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "image-edit-model",
    "image": "data:image/png;base64,...",
    "prompt": "turn the sky into dramatic storm clouds",
    "strength": 0.55,
    "response_format": "url"
  }'
```

### Inpaint Image

`POST /v1/images/inpaint`

Like edits, but `mask` is required.

```json
{
  "model": "inpaint-model",
  "image": "data:image/png;base64,...",
  "mask": "data:image/png;base64,...",
  "prompt": "replace the masked area with a carved wooden door",
  "strength": 0.99,
  "steps": 30,
  "response_format": "url"
}
```

### Upscale Image

`POST /v1/images/upscale`

```json
{
  "model": "realesrgan-x4plus",
  "image": "data:image/png;base64,...",
  "scale": 4,
  "response_format": "url"
}
```

### Depth Map

`POST /v1/images/depth`

```json
{
  "model": "depth-anything",
  "image": "data:image/png;base64,...",
  "response_format": "url"
}
```

### Segment Image

`POST /v1/images/segment`

```json
{
  "model": "sam-vit-h",
  "image": "data:image/png;base64,...",
  "points": [[420, 300]],
  "boxes": [[100, 100, 600, 700]],
  "response_format": "url"
}
```

### Deblur Image

`POST /v1/images/deblur`

```json
{
  "image": "data:image/png;base64,...",
  "strength": 0.5,
  "response_format": "url"
}
```

### Unpixelate Image

`POST /v1/images/unpixelate`

```json
{
  "model": "realesrgan-x4plus",
  "image": "data:image/png;base64,...",
  "scale": 4,
  "response_format": "url"
}
```

### Outfit Change

`POST /v1/images/outfit`

Fields:

- `model` required
- `image` or `video` optional input
- `prompt` required outfit/clothing description
- `negative_prompt`, `mask`, `steps`, `guidance_scale`, `strength`, `seed`, `response_format`

```json
{
  "model": "inpaint-model",
  "image": "data:image/png;base64,...",
  "prompt": "tailored navy velvet evening suit with silver embroidery",
  "negative_prompt": "distorted body, extra limbs",
  "steps": 30,
  "guidance_scale": 7.5,
  "strength": 0.92,
  "response_format": "url"
}
```

### Face Swap

`POST /v1/images/faceswap`

```json
{
  "source_face": "data:image/png;base64,...",
  "target": "data:image/png;base64,...",
  "target_type": "image",
  "response_format": "url"
}
```

For video targets, use `target_type: "video"`.

## Video

### Video Progress

`GET /v1/video/progress`

```bash
curl -s "$CODERAI_URL/v1/video/progress" \
  -H "Authorization: Bearer $CODERAI_TOKEN" | jq
```

### Generate Video

`POST /v1/video/generations`

Primary fields:

| Field | Type | Default | Description |
|---|---:|---:|---|
| `model` | string | required | Video model id |
| `prompt` | string | `""` | Text prompt |
| `negative_prompt` | string/null | `null` | Negative prompt |
| `width` | integer | `512` | Width |
| `height` | integer | `512` | Height |
| `num_frames` | integer/null | model default | Frame count |
| `fps` | integer/null | model default | Frames per second |
| `num_inference_steps` | integer/null | model default | Diffusion steps |
| `guidance_scale` | number/null | model default | CFG/guidance |
| `seed` | integer/null | random | Seed |
| `mode` | string | `t2v` | `t2v`, `i2v` (init_image, prompt dropped), `ti2v` (init_image + prompt), `v2v`, `interp`. The server gracefully falls back between Wan t2v/i2v pipelines when a model supports only one. |
| `image` / `init_image` | string/null | `null` | Initial/reference frame |
| `end_image` | string/null | `null` | End frame for interpolation |
| `video` | string/null | `null` | Input video for v2v/post-processing |
| `strength` | number/null | `null` | Denoising strength |
| `camera_motion` | string/null | `null` | `zoom-in`, `pan-left`, etc. |
| `character_profiles` | string[]/null | `null` | Saved character profiles |
| `loras` | array/null | `null` | Video LoRA adapters — see [LoRA references](#lora-references-in-requests) |
| `disable_safety_checker` | boolean | `false` | Null the diffusers safety checker (no effect on models without one, e.g. Wan) |
| `response_format` | string | `url` | `url` or `b64_mp4` |

Text-to-video example:

```bash
curl -s "$CODERAI_URL/v1/video/generations" \
  -H "Authorization: Bearer $CODERAI_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "video-model",
    "mode": "t2v",
    "prompt": "a slow dolly shot through a neon market in the rain",
    "negative_prompt": "low quality, flicker",
    "width": 768,
    "height": 432,
    "num_frames": 49,
    "fps": 12,
    "num_inference_steps": 30,
    "guidance_scale": 6.0,
    "seed": 9001,
    "response_format": "url"
  }' | jq
```

Image-to-video example:

```json
{
  "model": "i2v-model",
  "mode": "i2v",
  "prompt": "gentle camera push-in, hair and fabric moving in the wind",
  "init_image": "data:image/png;base64,...",
  "num_frames": 32,
  "fps": 8,
  "camera_motion": "zoom-in",
  "response_format": "url"
}
```

Video with generated audio, subtitles, dub, and post-processing:

```json
{
  "model": "video-model",
  "prompt": "a robot chef prepares pasta in a futuristic kitchen",
  "mode": "t2v",
  "num_frames": 49,
  "fps": 12,
  "add_audio": true,
  "audio_type": "ambient",
  "audio_prompt": "soft kitchen ambience, gentle synth pad",
  "generate_subtitles": true,
  "burn_subtitles": true,
  "subtitle_style": "minimal",
  "upscale_output": true,
  "upscale_factor": 2,
  "interpolate_output": true,
  "fps_multiplier": 2,
  "response_format": "url"
}
```

Multi-character dialog example:

```json
{
  "model": "video-model",
  "prompt": "two detectives talk in a dim archive room",
  "character_profiles": ["DetectiveA", "DetectiveB"],
  "dialogs": [
    {"character": "DetectiveA", "voice": "narrator_a", "text": "The file was never missing.", "lip_sync": true},
    {"character": "DetectiveB", "voice": "narrator_b", "text": "Then someone wanted us to think it was.", "lip_sync": true}
  ],
  "burn_subtitles": true,
  "response_format": "url"
}
```

### Upscale Video

`POST /v1/video/upscale`

```json
{
  "model": "realesrgan-video",
  "video": "data:video/mp4;base64,...",
  "upscale_factor": 2,
  "response_format": "url"
}
```

### Subtitle Video

`POST /v1/video/subtitle`

```json
{
  "model": "whisper-large-v3",
  "video": "data:video/mp4;base64,...",
  "language": "en",
  "translate": true,
  "target_lang": "it",
  "burn": false,
  "style": "default",
  "response_format": "srt"
}
```

`response_format` can be `srt`, `vtt`, `json`, or `burned_video`.

### Interpolate Video or Frames

`POST /v1/video/interpolate`

```json
{
  "model": "rife",
  "video": "data:video/mp4;base64,...",
  "fps_multiplier": 2,
  "response_format": "url"
}
```

Frame interpolation:

```json
{
  "model": "rife",
  "init_image": "data:image/png;base64,...",
  "end_image": "data:image/png;base64,...",
  "fps_multiplier": 4,
  "response_format": "url"
}
```

### Dub Video

`POST /v1/video/dub`

```json
{
  "model": "whisper-large-v3",
  "video": "data:video/mp4;base64,...",
  "source_lang": "en",
  "target_lang": "es",
  "voice_clone": true,
  "burn_subtitles": true,
  "response_format": "url"
}
```

## Video Composition

`POST /v1/video/compose` turns N pieces into one finished video. CoderAI could already
make every piece of a short reel; nothing turned the pieces into a reel, because that step
needs ffmpeg *and* facts only the server has — how long the narration it just synthesised
actually is, and when each word falls. A client on another machine cannot know either, so
composition lives here.

A request is an ordered list of **scenes**. Each scene has narration (text to synthesise,
or audio you already have) and one or more **visuals**; the scene lasts as long as its
narration plus its padding, and its visuals split that time between them. Voice, music,
captions, overlays and thumbnail are defaults you override per request.

| Field | Type | Default | Description |
|---|---:|---:|---|
| `scenes` | object[] | required | Ordered scenes (see below) |
| `canvas` | object | `1080×1920@30` | `width`, `height`, `fps`, `background` |
| `voice` | object | `engine: tts` | Default `VoiceSpec`: `engine` (`tts`/`clone`), `model`, `voice`, `voice_name`, `language`, `speed` |
| `visual_fit` | string | `cover` | `cover`, `contain` or `blur_pad` |
| `transition` | object | `{"type":"none","duration":0.3}` | Between scenes |
| `music` | object/null | `null` | Bed track, ducked under narration |
| `presenter` | object/null | `null` | A lip-synced presenter over the whole composition |
| `captions` | object/null | `null` | Burned-in captions from the real word timings |
| `overlays` | object[] | `[]` | Logos, lower thirds, text |
| `thumbnail` | object/null | `null` | Still frame to emit alongside the video |
| `output` | object | codec defaults | Container, codec, bitrate |
| `outputs` | string[] | `["video"]` | Which artefacts to produce |
| `async` | boolean | `true` | `true` queues and returns `202 {id, status}`; `false` blocks and returns the finished job |

Scene fields: `text` or `audio` (narration), `duration` (required when there is no
narration), `min_duration`, `padding`, `visuals[]`, `voice` (per-scene override — a second
speaker), `words[]` (caller-supplied word timings), `speakers[]` (a multi-character dialog:
who speaks when, each with their own voice and place on screen), `audio_mix`,
`clip_volume`, `presenter`.

Visual fields: `type` (`video`/`image`/`gradient`/`color`), `src`, `trim_start`,
`ken_burns`, `colors`, `color`, `keep_audio`, `audio_src`, `volume`.

Work is queued and rendered one at a time (a render is ffmpeg-bound; two at once are slower
than one after the other). A visual that cannot be fetched or decoded never fails the job:
it becomes a gradient and a line in `warnings`.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/video/compose` | Submit a composition |
| `GET` | `/v1/video/compose` | List composition jobs |
| `GET` | `/v1/video/compose/{job_id}` | Job status: stage, percentage, outputs, `warnings` |
| `GET` | `/v1/video/compose/{job_id}/log` | That job's own log |
| `POST` | `/v1/video/compose/{job_id}/cancel` | Stop it |

```bash
curl -s "$CODERAI_URL/v1/video/compose" \
  -H "Authorization: Bearer $CODERAI_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "canvas": {"width": 1080, "height": 1920, "fps": 30},
    "voice": {"engine": "tts", "model": "tts-1", "voice": "alloy"},
    "captions": {"enabled": true},
    "scenes": [
      {"text": "First, the setup.",
       "visuals": [{"type": "image", "src": "/v1/files/shot1.png",
                    "ken_burns": "in"}]},
      {"text": "Then the payoff.",
       "visuals": [{"type": "video", "src": "/v1/files/clip.mp4"}]}
    ]
  }' | jq
# → 202 {"id":"cmp_…","status":"queued"}
```

### Talking heads and presenters

`POST /v1/video/talking-head` is the one-shot form: a character (a saved profile or an
explicit portrait) saying one line, lip-synced — it IS a one-scene composition with a
full-screen presenter, so it runs on exactly the same code. It takes `character` or
`image`, `text` or `audio`, `voice`, `engine`, `motion` (`still` by default for a preview,
`generate` or `src` otherwise), `canvas` (720×720@25) and `async`; a preview is capped at
15 seconds. A `presenter` block on a compose request or scene is the same machinery inside
a larger composition, laid out `full`, `pip` or `split`.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/video/talking-head` | A character saying one line |
| `GET` | `/v1/video/talking-head/{job_id}` | Job status |
| `POST` | `/v1/video/talking-head/{job_id}/cancel` | Cancel it |
| `GET` | `/v1/video/presenter/engines` | Which lip-sync engines are installed (`sadtalker`, `wav2lip`, …) |

A lip-sync engine is an optional install, so ask `/v1/video/presenter/engines` before
promising a presenter; `engine: "auto"` picks whatever is there.

### Uploading media once

A client that is not on this machine has media of its own: footage, a logo, a music bed.
Inlining a 40 MB clip as base64 in each of ten renders is exactly the waste that makes an
API unusable.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/files/upload` | Upload media; returns a `sha256:` id and a `/v1/files/...` URL |
| `GET` | `/v1/files/blob/{hash}` | Does this content already exist here? (`200`/`404`) |

Storage is content-addressed on the sha256 of the bytes, so re-uploading the same file is
free and a client can ask before sending. Full detail in
[`docs/video-compose.md`](docs/video-compose.md).

## Audio

### Transcriptions

`POST /v1/audio/transcriptions`

This is an OpenAI-style multipart form endpoint.

Form fields:

| Field | Type | Default | Description |
|---|---:|---:|---|
| `model` | string | required | Whisper/transcription model |
| `file` | file | required | Audio/video file upload |
| `language` | string/null | `null` | Language hint |
| `prompt` | string/null | `null` | Context prompt |
| `response_format` | string | `json` | `json`, `verbose_json`, `text`, `srt`, `vtt` |
| `temperature` | number | `0.0` | Decoding temperature |

Example:

```bash
curl -s "$CODERAI_URL/v1/audio/transcriptions" \
  -H "Authorization: Bearer $CODERAI_TOKEN" \
  -F model="whisper-large-v3" \
  -F file=@speech.wav \
  -F language="en" \
  -F response_format="json" | jq
```

Text-only response:

```bash
curl -s "$CODERAI_URL/v1/audio/transcriptions" \
  -H "Authorization: Bearer $CODERAI_TOKEN" \
  -F model="whisper-large-v3" \
  -F file=@speech.wav \
  -F response_format="text"
```

### Text-to-Speech

`POST /v1/audio/speech`

Request fields:

- `model` required
- `input` required text
- `voice` default `af_sarah`
- `response_format` default `mp3`
- `speed` default `1.0`
- `voice_profile` optional saved profile name

Response:

```json
{
  "audio": "<base64-audio>"
}
```

Example:

```bash
curl -s "$CODERAI_URL/v1/audio/speech" \
  -H "Authorization: Bearer $CODERAI_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kokoro",
    "input": "Local inference is online.",
    "voice": "af_sarah",
    "response_format": "mp3",
    "speed": 1.0
  }' | jq -r .audio | base64 -d > speech.mp3
```

### Audio Generation Progress

`GET /v1/audio/progress`

```bash
curl -s "$CODERAI_URL/v1/audio/progress" \
  -H "Authorization: Bearer $CODERAI_TOKEN" | jq
```

### Generate Audio / Music / SFX

`POST /v1/audio/generate`

Request fields:

| Field | Type | Default |
|---|---:|---:|
| `model` | string | required |
| `prompt` | string | required |
| `duration` | number | `10.0` |
| `top_k` | integer | `250` |
| `top_p` | number | `0.0` |
| `temperature` | number | `1.0` |
| `cfg_coef` | number | `3.0` |
| `seed` | integer/null | `null` |
| `melody` | string/null | `null` |
| `voice_profile` | string/null | `null` |
| `response_format` | string | `url` |

Example:

```json
{
  "model": "facebook/musicgen-medium",
  "prompt": "warm lo-fi loop with brushed drums and soft Rhodes chords",
  "duration": 12,
  "temperature": 1.0,
  "cfg_coef": 3.0,
  "seed": 44,
  "response_format": "url"
}
```

Melody-conditioned example:

```json
{
  "model": "facebook/musicgen-melody",
  "prompt": "cinematic orchestral arrangement of the melody",
  "melody": "data:audio/wav;base64,...",
  "duration": 20,
  "response_format": "url"
}
```

### Voice Profiles

List voices:

`GET /v1/audio/voices`

Create voice profile:

`POST /v1/audio/voices`

Multipart fields:

- `name`
- `transcript`
- `description`
- `audio` file

```bash
curl -s "$CODERAI_URL/v1/audio/voices" \
  -H "Authorization: Bearer $CODERAI_TOKEN" \
  -F name="narrator_a" \
  -F transcript="This is the exact reference transcript." \
  -F description="Warm narrator voice" \
  -F audio=@reference.wav | jq
```

Get, patch, delete:

- `GET /v1/audio/voices/{name}`
- `PATCH /v1/audio/voices/{name}`
- `DELETE /v1/audio/voices/{name}`

Extract a voice profile from audio or video:

`POST /v1/audio/voices/extract`

```json
{
  "name": "speaker_from_clip",
  "description": "Extracted from interview clip",
  "video": "data:video/mp4;base64,...",
  "transcript": "Optional exact transcript for the selected speech segment."
}
```

### Voice Clone

`POST /v1/audio/clone`

Fields:

- `text` required output text
- `voice_name` optional saved profile
- `ref_audio` and `ref_text` optional inline reference
- `speed`, `seed`, `response_format`

Using saved voice:

```json
{
  "text": "The archive doors opened at midnight.",
  "voice_name": "narrator_a",
  "speed": 0.95,
  "seed": 10,
  "response_format": "url"
}
```

Using inline reference:

```json
{
  "text": "The system is ready.",
  "ref_audio": "data:audio/wav;base64,...",
  "ref_text": "This is the reference speaker transcript.",
  "response_format": "b64_wav"
}
```

### Voice Conversion

`POST /v1/audio/convert`

Fields:

- `source_audio` required
- `target_voice` or `voice_name` optional
- `f0_condition` singing-mode pitch conditioning
- `pitch_shift`
- `diffusion_steps`
- `length_adjust`
- `inference_cfg_rate`
- `response_format`

```json
{
  "source_audio": "data:audio/wav;base64,...",
  "voice_name": "singer_a",
  "f0_condition": true,
  "pitch_shift": 0,
  "diffusion_steps": 20,
  "response_format": "url"
}
```

### Audio Stems

`POST /v1/audio/stems`

```json
{
  "audio": "data:audio/wav;base64,...",
  "stem_mode": "vocals-instrumental",
  "response_format": "url",
  "fallback_mode": true
}
```

Supported requested split modes include:

- `vocals-instrumental`
- `4-stem`
- `drums-bass-other`

### Audio Cleanup

`POST /v1/audio/cleanup`

```json
{
  "audio": "data:audio/wav;base64,...",
  "noise_reduction": true,
  "normalize": true,
  "remove_hum": true,
  "repair_clicks": false,
  "response_format": "url",
  "fallback_mode": true
}
```

## Speaker Recognition and Diarization

Diarization answers *who spoke when*; speaker recognition answers *who that is*. They are
separate models and separate calls, and they compose: enroll a voiceprint once, then ask
diarization to label the turns with the enrolled names instead of `SPEAKER_00`.

### Diarization

`POST /v1/audio/diarization` (multipart)

| Field | Type | Default | Description |
|---|---:|---:|---|
| `file` | file | required | The audio |
| `model` | string/null | config default | Diarization model (pyannote-family) |
| `num_speakers` | integer/null | `null` | Exact count, when you know it |
| `min_speakers` / `max_speakers` | integer/null | `null` | Bounds, when you don't |
| `response_format` | string | `json` | `json` or `verbose_json` |
| `identify` | boolean | `false` | Relabel each turn with an enrolled speaker name |
| `identify_backend` | string | `ecapa` | Voiceprint backend: `ecapa` (speechbrain ECAPA-TDNN, ungated, 192-d), `pyannote` (may be gated, uses `HF_TOKEN`), `wespeaker` |
| `identify_threshold` | float | `0.25` | Cosine similarity below which a turn stays anonymous |

The same work is available inline on transcription: `POST /v1/audio/transcriptions` with
`diarize=true` (plus `num_speakers`, and `timestamp_granularities` for word timings)
returns speaker-labelled segments instead of flat text.

### Enrolled speakers

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/audio/speakers` | Enroll a voiceprint (`file`, `name`, `backend`, `model`) |
| `GET` | `/v1/audio/speakers` | List enrolled speakers |
| `DELETE` | `/v1/audio/speakers/{name}` | Delete one |
| `POST` | `/v1/audio/speaker-identify` | Who is this? (`file`, `backend`, `threshold`) → best match or none |
| `POST` | `/v1/audio/speaker-verify` | Is this the same speaker? (`file`, `name` **or** `reference`, `threshold`) → accept/reject + score |
| `POST` | `/v1/audio/speaker-embeddings` | The raw voiceprint vector(s) (`file`, `backend`, `model`, `window`, `step`) |

`speaker-embeddings` is the building block: with `window`/`step` it returns a vector per
window rather than one for the clip, which is what you want for your own clustering. All of
these run in the isolated pyannote venv and share its worker — and its VRAM-eviction
registration — with diarization, so enrolling and diarizing do not load two copies.

```bash
# enroll, then label a meeting's turns with real names
curl -s "$CODERAI_URL/v1/audio/speakers" \
  -H "Authorization: Bearer $CODERAI_TOKEN" \
  -F "file=@alice.wav" -F "name=alice" | jq

curl -s "$CODERAI_URL/v1/audio/diarization" \
  -H "Authorization: Bearer $CODERAI_TOKEN" \
  -F "file=@meeting.wav" -F "identify=true" -F "identify_threshold=0.3" | jq
```

An identification below the threshold is reported as unknown rather than as the nearest
name: a wrong name on a turn is worse than no name.

#### Verifying against a voiceprint you hold yourself

`speaker-verify` names its reference one of two ways, and exactly one must be present:

| Field | Type | Description |
|---|---:|---|
| `name` | string | An enrolled speaker. Looked up for the backend the clip was embedded with, so enrol and verify must pass the same `backend` |
| `reference` | string (JSON) | A voiceprint the **client** holds: a bare array, `{"embedding": [...]}`, or the whole response `speaker-embeddings` returned |

A caller that keeps its own voiceprints — because the speakers are its users, not this
install's operators — should not have to enrol them here to compare one, nor reimplement
cosine similarity and pick a threshold alone. With `reference` nothing is stored and
nothing is enrolled; the answer adds `"reference": "vector"` (against `"enrolled"` for the
`name` path) so a client can tell which comparison it got.

```bash
# keep the vector
REF=$(curl -s "$CODERAI_URL/v1/audio/speaker-embeddings" \
  -H "Authorization: Bearer $CODERAI_TOKEN" -F "file=@alice.wav" | jq -c .)
# compare a later clip against it, enrolling nothing
curl -s "$CODERAI_URL/v1/audio/speaker-verify" \
  -H "Authorization: Bearer $CODERAI_TOKEN" \
  -F "file=@unknown.wav" -F "reference=$REF" -F "threshold=0.45" | jq
```

A `reference` carries no record of which model produced it. A dimension mismatch is
refused with `400` naming both dimensions, because a 192-d ECAPA vector against a 512-d one
is a different space rather than a near miss. Two **same**-dimension vectors from different
backends cannot be caught that way and would compare as confident noise, so pairing the
reference with the right `backend` is the caller's responsibility. A malformed `reference`
is a `400` before the clip is embedded — a request that cannot succeed does not pay for a
GPU round-trip.

About the threshold: `0.25` is the shipped default and is deliberately permissive, which
suits labelling a meeting where a missed name costs little. Access control is the opposite
trade — a false accept is the expensive error — so raise it, and calibrate rather than
guess: `speaker-identify` returns `score` and ranked `candidates`, so record clips of the
target and of other people, look at the two score distributions, and put the threshold in
the gap between them.

### Audio engine discovery

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/v1/audio/speech/capabilities` | What a TTS model can steer (`?model=`): emotions, delivery styles, built-in voices, and whether its venv is ready |
| `GET` | `/v1/audio/clone/engines` | Which cloning engines this install can actually run, and what each needs |
| `POST` | `/v1/audio/watermark/detect` | `{"audio": "<base64 / data URI / URL>"}` — did this clip come out of this server? |

Ask `speech/capabilities` instead of assuming: the emotion and style vocabularies differ
per TTS family, and a parameter a model does not support is ignored rather than honoured.
It answers without loading anything — an already-loaded model is asked directly, otherwise
the answer is what was remembered from the last load, and `voices_source` says which. It
also reports `needs_venv` / `env_ready`, so a client can tell "this will answer now" from
"the first request spends minutes building an isolated venv". Note that
`/v1/audio/speakers` means something else entirely: enrolled voiceprints for recognition,
not a TTS model's built-in voices.

## Embeddings

`POST /v1/embeddings`

Request fields:

| Field | Type | Default | Description |
|---|---:|---:|---|
| `model` | string | required | Embedding model |
| `input` | string/string[] | required | Text input(s) |
| `image` | string/string[]/null | `null` | Optional image input(s) for multimodal embeddings |
| `encoding_format` | string | `float` | `float` or `base64` |
| `dimensions` | integer/null | `null` | Optional truncation size |
| `quantization` | string/null | `null` | TurboQuant vector quantization: `turbo`/`turbo8`/`turbo6`/`turbo4`/`turbo2` |

Example:

```bash
curl -s "$CODERAI_URL/v1/embeddings" \
  -H "Authorization: Bearer $CODERAI_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "BAAI/bge-small-en-v1.5",
    "input": ["first document", "second document"],
    "encoding_format": "float"
  }' | jq
```

Response shape:

```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "index": 0, "embedding": [0.01, -0.02]},
    {"object": "embedding", "index": 1, "embedding": [0.03, 0.04]}
  ],
  "model": "BAAI/bge-small-en-v1.5",
  "usage": {"prompt_tokens": 4, "total_tokens": 4}
}
```

Multimodal embedding example:

```json
{
  "model": "clip-embedding-model",
  "input": "a red sports car",
  "image": "data:image/png;base64,...",
  "encoding_format": "base64"
}
```

### TurboQuant embedding quantization

TurboQuant ([arXiv:2504.19874](https://arxiv.org/abs/2504.19874)) is a data-free,
inner-product-preserving vector quantizer: it randomly rotates each embedding so its
coordinates concentrate, then applies a per-coordinate scalar quantizer. Quantized
vectors keep their dot products / cosine similarity, so they can be stored 3–12×
smaller in a vector DB. Set `quantization` to enable it:

- `turbo`/`turbo8` = 8-bit (near-lossless, ~3×), `turbo6`, `turbo4` (~6×), `turbo2` (~12×).
- With `encoding_format: "float"` (default) the response returns the **lossy
  reconstructed** float vectors (same shape) — drop-in, behaves like a quantized store.
- With `encoding_format: "base64"` each `embedding` is the **compact packed bytes**
  (`[float16 norm][packbits(b-bit rotated codes)]`), and the response carries a
  top-level `quantization` block (`bits`, `seed`, `dim`, `dim_padded`, `radius`,
  `bytes_per_vector`, `layout`) describing how to decode them.

The implementation backend is chosen per embedding model in the admin **Models**
config (TurboQuant section): `builtin` (NumPy, always available) or `library`
(the optional `turboquant-py[torch]` package, which adds the paper's QJL stage).
TurboQuant must be enabled for the model, or a request `quantization` is rejected
with HTTP 400. Selecting the `library` backend when the package is not installed
also returns HTTP 400 rather than silently degrading.

```bash
curl -s "$CODERAI_URL/v1/embeddings" \
  -H "Authorization: Bearer $CODERAI_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "BAAI/bge-small-en-v1.5",
    "input": ["first document", "second document"],
    "quantization": "turbo4",
    "encoding_format": "base64"
  }' | jq
```

## Reranking

`POST /v1/rerank`

Scores query↔document relevance with a cross-encoder (e.g. `BAAI/bge-reranker-v2-m3`).
Embeddings retrieve candidates cheaply; a cross-encoder reads the query and the document
*together* and reorders them. Loaded natively through transformers and VRAM-managed like
every other model, so it takes part in eviction and thermal gating.

| Field | Type | Default | Description |
|---|---:|---:|---|
| `model` | string | required | Reranker model id |
| `query` | string | required | The search query |
| `documents` | string[] | required | Documents to score |
| `top_n` | integer/null | `null` | Return only the best N |
| `return_documents` | boolean/null | `false` | Echo the document text in each result |
| `max_length` | integer/null | model default | Truncation length for each query+document pair |

```bash
curl -s "$CODERAI_URL/v1/rerank" \
  -H "Authorization: Bearer $CODERAI_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"bge-reranker-v2-m3","query":"who pays the deductible?",
       "documents":["art. 2054 c.c. …","the policy excess clause …"],
       "top_n":1,"return_documents":true}' | jq
```

```json
{
  "object": "rerank.result",
  "model": "bge-reranker-v2-m3",
  "results": [{"index": 1, "relevance_score": 0.93, "document": "the policy excess clause …"}],
  "usage": {"total_tokens": 42}
}
```

`relevance_score` is the sigmoid of the cross-encoder logit, in `[0, 1]` and monotonic, so
ranking order is preserved whatever threshold you pick. The response shape mirrors the
common rerank APIs (Cohere / Jina).

## Retrieval Chunking

`POST /v1/chunk` splits a document into the pieces a retrieval index stores, with the
document's structure decided **per request** by a named or inline *structure template*.

This is not embedding windowing. Windowing makes one over-long input fit a model's context
and pools the pieces into one vector; this produces many chunks, each separately indexable,
each carrying its section, parent and references. Use both: chunk here, then embed each
chunk.

Length-only chunking cuts an argument wherever the character budget runs out, so a chunk
can straddle a section boundary and a retrieved passage mixes what the parties alleged with
what the court held. Sectioning first and length-chunking *inside* each section removes that
class of error and gives every chunk a label and a recoverable parent.

| Field | Type | Default | Description |
|---|---:|---:|---|
| `text` | string | required | The document (or use `/v1/chunk/file`) |
| `chunk_tokens` | integer/null | `800` | Target size in approximated tokens |
| `overlap_tokens` | integer/null | `120` | Carried tail; `0` = no carry |
| `chars_per_token` | integer/null | `4` | The approximation divisor; no tokenizer is loaded |
| `structural` | boolean/null | `true` | `false` = flat length chunking, no sectioning |
| `structure` | string/object/null | config default | Template name, or an inline template |
| `extract_refs` | string/object/false | template default | Reference extractor: a name, an inline object, or `false` |
| `require_text` | boolean/null | `true` | Refuse container bytes (a PDF, an OOXML zip) instead of chunking them |
| `include_offsets` | boolean/null | `false` | Add char offsets into the normalised source |
| `min_chunk_chars` | integer/null | template default | Absorb runt chunks below this |

```jsonc
{
  "chunks": [
    {"ord": 0, "text": "…", "n_chars": 2464,
     "section": "motivazione",              // null when unlabelled
     "parent": 2,                           // index into "sections", shared by its chunks
     "norm_refs": ["art.2043 c.c."],
     "start": 10433, "end": 12897}          // only with include_offsets
  ],
  "n_chunks": 37,
  "sections": [{"parent": 0, "section": "intestazione", "n_chars": 1204}],
  "structure": "it-legal",                  // the template actually applied
  "structural_applied": true,               // false = fell back to flat
  "params": {"chunk_tokens": 800, "overlap_tokens": 120,
             "chars_per_token": 4, "min_chunk_chars": 0}
}
```

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/chunk` | Chunk a document sent as JSON |
| `POST` | `/v1/chunk/file` | Chunk an uploaded document (same fields as form fields) |
| `GET` | `/v1/chunk/templates` | The structure templates on offer, with their section labels |

`parent` is the index into `sections`, so a caller can retrieve a chunk and then pull the
whole section it came from as context.

`require_text` defaults to **on** because a chunker that accepts a PDF or an OOXML
container embeds garbage the index cannot afterwards distinguish from content: on one
production corpus 0.59% of chunks held control characters and 0.04% raw OOXML — documents
whose container was chunked instead of its extracted text. Set it to `false` deliberately,
or extract the text first. A document over `chunking.max_chars` (8 MB) is a `413`
`too_large`, and an unknown `structure` name is an error, never a silent fallback to the
default template: a caller who asked for their own template and silently got someone
else's sectioning would index the wrong boundaries and never know.

The defaults are a **format, not a preference** — changing `chunk_tokens`,
`overlap_tokens` or `chars_per_token` moves every chunk boundary, which makes a re-chunked
index incomparable with the one it replaces.

A pure CPU text operation: token counts are characters divided by `chars_per_token`, so no
tokenizer is loaded, no model is needed and no GPU or pod is involved. Every image that runs
coderai serves this, including the light-core ones. Templates, the built-in `it-legal`
structure and the fidelity tests are in [`docs/chunking.md`](docs/chunking.md).

## Document OCR

`POST /v1/ocr` (multipart) transcribes an image or PDF with a purpose-built OCR engine —
PaddleOCR, docTR or Surya — returning faithful text plus per-line boxes and layout. Those
three are **not** vision LLMs. Two document VLMs are available for pages the
detector-based engines read badly: `engine=olmocr` (olmOCR-2, no boxes at all) and
`engine=chandra` (asked for layout, so it returns boxes and labelled regions too).

| Field | Type | Default | Description |
|---|---:|---:|---|
| `file` | file | required | Image or PDF (`files[]` on `/v1/ocr/batch`) |
| `engine` | string/null | config default | `paddle`, `doctr`, `surya`, `olmocr`, `chandra` |
| `dpi` | integer/null | config default | PDF rasterisation DPI |
| `lang` | string/null | `auto` | Document language; omit to let the engine decide |
| `detect` | string/null | config default | `off`, `layout`, `detector`, `both` |
| `layout` | boolean/null | config default | Run the layout pass (engines that hold a layout predictor) |
| `structured` | boolean | `false` | Structured-JSON extraction with a text model |
| `schema` | string/null | `null` | Extraction schema (a stored name, or inline JSON) |
| `max_tokens` | integer/null | engine default | Cap on the model's reply for **one page** |
| `max_continuations` | integer/null | engine default | Resumptions allowed for a page that hits the cap |
| `native_dpi` | boolean/null | config default | Pass an embedded scan through at *its* resolution |
| `page_image_format` | string/null | `auto` | `auto`, `png`, `jpeg` |
| `max_dpi_factor` | float/null | config default | Passthrough ceiling, as a multiple of the target DPI |

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/ocr` | OCR one image or PDF |
| `POST` | `/v1/ocr/batch` | OCR many files concurrently (`files[]`) |
| `GET` | `/v1/ocr/engines` | Which OCR engines this install offers, and whether each is ready |
| `GET` | `/v1/ocr/schemas` | List extraction schemas |
| `GET` | `/v1/ocr/schemas/{name}` | One schema |
| `POST` | `/v1/ocr/schemas` | Create or update a user schema |
| `DELETE` | `/v1/ocr/schemas/{name}` | Delete a user schema |

### A degraded read is not a successful one

A document whose every readable page came back empty is **not** a `200`. The orchestrator
has already measured ink on those pages, so "no text" is that run failing, not the document
being blank — and the two are indistinguishable in a `200` whose `text` is empty. A client
that treats an empty read as an unreadable scan then consumes the document from its queue
and writes it off; that happened to roughly a thousand documents in one night behind a lost
engine lease.

So: a document that produced *some* text is still a `200`, with `unread_pages` naming the
ones it did not read. A document that produced none is `502` with
`error.type: "ocr_unread"` and the same body. Set `ocr.unread_status` to `0` for the old
behaviour. On the VLM engines each page's `meta` also carries
`model_attempts` and `repeat_detected`, which separate a page truncated by a token cap from
one the model looped on, from one that is genuinely blank — three causes needing opposite
handling that a short page otherwise reports identically. Full subsystem detail, including throughput and the fleet layout, is in
[`docs/ocr.md`](docs/ocr.md).

## Moderation

`POST /v1/moderation` screens generated images and video before release — four independent
checks over an image, or over frames sampled from a video.

| Check | What it answers |
|---|---|
| `nsfw` | How explicit the picture is, with the classifier's own label scores passed through |
| `faces` | Where the faces are, one entry per face per sampled frame |
| `age` | An age estimate per face, as a point estimate **and** an interval |
| `face_match` | Whether each face matches a reference set — the consenting performer — by cosine similarity |

| Field | Type | Default | Description |
|---|---:|---:|---|
| `input` | object[] | required | Items: `{type: "image"\|"video", src, sample_fps?}` |
| `checks` | string[] | `["nsfw","age","faces"]` | Which checks to run |
| `face_match` | object/null | `null` | `{reference: [...], threshold, character_profiles: [...]}` |
| `nsfw_threshold` | float | `0.7` | Score at or above which a frame is flagged |

`GET /v1/moderation/models` reports which moderation models are present and loadable.

**It fails closed.** Every "safe" field in the response — `minor_detected: false`,
`flagged: false`, `unmatched_faces: 0` — is a statement that a model actually examined the
content and found nothing. If a model needed for a requested check is missing, broken or
unsure, the call returns an error or flags for review; it never returns the reassuring value
by default. The age interval, not the point estimate, is what decisions should be made on: a
point estimate alone invites treating "19" as settled. See
[`docs/moderation.md`](docs/moderation.md).

## Character Profiles

Character profiles are named collections of reference images used for visual identity consistency in image/video generation.

### Create or Replace Character

`POST /v1/characters`

```json
{
  "name": "Alice",
  "description": "Short-haired detective in a charcoal coat",
  "images": [
    {"label": "front", "data": "data:image/png;base64,..."},
    {"label": "side", "data": "data:image/png;base64,..."}
  ]
}
```

Response:

```json
{"ok": true, "name": "Alice", "image_count": 2}
```

### List Characters

`GET /v1/characters`

```json
{
  "characters": [
    {"name": "Alice", "description": "...", "image_count": 2, "created_at": 1781090000}
  ]
}
```

### Get Character

`GET /v1/characters/{name}`

Returns profile metadata plus base64 images.

### Patch Character

`PATCH /v1/characters/{name}`

```json
{
  "description": "Updated description",
  "add_images": [{"label": "close-up", "data": "data:image/png;base64,..."}],
  "remove_indices": [0]
}
```

### Delete Character

`DELETE /v1/characters/{name}`

### Generate Character References

`POST /v1/characters/generate`

Generates reference images from text and saves them as a profile.

```json
{
  "name": "CaptainNova",
  "description": "A calm starship captain",
  "prompt": "consistent character sheet, woman starship captain, front and side views, clean studio lighting",
  "model": "image-model",
  "n": 4,
  "steps": 30,
  "width": 768,
  "height": 768
}
```

### Extract Character from Media

`POST /v1/characters/extract`

```json
{
  "name": "InterviewGuest",
  "description": "Face crops extracted from source video",
  "videos": ["data:video/mp4;base64,..."],
  "max_images": 5
}
```

## Environment Profiles

Environment profiles are named collections of reference images used to condition scene/background style.

Routes mirror character profiles:

- `POST /v1/environments`
- `GET /v1/environments`
- `GET /v1/environments/{name}`
- `PATCH /v1/environments/{name}`
- `DELETE /v1/environments/{name}`
- `POST /v1/environments/generate`
- `POST /v1/environments/extract`

Create example:

```json
{
  "name": "OldLibrary",
  "description": "Warm wood, tall shelves, dust in sunset beams",
  "images": [
    {"label": "wide", "data": "data:image/png;base64,..."}
  ]
}
```

Generate example:

```json
{
  "name": "MarsHangar",
  "description": "Industrial red planet aircraft hangar",
  "prompt": "wide cinematic environment concept art of a Mars aircraft hangar, dust, red light, realistic",
  "model": "image-model",
  "n": 4,
  "width": 1024,
  "height": 768
}
```

Use in generation:

```json
{
  "model": "image-model",
  "prompt": "Alice stands beside a parked rover",
  "character_profiles": ["Alice"],
  "environment_profiles": ["MarsHangar"]
}
```

## LoRA Training and Registry

### Train LoRA

`POST /v1/loras/train`

Request fields:

| Field | Type | Default | Description |
|---|---:|---:|---|
| `name` | string | required | LoRA name |
| `base_model` | string | required | Base model to train against |
| `train_base_model` | string/null | `null` | Optional training model override |
| `target` | string | `image` | `image` or `video` |
| `quantize_4bit` | boolean | `true` | Quantized training where supported |
| `num_frames` | integer | `1` | Video/frame setting |
| `character` | string/null | `null` | Use saved character profile |
| `environment` | string/null | `null` | Use saved environment profile |
| `images` | string[]/null | `null` | Inline training images |
| `instance_prompt` | string/null | `null` | Instance prompt/token |
| `steps` | integer | `800` | Training steps |
| `rank` | integer | `16` | LoRA rank |
| `learning_rate` | number | `0.0001` | LR |
| `resolution` | integer | `512` | Training resolution |
| `seed` | integer | `42` | Seed |

Example:

```json
{
  "name": "alice_identity",
  "base_model": "image-model",
  "target": "image",
  "character": "Alice",
  "instance_prompt": "photo of alice_person",
  "steps": 800,
  "rank": 16,
  "learning_rate": 0.0001,
  "resolution": 768
}
```

Training is blocking and queued one-at-a-time.

### LoRA Progress

`GET /v1/loras/progress`

```bash
curl -s "$CODERAI_URL/v1/loras/progress" \
  -H "Authorization: Bearer $CODERAI_TOKEN" | jq
```

### LoRA Registry

- `GET /v1/loras` — list registered LoRAs (name, weight path, metadata)
- `GET /v1/loras/{name}` — fetch one registered LoRA
- `DELETE /v1/loras/{name}` — delete a registered LoRA

### Upload LoRA Weights

`POST /v1/loras/upload`

Upload a LoRA file into a **content-addressed (sha256) blob store** so a client on a
different machine can use it without sharing the server's filesystem. Accepts the file
in three ways:

- **multipart/form-data** with a `file` field,
- **JSON** `{"file": "<base64>"}` (a `data:` URI is also accepted; `data` is an alias),
- a **raw** request body (the bytes of the `.safetensors`).

Returns `{"id": "sha256:<hex>", "bytes": <n>, "existed": <bool>}`. Reference the returned
`id` in any image/video request via `"loras": [{"id": "sha256:<hex>", "weight": ...}]`.

```bash
curl -s "$CODERAI_URL/v1/loras/upload" \
  -H "Authorization: Bearer $CODERAI_TOKEN" \
  -F "file=@./alice_identity.safetensors" | jq
# → {"id":"sha256:1f3b…","bytes":18874368,"existed":false}
```

### Check Uploaded Blob

`GET /v1/loras/blob/{hash}`

Existence check for an uploaded blob — `200` with `{id, bytes, exists}` when present,
`404` when absent — so a client can skip re-uploading a file the server already has.
`hash` may be a bare hex sha256 or `sha256:<hex>`.

### LoRA references in requests

The `loras` array in image (`/v1/images/generations`) and video
(`/v1/video/generations`) requests accepts LoRA weights supplied in several ways. The
server resolves each entry in this **priority order**:

| Field | Example | Meaning |
|---|---|---|
| `id` | `"name:alice_identity"` | A registered/trained LoRA by name |
| `id` | `"sha256:1f3b…"` | An uploaded blob (from `/v1/loras/upload`) |
| `file` / `data` | `"<base64>"` or `data:` URI | Inline weights, sent with the request |
| `url` | `"https://…/lora.safetensors"` | Server downloads and caches it |
| `model` / `path` | `"/path/to/lora.safetensors"` or HF id | Legacy local path / HF id (shared filesystem only) |

Common fields: `weight` (float, scale; default `1.0`) and `name` (optional adapter name).

```json
{
  "model": "image-model",
  "prompt": "alice_person in a cyberpunk alley",
  "loras": [
    {"id": "name:alice_identity", "weight": 0.85},
    {"id": "sha256:1f3b9c…", "weight": 0.6, "name": "jacket"}
  ]
}
```

> The previous `{"model": "alice_identity"}` form still works, but prefer `id`
> (`"name:<registered>"`) or an uploaded `sha256:` blob so requests don't depend on the
> client and server sharing a filesystem.

## 2D / 3D / Spatial APIs

### Image to 3D

`POST /v1/images/to3d`

```json
{
  "image": "data:image/png;base64,...",
  "method": "mesh",
  "max_shift": 20,
  "response_format": "url"
}
```

`method` can include `stereo`, `anaglyph`, `depth`, or `mesh`.

### 3D to Image

`POST /v1/images/from3d`

```json
{
  "model_data": "data:model/gltf-binary;base64,...",
  "format": "glb",
  "camera_distance": 2.0,
  "camera_elevation": 30,
  "camera_azimuth": 45,
  "width": 768,
  "height": 768,
  "response_format": "url"
}
```

### Video to 3D

`POST /v1/video/to3d`

```json
{
  "video": "data:video/mp4;base64,...",
  "method": "anaglyph",
  "max_shift": 15,
  "response_format": "url"
}
```

### 3D to Video

`POST /v1/video/from3d`

```json
{
  "model_data": "data:model/gltf-binary;base64,...",
  "format": "glb",
  "frames": 36,
  "fps": 12,
  "camera_elevation": 20,
  "camera_distance": 2.5,
  "width": 768,
  "height": 768,
  "response_format": "url"
}
```

### Generate 3D Model

`POST /v1/3d/generate`

```json
{
  "prompt": "a stylized low-poly red dragon statue",
  "model": "3d-model",
  "steps": 64,
  "seed": 42,
  "response_format": "url"
}
```

Image-conditioned 3D generation:

```json
{
  "image": "data:image/png;base64,...",
  "model": "triposr",
  "response_format": "url"
}
```

## Built-In Pipelines

Pipelines chain existing endpoints server-side and aggregate `steps` and `data`.

Implementation caveat: `codai/api/pipelines.py` currently imports video helpers named `create_video_generation` and `create_video_dub`, while `codai/api/video.py` defines the route handlers as `video_generations` and `video_dub`. If those aliases are not added elsewhere at runtime, built-in video pipeline calls can fail even though the routes are registered. The lower-level video endpoints documented above are the canonical API surface.

### Image to Video Pipeline

`POST /v1/pipelines/image-to-video`

Steps:

1. Generate image with `image_model`
2. Animate it with `video_model`
3. Optionally add audio and upscale

```json
{
  "prompt": "a lonely lighthouse under aurora lights, cinematic",
  "image_model": "image-model",
  "video_model": "video-model",
  "image_size": "1024x1024",
  "image_steps": 30,
  "image_cfg": 7.0,
  "image_seed": 100,
  "num_frames": 32,
  "fps": 8,
  "num_inference_steps": 25,
  "guidance_scale": 6.5,
  "camera_motion": "zoom-in",
  "add_audio": true,
  "audio_type": "ambient",
  "audio_prompt": "distant waves, soft wind",
  "upscale_output": true,
  "response_format": "url"
}
```

### Video Dub Pipeline

`POST /v1/pipelines/video-dub`

```json
{
  "model": "whisper-large-v3",
  "video": "data:video/mp4;base64,...",
  "source_lang": "en",
  "target_lang": "de",
  "voice_clone": true,
  "burn_subtitles": true,
  "response_format": "url"
}
```

### Story Pipeline

`POST /v1/pipelines/story`

Steps:

1. LLM writes visual scene descriptions
2. Image model generates scene images
3. Video model animates the first scene
4. Optional TTS narration

```json
{
  "story": "A courier robot crosses a flooded city to deliver a seed vault key.",
  "text_model": "Qwen/Qwen3-8B",
  "image_model": "image-model",
  "video_model": "video-model",
  "tts_model": "kokoro",
  "tts_voice": "af_sarah",
  "num_scenes": 4,
  "num_frames": 32,
  "fps": 8,
  "response_format": "url"
}
```

### Audio Dub Pipeline

`POST /v1/pipelines/audio-dub`

Steps:

1. Transcribe source audio/video
2. Optionally translate transcript
3. Synthesize dubbed audio with voice cloning
4. If input is video, replace audio track

```json
{
  "video": "data:video/mp4;base64,...",
  "voice_name": "narrator_a",
  "source_lang": "en",
  "target_lang": "fr",
  "whisper_model": "whisper-large-v3",
  "speed": 1.0,
  "burn_subtitles": true,
  "response_format": "url"
}
```

## Custom Pipelines

Custom pipelines let clients define reusable multi-step workflows with template variables.

Implementation caveat: custom pipeline execution calls each handler with `(request, http_request)`. Some handlers in `codai/api/` accept only the request object, so step types whose handlers do not accept an HTTP request may need handler signature adjustments before they run reliably. Treat `/v1/pipelines/step-types` as the server's advertised builder schema and validate complex custom pipelines in your deployment.

### List Custom Pipelines

`GET /v1/pipelines/custom`

### List Step Types

`GET /v1/pipelines/step-types`

Supported step types include:

- `text_gen`
- `image_gen`
- `image_edit`
- `image_inpaint`
- `image_upscale`
- `image_deblur`
- `image_unpix`
- `image_outfit`
- `image_faceswap`
- `video_gen`
- `video_upscale`
- `video_sub`
- `video_interp`
- `video_dub`
- `tts`
- `stt`
- `audio_gen`
- `voice_clone`
- `voice_convert`

Template variables:

- `{{input}}` - pipeline runtime input
- `{{stepN.output}}` - extracted text/base output from step N
- `{{stepN.url}}` - first URL output from step N
- `{{stepN.<field>}}` - any extracted field from step N

### Create Custom Pipeline

`POST /v1/pipelines/custom`

```json
{
  "id": "poster-to-trailer",
  "name": "Poster to Trailer",
  "description": "Generate a poster concept, animate it, then create music.",
  "steps": [
    {
      "type": "text_gen",
      "label": "Write visual prompt",
      "params": {
        "model": "Qwen/Qwen3-8B",
        "system": "Write vivid visual prompts only.",
        "prompt": "Turn this idea into a cinematic image prompt: {{input}}"
      }
    },
    {
      "type": "image_gen",
      "label": "Generate poster",
      "params": {
        "model": "image-model",
        "prompt": "{{step0.output}}",
        "size": "1024x1024"
      }
    },
    {
      "type": "video_gen",
      "label": "Animate poster",
      "params": {
        "model": "video-model",
        "mode": "i2v",
        "prompt": "{{step0.output}}, slow cinematic movement",
        "init_image": "{{step1.url}}",
        "num_frames": 32,
        "fps": 8
      }
    },
    {
      "type": "audio_gen",
      "label": "Create soundtrack",
      "params": {
        "model": "musicgen",
        "prompt": "epic short trailer music for: {{input}}",
        "duration": 12
      },
      "continue_on_error": true
    }
  ]
}
```

### Update and Delete

- `PUT /v1/pipelines/custom/{pipeline_id}`
- `DELETE /v1/pipelines/custom/{pipeline_id}`

### Run Saved Pipeline

`POST /v1/pipelines/custom/{pipeline_id}/run`

```json
{
  "input": "a solar-powered train crossing the Sahara at night"
}
```

### Run Inline Pipeline

`POST /v1/pipelines/run`

Sends a `PipelineDefinition` directly without saving. The current implementation executes with an empty `{{input}}`, so include static params or use saved pipeline run when runtime input is required.

### Audio Understanding Pipeline

`POST /v1/pipelines/audio-understand`

Transcribes audio, then optionally asks a text model to summarize or reason over it.

```json
{
  "audio": "data:audio/wav;base64,...",
  "audio_model": "whisper-large-v3",
  "text_model": "Qwen/Qwen3-8B",
  "input": "Summarize action items and decisions.",
  "language": "en"
}
```

### Audio Music Dub Pipeline

`POST /v1/pipelines/audio-music-dub`

Current implementation returns a structured workflow with placeholder stages for stems, translation/adaptation, voice conversion, and remix.

```json
{
  "audio": "data:audio/wav;base64,...",
  "audio_model": "whisper-large-v3",
  "target_lang": "it",
  "source_lang": "en",
  "notes": "Preserve rhyme and chorus structure."
}
```

## Model and Weight Transfer

A rented pod normally downloads its own weights — from Hugging Face, or from a URL — which
costs nothing here and runs at datacenter speed. Some models have neither: a merge you made
locally, a file from a host that is gone, weights you are not going to publish. For those
the only way to get them onto another coderai is to send them.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/models/upload` | Stream weights to a coderai that cannot fetch them itself |
| `GET` | `/v1/models/uploaded` | Is this file already there, at the right size? (`200`/`404`) |
| `POST` | `/v1/models/register` | Teach a running instance about models it did not start with |
| `POST` | `/v1/loras/upload` | Upload adapter weights (content-addressed) |
| `GET` | `/v1/loras/blob/{hash}` | Does this adapter blob already exist here? |
| `POST` | `/v1/loras/register` | Serve an uploaded adapter under its own name |
| `GET` | `/v1/models/history` | Per-model load history and measured VRAM, merged across engines |

```text
POST /v1/models/upload?name=<file>&model_type=<section>[&alias=][&tar=1]
     body: the raw file, or a tar stream with tar=1, streamed to disk
GET  /v1/models/uploaded?name=<file>[&bytes=<n>]
```

The receiving side writes the bytes into its model directory and registers the entry, so
the model becomes loadable exactly as if it had been downloaded there. Sending 30 GB is a
real decision, not a default: coderai resolves a Hugging Face repo id or a URL first and
uploads only when a model's `source` says `upload`. Ask `/v1/models/uploaded` (or
`/v1/loras/blob/{hash}`) before sending — re-sending content that is already present is
the waste these endpoints exist to avoid.

`/v1/models/register` takes one model entry or a list of them, in the same shape
`models.json` uses, each with a `path` the *receiver* can fetch (a Hugging Face repo id or
a URL — a local path from another machine means nothing there). Registration is in memory
only, like a pod's boot seed, because a pod is disposable; it is idempotent, so
re-registering a known model never disturbs a loaded one. This is what makes a pod an
extension of the local system rather than a box that answers "not available" for the rest
of its life whenever a request names a model its owner had not thought of at boot.

`GET /v1/models/history` is where `measured_vram_gb` comes from; see
[`docs/memory-accounting.md`](docs/memory-accounting.md).

## Test Runs

`POST /v1/models/test`

Placement has a lot of moving parts: a model may be served here, from a URL, or on a RunPod
pod whose image, weights and adapters each have to resolve. Most of that fails minutes into
a pod boot, in a log nobody is watching. This runs the smallest real request for the
model's kind and reports what happened.

| Field | Type | Default | Description |
|---|---:|---:|---|
| `model` | string | required | The model to exercise |
| `where` | string | `auto` | `auto`, `local` (ignore remote config), `runpod` (refuse to fall back) |

It goes through the front like any client request, so it exercises the same routing, auth,
pod provisioning and staging a real request would — a test that simulated the path would
prove nothing about the path. `where: "runpod"` is the useful one: a pod that is not
actually reachable then fails the test instead of quietly passing locally.

`GET /v1/models/test/state` reports what this engine currently believes about the models it
has tested. A few kinds have no honest synthetic input — a face swap needs real faces, stem
separation needs audio worth separating — and those report a reachability check instead of a
generation, saying so rather than implying more than was proven.

## Remote GPUs: the Client API

The orchestrator grows and shrinks a pod pool by itself. These endpoints are for a client
that **knows what is coming** — a batch of a thousand documents — and would otherwise
discover the pool one cold start at a time. They remove the ramp; they do not replace the
autoscaler.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/runpod/scale` | Hold N pods warm for a model (a lease) |
| `GET` | `/v1/runpod/scale/{request_id}` | How a scale request that was still finding capacity ended |
| `GET` | `/v1/runpod/status` | Fleet status: pools, pods, per-pod state |
| `GET` | `/v1/runpod/spend` | Spend and live cost |
| `POST` | `/v1/runpod/release` | Terminate one pod by id |
| `POST` | `/v1/runpod/repurpose` | Move a rented pod to another pool or image, in place |

`POST /v1/runpod/scale`:

| Field | Type | Default | Description |
|---|---:|---:|---|
| `model` | string | required | Whose pod pool to scale |
| `pods` | integer | required | A **floor**: how many to hold ready. `0` releases the lease |
| `max_pods` | integer/null | `null` | How large the pool may *become* under load; may only narrow the operator's own number |
| `teardown` | boolean | `false` | Shed the surplus now instead of waiting out `scale_down_after_s` |
| `gpu_types` | string/string[]/null | `null` | Which cards this lease will accept; may only narrow the model's choice |
| `allow_other_gpus` | boolean/null | `null` | Whether a fallback card is acceptable |
| `wait_s` | float/null | config default | How long to block before answering "still finding capacity" |

```bash
# before a batch
curl -s "$CODERAI_URL/v1/runpod/scale" \
  -H "Authorization: Bearer $CODERAI_TOKEN" -H "Content-Type: application/json" \
  -d '{"model":"my-ocr-model","pods":8,"max_pods":32,"gpu_types":"NVIDIA A40"}' | jq
# after it
curl -s "$CODERAI_URL/v1/runpod/scale" \
  -H "Authorization: Bearer $CODERAI_TOKEN" -H "Content-Type: application/json" \
  -d '{"model":"my-ocr-model","pods":0,"teardown":true}' | jq
```

A call that is still finding capacity after `wait_s` answers
`{"accepted": true, "pending": true, "request_id": "…"}`, and
`GET /v1/runpod/scale/{request_id}` says how it ended — the provisioning continues either
way.

These are deliberately **not** admin-scoped, and the guards are what makes that safe: the
gate is per model (`allow_client_scale`), the number is clamped to the model's `max_pods`,
`gpu_types` and `max_pods` may only ever narrow the operator's own configuration and die
with the lease, the lease expires (`client_scale_ttl_s`, default 30 min) so a client that
dies holding four A40s stops paying for them by itself, and the account's $/hr and spend
caps still refuse to provision regardless of what is asked. There is deliberately no way
for a request to change a model's configuration. A pod with requests on it drains rather
than dying under them. See [`docs/runpod.md`](docs/runpod.md).

## Service and Operational Endpoints

Not in `/openapi.json` (they are registered with `include_in_schema=False`), but part of the
HTTP surface and the ones an operator actually curls.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/healthz` | Cheap liveness: touches no torch or model state, so it answers the instant the event loop is free |
| `GET` | `/health` | The same answer at the path stock vLLM uses, because clients hard-code it |
| `GET` | `/v1/health` | The same answer under the `/v1` prefix |
| `GET` | `/metrics` | Prometheus exposition |
| `GET` | `/boot` | What this process did on the way up, with timings |
| `GET` | `/logs` | This process's own output, from a cursor (`?since=&limit=`) |
| `GET` | `/coderai/capabilities` | The broker capability document: version, transports, OpenAI-compat matrix, hardware summary |

`/healthz` is how the front's engine supervisor distinguishes a *slow* engine (busy loading
a model — the event loop may be blocked, so the answer can be late but will come) from a
*dead* one (connection refused). It must stay trivial, which is why it reports nothing
useful about models.

`/boot` and `/logs` exist for rented pods, which are the one kind of machine whose logs
cannot be read from outside: a boot that failed looked exactly like a boot that was merely
slow, and the provider's log API answered HTTP 400 when we went looking. `/boot` is readable
the instant the port opens — well before the pod is useful — and includes the container
script's own phases; `/logs` covers the life after it, and the renting install drains it
into a file it keeps after the pod is gone. On a pod the bearer token is required for both,
so neither is a way to read an install's log without one.

`/metrics` is open when `server.metrics_public` is set, and otherwise takes the same light
credential as the other telemetry endpoints.

## Cluster Endpoints

The mesh between a head node and the machines it treats as engines. Authenticated with the
shared cluster token, not a user session.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/cluster/state` | This node's aggregate state: models, engines, RPC peers |
| `GET` | `/cluster/logs` | This node's log, for the head's cluster page |
| `GET` | `/cluster/rpc-servers` | The llama.cpp RPC endpoints this node exposes |
| `POST` | `/cluster/model-load` | The head tells this node to load a model |
| `POST` | `/cluster/model-unload` | The head tells this node to unload one |
| `POST` | `/cluster/reload-config` | The head tells this node which models it owns |

`reload-config` is the interesting one: entries this node's own catalogue lacks are
registered in memory on its primary engine, so the head can name a model this node never
heard of and have it served. See [`docs/cluster.md`](docs/cluster.md).

## Internal Endpoints

The control plane between the front proxy and its engine subprocesses, and between engines
that share a GPU. A client never calls these; they are documented because an operator
debugging VRAM contention or a thermal pause will see them in the logs.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/internal/engine-state` | This engine's loaded models, VRAM and queue depth |
| `POST` | `/internal/evict-vram` | A co-located sibling asks this engine to release VRAM it is holding |
| `POST` | `/internal/gpu-reserve` | A sibling reserves the card for an exclusive op (so this engine will not load a model and contend with its training VRAM) |
| `POST` | `/internal/gpu-release` | Lift that reservation |
| `POST` | `/internal/thermal-pause` | Pause at the next safe point — request entry, or between tokens |
| `POST` | `/internal/thermal-resume` | Lift the thermal pause once the hardware has cooled |
| `POST` | `/internal/reload-config` | Re-read `config.json` / `models.json` after an admin edit |

Temperatures are monitored centrally by the front's supervisor, which is why a thermal pause
arrives as a request rather than being each engine's own decision. See
[`docs/frontend-engine-split.md`](docs/frontend-engine-split.md).

## Admin HTML Routes

Admin pages are session-cookie based.

| Method | Path | Purpose | Auth |
|---|---|---|---|
| `GET` | `/login` | Login page | Public |
| `POST` | `/login` | Login form | Public |
| `GET` | `/logout` | Logout | Optional session |
| `GET` | `/admin/change-password` | Password change page | Logged-in |
| `POST` | `/admin/change-password` | Change password | Logged-in |
| `GET` | `/admin` | Dashboard | Logged-in |
| `GET` | `/admin/models` | Model management page | Admin |
| `GET` | `/admin/tokens` | Token page | Admin |
| `GET` | `/admin/users` | User page | Admin |
| `GET` | `/chat` | Chat UI | Logged-in |
| `GET` | `/admin/settings` | Settings page | Admin |
| `GET` | `/admin/archive` | Archive page | Admin |
| `GET` | `/admin/tasks` | Tasks page: running jobs, engine tiles, logs | Admin |
| `GET` | `/admin/cluster` | Cluster page: nodes, pools, reachability | Admin |
| `GET` | `/admin/runpod` | RunPod page: pools, pods, spend | Admin |

Static assets are mounted under `/static/admin/*`.

These pages are rendered by the **front proxy** (`codai/frontproxy/ui_pages.py`), with
sessions validated locally, so navigating the dashboard never waits on an engine that is
busy generating. The engine serves generation; the pages live on the front.

## Admin API

Admin APIs usually require a valid session cookie and admin role unless noted.

### Status, Users, Tokens

| Method | Path | Body/Query | Purpose |
|---|---|---|---|
| `GET` | `/admin/api/status` | none | System, model, VRAM, queue, recent activity status |
| `POST` | `/admin/api/users` | `{username,password,role}` | Create user |
| `DELETE` | `/admin/api/users/{user_id}` | path | Delete user |
| `GET` | `/admin/api/tokens` | none | List API tokens |
| `POST` | `/admin/api/tokens` | `{name, provider?}` | Create token |
| `DELETE` | `/admin/api/tokens/{token_id}` | path | Delete token |
| `POST` | `/admin/api/system/reload` | none | Reload config/system state |

Create token example after logging in with a session cookie:

```bash
curl -s "$CODERAI_URL/admin/api/tokens" \
  -b cookies.txt \
  -H "Content-Type: application/json" \
  -d '{"name":"automation","provider":"local"}' | jq
```

### Model and Cache Management

| Method | Path | Body/Query | Purpose |
|---|---|---|---|
| `GET` | `/admin/api/models` | none | List configured models |
| `POST` | `/admin/api/model-download` | `{model_id,file_pattern?}` | Start Hugging Face download |
| `GET` | `/admin/api/download-stream/{session_id}` | path | SSE download progress |
| `GET` | `/admin/api/downloads` | none | Active/recent downloads |
| `POST` | `/admin/api/download-cancel/{session_id}` | path | Cancel download |
| `POST` | `/admin/api/model-upload` | multipart chunk | Chunked model upload |
| `DELETE` | `/admin/api/models/{model_identifier}` | path | Remove cached model |
| `GET` | `/admin/api/hf-files` | `repo_id` | List HF repo files |
| `GET` | `/admin/api/cached-models` | none | Local cache inventory |
| `GET` | `/admin/api/cache-stats` | none | Disk/cache stats |
| `DELETE` | `/admin/api/cache` | `cache_type=all\|hf\|gguf` | Clear cache |
| `DELETE` | `/admin/api/cached-models/{model_id:path}` | `cache_type` | Delete cached model |
| `POST` | `/admin/api/model-enable` | `{path\|model_id,model_type}` | Enable model in config |
| `POST` | `/admin/api/model-disable` | `{path\|model_id,config_id?}` | Disable model |
| `GET` | `/admin/api/model-loaded-status` | none | Loaded model / pool info |
| `POST` | `/admin/api/model-load` | `{path}` | Load model now |
| `POST` | `/admin/api/model-unload` | `{path}` | Unload model |
| `POST` | `/admin/api/model-configure` | model config JSON | Configure model (incl. the `acceleration` block — see [Acceleration and Distillation](#acceleration-and-distillation)) |
| `GET` | `/admin/api/accel-presets` | none | Catalog of acceleration/distillation presets (Lightning, Lightx2v, Turbo, LCM, Hyper-SD) |

Download with SSE progress:

```bash
SESSION_ID=$(curl -s "$CODERAI_URL/admin/api/model-download" \
  -b cookies.txt \
  -H "Content-Type: application/json" \
  -d '{"model_id":"Qwen/Qwen3-8B"}' | jq -r .session_id)

curl -N "$CODERAI_URL/admin/api/download-stream/$SESSION_ID" -b cookies.txt
```

SSE events include `progress`, `done`, `error`, and `keepalive`.

### Settings and Archive Admin

| Method | Path | Body/Query | Purpose |
|---|---|---|---|
| `GET` | `/admin/api/settings` | none | Current config sections |
| `POST` | `/admin/api/settings` | partial settings JSON | Save settings |
| `GET` | `/admin/api/archive` | `limit`, `offset` | List archive entries |
| `GET` | `/admin/api/archive/{gen_id}` | path | Archive entry detail |
| `DELETE` | `/admin/api/archive/{gen_id}` | path | Delete archive entry |
| `GET` | `/admin/api/archive/{gen_id}/files/{filename}` | path | Download archive file |
| `GET` | `/admin/api/archive-settings` | none | Archive config and retention options |

Settings include server/backend/model/offload/vulkan/archive/thermal/broker/parser/system-prompt sections.

### Hugging Face Search and Metadata

| Method | Path | Query | Purpose |
|---|---|---|---|
| `GET` | `/admin/api/hf-search` | `q`, `gguf_mode`, `pipeline_tag`, `sort`, `sizes`, `arch`, `capabilities`, `component_type` | Search models |
| `GET` | `/admin/api/hf-model-files` | `model_id` | List GGUF/model files with size/quant metadata |
| `GET` | `/admin/api/hf-model-info` | `model_id` | Full HF model metadata summary |

Example:

```bash
curl -s "$CODERAI_URL/admin/api/hf-search?q=whisper&capabilities=speech_to_text" \
  -b cookies.txt | jq
```

### Admin Profile Proxies

Logged-in users can access profile metadata through admin routes:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/admin/api/characters` | List characters |
| `GET` | `/admin/api/characters/{name}` | Character detail |
| `GET` | `/admin/api/characters/{name}/thumbnail` | Character thumbnail |
| `DELETE` | `/admin/api/characters/{name}` | Delete character |
| `GET` | `/admin/api/environments` | List environments |
| `GET` | `/admin/api/environments/{name}` | Environment detail |
| `GET` | `/admin/api/environments/{name}/thumbnail` | Environment thumbnail |
| `DELETE` | `/admin/api/environments/{name}` | Delete environment |
| `GET` | `/admin/api/voices` | List voice profiles |
| `GET` | `/admin/api/voices/{name}` | Voice detail |
| `DELETE` | `/admin/api/voices/{name}` | Delete voice |

### Machine, Engines and Tasks

Everything on the dashboard and the tasks page. These take a *light* credential — a session
cookie or bearer that is present and well-formed — rather than a full round-trip validation
against the primary engine, because validating there would 401 exactly when an engine is
busy generating, making the operator's tiles disappear when they most want to see them.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/admin/api/system-stats` | Host CPU / RAM / disk / temperature |
| `GET` | `/admin/api/gpu-stats` | Per-GPU utilisation, VRAM and temperature |
| `GET` | `/admin/api/engines` | Engine nodes: name, health, VRAM, loaded models (the front's own registry) |
| `POST` | `/admin/api/engines/{eid}/restart` | Restart one engine subprocess |
| `GET` | `/admin/api/usage` | Requests, errors and time per API key / model / kind — the same numbers `/metrics` exports, as a table |
| `GET` | `/admin/api/whoami` | Validate the current admin session |
| `POST` | `/admin/api/batch` | Run several admin API calls in one request |

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/admin/api/tasks` | Poll running and finished tasks across *all* engines |
| `POST` | `/admin/api/tasks/{task_id}/cancel` | Cancel a task |
| `POST` | `/admin/api/tasks/{task_id}/interrupt` | Interrupt a running task |
| `POST` | `/admin/api/tasks/{task_id}/pause` | Pause it |
| `POST` | `/admin/api/tasks/{task_id}/resume` | Resume it |
| `POST` | `/admin/api/tasks/{task_id}/restart` | Restart a training task |
| `DELETE` | `/admin/api/tasks/{task_id}` | Remove a finished task from the list |

A per-task action has to reach the engine that *owns* the task, not whichever engine is
primary, or the UI reads "Task not found"; the front routes these rather than proxying them.

### Model Catalogue, Quantization and Engines

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/admin/api/model-add-known` | Add a known-but-not-downloaded reference to `models.json` |
| `POST` | `/admin/api/model-mark-download` | List a model for later download |
| `POST` | `/admin/api/model-unmark-download` | Take it off that list |
| `POST` | `/admin/api/model-free-disk` | Reclaim disk: delete a model's files, keep its config entry |
| `POST` | `/admin/api/download-cancel-all` | Cancel every active download |
| `POST` | `/admin/api/model-quantize` | Quantize a model to fast-kernel 4-bit (GPTQ/AWQ) |
| `GET` | `/admin/api/quantize-capabilities` | Whether GPTQ/AWQ quantization is available here |
| `GET` | `/admin/api/quantize-status` | Quantization job status |
| `GET` | `/admin/api/turboquant-info` | TurboQuant backend availability (`builtin` / `library`) |
| `GET` | `/admin/api/accel-loras` | Cached distill-LoRA files for acceleration |
| `GET` | `/admin/api/ds4/default-models` | Downloadable ds4 default models |
| `GET` | `/admin/api/colibri/default-models` | Downloadable colibri default models |

### Optional Subsystems and Isolated Venvs

Several engines need an interpreter of their own. These build, inspect and remove those
environments from the admin UI instead of leaving an operator to guess why a first request
took twenty minutes.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/admin/api/embedder-venvs` | Isolated embedder venv status |
| `POST` | `/admin/api/embedder-venv-install` | Build one |
| `POST` | `/admin/api/embedder-venv-remove` | Delete one |
| `GET` | `/admin/api/lipsync-engines` | Lip-sync engine status |
| `POST` | `/admin/api/lipsync-engine-install` | Install a lip-sync engine |
| `POST` | `/admin/api/lipsync-engine-remove` | Remove one |
| `POST` | `/admin/api/ocr/build-venv` | Build the Paddle / Surya isolated venv |
| `GET` | `/admin/api/ocr/build-venv/status` | How that build is going |
| `GET` | `/admin/api/ocr/schemas` | List OCR extraction schemas |
| `GET` | `/admin/api/ocr/schemas/{name}` | One schema |
| `POST` | `/admin/api/ocr/schemas` | Create or update a user schema |
| `DELETE` | `/admin/api/ocr/schemas/{name}` | Delete one |

See [`docs/isolated-worker-environments.md`](docs/isolated-worker-environments.md) for why
these exist and what `env_ready` means.

### RunPod Administration

The operator's half of the RunPod subsystem; the client-facing half is
[Remote GPUs: the Client API](#remote-gpus-the-client-api).

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/admin/api/runpod/status` | Account pods and live cost — **the account**, not just this install's pools |
| `GET` | `/admin/api/runpod/stats` | Pod usage and spend history |
| `GET` | `/admin/api/runpod/gpu-types` | GPU types and prices |
| `GET` | `/admin/api/runpod/pod-logs` | A pod's container / engine logs (`?pod_id=&tail=`), plus a console deep-link |
| `POST` | `/admin/api/runpod/pod-adopt` | Take an unmanaged pod under local management |
| `POST` | `/admin/api/runpod/pod-release` | Terminate one pod by id, now |
| `POST` | `/admin/api/runpod/seed-volumes` | Boot one pod per region so each regional volume gets a model's weights once |

`runpod/status` reads the **account**, which is the point: a pool never probes its own pods,
so a pod killed out of band can otherwise keep a warm floor "satisfied" forever, and a pod
whose engine died keeps billing with nothing owning it. Reconcile from the account listing.

### Cluster Administration

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/admin/api/cluster` | Everything this front can send work to, with reachability |
| `GET` | `/admin/api/cluster/logs` | Tail a cluster node's log |

Front-native: answered from the registry and config without touching an engine, so the
cluster page renders while every engine is mid-generation.

## Acceleration and Distillation

Image and video models can be configured to use a **distillation adapter** (Lightning,
Lightx2v / phased DMD, SDXL-Turbo, LCM-LoRA, Hyper-SD). When enabled, the distill LoRA
is **fused into the pipeline at load time** and the correct low step-count / low-guidance
/ scheduler defaults are applied at generation time — cutting inference from ~25–50 steps
to **1–8 steps at guidance ≈ 1.0** (a 5–10× speedup). It is orthogonal to per-request
character LoRAs, which still apply on top.

This is **per-model configuration** (set via `POST /admin/api/model-configure` or the
admin Models page), not a per-request field. The catalog of presets is served by
`GET /admin/api/accel-presets`.

The `acceleration` block in a model's config:

```json
"acceleration": {
  "enabled": true,
  "preset": "wan22_lightning_4step",
  "lora": "lightx2v/Wan2.2-Lightning",
  "lora_weight": 1.0,
  "steps": 4,
  "guidance_scale": 1.0,
  "flow_shift": 5.0,
  "scheduler": ""
}
```

- `enabled` — `false` or an absent block means no change to current behaviour.
- `preset` — a catalog key (below) or `"custom"`. When not `"custom"`, unset fields are
  filled from the preset; any explicit field overrides it.
- `lora` — distill LoRA path or HF repo (`repo` or `repo:weight_name.safetensors`).
  `null` for full-model presets such as SDXL-Turbo.
- `steps` / `guidance_scale` — defaults applied when the request omits them.
- `flow_shift` — optional Wan flow-match scheduler shift.
- `scheduler` — optional scheduler class override (e.g. `LCMScheduler`).

Preset catalog (`GET /admin/api/accel-presets`):

| Preset key | Applies to | Steps | Guidance | Notes |
|---|---|---:|---:|---|
| `wan22_lightning_4step` | video | 4 | 1.0 | Wan2.2 Lightning (4-step DMD) |
| `wan21_lightx2v_4step` | video | 4 | 1.0 | Wan2.1 Lightx2v (4-step) |
| `sdxl_lightning_4step` | image | 4 | 1.0 | SDXL-Lightning (4-step) |
| `sdxl_lightning_8step` | image | 8 | 1.0 | SDXL-Lightning (8-step) |
| `sdxl_turbo` | image | 4 | 1.0 | SDXL-Turbo (full model, 1–4 step) |
| `sdxl_lcm` | image | 6 | 1.5 | SDXL LCM-LoRA (`LCMScheduler`) |
| `hyper_sdxl_8step` | image | 8 | 1.0 | Hyper-SD SDXL (8-step) |
| `sd15_lcm` | image | 6 | 1.5 | SD1.5 LCM-LoRA (`LCMScheduler`) |

> The preset LoRA repo ids are best-effort defaults; override `lora` (and any numeric
> field) per model. A LoRA-fuse failure is logged and generation proceeds un-accelerated.
> sd.cpp models get the step/guidance defaults and optional `<lora:…>` prompt injection
> (more limited than diffusers).

### KV-cache quantization (GGUF text models)

GGUF/llama.cpp text models can quantize the **KV cache** to fit longer contexts in
less VRAM. This is **per-model configuration** (set via `POST /admin/api/model-configure`
or the admin Models UI), independent of the weight-quantization flags:

```json
"cache_type_k": "q8_0",
"cache_type_v": "q8_0"
```

Accepted values: `q8_0` (near-lossless, ~2× smaller KV), `q5_1`, `q5_0`, `q4_1`,
`q4_0` (smallest), or omit/blank for the default `f16`. A sub-8-bit **value** cache
(`q5_*`/`q4_*`) requires flash attention; CoderAI auto-enables it for that model.

## AISBF / Broker Integration

CoderAI exposes:

- `GET /coderai/capabilities`
- OpenAI-compatible `/v1/models` and `/v1/chat/completions`
- Native `/v1/*` endpoints that can be proxied by AISBF

AISBF broker mode uses outbound WebSocket connections from CoderAI to AISBF for NAT traversal. The canonical broker protocol is documented in `coderai-broker-implementation-reference.md`.

Global-scope broker URL template:

```text
wss://<aisbf-host>/api/coderai/wss?provider_id=<provider_id>&client_id=<client_id>&username=global&registration_token=<token>
```

User-scope broker URL template:

```text
wss://<aisbf-host>/api/u/<username>/coderai/wss?provider_id=<provider_id>&client_id=<client_id>&username=<username>&registration_token=<token>
```

Important broker fields:

- `provider_id` identifies the AISBF provider configuration.
- `client_id` must be stable and match the provider config.
- `username` is `global` or the AISBF username for user-scoped providers.
- `registration_token` is provider-scoped and required for admission.

AISBF can call operations such as `models.list`, `chat.completions`, `capabilities`, `register`, and `proxy`. Proxy operations can forward headers, query params, multipart form payloads, binary/base64 bodies, progress polling endpoints, and streaming envelopes.

## Error Handling

Common HTTP status codes:

| Status | Meaning |
|---:|---|
| `400` | Invalid request, missing required media, or incompatible fields |
| `401` | Missing/invalid token or session |
| `403` | Forbidden, unsafe file path, or insufficient role |
| `404` | Model, profile, file, pipeline, or archive entry not found |
| `422` | Validation error for strict fields |
| `429` | Rate limit or queue saturation |
| `500` | Generation/backend failure |
| `501` | Optional backend not installed |
| `503` | Model/backend unavailable or CUDA context poisoned |

Typical auth error:

```json
{
  "detail": {
    "message": "Invalid API key. Provide a valid Bearer token.",
    "type": "invalid_request_error",
    "code": "invalid_api_key"
  }
}
```

If a CUDA device-side assert or illegal memory access poisons the context, CoderAI fails fast with a `503` instructing that the process must be restarted.

## Complex Workflows

### Workflow 1: Consistent Character Image and Video

Goal: create a character, generate a scene image using that identity, then animate it.

1. Create character profile:

```bash
curl -s "$CODERAI_URL/v1/characters" \
  -H "Authorization: Bearer $CODERAI_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name":"Alice",
    "description":"Detective with short black hair and charcoal coat",
    "images":[{"label":"front","data":"data:image/png;base64,..."}]
  }'
```

2. Generate an image with the profile:

```bash
IMAGE_URL=$(curl -s "$CODERAI_URL/v1/images/generations" \
  -H "Authorization: Bearer $CODERAI_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model":"image-model",
    "prompt":"Alice in a rainy neon alley, cinematic detective noir",
    "character_profiles":["Alice"],
    "character_strength":0.75,
    "size":"1024x1024",
    "response_format":"url"
  }' | jq -r '.data[0].url')
```

3. Animate the image:

```bash
curl -s "$CODERAI_URL/v1/video/generations" \
  -H "Authorization: Bearer $CODERAI_TOKEN" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\":\"video-model\",
    \"mode\":\"i2v\",
    \"prompt\":\"Alice looks up as rain falls, subtle camera push-in\",
    \"init_image\":\"$IMAGE_URL\",
    \"num_frames\":32,
    \"fps\":8,
    \"camera_motion\":\"zoom-in\",
    \"response_format\":\"url\"
  }" | jq
```

### Workflow 2: Full Story Generation

Use the built-in story pipeline to generate a script, scene images, a short video, and narration.

```bash
curl -s "$CODERAI_URL/v1/pipelines/story" \
  -H "Authorization: Bearer $CODERAI_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "story":"A botanist finds a singing plant inside a crashed satellite.",
    "text_model":"Qwen/Qwen3-8B",
    "image_model":"image-model",
    "video_model":"video-model",
    "tts_model":"kokoro",
    "tts_voice":"af_sarah",
    "num_scenes":4,
    "num_frames":32,
    "fps":8,
    "response_format":"url"
  }' | jq
```

Output includes:

- `steps[0].text` generated scene script
- `steps[1].urls` generated images
- `data[0].video_url`
- `data[0].audio_url`

### Workflow 3: Multilingual Video Dubbing

1. Upload or encode the source video as a data URL.
2. Call the video dub pipeline.
3. Poll `/v1/video/progress` if needed.
4. Download output from returned URL.

```bash
curl -s "$CODERAI_URL/v1/pipelines/video-dub" \
  -H "Authorization: Bearer $CODERAI_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model":"whisper-large-v3",
    "video":"data:video/mp4;base64,...",
    "source_lang":"en",
    "target_lang":"ja",
    "voice_clone":true,
    "burn_subtitles":true,
    "response_format":"url"
  }' | jq
```

For lower-level control, use:

- `POST /v1/video/subtitle`
- `POST /v1/audio/clone`
- `POST /v1/video/dub`

### Workflow 4: Audio Meeting Summary

Transcribe a meeting and summarize action items with a text model.

```bash
curl -s "$CODERAI_URL/v1/pipelines/audio-understand" \
  -H "Authorization: Bearer $CODERAI_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "audio":"data:audio/wav;base64,...",
    "audio_model":"whisper-large-v3",
    "text_model":"Qwen/Qwen3-8B",
    "language":"en",
    "input":"Extract decisions, owners, deadlines, and unresolved questions."
  }' | jq
```

### Workflow 5: Train and Apply a Character LoRA

1. Build a character profile:

```json
{
  "name": "Mira",
  "description": "Explorer with copper curls and a green field jacket",
  "images": [{"label": "front", "data": "data:image/png;base64,..."}]
}
```

2. Train LoRA:

```bash
curl -s "$CODERAI_URL/v1/loras/train" \
  -H "Authorization: Bearer $CODERAI_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name":"mira_lora",
    "base_model":"image-model",
    "target":"image",
    "character":"Mira",
    "instance_prompt":"photo of mira_person",
    "steps":800,
    "rank":16,
    "resolution":768
  }' | jq
```

3. Poll progress:

```bash
watch -n 2 "curl -s '$CODERAI_URL/v1/loras/progress' -H 'Authorization: Bearer $CODERAI_TOKEN' | jq"
```

4. Generate with LoRA:

```json
{
  "model": "image-model",
  "prompt": "photo of mira_person exploring alien ruins, cinematic backlight",
  "loras": [{"model": "mira_lora", "weight": 0.8}],
  "response_format": "url"
}
```

### Workflow 6: Custom Pipeline for Automated Media Asset Creation

Create a reusable pipeline that converts a product idea into a slogan, hero image, promo video, and voiceover.

```bash
curl -s "$CODERAI_URL/v1/pipelines/custom" \
  -H "Authorization: Bearer $CODERAI_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "id":"product-media-kit",
    "name":"Product Media Kit",
    "description":"Slogan, image, video, and voiceover for a product concept.",
    "steps":[
      {
        "type":"text_gen",
        "label":"Write slogan and image prompt",
        "params":{
          "model":"Qwen/Qwen3-8B",
          "system":"Return a concise slogan, then a vivid image prompt.",
          "prompt":"Product concept: {{input}}"
        }
      },
      {
        "type":"image_gen",
        "label":"Hero image",
        "params":{
          "model":"image-model",
          "prompt":"{{step0.output}}",
          "size":"1024x1024",
          "response_format":"url"
        }
      },
      {
        "type":"video_gen",
        "label":"Promo animation",
        "params":{
          "model":"video-model",
          "mode":"i2v",
          "prompt":"premium product commercial, elegant camera motion, {{step0.output}}",
          "init_image":"{{step1.url}}",
          "num_frames":32,
          "fps":8,
          "response_format":"url"
        }
      },
      {
        "type":"tts",
        "label":"Voiceover",
        "params":{
          "model":"kokoro",
          "input":"{{step0.output}}",
          "voice":"af_sarah",
          "speed":1.0
        },
        "continue_on_error":true
      }
    ]
  }' | jq

curl -s "$CODERAI_URL/v1/pipelines/custom/product-media-kit/run" \
  -H "Authorization: Bearer $CODERAI_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"input":"A compact solar charger for hikers and emergency kits"}' | jq
```

## Practical Client Patterns

### Polling Progress While a Job Runs

Use a second terminal while a generation request is running:

```bash
while true; do
  curl -s "$CODERAI_URL/v1/video/progress" \
    -H "Authorization: Bearer $CODERAI_TOKEN" | jq -c
  sleep 2
done
```

### Python Chat Client

```python
import requests

base = "http://127.0.0.1:8776"
token = "your-api-token"

resp = requests.post(
    f"{base}/v1/chat/completions",
    headers={"Authorization": f"Bearer {token}"},
    json={
        "model": "Qwen/Qwen3-8B",
        "messages": [{"role": "user", "content": "Write a CLI release note."}],
        "temperature": 0.3,
    },
    timeout=300,
)
resp.raise_for_status()
print(resp.json()["choices"][0]["message"]["content"])
```

### Python Streaming Chat Client

```python
import json
import requests

base = "http://127.0.0.1:8776"
token = "your-api-token"

with requests.post(
    f"{base}/v1/chat/completions",
    headers={"Authorization": f"Bearer {token}"},
    json={
        "model": "Qwen/Qwen3-8B",
        "messages": [{"role": "user", "content": "Count to five slowly."}],
        "stream": True,
    },
    stream=True,
    timeout=300,
) as r:
    r.raise_for_status()
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            break
        event = json.loads(payload)
        delta = event["choices"][0].get("delta", {})
        print(delta.get("content", ""), end="", flush=True)
```

### OpenAI Python SDK Compatibility

For OpenAI-compatible text routes:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8776/v1",
    api_key="your-api-token",
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[{"role": "user", "content": "Explain local model routing."}],
)
print(response.choices[0].message.content)
```

## Endpoint Index

Every HTTP route the server defines — **257** of them — generated from the
decorators in `codai/` by [`tools/gen_endpoint_index.py`](tools/gen_endpoint_index.py).
`Source` names the module(s) that own the route: one owned by `frontproxy/*` is served
by the torch-free front, one owned by `api/*` by an engine process, and a route listed
in both is proxied or aggregated by the front (see
[`docs/frontend-engine-split.md`](docs/frontend-engine-split.md)).

Routes the code registers with `include_in_schema=False` are **not** in
`/openapi.json` — the service, cluster and internal groups below are mostly those.
They are still part of the HTTP surface, which is why they are listed here.

### Public `/v1`

| Method | Path | Purpose | Source |
|---|---|---|---|
| `POST` | `/v1/3d/generate` | Generate a 3D model from a prompt | `api/spatial` |
| `GET` | `/v1/archive` | List archived generations | `api/app` |
| `DELETE` | `/v1/archive/{filename}` | Delete an archived file | `api/app` |
| `POST` | `/v1/audio/cleanup` | Clean / restore audio | `api/audio_clean` |
| `POST` | `/v1/audio/clone` | Clone a voice / synthesize cloned speech | `api/voice_clone` |
| `GET` | `/v1/audio/clone/engines` | List voice cloning engines | `api/voice_clone` |
| `POST` | `/v1/audio/convert` | Voice conversion (speech-to-speech) | `api/voice_convert` |
| `POST` | `/v1/audio/diarization` | Speaker diarization (who spoke when) | `api/transcriptions` |
| `POST` | `/v1/audio/generate` | Generate audio, music or SFX | `api/audio_gen` |
| `GET` | `/v1/audio/progress` | Audio generation progress | `api/audio_gen` |
| `POST` | `/v1/audio/speaker-embeddings` | Extract a speaker embedding (voiceprint) | `api/transcriptions` |
| `POST` | `/v1/audio/speaker-identify` | Identify the speaker in an audio clip | `api/transcriptions` |
| `POST` | `/v1/audio/speaker-verify` | Verify a clip against an enrolled speaker or a reference voiceprint | `api/transcriptions` |
| `GET` | `/v1/audio/speakers` | List enrolled speakers | `api/transcriptions` |
| `POST` | `/v1/audio/speakers` | Enroll a speaker voiceprint | `api/transcriptions` |
| `DELETE` | `/v1/audio/speakers/{name}` | Delete an enrolled speaker | `api/transcriptions` |
| `POST` | `/v1/audio/speech` | Text-to-speech synthesis | `api/tts` |
| `GET` | `/v1/audio/speech/capabilities` | What a TTS model can steer | `api/tts` |
| `POST` | `/v1/audio/stems` | Separate audio into stems | `api/audio_stems` |
| `POST` | `/v1/audio/transcriptions` | Transcribe audio to text | `api/transcriptions` |
| `GET` | `/v1/audio/voices` | List voice profiles | `api/voice_clone` |
| `POST` | `/v1/audio/voices` | Create a voice profile | `api/voice_clone` |
| `POST` | `/v1/audio/voices/extract` | Extract a voice profile from a sample | `api/voice_clone` |
| `GET` | `/v1/audio/voices/{name}` | Get a voice profile | `api/voice_clone` |
| `PATCH` | `/v1/audio/voices/{name}` | Update a voice profile | `api/voice_clone` |
| `DELETE` | `/v1/audio/voices/{name}` | Delete a voice profile | `api/voice_clone` |
| `POST` | `/v1/audio/watermark/detect` | Detect a coderai audio watermark | `api/voice_clone` |
| `GET` | `/v1/characters` | List character profiles | `api/characters` |
| `POST` | `/v1/characters` | Create or replace a character profile | `api/characters` |
| `POST` | `/v1/characters/extract` | Extract a character from media | `api/characters` |
| `POST` | `/v1/characters/generate` | Generate character reference images | `api/characters` |
| `GET` | `/v1/characters/{name}` | Get a character profile | `api/characters` |
| `PATCH` | `/v1/characters/{name}` | Update a character profile | `api/characters` |
| `DELETE` | `/v1/characters/{name}` | Delete a character profile | `api/characters` |
| `POST` | `/v1/chat/completions` | Chat completions | `api/text` |
| `POST` | `/v1/chunk` | Split a document into retrieval chunks | `api/chunking` |
| `POST` | `/v1/chunk/file` | Split an uploaded document into chunks | `api/chunking` |
| `GET` | `/v1/chunk/templates` | List the structure templates on offer | `api/chunking` |
| `POST` | `/v1/completions` | Legacy text completions | `api/text` |
| `POST` | `/v1/embeddings` | Create embeddings | `api/embeddings` |
| `GET` | `/v1/environments` | List environment profiles | `api/environments` |
| `POST` | `/v1/environments` | Create or replace an environment profile | `api/environments` |
| `POST` | `/v1/environments/extract` | Extract an environment from media | `api/environments` |
| `POST` | `/v1/environments/generate` | Generate environment reference images | `api/environments` |
| `GET` | `/v1/environments/{name}` | Get an environment profile | `api/environments` |
| `PATCH` | `/v1/environments/{name}` | Update an environment profile | `api/environments` |
| `DELETE` | `/v1/environments/{name}` | Delete an environment profile | `api/environments` |
| `GET` | `/v1/files/blob/{hash}` | Check an uploaded file exists | `api/uploads` |
| `POST` | `/v1/files/upload` | Upload media (content-addressed) | `api/uploads` |
| `GET` | `/v1/files/{filename}` | Download a generated file | `api/app` |
| `GET` | `/v1/health` | Liveness alias under the `/v1` prefix | `api/app` |
| `POST` | `/v1/images/deblur` | Deblur an image | `api/images` |
| `POST` | `/v1/images/depth` | Estimate a depth map | `api/images` |
| `POST` | `/v1/images/edits` | Edit an image (instruction / img2img) | `api/images` |
| `POST` | `/v1/images/faceswap` | Swap faces between images | `api/faceswap` |
| `POST` | `/v1/images/from3d` | Render a 3D model to an image | `api/spatial` |
| `POST` | `/v1/images/generations` | Generate images (text-to-image) | `api/images` |
| `POST` | `/v1/images/inpaint` | Inpaint a masked region | `api/images` |
| `POST` | `/v1/images/outfit` | Change outfit / clothing | `api/images` |
| `GET` | `/v1/images/progress` | Image generation progress | `api/images` |
| `POST` | `/v1/images/segment` | Segment an image | `api/images` |
| `POST` | `/v1/images/to3d` | Image to 3D model | `api/spatial` |
| `POST` | `/v1/images/unpixelate` | Restore a pixelated image | `api/images` |
| `POST` | `/v1/images/upscale` | Upscale an image | `api/images` |
| `GET` | `/v1/loras` | List registered LoRAs | `api/loras` |
| `GET` | `/v1/loras/blob/{hash}` | Check an uploaded LoRA blob exists | `api/loras` |
| `POST` | `/v1/loras/jobs/{job_id}/cancel` | Cancel a training job | `api/loras` |
| `GET` | `/v1/loras/progress` | LoRA training progress | `api/loras` |
| `POST` | `/v1/loras/register` | Serve an uploaded adapter under its own name | `api/loras` |
| `POST` | `/v1/loras/train` | Train a new LoRA | `api/loras` |
| `POST` | `/v1/loras/upload` | Upload LoRA weights (content-addressed) | `api/loras` |
| `GET` | `/v1/loras/{name}` | Get a registered LoRA | `api/loras` |
| `DELETE` | `/v1/loras/{name}` | Delete a registered LoRA | `api/loras` |
| `GET` | `/v1/models` | List available models | `api/app`, `frontproxy/app`, `system_app` |
| `GET` | `/v1/models/history` | Model load / eviction history | `api/app`, `frontproxy/app` |
| `POST` | `/v1/models/register` | Register a model at runtime | `api/models_transfer` |
| `POST` | `/v1/models/test` | Test run a model where it is configured | `api/model_test` |
| `GET` | `/v1/models/test/state` | What this engine currently believes | `api/model_test` |
| `POST` | `/v1/models/upload` | Upload model weights | `api/models_transfer` |
| `GET` | `/v1/models/uploaded` | Check an uploaded model is present | `api/models_transfer` |
| `POST` | `/v1/moderation` | Screen images and video before release | `api/moderation` |
| `GET` | `/v1/moderation/models` | Moderation models and availability | `api/moderation` |
| `POST` | `/v1/ocr` | OCR an image or PDF with a dedicated OCR engine | `api/ocr` |
| `POST` | `/v1/ocr/batch` | OCR many images/PDFs concurrently | `api/ocr` |
| `GET` | `/v1/ocr/engines` | List the OCR engines this install offers | `api/ocr` |
| `GET` | `/v1/ocr/schemas` | List available extraction schemas | `api/ocr` |
| `POST` | `/v1/ocr/schemas` | Create or update a user extraction schema | `api/ocr` |
| `GET` | `/v1/ocr/schemas/{name}` | Get one extraction schema | `api/ocr` |
| `DELETE` | `/v1/ocr/schemas/{name}` | Delete a user extraction schema | `api/ocr` |
| `POST` | `/v1/pipelines/audio-dub` | Audio dubbing pipeline | `api/pipelines` |
| `POST` | `/v1/pipelines/audio-music-dub` | Dub a song into another language | `api/custom_pipelines` |
| `POST` | `/v1/pipelines/audio-understand` | Transcribe and analyze audio | `api/custom_pipelines` |
| `GET` | `/v1/pipelines/custom` | List saved custom pipelines | `api/custom_pipelines` |
| `POST` | `/v1/pipelines/custom` | Create a custom pipeline | `api/custom_pipelines` |
| `PUT` | `/v1/pipelines/custom/{pipeline_id}` | Update a custom pipeline | `api/custom_pipelines` |
| `DELETE` | `/v1/pipelines/custom/{pipeline_id}` | Delete a custom pipeline | `api/custom_pipelines` |
| `POST` | `/v1/pipelines/custom/{pipeline_id}/run` | Run a saved custom pipeline | `api/custom_pipelines` |
| `POST` | `/v1/pipelines/image-to-video` | Image-to-video pipeline | `api/pipelines` |
| `POST` | `/v1/pipelines/run` | Run an inline pipeline definition | `api/custom_pipelines` |
| `GET` | `/v1/pipelines/step-types` | List available pipeline step types | `api/custom_pipelines` |
| `POST` | `/v1/pipelines/story` | Story pipeline (multi-scene) | `api/pipelines` |
| `POST` | `/v1/pipelines/video-dub` | Video dubbing pipeline | `api/pipelines` |
| `POST` | `/v1/rerank` | Rerank documents against a query (cross-encoder) | `api/rerank` |
| `POST` | `/v1/runpod/release` | Terminate one pod by id | `api/app` |
| `POST` | `/v1/runpod/repurpose` | Move a rented pod to another pool or image, in place | `api/app` |
| `POST` | `/v1/runpod/scale` | Ask for warm pods on a model | `api/app` |
| `GET` | `/v1/runpod/scale/{request_id}` | How a scale request ended | `api/app` |
| `GET` | `/v1/runpod/spend` | RunPod spend and pod status | `api/app` |
| `GET` | `/v1/runpod/status` | RunPod fleet status | `api/app` |
| `GET` | `/v1/video/compose` | List composition jobs | `api/compose` |
| `POST` | `/v1/video/compose` | Compose a finished video from scenes | `api/compose` |
| `GET` | `/v1/video/compose/{job_id}` | Composition job status | `api/compose` |
| `POST` | `/v1/video/compose/{job_id}/cancel` | Cancel a composition job | `api/compose` |
| `GET` | `/v1/video/compose/{job_id}/log` | A composition job's own log | `api/compose` |
| `POST` | `/v1/video/dub` | Dub a video | `api/video` |
| `POST` | `/v1/video/from3d` | Render a 3D model to a video | `api/spatial` |
| `POST` | `/v1/video/generations` | Generate video | `api/video` |
| `POST` | `/v1/video/interpolate` | Interpolate video frames | `api/video` |
| `GET` | `/v1/video/presenter/engines` | Installed lip-sync engines | `api/compose` |
| `GET` | `/v1/video/progress` | Video generation progress | `api/video` |
| `POST` | `/v1/video/subtitle` | Subtitle / caption a video | `api/video` |
| `POST` | `/v1/video/talking-head` | A character saying one line | `api/compose` |
| `GET` | `/v1/video/talking-head/{job_id}` | Talking-head job status | `api/compose` |
| `POST` | `/v1/video/talking-head/{job_id}/cancel` | Cancel a talking-head job | `api/compose` |
| `POST` | `/v1/video/to3d` | Video to 3D model | `api/spatial` |
| `POST` | `/v1/video/upscale` | Upscale a video | `api/video` |

### Service and operational

| Method | Path | Purpose | Source |
|---|---|---|---|
| `GET` | `/boot` | What this process did on the way up, with timings and the container phases | `api/app` |
| `GET` | `/chat` | Chat / Studio UI | `frontproxy/ui_pages` |
| `GET` | `/coderai/capabilities` | Server capability document | `api/app` |
| `GET` | `/favicon.ico` | Favicon | `api/app`, `frontproxy/ui_pages` |
| `GET` | `/health` | vLLM-compatible liveness alias (clients hard-code this path) | `api/app` |
| `GET` | `/healthz` | Cheap liveness probe touching no model state; the supervisor polls it | `api/app`, `frontproxy/app`, `system_app` |
| `GET` | `/login` | Login page | `frontproxy/ui_pages` |
| `POST` | `/login` | Authenticate admin login | `admin/routes` |
| `GET` | `/logout` | Log out | `admin/routes` |
| `GET` | `/logs` | This process's own output from a cursor — the only way to read a pod's log | `api/app` |
| `GET` | `/metrics` | Prometheus exposition (open when `server.metrics_public`) | `frontproxy/app` |

### Admin HTML

| Method | Path | Purpose | Source |
|---|---|---|---|
| `GET` | `/admin` | Dashboard | `frontproxy/ui_pages` |
| `GET` | `/admin/archive` | Generation archive page | `frontproxy/ui_pages` |
| `GET` | `/admin/change-password` | Password change page | `frontproxy/ui_pages` |
| `POST` | `/admin/change-password` | Change admin password | `admin/routes` |
| `GET` | `/admin/cluster` | Cluster page (nodes, pools, reachability) | `frontproxy/ui_pages` |
| `GET` | `/admin/models` | Model management page | `frontproxy/ui_pages` |
| `GET` | `/admin/runpod` | RunPod page (pools, pods, spend) | `frontproxy/ui_pages` |
| `GET` | `/admin/settings` | Settings page | `frontproxy/ui_pages` |
| `GET` | `/admin/tasks` | Tasks page (running jobs, engine tiles, logs) | `frontproxy/ui_pages` |
| `GET` | `/admin/tokens` | API token page | `frontproxy/ui_pages` |
| `GET` | `/admin/users` | User page | `frontproxy/ui_pages` |

### Admin API

| Method | Path | Purpose | Source |
|---|---|---|---|
| `GET` | `/admin/api/accel-loras` | List cached distill-LoRA files for acceleration | `admin/routes` |
| `GET` | `/admin/api/accel-presets` | List acceleration / distillation presets | `admin/routes` |
| `GET` | `/admin/api/archive` | List archived generations | `frontproxy/admin_data` |
| `GET` | `/admin/api/archive-settings` | Archive retention settings | `frontproxy/admin_data` |
| `GET` | `/admin/api/archive/{gen_id}` | One archived generation | `frontproxy/admin_data` |
| `DELETE` | `/admin/api/archive/{gen_id}` | Delete an archived generation | `frontproxy/admin_data` |
| `GET` | `/admin/api/archive/{gen_id}/files/{filename}` | Fetch one file from an archived generation | `frontproxy/admin_data` |
| `POST` | `/admin/api/batch` | Run several admin API calls in one request | `frontproxy/app` |
| `DELETE` | `/admin/api/cache` | Clear the model cache | `admin/routes` |
| `GET` | `/admin/api/cache-stats` | Model cache statistics | `admin/routes` |
| `GET` | `/admin/api/cached-models` | List cached models | `admin/routes` |
| `DELETE` | `/admin/api/cached-models/{model_id:path}` | Evict a cached model | `admin/routes` |
| `GET` | `/admin/api/characters` | List characters | `admin/routes` |
| `GET` | `/admin/api/characters/{name}` | Get a character | `admin/routes` |
| `DELETE` | `/admin/api/characters/{name}` | Delete a character | `admin/routes` |
| `GET` | `/admin/api/characters/{name}/thumbnail` | Character thumbnail | `admin/routes` |
| `GET` | `/admin/api/cluster` | Everything this front can send work to, with reachability | `frontproxy/app` |
| `GET` | `/admin/api/cluster/logs` | Tail a cluster node's log | `frontproxy/app` |
| `GET` | `/admin/api/colibri/default-models` | List downloadable colibri default models | `admin/routes` |
| `POST` | `/admin/api/download-cancel-all` | Cancel all active downloads | `admin/routes` |
| `POST` | `/admin/api/download-cancel/{session_id}` | Cancel a download | `admin/routes` |
| `GET` | `/admin/api/download-stream/{session_id}` | Stream model download progress | `admin/routes` |
| `GET` | `/admin/api/downloads` | List active downloads | `admin/routes` |
| `GET` | `/admin/api/ds4/default-models` | List downloadable ds4 default models | `admin/routes` |
| `POST` | `/admin/api/embedder-venv-install` | Build an isolated embedder venv | `admin/routes` |
| `POST` | `/admin/api/embedder-venv-remove` | Delete an isolated embedder venv | `admin/routes` |
| `GET` | `/admin/api/embedder-venvs` | Isolated embedder venv status | `admin/routes` |
| `GET` | `/admin/api/engines` | Engine nodes: name, health, VRAM, loaded models (front's own registry) | `frontproxy/app` |
| `POST` | `/admin/api/engines/{eid}/restart` | Restart one engine subprocess | `frontproxy/app` |
| `GET` | `/admin/api/environments` | List environments | `admin/routes` |
| `GET` | `/admin/api/environments/{name}` | Get an environment | `admin/routes` |
| `DELETE` | `/admin/api/environments/{name}` | Delete an environment | `admin/routes` |
| `GET` | `/admin/api/environments/{name}/thumbnail` | Environment thumbnail | `admin/routes` |
| `GET` | `/admin/api/gpu-stats` | Per-GPU utilisation, VRAM and temperature | `frontproxy/app` |
| `GET` | `/admin/api/hf-files` | List files in a Hugging Face repo | `admin/routes` |
| `GET` | `/admin/api/hf-model-files` | List Hugging Face model files | `admin/routes` |
| `GET` | `/admin/api/hf-model-info` | Get Hugging Face model info | `admin/routes` |
| `GET` | `/admin/api/hf-search` | Search Hugging Face models | `admin/routes` |
| `POST` | `/admin/api/lipsync-engine-install` | Install a lip-sync engine | `admin/routes` |
| `POST` | `/admin/api/lipsync-engine-remove` | Remove a lip-sync engine | `admin/routes` |
| `GET` | `/admin/api/lipsync-engines` | Lip-sync engine status | `admin/routes` |
| `POST` | `/admin/api/model-add-known` | Register a model in config without downloading | `admin/routes` |
| `POST` | `/admin/api/model-configure` | Update a model's configuration | `admin/routes` |
| `POST` | `/admin/api/model-disable` | Disable a model | `admin/routes` |
| `POST` | `/admin/api/model-download` | Download a model | `admin/routes` |
| `POST` | `/admin/api/model-enable` | Enable a model | `admin/routes` |
| `POST` | `/admin/api/model-free-disk` | Delete a model's files but keep its config | `admin/routes` |
| `POST` | `/admin/api/model-load` | Load a model into memory | `admin/routes`, `frontproxy/app` |
| `GET` | `/admin/api/model-loaded-status` | Model load status | `admin/routes`, `frontproxy/app` |
| `POST` | `/admin/api/model-mark-download` | List a model for later download | `admin/routes` |
| `POST` | `/admin/api/model-quantize` | Quantize a model to fast-kernel 4-bit | `admin/routes` |
| `POST` | `/admin/api/model-unload` | Unload a model | `admin/routes`, `frontproxy/app` |
| `POST` | `/admin/api/model-unmark-download` | Remove a model from the download list | `admin/routes` |
| `POST` | `/admin/api/model-upload` | Upload a model file | `admin/routes` |
| `GET` | `/admin/api/models` | List configured models | `admin/routes` |
| `DELETE` | `/admin/api/models/{model_identifier}` | Remove a configured model | `admin/routes` |
| `POST` | `/admin/api/ocr/build-venv` | Build a Paddle/Surya isolated venv | `admin/routes` |
| `GET` | `/admin/api/ocr/build-venv/status` | OCR isolated-venv build status | `admin/routes` |
| `GET` | `/admin/api/ocr/schemas` | List OCR extraction schemas | `admin/routes` |
| `POST` | `/admin/api/ocr/schemas` | Create/update a user OCR schema | `admin/routes` |
| `GET` | `/admin/api/ocr/schemas/{name}` | Get one OCR extraction schema | `admin/routes` |
| `DELETE` | `/admin/api/ocr/schemas/{name}` | Delete a user OCR schema | `admin/routes` |
| `GET` | `/admin/api/quantize-capabilities` | GPTQ/AWQ quantization availability | `admin/routes` |
| `GET` | `/admin/api/quantize-status` | Quantization job status | `admin/routes` |
| `GET` | `/admin/api/runpod/gpu-types` | List RunPod GPU types + prices | `admin/routes` |
| `POST` | `/admin/api/runpod/pod-adopt` | Take an unmanaged RunPod pod under local management | `admin/routes` |
| `GET` | `/admin/api/runpod/pod-logs` | Fetch a RunPod pod's container/vLLM logs | `admin/routes` |
| `POST` | `/admin/api/runpod/pod-release` | Terminate one RunPod pod by id, now | `admin/routes` |
| `POST` | `/admin/api/runpod/seed-volumes` | Pre-fill a model's regional volumes | `admin/routes` |
| `GET` | `/admin/api/runpod/stats` | RunPod pod usage + spend stats | `admin/routes` |
| `GET` | `/admin/api/runpod/status` | RunPod account pods + live cost | `admin/routes` |
| `GET` | `/admin/api/settings` | Current server settings | `frontproxy/admin_data` |
| `POST` | `/admin/api/settings` | Update server settings | `admin/routes` |
| `GET` | `/admin/api/status` | System, model, VRAM, queue and recent-activity status | `frontproxy/app` |
| `GET` | `/admin/api/system-stats` | Host CPU / RAM / disk / temperature snapshot | `frontproxy/app` |
| `POST` | `/admin/api/system/reload` | Reload server configuration | `admin/routes` |
| `GET` | `/admin/api/tasks` | List active and recent tasks | `admin/routes`, `frontproxy/app` |
| `DELETE` | `/admin/api/tasks/{task_id}` | Remove a finished task | `admin/routes`, `frontproxy/app` |
| `POST` | `/admin/api/tasks/{task_id}/cancel` | Cancel a task | `admin/routes` |
| `POST` | `/admin/api/tasks/{task_id}/interrupt` | Interrupt a running task | `admin/routes` |
| `POST` | `/admin/api/tasks/{task_id}/pause` | Pause a running task | `admin/routes` |
| `POST` | `/admin/api/tasks/{task_id}/restart` | Restart a training task | `admin/routes` |
| `POST` | `/admin/api/tasks/{task_id}/resume` | Resume a paused task | `admin/routes` |
| `GET` | `/admin/api/tokens` | List API tokens | `frontproxy/admin_data` |
| `POST` | `/admin/api/tokens` | Create an API token | `frontproxy/admin_data` |
| `PATCH` | `/admin/api/tokens/{token_id}` | Rename / re-scope a token | `frontproxy/admin_data` |
| `DELETE` | `/admin/api/tokens/{token_id}` | Delete a token | `frontproxy/admin_data` |
| `GET` | `/admin/api/turboquant-info` | TurboQuant backend availability | `admin/routes` |
| `GET` | `/admin/api/usage` | Requests, errors and time per API key / model / kind | `frontproxy/app` |
| `GET` | `/admin/api/users` | List users | `frontproxy/admin_data` |
| `POST` | `/admin/api/users` | Create a user | `frontproxy/admin_data` |
| `DELETE` | `/admin/api/users/{user_id}` | Delete a user | `frontproxy/admin_data` |
| `GET` | `/admin/api/voices` | List voices | `admin/routes` |
| `GET` | `/admin/api/voices/{name}` | Get a voice | `admin/routes` |
| `DELETE` | `/admin/api/voices/{name}` | Delete a voice | `admin/routes` |
| `GET` | `/admin/api/whoami` | Validate the current admin session | `admin/routes` |

### Cluster (node ↔ head)

| Method | Path | Purpose | Source |
|---|---|---|---|
| `GET` | `/cluster/logs` | This node's log, for the head's cluster page | `frontproxy/app` |
| `POST` | `/cluster/model-load` | The head tells this node to load a model | `frontproxy/app` |
| `POST` | `/cluster/model-unload` | The head tells this node to unload a model | `frontproxy/app` |
| `POST` | `/cluster/reload-config` | The head tells this node which models it owns | `frontproxy/app` |
| `GET` | `/cluster/rpc-servers` | The llama.cpp RPC endpoints this node exposes | `frontproxy/app` |
| `GET` | `/cluster/state` | This node's aggregate state (models, engines, RPC peers) | `frontproxy/app` |

### Internal (front ↔ engine)

| Method | Path | Purpose | Source |
|---|---|---|---|
| `GET` | `/internal/engine-state` | This engine's loaded models, VRAM and queue depth | `api/app`, `system_app` |
| `POST` | `/internal/evict-vram` | A co-located sibling asks this engine to release VRAM | `api/app` |
| `POST` | `/internal/gpu-release` | Lift a sibling's GPU reservation | `api/app` |
| `POST` | `/internal/gpu-reserve` | A sibling reserves the card for an exclusive op | `api/app` |
| `POST` | `/internal/reload-config` | Re-read `config.json` / `models.json` after an admin edit | `api/app`, `system_app` |
| `POST` | `/internal/thermal-pause` | Front-driven thermal pause at the next safe point | `api/app` |
| `POST` | `/internal/thermal-resume` | Lift the thermal pause once the hardware cooled | `api/app` |
