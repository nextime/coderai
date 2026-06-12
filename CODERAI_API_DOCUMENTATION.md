# CoderAI API Documentation

This document describes the full HTTP API exposed by CoderAI, including OpenAI-compatible endpoints, native multimodal endpoints, profile/LoRA APIs, pipelines, admin APIs, examples, and end-to-end workflows.

The API is implemented with FastAPI in `codai/api/app.py` and routers under `codai/api/`, with admin routes under `codai/admin/routes.py`.

Interactive, always-up-to-date OpenAPI docs are served by the running server at
`/docs` (Swagger UI, linked as **API Docs** in the admin nav) and `/redoc`, with the
raw schema at `/openapi.json`. Every endpoint there carries a tag, summary, and
per-field descriptions generated from the code.

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

Static assets are mounted under `/static/admin/*`.

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
| `DELETE` | `/admin/api/cache` | `cache_type=all|hf|gguf` | Clear cache |
| `DELETE` | `/admin/api/cached-models/{model_id:path}` | `cache_type` | Delete cached model |
| `POST` | `/admin/api/model-enable` | `{path|model_id,model_type}` | Enable model in config |
| `POST` | `/admin/api/model-disable` | `{path|model_id,config_id?}` | Disable model |
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

### Public `/v1` and Discovery

| Method | Path |
|---|---|
| `GET` | `/v1/models` |
| `GET` | `/coderai/capabilities` |
| `GET` | `/v1/files/{filename}` |
| `GET` | `/v1/archive` |
| `DELETE` | `/v1/archive/{filename}` |
| `POST` | `/v1/chat/completions` |
| `POST` | `/v1/completions` |
| `GET` | `/v1/images/progress` |
| `POST` | `/v1/images/generations` |
| `POST` | `/v1/images/edits` |
| `POST` | `/v1/images/inpaint` |
| `POST` | `/v1/images/upscale` |
| `POST` | `/v1/images/depth` |
| `POST` | `/v1/images/segment` |
| `POST` | `/v1/images/deblur` |
| `POST` | `/v1/images/unpixelate` |
| `POST` | `/v1/images/outfit` |
| `POST` | `/v1/images/faceswap` |
| `GET` | `/v1/video/progress` |
| `POST` | `/v1/video/generations` |
| `POST` | `/v1/video/upscale` |
| `POST` | `/v1/video/subtitle` |
| `POST` | `/v1/video/interpolate` |
| `POST` | `/v1/video/dub` |
| `POST` | `/v1/audio/transcriptions` |
| `POST` | `/v1/audio/speech` |
| `GET` | `/v1/audio/progress` |
| `POST` | `/v1/audio/generate` |
| `GET` | `/v1/audio/voices` |
| `POST` | `/v1/audio/voices` |
| `GET` | `/v1/audio/voices/{name}` |
| `PATCH` | `/v1/audio/voices/{name}` |
| `DELETE` | `/v1/audio/voices/{name}` |
| `POST` | `/v1/audio/voices/extract` |
| `POST` | `/v1/audio/clone` |
| `POST` | `/v1/audio/convert` |
| `POST` | `/v1/audio/stems` |
| `POST` | `/v1/audio/cleanup` |
| `POST` | `/v1/embeddings` |
| `POST` | `/v1/characters` |
| `GET` | `/v1/characters` |
| `GET` | `/v1/characters/{name}` |
| `PATCH` | `/v1/characters/{name}` |
| `DELETE` | `/v1/characters/{name}` |
| `POST` | `/v1/characters/generate` |
| `POST` | `/v1/characters/extract` |
| `POST` | `/v1/environments` |
| `GET` | `/v1/environments` |
| `GET` | `/v1/environments/{name}` |
| `PATCH` | `/v1/environments/{name}` |
| `DELETE` | `/v1/environments/{name}` |
| `POST` | `/v1/environments/generate` |
| `POST` | `/v1/environments/extract` |
| `POST` | `/v1/loras/train` |
| `GET` | `/v1/loras/progress` |
| `GET` | `/v1/loras` |
| `GET` | `/v1/loras/{name}` |
| `DELETE` | `/v1/loras/{name}` |
| `POST` | `/v1/images/to3d` |
| `POST` | `/v1/images/from3d` |
| `POST` | `/v1/video/to3d` |
| `POST` | `/v1/video/from3d` |
| `POST` | `/v1/3d/generate` |
| `POST` | `/v1/pipelines/image-to-video` |
| `POST` | `/v1/pipelines/video-dub` |
| `POST` | `/v1/pipelines/story` |
| `POST` | `/v1/pipelines/audio-dub` |
| `GET` | `/v1/pipelines/custom` |
| `GET` | `/v1/pipelines/step-types` |
| `POST` | `/v1/pipelines/custom` |
| `PUT` | `/v1/pipelines/custom/{pipeline_id}` |
| `DELETE` | `/v1/pipelines/custom/{pipeline_id}` |
| `POST` | `/v1/pipelines/custom/{pipeline_id}/run` |
| `POST` | `/v1/pipelines/run` |
| `POST` | `/v1/pipelines/audio-understand` |
| `POST` | `/v1/pipelines/audio-music-dub` |

### Admin API

| Method | Path |
|---|---|
| `GET` | `/admin/api/status` |
| `POST` | `/admin/api/users` |
| `DELETE` | `/admin/api/users/{user_id}` |
| `GET` | `/admin/api/tokens` |
| `POST` | `/admin/api/tokens` |
| `DELETE` | `/admin/api/tokens/{token_id}` |
| `GET` | `/admin/api/models` |
| `POST` | `/admin/api/model-download` |
| `GET` | `/admin/api/download-stream/{session_id}` |
| `GET` | `/admin/api/downloads` |
| `POST` | `/admin/api/download-cancel/{session_id}` |
| `POST` | `/admin/api/model-upload` |
| `DELETE` | `/admin/api/models/{model_identifier}` |
| `GET` | `/admin/api/hf-files` |
| `GET` | `/admin/api/cached-models` |
| `GET` | `/admin/api/cache-stats` |
| `DELETE` | `/admin/api/cache` |
| `DELETE` | `/admin/api/cached-models/{model_id:path}` |
| `POST` | `/admin/api/model-enable` |
| `POST` | `/admin/api/model-disable` |
| `GET` | `/admin/api/model-loaded-status` |
| `POST` | `/admin/api/model-load` |
| `POST` | `/admin/api/model-unload` |
| `POST` | `/admin/api/model-configure` |
| `POST` | `/admin/api/system/reload` |
| `GET` | `/admin/api/settings` |
| `POST` | `/admin/api/settings` |
| `GET` | `/admin/api/archive` |
| `GET` | `/admin/api/archive/{gen_id}` |
| `DELETE` | `/admin/api/archive/{gen_id}` |
| `GET` | `/admin/api/archive/{gen_id}/files/{filename}` |
| `GET` | `/admin/api/archive-settings` |
| `GET` | `/admin/api/hf-search` |
| `GET` | `/admin/api/hf-model-files` |
| `GET` | `/admin/api/hf-model-info` |
| `GET` | `/admin/api/characters` |
| `GET` | `/admin/api/characters/{name}` |
| `GET` | `/admin/api/characters/{name}/thumbnail` |
| `DELETE` | `/admin/api/characters/{name}` |
| `GET` | `/admin/api/environments` |
| `GET` | `/admin/api/environments/{name}` |
| `GET` | `/admin/api/environments/{name}/thumbnail` |
| `DELETE` | `/admin/api/environments/{name}` |
| `GET` | `/admin/api/voices` |
| `GET` | `/admin/api/voices/{name}` |
| `DELETE` | `/admin/api/voices/{name}` |
