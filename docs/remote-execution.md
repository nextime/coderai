# Running anything somewhere else

Every model coderai serves can now be served from another machine instead: a
second coderai, a bare `llama-server`/vLLM/SGLang, a rented RunPod pod, or a
hosted OpenAI-compatible API. Nothing is downloaded locally, no local VRAM is
used, and clients see no difference.

There are four mechanisms, because the subsystems are shaped differently. Pick
by what you are moving.

| What you're moving | Mechanism | Where you configure it |
|---|---|---|
| An isolated-venv worker (vLLM, ds4, kt, h3, pyannote, MeloTTS, parler, Canary, CrisperWhisper) | `service_url` on the worker | engine config block, or `CODERAI_<NAME>_SERVICE_URL` |
| colibri / k3 (mux engines) | `service_url` + `tools/mux_service.py` on the far side | `colibri.service_url` / `k3.service_url` |
| Any text model, GGUF included | `service_url` on the model entry | models.json |
| Images, video, embeddings, rerank, OCR, TTS, STT, voice, audio, stems, 3D, pipelines | remote gateway | `remotes.endpoints` in config.json |

---

## 1. Workers with a service

Each isolated-venv worker's `ensure_service()` returns a base URL, so pointing
it at a URL is enough — it health-checks the far side and returns it instead of
building a venv, downloading weights, or spawning anything:

```json
{ "vllm": { "enabled": true, "service_url": "https://pod-abc-8000.proxy.runpod.net" } }
```

Env equivalents: `CODERAI_VLLM_SERVICE_URL`, `CODERAI_DS4_SERVICE_URL`,
`CODERAI_KT_SERVICE_URL`, `CODERAI_H3_SERVICE_URL`,
`CODERAI_PYANNOTE_SERVICE_URL`, `CODERAI_MELOTTS_SERVICE_URL`,
`CODERAI_PARLER_SERVICE_URL`, `CODERAI_CANARY_SERVICE_URL`,
`CODERAI_CRISPERWHISPER_SERVICE_URL`.

A wrong or dead URL fails immediately with
`configured service_url … is not answering its health check`, rather than as a
confusing timeout later. vLLM additionally verifies the remote actually serves
the requested model — the OCR path asks for surya-2 by name, and a generic
remote vLLM would otherwise be proxied silently.

## 2. colibri and k3

These two drive their C engine over a stdin/stdout mux protocol, so they have no
HTTP surface to redirect. `tools/mux_service.py` is that surface: run it on the
machine that holds the container, and point coderai at it.

On the remote host (needs the model container and the built engine):

```bash
python tools/mux_service.py --kind colibri --port 8092 \
    --model-dir /models/GLM-5.2-int4 --ctx 100000
# k3: --kind k3 --model-dir /models/Kimi-K3
```

In coderai:

```json
{ "colibri": { "enabled": true, "service_url": "http://engine-box:8092" } }
```

`codai/api/mux_remote.RemoteMuxEngine` presents the same surface as a local
`MuxEngine`, so streaming, stats, pause/resume and mid-turn cancellation all
behave identically — cancellation travels as a dropped connection.

## 3. Text models, including GGUF

GGUF models load in-process through llama-cpp-python; there is no service to
redirect. So point the *model* at a URL instead — add `service_url` to its
models.json entry and `codai/backends/remote_openai.py` proxies it:

```json
{
  "path": "/AI/guffcache/gemma-4-31B-Q4_0.gguf",
  "service_url": "http://gpu-box:8080",
  "served_model": "gemma-4-31b",
  "api_key": ""
}
```

`service_url` may include `/v1` or not. `served_model` is the name the remote
knows it by (defaults to the model's own name); `api_key` adds a bearer token.
The manager checks this before every engine route, and such a model is never
downloaded, cached, or charged local VRAM.

The far side can be anything OpenAI-compatible: another coderai, `llama-server`,
vLLM, SGLang, TGI, or a hosted API.

### On RunPod

A RunPod pod picks its server from the model:

```json
{ "runpod": { "mode": "pods", "engine": "auto",
              "hf_gguf": "bartowski/gemma-4-31B-GGUF:Q4_0", "ctx": 32768 } }
```

* `engine: auto` — an HF repo goes to vLLM (`vllm/vllm-openai`), a GGUF goes to
  llama.cpp (`ghcr.io/ggml-org/llama.cpp:server-cuda`).
* `engine: vllm | llamacpp` — force one.

GGUF does **not** simply go to vLLM. vLLM's GGUF loader is experimental: single
file only (sharded GGUFs must be merged first), a limited architecture list, and
it still wants the original repo for the tokenizer. llama.cpp's own server image
serves the whole GGUF catalogue as-is, and exposes the same `/v1` surface, so
the readiness probe and the proxy are unchanged.

`hf_gguf` is required for a llama.cpp pod and takes llama.cpp's own `-hf` form
(`user/repo:Q4_K_M`). A local `/AI/…/foo.gguf` path means nothing on a rented
machine, and coderai does not upload multi-GB weights.

## 4. Everything else — the remote gateway

Images, video, embeddings, rerank, OCR, TTS, STT, voice cloning, audio
generation, stems, 3D and pipelines all load their model in-process. Rather than
write a service for each, note that *another coderai already exposes all of
them* — so it can be the service.

`codai/api/remote_gateway.py` sits in front of the routes, and when a request is
addressed to something configured as remote it replays it verbatim to that
endpoint and streams the answer back:

```json
{
  "remotes": {
    "enabled": true,
    "api_key": "",
    "max_body_mb": 512,
    "endpoints": {
      "images": "http://gpu-box:8000",
      "video": "https://pod-xyz-8000.proxy.runpod.net",
      "embeddings": "http://cpu-box:8000"
    }
  }
}
```

Capability keys: `images`, `video`, `embeddings`, `rerank`, `ocr`, `tts`, `stt`,
`voice`, `speaker`, `audio_gen`, `stems`, `audio_clean`, `spatial`, `faceswap`,
`loras`, `characters`, `environments`, `pipelines`. Env equivalents:
`CODERAI_REMOTE_IMAGES_URL` and friends.

Precedence: a model's own `service_url` wins over the capability map.

Notes:

* `/v1/chat/completions` and `/v1/completions` are deliberately **not** gatewayed
  — they go through `RemoteOpenAIBackend`, which keeps usage accounting, tool
  parsing and the model manager's bookkeeping intact.
* Auth and rate limiting run *before* the gateway, so a request still has to be
  authorised here before it is forwarded.
* When nothing is configured as remote the middleware returns on its first
  branch and never buffers a body.
* Requests larger than `max_body_mb` are served locally rather than buffered in
  RAM; raise it if you forward large uploads.

## Deployment shapes this enables

* **One small coderai, many remote GPUs** — the local instance holds the API,
  auth, catalogue and UI; each capability points at a box or pod that owns one
  model family.
* **Per-engine pod images** — vLLM's and llama.cpp's own images serve LLMs with
  no coderai on the far side; the coderai image is only needed where a coderai
  subsystem runs (the mux service, or a gateway target).
* **Burst to RunPod** — pin a model to the `runpod` backend and it provisions,
  scales and reaps pods against the configured budget (see `docs/runpod.md`).
