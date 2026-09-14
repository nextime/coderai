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

A pod row with no `image` uses the published one for its capability —
`ghcr.io/nextime/coderai-<capability>:latest`, a trimmed coderai carrying only
that capability's dependencies — so turning a capability remote is one choice,
not an image name you have to know. `rerank` rides the `embeddings` image,
`speaker` the `stt` one, and `stems`/`audio_clean`/`audio_gen` the `audio` one.
`loras`, `characters`, `environments` and `pipelines` have no published image and
need one named explicitly. Point the whole set at your own build with
`CODERAI_CAPABILITY_IMAGE_REPO` / `CODERAI_CAPABILITY_IMAGE_TAG`, or override a
single row's `image`.

Capability keys: `images`, `video`, `embeddings`, `rerank`, `ocr`, `tts`, `stt`,
`voice`, `speaker`, `audio_gen`, `stems`, `audio_clean`, `spatial`, `faceswap`,
`loras`, `characters`, `environments`, `pipelines`. Env equivalents:
`CODERAI_REMOTE_IMAGES_URL` and friends.

Precedence: a model's own `service_url` wins over the capability map.

Every `/v1` endpoint is mapped to one of these capabilities — a test asserts it,
so a newly added endpoint cannot quietly stay local. Three are excluded on
purpose: `/v1/chat/completions` and `/v1/completions` (RemoteOpenAIBackend serves
those) and `/v1/models` (the catalogue is this instance's own).

What is **not** gatewayed, by design:

* `/admin/**` — administers *this* instance: its model list, its engines, its
  settings. To manage a remote coderai, open its own admin.
* `/internal/**` — the front↔engine control plane (VRAM reservations, thermal
  pause, config reload), already gated by the internal token.
* `/chat`, `/login`, `/logout`, `/healthz`, `/coderai/capabilities` — this
  instance's own UI and identity.

### Generated files

With the default `response_format: "url"` a render answers with a URL built from
the base of the machine that produced it — `http://pod-xyz:8000/v1/files/out.png`
— which the client usually cannot reach, and which 404s if fetched here because
the file is on the pod. So the gateway rewrites those URLs to point at this
instance and remembers which remote holds each file; a later
`GET /v1/files/<name>` is forwarded back there. Nothing to configure.

Two consequences worth knowing: the file lives on the pod, so **a pod reaped for
idleness takes its outputs with it** — fetch what you need, or ask for
`response_format: "b64_json"` and get the bytes inline. And the mapping is a
routing hint held in memory (bounded, newest 4096), not a store: it does not
survive a restart.

### Letting coderai rent the pod

A URL means you started the machine and you own its bill. To have coderai
provision, scale and reap the pod instead, give the capability a pod block —
the same fields a model's `runpod` block takes:

```json
{
  "remotes": {
    "endpoints": {"images": "runpod"},
    "pods": {
      "images": {
        "image": "registry.example.com/coderai:base",
        "min_vram_gb": 24, "max_hourly_usd": 0.6,
        "max_pods": 2, "idle_timeout_s": 300,
        "cost_limit_usd": 20, "cost_period": "day"
      }
    }
  }
}
```

`"endpoints": {"images": "runpod"}` is shorthand for "use the pod block for
images"; a capability listed under `pods` is routed there even without it.

The first matching request provisions a pod, the gateway borrows it for the
duration of the request and hands it back, and the scaler tears it down after
`idle_timeout_s` — so an idle capability costs nothing. Rate caps, rolling spend
limits, the global budget, the stale-pod reaper and the stats page all apply
exactly as they do for a RunPod-served LLM; capability pools show up there as
`capability:<name>`.

Pod engines (`engine`):

| value | image | ready when |
|---|---|---|
| `auto` | vLLM, or llama.cpp for a GGUF | `/v1/models` |
| `vllm` / `llamacpp` | forced | `/v1/models` |
| `coderai` | your `image` (required) | `/healthz` |
| `custom` | your `image`, `docker_args` verbatim | `health_path` |

`coderai` is the default for a capability pool: the far side is a whole coderai,
reached through the same API.

### Where the images come from

RunPod pulls the image from a registry over the public internet — it cannot use
anything on your machine. What you need depends on the engine:

| Engine | Image | You publish anything? |
|---|---|---|
| `vllm` | `vllm/vllm-openai:latest` | no — public |
| `llamacpp` | `ghcr.io/ggml-org/llama.cpp:server-cuda` | no — public |
| `coderai` / `custom` | yours | **yes** |

So LLM pods need no work at all. A capability pod needs a coderai image somewhere
RunPod can reach. Any registry works:

* **GHCR** (`ghcr.io/<you>/coderai`) — the one `DISTRIBUTION.md` targets, free for
  public images, tied to a GitHub token for private ones.
* **Docker Hub** — one free private repo; watch the pull-rate limits.
* **AWS ECR / Google Artifact Registry / Azure ACR** — if a pod runs in the same
  region the pull is fast and egress is cheap.
* **Your own registry** — needs to be reachable from RunPod's network and to
  serve valid TLS.

Publishing it:

```bash
docker tag coderai:base ghcr.io/<you>/coderai:base
docker login ghcr.io -u <you>          # a PAT with write:packages
docker push ghcr.io/<you>/coderai:base
```

Two warnings worth having in advance. The flattened `coderai:base` is ~26.6 GB —
the first push takes a while, and **every cold pod pays that pull** before it
answers `/healthz`, so raise `boot_timeout_s`/`load_timeout_s` and consider a
larger `container_disk_gb`. And a full coderai carries every subsystem; if a pod
only ever renders video, a trimmed image boots far faster and costs less per
cold start.

### Trimmed capability images

A pod that only renders video has no use for the STT, OCR, voice-cloning or LLM
stacks, and every cold pod pays for what it pulls. `packaging/runpod/` builds a
coderai carrying one capability's dependencies:

```bash
# all of them, with login, resume, push and the make-them-public URLs:
./packaging/runpod/publish_capability_images.sh

# or one at a time:
./packaging/runpod/build_capability_image.sh video ghcr.io/<you>/coderai-video:latest
```

`publish_capability_images.sh` is resumable — it skips images already built for
the current version, so re-running after a failure does not rebuild 7 GB of
torch. Useful knobs: `PROFILES="images video"`, `REBUILD=1`, `NO_PUSH=1`,
`NS=ghcr.io/<you>`, `YES=1` (no prompts).

If a profile's wheels need a compiler, list the apt packages in
`profiles/<profile>.build-deps`; they are installed and purged inside one layer,
so the toolchain never ships. `audio` does this for deepfilterlib's Rust
extensions, and costs ~50 MB more than a profile that needs nothing.

Profiles live in `packaging/runpod/profiles/` — `core.txt` (what any pod needs to
boot) plus one file per capability: images, video, tts, stt, voice, audio,
embeddings, ocr, faceswap. Add or trim entries there; the routers import lazily,
so a pod only needs the libraries its own endpoints touch.

The build fails if the profile is missing something the app imports, and the
script then boots the container and waits for `/healthz` — because a pod that
builds but never answers is a failure you would otherwise discover as a boot
timeout on a rented GPU.

Point the capability at it:

```json
"pods": {"video": {"image": "ghcr.io/<you>/coderai-video:latest"}}
```

### Sharing one pod between capabilities

A pod is a whole GPU, and one coderai can serve several capabilities from it.
Name a shared `pool` and they use the same pods and the same budget:

```json
"pods": {
  "images": {"pool": "media", "image": "ghcr.io/<you>/coderai-media:latest",
             "max_pods": 2, "max_hourly_usd": 0.6},
  "video":  {"pool": "media"},
  "tts":    {"pool": "media"}
}
```

Configure the pool once, on whichever capability carries the settings; the others
just reference it. Without `pool`, each capability rents its own card — three
capabilities, three GPUs, to do what one can.

**Private registry:** RunPod stores the credentials for you — *Settings →
Container Registry Credentials* — and gives each set an id. Put that id in
`runpod.registry_auth_id` (account-wide) or in a pod block's `registry_auth_id`
(per model or capability). Without it a private image fails to pull and the pod
never becomes healthy. Public images need none.

If a pod cannot be provisioned (RunPod disabled, budget cap hit, no GPU under
the ceiling) the request fails with 502. It does **not** quietly fall back to
running the model locally — that failure mode is only discovered by noticing
your own VRAM disappear.

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

## Per-model placement: local, remote, or burst

Three answers to "where does this model run", set per model.

**Always remote** — `backend: "runpod"` on the model entry, with a `runpod` block
choosing pods or serverless. No local VRAM, nothing downloaded here.

**Local only** — the default. Requests queue behind the running generation, up to
`max_instances` concurrently and `queue_max_waiting` in the queue.

**Local first, burst on overlap** — local serves what it can; overflow goes to
RunPod:

```json
{
  "path": "/AI/guffcache/gemma-4-31B-Q4_0.gguf",
  "max_instances": 1,
  "runpod_spillover": {
    "enabled": true,
    "on_busy": true,
    "target": {"mode": "pods", "engine": "llamacpp",
               "hf_gguf": "bartowski/gemma-4-31B-GGUF:Q4_0",
               "max_hourly_usd": 0.8, "idle_timeout_s": 180}
  }
}
```

Triggers, most eager first:

| Trigger | Fires when |
|---|---|
| `on_busy` | no free local slot right now — the second concurrent request offloads instead of queueing |
| `on_concurrency_full` | the wait queue itself is full (would otherwise be a 503) |
| `on_no_gpu` | no local GPU can serve the model at all |

`target.mode` is `pods` (a coderai-managed pod: provisioned on demand, health-
gated, scaled, reaped, billed against the same caps) or `serverless` (your own
RunPod endpoint id). `served_model` renames the model for the remote.

### Where a pod gets the weights

A pod cannot see this machine's disk, so a local path means nothing there. One of
these has to resolve (`source` on the model's runpod block):

| `source` | The pod gets the model by |
|---|---|
| `auto` (default) | an explicit `hf_repo`, a path that already IS a repo id, or a repo id **recovered from the HuggingFace cache path** (`models--Owner--Repo`), then `model_url` |
| `hf` | downloading `hf_repo` from HuggingFace |
| `url` | downloading `model_url` directly — how most one-off GGUFs are had |
| `upload` | receiving the weights from here |

`auto` covers most of a normal catalogue without anyone typing anything: weights
downloaded through coderai live in the HuggingFace cache, whose directory names
still say which repo they came from.

Not every server can take every source, and the mismatch is refused at
configuration time rather than after a pod has booted for five minutes:

* **vLLM** is launched `--model <id>` and can only take a HuggingFace repo id.
* **llama.cpp** takes `-hf user/repo:QUANT` or downloads a URL with `-mu`.
* **Upload works only on a coderai pod.** vLLM and llama.cpp need the file to
  exist before their container starts; a coderai pod loads models on demand, so
  it can be given them afterwards.

Uploading is a deliberate choice, not a fallback: it is the cold cost of every
NEW pod, since pod storage is disposable. On a fast symmetric link that may be
perfectly reasonable — a 30 GB model is a few minutes at gigabit — but a
HuggingFace repo id or a URL costs nothing here and pulls at datacenter speed. A
RunPod network volume is the better answer for weights used repeatedly.

### Keeping a pod warm

`keep_warm` on a model's runpod block (or a capability's) keeps one pod running
at all times, so no request waits for a cold boot — image pull plus weight
download, which is minutes. **Off by default, deliberately**: a warm pod bills
every hour of every day, including the ones nobody uses it. With it off, a pool
scales to zero and an idle model costs nothing.

Pods configured warm are started at startup rather than on first use, so the
first request after a restart does not pay the boot that `keep_warm` exists to
avoid.

### Several models on one pod

Two models that would each rent a card can share one, until it saturates:

```json
{"runpod": {"pool": "shared", "engine": "coderai",
            "image": "ghcr.io/<you>/coderai-images:0.2.0",
            "max_pods": 2, "scale_up_inflight_per_pod": 4}}
```

Give the second model the same `pool` name. Both requests land on the same pod —
each carries its own model name, and a coderai pod picks the model per request —
and the pool rents a second pod only once the first is carrying
`scale_up_inflight_per_pod` requests. The same `pool` field works on a burst
target, so two models' overflow shares one card.

**This needs a multi-model pod image.** A vLLM pod is launched `--model X` and a
llama.cpp pod `-hf one.gguf`: they serve exactly one model, so sharing is refused
with an explicit error rather than sending requests to a pod that would answer
with the wrong weights. Use `engine: coderai` (or a custom multi-model image) for
a shared pool.

The first model to create a shared pool sets its shape — image, GPU class,
budget, scaling — and later joiners do not overwrite it, so whichever model
happens to load last can't silently redefine the budget everyone is sharing.

The burst is a borrow: a pod is taken for one request and handed straight back,
so an overlap that never recurs is reaped after `idle_timeout_s` rather than
lingering. And a burst that cannot be opened — cold pod failed, budget cap hit,
misconfigured — falls back to the local queue rather than erroring: local is
always the safety net, never the other way round.

## Deployment shapes this enables

* **One small coderai, many remote GPUs** — the local instance holds the API,
  auth, catalogue and UI; each capability points at a box or pod that owns one
  model family.
* **Rent per capability, pay only while used** — capabilities with a pod block
  scale from zero on demand and are reaped when idle, under the same budgets as
  a RunPod-served LLM.
* **Per-engine pod images** — vLLM's and llama.cpp's own images serve LLMs with
  no coderai on the far side; the coderai image is only needed where a coderai
  subsystem runs (the mux service, or a gateway target).
* **Burst to RunPod** — pin a model to the `runpod` backend and it provisions,
  scales and reaps pods against the configured budget (see `docs/runpod.md`).
