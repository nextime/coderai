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
| Any model, on a machine you already have (always-on or started by command) | `host` backend | `"backend": "host"` on the model entry |

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
not an image name you have to know. `rerank` rides the `embeddings` image and
`stems`/`audio_clean`/`audio_gen` the `audio` one; `speaker` has an image of its
own, because the STT image shipped nothing for diarization or voiceprints.
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
boot) plus one file per capability. There are fifteen images, and they come in
two kinds:

| image | serves | core |
|---|---|---|
| `images`, `video`, `tts`, `stt`, `text`, `voice`, `audio`, `embeddings`, `ocr`, `faceswap` | one capability, in the main venv | full (torch + CUDA) |
| `speaker`, `tts-xtts`, `stt-nemo`, `stt-crisper`, `ocr-paddle` | a stack that **cannot** share the main venv | light (no torch) |

The routers import lazily, so a pod only needs the libraries its own endpoints
touch. The build fails if a profile is missing something the app imports, and
the script then boots the container and waits for `/healthz` — because a pod
that builds but never answers is a failure you would otherwise discover as a
boot timeout on a rented GPU.

#### Images for stacks that cannot share a venv

Some libraries genuinely cannot live beside torch 2.11 and transformers 5.x:
pyannote calls a torchaudio attribute removed in 2.11; coqui-tts imports a
transformers 4.x symbol; CrisperWhisper pins transformers 4.46 **and** torch 2.5;
NeMo wants its own torch; paddlepaddle brings its own CUDA runtime. Each was
measured, not assumed — three others that the local install keeps in isolated
venvs (Surya, speechbrain, pytorch_lightning) turned out to import cleanly and
went into the ordinary images instead.

The conflicting ones each get an image carrying the stack in a venv of its own,
declared beside the profile:

```
profiles/speaker.txt                  # what the API layer needs (no torch)
profiles/speaker.light                # marker: build on the LIGHT core
profiles/speaker.venv-pyannote.txt    # the venv: its own torch, pyannote, ...
profiles/speaker.venv-pyannote.check  # run at build time INSIDE that venv
```

The `.check` script matters more than it looks. `import pyannote.audio` passes
without matplotlib; instantiating a pipeline does not — and a bare import
check shipped a broken venv four times before this existed. A check imports
what the *real load* reaches (pipeline modules, model classes), without a GPU
or a token, which a build has neither of.

Why a *light* core: the first build of these images sat on the full core, so
each carried 7 GB of CUDA, torch and triton that its main venv never used —
beside the venv that did the work. They came out at 14–16 GB, and **RunPod
refuses images above a size line**: a 10.1 GB image boots, a 14.5 GB one is
"Exited by Runpod" two seconds after rent, on every machine, with no other
reason given (container disk, layer format and image config were each ruled
out). On the light core — coderai's Python dependencies and no GPU stack at
all, 1.2 GB — the same five images are 6.9–8.4 GB. A specialised container
carries one GPU stack, not two.

`speaker` used to point at the `stt` image on the assumption it was "the STT
deps plus a bit". That image shipped nothing for diarization or voiceprints;
a speaker pod booted with no way to embed a voice.

#### What a pod is told at launch

A pod is disposable, so a few decisions made here travel with it as environment
— never baked into an image, never written to its config:

- **`HF_TOKEN`**, for gated models. pyannote loaded from its venv, reached
  `from_pretrained`, and stopped at "the model is gated" until this was sent.
- **Licence acceptances** — Surya's (`CODERAI_OCR_SURYA_ACCEPT_LICENSE`) and
  coqui's CPML (`COQUI_TOS_AGREED`). Configuring the model here *is* the
  decision; a pod cannot make it, and coqui's alternative is an interactive
  prompt on a pipe that corrupts the worker protocol and then blocks forever.
- **Where its baked venvs are** (`CODERAI_PYANNOTE_VENV` etc.), or the worker
  tries to build one on a machine rented by the second.
- **Which OCR engine to enable.** OCR ships disabled, and each engine has a gate
  of its own on top; a pod rented to do OCR arrives with both open.

#### Surya on a pod

Surya 0.22 is a VLM (`surya-ocr-2`), not a self-contained OCR library. It needs
a server: vLLM, which it spawns *in Docker* (impossible inside a container), or
`llama-server` on a GGUF, which the image does not carry. Locally it rides
coderai's own vLLM engine. A pod asked for `surya` serves the request with
docTR — the gateway rewrites the request's `engine` field on the way out,
because the OCR route honours what the request asks for over the pod's
configured default — and logs why. Surya on a pod belongs with the engine
images, where a `llama-server` and the GGUF are the same work as colibri and
ds4.

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

### A machine you already have: the `host` backend

The capability images are ordinary containers. Nothing about them needs RunPod
— a pod is just a machine that pulls one. The same image runs on a box you own,
a server rented by the month, a second GPU down the hall, and coderai uses it
per model, with an endpoint and a token. On the model page it is **Runs on →
A machine of yours** under Placement; in `models.json`:

```json
{ "backend": "host",
  "host": { "url": "http://gpubox:8000", "api_key": "…" } }
```

That is the always-on shape: the container is already running and coderai
just talks to it. If it is down, coderai says so rather than starting nothing.
Add a command and it becomes on-demand:

```json
{ "backend": "host",
  "host": { "url": "http://gpubox:8000", "api_key": "…",
            "start_cmd": "ssh gpubox docker start coderai-images",
            "stop_cmd":  "ssh gpubox docker stop coderai-images",
            "boot_timeout_s": 120, "idle_timeout_s": 600 } }
```

`start_cmd` is whatever starts the thing — `docker run`, `ssh … docker start`,
a systemd unit, a script. coderai runs it, waits for `/healthz` up to
`boot_timeout_s`, uses the host, and runs `stop_cmd` after `idle_timeout_s`
with nothing in flight. It only ever stops a host it started itself; one that
was already up when coderai arrived is never touched. Both commands are
optional — `start_cmd` without `stop_cmd` starts on demand and leaves it running.

Deliberately absent: GPU search, price ranking, budgets, spot, capacity
fallback, the orphan reaper. Those exist because a cloud rents an anonymous
card by the second and can lose it. A host is a named machine you are
responsible for — that machinery would be wrong, not just unnecessary.

Run the container with the same token: `-e CODERAI_API_TOKEN=…`. A capability
image locks itself with it, and a plain `service_url` used to carry no token at
all — a hand-run image behind one answered 401 to everything. `service_token`
on a model entry fixes that for the `service_url` path too.

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

How each server gets there differs, and a combination that cannot work is
refused at configuration time rather than after a pod has booted for five
minutes:

* **llama.cpp** takes `-hf user/repo:QUANT`, or downloads a URL itself (`-mu`).
* **vLLM** is launched `--model <id>` and downloads nothing but a HuggingFace
  repo — *so the pod downloads it first*: with `source: url`, coderai overrides
  the image's entrypoint with a small script that fetches the file (curl, wget or
  python — whichever the image has), extracts it when `model_url_is_tar` is set,
  and then execs vLLM pointed at what it staged. This is the "external bounce"
  route: put the weights on any HTTP host the pod can reach — an object store,
  your own HTTPS server — and a vLLM pod can serve them.
* **Upload works only on a coderai pod.** vLLM and llama.cpp need the file to
  exist before their container starts and have no endpoint to receive one; a
  coderai pod loads models on demand, so it can be given them afterwards.

The same staging carries **adapters given as a URL**: a vLLM pod gets
`--lora-modules name=<staged path>`, a llama.cpp pod `--lora`/`--lora-scaled`.
So a local adapter can reach a vLLM pod after all — put it on a URL rather than
uploading it.

Staging uses RunPod's REST pod API, which is the only one that accepts
`dockerEntrypoint`/`dockerStartCmd`; pods that need no staging are created
through the GraphQL path exactly as before.

Uploading is a deliberate choice, not a fallback: it is the cold cost of every
NEW pod, since pod storage is disposable. On a fast symmetric link that may be
perfectly reasonable — a 30 GB model is a few minutes at gigabit — but a
HuggingFace repo id or a URL costs nothing here and pulls at datacenter speed. A
RunPod network volume is the better answer for weights used repeatedly.

### LoRA adapters

Image and video models have had LoRA support for a long time. Text models now do
too, locally and remotely — `lora_path` (a local path **or** a HuggingFace repo
id), `lora_model_dir`, or a `loras` list, with `lora_scale` for strength. A
**QLoRA** adapter needs nothing special: the quantisation describes how the base
model loads (`load_in_4bit`), and the adapter itself is an ordinary LoRA.

Each runtime applies them its own way, and what reaches a pod differs:

| Runtime | How | On a pod |
|---|---|---|
| transformers (HF) | PEFT `load_adapter`, several at once | via a coderai pod, adapters travel with the seed |
| llama.cpp (GGUF) | one adapter, **GGUF-converted** | needs the file; use a HF repo id or a coderai pod |
| vLLM | `--enable-lora --lora-modules` at launch | **HuggingFace repo ids only** |
| diffusers (image/video) | `load_lora_weights` | uploaded to the remote automatically, content-addressed |

The asymmetry is worth knowing: a **diffusion** LoRA is sent with the request
(hashed, uploaded once, referenced by content id), because that API carries
adapters per request. A **text** LoRA is part of how the model is *loaded*, so it
has to be on the pod before the model loads there.

**A local text adapter still works on RunPod** — it just decides which pod. A
model whose adapter exists only on this machine is routed to a **coderai text
pod** (`ghcr.io/<you>/coderai-text`), because that is the only pod image with an
endpoint that can receive one. The sequence:

1. the pod is created and told the adapter's `sha256:` id — computed here, so
   the two agree before the pod exists;
2. the adapter is sent to the pod's content-addressed blob store before the
   first request, once per pod;
3. the pod resolves `sha256:<hex>` back to a file and applies it with PEFT.

A **published** adapter (a HuggingFace repo id) skips all of that and goes to a
vLLM pod, which fetches it itself — cheaper and faster, so publish adapters you
use often. A PEFT adapter **directory** cannot be sent (the blob store holds
single files); publish that one.

Build and push the text image before using it:

```bash
PROFILES=text ./packaging/runpod/publish_capability_images.sh
```

### A pod is an extension of this system, not a fixed catalogue

A pod starts knowing only what it was seeded with: the one model a per-model pool
rented it for, or the models of its capability. That is not enough — a request
can name any model at any time, and a pod that answers "not available" for the
rest of its life is a dead end rather than an extension of the local system.

So when a remote refuses a model it does not know, coderai **teaches it and
retries once**:

1. the remote answers `Model 'x' is not available`;
2. the gateway resolves that model to something the remote can fetch — a
   HuggingFace repo id, a URL, or an upload when it exists nowhere else — and
   posts it to `/v1/models/register`;
3. the request is replayed, and the remote downloads and serves the model.

Registration is in memory (a pod is disposable) and idempotent, so a model
already loaded is never disturbed. Each model is taught to each pod once.

A model that resolves to none of those — a local path with no `hf_repo`,
`model_url` or `source: upload` — is refused with that reason rather than
registered as something the pod could never load.

### Network volumes (optional)

A pod's own disk dies with the pod, so every cold pod re-downloads its weights.
A RunPod **network volume** — created by hand in the RunPod console, coderai
never creates one — turns that into a one-off:

```json
{"runpod": {"network_volume_id": "abc123", "volume_mount_path": "/workspace"}}
```

Account-wide in Settings, or per pool in its pod block. What changes when one is
attached:

| | Without a volume | With one |
|---|---|---|
| Downloaded weights | re-fetched by every cold pod | fetched once, reused |
| Uploaded models / LoRA adapters | re-uploaded to every new pod | sent once |
| Staged downloads (`source: url`) | container disk | on the volume |

coderai points `HF_HOME`, the diffusers/transformers caches, `CODERAI_MODELS_DIR`
and the staging directory at the mount, which is what actually makes the volume
do anything — attaching one without redirecting the caches changes nothing.

**That also makes the volume your bounce host.** An upload lands on it and every
later pod sees the file, so "upload a QLoRA or a local model once, use it from
any pod" needs no third-party storage and no public IP.

Two constraints RunPod imposes, both handled for you:

* **Secure Cloud only** — `cloud_types` is forced to `SECURE`, because a
  COMMUNITY candidate would just fail the attach.
* **Same data center as the volume** — the volume's region is looked up once and
  pods are pinned there, rather than letting GPU price ranking pick a region
  where the attach fails as an unexplained capacity error.

A volume is attached at deploy time and cannot be added to a running pod. Cost is
about $0.07/GB/month. And RunPod warns that several workers **writing** the same
volume at once can corrupt it — coderai's writes are write-once (content-addressed
adapter blobs, named model files), but keep that in mind before pointing many
pods at one volume for anything else.

#### Dependencies on the volume, small image in the registry

A capability image is ~7 GB and nearly all of it is torch. With a volume those
libraries can live there instead, and the pod boots a **slim** image — OS,
Python and coderai only — that runs from a venv on the volume:

```json
{"runpod": {"network_volume_id": "abc123", "venv_on_volume": true}}
```

The first pod to use a given venv builds it (a few minutes of pip); every pod
after finds it ready and skips both the big pull and the install. The venv is
named after the capability by default, so `embeddings` and `video` pods get their
own; `venv_name` overrides that to share or separate them deliberately.

Build and push the slim image once:

```bash
PUSH=1 ./packaging/runpod/build_capability_image.sh slim ghcr.io/<you>/coderai-slim:latest
```

**The trade-off is real and unmeasured on your workload.** The pull gets much
smaller, but importing torch from network-backed storage is slower than from
local disk, so some of the saving comes back at import time. Whether it is a net
win depends on how often your pods cold-start. It is opt-in per pool, never a
default.

Two failure modes are handled rather than left to chance: the ready marker is
written **after** the install, so a pod that dies mid-build cannot leave a
half-built venv that later pods import from; and a lock directory (mkdir is
atomic) makes a second pod **wait** for the first rather than installing into the
same directory at the same time.

**Container images cannot come from a volume.** A pod pulls its image from a
registry before any volume exists; the volume holds model weights and adapters,
not the image.

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

## Test run

Placement has a lot of moving parts — image, weights, adapters, auth, budgets —
and most of them fail minutes into a pod boot, in a log nobody is watching. Every
model page has a **Test run** button, and the endpoint behind it:

```
POST /v1/models/test  {"model": "…", "where": "auto" | "local" | "runpod"}
```

It sends the smallest real request for the model's kind — a chat completion, a
256×256 image, an embedding, a rerank, a short TTS line — **through the front**,
so it exercises the same routing, auth, pod provisioning and staging a real
request would. A test that simulated the path would prove nothing about it.

`where` forces placement for that one request: `local` ignores any remote
configuration, `runpod` refuses to fall back to local — so a pod that is not
actually reachable fails the test instead of quietly passing here.

The answer says where it ran, how long it took, and a sample of the output:

```json
{"model": "qwen38", "where": "runpod", "target": "its own pod",
 "ran": "/v1/chat/completions", "ok": true, "seconds": 184.2, "sample": "OK"}
```

Some kinds have no cheap synthetic input: a video generation is minutes of GPU
and real money, a face swap needs faces, transcription needs audio. Those report
a **reachability** check instead and say plainly that no generation was run,
rather than implying more than was proven.

Expect a cold RunPod pod to take several minutes the first time — image pull plus
weight download — which is exactly what the test is there to surface before a
user hits it.

Two kinds are no longer reachability-only. `stt` and `ocr` inputs *can* be
synthesised honestly — espeak speaks a known sentence, PIL draws known words —
and the result is checked **against what went in**, so a model returning fluent
nonsense fails just as an empty one does. When the synthesis itself cannot run
(no espeak, no fonts) the probe falls back to the reachability check and says so,
rather than blaming the model for a gap on this side.

### Testing one model without moving the rest

`remotes.endpoints["images"] = "runpod"` is a **production routing switch**: it
moves every request of that kind. Using it to test a single model is how an
embeddings remote, left enabled after a test, quietly sent real traffic to a
rented pod while the local GPU sat idle — noticed only because the card was
quiet.

Use the per-model path instead. `tools/runpod_model_test.py` writes a `runpod`
block on the **single model entry** and restores `models.json` afterwards,
including on Ctrl-C:

```bash
python3 tools/runpod_model_test.py bge-m3
python3 tools/runpod_model_test.py --engine vllm Qwen/Qwen2.5-0.5B-Instruct
# pin the exact build under test; production keeps following :latest
python3 tools/runpod_model_test.py --image ghcr.io/nextime/coderai-stt:0.2.12 wav2vec2-en
```

Its siblings of the same capability do not move — per-model placement beats the
capability map, which is the same mechanism that lets one video model run local
while another runs on a pod.

**Writing the config is not enough to test it.** The front pushes a reload and
an engine takes it only when idle, so a harness that writes and immediately runs
is testing the *previous* configuration. That produced a whole pass of false
"nothing configures this to run remotely" failures, and could equally have
reported a remote pass for a request served locally. `GET /v1/models/test/state`
answers for the process that actually serves the request:

```json
{"pid": 41, "remotes_enabled": true, "capability_endpoints": [],
 "model": "bge-m3", "known": true, "capability": "embeddings",
 "placement": "pod", "backend": "runpod", "has_runpod_block": true}
```

The harness polls it until the placement it wrote is visible, and refuses to
test if it never becomes visible.

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
