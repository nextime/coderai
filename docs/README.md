# CoderAI documentation

**CoderAI — complete orchestration, distribution and escalation of remotizable advanced inference.**

Subsystem guides. The [top-level README](../README.md) is the overview and API
reference; these are the deep dives.

## Serving engines

| Doc | Subject |
|---|---|
| [frontend-engine-split.md](frontend-engine-split.md) | The torch-free front proxy, engine subprocesses, routing and assignment |
| [cluster.md](cluster.md) | Several machines as one: nodes as engines, mDNS + a shared token, one GGUF over llama.cpp RPC, vLLM/SGLang multi-node, prefix-cache routing, pools, Prometheus |
| [gguf-process-isolation.md](gguf-process-isolation.md) | Running GGUF models in isolated processes |
| [process-isolation-plans.md](process-isolation-plans.md) | Design notes on isolation strategies |
| [vllm.md](vllm.md) | vLLM as a first-class engine node (isolated venv, CUDA-only) |
| [llama-server.md](llama-server.md) | `llama-server` as a backend: the alternative front end for a GGUF — speculative decoding and real slots, which the in-process bindings cannot reach |
| [deepseek-ds4.md](deepseek-ds4.md) | DeepSeek-V4 via antirez's ds4 / DwarfStar |
| [glm-colibri.md](glm-colibri.md) | GLM-5.2 (and DeepSeek-V4 / Kimi-K3) via the colibri C engine |
| [kimi-k3.md](kimi-k3.md) | Kimi-K3 on CPU via kimi-k3-in-c |
| [ktransformers.md](ktransformers.md) | ktransformers / SGLang CPU+GPU heterogeneous MoE |
| [runpod.md](runpod.md) | Renting remote GPUs: pods, serverless, budgets, the reaper, spillover. §11 records the **not-yet-built** plan for other providers (Scaleway, Exoscale, OVHcloud, Hyperstack) |
| [remote-execution.md](remote-execution.md) | Serving any model from another machine: worker `service_url`, the mux service, remote text models, the capability gateway |

## Subsystems

| Doc | Subject |
|---|---|
| [video-compose.md](video-compose.md) | `POST /v1/video/compose`: scenes → a finished, uploadable reel (narration timing, Ken Burns, ducked music, burned captions, thumbnail) and `/v1/files/upload` |
| [ocr.md](ocr.md) | The `/v1/ocr` document-transcription subsystem |
| [chunking.md](chunking.md) | `POST /v1/chunk`: splitting a document for a retrieval index, with the structure decided per request by a template |
| [moderation.md](moderation.md) | `POST /v1/moderation`: screening generated images and video before release (explicitness, faces, age, performer match) — and why it fails closed |
| [expressive-tts.md](expressive-tts.md) | Expressive text-to-speech |
| [isolated-worker-environments.md](isolated-worker-environments.md) | Engines that need an interpreter of their own: the inherited `PYTHONHOME` that made three different failures, when a venv that is there cannot be used, and `env_ready` before the first request |
| [image-prompts.md](image-prompts.md) | Image prompts: the `(tag:1.3)` / `[tag]` / `BREAK` grammar, prompts past CLIP's 77 tokens, `clip_skip`, negative prompts, textual-inversion embeddings, size validation |
| [zimage-lora-training.md](zimage-lora-training.md) | LoRA training for image models |
| [text-to-music.md](text-to-music.md) | `POST /v1/audio/generate` with ACE-Step: style prompt + sung lyrics → a 48 kHz stereo track, the python-3.12 worker it needs, and the two image variants (`--with-acestep`) |
| [dtype-auto-selection.md](dtype-auto-selection.md) | How model dtype is chosen automatically |
| [vision-chat.md](vision-chat.md) | Images in a chat request: loading the vision tower the checkpoint's config asks for, the processor that makes a picture promptable, and why a model that cannot see refuses instead of answering blind |
| [memory-accounting.md](memory-accounting.md) | Where `measured_vram_gb` comes from: walking the loaded model (quantization needs no factor), costing the KV cache from the real GQA/sliding-window geometry, refusing a measurement from a failed load, the PSS-based host-RAM cap, idle unload, and `GET /v1/models/history` |
| [longcat-video.md](longcat-video.md) | LongCat-Video: continuation-native text/image-to-video, long video, audio-driven avatars |
| [minimax-h3.md](minimax-h3.md) | MiniMax-H3: video **and** native audio denoised in one packed sequence |
| [serving-a-model-this-box-cannot-hold.md](serving-a-model-this-box-cannot-hold.md) | What has to be true before a model that does not fit will serve — and why "it OOM'd" was the wrong diagnosis |
| [tools-as-services.md](tools-as-services.md) | Running the bundled tools detached, each in its own container, reaching coderai only over HTTP |
| [model-defaults-and-sources.md](model-defaults-and-sources.md) | A model's own `defaults` (steps/guidance/sampler/modes/licence), the sampler names and the flow-matching boundary, and where weights come from: repo, single file, `org/repo:file` or `civitai:<versionId>` |

## Deployment

| Doc | Subject |
|---|---|
| [runpod-new-model-images-plan.md](runpod-new-model-images-plan.md) | **Plan, not built:** pod images for the recently added models — a light-core `coderai-acestep` profile, and what `coderai-video` is missing for an LTX-2.3 GGUF |
| [orchestrator-runpod-guide.md](orchestrator-runpod-guide.md) | Install CoderAI as a **GPU-less RunPod orchestrator**: register models, scale pods, watch the cost. The operator's path for a host with no GPU |
| [install-from-packages.md](install-from-packages.md) | `docker pull ghcr.io/nextime/coderai` — the full image and the capability images, Linux, Windows, offline |
| [reverse-proxy-nginx.md](reverse-proxy-nginx.md) | nginx at root, on a subdomain, or under a sub-path |
| [admin-forward-auth.md](admin-forward-auth.md) | Trusted forward-auth: letting a reverse proxy that already authenticated a user hand that identity to the admin GUI |

## Measurements, field reports and shipped designs

| Doc | Subject |
|---|---|
| [ocr-client-throughput-guide.md](ocr-client-throughput-guide.md) | How to call `/v1/ocr/batch` for maximum throughput, measured through the real HTTP path on A40-class pods |
| [ocr-throughput-measurements-2026-10-18.md](ocr-throughput-measurements-2026-10-18.md) | The numbers behind that guide: throughput and cost on rented GPUs, always through the full orchestrator path |
| [pod-admission-control-plan.md](pod-admission-control-plan.md) | **Implemented:** a KV-aware admission gate instead of a guessed batch width |
| [runpod-regional-storage-plan.md](runpod-regional-storage-plan.md) | **Built in v0.3.55:** per-region network volumes, so multi-region renting and a slim pod image coexist |
| [digesta-field-report-2026-10-15.md](digesta-field-report-2026-10-15.md) | What Digesta hit in production on 15 Oct 2026: a surya lease returning empty pages as `200`, `scale` 404ing for every model, a duplicated `bge-m3`, chunk divergence; plus a measured three-engine comparison |

## Plans not yet built

| Doc | Subject |
|---|---|
| [vlm-lora-training-plan.md](vlm-lora-training-plan.md) | Training LoRA/QLoRA adapters for vision models from images and video — coderai serves them and cannot train them |
| [unlimited-ocr-plan.md](unlimited-ocr-plan.md) | `baidu/Unlimited-OCR` as a coderai OCR engine |
| [surya-block-router-plan.md](surya-block-router-plan.md) | **Deferred:** a density router between surya's full-page and block modes |
| [reference-projects.md](reference-projects.md) | Outside projects studied: what each does that we do not, and what was taken from them |

## Where else to look

- [`../README.md`](../README.md) — the overview, install, configuration and a condensed API reference.
- [`../CODERAI_API_DOCUMENTATION.md`](../CODERAI_API_DOCUMENTATION.md) — the full HTTP
  surface, with an [endpoint index](../CODERAI_API_DOCUMENTATION.md#endpoint-index)
  generated from the routes themselves (`tools/gen_endpoint_index.py`).
- `/docs` and `/redoc` on a running server — the live OpenAPI schema.
