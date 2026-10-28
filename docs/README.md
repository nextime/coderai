# CoderAI documentation

**CoderAI — complete orchestration, distribution and escalation of remotizable advanced inference.**

Subsystem guides. The [top-level README](../README.md) is the overview and API
reference; these are the deep dives.

## Serving engines

| Doc | Subject |
|---|---|
| [frontend-engine-split.md](frontend-engine-split.md) | The torch-free front proxy, engine subprocesses, routing and assignment |
| [gguf-process-isolation.md](gguf-process-isolation.md) | Running GGUF models in isolated processes |
| [process-isolation-plans.md](process-isolation-plans.md) | Design notes on isolation strategies |
| [vllm.md](vllm.md) | vLLM as a first-class engine node (isolated venv, CUDA-only) |
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
| [image-prompts.md](image-prompts.md) | Image prompts: the `(tag:1.3)` / `[tag]` / `BREAK` grammar, prompts past CLIP's 77 tokens, `clip_skip`, negative prompts, textual-inversion embeddings, size validation |
| [zimage-lora-training.md](zimage-lora-training.md) | LoRA training for image models |
| [text-to-music.md](text-to-music.md) | `POST /v1/audio/generate` with ACE-Step: style prompt + sung lyrics → a 48 kHz stereo track, the python-3.12 worker it needs, and the two image variants (`--with-acestep`) |
| [dtype-auto-selection.md](dtype-auto-selection.md) | How model dtype is chosen automatically |
| [model-defaults-and-sources.md](model-defaults-and-sources.md) | A model's own `defaults` (steps/guidance/sampler/modes/licence), the sampler names and the flow-matching boundary, and where weights come from: repo, single file, `org/repo:file` or `civitai:<versionId>` |

## Deployment

| Doc | Subject |
|---|---|
| [runpod-new-model-images-plan.md](runpod-new-model-images-plan.md) | **Plan, not built:** pod images for the recently added models — a light-core `coderai-acestep` profile, and what `coderai-video` is missing for an LTX-2.3 GGUF |
| [orchestrator-runpod-guide.md](orchestrator-runpod-guide.md) | Install CoderAI as a **GPU-less RunPod orchestrator**: register models, scale pods, watch the cost. The operator's path for a host with no GPU |
| [install-from-packages.md](install-from-packages.md) | `docker pull ghcr.io/nextime/coderai` — the full image and the nineteen capability images, Linux, Windows, offline |
| [reverse-proxy-nginx.md](reverse-proxy-nginx.md) | nginx at root, on a subdomain, or under a sub-path |

## Field reports

| Doc | Subject |
|---|---|
| [digesta-field-report-2026-10-15.md](digesta-field-report-2026-10-15.md) | What Digesta hit in production on 15 Oct 2026: a surya lease returning empty pages as `200`, `scale` 404ing for every model, a duplicated `bge-m3`, chunk divergence; plus a measured three-engine comparison |
