# CoderAI documentation

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
| [runpod.md](runpod.md) | Renting remote GPUs: pods, serverless, budgets, the reaper, spillover |
| [remote-execution.md](remote-execution.md) | Serving any model from another machine: worker `service_url`, the mux service, remote text models, the capability gateway |

## Subsystems

| Doc | Subject |
|---|---|
| [ocr.md](ocr.md) | The `/v1/ocr` document-transcription subsystem |
| [expressive-tts.md](expressive-tts.md) | Expressive text-to-speech |
| [zimage-lora-training.md](zimage-lora-training.md) | LoRA training for image models |
| [dtype-auto-selection.md](dtype-auto-selection.md) | How model dtype is chosen automatically |

## Deployment

| Doc | Subject |
|---|---|
| [reverse-proxy-nginx.md](reverse-proxy-nginx.md) | nginx at root, on a subdomain, or under a sub-path |
