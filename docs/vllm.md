# vLLM backend

> **Status: IMPLEMENTED (v0.1.86).** vLLM is a first-class managed engine backend,
> wired exactly like ds4/ktransformers: a subprocess in an ISOLATED venv, proxied over
> its OpenAI HTTP API, selected per-model via a `backend: "vllm"` pin or the `vllm.model_id`
> alias (never auto-claimed). Config: `codai/config.py::VllmConfig`; worker:
> `codai/api/vllm_worker.py`; proxy: `codai/backends/vllm.py`; manager wiring +
> front-proxy capability (`vllm` on GPU nodes) + admin card. It also serves **Surya2**
> for the OCR subsystem (`ocr.surya_serve = "vllm"`).
>
> Engine layering: nvidia/cuda/vulkan/opencl are **in-process base engines**; vLLM (like
> ds4/colibri/k3/kt) is a **managed external engine** — it must be out-of-process because
> it pins its own torch/CUDA (torch 2.13 / cu13) that conflicts with the main venv.
>
> **Use it like nvidia/radeon — from the model list.** Tag a model entry with
> `backend: vllm` and vLLM serves *that* model using its own `path`, each under its own
> name; multiple models can be tagged (they swap on the GPU via normal VRAM eviction). The
> `vllm.model_id`/`vllm.model_path` config fields are OPTIONAL — only for serving one model
> with no list entry (ds4/kt style). Engine-level config (`venv`, `gpu`, `ctx`,
> `gpu_memory_utilization`, `tensor_parallel_size`, `dtype`, `quantization`, `auto_build`)
> applies to whatever model it serves. `gpu` → `CUDA_VISIBLE_DEVICES` pins the card(s);
> CUDA-only.

## Why add vLLM

coderai's current inference backends serve **one request at a time per loaded
instance** — the Vulkan/CUDA llama.cpp path holds a per-instance generation lock,
and the big-MoE engines (ds4/colibri/k3/kt) are single-stream serve loops. To run
many requests concurrently today you must load **several instances** of a model and
let the manager fan requests across them. That works but wastes VRAM (N full copies)
and gives sub-linear throughput.

[vLLM](https://github.com/vllm-project/vllm) provides **continuous batching + paged
KV cache**: a single loaded model interleaves many in-flight requests, so aggregate
tokens/sec on one GPU (e.g. an RTX 3090, 24 GB) far exceeds N serialized llama.cpp
instances for the same VRAM. It exposes an **OpenAI-compatible HTTP server**
(`vllm serve` / `python -m vllm.entrypoints.openai.api_server`), so it slots into
coderai exactly like the existing **ds4 / ktransformers** HTTP-proxy pattern.

This is the right long-term answer for high-concurrency workloads — bulk OCR of a
large corpus, batch classification/extraction, high-QPS chat.

## How it would integrate (mirror the ds4/kt pattern)

- `codai/config.py` — `VllmConfig` dataclass (enabled, install_dir/model_path,
  host, port, ctx/`max_model_len`, `gpu_memory_utilization`, `tensor_parallel_size`,
  `max_num_seqs`, dtype/quantization, `served_model_name`/`model_id`, extra_args,
  extra_env, auto_build) + parent Config field + from_dict/to_dict.
- `codai/api/vllm_worker.py` — manage
  `python -m vllm.entrypoints.openai.api_server --host <h> --port <p>
  --model <path> --served-model-name <id> --max-model-len <ctx>
  --gpu-memory-utilization <f> [extra]` as a subprocess; health-gate on
  `/v1/models`; auto-pick a free port. Mirror `codai/api/kt_worker.py`.
- `codai/backends/vllm.py` — OpenAI HTTP proxy for `/v1/chat/completions`
  (+ stream), near-copy of `codai/backends/ds4.py` / `codai/backends/ktransformers.py`.
- Manager wiring — register `"vllm"` in `_ENGINE_BACKENDS`, add
  `get_active_vllm_config`, a `_vllm_name_claims` that returns **False** (pin/alias
  only, never auto-claim — same as `kt`, so it never steals models from other
  engines), a load branch, text-accept, and `/v1/models` surfacing.
- Front-proxy — `required_capability` params, `registry._DEFAULT_CAPS` (GPU nodes),
  and thread the config through assignment/app/engine_supervisor.
- Admin — `routes.py` config get/set, `settings.html` card, `models.html` backend
  dropdown option `vllm`.
- Packaging/docs — `auto_build` gated `pip install vllm` (heavy, CUDA-specific);
  document that vLLM needs a matching CUDA/torch build.

## Notes / caveats

- vLLM is **CUDA-first** (this box's 3090s are fine; AMD/Vulkan is not the target).
- Vision models: vLLM supports **Qwen2.5-VL** and other VLMs with continuous
  batching — so a vLLM backend would *also* be the high-throughput path if we ever
  serve a VLM for image tasks. (For OCR proper we integrate a dedicated OCR engine,
  not a VLM — see the OCR docs.)
- Selection stays pin/alias-only to avoid collisions with ds4/colibri/k3/kt, exactly
  like ktransformers.
