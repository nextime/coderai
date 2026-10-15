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

## Tool calling

vLLM parses a model's tool calls with a named parser, and **refuses any request
carrying `tools` unless it was launched with one**:

```
400 "auto" tool choice requires --enable-auto-tool-choice and
    --tool-call-parser to be set
```

That is a request-time error from a server that started cleanly, for a
capability the catalogue says the model has, naming flags the caller cannot
set. So coderai sets them.

A tool call is TEXT the model emits in its own trained format, so the parser
belongs to the model FAMILY rather than to the deployment. `tool_call_parser`
defaults to `auto`, which picks from the model name:

| family | parser |
|---|---|
| Qwen 2.5 / 3, Hermes finetunes | `hermes` |
| Qwen3-Coder | `qwen3_coder` |
| Llama 3.x | `llama3_json` |
| Mistral / Ministral | `mistral` |
| DeepSeek-V3 | `deepseek_v3` |
| GLM-4.5 / 4.6 | `glm45` |
| Granite, InternLM, Jamba | `granite`, `internlm`, `jamba` |

Longest match wins, so `qwen3-coder` is not swallowed by `qwen`.

Set it explicitly — globally, or per model in its `vllm` block — for anything
not on that list:

```json
{"id": "my-model", "backend": "vllm",
 "vllm": {"tool_call_parser": "llama3_json"}}
```

`off` disables tool calling for that model. Use it rather than letting a wrong
guess stand: a mismatched parser does not fail, it parses tool calls into
nonsense, and silently producing wrong tool ARGUMENTS is worse than refusing
the request.

Both flags are passed together or not at all — vLLM refuses to **start** with
`--enable-auto-tool-choice` and no parser, and an engine that will not boot is
worse than one that cannot do tool calls. When the family is unrecognised the
log says what to set, because otherwise the symptom lands on the client.

The `vllm` block travels to a RunPod pod with the model, so a pod serves it the
same way this install does.
