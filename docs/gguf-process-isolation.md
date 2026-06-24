# GGUF process isolation: keeping llama.cpp out of the torch CUDA context

> **Status (implemented, default on for auto-detected NVIDIA).** Controlled by
> `ServerConfig.isolate_gguf_engine` (default `True`). Implemented in
> `codai/frontproxy/engine_supervisor.py` `_build_engines()`. Related:
> `docs/process-isolation-plans.md`, `docs/frontend-engine-split.md`.

## Problem

When a GGUF model (llama.cpp / llama-cpp-python, CUDA build) and a torch/diffusers
model run in the **same process on the same NVIDIA GPU**, llama.cpp's CUDA backend
corrupts the CUDA context for PyTorch. The symptom: after a GGUF model has run,
the next torch kernel dies — observed as:

```
diffusers/pipelines/z_image/pipeline_z_image.py … encode_prompt → self.text_encoder(...)
transformers/models/qwen3/modeling_qwen3.py:414 → create_causal_mask → sdpa_mask
  → _ignore_causal_mask_sdpa → padding_mask.all()
torch.AcceleratorError: CUDA error: invalid argument
```

Real trace from a township match: Z-Image-Turbo keyframes succeeded for ~6 minutes,
then a `gemma-4-26B …-mmproj` **GGUF** loaded on the nvidia engine (`clip_ctx: CLIP
using CUDA0 backend`) to serve a `/v1/chat/completions`, and the **next** Z-Image
generation crashed on the first torch CUDA op. Every successful Z-Image run was
*before* llama.cpp touched CUDA0; the first one *after* crashed. The "asynchronously
reported" note confirms the failing op isn't really `padding_mask.all()` — that's
just the first torch kernel to hit the already-corrupted context.

### Why "evict + swap" in one process is not enough

Fully unloading the torch model before loading the GGUF model (and vice versa) does
**not** fix this: the corruption is to the **process's CUDA context**, which torch
holds for the process lifetime and cannot be torn down/recreated cleanly while the
process lives. Once llama.cpp has run in a process, every later torch kernel in that
process is at risk. The only robust fix is to never run llama.cpp and torch CUDA in
the same process.

## Design — co-located GGUF engine

coderai already runs a thin **front** proxy that supervises **engine** subprocesses
(one per GPU), routing requests by capability. We reuse that: on an NVIDIA GPU we run
**two** engine processes pinned to the **same card**:

- the **torch engine** (`nvidia`) — capabilities `{transformers, whisper, ds4}`
  (**`gguf` dropped**) — serves HF/diffusers (incl. Z-Image and its Qwen3 text
  encoder, which is torch, not llama.cpp).
- a sibling **gguf engine** (`<name>-gguf`, `backend=nvidia`, capabilities `{gguf}`)
  — serves llama.cpp GGUF models.

Same `CUDA_VISIBLE_DEVICES` / `CODERAI_ENGINE_GPUS`, **different process → different
CUDA context**. The torch engine never runs llama.cpp; the gguf engine never runs a
torch CUDA kernel (it's only ever assigned GGUF work), so neither can poison the
other — even though they share one physical card.

The gguf engine uses `backend="nvidia"` (not `"vulkan"`) so a GGUF load takes the
**proven CUDA-llama path**: the manager sets `original_backend="nvidia"` →
`VulkanBackend(original_backend="nvidia")` forces CUDA (`manager.py:352-355`). Its
torch CUDA context is created at import but never runs a kernel, so it can't be
corrupted.

### What we get for free (reused, not rebuilt)

- **Routing / assignment** — capability-based (`router.required_capability` →
  `gguf`/`transformers`; `assignment.compute_assignment` / `pick_engine` filter by
  `engine.capabilities`). GGUF → gguf engine; diffusers/HF → torch engine.
- **Front↔engine HTTP proxy** — streaming, auth, keepalive, token counting.
- **VRAM/eviction** — per-process, measured from *actual* free VRAM, so the two
  engines on one card see each other's allocations and self-bound.
- **Thermal** — both engines carry the same GPU selector, so the front's thermal
  monitor pauses both when the card is hot, with the existing cooperative-pause →
  SIGSTOP escalation (`_thermal_signal` → `os.killpg(os.getpgid(pid), SIG…)`; engines
  run via `setsid`). Crucially, SIGSTOPping the gguf engine **does not freeze** the
  torch engine — separate process groups give finer thermal control than the old
  single-process layout.

## Thermal suspend of the GGUF subprocess

No new mechanism: the gguf engine is a first-class engine, so the front already:
1. asks it to pause **cooperatively** (`POST /internal/thermal-pause`); the engine's
   streaming loop checks the pause flag between tokens/requests, and
2. **escalates to `SIGSTOP`** on its process group if it stays busy for
   `thermal.stop_escalate_checks` checks (a llama.cpp generation is a blocking C call
   that won't observe the cooperative flag mid-run; SIGSTOP freezes it regardless),
   then `SIGCONT` once cooled. Restart/shutdown wake a frozen engine first
   (`_thermal_resume_if_frozen`).

## Known trade-offs

- **Two processes per NVIDIA card.** Extra host RAM (the gguf engine imports the full
  stack) and a second, idle torch CUDA context (~0.3–0.6 GB VRAM). Acceptable for
  correctness; could be trimmed later by lazy-importing torch in gguf-only engines.
- **No cross-engine eviction.** Each engine evicts only its own models, so on a single
  shared card the gguf engine can't evict a resident diffusers model to reclaim VRAM
  (and vice versa). llama.cpp's auto CPU-offload (`n_gpu_layers`) and the global RAM
  cap keep this from hard-OOMing; tighter cross-engine VRAM coordination is a
  follow-up.
- **`engine_specs` users opt out.** When `engine_specs` is set the auto-split is
  skipped — declare the torch/gguf split yourself (two specs on the same card: one
  `backend: nvidia` with `capabilities` minus `gguf`, one `backend: nvidia` /
  `capabilities: ["gguf"]`).

## Validation

- `.gguf` request → routed to a `{gguf}`-capable engine; HF/diffusers request →
  the `{transformers,…}` torch engine (verified via `required_capability` +
  `Engine` capability defaults).
- A township match that interleaves `/v1/chat/completions` (gemma GGUF) and
  `/v1/images/generations` (Z-Image) no longer crashes the image step, because the
  GGUF model runs in a different process from the diffusers pipeline.
- Disable with `"server": { "isolate_gguf_engine": false }` to restore the old
  single-engine behaviour (and the crash).
