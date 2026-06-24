# dtype auto-selection: model-native default, forceable override

> **Status (planned — not yet implemented).** Today (0.1.9) unset `precision`
> resolves to a static **fp16 on CUDA / fp32 on CPU** fallback
> (`codai/backends/cuda.py:770-774`, `codai/models/hf_loading.py` `resolve_dtype`).
> This document specifies making the *default* model-native (read from the
> checkpoint) while keeping `precision` as an explicit force. Companion notes:
> `docs/frontend-engine-split.md`, memory `project_model_config_respect`.

## Problem

The HF nvidia engine (and the shared `hf_loading.py` path) pick a compute dtype
from a blanket fallback when the model's `precision` is unset. fp16 is wrong for
most modern checkpoints:

- Qwen3.x, Llama-3.x, Gemma, Mistral, etc. are trained and shipped in **bf16**.
  Loading them in fp16 risks activation/logit **overflow** (Qwen is notably
  prone to this) and silently diverges from the reference numerics.
- It also produced the observed warning: with `precision: null` the model loaded
  as **float32**, and Flash-Attention-2 (which only supports fp16/bf16) emitted
  *"Flash Attention 2 only supports torch.float16 and torch.bfloat16 dtypes, but
  the current dtype in Qwen3_5ForCausalLM is torch.float32"*.

The authoritative signal already exists and we ignore it: every HF checkpoint's
`config.json` carries **`torch_dtype`** (newer transformers: `dtype`) — exactly
the dtype the model was trained/saved in.

## Design

### Resolution order (highest priority wins)

1. **`precision` in `models.json`** — explicit force. Accepted values:
   `fp16`/`float16`, `bf16`/`bfloat16`, `fp32`/`float32`. The literal value
   `auto` (or unset / empty) means "fall through to the next step", **not**
   fp16.
2. **Checkpoint `config.json` `torch_dtype`** (fallback to `dtype` key for
   newer transformers schemas) — the model-native dtype. This is the
   "automatic based on the model" behaviour. Qwen/Llama/Gemma → bf16; genuinely
   fp16-native checkpoints → fp16.
3. **Final fallback** when the checkpoint has no usable dtype (missing,
   `"auto"`, or `float32` while loading on GPU):
   - **bf16 on CUDA** — every Ampere+ GPU in this fleet supports it, and it is
     strictly safer than fp16 (wider exponent, no overflow).
   - **fp32 on CPU** — no bf16/fp16 benefit without a GPU; keep full precision.

### FA2 guard

Flash-Attention-2 accepts only fp16/bf16. If the resolved dtype is **fp32** and
FA2 is requested (per-model or global flash flag), **promote to bf16** rather
than silently downgrading attention to sdpa. Rationale: the user asked for FA2;
honour it with the nearest valid dtype instead of quietly changing the attention
backend (which changes numerics/perf without telling anyone). Only fall back to
sdpa when FA2 is genuinely unavailable (kernel not built / not installed).

## Implementation sketch

- **`codai/models/hf_loading.py` — `resolve_dtype`:** extend the signature to
  accept the checkpoint source so step 2 can read it, e.g.
  `resolve_dtype(cfg, default=None, model_path=None)`. When `cfg['precision']`
  is unset/`auto`, read `<model_path>/config.json` → `torch_dtype`/`dtype`. If
  still unresolved, apply the CUDA→bf16 / CPU→fp32 fallback. Map dtype strings
  to `torch.*` via the existing alias table. Keep it a pure function (no model
  load) — it only reads the small `config.json`.
  - Cache the parsed `config.json` dtype per model dir to avoid re-reading on
    every load (optional; the file is tiny).
  - For GGUF/llama.cpp this is a no-op — quant type is baked into the file;
    `precision` does not apply there.
- **`codai/backends/cuda.py`:** the load path already has the model dir; pass it
  as `model_path` to `resolve_dtype` (currently calls it with only
  `{'precision': kwargs.get('precision')}` at ~line 769). Apply the FA2 guard
  where `attn_implementation` is chosen (~line 977-984): if `_cfg_dtype is
  torch.float32` and `self.use_flash_attn and self.flash_attn_available`, set
  `_cfg_dtype = torch.bfloat16` before `load_kwargs['dtype']` is set, and log
  the promotion.
- **Both call sites in `manager.py`** already forward `precision` via
  `kwargs['precision'] = config.get('precision')`; no change needed there beyond
  ensuring the model path is available to the backend (it already is).
- **Parity:** apply the same `resolve_dtype(..., model_path=...)` upgrade to the
  spatial/embedding/audio/vision builders in `hf_loading.py` so every HF path
  benefits, not just text.

## Config surface (no code change needed by operators)

- Force a specific dtype:
  ```json
  { "name": "Qwen/Qwen3.5-9B", "precision": "bf16" }
  ```
- Let the engine decide from the checkpoint: omit `precision` or set
  `"precision": "auto"`. With this design, `Qwen/Qwen3.5-9B` (config
  `torch_dtype: bfloat16`) auto-loads as **bf16** with no edit.

## Validation

- `Qwen/Qwen3.5-9B` with `precision` unset → loads bf16, **no FA2 dtype
  warning**.
- A genuinely fp16-native checkpoint (`config.json torch_dtype: float16`) with
  `precision` unset → loads fp16.
- `precision: "fp32"` + FA2 enabled → dtype promoted to bf16, promotion logged;
  FA2 stays active.
- CPU-only load with `precision` unset and no GPU → fp32.
- GGUF model → `precision` ignored, no regression.

## Related, deliberately out of scope here

- **Per-model `flash_attention: false` is currently overridden by the global
  `offload.flash_attention` flag** (`self.use_flash_attn` derives from the
  global). If per-model opt-out should win, that precedence fix is separate from
  dtype selection and tracked on its own.
- Switching the *global* unset-default to bf16 is subsumed by step 3 above and
  needs no separate flag once this lands.
