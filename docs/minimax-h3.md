# MiniMax-H3 (video + native audio)

H3 is a 33B flow-matching DiT that denoises **video and audio in one packed
sequence** — every generation comes back as an mp4 that already has a synchronised
soundtrack, not a dub.

It runs **in the engine, like every other video model**, whenever that engine's
diffusers can do H3 (>= 0.40): the pipeline is registered with the model manager,
so eviction, VRAM accounting, the RAM cap and the GPU swap gate all treat it
normally. Pin it to a CUDA engine (`"engine": "nvidia"` in models.json) — a 33B
bf16 DiT is not going to run on the Vulkan engine.

When the engine's diffusers is older, the same request is served by an isolated
venv worker instead, so the feature works either side of a diffusers upgrade:

```
/v1/video/generations ──► video.py::_generate_h3
                            ├─ in-engine  (diffusers >= 0.40)
                            │    _load_h3_pipeline → ModularPipeline(workflow=…)
                            └─ worker     (older diffusers)
                                 h3_worker → tools/h3_service.py (isolated venv)
```

The choice is automatic (`h3_worker.engine_supports_h3()`), and `"in_process":
true|false` in the model config forces either side. Both paths share the
checkpoint contracts in `tools/h3_common.py` — the frame/canvas arithmetic, the
workflow choice and the audio mux are identical, so the two routes produce the
same thing.

Why H3 needs its own branch at all, in either mode: diffusers ships it as Modular
Diffusers blocks only (`MiniMaxH3Blocks` / `MiniMaxH3ModularPipeline`), with no
`DiffusionPipeline` half — so `_detect_pipeline_class()` /
`PClass.from_pretrained` can't express it.

## Configure a model

Add an entry to `models.json`. Detection is by name (anything containing
`minimax-h3` / `minimax_h3`) or by an explicit backend:

```json
{
  "minimax-h3": {
    "type": "video",
    "backend": "h3",
    "model_path": "MiniMaxAI/MiniMax-H3",
    "offload_strategy": "group",
    "dtype": "bfloat16",
    "used_vram_gb": 22.0
  }
}
```

| key | meaning |
|---|---|
| `offload_strategy` | `group` (default), `leaf` (smaller groups, slower), or `""` for straight-to-GPU. The diffusers CPU-offload strategies have no modular equivalent and are normalised to `group`. |
| `device_map` | per-component device map (`auto`, `balanced`) instead of group offload |
| `used_vram_gb` | reserved before start, and what eviction frees. Default 22. |
| `in_process` | force in-engine (`true`) or the isolated worker (`false`); default: whatever the engine's diffusers supports |
| `h3_venv` | override the isolated venv location (worker mode only) |
| `engine` | pin the owning engine — use your CUDA one |
| `gpu_device` | pins `CUDA_VISIBLE_DEVICES` for the worker |

In worker mode the venv builds itself on first use from `requirements-h3.txt`
(several GB) under `/opt/coderai/h3_venv`, the `/cache` mount, or
`~/.coderai/h3_venv`. In-engine mode needs nothing extra — just diffusers >= 0.40
(which also wants `huggingface_hub >= 1.23`) in the engine's own venv.

## What the request translation does

H3 has hard checkpoint contracts, so `_generate_h3` adapts the request instead of
forwarding it, and reports every adaptation in the response `warnings`:

- **Guidance-distilled** — no negative prompt, no guidance scale. Both are dropped.
- **24 fps, fixed.** `num_frames` is snapped **up** to the next `17n + 5` the video
  VAE can decode, clamped to 5–15 s. `fps` on the request is ignored.
- **Canvas** axes are rounded down to a multiple of 32 (checkpoint short edge 768,
  area cap 768×1344).
- **Three workflows**, each loading its own ~61.7 GB transformer partition:
  `t2va` (prompt), `fl2va` (first and/or last keyframe, `transformer/`), `ref2va`
  (up to 12 references, `transformer_ref/`). The worker picks one per request and
  **reloads** when it changes — loading with no workflow pulls *both* partitions.
- **Identity is native**: `character_profiles` go in as `ref2va` image references
  (max 9), so H3 skips the identity-keyframe bridge other video models need. An
  explicit keyframe wins over references, because fl2va and ref2va are different
  checkpoints and a request cannot be both.
- `v2v` and `interp` are rejected — H3 has no such workflow.

Post-processing (upscale, interpolation, dialogs, lip sync, subtitles) runs on top
of the returned mp4 exactly as for any other video model, so a cloned voice line
still lands over H3's own soundtrack.

## LoRAs

`loras` on the request are forwarded to the worker and loaded with
`MiniMaxH3LoraLoaderMixin`, with `set_adapters` weights. Public recipes for
training one: ~32 face crops at 512px, rank/alpha 16, LR 1e-4, ~1000 steps with
the audio loss zeroed (stills carry no audio); or 50–200 clips of 3–15 s at
*exactly* 24.000 fps with the audio kept, rank 16, 1500–5000 steps.

## Operational notes

- **Weights are ~62 GB per partition.** Budget disk and first-run download time.
- Both modes register as evictable, VRAM-tracked models. In-engine, eviction drops
  the pipeline like any other; in worker mode it stops the whole subprocess, which
  is the only way that process's pages actually go back.
- In-engine, an H3 OOM or device-side assert poisons the engine's CUDA context and
  takes down every model on that engine — the reason the worker mode exists as a
  fallback while H3 is unproven.
- Switching workflow within one model reloads the other partition — batch requests
  by workflow if you care about wall-clock.
- The service logs under `[h3]`; failures surface with the worker's last output.
