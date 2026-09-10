# MiniMax-H3 (video + native audio)

H3 is a 33B flow-matching DiT that denoises **video and audio in one packed
sequence** — every generation comes back as an mp4 that already has a synchronised
soundtrack, not a dub. It is served through an isolated worker rather than the
normal diffusers loader, for one blunt reason: diffusers ships H3 as Modular
Diffusers blocks only (`MiniMaxH3Blocks` / `MiniMaxH3ModularPipeline`, 0.40+),
there is no `DiffusionPipeline` half, and the main venv is on diffusers 0.38.

```
/v1/video/generations ──► codai/api/video.py::_generate_h3
                            └─► codai/api/h3_worker.py   (venv + process + eviction)
                                  └─► tools/h3_service.py  (isolated venv, HTTP)
                                        └─► ModularPipeline.from_pretrained(..., workflow=…)
```

Same shape as the pyannote, NeMo-Canary and vLLM workers.

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
| `h3_venv` | override the isolated venv location |
| `gpu_device` | pins `CUDA_VISIBLE_DEVICES` for the worker |

The venv builds itself on first use from `requirements-h3.txt` (several GB) under
`/opt/coderai/h3_venv`, the `/cache` mount, or `~/.coderai/h3_venv`.

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
- The worker is registered as an evictable, VRAM-tracked model; eviction stops the
  whole subprocess, which is the only way the transformer's pages actually go back.
- Switching workflow within one model reloads the other partition — batch requests
  by workflow if you care about wall-clock.
- The service logs under `[h3]`; failures surface with the worker's last output.
