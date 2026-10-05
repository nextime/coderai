# LongCat-Video

[LongCat-Video](https://github.com/meituan-longcat/LongCat-Video) (Meituan, 13.6B dense,
MIT) does text-to-video, image-to-video, video-continuation, long video, interactive
generation and audio-driven avatars. It is natively pretrained on continuation, which is
why minutes-long output does not drift in colour or decay in quality the way chained
frame-tail conditioning on other models does.

## Why it runs in its own venv

There is **no diffusers pipeline** for it — diffusers merged LongCat-*Image*, not the
video model ([issue #12709](https://github.com/huggingface/diffusers/issues/12709) is
still open) — so coderai drives the upstream repo's own `LongCatVideoPipeline`. That
cannot happen in the server process:

| | coderai | LongCat |
|---|---|---|
| Python | 3.13 | **3.10** |
| torch | 2.11+cu130 | **2.6.0+cu124** |
| transformers | 5.x | **4.41.0** |
| numpy | 2.4 | **1.26.4** |
| opencv | opencv-contrib | `opencv-python` |

Four simultaneous conflicts. It therefore runs as a managed subprocess
(`tools/longcat_service.py`) in an isolated venv, the same shape as vLLM and the OCR
Paddle/Surya engines, with `codai/api/longcat_worker.py` owning the venv, the process and
the VRAM registration.

Two prerequisites, both bundled in the image and both needed on a source install:

1. **A Python 3.10 venv** from `requirements-longcat.txt`. coderai *cannot build this
   itself* — every venv bootstrap in `codai/` uses `sys.executable -m venv`, which is
   3.13. The image ships the standalone 3.10 it already carries for the lip-sync tools
   (`/opt/coderai/py310`) plus the built venv.
2. **The repo checkout.** `longcat_video` is a package *inside the repo*, not on PyPI, and
   it vendors its own Wan VAE and scheduler. Bundled as code only, no weights.

```bash
git clone https://github.com/meituan-longcat/LongCat-Video ~/.coderai/LongCat-Video
python3.10 -m venv ~/.coderai/longcat_venv
~/.coderai/longcat_venv/bin/python -m pip install -r requirements-longcat.txt
```

`flash-attn` is deliberately **not** installed: it is a source build needing ninja and a
CUDA toolchain, and upstream accepts xformers, which ships a wheel. Set
`longcat.attention = "flash"` once you have installed it yourself.

## Downloading the weights

Exactly like any other model — the models page, or
`POST /admin/api/model-download` with a `model_id` and an optional `file_pattern`. The
pattern matters here: the base repo is **~83 GB** across variants.

| Repo | What it is |
|---|---|
| `meituan-longcat/LongCat-Video` | base, bf16 |
| `meituan-longcat/LongCat-Video-Avatar` | audio-driven, wav2vec2 encoder |
| `meituan-longcat/LongCat-Video-Avatar-1.5` | Whisper-large-v3; the only INT8 variant, and it **requires** distilled sampling |
| `szwagros/LongCat-Video-Avatar-1.5-fp8`, `wavespeed/LongCat-Video-Avatar-1.5-e4m3` | FP8 e4m3, reportedly 12–16 GB |
| `Kijai/LongCat-Video_comfy` | FP8 scaled, ComfyUI layout |
| `vantagewithai/LongCat-Video-Avatar-*-GGUF-ComfyUI` | 2–8 bit GGUF (needs the ComfyUI gguf loader — not supported yet) |
| `fjkane/LongCat-Video-bf16` | bf16 mirror |

The checkpoint directory must contain `tokenizer/`, `text_encoder/`, `vae/`,
`scheduler/`, `dit/` and — for the distilled and refinement stages — `lora/`. coderai
checks that **before** loading, so a missing piece is reported immediately rather than as
a `from_pretrained` traceback minutes in.

## Configuring it

Model page → set **Backend** to `longcat`, which reveals the LongCat group: variant,
default quality, offload, segments, conditioning frames, context-parallel GPUs, the repo
checkout and venv, and the avatar fields. Global runtime settings (venv, the 3.10
interpreter, `auto_build`, attention backend, timeouts, the eviction drain) live in
**Settings → LongCat**.

## Generating

```jsonc
POST /v1/video/generations
{
  "model": "longcat",
  "prompt": "a cat walking through a neon-lit street",
  "quality": "fast",          // draft | fast | best
  "num_segments": 11          // or duration_seconds; ~11 segments is a minute at 15fps
}
```

**Three coarse-to-fine stages**, which `quality` selects:

| Preset | Stages | Steps |
|---|---|---|
| `draft` | base 480p | 50 |
| `fast` (default) | distilled | 16 |
| `best` | base + 720p refinement | 50 + 50 |

Pass `stage` (`base`/`distill`/`refinement`) instead to run **one** stage per request —
the resumable path; `refinement` needs frames from an earlier stage.

**Modes** map onto upstream's entry points: `t2v`, `i2v`/`ti2v` (needs `init_image`), and
`extend`/`v2v` (needs `cond_frames` — the same field the VACE path uses, so a client that
already chains clips needs no new vocabulary). The audio-driven avatar tasks are a
separate increment and are refused with a 400 rather than downgraded.

**Long video** is generated in segments. A call emits `num_frames` (93) but its first
`num_cond_frames` (13) re-render the previous segment's tail, so each segment adds the
remainder — which is why ~11 segments is about a minute. The response also returns the
tail, so a caller that prefers to chain requests itself can.

## VRAM, eviction and swap

The 13.6B DiT is ~27 GB at bf16 and the reported peak for a full profile is ~41.6 GB, but
**no official figure exists** and the real number moves with the variant, the stage and
the offload mode. So the configured reservation is a starting point and the service
reports the peak it actually reached (`torch.cuda.max_memory_reserved`), which is written
back per model+config and used for the next reservation. The high-water mark is kept: a
draft run touches far less than a 720p refinement.

LongCat takes its turn like any other model. It evicts others before loading, and
registers a releaser so others can evict it. The handover happens at a **segment
boundary**: a release asks the generation to yield there, so an eviction arriving
mid-request costs one segment instead of the whole generation, and the request still
returns — shorter, with a warning naming why — rather than failing. Bounded by
`longcat.evict_drain_timeout_s`.

Set `max_instances` on the entry to have video requests queue at the front as text does;
without it, video passes through unqueued as it always has.

## Remote and scaled

`longcat.service_url` (or `CODERAI_LONGCAT_SERVICE_URL`) points at a service running
elsewhere — another host, a container, a rented pod — and coderai proxies to it without
building or downloading anything. `distribute` fans independent generations across
engines and nodes. A `video-longcat` RunPod capability image needs a custom Dockerfile
(the generic one is python:3.12 + cu128, wrong on both axes) and is not built yet.

## Not done yet

- audio-driven avatar tasks; GGUF variants (they need the ComfyUI gguf loader)
- context-parallel multi-GPU (`cp_size` is accepted but not yet acted on)
- the `video-longcat` pod image
- LoRA/QLoRA training on top of it (feasible — upstream's own refinement expert is a LoRA)
- **no generation has been verified against real weights**
