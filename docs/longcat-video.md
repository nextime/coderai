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

## Multi-GPU

`cp_size` on the model entry splits the DiT's spatial dimensions across that many GPUs.
It is **N processes under torchrun**, not N threads: NCCL is initialised and
`init_context_parallel()` called *before* the model loads (it is what makes the DiT split
at all), and every rank must enter the same pipeline call or the collectives deadlock. So
rank 0 owns the HTTP socket and broadcasts each job to the others, which wait in a loop
for one. Asking for more ranks than there are visible GPUs is refused up front, because
NCCL's own failure for that names nothing useful.

`cp_size: 1` — the default — initialises no distributed group at all.

## Remote and scaled

`longcat.service_url` (or `CODERAI_LONGCAT_SERVICE_URL`) points at a service running
elsewhere — another host, a container, a rented pod — and coderai proxies to it without
building or downloading anything. `distribute` fans independent generations across
engines and nodes.

For rented GPUs there is a **`video-longcat`** capability image. The generic capability
builder is python:3.12 on the cu128 torch index, so it cannot express this image at all —
LongCat needs 3.10 on cu124, meaning two interpreters in one image: the light core runs
coderai, a 3.10 venv runs the model. It therefore carries its own Dockerfile (the same
escape hatch the `engines` profile uses), builds the venv from
`requirements-longcat.txt` and asserts 3.10 / torch 2.6 / transformers 4.41 at build
time, clones the pipeline source and checks it imports, and bakes **no weights**.

A model with `backend: longcat` is routed to that image automatically — placing it on the
plain `video` image would give a pod that boots, passes its health check, accepts the
request and has nothing to generate with. Build it with
`packaging/runpod/build_capability_image.sh video-longcat`, which runs the container and
polls `/healthz` for 120 s before declaring success.

## Variants and the avatar families

The avatar weights are **a different pipeline** upstream
(`LongCatVideoAvatarPipeline`), with its own methods and **two** guidance scales — how
hard to follow the prompt, and how hard to follow the audio. Its distilled pass is also
not the base model's: 8 steps against `dmd_lora`, where the base model's is 16 against
`cfg_step_lora`.

| | base | Avatar | Avatar-1.5 |
|---|---|---|---|
| audio encoder | — | chinese-wav2vec2-base | whisper-large-v3 |
| modes | t2v, i2v, extend | at2v, ai2v | at2v, ai2v |
| INT8 | no | no | **yes** (`base_model_int8/`) |
| distilled | `cfg_step_lora`, 16 steps | no | **required**, `dmd_lora`, 8 steps |

Unsupported combinations are **refused before loading** rather than quietly loading
something else: INT8 or the DMD distillation on anything but avatar-1.5, avatar-1.5
without distilled sampling, a base-model mode on avatar weights (or the reverse), and the
GGUF variants, which need ComfyUI's gguf loader.

FP8 loads the DiT as `float8_e4m3fn`. Community FP8 layouts differ between publishers
(Kijai's scaled format is not a plain e4m3 dump), so if a repo needs its own loader the
load fails with that reason instead of producing noise.

Audio-driven generation:

```jsonc
POST /v1/video/generations
{
  "model": "longcat-avatar-15",
  "mode": "ai2v",                 // at2v = audio+text; ai2v adds a reference image
  "prompt": "a news anchor reading the headlines",
  "audio_file": "<base64 or URL>",
  "init_image": "<base64 or URL>",
  "audio_guidance_scale": 4.0
}
```

The audio is resampled to 16 kHz (the encoder's rate) and embedded by the family's
encoder. `ref_img_index` (0–24; 30 reduces repeated actions) and `mask_frame_range` are
set on the model entry.

## LoRA / QLoRA training

Supported, through **SimpleTuner**, which coderai drives. It does not train in-process,
and the two reasons are worth knowing because they are not going to change on their own:

1. LongCat's venv is standalone (Python 3.10, torch 2.6, transformers 4.41), so a trainer
   there cannot import `codai.api.loras` the way the overlay trainer for H3/Krea does —
   that venv inherits the parent's site-packages, this one deliberately does not.
2. The upstream repo has **no way to create trainable LoRA layers**. Its DiT exposes
   `load_lora()`, `enable_loras()` and `disable_all_loras()` and nothing that initialises
   one, and the adapters it loads (`cfg_step_lora`, `refinement_lora`, `dmd_lora`) use its
   own key layout. Hand-rolling that layout unverified would produce adapters the pipeline
   cannot load — support in appearance only.

Upstream's own refinement expert *is* a LoRA on the base model, so the architecture is
well suited to this; it is the training-side plumbing that is missing, not the capability.

```bash
python3.10 -m venv ~/.coderai/longcat_venv-train
~/.coderai/longcat_venv-train/bin/python -m pip install -r requirements-longcat-train.txt
```

A separate venv from the inference one: SimpleTuner pins its own torch and the two would
fight. Then:

```jsonc
POST /v1/loras/train
{
  "name": "my-style",
  "base_model": "meituan-longcat/LongCat-Video",
  "target": "longcat",
  "dataset_config": "/data/my-clips/simpletuner.json",
  "steps": 800,
  "rank": 8
}
```

`dataset_config` is **required** and is a SimpleTuner data-backend config describing
captioned video clips (its quickstart suggests 50–100 clips of 10–30 s). coderai will not
synthesise one from the `images` field other targets use: a still-image dataset would
train something other than a video LoRA, and silently.

**QLoRA** is `longcat.train_base_precision` — `int8-quanto` (the default),
`int4-quanto` or `fp8-torchao`, no extra installs. A 13.6B transformer plus optimiser
state does not fit a consumer card at bf16, so the quantised base is the normal path
rather than the exception. Batch size 1, gradient checkpointing on, rank 4–8.

Two constraints are checked **before** a long run starts, because they are the VAE's and
not preferences: `(num_frames - 1)` must be divisible by 4 — the same 4n+1 rule coderai
already applies to Wan — and each side of the resolution must be divisible by 16.

Progress arrives on the same job/JSON-lines protocol the other trainers use, so
`/v1/loras/progress` and the Tasks page work unchanged.

## Not done yet

- GGUF variants (they need the ComfyUI gguf loader)
- **no generation has been verified against real weights**, avatar included
- **no training run has been executed** — the SimpleTuner driver is wired and its config
  and constraints are checked, but it has not been run against a real dataset
