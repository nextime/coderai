# Plan: train LoRA/QLoRA adapters for vision models, from images and video

coderai can **serve** a LoRA on a vision-language model, and cannot **train**
one. This is the plan to close that, for images and for video, the way coderai
supports anything else: orchestrated, configurable in the interface, and on
whatever infrastructure the operator has — a single consumer GPU or rented
cards.

It is written for the public release. Nothing here is scoped to a language, a
document type, a card or a deployment size.

## What exists today, exactly

Two unrelated things both called "LoRA", and neither is this:

| | what it is | where |
|---|---|---|
| `codai/api/loras.py` | a **diffusion** trainer: SDXL/Z-Image UNet-DiT and Wan/LTX2/H3 video transformers. Trains a *generator* to produce a subject. Its "QLoRA" is a 4-bit video transformer for fitting training on one card | `POST /v1/loras/train` |
| `codai/models/text_loras.py` | **serving and packaging** only: `configured_specs`, blob ids, bundles, portability. No training anywhere in it | — |

So an adapter for an LLM or a VLM has to be trained outside coderai and
registered by path. `diogene-alpha` and `digesta-alpha` arrived that way.

Serving such an adapter already works, on local engines and on rented pods: a
coderai pod is sent adapters **with the request**, and `_sibling_lora_specs`
registers every adapter of the same base at launch so one engine serves the
base and each adapter under its own name (v0.4.10/0.4.11 made that read every
model section, which is what lets a `vision_models` entry own one).

## What to build

A **supervised fine-tuning path for served models**, parallel to the diffusion
one rather than inside it — the two share almost nothing beyond the word.

- **Datasets are conversations with media.** The training unit is an
  OpenAI-shaped `messages` list whose user content carries `image_url` or a
  video reference, with the assistant turn as the target. That is the format
  the inference side already accepts (`codai/api/text.py` normalises
  `image_url`/`input_image`), so a dataset is recorded requests plus the answer
  they should have given. No new interchange format.
- **Video is frames plus a budget, not a new modality.** A VLM with a
  `video_token_id` takes sampled frames; the knobs are frames-per-clip,
  sampling, and the per-sample token budget. The ceiling is the context window,
  and the arithmetic is the same one `codai/ocr/resume.py` already states for
  output: the window is the ceiling, not the budget.
- **QLoRA by default, because of the base sizes.** A 27B base on one 40–48 GB
  card needs 4-bit. `text_loras.py` already records the constraint that makes
  the result loadable: *"QLoRA trained against a 4-bit base loads like any
  other, provided the base is loaded the way it was trained"*, which is what
  `load_in_4bit` on the entry is for.
- **The vision tower stays frozen.** vLLM applies adapters to the language
  model's linear layers, so an adapter targeting `model.visual.*` trains
  something that cannot then be served. Freezing the encoder is also the
  standard way VLMs are fine-tuned, so this is a constraint to *state and
  enforce* (refuse such a target with a clear reason) rather than a limitation
  to regret. Worth noting: AWQ/INT4 releases commonly leave the vision tower
  unquantised — `cyankiwi/Qwen3.8-27B-AWQ-INT4` has 111 `model.visual.*` layers
  in its `ignore` list — so the encoder is full precision either way.
- **Training is a job, like compose.** Progress, cancellation, resumption from
  a checkpoint and an eventual artefact in the blob store, so the adapter is
  registerable the moment it finishes. `loras.py` already has
  `_save_train_checkpoint`/`_apply_peft_checkpoint` to follow.
- **It must run where the operator's capacity is.** On a local GPU, and on a
  rented pod for a base too large for one — which means the dataset and the
  adapter have to travel, and the pod path must be a first-class target rather
  than an afterthought. `cluster/ddp.py` is the precedent for multi-card.

## Why it is worth doing

- A served VLM is the one model class coderai can orchestrate end to end
  *except* for adapting it, which is exactly what makes a general document or
  domain reader useful to an operator with their own corpus.
- Every piece except the trainer already exists: serving, per-request adapters,
  blob packaging, pod seeding of `loras`, multi-name engines, 4-bit bases, and
  a job system. This is the missing middle, not a new subsystem.
- It generalises past OCR. The same path adapts a VLM for classification,
  extraction, captioning or video QA, which is why it belongs in the product
  rather than in one install's scripts.

## Open questions, to answer with measurements rather than opinions

- Which trainer backend: peft + transformers directly, or a library that
  already handles multimodal collation. The deciding factor is what accepts the
  `messages`-with-media format without a bespoke converter.
- Whether a video sample is worth its token cost at all for the tasks operators
  actually have, measured on a real dataset before frames-per-clip gets a
  default.
- How to evaluate an adapter from inside coderai. Training something
  unmeasurable is how `fallback_engines` came to never fire: a degraded result
  that answers 200 is indistinguishable from a good one unless something
  measures it.
