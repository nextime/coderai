# Image prompts: weights, BREAK, long prompts and CLIP skip

A CLIP text encoder reads 77 tokens, two of which are the start and end
markers. That limit is the reason the Stable Diffusion ecosystem grew a prompt
syntax of its own, and the reason a server that ignores it renders something
other than what was asked for — without failing, which is the awkward part.

CoderAI encodes image prompts itself rather than handing the text to the
pipeline. Three things follow from that, on every CLIP-family image model
(SD 1.x, SD 2.x, SDXL and their fine-tunes):

* **No 77-token limit.** The prompt is split into as many 75-token windows as
  it needs, each encoded and concatenated. Nothing is truncated and nothing is
  dropped quietly.
* **Weights.** `(tag:1.3)` emphasises, `[tag]` attenuates, bare `(tag)` is
  1.1, nesting multiplies, and `\(` is a literal bracket.
* **BREAK.** Starts the next window, so tags describing different subjects are
  not blended by sharing one.

And `clip_skip`, which decides how deep in CLIP the embedding is read from.

## Why it matters

The booru-tag fine-tune families — Pony Diffusion, Illustrious, NoobAI and the
hundreds of models derived from them — are trained on prompts that open with a
run of quality tags and are read at `clip_skip` 2:

```
score_9, score_8_up, score_7_up, score_6_up, rating_safe, source_anime,
(masterpiece:1.2), 1girl, red hair, standing in a field of wheat, golden hour,
(detailed background:1.3) BREAK
wide shot, depth of field, film grain
```

Against a server that passes that text straight to a pipeline, three separate
things go wrong at once: the prompt is cut off partway through the quality tags
so the subject is never seen; the brackets and colons arrive as literal tokens,
i.e. as noise; and the embedding is read from the wrong layer, which washes the
output out. All three answer `200` with an image, which is why they are easy to
miss and worth stating.

## Requests

`POST /v1/images/generations`, `/v1/images/edits` and `/v1/images/inpaint` all
take:

| Field | Meaning |
|---|---|
| `negative_prompt` | What to avoid. The counterpart of the quality tags (`score_4, score_5, score_6, worst quality`) and as load-bearing as the prompt itself. |
| `clip_skip` | Read the embedding this many CLIP layers from the end. Omitted uses the model's configured value, then the pipeline's own default. |
| `prompt_weighting` | `false` keeps brackets literal, for a caller sending prose. The windowing stays on either way, so turning it off never costs tokens. |

```json
{
  "model": "pony-xl",
  "prompt": "score_9, score_8_up, (masterpiece:1.2), 1girl, red hair BREAK wide shot",
  "negative_prompt": "score_4, score_5, score_6, worst quality, blurry",
  "clip_skip": 2,
  "size": "832x1216"
}
```

## Per model, for clients that cannot ask

A client speaking the plain OpenAI image API has no `clip_skip` field to send,
and a model that needs 2 needs it on every request. So both settings are part
of the model's own configuration — **Models → Prompt Handling** in the admin
UI, or `clip_skip` / `prompt_weighting` on the model's entry in `models.json`.
A request that sends either one still wins; the config is the default for
everyone who does not.

There is also a server-wide default under `image` in `config.json`
(`clip_skip`, `prompt_weighting`), which the model's entry overrides in turn.

## GGUF models (sd.cpp)

sd.cpp parses the weight grammar and manages the token windows itself, inside
the engine, so a prompt reaches it as written and `clip_skip` is passed
through. Until v0.4.89 the negative prompt was not: the call hardcoded an empty
string, so for every GGUF image model whatever the caller asked to avoid was
discarded between the API and the engine.

## What is not covered

Flux, SD3 and Z-Image carry a T5 encoder with a much longer limit and
conventions of its own. CoderAI leaves those on their pipeline's native text
path — the weight grammar is a CLIP-era convention and a T5 reads it as text,
which is what those models expect. Negative prompts work there as the pipeline
supports them.

Textual-inversion embeddings (`easynegative` and friends) are not loaded by
name yet; a LoRA is the supported way to add learned tokens
(see [zimage-lora-training.md](zimage-lora-training.md)).

## Implementation

`codai/models/prompt_weighting.py` holds the parser, the windowing and the
encoder, and documents how `clip_skip` maps to a hidden-state index — SDXL
takes the penultimate layer by default and counts from there, SD 1.x takes the
final output and needs the layer norm applied by hand. Getting that mapping
wrong shifts every image, so it mirrors diffusers exactly.
`tests/test_prompt_weighting_and_clip_skip.py` pins the grammar table, the
window arithmetic, both layer conventions and the three defects above.
