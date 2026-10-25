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
| `clip_skip` | Which CLIP layer to read, counted from the end: **1 is the final layer, 2 the penultimate, n the n-th from the end** — A1111's `CLIP_stop_at_last_layers`, the number every model card quotes. Omitted uses the model's configured value, then the family's default (the penultimate layer on SDXL, the final one on SD 1.x). |
| `prompt_weighting` | `false` keeps brackets literal, for a caller sending prose. The windowing stays on either way, so turning it off never costs tokens. |
| `embeddings` | Textual inversions to load, so their trigger words work in the prompt — `["easynegative"]`. |
| `prompt_prefix` / `negative_prompt_prefix` | Override the model's standing prefix; `""` switches it off for this call. |

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

## Size

`size` is `"WIDTHxHEIGHT"` in pixels, or `"auto"` for the model's default.

A size that does not parse is a `400` saying so, rather than a silent
fall-back: `"1024"` used to generate a square nobody asked for, and the
response looked like a success. A side that is not a multiple of 8 is snapped
to the nearest one and logged, because the VAE downsamples by 8 and cannot
represent an odd side. A size above the install's ceiling is a `400` naming the
limit — `4096x4096` used to be accepted and come back as a CUDA OOM, which
reads like a busy GPU rather than a size the caller chose.

The ceiling is `image.max_side` (default 4096) and `image.max_pixels` (default
4.2 MP, which covers 2048×2048 and 2560×1440). Raise both if the card can take
it.

On `/v1/images/edits` and `/v1/images/inpaint`, omitting `size` keeps the
source photo's own aspect at about 1 MP on a 64-pixel grid. Both used to
default to a square, which squashed every non-square photo — and on inpaint it
pulled the mask out of alignment with the image as well.

## Standing prefixes: the ladder a model always wants

CoderAI never adds tags to a prompt by itself. The `score_*` ladder is yours to
send — in both directions — and what the sections above do is make sure it is
read correctly, not supply it.

When a model should always get it, that belongs to the model:
`prompt_prefix` and `negative_prompt_prefix` on its config entry (admin →
Models → Prompt Handling). They are comma-joined in front of whatever the
caller sent, which is what makes a Pony-family model usable from a client that
speaks the plain OpenAI image API and only sends a sentence:

```json
{ "prompt_prefix": "score_9, score_8_up, score_7_up, score_6_up",
  "negative_prompt_prefix": "score_4, score_5, score_6" }
```

A request can send its own `prompt_prefix` / `negative_prompt_prefix` to
override either, and an **empty string switches the prefix off for that call** —
what a caller writing its own complete prompt needs. The prefix is applied at
most once: a prompt that already opens with it is left alone, so pasting a full
prompt does not get the ladder twice. Both default to empty, so an install that
configures nothing sends exactly what the caller wrote.

The prefix is prompt text like any other, so the weight grammar works in it:
`(masterpiece:1.2), score_9` is a valid prefix.

## Learned prompt tokens (textual inversion)

A textual inversion is a few vectors trained to sit in CLIP's embedding table
under a word of their own. Writing that word is how much of the ecosystem's
negative prompting is built — `easynegative`, `bad-hands-5`, `FastNegativeV2`
— and until the file is loaded the word is an ordinary unknown token, so the
prompt quietly asks for nothing.

Three ways to get one loaded, all of which can be combined:

* `embeddings: ["easynegative"]` on the request, for an ad-hoc one.
* `textual_inversion` on the model's config (admin → Models → Prompt
  Handling), for the ones a model always wants.
* every file in `image.embeddings_dir`, discovered by filename — which is how
  A1111 works, and what a prompt copied off a model card assumes.

A reference is a bare name (looked up in the directory), a path to a
`.safetensors` / `.pt` / `.bin`, or a HuggingFace repo id. `ref:token` renames
the trigger word: `my-emb.safetensors:myneg`.

A name the caller gave that cannot be loaded is a `400` — rendering without it
would look like a success. A file in the directory that fails to load is a line
in the log, because one bad file should not break every request.

Two details that decide whether this works at all, both handled:

* **SDXL keeps two text encoders**, so an SDXL-format embedding holds a
  `clip_l` and a `clip_g` tensor that must go into the matching
  tokenizer/encoder pair. Loading only the first gives the trigger word an
  embedding in one encoder and nothing in the other.
* **A multi-vector embedding occupies several tokens**, registered as `tok`,
  `tok_1`, `tok_2`… A prompt only ever writes `tok`, so it is expanded before
  tokenizing. Without that, an eight-vector embedding contributes one vector
  and appears to do nothing.

## What is not covered

Flux, SD3 and Z-Image carry a T5 encoder with a much longer limit and
conventions of its own. CoderAI leaves those on their pipeline's native text
path — the weight grammar is a CLIP-era convention and a T5 reads it as text,
which is what those models expect. Negative prompts work there as the pipeline
supports them.

## Implementation

`codai/models/prompt_weighting.py` holds the parser, the windowing and the
encoder, and documents how `clip_skip` maps to a hidden-state index.

That mapping is worth knowing about, because diffusers' argument of the same
name counts something else. Its SDXL path reads `hidden_states[-2]` by default
and `hidden_states[-(clip_skip + 2)]` otherwise, so **diffusers' `clip_skip` is
layers skipped *beyond* the penultimate one SDXL already uses**. Passing a
caller's 2 straight through reads `hidden_states[-4]` — two layers deeper than
asked for, which on a Pony-family model is the difference between a photograph
and mush. CoderAI shipped exactly that in 0.4.89 and fixed it in 0.4.92; the
mapping now happens once, here, instead of being inherited. SD 1.x additionally
needs the final layer norm applied by hand when an earlier layer is taken,
since `hidden_states` are pre-norm.

SDXL also conditions on a pooled vector that the UNet cannot run without, and
diffusers refuses `prompt_embeds` that arrive without it. Where a pipeline does
not hand one over, it is rebuilt the way CLIP produces it — pooled at the
end-of-text token, then projected — and only if even that is impossible does
the encoder stand down and let the pipeline encode the prompt itself.
`codai/models/textual_inversion.py` resolves and loads the embeddings, and
tracks what a pipeline already carries — a pipeline outlives the request that
loaded it, and diffusers raises on registering the same trigger word twice.

`tests/test_prompt_weighting_and_clip_skip.py` pins the grammar table, the
window arithmetic and both layer conventions;
`tests/test_image_size_and_embeddings.py` pins the size rules and the
embedding resolution, including the SDXL two-encoder split and the
multi-vector expansion.
