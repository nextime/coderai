# Images in a chat request

A model that is listed as able to read images has to read them. If it cannot,
the request is refused — it is never answered blind.

## What used to happen

`/v1/models` listed `Qwen3.5-9B` with `["text_generation", "image_to_text"]`.
A chat whose user turn carried an `image_url` part came back, for every
picture, with a description of nothing:

```json
{"shows": "a blank white image with no visible content"}
```

Nine photographs of one video were indexed that way. They were real pictures —
the footage scorer rated them 0.53–0.59 — and the model was not hallucinating.
It was answering the prompt it had been given, which was the literal text
`[image_url content]`.

Two causes, measured on transformers 5.12 against `Qwen/Qwen3.5-9B`, whose
`config.json` carries a `vision_config` and an `image_token_id`:

| auto class | resolves to | holds the vision tower |
|---|---|---|
| `AutoModelForCausalLM` | `Qwen3_5ForCausalLM` | no |
| `AutoModelForImageTextToText` | `Qwen3_5ForConditionalGeneration` | yes |

The engine asked for the first one, so the tower was never built and its
weights were never read. And with no projector loaded, the chat path flattened
multipart content to a text placeholder, because that is all it could do with
a picture.

## What happens now

**The class comes from the checkpoint's config, not from its name.**
`codai/backends/hf_vision.py` reads the config: a nested `vision_config` or an
`image_token_id`, plus an `AutoModelForImageTextToText` mapping in the
installed transformers, means the model is loaded with the class that holds
its tower. Every `from_pretrained` in the CUDA backend goes through that
resolver, and any uncertainty — unreadable config, unknown family, missing
mapping — falls back to the text class, so a model that is not a vision model
loads exactly as it did before.

Deciding from the name does not work in either direction: `Qwen3.5-9B` carries
no "vl" or "vision" and is a vision model, while plenty of ids containing
"vision" are not image-text-to-text at all.

**A vision model also loads its processor.** The `AutoProcessor` — not the
tokenizer — owns the image preprocessing and knows how many placeholder tokens
this family expands one picture into. `supports_vision` is set only when both
halves are present: a class that holds a tower *and* a processor that can
preprocess for it. A tokenizer-shaped "processor" with no image processor does
not count; transformers returns one for some text checkpoints and it would
pass every duck-type check until the first picture arrived.

**An image request takes its own generation path.** The prompt is rendered by
the processor's chat template and then fed through `processor(text=…,
images=[…])`, which yields `pixel_values` and the grid metadata the tower
needs. Images travel as a separate list in the order they appear in the turn,
because that order is what binds a picture to its placeholder.

That path deliberately does not use the KV prefix cache. Image tokens are
expanded from the pixels themselves, so a prefix cached from another request
would be reused against the wrong picture.

**A picture sent to a model that cannot see is a 400.**

```
Model 'X' cannot read images: it is served without a vision tower (an HF
checkpoint with no vision_config or processor, or a GGUF with no mmproj).
Send the image to a model whose capabilities list image_to_text, or
configure an mmproj for this one.
```

A client can fall back from a refusal. It cannot fall back from a confident
wrong answer, which is why this is an error and not a warning.

**The listing agrees with the runtime.** A local GGUF text entry with no
`mmproj` no longer advertises `image_to_text`: llama.cpp needs that projector
file to turn pixels into embeddings, and without it the request would now be
refused. HF checkpoints keep the capability — they load their own tower when
they have one — and a remote or pod-served model is the remote's business.

## Configuring vision

* **HF checkpoint** (`text_models`, `vision_models`): nothing to set. If the
  config has a vision tower and the installed transformers knows the family,
  it is loaded and used.
* **GGUF**: set `mmproj` on the entry to the projector file. Without it the
  model is text-only, and is listed as text-only.
* **Remote / pod-served** (vLLM, RunPod): vision is the remote's, and is
  reported as supported.

## Checking it

```bash
curl -s $HOST/v1/models | jq -r '.data[] | select(.capabilities[]? == "image_to_text") | .id'
```

Then send one picture to one of those ids, and the same picture to a text
model. The first answers about the image; the second answers `400`. Neither
describes a blank page.
