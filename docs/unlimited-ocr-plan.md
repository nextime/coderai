# Plan: Unlimited-OCR as a coderai OCR engine

`baidu/Unlimited-OCR` (MIT, 3B params, BF16) is a document-layout VLM that emits
layout-tagged text with bounding boxes and block categories. This is the plan to
support it the way coderai supports any engine: orchestrated, swappable,
memory-managed, scalable where the operator configures it, and fully
configurable in the interface — on **every** infrastructure, not one.

It is written for the public release. Nothing here is scoped to a language, a
document type, a GPU model or a deployment size.

## Why it is worth supporting

- **It reaches a class of operator our other OCR VLMs do not.** Upstream states
  8 GB of VRAM is enough for BF16. That puts document-layout OCR on one
  consumer GPU, in-process, with no rented hardware and no cluster.
- **It is an in-process engine**, not a subprocess-per-page one. It serves over
  an OpenAI-compatible endpoint, so it follows `chandra`/`olmocr`: one thread
  per page, `instances` as a concurrency slot, no isolated venv, no lanes.
- **Its output carries structure.** `<|det|>type [bbox]<|/det|>` gives block
  category and geometry, which is what `OcrPage` already models.
- **It parses long documents in one pass** (32k context). Today every coderai
  OCR engine is strictly per-page; a document-level path is a capability the
  whole subsystem lacks, and this is a reason to design one.

## What upstream requires (and what each requirement costs us)

| Requirement | Source | Consequence for coderai |
|---|---|---|
| Architecture is **not in any released vLLM wheel**; use `vllm/vllm-openai:unlimited-ocr` | vLLM recipe | We pin vLLM 0.27.1, so this needs its own serve target. Same situation as `packaging/runpod/profiles/vllm-ocr.txt`. The engine must also **detect** a server that cannot serve the model and say so, because operators will point it at a stock vLLM. |
| Custom logits processor `vllm.model_executor.models.unlimited_ocr:NGramPerReqLogitsProcessor` | vLLM recipe | Server-side flag we cannot set on someone else's endpoint. The engine must detect its absence (runaway coordinate-token output) and report it as a server misconfiguration rather than a bad page. |
| Per-request `vllm_xargs`: `ngram_size=35`, `window_size=128` (**1024** for multi-page) | vLLM recipe | Request-shaping parameters → config knobs with documented defaults, sent as extra body fields. |
| Prompt must literally start `<image>`; `skip_special_tokens=False`; `temperature=0.0`; no chat template | model card + recipe | Each one wrong yields **empty output, not an error**. The engine asserts all three and names the one that is missing. |
| `--no-enable-prefix-caching`, `--mm-processor-cache-gb 0` | vLLM recipe | Documented in the serve target; harmless if an operator's endpoint differs. |
| Single image = crop ("gundam") mode; multi-image falls back to base mode | model card | The per-page and whole-document paths are not quality-equivalent. Say so; never silently switch. |

Unenumerated upstream: parameter-level language coverage ("multilingual", no
list), and any accuracy claim per language. Treated as unknown, not assumed.

## Phase 1 — the engine and the parser (no GPU, no image, offline-testable)

`codai/ocr/unlimited.py`, modelled on `codai/ocr/chandra.py`:

1. `UnlimitedEngine(OcrEngine)` with `load`/`cleanup`/`vram_gb`/
   `prelaunch_vram_gb`/`recognize_image`, so it joins the VRAM-eviction
   contract in both directions and takes its turn through the existing GPU swap
   gate.
2. **The `<|det|>` parser** — the only genuinely new logic. Parses
   `<|det|>type [bbox]<|/det|>text` into lines/regions carrying the block
   **category** (coderai currently drops category information from every
   engine; this is the first that supplies it). Tested against recorded output
   strings: well-formed, truncated mid-marker, missing bbox, unknown category,
   nested markers, coordinate-token runaway, and empty output.
3. **The empty-output guard.** Assert the three decode preconditions before the
   request and classify an empty reply, rather than returning a blank page:
   missing `<image>` prefix, `skip_special_tokens` left true, temperature not 0,
   or a server without the logits processor. Each gets its own message.
4. Config: `unlimited_*` fields mirroring the other engines (`enabled`,
   `instances`, `serve`, `server_url`, `model`, `langs`, `max_tokens`) plus
   `ngram_size` / `window_size` / `window_size_multipage`, each with upstream's
   default and a comment saying why that number.

Deliverable: the engine reads a recorded reply correctly and refuses an
unservable one with a named reason. Costs nothing to run.

## Phase 2 — serve modes, so every infrastructure is a first-class target

One `unlimited_serve` setting, the same shape surya already has:

- `local` — in-process through coderai's own vLLM backend on a local GPU. **The
  path that matters most for this model**, since 8 GB is the point.
- `server` — an endpoint the operator already runs (vLLM **or** SGLang;
  upstream documents both). URL and token in config; we start nothing.
- `model` — a coderai-managed model entry, which is what makes it routable,
  swappable and scalable like any other model.
- `pod` — a rented GPU, for operators who scale out.
- **No GPU at all** — the engine reports why it cannot serve, in
  `/v1/ocr/engines`, rather than failing at the first request.

Admin UI: full card in Settings → OCR, every knob above, each with a tooltip
carrying the upstream constraint it encodes.

## Phase 3 — the serve target for operators who want one

A profile following `packaging/runpod/profiles/vllm-ocr.txt`: baidu's image, the
logits processor, prefix caching and mm cache off, `--trust-remote-code`,
32k context. Built one at a time like every capability image.

## Phase 4 — document-level OCR, designed for the subsystem not for one engine

The 32k one-shot mode is this model's distinguishing feature, and coderai's OCR
contract is per-page end to end. So the work is a **general** capability:

- a document-level request path that submits N rasterised pages in one call;
- `OcrDocument` as the unit, with per-page results carved out of one reply;
- capability reporting, so a client can ask which engines can do it and fall
  back to per-page where they cannot;
- `window_size` switching to the multi-page value, and an honest statement that
  multi-image input uses base rather than crop mode.

Any engine that later gains the ability implements the same interface. This
phase is worth doing only once Phase 1-3 are serving, and should not be rushed
into the per-page path as a special case.

## What must be measured, not claimed

Language coverage, accuracy per document type, and tokens per page are
**measurements with conditions**, published as such. One operator's corpus
proves nothing about the next one's, so the engine reports what it did (tokens,
attempts, categories found) and the docs record measurements with the hardware,
the vLLM build and the corpus named. No language is designed in or out.
