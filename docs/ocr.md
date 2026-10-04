# OCR subsystem (dedicated OCR engines)

CoderAI OCRs documents with **purpose-built OCR engines** (detection + recognition) —
faithful transcription with per-line bounding boxes and layout, GPU batching, no
hallucinated text — and, for the pages those read badly, one **document VLM**. Four
engines are integrated and selectable per request:

| engine | id | license | how it runs |
|---|---|---|---|
| [docTR](https://github.com/mindee/doctr) | `doctr` | Apache-2.0 | **in-process** (uses the main venv's torch); works on GPU. Recommended on new-CUDA boxes. |
| [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) + PP-Structure | `paddle` | Apache-2.0 | **isolated venv subprocess** (layout + tables) |
| [Surya](https://github.com/VikParuchuri/surya) | `surya` | GPL (compatible with coderai's GPLv3) | **isolated venv subprocess**; best layout/reading-order; opt-in |
| [olmOCR-2](https://huggingface.co/allenai/olmOCR-2-7B-1025) | `olmocr` | Apache-2.0 | **a document VLM**, served by a coderai model, coderai's vLLM, or any OpenAI endpoint. **No bounding boxes.** |

Per document you get three outputs: **plain text**, **structured JSON fields**, and
**stamp/signature flags**.

**Which one?** `paddle`/`doctr`/`surya` give you boxes, layout regions and tables, which
is what you want when something downstream has to point at a place on the page (redaction,
stamp detection, a table's cells). `olmocr` gives you the best *reading* of a hard page —
old scans, maths, multi-column, dense tables — and nothing positional. Stamp/signature
detection in `layout` mode has nothing to work with under `olmocr`; `detector` mode (YOLO)
still works, because it looks at the image, not the engine's output.

## Why isolated venvs for Paddle & Surya

Their dependencies conflict with the main coderai venv, so each runs in its **own
virtualenv + subprocess worker** (`codai/ocr/workers/ocr_worker.py`, JSON over a pipe) —
the same isolation pattern coderai uses for the DINOv2-SALAD embedder:

- **PaddleOCR** pulls `opencv-contrib-python` (clashes with `opencv-python`/`cv2`), and
  `paddlepaddle-gpu` wheels **bundle their own CUDA runtime** — so a cu12x Paddle wheel
  runs on GPU even when the host is on **newer CUDA (e.g. CUDA 13)**, as long as the driver
  is recent. Isolation makes both facts safe.
- **Surya** caps `pillow<11`, incompatible with the main venv's `pillow>=12`.

docTR is torch-based and compatible, so it stays in-process.

## Install

**Main venv (docTR + detection + PDF + validation):**
```
pip install -r requirements-ocr.txt          # or: ./build.sh nvidia --ocr
```
`pypdfium2` is already in the base requirements, so image OCR works even without this. Any
engine whose dependency is missing returns **HTTP 503** instead of crashing.

**PaddleOCR isolated venv** (`ocr.paddle_venv`, default `~/.coderai/paddle_venv`):
```
python3 -m venv ~/.coderai/paddle_venv
~/.coderai/paddle_venv/bin/pip install -r requirements-ocr-paddle.txt   # GPU wheel via Paddle's index
```
**Surya isolated venv** (`ocr.surya_venv`, default `~/.coderai/surya_venv`):
```
python3 -m venv ~/.coderai/surya_venv
~/.coderai/surya_venv/bin/pip install -r requirements-surya.txt
```

### Surya serving modes (`ocr.surya_serve`) — IMPORTANT: the venv version depends on the mode

Surya changed architecture across versions, so the surya venv must hold the version that
matches the chosen serving mode:

| `surya_serve` | What it is | surya‑ocr version | How it runs |
|---|---|---|---|
| `local` (default) | classic detection + recognition on torch | **≤ 0.17.1** (`requirements-surya.txt` pins 0.17.1) | in the isolated venv, GPU |
| `vllm` | the latest **"Surya2"** VLM (Qwen3.5‑VL, `datalab-to/surya-ocr-2`) | **≥ 0.20** (e.g. 0.22.1) | served by coderai's **vLLM backend**; Surya attaches via `SURYA_INFERENCE_URL` |
| `llamacpp` | same Surya2 VLM | ≥ 0.20 | attaches to a `llama-server` at `ocr.surya_server_url` |

Surya2 (≥0.20) is a VLM that CANNOT run standalone in the venv — it requires an external
OpenAI server (vLLM/llama‑server). `vllm` mode is recommended: coderai serves
`ocr.surya_model` through the vLLM backend (continuous batching) and points Surya at it.
So: install **surya‑ocr 0.17.1** for `local`, or **surya‑ocr>=0.20** for `vllm`/`llamacpp` —
one per venv.
Or set `paddle_auto_build` / `surya_auto_build` to have coderai create the venv and install
on first use. In the OCI image these venvs are **baked in** at `/opt/coderai/paddle_venv` /
`/opt/coderai/surya_venv` (same as the `lipsync_venv` / `parler-venv`), so the image is
self-contained; the engines prefer a baked venv, then `~/.coderai/<name>_venv`, then the
config path. Surya loads only when `surya_accept_license` is set.

## olmOCR-2 (`olmocr`) — the document VLM

[olmOCR-2](https://huggingface.co/allenai/olmOCR-2-7B-1025) is AllenAI's Qwen2.5-VL-7B
fine-tune for document transcription (Apache-2.0, 82.4 on olmOCR-bench). It reads a whole
page at once and answers in its trained format: a YAML front matter block followed by the
page as markdown, with equations as LaTeX and tables as HTML. coderai sends olmOCR's own
"no-anchoring v4" prompt verbatim — paraphrasing it costs accuracy — renders the page at
`olmocr_longest_side` (1288 px, what it was trained on), parses the front matter into the
page's `meta`, and strips it from `text`. When the model reports the page rotated, coderai
turns it and asks once more (`olmocr_retry_rotation`), which is what the olmOCR pipeline
itself does.

Three ways to serve it (`ocr.olmocr_serve`):

| mode | what it means | needs |
|---|---|---|
| `model` (default) | the checkpoint is a **vision model in your models.json** and olmOCR goes through coderai's own model manager — VRAM accounting, eviction, quantisation and the thermal governor included. The only mode that works **without CUDA** (GGUF + mmproj on llama.cpp over Vulkan/ROCm). | `olmocr_model_id` |
| `vllm` | coderai's vLLM backend serves `olmocr_model` on its own instance (continuous batching). CUDA only. | the vLLM backend configured |
| `server` | attach to an OpenAI-compatible server already running it: `llama-server`, a remote vLLM, another coderai. | `olmocr_server_url` (+ `olmocr_api_key` if it wants one) |

Fields: `olmocr_enabled`, `olmocr_serve`, `olmocr_model_id`, `olmocr_model`
(default `allenai/olmOCR-2-7B-1025-FP8`), `olmocr_server_url`, `olmocr_api_key`,
`olmocr_instances` (pages in flight), `olmocr_longest_side`, `olmocr_max_tokens`,
`olmocr_temperature`, `olmocr_timeout`, `olmocr_retry_rotation`.

```bash
curl -F file=@scansione-1974.pdf -F engine=olmocr http://localhost:8776/v1/ocr
```

Each page's `meta` carries what the model reported: `primary_language`,
`is_rotation_valid`, `rotation_correction`, `is_table`, `is_diagram` (plus
`rotation_applied` when coderai re-asked a turned page). Engines that report no metadata
omit the field entirely, so nothing else in the response shape changed.

**A pod cannot serve it.** Like Surya-2, olmOCR needs a model server the RunPod OCR images
do not carry, so a pod asked for `olmocr` serves the request with docTR and says so in the
log. Point `olmocr_serve: "server"` at something the pod can reach if you want the real
thing out there.

## When an engine's server will not start

The VLM engines (`surya` in `vllm` mode, `olmocr` in `vllm` mode) boot a vLLM instance
that claims `gpu_memory_utilization` × the **whole card** before it will serve, and
refuses to start when that much is not free. Three rules follow, all enforced by the pool:

- **Room is made before the load, not after it.** The engine declares what it needs up
  front (`prelaunch_vram_gb`) and the model manager evicts for it first. Evicting
  afterwards is useless, because the load is what fails.
- **The share is relative to what is free, not to the card.** `vlm_gpu_memory_utilization`
  is the slice the OCR engine wants for *itself*, but vLLM's own flag is a fraction of
  **total** card memory and every other process' allocation counts against it. So the
  share is translated at launch: 0.35 on a 24 GB card with 3.6 GB of resident embedder
  becomes `--gpu-memory-utilization 0.50`, a 12 GB budget of which 8.4 GB is actually the
  engine's. Without the translation, 0.35 left surya-2 about 5 GB, and after weights,
  activation peak at `max_model_len=18432` and CUDA graphs it reported
  `Available KV cache memory: -2.81 GiB` and died — once every 20 minutes, for hours.
  The translation is capped (0.92) so the driver keeps its room; when the cap would bite,
  the evict-first pass above is what clears the card instead.
- **A failed build is not retried on every request.** For `ocr.build_retry_cooldown_s`
  (default 60 s) that engine fails fast with the reason the boot gave, instead of spending
  ~35 s booting and dying again per request. Without this, a card 11 GB short of Surya-2's
  demand produced 338 consecutive failed boots in one morning, each one a half-minute hang
  for the caller.

If you see `Free memory on device cuda:0 … is less than desired GPU memory utilization`,
set **`ocr.vlm_gpu_memory_utilization`** (Settings → OCR → "VLM engines' GPU share"): the
OCR VLM gets its own share of the card instead of the one `vllm.gpu_memory_utilization`
asks for on behalf of an LLM — 0.35 is plenty for a 7B page model, where the LLM default
of 0.9 (or 0.6) can be more than the card has left. Otherwise: give the card less other
resident work, or serve the engine with `model`/`server` mode, which needs no instance of
its own at all.

### A document is never refused because one engine is down

`ocr.fallback_engines` (default `auto`) makes a failing engine hand the document to the
other **enabled** engines rather than returning 503 — fewest-ways-to-fail first
(`paddle`, `doctr`, `olmocr`, `surya`), since a transcription from the second-best engine
beats no transcription. The response's `engine` field names the engine that actually read
the pages, so a fallback is visible to the caller, and the reason the first choice failed
is logged. This also covers a pool that is evicted *mid-document* by a model load: the
failed pages are re-read by the next engine instead of failing the request. Set an
explicit order (`"doctr, paddle"`) to override it, or `off` to get the original error
back.

## Enable & configure

Settings → **OCR** card, or `config.json` `"ocr"`. Key fields: `enabled`,
`default_engine`, `dpi`, `max_concurrency`, `lang`, `build_retry_cooldown_s`,
`vlm_gpu_memory_utilization`, `fallback_engines`; per-engine `*_enabled` /
`*_instances` / `*_use_gpu` / lang; `detect_mode` (+ `detect_model_path`, `detect_conf`);
`extract_enabled` / `extract_model_id` / `extract_schema` / `extract_validate`.

**Concurrency on one GPU** comes from loading multiple instances of an engine
(`paddle_instances` etc.) and fanning pages across them (bounded by `max_concurrency`).
OCR models are small, so a 24 GB card (e.g. RTX 3090) fits several instances — good
aggregate throughput without vLLM (a continuous-batching path is a future option; see
[vllm.md](vllm.md)).

## Endpoints

```
POST /v1/ocr            multipart: file=@doc.pdf [engine=] [dpi=] [detect=] [structured=] [schema=]
POST /v1/ocr/batch      multipart: files=@a.pdf files=@b.png [...same fields...]
GET/POST/DELETE /v1/ocr/schemas[/{name}]   manage extraction schemas
```

Response: `{engine, num_pages, text, pages[], stamps[], signatures[], structured}`. Each
page carries `lines[]` (text + bbox + conf), `regions[]` (layout), `tables[]`, and
`meta` when the engine reports page metadata (olmOCR does; the others do not). The
`ocr` **pipeline step type** chains OCR with `text_gen` in custom pipelines.

```bash
curl -F file=@sentenza.pdf -F structured=true -F detect=both -F schema=italian_sentenza \
     http://localhost:8776/v1/ocr
```

## Stamp / signature detection (`detect_mode`)

- `off` — none.
- `layout` — flags the engine's layout regions (figure/seal/stamp/signature markers,
  EN + IT: *timbro/sigillo/firma*). Free but coarse.
- `detector` — a small **YOLO** model (`detect_model_path`, needs `ultralytics`) trained
  to spot signatures/stamps; tight boxes + confidence, bucketed into `stamps` /
  `signatures`.
- `both` — layout + detector.

Detection *locates/flags* stamps and signatures — it does not *verify* a signature.

## Structured extraction — schemas are data, not code

Extraction feeds the OCR text to an existing coderai text model
(`extract_model_id`). The target schema is **not hardcoded**; it is resolved three ways:

1. **Inline** — the `schema` field starts with `{`/`[`: a fill-in template or a JSON
   Schema, used verbatim.
2. **Named** — a bare name (`italian_sentenza`): looked up in
   `<config_dir>/ocr_schemas/*.json` (user schemas) then the built-in seeds; user files
   shadow built-ins.
3. **Auto** — empty / `auto`: the model extracts salient key/value fields itself.

Built-in seeds: `italian_sentenza` (Italian judicial decision, template),
`generic_document` (template), `invoice` (JSON Schema, validated when `jsonschema` is
installed). Manage your own from the **Settings → OCR** card (picker + raw-JSON editor +
field builder) or the `/v1/ocr/schemas` API. A `jsonschema`-typed schema is validated
against the model output when `extract_validate` is on (`_validation_error` is surfaced,
never fatal).

Adding a new document type — *atto di citazione*, *decreto ingiuntivo*, contracts, … — is
just dropping a JSON file in `ocr_schemas/` (or POSTing it); no code change.
