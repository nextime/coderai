# OCR subsystem (dedicated OCR engines)

CoderAI OCRs documents with **purpose-built OCR engines** (detection + recognition) —
**not** a vision LLM. That means faithful transcription with per-line bounding boxes and
layout, GPU batching, and no hallucinated text. Three engines are integrated and
selectable per request:

| engine | id | license | how it runs |
|---|---|---|---|
| [docTR](https://github.com/mindee/doctr) | `doctr` | Apache-2.0 | **in-process** (uses the main venv's torch); works on GPU. Recommended on new-CUDA boxes. |
| [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) + PP-Structure | `paddle` | Apache-2.0 | **isolated venv subprocess** (layout + tables) |
| [Surya](https://github.com/VikParuchuri/surya) | `surya` | GPL (compatible with coderai's GPLv3) | **isolated venv subprocess**; best layout/reading-order; opt-in |

Per document you get three outputs: **plain text**, **structured JSON fields**, and
**stamp/signature flags**.

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

## Enable & configure

Settings → **OCR** card, or `config.json` `"ocr"`. Key fields: `enabled`,
`default_engine`, `dpi`, `max_concurrency`, `lang`; per-engine `*_enabled` /
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
page carries `lines[]` (text + bbox + conf), `regions[]` (layout), and `tables[]`. The
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
