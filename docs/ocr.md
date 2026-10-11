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
| `local` (default) | classic detection + recognition on torch | **≤ 0.17.1** | in the isolated venv, on a GPU of this box |
| `model` | the **"Surya2"** VLM as a MODEL this install serves (`ocr.surya_model_id` → `models.json`) | **≥ 0.20** (pinned 0.22.1) | wherever that entry says: a **RunPod pod rented on demand**, another host via `service_url`, or the local vLLM backend. **The only served mode that needs no GPU here.** |
| `vllm` | the same Surya2 VLM, served from coderai's own isolated vLLM venv | ≥ 0.20 | this box, CUDA required |
| `llamacpp` (alias `server`) | the same Surya2 VLM | ≥ 0.20 | attaches to a server you already run, at `ocr.surya_server_url` |

Surya2 (≥0.20) is a VLM that CANNOT run standalone in the venv — it needs an OpenAI
server to attach to. In the served modes the venv is therefore light: it renders the page
and speaks HTTP, and every GPU second is spent on the other side.

### `model` mode: Surya on whatever serves the model

Surya is a *client* — layout, reading order, guided-JSON decoding and logprobs are its
own, which is why it drives a model server directly instead of going back through
coderai's `/v1` (one proxy hop would drop the guided decoding and the logprobs with it).
What it needs from coderai is therefore an address, and `model` mode is where it asks for
one in the vocabulary the rest of the server already speaks:

```jsonc
// models.json — the VLM is an ordinary model entry, served like any other
"datalab-to/surya-ocr-2": { "backend": "runpod", "engine": "coderai-vllm", … }
```
```jsonc
// config.json
"ocr": { "surya_serve": "model", "surya_model_id": "datalab-to/surya-ocr-2" }
```

A word on vocabulary, because this subsystem borrows one. Everywhere else in coderai an
**engine** is the thing that *runs* a model — vLLM, llama.cpp, ds4. In `ocr.default_engine`
the same word covers two different kinds of thing: paddle and docTR really are runners
(own weights, no server), while **surya-2 and olmOCR are pipelines over a model something
else runs**. Their engine is vLLM. That is why they alone need to be told where a server
is, and why the answer is the same question the rest of coderai already answers for every
model: local, a cluster node, or a rented pod.

| where the model is served | how `model` mode finds it |
|---|---|
| this box, coderai's own vLLM | the vLLM backend's instance for that entry |
| a cluster node | the node that lists the id in its own `/v1/models`, with the cluster token |
| RunPod | the pod pool for that entry, provisioning if it is cold |
| any other server | the entry's `service_url` |

coderai resolves that id through the backend that owns it — provisioning a pod if the
pool is cold — and hands the worker `SURYA_INFERENCE_URL`, `SURYA_MODEL_CHECKPOINT` (the
served name; Surya refuses a server whose `/v1/models` disagrees) and `VLLM_API_KEY` (pods
are bearer-locked). Renting, autoscaling, the spend ledger and the idle reaper stay with
the model layer; the OCR side holds only a lease. When a scale-to-zero pod is reaped
between documents, the next page finds the endpoint unhealthy, re-resolves once — which
rents a fresh pod — and is read rather than lost.

This is the mode for a **GPU-less orchestrator**: an install whose job is to rent cards
has no card of its own, so `vllm` mode there can only ever fail (`NO_VLLM_VENV`), and a
URL pasted into `llamacpp` mode stops existing fifteen minutes after the last page.

So: install **surya‑ocr 0.17.1** for `local`, or **surya‑ocr>=0.20** for
`model`/`vllm`/`llamacpp` — one per venv.
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
- **The share is a starting point, and the launcher measures the rest.** An engine's real
  footprint is not knowable up front. Measured live on a 24 GB 3090, surya-ocr-2 needs
  **~10.9 GiB before a single KV block** (weights 1.37, CUDA graphs 0.05, and ~9.4 GiB of
  *vision-encoder profiling* peak); `vlm_gpu_memory_utilization` is a fraction of the
  card's TOTAL memory, so the 0.35 this once shipped was 8.4 GiB — under the floor, unable
  to start whatever else was resident. Rather than trusting a constant, when vLLM reports
  that the budget cannot hold the cache the launcher **escalates the share, re-evicts for
  the larger figure, retries, and records what booted**, keyed by model + card + context +
  batch limits (`~/.coderai/vllm_gmu_learned.json`). A different model, a bigger card or
  different limits rediscover their own value; the recorded one is reused on later starts,
  and a configured share lower than what is known to work is ignored. It also sizes
  **down**: a successful boot reports the cache it got, and `footprint = budget - cache`
  is the only direct measurement of an engine's real cost there is, so a share with
  clear slack is recorded smaller for the next start (keeping ~2 GiB of cache, and
  never at or below a share already proven too small — that floor is what stops it
  oscillating). Over-provisioning therefore costs one boot, not the life of the
  process. Nothing is added back for other residents — vLLM charges only its own
  allocations against `total × gmu` — so the evict-first pass is what guarantees the
  room.

- **The batch limits shrink the footprint a little; the encoder is what dominates.**
  `ocr.vlm_max_num_batched_tokens` (default 4096), `ocr.vlm_max_num_seqs` (16) and
  `ocr.vlm_max_model_len` (0 = inherit) stop a VLM OCR engine inheriting the *LLM*
  backend's sizing, and chunked prefill means capping tokens-per-batch costs no usable
  context. Measured effect on surya-ocr-2, though: the footprint fell only 11.13 → 10.88
  GiB, **all of it from capturing fewer CUDA graphs** (0.30 → 0.05). The ~9.4 GiB peak did
  not move, because for this model vLLM sizes its encoder cache at 114,688 tokens and
  profiles it *with one video item at maximum feature size*. Capping the multimodal limits
  would collapse that — and would also stop the instance serving video at all, so it is
  not done by default. Set it yourself via `vllm.extra_args` if that instance only ever
  sees still pages.

If you see `Free memory on device cuda:0 … is less than desired GPU memory utilization`,
set **`ocr.vlm_gpu_memory_utilization`** (Settings → OCR → "VLM engines' GPU share"): the
OCR VLM gets its own share of the card instead of the one `vllm.gpu_memory_utilization`
asks for on behalf of an LLM — 0.35 is plenty for a 7B page model, where the LLM default
of 0.9 (or 0.6) can be more than the card has left. Otherwise: give the card less other
resident work, or serve the engine with `model`/`server` mode, which needs no instance of
its own at all.

### On a rented pod

A pod image carries docTR and Surya in its venv and PaddleOCR in a baked one
(`packaging/runpod/profiles/ocr.txt`, `ocr-paddle.*`). What it does **not** carry is a
model server: Surya-2 and olmOCR-2 are VLMs, and locally they ride coderai's own vLLM
engine. Baking that in would mean a second torch in the image — the vLLM venv alone is
~10.5 GB against a ~12 GB ceiling — so it is deliberately absent.

Those engines are still remotizable, through their **served** modes, which need nothing
but HTTP: set `ocr.olmocr_serve = "server"` with `olmocr_server_url`, or
`ocr.surya_serve = "llamacpp"` with `surya_server_url`, pointing at an endpoint the pod
can reach — a vLLM pod, for instance. The renting install seeds the mode, the URL and the
model ids to the pod, and pins `fallback_engines` to the engines the image actually has,
so a failed page is never handed to one that cannot run there. Ask for a VLM engine with
no server configured and the pod says so in its log and serves the request with docTR
instead of dying at inference.

Engine selection on a pod follows the usual `alias or path or id`, so an entry aliased
`olmocr` over an HF repo path is recognised.

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
`vlm_gpu_memory_utilization`, `vlm_max_num_batched_tokens`, `vlm_max_num_seqs`,
`vlm_max_model_len`, `fallback_engines`, `evict_drain_timeout_s`;
per-engine `*_enabled` /
`*_instances` / `*_use_gpu` / lang; `detect_mode` (+ `detect_model_path`, `detect_conf`);
`extract_enabled` / `extract_model_id` / `extract_schema` / `extract_validate`.

`*_use_gpu` is a **requirement, not a preference**: with it on and no CUDA device
present, paddle and docTR refuse to load (503, naming the setting) rather than
reading pages on the CPU many times slower with nothing in the response to say so.
Set it to `false` to run on the CPU deliberately — that is fully supported, it just
has to be asked for.

**Concurrency on one GPU** comes from loading multiple instances of an engine
(`paddle_instances` etc.) and fanning pages across them (bounded by `max_concurrency`).
OCR models are small, so a 24 GB card (e.g. RTX 3090) fits several instances — good
aggregate throughput without vLLM (a continuous-batching path is a future option; see
[vllm.md](vllm.md)).

## Endpoints

```
POST /v1/ocr            multipart: file=@doc.pdf [engine=] [dpi=] [detect=] [structured=] [schema=]
POST /v1/ocr/batch      multipart: files=@a.pdf files=@b.png [...same fields...]
GET  /v1/ocr/engines    what this install offers, and what each engine returns
GET/POST/DELETE /v1/ocr/schemas[/{name}]   manage extraction schemas
```

`engine=` takes an engine name (`paddle`, `doctr`, `surya`, `olmocr`) **or** the id
of the model a pipeline drives, so `engine=surya` and
`engine=datalab-to/surya-ocr-2` reach the same place. Omit it and
`ocr.default_engine` decides — which may itself be written either way.

`GET /v1/ocr/engines` is how a client finds that out instead of guessing: each row
carries `enabled` (with a `reason` when it is not), `kind` (`engine` owns its
weights, `pipeline` drives a model), `aliases` (the model ids that also resolve to
it), `instances`, and `returns` — `{boxes, regions, tables, meta}`, so a caller
needing coordinates can see that olmOCR has none. The top level reports `default`
(what the operator wrote) and `default_engine` (what it resolves to). Where an
engine's model runs is deliberately absent: it is configuration, it can change
between two requests, and a client that learned it would come to depend on it.

Response: `{engine, num_pages, text, pages[], stamps[], signatures[], structured}`. Each
page carries `lines[]` (text + bbox + conf), `regions[]` (layout), `tables[]`, and
`meta` when the engine reports page metadata (olmOCR does; the others do not). The
`ocr` **pipeline step type** chains OCR with `text_gen` in custom pipelines.

### Where `conf` comes from on a served engine

A VLM pipeline has no detector score to report, so surya asks the model server for
`logprobs` and averages `exp(logprob)` over the generated tokens: that mean token
probability becomes each line's (and each layout region's) `conf`.

This matters because of surya's fallback. When the server returns no logprobs,
surya does not report "unknown" — it uses a literal **1.0**. Every line of every
page then looks perfectly confident, including the guesses, and nothing downstream
can tell which page needs a human. coderai therefore forwards `logprobs` on the way
out and returns `choices[0].logprobs` on the way back, on both the buffered and the
streamed path, for the vLLM and RunPod proxy backends (the engines that run the
model and can measure it). Local in-process backends still answer `null`, and a
request that did not ask for logprobs is sent exactly as before — they are not free.

In `model` mode the hop that must do this is the **pod's own coderai**, not the
orchestrator: surya attaches straight to the endpoint the pool hands it. A pod runs
only the code baked into its image, so an install whose pods predate this needs its
capability image rebuilt before the numbers appear.

### Scaling in `model` mode

An OCR pipeline drives one server address itself — that is what keeps the page
images, the strict schema and the confidence scores intact. One address cannot
spread load, though, and three things used to follow from that:

* a RunPod pool grows when its least-loaded pod reaches
  `scale_up_inflight_per_pod` **in-flight** requests, and in-flight is counted
  when a request passes through the pool. An attached engine never did, so the
  pool saw no load however hard OCR ran — and the engine had one pod's URL anyway;
* several cluster nodes serving the model resolved to whichever answered first,
  for ever, however busy it became;
* a local model set to burst to RunPod never did: that trigger lives on the proxy
  path an attached engine bypasses.

With **`ocr.serve_fanout`** on (the default) the engine is instead handed one
address of coderai's own, and each request is forwarded — **verbatim**, body bytes
in and body bytes out, so nothing is decided about it — to whichever server is
least busy:

| the model is configured… | a page goes to |
|---|---|
| `backend: runpod` | the pool, every request: it picks the least-loaded pod and rents another when they are all at the threshold |
| to run on this machine | the local server, until it is busier than the model's own `scale_up_inflight_per_pod`, then a pod (the burst `runpod_spillover` describes) |
| pinned to cluster nodes (`engine`) | the least busy of them, counting what each node reports for its **whole** install, so a node loaded with other work is left alone |
| as several of these at once | its configured home first — an explicit `backend` or `engine` is a decision, not a hint — then the rest as overflow |

Nothing here is discovered. A cluster node serves a model because the model is
**pinned** to it, with the same `engine` pin the front router already treats as a
hard constraint (a comma-separated pin names several nodes, so OCR can spread over
them). Asking every node what it happens to hold, and using whoever answers first,
would put work on a machine nobody chose and move it when a different node replied
first.

How busy is too busy is the model's own `scale_up_inflight_per_pod`, on the model
page — the same number the pool scales on, not a second OCR-shaped copy of it. A
model with no RunPod block has nothing to rent, so load alone never moves its
pages. Turning `ocr.serve_fanout` off restores the single direct address (one pod,
no growth), which is one hop fewer and loses nothing on an install with exactly
one server.

**None of this is visible to the caller.** A client asks for OCR with a model it
has heard of — the id it sees in `/v1/models` — or with an engine name, and both
work: `engine=surya` and `engine=datalab-to/surya-ocr-2` reach the same place.
Where that model runs is read from its configuration, exactly as it is for every
other request coderai serves.

The address is a lease: a secret minted per attachment, carried in the URL (the
engine's `/health` and `/v1/models` probes send no headers, so it cannot be a
bearer alone) and valid for nothing but that one gateway path.

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
