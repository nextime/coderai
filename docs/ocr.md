# OCR subsystem (dedicated OCR engines)

CoderAI OCRs documents with **purpose-built OCR engines** (detection + recognition) —
faithful transcription with per-line bounding boxes and layout, GPU batching, no
hallucinated text — and, for the pages those read badly, two **document VLMs**. Five
engines are integrated and selectable per request:

| engine | id | license | how it runs |
|---|---|---|---|
| [docTR](https://github.com/mindee/doctr) | `doctr` | Apache-2.0 | **in-process** (uses the main venv's torch); works on GPU. Recommended on new-CUDA boxes. |
| [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) + PP-Structure | `paddle` | Apache-2.0 | **isolated venv subprocess** (layout + tables) |
| [Surya](https://github.com/VikParuchuri/surya) | `surya` | GPL (compatible with coderai's GPLv3) | **isolated venv subprocess**; best layout/reading-order; opt-in |
| [olmOCR-2](https://huggingface.co/allenai/olmOCR-2-7B-1025) | `olmocr` | Apache-2.0 | **a document VLM**, served by a coderai model, coderai's vLLM, or any OpenAI endpoint. **No bounding boxes.** |
| [Chandra](https://huggingface.co/datalab-to/chandra-ocr-2) | `chandra` | code Apache-2.0, **weights modified OpenRAIL-M** (research/personal/under-$2M, no competitive use) | **a document VLM that returns layout** — boxes, labelled regions and tables in one pass. Served like olmOCR; opt-in. |

Per document you get three outputs: **plain text**, **structured JSON fields**, and
**stamp/signature flags**.

**Which one?** `paddle`/`doctr`/`surya` give you boxes, layout regions and tables, which
is what you want when something downstream has to point at a place on the page (redaction,
stamp detection, a table's cells). `olmocr` gives you the best *reading* of a hard page —
old scans, maths, multi-column, dense tables — and nothing positional. `chandra` is the
one that does both: it is a VLM, so it reads a hard page like olmOCR, but it is *asked*
for its layout, so it answers with one block per region carrying a box and a label. It is
also the only engine here whose licence has a revenue ceiling, which is the reason it is
not simply the default. Stamp/signature detection in `layout` mode has nothing to work
with under `olmocr`; it works under `chandra` (the regions are real) and under the three
detector-based engines, and `detector` mode (YOLO) works everywhere because it looks at
the image, not the engine's output.

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

## Chandra (`chandra`) — the document VLM that *also* gives you boxes

[Chandra](https://huggingface.co/datalab-to/chandra-ocr-2) is Datalab's document VLM — the
same lab as Surya, and a Qwen3-VL fine-tune. Like olmOCR it reads a whole page in one
call; unlike olmOCR it is **asked for its layout**, and that is the point of having it.

Its `ocr_layout` prompt tells the model to answer as one `<div>` per layout block carrying
`data-bbox="x0 y0 x1 y1"` (normalised 0–1000) and `data-label` from a fixed set — `Text`,
`Section-Header`, `Table`, `Figure`, `Form`, `Caption`, `Footnote`, `Equation-Block`,
`Code-Block`, `Chemical-Block`, `Page-Header`, `Page-Footer`, `Table-Of-Contents`,
`Diagram`, `List-Group`, `Complex-Block`, `Blank-Page`. coderai turns each block into all
three things a page carries: a **line** (its text), a **region** (its label and box) and,
for a table, a **table entry keeping the HTML** — because a table flattened to text has
lost the one thing that made it a table. Maths comes back as `<math>` KaTeX and chemistry
as `<chem>` SMILES, and both survive into the text as their source: unrendered LaTeX beats
a formula silently deleted.

The prompts are mirrored **verbatim** from `chandra/prompts.py`, down to the Python repr of
the allowed-tag list, because that wording is what the model was trained on. Nothing of
chandra's own package is installed: the engine is PIL plus an HTTP call, and the reply is
split with stdlib `html.parser` rather than pulling bs4 and markdownify into the main venv
for a tag split.

### Three reasons to reach for it

1. **You need boxes from a VLM.** `olmocr` has none at all; `chandra` has them for every
   block in one pass, with no separate detector.
2. **It is a different model family from Surya-2.** Surya-2 has its own 65k `WordLevel`
   vocabulary; Chandra has Qwen3-VL's tokenizer. So it is the useful *second opinion* on a
   page Surya reads badly — including a page whose accents come back as the
   unreadable-glyph marker (see below). Running the same page through both separates "this
   document's pixels are too few" from "this one model cannot read this glyph".
3. **90+ languages**, with no language input to set — `lang=` is accepted and does not
   steer it, same as surya-2 and olmOCR (`language_aware: false` in
   `/v1/ocr/engines`).

### Licence — read this before enabling it

The code is Apache-2.0 but the **weights are a modified OpenRAIL-M**: free for research,
personal use and companies under **$2M** funding/revenue, and **not for use competing with
Datalab's own API**. Commercial self-hosting beyond that needs a licence from Datalab.

That is a **narrower grant than Surya's GPL**, so the engine is gated the same way and the
gate behaves differently in one place that matters: asking a rented pod for `surya` by name
*is* the GPL acceptance (the deployment's own decision travelling to the pod), but
`chandra_accept_license` is only ever **inherited from a decision already recorded in this
install's config**. A pod cannot accept a licence on your behalf.

```yaml
ocr:
  chandra_enabled: true
  chandra_accept_license: true      # nothing loads until this is set deliberately
  chandra_serve: model
  chandra_model_id: chandra         # a vision model in models.json
```

### Serving it

The same three modes as olmOCR, and for the same reasons:

| mode | what it means | needs |
|---|---|---|
| `model` (default) | the checkpoint is a **vision model in your models.json** and Chandra goes through coderai's own model manager — VRAM accounting, eviction, the cluster/RunPod dispatcher. The only mode that works on a box with **no GPU of its own**, which is also the box whose whole job is renting one. | `chandra_model_id` |
| `vllm` | coderai's vLLM backend serves `chandra_model` on its own instance (continuous batching). CUDA only. | the vLLM backend configured |
| `server` | attach to an OpenAI-compatible server already running it. | `chandra_server_url` (+ `chandra_api_key` if it wants one) |

In `model` mode the request goes through `pod_http`, so a RunPod pod reached over direct
TCP — https to a bare IP, certificate signed by **this install's own CA** — is verified
against that CA. Plain `requests` fails it twice over with `CERTIFICATE_VERIFY_FAILED`,
which is exactly how olmOCR came to be unreachable before 0.3.22.

### DPI: the ceiling is Surya's, to the pixel

`chandra/model/util.py`'s `scale_to_fit` caps the input at **3072×2048 = 6,291,456
pixels** — the same numbers as surya-2 — and snaps dimensions to a **28-pixel grid**,
because a Qwen-family vision tower tokenises in 28×28 patches. An A4 page reaches that
ceiling at about **255 DPI**, so `chandra_dpi` defaults to **250** for precisely the reason
`surya_dpi` does, and rasterising higher costs time and memory for no extra pixels at the
model. coderai does the scaling itself (same algorithm, same grid) and reports the size the
model actually saw as the page's `width`/`height`, with the size you rasterised in
`meta.source_size` when they differ — the boxes are normalised against what the model saw,
and de-normalising them against the pre-scaling size would put every box on a scaled page
in the wrong place.

### Repetition loops

A VLM that loses its place on a page does not fail. It repeats the last thing it saw until
it runs out of tokens, and the server answers **200** with a page that looks plausible.
Chandra's own client treats that as retryable and so does coderai: `chandra_max_retries`
(default 2) re-asks when the reply ends in a repetition, raising the temperature each time,
because a greedy decode falls into the same loop twice. If it still loops after the last
retry the page is returned anyway with `repeat_detected: true` in its `meta` — a looping
transcription usually has a correct beginning, and the flag is what lets you decide instead
of being handed a silent mess. `meta.retries` records how many re-asks it took.

### Fields

`chandra_enabled`, `chandra_accept_license`, `chandra_serve`, `chandra_model_id`,
`chandra_model` (default `datalab-to/chandra-ocr-2`), `chandra_server_url`,
`chandra_api_key`, `chandra_instances` (pages in flight), `chandra_dpi`, `chandra_layout`
(off asks for plain HTML — faster, and then there are no boxes or regions at all),
`chandra_include_furniture` (keep `Page-Header`/`Page-Footer` in the text; on, because a
judicial document's header carries the court and the case number), `chandra_max_tokens`
(12384 — HTML runs far longer than the text it carries), `chandra_temperature` (0, greedy),
`chandra_top_p` (0.1), `chandra_max_retries`, `chandra_timeout`.

```bash
curl -F file=@sentenza.pdf -F engine=chandra http://localhost:8776/v1/ocr
```

**A pod cannot serve it either**, for the same reason as Surya-2 and olmOCR: the RunPod OCR
images carry no model server, and baking one in would put a second torch in a ~12 GB image.
A pod asked for `chandra` serves the request with docTR and says so in the log. Point
`chandra_serve: "server"` at something the pod can reach — including a vLLM pod — if you
want the real thing out there.

## When an engine's server will not start

The VLM engines in `vllm` mode (`surya`, `olmocr`, `chandra`) boot a vLLM instance
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

### olmOCR reaches a pod directly, so it needs the pod's CA itself

olmOCR failed against a rented pod with `CERTIFICATE_VERIFY_FAILED`. A pod on the
direct-TCP path serves HTTPS with a certificate signed by **this install's own CA**
(`codai/api/pod_tls.py`), at a public IP that was not known when the certificate was
issued — so stock `requests` fails it twice over.

Surya never hit this because surya attaches through the fan-out gateway, which
learned to trust pod certificates in 0.3.16. olmOCR talks to the upstream the
dispatcher chose **directly**, so it needed the same trust of its own, and the gap
only surfaced when olmOCR was asked to cover for surya. It now goes through
`codai/api/pod_http.py`, which verifies *https-to-a-bare-IP* against our CA with the
hostname check off and hands every other URL — localhost, a named server, another
coderai — to `requests` untouched.

The lesson generalises: anything that may be handed a **pod** address uses
`pod_http`, not `requests`. A localhost worker is unaffected either way, since
`pod_http` passes non-pod URLs straight through.

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

### A closed keep-alive socket is not a dead pod

A pod dropped from the pool is **terminated** — it is a rented GPU, and the bill stops
when it does. So what counts as "nothing is at that address" has to be narrow, and for a
while it was not.

`httpx.RemoteProtocolError("Server disconnected without sending a response")` was in the
dead-upstream set. That error does not mean the server is gone: it is what httpx raises
when a **pooled** connection was closed by the far end between two requests. uvicorn
closes an idle connection after 5s by default and nothing in coderai raised that, while
the OCR gateway held pooled connections for 60s — so the first page sent after any quiet
gap hit a socket the pod had already closed, and the pool killed the pod for it.

On 2026-10-12 that cost five A40s between 19:14 and 20:11. None of them was broken: each
pod's own log ended in `"POST /v1/chat/completions HTTP/1.1" 200 OK` at 4664 tokens/s,
seconds before it was reaped, and every replacement needed a ~6-minute image pull before
it could read a page. From outside it looked like OCR flapping; it was coderai killing
its own workers.

Three things changed, all on the orchestrator (no pod image is involved):

* a closed pooled connection is **retried on a fresh connection**, against the *same*
  upstream — re-choosing would send the page to a different pod, or rent one;
* a pod is **asked before it is dropped**: `/healthz` (then `/health`) on a new
  connection, with any answer below 500 counting as alive — a 401 is a server,
  authenticating. Only silence drops it;
* idle pooled connections are now released after 4s (`CODERAI_OCR_KEEPALIVE_EXPIRY_S`),
  under the server's own keep-alive, so the race is rare rather than routine. Reuse
  *within* a burst of pages — the reason the pool exists — is unaffected.

A pod that RunPod really has reclaimed mid-document is still dropped immediately, which
is what this logic was added for: left in the pool, every following page picks the corpse
and the in-flight count climbs with nothing draining.

## Enable & configure

Settings → **OCR** card, or `config.json` `"ocr"`. Key fields: `enabled`,
`default_engine`, `dpi`, `surya_dpi`, `chandra_dpi`, `max_concurrency`, `lang`,
`paddle_auto_lang`, `build_retry_cooldown_s`,
`vlm_gpu_memory_utilization`, `vlm_max_num_batched_tokens`, `vlm_max_num_seqs`,
`vlm_max_model_len`, `fallback_engines`, `evict_drain_timeout_s`;
per-engine `*_enabled` /
`*_instances` / `*_use_gpu` / lang; `detect_mode` (+ `detect_model_path`, `detect_conf`);
`extract_enabled` / `extract_model_id` / `extract_schema` / `extract_validate`.

`*_use_gpu` is a **requirement, not a preference**: with it on and no CUDA device
present, paddle, docTR and Surya's `local` mode refuse to load (503, naming the
setting) rather than reading pages on the CPU many times slower with nothing in
the response to say so. Set it to `false` to run on the CPU deliberately — that is
fully supported, it just has to be asked for.

Surya's `local` mode in particular has to be **asked for**. `surya_serve` defaults
to `model`, not `local`: an install that said nothing used to get the one mode that
runs the model on this machine — which on a GPU-less orchestrator meant a silent
CPU run of a VLM pipeline, i.e. minutes a page while looking healthy. `model`
adapts to wherever the checkpoint is actually served and tells you to set
`surya_model_id` when it has not been told.

**Concurrency**, measured on one A40 (same page, same install):

| N in flight | surya req/s | surya p50 | chandra pages/s | chandra p50 |
|---|---|---|---|---|
| 1 | 0.93 | 1.1s | 0.21 | 4.6s |
| 8 | 3.79 | 2.0s | 1.31 | 6.1s |
| **16** | **7.11** | **2.2s** | **1.88** | **8.4s** |
| 24 | 5.89 | 4.1s | 2.09 | 10.6s |
| 32 | 5.44 + 13 *rejected* | 3.5s | 1.44 + 21 *rejected* | 7.5s |
| 48 | **all 48 rejected** | — | — | — |

> **The failures in the last two rows were not the GPU.** They were HTTP 429 from
> coderai's own rate limiter — `/v1/chat/completions` was capped at 60 requests per
> 60 seconds per client IP, as a module constant with no config. These ladders ran
> every level inside one window, so the rejections begin exactly where the
> cumulative count crosses 60: surya 16+24=40 then 20 of 32 pass and 12 are
> refused (13 measured); chandra and olmOCR 8+16+24=48 then 12 pass and 20 are
> refused (21 measured, twice). Re-running one level alone returns 32/32 OK.
>
> So the reading once given here — "both peak at 16 and both fail outright by 32,
> a cliff" — was wrong: the engines never fell over, and the true knee is **above**
> what these numbers show. A pod now ships with the limiter off
> (`CODERAI_SERVER_RATE_LIMIT_ENABLED=0`), since a pod has exactly one client — the
> install that rented it — and its admission is governed by
> `pod_max_parallel_requests` and `pod_queue_max_size` instead. **Any ladder like
> this must either keep each level under the limit or pace itself across windows.**

`runpod.scale_up_inflight_per_pod` is 16 for all three engines. That remains a
defensible spill point — latency is already climbing at 16 on these cards — but it
is a *chosen* setting, not the measured edge of the hardware.

Three settings bound this and they are easy to confuse:

* `runpod.scale_up_inflight_per_pod` (**per model**, in its models.json entry) — when
  the pool rents *another* pod. Read per model by `placement.spill_threshold`.
* `ocr.<engine>_instances` — how many pages that engine can have in flight. **This is
  the real limiter**, and for the subprocess engines it is what the box can host: a
  surya worker is a separate interpreter at ~750 MB, so 64 of them is ~45 GB.
  chandra and olmOCR run in-process and cost a thread.
* `ocr.max_concurrency` — a global **ceiling** across every engine, not a throughput
  knob. It shipped at 4, which made it the binding limit everywhere; it is 64 now.
  Each admitted page holds a rasterised bitmap (~18 MB for A4 at 250 DPI), so the
  ceiling is finite on purpose.

A note on the surya figures: surya issues one request per *layout block*, so its
"req/s" is blocks, not pages — a real page is several requests and its page rate is
correspondingly lower. The subprocess protocol is also synchronous (one `cmd: ocr`,
one reply), so a worker handles one page at a time and concurrency is the process
count. In the served modes that worker only scales an image and makes an HTTP call,
so this is the thing to change if surya ever needs fleet-scale concurrency.
OCR models are small, so a 24 GB card (e.g. RTX 3090) fits several instances — good
aggregate throughput without vLLM (a continuous-batching path is a future option; see
[vllm.md](vllm.md)).

## Endpoints

```
POST /v1/ocr            multipart: file=@doc.pdf [engine=] [lang=] [dpi=] [detect=] [structured=] [schema=]
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

### `lang=` — and why the default is auto-recognition

A client names the document's language; omitting it (or sending `lang=auto`) means
**let the engine recognise what it finds**. The default is deliberately not a
language: a server that has not been told what it is reading should not assume one.

The parameter does not mean the same thing to every engine, and the response says
which it was:

| engine | `lang=` | why |
|---|---|---|
| `paddle` | **steers it** | loads a language-specific recognition model |
| `doctr` | accepted, no effect | recognition is Latin-script-wide |
| `surya` | accepted, no effect | VLM; its whole prompt is `"OCR this block image to HTML."` |
| `olmocr` | accepted, no effect | VLM; fixed prompt |
| `chandra` | accepted, no effect | VLM; fixed prompt (90+ languages, no language input) |

So every response carries both the language applied and whether the engine that
served could act on it — silently ignoring `lang` would leave a caller believing it
steered something:

```json
{ "engine": "surya", "lang": "it", "lang_applied": false, ... }
```

`/v1/ocr/engines` reports the same per engine (`language_aware`), so a client can
choose before sending.

**Precedence** is request > `ocr.<engine>_lang` > `ocr.lang` > auto. The request
beats the engine-specific setting on purpose: that setting is an install's default,
while a caller naming a language knows something about the document in hand. Asking
for `auto` explicitly is *not* the same as sending nothing — an install configured
for Italian can still be asked to work it out.

**PaddleOCR is the exception that shapes the design.** It has no auto mode and its
options are sent once at load (`{"cmd": "load", "opts": ...}`), so an instance is
built for one language and cannot be re-languaged. Two consequences:

* `auto` becomes `ocr.paddle_auto_lang` (default `latin`). That is a *script family*,
  not detection — it covers Italian, French, Spanish, German, Portuguese and the rest
  of the Latin alphabet, and most ISO codes PaddleOCR accepts resolve to it anyway.
  PaddleOCR's own default is `ch`, a worse guess for most documents than no guess.
  Another script needs naming: `lang=arabic|japan|cyrillic|devanagari`.
* paddle gets **one pool per language**, keyed on the model it is built with (so
  `auto` and `latin` share a pool rather than loading it twice). The multilingual
  engines keep a single pool whatever is asked, which is not tidiness: a
  language-keyed pool is a second set of loaded instances holding VRAM, and a
  production install runs 24 surya instances.

Note that `ocr.surya_langs` is a **legacy** setting. It reaches only the ≤0.17
recognition API (`rec(images, langs, det_predictor)`); surya-2 has no language input,
so on the pod path it does nothing.

### `surya_token_trace` — whose fault is an unreadable character?

"Surya's path" is two things, and only one of them is ours:

1. the **model** emits a broken byte sequence — then the text faithfully renders a wrong
   answer and nothing on our side can fix it;
2. the model emits a **correct** sequence and the serving stack mis-assembles it — then
   the model was right and the bug is ours.

The finished text cannot tell them apart, for the reason above. The *bytes* can, so
`ocr.surya_token_trace` surfaces them. It is **off by default**: it is a diagnostic, it
adds response weight, and it reaches into surya's internals.

It has to, because surya drops the data on the floor. Its recognition predictor really
does pass the per-token logprobs on — `BlockOCRResult(..., raw_logprobs=out.logprobs)` —
but the model never declares that field and pydantic's default is `extra="ignore"`, so
the value is constructed and silently discarded (their own comment calls it "phase 2
use"). Verified by construction: the attribute does not exist on the result. coderai
therefore takes it one layer earlier, wrapping `_generate_one`, which still has it. The
wrapper only reads, and swallows every error — a diagnostic that could break a page would
be far worse than no diagnostic.

With it on, a document that lost characters carries its own diagnosis under
`unreadable`, with a one-word `blame` for the impatient:

```json
{ "unreadable": {
    "count": 29,
    "blame": "the model",
    "trace": [
      { "page": 0, "calls": 7, "with_logprobs": 7, "conclusive": true,
        "kinds": {"invalid-utf8": 29},
        "why": "the model's own byte stream is broken at the places listed: ...",
        "findings": [
          { "call": 3, "index": 41, "kind": "invalid-utf8",
            "token":   {"token": "Ã", "bytes": [195], "logprob": -0.0021},
            "why":     "byte 0xC3 does not form valid UTF-8 here",
            "context": [ {"token": "p", "bytes": [112]}, {"token": "i", "bytes": [105]},
                         {"token": "Ã", "bytes": [195]}, {"token": " ", "bytes": [32]} ] }
        ] } ] } }
```

How to read it:

| `conclusive` | findings | verdict (`blame`) |
|---|---|---|
| `true` | `kind: "invalid-utf8"` — a lone `[195]` with nothing to complete it | **the model** emitted a broken sequence — report it to Datalab |
| `true` | `kind: "model-emitted"` — the token *is* the replacement character | **the model** chose the marker, or something upstream substituted it before we saw a byte |
| `true` | **none**, on a page full of markers | **the serving path** — the bytes were sound and the loss happened after them, so it is ours |
| `false` | — | **unknown**. `with_logprobs: 0` means the upstream returned none, so there was nothing to read |

**That last row is the point of the envelope.** An empty finding list has two
incompatible meanings — "the bytes were sound" and "there were no bytes to look at" —
and reporting a bare list would make the second read as the first: a confident verdict
drawn from missing data. So the trace always says how many calls it saw, how many
carried logprobs, and whether it is `conclusive`. surya asks for logprobs by default
(`SURYA_INFERENCE_LOGPROBS = True`) and coderai's gateway forwards the body verbatim
precisely so they survive, but an upstream that declines them is a configuration rather
than a bug, and has to be visible instead of silently inverted into a conclusion.

The scan reassembles
the whole byte stream and walks it as UTF-8 rather than judging tokens one at a time: per
token is meaningless here, since one character legitimately arrives as two or three
tokens and each is "invalid" alone. The first version of this did judge them pairwise and
duly flagged every accent *and* every em dash — exactly the false positive that would
make the diagnostic worthless.

Findings are capped (`TRACE_MAX`, 40 per page) with two tokens of context each side,
because a lead byte is only wrong in the light of what followed it.

A third, independent discriminator needs no code at all: serve the **same weights through
llama.cpp** (`datalab-to/surya-ocr-2-gguf`, with `surya_serve: "llamacpp"`). A different
detokenizer on identical weights — accents surviving there would convict vLLM outright.

### The verdict — measured, and it corrects an earlier one here

**`surya_repair_accents` cannot work on vLLM 0.27.1, and an earlier version of this
section claimed a verdict it had not earned.** Both facts come from the same
measurement, taken through the live gateway on 2026-10-12:

```
result.raw (what surya reads) : 3 markers   'nt ? propriet? della propriet? d'
reassembled from logprobs     : 0 markers   'nt  propriet della propriet ...'
entries with bytes=[]         : 257
any byte >= 0x80 in stream    : False
```

vLLM reports **`bytes: []`** for the byte-level tokens of surya-ocr-2's custom
vocabulary, and **not one byte ≥ 0x80 appears anywhere in the stream**. The accent
bytes are not corrupted in the logprobs — they are *absent* from them. So:

* **the repair has nothing to rebuild from.** It is not a tuning problem; the data
  does not exist. Its guards do refuse correctly (the ASCII-skeleton check catches
  the mismatch), so a page comes back with its markers rather than with the accents
  silently deleted — but it cannot fix anything on this stack.
* **the trace's first verdict was vacuous.** `_reassemble` accepted an empty byte
  list as data, built a string with the accents simply missing, found no invalid
  UTF-8 in it, and reported `conclusive: true — the byte stream decoded cleanly, the
  fault is in the serving path`. It was reading absence as cleanliness: the same
  failure the envelope was built to prevent for *missing logprobs*, one level
  further down. It now counts `tokens_without_bytes` and reports
  `conclusive: false` with `blame: "unknown"` when the server declines to give
  bytes.

What the measurement *does* establish: the loss is **inside the serving stack**,
which produced neither usable bytes nor usable text. It cannot say whether the
model's own output was sound, and the honest trace now says exactly that.

So on vLLM 0.27.1 the options are **`chandra`** (which reads the same pixels
perfectly) or **`coderai-vllm-ocr`** below.

### The repair, and why it is still shipped (`surya_repair_accents`)

The trace was run. On a page of crisp synthetic Italian, served through the Aruba
install on vLLM 0.27.1:

```
unreadable: count=29  blame='the serving path'
page 0: calls=1  with_logprobs=1  conclusive=True  kinds=None
   why: the byte stream decoded cleanly, so the characters were lost AFTER the
        model: the fault is in the serving path, not the model
```

**surya-ocr-2's own byte stream was valid UTF-8**, with every line reported at
`conf: 0.9605`. The model read the page correctly and something after it destroyed
the accents. Note what that does *not* implicate: surya-ocr-2 and chandra-ocr-2 are
the **same architecture** (`Qwen3_5ForConditionalGeneration`) on the same vLLM, and
chandra's accents survive — so this is not vLLM mishandling `qwen3_5`, it is vLLM
mishandling **surya's custom 65,425-entry byte-level vocabulary**, where one accented
letter is two tokens and either byte alone decodes to a marker. It was also a single
model call, so concurrency is not involved.

Since the correct bytes arrive in the same response's logprobs,
`ocr.surya_repair_accents` (**on by default**) rebuilds the reply from them. It is
applied inside the backend wrapper *before* surya parses the HTML, so the blocks, the
boxes, the plain text and the confidences downstream are all built from correct
characters — repairing further down would mean fixing the same text in several places
and missing one.

Rewriting a model's output is a serious thing to do, so it is guarded three ways and
refuses unless all three hold:

1. the reassembled byte stream decodes **strictly** as UTF-8;
2. the result contains **no marker of its own**;
3. its **ASCII skeleton is identical** to the original's.

The third is what makes this a repair rather than a substitution: it can only ever
change non-ASCII characters. A truncation, a different block or a shifted offset all
change the skeleton and the original is kept, markers and all. Nothing happens when
the text is clean — not even a reassembly attempt — and a page that was repaired says
so:

```json
{ "accents_repaired": { "blocks": 1, "recovered": 29,
                        "how": "rebuilt from the model's own per-token bytes, …" } }
```

Set it false to see exactly what the server sent.

**The other half: `coderai-vllm-ocr`.** A pod image pinned to **vLLM 0.20.1** — the
version `surya-ocr` itself spawns (`surya/settings.py:96`) — so the model can be served
on the stack its authors validate, the repair can be checked against a reference rather
than trusted, and an upstream bug report has a known-good comparison. Select it per
model with `runpod.engine: "coderai-vllm-ocr"`; build it with
`./packaging/runpod/build_capability_image.sh vllm-ocr`. It is **not** a drop-in for
`coderai-vllm`: 0.20.1 is seven minor releases behind, and anything else on that image
is pinned back with it, so rent it for the OCR model only.

### What `conf` means, per engine

Every engine now reports a **measured** confidence, and `/v1/ocr/engines` says what
one number covers before you send anything:

| engine | scope | source |
|---|---|---|
| `paddle` | line | recogniser score |
| `doctr` | line | recogniser score |
| `surya` | **block** | mean token probability |
| `chandra` | **page** | mean token probability |
| `olmocr` | **page** | mean token probability |

The scope is not cosmetic. A per-line score can single out the line to re-read; a
page-level one can only tell you the page went badly. surya calls the model once per
layout block, so its figure is genuinely per-block; chandra and olmOCR generate a
whole page in one call, so theirs is page-level — chandra puts that same number on
every line and region (which is what `unreadable`'s per-spot `conf` reads) and says
`meta.confidence_scope: "page"` so it is not mistaken for per-line precision.

All three VLM figures are the mean of `exp(logprob)` over the generation's tokens —
the same computation surya's own client does, so they are comparable with each
other rather than three different notions of "confidence". It costs one float per
token; `ocr.chandra_logprobs` and `ocr.olmocr_logprobs` turn the request off.

**A server may decline to report logprobs**, and then `conf` is `0.0` — which must
not be read as "the model was hopeless". `meta.confidence_measured` distinguishes
the two. That is the same distinction the unreadable-glyph trace needs and twice
got wrong (see the correction above): a zero that means *not measured* and a zero
that means *measured as bad* are different facts, and code that conflates them
produces confident nonsense.

(chandra used to hard-code `conf=0.0`, with a comment arguing that the model
reported none and inventing one would be read as a measurement. The first half was
right; the conclusion was wrong, because the number was obtainable all along by the
route surya already used.)

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

### Unreadable glyphs, and why Italian accents were the first to go

Accented letters came back from surya as `U+FFFD` — `più` as `pi?`, `facoltà` as
`facolt?` — nine of them in one page of raw JSON, as real bytes `EF BF BD`. That
looks exactly like a decoding bug, and it is not one.

`U+FFFD` is **a real entry in surya-ocr-2's vocabulary**: id 65421 of 65425, the
only id that decodes to it, with `<unk>` a separate token at 65424. The tokenizer
is `WordLevel` and character-level for Latin text, so `à` is a *single* token
(2160) and an accent cannot be split across two — no partial-UTF-8 artefact is
possible, and per-token decoding is byte-identical to decoding the whole string.
Surya reads its text only from `message.content`, never from the per-token
logprobs, and coderai's fan-out gateway forwards the upstream body as raw bytes.

So the model **predicted** "there is a character here I cannot identify", and
every hop delivered that faithfully.

The decisive detail is the **count**. A charset mis-decode produces *two* markers
per accented letter, because `à` is two UTF-8 bytes and each undecodable byte
becomes its own U+FFFD — `pi��`. What arrives is *one*: `pi�`. One marker is one
token. Every other hop was checked and is byte-faithful too: `_html_to_text` is
regex plus `html.unescape` and survives both literal and entity forms; the
isolated-venv worker returns `json.dumps` with the default `ensure_ascii=True`,
so what crosses the pipe is pure-ASCII `\uXXXX` escapes and the locale cannot
matter; the gateway forwards raw bytes. There is no charset conversion anywhere
in the path that could do this.

Two things follow.

**It is a resolution problem.** Surya caps its own input at 3072×2048 =
6,291,456 pixels in `scale_to_fit`, which is called with **defaults only** — no
surya setting raises it, `IMAGE_DPI_HIGHRES` included. Measured, for A4:

| `dpi` | rasterised | Mpixels | model sees | Mpixels |
|------:|-----------:|--------:|-----------:|--------:|
| 200 | 1654×2338 | 3.87 | 1652×2352 | 3.89 |
| 250 | 2067×2922 | 6.04 | 2072×2912 | 6.03 |
| 255 | 2108×2980 | 6.28 | 2100×2968 | 6.23 |
| 300 | 2481×3507 | 8.70 | 2100×2968 | 6.23 |
| 400 | 3308×4676 | 15.47 | 2100×2968 | 6.23 |

The shared `ocr.dpi = 200` was therefore handing the model 62% of the pixels it
accepts, and a diacritic sits above the x-height at ~3 pixels tall. `ocr.surya_dpi`
(default **250**) is surya's own rasterisation DPI for this reason; paddle and
docTR have their own optima and keep `ocr.dpi`. Above ~255 nothing more reaches
the model, though supersampling down from 300–400 can still render glyphs more
cleanly, at the cost of rasterisation time and memory.

**Raising DPI is worth doing on its own merits, but it has not been shown to fix
U+FFFD.** A production document came back with the same markers at 200 and at
300, which is evidence against too-few-pixels being the whole story — a
perception failure would be stochastic, not every accent every time. Note what
rasterisation DPI can and cannot do: if the PDF *contains a scanned image*, the
page is re-sampled from that image, so asking for 300 DPI upscales a 150-DPI scan
and adds no detail that was never captured. `pdfimages -list doc.pdf` reports the
embedded resolution and is the first thing to check. The remaining candidates are
the document's own glyph quality and the serving stack (coderai's pods run vLLM
0.27.1, while surya's own `VLLM_DOCKER_IMAGE` pins `v0.20.1`); the way to tell
them apart is to OCR **perfect synthetic Italian** through the same path — clean
output indicts the document, U+FFFD on crisp vector glyphs indicts the stack.

**Correction, measured 2026-10-12.** An earlier version of this section argued that
U+FFFD could not be a byte-level artefact, because `à` was a single token and one marker
therefore meant the model had chosen it. **That was wrong.** surya-ocr-2's vocabulary is
**byte-level**, and an accented letter is two tokens:

```
'à' utf8=[195, 160]  ids=[2131, 2258]  pieces=['Ã', 'ł']
'è' utf8=[195, 168]  ids=[2131, 2104]  pieces=['Ã', '¨']

decode([2131])       = '?'     <- a lone lead byte gives exactly ONE marker
decode([2131, 2104]) = 'è'
```

So **one marker per accented letter is exactly the signature of a byte-level failure** —
the thing it was claimed to rule out. The vocabulary lookup was misread too: `vocab['à']
= 2160` does not mean the letter is token 2160; in a byte-mapped vocabulary that key is
the glyph standing for byte `0xA0`, and `decode([2160])` is a single marker as well. The
"one marker vs two" discriminator does not exist.

There is a second, cheaper discriminator that does: **run the page through `chandra`**.
It is a different model family with a different tokenizer (Qwen3-VL, against
surya-2's own 65k `WordLevel` vocabulary), reached over the same transport, at the
same pixel ceiling, from the same rasterised page. Accents intact under chandra
and lost under surya is as close to a controlled experiment as this gets, and it
points at the model rather than the document. Both losing them points back at the
pixels — and then `pdfimages -list` is the next thing to read.

`dpi` only affects **PDF** input. An uploaded image is used at the resolution it
arrives, so a low-resolution scan cannot be rescued by this setting.

**It must not be silent.** `pi?` indexes, searches and reads wrong for ever, and
for judicial documents a known unknown beats a quiet corruption. Every result now
counts its unreadable glyphs, and the key is **omitted entirely when there are
none**, so a clean response is byte-for-byte what it always was:

```json
{ "text": "...", "pages": [ { "index": 3, "unreadable": 2,
    "lines": [ { "text": "la pi\ufffd alta", "conf": 0.41, "unreadable": 1 } ] } ],
  "unreadable": {
    "count": 2,
    "pages": [ { "page": 3, "count": 2 } ],
    "spots": [ { "page": 3, "line": 0, "offset": 5, "conf": 0.41,
                 "bbox": [...], "context": "la pi\ufffd alta facolt" } ] } }
```

The `conf` on a spot is the **line's** — the finest granularity that exists, since
surya reports one mean token probability per block, not per character. It is also
the useful signal: a low-confidence line with an unreadable glyph is a line to
re-read at a higher DPI, while a high-confidence one is more likely a glyph that is
genuinely not in the model's repertoire. `spots` is capped (100) with a
`truncated` count, and the text itself is never rewritten — stripping the marker
would destroy the only record of where the loss is.

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

#### `*_instances` is a local worker count, not GPU concurrency

The two numbers look interchangeable on the OCR card and are not:

| setting | what it bounds | where it costs |
|---|---|---|
| `runpod.scale_up_inflight_per_pod`, `runpod.max_pods` | concurrent pages **per rented GPU**, and how many GPUs | the pod |
| `ocr.surya_instances` (also `paddle`, `doctr`) | **worker processes on this machine** | ~600 MB RSS and a core's rendering, each |
| `ocr.surya_lanes` | pages in flight **per worker process** (served modes) | almost nothing — see below |
| `ocr.chandra_instances`, `ocr.olmocr_instances` | concurrency slots, in-process | a slot |

Only surya, paddle and docTR spawn a process per instance — they run in their own
venv, driven over a pipe. `surya_instances: 64` was set on Aruba as though it were
the GPU-side figure, and on 2026-10-12 it did exactly what it says: **64 worker
processes, 37 GB of RSS, swap exhausted, load 25 on 16 cores**, on a host also
running some thirty other containers. A batch then behaved as the operator
reported — one flawless 19-second burst of 22 documents, then **no completions at
all for nine minutes** — while the rented A40 sat at 2% KV-cache usage with
`Waiting: 0`. The GPU was never the bottleneck. The host was.

It got worse on its own, because the workers' `/health` probe (5s) started timing
out under the load the workers themselves were creating, and a timed-out probe
counted as *"the endpoint is gone"*: coderai killed each worker and spawned a
replacement, which added load, which timed out more probes.

Both halves are now bounded:

* the process-per-instance engines are capped at **one worker per core, and never
  more than half of free RAM**, with a log line naming the figure it is running
  and why (`CODERAI_OCR_INSTANCE_CEILING` overrides it for an operator who has
  measured their own host);
* a health probe only means *gone* on a definite **404 "unknown OCR engine
  lease"** — which is the real case it was written for, a front that restarted. A
  timeout means busy, and the worker stays up: respawning cannot fix a front that
  is down, since the replacement asks that same front for its lease. A 503 means
  the pool has nothing rented this second, which the lease does not care about.

#### Is there anything on the page at all?

Three outcomes are indistinguishable in a short OCR result, and they lead to
completely different places:

| what happened | what it needs | how to tell |
|---|---|---|
| the engine could not read the page | a better engine, or the page is bad | ink present, no text |
| the reply was cut off | a bigger budget | `model_attempts.truncated` |
| **the page is blank** | fix the document upstream | `page_ink.blank` |

The third was not detectable at all, and it cost most of a day. A four-page
court PDF came back with **47 characters** and was reported as an OCR failure.
It was not. Every page was a ~900-byte CCITT Group 4 image — which is what an
*empty* page costs, a couple of bytes per all-white row — carrying **0.13% ink**,
all of it a diagonal scan edge and a diagonal "copia non ufficiale" watermark.
The 47 characters *were* the watermark. The engine had read the page perfectly.

So any page whose text comes back shorter than `ocr.blank_page_max_chars` (200)
now carries the one measurement that separates the three — the fraction of its
pixels that are not background, taken from an image already in memory:

```json
"page_ink": {"ink": 0.00137, "threshold": 0.005, "blank": true,
             "why": "0.137% of this page's pixels are not background, below
                     0.500% — there is essentially nothing on it to read. A short
                     result here is the page being empty, NOT the engine failing
                     to read it; look at the source document."}
```

The opposite verdict is just as useful: a short page **above** the floor says
there *is* something there, so the engine's failure is real and worth chasing.
Blank page indices are also listed at the top of the response as `blank_pages`,
because a caller deciding whether to accept a document should not have to walk
the pages to discover that it is empty.

`ocr.blank_page_ink` (0.5%) sits well under any page carrying text — a typical
text page at 200 DPI is several per cent — and comfortably above the edges,
specks and watermarks an empty scan still carries. It applies to **every
engine**: the measurement is of the image, not of the model.

This is a diagnostic, not a gate. Nothing is skipped, refused or altered because
of it; a blank page is still OCR'd and still returns whatever it returns.

---

## Finishing a page the model ran out of room for

`ocr.surya_max_tokens` raises the cap, but raising a cap only moves the cliff: a
dense enough page overflows any fixed number. `ocr.surya_max_continuations`
(default **2**) resumes instead — the text so far goes back as a prefilled
assistant turn with vLLM's `continue_final_message`, so the model carries on the
exact string rather than answering afresh about the same image. A server that
rejects the flag (llama.cpp) gets a plain "continue" instruction; a server that
rejects both leaves the page partial and honestly marked, because a page that
cannot be finished is still a page and raising would turn a degraded result into
no result.

The logprobs are concatenated with the text, in step. That is not bookkeeping:
`surya_repair_accents` rebuilds characters from the model's own per-token bytes,
so text from a later segment paired with logprobs from an earlier one would
corrupt exactly the accented characters the repair exists to fix. If they cannot
be kept aligned, the repair is given up on rather than fed a mismatch.

Resumption runs at temperature 0 — the opposite of the retry path, which
deliberately *raises* temperature to break a repetition. A page still unfinished
after its budget is marked `"complete": false`, so a caller can refuse a partial
rather than store it as read.

**The budget is per PAGE, not per document.** A fifty-page document is fifty
independent pages, each with its own budget, so what the number has to cover is
the densest *single* page — not the longest document.

The real ceiling is not this number but the model's context. Each resumption
re-sends the text so far as *input*, so with surya-ocr-2's 32,768-token window
and roughly 4,000 of it spent on the page image, about thirteen 2,048-token
segments exhaust it whatever is configured; raising `surya_max_tokens` lowers
that count in proportion. The default is **8**, which covers any plausible
single page while staying inside the window.

A generous budget is safe because a page that stops making progress stops
costing: a resumption that comes back repeating itself, or that returns the text
already in hand, ends the loop. That second case is also how a server that
**ignores** `continue_final_message` is caught — a model that starts the page
over returns what we already have, and without the check a budget of 8 would buy
eight full generations of the same page.

### Per-request budgets

One global cap cannot serve a public deployment: a four-page contract and a
fifty-page judgment need different budgets, and only the caller knows which it is
sending. `/v1/ocr` and `/v1/ocr/batch` therefore accept `max_tokens` and
`max_continuations` per request, overriding the configured values for that call
only. Omitting them means "use the configured value" — **not** zero, which is a
meaningful setting for both (no override, and resumption off).

They travel *with the page* rather than with the engine load, because a pool is
built once and then serves every caller, and they are applied **thread-locally**
in the worker, because several pages are in flight in one process and the hooks
they feed are module-level — a per-request value in a module cell would be
overwritten by the next page to arrive.

One caveat worth knowing, since it cost a day of wrong conclusions: FastAPI
drops unknown form fields **silently**. Before these parameters existed, a
caller could send `max_tokens=8000` on every request for a week and never be
told it did nothing.

---

## An empty page and a 71-character page are the same bug

Two failure modes from one live batch: ~98 pages of about **71 characters** and
**100 with no text at all**, against 13 that read perfectly. Neither is a model
that cannot read.

`surya-ocr-2` emits **block markup** — a category and a bounding box per block,
*before* that block's text. A reply stopped at its token cap is therefore
incomplete markup, and incomplete markup parses to a handful of blocks, or to
none:

| what the reply had room for | what the page looks like |
|---|---|
| all of it | the page, correctly |
| some blocks, then the cap | a short fragment — the uniform low chars-per-page |
| only markup, then the cap | **no text at all** |

The cap is surya's `chat_completions_batch(max_tokens_default=2048)`, which its
vLLM backend never overrides — so 2048 was the only value there has ever been.
A dense page can legitimately need several thousand tokens, and a batch measured
on 2026-10-13 averaged **2,644 generated tokens per request**, i.e. above the
cap.

`ocr.surya_max_tokens` raises it (0 = leave surya's 2048). It is opt-in because
a higher cap costs tokens on every page and the right value depends on the
documents. What is **not** optional is being able to tell the two cases apart,
so every page that hit the cap says so in its own result:

```json
"model_attempts": {
  "calls": 1, "tokens": [2048], "cap": 2048, "truncated": true,
  "why": "the reply stopped at the 2048-token cap, so the block markup it was in
          the middle of is incomplete and parses to less text than the page holds
          (or to none). This is a cut-off reply, NOT a page the model could not
          read — raise ocr.surya_max_tokens."
}
```

A page whose reply fitted carries nothing, and is byte-identical to what it
always returned. The per-attempt token counts are kept **per thread**, like the
token trace, so one lane never reports another lane's verdict.

### A diagnostic that does not arrive is not a diagnostic

`model_attempts` was added, shipped, and then hunted for in a live batch that
could not have contained it: `SubprocessOcrEngine._page_from_dict` copied the
worker's diagnostics onto `page.meta` **by name**, and it knew only
`unreadable_trace` and `accents_repaired`. The same reassembly had already
dropped `accents_repaired` once, so a working repair reported `null`.

It now forwards by exclusion — every key that is not an `OcrPage` field becomes
`page.meta`. A key-by-key copy means each new diagnostic is born broken and
nobody finds out until the moment they need it.

---

## A batch can be GPU-idle and still look stalled

Measured on the Digesta orchestrator on 2026-10-13, ten minutes of batch OCR at
four client workers, read from the pod's own drained log
(`pod-logs/<pod_id>.log`):

```
Running: 15-19 reqs, Waiting: 0 reqs, GPU KV cache usage: 5-7%
Avg generation throughput: ~1,900-2,100 tokens/s, every minute, for ten minutes
524 x "POST /v1/chat/completions HTTP/1.1" 200 OK, no aborts
2,644 generated tokens PER REQUEST (second pod: 4,010)
MM cache hit rate: 25-27%
```

The client reported a stall: one opening burst, then ~0 completions. Three
separate things are in those numbers, and only the last is a defect.

**The GPU was never the bottleneck.** 5-7% of the KV cache, `Waiting: 0` for ten
minutes. The card could have taken several times the load. So giving Surya a
card to itself does not help, and a co-loaded model on the same pod (pyannote,
in that run) costs nothing measurable — the limit was never VRAM or batch slots.
Check `Waiting:` and `KV cache usage:` in the pod log before moving anything.

**Oversubscription converts throughput into latency, not into stalling.** The
pool queues pages beyond `max_pods x scale_up_inflight_per_pod` and serves them
perfectly correctly, but every page's latency rises by the queue — and a client
that counts finished DOCUMENTS sees nothing at all until the first document's
LAST page lands. Spread 64 pages over sixteen documents and all sixteen advance
while none completes. That reads exactly like a stall. The pool therefore says, when the
pages are set up, how many pods they imply:

```
[ocr] engine 'surya': 64 pages at once spreads over 4 pods (16 in flight each);
      a pod takes ~12 min to become ready, so the first burst is served by
      whatever is already warm.
```

and, when the arithmetic exceeds `max_pods`, that the surplus simply queues:

```
[ocr] engine 'surya': 64 pages at once needs 4 pods at 16 in flight each, but
      max_pods is 2 — 32 will be served and the rest will queue.
```

A high `max_pods` does not help a burst: it is the pods already **warm** that
serve it, because the twelve minutes are spent before the first of the new ones
answers anything.

**The defect: 2,644 generated tokens for a page whose text is ~400.** Surya
re-runs a page whose generation repeated itself — `_should_retry` in
`surya.inference.backends.openai_client` — up to `max_retries` (3) more times,
each at a higher temperature and each able to run to its token cap. On a page
the model genuinely cannot read that is four full generations to reach the same
unreadable answer, and the repeat detector cannot distinguish "the model broke"
from "this page really does repeat", which a tax table does. The 25-27%
multimodal cache hit rate is the direct evidence: a quarter of all requests were
re-sends of an image the pod had already read.

`max_retries` is unreachable from Surya's settings — its vLLM backend calls
`chat_completions_batch` without it, so the default was the only value
available. CoderAI therefore hooks the decision, beside the token-trace hook it
already installs in that module, and `ocr.surya_max_retries` is the ceiling:

| value | meaning |
|---|---|
| `1` (default) | one re-run — keeps the rescue a retry does buy, drops the long tail |
| `0` | no re-runs; a repeating page is returned as it came |
| `-1` | whatever Surya decides (its own 3) |
| `n` | at most `n`, and never more than Surya would have allowed |

It is a ceiling, not a setting: asking for 10 where Surya allows 3 gets 3.

Every retry is logged, and a page that needed more than one call carries it in
the result — always, not only under `surya_token_trace`:

```json
"model_attempts": {"calls": 3, "why": "surya re-ran this page because the
                   generation repeated itself; each re-run is a full generation
                   at a higher temperature"}
```

A page read on the first attempt stays byte-identical to what it has always
returned. Without this, a page generated four times over and a page that was
merely slow look the same from outside — which is why ten minutes of this was
invisible.

CoderAI also pins `SURYA_INFERENCE_PARALLEL=1` in the served modes. A **lane**
is coderai's unit of concurrency; left unset, Surya sizes its own thread pool
from a built-in GPU table and fans out inside the worker, so `instances x lanes`
would stop being the real number of pages in flight the moment one call carried
more than one page.

---

## Lanes: concurrency without another process

The worker protocol was one request, one reply, strictly in order, so **pages in
flight was the process count** — and that is the whole reason 64 pages meant 64
processes and 37 GB.

In a served mode that memory buys nothing. The recognition VLM is on the pod; the
local predictor holds **no weights** (1.7 MB of tokenizer and config on disk, and
surya-2 returns its own boxes so the detection model is never fetched); and the
process spends a page rasterising it and then waiting on a socket. So a **lane**
is a worker *object* inside the process that already exists, with its own
predictor and its own page, sharing the one torch import:

```
effective concurrency = <engine>_instances x <engine>_lanes
```

`surya_lanes` defaults to **4** and applies to the served modes only — in `local`
mode a lane would be a second copy of the model on the card, so there is exactly
one. Lanes are built on demand, so a quiet install still pays for one.

Three details make it safe to drive one process concurrently:

* **every reply carries the id of the request it answers.** Out-of-order replies
  are the point — a short page must not wait behind a dense one — and attributing
  one page's text to another request would be worse than losing it. A reply whose
  id is unknown is dropped, never guessed at;
* **the token trace and the accent repairs are per THREAD.** surya's hook is
  module-level, one `_generate_one` per process, and its sink used to be a list
  belonging to whichever worker object installed it last. Shared between lanes it
  would have filed one page's logprobs — and `surya_repair_accents`' rewrites —
  under a different page;
* **a worker that dies fails its pages** instead of leaving them parked on a
  reply that is never coming: the wait is broken up to check the process is still
  there, and killing the engine wakes everything waiting.

A single-lane engine sends no ids and reads replies in order, exactly as before,
which is how paddle, docTR and surya-`local` are still driven.

To raise OCR throughput, raise lanes and the pod-side numbers, not the process
count. Instances still bound how much of this host OCR may take; lanes are what
turn one process into many pages.

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
