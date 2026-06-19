# Process-isolation plans: keeping the web UI responsive during model load/inference

> **Status (implemented):** Plan B + Multi-engine shipped. The front proxy lives in
> `codai/frontproxy/` (`app.py`, `engine_supervisor.py`, `registry.py`, `router.py`),
> engines run via `coderai --engine-only --internal-port N`, and default boot starts
> the front + one engine per GPU. Operator guide: `docs/frontend-engine-split.md`.
> Plan C (replicating session/config/queue ownership into the front) remains a
> follow-up — **except the broker, which has already moved into the front** (it
> registers once for the whole node, advertises aggregate VRAM torch-free, and
> dispatches brokered requests to engines through the router). Sessions/config/queue
> are the remaining Plan-C pieces.

## Problem statement

While a model loads (and, for some backends, while it generates), the web
interface and API become unresponsive.

Root cause: the server is a single process. GIL-heavy Python work blocks the
asyncio event loop that serves the UI/API. Specifically:

- **Transformers text** (`codai/backends/cuda.py`, `NvidiaBackend`) — both the
  `from_pretrained` **load** and token-by-token `model.generate` hold the GIL.
  Dispatching them via `asyncio.to_thread` does **not** free the loop, because
  `to_thread` only helps when the worker releases the GIL.
- **Diffusers** (image/video/audio, `codai/api/images.py`, `video.py`,
  `audio_gen.py`) — the `from_pretrained` **load** is GIL-heavy and freezes the
  UI. The denoise loop itself is mostly torch CUDA ops that *do* release the
  GIL, so the freeze is almost entirely the load.
- **Vulkan / GGUF** (`codai/backends/vulkan.py`, llama.cpp) — the native load
  **releases the GIL**, so this path does *not* freeze the UI. (This is why the
  existing defensive comments assume "the load releases the GIL during its C
  call" — true for llama.cpp, false for the transformers/diffusers paths.)

The fix is to ensure the process serving the UI/API is not the process whose GIL
is held by model work. Three architectures achieve this with very different
cost/benefit. This document captures all three so we can choose deliberately.

> Note: an unrelated, already-shipped fix lives in `cuda.py` — Gemma-class models
> whose attention head dimension exceeds FlashAttention-2's limit of 256 now fall
> back to SDPA (`_model_head_dim`), which fixed "requests silently stop" for those
> models. That is orthogonal to the process-isolation work below.

---

## Summary comparison

| | A: model worker (out-of-process models) | B: thin resilient proxy | C: full frontend/engine split |
|---|---|---|---|
| Process boundary | Python pipeline-call layer | HTTP layer | HTTP layer + state ownership |
| Serialization burden | **High** (torch generators, callbacks, tensors, PIL) | **Low** (already HTTP) | **Low** (already HTTP) |
| Engine/model code changes | Large (text clean, diffusers invasive) | **None** (engine ≈ current app) | Moderate (engine becomes pure executor) |
| Fixes which model types | One modality at a time | **All at once** | **All at once** |
| New moving parts | Worker harness + per-modality IPC | Reverse proxy + status cache + supervisor | Proxy + relocated coordination state + supervisor |
| Crash/CUDA-poison isolation | Per-model worker | **Engine restart, front survives** | **Engine restart, front survives** |
| Effort | Text: medium. Diffusers: very large. | **Small–medium** | Large |
| Recommended role | Fallback / not preferred | **First cut (do this)** | Eventual evolution of B |

**Recommendation:** ship **B**, evolve toward **C** if/when coordination state
needs to be authoritative in the front; keep **A** only as a documented
alternative (it is the worst fit for diffusers).

---

## Shared context (applies to B and C)

- Public surface is **plain HTTP + SSE**. No inbound websockets, no mounted
  sub-apps (verified). This makes a reverse-proxy split clean.
- `codai/broker/asgi_bridge.py` already drives the ASGI app from an external
  transport, so the app is already transport-decoupled in spirit.
- The front process must import **no** `torch` / `transformers` / `diffusers`,
  so its GIL is never held by model code and its event loop is always free.
- VRAM/GPU stats can be read by the front **without torch** via `nvidia-smi`
  and sysfs/`lspci` (the existing `api_status` already reads sysfs/`lspci` for
  the non-CUDA path).

---

## Plan A — Out-of-process model worker (models leave the API process)

The original approach: keep the API/UI in the main process, push the GIL-heavy
model into a child process behind a proxy backend.

### A.1 Generic worker harness
- `codai/backends/worker_client.py` — parent-side proxy implementing the
  `ModelBackend` interface; spawns the child, waits on `/health`, forwards calls.
- `codai/backends/text_worker.py` — child entrypoint
  (`python -m codai.backends.text_worker --port 0`) running a tiny local uvicorn
  that instantiates the **real** `NvidiaBackend` and exposes `/load`,
  `/generate`, `/generate_chat`, `/generate_stream` (SSE), `/generate_chat_stream`
  (SSE), `/context_size`, `/usage`, `/tokenize`, `/health`, `/shutdown`.
- Wire into `ModelManager.load_model` (`codai/models/manager.py:158`): when
  `backend_type == "nvidia"`, instantiate `WorkerTextBackend()` instead of
  `NvidiaBackend()`, behind a default-on config flag. Instance pools, eviction,
  VRAM delta accounting (`torch.cuda.mem_get_info` in the parent still sees the
  child's allocations) are untouched — each instance owns a subprocess.

### A.2 Text worker (clean)
- I/O is tiny (text / SSE tokens). Streaming maps directly to SSE.
- `cleanup()` terminates the subprocess → frees VRAM.
- Bonus: a device-side CUDA assert kills only the child; parent maps the error to
  the existing `cuda_context_poisoned` logic and respawns.

### A.3 Diffusers worker (very large — the blocker)
Diffusers cannot be a thin wrapper. Evidence in `codai/api/images.py`/`video.py`:
- Pipelines are stored **as live objects** in the shared registry
  (`multi_model_manager.models[model_key] = pipe`) and called inline at ~dozens
  of sites (txt2img, img2img, inpaint, upscale, depth, segmentation, video
  modes, audio_gen).
- Pipelines are **mutated in-process**: `apply_accel_to_pipeline(pipeline, accel)`
  (`images.py:345`), LoRA application, IP-Adapter wiring, scheduler swaps.
- `pipe(...)` call args are **not serializable**: `generator` (a
  `torch.Generator` bound to a device), `callback_on_step_end=_step_cb` (a live
  closure updating the in-process `_gen_progress`), `embed_kwargs` (prompt
  embedding tensors), IP-Adapter/character/environment **PIL reference images**
  (`images.py:877-899`).

Consequence: putting diffusers in a worker means **moving the entire generation
lifecycle into the worker** (load + accel + LoRA + IP-adapter + the call + output
extraction) and converting every call site to a **high-level** IPC request
(prompt, seed, steps, image bytes), serializing every input (PIL/tensors/masks/
control images), every output (images/frames/audio), and **relaying step
progress** back over IPC. Large, regression-prone rewrite of the media API.

### A.4 Assessment
- Text: medium effort, clean win.
- Diffusers: very large, fragile; payoff limited (denoise releases the GIL).
- **Not recommended** as the diffusers solution. Superseded by B.

---

## Plan B — Thin resilient reverse proxy (RECOMMENDED FIRST CUT)

Split at the HTTP boundary. The **engine** is the current app, essentially
unchanged, on an internal port. The **front** is a small async reverse proxy on
the public port whose event loop never freezes (no torch in its address space).

### B.1 Architecture
```
client ──HTTP/SSE──▶  front (public port)  ──HTTP/SSE──▶  engine (internal port, all models)
                      • no torch                          • current app, unchanged
                      • always-responsive                 • may freeze on GIL-heavy load
                      • status cache + timeouts           • does all GPU work
                      • supervises engine subprocess
```

### B.2 The one rule that makes it work
The front must answer **UI / status / admin** without synchronously
hard-depending on a possibly-frozen engine:
- UI / status / admin → short timeout on the engine call; on timeout serve a
  **last-known status cache** plus an "engine busy loading model X" flag.
- Generation (chat / image / video / SSE) → proxied with **long timeout**. That
  single request legitimately waits for the load; the rest of the UI stays live.

### B.3 New files
- `codai/frontproxy/__init__.py`
- `codai/frontproxy/app.py` — FastAPI app for the front:
  - Catch-all reverse-proxy route: streams request body (chunked uploads),
    forwards method/path/query/headers (incl. auth, rewriting `Host`), streams
    the response back (SSE and large binary), preserves status codes.
  - Status handler: proxies `/admin/api/status` with a short timeout; caches the
    last success; on timeout/refusal returns cache + `{ "engine": "loading"|"down" }`.
  - `/healthz` for the front itself.
- `codai/frontproxy/engine_supervisor.py` — spawn the engine subprocess
  (`python -m codai.main --internal-port …`), poll `/healthz` on the engine,
  restart on crash/exit (this is where CUDA-poison recovery becomes "respawn").
- HTTP client: `httpx.AsyncClient` with streaming, or `aiohttp`. Separate short-
  and long-timeout clients.

### B.4 Engine-side changes (minimal)
- `codai/main.py` / `codai/cli.py`: add `--internal-port` / `--engine-only` so
  the engine binds to localhost and the front owns the public port. Default boot
  launches front + engine; a flag preserves the legacy single-process mode.
- Add a cheap `/healthz` on the engine (no torch, returns immediately) so the
  supervisor can distinguish "loading" (slow) from "dead".

### B.5 Proxy correctness checklist (the real work)
- **SSE / streaming**: forward `text/event-stream` without buffering; flush per
  chunk; propagate client disconnect to cancel the upstream request.
- **Large uploads**: stream `model-upload` / image inputs (don't buffer whole
  body in memory).
- **Large downloads**: stream image/video/audio byte responses.
- **Auth / headers**: pass `Authorization`, cookies; rewrite `Host`; preserve
  `Content-Type`, `Content-Length`/chunked, `Content-Disposition`.
- **Timeouts**: short for status/UI; long (or none) for generation; map engine
  timeout to cached status, never to a hung front request.
- **Backpressure / limits**: bound concurrent in-flight proxied requests.
- **Redirects / error passthrough**: preserve 3xx/4xx/5xx and bodies.

### B.6 Limitations
- Does not speed up the one in-flight request waiting on a load; keeps the rest
  of the UI responsive.
- True concurrency across models needs multiple engines (see "Multi-engine").

### B.7 Effort: small–medium. Engine code essentially untouched; risk concentrated
in the proxy, which is testable in isolation.

---

## Plan C — Full frontend/engine split (eventual evolution of B)

Make the front authoritative for all pure-Python coordination state so it never
needs the engine even for status; the engine becomes a pure executor.

### C.1 What moves to the front
Relocate non-GPU, pure-Python concerns out of the engine into the front (each is
serialization-trap-free):
- **Sessions / auth / API tokens** (`codai/admin` session manager).
- **Config / models.json management** (the admin "models" CRUD, `config_manager`).
- **Request queue + metrics** (`codai/queue/manager.py`).
- **Progress + model-registry view**: the engine pushes events (loaded/unloaded,
  `_gen_progress` step updates, VRAM deltas) to the front over a control channel;
  the front holds the authoritative cache and serves status with zero engine
  dependency.

### C.2 Engine becomes pure executor
- Exposes only: load/unload, generate (all modalities), health, event stream.
- No session/config/queue logic; receives resolved requests from the front.

### C.3 Control channel
- A persistent engine→front event stream (SSE or a small socket) for progress,
  load state, VRAM, and crash notifications. Front reconciles its cache; on
  engine restart, front re-syncs.

### C.4 Benefits
- Status/admin are instant and always correct, even mid-load.
- Clean seam for **multi-engine** orchestration.
- Strong fault isolation: engine crash never loses UI/session/queue state.

### C.5 Effort: large, but every moved piece is plain Python (no pipeline
serialization). Best approached incrementally on top of a shipped B.

---

## Multi-engine (future, enabled by B/C)

One engine per GPU (or per hot model). The front routes a request to the engine
that holds the target model (or asks an idle engine to load it). One engine
loading no longer blocks generation on another engine. Requires:
- Engine registry in the front (which engine holds which model, health, VRAM).
- A placement/eviction policy across engines (extends the current per-process
  VRAM logic to a fleet view).

---

## Decision log / open questions

- Confirm: default boot launches **front + engine** with a flag to retain the
  legacy single-process mode? (Recommended yes.)
- Confirm: HTTP client — `httpx` (already a likely dependency) vs `aiohttp`.
- Confirm: status staleness budget when the engine is mid-load (e.g. serve cache
  up to N seconds old, then show "engine loading").
- B → C migration order: sessions/tokens first (low risk), then config, then
  queue, then progress/registry (needs the control channel).

## Recommended sequencing

1. **B** — front proxy + engine supervisor + status cache. Fixes the freeze for
   all model types with no engine changes beyond `--internal-port`/`/healthz`.
2. (Optional, separate) **A.1/A.2** text worker — only if we want per-model fault
   isolation *within* an engine; otherwise B already solves the UI freeze.
3. **C** — incrementally move coordination state to the front.
4. **Multi-engine** — once C's registry exists.
