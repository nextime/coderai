# GLM-5.2 via colibri

CoderAI can serve **GLM-5.2** through **[colibri](https://github.com/JustVugg/colibri)**
by **JustVugg** — a brilliant, pure-C Mixture-of-Experts inference engine that treats
VRAM, RAM and disk as one memory hierarchy and streams expert weights on demand, so a
**744B-parameter** model runs on a single consumer GPU.

Unlike ds4 (which ships its own HTTP server), colibri's Python side is only a thin
gateway around the C engine. So coderai drives the **C engine binary directly** over
its stdin/stdout *mux* wire protocol (`docs/serve_protocol.md` in the colibri repo) —
we own the build, the process, the GLM-5.2 chat template and the protocol client; no
colibri Python runs at request time.

> **Hardware reality:** GLM-5.2 is huge (~429 GB int4 container). colibri runs the
> **dense layers on CPU** and streams **routed experts** (a small hot tier pinned in
> VRAM, the rest from NVMe). It is **CPU- and streaming-bound**, not GPU-bound — the
> GPU is a small accelerator. Keep the model on fast local storage (NVMe/ext4).

## The model

The model is a **directory** (int4 g64 shards + config + tokenizer + int8 MTP head),
not a single file — the gs64 build with the int8 MTP head from
[`mastouri/GLM-5.2-colibri-int4-g64-with-int8-mtp`](https://huggingface.co/mastouri/GLM-5.2-colibri-int4-g64-with-int8-mtp).
Download it from the Models page like any other HF repo (whole-repo snapshot → a
directory); coderai registers it as a `colibri`-backed model.

## Pre-compiled, not built at request time

The engine is a **pre-compiled** C binary (like ds4), never compiled inside the CUDA
*runtime* container (which has no `nvcc`). Build it on a host with the CUDA toolkit —
`build.sh --colibri` — with a **portable** GPU arch (SASS for Ampere→Blackwell). The
binary links `libcudart.so.13`, resolved in-container from `/opt/coderai/local-libs`
(the CUDA-13 runtime coderai already ships for PyTorch), so it adds **no** portability
constraint beyond what the existing stack requires.

## Tuning (env, via the per-model / global `extra_env`)

- **`CTX`** — context window. Sized from the model's `n_ctx`. The DSA KV reserve is
  **host RAM** (~330 KB/token), so max context is RAM-bound: ~64k is comfortable on a
  54 GB box; the model itself supports up to 1,048,576 positions given enough RAM.
- **`CUDA_DENSE=1`** — run the dense layers on the GPU (uses ~11 GB VRAM) instead of
  the CPU. The big lever for offloading CPU heat and actually using the 3090.
- **`CUDA_EXPERT_GB`** — VRAM budget for the hot expert tier.
- **`COLI_NO_OMP_TUNE=1`** / **`OMP_NUM_THREADS`** — tame CPU spin/heat when the CPU is
  mostly waiting on the GPU.

## Graceful thermal pause (coderai patch)

colibri decodes autonomously once a request is in flight, so coderai adds a small,
idempotent **`PAUSE`/`RESUME` serve-mux patch** (`packaging/patch-colibri.py`, applied
by `build.sh`): the engine idles the decode loop *between tokens* — keeping all KV
state — on `PAUSE`, and continues on `RESUME`. This lets the front's cooperative
thermal throttle cool the box without `SIGSTOP`-freezing the process. The patch is a
candidate to upstream.

## Files

- `codai/config.py` — `ColibriConfig`.
- `codai/api/colibri_worker.py` — clone/build, the `MuxEngine` protocol client, and the
  per-container engine registry.
- `codai/backends/colibri.py` — `ColibriBackend` + the GLM-5.2 chat template renderer.
- `codai/models/manager.py` — `colibri_should_handle()` routes matching models to
  `ColibriBackend`.
- `packaging/patch-colibri.py` — the `PAUSE`/`RESUME` serve-mux patch.

## Credits

GLM-5.2 support exists entirely thanks to **[colibri](https://github.com/JustVugg/colibri)**
by **[JustVugg](https://github.com/JustVugg)** — an ingenious piece of systems
engineering that makes a 744B model runnable on hardware it has no business running on.
CoderAI only drives it; the brilliance is theirs. Deep thanks.
