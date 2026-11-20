# `llama-server` as a backend (`llamasrv`)

CoderAI serves GGUF models **in-process**, through
[llama-cpp-python](https://github.com/abetlen/llama-cpp-python) — `libllama` loaded into
the engine process. That is the default and stays the default. This backend runs
llama.cpp's own **`llama-server`** binary as a managed subprocess instead, and proxies
OpenAI requests to it.

It is an **alternative front end to the same files**, not a replacement. Both paths load
the same GGUF; they differ in what they can do with it.

## Why have both

Two things live outside `libllama` and so cannot be reached from the Python bindings at
all:

- **Speculative decoding.** The draft-and-verify loop is in llama.cpp's
  `common/speculative.cpp`. `common` is a static library linked by the CLI tools — it is
  not part of `libllama`. A tree whose headline feature is drafting can therefore only be
  driven by running its server.
- **Real slots with continuous batching** (`-np`). The in-process path serialises on a
  per-instance generation lock, so concurrency there means more instances, each with its
  own copy of the weights.

If you need neither, use the in-process path: one process, one copy of the weights, no
port.

## Selection is pin-only

`llama-server` serves exactly the files the in-process path serves, so it **never claims a
model by name**. Enabling the backend changes where nothing goes. A model reaches it only
by:

```jsonc
// models.json
{ "id": "my-model.gguf", "path": "/models/my-model.gguf", "backend": "llamasrv" }
```

or by matching the configured `model_id`. A pin to a *disabled* backend routes **nowhere**
rather than falling back to the bindings: the model was pinned here for a reason, and
serving it without slots or drafting would look like the feature working badly rather than
being off.

Each pinned GGUF gets its own server process, keyed by file.

## Which llama.cpp gets built

`tree` decides:

| value | what is built |
|---|---|
| `bindings` (default) | the llama.cpp **vendored in the installed `llama-cpp-python` sdist** |
| `repo` | `repo_url` checked out at `git_ref` — a fork |

`bindings` is the one to keep. The server is then built from the very tree the in-process
path already runs, so the two are one engine with two front ends: a GGUF behaves
identically through either, and the only difference is batching. It is also what
`packaging/build-rpc-server.sh` does, for the same no-version-skew reason.

Pick `repo` for a tuned fork. Two hazards come with that, and the configuration refuses to
paper over either:

- **Pin the ref.** A fork's default branch moves under you. An empty `git_ref` is accepted
  but warns in the log, because the build that breaks next month will look like a broken
  fork rather than a moved ref.
- **Pin the architecture.** Several forks are tuned for one CUDA architecture. See below.

Each tree builds in its own directory (`~/.coderai/llamasrv/<slug>`), so switching trees or
bumping a ref never builds over the previous one.

> **Not an RPC peer.** This binary is deliberately never registered as a ggml RPC server.
> The RPC protocol is versioned and a client and server from different llama.cpp commits
> refuse each other, so `codai/backends/ggml_rpc.py` keeps talking to the `rpc-server`
> built from the bindings' own tree. A fork is a different tree.

## The architecture gate

`cuda_architectures` is what cmake gets as `CMAKE_CUDA_ARCHITECTURES`. Blank means "the
cards this host actually has", probed via NVML or torch
(`codai/backends/gpu_probe.py:compute_capabilities`), duplicates collapsed.

An explicit value is checked against the host, and `require_matching_arch` (**on by
default**) refuses the build when nothing matches. This fails closed on purpose: a tree
tuned for one architecture and built for another does not fail cleanly — it can compile,
load, and answer with degraded output. "It built" is not evidence of anything. A host where
no card can be probed *at all*, with an explicit architecture set, is also refused: that is
a probe failure, not a match.

With no CUDA and no explicit request, a CPU-only binary is built and the log says so. That
is a fallback, not a configuration.

## Enable & configure

Settings → **llama-server (GGUF, alternative to the in-process path)**, or `config.json`
`"llamasrv"`:

| field | meaning |
|---|---|
| `enabled` | turn the backend on (still pin-only) |
| `model_id` | an id/alias that routes here; blank = pins only |
| `tree` / `repo_url` / `git_ref` | which llama.cpp to build (above) |
| `cuda_architectures` | cmake SASS targets; blank = this host's cards |
| `require_matching_arch` | refuse to build for an architecture no local card has |
| `extra_cmake_args` | appended to the cmake configure line |
| `auto_build` | build `llama-server` on first use if missing (needs `cmake`) |
| `ctx` | `-c`; `0` = the per-model `n_ctx`, else the server's own default |
| `n_gpu_layers` | `-ngl`; `-1` = all |
| `parallel` | `-np` server slots; `0` = the server's auto |
| `flash_attn` | `-fa on\|off\|auto` |
| `cache_type_k` / `cache_type_v` | `-ctk` / `-ctv` |
| `spec_type` / `spec_draft_n_max` | `--spec-type` / `--spec-draft-n-max`; blank = the tree's default |
| `extra_args` | flags passed verbatim to `llama-server` |
| `extra_env` | free-form `KEY=VALUE` env for the subprocess |
| `api_key` | the bearer the server requires; blank mints a new one per launch |
| `service_url` | proxy to a `llama-server` running elsewhere; nothing is built or launched |
| `ready_timeout_s` | how long a cold load may take |

A per-model `llamasrv` block overrides most of these for that model. It deliberately
**cannot** set `require_matching_arch`, `install_dir`, `auto_build`, `service_url` or
`api_key`: those are install-level, and a per-model block must not switch off a safety
check for everyone sharing the build.

### The bearer

`llama-server` is launched with `--api-key`, minted per launch when none is configured. A
loopback port is not an authorisation — another local process should not be able to drive
the model by finding the port. The key is redacted in the launch log, which is tailed on
the Tasks page and ships in support bundles.

The proxy is inherited from the vLLM backend (`VllmBackend`), which talks to a server
started *without* a key; the bearer reaches it through a `_headers()` hook consulted at
every HTTP call site. A test asserts that, because auth reaching only some call sites is
worse than none: the model answers until the first streamed request.

## The dedicated pod

`coderai-llamasrv` is the RunPod capability image: a coderai pod with `llama-server`
**compiled into the image's toolchain stage**. The runtime carries no CUDA compiler, so
nothing is ever built on a rented machine.

```sh
packaging/runpod/build_capability_image.sh llamasrv
```

Rent it with `engine: "coderai-llamasrv"` on the model's `runpod` block, or let a local
model pinned `backend: llamasrv` burst to it — the pin travels.

It is still a coderai pod, not an upstream llama.cpp server image: TLS on the direct-TCP
path, a `/boot` record, seeding, local LoRA upload, the Tasks page.

Build a fork variant under its own tag:

```sh
docker build -f packaging/runpod/Dockerfile.capability-llamasrv \
  --build-arg LLAMASRV_REPO=https://github.com/owner/fork \
  --build-arg LLAMASRV_REF=v1.2.3 \
  --build-arg CUDA_ARCHS=86 \
  -t coderai-llamasrv-fork:latest .
```

A tuned fork usually wants exactly one architecture (`86` is GA102 — RTX 30-series and the
A40) and should be published under its own tag, because an image built for one
architecture must not be rented onto another card.

**The pod is never told how to build its server.** `tree`, `git_ref`,
`cuda_architectures`, `extra_cmake_args` and `require_matching_arch` are stripped from what
is forwarded: they describe a build the pod cannot perform, and a forwarded architecture
would claim a target the image was not compiled for. Only the run-time settings travel
(`ctx`, `parallel`, the cache types, the drafting flags, `extra_args`…). The image tells the
worker where its binary is through `CODERAI_LLAMASRV_BIN`; if that path is wrong the pod
says so instead of failing later inside cmake.

## Drafting: leave `spec_type` blank

`spec_type` maps to `--spec-type`, and on llamAmpere **leaving it blank is the
configuration**, not the absence of one. That tree's default is `auto`: it uses *the model's
own built-in drafter with measured settings for its family*, and the help text is explicit
that "any explicit value, including none, turns auto off".

So for a GGUF that carries its own MTP head, setting `spec_type` to the thing you want
*disables* the detection that would have chosen it. The accepted names are `draft-mtp`,
`draft-mtp-adaptive`, `ngram-mod` and `none`; set one only to override what `auto` picked,
or to turn drafting off.

Whether a GGUF has that head is a property of the file, not of its name. The NextN block is
the last layer's `nextn.*` tensors — for a 64-layer Qwen3.8-27B, `blk.64.nextn.eh_proj`,
`enorm`, `hnorm` and `shared_head_norm` beside a full `blk.64` attention and FFN. A file
whose publisher dropped it has no `blk.64` at all and drafts nothing, whatever the filename
says; check the tensors.

(`--mtp` is **not** this flag: in that tree it is a *download* option — fetch the MTP head
if the repo has one — and is not accepted by the server.)

## Vision: the projector comes from the model entry

llama.cpp loads a vision tower only when told, so a multimodal GGUF served without
`--mmproj` is a text model that accepts an image and describes **nothing** — it answers,
plausibly, about nothing it was shown. The projector is therefore read from the model's own
`mmproj` field, the same one the in-process path reads, and a model does not lose its sight
by being pinned to this engine.

- A declared-but-missing file is **skipped with a log line**, not passed through:
  `llama-server` refuses to start without it, which would turn "no vision" into "no model".
- An `--mmproj` (or `-mm`) already in `extra_args` wins, and ours is not added. The server
  takes the last one it is given, so appending ours would silently beat the override.
- **To serve a multimodal GGUF as text-only, clear the entry's `mmproj`** (and drop
  `image_to_text` from its `capabilities`, or `/v1/models` keeps advertising a vision model
  and clients will send images to it).

That last one is a real decision on a 24 GB card, because the projector and the context
window compete for the same memory. At `q8_0` KV a 27B Qwen3.8 costs ~36 KiB/token, so:

| projector | VRAM | context it costs |
|---|---:|---:|
| f16, ~0.9 GB | 884 MiB | ~25,000 tokens |
| F32, ~1.8 GB | 1757 MiB | ~50,000 tokens |

A coding model at a long context is usually better served without the tower; a model that
is actually sent images should give up the context instead. The same conversion at f16
rather than F32 halves the cost, which is worth checking before giving up 50K tokens.

## VRAM eviction

A `llama-server` holds the whole model in a **child process**, so it is the one engine whose
VRAM coderai cannot free by dropping an object. It takes part in eviction anyway, through
the same path as everything else and with no special case in the manager:

```
ensure_vram_for()        before the load, like any model: resident models are evicted
                         (and co-located sibling engines asked) to make room first
record_vram_delta()      after it: the footprint is the device's free-VRAM delta across
                         the load, which sees a subprocess exactly as it sees tensors
_evict_one()             later: drops the backend and calls cleanup() …
  -> cleanup()           … which stops this model's server, and the card is free
```

Three properties make that honest rather than approximate:

- **The card is free when the process is gone, not when it has been signalled.** `terminate`
  is waited for, and so is `kill`. Returning in between is what makes an eviction appear to
  free nothing — the manager measures no change, loads the next model anyway, and OOMs
  against weights that are still mapped.
- **A stop that matched nothing says so.** The service key includes the engine, and a key
  that matches no running server means the VRAM is still held by something nobody is
  tracking. That is logged, and `stop_service` returns `False`, instead of looking like
  success.
- **The server dies with its engine.** `PR_SET_PDEATHSIG` is set on the child, so an engine
  that is SIGKILLed — the OOM killer, a crash, a stray `pkill` — does not leave a
  `llama-server` holding a 16 GB model with nothing left that knows how to evict it. `atexit`
  covers the clean exit; only this covers the unclean one.

Because the footprint is measured from the device rather than from inside the process, an
idle llamasrv model is as evictable as an idle in-process one, and a GGUF served through the
server competes for the card on equal terms with a diffusers pipeline or a vLLM instance.

## Two engines: stock and a fork

`llamasrv` runs **stock** llama.cpp's `llama-server`. `llamampere` runs the same server
built from a **fork**. They are separate engines, each with its own enable flag, build
directory, server processes and engine node.

That is deliberate, and it is the whole point. The reason to run a fork is to find out
whether it is actually faster, and that needs both serving **at once** — one model on stock
llama.cpp, another on the fork, same card, same weights, measured against each other. A
single `tree` setting could only ever answer "which one is everything using", which is a
question nobody has.

```json
// config.json
{
  "llamasrv":      { "enabled": true, "tree": "bindings" },
  "llamampere": { "enabled": true, "tree": "repo",
                     "repo_url": "https://github.com/JakeATX/llamAmpere",
                     "git_ref": "v0.6.4", "cuda_architectures": "86" }
}
```
```json
// models.json — the same weights, one on each engine
{ "alias": "coder-stock", "path": "/AI/x.gguf", "backend": "llamasrv" }
{ "alias": "coder-fork",  "path": "/AI/x.gguf", "backend": "llamampere" }
```

A model refines its engine with a block named after that engine — `llamasrv` or
`llamampere` — so an override written for the fork cannot rewrite the stock engine's
launch. The fork's defaults point at llamAmpere, which targets Ampere; its
`cuda_architectures` deliberately does **not** default to "this host's cards", because
building a tuned tree for hardware its author never tested yields plausible output rather
than an error.

The same GGUF on both engines is **two servers**: the running-server key carries the engine,
not just the file. Keyed by path alone the second engine would be handed the first one's
server, and a comparison between them would silently be a comparison of one with itself.

## In the full image, and on a host that runs it

The image ships CUDA *runtime* libraries and no toolchain — no `nvcc`, no `cmake` — so a
server binary cannot be compiled on the machine that runs it. `auto_build` fails there, by
design and with a message saying so.

So the image bakes **one binary per engine**:

| engine | path in the image | built from |
|---|---|---|
| `llamasrv` | `/opt/coderai/llamasrv/bin/llama-server` | the llama.cpp vendored in the installed `llama-cpp-python` — the same tree as `rpc-server`, so the server, the RPC server and the in-process path are one llama.cpp |
| `llamampere` | `/opt/coderai/llamasrv-fork/bin/llama-server` | `LLAMAMPERE_REPO` at `LLAMAMPERE_REF` |

Each has a `BUILD_INFO` beside it naming the repo, ref and architectures. The launcher
exports `CODERAI_LLAMASRV_BIN` and `CODERAI_LLAMAMPERE_BIN` for those paths when the
files are there, and exports nothing for one that is absent — so an engine without a binary
reports itself unavailable instead of pointing at a file that does not exist. One variable
for both engines would have served whichever binary was baked last to every pinned model.

Both are built in `packaging/linux/Dockerfile.oci` (a from-scratch build) **and** in
`packaging/linux/Dockerfile.update-llama`, the overlay driven by
`update_oci_image.sh REBUILD_LLAMA=1`. The overlay matters: an install that only ever
receives the app overlay never rebuilds the bundle, and the binaries are exactly the part
the app overlay cannot refresh.

| build arg | default | meaning |
|---|---|---|
| `BUILD_LLAMAMPERE` | `1` | compile the fork at all |
| `LLAMAMPERE_REPO` | `https://github.com/JakeATX/llamAmpere` | which fork |
| `LLAMAMPERE_REF` | `v0.6.4` | the pinned tag — never a branch |
| `LLAMAMPERE_ARCHS` | `86` | baked SASS targets |

Two things about a baked binary are worth being blunt about.

**The architecture is not gated.** `require_matching_arch` guards a *build*, and with a
prebuilt binary there is no build to guard. An image built for `86` and run on another card
gets "no kernel image is available" from the CUDA runtime. Targeting other hardware means
rebuilding with your own `LLAMAMPERE_ARCHS`, not expecting a fallback.

**The `tree` / `repo_url` / `git_ref` settings are then not what is serving.** They describe
a build that did not happen on this machine, and nothing can verify that a prebuilt binary
is the tree they name — a stock llama.cpp and an Ampere fork both answer, one of them
without the speculative decoding the pin was for. That is why the launcher trusts only its
own paths and never a `llama-server` found on `PATH`, and why the worker logs the binary's
`BUILD_INFO` on first use: the ref actually in play is printed rather than assumed.

### Building a binary by hand for an existing install

An install whose image predates this — or one that bind-mounts a newer working tree into an
older image — can be handed a binary without rebuilding anything. Build it where the ABI
matches the image (same Ubuntu, glibc and CUDA) and drop it at the path the worker derives
from the config, `~/.coderai/llamasrv/<repo>-<ref>/build/bin/llama-server`:

```bash
REF=v0.6.4
DIR=~/.coderai/llamasrv/llamAmpere-$REF
git clone --depth 1 --branch $REF https://github.com/JakeATX/llamAmpere $DIR/src
docker run --rm -v $DIR:/work nvidia/cuda:12.6.3-devel-ubuntu24.04 bash -c '
  apt-get update -qq && apt-get install -y -qq cmake build-essential libcurl4-openssl-dev
  cmake -S /work/src -B /work/build -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
    -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \\
    -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -lcuda"
  cmake --build /work/build -j "$(nproc)" --target llama-server
  chown -R '"$(id -u):$(id -g)"' /work/build'
```

The flags are the ones `ensure_built` uses, plus one a **container** build cannot do
without. A devel container has no driver: `libcuda.so.1` is injected by the container
runtime at *run* time, so at *build* time ggml's calls into the CUDA driver API
(`cuMemCreate`, `cuGetErrorString`, …) resolve to nothing and the link fails — after the
entire compile has succeeded. The stub has to be **linked**, not merely searched:
`-L …/stubs` on its own fails identically. Building on a host that has the driver installed
needs none of this. `GGML_NATIVE=OFF` matters too: it keeps the build host's `-march` out of
a binary that may run on an older CPU. Because the path is the one derived from `tree`/`git_ref`, no
`CODERAI_*_BIN` variable is involved and the config still describes what is serving —
bumping `git_ref` then builds beside it rather than over it, and a stale binary cannot be
mistaken for the new ref.

## Choosing between the images

| | `coderai-llama` | `coderai-llamasrv` |
|---|---|---|
| how a GGUF is served | in-process, llama-cpp-python | `llama-server` subprocess |
| concurrency | one generation per instance | slots + continuous batching |
| speculative decoding | not reachable | yes |
| ggml RPC peer | yes (`rpc-server` shipped) | no |
| core | full | light (no torch) |

Reach for `coderai-llama` first. Reach for this one when a model wants drafting or slots.
