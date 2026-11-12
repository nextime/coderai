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

## Choosing between the images

| | `coderai-llama` | `coderai-llamasrv` |
|---|---|---|
| how a GGUF is served | in-process, llama-cpp-python | `llama-server` subprocess |
| concurrency | one generation per instance | slots + continuous batching |
| speculative decoding | not reachable | yes |
| ggml RPC peer | yes (`rpc-server` shipped) | no |
| core | full | light (no torch) |

Reach for `coderai-llama` first. Reach for this one when a model wants drafting or slots.
