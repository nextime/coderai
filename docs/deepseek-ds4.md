# DeepSeek V4 via ds4

CoderAI can serve **DeepSeek V4** (Flash / PRO) through antirez's
[ds4 / DwarfStar](https://github.com/antirez/ds4) — a native (C/CUDA/Metal)
inference engine built specifically for DeepSeek V4 that ships its own
OpenAI-compatible HTTP server (`ds4-server`).

Because ds4 is a standalone binary (not a Python package), coderai owns its whole
lifecycle as an *external worker* — the same pattern used for Parler-TTS
(`codai/api/parler_worker.py`). When enabled, coderai builds ds4, downloads the
model weights, launches `ds4-server` as a managed subprocess, and proxies text
requests to it. Everything else in coderai (tool parsing, streaming, the chat UI)
keeps working unchanged.

> **Hardware:** DeepSeek V4 is large. Per upstream you want **96 GB+ RAM**
> (256 GB+ for the Q4 variant, 512 GB for PRO). First use also clones the repo,
> compiles a native binary, and downloads several GB of weights — it is slow.

## Enabling

Admin → **Settings → DeepSeek V4 (ds4)**:

- **Enable ds4** — turn the integration on.
- **Model id / alias** (default `deepseek-v4`) — any chat request whose model name
  equals this id, or contains `deepseek-v4` (case-insensitive), is routed to ds4
  instead of the normal NVIDIA/Vulkan backends. All other models are unaffected.
- **Weight variant** — passed to ds4's `download_model.sh`
  (`q2-imatrix`, `q2-q4-imatrix`, `q4-imatrix`, `pro-q2-imatrix`).
- **Build target** — `auto` detects CUDA (`cuda-generic`) / macOS (`metal`) /
  `cpu`; override for DGX Spark (`cuda-spark`).
- **Install dir** — where ds4 is cloned/built (default `~/.coderai/ds4`, or
  `$CODERAI_DS4_DIR`).
- **Auto build** — clone + `make` the `ds4-server` binary if it's missing.
- **Bind host / Port / Context** — `ds4-server --host/--port/--ctx`
  (port `0` auto-picks a free port).
- **Extra args** — passed verbatim to `ds4-server`, e.g.
  `--kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192`.

Then send a normal request:

```sh
curl localhost:8776/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "deepseek-v4",
  "messages": [{"role":"user","content":"Hello"}]
}'
```

The first such request triggers build → download → serve (with generous timeouts);
build and download logs are streamed with a `[ds4]` prefix. The subprocess is torn
down by the model manager's normal eviction and on server shutdown.

## Building ahead of time / packaging

Runtime auto-build works, but for reproducible installs (and Docker) you can build
ds4 during setup:

```sh
./build.sh all --ds4        # clones + builds ds4-server into ~/.coderai/ds4
```

The OCI image builder (`packaging/linux/build_oci_image.sh`) auto-discovers and
bundles the prebuilt `ds4-server` binary (and its shared libraries) the same way it
bundles `whisper-server`. Model **weights are not bundled** — they are downloaded
on first use inside the container. If only the binary is shipped (no repo scripts),
coderai shallow-clones the repo at first use to obtain `download_model.sh`.

## Implementation

- `codai/config.py` — `Ds4Config`.
- `codai/api/ds4_worker.py` — clone/build, weight download, `ds4-server` lifecycle.
- `codai/backends/ds4.py` — `Ds4Backend`, an OpenAI-API proxy implementing the
  `ModelBackend` interface.
- `codai/models/manager.py` — `ds4_should_handle()` routes matching models to
  `Ds4Backend`; `is_allowed_model()` accepts the ds4 model id.
