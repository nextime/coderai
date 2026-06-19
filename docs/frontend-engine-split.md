# Frontend/engine split (responsive UI + multi-engine)

CoderAI boots as two layers so heavy model work never freezes the web interface:

- **front** — a thin reverse proxy on the public host/port. It imports no
  torch/transformers/diffusers, so its event loop is always free. It streams
  requests/responses (including SSE) to the engines and serves an aggregated,
  cached status/tasks view.
- **engine(s)** — the real CoderAI app (the current server), bound to internal
  localhost ports, doing all GPU/model work. One engine per GPU by default; each is
  pinned with `CUDA_VISIBLE_DEVICES` so inside it the GPU is always `cuda:0` and the
  existing per-process VRAM/eviction logic is unchanged.

```
client ─HTTP/SSE─▶ front (public) ─┬─ engine#0  (CUDA_VISIBLE_DEVICES=0, :8780)
                   • no torch       ├─ engine#1  (CUDA_VISIBLE_DEVICES=1, :8781)
                   • always live    └─ …
```

See `docs/process-isolation-plans.md` for the design rationale (this is Plan B +
multi-engine).

## Modes

| Launch | Result |
|---|---|
| `coderai` (default) | Front on the public port; auto-spawns one engine per GPU |
| `coderai --single-process` | Legacy: one process, full app on the public port |
| `coderai --engine-only --internal-port N` | One engine on `127.0.0.1:N` (the front launches these for you) |

`--engine-only` is not meant to be run by hand; the front's supervisor manages it.

## Config (`config.json` → `server`)

| Key | Default | Meaning |
|---|---|---|
| `single_process` | `false` | Force legacy one-process mode |
| `internal_port_base` | `8780` | First engine's internal port (+1 per extra engine) |
| `engines` | `0` | Number of engines; `0` = auto (one per GPU, min 1) |
| `engine_gpus` | `null` | Explicit GPU indices, e.g. `[0, 1]`; `null` = auto-detect (NVIDIA) |
| `engine_specs` | `null` | Explicit heterogeneous engines (see below). Overrides `engines`/`engine_gpus` |
| `proxy_status_timeout` | `2.0` | Short timeout (s) for status/UI proxying |
| `proxy_max_inflight` | `64` | Max concurrent proxied requests through the front |

### Heterogeneous engines (e.g. NVIDIA + Radeon)

Auto-detection only finds NVIDIA cards and assumes one backend, and CUDA vs Vulkan
device **enumeration is inconsistent** — so for a mixed setup, declare each engine
with its own backend and env block via `engine_specs`. Each engine is its own
process: the front applies the env at spawn, forces the backend
(`CODERAI_ENGINE_BACKEND`), and routes models only to capability-compatible engines.

- **Capabilities** (default from backend): `nvidia` → `["transformers","gguf"]`
  (CUDA for transformers, GGUF via llama.cpp — which itself may use CUDA or Vulkan);
  `vulkan` → `["gguf"]`. Override per engine with `"capabilities": [...]`.
- **Routing:** a transformers/safetensors model goes only to a `transformers`-capable
  (NVIDIA) engine; a GGUF goes to whichever compatible engine already holds it, else
  the least-loaded GGUF-capable engine (NVIDIA *or* Radeon).

Example `config.json` → `server.engine_specs` for an NVIDIA (`cuda:0`) + Radeon
(Vulkan device 1) box, where the NVIDIA engine also serves GGUF via the NVIDIA
Vulkan ICD:

```json
"engine_specs": [
  {
    "name": "nvidia",
    "backend": "nvidia",
    "env": {
      "CUDA_VISIBLE_DEVICES": "0",
      "RADEON_VISIBLE_DEVICES": "",
      "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/nvidia_icd.json",
      "GGML_VK_VISIBLE_DEVICES": "0"
    }
  },
  {
    "name": "radeon",
    "backend": "vulkan",
    "env": {
      "CUDA_VISIBLE_DEVICES": "",
      "GGML_VK_VISIBLE_DEVICES": "1"
    }
  }
]
```

The first spec is the **primary** engine (owns admin/auth/config). Empty-string env
values are honoured (`CUDA_VISIBLE_DEVICES=""` hides all CUDA cards from the Radeon
engine). `internal_port_base` assigns ports in order (8780, 8781, …).

#### An engine can own several GPUs

"One engine per GPU" is only the auto-detect default. An engine owns whatever its
`env` exposes, so to run a single large model **across two NVIDIA cards**, give one
engine both — list both CUDA UUIDs — and the NVIDIA backend shards the model over
them automatically (`device_map`/accelerate `max_memory` across every visible CUDA
device; tune per-model with `max_gpu_percent` / `balanced_gpu_percent` / `max_vram`).

Example: 2× NVIDIA (one sharding engine) + 1× Radeon:

```json
"engine_specs": [
  {
    "name": "nvidia-dual",
    "backend": "nvidia",
    "env": {
      "CUDA_VISIBLE_DEVICES": "GPU-<uuidA>,GPU-<uuidB>",
      "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
      "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/nvidia_icd.json",
      "GGML_VK_VISIBLE_DEVICES": "0"
    }
  },
  {
    "name": "radeon",
    "backend": "vulkan",
    "env": {
      "CUDA_VISIBLE_DEVICES": "",
      "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/radeon_icd.json",
      "GGML_VK_VISIBLE_DEVICES": "0"
    }
  }
]
```

Use **GPU UUIDs** (from `nvidia-smi --query-gpu=uuid --format=csv`) rather than
indices so the assignment survives reboots/reordering. The front reports such an
engine's VRAM as the **sum across its GPUs** (with a per-device breakdown in
`/internal/engine-state` and `x_engines`).

## Choosing which card runs a model

When a model is compatible with more than one engine (e.g. a GGUF that runs on both
the NVIDIA and Radeon engines), the card is chosen by this precedence:

1. **Per-model pin** — set `engine` on the model (Models page → *Engine / card*, or
   the `"engine"` field in `models.json`) to a declared engine name. Honoured only
   if that engine can serve the model's format.
2. **Already resident** — the engine that already has the model loaded (avoids a
   reload).
3. **Default engine** — `server.default_engine` (Settings → *Default engine*), used
   when the model is compatible with several engines.
4. **Least-loaded** compatible engine.

`default_engine` and the per-model *Engine / card* control only appear in the UI
when 2+ engines are declared.

**Bad pins are reported, not silently ignored.** Saving a per-model engine (or the
default engine) that is unknown, or that can't run the model's format (e.g. a
transformers model pinned to a Vulkan/Radeon engine), returns a warning in the admin
UI. At request time the front also logs a one-line warning (deduped per
model+engine) before falling back to a compatible engine.

## Routing

- **Inference** (`POST /v1/...` carrying a `model`) → chosen per the precedence
  above, restricted to capability-compatible engines. This is what lets one model
  load on engine A while engine B keeps generating.
- **Admin / auth / config / UI / status / tasks** → the **primary** engine
  (engine#0). Sessions and `models.json` writes are per-process today, so pinning
  these keeps sessions consistent without a shared store.
- **Status / tasks pollers** use a short timeout with a cached/empty fallback, so a
  momentarily-blocked engine loop can never hang the dashboard. The front overlays
  cross-engine VRAM totals (`vram`) and running tasks (tagged with their `engine`).

## Thermal protection

Thermal cooldowns are scoped to match how work is distributed:

- **CPU too hot → everything pauses.** CPU temperature is read globally, and every
  engine gates on it, so all tasks back off until the CPU cools.
- **A GPU too hot → only that GPU's engine pauses.** Each engine reads only the
  cards it owns (the front sets `CODERAI_ENGINE_GPUS` — NVIDIA UUIDs and/or a vendor
  keyword), so a hot NVIDIA card pauses the NVIDIA engine while the Radeon engine
  keeps generating, and vice-versa. Each engine is its own process with its own
  cooldown state, so they're naturally independent.

Granularity is per-engine: if one engine owns several GPUs, a single hot card pauses
that engine's work on all of its cards (they share one process). In single-process
mode the GPU check covers all cards.

**Per-card thresholds.** Each card is judged against its own vendor's limit:
`thermal.gpu_high`/`gpu_resume` are the defaults, and `thermal.gpu_overrides`
(`{"amd": {"high": 95, "resume": 92}}`) raises/lowers them per vendor — so a Radeon
can run hotter than an NVIDIA card. Settings → Thermal renders one override row per
GPU vendor **detected on the machine** (never a hardcoded list).

**Which engine is cooling** is shown on the Tasks page banner (each engine reports
its cooldown via `/internal/engine-state`; the front names the cooling engine and
whether it's a GPU or CPU pause).

## Concurrency (per-engine)

Each engine is its own process with its own request queue, so concurrency limits
apply **per-engine** and total throughput is the sum across engines:

- **Max parallel requests** (`server.max_parallel_requests`) — how many requests an
  engine runs at once.
- **Max instances per model** (`models.max_model_instances`) — concurrent copies of
  one model (needed to run several requests against the *same* model at once).

Both take **per-engine overrides** (`*_overrides`, keyed by engine name, e.g.
`{"nvidia": 4, "radeon": 1}`) so a bigger card runs more in parallel than a smaller
one. Settings → Concurrency shows the defaults plus one override row per running
engine. The front resolves each engine's value and passes it down at spawn
(`CODERAI_MAX_PARALLEL` / `CODERAI_MAX_MODEL_INSTANCES`).

## Managing engines

The Tasks page shows an **Engines** panel (front mode only) with each engine's
health, VRAM and loaded-model count, and a **Restart** button — use it to kill an
engine that's wedged/looping; the supervisor respawns it immediately while the front
and other engines keep serving. Backed by `GET /admin/api/engines` and
`POST /admin/api/engines/{id}/restart` on the front (authorized against the primary
engine's session).

## Shared host-RAM cap

`offload.max_ram_gb` is a single **server-wide** ceiling shared by all engines, not
split into per-engine slices. The front sets `CODERAI_FRONT_PID` on each engine, so
every engine measures the same fleet-wide RSS (front + all engines + their workers)
and enforces the one cap against that total. When the combined usage crosses the
cap, each engine runs its normal mitigation/eviction (dropping its idle LRU models),
so whichever engine holds idle models frees them for the shared budget; busy models
aren't evicted. An idle engine uses ~0 of the budget; a busy one can use most of it.

VRAM is naturally per-card (each engine sees only its own GPUs via
`CUDA_VISIBLE_DEVICES`), and model eviction on swap is unchanged *within* an engine.

## Broker (runs in the front)

The AISBF broker client runs **in the front**, not in a model engine — it's
coordination/protocol work, so binding it to a GPU process would stall it whenever
that engine loads a model. Benefits:

- Never stalls during a model load (the front's loop is always free).
- One registration for the whole node, regardless of engine count.
- Advertises **aggregate** hardware: `build_hardware_summary` is torch-free in the
  front (via `gpu_stats()`), so it reports the total VRAM across *every* card.
- Brokered requests dispatch through the **same router/proxy** as HTTP — a brokered
  GGUF request can land on the Radeon engine, a transformers one on NVIDIA.

Engines run no broker client (`main.py` disables it under `--engine-only`); only
single-process mode keeps the broker in-process. Implementation:
`FrontProxy.start_broker` / `broker_execute` (`codai/frontproxy/app.py`) +
`execute_broker_request(..., executor=...)` (`codai/broker/dispatcher.py`).

## Model assignment (one owner per model)

With multiple engines, the front assigns each configured model to exactly **one**
owner engine and routes accordingly, so a model is never served from two engines:

- **Owner precedence:** per-model `engine` pin → default engine → balanced
  round-robin across capability-compatible engines.
- Routing honours the assignment first (`registry.engine_for_assigned`); unassigned
  / ad-hoc models fall back to capability routing.
- `/v1/models` (and the broker's model list) is the **union** across engines, deduped
  — the full catalogue with no duplicates.
- Engines aren't pruned, so the admin Models page (served from the primary) still
  shows the complete configuration.
- **Two configs of one model** can run on different engines if they have distinct
  aliases (the assignment keys on the routable id: alias → path); configs sharing a
  path with no distinct alias collapse to one owner.

## Security: engines are localhost-only + token-gated

Engines bind **127.0.0.1 only** (forced regardless of the configured host, which is
the front's public bind), and the front reaches them via `http://127.0.0.1:<port>`.
On top of that, the front generates a per-run secret, passes it to each engine via
`CODERAI_INTERNAL_TOKEN`, and stamps every engine request with an
`X-Coderai-Internal` header; an engine rejects (403) any request lacking it (and the
front strips client-supplied copies so the token can't be spoofed). So nothing else
on localhost can talk to an engine and bypass the front's auth/routing. Single-process
mode sets no token and is unaffected.

## Fault isolation

The supervisor polls each engine's auth-free, localhost-only
`/internal/engine-state`. If an engine exits (including a CUDA device-side assert),
it is **respawned**; the front and sibling engines keep serving. The front's own
`/healthz` reports per-engine readiness.

## Known limitations (follow-ups)

- Admin/config/session state is pinned to the primary engine (not yet replicated —
  that's "Plan C" in the design doc). Cross-engine **task visibility** works
  (merged read-only); cross-engine **session sharing** does not — all admin traffic
  intentionally lands on the primary.
- Placement is first-fit (model→least-loaded compatible engine); there is no live
  cross-engine rebalancing/migration yet.
- Capability routing keys off the model **name** (a `.gguf`/`gguf` name → GGUF, else
  transformers), matching the engine's own `is_gguf` heuristic. A transformers model
  whose name happens to contain "gguf" would be mis-routed — rename or declare an
  alias if that ever bites.
