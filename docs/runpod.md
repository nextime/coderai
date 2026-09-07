# RunPod — renting remote GPUs

CoderAI can serve a model on a **rented cloud GPU** instead of (or in addition to)
your local hardware, using [RunPod](https://runpod.io). The remote GPU is used
exactly like a local one: the model appears in `/v1/models`, answers
`/v1/chat/completions`, streams, and shows up in the Tasks page — the only
difference is that the weights never touch your disk and the GPU is billed by the
second.

Two modes, chosen **per model**:

| Mode | Who owns the lifecycle | Use it when |
|---|---|---|
| **Pods** | CoderAI — it provisions, health-checks, load-balances, scales and destroys full GPU containers | you want control over the GPU model, price, pool and idle timeout |
| **Serverless** | RunPod — it autoscales workers behind an endpoint you created in their console | you already have a serverless endpoint and just want CoderAI to proxy to it |

There is also **spillover**: a *local* model that bursts to RunPod only when the
local GPU can't take the request.

The RunPod backend uses **no local VRAM**. It runs as a thin HTTP proxy inside the
primary engine (the same shape as the `ds4` / `kt` / `vllm` HTTP backends) and
forwards OpenAI-format requests to the remote pod or endpoint. That also means a
CoderAI instance with **no local GPU and no local models at all** is a valid
deployment — see [RunPod-only deployments](#runpod-only-deployments).

---

## 1. Account setup

**Settings → RunPod** (the Settings page is split into category tabs; RunPod is
one of them).

| Field | Meaning |
|---|---|
| Enabled | Master switch. When off, nothing is provisioned and the maintenance loop does not run. |
| API key | Your RunPod API key. Masked in the UI, masked in every error message, never logged. |
| Default cloud type | `SECURE` or `COMMUNITY` — the default pool for models that don't pick one. |
| Default GPU type | Fallback RunPod `gpuTypeId` when a model doesn't name one. |
| Data center | Optional data-center id filter (blank = any). |
| Deployment id | A stable tag baked into every pod name. **Read [the reaper](#4-the-stale-pod-reaper) before changing this.** |
| Global max $/hr | Account-wide ceiling on the *sum* of all running RunPod pod rates. |
| Global spend cap + period | Account-wide cumulative budget (`hour`/`day`/`week`/`month`/`unlimited`). |
| API / REST / serverless base | Advanced overrides; leave at the defaults. |

Press **Test key & list GPUs** to verify the key and pull the live GPU catalogue
(ids, VRAM, on-demand and spot prices). Nothing is rented by this button.

In `config.json` the same block is:

```json
{
  "runpod": {
    "enabled": true,
    "api_key": "…",
    "cloud_type": "SECURE",
    "default_gpu_type": "",
    "data_center": "",
    "deployment_id": "default",
    "global_max_hourly_usd": 2.0,
    "global_cost_limit_usd": 100.0,
    "global_cost_period": "week"
  }
}
```

---

## 2. Putting a model on RunPod

Set the model's **backend** to `runpod` on the model page, then fill in the RunPod
sub-form that appears. In `models.json` that is a `backend` pin plus a `runpod`
block:

```json
{
  "text_models": [
    {
      "id": "cloud-qwen-72b",
      "backend": "runpod",
      "enabled": true,
      "runpod": {
        "mode": "pods",
        "served_model": "Qwen/Qwen3.5-72B-Instruct",
        "cloud_types": ["SECURE", "COMMUNITY"],
        "selection_criteria": "cheaper",
        "min_vram_gb": 48,
        "max_hourly_usd": 1.20,
        "allow_spot": false,
        "min_pods": 0,
        "max_pods": 2,
        "scale_up_inflight_per_pod": 4,
        "idle_timeout_s": 300,
        "boot_timeout_s": 300,
        "load_timeout_s": 600,
        "ctx": 32768,
        "cost_limit_usd": 50,
        "cost_period": "week"
      }
    }
  ]
}
```

### Per-model fields

**Mode**

| Key | Default | Meaning |
|---|---|---|
| `mode` | `pods` | `pods`, `serverless` or `auto` |

**GPU selection (pods)**

| Key | Default | Meaning |
|---|---|---|
| `cloud_types` | `["SECURE"]` | Pools to consider. Select both to widen availability. |
| `selection_criteria` | `cheaper` | `cheaper` = lowest price first. `faster` = SECURE before COMMUNITY, on-demand before spot, more VRAM, then price. |
| `gpu_type` | — | Pin an explicit RunPod `gpuTypeId` instead of searching. |
| `min_vram_gb` | `0` | Reject GPUs smaller than this. |
| `max_hourly_usd` | `0` (none) | Per-model $/hr ceiling for a single GPU. |
| `allow_spot` | `false` | Allow interruptible/spot instances (cheaper, can be reclaimed). |

**Serving (pods)**

| Key | Default | Meaning |
|---|---|---|
| `served_model` | — | The HF repo id the pod actually serves. **Required for pods.** |
| `image` | `vllm/vllm-openai:latest` | Container image. Must expose an OpenAI-compatible server. |
| `port` | `8000` | Port the server listens on inside the container. |
| `ctx` | — | Passed as vLLM's `--max-model-len`. |
| `container_disk_gb` | `40` | Must fit the image **and** the downloaded weights. |
| `volume_gb` | `0` | Optional persistent volume. |
| `env` | `{}` | Extra pod environment — put `HF_TOKEN` here for gated repos. |

**Scaling (pods)**

| Key | Default | Meaning |
|---|---|---|
| `min_pods` | `0` | Keep this many pods warm at all times. `0` = fully cold-start. |
| `max_pods` | `1` | Hard ceiling on concurrent pods for this model. |
| `scale_up_inflight_per_pod` | `4` | Add a pod when in-flight requests per pod exceeds this. |
| `idle_timeout_s` | `300` | Destroy a pod this long after its **last** request. |
| `boot_timeout_s` | `300` | Give up if the pod never exposes its port. |
| `load_timeout_s` | `600` | Give up if the server never answers `/v1/models`. |

`boot_timeout_s` and `load_timeout_s` exist because a 70B model can spend many
minutes pulling the image and downloading weights before it ever answers. Raise
them for big models; the defaults suit a ~7–14B model on a warm image.

**Serverless**

| Key | Default | Meaning |
|---|---|---|
| `endpoint_id` | — | An existing RunPod serverless endpoint id. **Required for serverless.** |
| `min_workers` / `max_workers` | `0` / `1` | Advisory worker bounds. |

**Cost**

| Key | Default | Meaning |
|---|---|---|
| `cost_limit_usd` | `0` (unlimited) | Cumulative spend budget for this model. |
| `cost_period` | `unlimited` | `hour`, `day`, `week`, `month` or `unlimited` — a **trailing rolling window**, not a calendar period. |

### Models that must never be downloaded locally

A model pinned to `backend: "runpod"` is skipped by the local download/cache path
entirely — CoderAI logs `Model 'X' is RunPod-served — skipping local download/cache`
and never fetches weights. So a RunPod model can be configured, enabled and served
on a machine that has neither the disk space nor the GPU for it.

---

## 3. Request flow

```
client ──▶ front proxy ──▶ primary engine ──▶ RunpodBackend ──▶ https://<pod>-8000.proxy.runpod.net/v1
                                                            └─▶ https://api.runpod.ai/v2/<eid>/openai/v1
```

**Cold start.** The first request for a cold pods-model provisions a pod, waits for
the port, then waits for the server to answer `/v1/models`, and only then forwards.
The request is held for the whole boot — expect minutes, not seconds, on a cold
large model. Subsequent requests reuse the same pod; the pool load-balances across
pods and destroys each one `idle_timeout_s` after its own last request.

**GPU fallback.** RunPod frequently reports a GPU as available and then refuses the
create with *"There are no longer any instances available"*. CoderAI ranks all
matching GPU candidates by the selection criteria and walks down the list on a
capacity miss, so provisioning survives a transient shortage.

**Stuck-boot retry.** If a pod is created but never becomes usable within its
timeouts, CoderAI dumps the pod's container/vLLM log (so an OOM or a bad
`--max-model-len` is visible rather than silent), terminates it, and retries on a
different machine — up to 3 attempts.

**The global concurrency gate does not apply.** A RunPod model is deliberately
exempt from the global `max_model_instances` admission gate: the remote GPU is not
a contended local resource, so a burst of RunPod traffic must not queue behind
local work. Per-model limits still apply.

---

## 4. The stale-pod reaper

A rented pod you forgot about bills forever. CoderAI therefore treats
"identify and destroy pods that should not exist" as a standing invariant, not a
cleanup step.

Every pod CoderAI creates is named `coderai-<deployment_id>-<model>-<random>`. A
maintenance loop on the primary engine runs every ~15s and, every other tick
(~30s, plus once at startup), lists the account's pods and **terminates any pod
carrying this deployment's tag that is not in a live pool and not currently being
provisioned**. That covers pods orphaned by a crash, a kill -9, or a container
restart mid-provision.

Consequences worth knowing:

- Pods from **another** `deployment_id` are never touched, so two CoderAI
  instances can share one RunPod account safely — **as long as they use different
  deployment ids**. Two instances sharing an id will reap each other's pods.
- Pods you created **by hand** in the RunPod console are never touched (they don't
  carry the tag).
- The reaper runs only on the primary engine — the node that actually hosts the
  pools.

`atexit` also tears down every pool on a clean shutdown.

---

## 5. Budgets and the ledger

Two independent guards:

- **Rate cap** (`max_hourly_usd` per model, `global_max_hourly_usd` account-wide) —
  refuses to *start* a pod whose price, added to the current running total, would
  exceed the ceiling.
- **Spend budget** (`cost_limit_usd` + `cost_period` per model, and the global
  pair) — refuses to start a pod once cumulative spend in the trailing window is
  exhausted.

Spend is persisted to `<config>/runpod_ledger.json`. A pod's final cost is booked
when it is torn down; the live cost of still-running pods is added on top for
enforcement and display, so nothing is double-counted.

When a budget blocks provisioning, the request fails rather than silently
overspending.

---

## 6. Spillover — local first, cloud on overflow

A **local** model can declare a RunPod burst target. The local GPU stays primary;
individual requests spill to the cloud only on the triggers you enable.

```json
{
  "id": "my-local-model",
  "backend": "gguf",
  "runpod_spillover": {
    "enabled": true,
    "on_concurrency_full": true,
    "on_no_gpu": true,
    "on_local_error": false,
    "target": { "mode": "serverless", "endpoint_id": "abc123", "served_model": "Qwen/Qwen3.5-72B-Instruct" }
  }
}
```

| Trigger | Fires when |
|---|---|
| `on_concurrency_full` | the local queue for this model is full |
| `on_no_gpu` | no local engine can serve the request (no GPU matching the model's needs) |
| `on_local_error` | the local attempt fails |

The spill is a guarded reverse-proxy in the front: it rewrites the request's
`model` to the target's `served_model`, adds the RunPod bearer token, and forwards.
If the spill itself fails, the request falls back to the local path rather than
erroring.

**Current limits (honest scope):** spillover is implemented on the front's direct
request path with a **serverless** target. A pods target, the broker path, the
streaming-concurrency trigger and `on_local_error` are not wired yet.

---

## 7. Observability

- **Admin → RunPod** — the stats page: live cost tiles, running pods (including
  ones still `provisioning…`), a per-pod console deep-link, a log viewer, and
  spend broken down by model. Polls every 10s.
- **Tasks page** — a RunPod engine box alongside the local engines, showing RunPod
  work in the same place as everything else. Appears only when RunPod is enabled.
- **Pod logs** — the stats page's "view" action fetches the pod's container/vLLM
  log. RunPod's log API is inconsistent, so CoderAI tries several routes and falls
  back to the console deep-link, which is always reliable.
- Every created pod logs its **pod id and console URL immediately**, before the
  boot wait — so a pod is traceable even if the boot then fails.

Admin API:

| Endpoint | Purpose |
|---|---|
| `GET /admin/api/runpod/gpu-types` | Live GPU catalogue + prices |
| `GET /admin/api/runpod/status` | Pools, pods and their states |
| `GET /admin/api/runpod/stats` | Cost tiles + per-model spend |
| `GET /admin/api/runpod/pod-logs` | Container / vLLM log for one pod |

---

## 8. RunPod-only deployments

CoderAI can run as a pure orchestrator: **no local GPU, no local models, no
weights on disk.** Every node's capability set includes `runpod` (including an
explicit `cpu` node and an unrecognised backend), so a request for a RunPod model
always finds a host node instead of failing with "no engine can serve this".

Verified behaviour on such an instance: it boots, skips all local downloads, lists
the RunPod models in `/v1/models` with `backend: runpod`, starts the maintenance
loop and reaper, and forwards chat requests to RunPod.

---

## 9. Operational notes

These come from live testing on a real account, not from the docs:

- **Community cloud is the flaky tier.** Expect "no longer any instances
  available" and pods that are created but never boot. Secure cloud booted
  reliably. Selecting both pools with `selection_criteria: "faster"` prefers
  Secure and falls back to Community only when needed.
- **Cold start is expensive in wall-clock, not money.** A successful cold pods
  request in testing took ~9.5 minutes end to end (through two unavailable GPUs
  and one failed boot) and cost about half a cent.
- **Set `idle_timeout_s` deliberately.** Too short and you re-pay the cold start;
  too long and you rent an idle GPU. For interactive use, a few minutes is sane;
  for batch work, `min_pods: 1` for the duration of the batch is cheaper than
  repeated cold starts.
- **Gated models need `HF_TOKEN` in the pod's `env`** — otherwise the pod boots,
  fails the download, and dies at `load_timeout_s`.
- `container_disk_gb` must fit the image *plus* the weights. Under-sizing it is a
  common cause of a pod that boots and then never answers.
