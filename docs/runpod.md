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

On the model page, under **Placement — where this model runs**, set **Runs on**
to *RunPod only* and fill in the RunPod block that appears. (It is the same
thing as choosing `runpod` in the compute-backend dropdown; the two stay in
sync.) In `models.json` that is a `backend` pin plus a `runpod` block:

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
| `gpu_count` | `1` | GPUs per pod (1–8). A multi-GPU pod is one machine with N cards: `min_vram_gb` is checked against the total, `max_hourly_usd` against the whole pod, and the engine is told to shard across every card (vLLM `--tensor-parallel-size N`, llama.cpp `--split-mode layer`). For a model too big for any single card. |
| `min_vram_gb` | `0` | Reject pods with less total VRAM than this (all `gpu_count` cards together). |
| `max_hourly_usd` | `0` (none) | Per-model $/hr ceiling for the whole pod (price × `gpu_count`). |
| `allow_spot` | `false` | Allow interruptible/spot instances (cheaper, can be reclaimed). |

**Serving (pods)**

| Key | Default | Meaning |
|---|---|---|
| `served_model` | — | The HF repo id the pod actually serves. **Required for pods.** |
| `image` | `vllm/vllm-openai:latest` | Container image. Must expose an OpenAI-compatible server. |
| `port` | `8000` | Port the server listens on inside the container. |
| `ctx` | — | Passed as vLLM's `--max-model-len`. |
| `container_disk_gb` | `40` | Grown automatically to fit the weights unless a network volume holds them; an explicit larger value is respected. |
| `weights_gb` | — | State the model's size and the estimate is skipped. The only honest answer for a URL download, an upload or a local file — the estimator cannot see those — and for an HF repo it beats a heuristic that errs high. Also accepted on the model entry itself. |
| `volume_gb` | `0` | Optional persistent volume. |
| `env` | `{}` | Extra pod environment — put `HF_TOKEN` here for gated repos. |

**Scaling (pods)**

| Key | Default | Meaning |
|---|---|---|
| `min_pods` | `0` | Keep this many pods warm at all times. `0` = fully cold-start. |
| `max_pods` | `1` | Hard ceiling on concurrent pods for this model. |
| `scale_up_inflight_per_pod` | `4` | Add a pod when in-flight requests per pod exceeds this. |
| `idle_timeout_s` | `300` | Destroy a pod this long after its **last** request. |
| `boot_timeout_s` | `900` | Give up if the pod never exposes its port. The port appears only *after* the image is pulled, so this covers the download — 15 GB at a cold machine's ~25 MB/s is ten minutes. |
| `load_timeout_s` | `600` | Give up if the server never answers `/v1/models`. |

`boot_timeout_s` and `load_timeout_s` exist because a 70B model can spend many
minutes pulling the image and downloading weights before it ever answers. Raise
them for big models. While waiting, the log reports status and uptime every 60s
rather than going silent, and a container that `EXITED` is failed fast with
RunPod's own reason instead of waited out.

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

### Engine pods — ds4, colibri, k3 on a rented GPU

> ⚠ **The weights are the cost, not the GPU.** Models on these engines are
> 100 GB and up — DeepSeek-V4 is ~154 GB, Kimi-K3 is measured in terabytes.
> A pod with **no network volume downloads all of it on every cold start**,
> which is typically one to two hours of rental per boot, before it answers
> anything. The model page shows this warning the moment you pick one of these
> engines; the provisioning log repeats it; the test run reports it. Attach a
> network volume, and put the weights on it once.

The native MoE engines run on RunPod through the **engines image**
(`ghcr.io/nextime/coderai-engines`), a coderai pod that carries `ds4-server`,
colibri's `colibri` / `deepseek_v4` / `kimi_k3` and kimi-k3-in-c's `k3`. They
are **compiled in the image**, not copied from your machine: the local
`ds4-server` is built for the one card here (sm_86) against the host's CUDA 13,
and the local colibri has no GPU backend at all — neither would run on an A100.
The image builds them in a CUDA 12.8 toolchain stage for every generation RunPod
rents (sm_80 A100, sm_86, sm_89 L40S/4090, sm_90 H100, sm_120a RTX 50 / RTX PRO
Blackwell) and a portable x86-64-v3 CPU baseline, then keeps only the binaries.
The runtime side is the light core plus NVIDIA's cuda-runtime and cuBLAS 12.8
wheels — the engines are C, coderai only drives them, so there is no torch in
it: 2.4 GB, the smallest capability image.

```bash
./packaging/runpod/build_capability_image.sh engines ghcr.io/nextime/coderai-engines:0.2.18
```

It stages the engine sources from `~/.coderai/{ds4,colibri,kimi-k3-in-c}` (or
`CODERAI_DS4_DIR` etc.), so the pod runs the same engine code the local install
does.

**Naming the engine.** A RunPod-only model has `backend: runpod` — that slot is
taken — so the engine goes on the runpod block, beside `vllm` and `llamacpp`:

```json
{ "path": "DeepSeek-V4-Pro-Q4K.gguf", "backend": "runpod",
  "runpod": { "engine": "ds4", "network_volume_id": "abc123",
              "volume_path": "models/DeepSeek-V4-Pro-Q4K.gguf",
              "min_vram_gb": 80, "boot_timeout_s": 900, "load_timeout_s": 3600 } }
```

A *local* model pinned to `backend: ds4` / `colibri` / `k3` that bursts to
RunPod carries the pin itself; either way it lands on the engines image with
that engine switched on (`CODERAI_DS4_ENABLED=1`) and **this install's settings
for it** forwarded as `CODERAI_DS4_CONFIG` — context, expert cache, extra args,
the ds4 download variant — minus anything local (install dir, ports, paths).
`ds4.auto_download` is forced on for the pod: a pod that cannot fetch its own
weights answers nothing.

**Where the weights come from, per engine:**

| Engine | On the pod | Without `volume_path` |
|---|---|---|
| `ds4` | loads `volume_path` if set | runs `download_model.sh <model_variant>` into `<volume>/cache/ds4` (or the container disk — see the warning) |
| `colibri` | loads the container directory at `volume_path` | **never downloads**: fails with "no model container resolved" |
| `k3` | loads the checkpoint directory at `volume_path` | **never downloads**: the checkpoint is ~1.56 TB |
| `kt` | loads the HF model directory at `volume_path` | SGLang downloads the HF repo id into the volume's HF cache |

`volume_path` is relative to the volume mount (`/workspace` by default) or
absolute. Put the weights there once — from a pod, with `scp` through the RunPod
console, or by letting one ds4 pod download and every later one attach.

**ktransformers has an image of its own**, `coderai-engines-kt`: SGLang with
the kt integration plus `kt-kernel`, in a venv of their own on the light core
(they pin their own torch). `kt-kernel` dispatches at runtime — AMX, AVX-512,
AVX2 (llamafile), AMD BLIS — so one image serves whatever CPU the pod host
has; AMX hosts are the fast ones. `engine: kt` on the runpod block selects it;
the pod launches SGLang from that venv (`CODERAI_KT_VENV`). The model path may
be a HuggingFace repo id — SGLang downloads it into the volume's HF cache — or a
directory on the volume; `ktransformers.extra_args` travels for the KT knobs
(`--kt-cpuinfer`, `--kt-weight-path` …). Note the kt backend has not been run
end to end on this machine either: the image is import-checked at build and
`/healthz`-checked at boot, and the first real inference will be on a pod.

**The engines are CPU-hungry too.** colibri's DeepSeek-V4 and Kimi-K3 engines
and `k3` stream experts through RAM by design (no CUDA path); pick a pod with
the RAM they need, not just the VRAM — `min_vram_gb` says nothing about RAM.

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
- **There is no RunPod pod-log API.** This was checked against their published
  OpenAPI spec (`rest.runpod.io/v1/openapi.json`): 23 routes, covering pods,
  billing, endpoints and volumes, and not one of them serves logs. GraphQL
  refuses introspection and its documented `Pod` type exposes no log field
  either. The web console is a human surface, not an API. Code that "tried
  several routes" was guessing at URLs that never existed, which is why it
  always ended in `HTTP 400`.
- **A coderai pod reports on itself instead.** `packaging/runpod/boot.sh` is the
  entrypoint and prints timestamped phases from the first instant the container
  exists:

  ```
  [boot +0s] container started
  [boot +0s] profile=embeddings image=…
  [boot +0s] seed models: [{"path":"BAAI/bge-m3",…}]   ← or NONE
  [boot +1s] gpu: NVIDIA RTX 2000 Ada, 16376 MiB
  [boot +1s] starting uvicorn on 0.0.0.0:8000
  ```

  The same record — plus the application's own phases — is served at **`GET
  /boot`**, readable the instant the port opens and long before the pod is
  useful. A failed boot asks the pod for it before falling back to a console
  link. That seed line alone identifies the most common pod failure ("Model 'x'
  is not available. Use one of: " with nothing after the colon) in one look.
- **The pull window is diagnosed from outside**, because no code of ours is
  running during it: `wait_ready` reports status and uptime every 60 seconds
  (`still no port after 180s (status=RUNNING uptime=42s) — image pull in
  progress`) rather than going silent for minutes.
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
- **The port appears only after the image is pulled**, so the boot budget has to
  cover the download. This was originally 300s on the assumption the container
  was already there, and it broke exactly as you would expect: the same 7.3 GB
  capability image opened its port in **133s** on a machine that had the layers
  cached, and failed repeatedly at 300s on cold ones — each failure renting a
  *fresh* machine that pulled the whole image again, three times over. The
  default is now **900s** (`boot_timeout_s` per model). Being wrong by fifteen
  minutes on a $0.25/hr card costs six cents; being wrong by too little cost
  three pods and still failed.
- **RunPod refuses images above a size line.** Measured: a 10.1 GB image boots;
  a 14.5 GB one is `Exited by Runpod` two seconds after rent, on every machine,
  with no other reason given. Container disk (tried 80 GB), layer format and
  the full image config were each ruled out — size was the only difference.
  The specialised capability images sit at 6.9–8.4 GB on a torch-free core for
  exactly this reason; the ordinary ones at 8.9–10.1 GB are close to the line,
  and `ocr` (9.7 GB) is the one to watch if anything is added to it.
- **`EXITED` at uptime 0s is a dead pod, not a slow pull.** `wait_ready` used to
  treat only `TERMINATED`/`FAILED` as terminal and waited out the whole budget
  on a container that had died in its first second — three machines in a row,
  printing "image pull in progress". It now fails fast past a short restart
  grace and reports RunPod's own `lastStatusChange`, the one field that says
  why. "Exited by Runpod" means the platform refused the image; "exited with
  code N" means the process died.
- **The container disk is sized for the weights before renting.** A pod that
  fills its 40 GB disk mid-download dies at `load_timeout_s` having paid for
  every minute. One HuggingFace metadata call gives the size; the disk grows to
  weights + ~25 GB (unpacked image and download cache) and never shrinks an
  explicit `container_disk_gb`. It counts what a load will *fetch*, not the
  whole repo — SDXL's repo is 77 GB, of which a diffusers load pulls 28; the
  rest is the same weights again as Flax, legacy `.bin` and ONNX. With a
  network volume attached the weights land there, so the disk stays put.
- **The build disk can fill with layers Docker cannot see.** Twice, 500 GB of
  overlay2 belonged to old flattened bases pinned by container mount records
  the daemon had lost; `docker system df` said 90 GB. `sudo
  tools/docker_leak_audit.py` reports them, `--fix` removes them with the
  daemon stopped. Push a capability image and delete it locally — every one
  of them is rebuildable from the two cores.
- **A machine may serve a cached `:latest`.** An images pod reported a
  dependency missing that had already been published under that tag. Pin an
  immutable version tag when a result has to mean something — the test harness
  takes `--image ghcr.io/nextime/coderai-<capability>:<version>` for this, while
  production keeps following `:latest`.
- **Set `idle_timeout_s` deliberately.** Too short and you re-pay the cold start;
  too long and you rent an idle GPU. For interactive use, a few minutes is sane;
  for batch work, `min_pods: 1` for the duration of the batch is cheaper than
  repeated cold starts.
- **Gated models need `HF_TOKEN` in the pod's `env`** — otherwise the pod boots,
  fails the download, and dies at `load_timeout_s`.
- `container_disk_gb` must fit the image *plus* the weights. Under-sizing it is a
  common cause of a pod that boots and then never answers.
