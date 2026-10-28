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

## A restart mid-boot used to freeze the pool

Observed live 2026-10-14: six pods billing on the account, none in coderai's
interface, and this every fifteen seconds for seven minutes —

```
[runpod] another engine is booting a pod for 'datalab-to/surya-ocr-2' — waiting for it
```

Nothing was booting. A deploy had restarted coderai while a pod was mid-boot,
leaving that pod's URL-less registry entry behind with no owner. Two separate
decisions turned a dead marker into a frozen pool:

* `sibling_is_provisioning()` judged the entry's liveness against
  `REAP_GRACE_SECONDS` (1200s). That constant answers a different question — how
  long the reaper waits before killing an unowned pod — and the registry has a
  **60-second heartbeat**, so a dead owner is detectable in seconds. Now
  `PROVISION_STALE_SECONDS` (180s, three missed beats).
* the sibling check ran **before** `_adopt_shared_pod()`. So the recovery path
  for orphaned pods sat behind the marker that orphaning creates, and five
  running, registered, billing pods were never adopted. Adoption now runs first:
  a pod already serving this pool beats both waiting for a boot and renting.

Money, same incident: a pod registered at *creation* recorded no `$/hr` (the rate
was passed only at the ready-registration), so a boot interrupted before ready
could only ever be reaped with `the registry records no rate for it`. The price
is known at creation and is recorded there. And `hourly_usd` is now carried over
on re-registration like `image` and `gpu` were — an adoption that did not know
the rate used to turn a billing pod free.


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
    "network_volumes": {"EU-SE-1": "vol-a", "EU-RO-1": "vol-b"},
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
| `direct_tcp` | *auto* | **Auto = direct for coderai images, proxy for vLLM / llama.cpp / custom.** Talk to the pod at its public `ip:port` instead of through RunPod's HTTPS proxy. The proxy (`<pod>-<port>.proxy.runpod.net`) sits behind **Cloudflare**: TLS for free, but a 100 s idle limit per request — a slow *non-streaming* request (a long render, a cold 150 GB engine's first token) can be cut off with a 524 while the pod is still working — and a request-body cap. Direct TCP has neither limit. **coderai pods bring their own TLS on this path**: a certificate from this install's CA (`<config>/runpod/tls/ca.pem`, created on first use with `openssl`) is issued per pod and sent as environment; the pod serves HTTPS on it and this side verifies against that CA — so the token and the payloads stay encrypted. vLLM's and llama.cpp's images cannot take the certificate and stay plain HTTP on this path (the log says so). |
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
| `pod_max_parallel_requests` | `0` (the pod's own default, 2) | How many requests the **pod's** coderai server admits at once. The default is sized for a shared workstation and throttles a dedicated card long before its GPU is busy — requests over the limit get `429` while vLLM's `max_num_seqs` sits unused. Set it to roughly the remote engine's batch width. Rides to the pod as `CODERAI_SERVER_MAX_PARALLEL_REQUESTS`. |
| `pod_queue_max_size` | `0` (the pod's own default, 6) | How deep the pod queues admitted-but-waiting requests. Rides as `CODERAI_SERVER_QUEUE_MAX_SIZE`. |
| `data_center` | — | Pin this model's pods to one RunPod data centre (region), e.g. `EU-RO-1`. Blank uses the account-wide setting. A network volume still wins over both: a volume lives in one region and a pod elsewhere cannot attach it. |
| `network_volumes` | — | One network volume **per region**: `{"EU-SE-1": "vol-a", "EU-RO-1": "vol-b"}`, or `"EU-SE-1=vol-a,EU-RO-1=vol-b"`. This is how a model keeps BOTH a region fallback list and persistent storage — a single `network_volume_id` pins every pod to its own region and the `data_centers` list is discarded. The region is chosen first and that region's volume is attached. A region with no volume here is dropped from the list and said so, never rented without storage. Supersedes `network_volume_id`; a one-entry map behaves exactly like it. Falls back to the account-wide map. |
| `schedule_enabled` | `false` | Apply the warm floor (`min_pods`/`keep_warm`) only inside a daily window. Outside it the floor is `0` and the warm pod is terminated, so a model used in office hours stops billing overnight. It does **not** block requests: one arriving out of hours cold-starts a pod exactly as `min_pods: 0` always has. Ignored unless both `schedule_start` and `schedule_end` are set. |
| `schedule_start` / `schedule_end` | — | `HH:MM`. The start is inclusive, the end exclusive. An end **earlier** than the start runs overnight and belongs to the day it started on, so `22:00`–`06:00` on `fri` covers Friday night into Saturday morning. |
| `schedule_days` | every day | Which days the window opens on: `0`=Mon … `6`=Sun, or names (`mon,tue`). Empty means every day. |
| `schedule_tz` | server local | IANA zone for the window, e.g. `Europe/Rome`. An unknown name falls back to the server's local time rather than silently shifting to UTC. |
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

### The LLM servers as coderai pods

`engine: vllm` and `engine: llamacpp` run the **upstream** images
(`vllm/vllm-openai`, `ghcr.io/ggml-org/llama.cpp:server-cuda`): well-tested,
but not coderai — no TLS on the direct-TCP path, no `/boot`, no seeding, no
local LoRAs, nothing on the Tasks page. Two more images run the same servers
*inside* a coderai pod:

| `engine` | Image | What it is |
|---|---|---|
| `coderai-vllm` | `coderai-vllm` (light core + vLLM `+cu129` venv, 12.4 GB — PyPI's wheel is CUDA 13 and dies on RunPod's 12.8 drivers) | vLLM driven by coderai's vLLM backend. Your local **vLLM settings travel** (`dtype`, `quantization`, `gpu_memory_utilization`, `max_num_seqs`, `extra_args`); tensor parallelism follows the pod's `gpu_count`. |
| `coderai-llama` | `coderai-llama` (light core + llama-cpp-python CUDA, 5.3 GB; the GGUF backend asks NVML about the card, not torch) | A GGUF on coderai's own CUDA backend — llama-cpp-python compiled for every pod GPU generation. Local LoRAs work. |

`coderai-text` remains the transformers pod (HF safetensors, 4/8-bit, PEFT
adapters). Throughput under concurrency is vLLM's game on either image.

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
(`--kt-cpuinfer`, `--kt-weight-path` …). The image carries `gcc` because Triton
JIT-compiles SGLang's kernels on the pod. Note the kt backend has not been run
end to end on this machine either: the image is import-checked at build and
`/healthz`-checked at boot, and the first real inference will be on a pod.

**The engines are CPU-hungry too.** colibri's DeepSeek-V4 and Kimi-K3 engines
and `k3` stream experts through RAM by design (no CUDA path); pick a pod with
the RAM they need, not just the VRAM — `min_vram_gb` says nothing about RAM.

---

## 2b. Several LoRAs on one rented card

A LoRA is tens of megabytes of low-rank deltas over a base that may be tens of
gigabytes. Renting a card per adapter pays for the base weights again every
time; the point of a **pool** is that you do not.

Give the base and each adapter the same `runpod.pool`. One pod loads the base
once, vLLM registers every adapter at launch, and a request picks one by name:

```json
{"id": "my-base",    "runpod": {"pool": "team-a", "served_model": "my-base",
                                "hf_repo": "vendor/Some-27B"}}
{"id": "tuned-alpha", "loras": [{"name": "tuned-alpha", "path": "sha256:…"}],
                      "runpod": {"pool": "team-a", "served_model": "tuned-alpha",
                                 "hf_repo": "vendor/Some-27B"}}
```

```
POST /v1/chat/completions {"model": "tuned-alpha", …}
POST /v1/chat/completions {"model": "my-base",     …}
```

Both land on the same pod and the same loaded base. Requests naming *different*
adapters batch together in one forward pass — that is `--max-loras`, which the
launch sets explicitly because its default of 1 makes them serialise instead,
costing latency rather than raising an error.

### Sending an adapter from a client

An adapter trained elsewhere gets in with two calls. A PEFT adapter is a
DIRECTORY (`adapter_config.json` plus weights), so it travels as a tar — the
tar magic is detected, no flag to pass:

```bash
tar -cf adapter.tar -C /path/to/tuned-alpha .
curl -X POST $BASE/v1/loras/upload -H "Authorization: Bearer $TOKEN" \
     -F file=@adapter.tar
# -> {"id": "sha256:2e820c…", "bytes": 166821376, "existed": false}

curl -X POST $BASE/v1/loras/register -H "Authorization: Bearer $TOKEN" \
     -H 'content-type: application/json' -d '{
       "name": "tuned-alpha", "base": "my-base", "adapter": "sha256:2e820c…"}'
# -> {"ok": true, "state": "registered", "pool": "team-a"}
```

The store is content-addressed, so re-uploading an unchanged adapter costs one
existence check (`GET /v1/loras/blob/<hex>`, 200 or 404) instead of the bytes.
Registering inherits the base's pool, placement and budgets, and sets
`served_model` to the new name; it deliberately does NOT inherit `min_pods` or
`keep_warm`, because an adapter rides the base's pod rather than asking for one
of its own. Re-registering the same name with a new blob is how a retrain ships.

An adapter that lives in a HuggingFace repo needs no upload at all — give
`adapter` the repo id and the pod fetches it itself, at datacenter speed.

### Adding one to a pod that is already running

`--enable-lora` cannot be switched on after vLLM has started, so an engine that
launched without it must RELOAD the base model to accept its first adapter —
minutes, on a card billed by the hour. A pod told it serves a pool therefore
launches with LoRA enabled **even when no adapter is known yet**
(`CODERAI_VLLM_ENABLE_LORA`): the capacity flags are what must be decided at
launch, not the list of adapters.

With the door open, a new adapter is handed to the running engine
(`/v1/load_lora_adapter`) and is servable immediately, with no reload. Where
that is not possible — an engine started without `--enable-lora`, an older
vLLM, an adapter whose rank exceeds `--max-lora-rank` — coderai falls back to
restarting the engine with the adapter included, and says so in the log.

> A pod rented an hour ago may be running an older image than the one this
> install would launch today, so the runtime path is always tried and never
> assumed.

### Updating an adapter after a retrain

Register the same name again with the new blob. The reply says `"state":
"updated"` rather than `"registered"`, which is the confirmation that it
replaced an adapter instead of adding a second one.

Versioning is by CONTENT, which is what makes this safe to run on every
retrain: an adapter whose bytes have not changed is recognised as the same one
and nothing reloads. When they HAVE changed, the new bytes are sent to the pod,
the old adapter is dropped from the running engine and the new one takes its
name — the base model is not reloaded and nothing else on that pod is
interrupted.

Both halves of that needed a version in the key, and neither had one at first:

* the orchestrator cached "this pod already has this adapter" by NAME, so a
  retrain never sent the new bytes;
* and the engine was asked "do you serve this name?", which a stale adapter
  answers yes to. It kept the weights it was launched with and answered
  confidently from them. The update looked applied — the catalogue had it and
  the name worked — and the only visible symptom was output identical to the
  previous version.

An engine coderai has no record of is never guessed at: unloading an adapter on
a guess is worse than serving the one that is there.

## 3. Request flow

```
client ──▶ front proxy ──▶ primary engine ──▶ RunpodBackend ──▶ https://<public-ip>:<port>/v1   (coderai images: direct, pod TLS)
                                                            or https://<pod>-8000.proxy.runpod.net/v1 (vLLM / llama.cpp images)
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

"Not in a live pool" is decided from a shared registry file, and an entry in it
**shields** the pod it names from reaping — a sibling engine must not kill a pod
another process is managing. So every process heartbeats its own entries once a
minute, and an entry goes quiet for `CODERAI_RUNPOD_REAP_GRACE` (1200 s) when
its process is gone. Without that heartbeat an entry vouched for its pod for
the life of the install: observed 2026-10-13, where an unattended upgrade
restarted coderai and the surviving qwen38 pod was then both unmanaged (its
pool's warm floor was 0 outside its schedule window, so nothing adopted it and
nothing applied `idle_timeout_s` to it) and un-reapable (the dead process's
entry still vouched for it). It billed on. A pod the reaper kills is now also
written to the ledger, from the `created` timestamp in its entry — those hours
used to be invisible, because only a pool-tracked termination recorded spend.

One caveat that follows: a process whose maintenance loop is dead but which is
still running stops heartbeating, and a sibling will reap its pods after the
grace. That is the intended trade — twenty minutes of silence from the loop
that is supposed to manage a pod is itself the failure.

### Inheriting a pod instead of renting one

A pool reuses a pod already running for it — a sibling engine's, or its own from
before a restart — rather than renting a second card. Three things must match,
and all three are checked against the registry before the pod is taken over:

| must match | why |
|---|---|
| the pool (model) | it has to serve what the caller asked for |
| the **image** | the same model on a pinned tag and on `:latest` is different CODE, and inheriting the wrong one is invisible from outside |
| the **card** | a pod rented as a fallback is on a card the configuration accepts only when the preferred one is unavailable; inheriting it would serve a lease that asked for one card *and nothing else* on another |

The pod's own bearer token is adopted with it (it answers to nothing else), and
an entry recording neither image nor card — written before this check existed —
is refused rather than trusted: one extra pod once, against a silent answer from
the wrong code. Every refusal says which pod and why.

None of those three says anything about the pod's **catalogue**, and a *shared*
pool promises several names over one copy of the weights. A pod rented for one
of them has never heard of the others, so the first request naming a sibling
got a 404 — and because the pool keeps the pod warm, that 404 was permanent.
The pod is now taught instead: the adapter bytes are pushed to it, the entry is
registered, and the SAME request is retried on the SAME pod. Renting a second
card to serve a name the running one can hold is the answer this avoids.

```
[runpod] pod did not know 'tuned-alpha' (Model 'tuned-alpha' is not available.
         Use one of: base-awq) — taught it, retrying
```

Teaching is tried once per pod per model. A pod that still refuses afterwards
fails with its OWN message rather than with `404 Client Error: Not Found for
url: …`, which said nothing about the catalogue and read like a network fault.

#### A name is not a path

A LoRA and the base it adapts load from the same weights, so a model entry's
`path` cannot identify it — the adapter and its base have the same one. Identity
is **(weights, name)** everywhere a model is seeded, registered or served, and
every engine serves one set of weights under every name configured for it: the
base under its own, each adapter through vLLM's `--lora-modules`. Keying on the
path alone made the second name vanish into the first; keying the ENGINE on the
name as well would have started a second copy of the model on the same card.

Every pod's hours reach the ledger, whichever way it ends: a pool retires one
(`_terminate`), RunPod takes one away (`forget`), the reaper kills a stale one,
or it disappears entirely — terminated from the console, or straight through the
RunPod API. The last two have no pool to close the bill, so it is computed from
the registry's own `created` and recorded against the pool the pod belonged to,
marked `reaped` with the reason. A cost that is invisible cannot be budgeted
against, and the surprise arrives later, on the invoice.

Consequences worth knowing:

- Pods from **another** `deployment_id` are never touched, so two CoderAI
  instances can share one RunPod account safely — **as long as they use different
  deployment ids**. Two instances sharing an id will reap each other's pods.
- Pods you created **by hand** in the RunPod console are never touched (they don't
  carry the tag).
- The reaper runs only on the primary engine — the node that actually hosts the
  pools.

`atexit` also tears down every pool on a clean shutdown.

### Growing the fleet: demand, not one pod per boot

Scale-up is triggered in `acquire()`. It used to be gated on a boolean,
`_provisioning`, held for the whole provision — and a provision is a pod boot,
~12 minutes. So however much work was queued, **the fleet grew by at most one
pod per boot**: a burst needing four pods took forty minutes to be served by
four, whatever `max_pods` said. Measured on 2026-10-13 with 64 pages in flight
at `scale_up_inflight_per_pod: 16` (four pods of demand) and `max_pods: 64`: two
pods, ready ten minutes apart.

The grower also could not see the *size* of the demand. It asked
`min(p.inflight) >= threshold`, which answers "is the fleet full?" — so the only
request it could formulate was "one more". The requests blocked in `acquire()`
waiting for a slot, which are the actual measure of unmet demand, were invisible.

Both are now counts rather than flags:

- **`_booting`** — pods this process is provisioning right now. Booting pods
  count as capacity, so the same demand is never provisioned for twice.
- **`_waiting`** — callers blocked for a slot, added to what the pods carry to
  give the demand in pods: `ceil((inflight + waiting) / scale_up_inflight_per_pod)`.

Growth continues while `pods + booting < demand`, bounded by `max_pods` and by
**`scale_up_max_parallel_boots`** (default 4). That last one is a real limit and
not a formality: the pods the demand justifies are also the GPUs that start
billing together, so it is the knob that says how fast an operator is willing to
spend to absorb a burst. The per-pod cost caps still refuse each pod
individually.

The original single-flight trigger is kept beside it — every pod at its
in-flight threshold grows **one** more, single-flight, because
`max_inflight_per_pod` defaults to unlimited and requests are then never
blocked, so `_waiting` stays 0 and a merely-saturated fleet would never grow at
all. A prediction gets one pod; measured demand may get several.

Growth says so before it spends:

```
[runpod] 'datalab-to/surya-ocr-2': growing to meet 4 pod(s) of demand — 1 up, 2 booting
```

Two things found while fixing this, both worth knowing:

- **A pod's boot time bounds the first burst, not `max_pods`.** Twelve minutes
  pass before the first new pod answers anything, so a burst is served by
  whatever is already warm. Raising `max_pods` does not help it.
- **`acquire()` ignored its own timeout** when a *sibling process* was booting a
  pod for the same pool: it waited a flat 15 s and looped straight back into the
  same decision, never re-checking the deadline, for as long as the sibling's
  registry entry survived (up to `REAP_GRACE_SECONDS`, 15 minutes). It now
  raises when the caller's deadline passes.

### The same sweep reconciles the pools (a pod that dies behind our back)

The listing is not only used to find pods that should not exist. It is also the
only place a pod that **should** exist and no longer does becomes visible.

Nothing in a pool probes its own pods on a timer. A `PodHandle`'s `healthy` flag
goes false when something tried to **use** the pod and failed, or when the pool
itself retired it — which means a pod with no traffic on it is never discovered
dead. That is exactly a warm-floor pod. So a `keep_warm` pod terminated from the
RunPod console, reclaimed as spot, or lost with its host left the pool believing
its floor was satisfied, **indefinitely and silently**: `maintain()` counted the
handle, found `healthy >= min_pods`, and neither topped up nor logged. The first
real request then paid a full cold boot — around 12 minutes on the Digesta
orchestrator — which is the precise cost `keep_warm` is bought to avoid.

Each successful listing is now reconciled against what the pools believe:

- A pod RunPod reports as `TERMINATED`/`EXITED`/`DEAD`/`FAILED` is dropped
  immediately. That is RunPod telling us directly; a second opinion adds nothing.
- A pod simply **missing** from the listing is dropped only after it is missing
  from **two listings in a row** (~30s apart). The API is eventually consistent
  and pages; one response that omits a running pod used to be enough to
  unregister it, and believing it costs a second rented GPU for work already
  being done. A pod that reappears resets the count.
- A pod that **served a request in the last 60s** is kept whatever the listing
  says — traffic is better evidence than an API page. This cannot mask the bug
  above, because a warm-floor pod has no traffic to speak for it.
- A pod still being **provisioned** is shielded, and a **failed** listing never
  reaches the reconcile at all: `reap_orphans` returns early on a `RunpodError`,
  so an API outage cannot empty every pool and re-rent the fleet.

Dropping such a handle does **not** call the API — there is nothing left to
terminate — but it does close the bill: the ledger entry is written exactly as a
real termination writes it, estimated up to the moment we noticed, because a
pod that died out of band was billed right up to its death. A borrowed
(`(shared)`) pod is dropped without a ledger entry; its owner holds the bill.

Once the handle is gone, the next scaler pass sees the pool below its floor and
says so before it spends:

```
[runpod] pod qb9ap… for 'pool:digesta-qwen38' is gone from RunPod (status TERMINATED)
         — dropped from the pool; billed ~$0.412
[runpod] 'pool:digesta-qwen38': warm floor 1, 0 pod(s) healthy — renting one
```

A pool with `min_pods: 0` re-rents nothing: a pod dying out of band there is
simply gone, and renting a replacement nobody asked for is how an idle install
bills overnight.

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
| `GET /v1/runpod/spend` | The same pods, ledger, caps and schedule status over the **served** API, for an external monitor. Needs an API key marked `admin` (Admin → Tokens); an ordinary key gets `403`. |
| `GET /admin/api/runpod/pod-logs` | Container / vLLM log for one pod |

---

## 8. Several regions, and storage in each

RunPod ties a network volume to one data centre: a pod elsewhere cannot attach
it. So a single `network_volume_id` forces a choice nobody should have to make
— rent wherever a card is free, and re-download the weights onto every cold
pod, or keep the storage and wait for capacity in one region. On a contended
card that is the difference between a lease that fills and one that waits.

`network_volumes` removes the choice. Give each region its own volume, on the
account block or a model's:

```json
{"runpod": {"network_volumes": {"EU-SE-1": "vol-a", "EU-RO-1": "vol-b"},
            "data_centers": ["EU-SE-1", "EU-RO-1", "EU-FR-1"]}}
```

The scaler pairs every candidate card with every allowed region (as it already
did), and the volume now follows the region the candidate was ranked for. So in
the example above a pod in `EU-RO-1` attaches `vol-b`, and `EU-FR-1` is dropped
from the list with a line saying why — a pod rented there would have no storage,
re-download everything and boot minutes slower, which is the cost the volumes
were added to remove. If no allowed region has a volume, provisioning refuses
and names both sets rather than quietly placing the data somewhere else.

`network_volume_id` keeps working and keeps meaning exactly what it meant; a
one-entry map behaves identically to it. A model's map replaces the account's
whole, not merged: a model that lists its regions is stating where its data may
live, and inheriting an extra region from the account would put weights
somewhere it never named.

### Seeding a new region

A volume in a new region is empty, and the first pod there pays for the whole
download — the cost the volume exists to remove, landing on whoever happens to
send the first request. Pay it deliberately instead:

```
POST /admin/api/runpod/seed-volumes
{"model": "datalab-to/surya-ocr-2", "regions": ["EU-RO-1"]}
```

It boots one ordinary pod per region, in order, with that region's volume
attached — booting it *is* the download, because the caches already point at
the volume — then terminates each one. No new pod shape and nothing to
maintain. Sequential on purpose: N regions at once means N cards billing for
the same minutes, and nobody is waiting on this. Admin-only, it rents cards,
and the account's $/hr and spend caps still apply. It returns as soon as the
work starts; the per-region result goes to the server log.

The **venv** needs no seeding. `venv_on_volume` builds it under a lock with its
`.ready` marker written last, so the first pod in a region builds it once and
every pod after finds it — which is what makes the dependency-free slim image
usable in every region instead of pulling a 12–27 GB capability image.

### Why not a global volume

RunPod's **global** volume (beta) is region-independent and `global_volume_id`
supports it, but it cannot be the answer: the type cannot be created through
the public API at all (`POST /v2/network-volumes` accepts only
`STANDARD|HIGH_PERFORMANCE` and demands a `dataCenter`), so it exists only for
an operator who made one by hand in the console. It is also object storage —
no locking, no atomic rename, eventual consistency — which is fine for
read-mostly weights and wrong for anything that writes.

---

## 9. RunPod-only deployments

CoderAI can run as a pure orchestrator: **no local GPU, no local models, no
weights on disk.** Every node's capability set includes `runpod` (including an
explicit `cpu` node and an unrecognised backend), so a request for a RunPod model
always finds a host node instead of failing with "no engine can serve this".

Verified behaviour on such an instance: it boots, skips all local downloads, lists
the RunPod models in `/v1/models` with `backend: runpod`, starts the maintenance
loop and reaper, and forwards chat requests to RunPod.

---

## 10. Operational notes

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

---

## 11. Other GPU cloud providers — not built yet

RunPod is the only rentable-GPU provider CoderAI implements. The intent is for it
to be **one** of several: Scaleway, Exoscale, OVHcloud and Hyperstack all rent GPUs
by the hour or the second, and the same shape of integration should work for any of
them, as well as for providers not named here.

Nothing below is implemented. It is recorded so the next person does not have to
re-derive where the seam is.

### Where the seam already is

Most of what makes a rented GPU useful is **not** RunPod-specific. The pool logic —
demand-based growth, the stale-pod reaper, budgets and the ledger, spillover, shared
pools, LoRA fan-out, region selection — lives above the provider and asks it for a
small number of things. `RunpodClient` is effectively that interface already:

| Operation | What a second provider would have to answer |
|---|---|
| `ping` | credentials are valid |
| `list_gpu_types` / `pick_gpu` | catalogue with VRAM and $/hr, so price and size caps mean something |
| `create_pod` | run this image, expose this port, this much disk, these env vars |
| `get_pod` / `list_pods` | state, and the address once there is one |
| `wait_ready` | when the port answers — and *why* it never did |
| `stop_pod` / `terminate_pod` | stop billing, idempotently |
| `get_pod_logs` | the corpse's own log, for a pod that died before answering |
| `list_network_volumes` | per-region persistent storage, where offered |

A provider that can do those eight things can in principle be driven by the existing
fleet code. The work is a second client plus a way to choose one per pool — not a
second orchestrator.

### What is genuinely provider-specific

The traps in §10 were nearly all discovered live, and most of them are *categories*
of problem that any provider will have its own version of:

- **An image size ceiling.** RunPod refuses images above a line that is not
  documented (measured: 10.1 GB boots, 14.5 GB is killed two seconds after rent with
  no reason given). Every provider will have some limit; none of them will tell you
  what it is until an image stops booting.
- **What the boot budget has to cover.** The port appears only after the image is
  pulled, so a budget sized for "container already present" fails on cold machines
  and rents a fresh one on each retry.
- **How the port is reached.** RunPod offers both a proxy hostname and direct TCP,
  and they behave differently under TLS and keep-alive. Another provider may offer
  only one, or a load balancer with its own idle timeout.
- **Rate limits with their own units.** RunPod meters GraphQL per day and REST v2
  separately; coderai budgets calls against both. A provider metering per minute, or
  per account rather than per key, needs different accounting.
- **Terminal states that are not obviously terminal.** `EXITED` at uptime 0s is a
  dead pod, not a slow pull. Each provider words this differently, and guessing
  wrong means waiting out the whole budget on a container that died immediately.
- **Regional storage.** Volumes tend to be region-scoped, console-only, or both.

### Worth deciding early

- **Cost comparison is the point of multi-provider**, and it only works if the
  catalogue normalises: $/hr for a card of a given VRAM, in a given region, including
  whatever the provider charges for disk and egress. A pool that can rent from two
  providers should be able to prefer on price without the caller knowing which.
- **Credentials are per provider**, and the existing rules hold: the key is used from
  a variable, never printed, never committed, and never baked into an image.
- **A failed provider should degrade, not block.** The reaper reconciles from the
  account listing rather than probing, because a provider that is up but lying is the
  normal failure. That reasoning has to be re-done per provider, not inherited.
