# CoderAI as a GPU-less RunPod orchestrator

A practical guide for an operator who has to **install** CoderAI on a machine with
no GPU, point it at RunPod, register a handful of models, let them **scale**, and
**watch what they cost**. Written for the Digesta deployment; nothing here is
Digesta-specific.

The shape of the thing: **one CoderAI install — the full image — on your own
host** answers `/v1/*` and serves the admin GUI. It needs no GPU. When a request
arrives for a model it rents a RunPod pod, waits for it to come up, proxies to
it, load-balances across pods, destroys them when they go idle, and bills every
hour into a local ledger with caps you set. Your callers only ever talk to the
orchestrator; the pods are its business, not theirs.

```
 Digesta ──► CoderAI orchestrator ─────────────────────► RunPod pods (A40, …)
             ghcr.io/nextime/coderai                       ghcr.io/nextime/
             (the FULL image, port 8776)                    coderai-<capability>
             /v1/chat/completions                          or vLLM / llama.cpp
             /v1/embeddings
             /v1/ocr                                       rented, configured,
             /admin  (GUI)                                  routed and destroyed
                                                            BY the orchestrator
```

Those two columns are different software with different jobs. You install the
left one. You never install the right one.

---

## 1. Install

> **Install the FULL image. Never a capability image.**
>
> `ghcr.io/nextime/coderai` is the orchestrator: it has the admin GUI, the
> supervised engine, the RunPod pools, the ledger and the reaper.
> `ghcr.io/nextime/coderai-<capability>` images are **pods** — what CoderAI
> rents and launches *for* you. Running one by hand as your front is the single
> most expensive wrong turn available here: a pod image has no GUI, and the
> admin pages are not behind the API's bearer check, so it is not something to
> expose either. You pull a capability image exactly never; CoderAI pulls them
> on RunPod.

```bash
docker pull ghcr.io/nextime/coderai:latest
```

Verify the signature before you run it — every published image is signed:

```bash
cosign verify \
  --key https://raw.githubusercontent.com/nextime/coderai/master/packaging/cosign.pub \
  ghcr.io/nextime/coderai:latest
```

Run it. Two mounts matter: a **config** directory (settings, model list, API
tokens — keep it on persistent storage and back it up) and a **models/cache**
directory. The full image listens on **8776**.

```bash
mkdir -p ~/coderai/config ~/coderai/models ~/coderai/cache

docker run -d --name coderai --restart unless-stopped \
  -p 8776:8776 \
  -e CODERAI_CONFIG_DIR=/config \
  -v ~/coderai/config:/config \
  -v ~/coderai/models:/models \
  -v ~/coderai/cache:/cache \
  ghcr.io/nextime/coderai:latest
```

> ### ⚠ Your config file goes in a **subdirectory**, and this costs people hours
>
> The entrypoint does not use `$CODERAI_CONFIG_DIR` directly. It creates
> **`$CODERAI_CONFIG_DIR/coderai/`** and symlinks `~/.coderai` to it, because the
> app resolves its config from the HOME-style path
> (`coderai-entrypoint` lines 49 and 70). So with the mount above, the file you
> must edit is:
>
> ```
> ~/coderai/config/coderai/config.json      ← the one that is read
> ~/coderai/config/config.json              ← IGNORED. Silently.
> ```
>
> Put settings in the parent and nothing happens: on first start the app writes
> a **default** `coderai/config.json` with `backend: null`, `engine_specs: null`
> and `engines: null`, reads that, and your carefully set `backend.type: "cpu"`
> is never seen — so the engine auto-detects, finds no GPU and aborts with
> `No supported backend detected`. It looks exactly like the front ignoring
> `engine_specs`. It isn't; the file was never read.
>
> Verify which file is live before you debug anything else:
>
> ```bash
> docker exec coderai sh -c 'readlink -f ~/.coderai; cat ~/.coderai/config.json'
> ```
>
> The same applies to `auth.json`, `models.json` and the rest — they all live in
> that subdirectory. Mounting your config directly at `/config/coderai` also
> works and avoids the whole question.

On a machine **with** a GPU add `--gpus all` (NVIDIA) or `--device /dev/dri`
(AMD/Intel, Vulkan) and you are done.

### On a host with no GPU

This is the normal shape for an orchestrator, and it needs two settings — not
because anything is broken, but because the defaults assume you have a card.

In `<config>/config.json`:

```json
"backend":  { "type": "cpu" },
"server":   { "engine_specs": [ { "name": "cpu", "backend": "cpu", "primary": true } ] }
```

Both matter, and here is exactly why:

- **`backend.type` must be `cpu`, not `auto`.** With `auto`, `main.py` looks for
  NVIDIA, then Vulkan, then OpenCL, and on finding none exits with
  `Error: No supported backend detected`. Set explicitly to `cpu`, that search
  is skipped entirely and the engine starts.
- **`engine_specs` must name the engine explicitly.** Left to auto-detect, the
  front creates a no-GPU engine named `cpu` but launches it with
  `backend="auto"` (`engine_supervisor.py:311`) — which hits the abort above
  regardless of what `backend.type` says. The log then reads `[cpu] Available
  backends: {'cpu': True}` followed immediately by `No supported backend
  detected`, which looks self-contradictory until you know that `[cpu]` is the
  engine's *name* and `auto` was its *backend*. An explicit spec is what makes
  the two agree.

> **`server.engines = 0` does not mean "no engines".** It means **auto** — one
> per detected GPU, minimum 1 (`config.py:70`). There is no setting for "no
> local engine", and you do not want one: the RunPod pools, the scaler and the
> stale-pod reaper all live *inside* the primary engine. One idle CPU engine is
> not a workaround, it is the component that manages your pods.

With those two settings the engine comes up, registers nothing locally, and
every model you configure with `backend: runpod` is served from a rented pod.

Check it:

```bash
curl -fsS http://127.0.0.1:8776/healthz     # {"ok":true,"pid":…}
curl -fsS http://127.0.0.1:8776/v1/models   # [] until you add models
```

Then open `http://127.0.0.1:8776/admin`. It redirects to `/login`; sign in with
the credentials from your config (`<config>/auth.json`), change the password,
and create an API token (Admin → Tokens) before anything calls `/v1/*`.

### Behind nginx, under a sub-path

The app honours `X-Forwarded-Prefix`, so it can live under `/coderai/` without
rewriting links:

```nginx
location ^~ /coderai/ {
    proxy_pass         http://127.0.0.1:8776/;
    proxy_set_header   Host              $host;
    proxy_set_header   X-Forwarded-Prefix /coderai;
    proxy_set_header   X-Forwarded-Proto $scheme;
    proxy_read_timeout 3600s;   # generation is slow; do not cut it off
}
```

`proxy_read_timeout` matters: a long render or a cold engine's first token can
take minutes, and nginx's 60 s default will kill it mid-answer.

**Three things will bite you on this path.**

**1. `auth_basic` and Bearer tokens collide.** They are the same HTTP header.
If you put `auth_basic` on `/coderai/`, nginx demands `Authorization: Basic …`
and rejects a client that sends `Authorization: Bearer sk-coderai-…` with 401
*before* the request ever reaches CoderAI. So basic-auth the **GUI** and leave
the **API** on its own tokens:

```nginx
# The GUI: a browser, protected by basic auth.
location ^~ /coderai/ {
    auth_basic           "coderai";
    auth_basic_user_file /etc/nginx/coderai.htpasswd;
    proxy_pass           http://127.0.0.1:8776/;
    proxy_set_header     Host               $host;
    proxy_set_header     X-Forwarded-Prefix /coderai;
    proxy_set_header     X-Forwarded-Proto  $scheme;
    proxy_read_timeout   3600s;
}

# The API: no basic auth — CoderAI's own Bearer tokens guard it.
location ^~ /coderai/v1/ {
    auth_basic           off;
    proxy_pass           http://127.0.0.1:8776/v1/;
    proxy_set_header     Host               $host;
    proxy_set_header     X-Forwarded-Prefix /coderai;
    proxy_set_header     X-Forwarded-Proto  $scheme;
    proxy_read_timeout   3600s;
    proxy_buffering      off;     # or streamed responses arrive in one lump
}
```

`proxy_buffering off` on the API path matters for streaming: with buffering on,
nginx holds a token-by-token response and delivers it all at once.

**2. Leave the readiness probes unauthenticated.** `/healthz`, `/health` and
`/v1/health` must answer 200 without credentials. Every health checker — and
surya-ocr's vLLM client — reads a 401 as *backend down*. They are deliberately
exempt from the Bearer check inside CoderAI; do not put basic auth in front of
them either.

**3. Make sure your API tokens are actually enforced.** CoderAI's Bearer check
falls **open** when it has no user database to check against and no
`CODERAI_API_TOKEN` in the environment — reasonable on a loopback workstation,
dangerous the moment the app is published. Before 0.2.83 a bare
`uvicorn codai.api.app:app` start did not initialise that database, so `/v1/*`
was served *without credentials*. From 0.2.83 the app initialises it from the
config dir at import and logs `[api] API tokens enforced from <dir>`.

Verify it yourself, from the host, before you expose anything:

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8776/v1/models
#   401  <- correct
#   200  <- OPEN: upgrade to 0.2.83, or set CODERAI_API_TOKEN and restart
```

On an older image, `-e CODERAI_API_TOKEN=<something-long>` closes it: with an
env token configured the middleware refuses anything that does not match.

**The GUI itself works under a sub-path** — the login form posts to
`/coderai/login`, assets resolve under `/coderai/static/admin/`, and the page's
`ROOT_PATH` is `/coderai`, so no nginx rewriting is needed.

---

## 2. Point it at RunPod

Admin GUI → **Settings → RunPod**, or edit `<config>/config.json`:

```json
"runpod": {
  "enabled": true,
  "api_key": "rpa_…",
  "cloud_type": "SECURE",
  "data_center": "",
  "global_max_hourly_usd": 20.0,
  "global_cost_limit_usd": 40.0,
  "global_cost_period": "day"
}
```

Set the two global caps **before** you add a model. They are the backstop: the
orchestrator refuses to provision when the next pod would push the account over
`global_max_hourly_usd`, or when spend in the trailing `global_cost_period`
window has reached `global_cost_limit_usd`. A trailing window, not a calendar
one.

`data_center` pins every pod to one region; leave it blank and set the region
per model instead (§3).

> The API key is account-wide and can rent machines. Treat it as a secret, keep
> it out of git, and rotate it if it is ever printed.

---

## 3. Register a model

Admin GUI → **Models → Add**, set **backend: runpod**, then fill the RunPod
block. Everything below is also a key in `<config>/models.json` under the
model's `runpod` object, and everything below is editable from the model page.

A text model on vLLM:

```json
{
  "name": "Qwen/Qwen3-8B-Instruct",
  "backend": "runpod",
  "runpod": {
    "mode": "pods",
    "engine": "vllm",
    "served_model": "Qwen/Qwen3-8B-Instruct",
    "min_vram_gb": 48,
    "max_hourly_usd": 1.50,
    "gpu_count": 1,
    "cloud_types": ["SECURE"],
    "data_center": "EU-SE-1",
    "direct_tcp": "auto",

    "min_pods": 0,
    "max_pods": 4,
    "scale_up_inflight_per_pod": 8,
    "idle_timeout_s": 300,
    "boot_timeout_s": 900,
    "load_timeout_s": 600,

    "pod_max_parallel_requests": 48,
    "pod_queue_max_size": 256,

    "cost_limit_usd": 50,
    "cost_period": "day"
  }
}
```

What each group does:

**Choosing the machine.** `min_vram_gb` and `max_hourly_usd` are the limits the
search works inside — `max_hourly_usd` is for the **whole pod**, so it already
accounts for `gpu_count`. `gpu_type` pins an explicit RunPod `gpuTypeId` if you
would rather not search. `selection_criteria` is `cheaper` (default) or
`faster`. `allow_spot` lets interruptible machines in: cheaper, reclaimable.

**Reaching it.** `direct_tcp: "auto"` is right for CoderAI images: it talks to
the pod at its public `ip:port`, and the pod serves HTTPS with a certificate from
this install's own CA. **Prefer this over RunPod's proxy**, which sits behind
Cloudflare and cuts any non-streaming request at 100 s — a long OCR page or a
cold model's first token will fail with a 524 while the pod is still working.
vLLM's and llama.cpp's own images cannot take the certificate and stay on plain
HTTP, which the log says plainly.

**Region.** `data_center` pins this model's pods. A network volume overrides it,
because a volume lives in one region and a pod elsewhere cannot attach it.

---

## 4. A network volume, and one recipe per kind of model

### You never configure a pod — CoderAI does

This is the part worth being explicit about, because it is the easiest thing to
get wrong: **you do not set anything up on a pod, and you do not send requests
to one.** A pod is disposable. CoderAI rents it, configures it, routes to it and
destroys it. Everything you set is on the *model*, here, in `models.json` or the
model page.

When CoderAI provisions a pod it sets, by itself:

| on the pod | from where |
|---|---|
| its **bearer token** — a fresh `cra-…` per pool, so a pod is never open on a public proxy URL | generated unless you set `api_key`, or opt out with `allow_open_pod` |
| **TLS**, when the path is direct TCP — a certificate from this install's own CA, which only this install verifies | automatic for CoderAI images |
| which **models** to serve, and where to get the weights | `served_model`, `source`, `hf_repo`, `volume_path` |
| the **engine** and its launch arguments, including `--tensor-parallel-size`, `--enable-lora` and the LoRA modules | `engine`, `gpu_count`, the adapters on the model |
| its **admission limits** | `pod_max_parallel_requests`, `pod_queue_max_size` |
| the **volume** mount | `network_volume_id`, `volume_mount_path` |
| **teardown** — idle pods destroyed, and a reaper that kills tagged pods no live pool is tracking, so a crash cannot leave one billing | `idle_timeout_s`, automatic |

So: no SSH into a pod, no editing config on a pod, no pointing a client at a pod
URL. If you find yourself wanting to drive a pod directly to get throughput, the
answer is `max_pods` and `scale_up_inflight_per_pod` instead — client-side
concurrency against one pod produces 429s, not speed.

The one thing to keep in your own hands is the **orchestrator's** front door:
its API tokens (Admin → Tokens) and what your reverse proxy exposes.

### Put the weights on a volume — this is the single biggest lever

**The weights are the cost, not the GPU.** A pod with no network volume
downloads its model on **every cold start**, and you pay rental for all of it
before it answers anything. A 15 GB repo at a cold machine's ~25 MB/s is ten
minutes; a 150 GB MoE is one to two hours *per boot*. Attach a RunPod network
volume, put the weights on it once, and a cold start becomes a mount.

Create the volume in the RunPod console (Storage → Network Volume), **in the
region you intend to rent in**, then give its id to the model:

```json
"runpod": {
  "network_volume_id": "abc123xyz",
  "volume_mount_path": "/workspace",
  "volume_path": "models/Qwen3-8B-Instruct"
}
```

- `network_volume_id` — the volume. Volumes are **Secure Cloud only**, so
  naming one pins `cloud_types` to `["SECURE"]` automatically.
- `volume_mount_path` — where it appears in the pod. `/workspace` by default.
- `volume_path` — where the weights are **on** the volume, relative to the
  mount (or absolute). Leave it unset and the engine downloads into the volume's
  HF cache instead, which still beats the container disk because the next pod
  reuses it.
- `container_disk_gb` — grown automatically to fit the weights **unless** a
  volume holds them. With a volume you can leave it at the default.

**The region is not optional once a volume exists.** A volume lives in one data
centre and a pod elsewhere cannot attach it, so the volume's region overrides
both `data_center` and the account setting. If you want pods in two regions, you
need a volume in each.

A practical layout for a shared volume:

```
/workspace/
  models/            weights, one directory per model
  loras/             adapters (PEFT dirs, or GGUF-converted for llama.cpp)
  hf/                HF cache for anything downloaded on demand
  venvs/             engine venvs, when you use venv_on_volume
```

### Share one volume — and one pool — across models

Two models naming the same `pool` share the **same pods** instead of renting a
card each. This only works when the pod's server can serve more than one model:
a **CoderAI pod** picks the model per request, while a vLLM pod is launched
`--model X` and can only ever serve that one. So pool CoderAI-engine models
(`engine: coderai`, e.g. the capability images) and leave vLLM/llama.cpp models
on their own pods.

```json
"runpod": { "pool": "digesta-cpu-bound", "engine": "coderai", "max_pods": 3 }
```

With `venv_on_volume: true` the pod's Python dependencies live on the volume and
a small image boots and uses them: the first pod builds the venv (a few
minutes), every pod after skips both the multi-GB image pull and the install.
Opt in per pool — it trades a fast pull for slower imports off network storage.

### LLM — vLLM, for throughput

```json
{
  "name": "Qwen/Qwen3-8B-Instruct",
  "backend": "runpod",
  "runpod": {
    "mode": "pods", "engine": "vllm",
    "served_model": "Qwen/Qwen3-8B-Instruct",
    "network_volume_id": "abc123xyz",
    "volume_path": "models/Qwen3-8B-Instruct",
    "min_vram_gb": 48, "max_hourly_usd": 1.50, "gpu_count": 1,
    "ctx": 32768,
    "max_pods": 4, "scale_up_inflight_per_pod": 8, "idle_timeout_s": 300,
    "pod_max_parallel_requests": 48, "pod_queue_max_size": 256,
    "cost_limit_usd": 50, "cost_period": "day"
  }
}
```

`ctx` becomes vLLM's `--max-model-len`. For a model too big for one card, raise
`gpu_count`: `min_vram_gb` is then checked against the **total**,
`max_hourly_usd` against the **whole pod**, and vLLM is told to shard with
`--tensor-parallel-size N`.

### LLM — a GGUF on llama.cpp

```json
"runpod": {
  "engine": "llamacpp",
  "hf_gguf": "unsloth/Qwen3-8B-GGUF:Q4_K_M",
  "network_volume_id": "abc123xyz",
  "min_vram_gb": 24, "max_hourly_usd": 0.60
}
```

`hf_gguf` is `repo:quant`. A local `/AI/…/foo.gguf` path means nothing on a
rented machine — either publish it or put it on the volume and use
`volume_path`.

### Embeddings

Small, fast, and usually the thing you want **warm** rather than cold — an
embedding call that waits two minutes for a pod is useless in a RAG path.

```json
{
  "name": "BAAI/bge-m3",
  "backend": "runpod",
  "runpod": {
    "engine": "coderai", "image": "ghcr.io/nextime/coderai-embeddings:latest",
    "served_model": "BAAI/bge-m3",
    "network_volume_id": "abc123xyz", "volume_path": "models/bge-m3",
    "min_vram_gb": 16, "max_hourly_usd": 0.40,
    "pool": "digesta-small",
    "min_pods": 1, "max_pods": 2,
    "schedule_enabled": true,
    "schedule_start": "08:00", "schedule_end": "20:00",
    "schedule_days": "mon,tue,wed,thu,fri", "schedule_tz": "Europe/Rome",
    "pod_max_parallel_requests": 16, "pod_queue_max_size": 64
  }
}
```

`min_pods: 1` with a schedule is the pattern: instant during working hours,
nothing billed overnight.

### OCR

Two shapes. The **VLM path** is an ordinary chat model doing OCR — nothing
special, configure it like any LLM above. The **native engine path** uses
`/v1/ocr` with a real OCR engine:

```json
{
  "name": "datalab-to/surya-ocr-2",
  "backend": "runpod",
  "runpod": {
    "engine": "vllm",
    "served_model": "datalab-to/surya-ocr-2",
    "network_volume_id": "abc123xyz", "volume_path": "models/surya-ocr-2",
    "min_vram_gb": 40, "max_hourly_usd": 1.20,
    "direct_tcp": "auto",
    "max_pods": 8, "scale_up_inflight_per_pod": 16,
    "pod_max_parallel_requests": 48, "pod_queue_max_size": 256,
    "cost_limit_usd": 100, "cost_period": "day"
  }
}
```

Then configure the OCR engine in **Settings → OCR** on the orchestrator:

```json
"ocr": {
  "surya_serve": "vllm",
  "surya_server_url": "",
  "surya_auto_build": true
}
```

`surya_serve: vllm` puts the model in vLLM on the pod and leaves the Surya
**client** on the orchestrator. That client needs an isolated venv, because
surya-ocr caps `pillow<11` and CoderAI needs `pillow>=12` — they cannot share
one.

**`surya_auto_build: true` is the setting people miss.** It defaults to
**false**, so on first use the engine reports a missing venv instead of
building one, and you end up installing it by hand. Switched on, the engine
creates the venv from `requirements-surya.txt` on first use (a non-blocking
build — the first request reports "building", later ones work) and puts it on
the **persistent cache mount** at `<cache>/ocr/surya_venv`, so it survives
restarts and is built once. If you would rather pre-build it, that is
`python3 -m venv <cache>/ocr/surya_venv` plus
`pip install -r requirements-surya.txt`; `ocr.surya_venv` overrides the
location.

Keep `direct_tcp` so a slow page is not cut off at 100 s by RunPod's proxy, and
set the pod admission numbers — this is the exact path that was measured at
~6 pages/s with the defaults while the GPU idled.

For scanned-document throughput, **batching beats concurrency**: send
`/v1/ocr/batch` with many files and let `scale_up_inflight_per_pod` add pods.

### Engine pods — the very large MoE models

`ds4`, `colibri`, `k3`, `kt` run through `ghcr.io/nextime/coderai-engines`. For
these a volume is not an optimisation, it is a requirement: `colibri` and `k3`
**never download** and fail outright without `volume_path`.

```json
"runpod": {
  "engine": "ds4", "network_volume_id": "abc123xyz",
  "volume_path": "models/DeepSeek-V4-Pro-Q4K.gguf",
  "min_vram_gb": 80, "gpu_count": 2, "max_hourly_usd": 4.00
}
```

### LoRA and QLoRA adapters

Adapters are configured on the model entry (the LoRA section of the model page),
and how they reach the pod depends on where they live:

- **Portable** — an adapter the pod can resolve itself, i.e. a HuggingFace repo
  id or a URL. It travels as a reference and the engine fetches it. For a vLLM
  pod the orchestrator stages it and adds `--enable-lora --lora-modules
  <name>=<path> --max-lora-rank 64` to the launch automatically.
- **Local-only** — an adapter that exists only on your disk (one you trained
  yourself). It cannot be resolved remotely, so the orchestrator **forces a
  CoderAI pod** (`engine: coderai`) and sends it the adapter, because vLLM and
  llama.cpp resolve adapters at launch and have no endpoint to receive one. The
  provisioning log says so explicitly when it happens.

The durable answer for adapters you train is the **volume**: put them under
`/workspace/loras/<name>` and reference that path, and every pod has them with
no transfer and no pod-type constraint.

A QLoRA adapter is an ordinary PEFT adapter — what makes it QLoRA is that the
*base* was quantised during training. Two things follow:

- set the base model's `quantization` to match what the adapter was trained
  against (e.g. `"quantization": "bitsandbytes"` for an nf4 base). An adapter
  trained on an nf4 base and merged onto an fp16 base loads without error and
  quietly degrades — there is no exception to catch, only worse output;
- **llama.cpp takes exactly one adapter and it must be GGUF-converted.** A PEFT
  safetensors directory is not loadable there. Use a CoderAI or vLLM pod for
  PEFT adapters, or convert first.

`max_lora_rank` defaults to 64 on the vLLM launch; a higher-rank adapter needs
it raised through `docker_args`.

### Sanity checklist before you add the tenth model

- Does it name a volume, and is the volume in the region you rent in?
- Is `max_hourly_usd` set, and does it account for `gpu_count`?
- Is `cost_limit_usd` + `cost_period` set, so one model cannot eat the account?
- Are the **pod** admission numbers set, or will it 429 at 16 concurrent?
- Should it be warm during working hours (`min_pods` + a schedule) or cold?
- If it shares a `pool`, is it a CoderAI-engine model? (vLLM pods cannot pool.)

## 5. Scale

Four numbers decide the shape of a pool:

| key | meaning |
|---|---|
| `min_pods` | kept warm at all times. `0` = fully cold-start. A warm pod bills every hour of every day. |
| `max_pods` | hard ceiling on concurrent pods for this model. |
| `scale_up_inflight_per_pod` | add a pod once the least-loaded one already has this many requests in flight. **Without this the pool only grows when it has no healthy pod at all**, so fifty concurrent requests would pile onto pod #1. |
| `idle_timeout_s` | destroy a pod this long after its last request. |
| `allow_client_scale` | let a client lease a warm floor up to `max_pods` through `POST /v1/runpod/scale`. Off by default. |
| `client_scale_ttl_s` | how long such a lease lasts unless renewed (default 1800). A lease that never expired would be a bill that never stopped. |
| `data_centers` | several acceptable regions, tried in order on a capacity miss — how you say "anywhere in the EU". A single `data_center` pins one region; blank lets RunPod pick any region **on earth**. |
| `global_volume_id` | a RunPod **global** volume (region-independent). Unlike `network_volume_id` it does NOT pin the pod's region, which is the only reason to use one. |
| `gpu_types` | ordered GPU **preference**, best first (comma-separated; names have spaces, so commas only). |
| `allow_other_gpus` | when no preferred card is free, rent another that still fits `min_vram_gb` and `max_hourly_usd`. Off by default. |

**Then the two that are easy to miss**, and the reason a pod can look slow while
its GPU is idle:

| key | meaning |
|---|---|
| `pod_max_parallel_requests` | how many requests the **pod's own** server admits at once. Default is **2**. |
| `pod_queue_max_size` | how deep the pod queues admitted-but-waiting requests. Default is **6**. |

Those defaults are sized for a shared workstation. On a dedicated card they are
the bottleneck: the app's admission gate saturates long before the GPU does and
everything over the limit gets `429`, while vLLM's `max_num_seqs` sits unused.
Set `pod_max_parallel_requests` to roughly the remote engine's batch width (for
vLLM, its `max_num_seqs`) and give the queue some depth. Measured on one A40
serving an OCR VLM: **~6 pages/s** with the defaults versus a GPU that was never
saturated.

Do not compensate by driving a single pod hard from the client. Send everything
to the orchestrator and let it fan out — that is what `scale_up_inflight_per_pod`
and `max_pods` are for. Client-side concurrency against one pod just produces
429s.

### Only pay during working hours

A warm pod (`min_pods ≥ 1`) bills around the clock. To keep one ready only when
anyone is actually working:

```json
"schedule_enabled": true,
"schedule_start": "08:00",
"schedule_end": "20:00",
"schedule_days": "mon,tue,wed,thu,fri",
"schedule_tz": "Europe/Rome"
```

Outside the window the warm floor drops to 0 and the pod is **terminated**, so
billing stops. It does **not** block requests: one arriving at 03:00 still
cold-starts a pod, exactly as `min_pods: 0` always has. An end earlier than the
start runs overnight and belongs to the day it started on, so `22:00`–`06:00` on
`fri` covers Friday night into Saturday morning. Leave `schedule_days` empty for
every day.

### Weights in one place, pods in any region

A **network volume** lives in exactly one data centre and a pod elsewhere
cannot attach it, so `network_volume_id` also decides where every pod runs —
coderai pins it for you and says so (`volume … is in X — pinning pods there`).
That is the right trade when you want fast starts in one region.

A **global volume** (RunPod beta, Sept 2026) is region-independent storage
backed by object storage. Store the weights once and rent a card wherever one
is free:

```json
"global_volume_id": "gv-abc123",
"data_centers": "EU-RO-1,EU-FR-1,EUR-NO-1,EUR-IS-3"
```

`data_centers` is the other half, and it matters more than it looks. A single
`data_center` pins one region, and leaving it blank lets RunPod pick **any
region on earth** — rarely what you want for data that may not leave the EU.
A list is tried in order, so the first entry is the preferred region and the
scaler only moves on when it cannot get a card there. Each (card, region) pair
becomes its own candidate, so the capacity fallback that already walks GPU
options walks regions too.

Caveats worth knowing before you rely on it:

* **It has to be created in the RunPod console** — Storage → **+ New volume** →
  **Global volume**. The public API still refuses: `POST /v2/network-volumes`
  allows only `STANDARD` and `HIGH_PERFORMANCE` and demands a `dataCenter`.
* **Attaching is console-documented only.** No field appears in the Pod API
  reference, and the v1 REST validator accepts unknown body keys silently — so
  a wrong name cannot be caught by probing, it just yields a pod with no volume
  that re-downloads its weights. coderai sends `globalNetworkVolumeId`,
  overridable with `CODERAI_RUNPOD_GLOBAL_VOLUME_FIELD`. **Verify on the first
  pod** that the weights are really at the mount.
* **Read-heavy only.** No file locking, no atomic rename, eventual consistency,
  last-write-wins on concurrent writes. Fine for weights, adapters, tokenizers
  and configs; wrong for checkpoints — use a network volume to train.
* RunPod mounts it at `/workspace` when it is the only volume, and moves it to
  `/workspace-global` if a network volume is attached too. coderai therefore
  points the pod env at the global volume only when no network volume is set.
* GPU pods and GPU serverless only; CPU is not supported. And if the account
  balance reaches $0 the volume is flagged and permanently deleted after 15
  days.

---

### When the card you asked for is not available

`gpu_type` is a **filter**, not a preference: `"NVIDIA A40"` has always meant
"an A40 or nothing", so when RunPod has no A40 free the request fails rather
than taking a card that would have served it just as well.

To let it fall through:

```json
"gpu_types": "NVIDIA A40,NVIDIA RTX A6000",
"allow_other_gpus": true,
"min_vram_gb": 40,
"max_hourly_usd": 1.20
```

* `gpu_types` is an ordered **preference**. A preferred card always ranks ahead
  of a fallback, whatever `selection_criteria` says — the point of naming one is
  to get it when it is there.
* `allow_other_gpus` is what permits anything else. It is **off by default**: a
  pinned card is sometimes pinned for a reason, and quietly renting a different
  one would be a surprise on the invoice.
* **`min_vram_gb` and `max_hourly_usd` are the real constraints** once the
  fallback is on, and this is the part that catches people. With
  `max_hourly_usd: 0.60`, an A40 at $0.59 fits and nothing else does — so the
  fallback finds nothing and you are back where you started. Raise the ceiling
  to the most you are willing to pay for the *alternative*, and keep
  `min_vram_gb` honest so a cheap 16 GB card cannot win a job that needs 48.
* A fallback is never silent. The provisioning line says
  `[FALLBACK — no preferred card available]`, and each ranked option carries a
  `preferred` flag.

Name matching is exact on the id or display name, case-insensitive, and
tolerates a missing `NVIDIA ` prefix — so `A40` finds `NVIDIA A40`. It is
deliberately **not** a substring match: `A40` would otherwise also match
`NVIDIA RTX A4000`, a 16 GB card, for a job that asked for 48.

---

### Let a client ask for pods

The autoscaler reacts to load: it adds a pod once the least-loaded one already
has `scale_up_inflight_per_pod` requests in flight. That is the right behaviour
for traffic nobody predicted, and the wrong one for a client that KNOWS what is
coming — a batch of a thousand documents discovers the pool one cold start at a
time.

So a client can say so, on a model that opted in:

```json
"allow_client_scale": true,
"client_scale_ttl_s": 1800
```

```bash
curl -fsS -X POST http://127.0.0.1:8776/v1/runpod/scale \
     -H "Authorization: Bearer sk-coderai-…" -H "Content-Type: application/json" \
     -d '{"model": "qwen38-awq", "pods": 3}'
```

```json
{"model": "qwen38-awq", "requested": 3, "granted": 3,
 "max_pods": 3, "expires_in_s": 1800, "clamped": false}
```

It is a **lease**, bounded three ways, because a client that can raise the floor
can raise the bill:

| bound | what it does |
|---|---|
| `allow_client_scale` | off by default — no model is scalable by a client unless you said so |
| `max_pods` | asking for more returns the ceiling with `"clamped": true`, not an error: the client asked for "as many as you can" |
| `client_scale_ttl_s` | the lease expires (default 30 min) unless renewed, so a client that dies holding four A40s stops paying for them by itself |

`"pods": 0` releases it early. The lease and a warm-pod schedule are **maxed,
never summed** — both mean "hold this many ready" — so a client cannot lower a
floor you configured, and a closed window cannot cancel a lease the client is
relying on. The account's $/hr and spend caps still refuse to provision whatever
is asked here, and `GET /v1/runpod/status` reports the live lease per model
(`client_pods`, `client_lease_expires_in_s`) so an unexpected bill has a visible
cause.

Deliberately not admin-scoped: the gate is the per-model flag, not the key. An
ordinary API key may scale a model you marked scalable, and nothing else.

---

## 6. Watch the cost

Three layers, and you want all three.

**Caps that refuse to spend.** Per model, `cost_limit_usd` with `cost_period`
(`hour`/`day`/`week`/`month`, a trailing window). Account-wide,
`global_cost_limit_usd` and `global_max_hourly_usd`. When a cap is hit,
provisioning stops rather than quietly continuing.

**The GUI.** Admin → **RunPod** shows live pods, their hourly rate, accumulated
uncommitted cost, per-model spend from the ledger, and — for scheduled models —
whether each is inside its window and when that next flips.

**The API, for your own monitoring.** `GET /v1/runpod/status` (or
`/v1/runpod/spend` — the same document under two names, so pick the one that
reads right where you call it) answers both "what is the fleet doing" and "what
is it costing":

```bash
curl -fsS -H "Authorization: Bearer sk-coderai-…" \
     http://127.0.0.1:8776/v1/runpod/spend
```

```json
{
  "summary": {
    "pods_total": 3, "pods_ready": 2, "pods_booting": 1, "pods_unhealthy": 0,
    "gpus_in_use": 2, "inflight": 2, "spot_pods": 1, "hourly_usd": 2.18,
    "by_model":  {"datalab-to/surya-ocr-2": 2, "Qwen/Qwen3-8B-Instruct": 1},
    "by_region": {"EU-SE-1": 2, "EU-RO-1": 1},
    "by_gpu":    {"A40": 2, "A100": 1},
    "by_cloud_type": {"SECURE": 3}
  },
  "pods": [{"model":"datalab-to/surya-ocr-2","pod_id":"x0nr…","state":"ready",
            "healthy":true,"inflight":2,"gpu":"A40","gpu_count":1,
            "data_center":"EU-SE-1","cloud_type":"SECURE","is_spot":false,
            "pool":"","engine":"vllm","hourly_usd":1.09,"uptime_s":940,
            "live_cost_usd":0.28,"console_url":"https://…"}],
  "ledger": {"per_model": {"…": {"day": 12.40}}, "global": {"day": 31.05}},
  "caps":   {"global_max_hourly_usd":20.0,"global_cost_limit_usd":40.0,
             "global_cost_period":"day","enabled":true},
  "schedules": [{"model":"…","in_window":true,"next_change":"2026-10-08T20:00",
                 "effective_min_pods":1,"healthy_pods":1}],
  "live_hourly_usd": 1.09,
  "live_uncommitted_usd": 0.28
}
```

`summary` is derived from `pods`, so the two can never disagree. Three
distinctions worth knowing when you build a dashboard on this:

- **`pods_ready` vs `pods_total`** — a booting pod bills but serves nothing, so
  `summary.hourly_usd` counts only pods that are actually serving. For real
  spend use the ledger plus `live_uncommitted_usd`.
- **`pods_unhealthy`** is a pod past boot that is failing its probe: it is
  costing money and answering nothing. A non-zero value here is the thing to
  alert on.
- **`gpus_in_use`** counts cards, not pods, so a `gpu_count: 2` pod counts twice.

This endpoint needs an **admin-scoped** API key — Admin → **Tokens**, tick
*Admin scope* when generating it (or toggle it on an existing one). An ordinary
key gets `403`: a key that may ask a model a question has no business reading the
account's spend. Give Digesta's monitor an admin-scoped key and a plain key for
inference, not one key for both.

`live_uncommitted_usd` is what is running but not yet written to the ledger —
add it to the ledger figure for "spent so far".

### Pods CoderAI did not start

The ledger only knows about pods CoderAI rented. A pod created by hand in the
RunPod console is **not** in it, is not covered by the caps, and is not reaped.
If the figures look low, check the console too.

---

## 7. Calling it

Standard OpenAI shapes, plus OCR:

```bash
# chat
curl -fsS http://127.0.0.1:8776/v1/chat/completions \
  -H "Authorization: Bearer sk-coderai-…" -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-8B-Instruct","messages":[{"role":"user","content":"ciao"}]}'

# embeddings
curl -fsS http://127.0.0.1:8776/v1/embeddings \
  -H "Authorization: Bearer sk-coderai-…" -H "Content-Type: application/json" \
  -d '{"model":"BAAI/bge-m3","input":["una frase"]}'

# OCR — multipart, not JSON
curl -fsS http://127.0.0.1:8776/v1/ocr \
  -H "Authorization: Bearer sk-coderai-…" \
  -F file=@page.pdf -F engine=surya -F dpi=200

# many files at once
curl -fsS http://127.0.0.1:8776/v1/ocr/batch \
  -H "Authorization: Bearer sk-coderai-…" \
  -F files=@a.pdf -F files=@b.pdf -F engine=surya
```

**The first request to a cold model is slow** — the pod has to be rented, the
image pulled and the weights downloaded, which is minutes, covered by
`boot_timeout_s` and `load_timeout_s`. Give your client a timeout that allows
for it, or keep `min_pods: 1` during working hours (§5).

### The native Surya OCR engine

`engine=surya` runs the Surya client in an isolated venv on the **orchestrator**
and the model in vLLM on the pod. The venv ships in the full image — it cannot share the main one, because Surya caps `pillow<11` while
CoderAI needs `pillow>=12`. If you are on an older image you had to build it by
hand into the cache mount; pull the new image and delete that workaround.

Surya's vLLM client probes `{service_url}/health`, which this app serves from
0.2.83. On an older image that probe 500s and the engine never becomes ready.

---

## 8. When something is wrong

```bash
docker logs --tail 200 coderai                     # provisioning, boots, caps, 429s
curl -fsS localhost:8776/healthz                    # is the app alive
curl -fsS localhost:8776/v1/models                  # is the model registered
curl -fsS -H "Authorization: Bearer <admin-key>" \
     localhost:8776/v1/runpod/spend                 # pods, spend, schedules
```

| symptom | likely cause |
|---|---|
| `/v1/*` returns 401 | missing or wrong `Authorization: Bearer`. Create a token in Admin → Tokens. |
| `/v1/runpod/spend` returns 403 | the key is valid but not admin-scoped. |
| First request times out | cold start. Raise the client timeout; check `boot_timeout_s`/`load_timeout_s` and the log. |
| Requests 429 under load | `pod_max_parallel_requests` too low, or `max_pods`/`scale_up_inflight_per_pod` too low. |
| Pod boots then never serves | wrong `served_model`, or `direct_tcp` forced on an image that cannot take the certificate. Read the log. |
| Long requests die at ~100 s with 524 | you are on RunPod's Cloudflare proxy. Use `direct_tcp`. |
| "budget cap hit — not provisioning" | a `cost_limit_usd` or a global cap. Working as intended; raise it or wait for the window to roll. |
| GUI 404s at `/admin` | an image older than 0.2.83. |
| Pods appear that CoderAI did not start | someone used the console. Not in the ledger, not capped, not reaped. |

Pods CoderAI rents are tagged with this install's deployment id, and a reaper
destroys tagged pods no live pool is tracking — so a crash or a restart does not
leave a machine billing forever. Pods it did not rent are never touched.

---

## 9. Keeping it running, and upgrading in place

### Run it as a service — systemd

The production host uses systemd. A plain unit for the orchestrator:

```ini
# /etc/systemd/system/coderai.service
[Unit]
Description=CoderAI RunPod orchestrator
After=network-online.target docker.service
Wants=network-online.target
Requires=docker.service

[Service]
Type=exec
# --rm + ExecStartPre makes a restart idempotent: a leftover container of the
# same name would otherwise make the next start fail on a name conflict.
ExecStartPre=-/usr/bin/docker rm -f coderai
ExecStart=/usr/bin/docker run --rm --name coderai \
    -p 8776:8776 \
    -e CODERAI_CONFIG_DIR=/config \
    -e CODERAI_CACHE_DIR=/cache \
    -v /srv/coderai/config:/config \
    -v /srv/coderai/cache:/cache \
    ghcr.io/nextime/coderai:latest
ExecStop=/usr/bin/docker stop -t 30 coderai
Restart=always
RestartSec=10
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now coderai
systemctl status coderai
journalctl -u coderai -f        # provisioning, boots, caps, 429s
```

With **podman** instead of docker, prefer a rootless quadlet — systemd generates
the unit from it, so there is no `docker run` line to keep in sync:

```ini
# ~/.config/containers/systemd/coderai.container
[Unit]
Description=CoderAI RunPod orchestrator

[Container]
Image=ghcr.io/nextime/coderai:latest
ContainerName=coderai
PublishPort=8776:8776
Environment=CODERAI_CONFIG_DIR=/config
Environment=CODERAI_CACHE_DIR=/cache
Volume=%h/coderai/config:/config:Z
Volume=%h/coderai/cache:/cache:Z

[Service]
Restart=always

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload
systemctl --user start coderai
loginctl enable-linger "$USER"   # or it stops when you log out
```

`Type=exec` with `--rm` (or the quadlet's own lifecycle) means the container is
recreated on every start, so an image you just pulled is actually picked up —
a `docker start` of an old container would not.

### Run it as a service — sysvinit

On a host without systemd, an init script does the same job. The one this
project uses on its build machine (`/etc/init.d/coderai`) is worth copying as a
pattern for two reasons it documents in its own comments:

- **the container, not the launcher, is the source of truth** for "is it
  running" — dockerd owns the container, so it outlives the shell that started
  it, and `docker run --rm --name coderai` dies instantly on a name conflict if
  you don't check;
- a **boot-time shell has a different PATH** than your interactive one, so the
  script spells out the path to its runner instead of relying on `.bashrc`.

```sh
#!/bin/sh
### BEGIN INIT INFO
# Provides:          coderai
# Required-Start:    $local_fs $remote_fs $network docker
# Required-Stop:     $local_fs $remote_fs $network docker
# Default-Start:     2 3 4 5
# Default-Stop:      0 1 6
### END INIT INFO
PATH=/sbin:/usr/sbin:/bin:/usr/bin
RUNAS=youruser
CONTAINER=coderai
CMD='/home/youruser/bin/coderai docker'
. /lib/lsb/init-functions

container_running() {
    [ -n "$(docker ps --filter "name=^${CONTAINER}\$" --filter status=running \
            --format '{{.ID}}' 2>/dev/null | head -n 1)" ]
}
case "$1" in
    start)   container_running && { echo "already running"; exit 0; }
             su - "$RUNAS" -c "$CMD -d" ;;
    stop)    docker stop -t 30 "$CONTAINER" ;;
    restart) "$0" stop; sleep 3; "$0" start ;;
    status)  container_running && echo "RUNNING" || { echo "not running"; exit 3; } ;;
    *)       echo "Usage: $0 {start|stop|restart|status}" >&2; exit 2 ;;
esac
```

`update-rc.d coderai defaults` to enable it.

### A wrapper script for the run line

Rather than keep a long `docker run` in the unit, put it in one script the unit
and you both call. The build machine's `~/bin/coderai` is the example — it
wraps the host runner `coderai-docker`, and the parts worth stealing are:

```bash
#!/bin/bash
# One place for the run line, so the init script, systemd and you agree.

if [ x"$1" == x"docker" ] ; then
   detach=""
   [ x"$2" == x"-d" ] && detach="--detach"

   # Live-code mode: with a flag FILE present, bind-mount the working tree over
   # the image's copy so a restart picks up edits with no rebuild. A file, not
   # an env var, because the init script launches this through `su -`, which
   # does not carry the environment.
   devmap=""
   if [ -f /srv/coderai/.dev-code ] ; then
      devmap="--map /srv/coderai/codai:/opt/coderai/app/codai"
      echo "coderai: LIVE CODE (remove .dev-code to use the image's)"
   fi

   coderai-docker --user --host 0.0.0.0 --port 8776 $devmap \
                  --map /srv/data \
                  $detach
else
   # the same entry point without a container, for development
   exec /srv/coderai/coderai "$@"
fi
```

Two details that matter in production: a **flag file** rather than an
environment variable, because a service manager's shell carries neither your
env nor your PATH; and keeping the bind-mount list in one place, because a mount
that exists in your manual run but not in the unit is a bug you only find after
a reboot.

### Upgrading in place, without pulling 25 GB

The image can refresh **its own application code** from git instead of being
replaced. `coderai-docker --upgrade` (the host runner) drives
`coderai-upgrade` inside the container, which:

1. shallow-clones the configured branch (default `production`);
2. compares that tree's `codai/__init__.py __version__` with the version baked
   into the image, and stops if the image is already current;
3. replaces `/opt/coderai/app` with the fetched code;
4. re-runs pip when the fetched code's declared dependencies changed, so new
   packages land in the image's python env;
5. exits `0` if it changed something, `10` if nothing was needed — and the host
   runner `docker commit`s the container back onto **the same image tag** only
   on `0`, so there is no Dockerfile rebuild and no extra overlay image.

```bash
coderai-docker --upgrade                      # to the production branch
coderai-docker --upgrade --upgrade-ref v0.2.83
coderai-docker --upgrade --force              # even if not strictly newer
coderai-docker --upgrade --no-pip             # code only (offline host)
coderai-docker --upgrade --ssh-key ~/.ssh/id_ed25519   # private repo over SSH
```

Knobs, all optional, passed as env by the runner: `CODERAI_UPGRADE_REPO`,
`CODERAI_UPGRADE_REF`, `CODERAI_UPGRADE_FORCE`, `CODERAI_UPGRADE_SKIP_PIP`,
`CODERAI_UPGRADE_SSH_KEY`. If the default remote is unreachable — expired
certificate, DNS, an outage — it falls back to the public GitHub mirror; a repo
you named explicitly is never silently replaced.

**When to use which.** `--upgrade` is right for a code-only fix on a live
deployment: seconds, no large transfer, and it works on a metered link. Pull a
new image when the **dependencies** change substantially, when you want the
signature of a published build, or when you want to be certain of what you are
running — `--upgrade` mutates a tag in place, so after it, `coderai:latest`
on your host is no longer byte-identical to the registry's. Record what you did.
For an orchestrator fleet, pulling the signed image is the auditable path and
`--upgrade` is the emergency one.

Restart after an upgrade (`systemctl restart coderai`) — the code is swapped in
the image, not in the running process.

### Upgrading unattended, hourly

Two installs follow this repo: **zeiss** (nexlab) and the **Digesta production
host in Aruba**. The Aruba one upgrades itself, hourly, from `production`, with
nobody logged in. That is `packaging/linux/launcher/coderai-autoupgrade` plus the
two units in `packaging/linux/systemd/`.

```bash
install -D -m755 packaging/linux/launcher/coderai-autoupgrade ~/.local/bin/
install -D -m644 packaging/linux/systemd/coderai-autoupgrade.service \
                 packaging/linux/systemd/coderai-autoupgrade.timer \
                 ~/.config/systemd/user/
install -D -m600 packaging/linux/systemd/autoupgrade.conf.example \
                 ~/.config/coderai/autoupgrade.conf
$EDITOR ~/.config/coderai/autoupgrade.conf      # see the two traps below
systemctl --user daemon-reload
systemctl --user enable --now coderai-autoupgrade.timer
sudo loginctl enable-linger "$USER"             # NOT optional on a server
```

**Run it once by hand first** and read the output — `~/.local/bin/coderai-autoupgrade`.
It prints the configuration it actually resolved, and whether the idle check is
on. An unattended job that was never watched once is a guess.

**It skips the upgrade entirely while CoderAI is serving.** Checked before the
image is touched, not merely before the restart, so a busy host never reaches the
state where new code waits in the image. Two probes, the higher count wins:

| probe | what it sees |
|---|---|
| `<base>/metrics` → `coderai_engine_inflight` | every request the front has in flight, pod-served included |
| `<base>/v1/runpod/status` → `summary.inflight` | what the rented pods themselves report |

The RunPod one matters most on an orchestrator where **every** model is remote:
the local engines are idle by definition, so the pods' own count is the only
signal that work is happening. Interrupting a pod-served request loses the work
*and* keeps paying for the pod that was doing it.

The check **fails closed**. No token, wrong URL, unreachable endpoint → it skips
and says so, because an unreadable probe almost always means a misconfiguration
rather than an idle server. `REQUIRE_IDLE_CHECK="0"` opts out, and then it will
interrupt live requests.

By default a live request is **never** interrupted (`MAX_DEFERRALS="0"`): if the
image was upgraded and a request arrives before the restart, the new code waits
and every run says so. Set it above zero to allow a restart after that many
hourly ticks.

**It decides by reading the image, not by an exit code.** `coderai-docker
--upgrade` exits `0` both when it upgraded and when the image was already
current, so an exit-code-driven loop would restart the service every hour
forever. The script reads `__version__` out of the image before and after; a
change is the only thing that triggers a restart.

**It rolls back.** The pre-upgrade image is tagged first (a tag, so it costs
nothing). If the service is not healthy within `HEALTH_TIMEOUT` after the
restart, the tag is put back, the service is restarted again, and the run exits
non-zero with one line saying whether it recovered. A service left down says
"Needs a human".

Two traps worth knowing before they cost you an evening:

- **The config file is sourced as shell, so quote anything with a space.**
  `RESTART_CMD=systemctl --user restart coderai.service` assigns only
  `systemctl` and then *runs* `--user restart coderai.service`. The restart then
  restarts nothing, the health check fails, and a perfectly good upgrade gets
  rolled back. Hence the quoting in the example, and hence the resolved
  `restart=[…]` line in every run's log.
- **`loginctl enable-linger`.** A systemd *user* timer only fires while the user
  has a session. Without lingering it is silently inert after a reboot — the one
  failure mode here that leaves no trace anywhere.

`HEALTH_URL` is the only address to get right; `/metrics` and
`/v1/runpod/status` are derived from it. On the Digesta host the container's
8776 is published on 8777, so it is `http://127.0.0.1:8777/healthz`.

State and history live in `${XDG_STATE_HOME:-~/.local/state}/coderai`:
`autoupgrade.log` (trimmed, since this runs 24 times a day), `restart-pending`
and `deferrals`. The journal has the per-run copy: `journalctl --user -u
coderai-autoupgrade`.

One thing to keep in mind: each real upgrade commits another layer onto the tag,
so after many of them the image has drifted from anything in the registry. Pull
the signed image occasionally to re-base it — `--upgrade` is for carrying fixes
between releases, not a substitute for them.

## 10. Reference

| | |
|---|---|
| Image | `ghcr.io/nextime/coderai:latest` — the orchestrator. Port **8776**. Capability images are pods; never install one by hand |
| Config | `<config>/config.json`, `<config>/models.json`, `<config>/auth.json` |
| Signing key | `packaging/cosign.pub` in the repo |
| Full RunPod key reference | `docs/runpod.md` |
| Remote execution / engines | `docs/remote-execution.md` |
| Other capability images | `docs/install-from-packages.md` |
| In-image upgrader | `packaging/linux/launcher/coderai-upgrade`, driven by `coderai-docker --upgrade` |
| Unattended upgrade | `packaging/linux/launcher/coderai-autoupgrade` + `packaging/linux/systemd/` (hourly, skips while serving, rolls back) |
| Host runner | `packaging/linux/run_oci.sh` (installed as `coderai-docker`) |
