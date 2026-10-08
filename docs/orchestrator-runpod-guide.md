# CoderAI as a GPU-less RunPod orchestrator

A practical guide for an operator who has to **install** CoderAI on a machine with
no GPU, point it at RunPod, register a handful of models, let them **scale**, and
**watch what they cost**. Written for the Digesta deployment; nothing here is
Digesta-specific.

The shape of the thing: one small CoderAI process on your own host answers
`/v1/*` and serves the admin GUI. It owns no GPU. When a request arrives for a
model it rents a RunPod pod, waits for it to come up, proxies to it, load-balances
across pods, destroys them when they go idle, and bills every hour into a local
ledger with caps you set. Your callers only ever talk to the orchestrator.

```
 Digesta ──► CoderAI orchestrator (your host, no GPU) ──► RunPod pods (A40, …)
             /v1/chat/completions                          vLLM / llama.cpp /
             /v1/embeddings                                 a full CoderAI
             /v1/ocr
             /admin  (GUI)
```

---

## 1. Install

Use the **light capability image**. It is torch-free, boots in seconds on a
GPU-less host, and since 0.2.83 it serves the admin GUI as well as the API.

```bash
docker pull ghcr.io/nextime/coderai-ocr:latest
```

Verify the signature before you run it — every published image is signed:

```bash
cosign verify \
  --key https://raw.githubusercontent.com/nextime/coderai/master/packaging/cosign.pub \
  ghcr.io/nextime/coderai-ocr:latest
```

Run it. Two mounts matter: a **config** directory (settings, model list, API
tokens — keep it on persistent storage and back it up) and a **cache**
directory (anything the orchestrator builds at runtime, such as an isolated
engine venv; it must survive restarts or it gets rebuilt).

```bash
mkdir -p ~/coderai/config ~/coderai/cache

docker run -d --name coderai --restart unless-stopped \
  -p 8000:8000 \
  -e CODERAI_CONFIG_DIR=/config \
  -e CODERAI_CACHE_DIR=/cache \
  -v ~/coderai/config:/config \
  -v ~/coderai/cache:/cache \
  ghcr.io/nextime/coderai-ocr:latest
```

The capability image listens on **8000**. (The full `ghcr.io/nextime/coderai`
image listens on 8776 and expects a local GPU — do not use it for this job.)

Check it:

```bash
curl -fsS http://127.0.0.1:8000/healthz     # {"ok":true,"pid":…}
curl -fsS http://127.0.0.1:8000/health      # same — the vLLM-convention alias
curl -fsS http://127.0.0.1:8000/v1/models   # [] until you add models
```

Then open `http://127.0.0.1:8000/admin`. It redirects to `/login`; the first
credentials are created on first run and printed in the container log
(`docker logs coderai`), and you are asked to change the password.

### Behind nginx, under a sub-path

The app honours `X-Forwarded-Prefix`, so it can live under `/coderai/` without
rewriting links:

```nginx
location ^~ /coderai/ {
    proxy_pass         http://127.0.0.1:8000/;
    proxy_set_header   Host              $host;
    proxy_set_header   X-Forwarded-Prefix /coderai;
    proxy_set_header   X-Forwarded-Proto $scheme;
    proxy_read_timeout 3600s;   # generation is slow; do not cut it off
}
```

`proxy_read_timeout` matters: a long render or a cold engine's first token can
take minutes, and nginx's 60 s default will kill it mid-answer.

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

## 4. Scale

Four numbers decide the shape of a pool:

| key | meaning |
|---|---|
| `min_pods` | kept warm at all times. `0` = fully cold-start. A warm pod bills every hour of every day. |
| `max_pods` | hard ceiling on concurrent pods for this model. |
| `scale_up_inflight_per_pod` | add a pod once the least-loaded one already has this many requests in flight. **Without this the pool only grows when it has no healthy pod at all**, so fifty concurrent requests would pile onto pod #1. |
| `idle_timeout_s` | destroy a pod this long after its last request. |

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

---

## 5. Watch the cost

Three layers, and you want all three.

**Caps that refuse to spend.** Per model, `cost_limit_usd` with `cost_period`
(`hour`/`day`/`week`/`month`, a trailing window). Account-wide,
`global_cost_limit_usd` and `global_max_hourly_usd`. When a cap is hit,
provisioning stops rather than quietly continuing.

**The GUI.** Admin → **RunPod** shows live pods, their hourly rate, accumulated
uncommitted cost, per-model spend from the ledger, and — for scheduled models —
whether each is inside its window and when that next flips.

**The API, for your own monitoring.** `GET /v1/runpod/spend` returns the same
figures as JSON:

```bash
curl -fsS -H "Authorization: Bearer sk-coderai-…" \
     http://127.0.0.1:8000/v1/runpod/spend
```

```json
{
  "pods":   [{"model":"…","pod_id":"…","hourly_usd":1.09,"uptime_s":940,
              "live_cost_usd":0.28,"healthy":true,"inflight":2}],
  "ledger": {"per_model": {"…": {"day": 12.40}}, "global": {"day": 31.05}},
  "caps":   {"global_max_hourly_usd":20.0,"global_cost_limit_usd":40.0,
             "global_cost_period":"day","enabled":true},
  "schedules": [{"model":"…","in_window":true,"next_change":"2026-10-08T20:00",
                 "effective_min_pods":1,"healthy_pods":1}],
  "live_hourly_usd": 1.09,
  "live_uncommitted_usd": 0.28
}
```

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

## 6. Calling it

Standard OpenAI shapes, plus OCR:

```bash
# chat
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-coderai-…" -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-8B-Instruct","messages":[{"role":"user","content":"ciao"}]}'

# embeddings
curl -fsS http://127.0.0.1:8000/v1/embeddings \
  -H "Authorization: Bearer sk-coderai-…" -H "Content-Type: application/json" \
  -d '{"model":"BAAI/bge-m3","input":["una frase"]}'

# OCR — multipart, not JSON
curl -fsS http://127.0.0.1:8000/v1/ocr \
  -H "Authorization: Bearer sk-coderai-…" \
  -F file=@page.pdf -F engine=surya -F dpi=200

# many files at once
curl -fsS http://127.0.0.1:8000/v1/ocr/batch \
  -H "Authorization: Bearer sk-coderai-…" \
  -F files=@a.pdf -F files=@b.pdf -F engine=surya
```

**The first request to a cold model is slow** — the pod has to be rented, the
image pulled and the weights downloaded, which is minutes, covered by
`boot_timeout_s` and `load_timeout_s`. Give your client a timeout that allows
for it, or keep `min_pods: 1` during working hours (§4).

### The native Surya OCR engine

`engine=surya` runs the Surya client in an isolated venv on the **orchestrator**
and the model in vLLM on the pod. The venv is baked into the `coderai-ocr` image
from 0.2.83 — it cannot share the main one, because Surya caps `pillow<11` while
CoderAI needs `pillow>=12`. If you are on an older image you had to build it by
hand into the cache mount; pull the new image and delete that workaround.

Surya's vLLM client probes `{service_url}/health`, which this app serves from
0.2.83. On an older image that probe 500s and the engine never becomes ready.

---

## 7. When something is wrong

```bash
docker logs --tail 200 coderai                     # provisioning, boots, caps, 429s
curl -fsS localhost:8000/healthz                    # is the app alive
curl -fsS localhost:8000/v1/models                  # is the model registered
curl -fsS -H "Authorization: Bearer <admin-key>" \
     localhost:8000/v1/runpod/spend                 # pods, spend, schedules
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

## 8. Reference

| | |
|---|---|
| Image | `ghcr.io/nextime/coderai-ocr:latest` (GPU-less orchestrator + GUI), port 8000 |
| Config | `<config>/config.json`, `<config>/models.json`, `<config>/auth.json` |
| Signing key | `packaging/cosign.pub` in the repo |
| Full RunPod key reference | `docs/runpod.md` |
| Remote execution / engines | `docs/remote-execution.md` |
| Other capability images | `docs/install-from-packages.md` |
