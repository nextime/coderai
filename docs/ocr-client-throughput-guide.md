# How to call coderai's OCR for maximum throughput

Measured 2026-10-18 on rented A40-class pods, through `POST /v1/ocr/batch`.
Every number came from the real HTTP path.

**Supersedes the earlier note.** Two things in it were wrong: there is no
request-count ceiling on the OCR path, and the pages-per-request escalation it
implied is unnecessary. Corrections are marked below.

---

## 1. The request — one page per call

```
POST /v1/ocr/batch
Authorization: Bearer <token>
Content-Type: multipart/form-data

files              ONE file per call
engine             surya
detect             off
dpi                250
max_tokens         4096                  <- was 12288
max_continuations  1                     <- was 3
```

One file per call, always. There is no reason to pack:

* a batch returns only when its slowest document returns, so one degenerate
  page holds the rest hostage;
* a failure anywhere costs the whole batch;
* it saves no memory — bytes in flight are `pages × bytes per page` either way.

`max_tokens` and `max_continuations` are per-request fields. Changing them needs
no deploy on either side, and they are worth +33% throughput and −25% cost.

## 2. Concurrency — one rule

```
in_flight = 32 × pods_ready        (capped at 8192)
```

Read `pods_ready` from `GET /v1/runpod/status` → `summary.pods_ready` and
recompute as the fleet grows and shrinks. One request carries one page, so the
request count *is* the page count.

**CORRECTION.** The earlier note said the server admits 256 concurrently and
queues 2048, so above ~2300 outstanding requests you would be refused. That is
false for OCR: `server.max_parallel_requests` and `server.queue_max_size` gate
`/v1/chat/completions` only. On the OCR path every request you open is accepted
immediately. The limits that do exist are pages, not requests:

| limit | value | what it is |
|---|---|---|
| `max_inflight_per_pod × pods` | 32 × pods | pages a pod will take at once |
| `ocr.max_concurrency` | 8192 | pages admitted into the OCR manager |
| pod thread pool | 8192 | one thread per page in flight to a pod |

All three line up at **4096 pages = 128 pods**, which is the configured
`max_pods`. Asking for more than 8192 does not fail fast — the surplus waits for
a pod lease and eventually times out, which is worse than being refused. Stay at
or under `32 × pods_ready`.

## 3. Memory — yours to bound, not ours

The upload is read into memory when the request arrives, **before** any page
limit applies. So the bytes held are set by what you send, not by a server
setting:

```
8192 single-page requests × ~175 KB        ≈ 1.4 GB
  + the per-page PDF split from each       ≈ 2.8 GB total      fine
same 8192 slots at 4 files per call        ≈ 11 GB             avoid
```

Another reason for one file per call.

## 4. Expected rates

Per pod, single-page documents, A40-class card at $0.59/hr:

| setting | pages/s | pages/$ | $/page |
|---|---|---|---|
| `max_tokens` 12288, `max_continuations` 3 | 0.413 | 2,523 | $0.000396 |
| **`max_tokens` 4096, `max_continuations` 1** | **0.549** | **3,349** | **$0.000298** |

Multiply by `pods_ready`. At the 128-pod ceiling that is ~70 pages/s, ~253,000
pages/hour.

Page latency is **not** uniform: p50 44 s, p95 169 s, worst ~450 s. Set the
client timeout to at least **600 s per request** and do not treat a slow page as
a failure.

## 5. Which engine

| engine | pages/s | $/page | success | loops | generations/page |
|---|---|---|---|---|---|
| `surya` | 0.549 | $0.000298 | 98% | 4% | 1.73 |
| `chandra` | 0.251 | $0.000651 | **100%** | **0%** | **1.02** |

`surya` for bulk. Its failures are mechanically identifiable, so the cheapest
route to high quality is surya first, chandra only where needed:

```
re-read with chandra when:
    text shorter than 50 characters                 (empty)
    OR share of distinct 20-char windows < 0.5      (degenerate repetition)
```

~6% of pages qualify, so combined cost is ~$0.000319/page — near chandra's
quality at near surya's price. No human triage: both tests are mechanical.

## 6. Reading the response

* **A 200 is not necessarily a good page.** A looping or truncated page answers
  200. Check `num_pages`, text length, and the repetition test above.
* **`fallback_engines` is off**, deliberately: a degraded page is never silently
  re-read by another engine. You decide what to re-read.
* **Expect ~2% of pages to error** (502 from a pod, seen at that rate). Retry
  them — they are not deterministic failures.
* **~10% of this archive is unreadable by anyone**: 487 of 4,961 PDFs have no
  `%PDF` header or no `%%EOF`. Screen locally before sending; each one
  otherwise costs a request and returns an error.

## CORRECTION, 2026-10-19: it is 32 per pod, not 64

An earlier version of this note said `in_flight = 64 × pods_ready`. That was
wrong, and it was measured wrong: **a pod serves ~32 pages concurrently**
(observed `Running: 29-32 reqs, Waiting: 0` on every pod of a live fleet), and
the 0.557 pages/s figure below was itself measured at a concurrency of 32 on one
pod. 64 was the configured ceiling, not an observed capacity.

Sending 64 does not queue politely inside the orchestrator. It queues in the
POD'S SOCKET BACKLOG, and when that overflows the pod starts refusing
connections while remaining perfectly healthy. On 2026-10-19 that cost eight
A40s: the orchestrator read the refusals as "the pod is gone" and terminated
them mid-work with 522 pages in flight. coderai no longer terminates on a
refused connection (v0.4.37), but there is still nothing to gain by offering a
pod twice what it will take.

**Send `32 × pods_ready`.** It is the measured optimum, not a safety margin.

## 7. Before a large run

1. Lease the fleet you want:

   ```json
   POST /v1/runpod/scale
   {"model": "datalab-to/surya-ocr-2", "pods": 8, "max_pods": 8}
   ```

   **Send both.** `pods` is a FLOOR — "hold at least this many ready". It says
   nothing about how large the fleet may become, and the autoscaler will keep
   growing toward the configured ceiling for as long as pages keep arriving.
   `max_pods` is the ceiling for the life of your lease, and it is the field
   that bounds what a stage costs. It may only ever narrow the operator's
   configured ceiling; a larger number is clamped and reported in `notes`.

   A cold pod needs **~12 minutes** to pull its image and get weights
   resident, so the first pages of an unscaled run are slow.

2. Poll `GET /v1/runpod/status` until `summary.pods_ready` matches. The
   per-model rows under `schedules` report `effective_max_pods` (the ceiling
   actually in force), `client_max_pods` (your lease's own) and
   `client_lease_expires_in_s`.
3. Then feed, holding `in_flight = 32 × pods_ready`, recomputed as that number
   changes.

### Staging a campaign

Each stage is one call, no restart in between:

```json
{"pods":   8, "max_pods":   8}     ← stage 1
{"pods":  32, "max_pods":  32}     ← stage 2
{"pods":  64, "max_pods":  64}     ← stage 3
{"pods": 128, "max_pods": 128}     ← the rest
{"pods": 0, "teardown": true}      ← done; give the fleet back now
```

`{"pods": 0, "max_pods": N}` is a cap with no warm floor — useful for a long
tail where you want the ceiling but do not need pods held ready. `{"pods": 0}`
alone releases the whole lease. The lease expires by itself after 30 minutes,
so a client that dies does not leave pods billing; **re-send the current stage
every few minutes** for a run longer than that.

### Watch the right cost number

`live_hourly_usd` from `GET /v1/runpod/status` counts **every pod that bills**,
including the ones still booting — RunPod charges from creation, and a pod is
not healthy for its first ~12 minutes. Alongside it:

* `live_hourly_serving_usd` — the part that is actually reading pages
* `live_hourly_booting_usd` — the part that is paid boot producing nothing

A high booting share means the ramp is too steep: you are paying for cards that
cannot help yet. Settle a stage before asking for the next one. (Before
v0.4.34 `live_hourly_usd` counted only healthy pods, so a booting fleet
reported as free — it read $0.00 while 13 pods billed $7.67/hr.)

If a `degraded` key appears in the status document, one of its three sources
did not answer in time and that part of the answer is stale — it is not a
report of an idle fleet.

Pods are not kept warm between campaigns (`min_pods: 0`, `keep_warm: false`), so
an idle fleet costs nothing and every run starts cold by design.
