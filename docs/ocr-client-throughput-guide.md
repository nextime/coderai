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
offered   = 64 × pods_ready        (the server dispatches ~32/pod)
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
| `max_inflight_per_pod × pods` | ~32 × pods | dispatched at once; discovered per pod |
| `ocr.max_concurrency` | 8192 | pages admitted into the OCR manager |
| pod thread pool | 8192 | one thread per page in flight to a pod |

All three line up at **4096 pages = 128 pods**, which is the configured
`max_pods`. Asking for more than 8192 does not fail fast — the surplus waits for
a pod lease and eventually times out, which is worse than being refused. Stay at
or under `64 × pods_ready`.

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
| `max_tokens` 12288, `max_continuations` 3 | 0.416 | 2,523 | $0.000396 |
| **`max_tokens` 4096, `max_continuations` 1** | **0.557** | **3,349** | **$0.000298** |

Multiply by `pods_ready`. At the 128-pod ceiling that is ~71 pages/s, ~257,000
pages/hour, and 996,760 pages in ~3.9 hours for ~$298.

Page latency is **not** uniform: p50 44 s, p95 169 s, worst ~450 s. Set the
client timeout to at least **600 s per request** and do not treat a slow page as
a failure.

### How these numbers were taken, and what they are not

`0.557` is **concurrency ÷ mean page latency** — 32 pages in flight on one pod
divided by a 57.5 s mean. That is the steady-state figure and the only one of
the three that extrapolates per pod. For the same run:

| | pages/s | why it is wrong for planning |
|---|---|---|
| 100 pages ÷ 450 s total elapsed | 0.222 | includes the tail draining alone |
| **32 ÷ 57.5 s mean latency** | **0.557** | steady state — use this |
| 80 pages ÷ 64 s at the 80% mark | 1.242 | completion order favours short pages |

They come from a controlled bench: one pod, one client, nothing else running.

**A live run on 2026-10-19 got 0.051 pages/s per pod** — eleven times slower —
at the same `4096/1`, same card, same ~29 pages in flight per pod. The cause is
under investigation and is not yet known; the pod image had changed between the
bench and the run. Until that is resolved, treat `0.557` as what the hardware
*can* do rather than what your next run *will* do, and watch
`live_hourly_usd ÷ pages completed` rather than assuming $0.000298.

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

### Retry policy — bounded, and different per code

**An earlier revision of this note said to retry 404 and 502 "indefinitely".
That was wrong.** It would spin forever on a document that reliably fails, and
forever on a 404 that is not a restart — in both cases paying for pods and
producing nothing, which is the failure it was trying to prevent.

What separates the codes is what each one says about the server's health:

```
429  the server is HEALTHY and full. It told you when to come back.
     -> wait Retry-After, then retry. No attempt cap needed.
     -> sanity bound: if 429 for >15 min with zero pages completing, alert.

404  the server is RESTARTING. A real "endpoint does not exist" would have
     been 404 from your very first request, not after thousands of 200s.
     -> retry with backoff for up to 5 minutes.
     -> past 5 minutes it is not a restart. STOP the run and alert: a
        route that is gone will not come back by being asked again.

502  the POD failed this page. Usually transient; sometimes the document.
     -> up to 5 attempts per page with backoff.
     -> then PARK the page in a failed list and move on. Never block the
        corpus on one file — ~3-5% of pages 502 at least once, and a single
        unparkable document stops a million-page run.

400, 413  the document is the problem, not the server.
     -> park it immediately. Do not retry.
```

**And a circuit breaker, which matters more than any of the above.** If more
than 20% of the last 200 pages ended parked or erroring, stop feeding and
alert. At that point the fleet is costing money and producing nothing, and no
per-page retry rule can detect it — only the aggregate can.

The failure this policy exists to prevent, from 2026-10-19: a restart answered
618 requests with 404 over ~40 seconds, the client treated 404 as fatal and
stopped, and nine pods sat ready and billing $5.31/hr until a human noticed. A
five-minute 404 window covers any restart; it does not cover a route that is
genuinely gone, and it should not.

### Status fields worth watching

From `GET /v1/runpod/status`:

| field | meaning |
|---|---|
| `summary.pods_ready` | serving now — size your window from this |
| `summary.pods_quarantined` | set aside after refusing a connection, being re-probed. **Not gone, not booting.** They come back by themselves |
| `pods[].capacity` | what the server will dispatch to that pod at once (~32, discovered from the pod) |
| `pods[].soft_limit` | non-zero means that pod refused a connection and is being fed more gently |
| `live_hourly_usd` | **every pod that bills**, including ones still booting |
| `live_hourly_booting_usd` | the part that is paid boot producing nothing. A high share means the ramp is too steep |
| `degraded` | one of the three sources was slow; that part of the answer is stale. **Not** a report of an idle fleet |

## CORRECTION, 2026-10-19: keep sending 64 — the server manages it

An earlier revision of this note told you to send `32 × pods_ready`. **Ignore
that. Keep sending `64 × pods_ready`.**

The reasoning behind the 32 was half right and the conclusion was wrong. A pod
does serve about 32 pages at once — observed `Running: 29-32 reqs, Waiting: 0`
on every pod of a live fleet. But that is a fact about how many pages the server
should *dispatch to one pod*, not about how many you should *offer*. Offering
only what the pods can hold means that the instant a page finishes there is
nothing queued to replace it, so every pod idles for one round trip between
every page, and you lose exactly the throughput the queue exists to protect.

**A deep offer is what keeps the fleet at full speed.** So the split of
responsibility is:

| | who | how much |
|---|---|---|
| offered by you | client | `64 × pods_ready` |
| dispatched to each pod | server | ~32, discovered from the pod itself |
| held queued, ready to dispatch | server | the rest |

The server bounds per-pod dispatch from what each pod has shown it will accept,
halving that pod's width if it ever refuses a connection and walking it back up
as pages succeed (v0.4.38). You do not need to know, or track, what a pod can
take — and you cannot overload one by offering more.

The admission budget is sized for exactly this offer:

```
ocr.max_concurrency   4096    = 32 × 128 pods, dispatched
ocr.max_queued_pages  4096    = 32 × 128 pods, queued behind them
total accepted        8192    = 64 × 128 pods
```

### When you do get a 429

Only past that total. It means you are offering more than even the queue is
meant to hold — nothing was lost and nothing was read twice:

```
Retry-After: 2
X-OCR-Pages-Running: 4096     pages dispatched right now
X-OCR-Pages-Queued:  4096     accepted, waiting for a pod
X-OCR-Page-Width:    4096     ocr.max_concurrency
X-OCR-Queue-Limit:   4096     ocr.max_queued_pages
```

**Honour the `Retry-After`.** A client that does finishes sooner than one that
retries immediately, because its retries stop competing with work already
accepted. Those four headers are also the live tuning signal: `Pages-Queued`
well below the limit means the fleet is keeping up with you and you could offer
more; pinned at the limit means the surplus is buying nothing.

Why this changed: before v0.4.38 nothing bounded per-pod dispatch
(`max_inflight_per_pod` defaults to unlimited), so a deep offer did not queue in
the orchestrator — it queued in the pods' socket backlogs, and an overflowing
backlog makes a healthy pod refuse connections. Those refusals were read as dead
pods and cost eight A40s mid-work. The queue now forms where it belongs.

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
3. Then feed, offering `64 × pods_ready`, recomputed as that number
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
