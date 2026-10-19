# OCR throughput and cost on rented GPUs — measurements, 2026-10-18

All figures measured through the **full orchestrator path** (`POST /v1/ocr/batch`
on coderai), never against a pod directly: a number taken by bypassing the
orchestrator describes a path no client will use.

Corpus: 500 single-page PDFs (`~/antani2`), median 89 KB, p95 589 KB, all
structurally valid. 100 of them per rung, same 100 every time.

---

## 1. How throughput is computed, and why the obvious ways are wrong

Three numbers can be derived from one rung and only one of them extrapolates:

| method | 100-page rung | what it measures |
|---|---|---|
| completions ÷ elapsed, at 80/100 | 0.526 pages/s | the **fastest** 80 pages — completion-order bias |
| completions ÷ total elapsed | 0.085 pages/s | includes a **draining tail** with the card idle |
| **concurrency ÷ mean latency** | **0.413 pages/s** | steady state with the queue full |

The third is the right one for a long campaign: a million-page queue never
drains, so the tail idle never happens, and no page is privileged by finishing
first. Every `pages/s` below is `slots ÷ mean page latency`.

---

## 2. Page budget — surya on one A40 ($0.59/hr)

Concurrency 32, `dpi=250`, `detect=off`.

| `max_tokens` / `max_continuations` | pages/s | pages/$ | $/page | 996,760 pages | chars med | mean | p95 | worst page |
|---|---|---|---|---|---|---|---|---|
| 12288 / 3 *(sent today)* | 0.413 | 2,523 | $0.000396 | **$395** | 1,952 | 77.4 s | 133 s | 1,137 s |
| **4096 / 1** ← best | **0.549** | **3,349** | **$0.000298** | **$298** | 1,970 | 58.3 s | 169 s | 450 s |
| 2048 / 1 | 0.500 | 3,052 | $0.000327 | $327 | 1,970 | 64.0 s | 281 s | 598 s |

**+33% throughput, −25% cost, identical text.** Quality is unchanged across all
three: `uniq20` median 1.00, zero U+FFFD, 97–98 of 100 pages ok.

### Why there is an optimum rather than "lower is better"

2048 is worse than 4096. Below ~4096 tokens a legitimately long page no longer
fits in one generation and needs a continuation, and prefix caching is **off**
for this model (`enable_prefix_caching=False` in the pod's vLLM config), so each
continuation re-pays the full image prefill. It shows up precisely where
expected: p95 latency rises 169 s → 281 s while the median page is unchanged.

### Where the budget actually goes

The budget never binds a normal page — the median page is 1,952 characters,
roughly 600 tokens. It bounds the damage a **degenerate** one can do:

```
slowest 5% of pages consumed 43% of card time at cap 12288, 27% at cap 4096

consiglio_stato-sen-2013-201300001_p0003   12288: 1137s  38,014 chars  uniq20 0.040
                                            4096:  450s  17,586 chars  uniq20 0.071
dgn_cc_0151460094_2996246_p0045            12288: 1078s  25,282 chars  uniq20 0.045
                                            4096:  380s  11,714 chars  uniq20 0.048
```

`uniq20` is the share of distinct 20-character windows; 0.04 means 96% of the
output is repetition. At `12288 × (1+3)` one looping page may emit ~49,000
tokens. These pages still loop — the engine's repetition problem is untouched —
but each costs ~2.5× less and writes ~20,000 fewer characters of garbage.

**Recommended client setting: `max_tokens=4096`, `max_continuations=1`.** Both
are per-request fields: no deploy, no restart.

---

## 3. Engine comparison — surya vs chandra, same card, same budget, same pages

| engine | card | pages/s | pages/$ | $/page | chars med | notes |
|---|---|---|---|---|---|---|
| surya | A40 | 0.549 | 3,349 | $0.000298 | 1,970 | cap 4096 / cont 1 |
| chandra | A40 | *pending* | | | | cap 4096 / cont 1, 128 instances |

Chandra could not be measured at all until the pod image was rebuilt — see §5.

---

## 4. Quality, surya at cap 12288 (96 pages inspected)

```
chars    min 0   p25 1,213   median 1,952   p75 2,614   max 12,748
uniq20   median 1.000   p25 0.990          (low = repeating)
U+FFFD   0 on every page
looping (uniq20 < 0.5)   1 of 96
empty   (<50 chars)      7 of 96
```

**Zero replacement characters** across every page — the accent corruption this
corpus was suspected of is absent, with `surya_repair_accents` on.

Of the 7 empty pages, 4 have near-blank sources (1.7–12.7 KB) and **3 do not**
(9.6 KB, 9.6 KB, 59.9 KB) — ~3% of real pages returning nothing, which is now a
larger quality question than looping and is not yet explained.

Corpus hygiene, separately: **487 of 4,961 PDFs in the multi-page archive
(~10%) are structurally broken** — no `%PDF` header or no `%%EOF`. They fail on
arrival and cost nothing, but they will not be read either.

---

## 5. Defects found and fixed while measuring

Every one of these had to be fixed before any number could be trusted; each was
only visible once the previous one was gone.

| ver | defect | evidence |
|---|---|---|
| 0.4.24 | `/admin/api/system/reload` built a bare `Request({})` and raised on every call — it had never reloaded anything | `KeyError 'type'` |
| 0.4.25 | admission gate defaulted to **2 in flight / 6 queued**; a pod was handed 64 through env while the front in front of it admitted 2 | pod at `Running: 1, Waiting: 0`; 219 pages in 1210 s, same as reading them one at a time |
| 0.4.26 | the 0.4.25 diagnostic compared *requests* against *pages* and fired against the value coderai itself had just derived | warned "only 256 request(s)" about its own choice |
| 0.4.27 | `max_pods` counted only **healthy** pods, so an adopted pod that was briefly unhealthy left room under a cap of 1 | two A40s on a `max_pods=1` pool, $1.18/hr |
| 0.4.28 | `pod.inflight` was a ratchet — raised in `_pick`, lowered only in `release`, no TTL — so one lost page took a pod out of service permanently | pool said 64/64 in flight; pod said `Running: 31` at 10.6% KV |
| 0.4.29 | two pod-registration paths with no identity check between them | precautionary; see commit for what was and was not concluded |
| 0.4.30 | corrected an unsupported claim in 0.4.29 about what caused the double billing | — |
| 0.4.31 | `CODERAI_OCR_CHANDRA_INSTANCES` / `OLMOCR_INSTANCES` were **sent to every pod and read by nobody**; a pod has no config file, so those engines silently ran one instance | `[ocr] engine 'chandra': 1 instance(s) ready` → after the fix, `128 instance(s) ready`, `Running: 1` → `12–13` |

Pod image `ghcr.io/nextime/coderai-vllm-ocr:0.4.31` rebuilt, scanned, pushed and
signed — `sha256:9f96db69…`, cosign `{"git":"9632e66","version":"0.4.31"}`.

### Settings that matter, and were wrong by default

```
server.max_parallel_requests   2  →  sized to the pool
server.queue_max_size          6  →  sized with it
```

The gate must be derived from pool capacity, not fixed:

```
healthy_pods × max_inflight_per_pod ÷ pages_per_document
```

The same install needs **8** for ~9-page documents and **64** for single-page
ones. 0.4.25 derives it from `ocr.max_concurrency`, which does not know how many
pods exist; deriving from live pool capacity is still open.

---

## 6. Extrapolation to 996,760 pages

At the best measured setting (surya, A40, cap 4096 / cont 1):

```
pages/s per pod   0.549          →  1,976 pages/hour
$/page            $0.000298
GPU cost          ~$298
pod-hours         504
```

Cost is **independent of fleet size** — pages per dollar is fixed, so more pods
buy time, not money. Only boot overhead scales (~12 min per pod before it serves
a page, ≈ $0.12):

| fleet | wall clock | boot overhead |
|---|---|---|
| 1 pod | 504 h (21 days) | $0.12 |
| 16 | 32 h | $1.9 |
| 32 | 16 h | $3.8 |
| 64 | 7.9 h | $7.6 |
| 128 | **3.9 h** | $15.1 |

**Not included:** the ~10% structurally broken PDFs; the ~3% of real pages
returning empty; and the fact that these are single-page documents — a denser
page costs more tokens, so treat $298 as the floor for this page profile.

---

## 7. Open

- **A100-SXM4-80GB sweep for surya** (concurrency 32/64/96 at cap 4096/cont 1).
  An A100 SXM is ~3.4× an A40's price, so it must beat 3.4× on pages/s to win on
  pages/$. The A40 sat pinned at `Running: 32` with KV at only 12%, so its
  ceiling may not be VRAM at all — in which case the extra 32 GB buys nothing.
- **chandra** at the same budget and card (run in progress).
- **~3% of real pages return empty** — unexplained.
- **2 pages per 100 answered 502** from the pod in two consecutive runs.
- **Derive the admission gate from live pool capacity** rather than from
  `ocr.max_concurrency`.
- **Early loop detection.** A page is only judged degenerate *after* it has run
  the budget to its cap. `uniq20` collapses to ~0.04 on those pages, so the
  signal is cheap and available mid-stream; aborting there would cut the worst
  case far below what any budget can.
- `enable_prefix_caching=False` on the pod's vLLM makes every continuation
  re-pay the image prefill. Worth checking whether this model supports it.
