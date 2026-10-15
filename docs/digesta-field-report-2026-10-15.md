# Field report from Digesta — 15 Oct 2026

What a client hit in one night of production OCR, chunking and embedding against
the production orchestrator on `ru002848`. Every item was reproduced; each one
says what was run, what came back, and where in this tree the relevant code sits.

Written by the Digesta side. Where a cause is a guess it says so; where a number
is measured it says when and on what.

**Read [P1-a](#p1-a--surya-returns-200-with-empty-pages-after-a-lost-lease) first.**
It is a silent-wrong-answer bug, it is the most expensive thing here, and it
invalidates some of our earlier engine measurements (see
[What we can no longer claim](#what-we-can-no-longer-claim)).

It affects **surya only**: on the same documents, in the same minutes, chandra
and olmOCR both returned text normally. So OCR is not broken across the board —
one engine is, silently.

| # | Severity | Area | One line |
|---|---|---|---|
| [P1-a](#p1-a--surya-returns-200-with-empty-pages-after-a-lost-lease) | **P1** | OCR | surya returns `200 OK` with empty pages after a lost lease; chandra and olmOCR are fine |
| [P1-b](#p1-b--runpodscale-404s-for-every-model-cannot-scale-or-tear-down) | **P1** | RunPod | `/v1/runpod/scale` 404s for every model — cannot scale **or tear down** |
| [P2-a](#p2-a--bge-m3-runs-twice-alias-and-canonical-id-are-two-models) | P2 | RunPod | `bge-m3` and `BAAI/bge-m3` each rent their own A40 |
| [P2-b](#p2-b--an-adopted-pool-pod-cannot-be-released) | P2 | RunPod | An adopted pool pod cannot be released through the API |
| [P2-c](#p2-c--v1chunk-no-longer-matches-the-spec) | P2 | Chunking | `/v1/chunk` boundaries diverge; a 162-char fragment; `norm_refs` wrong |
| [P3-a](#p3-a--scale-blocks-for-minutes-and-starves-status) | P3 | RunPod | `scale` blocks for 10+ minutes and starves `/v1/runpod/status` |
| [P3-b](#p3-b--a-trickling-response-never-hits-the-read-timeout) | P3 | OCR | A trickling response never hits the client read timeout |
| [OK](#what-works) | — | — | `/v1/embeddings` now correct; the stale-pod reaper works |

Environment: orchestrator `v0.3.85` (`4dbe7f0`), pods on `A40 SECURE`, engine
`coderai-vllm-ocr` (rasterisation on the pod), client `127.0.0.1:8777`, OCR at
300 DPI with `max_tokens=3072`, `max_continuations=4`.

---

## P1-a — surya returns `200` with empty pages after a lost lease

### What we ran

One `POST /v1/ocr`, `engine=surya`, a 10-page scanned PDF (Consiglio di Stato,
1951, typewritten), `dpi=300`.

**Not a problem with difficult old scans.** We re-ran it on one of the newest
documents in the corpus — a 2026 `merito_gdp` PCT filing, one page — and got the
same thing: `200 OK`, **0 characters**, in 5.6 seconds. The orchestrator logged
86 further `unknown OCR engine lease` errors during that eight-minute window.
Nothing here depends on the document; the age of the first test was our own
poor choice of sample, not the cause.

### What came back

`HTTP 200`. `num_pages: 10`. Every page `text: ""`, `lines: []`, `regions: []`.
The whole document's `text` was eighteen newline characters. No error field
anywhere in the response.

The per-page `meta` is where it gets interesting — this is verbatim:

```json
{
  "model_attempts": {
    "calls": 4, "tokens": [0, 0, 0, 0], "cap": 3072, "max_continuations": 4,
    "why": "surya re-ran this page because the generation repeated itself; each re-run is a full generation at a higher temperature"
  },
  "page_ink": {
    "ink": 0.20343, "threshold": 0.005, "blank": false,
    "why": "20.343% of this page's pixels are not background, so there IS something on it -- a short result here is the engine failing to read it, not an empty page."
  }
}
```

So the orchestrator **correctly diagnosed its own failure** — 20% ink, not
blank, "the engine failing to read it, not an empty page" — and still answered
`200` with empty text and no error.

### What was actually happening

From the orchestrator's log, hundreds of times per second:

```
[remote-gateway] /v1/ocr/attach/<token>/v1/chat/completions -> https://194.68.245.192:22193 (model 'datalab-to/surya-ocr-2' on its own RunPod pod)
Inference error: Error code: 404 - {'error': {'message': 'unknown OCR engine lease', 'type': 'invalid_request_error'}}
[surya] page repeated itself — retry 1/1 (temperature rises; a retried page costs another full generation)
```

120 occurrences of `unknown OCR engine lease` in 400 log lines. That string is
minted by `_unknown_lease()` in `codai/ocr/gateway.py:313`.

`codai/ocr/surya.py:281` already documents what it means:

> It answers 404 "unknown OCR engine lease" when the lease really is gone
> (the front restarted), and that is the one case re-resolving fixes.

The front **had** restarted, about three minutes before our call (the container
reported `Up 3 minutes` at 10:45). So this is precisely the documented
re-resolve case — and the re-resolve did not happen. Instead:

1. the 404 propagated as a page-level generation failure;
2. `[surya] page repeated itself — retry 1/1` fired on it, so the repetition
   detector **misread an HTTP 404 as a repeated generation**;
3. four attempts per page, `tokens: [0,0,0,0]`, each a full paid generation;
4. the empty result was returned as a successful `200`.

### Only surya

In the same minutes, on the same two documents, **chandra and olmOCR both
returned text normally** (1,472 / 946 and 2,990 / 2,351 characters — see the
[engine comparison](#engine-comparison-on-current-documents)). Every
`unknown OCR engine lease` line in the log names
`model 'datalab-to/surya-ocr-2'`. So this is one engine's lease handling, not
the OCR subsystem as a whole — which is also why it went unnoticed: the other
two tiers of the cascade work.

### Why it costs us so much

Our pipeline treats a page with no text as an unreadable scan. A `200` with
empty pages is therefore indistinguishable from a genuinely blank page, so the
documents are consumed from the work queue and written off. Last night that
pattern marked roughly 1,000 documents failed in one run.

It also burns GPU time: a broken lease becomes a hot retry loop against a rented
A40 rather than one clean error.

### What we would like

1. **Never return `200` for a page the orchestrator itself judged unread.** When
   `page_ink.blank` is `false` and the engine produced no tokens, that is an
   error — `503` or a per-page `error` field, anything a client can branch on.
   You already compute the judgement; it just does not reach the response.
2. **Do not count a transport error as a repetition.** `tokens: [0,0,0,0]` with
   a 404 upstream is not a repeating generation, and the temperature-raising
   retry cannot help it.
3. **Re-resolve the lease on 404**, as `surya.py:281` says is correct, instead
   of surfacing it as a page failure.
4. Ideally, hold OCR requests briefly across a front restart, or fail them
   loudly, rather than serving them against leases that no longer exist.

---

## P1-b — `/v1/runpod/scale` 404s for every model (cannot scale **or tear down**)

Every model id we tried, including ones with pods **currently running and
serving**:

```
POST /v1/runpod/scale {"model":"datalab-to/surya-ocr-2","pods":0,"teardown":true}
  -> 404 {"detail":"'datalab-to/surya-ocr-2' has no runpod block"}
POST /v1/runpod/scale {"model":"datalab-to/chandra-ocr-2","pods":0,"teardown":true}
  -> 404 {"detail":"'datalab-to/chandra-ocr-2' has no runpod block"}
POST /v1/runpod/scale {"model":"allenai/olmOCR-2-7B-1025-FP8","pods":1}
  -> 404 {"detail":"'allenai/olmOCR-2-7B-1025-FP8' has no runpod block"}
POST /v1/runpod/scale {"model":"BAAI/bge-m3","pods":1}
  -> 404 {"detail":"'BAAI/bge-m3' has no runpod block, and no pod pool is running
           under that name either. Pass a model id, or a pool key that
           /v1/runpod/status is currently showing"}
```

Meanwhile `/v1/runpod/status` reports those very models with ready pods:

```json
{"pods_total": 6, "pods_ready": 6, "hourly_usd": 3.54,
 "by_model": {"pool:digesta-qwen38": 1, "bge-m3": 1, "BAAI/bge-m3": 1,
              "datalab-to/surya-ocr-2": 1, "datalab-to/chandra-ocr-2": 1,
              "allenai/olmOCR-2-7B-1025-FP8": 1}}
```

Earlier in the same session, `scale` with `pods: 1` for surya and chandra
**did** work and provisioned those pods. Between then and the teardown attempt
the orchestrator restarted. So a `scale` config appears to be lost on restart
while the pods survive — leaving pods that nothing can turn off.

### The clearest form of it

Reproduced at 12:58, with no restart in between — `scale` accepted this exact
model id fifteen minutes earlier to CREATE the pod:

```
POST /v1/runpod/scale {"model":"datalab-to/chandra-ocr-2","pods":0,"teardown":true}
  -> 404 {"detail":"'datalab-to/chandra-ocr-2' has no runpod block, and no pod
           pool is running under that name either. Pass a model id, or a pool
           key that /v1/runpod/status is currently reporting."}

GET /v1/runpod/status
  -> "by_model": {"pool:digesta-qwen38": 1, "datalab-to/chandra-ocr-2": 2}
```

The error asks for "a pool key that `/v1/runpod/status` is currently
reporting". That is precisely the string that was passed, and `status` is
reporting it, with two pods. Whatever name `teardown` wants, there is no way to
discover it from the API.

Note also the **2**: we asked for one chandra pod and got two, the second still
booting. We never requested it.

**This is the one with a running meter.** We could not stop six pods at
$3.54/hr from the API; the only way out is the RunPod console. If a name in
`status.by_model` is not a valid `scale` target, the two should at least agree,
or the error should say which name to pass. Please also consider making
`teardown` accept whatever `status` prints.

---

## P2-a — `bge-m3` runs twice: alias and canonical id are two models

`/v1/runpod/status` showed both, each with its own A40:

```
"by_model": {"bge-m3": 1, "BAAI/bge-m3": 1, ...}
```

and the gateway log confirms two distinct pods:

```
[remote-gateway] /v1/embeddings -> https://194.68.245.75:22041 (model 'bge-m3' on its own RunPod pod)
[remote-gateway] /v1/embeddings -> https://194.68.245.2:22051 (model 'BAAI/bge-m3' on its own RunPod pod)
```

`/v1/models` lists both names, so a client has no way to know they are the same
weights. One of them is pure waste (~$0.59/hr). Aliases should resolve to one
pool before anything is rented.

---

## P2-b — an adopted pool pod cannot be released

```
POST /v1/runpod/scale {"model":"pool:digesta-qwen38","pods":0,"teardown":true}
  -> 404 {"detail":"'pool:digesta-qwen38' has no runpod block"}
```

None of the orchestrator's paths can terminate it. That pod has now been
billing for about twelve hours because the only way to stop it is the RunPod
console. An adopted pod that `status` reports should be releasable the same way
it is reported.

---

## P2-c — `/v1/chunk` no longer matches the spec

Same input text, same parameters (`template=it-legal`, `chunk_tokens=800`,
`overlap_tokens=120`), compared against our in-house implementation that
[chunking.md](chunking.md) specifies:

| | chunks | lengths |
|---|---|---|
| Digesta | 2 | 2736, 3525 |
| coderai | 3 | 2448, 3166, **162** |

No boundary matches. Earlier in this work the boundaries matched byte-for-byte
on three real documents (9/9, 16/16, 11/11 chunks, with identical sections and
parents), so this is a regression since then.

Two specifics:

- **The 162-character chunk** is the `P.Q.M.` (the operative part). A fragment
  that size is noise in a retrieval index. Our rule, and what
  [chunking.md](chunking.md) describes, is that a short trailing section merges
  into the previous chunk rather than standing alone.
- **`it-legal` and `it-legal-strict` return byte-identical output** on this
  input. If the strict variant is meant to differ, it is not differing.

Still outstanding from the earlier round, unchanged:

- `norm_refs` emits bare article numbers without a code (`art.380`, `art.360`,
  `art.115`). The spec requires a code suffix (`c.c.`, `c.p.c.`, `c.p.`,
  `cost.`) precisely because a bare "art. 360" is ambiguous — these are false
  positives in the field whose only purpose is exact-match filtering.
- `norm_refs` omits acts entirely: we extract `d.p.r.115/02` and
  `d.lgs.446/1998` where coderai returns neither.

---

## P3-a — `scale` blocks for minutes and starves `status`

A single `POST /v1/runpod/scale {"pods": 1}` did not return within **616
seconds** for olmOCR and **351 seconds** for bge-m3 (it then 404'd — see P1-b).
Clients time out at 60 s and at 300 s. While a search is in progress,
`/v1/runpod/status` also stops answering, so a slow rental and an outage look
identical from outside. That ambiguity cost us two wrong diagnoses in one night.

An immediate `202` with a request id, plus a `status` that keeps answering,
would make this observable.

## P3-b — a trickling response never hits the read timeout

A chandra `POST /v1/ocr` with an httpx `timeout=420` stayed open for about
nineteen minutes. An httpx read timeout is per-read, so a response that dribbles
bytes never trips it, and a client cannot bound the call without a separate
wall-clock guard. A server-side cap, or at least documenting that clients need
their own total-time guard, would help.

---

## What works

- **`/v1/embeddings` is correct now.** Three inputs, `bge-m3`: dense 1024-dim,
  `sparse_embedding` with real `indices`/`values` (9–10 terms), plus
  `colbert_embedding`. Both the alias and the canonical id answer. This was
  broken earlier (it tried to load the model in-process on the GPU-less host
  and returned `Found no NVIDIA driver on your system`); `4dbe7f0`
  ("embeddings: detecting the head is not having it") appears to have fixed it.
  We only discard `colbert_embedding` because storing it would cost us ~2 TB.
- **The stale-pod reaper works.** After a series of restarts it found and
  terminated three untracked pods (~$0.26 each) that would otherwise have
  billed silently.
- **Surya's loop detection and its refusal to resume a repeating reply are the
  right behaviour**, and the `page_ink` diagnostic is genuinely good — it is
  what let us prove P1-a rather than guess at it. The problem in P1-a is only
  that the judgement never reaches the client.

---

## Engine comparison on current documents

The first same-document comparison we have managed to complete. Two recent
`merito_gdp` PCT filings, all three engines, 300 DPI, within the same few
minutes, pods warm:

| document | pages | surya | chandra | olmOCR |
|---|---|---|---|---|
| `dgn:gdp:1030720152:4284956` | 1 | **0 ch** · 5.6 s | 1,472 ch · 28.1 s | 946 ch (64%) · 7.1 s |
| `dgn:gdp:1030720152:4157356` | 2 | **0 ch** · 6.7 s | 2,990 ch · 25.0 s | 2,351 ch (79%) · 8.6 s |

Reading it, with chandra as the reference:

- **surya is unusable right now** — zero characters on both, the P1-a lease
  failure, and it fails fast rather than slowly.
- **chandra produces the most text** and is the slowest: 25–28 s for one or two
  pages.
- **olmOCR returns 64–79% of chandra's text at 3–4× the speed.** That is a far
  smaller gap than the 4–8× deficit we measured last night on 1951 scans, which
  is one reason we are retracting that figure below.

Two documents is a small sample and both are born-digital PCT filings rather
than hard scans, so treat the ratios as indicative. We would rather publish two
measured documents than repeat last night's numbers.

## What we can no longer claim

Measurements we reported earlier that today's results overturn or undermine.

- **"surya loops on 1951–1971 typewritten CdS scans" — withdrawn as a cause.**
  surya now returns zero characters on *every* document we try, including
  one-page 2026 PCT filings, and the log shows the reason is the lease 404, not
  repetition. The loop detector did fire, but on the error rather than on a
  repeating generation. Whatever genuine difficulty those old scans present is
  buried under this bug and cannot be measured until it is fixed. In particular,
  the `ocr_troncato` epidemic we blamed on the model (31 of 32 failures in one
  job) should be re-attributed to the lease path.
- **"olmOCR returns 4–8× less text than chandra on identical pages" — not
  reproduced.** On current documents it returns 64–79% of chandra's text (see
  the table above), which is a different conclusion: a reasonable quality/speed
  trade, not a broken tier. The 4–8× figures came from 1951 scans in a run that
  also contained the lease failure. We are not claiming the old numbers were
  wrong about those particular pages — only that they do not generalise and we
  should not have presented them as an engine verdict.
- Last night's ~1,000-document, 6-success run should not be read as an engine
  verdict at all.

Our own controller was also at fault in that run — gates that skipped their
checks on thin data and a cancel-then-submit race — so please do not take the
failure rate as a coderai number either.

## What we would like to re-run once P1-a and P1-b are fixed

1. surya against chandra and olmOCR on identical documents, old scans **and**
   modern filings — the comparison above has surya at zero, so it says nothing
   about surya's quality.
2. DPI 300 vs 250 vs 200 on output quality.
3. `/v1/chunk` against the [chunking.md](chunking.md) fixtures, to confirm the
   boundaries match again.
4. Then the full perimeter: ~103k Diogene documents, chunked and embedded.
