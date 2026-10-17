# Deferred: a density router between surya's full-page and block modes

**Status: noted, not implemented.** Decide after the next campaign is measured
with the corrected token budget (see "What has to happen first").

## The idea

Run layout detection first (cheap), count the blocks on the page, and choose the
recognition mode from the count:

* **few blocks** — full-page mode, one generation, lowest overhead
* **many blocks** — block mode, one short generation per block, which cannot
  run away and which continuous batching can overlap

A threshold of ~50 blocks was suggested. If we build this, the threshold must be
a **config setting, not a constant**: block count per page is a property of the
corpus, and an install reading dense multi-column judgments has a different
knee from one reading invoices.

## Why it is attractive, in our own numbers

The strongest argument is not reliability, it is **GPU utilisation**. Measured
on live pods while the orchestrator believed each held ~17 pages:

```
Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%   (6 of 8 samples)
Running: 4 reqs,                  GPU KV cache usage: 1.4%   (the busiest)
Avg generation throughput: 489 / 0.0 / 463.9 / 183.0 / 0.0 tokens/s
```

One full-page generation is a **single sequence**: it cannot be batched with
itself, so a pod with one page in flight runs one sequence and the card idles.
Block mode would submit ~40 short sequences per page, which is exactly what
continuous batching is for. That is consistent with the per-card measurement
too: an A100 PCIe delivered **1.14x** an A40, not the 2.8x its bandwidth
suggests — a workload that does not respond to a faster card but might respond
to a fuller batch.

`SURYA_INFERENCE_BACKEND=vllm` is **already set** (`codai/ocr/surya.py`), so the
continuous batching the idea depends on is in place. Nothing to do there.

## Why it is NOT the first thing to try

The repetition that block mode would prevent appeared **at the truncation
boundary**, not spontaneously:

```
[surya] not resuming a reply that is already repeating itself
        — it stopped at the 3072-token cap
```

A continuation re-sends everything already produced, so a cut-off page was fed
~3072 tokens of its own output and asked to continue — a reliable way to induce
a loop. Measured across 594 continuations: **532 pages reached segment 2**, 44
segment 3, 11 segment 4, 7 segment 5. That is a budget marginally too small,
not a mode that cannot read dense pages.

With `surya_max_tokens: 12288` and `surya_max_tokens_floor: 8192` (v0.4.22),
~90% of those pages should finish in ONE generation with no resumption, and a
large part of the looping should disappear on its own. Rebuilding the page path
first would risk fixing something the budget already fixed.

## What has to happen first

Measure one campaign with the corrected budget and record:

1. **pages/s** overall, and per pod.
2. **the repeat rate** — `repeat_detected` per page, and how many pages reach
   segment 2 at all. If segment-2 pages collapse from 90% to a few percent, the
   loops were the cap.
3. **GPU utilisation** on a pod — `Running: N reqs` and `GPU KV cache usage`. If
   the cards are STILL idle at 1-2% with pages in flight, that is the case for
   the router, independent of loops, and it becomes the priority.

Then: if (2) is fixed and (3) is still bad, build the router. If (2) is not
fixed, build the router sooner — full-page mode genuinely cannot read those
pages.

## If we build it

* Layout detection is already available in the worker (`self._det`,
  `self._det_proc`), so the pass is not new machinery.
* The call ladder in `codai/ocr/workers/ocr_worker.py::_predict` exists because
  the predictor API drifted across surya versions. A block-mode call has to join
  that ladder the same way — `layout_results=` is not available in every version
  we support, and a router that raises on an older surya is worse than no router.
* The mode chosen must appear in the page diagnostics. A page read in block mode
  and a page read whole are different products, and a reader of the result
  should not have to guess which it got.
* Keep full-page as the default for sparse pages: it is fewer round trips and,
  on a page that fits one generation, strictly cheaper.
