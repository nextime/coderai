# Pod admission control: a KV-aware gate instead of a guessed batch width

Status: **implemented** (M0-M3). `codai/api/admission.py`, with the gate in
`codai/queue/manager.py` and the measurement in `codai/api/vllm_worker.py`.
M4 and M5 below are not.

## The problem

vLLM admits up to `max_num_seqs` sequences without checking whether the KV
cache can hold them at the length they will reach. It discovers the shortfall
during *decode*, and its only recourse is to evict half-generated requests and
recompute them later.

The failure is expensive and almost invisible:

* Throughput collapses while every counter still looks busy — the engine is
  working, it is just doing the same work twice.
* It does not show up in GPU utilisation, and it cannot show up in VRAM: vLLM
  claims its `gpu_memory_utilization` share at boot, so a pod reads ~90% VRAM
  from the moment it starts whether it is serving one request or sixty-four
  (see `codai/api/serving_load.py`, module docstring).
* `vllm:num_preemptions` is the only direct signal, and nothing acts on it.

Measured shape of the problem: an OCR fleet dispatching 37–40 pages per pod
preempted intermittently rather than constantly — the mark of a limit sitting
almost exactly at the working set, where ordinary variation in page length
decides whether a batch fits. On the model entries involved, `vllm.max_num_seqs`
was unset, which means vLLM's own default: far more sequences than the cache can
hold.

## Why the obvious fix is the wrong one

The obvious fix is to set `max_num_seqs` per model from the cache size vLLM
reports at boot. It does not survive contact with the way pods work.

1. **Chicken and egg.** `max_num_seqs` is a launch flag. `GPU KV cache size: N
   tokens` is printed *after* vLLM boots. The value can only ever be applied to
   a later start.
2. **One config field, N cards.** `max_num_seqs` lives on a model entry and
   applies to every pod of that model, but cache size is a property of the card.
   A single OCR entry in the field lists six acceptable GPU types across three
   architectures.
3. **Same VRAM is not same cache.** vLLM selects a different attention backend
   per architecture and reserves a different workspace, so two 48 GB cards of
   different generations do not leave the same cache. A VRAM-ratio prediction is
   a guess; only the boot line is a fact.
4. **The learned record dies with the pod.** `_learned_gmu_path()` is
   `~/.coderai/vllm_gmu_learned.json` — container disk, wiped with the pod. Every
   pod re-discovers from scratch, so there is nothing to reuse.
5. **The cache key would be fragile.** `_learned_gmu_key` keys on VRAM, not card
   name, so all 48 GB cards collide. For the gmu learner that is harmless
   (escalation self-corrects upward); for a batch width a collision that
   *overestimates* capacity produces exactly the preemption it was meant to
   prevent, and nothing corrects it. The key would also have to pin the vLLM
   version, because `engine: coderai-vllm` resolves to `:latest` — the image can
   move under an unchanged config.
6. **It is the shape of a bug we have already shipped.** A persisted,
   measurement-driven limit that only moves down is how `measured_n_gpu_layers`
   ratcheted 31 → 20 under transient contention and stuck there, spilling to CPU
   with every number looking healthy. It needed an anti-ratchet guard and a
   config reset.

## The design

Do admission control in coderai's own queue, in front of the engine, and adjust
it at runtime from what the engine reports.

Three properties follow from the choice of signal:

**Gate on cache occupancy, not request count.** A request count means something
different on every card, so it needs a per-card table. A *percentage of the
cache* does not. `vllm:kv_cache_usage_perc` is already scraped on the pod by
`serving_load.engine_load()`. No measurement travels, no key, no record.

**It is congestion control.** Occupancy is queue depth, the preemption delta is
the loss signal, and AIMD converges on an unknown link capacity by construction
— which is the same thing as converging on unknown hardware.

**Nothing is persisted.** The controller re-converges from scratch on every pod.
It therefore cannot ratchet down and stick: the worst case is one pod serving
below capacity for its own lifetime, and with `idle_timeout_s: 900` that pod is
gone within the quarter hour. This is the direct answer to failure (6) above.

## What already exists

| piece | where | state |
| --- | --- | --- |
| admission gate in front of the engine | `queue_manager.max_parallel_requests`, `.max_size` (`codai/queue/manager.py:51-52`) | plain mutable ints |
| the gate consulted per request | `_can_start_now` (`manager.py:235-238`) | reads the attribute live |
| waiters released when capacity frees | `_wake_waiters_locked` (`manager.py:262`) | exists |
| gate values reported | `get_metrics()` (`manager.py:201-208`) | exists |
| gate set on a pod from env | `CODERAI_SERVER_MAX_PARALLEL_REQUESTS` / `..._QUEUE_MAX_SIZE` (`codai/main.py:1466-1481`), emitted by `pod_plan` (`runpod_worker.py:1356-1359`) | boot only |
| KV occupancy, queue depth, preemptions | `serving_load._WANTED` (`serving_load.py:61`) | scraped from the engine's `/metrics` |
| preemptions carried across engine restart | `serving_load._CUMULATIVE` | exists |
| the figures shipped to the orchestrator | `pod_telemetry._fetch` → `/internal/engine-state` | exists |

The gate is already a live attribute that the request path reads on every
admission. It is simply never written after boot — stated outright in
`pool_launch_env`: *"A pod keeps the environment it BOOTED with … None of it can
be changed from outside a running pod."*

## What is new

**M0 — the controller.** A periodic task on the serving process that reads the
engine's load and moves `queue_manager.max_parallel_requests` between a floor and
a ceiling. No new transport, no new store.

**M1 — bounds and policy as config**, rideable as env so a pod gets them (a pod
carries no config file):

| setting | default | meaning |
| --- | --- | --- |
| `admission_control` | `auto` | `auto` = on when the engine reports KV occupancy, off otherwise |
| `admission_kv_target_pct` | `0.85` | raise the limit while occupancy is below this |
| `admission_kv_ceiling_pct` | `0.95` | lower it above this, preemptions or not |
| `admission_min_parallel` | `4` | floor; never below what keeps the engine fed |
| `admission_interval_s` | `5.0` | control period (metrics TTL is 2 s) |
| `admission_backoff` | `0.75` | multiplicative decrease factor |
| `admission_recover_after` | `3` | intervals of quiet before increasing again |

**M2 — the configured value becomes a ceiling, not a setpoint.** This is the
compatibility property that matters: `pod_max_parallel_requests` already means
"how hard this pod may be driven", so the controller only ever runs at or below
it. No existing install becomes more aggressive, and with `admission_control`
off behaviour is byte-for-byte today's.

**M3 — projected occupancy.** See its own section below; it is what takes
steady-state preemptions from *small but nonzero* to zero.

**M4 — the orchestrator learns to read `waiting`** so intentional queueing is not
mistaken for overload (see failure mode 4). NOT DONE.

**M5 — the text path.** The gate implemented here is the engine-side one
(`queue_manager`), which is what governs OCR and the pipeline paths. Text
generation is gated separately, at the front, by `_model_capacity` — and on a
pod `models.max_model_instances` is left at its default, so a pod serving
`/v1/chat/completions` is held at one request at a time regardless of
`pod_max_parallel_requests`. Worth fixing, and a different change from this one.
NOT DONE.

## The controller

Each interval, read `kv_cache_pct`, `running`, `waiting` and `preemptions` from
the engine, and compare `preemptions` with the previous reading:

* **preemption delta > 0** — multiplicative decrease: `limit = max(floor,
  int(limit * backoff))`. Start the recovery hold.
* **occupancy > ceiling_pct** — same decrease. A leading indicator: back off
  before the recompute rather than after it.
* **occupancy < target_pct, no preemptions, recovery hold expired** — additive
  increase: `limit = min(configured_ceiling, limit + 1)`.
* **otherwise** — hold.

Additive increase and multiplicative decrease, because the costs are asymmetric:
being one slot too low wastes a little throughput, being too high throws away
work already done.

Raising the limit must call `_wake_waiters_locked()`. Without it, newly
permitted capacity sits idle until some in-flight request happens to finish,
which looks exactly like the controller not working.

## M3 — projected occupancy

### The blind spot it closes

Occupancy is a snapshot of a quantity that is **still growing**, and the growth
is already committed. Worked example, target 85%:

* **T+0s** — 20 requests running, each has produced 2,000 of an eventual 8,000
  tokens. Occupancy reads **70%**. The gate sees 70 < 85 and admits more.
* **T+30s** — those 20 requests have each reached 8,000 tokens. Their footprint
  has quadrupled, occupancy is far past 100%, and the engine preempts.

The gate was right about the present and wrong about the future. The preemption
was already determined at admission time, and a controller reading current
occupancy could not see it. Feedback always lags: it creeps up, overshoots, pays
a preemption, backs off. It converges, but on *small but nonzero* preemptions
rather than none.

### Gate on the commitment, not the level

If each running request will eventually hold `E` tokens, then:

```
projected = (running + 1) × E / kv_tokens
```

Current occupancy drops out of the formula entirely. The question is no longer
"how full is the cache" but "if everything now running ran to its expected
length, what fraction would it need". That is feed-forward, and the AIMD loop
becomes the error-correction term rather than the primary mechanism.

### Where E comes from, and why it fits

| half | hardware-dependent? | source |
| --- | --- | --- |
| `kv_tokens` | **yes** | measured on the pod at boot (`_pump_logs` → `kv_capacity()`) |
| `E` | **no** — a page is a page | `admission_expected_tokens`, else the engine's counters |

`E` is a property of the WORKLOAD, so it does not change when the pod lands on
a different card. That is what keeps the whole feature free of a per-card
table. Configured wins, because a bounded workload knows its own worst case (an
image's tokens plus a generation cap) while a measurement can only average what
has already happened. Unset, it is derived from counters `serving_load` already
collects — `requests_done`, `prompt_tokens`, `generation_tokens` — as a
**window**, because `requests_done` accumulates from boot and a lifetime mean
keeps folding in requests from before the workload changed. A measured `E` is a
mean, and preemption is driven by the tail, so it carries a safety factor.

### It needs no telemetry, and this corrects an earlier claim

An earlier draft said M3 depended on shipping `kv_tokens` to the orchestrator.
**It does not.** `_pump_logs` already captures it into
`_services[svc_key]["meta"]`, and the controller runs in that same process, so
it reads a local dict. No transport, no learned record, no per-card key, no
orchestrator involvement. Shipping it upward (it rides on
`/internal/engine-state`) is for humans to look at.

### A veto, never a licence

M3 may **block** an increase the feedback controller would have allowed. It must
never **permit** one the feedback controller would have blocked.

The asymmetry is the point. A wrong `E` is more dangerous than no `E`:
underestimate it, projected occupancy reads low, the gate over-admits, and the
result is a *confident* version of the bug — harder to diagnose than the
original. Veto-only means the worst case is over-conservatism, which costs a
little throughput and nothing else.

### What it is conservative about

`running` is a count, not a state: it says how many sequences exist, not how far
along each is, so a batch that is nearly finished is treated like one just
started. That is the right direction to be wrong in, and it is the price of
working from aggregate metrics rather than per-request introspection.

### Headroom is still required

`kv_target_pct` stays well below full regardless. A slightly idle cache is much
cheaper than a recompute, and `E` is an estimate.

## Failure modes to design against

1. **Ratchet and stick** — the `measured_n_gpu_layers` bug. Answered by never
   persisting the value, always being able to increase, and reporting the current
   limit in `get_metrics()` so it is visible rather than inferred.
2. **Narrowing into 429s.** Past `max_size`, surplus requests are rejected, and
   clients give up on errors far faster than on latency. The controller moves the
   *parallel* limit only and leaves the queue deep, so the effect is latency. A
   pod configured 192 in flight / 512 queued already has the right shape.
3. **Two queues in series.** If both the gate and the engine throttle, latency is
   the sum and neither owner can reason about the total. Keep `max_num_seqs`
   unset so the engine accepts whatever arrives: the engine executes, coderai
   admits. This preserves the split that already exists rather than changing it.
4. **The orchestrator misreading intentional queueing.** `scale_up_inflight_per_pod`
   could see deliberate holding-back as overload and rent another pod to solve a
   self-imposed limit. Telemetry already carries `waiting`; the scale-up path has
   to actually distinguish the two.
5. **Blocking the event loop.** `serving_load.engine_load()` is synchronous
   (`requests.get`). The controller must reach it through `asyncio.to_thread`, or
   it reproduces the wedged front that `loop_watch` and
   `tests/test_no_blocking_calls_on_the_loop.py` exist to catch.
6. **A rewinding counter.** `preemptions` is cumulative and rewinds when the
   engine restarts. `serving_load._CUMULATIVE` already carries it across a
   restart, but the controller must treat a *decrease* as a restart and reset its
   baseline rather than reading it as negative loss.
7. **No metrics at all.** llama.cpp and other engines expose none of this. With
   no occupancy reading the controller must do nothing and leave the configured
   value untouched — degrade to today, never to a guess.

## Out of scope

* Deriving or learning `max_num_seqs`. Superseded by this design.
* Persisting learned records so they outlive a pod. Still worth doing for the
  **gmu** learner, which currently re-discovers on every pod, but it is now an
  unrelated piece of work.
* Deriving `max_num_seqs` at all. The engine is left at its own default on
  purpose: if both the gate and the engine throttle, latency is the sum of two
  queues and neither owner can reason about the total. The engine executes;
  coderai admits.

## How we will know it works

Both figures are already collected per pod, so the criterion is falsifiable
without new instrumentation:

* `preemptions` stops rising, **and**
* `requests_done` per unit time does not fall.

Either alone is meaningless: preemptions go to zero if the gate simply chokes the
engine, and throughput is unchanged if the controller never engages. The pair is
the claim.

## Tests

* AIMD: decrease on a preemption delta; decrease above the ceiling with no
  preemptions; increase only after the recovery hold; hold otherwise.
* Clamping: never below `admission_min_parallel`, never above the configured
  `pod_max_parallel_requests`.
* A decreasing `preemptions` reading is treated as an engine restart, not as
  negative loss.
* Absent occupancy (no engine, non-vLLM engine, unreachable `/metrics`) leaves
  the configured value untouched.
* `admission_control` off reproduces today's behaviour exactly.
* Raising the limit wakes waiters: a queued request starts without any in-flight
  request finishing.
* The controller does not run blocking I/O on the loop (extend the existing AST
  test's coverage to the new module).
* The env names reach a pod through `pod_plan`.

## Where it lives, and why there

The controller runs in the process that SERVES — on a pod, the pod's own engine
process — because that is where the engine's metrics and its measured cache size
already are. Nothing travels and nothing is seeded.

Two placement details are load-bearing and were both found the hard way:

**The policy is read by `ConfigManager`, not by `codai/main.py`.** A pod never
runs `main.py`: its `boot.sh` execs `python -m uvicorn codai.api.app:app`. Both
`CODERAI_SERVER_MAX_PARALLEL_REQUESTS` and `..._QUEUE_MAX_SIZE` had been emitted
by `pod_plan` since pods shipped and read only in `main.py`, so on every rented
pod they were set and then ignored — the shared-workstation defaults of 2 in
flight / 6 queued stood, and the gate those comments describe was never applied
to the machine it was written for. `ConfigManager.load()` runs under both entry
points. Pinned by `tests/test_the_pod_gate_env_is_actually_read.py`.

**The controller starts in the lifespan**, for the same reason the seed
registration does.

Only an explicitly-set policy key is forwarded to a pod. An unset knob must
arrive absent rather than as today's default restated, or a pod would be pinned
to a default it was never configured with and would not follow a later change to
it.

## Files

| file | what |
| --- | --- |
| `codai/api/admission.py` | `decide()` (pure), `AdmissionController`, `Policy`/`State` |
| `codai/queue/manager.py` | `dynamic_cap`, `effective_parallel_limit()`, `set_dynamic_cap()` |
| `codai/api/vllm_worker.py` | `_MAX_CONCURRENCY` regex, `kv_capacity()` |
| `codai/config.py` | the `admission_*` settings, and the server env table |
| `codai/api/app.py` | starts it; reports it on `/internal/engine-state` |
| `codai/api/runpod_worker.py` | `_pod_admission_env()`, forwarded in `_plan` |
| `tests/test_admission_control_is_kv_aware.py` | 27 tests |
| `tests/test_the_pod_gate_env_is_actually_read.py` | 12 tests |
