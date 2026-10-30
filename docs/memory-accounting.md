# How coderai works out what a model needs

Every placement decision — resident on the GPU, accelerate CPU offload, how
much to evict before a load, whether to spill to disk — is made against one
number per model. This is where that number comes from, and what it does and
does not include.

## The model is measured, not the process

`measured_vram_gb` is a model's **full footprint**, and it is deliberately
placement-independent: offload moves weights between GPU and host, it does not
change how much the model needs. The split is reported separately.

It is obtained by walking the loaded object and adding up the bytes its tensors
really occupy (`codai/models/footprint.py`), plus the KV cache for the
configured context, plus a runtime reserve for activations and the decode
spike.

Quantization needs no correction factor, and that is the whole reason for
measuring the object rather than estimating from config:

* a bitsandbytes `Params4bit` **is** a `uint8` tensor, so `numel ×
  element_size` already yields the packed size;
* GPTQ and AWQ keep `qweight` / `qzeros` / `scales` in **buffers**, so buffers
  are walked as well as parameters — a parameters-only walk reports a 4-bit
  model as very nearly empty;
* bitsandbytes keeps `absmax`, the code book and (under double quantization) a
  nested second-level state in `quant_state`, outside `parameters()`. That is
  real memory and is counted explicitly.

Tied weights — an embedding shared with the LM head — are one allocation and
are deduplicated by storage pointer. An accelerate `meta` placeholder, left
where the real weights were moved to host RAM or disk, is sized from its
declared dtype and flagged, because the model still needs those bytes.

A diffusers pipeline is not an `nn.Module`; it is walked through its
components, skipping the schedulers and tokenizers that hold nothing. A
text/vision model is registered as a `ModelManager` wrapper, so its real module
and its config are reached through `.backend.model`.

## Context is costed from the geometry

The KV cache is sized from the number of **key/value** heads, not attention
heads. Under GQA many query heads share one KV head, so sizing by attention
heads overstates the cache by the grouping factor — fourfold on a 32/8 model.
Sliding-window layers are sized to their window rather than the full context
when `swa_full` is off: a model that alternates a windowed layer with a global
one would otherwise be overstated on half its layers.

**A computed KV figure is an estimate and is labelled as one.** Per-layer
KV-head arrays, shared KV and architecture-specific cache layouts are not
modelled, and predicting this from GGUF metadata has been wrong by 5× in both
directions on these very models. A measured `measured_kv_gb` for the context in
question — llama.cpp reports its own KV buffer size exactly — always wins.

## What a GGUF needs

A GGUF is loaded as stored, with no dtype conversion, so its weights **are** the
file, byte for byte. llama.cpp owns that memory and there is no torch object to
walk, so the file size is used, plus the KV cache and the reserve. A companion
vision projector (`mmproj`) loads on top and is added.

## A measurement that cannot be trusted is refused

`force_vram_update` refreshes these fields on every load, so a bad reading
becomes authoritative and is planned against. Three checks stand in the way:

1. **An empty load.** If the object is walkable and holds no weights at all,
   the load did not complete. There is nothing to fall back to — a size taken
   from process memory measures everything *except* this model — so nothing is
   recorded, and the refusal is written to the history with its reason.
2. **An implausible size.** No real model is under ~10 MB.
3. **A collapse against the previous reading.** Less than half of what the
   *same quantization signature* measured before is a partial load. A genuine
   drop arrives together with a signature change — adding 4-bit quantization
   to a model's transformers took one from 40.04 GB to 11.73 GB — and is let
   through. The signature covers precision, `load_in_4bit`, `load_in_8bit` and
   `component_quantization`.

The log names which check fired, because the three call for opposite responses.

## Why not process memory

The footprint used to be a process-RSS delta across the load. RSS is the wrong
instrument: it counts file-backed pages that the kernel maps when the loader
reads a mmap'd safetensors or GGUF, and summed over a process tree it counts
pages **shared** between those processes once per process — the interpreter,
torch and the CUDA runtime are mapped into every engine. As a delta it also
absorbs anything else that loaded, evicted or faulted pages in the same window.

Measured against ground truth on a 5-engine host:

```
gemma-2-9b-it.Q4_K_M    5.37 GB on disk  ->  reported 30.81 GB   (5.7x)
LTX-2.5, six loads, one unchanged config:
    22.11, 22.11, 23.28, 4.50, 19.02, 29.80 GB                   (6.6x)
```

Every other model on that host sat at 1.0–1.2×. The 4.50 GB reading came from a
load that placed a single layer and then raised; `Full-GPU load looks viable
(4.5 GB peak need)` was planned off it four seconds later.

## The host-RAM cap

`offload.max_ram_gb` is enforced against proportional set size
(`Pss_Anon + Pss_Shmem`, from `/proc/<pid>/smaps_rollup`), summed over the
front, the engines and their workers. PSS has already divided each shared page
by the number of processes sharing it, so summing it across a tree counts every
page exactly once; `Pss_File` is excluded because mapped file pages are page
cache that the kernel reclaims by itself and that evicting a model cannot free.

The cap can only be met by releasing memory something actually holds, so the
metric has to be one that releasing memory moves.

## A load that cannot fit in host RAM is refused

The cap above is a *pressure* mechanism: a watcher samples the fleet's memory
and, past `offload.ram_watch_soft_fraction` of the cap, runs a ladder — collect,
trim, drop the upscaler cache, drop prefix caches, evict idle models. Every rung
releases memory that is already held. None of them can decline a load that is at
that moment streaming weights into the host.

That gap has a cost, measured on a 54 GB host running four engines:

```
Oct 29 23:03  Killed coderai-nvidia-gguf   anon-rss 46.3 GB
Oct 30 03:10  Killed coderai-nvidia        anon-rss 47.7 GB
Oct 30 03:36  Killed coderai-nvidia        anon-rss 31.6 GB
Oct 30 03:49  Killed coderai-nvidia        anon-rss 32.7 GB
front: engine#0 exited (code -9); respawning (1/5 in 120s)
```

The watcher was awake throughout and logged
`RSS 50.6/43.0 GB (118%) — mitigation [gc+trim+evict_idle] → 53.7 GB`: it ran
its ladder and the number went up anyway. The kernel settled it, and SIGKILL
takes the engine, not the request — so the client that asked for the oversized
model saw a dropped connection, and so did every unrelated request in flight,
plus everything that arrived during the 120-second respawn. One bad load is
reported as the whole server being unreliable.

So `offload.ram_admission` (default on) adds the missing half: before a load
starts, estimate what it will cost the host and refuse it if it does not fit.
`codai/models/host_ram.py` holds the arithmetic, `ensure_host_ram_for()` the
decision, and it sits beside `ensure_vram_for()` on every load path.

The estimate follows what actually determines the cost:

| situation | host-RAM cost |
| --- | --- |
| the model loaded before | `measured_ram_gb` — what it really parked on the host |
| `offload_strategy` `model` / `sequential` / `cpu` | the whole footprint; that is what was asked for |
| `offload_strategy: none`, or `no_ram` | the load buffer only — it fails on VRAM instead |
| anything else | the **spill**: footprint minus free VRAM |
| footprint unknown | the load buffer only, and the load proceeds unguarded |

plus a 2 GB staging allowance for the shard buffer, the safetensors read window
and dequantisation scratch.

The budget is the tighter of what the kernel says is available (`MemAvailable`,
which already counts reclaimable cache) and what is left under `max_ram_gb`,
less `offload.ram_admission_floor_gb` (default 4 GB) kept free for the OS and
the sibling engines.

Refusal is the last step, never the first: the ladder runs, the budget is
re-measured, and a load that fits afterwards is admitted. Only then does the
request fail — as a 503 naming the model, the estimate, the budget, the
shortfall, and what to change on the card. A model whose footprint could not be
determined is never refused on a number nobody has; it loses the protection and
says so in the log.

### Sizing that skips itself is worse than no sizing

The same night, the GGUF path printed this before loading a 35B:

```
(auto-offload sizing skipped: cannot access local variable '_ck'
 where it is not associated with a value)
```

The auto-offload step reads the configured KV-cache types to decide how much
headroom to reserve, and it read them from a binding made a hundred lines
further down, where `llama_kwargs` is assembled. Every load that reached the
branch raised `UnboundLocalError`, the handler reported it as *skipped*, and
nothing reduced `n_gpu_layers`. llama.cpp was then asked to keep every layer on
a card that could not hold them and answered with a **43,562 MiB CPU KV
buffer** — which is what took the host down.

Two things follow. The read is hoisted above its use, with a test asserting the
source order rather than a line number. And the handler no longer says
*skipped*: a step that did not run leaves the load UNSIZED, and the message says
that, because "skipped" invites the reader to assume nothing needed doing.

## A resident model still needs room to run

Every reservation above is taken when a model LOADS. A model that is already
loaded asks for nothing — and a pass still costs more than the weights.

Measured on a production render, 2026-10-29: a speech model sat resident
between scenes while the TTS and visual models of the same job loaded around
it and filled the card. Then every scene logged

```
word timestamps failed (500: … CUDA out of memory. Tried to allocate 86.00 MiB)
```

and every caption in the render lost its word timings. 86 MiB is what a
resident model needs to run one window, and nothing in the system was keeping
it.

Three numbers per speech model, all measured after its first pass:

| | |
|---|---|
| `weights` | what the load put on the card |
| `activation` | what one pass needed **on top of what was already allocated** |
| reserve | `weights + activation`: what to free before loading it |

`activation` is what is now ensured free before each pass
(`ensure_free_vram`), and the reserve is what is freed before a load.

It is a delta on purpose. The process's absolute peak would charge a speech
model for every other model resident beside it — a 16 GB image model sitting
on the card during a transcription would be recorded as part of the speech
model's footprint, and every later load of it would evict the world. Both are
high-water marks, never averages: a one-word clip touches a fraction of what a
full 30 s window does, and an average under-reserves for the longer audio that
follows. Until a model has run once, a 1 GB floor stands in for `activation` —
deliberately modest, because the model is already resident and the failure
being prevented was 86 MiB.

The same checkpoint can be served in-process or by an isolated worker, and the
two used to disagree about its size: 3 GB in one path against 8 GB in the
other, where 8 GB was the measured one. They now share one function, so
eviction frees the same amount whichever way the model is running.

An OOM that still happens mid-pass frees the card and retries once. A pass is
seconds of work on audio that already exists; the expensive outcome is a
render whose captions have no timings.

## Idle unload

Eviction frees a model when something else needs the memory.
`offload.idle_unload_seconds` (default 1800) covers the case where nothing
does: the on-demand path keeps a model resident whenever there is room, which
left a 3.5B music model on 7.9 GB of a 24 GB card for hours after its last
request.

It is deliberately generous. Reloads are the expensive event — one production
day spent 2h04m of 21h30m inside model loads — so a short timeout trades a VRAM
saving nobody asked for against a reload everybody waits on. A model whose LRU
clock was never stamped is left alone rather than assumed idle.

## Seeing it happen

`GET /v1/models/history` returns what was loaded and evicted and why, newest
first, merged across engines — a swap is a cross-engine event, since one engine
evicts so a co-located one can load.

```
time, event ('load' | 'evict'), model, engine
  load:  seconds, vram_gb, offloaded_gb, strategy, measured
  evict: reason, freed_gb, expected_gb
```

It is in-memory and bounded (500 entries), so it is empty after a restart. It
exists because the alternative was reconstructing swaps from the log, and the
log rotates: on one host 78% of it was per-request debug, which filled the
2,000-line view in about four minutes.

## A pin is a choice of format, not a budget

The KV placement plan reads `cache_type_k` / `cache_type_v` / `kv_offload` from a
model as instructions it must not override: the format and the location are the
operator's call. For a while it read them as something more than that — as a
statement that whatever size resulted was acceptable — and skipped itself
entirely when any of them was set.

One model set `cache_type_k/v: q8_0` with `n_ctx: 128000`. Quantized KV at a long
context is exactly the configuration someone writes *because* they are worried
about size, and it is the configuration that most needs checking: 128k of q8_0 KV
measured 47.7 GB, llama.cpp placed it in host RAM because no GPU could hold it,
and the OOM killer took the engine — and every other model resident on it —
while the pages were still being faulted in.

The plan now honours a pin and still checks the result. A pinned type is never
requantized and a pinned placement is never moved; what changes is that when the
remaining levers cannot make it fit, the load is refused with a message naming
the pin, instead of being handed to the OOM killer. Refusing one model beats
losing the engine.

## Write the measurement down before the thing you measured kills you

Predicting a KV cache from GGUF metadata does not work — per-layer KV-head
arrays, sliding-window layers and shared KV put the estimate off by 5x in both
directions on the model above — so the plan acts only on a measured size, read
from llama.cpp's own `KV buffer size` log lines.

That measurement used to be taken after a successful load and printed as advice:
*record as measured_kv_gb: {...}*. But a KV cache large enough to need the plan is
large enough to kill the process before the load finishes, and there is no "after
the load" then, and nobody is reading the log either. The engine died, the front
respawned it, the next request repeated the same load, and the number that would
have prevented it was lost every single time.

It is now journalled the moment llama.cpp names the size, which is before the
allocation is faulted in and therefore before the kill. `measured_kv_gb` on the
model still wins; the journal (`~/.coderai/cache/kv_sizes.json`, keyed by model
path and context length) only answers when the config said nothing. A load that
dies now teaches the next one, so the second attempt refuses cleanly instead of
dying again.
