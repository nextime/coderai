# When a model does not fit, and what has to be true before it does

Written from one afternoon spent establishing why `Qwen3.8-Flash-Next` would not
serve. The conclusion was that nothing about it was a memory problem on the
machine it was tried on, and that the next machine needs three separate things
to be true, not one. This is the checklist, so the diagnosis is not repeated.

## The three independent requirements

A model needs all three. Fixing the one you notice first is the usual mistake.

**1. The engine must implement its architecture.** This is not about hardware
and no amount of memory changes it. Qwen3.8-Flash-Next is
`Qwen4ExpForConditionalGeneration` / `model_type: qwen4_exp`, and on the build
it was tried against:

```
llama.cpp implemented:  qwen35, qwen3moe, qwen3next, qwen3vlmoe
vLLM registered:        Qwen3NextForCausalLM, Qwen3NextMTP,
                        Qwen3VLForConditionalGeneration,
                        Qwen3VLMoeForConditionalGeneration
```

Neither carried it. llama.cpp says `unknown model architecture: '<name>'` and
the loader surfaces that; the architecture is also printed before the load, so
the name and the file are already side by side in the log. **To check a machine
before trying:** `strings …/libllama.so* | grep -ix <arch>` for the GGUF path,
and grep vLLM's `model_executor/models/registry.py` for the HF class name.

**2. The quantization must be one the engine can read.** Separate from the
architecture, and invisible in the metadata. A 27B "Flash" quant passed every
check an operator would make — right architecture, all 65 blocks present
including the MTP one, an mmproj beside it — and stored 402 of its 866 tensors
in **ggml type 143**, a rotation-folded format from a fork. The type ids live in
the tensor index; nothing announces them.

`codai/backends/gguf_preflight.py` now answers this from the header alone,
before eviction, and names the type and how many tensors use it. The bound is
llama.cpp's own `GGML_TYPE_COUNT`, so it tracks the installed library rather
than a list kept in the repo.

**3. The memory has to hold it.** Only here does the machine matter. For
Qwen3.8-Flash-Next (125B-A5B, 360 GB in bf16 across 131 shards):

| quant | size | | quant | size |
|---|---|---|---|---|
| UD-IQ1_S | 72.5 GB | | UD-IQ4_XS | 93.7 GB |
| UD-IQ1_M | 74.5 GB | | UD-Q4_K_XL | 111.3 GB |
| UD-Q2_K_XL | 78.9 GB | | Q8_0 | 188.2 GB |
| UD-IQ3_XXS | 82.0 GB | | BF16 | 354.0 GB |

An expert-pruned variant (`REAP128-86B`) exists at 48.9 GB, IQ4_XS only, and is
still `qwen4exp`. Budget VRAM + host RAM against the admission floor, not
against total RAM — see `memory-accounting.md`.

## The MTP file is not a model

`mtp-*.gguf` files hold the NextN / multi-token-prediction draft head **only**.
One was 2.7 GB and contained a single block, `blk.48` of a declared 49: no
embedding, no layers, nothing to run. It is an accessory to a base checkpoint,
not something to serve, and `gguf_mtp.block_count_override` deliberately refuses
to reinterpret one into a one-layer model.

Three different things get called "MTP support":

- **A quant that dropped its MTP block** but still declares it. llama.cpp asks
  for a block the file has not got and reports `missing tensor
  'blk.N.attn_norm.weight'`, which reads like a corrupt download. Handled —
  `gguf_mtp` corrects the declared count when the file's shape proves that is
  what happened.
- **Speculative decoding through MTP.** Available on **vLLM** today: the
  `speculative` setting passes raw JSON to `--speculative-config`, precisely so
  a draft method can be configured without a code change. Also on **ds4**
  (`--mtp`). **Not** on the llama.cpp path: `llama-cpp-python` exposes
  `draft_model`, but its only implementation is `LlamaPromptLookupDecoding`, a
  prompt-lookup n-gram drafter — there is no way to hand it a second GGUF, even
  though the C library has the plumbing.
- **An MTP block inside the main GGUF**, which the architecture's own code uses.
  This build wants it: it asked for `blk.64` rather than stopping at 63.

## Moving one of these to a bigger machine

1. Confirm requirement 1 on that machine, for the engine you intend to use.
   Nothing else matters until this passes.
2. Pick a quant and check requirement 3 against VRAM + host RAM, remembering the
   host-RAM admission floor.
3. Download the **base**, not the `MTP/` folder.
4. Set the card's `backend` to match the engine that will serve it. A GGUF path
   with `backend: vllm` fails at the front with a pin conflict before anything
   opens the file — which looks like an engine fault and is a config one.
5. For vLLM, MTP speculative decoding goes in `speculative` as JSON.

The preflight in requirement 2 runs by itself and will say if the quant is
unreadable, before it costs an eviction.
