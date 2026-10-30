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

# A projector that loads, injects, and says nothing

`Qwen3.8-27B` served images for months and described every one of them
identically. Worth recording in full, because four plausible explanations were
wrong and only an experiment settled it.

The symptom, at temperature 0 with two images that could not differ more:

```
red square   -> "…I see a red circle on a white background…"
blue circle  -> "…I see a red circle on a white background…"   byte-identical
no image     -> "But no image provided in prompt?"
```

That last line matters: the model **notices** when no image is present. So with
an image it receives something in the image slot, and the content of that
something never varies.

## What it was not

Each of these was checked and cleared:

| hypothesis | evidence against |
|---|---|
| mmproj/model mismatch | `clip.vision.projection_dim` 5120 = the model's `hidden_size` 5120; `mm.2.weight [4608, 5120]` |
| missing deepstack tensors | unsloth's reference mmproj is the same shape — 334 tensors, `is_deepstack_layers` all false, zero deepstack tensors — so this family does not use them |
| projector not implemented | `clip_graph_qwen3vl` is present in `libmtmd` |
| base model is text-only | `Qwen3_5ForConditionalGeneration`, has `vision_config`, `image_token_id: 248056` |
| vision markers absent | the GGUF tokenizer carries `<\|vision_start\|>` 248053, `<\|vision_end\|>` 248054, `<\|image_pad\|>` 248056, matching the base config exactly |
| the mmproj never loaded | `mmproj … (vision enabled)` plus 341 `clip_model_loader` lines |
| coderai's fast path dropping images | the guard in vulkan.py explicitly refuses the render path for vision requests, and it fires |
| the 1024-token floor | llama.cpp's own stated requirement, and the symptom predates applying it |

Two inferences made along the way were also wrong, and both came from reading
code and headers rather than running anything. "No mtmd output at request time,
so the handler is bypassed" ignored that the llama.cpp log callback is swapped to
a quiet one after the load, by design — runtime silence proves nothing.

## What settled it

Two experiments, not another reading of the headers.

**gemma-4-14b-8k, its own projector, the same two images:** `'Red square.'` and
`'Blue circle.'`, and `'Square, blue.'` with no image at all. So the pipeline —
fast-path guard, `create_chat_completion`, the mtmd chat handler, mmproj
loading, image injection — is sound end to end.

**The same Qwen model with a DIFFERENT publisher's projector** (unsloth's F16
rather than HauhauCS's BF16): identical constant answer. Two independent
conversions, one failure.

So the fault is upstream of coderai. Either llama.cpp's `qwen3vl_merger` graph is
wrong for this model, or both converters mislabel it — the base is `qwen3_5` /
`Qwen3_5ForConditionalGeneration` while the projector declares
`qwen3vl_merger`, so llama.cpp may simply be running the wrong graph.

## The fault is narrow, and there are working alternatives

`mmproj` is removed from the `qwen38` cards, which stops them advertising
`image_to_text` they cannot deliver — a model that answers confidently and
wrongly is worse than one that declines.

The capability is not lost. Measured on the same two images, same prompt:

| model | path | vision |
|---|---|---|
| gemma-4-31B-heretic | GGUF + `gemma4v` mmproj | `'Red square.'` / `'Blue circle.'` |
| gemma-4-14b-8k | GGUF + `gemma4v` mmproj | `'Red square.'` / `'Blue circle.'` |
| Qwen3.5-9B | transformers, built-in vision tower, no mmproj | `'red square'` / `'Blue Circle'` |
| qwen38, qwen38-128k | GGUF + `qwen3vl_merger` mmproj | the same sentence for both |

**gemma-4-31B-heretic is the one to reach for**: it is larger than the broken
27B, already on disk with its own projector, and exact on both images.

That table also narrows the fault to the `qwen3vl_merger` graph alone. `gemma4v`
is correct at two different model sizes, and Qwen3.5's OWN vision tower is
correct when transformers runs it instead of llama.cpp — so it is not the model,
not the family, and not our pipeline.

## The actual cause: our llama.cpp predates the fix

```
our build (0.3.30)  vendors llama.cpp e3a74b299   2026-06-16
the fix             b4aa7dd47  "mtmd : use align_corners for qwen3vl vision
                    position embedding interpolation" (#25781)   2026-07-21
0.3.36              vendors llama.cpp 0c1e57098   2026-10-01   <- contains it
```

Five weeks. Wrong position-embedding interpolation makes the vision features
spatially meaningless, which is why the output is constant rather than merely
wrong, and why `gemma4v` -- a different graph, untouched by that bug -- reads the
same images correctly through the same pipeline.

Everything observed fits: the projector pairs correctly, loads, and injects
tokens the model notices (it says so when there is no image); two publishers'
conversions fail identically because the defect is in the consumer, not the
artifact; and the image-token floor is irrelevant, confirmed by testing with it
at 1024 and at 0 and getting byte-identical answers.

`packaging/linux/Dockerfile.update-llama` is bumped to 0.3.36 for this reason.
It takes effect only on a rebuild, so until one runs, `mmproj` stays off the
qwen38 cards and vision goes to gemma-4-31B-heretic.

One method note, since it cost most of the investigation: three conclusions were
drawn from reading headers and code, and all three were wrong. The floor was
dismissed with "the symptom predates it" -- never checked, and it could not have
been, because the floor is applied automatically whenever the projector type
matches. Two projectors were compared while the one variable introduced by this
repo was held constant in both. What settled it was varying one thing at a time
against a known-good control.

## If you try a newer llama.cpp, do NOT pip install it

Six versions of llama-cpp-python have shipped since the one we run (0.3.30 vs
0.3.36), and a vendored-llama.cpp fix for this graph is plausible. But the wheel
here is a LOCAL SOURCE BUILD, and that is the whole point of it:

```
CMAKE_ARGS="-DGGML_CUDA=ON -DGGML_VULKAN=ON -DGGML_RPC=ON \
            -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS} -DGGML_NATIVE=OFF -DGGML_AVX2=ON"
```
(`packaging/linux/Dockerfile.update-llama`)

A PyPI wheel carries at most one backend. Installing one would take out the
nvidia engine, the radeon/Vulkan engine and the cluster's GGUF-over-RPC path
together — three capabilities, to chase one projector. The upgrade is a bump to
`LLAMA_CPP_PYTHON_VERSION` in that Dockerfile plus
`update_oci_image.sh REBUILD_LLAMA=1`, it changes the engine under every GGUF
model, and it wants the docker disk checked first (it has run to 90-98% before).

Given gemma-4-31B already answers correctly, that rebuild is worth doing when it
is wanted for its own sake, not as a fix for this.
