# A model's own settings, and where its weights come from

Two things an operator configures per model that clients then stop guessing at:
the settings the model was published with, and the file the weights live in.

## `defaults` — the card's own recommendation

```jsonc
{
  "alias": "Krea-2-Turbo",
  "path": "krea/Krea-2-Turbo",
  "defaults": {
    "steps": 8,
    "guidance_scale": 1.0,
    "sampler": "flow_euler",
    "modes": ["t2i"],
    "license": "Krea 2 licence — requires a content filter"
  }
}
```

These **fill in what a request omits and never override what it sends.** That
is the whole distinction from `call_kwargs`, which applies on every call
because the card says the model does not work without it.

Why it exists: a distilled model wants 8 steps at CFG 1 and its non-distilled
sibling wants 30 at CFG 4. Without this every client keeps its own table of
per-model settings, keyed on the model **name** — which breaks the moment a
model is renamed or a second quantization of it is added — and a client with no
table runs an 8-step model for 30 steps: four times the cost for a worse
picture.

Precedence, highest first:

1. what the request asked for;
2. a **fused** distillation preset detected on the pipeline (`acceleration`),
   because it knows the LoRA actually baked in;
3. `defaults` from the model entry;
4. **what the model's own name says**, when nothing above answered (images);
5. the generic floor (30 steps / CFG 7.5 for images, 25 / 7.5 for video).

### Where the card is read from

`defaults` lives in the models.json **entry**. The manager keeps runtime kwargs
flat and that entry under `_raw_cfg`, so only keys promoted by
`build_runtime_kwargs` exist at the top level — and `defaults` is not one of
them. Reading the flat config therefore finds nothing, which is exactly what
the image path did until 2026-10-29: `/v1/models` published a card's steps and
guidance while generation used the generic floor, so a client that read the
listing and trusted the server was wrong twice. `Z-Image-Turbo-IntoRealism`
(10 steps) and `Krea-2-RealDream` (8) had never had their cards applied, and
`call_kwargs` — MOSS-Image's `uncond_skip_layers` — sits in the same place.

`images.card_view()` merges the entry under the flat kwargs (the flat ones win
on a collision: they are what the loader was given), and both `model_defaults`
and `_model_call_kwargs` read through it. A test walks the installed catalogue
and asserts, for every card carrying `defaults`, that what `/v1/models`
publishes is what generation would use.

### 4: the name, when the card says nothing

A card with no `defaults` and no `acceleration` block used to fall straight to
the floor. Measured on a production fleet, 2026-10-29: `Z-Image-Turbo` — a
checkpoint distilled for 8 steps at CFG 1 — was served at 30 steps and CFG
7.5, about four times the work for a picture that also comes out over-guided,
because a distilled model given real classifier-free guidance looks burnt.
Roughly 72 s per 768×1344 still, for a model whose whole point is to take
seconds.

So when nothing better is available, the id itself is read:

| in the id | steps | CFG |
|---|---|---|
| `turbo`, `lightning`, `lightx2v`, `hyper-sd`, `tcd`, `distill` | 8 | 1.0 |
| `lcm` | 6 | 1.5 |
| `schnell`, `dmd2` | 4 | 1.0 |
| `sdxs` | 2 | 1.0 |

Only the last path segment is read, so an org named `turbo-labs` proves
nothing, and the word must stand on its own: `turbovision-xl` and `lcmx` do
not match. The choice is logged with the number it replaced, because an
unexplained step count is indistinguishable from a bug when a picture comes
out wrong.

This is a floor, not a recommendation: writing `defaults` on the card is still
the right thing to do, and it wins.

`/v1/models` publishes `defaults`, `modes` and `license` so a UI can show them
instead of inventing them. `modes` is read from `defaults.modes`, a top-level
`modes`, or a video entry's long-standing `video_subtypes` — whichever the
operator used.

### `sampler`

A request may carry `sampler`, and a model entry may default it:

```
euler  euler_a  heun  dpm / dpmpp_2m  dpmpp_sde  dpmpp_2s  ddim  ddpm
pndm  lms  deis  unipc  lcm  kdpm2  kdpm2_a
flow_euler  flow_heun  flow_lcm
```

The `flow_*` family is for flow-matching models (FLUX, SD3, Z-Image, LTX) and
**cannot be mixed with the rest.** Handing a FLUX-family pipeline a DDIM does
not give a worse picture, it gives noise, so crossing that boundary is refused:
a 400 when the request asked, a logged warning when a model card asked (the
request itself is fine and should not be failed over the operator's config).

An unknown name is a 400 listing the valid ones.

## Where the weights come from

Three shapes, all accepted wherever a model `path` is given:

| Reference | Means |
|---|---|
| `org/repo` | a HuggingFace repo in the normal diffusers/transformers layout |
| `/AI/models/someMix_v18.safetensors` | one checkpoint file on this machine |
| `org/repo:someMix_v18.safetensors` | one file out of a repo holding several versions |
| `civitai:1609320` | one checkpoint from Civitai, by **model version id** |

### `civitai:` references

A large part of the community's fine-tunes are published only on Civitai, as a
single ComfyUI-style checkpoint — exactly the shape `from_single_file` already
handles. The only thing missing was a way to fetch one, so an operator had to
download by hand and hard-code a path.

It is the **version** id, not the model id. A Civitai "model" is a family with
many versions, and an entry naming the family would silently change underneath
you when a new version was published.

Downloaded once into `<cache>/civitai/<id>/` and never fetched again. Most
models need a token:

```bash
export CIVITAI_TOKEN=…            # or write it to <config>/civitai_token
```

Without one the API answers with a **web page**, HTTP 200. A naive fetch saves
that to a `.safetensors` name and the problem surfaces hours later as an
unreadable tensor header, so this checks the content type and the size and
refuses, naming the missing token.

## A GGUF component that cannot be loaded is an error

`component_quantization` can point a pipeline component at a GGUF file:

```jsonc
"component_quantization": {
  "transformer": "/AI/guffcache/wan22Remix_i2vHighV30-Q4_K_M.gguf",
  "text_encoder": "4bit",
  "vae": "4bit"
}
```

If a component named here **fails** to load, the load now fails too. It used to
be skipped with a log line — and a skipped component is simply absent, so the
base pipeline loaded its own full-precision copy. The operator asked for a 14 GB
quantized transformer and got a 38 GB bf16 one, while the VRAM estimate still
said 14 GB and the planner therefore freed about a third of what the load
actually needed. A silent downgrade to something that does not fit is worse
than a refusal.

## Pre-quantized checkpoints need no `load_in_4bit`

A repo published already quantized — `…-bnb-4bit`, an AWQ or GPTQ build, an fp8
release — carries a `quantization_config` in its `config.json`, and
transformers/diffusers load those weights as they are. Leave `load_in_4bit`
unset: there is nothing to ask for.

The VRAM estimator detects this and stops normalising the on-disk size up to
the load precision. It did not, once, and a 16.3 GB 4-bit checkpoint was
estimated at **30.03 GB against a 24 GB card** — so nothing could ever free
enough, the pipeline ran partly on the CPU, and a few-step turbo model took 72
seconds an image.

Related: quantized weights cannot be CPU-offloaded at all. A quantized kernel
cannot read a CPU tensor, so such a model loads and then fails **every**
request with `Pointer argument (at 0) cannot be accessed from Triton`. That is
refused at load time, and a quantized model attempts full-GPU regardless of
what the estimate says.
