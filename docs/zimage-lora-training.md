# Native Z-Image (DiT) LoRA training

> **Status (v1 — needs one validation train).** Implemented in `codai/api/loras.py`
> `_train_dit()`, routed from `_train_lora_sync` when the base model is a Z-Image DiT.
> Companion: `docs/gguf-process-isolation.md` (unrelated), the SDXL/Wan trainers in the
> same file.

## Why
coderai's built-in LoRA trainer only targets **SDXL/SD1.x U-Nets** (`UNet2DConditionModel`
+ CLIP), producing `unet.`-prefixed keys. **Z-Image is a DiT** (`ZImageTransformer2DModel`,
`transformer.`-prefixed, Qwen3 text encoder), so an SDXL LoRA **silently no-ops** on it
("No LoRA keys associated to ZImageTransformer2DModel found with prefix='transformer'").
The old workaround (`lora_train_base_model` → train on a separate SDXL) can't produce a
LoRA that loads on Z-Image. `_train_dit` trains **natively** against Z-Image so the keys
match and the LoRA applies on the Z-Image pipeline.

No ai-toolkit / no separate venv: `ZImageTransformer2DModel`, `ZImagePipeline`, PEFT and
the Qwen3 text encoder are already in the main venv. It reuses the existing job /
progress / checkpoint / queue system unchanged.

## Recipe (reverse-engineered from diffusers `pipeline_z_image.py`)
Training must mirror inference exactly:
- **Prompt:** `tokenizer.apply_chat_template([{user, prompt}], add_generation_prompt, enable_thinking)`
  → Qwen3 `text_encoder(..., output_hidden_states=True).hidden_states[-2]`, masked to the
  real tokens → a **variable-length per-sample embedding**. The transformer takes a **list**.
- **Latents:** `AutoencoderKL`; scale `(vae.encode(img).sample() - shift_factor) * scaling_factor`
  (inverse of the pipeline's decode `latents/scale + shift`).
- **Transformer I/O is list-based:** input a LIST of `[C,1,H,W]` latents (the pipeline does
  `latents.unsqueeze(2).unbind(0)`), a normalized scalar timestep **`(1000 - t)/1000`**, and a
  prompt-embed LIST; the output is a list, `stack().squeeze(2)`.
- **Velocity sign:** the pipeline computes `noise_pred = -model_out` before the flow-match
  step, so the transformer's **raw output target is `x0 - noise`** (negated velocity). This
  is the #1 thing to get right — if the trained subject comes out inverted, flip to
  `target = noise - x0`.
- **Turbo timesteps:** Z-Image-Turbo is distilled (~8–9 fixed steps, guidance 0). Training
  samples timesteps from the **discrete turbo schedule** — `FlowMatchEulerDiscreteScheduler.set_timesteps(num_inference_steps=9, mu=calculate_shift(image_seq_len,…))` — and picks one of
  those ~9 `(t, sigma)` nodes per step, building `x_t = (1-sigma)·x0 + sigma·noise`. Uniform
  sigma sampling blurs the turbo model; staying on the nodes keeps it sharp.
- **Save:** `ZImagePipeline.save_lora_weights(transformer_lora_layers=…)` →
  `pytorch_lora_weights.safetensors` with `transformer.` keys; `_apply_loras`’
  `load_lora_weights` applies it on Z-Image at gen time.

LoRA target modules: `to_q/to_k/to_v/to_out.0` (same as the SDXL/Wan trainers). Defaults:
rank 16 (4–16 typical), lr 1e-4, resolution 512 (v1; 1024 is Z-Image's sweet spot but uses
more activation VRAM — enable gradient checkpointing if you raise it).

## How to use
1. **Remove the `lora_train_base_model` override** from the Z-Image entry in `models.json`
   (it forced SDXL training). With it gone, `base_model = train_base = Z-Image`, so the DiT
   branch fires.
2. Train via the township UI / `POST /v1/loras/train` exactly as before (character/env
   profile → images). The job routes to `_train_dit`.
3. The resulting LoRA loads on Z-Image automatically (`name:<lora>` references work).

## Validation checklist (do this once)
- Train one fighter: rank 8, ~800 steps, 512. Watch loss decrease.
- Generate with the trigger; confirm identity transfers **and** no "No LoRA keys" warning.
- If the subject is inverted/garbled → flip the velocity sign (`target = noise - x0`).
- If soft/blurry → the turbo-node sampling is the knob (step count / sigma source).

## Not yet
- Flux / SD3 DiTs (the route raises with guidance to use an SDXL override or external
  trainer). The same `_train_dit` shape extends to them with their own forward conventions.
- Gradient checkpointing for 1024-res / large-VRAM-saving training (needs the frozen-base
  `requires_grad` input hook, as in `_train_wan`).
