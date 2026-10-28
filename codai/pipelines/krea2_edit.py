# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Reference-image editing for Krea 2, which diffusers does not ship.

``Krea2Pipeline.__call__`` takes a prompt and no image, and there is no
``Krea2Img2ImgPipeline``: the family's only entry points in diffusers 0.42 are
that pipeline and the Krea2Turbo modular blocks. So a card naming Krea 2 with
an identity-edit LoRA had the weights ready and no way to be asked "change THIS
picture", and an edit request fell through to text-to-image, which quietly
ignored the image it was given.

The edit is done IN CONTEXT rather than by denoising from the reference the way
img2img does. The reference is VAE-encoded once, kept clean, and appended to
the transformer's ``hidden_states`` beside the canvas tokens; the model attends
across both and the reference is never noised, so identity survives the whole
trajectory instead of being worn away as the canvas is denoised. That is what
the family's edit LoRAs are trained against, and it is also why ``strength``
has no meaning here: there is no partial noising to control. The capability a
card should report is therefore ``instruction_edit``, not ``image_to_image``.

Telling the two token groups apart is the whole trick. ``position_ids`` is
``(seq, 3)``: text at the origin and image tokens carrying ``(0, h, w)``. The
reference gets the SAME ``(h, w)`` grid but ``1`` on the first axis, so the
rotary embedding separates "this pixel of the canvas" from "this pixel of the
reference" while keeping them spatially aligned. Without that the two groups
occupy identical coordinates and the model cannot tell which it is painting.
"""

from typing import Any, Callable, Dict, List, Optional, Union

import torch

__all__ = ["Krea2EditPipeline", "build_edit_view", "is_krea2_pipeline",
           "edit_position_ids"]


def _krea2_classes():
    """The diffusers Krea 2 classes, or ``None`` when this build lacks them."""
    try:
        import diffusers
        base = getattr(diffusers, "Krea2Pipeline", None)
        out = getattr(diffusers, "Krea2PipelineOutput", None)
        if out is None:                     # not exported in every build
            from diffusers.pipelines.krea2.pipeline_krea2 import (  # type: ignore
                Krea2PipelineOutput as out)
        return base, out
    except Exception:
        return None, None


def is_krea2_pipeline(pipe: Any) -> bool:
    """Whether ``pipe`` is a Krea 2 pipeline (or already an edit view of one)."""
    names = {c.__name__ for c in type(pipe).__mro__}
    return bool(names & {"Krea2Pipeline", "Krea2EditPipeline"})


_Base, _Output = _krea2_classes()

if _Base is None:                           # pragma: no cover - build without Krea 2
    class Krea2EditPipeline:                # type: ignore[no-redef]
        """Placeholder so importing this module never breaks a build."""

        def __init__(self, *a, **kw):
            raise RuntimeError(
                "this diffusers build has no Krea2Pipeline, so Krea 2 editing "
                "is unavailable")
else:

    class Krea2EditPipeline(_Base):         # type: ignore[misc,valid-type]
        """Krea 2, asked to change a picture instead of invent one.

        Built from a loaded ``Krea2Pipeline`` with
        :func:`build_edit_view`, so an install serving both text-to-image and
        editing holds one copy of the weights.
        """

        @torch.no_grad()
        def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            image: Any = None,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 8,
            sigmas: Optional[List[float]] = None,
            guidance_scale: float = 1.0,
            num_images_per_prompt: int = 1,
            generator=None,
            latents: Optional[torch.Tensor] = None,
            output_type: str = "pil",
            return_dict: bool = True,
            attention_kwargs: Optional[Dict[str, Any]] = None,
            max_sequence_length: int = 512,
            grounding_px: Optional[int] = None,
            callback_on_step_end: Optional[Callable] = None,
            callback_on_step_end_tensor_inputs: Optional[List[str]] = None,
            **kwargs,
        ):
            """``image`` is the picture to change; ``prompt`` says how.

            ``grounding_px`` is the longest side the reference is encoded at.
            It defaults to the output size, because a reference encoded on the
            canvas's own grid lines its tokens up with the canvas's: the same
            ``(h, w)`` coordinate means the same place in both. Raising it
            feeds the model more detail of the original at the cost of a longer
            sequence, which is quadratic in attention.
            """
            if image is None:
                raise ValueError(
                    "Krea 2 editing needs an `image` to change; use the plain "
                    "Krea2Pipeline for text-to-image")

            import numpy as np

            device = self._execution_device
            dtype = self.transformer.dtype

            # ---------------------------------------------------- geometry
            # Snapped to the grid the packer needs, exactly as the base
            # pipeline does: latents are patched p-square after an
            # vae_scale_factor-fold compression, so both sides must divide by
            # the product or the reshape in _pack_latents is invalid.
            multiple = self.vae_scale_factor * self.patch_size
            if height is None or width is None:
                src_w, src_h = _image_size(image)
                height = height or src_h
                width = width or src_w
            height = max(multiple, int(height) // multiple * multiple)
            width = max(multiple, int(width) // multiple * multiple)

            self._guidance_scale = guidance_scale
            self._attention_kwargs = attention_kwargs
            self._current_timestep = None
            self._interrupt = False

            batch_size = 1 if isinstance(prompt, str) or prompt is None else len(prompt)

            # ------------------------------------------------------- text
            prompt_embeds, prompt_embeds_mask = self.encode_prompt(
                prompt=prompt or "",
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )
            negative_prompt_embeds = negative_prompt_embeds_mask = None
            if self.do_classifier_free_guidance:
                negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                    prompt=negative_prompt or "",
                    device=device,
                    num_images_per_prompt=num_images_per_prompt,
                    max_sequence_length=max_sequence_length,
                )

            # ----------------------------------------------------- canvas
            num_channels_latents = self.transformer.config.in_channels // (
                self.patch_size * self.patch_size)
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt, num_channels_latents,
                height, width, dtype, device, generator, latents)
            canvas_len = latents.shape[1]
            grid_h = height // multiple
            grid_w = width // multiple

            # -------------------------------------------------- reference
            ref_px = int(grounding_px) if grounding_px else max(height, width)
            ref_h, ref_w = _fit_to_grid(image, ref_px, multiple)
            ref_tokens, ref_grid = self._encode_reference(
                image, ref_h, ref_w, dtype, device, generator)
            ref_tokens = ref_tokens.repeat(
                batch_size * num_images_per_prompt // ref_tokens.shape[0], 1, 1) \
                if ref_tokens.shape[0] != batch_size * num_images_per_prompt \
                else ref_tokens

            # ------------------------------------------------ coordinates
            position_ids = self._edit_position_ids(
                prompt_embeds.shape[1], grid_h, grid_w, ref_grid[0], ref_grid[1],
                device)

            # ------------------------------------------------- timesteps
            sigmas = (np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
                      if sigmas is None else sigmas)
            mu = 1.15
            if not getattr(self.transformer.config, "is_distilled", False):
                try:
                    from diffusers.pipelines.krea2.pipeline_krea2 import (  # type: ignore
                        calculate_shift)
                    mu = calculate_shift(
                        canvas_len,
                        self.scheduler.config.get("base_image_seq_len", 256),
                        self.scheduler.config.get("max_image_seq_len", 4096),
                        self.scheduler.config.get("base_shift", 0.5),
                        self.scheduler.config.get("max_shift", 1.15))
                except Exception:
                    mu = 1.15
            from diffusers.pipelines.krea2.pipeline_krea2 import (  # type: ignore
                retrieve_timesteps)
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu)

            # ---------------------------------------------------- denoise
            with self.progress_bar(total=num_inference_steps) as bar:
                for i, t in enumerate(timesteps):
                    if self.interrupt:
                        continue
                    self._current_timestep = t
                    timestep = (t / self.scheduler.config.num_train_timesteps) \
                        .expand(latents.shape[0]).to(latents.dtype)

                    # The reference is re-attached CLEAN at every step. It is
                    # context, not something being denoised: noising it would
                    # make the identity it carries decay exactly as the canvas
                    # resolves, which is the failure in-context editing exists
                    # to avoid.
                    hidden = torch.cat([latents, ref_tokens.to(latents.dtype)], dim=1)

                    noise_pred = self.transformer(
                        hidden_states=hidden,
                        encoder_hidden_states=prompt_embeds,
                        timestep=timestep,
                        position_ids=position_ids,
                        encoder_attention_mask=prompt_embeds_mask,
                        attention_kwargs=self._attention_kwargs,
                        return_dict=False,
                    )[0][:, :canvas_len]

                    if self.do_classifier_free_guidance:
                        neg = self.transformer(
                            hidden_states=hidden,
                            encoder_hidden_states=negative_prompt_embeds,
                            timestep=timestep,
                            position_ids=position_ids,
                            encoder_attention_mask=negative_prompt_embeds_mask,
                            attention_kwargs=self._attention_kwargs,
                            return_dict=False,
                        )[0][:, :canvas_len]
                        noise_pred = neg + guidance_scale * (noise_pred - neg)

                    latents = self.scheduler.step(
                        noise_pred, t, latents, return_dict=False)[0]

                    if callback_on_step_end is not None:
                        cb_kwargs = {k: locals()[k]
                                     for k in (callback_on_step_end_tensor_inputs or [])
                                     if k in locals()}
                        out = callback_on_step_end(self, i, t, cb_kwargs) or {}
                        latents = out.pop("latents", latents)
                    bar.update()

            self._current_timestep = None

            # ----------------------------------------------------- decode
            if output_type == "latent":
                out_image = latents
            else:
                lat = self._unpack_latents(latents, height, width).to(self.vae.dtype)
                mean, std_inv = self._vae_stats(lat.device, lat.dtype)
                lat = lat / std_inv + mean
                out_image = self.vae.decode(lat, return_dict=False)[0][:, :, 0]
                out_image = self.image_processor.postprocess(
                    out_image, output_type=output_type)

            self.maybe_free_model_hooks()
            if not return_dict:
                return (out_image,)
            return _Output(images=out_image)

        # ------------------------------------------------------- helpers

        def _vae_stats(self, device, dtype):
            """``(latents_mean, 1/latents_std)``, shaped for 5-D latents.

            The same pair the base pipeline uses to undo normalisation when it
            decodes; encoding a reference has to apply it in the other
            direction or the reference lands on a different scale from the
            canvas and the model reads it as noise.
            """
            z = self.vae.config.z_dim
            mean = torch.tensor(self.vae.config.latents_mean).view(1, z, 1, 1, 1) \
                .to(device, dtype)
            std_inv = (1.0 / torch.tensor(self.vae.config.latents_std)
                       .view(1, z, 1, 1, 1)).to(device, dtype)
            return mean, std_inv

        def _encode_reference(self, image, height, width, dtype, device, generator):
            """The reference as packed tokens, plus its ``(grid_h, grid_w)``."""
            px = self.image_processor.preprocess(image, height=height, width=width)
            px = px.to(device=device, dtype=self.vae.dtype)
            if px.ndim == 4:                       # (B,C,H,W) -> add the frame axis
                px = px.unsqueeze(2)
            posterior = self.vae.encode(px)
            lat = getattr(posterior, "latent_dist", None)
            lat = lat.mode() if lat is not None else posterior[0]
            # `mode()`, not `sample()`: a reference is a fixed fact about the
            # request, so sampling it would make the same edit of the same
            # picture differ run to run for no reason the caller asked for.
            mean, std_inv = self._vae_stats(lat.device, lat.dtype)
            lat = (lat - mean) * std_inv
            lat = lat[:, :, 0]                     # drop the frame axis
            b, c, lh, lw = lat.shape
            p = self.patch_size
            lh = lh // p * p
            lw = lw // p * p
            lat = lat[:, :, :lh, :lw].to(dtype)
            tokens = self._pack_latents(lat, b, c, lh, lw)
            return tokens, (lh // p, lw // p)

        _edit_position_ids = staticmethod(lambda *a, **kw: edit_position_ids(*a, **kw))



def edit_position_ids(text_len: int, grid_h: int, grid_w: int,
                      ref_h: int, ref_w: int, device=None):
    """Coordinates for text, then the canvas, then the reference.

    ``position_ids`` is ``(seq, 3)``. The canvas keeps the base pipeline's
    ``(0, h, w)``. The reference repeats its own grid but sits at ``1`` on the
    first axis, which is what lets the model tell the two apart: identical
    coordinates would make a reference token and the canvas token aligned with
    it indistinguishable to the rotary embedding, and the model could not tell
    which of the two it is painting.

    Module-level and pure, so it can be checked without the 12B weights and on
    a build whose diffusers ships no Krea 2 at all.
    """
    text_ids = torch.zeros(text_len, 3, device=device)

    def _grid(h, w, plane):
        ids = torch.zeros(h, w, 3, device=device)
        ids[..., 0] = plane
        ids[..., 1] = torch.arange(h, device=device)[:, None]
        ids[..., 2] = torch.arange(w, device=device)[None, :]
        return ids.reshape(h * w, 3)

    return torch.cat([text_ids, _grid(grid_h, grid_w, 0),
                      _grid(ref_h, ref_w, 1)], dim=0)


def _image_size(image):
    """``(width, height)`` of whatever the caller passed."""
    if hasattr(image, "size") and isinstance(getattr(image, "size"), tuple):
        return image.size
    if hasattr(image, "shape"):
        shape = image.shape
        if len(shape) >= 2:
            return int(shape[-1]), int(shape[-2])
    if isinstance(image, (list, tuple)) and image:
        return _image_size(image[0])
    return 1024, 1024


def _fit_to_grid(image, longest: int, multiple: int):
    """The reference's size, scaled to ``longest`` and snapped to the grid."""
    w, h = _image_size(image)
    scale = float(longest) / float(max(w, h) or longest)
    out_h = max(multiple, int(round(h * scale)) // multiple * multiple)
    out_w = max(multiple, int(round(w * scale)) // multiple * multiple)
    return out_h, out_w


def build_edit_view(base):
    """``base`` as an edit pipeline, sharing its weights, or ``None``.

    ``from_pipe`` rather than a second load: the components are already in
    VRAM and a Krea 2 transformer is 12B, so loading again to serve editing
    would cost a second copy of it and an eviction on every alternation
    between the two tasks.
    """
    if _Base is None or not is_krea2_pipeline(base):
        return None
    if isinstance(base, Krea2EditPipeline):
        return base
    try:
        return Krea2EditPipeline.from_pipe(base)
    except Exception:
        # Last resort: the same components, named explicitly. from_pipe can
        # refuse over an unexpected component set (an offload hook, a LoRA
        # adapter); the parts themselves are all we need.
        try:
            return Krea2EditPipeline(
                transformer=base.transformer, vae=base.vae,
                text_encoder=base.text_encoder, tokenizer=base.tokenizer,
                scheduler=base.scheduler)
        except Exception:
            return None
