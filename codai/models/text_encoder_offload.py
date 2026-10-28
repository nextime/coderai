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

"""Encode the prompt, then give the card back before denoising.

A text encoder runs ONCE per request, before the first denoising step, and is
then dead weight for the rest of the generation -- but it stays resident and
keeps its VRAM. On the models where that matters it is not a rounding error:
Z-Image's encoder is 7.5 GiB of a 19.2 GiB pipeline (measured on the cached
snapshot), and LTX-2.3's 4-bit Gemma-3-12B is ~6.5 GiB of a card that then has
to hold a 15 GiB transformer. PinkCherry declaring 24.0 GB on a 24 GiB card is
that sum, and it is why every LTX render was pushed to CPU offload.

So: encode first, free the encoder, then denoise with the embeddings. The
transformer gets the whole card, and the model that "did not fit" fits.

This is what the Motus requirements call "sequential text-encoder offload
(LTX, Krea 2)". It existed only as a side effect before -- diffusers'
``enable_model_cpu_offload`` happens to move each component out after use,
which is why offloaded pipelines worked at all, and why nobody noticed the
resident path had no equivalent.

The embedding argument names differ per family and there is no common base
class to ask, so they are discovered from the pipeline's own signature rather
than hard-coded: Krea 2 takes ``prompt_embeds`` + ``prompt_embeds_mask``, LTX-2
takes ``prompt_embeds`` + ``prompt_attention_mask``, and a family that takes
neither is simply left alone.
"""

import inspect
from typing import Any, Dict, Optional, Tuple

__all__ = ["supported", "encode_then_free"]


def _call_params(pipe) -> set:
    try:
        return set(inspect.signature(pipe.__call__).parameters)
    except (TypeError, ValueError):
        return set()


def _mask_arg(params: set) -> Optional[str]:
    """The name this family uses for the prompt's attention mask, if any."""
    for name in ("prompt_embeds_mask", "prompt_attention_mask"):
        if name in params:
            return name
    return None


def supported(pipe) -> bool:
    """Whether this pipeline can be driven from pre-computed embeddings.

    Requires both an ``encode_prompt`` to call and a ``prompt_embeds`` to hand
    back. A pipeline missing either cannot have its encoder freed early, and
    pretending otherwise would mean generating from an empty prompt.
    """
    if not hasattr(pipe, "encode_prompt") or not hasattr(pipe, "text_encoder"):
        return False
    return "prompt_embeds" in _call_params(pipe)


def _free(pipe, comp_names=("text_encoder", "text_encoder_2")) -> float:
    """Move the named components off the GPU. Returns GiB freed (best effort).

    Moved to CPU rather than deleted: the pipeline is cached and will serve the
    next request, which needs the encoder again. Re-encoding on CPU would be
    slow, and re-loading it from disk slower still -- so it waits in host RAM,
    which is the cheap resource here.
    """
    freed = 0.0
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        before = torch.cuda.memory_allocated()
        for name in comp_names:
            comp = getattr(pipe, name, None)
            if comp is None or not hasattr(comp, "to"):
                continue
            try:
                # A bnb-quantized module refuses .to('cpu') on some versions;
                # that is not a failure worth aborting the request for.
                comp.to("cpu")
            except Exception:
                continue
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        freed = max(0.0, (before - torch.cuda.memory_allocated()) / (1024 ** 3))
    except Exception:
        return 0.0
    return freed


def encode_then_free(pipe, kwargs: Dict[str, Any], *,
                     log=None) -> Tuple[Dict[str, Any], float]:
    """Encode the prompt in ``kwargs``, free the encoder, return new kwargs.

    The returned kwargs carry the embeddings instead of ``prompt`` /
    ``negative_prompt``, so the caller then invokes the pipeline exactly as it
    would have. On any failure the original kwargs come back untouched and the
    generation proceeds the ordinary way -- a VRAM optimisation must never be
    the reason a request fails.
    """
    def _say(msg):
        if log:
            try:
                log(msg)
            except Exception:
                pass
        print(msg, flush=True)

    if not supported(pipe):
        return kwargs, 0.0
    params = _call_params(pipe)
    mask_arg = _mask_arg(params)
    enc_params = set()
    try:
        enc_params = set(inspect.signature(pipe.encode_prompt).parameters)
    except (TypeError, ValueError):
        return kwargs, 0.0

    prompt = kwargs.get("prompt")
    if prompt is None:
        return kwargs, 0.0
    negative = kwargs.get("negative_prompt")

    enc_kwargs: Dict[str, Any] = {"prompt": prompt}
    if "device" in enc_params:
        enc_kwargs["device"] = getattr(pipe, "_execution_device", None)
    if "max_sequence_length" in enc_params and "max_sequence_length" in kwargs:
        enc_kwargs["max_sequence_length"] = kwargs["max_sequence_length"]
    # LTX-2 encodes the negative prompt in the same call; Krea 2 needs a second
    # one. Asked of the signature so neither family is special-cased.
    both_in_one = "negative_prompt" in enc_params
    if both_in_one:
        enc_kwargs["negative_prompt"] = negative
        if "do_classifier_free_guidance" in enc_params:
            gs = kwargs.get("guidance_scale")
            enc_kwargs["do_classifier_free_guidance"] = bool(
                gs is not None and float(gs) > 1.0)

    try:
        import torch
        with torch.no_grad():
            out = pipe.encode_prompt(**enc_kwargs)
    except Exception as exc:
        _say(f"  [text-encoder-offload] encoding failed ({exc}); "
             f"generating the ordinary way")
        return kwargs, 0.0

    if not isinstance(out, (tuple, list)) or not out:
        return kwargs, 0.0

    new = dict(kwargs)
    new.pop("prompt", None)
    new.pop("negative_prompt", None)
    new["prompt_embeds"] = out[0]
    try:
        if both_in_one:
            # (prompt_embeds, negative_prompt_embeds, prompt_mask, negative_mask)
            if len(out) > 1 and out[1] is not None:
                new["negative_prompt_embeds"] = out[1]
            if mask_arg and len(out) > 2 and out[2] is not None:
                new[mask_arg] = out[2]
            neg_mask = "negative_" + mask_arg if mask_arg else None
            if neg_mask and neg_mask in params and len(out) > 3 and out[3] is not None:
                new[neg_mask] = out[3]
        else:
            # (prompt_embeds, prompt_mask) -- the negative needs its own call.
            if mask_arg and len(out) > 1 and out[1] is not None:
                new[mask_arg] = out[1]
            if negative is not None and "negative_prompt_embeds" in params:
                neg_kwargs = dict(enc_kwargs)
                neg_kwargs["prompt"] = negative
                with torch.no_grad():
                    nout = pipe.encode_prompt(**neg_kwargs)
                if isinstance(nout, (tuple, list)) and nout:
                    new["negative_prompt_embeds"] = nout[0]
                    neg_mask = "negative_" + mask_arg if mask_arg else None
                    if neg_mask and neg_mask in params and len(nout) > 1:
                        new[neg_mask] = nout[1]
    except Exception as exc:
        _say(f"  [text-encoder-offload] could not assemble the embeddings "
             f"({exc}); generating the ordinary way")
        return kwargs, 0.0

    freed = _free(pipe)
    if freed > 0.05:
        _say(f"  [text-encoder-offload] prompt encoded, encoder moved off the "
             f"GPU — {freed:.1f} GiB back for the denoiser")
    else:
        _say("  [text-encoder-offload] prompt encoded; the encoder was not on "
             "the GPU, so nothing was freed")
    return new, freed
