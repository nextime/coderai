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

Measured on LTX-2.3-PinkCherry (480x480, 25 frames, 8 steps, RTX 3090):

    without the offload   369.4 s   peak 23718 MiB   91% util
    with the offload      345.3 s   peak 24098 MiB   92% util
                          encoder moved off the GPU: 7.3 GiB freed

Read it as headroom, not as a speed-up: the card is 24 GiB and the denoiser
fills whatever it is given either way, so peak VRAM sits at the ceiling in
both runs and the 24 s is one sample, not a benchmark. What the 7.3 GiB buys
is that the transformer no longer has to share the card with a dead encoder --
which is what had been pushing these renders onto CPU offload, where the same
job took tens of minutes. The encoder's cost moves to host RAM (availability
fell to 11.3 GB on this box during the run), which is the intended trade.

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


def _resident_on_gpu(comp) -> bool:
    """Whether any of this component's parameters currently live on a GPU."""
    try:
        for p in comp.parameters():
            return bool(getattr(getattr(p, "device", None), "type", "") == "cuda")
    except Exception:
        pass
    try:
        return bool(getattr(getattr(comp, "device", None), "type", "") == "cuda")
    except Exception:
        return False


def _free(pipe, comp_names=("text_encoder", "text_encoder_2")) -> Tuple[float, list]:
    """Move the named components off the GPU.

    Returns ``(GiB freed, notes)``. The notes matter: these encoders are the
    quantized ones (LTX-2.3's is a 4-bit Gemma-3-12B), and a bnb-quantized
    module refuses ``.to('cpu')`` on some versions. Without distinguishing
    "refused to move" from "was not resident" both arrive as 0.0 GiB freed and
    the log blames the wrong thing -- so each component says which it was.

    Moved to CPU rather than deleted: the pipeline is cached and will serve the
    next request, which needs the encoder again. Re-encoding on CPU would be
    slow, and re-loading it from disk slower still -- so it waits in host RAM,
    which is the cheap resource here.
    """
    notes = []
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0, ["no CUDA device"]
        before = torch.cuda.memory_allocated()
        for name in comp_names:
            comp = getattr(pipe, name, None)
            if comp is None:
                continue
            if not hasattr(comp, "to"):
                notes.append(f"{name}: not movable")
                continue
            if not _resident_on_gpu(comp):
                notes.append(f"{name}: already off the GPU")
                continue
            try:
                comp.to("cpu")
                notes.append(f"{name}: moved to CPU")
            except Exception as exc:
                notes.append(f"{name}: REFUSED the move ({type(exc).__name__}: "
                             f"{str(exc)[:120]}) — still holding its VRAM")
                continue
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        freed = max(0.0, (before - torch.cuda.memory_allocated()) / (1024 ** 3))
    except Exception as exc:
        return 0.0, [f"freeing failed: {type(exc).__name__}: {exc}"]
    return freed, notes


def _assign_outputs(out, params: set, mask_arg: Optional[str]) -> Dict[str, Any]:
    """Name the tensors ``encode_prompt`` returned.

    The order is NOT the same across families and is not discoverable from the
    signature, so it is read off the values themselves. Both families return
    the positive pair first and the negative pair after it, mask immediately
    following its embeddings:

        Krea 2  -> (prompt_embeds, prompt_embeds_mask)
        LTX-2   -> (prompt_embeds, prompt_attention_mask,
                    negative_prompt_embeds, negative_prompt_attention_mask)

    A mask is an integer/bool tensor and embeddings are floating point, so the
    two are told apart rather than assumed by position -- a family that returns
    embeddings with no mask, or only a positive pair, lands correctly too.

    Guessing the position is what broke the first live run: the code assumed
    (embeds, neg_embeds, mask, neg_mask), read the mask from slot 2 and handed
    the pipeline a negative-embeddings tensor as the attention mask.
    """
    def _is_mask(t) -> bool:
        dt = getattr(t, "dtype", None)
        if dt is None:
            return False
        is_float = getattr(dt, "is_floating_point", None)
        if is_float is None:
            try:
                import torch
                is_float = dt.is_floating_point
            except Exception:
                return False
        return not bool(is_float)

    tensors = [t for t in out if t is not None]
    assigned: Dict[str, Any] = {}
    # Slot names in the order the families emit them. "pooled" covers the
    # SDXL-shaped families, which return four float tensors and no mask.
    slots = ["prompt_embeds", mask_arg, "negative_prompt_embeds",
             ("negative_" + mask_arg) if mask_arg else None,
             "pooled_prompt_embeds", "negative_pooled_prompt_embeds"]
    i = 0
    for name in slots:
        if i >= len(tensors):
            break
        if name is None:
            continue
        t = tensors[i]
        wants_mask = bool(mask_arg and name.endswith(mask_arg))
        if wants_mask != _is_mask(t):
            # This family did not emit this slot; leave it and try the next
            # name against the same tensor.
            continue
        if name == "prompt_embeds" or name in params:
            assigned[name] = t
        i += 1

    # Everything the encoder returned has to be named, or the optimisation is
    # abandoned. A family whose extra tensors we silently dropped would not
    # degrade quietly -- it would FAIL the request, because the pipeline
    # validates that the embeddings it was handed are complete. SDXL is the
    # worked example: it returns (embeds, neg_embeds, pooled, neg_pooled), and
    # dropping the pooled pair raises "Must provide pooled_prompt_embeds".
    #
    # So an unrecognised shape costs nothing beyond a logged line and the
    # ordinary generation path, and this module stays safe to enable on a
    # family nobody has tried yet.
    if i < len(tensors):
        raise ValueError(
            f"encode_prompt returned {len(tensors)} tensors and only {i} could "
            f"be named; refusing to hand the pipeline partial embeddings")
    return assigned


def _ensure_on_device(pipe, comp_names=("text_encoder", "text_encoder_2")) -> list:
    """Bring the named components back to the pipeline's execution device.

    REQUIRED, not an optimisation. The pipeline is cached and served again, so
    a previous request's offload has left the encoder on the CPU. Encoding then
    indexes cuda input ids against cpu weights:

        RuntimeError: Expected all tensors to be on the same device, but got
        index is on cuda:0, different from other tensors on cpu

    and -- the part that made this a 500 rather than a slow path -- the damage
    OUTLIVES this function. Falling back to "generate the ordinary way" hands
    the request to a pipeline whose own encode_prompt hits the same mismatch,
    so the request fails anyway. Measured in production 2026-10-28: the first
    request succeeded and every one after it returned 500 until the model was
    evicted.

    So the offload is a per-request CYCLE -- bring the encoder on, encode, move
    it off -- not a one-way move.
    """
    notes = []
    try:
        import torch
    except Exception:
        return notes
    dev = getattr(pipe, "_execution_device", None)
    if dev is None:
        try:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except Exception:
            return notes
    for name in comp_names:
        comp = getattr(pipe, name, None)
        if comp is None or not hasattr(comp, "to"):
            continue
        if _resident_on_gpu(comp):
            continue
        try:
            comp.to(dev)
            notes.append(f"{name}: brought back to {dev}")
        except Exception as exc:
            notes.append(f"{name}: could NOT be brought back to {dev} "
                         f"({type(exc).__name__}: {str(exc)[:120]})")
    return notes


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

    # Every bail-out says why. The first version returned silently, and when
    # it did nothing on a real LTX render there was no way to tell "flag off"
    # from "pipeline unsupported" from "no prompt in kwargs" -- the feature
    # looked inert with no evidence either way.
    if not supported(pipe):
        _say(f"  [text-encoder-offload] {type(pipe).__name__} cannot be driven "
             f"from embeddings (encode_prompt={hasattr(pipe, 'encode_prompt')}, "
             f"text_encoder={hasattr(pipe, 'text_encoder')}, "
             f"prompt_embeds={'prompt_embeds' in _call_params(pipe)}) — skipped")
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
        _say("  [text-encoder-offload] no `prompt` in the call kwargs "
             "(embeddings already supplied?) — skipped")
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

    # Put the encoder back before using it: a previous request on this cached
    # pipeline may have left it on the CPU.
    back = _ensure_on_device(pipe)
    if back:
        _say(f"  [text-encoder-offload] {'; '.join(back)}")

    try:
        import torch
        with torch.no_grad():
            out = pipe.encode_prompt(**enc_kwargs)
    except Exception as exc:
        # Leave the pipeline able to generate on its own: the ordinary path
        # calls this same encoder, so handing it back on the CPU would turn a
        # skipped optimisation into a failed request.
        _ensure_on_device(pipe)
        _say(f"  [text-encoder-offload] encoding failed ({exc}); "
             f"generating the ordinary way")
        return kwargs, 0.0

    if not isinstance(out, (tuple, list)) or not out:
        return kwargs, 0.0

    new = dict(kwargs)
    new.pop("prompt", None)
    new.pop("negative_prompt", None)
    try:
        assigned = _assign_outputs(out, params, mask_arg)
    except Exception as exc:
        _ensure_on_device(pipe)
        _say(f"  [text-encoder-offload] could not assemble the embeddings "
             f"({exc}); generating the ordinary way")
        return kwargs, 0.0
    if "prompt_embeds" not in assigned:
        _ensure_on_device(pipe)
        _say("  [text-encoder-offload] encode_prompt returned no embeddings; "
             "generating the ordinary way")
        return kwargs, 0.0
    if mask_arg and mask_arg not in assigned:
        # The pipeline rejects prompt_embeds without its mask, so handing back
        # embeddings we could not pair with one would fail the request. This
        # is exactly what happened on the first live LTX-2.3 run: the mask was
        # read from the wrong tuple slot and the call raised "Must provide
        # `prompt_attention_mask` when specifying `prompt_embeds`".
        _ensure_on_device(pipe)
        _say(f"  [text-encoder-offload] encode_prompt returned no {mask_arg}; "
             f"generating the ordinary way")
        return kwargs, 0.0
    new.update(assigned)

    # A family that does not take `negative_prompt` in encode_prompt encodes the
    # negative in a SECOND call -- and its __call__ will make that call itself if
    # we hand it no negative embeddings, against an encoder this function is
    # about to move to the CPU. Krea 2 is the worked example: with guidance on
    # and no negative supplied it substitutes "" and encodes it at denoise time,
    # which would run on the wrong device.
    #
    # So the negative is always encoded here for those families, "" included.
    # Dropping it instead would be worse than the device error: the request
    # would succeed with no negative guidance at all and nothing would say so.
    if not both_in_one and "negative_prompt_embeds" in params:
        try:
            neg_kwargs = dict(enc_kwargs)
            neg_kwargs["prompt"] = negative if negative is not None else ""
            with torch.no_grad():
                nout = pipe.encode_prompt(**neg_kwargs)
            if isinstance(nout, (tuple, list)) and nout:
                neg_assigned = _assign_outputs(nout, params, mask_arg)
                remap = {"prompt_embeds": "negative_prompt_embeds"}
                if mask_arg:
                    remap[mask_arg] = "negative_" + mask_arg
                if "pooled_prompt_embeds" in neg_assigned:
                    remap["pooled_prompt_embeds"] = \
                        "negative_pooled_prompt_embeds"
                for src, dst in remap.items():
                    if src in neg_assigned and dst in params:
                        new[dst] = neg_assigned[src]
                if "negative_prompt_embeds" not in new:
                    raise ValueError("no negative embeddings came back")
        except Exception as exc:
            _ensure_on_device(pipe)
            _say(f"  [text-encoder-offload] could not encode the negative "
                 f"prompt ({exc}); generating the ordinary way")
            return kwargs, 0.0

    freed, notes = _free(pipe)
    detail = "; ".join(notes) if notes else "no text encoder to move"
    if freed > 0.05:
        _say(f"  [text-encoder-offload] prompt encoded, encoder moved off the "
             f"GPU — {freed:.1f} GiB back for the denoiser ({detail})")
    else:
        _say(f"  [text-encoder-offload] prompt encoded but no VRAM came back "
             f"({detail})")
    return new, freed
