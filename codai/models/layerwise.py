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

"""Keep the weights small and compute at full width: layerwise casting.

A community fine-tune is very often published in fp8 -- one safetensors file of
about 13 GB for a 12B transformer. Loaded with ``torch_dtype=bfloat16`` it
becomes 26 GB, because that is what upcasting means, and the file that fit the
card on disk no longer fits it in VRAM. Measured 2026-11-01:

* Krea-2-RealDream (``civitai:3396368:3304483``, 12.9 GB on disk) measured
  31.07 GB and died mid-generation on cuBLAS's workspace;
* Z-Image-Turbo-IntoRealism (fp8, 5.9 GB on disk) is upcast the same way.

Neither needs to be. diffusers' ``enable_layerwise_casting`` stores the weights
at the narrow dtype and upcasts each module to the compute dtype on its own
forward, so the resident cost is the file's cost and the arithmetic still
happens in bf16. That matters especially on Ampere, which has no fp8 matmul at
all: fp8 there is a STORAGE format and nothing else.

Two things make this safe rather than merely small:

* diffusers sets ``_skip_layerwise_casting_patterns`` on its models, which
  leaves patch embeddings, positional embeddings and normalisation layers at
  full width. Those are where narrow weights actually hurt, and the defaults
  are used rather than overridden.
* The load dtype and the compute dtype are decided separately. A component with
  a storage dtype configured is LOADED at that dtype, so the peak is the narrow
  footprint too -- casting after a bf16 load would save the resident cost and
  still need 26 GB to get there, which for RealDream is the whole problem.

Quality is not free: the diffusers docs call the degradation "negligible,
mostly stemming from weight casting in normalization and modulation layers",
and this is opt-in per model for that reason.
"""

from typing import Any, Dict, Optional, Tuple

#: Accepted spellings for a storage dtype, mapped to the torch attribute name.
#: ``true`` means "the usual one", which is what every fp8 checkpoint in the
#: wild is published as.
_DTYPES = {
    "f8": "float8_e4m3fn",
    "fp8": "float8_e4m3fn",
    "f8_e4m3fn": "float8_e4m3fn",
    "float8_e4m3fn": "float8_e4m3fn",
    "e4m3": "float8_e4m3fn",
    "f8_e5m2": "float8_e5m2",
    "float8_e5m2": "float8_e5m2",
    "e5m2": "float8_e5m2",
}

#: The components worth casting when the config just says "yes". The heavy
#: ones: everything else is rounding error against them, and the VAE is left
#: alone on purpose -- it is the last thing to touch the latents and a narrow
#: decode is visible in a way a narrow denoise step is not.
_DEFAULT_COMPONENTS = ("transformer", "unet", "text_encoder", "text_encoder_2")


def _norm(cfg) -> Dict[str, Any]:
    return cfg if isinstance(cfg, dict) else {}


def _resolve_dtype(value):
    """A configured spelling -> a torch dtype, or None when unavailable."""
    if value is None or value is False:
        return None
    name = "float8_e4m3fn" if value is True else _DTYPES.get(
        str(value).strip().lower())
    if not name:
        return None
    try:
        import torch
    except Exception:
        return None
    return getattr(torch, name, None)


def spec(cfg) -> Tuple[Dict[str, Any], str]:
    """``(component -> storage dtype, description)`` for this model's config.

    ``layerwise_casting`` is either a dtype for the usual heavy components::

        "layerwise_casting": "f8"

    or a mapping when one component should differ from the rest::

        "layerwise_casting": {"transformer": "f8"}

    Empty when the model asks for nothing, so an ordinary load is untouched.
    """
    c = _norm(cfg)
    raw = c.get("layerwise_casting")
    if raw is None or raw is False or raw == {} or raw == "":
        return {}, ""
    out: Dict[str, Any] = {}
    if isinstance(raw, dict):
        for name, value in raw.items():
            dt = _resolve_dtype(value)
            if dt is not None:
                out[str(name)] = dt
    else:
        dt = _resolve_dtype(raw)
        if dt is not None:
            out = {name: dt for name in _DEFAULT_COMPONENTS}
    if not out:
        return {}, ""
    desc = ", ".join(f"{k}:{str(v).replace('torch.', '')}"
                     for k, v in sorted(out.items()))
    return out, desc


def storage_dtype_for(cfg, component: str):
    """The storage dtype configured for ONE component, or None.

    Read by the component loader so the weights are read at their own width
    instead of being upcast on the way in. A component the config does not name
    is not affected, which is why this answers per component rather than for
    the pipeline.
    """
    return (spec(cfg)[0] or {}).get(str(component))


def apply(pipeline, cfg, compute_dtype) -> str:
    """Enable layerwise casting on every configured component. Returns a note.

    Best effort per component and never fatal: a pipeline that has loaded and
    can generate must not be failed by an optimisation. A component that cannot
    do it says so and keeps the weights it has.
    """
    wanted, _desc = spec(cfg)
    if not wanted:
        return ""
    done, skipped = [], []
    for name, storage_dtype in sorted(wanted.items()):
        comp = getattr(pipeline, name, None)
        if comp is None:
            continue            # not a component of this pipeline at all
        fn = getattr(comp, "enable_layerwise_casting", None)
        if not callable(fn):
            skipped.append(f"{name} (this diffusers build cannot)")
            continue
        try:
            fn(storage_dtype=storage_dtype, compute_dtype=compute_dtype)
            done.append(f"{name}:{str(storage_dtype).replace('torch.', '')}")
        except Exception as exc:
            skipped.append(f"{name} ({exc})")
    note = ""
    if done:
        note = (f"storing {', '.join(done)} and computing in "
                f"{str(compute_dtype).replace('torch.', '')}")
    if skipped:
        note += ("; not applied to " if note else "not applied to ") + \
            ", ".join(skipped)
    return note
