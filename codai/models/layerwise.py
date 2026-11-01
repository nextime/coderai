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
    """The per-model config, with a forwarded ``_raw_cfg`` unwrapped.

    Delegated to :func:`codai.models.hf_loading._norm` rather than reimplemented:
    a model's config reaches the image loader as TRANSLATED kwargs with the
    original entry nested under ``_raw_cfg``, and every other reader of it
    unwraps that. This module did not, so ``layerwise_casting`` was invisible
    here and no casting hooks were installed -- while the storage dtype, read
    through the component loader's already-unwrapped copy, applied. The weights
    loaded as fp8 and nothing upcast them for the matmul, which on Ampere is a
    dtype the hardware cannot multiply at all.
    """
    try:
        from codai.models.hf_loading import _norm as _hf_norm
        return _hf_norm(cfg)
    except Exception:
        if not isinstance(cfg, dict):
            return {}
        raw = cfg.get("_raw_cfg")
        merged = dict(cfg)
        if isinstance(raw, dict):
            for k, v in raw.items():
                merged.setdefault(k, v)
        return merged


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


#: Attribute a component carries listing the modules the FILE stored narrow.
_NARROW_ATTR = "_coderai_narrow_modules"


def remember_narrow(component, storage_dtype, path=None,
                    component_cls=None) -> int:
    """Record which modules the FILE stored at ``storage_dtype``, and widen the rest.

    Sourced from the file's header, NOT from the loaded module's dtypes. Those
    two are not the same thing, and the difference cost three attempts:

    * ``torch_dtype=<storage>`` casts the WHOLE component, so the F16 time
      embeddings and F32 norms become fp8 too. diffusers' default skip patterns
      then leave exactly those out of the casting hooks, and the first matmul
      says "mat1 and mat2 must have the same dtype, but got BFloat16 and
      Float8_e4m3fn" naming ``time_embed.linear_1``.
    * ``torch_dtype=None`` does not preserve anything either. The meta loader
      ends at accelerate's ``set_module_tensor_to_device``, which does
      ``if dtype is None: value = value.to(old_value.dtype)`` -- so every
      incoming fp8 tensor is cast to whatever the freshly-built model already
      holds. Nothing was narrow afterwards, this function recorded zero, and
      the footprint stayed 31.07 GB against the 17.22 GB the same model
      measures when loaded as fp8.

    So load as ``storage_dtype`` (narrow, which is the point) and then put back
    the tensors the file stored WIDER, re-read from the file at their own
    precision. That is exact: nothing goes through fp8 that was not fp8 on
    disk. Those tensors are the small ones -- everything that is not a big
    linear is a rounding error against them.

    Returns the number of modules recorded as narrow. With no ``path`` it falls
    back to asking the module what it holds, which is right for a component
    that was not read from a single file.
    """
    if path is None or component_cls is None:
        paths = [name for name, module in component.named_modules()
                 if any(p.dtype == storage_dtype
                        for p in module.parameters(recurse=False))]
        try:
            setattr(component, _NARROW_ATTR, (str(storage_dtype), tuple(paths)))
        except Exception:
            return 0
        return len(paths)

    from codai.models import scaled_fp8 as _sfp8
    import torch

    file_dtypes = _sfp8.dtypes_in_file(path)
    renamed = _sfp8.rename_keys(component_cls, file_dtypes.keys())

    narrow_modules, wide_keys = set(), []
    for file_key, dtype_name in file_dtypes.items():
        model_key = renamed.get(file_key)
        if not model_key or "." not in model_key:
            continue
        module_path = model_key.rsplit(".", 1)[0]
        # Compared as torch dtypes, not as strings: safetensors says "F8_E4M3"
        # where torch says "float8_e4m3fn", and no amount of case folding makes
        # those equal.
        if _sfp8.torch_dtype_of(dtype_name) is storage_dtype:
            narrow_modules.add(module_path)
        else:
            wide_keys.append((file_key, model_key))

    # Put the wider tensors back at their own precision, from the file.
    restored = 0
    if wide_keys:
        try:
            from safetensors import safe_open
            with safe_open(path, framework="pt") as fh:
                for file_key, model_key in wide_keys:
                    module_path, param_name = model_key.rsplit(".", 1)
                    try:
                        module = component.get_submodule(module_path)
                    except Exception:
                        continue
                    target = getattr(module, param_name, None)
                    if target is None:
                        continue
                    value = fh.get_tensor(file_key)
                    if not value.is_floating_point():
                        continue
                    if tuple(value.shape) != tuple(target.shape):
                        continue
                    target.data = value.to(target.data.device)
                    restored += 1
        except Exception as exc:
            print(f"  [layerwise] could not restore the file's wider tensors "
                  f"({exc}); they keep the load dtype")

    try:
        setattr(component, _NARROW_ATTR,
                (str(storage_dtype), tuple(sorted(narrow_modules))))
    except Exception:
        return 0
    if restored:
        print(f"  [layerwise] {len(narrow_modules)} module(s) are "
              f"{str(storage_dtype).replace('torch.', '')} in the file; "
              f"{restored} wider tensor(s) restored from it at their own "
              f"precision")
    return len(narrow_modules)


def _hook_one(component, storage_dtype, compute_dtype):
    """Hook the modules that ARE stored narrow; bring the rest to compute dtype.

    Per module, because a checkpoint is not uniform. Real Dream's fp8 build is
    256 fp8 linears plus F16 time-embedding projections and F32 norms, and
    casting the WHOLE component to fp8 -- which is what passing torch_dtype did
    -- downcast tensors the file never stored that way. diffusers' own
    ``enable_layerwise_casting`` then skipped exactly those precision-critical
    modules by default, so they kept an fp8 weight that nothing upcast and the
    first matmul said

        mat1 and mat2 must have the same dtype, but got BFloat16 and
        Float8_e4m3fn

    naming ``time_embed.linear_1`` -- a module that was F16 on disk. Measured
    2026-11-01.

    So: ask each module what dtype it actually holds. One holding the storage
    dtype gets the casting hook (narrow at rest, compute dtype inside its own
    forward). One holding anything else is brought to the compute dtype
    outright, which is cheap -- everything that is not a big linear is a
    rounding error against them -- and leaves nothing for a matmul to trip on.

    ``skip_modules_pattern`` is not used at all: the per-module hook has no
    pattern matching, so nothing can be excluded behind our back. That mattered
    because diffusers appends the model's own
    ``_skip_layerwise_casting_patterns`` even to an empty pattern tuple.
    """
    from diffusers.hooks.layerwise_casting import apply_layerwise_casting_hook

    # The modules the FILE stored narrow, if that was recorded before the
    # pipeline cast them. Without it we can only ask what each module holds
    # NOW, which is the compute dtype for all of them once the pipeline has
    # been through.
    remembered = None
    try:
        stash = getattr(component, _NARROW_ATTR, None)
        if stash and stash[0] == str(storage_dtype):
            remembered = set(stash[1])
    except Exception:
        remembered = None

    hooked = recast = restored = 0
    for _name, module in component.named_modules():
        params = list(module.parameters(recurse=False))
        if not params:
            continue
        if remembered is not None and _name in remembered:
            # Put it back to the width the file had it at, then hook it.
            for p in params:
                if p.dtype != storage_dtype:
                    p.data = p.data.to(storage_dtype)
                    restored += 1
            apply_layerwise_casting_hook(module, storage_dtype, compute_dtype,
                                         False)
            hooked += 1
            continue
        if remembered is None and any(p.dtype == storage_dtype for p in params):
            apply_layerwise_casting_hook(module, storage_dtype, compute_dtype,
                                         False)
            hooked += 1
            continue
        for p in params:
            if p.dtype.is_floating_point and p.dtype != compute_dtype:
                p.data = p.data.to(compute_dtype)
                recast += 1
    return hooked, recast + restored


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
        if not hasattr(comp, "named_modules"):
            skipped.append(f"{name} (not a module)")
            continue
        try:
            hooked, recast = _hook_one(comp, storage_dtype, compute_dtype)
        except Exception as exc:
            skipped.append(f"{name} ({exc})")
            continue
        if not hooked:
            skipped.append(f"{name} (nothing stored at "
                           f"{str(storage_dtype).replace('torch.', '')})")
            continue
        done.append(f"{name}: {hooked} module(s) kept at "
                    f"{str(storage_dtype).replace('torch.', '')}"
                    + (f", {recast} brought to "
                       f"{str(compute_dtype).replace('torch.', '')}"
                       if recast else ""))
    note = ""
    if done:
        note = "; ".join(done) + (f"; computing in "
                                  f"{str(compute_dtype).replace('torch.', '')}")
    if skipped:
        note += ("; not applied to " if note else "not applied to ") + \
            ", ".join(skipped)
    return note
