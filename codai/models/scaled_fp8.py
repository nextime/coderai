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

"""Checkpoints whose weights carry a separate per-tensor scale.

A large part of the community publishes its fine-tunes quantized with the
scale kept beside the weight rather than folded into it::

    blocks.0.attn.wq.weight         F8_E4M3  [6144, 6144]
    blocks.0.attn.wq.weight_scale   BF16     []            a scalar

diffusers has no idea what ``weight_scale`` is -- the string does not appear in
single_file_utils.py or single_file_model.py at all. Its converter renames every
key and passes it through, so the scale arrives as
``transformer_blocks.0.attn.to_q.weight_scale``, which the model does not have,
and it is dropped as an unexpected key. The weight is then used UNSCALED.

That is not a rounding error. Measured on Real Dream's Krea 2 V3 fp8 build,
2026-11-01: 256 scales, min 0.000839, median 0.00249, max 0.0125, and not one
of them within 1% of 1.0. Dropping them leaves every weight about four hundred
times too large, which is noise or NaN rather than a degraded image.

THE SCALE DOES NOT HAVE TO TOUCH THE WEIGHTS. For a linear layer it is a
scalar, so

    y = (W * s) @ x  ==  s * (W @ x)

and applying it to the OUTPUT costs nothing: the weights stay fp8, the
footprint stays the size of the file, and the arithmetic is exact. Dequantizing
instead -- multiplying it into the weights at load -- would be correct too, and
would cost 25 GB of bf16 for a 12.9 GB file, or the same memory as this with
worse quality if those weights were then re-cast to fp8 without a scale.

Two things keep this honest:

* The renaming is done by DIFFUSERS' OWN converter, looked up in
  SINGLE_FILE_LOADABLE_CLASSES by the component's class name, so this cannot
  drift away from how the weights themselves are renamed.
* Every scale must find a module. One that does not is raised, not logged: a
  checkpoint whose scales were silently half-applied is worse than one that
  refused to load, because it answers 200 with a plausible-looking image.
"""

import json
import struct
from typing import Any, Dict, List, Tuple

#: The suffix a scale carries in these checkpoints.
SCALE_SUFFIX = ".weight_scale"


def _read_scalar(raw: bytes, dtype: str):
    """One scalar out of its raw bytes, for the dtypes scales come in."""
    if dtype == "F32":
        return struct.unpack("<f", raw[:4])[0]
    if dtype == "F64":
        return struct.unpack("<d", raw[:8])[0]
    if dtype == "BF16":
        # bf16 is the top half of a float32.
        return struct.unpack("<f", struct.pack(
            "<I", struct.unpack("<H", raw[:2])[0] << 16))[0]
    if dtype == "F16":
        return struct.unpack("<e", raw[:2])[0]
    raise ValueError(f"scale in an unexpected dtype: {dtype}")


def scales_in_file(path: str) -> Dict[str, float]:
    """``{weight key -> scale}`` read from a safetensors file's header.

    Only the header and a handful of scalars are read: the weights are
    gigabytes and none of them are needed to answer this.
    """
    out: Dict[str, float] = {}
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
        base = 8 + n
        header.pop("__metadata__", None)
        for key, meta in header.items():
            if not key.endswith(SCALE_SUFFIX) or not isinstance(meta, dict):
                continue
            if meta.get("shape"):
                # Per-channel scales are a different arrangement and the
                # output trick does not apply unchanged; say so rather than
                # applying a vector as if it were a number.
                raise ValueError(
                    f"{key} is not a scalar (shape {meta['shape']}): this "
                    f"checkpoint scales per channel, which needs the scale "
                    f"applied along a dimension rather than to the output")
            start, end = meta["data_offsets"]
            fh.seek(base + start)
            out[key] = _read_scalar(fh.read(end - start), meta["dtype"])
    return out


def module_scales(component_cls, raw: Dict[str, float]) -> Dict[str, float]:
    """``{module path -> scale}`` in the loaded model's own naming.

    Renamed by the converter diffusers itself uses for this class, so the
    scales follow exactly the renaming their weights follow. A class diffusers
    cannot load from a single file has no converter and no scales to map.
    """
    if not raw:
        return {}
    try:
        from diffusers.loaders.single_file_model import (
            SINGLE_FILE_LOADABLE_CLASSES)
    except Exception as exc:
        raise RuntimeError(
            f"cannot reach diffusers' single-file converter registry to "
            f"rename this checkpoint's scales: {exc}") from exc
    name = getattr(component_cls, "__name__", str(component_cls))
    entry = SINGLE_FILE_LOADABLE_CLASSES.get(name) or {}
    convert = entry.get("checkpoint_mapping_fn")
    if convert is None:
        raise RuntimeError(
            f"{name} has no single-file converter in diffusers, so the scales "
            f"in this checkpoint cannot be renamed to match its weights")
    renamed = convert({k: v for k, v in raw.items()})
    out: Dict[str, float] = {}
    for key, value in (renamed or {}).items():
        if key.endswith(SCALE_SUFFIX):
            out[key[:-len(SCALE_SUFFIX)]] = value
    return out


def install(model, scales: Dict[str, float]) -> Tuple[int, List[str]]:
    """Multiply each scaled module's output by its scale. ``(applied, missing)``.

    A forward hook, so it runs after the module's own forward -- and therefore
    after layerwise casting has upcast the weights for the matmul, which is the
    order that makes fp8 storage and exact arithmetic coexist.
    """
    if not scales:
        return 0, []
    by_name = dict(model.named_modules())
    applied, missing = 0, []
    for path, scale in sorted(scales.items()):
        module = by_name.get(path)
        if module is None:
            missing.append(path)
            continue
        module.register_forward_hook(_scaler(float(scale)))
        applied += 1
    return applied, missing


def _scaler(scale: float):
    """A forward hook that scales whatever the module returned."""
    def _hook(_module, _args, output):
        if isinstance(output, tuple):
            return (output[0] * scale,) + output[1:]
        return output * scale
    return _hook


def apply_to_component(component, component_cls, path: str) -> str:
    """Read, rename and install a single file's scales. Returns a note, or "".

    Raises when a scale cannot be placed. A checkpoint whose scales were half
    applied answers 200 with a plausible image and is the worst of the three
    possible outcomes.
    """
    raw = scales_in_file(path)
    if not raw:
        return ""
    scales = module_scales(component_cls, raw)
    applied, missing = install(component, scales)
    if missing:
        raise RuntimeError(
            f"{len(missing)} of {len(scales)} weight scales in {path} did not "
            f"match a module of {getattr(component_cls, '__name__', '?')} "
            f"(first: {missing[0]!r}). Applying some and not others would "
            f"serve a model whose weights are partly four hundred times too "
            f"large, so this refuses instead.")
    lo = min(scales.values())
    hi = max(scales.values())
    return (f"{applied} per-tensor weight scales applied to the module outputs "
            f"(range {lo:.3g}–{hi:.3g}); weights stay as stored")
