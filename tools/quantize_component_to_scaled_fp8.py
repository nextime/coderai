#!/usr/bin/env python3
# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Cut a diffusers component folder down to one scaled-fp8 single file.

Why this exists. A pipeline whose transformer ships as 35 GB of bf16 shards
cannot be served from a 24 GB card at all, and quantizing it at load (the
`fp8` mode in component_quantization) re-reads and re-quantizes all 35 GB on
every cold start -- the per-component cache only covers bitsandbytes, so there
is nothing to reuse. Writing the quantized weights ONCE turns that into a file
roughly half the size that loads directly.

The output is the shape coderai already reads natively -- weights at
``F8_E4M3`` with a scalar ``<module>.weight_scale`` beside each one -- so it
needs no new loader. ``component_models`` points at it, ``from_single_file``
reads it, :mod:`codai.models.scaled_fp8` installs the scales as forward hooks
on the module OUTPUTS, and :mod:`codai.models.layerwise` records which modules
are narrow so the pipeline does not quietly upcast them back to bf16.

Three things make the arithmetic exact rather than approximate.

* **The scale lands on the output, so the BIAS must be divided by it too.**
  ``scaled_fp8.install`` hooks the whole module output, giving
  ``s * (W' @ x + b')``. For that to equal ``W @ x + b`` the file has to store
  ``W' = W / s`` *and* ``b' = b / s``. Storing only the weight scaled leaves
  every bias ``s`` times too small, and since ``s`` is around 0.002 on real
  checkpoints that is not a subtle error. The bias stays bf16; only the weight
  goes to fp8.
* **Only ``nn.Linear`` is touched, decided by asking the class, not by the
  key name.** The component is instantiated on the meta device (no weights, no
  VRAM) purely to look up what each key's module actually is. Output scaling is
  valid for a linear map; it is not valid for anything that applies a
  nonlinearity to the result, and a 2-D ``.weight`` is not by itself proof of a
  Linear.
* **Everything else is copied byte-for-byte.** Norm weights, the
  ``scale_shift_table`` parameters, the patchify/proj_out projections and every
  1-D tensor stay exactly as they were, which is also what the community GGUF
  builds of these models do.

Streaming on purpose. safetensors has no incremental writer in Python, and
``save_file`` wants the whole dict in memory -- 17 GB for this transformer, on a
box that is also running the server. The header can be computed from shapes and
dtypes alone, so it is written first and the tensor data streamed after it, one
tensor at a time. Peak memory is the largest single tensor. The output order is
grouped per module (``weight``, ``weight_scale``, ``bias``) so a module's scale
is known when its bias is written, and nothing has to be read twice.

Usage::

    python3 tools/quantize_component_to_scaled_fp8.py \\
        --src  /AI/hfcache/.../LTX-2.3-Diffusers/snapshots/<rev>/transformer \\
        --out  /AI/guffcache/LTX-2.3-transformer-fp8.safetensors

Then point an entry at it::

    "component_models": {"transformer": "/AI/.../LTX-2.3-transformer-fp8.safetensors"}
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import struct
import sys

#: Largest finite value of float8_e4m3fn. The scale maps a tensor's peak
#: magnitude onto this, so the whole fp8 range is used instead of the tiny slice
#: a raw cast would land in -- weights of ~0.002 cast straight to e4m3 fall into
#: its subnormals and keep one or two mantissa bits.
E4M3_MAX = 448.0

#: safetensors dtype names for what is written.
ST_NAME = {"torch.bfloat16": "BF16", "torch.float16": "F16",
           "torch.float32": "F32", "torch.float64": "F64",
           "torch.float8_e4m3fn": "F8_E4M3", "torch.float8_e5m2": "F8_E5M2",
           "torch.int64": "I64", "torch.int32": "I32", "torch.int8": "I8",
           "torch.uint8": "U8", "torch.bool": "BOOL"}

SCALE_SUFFIX = ".weight_scale"


def _dtype_name(t) -> str:
    name = ST_NAME.get(str(t.dtype))
    if name is None:
        raise SystemExit(f"no safetensors dtype name for {t.dtype}")
    return name


def _nbytes(t) -> int:
    return t.numel() * t.element_size()


def _raw(t) -> bytes:
    """A tensor's bytes, for dtypes numpy cannot represent (bf16, fp8)."""
    import torch
    return t.contiguous().view(torch.uint8).numpy().tobytes()


def _linear_weights(src: str):
    """``{key -> True}`` for every ``.weight`` whose module is an nn.Linear.

    Built by instantiating the component's own class on the meta device from
    its config.json -- no weights are read and no memory is allocated, and the
    answer comes from the class rather than from a guess about the key name.
    """
    import torch
    import torch.nn as nn
    import diffusers

    cfg_path = os.path.join(src, "config.json")
    if not os.path.isfile(cfg_path):
        raise SystemExit(f"{src} has no config.json, so its class is unknown")
    cfg = json.load(open(cfg_path))
    cls_name = cfg.get("_class_name")
    cls = getattr(diffusers, cls_name, None)
    if cls is None:
        raise SystemExit(f"diffusers has no class {cls_name!r}")
    c = dict(cfg)
    c.pop("_class_name", None)
    c.pop("_diffusers_version", None)
    with torch.device("meta"):
        model = cls.from_config(c)
    out = {}
    for path, module in model.named_modules():
        if isinstance(module, nn.Linear):
            out[f"{path}.weight"] = True
    return out, cls_name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True,
                    help="diffusers component folder (config.json + shards)")
    ap.add_argument("--out", required=True, help="single .safetensors to write")
    ap.add_argument("--include", default="",
                    help="only quantize modules whose path starts with one of "
                         "these comma-separated prefixes (default: all Linears)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be written and stop")
    args = ap.parse_args()

    import torch
    from safetensors import safe_open

    shards = sorted(glob.glob(os.path.join(args.src, "*.safetensors")))
    if not shards:
        raise SystemExit(f"no .safetensors in {args.src}")
    linears, cls_name = _linear_weights(args.src)
    prefixes = tuple(p for p in (s.strip() for s in args.include.split(",")) if p)

    # ── index every source tensor: which shard, what shape, what dtype ──────
    where, meta = {}, {}
    for s in shards:
        with safe_open(s, framework="pt") as f:
            for k in f.keys():
                sl = f.get_slice(k)
                where[k] = s
                meta[k] = (tuple(sl.get_shape()), sl.get_dtype())
    print(f"source: {len(meta)} tensors in {len(shards)} shard(s), class {cls_name}")

    def _quantizable(key: str) -> bool:
        if key not in linears:
            return False
        if prefixes and not key.startswith(prefixes):
            return False
        return len(meta[key][0]) == 2

    quant = [k for k in meta if _quantizable(k)]
    qset = set(quant)
    print(f"to quantize: {len(quant)} Linear weight(s); "
          f"copied verbatim: {len(meta) - len(quant)}")

    # ── output order: each module's weight, its scale, then its bias ────────
    #
    # Grouped so a module's scale is known by the time its bias is written, and
    # nothing is read twice. The owned biases are collected FIRST: ".bias" sorts
    # before ".weight", so without that they get emitted once in their own right
    # and again inside the group, and a duplicate key silently overwrites its
    # own header entry -- safetensors then refuses the file with "invalid offset
    # for tensor ..." because the offsets no longer tile the data. (Caught by the
    # round-trip test, which is the only reason this comment is here.)
    owned_bias = {k[: -len(".weight")] + ".bias" for k in qset}
    owned_bias &= set(meta)
    order, seen = [], set()
    for k in sorted(meta):
        if k in seen or k in owned_bias:
            continue
        order.append(k)
        seen.add(k)
        if k in qset:
            base = k[: -len(".weight")]
            order.append(base + SCALE_SUFFIX)
            b = base + ".bias"
            if b in owned_bias:
                order.append(b)
                seen.add(b)
    assert len(order) == len(meta) + len(quant), (
        f"ordering emitted {len(order)} entries for {len(meta)} tensors "
        f"plus {len(quant)} scales")

    # ── header: shapes and dtypes only, so it needs no tensor data ──────────
    header, off = {}, 0
    for k in order:
        if k.endswith(SCALE_SUFFIX):
            shape, nb, dt = [], 4, "F32"
        else:
            shape = list(meta[k][0])
            if k in qset:
                dt, esz = "F8_E4M3", 1
            else:
                dt = ST_NAME.get("torch." + meta[k][1].split(".")[-1]) \
                     or meta[k][1].upper()
                esz = {"F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1, "BOOL": 1,
                       "BF16": 2, "F16": 2, "F32": 4, "I32": 4,
                       "F64": 8, "I64": 8}.get(dt)
                if esz is None:
                    raise SystemExit(f"unhandled source dtype {meta[k][1]!r} for {k}")
            n = 1
            for d in shape:
                n *= d
            nb = n * esz
        header[k] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + nb]}
        off += nb
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * ((8 - len(blob) % 8) % 8)          # data starts 8-aligned
    total = 8 + len(blob) + off
    src_bytes = sum(os.path.getsize(s) for s in shards)
    print(f"output: {len(order)} tensors, {total / 2**30:.2f} GiB "
          f"(source {src_bytes / 2**30:.2f} GiB, "
          f"{100 * total / src_bytes:.0f}%)")
    if args.dry_run:
        return 0

    tmp = args.out + ".partial"
    opened, written, scales = {}, 0, []
    try:
        with open(tmp, "wb", buffering=1024 * 1024) as w:
            w.write(struct.pack("<Q", len(blob)))
            w.write(blob)
            cur_scale = None
            for k in order:
                if k.endswith(SCALE_SUFFIX):
                    assert cur_scale is not None, k
                    w.write(struct.pack("<f", cur_scale))
                    scales.append(cur_scale)
                    continue
                path = where[k]
                if path not in opened:
                    for h in opened.values():
                        h.__exit__(None, None, None)
                    opened.clear()
                    h = safe_open(path, framework="pt")
                    h.__enter__()
                    opened[path] = h
                t = opened[path].get_tensor(k)
                if k in qset:
                    amax = float(t.abs().to(torch.float32).max())
                    cur_scale = (amax / E4M3_MAX) if amax > 0 else 1.0
                    q = (t.to(torch.float32) / cur_scale).clamp(
                        -E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
                    w.write(_raw(q))
                elif cur_scale is not None and k.endswith(".bias") \
                        and (k[: -len(".bias")] + ".weight") in qset:
                    # Scaled with its weight: the hook multiplies the module's
                    # whole output, bias included.
                    b = (t.to(torch.float32) / cur_scale).to(t.dtype)
                    w.write(_raw(b))
                else:
                    w.write(_raw(t))
                written += 1
                if written % 500 == 0:
                    print(f"  {written}/{len(meta)} tensors…", flush=True)
    finally:
        for h in opened.values():
            h.__exit__(None, None, None)
    got = os.path.getsize(tmp)
    if got != total:
        os.unlink(tmp)
        raise SystemExit(f"wrote {got} bytes, header promised {total} — removed")
    os.replace(tmp, args.out)
    print(f"wrote {args.out}  ({got / 2**30:.2f} GiB)")
    if scales:
        print(f"  {len(scales)} scales, range {min(scales):.3g}–{max(scales):.3g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
