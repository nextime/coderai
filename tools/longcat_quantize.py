#!/usr/bin/env python3
"""Quantise a LongCat-Video DiT to INT8, in LongCat's own isolated venv.

Why this exists as a separate script: the quantisation helpers live in the
upstream package, which needs Python 3.10 + torch 2.6 — coderai's own interpreter
is 3.13, so this cannot run in-process any more than training can.

What it produces is exactly what the inference path already knows how to load: a
``base_model_int8/`` directory beside the bf16 ``dit/``, in upstream's own format
(per-channel symmetric weight-only INT8, sharded safetensors plus an index). The
model then loads at roughly half the weight footprint — which is what lets a 13.6B
DiT sit on a 24 GB card instead of streaming across PCIe every step.

It is NOT automatic. Quantisation is lossy and slow, so it is an action someone
chooses on the model page, and the bf16 weights are never touched or replaced.

Invoked by codai/api/longcat_worker.py, never by hand:

    <longcat venv python> tools/longcat_quantize.py --job /path/to/job.json
"""

import argparse
import gc
import json
import os
import sys
import time
import traceback


def _linear_paths(source: str, subfolder: str) -> tuple:
    """Which parameters belong to a Linear that upstream would quantise.

    The model is built on the META device: that allocates nothing, so a 13.6B
    transformer costs no memory here, and it gives the EXACT same set of modules
    upstream's quantize_model() would walk — including its skip patterns — rather
    than a guess from tensor shapes. Returns (paths_to_quantise, paths_with_bias).
    """
    import torch
    from longcat_video.modules.quantization import DEFAULT_SKIP_PATTERNS
    from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
    import torch.nn as nn

    with open(os.path.join(source, subfolder, "config.json"), encoding="utf-8") as fh:
        config = json.load(fh)
    for drop in ("_class_name", "architectures", "_diffusers_version", "model_max_length"):
        config.pop(drop, None)

    with torch.device("meta"):
        model = LongCatVideoTransformer3DModel(**config)

    quantise, with_bias = set(), set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not any(
                pat in name for pat in DEFAULT_SKIP_PATTERNS):
            quantise.add(name)
            if module.bias is not None:
                with_bias.add(name)
    del model
    return quantise, with_bias


def _quantise_streaming(source: str, subfolder: str, out_dir: str, emit) -> int:
    """Quantise tensor by tensor, writing shards as they fill.

    The whole point: never hold the model. Materialising it needed ~42 GB — the
    bf16 weights, the INT8 copies built while the originals were still referenced,
    and a full state dict at save time — which exhausted a 54 GB host. Here the
    peak is one tensor plus the shard being accumulated.

    The output is byte-for-byte the layout upstream's load_quantized_dit() expects:
    <path>.weight_int8 (int8), <path>.weight_scale (float32, per output channel),
    <path>.bias (bfloat16), everything else passed through as bf16.
    """
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    src_dir = os.path.join(source, subfolder)
    shards = sorted(f for f in os.listdir(src_dir) if f.endswith(".safetensors"))
    if not shards:
        raise RuntimeError(f"no safetensors in {src_dir}")

    quantise, _with_bias = _linear_paths(source, subfolder)
    emit(step=2, total=4,
         message=f"quantising {len(quantise)} Linear layers, streaming")

    os.makedirs(out_dir, exist_ok=True)
    max_shard = 4 * 1024 * 1024 * 1024
    out_shards, cur, cur_bytes, weight_map = [], {}, 0, {}
    total_size = 0

    def _flush():
        nonlocal cur, cur_bytes
        if not cur:
            return
        name = f"quantized_model-{len(out_shards) + 1:05d}-of-PLACEHOLDER.safetensors"
        path = os.path.join(out_dir, name)
        save_file(cur, path)
        out_shards.append(name)
        for k in cur:
            weight_map[k] = name
        cur, cur_bytes = {}, 0
        gc.collect()

    def _add(key, tensor):
        nonlocal cur_bytes, total_size
        nbytes = tensor.numel() * tensor.element_size()
        if cur_bytes + nbytes > max_shard and cur:
            _flush()
        cur[key] = tensor
        cur_bytes += nbytes
        total_size += nbytes

    seen = 0
    for shard in shards:
        with safe_open(os.path.join(src_dir, shard), framework="pt", device="cpu") as fh:
            for key in fh.keys():
                tensor = fh.get_tensor(key)
                module = key.rsplit(".", 1)[0]
                leaf = key.rsplit(".", 1)[-1]
                if module in quantise and leaf == "weight":
                    w = tensor.float()
                    # Per-channel symmetric, exactly as QuantizedLinear.from_linear.
                    scale = w.abs().amax(dim=1).clamp(min=1e-8) / 127.0
                    q = (w / scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
                    _add(f"{module}.weight_int8", q.contiguous())
                    _add(f"{module}.weight_scale", scale.float().contiguous())
                    del w, q, scale
                elif module in quantise and leaf == "bias":
                    _add(key, tensor.to(torch.bfloat16).contiguous())
                else:
                    _add(key, (tensor.to(torch.bfloat16)
                               if tensor.is_floating_point() else tensor).contiguous())
                del tensor
                seen += 1
                if seen % 200 == 0:
                    emit(step=2, total=4,
                         message=f"quantising — {seen} tensors, "
                                 f"{total_size / 1e9:.1f} GB written so far")
                    gc.collect()
    _flush()

    # The placeholder is only knowable once every shard is written.
    final = []
    for i, name in enumerate(out_shards, 1):
        fixed = f"quantized_model-{i:05d}-of-{len(out_shards):05d}.safetensors"
        if fixed != name:
            os.replace(os.path.join(out_dir, name), os.path.join(out_dir, fixed))
            for k, v in weight_map.items():
                if v == name:
                    weight_map[k] = fixed
        final.append(fixed)

    with open(os.path.join(out_dir, "quantized_model.safetensors.index.json"),
              "w", encoding="utf-8") as fh:
        json.dump({"metadata": {"total_size": total_size},
                   "weight_map": weight_map}, fh, indent=2)
    src_config = os.path.join(src_dir, "config.json")
    if os.path.exists(src_config):
        import shutil
        shutil.copy2(src_config, os.path.join(out_dir, "config.json"))
    with open(os.path.join(out_dir, "quantization_config.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"quantization_method": "int8_per_channel_symmetric",
                   "skip_patterns": sorted(
                       __import__("longcat_video.modules.quantization",
                                  fromlist=["x"]).DEFAULT_SKIP_PATTERNS),
                   "description": "Weight-only INT8 quantization with per-channel "
                                  "symmetric scaling"}, fh, indent=2)
    return total_size


def _available_gb() -> float:
    """Host memory actually available, or 0.0 if it cannot be read."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    return 0.0


def _needed_gb(source: str, subfolder: str) -> float:
    """A floor for the peak, from the weights on disk.

    The shards are fp32, so the bf16 model is about half of them; the peak then
    holds that plus the INT8 copy plus the state dict the save builds. 1.5x the
    bf16 size, with a 4 GB floor for the interpreter and torch itself.
    """
    d = os.path.join(source, subfolder)
    try:
        on_disk = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d)
                      if f.endswith(".safetensors"))
    except OSError:
        return 0.0
    bf16 = on_disk / 2 / (1024 ** 3)
    return bf16 * 1.5 + 4


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Quantise a LongCat DiT to INT8")
    ap.add_argument("--job", required=True, help="path to the job JSON")
    args = ap.parse_args(argv)

    job = json.load(open(args.job, encoding="utf-8"))
    progress_path = args.job + ".progress"
    result_path = args.job + ".result"

    def emit(**kw):
        kw["at"] = time.time()
        with open(progress_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(kw) + "\n")

    def finish(ok, **kw):
        with open(result_path, "w", encoding="utf-8") as fh:
            json.dump({"ok": ok, **kw}, fh)
        return 0 if ok else 1

    try:
        source = os.path.expanduser(job["checkpoint_dir"])
        subfolder = job.get("subfolder") or "dit"
        out_name = job.get("out_subfolder") or "base_model_int8"
        out_dir = os.path.join(source, out_name)

        src = os.path.expanduser(job.get("source_dir") or "")
        if src and src not in sys.path:
            sys.path.insert(0, src)

        # Imported for their side effect of failing EARLY and clearly if the venv or
        # the vendored package is wrong — the streaming path pulls them in itself.
        import torch                                               # noqa: F401
        from longcat_video.modules.quantization import QuantizedLinear  # noqa: F401

        if os.path.isdir(out_dir) and not job.get("overwrite"):
            return finish(True, path=out_dir, skipped="already quantised")

        # The peak is now one tensor plus the shard being accumulated (4 GB), not
        # the model — so the old "~42 GB needed" ceiling no longer applies. The
        # floor is kept as a sanity check, not as the real constraint.
        need_gb = float(job.get("min_free_gb") or 0) or 8.0
        free_gb = _available_gb()
        if free_gb and free_gb < need_gb:
            return finish(False, error=(
                f"not enough host memory: ~{need_gb:.0f} GB needed, "
                f"{free_gb:.0f} GB available"))

        emit(step=1, total=4, message="reading the checkpoint layout")
        # NOT loaded: the model is built on the meta device purely to learn which
        # parameters belong to a Linear, then the weights are streamed tensor by
        # tensor. Materialising it is what exhausted a 54 GB host.
        size = _quantise_streaming(source, subfolder, out_dir, emit)
        emit(step=3, total=4, message=f"wrote {size / 1e9:.1f} GB to {out_name}/")

        on_disk = sum(
            os.path.getsize(os.path.join(out_dir, f))
            for f in os.listdir(out_dir)
            if os.path.isfile(os.path.join(out_dir, f)))
        emit(step=4, total=4, message=f"done — {on_disk / 1e9:.1f} GB in {out_name}/")
        return finish(True, path=out_dir, bytes=on_disk)
    except Exception as exc:                                   # noqa: BLE001
        emit(message=f"failed: {exc}")
        return finish(False, error=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc()[-4000:])


if __name__ == "__main__":
    raise SystemExit(main())
