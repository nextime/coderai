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

        import torch
        from longcat_video.modules.quantization import (
            quantize_model, save_quantized_state_dict)
        from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel

        if os.path.isdir(out_dir) and not job.get("overwrite"):
            return finish(True, path=out_dir, skipped="already quantised")

        # Refuse rather than take the machine down. Measured the hard way: on a 54 GB
        # host this filled RAM AND all 4 GB of swap, and had to be killed to keep the
        # server alive. The peak is roughly the bf16 model plus the INT8 copy plus a
        # full state dict at save time, so ~2x the bf16 size is the honest floor.
        need_gb = float(job.get("min_free_gb") or 0) or _needed_gb(source, subfolder)
        free_gb = _available_gb()
        if free_gb and free_gb < need_gb:
            return finish(False, error=(
                f"not enough host memory to quantise: ~{need_gb:.0f} GB needed, "
                f"{free_gb:.0f} GB available. The bf16 model, the INT8 copy and the "
                f"state dict are all resident at the peak. Free memory (stop other "
                f"models) or raise min_free_gb if you know better."))

        emit(step=1, total=4,
             message=f"loading the bf16 DiT from {subfolder}/ (~{need_gb:.0f} GB needed)")
        # On CPU on purpose: the point of this job is to make a model that FITS the
        # card, so getting there must not need the card. low_cpu_mem_usage streams the
        # shards through a meta-device skeleton instead of building a second full copy
        # while loading.
        dit = LongCatVideoTransformer3DModel.from_pretrained(
            source, subfolder=subfolder, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True)
        dit = dit.to("cpu").eval()
        gc.collect()

        emit(step=2, total=4, message="quantising every Linear to INT8")
        with torch.no_grad():
            dit = quantize_model(dit)
        # The bf16 weights the QuantizedLinear layers replaced are garbage now; they
        # are ~13 GB and the save below allocates again, so collect before it.
        gc.collect()

        emit(step=3, total=4, message=f"writing {out_name}/")
        save_quantized_state_dict(
            dit, out_dir, config_source_dir=os.path.join(source, subfolder))
        del dit
        gc.collect()

        size = sum(
            os.path.getsize(os.path.join(out_dir, f))
            for f in os.listdir(out_dir)
            if os.path.isfile(os.path.join(out_dir, f)))
        emit(step=4, total=4, message=f"done — {size / 1e9:.1f} GB in {out_name}/")
        return finish(True, path=out_dir, bytes=size)
    except Exception as exc:                                   # noqa: BLE001
        emit(message=f"failed: {exc}")
        return finish(False, error=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc()[-4000:])


if __name__ == "__main__":
    raise SystemExit(main())
