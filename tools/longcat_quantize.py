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
import json
import os
import sys
import time
import traceback


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

        emit(step=1, total=4, message=f"loading the bf16 DiT from {subfolder}/")
        # On CPU on purpose: the bf16 DiT is ~27 GB and the point of this job is to
        # make it FIT the card, so it must not need the card to get there.
        dit = LongCatVideoTransformer3DModel.from_pretrained(
            source, subfolder=subfolder, torch_dtype=torch.bfloat16)
        dit = dit.to("cpu").eval()

        emit(step=2, total=4, message="quantising every Linear to INT8")
        dit = quantize_model(dit)

        emit(step=3, total=4, message=f"writing {out_name}/")
        save_quantized_state_dict(
            dit, out_dir, config_source_dir=os.path.join(source, subfolder))

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
