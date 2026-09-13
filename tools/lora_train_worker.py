#!/usr/bin/env python3
# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Run one LoRA training job in a venv that has a newer diffusers.

MiniMax-H3 and Krea 2 need diffusers >= 0.40 while the server runs 0.38, and
diffusers is imported once per process — so those two architectures train here,
in the overlay venv (codai/api/h3_worker.overlay_python), instead of in the engine.

Invoked by codai/api/loras.py, never by hand:

    <overlay python> tools/lora_train_worker.py --job /path/to/job.json

The job file carries the request fields plus the resolved image paths; progress is
appended to <job>.progress as JSON lines so the parent can surface it, and the
final result is written to <job>.result.
"""

import argparse
import json
import os
import sys
import time
import traceback

# The server's own modules — importable because the overlay inherits the parent's
# site-packages; only diffusers/huggingface_hub differ.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Out-of-process LoRA trainer")
    ap.add_argument("--job", required=True, help="Path to the job JSON")
    args = ap.parse_args(argv)

    job = json.load(open(args.job))
    progress_path = args.job + ".progress"
    result_path = args.job + ".result"

    def emit(**kw):
        kw["at"] = time.time()
        with open(progress_path, "a") as fh:
            fh.write(json.dumps(kw) + "\n")

    try:
        import diffusers
        emit(status="starting", message=f"diffusers {diffusers.__version__} "
                                        f"(overlay), arch={job['arch']}")
        from PIL import Image
        from codai.api import loras

        # Progress from inside the trainer goes to the job file, so the parent can
        # poll it without sharing memory.
        def _set_progress(**kw):
            emit(**kw)
        loras._set_progress = _set_progress
        loras._check_train_cancel = lambda: None   # cancellation is the parent's job

        images = [Image.open(p).convert("RGB") for p in job["images"]]
        if not images:
            raise RuntimeError("no reference images were passed to the trainer")

        class _Req:
            pass
        req = _Req()
        for k, v in job["request"].items():
            setattr(req, k, v)

        result = loras._train_flow_dit(
            job["arch"], req, job["base_path"], images, job["instance_prompt"],
            int(job["steps"]), int(job["rank"]), int(job["resolution"]),
            float(job["lr"]), int(job["seed"]), job["device"])
        json.dump({"ok": True, "result": result}, open(result_path, "w"))
        emit(status="done", message="training finished")
        return 0
    except Exception as exc:
        json.dump({"ok": False, "error": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc()[-2000:]}, open(result_path, "w"))
        emit(status="error", message=str(exc)[:300])
        print(traceback.format_exc(), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
