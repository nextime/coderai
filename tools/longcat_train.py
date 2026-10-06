#!/usr/bin/env python3
# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Drive a LongCat-Video LoRA / QLoRA training run through SimpleTuner.

coderai does NOT implement this loop. Two reasons, both hard:

1. LongCat's venv is standalone — Python 3.10, torch 2.6, transformers 4.41 — so a
   trainer there cannot ``from codai.api import loras`` the way
   ``tools/lora_train_worker.py`` does. That overlay venv inherits the parent's
   site-packages; this one deliberately does not.
2. The upstream repo offers no way to CREATE LoRA layers. Its DiT exposes
   ``load_lora()`` / ``enable_loras()`` / ``disable_all_loras()`` and nothing that
   initialises trainable ones, and the adapters it does load (``cfg_step_lora``,
   ``refinement_lora``, ``dmd_lora``) use its own key layout. Reimplementing that layout
   without being able to verify it would produce adapters the pipeline cannot load —
   worse than no support, because it would look like support.

SimpleTuner implements LongCat-Video properly (``model_family: "longcat_video"``, LoRA
and quantised LoRA via int8-quanto / int4-quanto / fp8-torchao), so this writes its
config and runs it, translating progress into the same JSON-lines protocol
``tools/lora_train_worker.py`` uses so the existing job records and the Tasks page work
unchanged.

Invoked by codai/api/longcat_worker.py, never by hand:

    <train venv python> tools/longcat_train.py --job /path/to/job.json
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

# SimpleTuner reports steps as "step 120/800" or a tqdm-style "120/800"; either is
# enough to drive a progress bar.
_STEP_RE = re.compile(r"(?:step[^\d]{0,3})?(\d+)\s*/\s*(\d+)", re.IGNORECASE)
_LOSS_RE = re.compile(r"loss[=:\s]+([0-9]*\.?[0-9]+)", re.IGNORECASE)


def build_config(job: dict) -> dict:
    """The SimpleTuner config for one job.

    Only the keys its LongCat-Video quickstart calls for, plus what the request asked.
    `pretrained_model_name_or_path` is left unset on purpose: the flavour selects the
    checkpoint."""
    req = job.get("request") or {}
    cfg = {
        "model_type": "lora",
        "model_family": "longcat_video",
        "model_flavour": job.get("flavour") or "final",
        "output_dir": job["output_dir"],
        "data_backend_config": job["dataset_config"],
        # The 13.6B transformer leaves no room for a larger batch, and gradient
        # checkpointing is what makes it fit at all.
        "train_batch_size": 1,
        "gradient_checkpointing": bool(job.get("gradient_checkpointing", True)),
        "lora_rank": int(job.get("rank") or 8),
        "max_train_steps": int(job.get("steps") or 800),
        "learning_rate": float(job.get("lr") or 1e-4),
        "seed": int(job.get("seed") or 42),
        "validation_resolution": job.get("resolution") or "480x832",
        "validation_num_video_frames": training_frames(job.get("num_frames") or 93),
        "tracker_project_name": "coderai-longcat-lora",
        # SimpleTuner has no default for this — it is a REQUIRED argument, and
        # omitting it fails at argparse with "the following arguments are
        # required: --optimizer", long before anything is loaded.
        #
        # adamw_bf16 keeps the optimiser state in bf16 rather than fp32, which is
        # what makes a 13.6B base trainable on one consumer card at all: Adam
        # normally carries two fp32 moments per parameter. The quantised base
        # (base_model_precision) covers the weights; this covers the state.
        "optimizer": job.get("optimizer") or "adamw_bf16",
    }
    # QLoRA: the quantised base. Without it the 13.6B transformer plus optimiser state
    # does not fit a consumer card.
    precision = str(job.get("base_precision") or "").strip()
    if precision:
        cfg["base_model_precision"] = precision
    if req.get("instance_prompt"):
        cfg["instance_prompt"] = req["instance_prompt"]
    return cfg


def training_frames(num_frames: int) -> int:
    """The frame count SimpleTuner will actually train at.

    TWO rules apply and they are not the same. Generation obeys the VAE's 4n+1 —
    93, 173, 333 are whole LongCat segments. TRAINING is stricter: SimpleTuner's
    longcat_video family rounds DOWN to 8n+1 (frames % 8 == 1) and says nothing,
    so asking for 93 trains at 89 and the number in the config is not the number
    that ran. Rounding here means the config states what will happen.
    """
    n = int(num_frames or 0)
    if n < 1:
        return 1
    if n % 8 == 1:
        return n
    return max(((n - 1) // 8) * 8 + 1, 1)


def frame_problems(num_frames: int, resolution: str) -> list:
    """Constraints checked before a long run starts.

    These are the model's, not preferences. Note the frame rule here is the
    TRAINING one (8n+1), which is stricter than the 4n+1 the VAE imposes on
    generation — see training_frames(). Each side of the resolution must be
    divisible by 16."""
    problems = []
    n = int(num_frames or 0)
    if n < 1 or (n - 1) % 8 != 0:
        problems.append(
            f"num_frames must be 8n+1 for training (89, 97, 105 …); got {n}. "
            f"Generation's rule is 4n+1, which is looser — 93 is a valid clip "
            f"length but not a valid training length.")
    try:
        w, h = (int(x) for x in str(resolution).lower().split("x"))
    except Exception:
        return problems + [f"resolution must look like 480x832; got {resolution!r}"]
    for name, v in (("width", w), ("height", h)):
        if v % 16:
            problems.append(f"{name} must be divisible by 16 (the VAE stride); got {v}")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="LongCat-Video LoRA trainer (SimpleTuner)")
    ap.add_argument("--job", required=True, help="path to the job JSON")
    args = ap.parse_args(argv)

    job = json.load(open(args.job))
    progress_path = args.job + ".progress"
    result_path = args.job + ".result"

    def emit(**kw):
        kw["at"] = time.time()
        with open(progress_path, "a") as fh:
            fh.write(json.dumps(kw) + "\n")

    try:
        # Round to the training rule BEFORE validating, so a clip length that is
        # legal for generation (4n+1) does not block a run — it is adjusted and
        # said out loud, rather than silently changed inside SimpleTuner.
        asked = int(job.get("num_frames") or 93)
        trains_at = training_frames(asked)
        if trains_at != asked:
            job["num_frames"] = trains_at
            emit(message=f"{asked} frames is not 8n+1, which training requires — "
                         f"using {trains_at} (generation's 4n+1 rule is looser)")
            print(f"[longcat-train] num_frames {asked} -> {trains_at} (8n+1)",
                  flush=True)

        problems = frame_problems(job.get("num_frames") or 93,
                                  job.get("resolution") or "480x832")
        if problems:
            raise RuntimeError("; ".join(problems))

        workdir = Path(args.job).parent
        cfg = build_config(job)
        cfg_path = workdir / "simpletuner.json"
        cfg_path.write_text(json.dumps(cfg, indent=2))
        emit(status="starting",
             message=f"SimpleTuner, LongCat-Video LoRA rank {cfg['lora_rank']}"
                     + (f", base {cfg['base_model_precision']}"
                        if cfg.get("base_model_precision") else ", bf16 base"))

        # SimpleTuner does NOT take --config. Its loader picks a configuration
        # BACKEND and then asks that backend where to look; the JSON one reads
        # CONFIG_PATH (a file, or a directory holding config.json) and otherwise
        # defaults to ./config/config.json relative to the working directory —
        # which is what produced
        #   ValueError: JSON configuration file not found. Paths tried: config/config.json
        # So the backend is named explicitly and the path handed over in the
        # environment, rather than as an argument it ignores.
        env = os.environ.copy()
        env["SIMPLETUNER_CONFIG_BACKEND"] = "json"
        env["CONFIG_PATH"] = str(cfg_path)
        # Belt and braces: the default lookup is ./config/config.json relative to
        # cwd, so the same file is placed there too. A loader change that ignored
        # CONFIG_PATH would still find it.
        fallback = workdir / "config"
        fallback.mkdir(parents=True, exist_ok=True)
        (fallback / "config.json").write_text(json.dumps(cfg, indent=2))

        cmd = [sys.executable, "-m", "simpletuner.train"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, cwd=str(workdir), env=env)
        total = int(cfg["max_train_steps"])
        last = -1
        tail = []
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            tail.append(line[:300])
            del tail[:-25]
            print(f"[longcat-train] {line}", flush=True)
            m = _STEP_RE.search(line)
            if m:
                step, seen_total = int(m.group(1)), int(m.group(2))
                if seen_total == total and step != last:
                    last = step
                    loss = _LOSS_RE.search(line)
                    emit(status="training", step=step, total=total,
                         message=(f"step {step}/{total}"
                                  + (f", loss {loss.group(1)}" if loss else "")))
        rc = proc.wait()
        if rc != 0:
            # The last few lines of a traceback are frame noise — the line that
            # says what went wrong is usually further up. Prefer an exception or
            # error line if one is in the tail; fall back to the raw end.
            import re as _re
            _signal = [t for t in tail
                       if _re.search(r"(Error|error:|Exception|required:|No such|not found)", t)]
            _detail = " | ".join((_signal or tail)[-4:])
            raise RuntimeError(f"SimpleTuner exited {rc}: {_detail}")

        # The adapter SimpleTuner wrote. Reported as a path rather than moved: the
        # parent owns where a LoRA is registered.
        out = Path(cfg["output_dir"])
        found = sorted(out.rglob("*.safetensors"), key=lambda p: p.stat().st_mtime)
        if not found:
            raise RuntimeError(f"training finished but no .safetensors appeared under "
                               f"{out}")
        json.dump({"ok": True, "result": {"path": str(found[-1]),
                                          "steps": total,
                                          "rank": cfg["lora_rank"]}},
                  open(result_path, "w"))
        emit(status="done", message=f"trained {found[-1].name}")
        return 0
    except Exception as exc:
        json.dump({"ok": False, "error": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc()[-2000:]},
                  open(result_path, "w"))
        emit(status="error", message=str(exc)[:300])
        print(traceback.format_exc(), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
