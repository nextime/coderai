#!/usr/bin/env python3
# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
# Licensed under the GNU General Public License v3 or later.
"""What a rented training pod runs. Fetched from the orchestrator, not baked.

Served by ``GET /v1/loras/train/text/{job}/runner`` so it can change without
republishing a 22 GB image, and so the pod and the orchestrator can never be
two different versions of this protocol.

The pod's whole life, and note that every step is OUTBOUND -- nothing listens,
nothing is addressable:

    1. ask what to train            GET  …/{job}/shard?shard=N
    2. fetch the dataset once       GET  …/datasets/{id}/content
    3. fetch the round's starting
       adapter (none on round 0)    GET  …/{job}/round/{k}/adapter
    4. train this round's slice     (locally, on the card we rented)
    5. hand the adapter in          POST …/{job}/round?shard=N&round=k
    6. go to 1

It exits when the orchestrator says the job is finished, and it exits non-zero
on an error it cannot retry -- a pod that stays alive doing nothing is a pod
that bills for nothing.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request

URL = (os.environ.get("CODERAI_URL") or "").rstrip("/")
JOB = os.environ.get("CODERAI_TRAIN_JOB") or ""
SHARD = int(os.environ.get("CODERAI_TRAIN_SHARD") or "0")
SECRET = os.environ.get("CODERAI_TRAIN_SECRET") or ""
BASE = os.environ.get("CODERAI_TRAIN_BASE") or ""
WORK = os.environ.get("CODERAI_TRAIN_WORKDIR") or "/workspace/train"


def _say(msg):
    print(f"[pod {SHARD}] {msg}", flush=True)


def _req(path, *, method="GET", data=None, out_path=None, timeout=1800):
    req = urllib.request.Request(
        URL + path, data=data, method=method,
        headers={"Authorization": f"Bearer {SECRET}",
                 "Content-Type": "application/octet-stream"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if out_path:
            with open(out_path, "wb") as fh:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
            return out_path
        body = resp.read()
    return json.loads(body) if body else {}


def _retrying(fn, what, attempts=5, wait=10):
    """Retry a transient failure; give up loudly on a permanent one.

    A 409 means this pod is out of step and must re-sync, not retry -- so it is
    raised rather than slept over.
    """
    for i in range(1, attempts + 1):
        try:
            return fn()
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 404, 409):
                raise
            _say(f"{what}: HTTP {exc.code}, attempt {i}/{attempts}")
        except Exception as exc:
            _say(f"{what}: {exc}, attempt {i}/{attempts}")
        if i < attempts:
            time.sleep(wait * i)
    raise RuntimeError(f"{what} failed after {attempts} attempts")


def main() -> int:
    if not (URL and JOB and SECRET):
        _say("CODERAI_URL, CODERAI_TRAIN_JOB and CODERAI_TRAIN_SECRET are required")
        return 2
    os.makedirs(WORK, exist_ok=True)
    sys.path.insert(0, "/opt/trainer")
    from codai.api import text_lora_dataset as tld
    from codai.api import text_lora_trainer as tlt

    dataset_file = os.path.join(WORK, "dataset.jsonl")
    examples = None
    tokenizer = None
    model = None

    while True:
        plan = _retrying(lambda: _req(f"/v1/loras/train/text/{JOB}/shard?shard={SHARD}"),
                         "asking for work")
        if plan.get("finished"):
            _say("the job is finished")
            return 0
        rnd = int(plan["round"])
        st = plan.get("settings") or {}
        settings = tlt.QLoraSettings(
            rank=int(st.get("rank", 16)), alpha=int(st.get("alpha", 32)),
            dropout=float(st.get("dropout", 0.05)),
            learning_rate=float(st.get("learning_rate", 1e-4)),
            max_len=int(st.get("max_len", 8192)),
            accumulation_steps=int(st.get("accumulation_steps", 8)),
            enable_thinking=bool(st.get("enable_thinking", False)),
            seed=int(st.get("seed", 42)))

        # The dataset arrives once and is kept: it is the same file every round,
        # and re-fetching 119 MB per round would be most of a short round.
        if not os.path.isfile(dataset_file):
            ident = plan.get("dataset_id") or ""
            _say(f"fetching the dataset {ident[:20]}…")
            _retrying(lambda: _req(f"/v1/loras/datasets/{ident}/content",
                                   out_path=dataset_file), "fetching the dataset")

        if tokenizer is None:
            from transformers import AutoTokenizer
            base = BASE or plan.get("base") or ""
            _say(f"loading the tokenizer for {base}")
            tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)

        if examples is None:
            _say("encoding the dataset")
            rows = list(tld.load_jsonl(dataset_file,
                                       field_map=plan.get("field_map") or None))
            examples, stats = tld.prepare(tokenizer, rows,
                                          max_len=settings.max_len,
                                          enable_thinking=settings.enable_thinking)
            _say(f"encoded: {stats.describe()}")

        mine = [examples[i] for i in plan.get("indices") or [] if i < len(examples)]
        if not mine:
            _say(f"round {rnd}: no examples for this shard; waiting")
            time.sleep(30)
            continue

        resume = plan.get("resume_round")
        start_from = None
        if resume is not None:
            start_from = os.path.join(WORK, f"round{resume}.safetensors")
            if not os.path.isfile(start_from):
                _say(f"fetching the round {resume} average to start from")
                _retrying(lambda: _req(
                    f"/v1/loras/train/text/{JOB}/round/{resume}/adapter",
                    out_path=start_from), "fetching the average")

        out_dir = os.path.join(WORK, f"out-round{rnd}")
        _say(f"round {rnd}: training {len(mine)} examples")
        result = tlt.run_training(base=BASE or plan.get("base") or "",
                                  encoded=mine, settings=settings,
                                  out_dir=out_dir, resume=False,
                                  init_adapter=start_from)
        adapter = os.path.join(out_dir, "adapter_model.safetensors")
        if not os.path.isfile(adapter):
            _say(f"round {rnd}: no adapter at {adapter}")
            return 1
        with open(adapter, "rb") as fh:
            blob = fh.read()
        trained = int(result.get("examples_trained") or 0)
        _say(f"round {rnd}: handing in {len(blob) / 1e6:.0f} MB "
             f"({trained} examples trained)")
        try:
            _retrying(lambda: _req(
                f"/v1/loras/train/text/{JOB}/round?shard={SHARD}&round={rnd}"
                f"&examples={trained}", method="POST", data=blob),
                "handing in the adapter")
        except urllib.error.HTTPError as exc:
            if exc.code == 409:
                # Out of step: the orchestrator has moved on. Re-ask rather than
                # retry, which is what the 409 is for.
                _say(f"round {rnd} is closed; re-syncing")
                continue
            raise


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"[pod] fatal: {exc}", flush=True)
        # Non-zero on purpose: a pod that stays alive doing nothing bills for
        # nothing, and the orchestrator reaps on silence anyway.
        sys.exit(1)
