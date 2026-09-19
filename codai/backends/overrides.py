# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Per-model blocks laid over a global engine config.

The global ``vllm``/``ktransformers`` sections are templates; a model's own
``vllm``/``kt`` block in models.json sets what differs for THAT model —
parallelism, the machines it spans, memory fraction, extra flags. Only the
listed fields may be overridden: a model must never redirect the engine to a
different install, venv or service.
"""

import dataclasses

VLLM_FIELDS = ("ctx", "gpu", "gpu_memory_utilization", "tensor_parallel_size",
               "pipeline_parallel_size", "distributed_executor_backend", "ray_address",
               "ray_port", "nodes", "nodes_ready_timeout_s", "max_num_seqs", "dtype",
               "quantization", "extra_args", "extra_env")

KT_FIELDS = ("ctx", "kt_weight_path", "nnodes", "dist_init_addr", "tp_size", "nodes",
             "extra_args", "extra_env")


def apply(cfg, block: dict, fields: tuple, label: str = ""):
    """``cfg`` with the allowed keys of ``block`` replaced; logs what changed."""
    if not isinstance(block, dict) or not block:
        return cfg
    overrides = {}
    for k in fields:
        if k not in block:
            continue
        v = block[k]
        if v in (None, ""):
            continue
        cur = getattr(cfg, k, None)
        if isinstance(cur, bool):
            v = str(v).strip().lower() in ("1", "true", "yes", "on") if isinstance(v, str) else bool(v)
        elif isinstance(cur, int):
            try:
                v = int(v)
            except (TypeError, ValueError):
                continue
        elif isinstance(cur, float):
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
        overrides[k] = v
    if not overrides:
        return cfg
    try:
        out = dataclasses.replace(cfg, **overrides)
    except Exception as exc:
        print(f"[{label or 'engine'}] per-model overrides not applied: {exc}", flush=True)
        return cfg
    print(f"[{label or 'engine'}] per-model overrides: "
          + ", ".join(f"{k}={v!r}" for k, v in overrides.items()), flush=True)
    return out


def model_block(model_name: str, key: str) -> dict:
    """The named block of this model's models.json entry, or {}."""
    try:
        from codai.models.manager import _model_entry_for
        entry = _model_entry_for(model_name) or {}
    except Exception:
        return {}
    b = entry.get(key)
    return b if isinstance(b, dict) else {}
