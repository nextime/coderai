# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""A diffusion pipeline's parts on different machines.

A video model is several models: a text encoder (UMT5-XXL, 11 GB in fp16),
one or two denoising experts (Wan 2.2: high-noise then low-noise, ~10 GB
each at 4-bit) and a VAE. They run one after the other, and what passes
between them is small — prompt embeddings, a latent tensor of a few MB —
so each can live on a different machine and the run is a relay over a LAN,
not a tensor-parallel all-reduce. Per model, ``components`` in models.json:

    {"text_encoder": "box2", "low_noise": "box3", "vae": "box3"}

Each value is a cluster node name (or a URL block). The machine that took
the request keeps what is not named: it encodes the prompt unless
``text_encoder`` says otherwise, runs the high-noise expert (or the whole
loop when the model has one expert), hands the latents to ``low_noise``'s
machine at the expert boundary, and decodes them unless ``vae`` says
otherwise — in which case the far machine returns the finished video.

The far side runs the ordinary /v1/video/generations with a ``_handoff``
block: the same request, so model loading, acceleration and LoRAs happen
exactly as for a normal generation there; only what the loop starts from
and what it returns differ (codai/api/video.py, _generate_video).
"""

import base64
import io
from typing import Optional

_KEYS = ("text_encoder", "low_noise", "vae")


def parse_components(raw) -> dict:
    """{"text_encoder": node, "low_noise": node, "vae": node} — blank = here."""
    d = raw if isinstance(raw, dict) else {}
    out = {}
    for k in _KEYS:
        v = d.get(k)
        if isinstance(v, str) and v.strip() and v.strip().lower() not in ("local", "here", "auto"):
            out[k] = v.strip()
        elif isinstance(v, dict) and v.get("url"):
            out[k] = dict(v)
    return out


def components_for(model_name: str) -> dict:
    """This model's placement, from its models.json entry."""
    try:
        from codai.models.manager import _model_entry_for
        entry = _model_entry_for(model_name) or {}
    except Exception:
        return {}
    return parse_components(entry.get("components"))


# ------------------------------------------------------------- tensors
def tensor_to_b64(t) -> str:
    """A tensor as safetensors bytes, base64 — dtype and shape travel with it."""
    from safetensors.torch import save
    return base64.b64encode(save({"t": t.detach().contiguous().cpu()})).decode()


def b64_to_tensor(s: str, device=None, dtype=None):
    from safetensors.torch import load
    t = load(base64.b64decode(s))["t"]
    if dtype is not None:
        t = t.to(dtype)
    if device is not None:
        t = t.to(device)
    return t


# --------------------------------------------------------------- remote
def _peer(spec) -> dict:
    """A node name or a {url, api_key} block → {name, url, api_key, verify}."""
    if isinstance(spec, dict):
        return {"name": spec.get("name") or spec.get("url"), "url": str(spec["url"]).rstrip("/"),
                "api_key": spec.get("api_key", ""), "verify": spec.get("verify", "system")}
    from codai.cluster.ddp import resolve_peers
    return resolve_peers([{"name": spec}])[0]


def call_handoff(spec, request_json: dict, handoff: dict, timeout: float = 3600.0) -> dict:
    """POST the same generation request with a ``_handoff`` block to the
    machine that owns a component; returns the JSON it answers."""
    from codai.api import pod_http
    peer = _peer(spec)
    body = dict(request_json)
    body["_handoff"] = handoff
    headers = {"Content-Type": "application/json"}
    if peer.get("api_key"):
        headers["Authorization"] = f"Bearer {peer['api_key']}"
    r = pod_http.post(peer["url"] + "/v1/video/generations", json=body, headers=headers,
                      timeout=timeout, verify=(False if peer.get("verify") == "off" else True))
    if r.status_code != 200:
        raise RuntimeError(f"{peer['name']}: {r.status_code} {(r.text or '')[:300]}")
    return r.json()


def request_json(request) -> dict:
    """The generation request as the far side must receive it — every field
    the client sent, minus anything a client never sends."""
    try:
        d = request.model_dump(exclude_none=True)
    except Exception:
        d = dict(getattr(request, "__dict__", {}))
    d.pop("_handoff", None)
    return d


# ---------------------------------------------------- the wan step split
def boundary_step(pipe, kw: dict) -> Optional[int]:
    """The first loop index that uses the low-noise expert, or None when the
    model has a single expert. Computed from the scheduler exactly as the
    pipeline will (set_timesteps + the boundary ratio)."""
    ratio = getattr(getattr(pipe, "config", None), "boundary_ratio", None)
    if ratio is None or getattr(pipe, "transformer_2", None) is None:
        return None
    import copy
    sched = copy.deepcopy(pipe.scheduler)
    sched.set_timesteps(int(kw.get("num_inference_steps") or 25))
    boundary = float(ratio) * float(sched.config.num_train_timesteps)
    for i, t in enumerate(sched.timesteps):
        if float(t) < boundary:
            return i
    return None


class SlicedTimesteps:
    """Make the pipeline's scheduler start at loop index ``start``: after the
    pipeline's own ``set_timesteps`` the timestep and sigma tables are cut
    so the loop runs only the remaining steps and ``step()`` still finds
    each timestep's sigma pair."""

    def __init__(self, scheduler, start: int):
        self.scheduler = scheduler
        self.start = int(start)
        self._orig = None

    def __enter__(self):
        sched, start = self.scheduler, self.start
        self._orig = sched.set_timesteps

        def _set(*a, **k):
            self._orig(*a, **k)
            sched.timesteps = sched.timesteps[start:]
            if getattr(sched, "sigmas", None) is not None:
                sched.sigmas = sched.sigmas[start:]
            if hasattr(sched, "_step_index"):
                sched._step_index = None
            if hasattr(sched, "_begin_index"):
                sched._begin_index = None
        sched.set_timesteps = _set
        return self

    def __exit__(self, *exc):
        self.scheduler.set_timesteps = self._orig
        return False
