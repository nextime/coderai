# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Another coderai as an engine of this one.

The front already knows how to drive N engines it cannot see inside: it polls
``/internal/engine-state``, hands each one the models it owns, and proxies
requests by capability and load. A node is that, one network hop away. The
node's own front answers the same questions for its whole install — the
union of its engines' loaded models, the sum of their VRAM, every capability
any of them has — on ``/cluster/state``, guarded by an API token of the node.
So the head needs a URL, a token, and a way to trust the node's certificate;
this module turns a ``cluster.nodes`` entry into exactly that.
"""

import ssl
from dataclasses import dataclass, field
from typing import List, Optional

import httpx


@dataclass
class NodeSpec:
    name: str
    url: str
    api_key: str = ""
    verify: str = "system"          # system | off | pem
    ca_pem: str = ""                # the node's certificate (or CA) when verify=pem
    capabilities: List[str] = field(default_factory=list)   # blank = as reported
    enabled: bool = True
    timeout_s: float = 4.0

    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def verify_arg(self):
        """What httpx's ``verify=`` takes for this node."""
        if self.verify == "off":
            return False
        if self.verify == "pem" and self.ca_pem.strip():
            ctx = ssl.create_default_context()
            ctx.load_verify_locations(cadata=self.ca_pem)
            return ctx
        return True


def parse_nodes(raw) -> List[NodeSpec]:
    """``cluster.nodes`` → specs, skipping entries that cannot be used."""
    out = []
    seen = set()
    for i, n in enumerate(raw or []):
        if not isinstance(n, dict):
            continue
        url = str(n.get("url") or "").strip().rstrip("/")
        if not url:
            continue
        name = str(n.get("name") or "").strip() or f"node{i + 1}"
        if name.lower() in seen:
            continue
        seen.add(name.lower())
        caps = n.get("capabilities")
        if isinstance(caps, str):
            caps = [c.strip() for c in caps.split(",") if c.strip()]
        spec = NodeSpec(
            name=name, url=url,
            api_key=str(n.get("api_key") or "").strip(),
            verify=str(n.get("verify") or "system").strip().lower() or "system",
            ca_pem=str(n.get("ca_pem") or ""),
            capabilities=[str(c).strip().lower() for c in (caps or []) if str(c).strip()],
            enabled=bool(n.get("enabled", True)),
        )
        try:
            spec.timeout_s = float(n.get("timeout_s") or spec.timeout_s)
        except (TypeError, ValueError):
            pass
        out.append(spec)
    return out


def enabled_nodes(cluster_cfg) -> List[NodeSpec]:
    if cluster_cfg is None or not getattr(cluster_cfg, "enabled", False):
        return []
    specs = parse_nodes(getattr(cluster_cfg, "nodes", None))
    t = float(getattr(cluster_cfg, "poll_timeout_s", 4.0) or 4.0)
    for s in specs:
        s.timeout_s = t
    return [s for s in specs if s.enabled]


def sync_client(spec: NodeSpec) -> httpx.Client:
    """A short-timeout client for state polls and control pushes."""
    return httpx.Client(timeout=spec.timeout_s, headers=spec.headers(),
                        verify=spec.verify_arg())


def async_clients(spec: NodeSpec):
    """(short, long) async clients the front proxies through — the same two
    shapes it keeps for local engines, but carrying this node's token and
    trust settings."""
    short = httpx.AsyncClient(timeout=spec.timeout_s, headers=spec.headers(),
                              verify=spec.verify_arg())
    long = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=None, write=None, pool=None),
        headers=spec.headers(), verify=spec.verify_arg())
    return short, long


# --------------------------------------------------------------- node side
def aggregate_state(registry, rpc_endpoints: Optional[list] = None,
                    node_name: str = "") -> dict:
    """What this install looks like as ONE engine, from its front's registry.

    The head merges this into its own registry: loaded models across every
    engine here, VRAM summed, tasks concatenated, capabilities unioned — so a
    node with an NVIDIA and a Radeon engine advertises both ``transformers``
    and ``gguf`` and the head sends it either kind of model; the node's own
    front then routes to the right card, as it always did.
    """
    loaded, info, tasks, caps = [], [], [], set()
    total = free = 0.0
    any_vram = False
    healthy = False
    cooling = None
    inflight = 0
    engines = []
    for e in registry.all():
        if getattr(e, "role", "engine") == "system" or getattr(e, "remote", False):
            continue      # a node reports ITS cards, never nodes of its own
        engines.append({"name": e.name, "backend": e.backend, "healthy": e.healthy,
                        "vram": e.vram, "loaded": sorted(e.loaded_models)})
        if not e.healthy:
            continue
        healthy = True
        caps |= set(e.capabilities or ())
        loaded.extend(e.loaded_models)
        info.extend(e.loaded_info or [])
        tasks.extend(e.tasks or [])
        inflight += int(getattr(e, "inflight", 0) or 0)
        if e.vram:
            any_vram = True
            try:
                total += float(e.vram.get("total") or 0.0)
                free += float(e.vram.get("free") or 0.0)
            except (TypeError, ValueError):
                pass
        if e.cooling and cooling is None:
            cooling = e.cooling
    # A node never rents pods on the head's behalf and never nests nodes.
    caps.discard("runpod")
    return {
        "node": node_name,
        "healthy": healthy,
        "loaded_models": sorted(set(loaded)),
        "loaded_info": info,
        "tasks": tasks,
        "vram": ({"total": round(total, 2), "free": round(free, 2),
                  "used": round(max(0.0, total - free), 2)} if any_vram else None),
        "cooling": cooling,
        "capabilities": sorted(caps),
        "inflight": inflight,
        "engines": engines,
        "rpc_servers": list(rpc_endpoints or []),
    }


#: Fields that describe THIS head's placement, not the model: a node must not
#: pin the model to an engine of the head, rent pods for it, or spread it over
#: cards the head sees. ``rpc_servers`` stays — the node loads the weights and
#: is the one that spreads them.
_HEAD_ONLY = ("engine", "engine_fallback", "runpod", "host", "gpu_split",
              "tensor_split", "split_strategy", "split_secondary_cap_gb",
              "node_path", "node_paths", "cluster_source", "placement")


def portable_entry(entry: dict) -> dict:
    """A models.json entry as the head sends it: the entry itself, plus where
    else the weights can be fetched from when the path means nothing there."""
    e = dict(entry)
    try:
        from codai.api.runpod_worker import parse_model_runpod, resolve_model_source
        block = e.get("runpod") if isinstance(e.get("runpod"), dict) else {}
        kind, value = resolve_model_source(e, parse_model_runpod(block))
        if kind in ("hf", "url") and value:
            e["cluster_source"] = value
    except Exception:
        pass
    return e


def localize_entries(entries: list, node_name: str = "") -> tuple:
    """(payload for /v1/models/register, refused) on the NODE: pick the path
    this machine can use, drop the head's placement fields."""
    import os
    payload, refused = [], []
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        path = str(entry.get("path") or "").strip()
        paths = entry.get("node_paths") if isinstance(entry.get("node_paths"), dict) else {}
        override = str(paths.get(node_name) or entry.get("node_path") or "").strip()
        src = str(entry.get("cluster_source") or "").strip()
        chosen = ""
        for cand in (os.path.expanduser(override) if override else "", path):
            if cand and (not cand.startswith("/") or os.path.exists(cand)):
                chosen = cand
                break
            if cand and cand.startswith("/") and os.path.exists(os.path.expanduser(cand)):
                chosen = os.path.expanduser(cand)
                break
        if not chosen and src:
            chosen = src
        if not chosen:
            refused.append(f"{entry.get('alias') or path}: {path!r} does not exist on "
                           f"this node and no HuggingFace id/URL is known — set "
                           f"'Path on the node' on the head's model page")
            continue
        drop = set(_HEAD_ONLY)
        if entry.get("rpc_servers"):
            # The split describes the node's devices plus its RPC peers, in the
            # order the node's llama.cpp sees them — it belongs to the node.
            drop -= {"gpu_split", "tensor_split", "split_strategy", "split_secondary_cap_gb"}
        out = {k: v for k, v in entry.items() if k not in drop}
        out["path"] = chosen
        payload.append(out)
    return payload, refused
