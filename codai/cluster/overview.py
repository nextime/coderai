# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Everything this front can send work to, on one page.

Local engines, cluster nodes, this machine's and the nodes' RPC servers, the
hosts named by ``backend: host`` models, the capability remotes, the pods —
each with whether it answers right now. Built in a worker thread from the
front's registry and a few short probes; probes are cached so the page can
refresh every few seconds without hammering anything.
"""

import json
import os
import threading
import time

_probe_cache = {}          # url -> (expires, ok, detail)
_probe_lock = threading.Lock()
_PROBE_TTL = 20.0


def _probe(url: str, path: str = "/healthz", api_key: str = "",
           timeout: float = 3.0) -> tuple:
    """(ok, detail) for a base URL, cached for a while."""
    key = (url, path, bool(api_key))
    now = time.time()
    with _probe_lock:
        hit = _probe_cache.get(key)
        if hit and hit[0] > now:
            return hit[1], hit[2]
    ok, detail = False, ""
    try:
        from codai.api import pod_http
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        t0 = time.time()
        r = pod_http.get(url.rstrip("/") + path, headers=headers, timeout=timeout)
        ok = r.status_code == 200
        detail = f"HTTP {r.status_code} in {(time.time() - t0) * 1000:.0f} ms"
    except Exception as exc:
        detail = str(exc)[:160]
    with _probe_lock:
        _probe_cache[key] = (now + _PROBE_TTL, ok, detail)
    return ok, detail


def _models_data(front) -> dict:
    path = getattr(front, "_models_path", None)
    if not path:
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _host_targets(front) -> list:
    """Every host named by a ``backend: host`` model, probed."""
    out = []
    data = _models_data(front)
    for cat, lst in data.items():
        if not isinstance(lst, list):
            continue
        for m in lst:
            if not isinstance(m, dict) or str(m.get("backend") or "").lower() != "host":
                continue
            block = m.get("host") if isinstance(m.get("host"), dict) else {}
            try:
                from codai.api.host_worker import parse_hosts
                slots = parse_hosts(block)
            except Exception:
                slots = []
            name = m.get("alias") or os.path.basename(str(m.get("path") or "")) or "?"
            for h in slots:
                ok, detail = _probe(h.url, h.health_path, h.api_key)
                out.append({"model": name, "url": h.url, "on_demand": bool(h.start_cmd),
                            "ok": ok, "detail": detail})
    return out


def _remote_targets(front) -> list:
    out = []
    try:
        from codai.api.remote_gateway import capability_endpoint_lists
        eps = capability_endpoint_lists()
    except Exception:
        eps = {}
    rem = getattr(front.config, "remotes", None)
    key = (getattr(rem, "api_key", "") or "") if rem else ""
    for cap, urls in sorted(eps.items()):
        for u in urls:
            if u.lower() == "runpod":
                out.append({"capability": cap, "url": "runpod", "ok": None,
                            "detail": "served by a pod pool"})
                continue
            ok, detail = _probe(u, "/healthz", key)
            out.append({"capability": cap, "url": u, "ok": ok, "detail": detail})
    return out


def _pods() -> list:
    try:
        from codai.api.runpod_worker import pods_status
        return pods_status()
    except Exception:
        return []


def _rpc_reachable(endpoint: str) -> bool:
    import socket
    host, _, port = endpoint.rpartition(":")
    try:
        with socket.create_connection((host, int(port)), timeout=1.5):
            return True
    except Exception:
        return False


def _rpc_targets(front, local_rpc: list) -> list:
    """This machine's RPC servers and every node's, one row each."""
    out = []
    for r in local_rpc or []:
        d = dict(r)
        d["node"] = "(this machine)"
        d["reachable"] = bool(d.get("listening"))
        out.append(d)
    for e in front.registry.remotes():
        for r in (e.rpc_servers or []):
            d = dict(r)
            d["node"] = e.name
            ep = d.get("endpoint") or ""
            d["reachable"] = _rpc_reachable(ep) if ep else False
            out.append(d)
    return out


def build_overview(front, local_rpc: list) -> dict:
    engines = front.engines_list()
    local = [e for e in engines if not e.get("remote")]
    nodes = [e for e in engines if e.get("remote")]
    ccfg = getattr(front.config, "cluster", None)
    return {
        "generated_at": time.time(),
        "cluster_enabled": bool(getattr(ccfg, "enabled", False)) if ccfg else False,
        "node_name": (getattr(ccfg, "node_name", "") or "") if ccfg else "",
        "serve": bool(getattr(ccfg, "serve", True)) if ccfg else True,
        "engines": local,
        "nodes": nodes,
        "rpc_servers": _rpc_targets(front, local_rpc),
        "hosts": _host_targets(front),
        "remotes": _remote_targets(front),
        "pods": _pods(),
    }
