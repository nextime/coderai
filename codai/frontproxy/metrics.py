# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Prometheus exposition for the front — ``GET /metrics`` — and the per-key
usage tables behind it.

No client library: the text format is a few lines per sample and writing it
by hand keeps the front's dependency list where it is. Counters live for
the life of the front process (Prometheus expects exactly that and takes
``rate()`` over them); everything else is read from the registry, the
supervisor and the RunPod ledger at scrape time.

What is exported, all prefixed ``coderai_``:

* ``requests_total{key,model,kind,status,engine}`` and
  ``request_seconds_sum/_count`` with the same labels — every inference the
  front completed, attributed to the API key that made it (its name on the
  Tokens page; ``session`` for a browser; ``anonymous`` when open);
* ``engine_up{engine,backend,remote}``, ``engine_inflight``,
  ``engine_vram_free_bytes`` / ``_total_bytes``, ``engine_loaded_models``;
* ``node_up{node}``, ``node_recoveries_total``;
* ``rpc_server_up{endpoint}``; ``discovery_peers`` / ``discovery_members``;
* ``runpod_spend_usd{model,period}`` (trailing hour/day/week/month), ``runpod_pods``;
* ``info{version}`` = 1.
"""

import threading
import time
from collections import defaultdict


def _esc(v) -> str:
    return str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _labels(d: dict) -> str:
    if not d:
        return ""
    return "{" + ",".join(f'{k}="{_esc(v)}"' for k, v in d.items()) + "}"


class Metrics:
    """Counters the front feeds from ``_record_activity``."""

    def __init__(self):
        self._lock = threading.Lock()
        self._n = defaultdict(int)         # (key, model, kind, status, engine) -> count
        self._t = defaultdict(float)       # same key -> seconds
        self.started = time.time()

    def observe(self, key: str, model: str, kind: str, status: int, engine: str,
                seconds: float) -> None:
        k = (key or "anonymous", model or "", kind or "text", int(status or 0), engine or "")
        with self._lock:
            self._n[k] += 1
            self._t[k] += max(0.0, float(seconds or 0.0))

    # ------------------------------------------------------------ tables
    def usage_tables(self) -> dict:
        """Per key, per model and per kind — for the Cluster page."""
        with self._lock:
            items = [(k, self._n[k], self._t[k]) for k in self._n]
        by_key, by_model = {}, {}
        for (key, model, kind, status, engine), n, t in items:
            for table, name in ((by_key, key), (by_model, model or "(none)")):
                row = table.setdefault(name, {"requests": 0, "errors": 0, "seconds": 0.0,
                                              "kinds": {}, "engines": {}})
                row["requests"] += n
                row["seconds"] += t
                if status >= 400 or status == 0:
                    row["errors"] += n
                row["kinds"][kind] = row["kinds"].get(kind, 0) + n
                if engine:
                    row["engines"][engine] = row["engines"].get(engine, 0) + n
        for table in (by_key, by_model):
            for row in table.values():
                row["seconds"] = round(row["seconds"], 1)
        return {"since": self.started, "by_key": by_key, "by_model": by_model}

    # ---------------------------------------------------------- exposition
    def counter_lines(self) -> list:
        with self._lock:
            items = [(k, self._n[k], self._t[k]) for k in sorted(self._n)]
        out = ["# HELP coderai_requests_total Inference requests completed by the front.",
               "# TYPE coderai_requests_total counter"]
        for (key, model, kind, status, engine), n, _ in items:
            out.append("coderai_requests_total" + _labels(
                {"key": key, "model": model, "kind": kind, "status": status, "engine": engine})
                + f" {n}")
        out += ["# HELP coderai_request_seconds Wall time of completed inference requests.",
                "# TYPE coderai_request_seconds summary"]
        for (key, model, kind, status, engine), n, t in items:
            lab = {"key": key, "model": model, "kind": kind, "status": status, "engine": engine}
            out.append("coderai_request_seconds_sum" + _labels(lab) + f" {t:.3f}")
            out.append("coderai_request_seconds_count" + _labels(lab) + f" {n}")
        return out


def render(front) -> str:
    """The whole /metrics page."""
    from codai import __version__
    lines = ["# HELP coderai_info CoderAI version.", "# TYPE coderai_info gauge",
             "coderai_info" + _labels({"version": __version__}) + " 1"]
    m = getattr(front, "metrics", None)
    if m is not None:
        lines += m.counter_lines()
        lines += ["# HELP coderai_front_start_time_seconds When this front started.",
                  "# TYPE coderai_front_start_time_seconds gauge",
                  f"coderai_front_start_time_seconds {m.started:.0f}"]

    # Engines and nodes, from the registry.
    up, infl, vfree, vtot, loaded, nup = [], [], [], [], [], []
    for e in front.registry.all():
        if getattr(e, "role", "engine") == "system":
            continue
        remote = bool(getattr(e, "remote", False))
        lab = {"engine": e.name, "backend": e.backend or "", "remote": "1" if remote else "0"}
        up.append("coderai_engine_up" + _labels(lab) + f" {1 if e.healthy else 0}")
        infl.append("coderai_engine_inflight" + _labels(lab) + f" {int(getattr(e, 'inflight', 0) or 0)}")
        v = e.vram or {}
        if v:
            vfree.append("coderai_engine_vram_free_bytes" + _labels(lab)
                         + f" {float(v.get('free', 0) or 0) * (1 << 30):.0f}")
            vtot.append("coderai_engine_vram_total_bytes" + _labels(lab)
                        + f" {float(v.get('total', 0) or 0) * (1 << 30):.0f}")
        loaded.append("coderai_engine_loaded_models" + _labels(lab) + f" {len(e.loaded_models)}")
        if remote:
            nup.append("coderai_node_up" + _labels({"node": e.name}) + f" {1 if e.healthy else 0}")
    for help_, typ, rows in (
            ("Engine (or node) answering its health poll.", "gauge", up),
            ("Requests currently proxied to the engine.", "gauge", infl),
            ("Free VRAM the engine reports.", "gauge", vfree),
            ("Total VRAM the engine reports.", "gauge", vtot),
            ("Models resident on the engine.", "gauge", loaded),
            ("Cluster node reachable.", "gauge", nup)):
        if rows:
            name = rows[0].split("{", 1)[0]
            lines += [f"# HELP {name} {help_}", f"# TYPE {name} {typ}"] + rows

    sup = getattr(front, "supervisor", None)
    lines += ["# HELP coderai_node_recoveries_total Nodes that came back and had their models re-pushed.",
              "# TYPE coderai_node_recoveries_total counter",
              f"coderai_node_recoveries_total {int(getattr(sup, '_node_recoveries', 0) or 0)}"]

    # RPC servers this machine runs.
    mgr = getattr(sup, "rpc_manager", None) if sup else None
    if mgr is not None:
        try:
            rows = []
            for ep in mgr.endpoints():
                rows.append("coderai_rpc_server_up" + _labels({"endpoint": ep.get("endpoint", "")})
                            + f" {1 if ep.get('listening') else 0}")
            if rows:
                lines += ["# HELP coderai_rpc_server_up llama.cpp rpc-server process running.",
                          "# TYPE coderai_rpc_server_up gauge"] + rows
        except Exception:
            pass

    # Discovery.
    disc = getattr(sup, "discovery", None) if sup else None
    if disc is not None:
        try:
            peers = disc.peers()
            lines += ["# HELP coderai_discovery_peers coderai installs seen on the LAN.",
                      "# TYPE coderai_discovery_peers gauge",
                      f"coderai_discovery_peers {len(peers)}",
                      "# HELP coderai_discovery_members Peers holding this cluster's token.",
                      "# TYPE coderai_discovery_members gauge",
                      f"coderai_discovery_members {sum(1 for p in peers if p.get('member'))}"]
        except Exception:
            pass

    # RunPod spend.
    try:
        from codai.api import runpod_ledger, runpod_worker
        tot = runpod_ledger.totals()
        rows = []
        for model, rec in (tot.get("per_model") or {}).items():
            if not isinstance(rec, dict):
                continue
            for period, usd in rec.items():
                try:
                    rows.append("coderai_runpod_spend_usd" + _labels({"model": model, "period": period})
                                + f" {float(usd or 0):.4f}")
                except (TypeError, ValueError):
                    continue
        for period, usd in (tot.get("global") or {}).items():
            try:
                rows.append("coderai_runpod_spend_usd" + _labels({"model": "", "period": period})
                            + f" {float(usd or 0):.4f}")
            except (TypeError, ValueError):
                continue
        if rows:
            lines += ["# HELP coderai_runpod_spend_usd Money spent on rented pods in the trailing period.",
                      "# TYPE coderai_runpod_spend_usd gauge"] + rows
        try:
            npods = len(runpod_worker.pods_status() or [])
        except Exception:
            npods = None
        if npods is not None:
            lines += ["# HELP coderai_runpod_pods Rented pods currently running.",
                      "# TYPE coderai_runpod_pods gauge", f"coderai_runpod_pods {npods}"]
    except Exception:
        pass

    # Queue / active requests as the overview counts them.
    try:
        active = sum(int(getattr(e, "inflight", 0) or 0) for e in front.registry.all())
        lines += ["# HELP coderai_requests_active Requests in flight through the front.",
                  "# TYPE coderai_requests_active gauge", f"coderai_requests_active {active}"]
    except Exception:
        pass
    return "\n".join(lines) + "\n"
