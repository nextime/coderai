#!/usr/bin/env python3
# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
# GPLv3 - see the project LICENSE.

"""Regenerate the ``## Endpoint Index`` section of ``CODERAI_API_DOCUMENTATION.md``.

The index used to be maintained by hand, and drifted: 135 of the 257 routes the
server defines were missing from it, including whole subsystems (``/v1/ocr``,
``/v1/chunk``, ``/v1/rerank``, ``/v1/moderation``, speaker recognition, video
composition, the client RunPod API). A hand-kept list of a surface this size is
wrong the week after it is written, so it is generated instead.

The purpose column comes from each route's own ``summary=`` where it has one.
Routes registered with ``include_in_schema=False`` (service probes, the cluster
mesh, the front↔engine control plane) carry no summary by design — they are not
in ``/openapi.json`` — so their one-line descriptions live in :data:`OVERRIDES`
below, which is the only hand-written part of the index.

Usage::

    python3 tools/gen_endpoint_index.py            # print the section
    python3 tools/gen_endpoint_index.py --write     # splice it into the doc
    python3 tools/gen_endpoint_index.py --check     # exit 1 if the doc is stale
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOC = ROOT / "CODERAI_API_DOCUMENTATION.md"
PKG = ROOT / "codai"

HEADING = "## Endpoint Index"

#: One line per route that has no ``summary=`` in the code, because it is
#: deliberately absent from the OpenAPI schema. Keyed ``"METHOD /path"``.
OVERRIDES = {
    # Service and operational
    "GET /health": "vLLM-compatible liveness alias (clients hard-code this path)",
    "GET /healthz": "Cheap liveness probe touching no model state; the supervisor polls it",
    "GET /v1/health": "Liveness alias under the `/v1` prefix",
    "GET /metrics": "Prometheus exposition (open when `server.metrics_public`)",
    "GET /boot": "What this process did on the way up, with timings and the container phases",
    "GET /logs": "This process's own output from a cursor — the only way to read a pod's log",
    "GET /favicon.ico": "Favicon",
    "GET /chat": "Chat / Studio UI",
    "GET /login": "Login page",
    # Admin HTML
    "GET /admin": "Dashboard",
    "GET /admin/models": "Model management page",
    "GET /admin/tokens": "API token page",
    "GET /admin/users": "User page",
    "GET /admin/settings": "Settings page",
    "GET /admin/archive": "Generation archive page",
    "GET /admin/tasks": "Tasks page (running jobs, engine tiles, logs)",
    "GET /admin/cluster": "Cluster page (nodes, pools, reachability)",
    "GET /admin/runpod": "RunPod page (pools, pods, spend)",
    "GET /admin/change-password": "Password change page",
    # Admin API
    "GET /admin/api/status": "System, model, VRAM, queue and recent-activity status",
    "GET /admin/api/system-stats": "Host CPU / RAM / disk / temperature snapshot",
    "GET /admin/api/gpu-stats": "Per-GPU utilisation, VRAM and temperature",
    "GET /admin/api/engines": "Engine nodes: name, health, VRAM, loaded models (front's own registry)",
    "POST /admin/api/engines/{eid}/restart": "Restart one engine subprocess",
    "GET /admin/api/usage": "Requests, errors and time per API key / model / kind",
    "POST /admin/api/batch": "Run several admin API calls in one request",
    "GET /admin/api/cluster": "Everything this front can send work to, with reachability",
    "GET /admin/api/cluster/logs": "Tail a cluster node's log",
    "GET /admin/api/tasks": "Poll running and finished tasks across engines",
    "DELETE /admin/api/tasks/{task_id}": "Remove a finished task from the list",
    "GET /admin/api/users": "List users",
    "POST /admin/api/users": "Create a user",
    "DELETE /admin/api/users/{user_id}": "Delete a user",
    "GET /admin/api/tokens": "List API tokens",
    "POST /admin/api/tokens": "Create an API token",
    "PATCH /admin/api/tokens/{token_id}": "Rename / re-scope a token",
    "DELETE /admin/api/tokens/{token_id}": "Delete a token",
    "GET /admin/api/settings": "Current server settings",
    "GET /admin/api/archive": "List archived generations",
    "GET /admin/api/archive/{gen_id}": "One archived generation",
    "DELETE /admin/api/archive/{gen_id}": "Delete an archived generation",
    "GET /admin/api/archive/{gen_id}/files/{filename}": "Fetch one file from an archived generation",
    "GET /admin/api/archive-settings": "Archive retention settings",
    # Cluster mesh
    "GET /cluster/state": "This node's aggregate state (models, engines, RPC peers)",
    "GET /cluster/logs": "This node's log, for the head's cluster page",
    "GET /cluster/rpc-servers": "The llama.cpp RPC endpoints this node exposes",
    "POST /cluster/model-load": "The head tells this node to load a model",
    "POST /cluster/model-unload": "The head tells this node to unload a model",
    "POST /cluster/reload-config": "The head tells this node which models it owns",
    # Front ↔ engine control plane
    "GET /internal/engine-state": "This engine's loaded models, VRAM and queue depth",
    "POST /internal/evict-vram": "A co-located sibling asks this engine to release VRAM",
    "POST /internal/gpu-reserve": "A sibling reserves the card for an exclusive op",
    "POST /internal/gpu-release": "Lift a sibling's GPU reservation",
    "POST /internal/thermal-pause": "Front-driven thermal pause at the next safe point",
    "POST /internal/thermal-resume": "Lift the thermal pause once the hardware cooled",
    "POST /internal/reload-config": "Re-read `config.json` / `models.json` after an admin edit",
}

#: Output order. The first predicate that matches wins, so the catch-all
#: ("Service and operational") must stay last even though it prints second.
GROUPS = [
    ("Admin API", lambda p: p.startswith("/admin/api/")),
    ("Admin HTML", lambda p: p.startswith("/admin")),
    ("Cluster (node \u2194 head)", lambda p: p.startswith("/cluster/")),
    ("Internal (front \u2194 engine)", lambda p: p.startswith("/internal/")),
    ("Public `/v1`", lambda p: p.startswith("/v1/")),
    ("Service and operational", lambda p: True),
]

#: The order the groups are PRINTED in, which is the order a reader wants.
PRINT_ORDER = ["Public `/v1`", "Service and operational", "Admin HTML",
               "Admin API", "Cluster (\u2194 head)".replace("(", "(node "),
               "Internal (\u2194 engine)".replace("(", "(front ")]

_DEC = re.compile(r"@(?:app|router)\.(get|post|put|patch|delete)\(")
_METHOD_ORDER = {"GET": 0, "POST": 1, "PUT": 2, "PATCH": 3, "DELETE": 4}


def _read_string(buf: str, i: int) -> tuple[str, int]:
    """Read the Python string literal starting at ``buf[i]``; return it and the
    index just past its closing quote. Honours escapes, so an apostrophe inside
    a double-quoted summary does not end it early."""
    quote = buf[i]
    out: list[str] = []
    j = i + 1
    while j < len(buf):
        c = buf[j]
        if c == "\\":
            out.append(buf[j + 1:j + 2])
            j += 2
            continue
        if c == quote:
            return "".join(out), j + 1
        out.append(c)
        j += 1
    return "".join(out), j


def _summary(buf: str) -> str:
    """The ``summary=`` value, concatenating implicitly joined string pieces."""
    m = re.search(r"summary\s*=\s*", buf)
    if not m:
        return ""
    i = m.end()
    parts: list[str] = []
    while i < len(buf):
        c = buf[i]
        if c in "'\"":
            value, i = _read_string(buf, i)
            parts.append(value)
            continue
        if c in " \t\r\n(":
            i += 1
            continue
        break
    return "".join(parts)


def collect() -> "collections.OrderedDict":
    """Every ``(METHOD, path)`` the package registers, with summary and sources."""
    routes: collections.OrderedDict = collections.OrderedDict()
    for path in sorted(PKG.rglob("*.py")):
        lines = path.read_text().splitlines()
        i = 0
        while i < len(lines):
            m = _DEC.search(lines[i])
            if not m:
                i += 1
                continue
            # Gather the whole decorator call, which may span lines.
            buf = lines[i][m.end() - 1:]
            depth = buf.count("(") - buf.count(")")
            j = i
            while depth > 0 and j + 1 < len(lines):
                j += 1
                buf += "\n" + lines[j]
                depth += lines[j].count("(") - lines[j].count(")")
            # Skip whitespace (and a newline) between "(" and the path, so a
            # decorator whose first argument is on the next line is not missed —
            # a route the index silently omitted would be the exact drift this
            # generator exists to prevent.
            k = 1
            while k < len(buf) and buf[k] in " \t\r\n":
                k += 1
            if buf[:1] == "(" and buf[k:k + 1] in "'\"":
                route, _ = _read_string(buf, k)
                entry = routes.setdefault((m.group(1).upper(), route),
                                          {"summary": "", "sources": []})
                summary = _summary(buf)
                if summary and not entry["summary"]:
                    entry["summary"] = summary
                rel = str(path.relative_to(ROOT))
                if rel not in entry["sources"]:
                    entry["sources"].append(rel)
            i = j + 1
    return routes


def _group_of(route: str) -> str:
    for name, match in GROUPS:
        if match(route):
            return name
    return GROUPS[-1][0]


def _source(rel: str) -> str:
    return rel.replace("codai/", "").removesuffix(".py")


def render(routes) -> str:
    grouped = collections.defaultdict(list)
    for (method, route), entry in routes.items():
        grouped[_group_of(route)].append((method, route, entry))
    missing = [f"{m} {r}" for (m, r), e in routes.items()
               if not e["summary"] and f"{m} {r}" not in OVERRIDES]
    out = [
        HEADING,
        "",
        f"Every HTTP route the server defines — **{len(routes)}** of them — generated from the",
        "decorators in `codai/` by [`tools/gen_endpoint_index.py`](tools/gen_endpoint_index.py).",
        "`Source` names the module(s) that own the route: one owned by `frontproxy/*` is served",
        "by the torch-free front, one owned by `api/*` by an engine process, and a route listed",
        "in both is proxied or aggregated by the front (see",
        "[`docs/frontend-engine-split.md`](docs/frontend-engine-split.md)).",
        "",
        "Routes the code registers with `include_in_schema=False` are **not** in",
        "`/openapi.json` — the service, cluster and internal groups below are mostly those.",
        "They are still part of the HTTP surface, which is why they are listed here.",
        "",
    ]
    unknown = [n for n, _ in GROUPS if n not in PRINT_ORDER]
    if unknown:                      # a new group with nowhere to print it
        raise SystemExit(f"PRINT_ORDER is missing {unknown}")
    for name in PRINT_ORDER:
        rows = grouped.get(name) or []
        if not rows:
            continue
        out += [f"### {name}", "", "| Method | Path | Purpose | Source |", "|---|---|---|---|"]
        for method, route, entry in sorted(
                rows, key=lambda r: (r[1], _METHOD_ORDER.get(r[0], 9))):
            purpose = entry["summary"] or OVERRIDES.get(f"{method} {route}") or "—"
            sources = ", ".join(f"`{_source(s)}`" for s in entry["sources"])
            out.append(f"| `{method}` | `{route}` | {purpose} | {sources} |")
        out.append("")
    if missing:
        out += ["> Undescribed routes (add them to `OVERRIDES` in the generator): "
                + ", ".join(f"`{x}`" for x in sorted(missing)), ""]
    return "\n".join(out).rstrip() + "\n"


def splice(doc_text: str, section: str) -> str:
    start = doc_text.index(HEADING)
    # The index is the last section in the document; stop at the next H2 if any.
    nxt = doc_text.find("\n## ", start + len(HEADING))
    tail = doc_text[nxt + 1:] if nxt != -1 else ""
    return doc_text[:start] + section + ("\n" + tail if tail else "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="splice into the doc")
    ap.add_argument("--check", action="store_true", help="exit 1 if the doc is stale")
    args = ap.parse_args()

    section = render(collect())
    if not (args.write or args.check):
        sys.stdout.write(section)
        return 0
    text = DOC.read_text()
    new = splice(text, section)
    if args.check:
        if new != text:
            print(f"{DOC.name}: the endpoint index is stale — run "
                  f"`python3 tools/gen_endpoint_index.py --write`", file=sys.stderr)
            return 1
        print(f"{DOC.name}: endpoint index up to date")
        return 0
    if new != text:
        DOC.write_text(new)
        print(f"{DOC.name}: endpoint index rewritten")
    else:
        print(f"{DOC.name}: endpoint index already up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
