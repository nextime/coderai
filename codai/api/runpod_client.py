# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Thin RunPod API client (GPU catalog + Pod lifecycle).

coderai talks to RunPod over its GraphQL API (https://graphql-spec.runpod.io) — the most
stable surface for GPU pricing and pod create/query/stop/terminate. Serverless invocation
uses the separate ``/v2/{endpoint}`` REST base and is handled in the worker, not here.

This module is pure transport: no scaling/policy logic (that lives in
:mod:`codai.api.runpod_worker`). The API key comes from the caller (RunpodConfig) and is
NEVER logged — only masked in error strings.
"""

import os
import threading
import time
from typing import Optional


#: Request field that attaches a GLOBAL volume (RunPod's region-independent
#: storage, beta since Sept 2026). RunPod documents global volumes as console
#: only — no field appears in the Pod API reference — and the v1 REST validator
#: accepts unknown body keys silently, so this cannot be confirmed by probing:
#: a wrong name yields a pod that boots with no volume and re-downloads its
#: weights, which is expensive and invisible. Hence the override, so the name
#: can be corrected without a release once RunPod publishes it. VERIFY on the
#: first pod that the weights are actually at the mount before relying on it.
GLOBAL_VOLUME_FIELD = os.environ.get("CODERAI_RUNPOD_GLOBAL_VOLUME_FIELD",
                                     "globalNetworkVolumeId")


class RunpodError(RuntimeError):
    """A RunPod API call failed (network, auth, or GraphQL error)."""


class RunpodTransient(RunpodError):
    """The API could not be ASKED — it says nothing about the pod.

    A 503 from RunPod's control plane, a timeout, a dropped connection: the
    question never got an answer. That is a different fact from "the pod is
    bad", and conflating the two cost a live pod once already. A boot wait got
    HTTP 503 while a 9.6 GB image was still being pulled, read it as a boot
    failure, tore the pod down through the SAME unavailable API — which also
    failed — and left an untracked GPU billing until someone looked at the
    account. The request that started it answered 502 while the machine it had
    paid for was still coming up.

    A subclass, so every existing ``except RunpodError`` keeps working; callers
    that can afford to wait catch this one first and retry.
    """


#: Said once per process, not once per pod: the scaler would otherwise repeat it
#: every boot.
_GLOBAL_VOLUME_WARNED = False

#: The GPU catalogue, shared and briefly cached: (expiry, rows).
#:
#: Every `_rank_gpus` asked for it, and `_rank_gpus` runs once per pod being
#: provisioned. With 32 pods booting in parallel that is 32 identical catalogue
#: queries and 32 large JSON bodies parsed, all within a few seconds, and the
#: parsing is Python -- so it holds the GIL and starves whatever else the
#: process is trying to do. On a live run that was the event loop carrying the
#: page forwards: 1,754 pages in flight and not one socket opened to a pod.
#:
#: The catalogue is a price list. It does not change inside a boot storm.
_GPU_CATALOG: tuple = (0.0, None)
_GPU_CATALOG_LOCK = __import__("threading").Lock()
GPU_CATALOG_TTL_S = 60.0


def _global_field_refused(text: str) -> bool:
    """Whether a create failed *because* of the global-volume field.

    RunPod words it two ways — "not in input schema" (v1) and "additional
    properties … not allowed" (v2) — and both name the field.
    """
    t = (text or "")
    return GLOBAL_VOLUME_FIELD in t and (
        "not in input schema" in t or "not allowed" in t or "Extra input keys" in t)


def _field_refused_gql(text: str, field: str) -> bool:
    """GraphQL's wording for an input field its schema does not define."""
    t = (text or "")
    return bool(field) and field in t and (
        "is not defined by type" in t or "invalid value" in t or "Unknown" in t)


def _global_field_refused_gql(text: str) -> bool:
    """GraphQL's wording for a global-volume field its schema does not define."""
    return _field_refused_gql(text, GLOBAL_VOLUME_FIELD)


#: Ask the host for a public IP whenever the pod is reached directly.
#:
#: Direct TCP connects to the pod's OWN ip:port, so the machine has to be able
#: to publish one. Community-cloud hosts often cannot, and RunPod does not say
#: so at create time: it takes the pod, publishes only its own agent port
#: (``19123/http``) on a 100.64/10 carrier-grade-NAT address, and never
#: publishes the declared port at all. The result is a pod that bills from the
#: moment it is created and can never be reached, until the boot wait gives up
#: and the next candidate is tried -- which draws from the same hosts.
#:
#: Measured 2026-11-08 on a pool restricted to five cards: ten pods rented in
#: half an hour, nine terminated, $1.21 billed for nothing, every one of them
#: "RunPod published 19123/http but NOT the declared port 8000". Asking for the
#: capability turns that into a placement RunPod refuses up front.
#:
#: Secure cloud always has a public IP, so this only ever narrows community
#: offers -- which is exactly the set that cannot serve a direct-TCP pod.
PUBLIC_IP_FIELD = "supportPublicIp"


def _warn_global_volume_unsupported(volume_id: str) -> None:
    global _GLOBAL_VOLUME_WARNED
    if _GLOBAL_VOLUME_WARNED:
        return
    _GLOBAL_VOLUME_WARNED = True
    print(f"[runpod] global volume {volume_id!r} is configured but RunPod's API "
          f"rejects {GLOBAL_VOLUME_FIELD!r}: global volumes are console-only for "
          "now (checked against v1 REST, GraphQL, v2 pods and templates). Pods "
          "will boot WITHOUT it and download their weights each time. Set "
          "CODERAI_RUNPOD_GLOBAL_VOLUME_FIELD once RunPod publishes the field "
          "name.", flush=True)


def _mask(key: str) -> str:
    if not key:
        return "<none>"
    return (key[:4] + "…" + key[-4:]) if len(key) > 10 else "****"


# A pod's public HTTP proxy hostname is derived from its id + the container port.
# That hostname is fronted by Cloudflare: TLS for free, but a 100 s idle limit
# per request and a request-body cap. See direct_tcp_url for the other way in.
def pod_proxy_url(pod_id: str, port: int) -> str:
    return f"https://{pod_id}-{int(port)}.proxy.runpod.net"


def direct_tcp_url(ports: list, port: int, tls: bool = False) -> str:
    """The pod's public ip:port for ``port`` exposed as `<port>/tcp`, or ''.

    A TCP port is mapped straight to the machine — no RunPod proxy, no
    Cloudflare in front, so no 100 s idle timeout on a slow non-streaming
    request and no body-size cap. Plain HTTP, though: the bearer token
    travels unencrypted, so use it where the network is trusted or the
    request must not be cut off. The public port is random per pod, which is
    why it is read from the runtime rather than derived.
    """
    for p in ports or []:
        try:
            if int(p.get("privatePort") or 0) != int(port):
                continue
        except (TypeError, ValueError):
            continue
        if str(p.get("type") or "").lower() != "tcp" or not p.get("isIpPublic"):
            continue
        ip, pub = p.get("ip"), p.get("publicPort")
        if ip and pub:
            return f"{'https' if tls else 'http'}://{ip}:{int(pub)}"
    return ""


# Deep link to the pod in the RunPod web console (container + vLLM logs live there).
def pod_console_url(pod_id: str) -> str:
    return f"https://www.runpod.io/console/pods/{pod_id}"


#: How long after the CONTAINER has started (RunPod's own ``uptime``) a
#: declared port may still be unpublished before the pod is called unreachable.
#: Generous: the evidence for the opposite verdict was a pod that reached 578s
#: of container uptime with no mapping, and a host that is going to publish one
#: does it at container start.
PORT_PUBLISH_GRACE_S = float(os.environ.get("CODERAI_RUNPOD_PORT_GRACE", "120"))


def _is_routable(ip: str) -> bool:
    """True for an address the public internet can actually reach.

    The case this exists for: RunPod publishes its own agent port on
    100.64.0.0/10 -- carrier-grade NAT -- on every pod, so "a port mapping
    exists" is not the same as "a port we can reach exists". A direct-TCP pod
    whose mapping is in CGNAT or RFC1918 space can never be connected to from
    here, and waiting for it is waiting for something that cannot happen.
    """
    import ipaddress
    try:
        return bool(ipaddress.ip_address(str(ip or "").strip()).is_global)
    except ValueError:
        return False


def classify_boot(info: dict, port: int, direct_tcp: bool = True,
                  grace_s: float = None) -> tuple:
    """What a booting pod is really doing: ``(verdict, why)``.

    Measured 2026-11-02: an RTX PRO 6000 pod sat "booting" for 29 minutes at
    $1.69/hr and was then terminated for not exposing its port. It had been
    RUNNING the whole time; RunPod had published its own ``19123/http`` mapping
    on 100.65.33.207 and *never published the declared 8000/tcp at all*. A
    second one behaved identically. Meanwhile three pods on other hosts each
    had 8000/tcp on a globally routable address within seconds of the container
    starting.

    So there was nothing to connect to, and nothing that any timeout could
    change -- but the boot loop reported "image pull in progress" every 60
    seconds, because that string was printed for ANY status that was not
    EXITED. It cost the full ``boot_timeout_s`` per attempt, three attempts per
    provision, in a loop, while pages waited for pods. The information needed
    to tell the states apart was in the answer all along.

    The verdicts, which have opposite correct responses:

    * ``pulling`` -- no mapping yet and the container has not started (or is
      inside the publish grace). Wait: a multi-GB image pull on a cold machine
      is minutes of exactly this.
    * ``starting`` -- the declared port is published on a routable address and
      the connection is refused. Wait: that is a server still coming up.
    * ``unreachable`` -- the container is running, past the grace, and the
      declared port is either absent from the mapping or published somewhere
      unroutable. GIVE UP NOW. The machine is the problem, and the next
      candidate is the only thing that can help.
    * ``ready`` -- the caller's own check already passed.
    """
    grace = PORT_PUBLISH_GRACE_S if grace_s is None else float(grace_s)
    if info.get("ready"):
        return "ready", ""
    try:
        uptime = float(info.get("uptime_s") or 0.0)
    except (TypeError, ValueError):
        uptime = 0.0
    raw = info.get("published_ports")
    if raw is None:
        raw = info.get("ports") or []
    ports = [p for p in raw if isinstance(p, dict)]
    started = uptime > grace

    def _priv(p):
        try:
            return int(p.get("privatePort") or p.get("private") or 0)
        except (TypeError, ValueError):
            return 0

    if not ports:
        if started:
            return "unreachable", (
                f"the container has been running {uptime:.0f}s and RunPod has "
                f"published no port mapping at all")
        return "pulling", "no port mapping yet — the image is still being pulled"

    mine = [p for p in ports if _priv(p) == int(port)]
    if not mine:
        others = ", ".join(f"{_priv(p)}/{p.get('type') or '?'}" for p in ports)
        if started:
            # The create request asks for `supportPublicIp` on a direct-TCP
            # pod (see PUBLIC_IP_FIELD), so a host that lands here anyway
            # either ignored the ask or is a community host RunPod placed
            # before honouring it. Say which, or the reader re-diagnoses the
            # whole class from scratch — it took nine dead pods once already.
            hint = (" — the pod asked for a public ip and this host still "
                    "published only RunPod's own agent port, so it cannot "
                    "serve a direct-TCP pod" if direct_tcp else
                    " — this host does not expose it")
            return "unreachable", (
                f"RunPod published {others} but NOT the declared port {port}, "
                f"and the container has been running {uptime:.0f}s{hint}")
        return "pulling", (f"port {port} is not published yet (RunPod has "
                           f"{others} so far)")

    if direct_tcp and not any(_is_routable(p.get("ip")) for p in mine):
        ips = ", ".join(str(p.get("ip") or "(no ip)") for p in mine) or "(no ip)"
        if started:
            return "unreachable", (
                f"port {port} is published on {ips}, which is not globally "
                f"routable (carrier-grade NAT or private space) — a direct-TCP "
                f"pod there cannot be reached from here")
        return "pulling", f"port {port} is on {ips}, not a routable address yet"

    return "starting", (f"port {port} is published and routable; the server "
                        f"inside has not accepted a connection yet "
                        f"({uptime:.0f}s up)")


def _machine_fields(p: dict) -> dict:
    """``{gpu, gpu_id, gpu_count, vram_gb}`` from a pod's machine, defensively.

    RunPod reports the card on the MACHINE, and a pod whose machine record is
    missing (one still provisioning, one being torn down) must not break a
    listing the reaper depends on -- so every field degrades to a blank rather
    than raising.
    """
    m = (p.get("machine") or {}) if isinstance(p, dict) else {}
    gt = (m.get("gpuType") or {}) if isinstance(m, dict) else {}
    try:
        n = max(1, int(p.get("gpuCount") or 1))
    except (TypeError, ValueError):
        n = 1
    try:
        per_card = float(gt.get("memoryInGb") or 0.0)
    except (TypeError, ValueError):
        per_card = 0.0
    return {"gpu": str(m.get("gpuDisplayName") or ""),
            "gpu_id": str(m.get("gpuTypeId") or ""),
            "gpu_count": n,
            # The pod's TOTAL VRAM: a 2-card pod holds twice the KV cache.
            "vram_gb": per_card * n}


def get_pod_machine(client, pod_id: str) -> dict:
    """``{gpu, gpu_id, gpu_count, vram_gb}`` for one pod, or blanks.

    Used when ADOPTING a pod this process did not rent: the registry may know
    the card (we recorded it when we rented it), and when it does not, the only
    other source is the account.
    """
    try:
        return {k: v for k, v in (client.get_pod(pod_id) or {}).items()
                if k in ("gpu", "gpu_id", "gpu_count", "vram_gb")}
    except Exception:
        return {}


#: [epoch seconds] before which GraphQL is not called at all. See _gql.
_GQL_BACKOFF_UNTIL = [0.0]
#: Never believe a ban longer than this (a unit we misread must not mean "never").
_GQL_BACKOFF_MAX_S = 6 * 3600


def _note_rate_limit(r) -> None:
    """Remember a 429's retryAfter (seconds) or Retry-After header."""
    wait = 0.0
    try:
        wait = float((r.json() or {}).get("retryAfter") or 0)
    except Exception:
        pass
    if not wait:
        try:
            wait = float(r.headers.get("Retry-After") or 0)
        except Exception:
            wait = 0.0
    wait = max(5.0, min(_GQL_BACKOFF_MAX_S, wait or 30.0))
    until = time.time() + wait
    if until > _GQL_BACKOFF_UNTIL[0] + 60:
        print(f"[runpod] RunPod rate-limited the API key for {wait:.0f}s "
              f"(until {time.strftime('%H:%M:%S', time.localtime(until))}) — "
              f"no GraphQL calls until then; renting, terminating and status "
              f"checks that need it will report this instead", flush=True)
    _GQL_BACKOFF_UNTIL[0] = max(_GQL_BACKOFF_UNTIL[0], until)


def _catalog_path() -> str:
    try:
        from codai.api.runpod_worker import _pod_registry_path
        return os.path.join(os.path.dirname(_pod_registry_path()),
                            "runpod-gpu-catalog.json")
    except Exception:
        return ""


def _save_catalog(rows: list) -> None:
    """Keep the last good catalogue on disk, for when GraphQL is rate-limited."""
    path = _catalog_path()
    if not path or not rows:
        return
    try:
        import json
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({"saved_at": time.time(), "rows": rows}, fh)
        os.replace(tmp, path)
    except Exception:
        pass


def _load_saved_catalog() -> list:
    path = _catalog_path()
    try:
        import json
        with open(path) as fh:
            d = json.load(fh)
        rows = d.get("rows") or []
        age_h = (time.time() - float(d.get("saved_at") or 0)) / 3600
        print(f"[runpod] GraphQL is rate-limited — using the GPU catalogue saved "
              f"{age_h:.1f} h ago ({len(rows)} card types) to choose and price "
              f"cards", flush=True)
        return list(rows)
    except Exception:
        return []


_GQL_COUNTS: dict = {}
_GQL_COUNT_SINCE = [time.time()]
_GQL_REPORT_EVERY_S = 600.0
_SAID: set = set()


def _say_once(key: str, msg: str) -> None:
    if key not in _SAID:
        _SAID.add(key)
        print(msg, flush=True)


def _count_gql(query: str) -> None:
    """Count GraphQL calls by operation and report the rate every 10 min:
    a ban arrives without warning, so the volume that earns one must be seen
    before it does."""
    import re
    m = re.search(r"\b(query|mutation)\b\s*(\w*)[^{]*\{\s*(\w+)", query or "")
    op = (m.group(3) if m else "?")
    _GQL_COUNTS[op] = _GQL_COUNTS.get(op, 0) + 1
    now = time.time()
    span = now - _GQL_COUNT_SINCE[0]
    if span >= _GQL_REPORT_EVERY_S:
        total = sum(_GQL_COUNTS.values())
        top = ", ".join(f"{k} {v}" for k, v in
                        sorted(_GQL_COUNTS.items(), key=lambda kv: -kv[1])[:6])
        print(f"[runpod] GraphQL: {total} calls in {span/60:.0f} min "
              f"({total / (span / 60):.1f}/min) — {top}", flush=True)
        _GQL_COUNTS.clear()
        _GQL_COUNT_SINCE[0] = now


def gql_call_counts() -> dict:
    return dict(_GQL_COUNTS)


def _card_for(gpu_type_id: str) -> dict:
    """{gpuDisplayName, gpuTypeId, gpuType{memoryInGb}} for a REST pod's
    machine, from the catalogue (in memory, else saved) -- REST names the card
    type and nothing else."""
    rows = (_GPU_CATALOG[1] or []) or _load_saved_catalog_quiet()
    for g in rows:
        if g.get("id") == gpu_type_id:
            return {"gpuDisplayName": g.get("display_name") or gpu_type_id,
                    "gpuTypeId": gpu_type_id,
                    "gpuType": {"memoryInGb": g.get("memory_gb") or 0}}
    return {"gpuDisplayName": gpu_type_id, "gpuTypeId": gpu_type_id, "gpuType": {}}


def _load_saved_catalog_quiet() -> list:
    try:
        import json
        with open(_catalog_path()) as fh:
            return list((json.load(fh) or {}).get("rows") or [])
    except Exception:
        return []


_V2_BACKOFF_UNTIL = [0.0]
_V2_QUOTA = {"text": "", "at": 0.0}


def _note_v2_quota(r) -> None:
    """Keep v2's live quota ("minute";r=179;t=41, "hour";..., "day";...)."""
    q = r.headers.get("RateLimit") if hasattr(r, "headers") else None
    if q:
        _V2_QUOTA.update(text=str(q), at=time.time())


def v2_quota() -> dict:
    return dict(_V2_QUOTA)


#: v2 phases -> the desiredStatus words the pool already understands.
_V2_STATUS = {"PROVISIONING": "RUNNING", "STARTING": "RUNNING", "RUNNING": "RUNNING",
              "EXITED": "EXITED", "ERROR": "FAILED", "TERMINATED": "TERMINATED"}


def _env_map(raw) -> dict:
    """A pod's environment as a dict, from any of the three shapes RunPod uses.

    REST v2 answers with a map, GraphQL with a list of ``"KEY=VALUE"`` strings,
    and a create request is BUILT from a list of ``{key, value}`` objects -- so
    all three turn up depending on which end of the API you are standing on. A
    value may itself contain "=", so the split is on the first one only.
    """
    out = {}
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict) and item.get("key") is not None:
                out[str(item["key"])] = str(item.get("value") or "")
            elif isinstance(item, str) and "=" in item:
                k, v = item.split("=", 1)
                out[k.strip()] = v
    return out


def _v2_pod_row(p: dict, pod_id: str = "", with_ports: bool = False) -> dict:
    """A REST v2 Pod in the shape get_pod / list_pods return from GraphQL."""
    p = p if isinstance(p, dict) else {}
    gpu = p.get("gpu") or {}
    machine = _card_for(str(gpu.get("id") or "")) if gpu.get("id") else {}
    phase = str(p.get("status") or "")
    rt = p.get("runtime") or {}
    row = {
        **_machine_fields({"machine": machine, "gpuCount": gpu.get("count") or 1}),
        "id": p.get("id") or pod_id, "name": p.get("name"),
        "status": _V2_STATUS.get(phase, phase), "phase": phase,
        "cost_per_hr": p.get("cost"), "image": p.get("image"),
        "createdAt": p.get("createdAt"), "last_change": "",
        "data_center": p.get("dataCenterId"),
    }
    if isinstance(rt.get("uptime"), (int, float)):
        row["uptime_s"] = float(rt["uptime"])
    # Pod RAM/CPU in use -- what ran an A40 pod out of memory under load.
    if isinstance(rt.get("memory"), dict):
        row["memory_util"] = rt["memory"].get("util")
    if isinstance(rt.get("cpu"), dict):
        row["cpu_util"] = rt["cpu"].get("util")
    if with_ports:
        ports = [{"ip": pt.get("ip") or "", "isIpPublic": bool(pt.get("ip")),
                  "privatePort": int(pt.get("private") or 0),
                  "publicPort": int(pt.get("public") or 0),
                  "type": pt.get("type") or "tcp"}
                 for pt in (rt.get("ports") or [])
                 if isinstance(pt, dict) and pt.get("public")]
        row["ports"] = ports
        row["ready"] = bool(ports)
        row.setdefault("uptime_s", 0)
    return row


#: GraphQL calls this process may make per rolling hour before it routes
#: everything REST v2 can do to REST v2. RunPod's own GraphQL limits (live
#: headers, 2026-10-20) are 3,600/h but 43,200/DAY -- an average of 1,800/h --
#: and the daily one is the one that ran out: a ban until midnight UTC. 1,500/h
#: stays under the daily average with room for other clients of the same key.
#: CODERAI_RUNPOD_GQL_HOURLY_BUDGET overrides; 0 disables the budget.
GQL_HOURLY_BUDGET = int(os.environ.get("CODERAI_RUNPOD_GQL_HOURLY_BUDGET", "1500") or 0)
_GQL_RECENT: "collections.deque" = None
_GQL_RECENT_LOCK = threading.Lock()
_GQL_OVER_SAID = [0.0]


def _gql_record_call() -> None:
    global _GQL_RECENT
    with _GQL_RECENT_LOCK:
        if _GQL_RECENT is None:
            import collections
            _GQL_RECENT = collections.deque()
        _GQL_RECENT.append(time.time())


def _gql_calls_last_hour() -> int:
    with _GQL_RECENT_LOCK:
        if not _GQL_RECENT:
            return 0
        cut = time.time() - 3600
        while _GQL_RECENT and _GQL_RECENT[0] < cut:
            _GQL_RECENT.popleft()
        return len(_GQL_RECENT)


def _gql_over_budget() -> bool:
    if GQL_HOURLY_BUDGET <= 0:
        return False
    n = _gql_calls_last_hour()
    if n < GQL_HOURLY_BUDGET:
        return False
    now = time.time()
    if now - _GQL_OVER_SAID[0] > 600:
        _GQL_OVER_SAID[0] = now
        print(f"[runpod] GraphQL budget reached ({n} calls in the last hour, budget "
              f"{GQL_HOURLY_BUDGET}/h) — routing everything REST v2 can do to REST "
              f"v2 until it drops; only spot rentals still need GraphQL", flush=True)
    return True


def _gql_banned() -> bool:
    """True while GraphQL should not be used where REST v2 can do the job:
    RunPod's rate limit is in force (see _note_rate_limit), or this process has
    spent its own hourly GraphQL budget."""
    return _GQL_BACKOFF_UNTIL[0] > time.time() or _gql_over_budget()


class RunpodClient:
    def __init__(self, cfg):
        self._cfg = cfg
        self._api_key = (getattr(cfg, "api_key", "") or "").strip()
        self._gql_base = (getattr(cfg, "api_base", "") or "https://api.runpod.io/graphql").strip()
        if not self._api_key:
            raise RunpodError("RunPod API key not configured (config.runpod.api_key).")

    # ------------------------------------------------------------------ #
    # transport
    # ------------------------------------------------------------------ #
    def _gql(self, query: str, variables: Optional[dict] = None,
             timeout: float = 30.0) -> dict:
        import requests
        # A 429 from RunPod carries `retryAfter` in SECONDS (watched it fall by
        # 20 every 20 s from 11,871), and it has been hours long. Asking again
        # inside that window only re-confirms the ban, so callers get the answer
        # here, without a request.
        _count_gql(query)
        _gql_record_call()
        until = _GQL_BACKOFF_UNTIL[0]
        if until > time.time():
            raise RunpodTransient(
                f"RunPod rate-limited this API key ({_mask(self._api_key)}); "
                f"not calling it again until "
                f"{time.strftime('%H:%M:%S', time.localtime(until))} "
                f"({until - time.time():.0f}s)")
        # RunPod authenticates GraphQL via the ?api_key= query param.
        url = self._gql_base + ("&" if "?" in self._gql_base else "?") + "api_key=" + self._api_key
        try:
            r = requests.post(url, json={"query": query, "variables": variables or {}},
                              timeout=timeout,
                              headers={"Content-Type": "application/json"})
        except requests.RequestException as exc:
            # No answer at all — transient by definition.
            raise RunpodTransient(f"RunPod API unreachable ({_mask(self._api_key)}): {exc}")
        if r.status_code == 401:
            raise RunpodError(f"RunPod rejected the API key ({_mask(self._api_key)}) — 401.")
        if r.status_code >= 400:
            # 5xx is RunPod's own trouble and 408/429 is "ask again"; neither is
            # a statement about what we asked for. A 4xx otherwise IS: a bad id,
            # a malformed query, a refused request — retrying that only wastes
            # the deadline.
            cls = (RunpodTransient
                   if r.status_code >= 500 or r.status_code in (408, 425, 429)
                   else RunpodError)
            if r.status_code == 429:
                _note_rate_limit(r)
            raise cls(f"RunPod API HTTP {r.status_code}: {r.text[:300]}")
        try:
            body = r.json()
        except ValueError:
            raise RunpodError(f"RunPod API returned non-JSON: {r.text[:200]}")
        if body.get("errors"):
            msg = "; ".join(e.get("message", str(e)) for e in body["errors"])
            raise RunpodError(f"RunPod GraphQL error: {msg}")
        return body.get("data") or {}

    def _rest_boot_state(self, pod_id: str, port: int) -> dict:
        """get_pod's answer from REST v2, with `ready` meaning the pod's mapped
        port ACCEPTS a TCP connection. Falls back to GraphQL if v2 fails
        outside a ban."""
        import socket
        try:
            row = _v2_pod_row(self._rest("GET", f"/pods/{pod_id}"), pod_id, with_ports=True)
        except (RunpodError, RunpodTransient):
            if _gql_banned():
                raise
            return self.get_pod(pod_id)
        ok = False
        for pt in row.get("ports") or []:
            if (int(pt.get("privatePort") or 0) == int(port) and pt.get("ip")
                    and pt.get("publicPort")):
                try:
                    with socket.create_connection((pt["ip"], int(pt["publicPort"])),
                                                  timeout=3):
                        ok = True
                except OSError:
                    ok = False
        row["ready"] = ok and row.get("status") == "RUNNING"
        # What RunPod has published, kept whatever `ready` says. `ports` is
        # cleared below so no caller can build a URL to a port that does not
        # accept yet -- but clearing it also threw away the only evidence that
        # distinguishes "the server has not bound yet" from "this host will
        # never publish the port at all", which is a 30-minute difference at
        # the pod's hourly rate. See `classify_boot`.
        row["published_ports"] = list(row.get("ports") or [])
        if not row["ready"]:
            row["ports"] = []
        return row

    def _rest(self, method: str, path: str, body: Optional[dict] = None,
              timeout: float = 30.0, params: Optional[dict] = None):
        """One call to RunPod's REST API v2 (https://api.runpod.io/v2).

        v2 has its OWN per-user quota -- measured 2026-10-20: GraphQL 43,200/day
        exhausted (429, Retry-After 3939 s) while v2 reported 86,399 of 86,400
        left and served creates and terminates. v1 (rest.runpod.io/v1) is
        deprecated, retired 2026-11-15, and its writes failed throughout that
        ban. Raises RunpodTransient (5xx/408/429, unreachable) or RunpodError.
        """
        import requests
        if _V2_BACKOFF_UNTIL[0] > time.time():
            raise RunpodTransient(
                f"RunPod REST v2 rate-limited this key; not calling it until "
                f"{time.strftime('%H:%M:%S', time.localtime(_V2_BACKOFF_UNTIL[0]))}")
        base = (getattr(self._cfg, "rest_v2_base", "") or "https://api.runpod.io/v2").rstrip("/")
        try:
            r = requests.request(method, base + path, json=body, params=params,
                                 timeout=timeout,
                                 headers={"Authorization": f"Bearer {self._api_key}",
                                          "Content-Type": "application/json"})
        except requests.RequestException as exc:
            raise RunpodTransient(f"RunPod REST v2 unreachable: {exc}")
        _note_v2_quota(r)
        if r.status_code >= 400:
            if r.status_code == 429:
                try:
                    wait = float(r.headers.get("Retry-After") or 30)
                except ValueError:
                    wait = 30.0
                _V2_BACKOFF_UNTIL[0] = time.time() + max(1.0, min(wait, _GQL_BACKOFF_MAX_S))
            cls = (RunpodTransient
                   if r.status_code >= 500 or r.status_code in (408, 425, 429)
                   else RunpodError)
            raise cls(f"RunPod REST v2 {method} {path} HTTP {r.status_code}: "
                      f"{(r.text or '')[:300]}")
        if not r.content:
            return {}
        try:
            return r.json()
        except ValueError:
            return {}

    def _v2_pods(self) -> list:
        """Every pod on the account, following v2 pagination."""
        out, cursor = [], None
        for _ in range(50):
            d = self._rest("GET", "/pods", params={"cursor": cursor} if cursor else None)
            out.extend(d.get("pods") or [] if isinstance(d, dict) else [])
            pg = (d.get("pagination") or {}) if isinstance(d, dict) else {}
            if not pg.get("hasNextPage") or not pg.get("nextCursor"):
                break
            cursor = pg["nextCursor"]
        return out

    def _list_gpu_types_v2(self) -> list:
        d = self._rest("GET", "/catalog/gpus")
        out = []
        for g in (d.get("gpus") or [] if isinstance(d, dict) else []):
            pr = g.get("price") or {}
            sec = pr.get("secure") if g.get("secure") else None
            com = pr.get("community") if g.get("community") else None
            out.append({"id": g.get("id"), "display_name": g.get("name") or g.get("id"),
                        "memory_gb": g.get("memory"), "secure_price": sec,
                        "community_price": com,
                        "on_demand_price": sec if sec is not None else com,
                        # v2 publishes no interruptible price: a spot option is
                        # simply not offered from this catalogue.
                        "spot_price": None})
        return out

    def ping(self) -> dict:
        """Cheap auth check — returns {"id","email"?} for the account or raises."""
        data = self._gql("query { myself { id } }", timeout=15.0)
        return data.get("myself") or {}

    # ------------------------------------------------------------------ #
    # GPU catalog + selection
    # ------------------------------------------------------------------ #
    def list_gpu_types(self, fresh: bool = False) -> list:
        """Return the RunPod GPU catalog with pricing, normalized:

        [{id, display_name, memory_gb, secure_price, community_price,
          on_demand_price, spot_price}]  (prices are USD/hr for 1 GPU; None if N/A)

        Cached for `GPU_CATALOG_TTL_S` across every caller in the process — see
        `_GPU_CATALOG`. `fresh=True` forces a fetch, for a caller that is about
        to show prices to a person rather than pick a card in a boot storm.
        """
        global _GPU_CATALOG
        if not fresh:
            until, rows = _GPU_CATALOG
            if rows is not None and until > time.time():
                return list(rows)
        q = """
        query GpuTypes {
          gpuTypes {
            id
            displayName
            memoryInGb
            securePrice
            communityPrice
            secureSpotPrice
            communitySpotPrice
          }
        }"""
        if _gql_banned():
            # Pricing is what keeps a pod inside its $/hr ceiling, and an
            # unpriced card is never chosen. During a GraphQL ban REST v2's
            # catalogue is priced too (secure/community; no spot); failing
            # that, the last catalogue this install saw.
            try:
                rows = self._list_gpu_types_v2()
                if rows:
                    with _GPU_CATALOG_LOCK:
                        _GPU_CATALOG = (time.time() + GPU_CATALOG_TTL_S, list(rows))
                    return rows
            except (RunpodError, RunpodTransient):
                pass
            saved = _load_saved_catalog()
            if saved:
                return saved
        data = self._gql(q)
        out = []
        for g in (data.get("gpuTypes") or []):
            secure = g.get("securePrice")
            community = g.get("communityPrice")
            on_demand = secure if secure is not None else community
            spot = g.get("secureSpotPrice")
            if spot is None:
                spot = g.get("communitySpotPrice")
            out.append({
                "id": g.get("id"),
                "display_name": g.get("displayName") or g.get("id"),
                "memory_gb": g.get("memoryInGb"),
                "secure_price": secure,
                "community_price": community,
                "on_demand_price": on_demand,
                "spot_price": spot,
            })
        with _GPU_CATALOG_LOCK:
            _GPU_CATALOG = (time.time() + GPU_CATALOG_TTL_S, list(out))
            _save_catalog(out)
        return out

    def pick_gpu(self, *, min_vram_gb: float = 0.0, max_hourly_usd: float = 0.0,
                 allow_spot: bool = False, cloud_type: str = "SECURE",
                 explicit_gpu_type: str = "") -> dict:
        """Choose the cheapest GPU that meets the constraints.

        Returns {gpu_type_id, display_name, memory_gb, price, is_spot} or raises
        RunpodError if nothing qualifies under the price ceiling.
        """
        catalog = self.list_gpu_types()
        secure = (cloud_type or "SECURE").upper() == "SECURE"

        def _price(g, spot):
            if spot:
                return g.get("spot_price")
            return g.get("secure_price") if secure else g.get("community_price")

        # Explicit GPU type: honor it, still enforce the price ceiling.
        if explicit_gpu_type:
            for g in catalog:
                if g["id"] == explicit_gpu_type or g["display_name"] == explicit_gpu_type:
                    for spot in ((True, False) if allow_spot else (False,)):
                        p = _price(g, spot)
                        if p is not None and (max_hourly_usd <= 0 or p <= max_hourly_usd):
                            return {"gpu_type_id": g["id"], "display_name": g["display_name"],
                                    "memory_gb": g["memory_gb"], "price": p, "is_spot": spot}
                    raise RunpodError(
                        f"GPU {explicit_gpu_type!r} exceeds the ${max_hourly_usd}/hr ceiling "
                        f"(cheapest {cloud_type} price ${_price(g, allow_spot)}).")
            raise RunpodError(f"GPU type {explicit_gpu_type!r} not found in the RunPod catalog.")

        # Otherwise pick cheapest qualifying by VRAM + price. Prefer spot when allowed.
        candidates = []
        for g in catalog:
            mem = g.get("memory_gb") or 0
            if min_vram_gb and mem < min_vram_gb:
                continue
            for spot in ((True, False) if allow_spot else (False,)):
                p = _price(g, spot)
                if p is None or p <= 0:
                    continue
                if max_hourly_usd > 0 and p > max_hourly_usd:
                    continue
                candidates.append((p, spot, g))
        if not candidates:
            raise RunpodError(
                f"No {cloud_type} GPU with ≥{min_vram_gb}GB VRAM under "
                f"${max_hourly_usd}/hr (spot={'on' if allow_spot else 'off'}).")
        candidates.sort(key=lambda c: (c[0], not c[1]))  # cheapest, spot-first on tie
        price, spot, g = candidates[0]
        return {"gpu_type_id": g["id"], "display_name": g["display_name"],
                "memory_gb": g["memory_gb"], "price": price, "is_spot": spot}

    # ------------------------------------------------------------------ #
    # Pod lifecycle
    # ------------------------------------------------------------------ #
    def create_pod(self, *, name: str, image: str, gpu_type_id: str, port: int,
                   cloud_type: str = "SECURE", gpu_count: int = 1,
                   container_disk_gb: int = 40, volume_gb: int = 0,
                   volume_mount_path: str = "/workspace", env: Optional[dict] = None,
                   docker_args: str = "", is_spot: bool = False,
                   bid_per_gpu: float = 0.0, data_center_id: str = "",
                   global_volume_id: str = "",
                   registry_auth_id: str = "",
                   entrypoint: Optional[list] = None,
                   start_cmd: Optional[list] = None,
                   network_volume_id: str = "", direct_tcp: bool = False) -> str:
        """Provision a pod (on-demand or interruptible/spot). Returns the pod id.

        ``direct_tcp`` exposes the port as `<port>/tcp` (public ip + random
        public port, no proxy, no Cloudflare) instead of `<port>/http`.

        ``entrypoint``/``start_cmd`` override the image's ENTRYPOINT/CMD, which is
        what lets a pod fetch something before its server starts — an image whose
        ENTRYPOINT is the server itself (vllm/vllm-openai) otherwise treats every
        argument as the server's own. Only the REST API accepts those, so the
        request goes there when they are given and stays on GraphQL otherwise.
        """
        if (entrypoint or start_cmd) or _gql_banned():
            # REST too while GraphQL is rate-limited: RunPod's bans have been
            # hours long, and REST kept answering through them.
            return self._create_pod_rest(
                global_volume_id=global_volume_id, docker_args=docker_args,
                network_volume_id=network_volume_id,
                name=name, image=image, gpu_type_id=gpu_type_id, port=port,
                cloud_type=cloud_type, gpu_count=gpu_count,
                container_disk_gb=container_disk_gb, volume_gb=volume_gb,
                volume_mount_path=volume_mount_path, env=env,
                is_spot=is_spot, bid_per_gpu=bid_per_gpu,
                data_center_id=data_center_id, registry_auth_id=registry_auth_id,
                entrypoint=entrypoint, start_cmd=start_cmd, direct_tcp=direct_tcp)
        env_list = [{"key": str(k), "value": str(v)} for k, v in (env or {}).items()]
        ports = f"{int(port)}/{'tcp' if direct_tcp else 'http'}"
        common = {
            "cloudType": (cloud_type or "SECURE").upper(),
            "gpuCount": int(gpu_count),
            "gpuTypeId": gpu_type_id,
            "name": name,
            "imageName": image,
            "containerDiskInGb": int(container_disk_gb),
            "volumeInGb": int(volume_gb),
            "volumeMountPath": volume_mount_path,
            "ports": ports,
            "dockerArgs": docker_args or "",
            "env": env_list,
        }
        if data_center_id:
            common["dataCenterId"] = data_center_id
        # A private image needs credentials RunPod holds for you: create them under
        # Settings -> Container Registry Credentials and paste the id here. Public
        # images (vLLM, llama.cpp, a public coderai) need none.
        if registry_auth_id:
            common["containerRegistryAuthId"] = registry_auth_id
        # A network volume persists across pods: weights downloaded once are
        # there for every later pod, which is the difference between paying a
        # cold model download per pod and paying it once.
        if network_volume_id:
            common["networkVolumeId"] = network_volume_id
        if global_volume_id:
            common[GLOBAL_VOLUME_FIELD] = global_volume_id
        # See PUBLIC_IP_FIELD: a direct-TCP pod on a host that cannot publish a
        # public IP is unreachable from the moment it is created, and bills
        # anyway. Confirmed on PodFindAndDeployOnDemandInput; the spot input
        # could not be read, so a refusal there is caught below.
        if direct_tcp:
            common[PUBLIC_IP_FIELD] = True
        if is_spot:
            common["bidPerGpu"] = float(bid_per_gpu)
            mutation = "podRentInterruptable"
            var_type = "PodRentInterruptableInput"
        else:
            mutation = "podFindAndDeployOnDemand"
            var_type = "PodFindAndDeployOnDemandInput"
        q = f"""
        mutation Deploy($input: {var_type}!) {{
          {mutation}(input: $input) {{ id imageName machineId }}
        }}"""
        try:
            data = self._gql(q, {"input": common}, timeout=90.0)
        except RunpodError as exc:
            # Same guard as the REST path: GraphQL calls the global-volume field
            # an undefined input field, and a configured global volume must not
            # fail every create while RunPod keeps it console-only. The
            # public-ip field gets the same treatment, because the spot input
            # type could not be read from RunPod's published schema -- better a
            # pod rented without the hint than no pod at all.
            dropped = []
            if (global_volume_id and GLOBAL_VOLUME_FIELD in common
                    and _global_field_refused_gql(str(exc))):
                _warn_global_volume_unsupported(global_volume_id)
                common.pop(GLOBAL_VOLUME_FIELD, None)
                dropped.append(GLOBAL_VOLUME_FIELD)
            if (PUBLIC_IP_FIELD in common
                    and _field_refused_gql(str(exc), PUBLIC_IP_FIELD)):
                print(f"[runpod] this API rejects {PUBLIC_IP_FIELD!r} on "
                      f"{var_type}: renting without it, so the host may not "
                      f"publish a public ip and a direct-TCP pod may be "
                      f"unreachable", flush=True)
                common.pop(PUBLIC_IP_FIELD, None)
                dropped.append(PUBLIC_IP_FIELD)
            if not dropped:
                raise
            data = self._gql(q, {"input": common}, timeout=90.0)
        node = data.get(mutation) or {}
        pod_id = node.get("id")
        if not pod_id:
            raise RunpodError(f"RunPod did not return a pod id for {name!r} (capacity? {data}).")
        return pod_id

    def _create_pod_rest(self, *, name, image, gpu_type_id, port, cloud_type,
                         gpu_count, container_disk_gb, volume_gb, volume_mount_path,
                         env, is_spot, bid_per_gpu, data_center_id,
                         registry_auth_id, entrypoint, start_cmd,
                         network_volume_id: str = "", direct_tcp: bool = False,
                         global_volume_id: str = "", docker_args: str = "") -> str:
        """Create a pod through REST API v2 (POST /v2/pods).

        Used for entrypoint/cmd overrides (GraphQL has none) and whenever
        GraphQL is rate-limited. v2 has no interruptible bidding, so a spot
        request is refused here rather than silently rented on-demand.
        """
        if is_spot:
            raise RunpodError("REST v2 cannot rent an interruptible (spot) pod; "
                              "that needs GraphQL")
        body = {
            "name": name,
            "image": image,
            "gpu": {"id": gpu_type_id, "count": int(gpu_count)},
            "cloud": (cloud_type or "SECURE").upper(),
            "disk": int(container_disk_gb),
            "ports": [f"{int(port)}/{'tcp' if direct_tcp else 'http'}"],
            "env": {str(k): str(v) for k, v in (env or {}).items()},
        }
        # See PUBLIC_IP_FIELD. v2 documents this on PodCreateInput with no
        # default, and "If null, the Pod might not have a public IP address".
        if direct_tcp:
            body[PUBLIC_IP_FIELD] = True
        if docker_args:
            body["args"] = docker_args
        if entrypoint:
            body["entrypoint"] = list(entrypoint)
        if start_cmd:
            body["cmd"] = list(start_cmd)
        if data_center_id:
            body["dataCenterIds"] = [data_center_id]
        if registry_auth_id:
            body["registry"] = registry_auth_id
        mounts = {}
        if int(volume_gb or 0) > 0:
            mounts["persistent"] = {"size": int(volume_gb),
                                    "path": volume_mount_path or "/workspace"}
        if network_volume_id:
            mounts["network"] = [{"volumeId": network_volume_id,
                                  "path": volume_mount_path or "/workspace"}]
        if global_volume_id:
            mounts["global"] = [{"volumeId": global_volume_id}]
        if mounts:
            body["mounts"] = mounts
        try:
            d = self._rest("POST", "/pods", body=body, timeout=90.0)
        except RunpodError as exc:
            # v2 documents global volumes as a mount, but a refused one must not
            # fail every create: rent without it (the weights download instead)
            # and say so once -- the same guard the GraphQL path keeps.
            if not (global_volume_id and "global" in str(exc).lower()
                    and (body.get("mounts") or {}).get("global")):
                raise
            _warn_global_volume_unsupported(global_volume_id)
            body["mounts"].pop("global", None)
            if not body["mounts"]:
                body.pop("mounts")
            d = self._rest("POST", "/pods", body=body, timeout=90.0)
        pod_id = (d or {}).get("id") if isinstance(d, dict) else None
        if not pod_id:
            raise RunpodError(f"RunPod REST v2 create returned no pod id: {str(d)[:300]}")
        return pod_id

    def list_network_volumes(self) -> list:
        """The account's network volumes: [{id, name, size, dataCenterId}, …].

        Needed because a pod must run in the volume's OWN data center — a volume
        in EU-RO-1 cannot be attached to a pod anywhere else, and the failure
        arrives as an unhelpful capacity error.
        """
        try:
            d = self._rest("GET", "/network-volumes")
            rows = d.get("networkVolumes") or [] if isinstance(d, dict) else []
            # v2 calls it dataCenter; callers read dataCenterId (the v1 name).
            return [dict(v, dataCenterId=v.get("dataCenterId") or v.get("dataCenter"))
                    for v in rows if isinstance(v, dict)]
        except (RunpodError, RunpodTransient) as exc:
            print(f"[runpod] could not list network volumes: {exc}", flush=True)
            return []

    def get_pod(self, pod_id: str) -> dict:
        """Return {id, status, uptime_s, cost_per_hr, ports:[...], ready:bool, url?}."""
        try:
            return _v2_pod_row(self._rest("GET", f"/pods/{pod_id}"), pod_id,
                               with_ports=True)
        except (RunpodError, RunpodTransient) as exc:
            if _gql_banned() or "HTTP 404" in str(exc):
                raise
            _say_once("v2-get", f"[runpod] REST v2 pod lookup failed ({exc}); "
                                f"using GraphQL for it")
        q = """
        query Pod($input: PodFilter!) {
          pod(input: $input) {
            id
            desiredStatus
            lastStatusChange
            costPerHr
            gpuCount
            machine { gpuDisplayName gpuTypeId gpuType { memoryInGb } }
            runtime {
              uptimeInSeconds
              ports { ip isIpPublic privatePort publicPort type }
            }
          }
        }"""
        data = self._gql(q, {"input": {"podId": pod_id}}, timeout=20.0)
        p = data.get("pod") or {}
        rt = p.get("runtime") or {}
        ports = rt.get("ports") or []
        return {
            **_machine_fields(p),
            "id": p.get("id") or pod_id,
            "status": p.get("desiredStatus"),
            # The one field that says WHY a pod is EXITED. It reads like
            # "Rented by User: ..." or "container exited with code 1" — and it
            # was never fetched, so a dead pod produced three hundred seconds
            # of "still no port" and no reason at all.
            "last_change": p.get("lastStatusChange") or "",
            "uptime_s": rt.get("uptimeInSeconds") or 0,
            "cost_per_hr": p.get("costPerHr"),
            "ports": ports,
            # A pod is "ready" for us once its http proxy port is exposed.
            "ready": bool(ports),
        }

    def get_pod_launch(self, pod_id: str) -> dict:
        """How a RUNNING pod was started: ``{env: {...}, args: str, cmd: [...]}``.

        This is the one lookup that can recover a pod's BEARER TOKEN, which is
        what makes a pod rented by another host adoptable at all: without it the
        pod answers 401 to everything and the only honest thing to do with it is
        terminate it.

        It is deliberately NOT part of `get_pod`/`list_pods`. Those rows are
        rendered on the admin page and logged, and a token that reaches either is
        a token leaked -- so it travels on a path of its own, with one caller.

        Two shapes, because the token is put in two places (see `pod_plan`): a
        coderai pod reads it from the environment, while vLLM and llama.cpp take
        it as `--api-key` and it is therefore in the pod's ARGS. REST v2 names
        those `env`/`args`/`cmd`; GraphQL has `env` as a list of "K=V" strings
        and calls the args `dockerArgs`. All of it is normalised here so the
        caller sees one shape.
        """
        out = {"env": {}, "args": "", "cmd": []}
        try:
            d = self._rest("GET", f"/pods/{pod_id}")
            d = d if isinstance(d, dict) else {}
            out["env"] = _env_map(d.get("env"))
            out["args"] = str(d.get("args") or "")
            cmd = d.get("cmd")
            out["cmd"] = [str(c) for c in cmd] if isinstance(cmd, list) else []
            if out["env"] or out["args"] or out["cmd"]:
                return out
        except (RunpodError, RunpodTransient) as exc:
            if _gql_banned() or "HTTP 404" in str(exc):
                raise
            _say_once("v2-launch", f"[runpod] REST v2 could not report how a pod "
                                   f"was started ({exc}); using GraphQL for it")
        if _gql_banned():
            return out
        q = """
        query Pod($input: PodFilter!) {
          pod(input: $input) { id env dockerArgs }
        }"""
        data = self._gql(q, {"input": {"podId": pod_id}}, timeout=20.0)
        p = (data.get("pod") or {}) if isinstance(data, dict) else {}
        out["env"] = _env_map(p.get("env"))
        out["args"] = str(p.get("dockerArgs") or "")
        return out

    #: The state transitions REST v2 accepts on a pod. `restart` is the only
    #: one that keeps the machine: v2 documents it as restarting a RUNNING
    #: pod's container IN PLACE, while `stop` is documented as "releasing
    #: GPU/CPU compute while keeping its disk" -- i.e. the card goes back to
    #: RunPod's pool and a later `start` takes whatever is free then. When the
    #: card you are standing on is sold out, that difference is the whole game.
    POD_ACTIONS = ("start", "stop", "restart", "terminate")

    def edit_pod(self, pod_id: str, *, image: str = "",
                 env: Optional[dict] = None, timeout: float = 60.0) -> dict:
        """Change a RUNNING pod's image and/or environment in place.

        What this buys: a rented card you already hold, running the wrong
        image. Renting the right one means giving this one back first, and a
        card that is sold out does not come back -- measured 2026-11-02, an OCR
        pool failed to deploy a candidate 301 times in one hour while an idle
        A40 sat in another pool on this account. Re-imaging the pod you have
        skips the queue you cannot win.

        ``image`` and ``env`` are both documented as mutable on
        ``PATCH /v2/pods/{id}``, and a PATCH leaves omitted fields untouched --
        but ``env``, when given, REPLACES the pod's environment rather than
        merging into it. So a caller changing env must pass the whole set the
        pod should boot with, which is exactly what ``pod_plan`` produces.

        The change lands on the container's next start, so a caller that wants
        it running now follows this with ``pod_action(pod_id, "restart")``.

        REST v2 only. GraphQL has no documented equivalent, and v1 retires
        2026-11-15 -- a write path that is about to be switched off is not
        somewhere to put the one call that moves a production pod.
        """
        pid = str(pod_id or "").strip()
        if not pid:
            raise RunpodError("edit_pod: no pod id given")
        body: dict = {}
        if image:
            body["image"] = str(image)
        if env is not None:
            body["env"] = {str(k): str(v) for k, v in (env or {}).items()}
        if not body:
            raise RunpodError("edit_pod: nothing to change (pass image and/or env)")
        d = self._rest("PATCH", f"/pods/{pid}", body=body, timeout=timeout)
        return _v2_pod_row(d if isinstance(d, dict) else {}, pid, with_ports=True)

    def pod_action(self, pod_id: str, action: str, timeout: float = 60.0) -> dict:
        """Trigger one state transition on a pod. Returns the updated pod row.

        Read :attr:`POD_ACTIONS` before choosing one: `restart` keeps the
        machine and `stop` hands the card back. ``terminate`` is accepted here
        for completeness but :meth:`terminate_pod` is the one to call for it --
        it retries, because a pod that cannot be terminated bills.
        """
        pid = str(pod_id or "").strip()
        act = str(action or "").strip().lower()
        if not pid:
            raise RunpodError("pod_action: no pod id given")
        if act not in self.POD_ACTIONS:
            raise RunpodError(f"pod_action: {action!r} is not one of "
                              f"{', '.join(self.POD_ACTIONS)}")
        d = self._rest("POST", f"/pods/{pid}/action", body={"action": act},
                       timeout=timeout)
        # `terminate` answers 204 with no body; the others return the pod.
        return _v2_pod_row(d if isinstance(d, dict) else {}, pid, with_ports=True)

    def stop_pod(self, pod_id: str) -> None:
        q = "mutation($input: PodStopInput!) { podStop(input: $input) { id } }"
        try:
            self._gql(q, {"input": {"podId": pod_id}}, timeout=30.0)
        except RunpodError:
            raise

    def terminate_pod(self, pod_id: str, attempts: int = 4) -> None:
        """Stop a pod, retrying while the API is merely unreachable.

        The one call that must not give up quietly. Every other failed call
        costs a retry; this one costs a GPU that bills until a human notices.
        The case that matters is precisely the one where a single try fails:
        the control plane is having trouble, which is also what made the caller
        decide to tear the pod down in the first place — so the teardown meets
        the same 503 that caused it.

        Raises after the last attempt. A caller that swallows that exception is
        choosing to leak a pod, and should say so in the log instead.
        """
        # REST v2 first: its own quota, and the one call that must not wait out
        # a GraphQL ban (a pod that cannot be terminated bills). 404 = gone.
        try:
            self._rest("DELETE", f"/pods/{pod_id}")
            return
        except RunpodError as exc:
            if "HTTP 404" in str(exc):
                return
            if _gql_banned():
                raise
        except RunpodTransient:
            if _gql_banned():
                raise
        q = "mutation($input: PodTerminateInput!) { podTerminate(input: $input) }"
        delay = 2.0
        for attempt in range(1, max(1, attempts) + 1):
            try:
                self._gql(q, {"input": {"podId": pod_id}}, timeout=30.0)
                return
            except RunpodTransient as exc:
                if attempt >= attempts:
                    raise
                print(f"[runpod] terminate {pod_id}: {exc} — retrying in "
                      f"{delay:.0f}s ({attempt}/{attempts})", flush=True)
                time.sleep(delay)
                delay = min(delay * 2, 15.0)

    def list_pods(self) -> list:
        """List the account's pods: [{id, name, status, cost_per_hr, image}]."""
        # REST FIRST, always. This is called every minute by the reaper and
        # the survivor sweep, and REST reads kept answering through a 3.3 h
        # GraphQL ban -- the fewer GraphQL calls, the less likely the ban.
        try:
            return [_v2_pod_row(p, p.get("id")) for p in self._v2_pods()
                    if isinstance(p, dict)]
        except (RunpodError, RunpodTransient) as exc:
            if _gql_banned():
                raise
            _say_once("rest-list", f"[runpod] REST pod listing failed ({exc}); "
                                   f"using GraphQL for it")
        # runtime.uptimeInSeconds is what makes the reaper's grace period real:
        # without an age every pod looked undatable, and the "never reap a pod
        # younger than GRACE" backstop abstained on all of them.
        # lastStatusChange carries the RENTAL time ("Rented by User: Thu Oct 15
        # 2026 19:36:58 GMT+0000"), which is the pod's real age. uptime alone is
        # the current RUN's age and resets on every container restart, so a pod
        # stuck in a restart loop reported 234s when it was 11 minutes old and
        # could never cross the reaper's 20-minute grace. The one pod that most
        # needs reaping — one nothing is tracking, failing over and over — was
        # the one pod the reaper could never see.
        q = """
        query Pods {
          myself {
            pods { id name desiredStatus costPerHr imageName lastStatusChange
                   gpuCount
                   machine { gpuDisplayName gpuTypeId gpuType { memoryInGb } }
                   runtime { uptimeInSeconds } }
          }
        }"""
        data = self._gql(q)
        me = data.get("myself") or {}
        out = []
        for p in (me.get("pods") or []):
            row = {
                "id": p.get("id"), "name": p.get("name"),
                "status": p.get("desiredStatus"), "cost_per_hr": p.get("costPerHr"),
                "image": p.get("imageName"),
                # WHICH CARD. A pool that cannot name a pod's card cannot size
                # its concurrency, weight its share of the load, or report the
                # fleet honestly -- adopted pods were listed as "(adopted)" and
                # counted as one anonymous kind.
                **_machine_fields(p),
                "last_change": p.get("lastStatusChange") or "",
            }
            # Absent while a pod is still being scheduled, which is exactly when
            # it must not be mistaken for a leftover — the reaper now treats a
            # missing age as "too young to judge".
            rt = p.get("runtime") or {}
            up = rt.get("uptimeInSeconds")
            if isinstance(up, (int, float)) and up >= 0:
                row["uptime_s"] = float(up)
            out.append(row)
        return out

    def get_pod_logs(self, pod_id: str, tail: int = 200) -> dict:
        """Best-effort container/vLLM logs for a pod via the REST API.

        There is no public pod-log API. RunPod's REST OpenAPI spec lists 23
        routes (pods, billing, endpoints, volumes) and not one of them serves
        logs; GraphQL refuses introspection, and its documented Pod type exposes
        none either. The console is a human surface, not an API.

        The routes below are kept because they cost one request on a path that
        only runs when a boot already failed, and RunPod has moved this surface
        before. The real answer is the pod's own /boot record — see
        RunpodPodPool._dump_pod_logs, which asks the pod first. Returns
        {console_url, logs?: str, error?: str}."""
        import requests
        out = {"console_url": pod_console_url(pod_id)}
        rest = (getattr(self._cfg, "rest_base", "") or "https://rest.runpod.io/v1").rstrip("/")
        hdrs = {"Authorization": f"Bearer {self._api_key}"}
        # RunPod's log surface has moved around; try the plausible routes in order and
        # keep the first that answers. The console link is the always-available path.
        attempts = [
            (f"{rest}/pods/{pod_id}/logs", None),
            (f"{rest}/pods/{pod_id}/logs", {"limit": tail}),
            (f"{rest}/pods/{pod_id}/containerLogs", None),
        ]
        errors = []
        for url, params in attempts:
            try:
                r = requests.get(url, headers=hdrs, params=params, timeout=15)
            except requests.RequestException as exc:
                errors.append(f"{url.rsplit('/', 1)[-1]}: unreachable ({exc})")
                continue
            if r.status_code == 200:
                try:
                    body = r.json()
                    out["logs"] = body if isinstance(body, str) else \
                        (body.get("logs") or body.get("data") or str(body))
                except ValueError:
                    out["logs"] = r.text
                return out
            errors.append(f"HTTP {r.status_code}")
        out["error"] = ("RunPod publishes no pod-log API (" + "; ".join(errors[:3])
                        + "); a coderai pod serves its own boot record at /boot — "
                        "for other images use the console link")
        return out

    def wait_ready(self, pod_id: str, port: int, ready_timeout: float = 900.0,
                   poll_every: float = 5.0, direct_tcp: bool = False,
                   tls: bool = False) -> str:
        """Poll until the pod exposes its port; return the base URL to talk to.

        The proxy URL is derivable the moment any port shows; a direct TCP
        mapping is only usable once the runtime reports the public ip:port,
        so the wait continues until it does.

        Only waits for the port to be exposed — the HTTP service inside (vLLM) is
        health-checked separately by the worker."""
        deadline = time.time() + ready_timeout
        started = time.time()
        next_report = started + 60.0
        last = {}
        # A boot wait is MINUTES long — a cold machine pulling a 10 GB image is
        # the normal case after every image release — so it is the call most
        # likely to be in flight when the control plane hiccups. Asking and not
        # being answered is not the pod failing; keep asking until the deadline
        # the caller set, and only then say the API was the thing that broke.
        api_down_since = 0.0
        last_transient = None
        said_port_open = False
        while time.time() < deadline:
            try:
                # A direct-TCP pod is polled over REST + a TCP connect: polling
                # GraphQL every 5 s per booting pod is what a 64-pod scale-up
                # turns into hundreds of GraphQL calls a minute, and RunPod
                # answered that with a 3.3 h ban. The connect is also the
                # stronger signal: RunPod refuses it until the server inside
                # listens, while GraphQL lists the port when the container starts.
                info = (self._rest_boot_state(pod_id, port) if direct_tcp
                        else self.get_pod(pod_id))
            except RunpodTransient as exc:
                last_transient = exc
                if not api_down_since:
                    api_down_since = time.time()
                    print(f"[runpod] pod {pod_id}: RunPod API not answering "
                          f"({exc}) — still waiting; this says nothing about "
                          f"the pod", flush=True)
                time.sleep(poll_every)
                continue
            if api_down_since:
                print(f"[runpod] pod {pod_id}: RunPod API answering again after "
                      f"{time.time() - api_down_since:.0f}s", flush=True)
                api_down_since = 0.0
            last = info
            if info.get("status") in ("TERMINATED", "FAILED"):
                raise RunpodError(f"Pod {pod_id} entered status {info.get('status')} before ready.")
            if info.get("status") == "EXITED" and time.time() - started > 45:
                # The container started and died. A pod reported EXITED with
                # uptime 0s from its very first poll and the loop waited out the
                # full budget on it — three machines in a row — while printing
                # "image pull in progress", which it never was. A short grace
                # period covers RunPod's own restart-on-exit; past that, EXITED
                # means the process crashed on boot and no amount of waiting
                # changes that. Fail fast so the retry lands on the real cause.
                raise RunpodError(
                    f"Pod {pod_id} container EXITED on boot (uptime "
                    f"{info.get('uptime_s')}s) — the image pulled but the process "
                    f"died immediately. RunPod says: "
                    f"{info.get('last_change') or '(no reason given)'}")
            if info.get("ready"):
                if not direct_tcp:
                    return pod_proxy_url(pod_id, port)
                direct = direct_tcp_url(info.get("ports"), port, tls=tls)
                if direct:
                    return direct
            # WHICH of the three boot states this is, from RunPod's own answer,
            # because they do not have the same correct response. Waiting is
            # right for a pull and for a server still binding; it is pure waste
            # for a host that will not publish the port, and that one used to be
            # indistinguishable -- a full boot_timeout_s of it, per attempt.
            verdict, why = classify_boot(info, port, direct_tcp)
            if verdict == "unreachable":
                raise RunpodError(
                    f"Pod {pod_id} cannot be reached and more waiting will not "
                    f"change that: {why}. Giving up after "
                    f"{time.time() - started:.0f}s instead of waiting out "
                    f"{ready_timeout:.0f}s. This is the MACHINE, not the image "
                    f"or the model: retry on another candidate, and if one card "
                    f"keeps doing it, take it out of `gpu_types`.")
            if verdict == "starting" and not said_port_open:
                said_port_open = True
                print(f"[runpod] pod {pod_id}: port {port} is published and "
                      f"routable after {time.time() - started:.0f}s — waiting "
                      f"for the server inside to accept connections", flush=True)
            # A multi-GB image pull is minutes of silence otherwise, which reads
            # exactly like a hung pod. Say what is actually being waited on --
            # and say the truth: this line claimed "image pull in progress" for
            # any status that was not EXITED, including a pod whose own uptime
            # proved the pull had finished eight minutes earlier.
            if time.time() >= next_report:
                next_report = time.time() + 60.0
                if info.get("status") == "EXITED":
                    why = (f"container has EXITED "
                           f"({info.get('last_change') or 'no reason given'})")
                print(f"[runpod] pod {pod_id}: still no port after "
                      f"{time.time() - started:.0f}s "
                      f"(status={info.get('status')} uptime={info.get('uptime_s')}s) "
                      f"— {why}", flush=True)
            time.sleep(poll_every)
        if api_down_since and not last:
            # The whole wait went by without one answer. Blaming the pod here
            # would be a guess, and the caller tears down on a boot failure.
            raise RunpodTransient(
                f"Pod {pod_id}: RunPod's API did not answer once in "
                f"{ready_timeout:.0f}s ({last_transient}) — the pod's state is "
                f"unknown, not known-bad.")
        _verdict, _why = classify_boot(last, port, direct_tcp)
        raise RunpodError(
            f"Pod {pod_id} did not expose port {port} within {ready_timeout}s "
            f"(last status={last.get('status')} uptime={last.get('uptime_s')}s; "
            f"{_why or 'no further detail'}). A cold machine pulling a "
            f"multi-GB image is the usual cause — raise `boot_timeout_s` on the "
            f"model if this repeats. Note that raising it does NOT help a host "
            f"that never publishes the port; that case is reported as "
            f"unreachable long before this.")
