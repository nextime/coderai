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


def _global_field_refused_gql(text: str) -> bool:
    """GraphQL's wording for an input field its schema does not define."""
    t = (text or "")
    return GLOBAL_VOLUME_FIELD in t and (
        "is not defined by type" in t or "invalid value" in t or "Unknown" in t)


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


def _gql_banned() -> bool:
    """True while RunPod's GraphQL rate limit is in force (see _note_rate_limit)."""
    return _GQL_BACKOFF_UNTIL[0] > time.time()


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
            # fail every create while RunPod keeps it console-only.
            if not (global_volume_id and GLOBAL_VOLUME_FIELD in common
                    and _global_field_refused_gql(str(exc))):
                raise
            _warn_global_volume_unsupported(global_volume_id)
            common.pop(GLOBAL_VOLUME_FIELD, None)
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
            # A multi-GB image pull is minutes of silence otherwise, which reads
            # exactly like a hung pod. Say what is actually being waited on.
            if time.time() >= next_report:
                next_report = time.time() + 60.0
                why = ("image pull in progress" if info.get("status") != "EXITED"
                       else f"container has EXITED ({info.get('last_change') or 'no reason given'})")
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
        raise RunpodError(
            f"Pod {pod_id} did not expose port {port} within {ready_timeout}s "
            f"(last status={last.get('status')} uptime={last.get('uptime_s')}s). "
            f"A cold machine pulling a multi-GB image is the usual cause — raise "
            f"`boot_timeout_s` on the model if this repeats.")
