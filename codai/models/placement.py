# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
# GPLv3 - see the project LICENSE.

"""Where should a request for this model run — asked per request, for any model.

A caller names a model. Where it runs is coderai's business, and the answer comes
from how the model is configured: on this machine, on a cluster node it is pinned
to, on a rented pod, or on this machine until it is busy and then a rented pod.
Nothing about that is specific to one kind of model or one endpoint.

For requests that pass THROUGH coderai this has always been true — the front
router honours the model's ``engine`` pin, treats a cluster node as an engine by
name, reaches RunPod through the model's backend, and bursts to a pod when the
model says to. This module is the same decision for a caller that cannot go
through that path: a component that must hold an OpenAI **address** and speak to
the model itself.

OCR is why it exists. Surya-2 and olmOCR are pipelines, not models: they drive a
model server directly, sending multipart images, a strict ``json_schema`` and
``logprobs: true`` — all three of which were silently dropped the last time OCR
went through the backend abstraction, which parses a request and re-emits it. So
the pipeline keeps its direct address, and :mod:`codai.ocr.gateway` makes that
address ours and forwards the bytes unchanged. What it cannot do is choose, and
choosing is what this module does:

* a RunPod pool grows when its least-loaded pod reaches
  ``scale_up_inflight_per_pod`` in-flight requests, counted as a request passes
  through the pool. A caller holding one pod's URL never passed through, so the
  pool saw no load however hard it was driven — and had the caller wanted a
  second pod, it had no address for one;
* a model pinned to several cluster nodes went to whichever answered first, for
  ever, however busy that node became;
* a local model configured to burst to RunPod under load never did: that trigger
  lives on the proxy path such a caller bypasses.

Any future component that needs "an endpoint for model X, placed as configured"
should ask here rather than grow its own answer.
"""

import threading
import time
from typing import Callable, List, Optional

from codai.ocr.base import OcrError

#: How long a discovered candidate set is reused before it is rebuilt. Short,
#: because a node coming back or a model being unloaded should take effect
#: without restarting coderai, and discovery is a handful of local HTTP calls.
CANDIDATES_TTL_S = 30.0

#: How long a node's reported load is trusted. A node reports the load of its
#: WHOLE install, not just our OCR traffic, so this is the only way to see a node
#: that is busy with something else.
NODE_LOAD_TTL_S = 5.0

_lock = threading.Lock()
_candidates: dict = {}          # model id -> (expires_at, [Candidate])
_inflight: dict = {}            # candidate key -> our own in-flight count
_node_load: dict = {}           # node url -> (expires_at, inflight)


class Candidate:
    """One place that can serve the model.

    ``acquire`` exists for a pool, where choosing is not free: it picks the
    least-loaded pod, counts the request against it (which is what makes the pool
    grow) and may rent a new pod. Everything else is a fixed address.
    """

    def __init__(self, key: str, kind: str, base: str, api_key: str = "",
                 priority: int = 0, acquire: Optional[Callable] = None,
                 node_url: str = "", primary: bool = False):
        self.key = key
        self.kind = kind                # "local" | "node" | "pod"
        self.base = base                # ends /v1 (empty for a lazy pool)
        self.api_key = api_key
        self.priority = priority        # lower wins ties
        self.acquire = acquire
        self.node_url = node_url
        #: This is where the operator SAID the model runs — its own `backend`.
        #: An explicit pin is a decision, not a hint: discovery may add places to
        #: overflow to, and must never override the configured home.
        self.primary = primary

    def __repr__(self):                 # pragma: no cover - debugging aid
        return f"<Candidate {self.kind} {self.key}>"


class Lease:
    """One chosen upstream for one request. Always released."""

    def __init__(self, base: str, api_key: str, key: str, kind: str,
                 release: Optional[Callable] = None,
                 dead: Optional[Callable] = None):
        self.base = base
        self.api_key = api_key
        self.key = key
        self.kind = kind
        self._release = release
        self._dead = dead
        self._done = False

    def release(self) -> None:
        if self._done:
            return
        self._done = True
        _dec(self.key)
        if self._release is not None:
            try:
                self._release()
            except Exception:
                pass

    def dead(self) -> None:
        """Nothing answered at this address: tell the pool, then release.

        A rented pod can vanish under us — RunPod reclaimed one mid-document
        today, and the pool went on reporting it healthy. Every page then chose
        the same corpse, the in-flight count climbed and nothing drained, which
        is what a client sees as "requests neither complete nor time out". The
        proxy backend has always reported this (RunpodBackend._unreachable);
        a caller holding a lease has to do the same.
        """
        if self._dead is not None:
            try:
                self._dead()
            except Exception:
                pass
            self._dead = None
        forget()          # the candidate set named a pod that is gone
        self.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
        return False


def _inc(key: str) -> None:
    with _lock:
        _inflight[key] = _inflight.get(key, 0) + 1


def _dec(key: str) -> None:
    with _lock:
        n = _inflight.get(key, 0) - 1
        if n > 0:
            _inflight[key] = n
        else:
            _inflight.pop(key, None)


def inflight(key: str = "") -> int:
    """Our in-flight count for one candidate, or the total."""
    with _lock:
        if key:
            return int(_inflight.get(key, 0))
        return sum(_inflight.values())


def forget(model_id: str = "") -> None:
    """Drop discovered candidates (config changed, a node came or went)."""
    with _lock:
        if model_id:
            _candidates.pop(model_id.strip(), None)
        else:
            _candidates.clear()
        _node_load.clear()


# ------------------------------------------------------------------ discovery #
def _local_candidate(model_id: str) -> Optional[Candidate]:
    """This install's own vLLM, when it is what serves the model."""
    try:
        from codai.models import manager as mm
        if mm.resolve_engine_backend(model_id) != "vllm":
            return None
        from codai.api import runpod_worker, vllm_worker
        from codai.models.manager import get_active_vllm_config
        vcfg = get_active_vllm_config()
        path = (runpod_worker._model_path_for(model_id) or "").strip() or model_id
        base = vllm_worker.ensure_service(vcfg, model_path=path, served_name=model_id)
    except Exception:
        return None
    if not base:
        return None
    return Candidate(key=f"local:{model_id}", kind="local",
                     base=base.rstrip("/") + "/v1", priority=0, primary=True)


def model_entry(model_id: str) -> dict:
    """The model's own entry in models.json, or {}."""
    try:
        from codai.admin.routes import config_manager
        md = getattr(config_manager, "models_data", {}) or {}
        for lst in md.values():
            if not isinstance(lst, list):
                continue
            for m in lst:
                if isinstance(m, dict) and str(m.get("id") or "") == model_id:
                    return m
    except Exception:
        return {}
    return {}


def _pinned_engines(model_id: str) -> set:
    """The engines this model is PINNED to, lower-cased, or an empty set.

    ``engine`` on a model entry is what the front router already treats as a hard
    constraint — the model runs there or not at all. A cluster node is an engine
    by that name, so this is also how an operator says "serve it on box2".
    """
    pin = str((model_entry(model_id) or {}).get("engine") or "").strip().lower()
    if not pin:
        return set()
    # A comma-separated pin names several (so OCR can spread over them); the
    # front treats the whole string as one engine name, so a single pin — the
    # common case — behaves identically in both places.
    return {part.strip() for part in pin.split(",") if part.strip()}


def _node_candidates(model_id: str) -> List[Candidate]:
    """The cluster nodes this model is PINNED to, that do hold it.

    Nodes are never discovered. Asking every node what it has, and using whoever
    answered, is configuration by probe: it would send pages to a machine nobody
    chose, and silently move them when a different node happened to reply first.
    A node serves OCR because the operator pinned the model to it.

    All the pinned ones are returned, not the first that answers: which of them
    takes a given page is decided per request, by load.
    """
    out: List[Candidate] = []
    pinned = _pinned_engines(model_id)
    if not pinned:
        return out
    try:
        from codai.admin.routes import config_manager
        from codai.cluster.nodes import enabled_nodes, sync_client
        ccfg = getattr(getattr(config_manager, "config", None), "cluster", None)
        if ccfg is None or not getattr(ccfg, "enabled", False):
            return out
        token = (getattr(ccfg, "token", "") or "").strip()
        for spec in enabled_nodes(ccfg):
            if str(getattr(spec, "name", "") or "").strip().lower() not in pinned:
                continue
            if token and not spec.api_key:
                spec.api_key = token
            root = spec.url.rstrip("/")
            base = root if root.endswith("/v1") else root + "/v1"
            try:
                # sync_client carries the node's token and trust settings but no
                # base_url, so the address is spelled out.
                with sync_client(spec) as client:
                    reply = client.get(base + "/models")
                    if reply.status_code != 200:
                        continue
                    ids = {str(m.get("id") or "")
                           for m in (reply.json().get("data") or [])}
            except Exception:
                continue
            if model_id in ids:
                out.append(Candidate(key=f"node:{spec.name or root}", kind="node",
                                     base=base, api_key=spec.api_key, priority=1,
                                     node_url=root, primary=True))
            else:
                # Pinned to a node that does not hold it: say so, rather than
                # quietly serving the page from somewhere the operator did not
                # choose (or renting a card for it).
                print(f"[ocr] {model_id!r} is pinned to node {spec.name!r}, which "
                      f"does not list it — skipping that node", flush=True)
    except Exception:
        return out
    return out


def _pod_candidate(model_id: str) -> Optional[Candidate]:
    """The model's RunPod pool, as a lazy candidate.

    Nothing is rented here: a pool that exists is a candidate, and the renting
    happens in :meth:`PodPool.acquire` only when a request actually picks it.
    """
    try:
        from codai.models import manager as mm
        if not mm.runpod_should_handle(model_id):
            return None
        from codai.backends.runpod import RunpodBackend
        backend = RunpodBackend(mm.get_active_runpod_config())
        backend.load_model(model_id)
    except Exception:
        return None

    def _acquire():
        base, release = backend._acquire_endpoint()
        auth = (backend._headers or {}).get("Authorization", "")
        key = auth[7:].strip() if auth[:7].lower() == "bearer " else ""
        # _unreachable() drops the pod the backend just handed out, which is
        # what stops the next page choosing the same dead one.
        return base, key, release, backend._unreachable

    # `backend: runpod` on the entry means the pool IS this model's home, so it
    # comes before a node that happens to list the same id.
    return Candidate(key=f"pod:{model_id}", kind="pod", base="", priority=2,
                     acquire=_acquire, primary=True)


def _spill_pod_candidate(model_id: str) -> Optional[Candidate]:
    """A LOCAL model's burst-to-RunPod target, as a pod candidate.

    ``runpod_spillover`` on a model entry is honoured by the front proxy for
    requests passing through it. An attached OCR engine bypasses the proxy, so the
    same intent is read here: the operator said "when the local one is busy, rent".
    """
    try:
        from codai.admin.routes import config_manager
        from codai.api.runpod_worker import (get_model_pod_pool, parse_model_runpod)
        entry = model_entry(model_id)
        if not entry:
            return None
        sp = entry.get("runpod_spillover")
        if not isinstance(sp, dict) or not sp.get("enabled"):
            return None
        # on_busy is exactly this situation. A spillover configured only for
        # on_no_gpu / on_local_error is not a load policy, so it is not used here.
        if not (sp.get("on_busy") or sp.get("on_concurrency_full")):
            return None
        target = sp.get("target") if isinstance(sp.get("target"), dict) else {}
        if (target.get("mode") or "serverless").lower() != "pods":
            return None
        rp = getattr(getattr(config_manager, "config", None), "runpod", None)
        if rp is None or not getattr(rp, "enabled", False):
            return None
        mcfg = parse_model_runpod(target)
        pool = get_model_pod_pool(model_id, entry, target)
    except Exception:
        return None
    if pool is None:
        return None

    def _acquire():
        pod, url = pool.acquire()
        key = getattr(pool, "api_key", "") or ""
        return (url.rstrip("/") + "/v1", key, lambda: pool.release(pod),
                lambda: pool.discard(pod, "connection refused"))

    return Candidate(key=f"spill:{model_id}", kind="pod", base="", priority=3,
                     acquire=_acquire)


def candidates(model_id: str) -> List[Candidate]:
    """Everywhere this model can be served right now, cheapest hop first."""
    model_id = (model_id or "").strip()
    now = time.time()
    with _lock:
        hit = _candidates.get(model_id)
        if hit and hit[0] > now:
            return list(hit[1])

    found: List[Candidate] = []
    local = _local_candidate(model_id)
    if local is not None:
        found.append(local)
    found.extend(_node_candidates(model_id))
    pod = _pod_candidate(model_id)
    if pod is not None:
        found.append(pod)
    spill = _spill_pod_candidate(model_id)
    if spill is not None:
        found.append(spill)

    with _lock:
        _candidates[model_id] = (now + CANDIDATES_TTL_S, list(found))
    return found


# -------------------------------------------------------------------- picking #
def _node_reported_load(cand: Candidate) -> int:
    """What a node says its whole install is doing, cached briefly.

    A node serves more than our OCR pages; counting only what we sent it would
    route onto a node that is saturated with someone else's work.
    """
    if cand.kind != "node" or not cand.node_url:
        return 0
    now = time.time()
    with _lock:
        hit = _node_load.get(cand.node_url)
        if hit and hit[0] > now:
            return int(hit[1])
    load = 0
    try:
        from codai.admin.routes import config_manager
        from codai.cluster.nodes import enabled_nodes, sync_client
        ccfg = getattr(getattr(config_manager, "config", None), "cluster", None)
        for spec in enabled_nodes(ccfg):
            if spec.url.rstrip("/") != cand.node_url:
                continue
            with sync_client(spec) as client:
                reply = client.get(cand.node_url + "/cluster/state")
                if reply.status_code == 200:
                    load = int(reply.json().get("inflight") or 0)
            break
    except Exception:
        load = 0
    with _lock:
        _node_load[cand.node_url] = (now + NODE_LOAD_TTL_S, load)
    return load


def _load_of(cand: Candidate) -> int:
    return inflight(cand.key) + _node_reported_load(cand)


def choose(model_id: str, spill_after: int = 0) -> Lease:
    """A lease on the least-loaded server for the model.

    ``spill_after`` is the in-flight count at which a fixed server (the local one,
    a node) is considered busy enough to prefer renting instead — the
    local-plus-spillover policy, expressed where OCR can act on it. 0 means never
    spill on load alone; a pool is then used only when it is all there is.
    """
    cands = candidates(model_id)
    if not cands:
        raise OcrError(
            f"model {model_id!r} is not served over HTTP by this install, so an OCR "
            f"engine cannot attach to it. Pin the model to a backend that has an "
            f"endpoint (backend: \"runpod\", a cluster node that holds it, or a "
            f"service_url), or pick a serve mode that fits this box.",
            status=503)

    last_exc = None

    # 1. A pool that is this model's configured home. `backend: runpod` means
    #    "run it on RunPod", so that is where it runs — and because every request
    #    goes through acquire(), the pool sees the real concurrency, spreads the
    #    pages over the pods it has and grows when they are all at their
    #    threshold. This is "on runpod only", scaling.
    for cand in [c for c in cands if c.primary and c.acquire is not None]:
        lease = _take(cand)
        if lease is not None:
            return lease
        last_exc = "pool unavailable"

    # 2. Fixed servers — this machine and the cluster nodes that hold the model —
    #    compete on load, so a head with two nodes sends each page to whichever is
    #    least busy, counting what the node itself reports so a node loaded with
    #    someone else's work is left alone.
    fixed = sorted((c for c in cands if c.acquire is None),
                   key=lambda c: (_load_of(c), 0 if c.primary else 1, c.priority))
    pools = sorted((c for c in cands if c.acquire is not None),
                   key=lambda c: (c.priority, _load_of(c)))
    if fixed:
        best = fixed[0]
        if not pools or spill_after <= 0 or _load_of(best) < spill_after:
            _inc(best.key)
            return Lease(best.base, best.api_key, best.key, best.kind)

    # 3. Everything we own is busy: rent. This is "local + spillover" — the
    #    overflow goes to a pool instead of queueing behind a saturated GPU.
    for cand in pools:
        lease = _take(cand)
        if lease is not None:
            return lease
        last_exc = "pool unavailable"

    if fixed:
        # Busy, and nothing could be rented (no budget, no capacity): the busy
        # server still serves, queued, rather than failing the page.
        best = fixed[0]
        _inc(best.key)
        return Lease(best.base, best.api_key, best.key, best.kind)

    raise OcrError(f"model {model_id!r}: no OCR server could be reached "
                   f"({last_exc})", status=503)


def _take(cand: Candidate) -> Optional[Lease]:
    """Acquire a pool candidate, or None when it cannot provide a server."""
    try:
        got = cand.acquire()
    except Exception as exc:
        print(f"[ocr] {cand.key} cannot serve this page: {exc}", flush=True)
        return None
    # A pool may also hand back a way to say "this one is dead"; older
    # three-value forms simply have none.
    base, key, release = got[0], got[1], got[2]
    dead = got[3] if len(got) > 3 else None
    _inc(cand.key)
    return Lease(base, key, cand.key, cand.kind, release, dead)


def spill_threshold(model_id: str) -> int:
    """In-flight requests at which a server we own is busy enough to rent instead.

    This is the model's own ``scale_up_inflight_per_pod`` — the number already
    configured, per model, to mean "one server should not hold more than this
    before another is needed". Reusing it keeps one knob per model instead of a
    parallel OCR-shaped one, and it is already in the model page.

    0 when the model has no RunPod block to burst to: nothing to rent, so load
    alone never changes where a page goes.
    """
    entry = model_entry(model_id)
    for block in (entry.get("runpod"),
                  (entry.get("runpod_spillover") or {}).get("target")):
        if isinstance(block, dict):
            try:
                n = int(block.get("scale_up_inflight_per_pod") or 0)
            except (TypeError, ValueError):
                n = 0
            if n > 0:
                return n
            if block is entry.get("runpod") or block:
                return 4          # the documented default for a pool that exists
    return 0


def any_candidate(model_id: str) -> bool:
    """Whether anything at all can serve the model — without renting a thing."""
    try:
        return bool(candidates(model_id))
    except Exception:
        return False
