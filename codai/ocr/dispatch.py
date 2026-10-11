# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
# GPLv3 - see the project LICENSE.

"""Which server should serve THIS OCR page — asked once per request.

:mod:`codai.ocr.served_model` answers "where is this model served" once and hands
the answer to the OCR engine, which then speaks to that one address for as long as
it lives. That is the right shape for fidelity (the engine drives the server
directly, so nothing rewrites its images, its json_schema or its logprobs) and the
wrong shape for load:

* a RunPod pool grows when its least-loaded pod reaches
  ``scale_up_inflight_per_pod`` in-flight requests, and in-flight is counted in
  :meth:`RunpodBackend._acquire_endpoint`. An attached engine never goes through
  it, so the pool saw zero load however hard OCR was running, never grew, and the
  engine would not have used a second pod anyway — it had one URL;
* a model served on several cluster nodes resolved to whichever node answered
  first, for ever, however busy that node became;
* a local model that was configured to burst to RunPod under load never did: the
  spillover trigger lives on the proxy path the engine bypasses.

So the engine is given a stable address that belongs to US, and this module picks
the upstream per request: the least loaded of the local server, the cluster nodes
that hold the model, and the pod pool — borrowing a pod through
:meth:`PodPool.acquire` so the pool sees the real concurrency and grows on it.

The fidelity argument survives because :mod:`codai.ocr.gateway` forwards bytes.
The earlier losses came from coderai *parsing* a request and re-emitting it through
the backend abstraction; a hop that copies the body verbatim decides nothing.
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
                 release: Optional[Callable] = None):
        self.base = base
        self.api_key = api_key
        self.key = key
        self.kind = kind
        self._release = release
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


def _node_candidates(model_id: str) -> List[Candidate]:
    """Every enabled cluster node whose own catalogue lists the model.

    All of them, not the first one that answers: which node serves a given page
    is decided per request, by load.
    """
    out: List[Candidate] = []
    try:
        from codai.admin.routes import config_manager
        from codai.cluster.nodes import enabled_nodes, sync_client
        ccfg = getattr(getattr(config_manager, "config", None), "cluster", None)
        if ccfg is None or not getattr(ccfg, "enabled", False):
            return out
        token = (getattr(ccfg, "token", "") or "").strip()
        for spec in enabled_nodes(ccfg):
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
                                     node_url=root))
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
        return base, key, release

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
        md = getattr(config_manager, "models_data", {}) or {}
        entry = None
        for lst in md.values():
            if not isinstance(lst, list):
                continue
            for m in lst:
                if isinstance(m, dict) and str(m.get("id") or "") == model_id:
                    entry = m
                    break
        if entry is None:
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
        return (url.rstrip("/") + "/v1", key, lambda: pool.release(pod))

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
        base, key, release = cand.acquire()
    except Exception as exc:
        print(f"[ocr] {cand.key} cannot serve this page: {exc}", flush=True)
        return None
    _inc(cand.key)
    return Lease(base, key, cand.key, cand.kind, release)


def any_candidate(model_id: str) -> bool:
    """Whether anything at all can serve the model — without renting a thing."""
    try:
        return bool(candidates(model_id))
    except Exception:
        return False
