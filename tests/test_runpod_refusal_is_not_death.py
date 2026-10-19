"""A refused connection is backpressure, not a verdict that the pod is gone.

`discard()` terminated the pod, on the reasoning that a connection can only be
refused by a pod that no longer exists. A pod whose listen backlog is full
refuses connections while being perfectly alive and busy -- and refuses them on
/healthz too, so the "is anything still there?" probe added to protect against
exactly this agrees that it has vanished.

Observed 2026-10-19, mid-campaign, ~58 concurrent connections per pod:

    08:36:13  pod 2n69g0vt47h0hl terminated (connection refused); billed ~$0.417
    08:36:30  pod c79lv31ayh28l1 terminated (connection refused); billed ~$0.449
    ... eight healthy A40s in 69 seconds, 522 pages in flight on them ...
    08:37:22  pod n9fpbskd7owl2h terminated (connection refused); billed ~$0.458

The warm floor then re-rented eight more, which booted ~12 minutes, began
refusing, and were killed again: $3.72 of pods billed -- more than the whole
campaign's spend -- on boots that never produced a page. Measured throughput
0.577 pages/s against 0.557 PER POD.

What these tests pin, in order of what getting it wrong costs:

  1. a refusal NEVER terminates. This is the defect; everything else is detail.
  2. a pod with work in flight is never terminated by this path at all, however
     many probes fail -- killing it loses those pages and pays for them.
  3. a quarantined pod still counts toward the warm floor, or the pool rents
     ~700 s of boot to cover a 45 s gap and ends up with two cards for one.
  4. a pod that answers again comes back by itself.
  5. a pod that really is dead is still eventually terminated -- the fix must
     not turn a leak into the new failure mode.
"""
import threading
import time
import types

import pytest

from codai.api import runpod_worker as rw


def _pool(max_pods=8, min_pods=0):
    class _Acct:
        enabled = True

    p = rw.RunpodPodPool.__new__(rw.RunpodPodPool)
    p._cv = threading.Condition()
    p.model_key = "datalab-to/surya-ocr-2"
    p._closed = False
    p._booting = 0
    p._waiting = 0
    p._surplus_since = 0.0
    p._denied = []
    p.pods = []
    p.account = _Acct()
    p.api_key = "k"
    cfg = rw.RunpodModelConfig()
    cfg.max_pods = max_pods
    cfg.min_pods = min_pods
    p.mcfg = cfg
    p._bump_cards = lambda: None
    p._terminate = lambda pod, reason: _TERMINATED.append((pod.pod_id, reason))
    return p


def _pod(pod_id="p1", inflight=0, url="https://10.0.0.1:2222"):
    return rw.PodHandle(pod_id=pod_id, url=url, hourly_usd=0.59,
                        started_at=time.time(), gpu="A40", inflight=inflight)


_TERMINATED: list = []


@pytest.fixture(autouse=True)
def _clear():
    _TERMINATED.clear()
    yield
    _TERMINATED.clear()


# --- 1. the regression: a refusal must not kill ---------------------------

def test_discard_does_not_terminate():
    """The eight A40s. This is the whole point."""
    p = _pool()
    pod = _pod()
    p.pods = [pod]
    p.discard(pod, "connection refused")
    assert _TERMINATED == [], "a refused connection terminated the pod"


def test_discard_keeps_the_pod_in_the_pool():
    """It is still ours and still billing. A pool that forgets it rents a
    replacement for a card it is already paying for."""
    p = _pool()
    pod = _pod()
    p.pods = [pod]
    p.discard(pod, "connection refused")
    assert pod in p.pods


def test_discard_takes_it_out_of_rotation():
    """Right, and unchanged: the next page must not choose it."""
    p = _pool()
    pod = _pod()
    p.pods = [pod]
    p.discard(pod, "connection refused")
    assert pod.healthy is False
    assert pod.quarantined_until > time.time()


def test_discard_is_a_no_op_for_a_pod_we_do_not_hold():
    p = _pool()
    p.discard(_pod("stranger"), "connection refused")
    assert _TERMINATED == []


# --- 2. work in flight is sacred -----------------------------------------

def test_a_pod_with_pages_on_it_is_never_terminated_by_this_path():
    """522 pages were in flight on the pods that were destroyed."""
    p = _pool()
    pod = _pod(inflight=64)
    p.pods = [pod]
    p.discard(pod, "connection refused")
    p._pod_reachable = lambda _p: False
    for _ in range(10):                       # far past the strike limit
        pod.quarantined_until = time.time() - 1
        p._recheck_quarantined(time.time())
    assert _TERMINATED == [], "a pod carrying pages was terminated"
    assert pod in p.pods


def test_in_flight_work_defers_the_probe_rather_than_passing_it():
    """It must not accumulate strikes while it is excused, or it dies the
    moment the last page drains."""
    p = _pool()
    pod = _pod(inflight=1)
    p.pods = [pod]
    p.discard(pod, "connection refused")
    p._pod_reachable = lambda _p: False
    for _ in range(5):
        pod.quarantined_until = time.time() - 1
        p._recheck_quarantined(time.time())
    assert pod.probe_failures == 0


# --- 3. the floor must not rent a replacement ----------------------------

def test_a_quarantined_pod_counts_toward_the_warm_floor():
    """Renting over a 45 s gap costs ~700 s of boot and leaves two cards where
    the floor asked for one -- the re-rent loop, in one line."""
    src = rw.RunpodPodPool.maintain.__doc__ or ""
    import inspect
    body = inspect.getsource(rw.RunpodPodPool.maintain)
    assert ("healthy_in_hand = len([p for p in self.pods\n"
            "                                       if p.healthy or p.quarantined_until])"
            in body), "the warm-floor deficit ignores quarantined pods"


def test_the_recheck_runs_before_anything_sizes_the_fleet():
    import inspect
    body = inspect.getsource(rw.RunpodPodPool.maintain)
    assert body.index("_recheck_quarantined") < body.index("_scale_down")


# --- 4. recovery is automatic -------------------------------------------

def test_a_pod_that_answers_again_returns_to_rotation():
    p = _pool()
    pod = _pod()
    p.pods = [pod]
    p.discard(pod, "connection refused")
    p._pod_reachable = lambda _p: True
    pod.quarantined_until = time.time() - 1
    p._recheck_quarantined(time.time())
    assert pod.healthy is True
    assert pod.quarantined_until == 0.0
    assert pod.probe_failures == 0
    assert _TERMINATED == []


def test_a_quarantine_is_not_rechecked_before_it_expires():
    p = _pool()
    pod = _pod()
    p.pods = [pod]
    p.discard(pod, "connection refused")
    probed = []
    p._pod_reachable = lambda _p: probed.append(1) or True
    p._recheck_quarantined(time.time())        # still cooling
    assert probed == []
    assert pod.healthy is False


def test_recovery_clears_the_strikes_so_later_blips_start_fresh():
    p = _pool()
    pod = _pod()
    p.pods = [pod]
    p.discard(pod, "connection refused")
    p._pod_reachable = lambda _p: False
    pod.quarantined_until = time.time() - 1
    p._recheck_quarantined(time.time())
    assert pod.probe_failures == 1
    p._pod_reachable = lambda _p: True
    pod.quarantined_until = time.time() - 1
    p._recheck_quarantined(time.time())
    assert pod.probe_failures == 0


def test_rediscovering_a_pod_clears_its_quarantine():
    """_add_pod is positive evidence the pod is there, which is what the
    quarantine was waiting for."""
    p = _pool()
    pod = _pod()
    p.pods = [pod]
    p.discard(pod, "connection refused")
    p._add_pod(_pod(pod.pod_id, url=pod.url))
    assert pod.quarantined_until == 0.0
    assert pod.healthy is True


# --- 5. a pod that really is dead still goes ----------------------------

def test_a_pod_that_never_answers_with_no_work_is_terminated():
    """The fix must not turn a terminating bug into a leaking one."""
    p = _pool()
    pod = _pod()
    p.pods = [pod]
    p.discard(pod, "connection refused")
    p._pod_reachable = lambda _p: False
    for _ in range(rw.QUARANTINE_STRIKES):
        pod.quarantined_until = time.time() - 1
        p._recheck_quarantined(time.time())
    assert [t[0] for t in _TERMINATED] == [pod.pod_id]
    assert pod not in p.pods


def test_it_takes_several_strikes_not_one():
    p = _pool()
    pod = _pod()
    p.pods = [pod]
    p.discard(pod, "connection refused")
    p._pod_reachable = lambda _p: False
    pod.quarantined_until = time.time() - 1
    p._recheck_quarantined(time.time())
    assert _TERMINATED == [], "one failed probe is still one failed probe"
    assert rw.QUARANTINE_STRIKES >= 2


def test_the_give_up_window_is_minutes_not_seconds():
    """Long enough that a full backlog has certainly drained."""
    assert rw.QUARANTINE_S * rw.QUARANTINE_STRIKES >= 90


def test_an_unreachable_probe_is_anything_that_does_not_answer():
    """A pod that answers an ERROR exists, and must not be destroyed here."""
    p = _pool()
    pod = _pod()
    p.pods = [pod]
    calls = []

    class _R:
        status_code = 503

    import codai.api.pod_http as ph
    old = ph.get
    ph.get = lambda url, **kw: calls.append(url) or _R()
    try:
        assert p._pod_reachable(pod) is True
    finally:
        ph.get = old
    assert calls and calls[0].endswith("/healthz")


def test_a_probe_that_raises_is_unreachable():
    p = _pool()
    pod = _pod()
    import codai.api.pod_http as ph
    old = ph.get

    def _boom(url, **kw):
        raise OSError("connection refused")
    ph.get = _boom
    try:
        assert p._pod_reachable(pod) is False
    finally:
        ph.get = old


# --- the operator can see it --------------------------------------------

def test_the_snapshot_reports_quarantine_as_its_own_state():
    src = open("codai/api/runpod_worker.py").read()
    assert '"state": ("quarantined"' in src
    assert '"quarantined_for_s"' in src
    assert '"pods_quarantined": len(quarantined)' in src


def test_a_quarantined_pod_is_not_counted_as_booting():
    """It told an operator a settled fleet was still starting, and told a
    client to wait for pods that were already here."""
    import inspect
    body = inspect.getsource(rw.fleet_summary)
    assert 'p.get("state") not in ("ready", "quarantined")' in body
