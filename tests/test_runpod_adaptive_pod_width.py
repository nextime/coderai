"""A client may offer any concurrency; the pool must only ever send a pod what
that pod will take.

`max_inflight_per_pod` defaults to 0 -- unlimited -- so out of the box the pool
dispatches every page it is handed. A client that offers more than a pod can
serve therefore does NOT queue in the orchestrator, where the queue would be
counted, reported as demand and bounded. It queues in the POD'S LISTEN BACKLOG,
and an overflowing backlog makes a perfectly healthy pod refuse connections.

Measured 2026-10-19: a client offering 64 pages per pod, against pods serving
~32 (`Running: 29-32 reqs, Waiting: 0`), produced refusals that were read as
dead pods and cost eight A40s. Nothing in any configuration said 64 was too
many, and nothing could have: the right number is a property of the pod, the
model and the card, not of the install.

So the pod is asked. A refusal halves what we send it, FROM THE IN-FLIGHT COUNT
AT THE REFUSAL -- the observed breaking point rather than a guess -- and
successes walk it back up one slot at a time. Multiplicative decrease, additive
increase, because the errors are not symmetric: overshooting costs refused
connections and a misdiagnosed fleet, undershooting costs a little latency.

The properties pinned here:

  1. a refusal narrows that pod, and narrows it from what it was carrying
  2. the narrowing is what `_pick` enforces, or it is decoration
  3. it never collapses to uselessness, and never exceeds the operator's own
     ceiling
  4. an install this never fires on behaves exactly as before
  5. a narrowed pod recovers by itself
"""
import threading
import time

import pytest

from codai.api import runpod_worker as rw


def _pool(max_inflight=0, max_pods=8):
    class _Acct:
        enabled = True

    p = rw.RunpodPodPool.__new__(rw.RunpodPodPool)
    p._cv = threading.Condition()
    p.model_key = "m"
    p._closed = False
    p._booting = 0
    p._waiting = 0
    p._surplus_since = 0.0
    p._denied = []
    p.pods = []
    p._affinity = {}
    cfg = rw.RunpodModelConfig()
    cfg.max_pods = max_pods
    cfg.max_inflight_per_pod = max_inflight
    p.mcfg = cfg
    p._bump_cards = lambda: None
    p._say_no_pod = lambda *a, **k: None
    return p


def _pod(pod_id="p1", inflight=0):
    return rw.PodHandle(pod_id=pod_id, url="https://10.0.0.1:2222",
                        hourly_usd=0.59, started_at=time.time(), gpu="A40",
                        inflight=inflight)


# --- 1. the pod's own behaviour sets its width ---------------------------

def test_a_refusal_narrows_this_pod():
    p = _pool(max_inflight=0)
    pod = _pod(inflight=58)
    p.pods = [pod]
    new = p.note_refusal(pod)
    assert new == 29, "halve the OBSERVED load, not a configured number"
    assert pod.soft_limit == 29


def test_the_live_numbers():
    """64 configured, 58 in flight when it refused, pods serving ~32."""
    p = _pool(max_inflight=64)
    pod = _pod(inflight=58)
    p.pods = [pod]
    p.note_refusal(pod)
    assert p.pod_capacity_now(pod) == 29
    assert 24 <= p.pod_capacity_now(pod) <= 34, "lands near the measured width"


def test_a_second_refusal_narrows_further():
    p = _pool()
    pod = _pod(inflight=32)
    p.pods = [pod]
    assert p.note_refusal(pod) == 16
    pod.inflight = 16
    assert p.note_refusal(pod) == 8


def test_a_refusal_with_nothing_in_flight_still_narrows():
    """The counter can be stale; the refusal is still real."""
    p = _pool()
    pod = _pod(inflight=0)
    p.pods = [pod]
    assert p.note_refusal(pod) >= rw.SOFT_LIMIT_FLOOR


def test_narrowing_is_per_pod():
    """One overloaded pod must not throttle the whole fleet."""
    p = _pool()
    a, b = _pod("a", inflight=40), _pod("b")
    p.pods = [a, b]
    p.note_refusal(a)
    assert p.pod_capacity_now(a) == 20
    assert p.pod_capacity_now(b) == 0, "untouched pod keeps the pool's setting"


# --- 2. _pick enforces it -----------------------------------------------

def test_pick_refuses_a_pod_that_is_at_its_discovered_width():
    p = _pool(max_inflight=0)
    pod = _pod(inflight=40)
    p.pods = [pod]
    p.note_refusal(pod)                 # width now 20, inflight 40
    assert p._pick([pod], "") is None, "dispatched past the discovered width"


def test_pick_uses_a_pod_below_its_discovered_width():
    p = _pool(max_inflight=0)
    pod = _pod(inflight=40)
    p.pods = [pod]
    p.note_refusal(pod)
    pod.inflight = 5
    assert p._pick([pod], "") is pod


def test_pick_reads_the_narrowed_capacity_not_the_raw_one():
    import inspect
    src = inspect.getsource(rw.RunpodPodPool._pick)
    assert "self.pod_capacity_now(p)" in src
    assert "cap = self.pod_capacity(p)" not in src


# --- 3. bounded both ends ------------------------------------------------

def test_it_never_collapses_to_uselessness():
    """A pod worth one page at a time cannot drain a queue."""
    p = _pool()
    pod = _pod(inflight=2)
    p.pods = [pod]
    for _ in range(20):
        p.note_refusal(pod)
        pod.inflight = pod.soft_limit
    assert pod.soft_limit == rw.SOFT_LIMIT_FLOOR
    assert rw.SOFT_LIMIT_FLOOR >= 2


def test_it_never_exceeds_the_operator_ceiling():
    """The discovered width narrows the configured one; it cannot widen it."""
    p = _pool(max_inflight=16)
    pod = _pod(inflight=8)
    p.pods = [pod]
    pod.soft_limit = 999
    assert p.pod_capacity_now(pod) <= 16


def test_recovery_stops_at_the_operator_ceiling():
    p = _pool(max_inflight=16)
    pod = _pod()
    p.pods = [pod]
    pod.soft_limit = 16
    for _ in range(rw.SOFT_LIMIT_RECOVER_EVERY * 3):
        pod.inflight = 1
        p.release(pod)
    assert pod.soft_limit == 16


# --- 4. nothing changes for an install this never fires on --------------

def test_an_untouched_pod_is_unbounded_when_the_operator_said_so():
    p = _pool(max_inflight=0)
    pod = _pod(inflight=10000)
    p.pods = [pod]
    assert p.pod_capacity_now(pod) == 0
    assert p._pick([pod], "") is pod


def test_an_untouched_pod_keeps_its_configured_capacity():
    p = _pool(max_inflight=32)
    pod = _pod()
    p.pods = [pod]
    assert p.pod_capacity_now(pod) == p.pod_capacity(pod)


def test_release_does_not_invent_a_limit():
    """A pod that was never narrowed must not acquire a width from succeeding."""
    p = _pool(max_inflight=0)
    pod = _pod(inflight=1)
    p.pods = [pod]
    for _ in range(rw.SOFT_LIMIT_RECOVER_EVERY * 2):
        pod.inflight = 1
        p.release(pod)
    assert pod.soft_limit == 0


# --- 5. a narrowed pod recovers -----------------------------------------

def test_successes_walk_the_width_back_up():
    p = _pool(max_inflight=64)
    pod = _pod(inflight=32)
    p.pods = [pod]
    p.note_refusal(pod)
    assert pod.soft_limit == 16
    for _ in range(rw.SOFT_LIMIT_RECOVER_EVERY):
        pod.inflight = 1
        p.release(pod)
    assert pod.soft_limit == 17


def test_recovery_is_additive_not_immediate():
    """Jumping straight back to the configured number would re-provoke the
    limit that was just discovered."""
    p = _pool(max_inflight=64)
    pod = _pod(inflight=32)
    p.pods = [pod]
    p.note_refusal(pod)
    for _ in range(rw.SOFT_LIMIT_RECOVER_EVERY - 1):
        pod.inflight = 1
        p.release(pod)
    assert pod.soft_limit == 16, "moved before enough pages had succeeded"


def test_a_refusal_resets_the_recovery_counter():
    p = _pool(max_inflight=64)
    pod = _pod(inflight=32)
    p.pods = [pod]
    p.note_refusal(pod)
    for _ in range(rw.SOFT_LIMIT_RECOVER_EVERY - 1):
        pod.inflight = 1
        p.release(pod)
    pod.inflight = 16
    p.note_refusal(pod)
    assert pod.since_limit_change == 0


# --- the operator can see a narrowed fleet ------------------------------

def test_the_snapshot_reports_the_width_in_force():
    """A fleet running at half its configured width is the first thing to know
    when throughput is below expectation."""
    src = open("codai/api/runpod_worker.py").read()
    assert '"soft_limit": int(getattr(p, "soft_limit", 0) or 0)' in src
    assert '"capacity": pool.pod_capacity_now(p)' in src


def test_discard_learns_the_width_before_setting_the_pod_aside():
    """Afterwards it would read an in-flight count the quarantine had already
    begun to unwind, and learn the wrong number."""
    import inspect
    src = inspect.getsource(rw.RunpodPodPool.discard)
    assert src.index("note_refusal") < src.index("quarantined_until = now")
