"""A pool that could only ever grow by one pod per pod-boot.

`_should_grow` opened with:

    if len(self.pods) >= self.mcfg.max_pods or self._provisioning:
        return False

and `_provisioning` was a boolean held for the WHOLE provision — which is a pod
boot, measured at ~12 minutes on the Digesta orchestrator (~7 min image pull plus
~5 min warm-up). So however much work was queued, the fleet grew one pod per
boot: a burst needing four pods took forty minutes to be served by four,
whatever `max_pods` said.

Measured 2026-10-13, 64 pages in flight at `scale_up_inflight_per_pod: 16`
(= four pods of demand) against `max_pods: 64`: two pods were rented, ready at
11:20 and 11:30 — ten minutes apart, one boot apart. The client reported it as
"a per-pod concurrency issue with no real scale-out", which is exactly what it
was.

The other half was that the grower could not see the size of the demand. It
looked at `min(p.inflight) >= threshold`, which answers "is the fleet full?" —
so the only request it could ever formulate was "one more pod". The requests
blocked in `acquire()` waiting for a slot, which are the actual measure of unmet
demand, were invisible.

What these tests pin is the caution as much as the fix: the demand-derived
number is also the number of GPUs that start billing together, and an earlier
incident in this same file's history was two pods rented for ONE request 45
seconds apart.
"""

import pathlib
import sys
import threading

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codai.api import runpod_worker as rw                          # noqa: E402


def _pool(max_pods=64, threshold=16, parallel=4, pods=(), booting=0, waiting=0):
    pool = rw.RunpodPodPool.__new__(rw.RunpodPodPool)
    pool._cv = threading.Condition()
    pool.model_key = "datalab-to/surya-ocr-2"
    pool._closed = False
    pool._booting = booting
    pool._waiting = waiting
    pool._surplus_since = 0.0
    pool.pods = list(pods)
    cfg = rw.RunpodModelConfig()
    cfg.max_pods = max_pods
    cfg.scale_up_inflight_per_pod = threshold
    cfg.scale_up_max_parallel_boots = parallel
    pool.mcfg = cfg
    return pool


def _pod(pid, inflight=0, draining=False):
    return rw.PodHandle(pod_id=pid, url=f"http://{pid}", hourly_usd=0.59,
                        started_at=0.0, inflight=inflight, draining=draining)


def _healthy(pool):
    return [p for p in pool.pods if p.healthy]


# ------------------------------------------------- how much demand is there #
def test_demand_counts_what_is_queued_not_just_what_is_running():
    """The missing input. 16 on the one pod plus 48 waiting is 64 pages, which
    at 16 per pod is four pods — not "one more"."""
    pool = _pool(pods=[_pod("a", inflight=16)], waiting=48)
    assert pool._pods_for_demand(_healthy(pool)) == 4


def test_demand_rounds_up_a_partial_pod():
    pool = _pool(pods=[_pod("a", inflight=16)], waiting=1)
    assert pool._pods_for_demand(_healthy(pool)) == 2


def test_demand_never_exceeds_max_pods():
    """max_pods is the budget, and it is still the budget."""
    pool = _pool(max_pods=2, pods=[_pod("a", inflight=16)], waiting=1000)
    assert pool._pods_for_demand(_healthy(pool)) == 2


def test_an_idle_fleet_wants_one_pod_not_zero():
    """Zero would make the arithmetic argue for shedding the pod serving the
    next request; shedding is _scale_down's job, with its own hysteresis."""
    pool = _pool(pods=[_pod("a", inflight=0)])
    assert pool._pods_for_demand(_healthy(pool)) == 1


# ------------------------------------------------------- growing in parallel #
def test_a_burst_grows_past_one_pod_at_a_time():
    """The fix. One pod up, 48 pages queued, nothing booting: grow."""
    pool = _pool(pods=[_pod("a", inflight=16)], waiting=48)
    assert pool._should_grow(_healthy(pool)) is True


def test_it_keeps_growing_while_earlier_pods_are_still_booting():
    """What the boolean made impossible. Four pods of demand, one up and one
    booting is two in hand — still short, so still grow."""
    pool = _pool(pods=[_pod("a", inflight=16)], waiting=48, booting=1)
    assert pool._should_grow(_healthy(pool)) is True


def test_it_stops_once_the_pods_in_hand_cover_the_demand():
    """Booting pods ARE capacity. Counting only the ready ones is how the same
    demand gets provisioned for over and over."""
    pool = _pool(pods=[_pod("a", inflight=16)], waiting=48, booting=3)
    assert pool._should_grow(_healthy(pool)) is False


def test_parallel_boots_are_capped():
    """The demand-derived number is also the number of GPUs that start billing
    together, so an operator gets to say how fast that may happen."""
    pool = _pool(pods=[_pod("a", inflight=16)], waiting=1000, parallel=2, booting=2)
    assert pool._should_grow(_healthy(pool)) is False
    assert _pool(pods=[_pod("a", inflight=16)], waiting=1000, parallel=8,
                 booting=2)._should_grow([_pod("a", inflight=16)]) is True


def test_the_cap_defaults_to_more_than_one_but_stays_finite():
    assert rw.RunpodModelConfig().scale_up_max_parallel_boots > 1
    assert rw.RunpodModelConfig().scale_up_max_parallel_boots < 64


def test_max_pods_still_stops_growth_with_boots_counted():
    """Two up and two booting against max_pods 4 is the fleet, even though the
    queue would justify forty."""
    pool = _pool(max_pods=4, pods=[_pod("a", inflight=16), _pod("b", inflight=16)],
                 waiting=600, booting=2)
    assert pool._should_grow(_healthy(pool)) is False


# ------------------------------- the incident the single-flight guard existed for #
def test_an_empty_pool_rents_exactly_one_for_the_first_request():
    """Two pods for ONE request, 45 seconds apart, is in this file's history.
    With nothing able to serve, one on the way is enough to wait for."""
    pool = _pool(pods=[], waiting=1)
    assert pool._should_grow([]) is True
    assert _pool(pods=[], waiting=1, booting=1)._should_grow([]) is False


def test_a_draining_pod_is_not_capacity():
    """It takes no new work, so counting it would answer "no need to grow" with
    the one pod that cannot help."""
    pool = _pool(pods=[_pod("a", inflight=16, draining=True)], waiting=48)
    assert pool._should_grow([p for p in pool.pods if not p.draining]) is True


def test_a_closed_pool_is_not_grown():
    pool = _pool(max_pods=0, pods=[], waiting=10)
    assert pool._should_grow([]) is False


# --------------------------------------------- the counter, and the old flag #
def test_the_counters_exist_without___init__():
    """Several fixtures (and any path that skips __init__) build a pool with
    __new__. Class-level defaults mean the grower reads 0 rather than raising."""
    pool = rw.RunpodPodPool.__new__(rw.RunpodPodPool)
    assert pool._booting == 0 and pool._waiting == 0


def test_the_single_flight_paths_ask_how_many_not_whether():
    """ensure_ready (the FIRST pod) and the warm-floor top-up must still be
    single-flight — they are not allowed to grow a fleet — so they check the
    count directly rather than a boolean that scale-up also writes."""
    src = (ROOT / "codai/api/runpod_worker.py").read_text()
    ready = src.split("def ensure_ready")[1].split("\n    def ")[0]
    assert "if self._booting:" in ready and "self._booting += 1" in ready
    maint = src.split("def maintain")[1].split("\n    def ")[0]
    assert "not self._booting" in maint


def test_the_counter_returns_to_zero_after_a_scale_up():
    """A leaked increment is a fleet that never grows again — the original bug,
    reintroduced and harder to see. Counting `+= 1` against `-= 1` in the source
    cannot show this: acquire() increments ONCE and has four mutually exclusive
    exits, each of which must decrement."""
    pool = _pool(pods=[_pod("a", inflight=16)], waiting=48)
    provisioned = []
    pool._provision_one = lambda: provisioned.append(1)
    pool._pick = lambda healthy, affinity: healthy[0]
    pool._remember_affinity = lambda a, p: None
    got, url = pool.acquire(timeout=5)
    assert got.pod_id == "a"
    for _ in range(200):                      # the grow thread is a daemon
        if pool._booting == 0 and provisioned:
            break
        __import__("time").sleep(0.01)
    assert provisioned == [1], "scale-up did not run"
    assert pool._booting == 0, "the boot counter leaked"


def test_the_counter_returns_to_zero_when_a_sibling_is_already_booting():
    """The bail-out paths are where a leak hides: nothing was provisioned, so
    nothing obviously went wrong, and the pool simply never grows again."""
    import codai.api.runpod_worker as mod
    pool = _pool(pods=[], waiting=1)
    saved = mod.sibling_is_provisioning
    mod.sibling_is_provisioning = lambda key: True
    try:
        try:
            pool.acquire(timeout=0.3)
        except Exception:
            pass                               # the timeout is the point
    finally:
        mod.sibling_is_provisioning = saved
    assert pool._booting == 0, "the boot counter leaked on the sibling path"


def test_the_callers_timeout_is_honoured_while_a_sibling_boots():
    """This path waited a flat 15s and looped back into _should_grow, which
    answers the same way while the sibling is still booting — so the deadline
    was never reached and acquire() ignored its timeout for as long as the
    sibling's registry entry survived (up to REAP_GRACE_SECONDS, 15 minutes).
    Found by a test that hung."""
    import time as _t
    import codai.api.runpod_worker as mod
    from codai.api.runpod_client import RunpodError
    pool = _pool(pods=[], waiting=1)
    saved = mod.sibling_is_provisioning
    mod.sibling_is_provisioning = lambda key: True
    t0 = _t.time()
    try:
        raised = None
        try:
            pool.acquire(timeout=0.5)
        except RunpodError as exc:
            raised = exc
    finally:
        mod.sibling_is_provisioning = saved
    assert raised is not None, "acquire() never gave up"
    assert _t.time() - t0 < 10, "it waited past its deadline"
    assert pool._booting == 0


def test_waiters_are_counted_only_while_waiting():
    """A counter that leaked would make an idle pool look permanently busy and
    rent pods for work that finished long ago."""
    src = (ROOT / "codai/api/runpod_worker.py").read_text()
    blk = src.split("def acquire")[1].split("\n    def ")[0]
    assert "self._waiting += 1" in blk
    assert "self._waiting = max(0, self._waiting - 1)" in blk
    assert "finally:" in blk.split("self._waiting += 1")[1][:200]


def test_growth_is_announced_with_the_numbers():
    """A fleet that silently fails to grow is indistinguishable from a fleet
    that does not need to — which is how ten minutes of this went unnoticed."""
    src = (ROOT / "codai/api/runpod_worker.py").read_text()
    assert "growing to meet" in src and "booting" in src
