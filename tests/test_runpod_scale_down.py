"""A fleet that grew for a burst has to give the pods back.

Scaling up was the only half that existed. The pool adds a pod once every pod it
has carries `scale_up_inflight_per_pod` requests, and pods went away again only
through the idle reaper — which fires on a pod with NO work and no request for
`idle_timeout_s`.

That leaves the expensive case uncovered. A pool that grew to absorb a burst and
is now serving a trickle keeps every pod: each one is touched often enough that
`last_used` never ages out, each is a fraction busy, and the bill stays at the
peak for as long as the trickle lasts. With `max_pods: 64` on an A40 that is the
difference between a few dollars an hour and a few hundred.

So the pool also asks how many pods the CURRENT concurrency justifies — the
scale-up arithmetic read backwards — and retires the excess. The rules that
matter are not "fewer pods": they are that work is never dropped to save money,
that a momentary dip does not cost a pod needed seconds later, and that the warm
floor is still a floor.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codai.api import runpod_worker as rw                         # noqa: E402


def _pool(max_pods=8, min_pods=0, threshold=4, after=180, pods=()):
    """A PodPool with its scaling state and nothing else."""
    pool = rw.RunpodPodPool.__new__(rw.RunpodPodPool)
    import threading
    pool._cv = threading.Condition()
    pool.model_key = "datalab-to/surya-ocr-2"
    pool._closed = False
    pool._provisioning = False
    pool._surplus_since = 0.0
    pool.pods = list(pods)
    cfg = rw.RunpodModelConfig()
    cfg.max_pods = max_pods
    cfg.min_pods = min_pods
    cfg.scale_up_inflight_per_pod = threshold
    cfg.scale_down_after_s = after
    cfg.schedule_enabled = False
    pool.mcfg = cfg
    pool.effective_min_pods = lambda: min_pods
    killed = []
    pool._terminate = lambda pod, why: killed.append((pod.pod_id, why))
    pool._killed = killed
    return pool


def _pod(pid, inflight=0, last_used=0.0, draining=False):
    return rw.PodHandle(pod_id=pid, url=f"http://{pid}", hourly_usd=0.6,
                        started_at=0.0, inflight=inflight, last_used=last_used,
                        draining=draining)


def _settle(pool, now=1000.0):
    """Two passes: the first notices the surplus, the second acts on it once the
    window has passed. That delay is the point, so the tests go through it."""
    pool._scale_down(now)
    pool._scale_down(now + pool.mcfg.scale_down_after_s + 1)


# ------------------------------------------------------------ the arithmetic #
def test_how_many_pods_the_load_needs():
    """One pod per scale_up_inflight_per_pod requests — up-scaling, backwards."""
    pool = _pool(threshold=4)
    assert pool._pods_needed([_pod("a", 4), _pod("b", 4)]) == 2
    assert pool._pods_needed([_pod("a", 1), _pod("b", 0)]) == 1
    assert pool._pods_needed([_pod("a", 5), _pod("b", 0)]) == 2   # ceil, not floor


def test_no_traffic_needs_no_pods_but_the_floor_still_holds():
    assert _pool(min_pods=0).\
        _pods_needed([_pod("a", 0)]) == 0
    assert _pool(min_pods=2)._pods_needed([_pod("a", 0)]) == 2


def test_it_never_asks_for_more_than_the_ceiling():
    """The ceiling is the budget."""
    assert _pool(max_pods=3, threshold=1)._pods_needed([_pod("a", 99)]) == 3


# ------------------------------------------------------------- shedding pods #
def test_a_fleet_larger_than_the_load_sheds_one():
    pool = _pool(pods=[_pod("busy", 4), _pod("idle", 0), _pod("spare", 0)])
    _settle(pool)
    assert len(pool._killed) == 1
    assert pool._killed[0][1] == "scaled down"
    assert "busy" in [p.pod_id for p in pool.pods]


def test_only_one_pod_goes_per_pass():
    """Each retirement changes the arithmetic, and a fleet that shed everything
    on one quiet moment would cold-start for minutes on the next burst."""
    pool = _pool(pods=[_pod("a", 1), _pod("b", 0), _pod("c", 0), _pod("d", 0)])
    _settle(pool)
    assert len(pool._killed) == 1
    assert len(pool.pods) == 3


def test_the_least_loaded_pod_is_the_one_that_goes():
    pool = _pool(threshold=4, pods=[_pod("heavy", 3), _pod("light", 1),
                                    _pod("empty", 0)])
    _settle(pool)
    assert pool._killed == [("empty", "scaled down")]


# --------------------------------------------- never drop work to save money #
def test_a_pod_with_work_is_drained_not_killed():
    """Killing a pod mid-request would make the caller pay for that work twice —
    once on the pod that died and once on the retry."""
    pool = _pool(threshold=10, pods=[_pod("a", 1), _pod("b", 2)])
    _settle(pool)
    assert pool._killed == []
    assert sum(1 for p in pool.pods if p.draining) == 1
    assert [p.pod_id for p in pool.pods if p.draining] == ["a"]


def test_a_drained_pod_is_terminated_once_it_empties():
    """And straight away, rather than waiting out idle_timeout_s — which is the
    cost it was taken out of service to stop paying."""
    draining = _pod("old", 2, draining=True)
    pool = _pool(threshold=10, pods=[_pod("a", 1), draining])
    draining.inflight = 0
    pool._scale_down(1000.0)
    assert ("old", "scaled down") in pool._killed
    assert "old" not in [p.pod_id for p in pool.pods]


def test_a_draining_pod_is_given_no_new_work():
    pool = _pool(pods=[_pod("drain", 0, draining=True), _pod("live", 3)])
    chosen = pool._pick([p for p in pool.pods if p.healthy], "")
    assert chosen.pod_id == "live"


def test_a_draining_pod_is_not_mistaken_for_spare_capacity():
    """_should_grow reads the LEAST loaded pod. An empty drainer would answer
    "no need to grow" with the one pod that cannot take the work."""
    pool = _pool(max_pods=4, threshold=4,
                 pods=[_pod("drain", 0, draining=True), _pod("live", 4)])
    assert pool._should_grow([p for p in pool.pods if p.healthy]) is True


def test_everything_draining_still_counts_as_needing_a_pod():
    pool = _pool(max_pods=4, pods=[_pod("drain", 1, draining=True)])
    assert pool._should_grow([p for p in pool.pods if p.healthy]) is True


# ----------------------------------------------------- not on a momentary dip #
def test_a_surplus_must_persist_before_it_costs_a_pod():
    """A burst that dips for a second must not cost a pod that is wanted again a
    second later: a cold one takes minutes to replace."""
    pool = _pool(after=180, pods=[_pod("a", 1), _pod("b", 0)])
    pool._scale_down(1000.0)                     # first sighting
    assert pool._killed == []
    pool._scale_down(1000.0 + 179)               # still inside the window
    assert pool._killed == []
    pool._scale_down(1000.0 + 181)
    assert len(pool._killed) == 1


def test_pressure_returning_cancels_the_countdown():
    pool = _pool(threshold=4, after=180, pods=[_pod("a", 0), _pod("b", 0)])
    pool._scale_down(1000.0)
    pool.pods[0].inflight = 4                    # the burst is back
    pool.pods[1].inflight = 4
    pool._scale_down(1100.0)
    assert pool._surplus_since == 0.0
    pool._scale_down(2000.0)                     # and the clock starts over
    assert pool._killed == []


def test_the_window_restarts_after_each_retirement():
    """Otherwise one long quiet spell would walk the whole fleet down in a single
    pass, which is the same mistake as shedding everything at once."""
    pool = _pool(pods=[_pod("a", 1), _pod("b", 0), _pod("c", 0)])
    _settle(pool)
    assert len(pool._killed) == 1
    pool._scale_down(1000.0 + pool.mcfg.scale_down_after_s + 2)
    assert len(pool._killed) == 1                # not yet: the window restarted


# ------------------------------------------------------------- the floor holds #
def test_the_warm_floor_is_never_scaled_through():
    """min_pods means "keep this many ready", and an idle pool is exactly when
    that promise matters."""
    pool = _pool(min_pods=2, pods=[_pod("a", 0), _pod("b", 0)])
    _settle(pool)
    assert pool._killed == []
    assert len(pool.pods) == 2


def test_zero_disables_shedding_entirely():
    """An install that would rather hold its pods can say so, and then only the
    idle reaper takes them."""
    pool = _pool(after=0, pods=[_pod("a", 1), _pod("b", 0)])
    _settle(pool)
    assert pool._killed == []


def test_a_closed_pool_sheds_nothing():
    pool = _pool(pods=[_pod("a", 1), _pod("b", 0)])
    pool._closed = True
    _settle(pool)
    assert pool._killed == []


def test_a_pool_with_nothing_in_flight_is_left_to_the_idle_reaper():
    """The important boundary. The idle reaper walks a quiet pool down to the
    warm floor on idle_timeout_s — a number the operator chose, weighing the bill
    against a cold start. Shedding here as well would quietly replace 900s with
    scale_down_after_s and make that setting a lie."""
    pool = _pool(min_pods=0, pods=[_pod("a", 0), _pod("b", 0), _pod("c", 0)])
    _settle(pool)
    assert pool._killed == []
    assert len(pool.pods) == 3


def test_one_request_is_enough_for_shedding_to_apply():
    """The pool is serving, just not with this many pods — which is exactly the
    case the idle reaper cannot see, because every pod looks recently used."""
    pool = _pool(threshold=4, pods=[_pod("a", 1), _pod("b", 0)])
    _settle(pool)
    assert len(pool._killed) == 1
