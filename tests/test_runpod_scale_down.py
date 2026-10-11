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


def _pool(max_pods=8, min_pods=0, threshold=4, after=180, pods=(),
          costliest_first=False):
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
    cfg.scale_down_costliest_first = costliest_first
    cfg.schedule_enabled = False
    pool.mcfg = cfg
    pool.effective_min_pods = lambda: min_pods
    killed = []
    pool._terminate = lambda pod, why: killed.append((pod.pod_id, why))
    pool._killed = killed
    return pool


def _pod(pid, inflight=0, last_used=0.0, draining=False, hourly_usd=0.6):
    return rw.PodHandle(pod_id=pid, url=f"http://{pid}", hourly_usd=hourly_usd,
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


# ------------------------------------------------- which of the surplus pods goes #
# A pool rents at whatever the market offered, so one fleet can hold a $0.35 card
# and a $0.79 one doing the same work. Shedding the cheap one saves less than half
# as much, and the operator may want to say so.


def test_by_default_the_least_loaded_pod_goes():
    pool = _pool(threshold=10, pods=[_pod("busy", 3), _pod("quiet", 1)])
    assert pool._scale_down_victim(pool.pods).pod_id == "quiet"


def test_among_equals_the_coldest_cache_goes():
    """Retiring the pod used least recently throws away the least work: its
    prefix cache is the one with the least in it. This tie-break was inverted
    when scale-down was written — it picked the pod used MOST recently."""
    pool = _pool(pods=[_pod("warm", 0, last_used=900.0),
                       _pod("cold", 0, last_used=100.0)])
    assert pool._scale_down_victim(pool.pods).pod_id == "cold"


def test_with_the_rule_on_the_dearest_pod_goes_first():
    pool = _pool(costliest_first=True,
                 pods=[_pod("cheap", 0, hourly_usd=0.35),
                       _pod("dear", 0, hourly_usd=0.79)])
    assert pool._scale_down_victim(pool.pods).pod_id == "dear"


def test_price_leads_even_when_the_dear_pod_is_the_busier_one():
    """It is drained, not dropped — so choosing it costs nothing but time, and
    saves more than shedding the cheap one."""
    pool = _pool(costliest_first=True, threshold=10,
                 pods=[_pod("cheap", 0, hourly_usd=0.35),
                       _pod("dear", 2, hourly_usd=0.79)])
    assert pool._scale_down_victim(pool.pods).pod_id == "dear"


def test_at_the_same_price_the_rule_falls_back_to_the_idlest():
    pool = _pool(costliest_first=True, threshold=10,
                 pods=[_pod("a", 2, hourly_usd=0.6), _pod("b", 0, hourly_usd=0.6)])
    assert pool._scale_down_victim(pool.pods).pod_id == "b"


def test_at_the_same_price_and_load_the_coldest_cache_goes():
    pool = _pool(costliest_first=True,
                 pods=[_pod("warm", 0, last_used=900.0, hourly_usd=0.6),
                       _pod("cold", 0, last_used=100.0, hourly_usd=0.6)])
    assert pool._scale_down_victim(pool.pods).pod_id == "cold"


def test_the_rule_decides_which_pod_goes_never_whether_one_goes():
    """A fleet the load still needs keeps every pod, dear ones included."""
    pool = _pool(costliest_first=True, threshold=4,
                 pods=[_pod("a", 4, hourly_usd=0.79), _pod("b", 4, hourly_usd=0.35)])
    _settle(pool)
    assert pool._killed == []
    assert not any(p.draining for p in pool.pods)


def test_a_dear_pod_with_work_is_still_drained_not_killed():
    pool = _pool(costliest_first=True, threshold=10,
                 pods=[_pod("cheap", 0, hourly_usd=0.35),
                       _pod("dear", 2, hourly_usd=0.79)])
    _settle(pool)
    assert pool._killed == []
    assert [p.pod_id for p in pool.pods if p.draining] == ["dear"]


# ------------------------------------------- a client finishing a batch KNOWS #
# The automatic path waits out scale_down_after_s and retires one pod per pass,
# because it is inferring from a momentary reading and a cold pod costs minutes.
# A client that has just finished a batch is not inferring — and until now it had
# no way to say so: releasing its lease only stopped the pool wanting pods, it
# did not give them back.


def test_a_client_can_shed_the_whole_surplus_at_once():
    """Not one per pass: the window exists to protect against guessing, and this
    caller is not guessing."""
    pool = _pool(pods=[_pod("a", 0), _pod("b", 0), _pod("c", 0), _pod("d", 0)])
    out = pool.shed_now()
    assert len(out["terminated"]) == 4
    assert pool.pods == []


def test_the_operators_floor_still_holds():
    """A client may give back what it asked for, never what the operator pays to
    keep warm."""
    pool = _pool(min_pods=2, pods=[_pod("a", 0), _pod("b", 0), _pod("c", 0)])
    out = pool.shed_now()
    assert len(out["terminated"]) == 1
    assert out["kept"] == 2 and out["floor"] == 2


def test_a_busy_pod_drains_instead_of_dying():
    """Paying for a pod another minute is cheaper than failing someone's page."""
    pool = _pool(threshold=10, pods=[_pod("busy", 3), _pod("idle", 0)])
    out = pool.shed_now()
    assert out["terminated"] == ["idle"]
    assert out["draining"] == [] or "busy" not in out["terminated"]


def test_everything_busy_is_all_drained_not_killed():
    pool = _pool(threshold=10, pods=[_pod("a", 1), _pod("b", 2), _pod("c", 3)])
    out = pool.shed_now()
    assert out["terminated"] == []
    assert len(out["draining"]) == 2          # one pod covers 6 requests at 10
    assert all(p.healthy for p in pool.pods)  # none dropped work


def test_what_the_load_still_needs_is_kept():
    """`teardown` is not "kill everything": it sheds the SURPLUS."""
    pool = _pool(threshold=4, pods=[_pod("a", 4), _pod("b", 4), _pod("c", 0)])
    out = pool.shed_now()
    assert out["kept"] == 2
    assert len(out["terminated"]) == 1


def test_the_cost_rule_applies_to_a_forced_teardown_too():
    pool = _pool(costliest_first=True,
                 pods=[_pod("cheap", 0, hourly_usd=0.35),
                       _pod("dear", 0, hourly_usd=0.79)])
    pool.mcfg.min_pods = 1
    pool.effective_min_pods = lambda: 1
    out = pool.shed_now()
    assert out["terminated"] == ["dear"]


def test_the_surplus_timer_is_reset_so_the_next_pass_starts_clean():
    pool = _pool(pods=[_pod("a", 0), _pod("b", 0)])
    pool._surplus_since = 123.0
    pool.shed_now()
    assert pool._surplus_since == 0.0


# ------------------------------- a release that could not be honoured is kept #
# A client asked for the fleet back while a page was still being read. The pods
# could not go yet — correctly, work is never dropped to save money — but the
# intent was then LOST: the pool fell back to idle_timeout_s (900s on the Digesta
# orchestrator) and billed for a fleet nobody had asked to keep. Seen live:
# "0 terminated, 0 draining, 1 kept", and the pod was still there four minutes
# later with nothing running on it.


def test_a_release_blocked_by_work_is_remembered():
    pool = _pool(threshold=10, pods=[_pod("busy", 2)])
    out = pool.shed_now()
    assert out["kept"] == 1
    assert out["retire_when_idle"] is True
    assert pool._release_pending is True


def test_demand_is_reported_as_demand_not_as_a_floor():
    """The log and the reply said "floor 1" when the 1 came from work in flight,
    which reads as though the operator had asked for the pod."""
    pool = _pool(threshold=10, min_pods=0, pods=[_pod("busy", 2)])
    out = pool.shed_now()
    assert out["needed"] == 1          # demand
    assert out["floor"] == 0           # configuration


def test_nothing_is_remembered_when_the_release_fully_succeeded():
    pool = _pool(pods=[_pod("idle", 0)])
    out = pool.shed_now()
    assert out["retire_when_idle"] is False
    assert pool._release_pending is False


def test_asking_for_pods_again_cancels_the_pending_release():
    """The client changed its mind; retiring what it just asked for is absurd."""
    pool = _pool(threshold=10, pods=[_pod("busy", 2)])
    pool.shed_now()
    assert pool._release_pending is True
    pool.set_client_floor(2)
    assert pool._release_pending is False


def test_releasing_to_zero_leaves_the_intent_standing():
    pool = _pool(threshold=10, pods=[_pod("busy", 2)])
    pool.shed_now()
    pool.set_client_floor(0)
    assert pool._release_pending is True


def test_a_pending_release_collapses_the_idle_timeout():
    """The mechanism: the pod is retired as soon as it falls idle instead of
    waiting out idle_timeout_s — the same treatment a closed schedule window
    already gets, for the same reason."""
    import inspect
    src = inspect.getsource(rw.RunpodPodPool.maintain)
    assert "elif self._release_pending:" in src
    assert "idle_to = min(idle_to, 1)" in src
