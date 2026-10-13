"""A warm floor satisfied by a machine that no longer exists.

`keep_warm` exists to pay for an idle GPU so that the first real request does
not wait out a cold boot — on the Digesta orchestrator, ~12 minutes of image
pull and warm-up. It stopped doing that whenever a pod went away behind the
pool's back: terminated from the RunPod console, reclaimed as spot, lost with
its host. The pool kept the floor "satisfied" indefinitely and said nothing.

The cause is one line that was never written: nothing re-checks a pod's health.
A `PodHandle.healthy` flag goes false only when something tried to USE the pod
and failed (`discard`), or when the pool itself retired it. A warm-floor pod is
precisely the pod with no traffic on it, so there is no use to fail — and
`maintain()` counted it, found `healthy_n >= keep_min`, and never topped up and
never had anything to log. (`RunpodPodPool`'s own docstring claimed the scaler
"re-checks health" for months. It never has.)

The fix reconciles pools against the account listing the stale-pod reaper
already fetches every ~30 seconds: authoritative, free, and incapable of
mistaking a busy pod for a dead one. What these tests pin is the caution around
it, because the failure mode in the other direction is forgetting a pod that is
alive and billing: only positive evidence acts, an absence needs two listings in
a row, a failed listing is not an empty one, and recent traffic outvotes the
API.
"""

import pathlib
import sys
import threading

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codai.api import runpod_worker as rw                          # noqa: E402


def _pool(key="pool:digesta-qwen38", min_pods=1, pods=()):
    pool = rw.RunpodPodPool.__new__(rw.RunpodPodPool)
    pool._cv = threading.Condition()
    pool.model_key = key
    pool._closed = False
    pool._provisioning = False
    pool._surplus_since = 0.0
    pool._release_pending = False
    pool._sched_closed_logged = False
    pool.pods = list(pods)
    cfg = rw.RunpodModelConfig()
    cfg.min_pods = min_pods
    cfg.keep_warm = min_pods >= 1
    cfg.idle_timeout_s = 300
    cfg.schedule_enabled = False
    pool.mcfg = cfg
    pool.effective_min_pods = lambda: min_pods
    pool.account = None
    return pool


def _pod(pid, last_used=0.0, inflight=0, hourly_usd=0.59, gpu="A40"):
    return rw.PodHandle(pod_id=pid, url=f"https://{pid}:22000", hourly_usd=hourly_usd,
                        started_at=0.0, gpu=gpu, inflight=inflight,
                        last_used=last_used)


def _install(monkeypatch, pool, ledger=None, unregistered=None):
    """Put `pool` in the module's pool table and stub the two side effects
    `forget` has: the spend ledger and the shared registry."""
    monkeypatch.setattr(rw, "_pools", {pool.model_key: pool}, raising=False)
    monkeypatch.setattr(rw, "_provisioning_ids", set(), raising=False)
    monkeypatch.setattr(rw, "_absent_before", set(), raising=False)
    monkeypatch.setattr(rw, "unregister_pod",
                        lambda pid: (unregistered if unregistered is not None
                                     else []).append(pid))

    class _Ledger:
        @staticmethod
        def record_spend(key, cost, meta):
            if ledger is not None:
                ledger.append((key, round(cost, 4), meta.get("pod_id")))

    monkeypatch.setitem(sys.modules, "codai.api.runpod_ledger", _Ledger)


# -------------------------------------------- what the pool knows on its own #
def test_nothing_in_the_pool_ever_rechecks_a_pods_health():
    """The whole bug in one assertion. If a health probe is ever added, this
    test should fail and be replaced — but it must not be *assumed*."""
    src = (ROOT / "codai/api/runpod_worker.py").read_text()
    body = src.split("def maintain(self)")[1].split("\n    def ")[0]
    assert "healthy = True" not in body          # it never revives one either
    # The only ways a handle leaves a pool: use failed, retirement, reconcile.
    assert "reconcile" in src


def test_a_dead_warm_pod_keeps_the_floor_satisfied_without_reconcile():
    """Before the fix, this was the end of the story: one healthy-looking
    handle, floor of 1, nothing to do, nothing to say."""
    pool = _pool(min_pods=1, pods=[_pod("ghost")])
    healthy = len([p for p in pool.pods if p.healthy])
    assert healthy == 1 and healthy >= pool.effective_min_pods()


# ------------------------------------------------- only positive evidence acts #
def test_an_explicitly_terminated_pod_is_dropped_at_once(monkeypatch):
    """RunPod saying TERMINATED is RunPod telling us directly. No confirmation
    round, because there is nothing a second listing could add."""
    pool = _pool(pods=[_pod("ghost")])
    _install(monkeypatch, pool)
    gone = rw.reconcile_pools({"ghost": "TERMINATED"})
    assert gone == {"ghost"}
    assert pool.pods == []


def test_every_finished_status_counts_as_finished(monkeypatch):
    for status in ("TERMINATED", "EXITED", "DEAD", "FAILED", "terminated"):
        pool = _pool(pods=[_pod("ghost")])
        _install(monkeypatch, pool)
        assert rw.reconcile_pools({"ghost": status}) == {"ghost"}, status


def test_a_running_pod_is_left_alone(monkeypatch):
    pool = _pool(pods=[_pod("good")])
    _install(monkeypatch, pool)
    assert rw.reconcile_pools({"good": "RUNNING"}) == set()
    assert [p.pod_id for p in pool.pods] == ["good"]


# ------------------------------------------------- absence needs two listings #
def test_one_listing_without_the_pod_is_not_enough(monkeypatch):
    """The RunPod API is eventually consistent and pages; a pod can be missing
    from one response and running perfectly well. Believing it costs a second
    rented GPU for work already being done."""
    pool = _pool(pods=[_pod("maybe")])
    _install(monkeypatch, pool)
    assert rw.reconcile_pools({"other": "RUNNING"}) == set()
    assert [p.pod_id for p in pool.pods] == ["maybe"]


def test_two_listings_without_it_settle_it(monkeypatch):
    pool = _pool(pods=[_pod("maybe")])
    _install(monkeypatch, pool)
    rw.reconcile_pools({"other": "RUNNING"})
    assert rw.reconcile_pools({"other": "RUNNING"}) == {"maybe"}
    assert pool.pods == []


def test_a_pod_that_reappears_starts_the_count_again(monkeypatch):
    """A flicker must not accumulate into a death over hours."""
    pool = _pool(pods=[_pod("flicker")])
    _install(monkeypatch, pool)
    rw.reconcile_pools({})                       # absent once
    rw.reconcile_pools({"flicker": "RUNNING"})   # back
    assert rw.reconcile_pools({}) == set()       # absent once again, not twice
    assert [p.pod_id for p in pool.pods] == ["flicker"]


def test_recent_traffic_outvotes_the_listing(monkeypatch):
    """A pod that answered a request seconds ago exists. This guard cannot mask
    the bug being fixed: a warm-floor pod has no traffic to speak for it."""
    import time
    pool = _pool(pods=[_pod("busy", last_used=time.time())])
    _install(monkeypatch, pool)
    rw.reconcile_pools({})
    assert rw.reconcile_pools({}) == set()
    assert [p.pod_id for p in pool.pods] == ["busy"]


def test_traffic_does_not_save_a_pod_runpod_calls_terminated(monkeypatch):
    import time
    pool = _pool(pods=[_pod("busy", last_used=time.time(), inflight=3)])
    _install(monkeypatch, pool)
    assert rw.reconcile_pools({"busy": "TERMINATED"}) == {"busy"}


def test_a_pod_still_provisioning_is_shielded(monkeypatch):
    """It may not be in the listing yet; the pool is mid-create."""
    pool = _pool(pods=[_pod("booting")])
    _install(monkeypatch, pool)
    monkeypatch.setattr(rw, "_provisioning_ids", {"booting"}, raising=False)
    rw.reconcile_pools({})
    assert rw.reconcile_pools({}) == set()
    assert [p.pod_id for p in pool.pods] == ["booting"]


# --------------------------------------------------- a failed listing is not data #
def test_a_listing_that_failed_never_reaches_the_reconcile(monkeypatch):
    """`reap_orphans` returns on RunpodError before anything is judged.
    Otherwise an API outage would empty every pool and re-rent the fleet."""
    from codai.api import runpod_client

    class _Acct:
        enabled, api_key = True, "k"

    monkeypatch.setattr("codai.models.manager.get_active_runpod_config",
                        lambda: _Acct, raising=False)

    def _boom(self):
        raise runpod_client.RunpodError("api down")

    monkeypatch.setattr(runpod_client.RunpodClient, "list_pods", _boom, raising=False)
    monkeypatch.setattr(runpod_client.RunpodClient, "__init__",
                        lambda self, acct: None, raising=False)
    called = []
    monkeypatch.setattr(rw, "reconcile_pools",
                        lambda *a, **k: called.append(a) or set())
    assert rw.reap_orphans() == 0
    assert called == []


# ------------------------------------------------------------- closing the bill #
def test_forgetting_a_pod_does_not_call_the_api(monkeypatch):
    """There is nothing left to terminate, and the call would only error."""
    pool = _pool(pods=[_pod("ghost")])
    _install(monkeypatch, pool)

    class _Client:
        def __init__(self, acct):
            raise AssertionError("forget must not talk to RunPod")

    monkeypatch.setattr("codai.api.runpod_client.RunpodClient", _Client)
    rw.reconcile_pools({"ghost": "TERMINATED"})
    assert pool.pods == []


def test_the_bill_is_still_recorded_and_the_registry_cleaned(monkeypatch):
    """A pod that died out of band was billed right up to its death. Dropping
    the handle without a ledger entry would make the GPU look free."""
    ledger, unregistered = [], []
    pod = _pod("ghost", hourly_usd=0.59)
    pod.started_at = 0.0
    pool = _pool(pods=[pod])
    _install(monkeypatch, pool, ledger=ledger, unregistered=unregistered)
    rw.reconcile_pools({"ghost": "TERMINATED"})
    assert unregistered == ["ghost"]
    assert len(ledger) == 1
    key, cost, pid = ledger[0]
    assert (key, pid) == (pool.model_key, "ghost") and cost > 0


def test_a_borrowed_pod_is_dropped_but_not_billed_here(monkeypatch):
    """Its owner holds the lifecycle and the bill; double-counting a shared pod
    would charge one GPU to two pools."""
    ledger = []
    pool = _pool(pods=[_pod("shared", gpu="(shared)")])
    _install(monkeypatch, pool, ledger=ledger)
    rw.reconcile_pools({"shared": "TERMINATED"})
    assert pool.pods == [] and ledger == []


def test_forget_is_idempotent(monkeypatch):
    """Two sweeps racing on the same corpse must bill it once."""
    ledger = []
    pod = _pod("ghost")
    pool = _pool(pods=[pod])
    _install(monkeypatch, pool, ledger=ledger)
    assert pool.forget(pod, "status TERMINATED") is True
    assert pool.forget(pod, "status TERMINATED") is False
    assert len(ledger) == 1


# ----------------------------------------------------- and then it re-rents #
def test_the_floor_is_re_established_once_the_ghost_is_gone(monkeypatch):
    """The point of the whole exercise: maintain() can now see the pool is
    below its floor, and says so before it spends."""
    pool = _pool(min_pods=1, pods=[_pod("ghost")])
    _install(monkeypatch, pool)
    rw.reconcile_pools({"ghost": "TERMINATED"})

    rented = []
    pool._adopt_shared_pod = lambda: False
    pool._provision_one = lambda: rented.append("rented")
    pool._drain_pod_log = lambda p: None
    pool._scale_down = lambda now: None
    monkeypatch.setattr(rw, "sibling_is_provisioning", lambda key: False)
    out = []
    monkeypatch.setattr("builtins.print", lambda *a, **k: out.append(" ".join(map(str, a))))
    pool.maintain()
    monkeypatch.undo()
    assert rented == ["rented"]
    assert any("warm floor 1" in line and "renting one" in line for line in out), out


def test_a_sibling_already_renting_is_one_pod_not_none(monkeypatch):
    """Topping up against a booting pod is how a warm floor of 1 became two."""
    pool = _pool(min_pods=1, pods=[])
    _install(monkeypatch, pool)
    rented = []
    pool._adopt_shared_pod = lambda: False
    pool._provision_one = lambda: rented.append("rented")
    pool._drain_pod_log = lambda p: None
    pool._scale_down = lambda now: None
    monkeypatch.setattr(rw, "sibling_is_provisioning", lambda key: True)
    pool.maintain()
    assert rented == []


def test_an_existing_pod_elsewhere_is_adopted_before_renting(monkeypatch):
    """Cheaper than a GPU, and avoids paying twice for one job."""
    pool = _pool(min_pods=1, pods=[])
    _install(monkeypatch, pool)
    rented = []
    pool._adopt_shared_pod = lambda: True
    pool._provision_one = lambda: rented.append("rented")
    pool._drain_pod_log = lambda p: None
    pool._scale_down = lambda now: None
    monkeypatch.setattr(rw, "sibling_is_provisioning", lambda key: False)
    pool.maintain()
    assert rented == []


def test_no_floor_means_no_re_rent(monkeypatch):
    """A pod dying out of band on a min_pods=0 pool is simply gone. Renting a
    replacement nobody asked for is how an idle install bills overnight."""
    pool = _pool(min_pods=0, pods=[_pod("ghost")])
    _install(monkeypatch, pool)
    rw.reconcile_pools({"ghost": "TERMINATED"})
    rented = []
    pool._adopt_shared_pod = lambda: False
    pool._provision_one = lambda: rented.append("rented")
    pool._drain_pod_log = lambda p: None
    pool._scale_down = lambda now: None
    monkeypatch.setattr(rw, "sibling_is_provisioning", lambda key: False)
    pool.maintain()
    assert rented == []


# ------------------------------------------------------- the registry as well #
def test_registry_entries_for_vanished_pods_are_dropped_too(monkeypatch):
    """They are shielded from reaping by being "known", so without this they
    accumulate for the life of the install — and a stale one with a URL is
    handed out by find_shared_pod as if it were a live pod."""
    pool = _pool(min_pods=0, pods=[])
    _install(monkeypatch, pool)
    rw.reconcile_pools({}, candidates={"orphan"})
    assert rw.reconcile_pools({}, candidates={"orphan"}) == {"orphan"}
