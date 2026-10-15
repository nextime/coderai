"""Giving pods back: the three holes a live run fell into.

1. A boot could not be cancelled. `_booting` is an INTEGER, not a list of pod
   objects -- a pod being created does not exist here yet -- so `shed_now`
   could not see one, could not count one and could not stop one. A client that
   asked for a pod, got four on the way and released them was told
   `kept:0, needed:1, floor:1` and paid all four boots out in full.

2. A pod could not be released at all unless a POOL floor could be lowered to
   lose it. `'pool:<name>' has no runpod block` -- from code whose own comment
   says the caller cannot look that key up -- and nothing anywhere could
   terminate one pod by id. An adopted pod, a pod left by a crash, a pod whose
   pool was reconfigured: RunPod console only.

3. Asking for N could produce more than N, because `ensure_ready` counted one
   boot while starting `want` of them.
"""

import pathlib
import sys
import threading

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codai.api import runpod_worker as rw                         # noqa: E402


def _pod(pid, inflight=0, last_used=0.0, draining=False, hourly_usd=0.6,
         started_at=0.0):
    return rw.PodHandle(pod_id=pid, url=f"http://{pid}", hourly_usd=hourly_usd,
                        started_at=started_at, inflight=inflight,
                        last_used=last_used, draining=draining)


def _pool(max_pods=8, min_pods=0, threshold=4, pods=(), booting=0,
          allow_client_scale=True):
    pool = rw.RunpodPodPool.__new__(rw.RunpodPodPool)
    pool._cv = threading.Condition()
    pool.model_key = "pool:digesta-qwen38"
    pool._closed = False
    pool._provisioning = False
    pool._surplus_since = 0.0
    pool._booting = booting
    pool._cancel_epoch = 0
    pool.pods = list(pods)
    cfg = rw.RunpodModelConfig()
    cfg.max_pods = max_pods
    cfg.min_pods = min_pods
    cfg.scale_up_inflight_per_pod = threshold
    cfg.scale_down_after_s = 180
    cfg.schedule_enabled = False
    cfg.allow_client_scale = allow_client_scale
    pool.mcfg = cfg
    pool.effective_min_pods = lambda: min_pods
    pool._killed = []
    pool._terminate = lambda pod, why: pool._killed.append((pod.pod_id, why))
    return pool


# ----------------------------------------------- a boot that can be stopped #
def test_shedding_bumps_the_cancel_epoch():
    pool = _pool(booting=4)
    before = pool._cancel_epoch
    pool.shed_now()
    assert pool._cancel_epoch == before + 1


def test_a_boot_started_before_the_shed_is_cancelled():
    pool = _pool(booting=1)
    epoch = pool._cancel_epoch          # what the boot captured at entry
    assert pool._boot_cancelled(epoch) is False
    pool.shed_now()
    assert pool._boot_cancelled(epoch) is True


def test_a_boot_started_after_the_shed_is_not():
    pool = _pool()
    pool.shed_now()
    epoch = pool._cancel_epoch          # a boot starting now captures this
    assert pool._boot_cancelled(epoch) is False


def test_the_boots_in_flight_are_counted_and_reported():
    """`kept:0, needed:1, floor:1` while four cards were booting said nothing
    had happened. The count is the difference between that and 'four stopped'."""
    pool = _pool(booting=4)
    out = pool.shed_now()
    assert out["cancelled_boots"] == 4


def test_booting_pods_count_as_capacity_in_the_surplus():
    """A pod on the way is capacity arriving. Measuring surplus only against
    READY pods is why a client holding four boots was told it had none."""
    pool = _pool(min_pods=1, pods=[_pod("a"), _pod("b")], booting=2)
    out = pool.shed_now()
    # 2 ready + 2 booting against a floor of 1: the ready surplus goes, the
    # boots are cancelled.
    assert out["cancelled_boots"] == 2
    assert len(out["terminated"]) >= 1


def test_a_shed_with_nothing_booting_reports_zero_not_nothing():
    pool = _pool(min_pods=0, pods=[_pod("a")])
    assert pool.shed_now()["cancelled_boots"] == 0


def test_only_anticipatory_boots_are_cancellable():
    """A boot a request is BLOCKED on is the one thing a shed must not take:
    the request is what the pod is for. Only the warm floor and background
    growth pass cancellable."""
    src = (ROOT / "codai/api/runpod_worker.py").read_text()
    warm = src.split("def _warm_provision_one")[1].split("\n    def ")[0]
    grow = src.split("def _grow_in_background")[1].split("\n    def ")[0]
    acquire = src.split("def acquire")[1].split("\n    def ")[0]
    ready = src.split("def ensure_ready")[1].split("\n    def ")[0]
    assert "cancellable=True" in warm
    assert "cancellable=True" in grow
    assert "cancellable" not in acquire
    assert "cancellable" not in ready


def test_a_cancelled_boot_is_not_retried_on_another_machine():
    """A boot FAILURE retries elsewhere; a cancellation must not, or the shed
    rents exactly the card it just gave back."""
    src = (ROOT / "codai/api/runpod_worker.py").read_text()
    blk = src.split("def _provision_one")[1].split("\n    def ")[0]
    assert "except BootCancelled:" in blk
    # The BOOT's handler, not the one inside `_give_back`: the retry-on-another
    # -machine path is the one that must not see a cancellation.
    boot = blk.split("self._dump_pod_logs")[0]
    assert boot.rindex("except BootCancelled:") < boot.rindex("except Exception as exc:")
    assert "retrying on the next machine" in blk


def test_a_cancelled_boot_is_benign_at_both_call_sites():
    src = (ROOT / "codai/api/runpod_worker.py").read_text()
    for name in ("_warm_provision_one", "_grow_in_background"):
        blk = src.split(f"def {name}")[1].split("\n    def ")[0]
        assert "except BootCancelled:" in blk, name


# ------------------------------------------------ a pod that can be dropped #
def test_a_live_pool_can_be_addressed_by_the_key_status_reports():
    """`pool:digesta-qwen38` is what /v1/runpod/status calls it, and it was the
    one name the scale endpoint refused."""
    pool = _pool()
    rw._pools["pool:digesta-qwen38"] = pool
    try:
        assert rw.get_live_pool("pool:digesta-qwen38") is pool
        out = rw._scale_live_pool(pool, "pool:digesta-qwen38", 0, teardown=True)
        assert out["model"] == "pool:digesta-qwen38"
        assert "teardown" in out
    finally:
        rw._pools.pop("pool:digesta-qwen38", None)


def test_a_pool_that_did_not_opt_in_is_still_refused():
    pool = _pool(allow_client_scale=False)
    try:
        rw._scale_live_pool(pool, "pool:x", 1)
    except PermissionError as exc:
        assert "allow_client_scale" in str(exc)
    else:
        raise AssertionError("a pool that did not opt in was scaled")


def test_an_unknown_name_says_what_WOULD_work():
    """The old message named a condition the caller could not check."""
    import pytest
    rw._pools.pop("nope", None)
    with pytest.raises(LookupError) as exc:
        rw.set_client_pods("nope", 1)
    msg = str(exc.value)
    assert "no pod pool is running" in msg and "/v1/runpod/status" in msg


def test_get_live_pool_is_exact_not_fuzzy():
    pool = _pool()
    rw._pools["pool:digesta-qwen38"] = pool
    try:
        assert rw.get_live_pool("digesta-qwen38") is None
        assert rw.get_live_pool("") is None
        assert rw.get_live_pool(None) is None
    finally:
        rw._pools.pop("pool:digesta-qwen38", None)


def test_release_pod_needs_a_pod_id():
    import pytest
    with pytest.raises(LookupError):
        rw.release_pod("")


def test_the_release_route_exists_and_is_admin_scoped():
    """Leasing warm pods is bounded and expires; terminating a named pod is
    neither, and it may be serving someone else's request."""
    src = (ROOT / "codai/api/app.py").read_text()
    assert '@app.post("/v1/runpod/release"' in src
    blk = src.split('@app.post("/v1/runpod/release"')[1].split("\n@app.")[0]
    assert "_admin_scoped_token(request)" in blk


# ------------------------------------------- scale stops blocking the server #
def test_scale_runs_off_the_event_loop():
    """It called into the pool from the coroutine, so while it walked dozens of
    (card, region) candidates the whole server was blocked -- and
    /v1/runpod/status stopped answering during exactly the operation someone
    would poll it about. A slow rental looked identical to an outage."""
    src = (ROOT / "codai/api/app.py").read_text()
    blk = src.split("async def runpod_scale(")[1].split("\n@app.")[0]
    assert "asyncio.to_thread" in blk
    assert "runpod_worker.set_client_pods(\n" not in blk


def test_a_slow_scale_answers_pending_rather_than_hanging():
    src = (ROOT / "codai/api/app.py").read_text()
    blk = src.split("async def runpod_scale(")[1].split("\n@app.")[0]
    assert "asyncio.wait_for" in blk and "asyncio.shield" in blk
    assert '"pending": True' in blk and '"request_id"' in blk


def test_the_work_continues_after_the_caller_stops_waiting():
    """`shield` is the whole point: the timeout ends the WAIT, not the scaling.
    Cancelling the task would leave a half-made lease and possibly a pod."""
    src = (ROOT / "codai/api/app.py").read_text()
    blk = src.split("async def runpod_scale(")[1].split("\n@app.")[0]
    assert "asyncio.shield(task)" in blk


def test_a_caller_that_gave_up_can_still_find_out_how_it_ended():
    src = (ROOT / "codai/api/app.py").read_text()
    assert '@app.get("/v1/runpod/scale/{request_id}"' in src


def test_the_job_table_cannot_grow_without_bound():
    from codai.api import app as A
    A._scale_jobs.clear()
    for i in range(A.SCALE_JOBS_MAX + 20):
        A._scale_job_put(f"j{i}", {"id": f"j{i}", "state": "done"})
    assert len(A._scale_jobs) <= A.SCALE_JOBS_MAX
    # The newest survive; the oldest are the ones nobody is coming back for.
    assert f"j{A.SCALE_JOBS_MAX + 19}" in A._scale_jobs
    A._scale_jobs.clear()
