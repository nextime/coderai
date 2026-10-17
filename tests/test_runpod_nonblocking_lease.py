"""A lease must not wait in a thread.

0.4.18 bounded a lease attempt at 20 s and ran it on a 256-thread pool, which
made the LEASE STAGE the ceiling on everything downstream: 256 threads at one
attempt per 20 s passes ~13 pages a second however many pods are up, and a
thread that drew the growth branch sat through an entire ~9-minute boot.

Measured consequence, on a live 5,000-document run: **zero** pages dispatched
to 50 healthy pods in 41 minutes. Not one `POST /v1/ocr` in any pod's log, GPUs
at `KV 0.0%`, 811 pages waiting, and the scaler reading that unservable queue
as demand and renting to 73 pods at $43/hr. `/healthz` answered in 1.6 ms
throughout, so nothing looked broken from outside.

So `acquire(grow_inline=False)` answers now or raises `NoPodRightNow`, and the
caller waits on its event loop, where waiting is free. The pool keeps a demand
signal by counting DENIALS over a window equal to the caller's retry interval,
so the count is distinct waiting pages rather than the attempts they have made.
"""

import threading
import time
import types

import pytest

from codai.api import runpod_worker as rw


def _mcfg(**kw):
    base = dict(max_pods=128, min_pods=0, max_inflight_per_pod=0,
                scale_up_inflight_per_pod=32, scale_up_max_parallel_boots=32,
                gpu_types=["NVIDIA A40"], gpu_weights={"NVIDIA A40": 1.0},
                gpu_limits=None, sticky_sessions=True, health_path="/v1/models",
                env={})
    base.update(kw)
    return types.SimpleNamespace(**base)


def _pool(mcfg=None, pods=()):
    p = rw.RunpodPodPool.__new__(rw.RunpodPodPool)
    p.mcfg = mcfg or _mcfg()
    p._cv = threading.Condition(threading.RLock())
    p._card_cache, p._card_gen, p._denied = {}, 0, []
    p._booting = p._waiting = 0
    p._affinity = {}
    p.model_key = "m"
    p.pods = list(pods)
    p.grew = 0

    def _grow():
        p.grew += 1
    p._grow_in_background = _grow
    return p


def _pod(inflight=0, healthy=True, draining=False):
    return types.SimpleNamespace(gpu="NVIDIA A40", vram_gb=48.0,
                                 inflight=inflight, healthy=healthy,
                                 draining=draining, pod_id="p",
                                 url="https://10.0.0.1:1", last_used=0.0)


# ------------------------------------------------------------ answers now #

def test_an_empty_pool_raises_at_once_instead_of_renting_here():
    p = _pool()
    t0 = time.monotonic()
    with pytest.raises(rw.NoPodRightNow):
        p.acquire(timeout=600.0, grow_inline=False)
    assert time.monotonic() - t0 < 0.5, "it waited"
    assert p.grew == 1, "growth must still be triggered, in the background"


def test_it_does_not_hold_the_thread_for_the_timeout():
    """THE REGRESSION: 20 s per attempt x 256 threads = ~13 pages/s."""
    p = _pool()
    t0 = time.monotonic()
    for _ in range(50):
        with pytest.raises(rw.NoPodRightNow):
            p.acquire(timeout=1200.0, grow_inline=False)
    took = time.monotonic() - t0
    assert took < 1.0, f"50 denials took {took:.1f}s — still waiting somewhere"


def test_a_free_pod_is_still_handed_out():
    p = _pool(pods=[_pod()])
    pod, url = p.acquire(timeout=600.0, grow_inline=False)
    assert pod is not None and url
    assert pod.inflight == 1


def test_the_booting_count_is_not_leaked_when_it_raises():
    """A leaked count stops `_grow_reason` firing for everyone."""
    p = _pool()
    for _ in range(5):
        with pytest.raises(rw.NoPodRightNow):
            p.acquire(timeout=600.0, grow_inline=False)
    assert p._booting == 0


def test_blocking_mode_is_untouched_for_callers_that_want_it():
    """A caller with a thread to spare and no retry loop still waits."""
    p = _pool(_mcfg(max_pods=0), pods=[_pod()])
    pod, _url = p.acquire(timeout=5.0)
    assert pod is not None


# ------------------------------------------------- demand without threads #

def test_a_denial_is_counted_as_demand():
    p = _pool()
    with pytest.raises(rw.NoPodRightNow):
        p.acquire(grow_inline=False)
    assert p._queued() >= 1


def test_retries_of_one_page_are_not_counted_as_many_pages():
    """The runaway: reading its own retries as new work rented 73 pods."""
    p = _pool()
    p.DENY_WINDOW_S = 0.2
    for _ in range(20):                 # one page, retrying fast
        with pytest.raises(rw.NoPodRightNow):
            p.acquire(grow_inline=False)
        time.sleep(0.03)
    # Within one window only the attempts made in that window count.
    assert p._queued() <= 10, p._queued()


def test_the_demand_decays_when_the_pages_stop_asking():
    p = _pool()
    p.DENY_WINDOW_S = 0.1
    with pytest.raises(rw.NoPodRightNow):
        p.acquire(grow_inline=False)
    assert p._queued() >= 1
    time.sleep(0.25)
    assert p._queued() == 0


def test_threads_and_denials_are_not_added_together():
    """The same page counted two ways would double the fleet."""
    p = _pool()
    p._waiting = 7
    for _ in range(3):
        with pytest.raises(rw.NoPodRightNow):
            p.acquire(grow_inline=False)
    assert p._queued() == max(7, p._denied_recent())


def test_the_record_does_not_grow_without_bound():
    p = _pool()
    for _ in range(5000):
        p._note_denied()
    assert len(p._denied) <= 4096


def test_demand_still_drives_the_fleet_size():
    """With no threads parked, denials must still rent pods."""
    p = _pool(pods=[_pod(inflight=1)])
    for _ in range(40):
        p._note_denied()
    assert p._pods_for_demand([p.pods[0]]) > 1
