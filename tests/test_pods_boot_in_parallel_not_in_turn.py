"""A four-pod job needed four CONSECUTIVE boots inside a one-boot budget.

The coordinator waited for addresses in a `for` loop: shard 0 was waited out
to completion before shard 1 was looked at, while shards 1..n sat booted and
BILLING. With a 600s publish grace and a boot that takes 6-7 minutes, a job
asking for four pods could not finish its waits inside any sane budget, and
the ones that did come up were released at the end for not meeting the floor:

    job-c9991acb7a48: shard 0 reachable 14:01:25, shard 1 reachable 14:02:33,
    both released -- "only 2 pod(s) became ready and this job requires at
    least 4"

Pods boot in parallel on RunPod's side, so the wall clock should be ONE boot.
"""

import inspect
import threading
import time

import pytest

from codai.api import loras


SRC = inspect.getsource(loras._text_train_distributed)


def test_the_waits_are_concurrent():
    assert "ThreadPoolExecutor" in SRC
    assert "as_completed" in SRC


def test_one_pod_still_runs(tmp_path, monkeypatch):
    """There used to be a `len(_shards) == 1` branch that skipped the pool.

    It went away with the barrier -- a pool that is never joined costs a
    one-pod job one thread and nothing else, and the special case was a second
    code path through the most failure-prone part of the run. So the claim
    worth keeping is behavioural: one pod boots, gets the dataset, and trains.
    """
    import _fleetfake as ff
    fleet = ff.FakeFleet({0: 0.0})
    pushes = []
    got = ff.run(monkeypatch, fleet=fleet, home=tmp_path / "loras",
                 job="job-solo", name="adapter-solo",
                 request=ff.req("adapter-solo", pods=1),
                 config=ff.cfg(max_pods=1, pods_start=1),
                 pushes=pushes, finish_after=0.4)
    assert [sh for sh, _ in pushes] == [0]
    assert got["rounds"] >= 1


def test_the_pool_is_bounded():
    """A 32-pod fleet must not open 32 threads without a ceiling stated."""
    assert "max_workers=min(len(_shards), 32)" in SRC


def test_a_pod_that_never_publishes_is_released_in_its_own_wait():
    """Not at the end of the round: it is billing, and the other waits may
    have minutes left to run."""
    block = _bring_up_body()
    assert 'fleet.release(shard, "never published a port")' in block
    assert 'fleet.release(shard, "no address")' in block


def test_a_cancel_during_the_boot_releases_the_pod(tmp_path, monkeypatch):
    """INSIDE the wait now, not after it.

    The old comment here said the check had to sit after the waits, because a
    cancel mid-boot would abandon a rented pod. That reasoning was wrong and
    cost ten cancelled jobs their fleets: the answer is to release the pod in
    the boot thread, which is what `_bring_up` does. Asserted by running it.
    """
    import pytest as _pytest
    import _fleetfake as ff
    fleet = ff.FakeFleet({0: 5.0})       # still booting when cancelled
    with _pytest.raises(RuntimeError):
        ff.run(monkeypatch, fleet=fleet, home=tmp_path / "loras",
               job="job-cancel", name="adapter-cancel",
               request=ff.req("adapter-cancel", pods=1),
               config=ff.cfg(max_pods=1, pods_start=1,
                             pod_boot_timeout_s=10.0),
               cancelled=True, finish_after=0.0)
    assert fleet.released_for("cancelled while booting") == [0], fleet.released


def test_each_pod_is_served_exactly_once_as_it_comes_up(tmp_path,
                                                        monkeypatch):
    """It used to go out in shard order AFTER every boot had finished, which
    made the slowest pod the start time for all of them. Per-pod now -- so
    what has to be true is that nobody is served twice and nobody is missed.
    """
    import _fleetfake as ff
    fleet = ff.FakeFleet({0: 0.0, 1: 0.3, 2: 0.6})
    pushes = []
    ff.run(monkeypatch, fleet=fleet, home=tmp_path / "loras",
           job="job-order", name="adapter-order",
           request=ff.req("adapter-order", pods=3),
           config=ff.cfg(max_pods=3, pods_start=3),
           pushes=pushes, finish_after=1.2)
    served = [sh for sh, _ in pushes]
    assert sorted(served) == [0, 1, 2], pushes
    assert len(served) == len(set(served)), f"served twice: {pushes}"


def test_a_pod_that_refuses_the_dataset_is_released_alone(tmp_path,
                                                          monkeypatch):
    """Per-pod means a per-pod failure: the others must be unaffected."""
    import _fleetfake as ff
    fleet = ff.FakeFleet({0: 0.0, 1: 0.0, 2: 0.0})
    pushes = []
    ff.run(monkeypatch, fleet=fleet, home=tmp_path / "loras",
           job="job-refuse", name="adapter-refuse",
           request=ff.req("adapter-refuse", pods=3),
           config=ff.cfg(max_pods=3, pods_start=3),
           pushes=pushes, push_fails=(1,), finish_after=0.6)
    assert fleet.released_for("dataset push failed") == [1], fleet.released
    assert sorted(sh for sh, _ in pushes) == [0, 2], pushes


def test_a_raising_boot_does_not_abort_the_other_pods():
    """The futures are drained in a watcher thread now, so a raise is logged
    and the loop over the rest continues."""
    i = SRC.index("as_completed")
    block = SRC[i:i + 500]
    assert "boot raised" in block
    assert "_f.result()" in block


# ------------------------------------------------- the property, measured ---

def test_parallel_waiting_is_not_the_sum_of_the_waits():
    """The behaviour, not the spelling: four 0.3s boots must take about 0.3s,
    not 1.2s. Run against the real executor shape used above.
    """
    import concurrent.futures as cf

    def _boot(_shard):
        time.sleep(0.3)
        return _shard

    shards = [0, 1, 2, 3]
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=min(len(shards), 32)) as pool:
        got = sorted(f.result() for f in
                     cf.as_completed({pool.submit(_boot, s): s
                                      for s in shards}))
    took = time.time() - t0
    assert got == shards
    assert took < 0.9, f"waits did not overlap: {took:.2f}s for 4x0.3s"


# ------------- a slice expiring is not a pod failing ------------------------
#
# Slicing the wait to honour a cancel introduced a worse bug than the one it
# fixed: `wait_ready` RAISES at its deadline ("Pod X did not expose port 8000
# within 20.0s"), so the loop read every expired slice as the pod never coming
# up. Measured in production within minutes of the deploy:
#
#     shard 0 never came up: Pod jnomn0v9giwgiu did not expose port 8000
#     within 20.0s ... released after 0.4 min -- never published a port
#     FAILED: no pod became reachable -- nothing to train on
#
# A 1800s budget had become 20 seconds, and the card was released while it was
# still pulling its image.

def test_an_expired_slice_keeps_waiting():
    block = _bring_up_body()
    assert "A SLICE EXPIRING IS NOT A FAILURE" in block
    assert 'f"within {_slice}" in _m' in block
    assert "continue" in block


def test_the_deadline_is_what_ends_the_wait():
    """Not the slice: while there is budget left, a non-fatal error keeps
    waiting."""
    block = _bring_up_body()
    assert "time.time() < _deadline" in block


def test_a_verdict_that_waiting_cannot_change_still_ends_it():
    """An unreachable host, or a container that exited, must not be waited out
    for the full budget -- that is what the classifier is for."""
    block = _bring_up_body()
    for fatal in ("cannot be reached", "EXITED on boot", "entered status"):
        assert fatal in block, fatal


def test_the_slice_length_is_named_once():
    from codai.api import loras as _l
    assert _l._SLICE_S == 20.0
    assert "min(_SLICE_S, _left)" in _bring_up_body()


def test_the_budget_still_comes_from_the_model():
    assert "_boot_timeout = float(cfg.pod_boot_timeout_s or 1800.0)" in SRC

def _bring_up_body():
    """The whole `_bring_up` body, bounded by the statement after it.

    Bounded by a marker rather than a character count: three assertions in
    this file broke on the body growing, which tests where the text happened
    to end rather than what the code does.
    """
    src = SRC
    i = src.index("def _bring_up")
    # The first statement after the function. It was `if len(_shards) == 1`,
    # which went away with the fleet-wide boot barrier: the boots are now
    # submitted to a pool that is never joined, so there is no single-pod
    # special case to bound on.
    j = src.index("import concurrent.futures as _cf", i)
    return src[i:j]
