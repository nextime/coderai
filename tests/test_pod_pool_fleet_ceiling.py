"""The pod thread pool must not be a silent ceiling on fleet size.

A page dispatched to a pod holds a pod-pool thread for its whole life, so the
pool's width -- not ocr.max_concurrency, not max_pods, not the cgroup -- decides
how many pods an orchestrator can actually feed:

    pods_feedable = pod_pool_width / max_inflight_per_pod

With the old _MAX_POD of 2048 and a 64-page-per-pod ceiling that was 32 pods,
whatever max_pods said. An operator who set max_pods=128 got 96 pods' worth of
nothing and no line anywhere explained it. That is a worse failure than a slow
one, because it looks like the pods are the problem.

What must NOT change: the pool is still bounded by the task budget, and the pod
pool still gives way first so the general pool can keep answering /healthz. A
ceiling that protects the box is right; a constant that caps the fleet on a box
with 16,000 spare tasks is not.
"""
import pytest

import codai.api.thread_pool as tp


PER_POD = 64        # max_inflight_per_pod in the measured configuration


def test_the_ceiling_can_feed_a_128_pod_fleet():
    assert tp._MAX_POD >= 128 * PER_POD, (
        "a %d-wide pod pool feeds only %d pods at %d pages each"
        % (tp._MAX_POD, tp._MAX_POD // PER_POD, PER_POD))


def test_the_old_ceiling_would_not_have():
    """Guards the test: 2048 really was a 32-pod limit."""
    assert 2048 // PER_POD == 32


def test_pod_pool_tracks_configured_concurrency(monkeypatch):
    """Two threads per in-flight page, so a page can wait on a pod and still
    have one for the forward."""
    monkeypatch.setattr(tp, "_configured_concurrency", lambda: 4096)
    monkeypatch.setattr(tp, "task_budget", lambda: 0)      # unbounded
    _general, pod = tp.pool_sizes()
    assert pod == min(tp._MAX_POD, 8192)


@pytest.mark.parametrize("want,expect_pods", [(1024, 32), (2048, 64), (4096, 128)])
def test_concurrency_translates_to_fleet_size(monkeypatch, want, expect_pods):
    monkeypatch.setattr(tp, "_configured_concurrency", lambda: want)
    monkeypatch.setattr(tp, "task_budget", lambda: 0)
    _general, pod = tp.pool_sizes()
    assert pod // PER_POD >= expect_pods


# --- the protections that must survive the raise ---

def test_the_task_budget_still_binds(monkeypatch):
    """A small cgroup must still win over the raised constant."""
    monkeypatch.setattr(tp, "_configured_concurrency", lambda: 4096)
    monkeypatch.setattr(tp, "task_budget", lambda: 1024)
    general, pod = tp.pool_sizes()
    assert general + pod <= 1024 - tp._TASK_HEADROOM + tp._MIN_POD, \
        "pools must fit the budget (floors aside)"
    assert pod < tp._MAX_POD, "the constant must not override a tight budget"


def test_the_pod_pool_gives_way_first(monkeypatch):
    """The general pool answers /healthz and the admin pages; an install that
    cannot be looked at cannot be fixed."""
    monkeypatch.setattr(tp, "_configured_concurrency", lambda: 4096)
    monkeypatch.setattr(tp, "task_budget", lambda: 2048)
    general, _pod = tp.pool_sizes()
    assert general >= tp._MIN_DEFAULT


def test_both_floors_survive_a_tiny_budget(monkeypatch):
    monkeypatch.setattr(tp, "_configured_concurrency", lambda: 4096)
    monkeypatch.setattr(tp, "task_budget", lambda: 100)
    general, pod = tp.pool_sizes()
    assert (general, pod) == (tp._MIN_DEFAULT, tp._MIN_POD)


def test_headroom_is_kept(monkeypatch):
    """A pool that spends the last task takes diagnosis away with it."""
    monkeypatch.setattr(tp, "_configured_concurrency", lambda: 100000)
    monkeypatch.setattr(tp, "task_budget", lambda: 4096)
    general, pod = tp.pool_sizes()
    assert general + pod <= 4096 - tp._TASK_HEADROOM
