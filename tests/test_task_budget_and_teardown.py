"""Two failures from one live 5,000-document run, and the budget behind both.

The run rented 48 warm pods on a client lease, then read 278 pages ON THE
ORCHESTRATOR -- a box with no GPU -- while every one of those cards sat idle.
Each page logged "the pod-side pipeline could not run (can't start new
thread)". The cause was a container task budget of 2048 (podman's
`--pids-limit` default) against a pod pool sized 2048 from
`ocr.max_concurrency: 4096`: the pool reached 1992 threads and the next one
could not be created.

Three things were wrong, and they are separable:

  1. The pools sized themselves from a config number without asking what the
     process is allowed. `task_budget()` asks.
  2. A lease took a pod-pool thread per in-flight page, so the width an
     install wants decided how many thread stacks it needs. A lease is a
     bounded wait, so a small pool serves many pages by cycling.
  3. "Cannot create a thread" was treated as "the pods cannot run the
     pipeline", and the fallback for that is the local pipeline -- which needs
     MORE of this box than the path that just failed for want of it.

And separately, from the same run: the client's own teardown sweep answered
`500` five times in three seconds, because releasing a pod that was already
gone raised. A teardown that was working looked like one that had failed.
"""

import types

import pytest

from codai.api import runpod_worker, thread_pool
from codai.ocr import manager as om
from codai.ocr.base import OcrError


# ------------------------------------------------------------ task budget #

def test_the_budget_is_read_and_is_a_number():
    n = thread_pool.task_budget()
    assert isinstance(n, int) and n >= 0


def test_a_tight_budget_shrinks_the_pools(monkeypatch):
    """The live case: 2048 tasks total, 4096 asked for."""
    monkeypatch.setattr(thread_pool, "_configured_concurrency", lambda: 4096)
    monkeypatch.setattr(thread_pool, "task_budget", lambda: 2048 - 30)
    general, pod = thread_pool.pool_sizes()
    assert general + pod <= 2048 - 30 - thread_pool._TASK_HEADROOM
    assert pod >= thread_pool._MIN_POD
    assert general >= thread_pool._MIN_DEFAULT


def test_an_unbounded_budget_leaves_the_sizes_alone(monkeypatch):
    monkeypatch.setattr(thread_pool, "_configured_concurrency", lambda: 4096)
    monkeypatch.setattr(thread_pool, "task_budget", lambda: 0)
    general, pod = thread_pool.pool_sizes()
    assert pod == thread_pool._MAX_POD
    assert general == thread_pool._MAX_DEFAULT


def test_the_general_pool_is_protected_so_the_box_stays_diagnosable(monkeypatch):
    """An install that cannot answer /healthz cannot be looked at."""
    monkeypatch.setattr(thread_pool, "_configured_concurrency", lambda: 4096)
    monkeypatch.setattr(thread_pool, "task_budget", lambda: 1200)
    general, pod = thread_pool.pool_sizes()
    assert general >= thread_pool._MIN_DEFAULT
    assert general + pod <= 1200 - thread_pool._TASK_HEADROOM


def test_an_impossibly_tight_budget_still_starts(monkeypatch):
    """Serving slowly beats not serving."""
    monkeypatch.setattr(thread_pool, "_configured_concurrency", lambda: 4096)
    monkeypatch.setattr(thread_pool, "task_budget", lambda: 40)
    assert thread_pool.pool_sizes() == (thread_pool._MIN_DEFAULT,
                                        thread_pool._MIN_POD)


def test_the_lease_pool_is_not_sized_per_page(monkeypatch):
    """The whole point: page width must not decide thread count."""
    monkeypatch.setattr(thread_pool, "_configured_concurrency", lambda: 100000)
    monkeypatch.setattr(thread_pool, "_lease_pool", None)
    monkeypatch.setattr(thread_pool, "task_budget", lambda: 0)
    ex = thread_pool.lease_executor()
    assert ex._max_workers <= thread_pool._LEASE_WORKERS


def test_the_lease_pool_also_respects_the_budget(monkeypatch):
    monkeypatch.setattr(thread_pool, "_lease_pool", None)
    monkeypatch.setattr(thread_pool, "task_budget", lambda: 400)
    ex = thread_pool.lease_executor()
    assert 16 <= ex._max_workers <= (400 - thread_pool._TASK_HEADROOM) // 8 + 1


# ------------------------------------------- never fall back to this box #

@pytest.mark.parametrize("exc", [
    RuntimeError("can't start new thread"),
    RuntimeError("can not start new thread"),
    MemoryError(),
    OSError("Cannot allocate memory"),
    OSError("Resource temporarily unavailable"),
    OSError("Too many open files"),
])
def test_local_exhaustion_is_recognised(exc):
    assert om._is_local_exhaustion(exc) is True


@pytest.mark.parametrize("exc", [
    RuntimeError("pod answered 503: no model"),
    ValueError("bad PDF"),
    OSError("connection refused"),
    TimeoutError("read timed out"),
])
def test_an_ordinary_failure_still_falls_back(exc):
    """An optimisation that cannot run must cost latency, not an error."""
    assert om._is_local_exhaustion(exc) is False


def test_the_message_names_the_limit_and_both_ways_out():
    """An operator reading it should not have to find this commit."""
    src = open(om.__file__).read()
    i = src.index("out of local resources")
    window = src[i:i + 1400]
    assert "pids-limit" in window and "TasksMax" in window
    assert "max_concurrency" in window
    assert "2048" in window           # podman's default, which is the trap


# --------------------------------------------------- teardown idempotence #

@pytest.mark.parametrize("exc", [
    RuntimeError("404 Not Found"),
    RuntimeError("pod does not exist"),
    RuntimeError("No such pod: abc"),
    RuntimeError("pod has already been terminated"),
    RuntimeError("pod not found"),
])
def test_an_already_gone_pod_is_recognised(exc):
    assert runpod_worker._already_gone(exc) is True


@pytest.mark.parametrize("exc", [
    RuntimeError("500 Internal Server Error"),
    RuntimeError("connection refused"),
    RuntimeError("unauthorized"),
    RuntimeError("rate limited"),
])
def test_a_real_failure_is_still_a_failure(exc):
    """Reporting a live pod as gone leaves it billing."""
    assert runpod_worker._already_gone(exc) is False


def _release_against(monkeypatch, terminate):
    """release_pod with no pools, a stub account and a stubbed terminate."""
    monkeypatch.setattr(runpod_worker, "_pools", {}, raising=False)

    class _Client:
        def __init__(self, _acct):
            pass

        def terminate_pod(self, pid):
            return terminate(pid)

    import codai.api.runpod_client as rc
    monkeypatch.setattr(rc, "RunpodClient", _Client)
    import codai.models.manager as mm
    monkeypatch.setattr(mm, "get_active_runpod_config",
                        lambda: types.SimpleNamespace(enabled=True))
    return runpod_worker.release_pod("podxyz", "released by client")


def test_releasing_a_pod_that_is_already_gone_succeeds(monkeypatch, capsys):
    """THE TEARDOWN BUG: this answered 500, five times in three seconds."""
    def _gone(_pid):
        raise RuntimeError("404 Not Found: pod does not exist")

    out = _release_against(monkeypatch, _gone)
    assert out["already_gone"] is True
    assert out["terminated"] is False
    assert "no longer exists" in out["detail"]
    assert "already gone" in capsys.readouterr().out


def test_releasing_a_live_pod_terminates_it(monkeypatch):
    seen = []
    out = _release_against(monkeypatch, lambda pid: seen.append(pid))
    assert out["terminated"] is True
    assert seen == ["podxyz"]


def test_a_real_termination_failure_still_raises(monkeypatch):
    def _boom(_pid):
        raise RuntimeError("500 upstream error")

    with pytest.raises(RuntimeError):
        _release_against(monkeypatch, _boom)
