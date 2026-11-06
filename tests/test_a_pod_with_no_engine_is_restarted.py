# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""A pod can answer every probe and still serve nothing.

On 2026-11-06 two chandra pods came back from a re-image with their vLLM never
started. They answered /healthz and /internal/engine-state, counted toward the
warm floor as "ready", and failed every page with `Connection refused` on their
own loopback -- for hours, while four pods carried the whole corpus.

The trap in fixing it is that "has served nothing" describes a HEALTHY idle pod
just as well: measured the same day, an unused qwen38 pod reported
`running: 0`, `kv_cache_pct: 0` and `requests_done: 1` while holding its model
in 45.2 GB of VRAM. Keying on idleness would have restarted every warm pod in
the fleet, which is why `loaded_models` is the discriminator and why the idle
case has a test of its own here.
"""
import time

import pytest

from codai.api import runpod_worker as rw


def _pod(**kw):
    d = dict(pod_id="p1", url="https://pod.example/v1", hourly_usd=1.0,
             started_at=time.time() - 10_000.0)
    d.update(kw)
    return rw.PodHandle(**d)


# --- the discriminator ----------------------------------------------------

def test_a_pod_holding_its_model_is_not_engineless():
    assert rw.RunpodPodPool._has_no_engine(
        {"loaded_models": ["datalab-to/chandra-ocr-2"],
         "vram": {"used": 43.2}, "load": {"requests_done": 4289}}) is False


def test_an_idle_pod_that_still_holds_its_model_is_not_engineless():
    """The false positive this check exists to avoid: no work, model resident."""
    assert rw.RunpodPodPool._has_no_engine(
        {"loaded_models": ["Qwen3.8-27B-AWQ-INT4"],
         "vram": {"used": 45.21, "free": 3.1},
         "load": {"running": 0, "kv_cache_pct": 0.0, "requests_done": 1}}) is False


def test_a_pod_with_no_model_and_no_vram_is_engineless():
    assert rw.RunpodPodPool._has_no_engine(
        {"loaded_models": [], "vram": {"used": 0.0},
         "load": {"running": 0, "kv_cache_pct": 0.0, "requests_done": 0,
                  "generation_tokens": 0}}) is True


def test_vram_in_use_vetoes_the_verdict_even_with_no_model_listed():
    """Something IS resident. Whatever is not reporting it, do not restart."""
    assert rw.RunpodPodPool._has_no_engine(
        {"loaded_models": [], "vram": {"used": 40.0}, "load": {}}) is False


def test_a_pod_that_has_served_is_never_engineless():
    assert rw.RunpodPodPool._has_no_engine(
        {"loaded_models": [], "vram": {"used": 0.0},
         "load": {"requests_done": 7}}) is False


def test_an_image_that_does_not_report_loaded_models_says_nothing():
    """An older pod image has no such field. Silence must not mean 'empty'."""
    assert rw.RunpodPodPool._has_no_engine({"load": {"requests_done": 0}}) is False
    assert rw.RunpodPodPool._has_no_engine({}) is False


# --- the sweep ------------------------------------------------------------

class _Pool:
    """The two methods under test, over a hand-built pod list."""

    _engine_state = rw.RunpodPodPool._engine_state
    _has_no_engine = staticmethod(rw.RunpodPodPool._has_no_engine)
    _restart_engineless_pods = rw.RunpodPodPool._restart_engineless_pods
    _recheck_restarted_pods = rw.RunpodPodPool._recheck_restarted_pods

    def __init__(self, pods, state):
        import threading
        self.pods = list(pods)
        self.model_key = "m"
        self.api_key = "k"
        self.account = object()
        self._cv = threading.Condition()
        self._state = state
        self.restarted = []
        self.terminated = []

    # seams
    def _engine_state(self, pod):                     # noqa: F811
        return self._state.get(pod.pod_id)

    def _bump_cards(self):
        pass

    def _restart_pod_now(self, pod):
        self.restarted.append(pod.pod_id)

    def _terminate(self, pod, reason):
        self.terminated.append((pod.pod_id, reason))


DEAD = {"loaded_models": [], "vram": {"used": 0.0},
        "load": {"requests_done": 0, "generation_tokens": 0}}
LIVE = {"loaded_models": ["m"], "vram": {"used": 40.0},
        "load": {"requests_done": 5}}


@pytest.fixture(autouse=True)
def _controller(monkeypatch):
    monkeypatch.setattr(rw._share, "is_follower", lambda: False)


def _run(pool, now=None):
    # Threads are a seam here; call the restart inline so the test is ordered.
    real = rw.threading.Thread
    class _Now:
        def __init__(self, target=None, args=(), **kw):
            self._t, self._a = target, args
        def start(self):
            self._t(*self._a)
    rw.threading.Thread = _Now
    try:
        pool._restart_engineless_pods(now or time.time())
    finally:
        rw.threading.Thread = real


def test_an_engineless_pod_the_pool_used_is_restarted_and_taken_out():
    now = time.time()
    p = _pod(last_used=now - 600.0, started_at=now - 5000.0)
    pool = _Pool([p], {"p1": DEAD})
    _run(pool, now)
    assert pool.restarted == ["p1"]
    assert p.healthy is False          # out of rotation, so the floor sees it
    assert p.restarting_until > now
    assert p.engine_restarts == 1
    assert pool.terminated == []


def test_a_pod_nobody_asked_is_left_alone():
    """last_used == 0: never dispatched to, so an empty engine proves nothing."""
    now = time.time()
    p = _pod(last_used=0.0, started_at=now - 5000.0)
    pool = _Pool([p], {"p1": DEAD})
    _run(pool, now)
    assert pool.restarted == [] and p.healthy is True


def test_a_pod_still_inside_its_boot_window_is_left_alone():
    now = time.time()
    p = _pod(last_used=now - 600.0, started_at=now - 10.0)
    pool = _Pool([p], {"p1": DEAD})
    _run(pool, now)
    assert pool.restarted == [] and p.healthy is True


def test_a_pod_with_work_in_flight_is_never_restarted():
    now = time.time()
    p = _pod(last_used=now - 600.0, started_at=now - 5000.0, inflight=3)
    pool = _Pool([p], {"p1": DEAD})
    _run(pool, now)
    assert pool.restarted == [] and p.healthy is True


def test_a_pod_carrying_a_workers_pages_is_never_restarted():
    """_load() counts the workers' report too, or a busy pod reads as idle."""
    now = time.time()
    p = _pod(last_used=now - 600.0, started_at=now - 5000.0, worker_inflight=9)
    pool = _Pool([p], {"p1": DEAD})
    _run(pool, now)
    assert pool.restarted == [] and p.healthy is True


def test_a_silent_pod_is_left_to_the_quarantine_path():
    """No answer is not evidence of no engine -- that mistake cost 8 A40s."""
    now = time.time()
    p = _pod(last_used=now - 600.0, started_at=now - 5000.0)
    pool = _Pool([p], {"p1": None})
    _run(pool, now)
    assert pool.restarted == [] and p.healthy is True


def test_a_healthy_pod_is_left_alone():
    now = time.time()
    p = _pod(last_used=now - 600.0, started_at=now - 5000.0)
    pool = _Pool([p], {"p1": LIVE})
    _run(pool, now)
    assert pool.restarted == [] and p.healthy is True


def test_a_quarantined_or_draining_pod_is_not_this_paths_business():
    now = time.time()
    a = _pod(pod_id="a", last_used=now - 600.0, started_at=now - 5000.0,
             quarantined_until=now + 60.0)
    b = _pod(pod_id="b", last_used=now - 600.0, started_at=now - 5000.0,
             draining=True)
    pool = _Pool([a, b], {"a": DEAD, "b": DEAD})
    _run(pool, now)
    assert pool.restarted == []


def test_restarts_are_bounded_then_the_pod_is_given_up_on():
    now = time.time()
    p = _pod(last_used=now - 600.0, started_at=now - 5000.0,
             engine_restarts=rw.ENGINELESS_MAX_RESTARTS)
    pool = _Pool([p], {"p1": DEAD})
    _run(pool, now)
    assert pool.restarted == []
    assert [t[0] for t in pool.terminated] == ["p1"]
    assert p not in pool.pods


def test_a_restarted_pod_that_comes_back_with_an_engine_returns_to_rotation():
    now = time.time()
    p = _pod(last_used=now - 600.0, started_at=now - 100.0,
             healthy=False, restarting_until=now - 1.0)
    pool = _Pool([p], {"p1": LIVE})
    pool._recheck_restarted_pods(now)
    assert p.healthy is True and p.restarting_until == 0.0


def test_a_restarted_pod_still_empty_stays_out_and_is_reconsidered():
    now = time.time()
    p = _pod(last_used=now - 600.0, started_at=now - 100.0,
             healthy=False, restarting_until=now - 1.0)
    pool = _Pool([p], {"p1": DEAD})
    pool._recheck_restarted_pods(now)
    assert p.healthy is False and p.restarting_until == 0.0


def test_a_worker_never_restarts_the_shared_fleet(monkeypatch):
    monkeypatch.setattr(rw._share, "is_follower", lambda: True)
    now = time.time()
    p = _pod(last_used=now - 600.0, started_at=now - 5000.0)
    pool = _Pool([p], {"p1": DEAD})
    _run(pool, now)
    assert pool.restarted == [] and p.healthy is True


def test_maintain_checks_for_an_engine_before_counting_the_warm_floor():
    import inspect
    src = inspect.getsource(rw.RunpodPodPool.maintain)
    assert "_restart_engineless_pods" in src
    assert src.index("_restart_engineless_pods") < src.index("_scale_down")


# --- cost of the check itself --------------------------------------------

def test_the_sweep_probes_only_a_few_pods_per_pass_longest_idle_first():
    """The probes run inline on the scaler thread, so the pass must be bounded.

    An idle fleet all blackholing TCP would otherwise hold `maintain()` for one
    timeout per pod -- nothing rents, nothing reaps and the registry heartbeat
    does not get written, which is the exact shape of the inline-warm-boot stall
    this file already paid for once. Longest-idle first so a truncated list
    still covers the fleet across passes instead of starving whoever is last.
    """
    now = time.time()
    pods = [_pod(pod_id=f"p{i}", last_used=now - (200.0 + i * 10.0),
                 started_at=now - 5000.0) for i in range(9)]
    asked = []

    class _Counting(_Pool):
        def _engine_state(self, pod):
            asked.append(pod.pod_id)
            return LIVE                    # healthy: nothing to restart

    pool = _Counting(pods, {})
    _run(pool, now)
    assert len(asked) == rw.ENGINELESS_PROBES_PER_PASS
    # p8 has been idle longest (now - 280), p5 the shortest of the four.
    assert asked == ["p8", "p7", "p6", "p5"]
    assert pool.restarted == []


def test_the_engine_probe_cannot_hold_the_scaler_thread_for_long():
    """A pod that needs more than a few seconds to read its own state has
    nothing to say that this check should act on."""
    from codai.api import pod_http
    seen = {}

    class _Reply:
        status_code = 200
        @staticmethod
        def json():
            return dict(LIVE)

    def _get(url, headers=None, timeout=None):
        seen.update(url=url, timeout=timeout)
        return _Reply()

    old = pod_http.get
    pod_http.get = _get
    try:
        got = rw.RunpodPodPool._engine_state(
            _Pool([], {}), _pod(url="https://pod.example/v1"))
    finally:
        pod_http.get = old
    assert got == LIVE
    assert seen["url"] == "https://pod.example/internal/engine-state"
    assert 0 < seen["timeout"] <= 5.0
    assert (rw.ENGINELESS_PROBES_PER_PASS
            * rw.ENGINELESS_PROBE_TIMEOUT_S) <= 30.0
