"""Pods that outlive an orchestrator restart are taken back, busy or not.

Measured 2026-10-19: a restart under load left 29 pods ($17.11/hr) running
OUTSIDE the pool. Two causes, both pinned here:

  1. adoption was on demand only -- a warm floor or a waiting request. The
     restart dropped the client's lease (floor 0) and the client was paused,
     so nothing ever asked.
  2. the adoption probe was `/v1/models` with a 4 s timeout, which goes
     through the pod's busy engine. Pods still finishing 64 pages each failed
     it; `/healthz` on the same pods answered in 0.1 s.
"""
import inspect
import threading
import time

from codai.api import runpod_worker as rw


def _pool():
    p = rw.RunpodPodPool.__new__(rw.RunpodPodPool)
    p._cv = threading.Condition()
    p.model_key = "m"
    p.pods = []
    return p


def test_maintain_adopts_survivors_every_pass():
    src = inspect.getsource(rw.RunpodPodPool.maintain)
    assert "self._adopt_survivors(now)" in src


def test_survivors_are_adopted_until_none_is_left():
    p = _pool()
    calls = []
    p._adopt_shared_pod = lambda: calls.append(1) or len(calls) <= 5
    p._adopt_survivors(time.time())
    assert len(calls) == 6, "keeps adopting until there is nothing left"


def test_survivor_sweeps_are_rate_limited():
    """Each attempt probes every registry candidate; a dead one costs a full
    timeout every time."""
    p = _pool()
    calls = []
    p._adopt_shared_pod = lambda: calls.append(1) or False
    now = time.time()
    p._adopt_survivors(now)
    p._adopt_survivors(now + 1)
    assert len(calls) == 1
    p._adopt_survivors(now + rw.SURVIVOR_ADOPT_EVERY_S + 1)
    assert len(calls) == 2


def test_a_failing_adoption_does_not_break_maintenance():
    p = _pool()

    def _boom():
        raise RuntimeError("registry unreadable")
    p._adopt_shared_pod = _boom
    p._adopt_survivors(time.time())          # must not raise


def test_a_busy_pod_is_adopted_on_healthz_when_its_token_is_recorded(monkeypatch):
    monkeypatch.setattr(rw, "_registry_update", lambda fn: {
        "busy1": {"pool": "m", "url": "https://10.0.0.1:2222/v1",
                  "key": "tok", "image": "img", "gpu": "A40",
                  "hourly_usd": 0.59}})
    monkeypatch.setattr(rw, "_is_this_process", lambda info: False)
    seen = []

    def _health(url, timeout=4.0, path="/v1/models", api_key=""):
        seen.append((path, timeout))
        return path == "/healthz"            # engine too busy; front answers
    monkeypatch.setattr(rw, "_pod_health_ok", _health)
    pid, url, key, rate = rw.find_shared_pod("m", "/v1/models", "",
                                             image="img", gpus=["A40"])
    assert pid == "busy1"
    assert ("/v1/models", 15.0) in seen, "the engine probe gets a real timeout"
    assert ("/healthz", 15.0) in seen


def test_without_a_recorded_token_healthz_alone_is_not_enough(monkeypatch):
    """/healthz answers anybody. Without the launch token on record we cannot
    know we can USE the pod, so it is not adopted on /healthz alone."""
    monkeypatch.setattr(rw, "_registry_update", lambda fn: {
        "anon": {"pool": "m", "url": "https://10.0.0.1:2222/v1",
                 "image": "img", "gpu": "A40"}})
    monkeypatch.setattr(rw, "_is_this_process", lambda info: False)
    monkeypatch.setattr(rw, "_pod_health_ok",
                        lambda url, timeout=4.0, path="/v1/models", api_key="":
                        path == "/healthz")
    pid, url, _k, _r = rw.find_shared_pod("m", "/v1/models", "",
                                          image="img", gpus=["A40"])
    assert pid is None


def test_a_pod_already_in_the_pool_is_not_offered_again(monkeypatch):
    """The sweep re-found one pod 15-18 times in a second and never reached
    the four behind it."""
    monkeypatch.setattr(rw, "_registry_update", lambda fn: {
        "held": {"pool": "m", "url": "https://10.0.0.1:1/v1", "key": "k",
                 "image": "img", "gpu": "A40"},
        "next": {"pool": "m", "url": "https://10.0.0.2:1/v1", "key": "k",
                 "image": "img", "gpu": "A40"}})
    monkeypatch.setattr(rw, "_is_this_process", lambda info: False)
    monkeypatch.setattr(rw, "_pod_health_ok", lambda *a, **k: True)
    pid, *_ = rw.find_shared_pod("m", "/v1/models", "", image="img",
                                 gpus=["A40"], exclude={"held"})
    assert pid == "next"


def test_adopting_a_duplicate_is_not_reported_as_progress():
    src = inspect.getsource(rw.RunpodPodPool._adopt_shared_pod)
    assert "added = self._add_pod(handle)" in src
    assert "return bool(added)" in src
    assert "exclude=held" in src
