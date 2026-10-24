"""With `ocr.workers` on, a waiting page must still be able to start a pod.

The shape of the defect: an OCR worker never rents (by design -- the controller
owns the fleet), and the controller only ever decided to grow inside its own
`acquire()`, which it never calls while the workers dispatch every page. The one
signal workers did send was keyed by POD -- `{pod_id: {inflight, ...}}` -- which
is precisely what a cold fleet has none of, so a worker holding pages with zero
pods reported `{}` and the controller read an idle install.

Consequences, both observed on a live install on 2026-10-24 with `workers: 2`,
`min_pods: 0` and `pod_pipeline: require`: a cold fleet never booted for any
load, and a page waited out `ocr.pod_pipeline_wait_s` and 503'd. Setting
`workers: 0` rented a pod for the SAME waiting request within 20 seconds.

A warm fleet had the same defect one step along: it could never grow past
`min_pods`, so "scale to 128 pods" was unreachable whenever the workers were on.
"""
import json
import time
import types

import pytest

from codai.api import runpod_fleet_share as share


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Point the share files at a temporary directory."""
    monkeypatch.setattr(share, "_path", lambda name: str(tmp_path / name),
                        raising=True)
    return tmp_path


# -- the demand channel ---------------------------------------------------

def test_a_worker_can_report_demand_with_no_pods_at_all(store):
    """The whole point: the signal is keyed by POOL, not by pod."""
    share.write_demand("w0", "surya-pool", 7)
    assert share.read_demand("surya-pool") == 7


def test_demand_from_several_workers_adds_up(store):
    """Different workers hold DIFFERENT pages, so these sum."""
    share.write_demand("w0", "surya-pool", 4)
    share.write_demand("w1", "surya-pool", 6)
    assert share.read_demand("surya-pool") == 10


def test_another_pools_demand_does_not_leak(store):
    share.write_demand("w0", "surya-pool", 4)
    share.write_demand("w0", "chandra-pool", 1)
    assert share.read_demand("surya-pool") == 4


def test_zero_clears_the_entry(store):
    share.write_demand("w0", "surya-pool", 3)
    share.write_demand("w0", "surya-pool", 0)
    assert share.read_demand("surya-pool") == 0


def test_a_dead_workers_demand_expires(store, monkeypatch):
    """A worker killed holding pages must not keep a card rented forever."""
    share.write_demand("w0", "surya-pool", 5)
    path = share._path(share.DEMAND)
    d = json.load(open(path))
    for v in d.values():
        v["at"] = time.time() - (share.REPORT_MAX_AGE_S + 60)
    json.dump(d, open(path, "w"))
    assert share.read_demand("surya-pool") == 0


def test_demand_is_a_separate_file_from_the_pod_reports(store):
    """They are read by different code paths and keyed differently."""
    share.write_demand("w0", "surya-pool", 2)
    share.write_reports("w0:surya-pool", {"pod-1": {"inflight": 3}})
    assert share.read_demand("surya-pool") == 2
    assert "pod-1" in share.read_reports()


# -- the controller acts on it -------------------------------------------

class _Pool:
    """Just enough of RunpodPool for the two methods under test."""

    def __init__(self, waiting=0, denied=0, worker_waiting=0):
        import threading
        from codai.api.runpod_worker import RunpodPodPool as RunpodPool
        self._cv = threading.Condition()
        self._waiting = waiting
        self._worker_waiting = worker_waiting
        self._denied = denied
        self.pods = []
        self._booting = 0
        self.model_key = "surya-pool"
        self.grew = []
        self._queued = RunpodPool._queued.__get__(self)
        self._grow_for_workers = RunpodPool._grow_for_workers.__get__(self)

    def _denied_recent(self):
        return self._denied

    def _grow_reason(self, healthy):
        return "nothing can serve a request yet — renting the first pod"

    def _grow_in_background(self):
        self.grew.append(True)


def test_worker_demand_counts_as_queued_work():
    p = _Pool(worker_waiting=9)
    assert p._queued() == 9


def test_the_three_queue_signals_are_not_summed():
    """One page parked here, turned away, and parked in a worker may be one page."""
    p = _Pool(waiting=3, denied=4, worker_waiting=5)
    assert p._queued() == 5


def test_the_scaler_grows_when_a_worker_is_waiting(monkeypatch):
    from codai.api import runpod_worker as rw
    monkeypatch.setattr(rw._share, "is_follower", lambda: False)
    p = _Pool(worker_waiting=4)
    p._grow_for_workers()
    assert p.grew, "a waiting page in a worker must be able to start a pod"
    assert p._booting == 1, "the boot must be counted, or the next pass rents again"


def test_the_scaler_does_nothing_when_no_worker_is_waiting(monkeypatch):
    from codai.api import runpod_worker as rw
    monkeypatch.setattr(rw._share, "is_follower", lambda: False)
    p = _Pool(worker_waiting=0)
    p._grow_for_workers()
    assert not p.grew


def test_a_worker_never_rents_even_with_demand(monkeypatch):
    """Unchanged rule: the controller owns the fleet."""
    from codai.api import runpod_worker as rw
    monkeypatch.setattr(rw._share, "is_follower", lambda: True)
    p = _Pool(worker_waiting=99)
    p._grow_for_workers()
    assert not p.grew


def test_growth_still_respects_the_pools_own_refusal(monkeypatch):
    """max_pods, parallel-boot caps and "one is already coming" all live in
    _grow_reason, and this path must not go around them."""
    from codai.api import runpod_worker as rw
    monkeypatch.setattr(rw._share, "is_follower", lambda: False)
    p = _Pool(worker_waiting=50)
    p._grow_reason = lambda healthy: ""
    p._grow_for_workers()
    assert not p.grew
    assert p._booting == 0
