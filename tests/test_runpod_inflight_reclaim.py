"""A pod's in-flight counter must be able to come down on its own.

`pod.inflight` is raised in `_pick` and lowered only in `release`. That makes
it a ratchet: any dispatch path that fails to release raises it for the life of
the pod, and once it reaches max_inflight_per_pod the pod is refused all work
while its card sits idle -- reported as "at capacity", which is the most
misleading thing the pool can say.

Observed 2026-10-18 on a single-pod fleet: the pool reported inflight=64 of 64
with 63 pages queued, while the pod itself reported `Running: 31, Waiting: 0`
at 10.6% KV cache. Half the slots did not exist, and nothing in the codebase
could ever have given them back.

Two leaks have been found and fixed (a lease dropped during acquisition in
0.4.23; a page whose client disconnected). This sweep exists for the third one,
which has not been found. It is a safety net, not a timeout: returning a slot
that IS in use would hand a pod more work than its cap allows, so the bound is
deliberately far past the slowest legitimate page.
"""
import time
import pytest

import codai.api.runpod_worker as rw


def _pod(inflight, ages=(), pod_id="p1"):
    """A pod holding `inflight` slots, taken `ages` seconds ago."""
    p = rw.PodHandle(pod_id=pod_id, url="https://x", hourly_usd=0.5,
                     started_at=time.time())
    p.inflight = inflight
    now = time.time()
    p.inflight_since = [now - a for a in ages]
    return p


class _Pool:
    """Just enough pool to run the sweep against."""
    def __init__(self, pods):
        import threading
        self.pods = list(pods)
        self._cv = threading.Condition()

    _reclaim_stranded_inflight = rw.RunpodPodPool._reclaim_stranded_inflight


def sweep(pods, now=None):
    pool = _Pool(pods)
    pool._reclaim_stranded_inflight(now if now is not None else time.time())
    return pool


# --- the observed incident ---

def test_a_fully_stranded_pod_is_freed():
    """inflight=64 with every slot older than the TTL: the pod is usable again."""
    pod = _pod(64, ages=[rw.INFLIGHT_TTL_S + 60] * 64)
    sweep([pod])
    assert pod.inflight == 0
    assert pod.inflight_since == []


def test_live_slots_survive_and_phantoms_do_not():
    """31 real pages running, 33 stranded -- the real ones must not be touched."""
    ages = [5.0] * 31 + [rw.INFLIGHT_TTL_S + 120] * 33
    pod = _pod(64, ages=ages)
    sweep([pod])
    assert pod.inflight == 31
    assert len(pod.inflight_since) == 31


# --- it must never cancel real work ---

def test_nothing_is_freed_before_the_ttl():
    pod = _pod(64, ages=[rw.INFLIGHT_TTL_S - 1] * 64)
    sweep([pod])
    assert pod.inflight == 64


def test_a_busy_pod_is_untouched():
    pod = _pod(8, ages=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    sweep([pod])
    assert pod.inflight == 8


def test_the_ttl_is_generous_enough_for_a_slow_page():
    """A full budget plus continuations and a retry on a contended card.
    12288 tokens at the ~300 tok/s a single stream gets is ~41s; with three
    continuations and a retry that is minutes, not seconds."""
    assert rw.INFLIGHT_TTL_S >= 600


# --- the counter and the list cannot disagree ---

def test_release_drops_one_timestamp():
    pod = _pod(3, ages=[30.0, 20.0, 10.0])
    pool = _Pool([pod])
    rw.RunpodPodPool.release(pool, pod)
    assert pod.inflight == 2
    assert len(pod.inflight_since) == 2


def test_release_drops_the_oldest_timestamp():
    """The remaining list must still say how long the rest have been held."""
    pod = _pod(3, ages=[300.0, 20.0, 10.0])
    pool = _Pool([pod])
    rw.RunpodPodPool.release(pool, pod)
    ages = sorted(time.time() - t for t in pod.inflight_since)
    assert max(ages) < 100, "the 300s-old slot should have been the one dropped"


def test_release_below_zero_is_impossible():
    pod = _pod(0, ages=[])
    pool = _Pool([pod])
    rw.RunpodPodPool.release(pool, pod)
    assert pod.inflight == 0
    assert pod.inflight_since == []


def test_the_counter_is_never_left_above_the_surviving_slots():
    """A counter that drifted above its own timestamps is corrected down."""
    pod = _pod(64, ages=[1.0, 2.0])      # counter says 64, list says 2
    sweep([pod])
    assert pod.inflight <= len(pod.inflight_since)


# --- housekeeping ---

def test_a_pod_with_no_slots_is_skipped():
    pod = _pod(0, ages=[])
    sweep([pod])
    assert pod.inflight == 0


def test_every_pod_in_the_pool_is_swept():
    a = _pod(4, ages=[rw.INFLIGHT_TTL_S + 10] * 4, pod_id="a")
    b = _pod(4, ages=[rw.INFLIGHT_TTL_S + 10] * 4, pod_id="b")
    sweep([a, b])
    assert (a.inflight, b.inflight) == (0, 0)


def test_a_zero_ttl_disables_the_sweep(monkeypatch):
    """An operator must be able to turn a safety net off."""
    monkeypatch.setattr(rw, "INFLIGHT_TTL_S", 0.0)
    pod = _pod(64, ages=[99999.0] * 64)
    sweep([pod])
    assert pod.inflight == 64


def test_the_sweep_runs_in_maintain():
    src = open("codai/api/runpod_worker.py").read()
    assert "self._reclaim_stranded_inflight(now)" in src
    blk = src.split("def maintain")[1].split("def close")[0]
    assert "_reclaim_stranded_inflight" in blk, "must be wired into the scaler pass"


def test_a_counter_with_no_timestamps_at_all_is_reconciled():
    """The upgrade case, and the live incident.

    A pod already in the pool when this code arrives carries a counter and no
    timestamp list. It is also precisely the pod that is locked out at its cap.
    Skipping it would leave the one pod this sweep exists to free.
    """
    pod = _pod(64, ages=[])
    sweep([pod])
    assert pod.inflight == 0


def test_a_drifted_counter_is_pulled_down_to_its_evidence():
    """Counter says 64, two slots recorded and both fresh: 62 are fiction."""
    pod = _pod(64, ages=[1.0, 2.0])
    sweep([pod])
    assert pod.inflight == 2
    assert len(pod.inflight_since) == 2


def test_reconciliation_does_not_need_anything_to_be_stale():
    """The first version only looked at pods with an expired slot, so a pod
    whose counter had drifted upward with no stale entry stayed locked out."""
    pod = _pod(10, ages=[1.0])
    sweep([pod])
    assert pod.inflight == 1
