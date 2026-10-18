"""A pool must never hold two handles for one pod.

Two paths append pods -- a fresh rent and an adoption -- and neither checked
whether the pod was already there. On a restart both fire for the same pod in
the same second; the log showed

    [runpod] adopting pod j74fxkxr9vs3fg's bearer token
    [runpod] reclaim: took back a pod for 'datalab-to/surya-ocr-2' ...

a second apart. Every capacity question is answered by walking self.pods, so
one card counted twice would claim 2 x max_inflight_per_pod slots, and max_pods
-- enforced against that same count -- could be satisfied by two entries for
one card.

Precautionary rather than a diagnosed fault, and the distinction is worth
keeping straight: a snapshot was once seen with two rows for one pod id, but
the model key was not recorded, so two POOLS sharing one pod was never ruled
out -- and that case this does not fix. The occasion when a max_pods=1 pool
really did bill for two A40s was two distinct pod ids, caused by the warm-floor
top-up counting only healthy pods against the cap. What justifies this guard by
itself is that two paths appended pods with no identity check between them, and
the in-flight staleness sweep cannot catch a duplicate because both entries
hold legitimate fresh timestamps.
"""
import time
import threading
import pytest

import codai.api.runpod_worker as rw


def _handle(pod_id="p1", url="https://a", inflight=0):
    h = rw.PodHandle(pod_id=pod_id, url=url, hourly_usd=0.59,
                     started_at=time.time(), gpu="A40", healthy=True)
    h.inflight = inflight
    return h


class _Pool:
    def __init__(self):
        self.pods = []
        self._cv = threading.Condition()
        self.model_key = "m"
        self.cards = {}

    def _bump_cards(self):
        pass

    _add_pod = rw.RunpodPodPool._add_pod


# --- the regression ---

def test_the_same_pod_is_not_added_twice():
    pool = _Pool()
    assert pool._add_pod(_handle()) is True
    assert pool._add_pod(_handle()) is False
    assert len(pool.pods) == 1


def test_rent_then_adopt_of_one_pod_leaves_one_entry():
    """The restart sequence: both paths fire for the same machine."""
    pool = _Pool()
    pool._add_pod(_handle(url="https://first"))
    pool._add_pod(_handle(url="https://first"))
    assert [p.pod_id for p in pool.pods] == ["p1"]


def test_capacity_is_not_doubled_by_a_duplicate():
    """The observed consequence: one card claiming two pods' worth of slots."""
    pool = _Pool()
    pool._add_pod(_handle(inflight=64))
    pool._add_pod(_handle(inflight=64))
    assert sum(p.inflight for p in pool.pods) == 64, \
        "a duplicate made one pod's 64 in-flight pages look like 128"


# --- the existing handle must survive, because it holds the accounting ---

def test_the_live_handle_is_kept_not_replaced():
    """Replacing it would strand every in-flight slot it holds."""
    pool = _Pool()
    first = _handle(inflight=17)
    pool._add_pod(first)
    pool._add_pod(_handle(inflight=0))
    assert pool.pods[0] is first
    assert pool.pods[0].inflight == 17


def test_a_changed_address_is_refreshed_onto_the_live_handle():
    """RunPod remaps ports; the pod id is the identity, the url is not."""
    pool = _Pool()
    first = _handle(url="https://old:1", inflight=5)
    pool._add_pod(first)
    pool._add_pod(_handle(url="https://new:2"))
    assert len(pool.pods) == 1
    assert pool.pods[0] is first
    assert pool.pods[0].url == "https://new:2"
    assert pool.pods[0].inflight == 5, "accounting must survive a re-address"


def test_a_returning_pod_is_marked_healthy_again():
    pool = _Pool()
    first = _handle()
    pool._add_pod(first)
    first.healthy = False
    pool._add_pod(_handle())
    assert first.healthy is True


# --- different pods are still different pods ---

def test_distinct_pods_are_both_added():
    pool = _Pool()
    assert pool._add_pod(_handle(pod_id="a")) is True
    assert pool._add_pod(_handle(pod_id="b")) is True
    assert len(pool.pods) == 2


def test_many_distinct_pods_all_land():
    pool = _Pool()
    for i in range(8):
        assert pool._add_pod(_handle(pod_id=f"p{i}")) is True
    assert len({p.pod_id for p in pool.pods}) == 8


def test_concurrent_adds_of_one_pod_still_yield_one_entry():
    """Rent and adopt genuinely race; the guard is under the pool lock."""
    pool = _Pool()
    added = []

    def add():
        added.append(pool._add_pod(_handle()))

    ts = [threading.Thread(target=add) for _ in range(12)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(pool.pods) == 1
    assert added.count(True) == 1, "exactly one caller should have added it"


# --- there must be only one way in ---

def test_the_worker_has_a_single_append_site():
    """Two unguarded append sites are how this happened; the guard is only
    worth anything if nothing bypasses it."""
    src = open("codai/api/runpod_worker.py").read()
    assert src.count("self.pods.append(") == 1, \
        "every path into the pool must go through _add_pod"
    assert "def _add_pod" in src
    assert "self._add_pod(handle)" in src, "the adoption path"
    assert "self._add_pod(h)" in src, "the fresh-rent path"
