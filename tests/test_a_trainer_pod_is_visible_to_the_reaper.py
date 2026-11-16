"""A trainer pod belonged to nothing that could clean it up.

It is named `coderai-train-job-<id>-s0`, so `reap_orphans`'s
`coderai-<deployment_id>-` prefix never matched it, and it lived only in the
fleet's own record file -- not the registry that startup reclaim reads. While
the job ran, nothing noticed. When the process died (a crash, or one of
several forced upgrades in a day) the pod billed on with no owner and nothing
to collect it.

The same absence had a second cost: the admin pod table calls anything outside
a POOL "unmanaged", and a trainer pod never joins a pool. Three healthy pods
of three live training jobs were being displayed as abandoned -- and acting on
that label would have destroyed the run.

One registry entry fixes both ends. While the owner heartbeats,
`_known_pod_ids` counts the pod and the page stops lying about it; once the
owner is gone the entry stops vouching after the reap grace and the pod is
collected like any other leftover.
"""

import inspect

import pytest

from codai.api import text_lora_fleet as flt


class Client:
    def __init__(self):
        self.terminated = []

    def create_pod(self, **kw):
        return "pod-1"

    def terminate_pod(self, pod_id):
        self.terminated.append(pod_id)


@pytest.fixture
def registry(monkeypatch):
    """Capture what the fleet writes to the deployment registry."""
    seen = {"registered": [], "forgotten": []}
    import codai.api.runpod_worker as rw
    monkeypatch.setattr(rw, "register_pod",
                        lambda pod_id, **kw: seen["registered"].append((pod_id, kw)))
    monkeypatch.setattr(rw, "unregister_pod",
                        lambda pod_id: seen["forgotten"].append(pod_id))
    return seen


def _fleet(client=None, tmp=None, monkeypatch=None):
    spec = flt.TrainerPodSpec(image="img:1", gpu_types=["A40"],
                              hourly_usd=0.59)
    f = flt.TrainerFleet(client or Client(), "job-abc", spec)
    if monkeypatch is not None:
        monkeypatch.setattr(f, "_flush", lambda: None)
    return f


# ----------------------------------------------------------- on renting ----

def test_a_rented_pod_is_registered(registry, monkeypatch):
    f = _fleet(monkeypatch=monkeypatch)
    pod = f._rent_one(0)
    assert pod is not None
    assert registry["registered"], "the pod was not recorded as the deployment's"
    pod_id, kw = registry["registered"][0]
    assert pod_id == "pod-1"


def test_the_entry_names_the_owning_job(registry, monkeypatch):
    """So a stale entry says which run died, and the page can say who owns it."""
    f = _fleet(monkeypatch=monkeypatch)
    f._rent_one(0)
    _pid, kw = registry["registered"][0]
    assert kw["pool_key"] == "text-lora:job-abc"


def test_the_entry_carries_the_bearer_and_the_price(registry, monkeypatch):
    """The bearer so an adopted pod is not answered 401 for the rest of its
    paid life; the price so a stale pod's cost can be stated."""
    f = _fleet(monkeypatch=monkeypatch)
    pod = f._rent_one(0)
    _pid, kw = registry["registered"][0]
    assert kw["api_key"] == pod.secret and kw["api_key"]
    assert kw["hourly_usd"] == 0.59
    assert kw["image"] == "img:1"


def test_a_registry_failure_does_not_fail_the_rental(monkeypatch, capsys):
    """Bookkeeping must not cost a rented card -- but it must be said, because
    the consequence is a pod that leaks."""
    import codai.api.runpod_worker as rw
    monkeypatch.setattr(rw, "register_pod",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("nope")))
    f = _fleet(monkeypatch=monkeypatch)
    assert f._rent_one(0) is not None
    out = capsys.readouterr().out
    assert "could not register pod" in out
    assert "not be reclaimed or reaped" in out


# --------------------------------------------------------- on releasing ----

def test_a_released_pod_is_forgotten(registry, monkeypatch):
    f = _fleet(monkeypatch=monkeypatch)
    f._rent_one(0)
    assert f.release(0, "done") is True
    assert registry["forgotten"] == ["pod-1"]


def test_a_failed_terminate_keeps_the_registry_entry(registry, monkeypatch):
    """The pod may still be running. Dropping the entry would hide it from the
    one sweep meant to find it -- the exact leak this closes."""
    class Stubborn(Client):
        def terminate_pod(self, pod_id):
            raise RuntimeError("RunPod says no")

    f = _fleet(Stubborn(), monkeypatch=monkeypatch)
    f._rent_one(0)
    assert f.release(0, "done") is False
    assert registry["forgotten"] == [], "a possibly-live pod was forgotten"


def test_the_unregister_happens_after_the_terminate():
    src = inspect.getsource(flt.TrainerFleet.release)
    assert src.index("terminate_pod") < src.index("_unregister")
    assert "failed terminate" in src and "still running" in src


# -------------------------------------------- how the page labels it -------
#
# Registering was not enough on its own: the registry row said
# `state: "unowned"` with the note "held by no pool here … and still billing",
# which is true of a fleet pod and tells the reader the opposite of the truth.

from codai.api import runpod_worker as rw


def test_the_prefix_is_defined_once():
    """Writer and readers cannot drift: a prefix typed twice is a pod labelled
    unowned while a job is using it."""
    from codai.api.text_lora_fleet import _fleet_pool_prefix
    assert _fleet_pool_prefix() == rw.FLEET_POOL_PREFIX


def test_a_live_fleet_pod_is_labelled_training():
    src = inspect.getsource(rw.pods_status)
    i = src.index("fleet_job and mine")
    block = src[i:i + 500]
    assert 'state = "training"' in block
    assert "belongs to a job, not a pool" in block


def test_a_live_fleet_pod_is_not_called_still_billing():
    """That phrase is what made three working pods look like a leak."""
    src = inspect.getsource(rw.pods_status)
    i = src.index("fleet_job and mine")
    assert "still billing" not in src[i:i + 500]


def test_a_dead_jobs_pod_names_the_job_that_died():
    src = inspect.getsource(rw.pods_status)
    i = src.index("Not ours and past the grace")
    block = src[i:i + 400]
    assert "whose process is gone" in block
    assert "still billing" in block


def test_a_non_fleet_registry_pod_keeps_the_old_wording():
    """A serving pool's orphan is a different situation and still reads that
    way."""
    src = inspect.getsource(rw.pods_status)
    assert "in the registry but held by no pool here" in src
