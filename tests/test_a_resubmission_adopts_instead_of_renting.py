"""Insisting cost a fleet per attempt.

Measured on this install: eleven concurrent training pools, 35 pods,
$21.25/hr, for a job that uses five. The cause was one line -- the pool was
keyed to the JOB id:

    pool_key = f"{_fleet_pool_prefix()}{self.job_id}"

Every POST makes a new job, so a new pool, so a new rental; the previous
attempt's pods were adopted by nobody and kept billing. The seat's retry loop
(asked for by the operator) then multiplied it every five minutes.

Three things were needed and none of them is the retry loop:

1. a pool name that SURVIVES a resubmission, so an attempt can adopt what the
   last one left;
2. one run per adapter, because two jobs averaging into one adapter overwrite
   each other whoever pays for the pods;
3. a cancel that is honoured DURING the boot wait -- ten cancelled jobs kept
   their fleets, one grew back to five pods after being cancelled, and 17 pods
   had to be terminated by hand.
"""

import inspect
import time

import pytest

from codai.api import loras
from codai.api import text_lora_fleet as flt


DIST = inspect.getsource(loras._text_train_distributed)


@pytest.fixture
def registry(monkeypatch):
    """A registry holding one reachable pod of an earlier attempt."""
    state = {"pod-old": {"pool": "text-lora:adapter-a",
                         "url": "http://1.2.3.4:41234", "api_key": "s3cret",
                         "hourly_usd": 0.59, "gpu": "NVIDIA A40",
                         "created": time.time() - 600}}
    import codai.api.runpod_worker as rw
    monkeypatch.setattr(rw, "_registry_update",
                        lambda fn: (fn(state), dict(state))[1])
    monkeypatch.setattr(rw, "register_pod", lambda *a, **k: None)
    monkeypatch.setattr(flt, "poll_status", lambda pod: {"state": "idle"})
    return state


def _fleet(monkeypatch, pool_name="adapter-a", job="job-new"):
    spec = flt.TrainerPodSpec(image="i", gpu_types=["A40"], hourly_usd=0.59)
    f = flt.TrainerFleet(object(), job, spec, pool_name=pool_name)
    monkeypatch.setattr(f, "_flush", lambda: None)
    return f


# ------------------------------------------------- 1. a durable pool name ---

def test_the_pool_is_named_for_something_that_survives_a_resubmission():
    src = inspect.getsource(flt.TrainerFleet._register)
    assert "self.pool_name" in src
    assert "self.job_id" not in src.split("pool_key=")[1].split(",")[0]


def test_the_job_passes_the_adapter_name():
    assert "pool_name=str(req.name or job_id)" in DIST


def test_the_pool_name_defaults_to_the_job_id():
    """A caller that gives none keeps the old behaviour rather than sharing a
    pool with every other job by accident."""
    spec = flt.TrainerPodSpec(image="i")
    assert flt.TrainerFleet(object(), "job-z", spec).pool_name == "job-z"


def test_a_previous_attempts_pod_is_adopted(registry, monkeypatch):
    f = _fleet(monkeypatch)
    got = f.adopt_existing()
    assert [p.pod_id for p in got] == ["pod-old"]
    assert sorted(f.pods) == [0]


def test_the_adopted_pod_keeps_its_own_bearer(registry, monkeypatch):
    """Each trainer pod is launched with a secret of its own; a different one
    is answered 401 for the rest of its paid life."""
    f = _fleet(monkeypatch)
    pod = f.adopt_existing()[0]
    assert pod.secret == "s3cret" and pod.url == "http://1.2.3.4:41234"


def test_a_pod_that_does_not_answer_is_not_adopted(registry, monkeypatch):
    """A registry entry can outlive its pod, and adopting a corpse fills a
    shard slot with something that never submits -- worse than renting,
    because it looks like progress."""
    monkeypatch.setattr(flt, "poll_status",
                        lambda pod: (_ for _ in ()).throw(RuntimeError("no")))
    f = _fleet(monkeypatch)
    assert f.adopt_existing() == []
    assert f.pods == {}


def test_another_pools_pod_is_left_alone(registry, monkeypatch):
    f = _fleet(monkeypatch, pool_name="a-different-adapter")
    assert f.adopt_existing() == []


def test_a_pod_with_no_address_is_not_adopted(registry, monkeypatch):
    registry["pod-old"]["url"] = ""
    f = _fleet(monkeypatch)
    assert f.adopt_existing() == []


def test_the_job_adopts_before_renting():
    i = DIST.index("adopted = fleet.adopt_existing()")
    assert i < DIST.index("started = fleet.provision(")
    assert "renting only what is " in DIST


# --------------------------------------------- 2. one run per adapter -------

def test_a_second_job_for_the_same_adapter_is_refused():
    src = inspect.getsource(loras._text_train_blocking)
    assert "_active_text_job_for(" in src
    assert "is already training" in src


def test_the_refusal_names_the_job_to_watch_and_how_to_cancel():
    src = inspect.getsource(loras._text_train_blocking)
    assert "Watch " in src and "/cancel" in src


@pytest.mark.parametrize("status,expected", [
    ("training", "job-a"), ("preparing", "job-a"),
    ("done", None), ("error", None), ("cancelled", None),
    ("interrupted", None),
])
def test_only_a_LIVE_job_blocks_a_resubmission(status, expected, monkeypatch):
    monkeypatch.setattr(loras, "_jobs",
                        {"job-a": {"target": "text", "name": "adapter-x",
                                   "status": status}})
    assert loras._active_text_job_for("adapter-x") == expected


def test_a_different_adapter_does_not_block(monkeypatch):
    monkeypatch.setattr(loras, "_jobs",
                        {"job-a": {"target": "text", "name": "adapter-x",
                                   "status": "training"}})
    assert loras._active_text_job_for("adapter-y") is None


def test_a_non_text_job_does_not_block(monkeypatch):
    monkeypatch.setattr(loras, "_jobs",
                        {"job-a": {"target": "image", "name": "adapter-x",
                                   "status": "training"}})
    assert loras._active_text_job_for("adapter-x") is None


# ------------------------------------- 3. cancel during the boot wait -------

def test_the_boot_wait_checks_for_cancellation():
    i = DIST.index("def _bring_up")
    block = DIST[i:i + 1600]
    assert "_train_cancelled(job_id)" in block
    assert "cancelled while booting" in block


def test_the_boot_wait_is_sliced_so_the_check_runs():
    i = DIST.index("def _bring_up")
    block = DIST[i:i + 1600]
    assert "min(20.0, _left)" in block


def test_the_pod_is_released_before_giving_up():
    """The reason the check used to sit after the waits: cancelling mid-boot
    would abandon a rented pod. The answer is to release it here."""
    i = DIST.index("_train_cancelled(job_id)")
    block = DIST[i:i + 300]
    assert 'fleet.release(shard, "cancelled while booting")' in block


def test_the_probe_does_not_raise():
    """A thread holding a rented card must release it and report, which it
    cannot do if the check throws."""
    doc = loras._train_cancelled.__doc__ or ""
    assert "not raise" in doc.lower()
    assert loras._train_cancelled("") is False
