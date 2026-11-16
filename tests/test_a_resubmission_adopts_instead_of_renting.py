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

def test_a_second_request_joins_the_run_instead_of_starting_one():
    """REVISED. Refusing a repeat request stopped the duplicate fleets and
    also threw away the only signal the caller had. A serving pool treats the
    same thing as a LEASE -- "asking for warm pods cancels a pending release:
    the client changed its mind" -- so a repeat request carries into the same
    pool and raises its target. A second RUN would still be wrong: two fleets
    averaging into one adapter directory, last round wins."""
    src = inspect.getsource(loras._text_train_blocking)
    assert "_active_text_job_for(" in src
    assert "carrying this request into that run" in src
    assert "note_train_lease(" in src


def test_the_answer_names_the_job_that_is_training():
    src = inspect.getsource(loras._text_train_blocking)
    assert '"joined": _already' in src
    assert "pods_target" in src


def test_the_lease_only_ever_raises_the_target():
    """A later request asking for fewer must not shrink a fleet the earlier
    one is relying on."""
    loras._forget_train_lease("j-x")
    assert loras.note_train_lease("j-x", 4) == 4
    assert loras.note_train_lease("j-x", 2) == 4
    assert loras.note_train_lease("j-x", 8) == 8
    loras._forget_train_lease("j-x")


def test_the_round_loop_reads_the_lease_every_round():
    assert "train_lease(job_id)" in DIST
    assert "target raised to " in DIST


def test_the_lease_dies_with_its_job():
    """Otherwise a finished run's target is handed to whatever reuses the
    id."""
    assert "_forget_train_lease(job_id)" in DIST


def test_the_operators_block_still_bounds_the_lease():
    """The lease is computed by narrowing the BLOCK, so a client cannot raise
    the target past what the operator allows."""
    src = inspect.getsource(loras._text_train_blocking)
    i = src.index("note_train_lease(")
    before = src[max(0, i - 400):i]
    assert "_flt.narrow(_blk" in before


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
    block = _bring_up_body()
    assert "_train_cancelled(job_id)" in block
    assert "cancelled while booting" in block


def test_the_boot_wait_is_sliced_so_the_check_runs():
    block = _bring_up_body()
    assert "min(_SLICE_S, _left)" in block


def test_the_pod_is_released_before_giving_up():
    """The reason the check used to sit after the waits: cancelling mid-boot
    would abandon a rented pod. The answer is to release it here."""
    block = _bring_up_body()
    assert 'fleet.release(shard, "cancelled while booting")' in block


def test_the_probe_does_not_raise():
    """A thread holding a rented card must release it and report, which it
    cannot do if the check throws."""
    doc = loras._train_cancelled.__doc__ or ""
    assert "not raise" in doc.lower()
    assert loras._train_cancelled("") is False

def _bring_up_body():
    """The whole `_bring_up` body, bounded by the statement after it.

    Bounded by a marker rather than a character count: three assertions in
    this file broke on the body growing, which tests where the text happened
    to end rather than what the code does.
    """
    src = DIST
    i = src.index("def _bring_up")
    j = src.index("if len(_shards) == 1", i)
    return src[i:j]


# --------- the join path, EXECUTED rather than read -------------------------
#
# Every test above checks the source text, and source text cannot catch an
# unbound local. The join branch read `_flt` while the import sat further down
# the function -- and a name assigned anywhere in a function is local
# throughout it, so the reference raised UnboundLocalError instead of falling
# back to the module global. Production found it four minutes after the
# deploy:
#
#     job-93b4994e0a96: FAILED: UnboundLocalError: cannot access local
#     variable '_flt' where it is not associated with a value
#
# This test calls the function.

def test_the_join_path_runs(monkeypatch, tmp_path):
    import types

    monkeypatch.setattr(loras, "_jobs",
                        {"job-live": {"target": "text",
                                      "name": "adapter-j",
                                      "status": "training"}})
    monkeypatch.setattr(loras, "_jobs_lock", __import__("threading").Lock())
    recorded = {}
    monkeypatch.setattr(loras, "_update_job",
                        lambda jid, **kw: recorded.update(kw))
    loras._forget_train_lease("job-live")

    req = types.SimpleNamespace(
        name="adapter-j", base_model="qwen38-train", pods=4, max_pods=8,
        min_pods=1, rounds=16, test_run=False, distributed=True,
        dataset_id="", field_map=None, dry_run=False)

    got = loras._text_train_blocking(req, "job-new", str(tmp_path / "ds.jsonl"))

    assert got["joined"] == "job-live"
    assert got["name"] == "adapter-j"
    assert isinstance(got["pods_target"], int)
    assert "already training" in got["detail"]
    # and the new job's own record says what became of it
    assert recorded.get("status") == "done"
    assert "joined job-live" in str(recorded.get("message"))
    loras._forget_train_lease("job-live")


def test_the_join_path_does_not_need_the_dataset(monkeypatch, tmp_path):
    """It returns before reading anything: the point of joining is that the
    work is already running elsewhere, and a missing file must not turn a join
    into a failure."""
    import types

    monkeypatch.setattr(loras, "_jobs",
                        {"job-live": {"target": "text", "name": "adapter-k",
                                      "status": "preparing"}})
    monkeypatch.setattr(loras, "_jobs_lock", __import__("threading").Lock())
    monkeypatch.setattr(loras, "_update_job", lambda jid, **kw: None)
    req = types.SimpleNamespace(
        name="adapter-k", base_model="qwen38-train", pods=1, max_pods=2,
        min_pods=1, rounds=1, test_run=False, distributed=None,
        dataset_id="", field_map=None, dry_run=False)
    got = loras._text_train_blocking(
        req, "job-new2", "/nonexistent/path/does-not-exist.jsonl")
    assert got["joined"] == "job-live"
    loras._forget_train_lease("job-live")
