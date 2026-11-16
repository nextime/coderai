"""Rented GPUs that run training, and are never left billing.

A serving pool rents a pod, waits for an OpenAI server, and terminates it after
N failed health probes. A training pod serves nothing and answers no probe, so
that machinery would rent it and then kill it as unreachable. Here the work is
the health signal: a pod calls in at the end of every round, and one that has
stopped finishing rounds has stopped being useful by definition.

The rule these tests are really about is the one that came from a real
incident: nothing is left billing with nobody owning it. So the pod id is on
disk BEFORE the pod exists, a terminate that fails keeps the pod in the record
rather than dropping it, and a new process adopts what the last one left.
"""

import json
import os

import pytest

from codai.api import text_lora_fleet as flt


class FakeClient:
    """Records what was asked of RunPod; fails on demand."""

    def __init__(self, fail_create=False, fail_terminate=False):
        self.created = []
        self.terminated = []
        self.fail_create = fail_create
        self.fail_terminate = fail_terminate
        self._n = 0

    def create_pod(self, **kw):
        if self.fail_create:
            raise RuntimeError("no capacity for A40 in any region")
        self._n += 1
        self.created.append(kw)
        return f"pod{self._n}"

    def terminate_pod(self, pod_id, attempts=4):
        if self.fail_terminate:
            raise RuntimeError("RunPod API 502")
        self.terminated.append(pod_id)


@pytest.fixture(autouse=True)
def _store(monkeypatch, tmp_path):
    from codai.api import loras
    monkeypatch.setattr(loras, "_loras_dir", lambda: str(tmp_path), raising=False)
    yield


def _spec(**kw):
    base = dict(image="ghcr.io/x/trainer:v1", gpu_types=["NVIDIA A40"],
                start_cmd=["python3", "-m", "trainer"], hourly_usd=0.45)
    base.update(kw)
    return flt.TrainerPodSpec(**base)


def _fleet(client=None, **kw):
    return flt.TrainerFleet(client or FakeClient(), "job-abc", _spec(**kw))


# ----------------------------------------------------------- what is rented --

def test_each_shard_gets_a_pod_told_which_shard_it_is():
    c = FakeClient()
    f = _fleet(c)
    pods = f.provision([0, 1, 2])
    assert len(pods) == 3 and len(c.created) == 3
    shards = sorted(int(k["env"]["CODERAI_TRAIN_SHARD"]) for k in c.created)
    assert shards == [0, 1, 2]
    assert all(k["env"]["CODERAI_TRAIN_JOB"] == "job-abc" for k in c.created)


def test_the_command_overrides_the_images_entrypoint():
    """The vLLM base's ENTRYPOINT is the server: a training command passed as
    arguments would start an OpenAI server with nonsense flags."""
    c = FakeClient()
    _fleet(c).provision([0])
    assert c.created[0]["start_cmd"] == ["python3", "-m", "trainer"]


def test_provisioning_is_idempotent_per_shard():
    c = FakeClient()
    f = _fleet(c)
    f.provision([0, 1])
    f.provision([0, 1])
    assert len(c.created) == 2, "a shard that already has a pod is not rented twice"


def test_a_refused_pod_does_not_leave_a_phantom_in_the_record():
    c = FakeClient(fail_create=True)
    f = _fleet(c)
    assert f.provision([0]) == []
    assert f.pods == {}
    assert flt.orphans() == [], "nothing is billing, so nothing is recorded"


# --------------------------------------------------- the budget ceiling ------

def test_provisioning_stops_at_the_cap_and_keeps_what_it_has():
    """A partial fleet is usable -- the round protocol averages what arrives --
    so a ceiling reached halfway is reported, not fatal, and never exceeded."""
    c = FakeClient()
    calls = {"n": 0}

    def _budget(rate):
        calls["n"] += 1
        return "" if calls["n"] <= 2 else "global $5/hr rate cap"

    f = flt.TrainerFleet(c, "job-abc", _spec(), budget_check=_budget)
    got = f.provision([0, 1, 2, 3])
    assert len(got) == 2 and len(c.created) == 2


def test_the_cap_is_consulted_before_renting_not_after():
    c = FakeClient()
    f = flt.TrainerFleet(c, "job-abc", _spec(),
                         budget_check=lambda rate: "spend limit reached")
    assert f.provision([0]) == []
    assert c.created == [], "nothing was rented to find out"


# ------------------------------------------------------------ the liveness ---

def test_a_checkin_is_the_health_signal():
    f = _fleet()
    f.provision([0])
    assert f.to_reap(now=f.pods[0].started_at + 100000) != []
    f.note_checkin(0, rounds_done=3)
    assert f.pods[0].rounds_done == 3
    assert f.to_reap(now=f.pods[0].last_checkin + 60) == []


def test_a_silent_pod_is_reaped():
    f = _fleet(max_silence_s=600)
    f.provision([0])
    t0 = f.pods[0].started_at
    assert f.to_reap(now=t0 + 300) == []
    assert [p.shard for p in f.to_reap(now=t0 + 700)] == [0]


def test_a_pod_that_keeps_checking_in_forever_still_dies_of_old_age():
    """The expensive failure is not a crash -- it is a loop that keeps
    reporting progress and billing for a week."""
    f = _fleet(max_silence_s=600, max_lifetime_s=3600)
    f.provision([0])
    t0 = f.pods[0].started_at
    for t in range(0, 3600, 300):
        f.pods[0].last_checkin = t0 + t
    assert [p.shard for p in f.to_reap(now=t0 + 3700)] == [0]


def test_a_pod_still_being_created_is_not_reaped():
    """Its id is a placeholder; terminating that would terminate nothing while
    reporting success."""
    f = _fleet()
    f.pods[0] = flt.TrainerPod(pod_id="pending-job-abc-0", shard=0,
                               started_at=0.0)
    assert f.to_reap(now=1e9) == []


# ------------------------------------------------- never billing unowned -----

def test_the_pod_id_is_on_disk_before_the_pod_exists():
    """A create that succeeds while the caller dies is the one case that bills
    all night, so the record goes down first."""
    seen = {}

    class Watching(FakeClient):
        def create_pod(self, **kw):
            seen["recorded_before_create"] = flt.orphans()
            return super().create_pod(**kw)

    _fleet(Watching()).provision([0])
    assert seen["recorded_before_create"], "nothing was recorded before renting"
    assert seen["recorded_before_create"][0]["shard"] == 0


def test_release_terminates_records_the_spend_and_forgets():
    c = FakeClient()
    f = _fleet(c)
    f.provision([0])
    assert f.release(0, "done") is True
    assert c.terminated == ["pod1"]
    assert f.pods == {}
    assert flt.orphans() == []


def test_a_terminate_that_fails_keeps_the_pod_in_the_record():
    """Dropping it here is precisely how a pod ends up running with nobody
    owning it. It stays listed so a later sweep still finds it."""
    c = FakeClient(fail_terminate=True)
    f = _fleet(c)
    f.provision([0])
    assert f.release(0) is False
    assert 0 in f.pods
    assert [o["pod_id"] for o in flt.orphans()] == ["pod1"]


def test_release_all_is_the_finally_block_guarantee():
    c = FakeClient()
    f = _fleet(c)
    f.provision([0, 1, 2])
    assert f.release_all("job finished") == 3
    assert sorted(c.terminated) == ["pod1", "pod2", "pod3"]
    assert flt.orphans() == []


def test_release_all_reports_what_it_could_not_kill():
    c = FakeClient(fail_terminate=True)
    f = _fleet(c)
    f.provision([0, 1])
    assert f.release_all() == 0
    assert len(flt.orphans()) == 2, "both still billing, both still named"


def test_a_new_process_adopts_what_the_last_one_left_running():
    """The orchestrator restarting mid-training must not lose the pods: they are
    rented, they are being paid for, and nothing in memory knows them."""
    c = FakeClient()
    first = _fleet(c)
    first.provision([0, 1])
    second = flt.TrainerFleet(FakeClient(), "job-abc", _spec())
    assert sorted(second.pods) == [0, 1]
    assert {p.pod_id for p in second.pods.values()} == {"pod1", "pod2"}
    # and it can kill them, which is the point of adopting them
    assert second.release_all("cleanup after restart") == 2


def test_orphans_spans_jobs():
    c = FakeClient()
    flt.TrainerFleet(c, "job-one", _spec()).provision([0])
    flt.TrainerFleet(c, "job-two", _spec()).provision([0])
    jobs = sorted(o["job"] for o in flt.orphans())
    assert jobs == ["job-one", "job-two"]


def test_the_record_survives_a_malformed_file(tmp_path):
    """A half-written state file must not make a fleet forget live pods -- it
    should degrade to 'adopt what parses'."""
    c = FakeClient()
    f = _fleet(c)
    f.provision([0])
    path = flt._state_path("job-abc")
    with open(path, "w") as fh:
        fh.write('{"job": "job-abc", "pods": [{"pod_id": "pod1", "shard": 0},')
    again = flt.TrainerFleet(FakeClient(), "job-abc", _spec())
    assert again.pods == {}, "unreadable means unknown, not a crash"


def test_the_cost_recorded_is_the_time_actually_rented():
    import time as _t
    c = FakeClient()
    f = _fleet(c, hourly_usd=3.6)
    f.provision([0])
    f.pods[0].started_at = _t.time() - 3600       # one hour ago
    cost = f.pods[0].cost_so_far()
    assert cost == pytest.approx(3.6, rel=0.01)


# ------------------------------------------------- the two hourly ceilings ---
#
# They are different limits and a run wants both: one bounds the bill, the other
# bounds the surprise. $0.90/hr means something quite different as a fleet total
# (one or two A40s, a day-scale run) than as a per-pod maximum (as many pods as
# the account caps allow), which is why neither is inferred from the other.

def test_a_fleet_ceiling_bounds_the_whole_bill():
    """0.90/hr total against 0.45/hr pods: two pods, and the third is refused
    before it is rented."""
    c = FakeClient()
    f = flt.TrainerFleet(c, "job-abc", _spec(hourly_usd=0.45),
                         max_fleet_hourly_usd=0.90)
    got = f.provision([0, 1, 2, 3])
    assert len(got) == 2, "0.45 + 0.45 = 0.90 exactly; a third would exceed it"
    assert len(c.created) == 2
    assert f.hourly_rate() == pytest.approx(0.90)


def test_the_fleet_ceiling_is_not_exceeded_by_a_rounding_error():
    """Exactly reaching the cap is allowed; a cent over is not."""
    c = FakeClient()
    f = flt.TrainerFleet(c, "job-abc", _spec(hourly_usd=0.30),
                         max_fleet_hourly_usd=0.90)
    assert len(f.provision([0, 1, 2, 3])) == 3
    assert f.hourly_rate() == pytest.approx(0.90)


def test_a_per_pod_ceiling_refuses_an_expensive_card():
    """What stops a capacity search landing on an H100 at six times the rate
    because the A40s were full."""
    c = FakeClient()
    f = flt.TrainerFleet(c, "job-abc",
                         _spec(hourly_usd=2.69, max_pod_hourly_usd=0.90))
    assert f.provision([0]) == []
    assert c.created == [], "refused before anything was rented"


def test_a_per_pod_ceiling_allows_as_many_as_the_account_permits():
    """The other reading of the same number: 0.90 per POD, no fleet total."""
    c = FakeClient()
    f = flt.TrainerFleet(c, "job-abc",
                         _spec(hourly_usd=0.45, max_pod_hourly_usd=0.90))
    assert len(f.provision([0, 1, 2, 3, 4, 5])) == 6


def test_both_ceilings_apply_together():
    c = FakeClient()
    f = flt.TrainerFleet(c, "job-abc",
                         _spec(hourly_usd=0.45, max_pod_hourly_usd=0.90),
                         max_fleet_hourly_usd=0.90)
    assert len(f.provision([0, 1, 2])) == 2


def test_the_refusal_says_which_ceiling_and_what_the_numbers_were():
    """A log line reading 'budget' tells an operator nothing about which knob
    to turn."""
    msgs = []
    c = FakeClient()
    f = flt.TrainerFleet(c, "job-abc", _spec(hourly_usd=0.45),
                         max_fleet_hourly_usd=0.90,
                         on_event=msgs.append)
    f.provision([0, 1, 2])
    joined = " ".join(msgs)
    assert "fleet ceiling" in joined and "$0.90" in joined and "$1.35" in joined


def test_the_account_caps_still_apply_on_top():
    """The per-fleet and per-pod ceilings are extra, not a replacement: the
    account's own $/hr and spend caps refuse regardless of what a job asks."""
    c = FakeClient()
    f = flt.TrainerFleet(c, "job-abc", _spec(hourly_usd=0.10),
                         max_fleet_hourly_usd=100.0,
                         budget_check=lambda rate: "global spend cap reached")
    assert f.provision([0]) == []


def test_adopted_pods_count_against_the_fleet_ceiling():
    """A restarted orchestrator must not rent a second fleet on top of the one
    it already has running."""
    c = FakeClient()
    first = flt.TrainerFleet(c, "job-abc", _spec(hourly_usd=0.45),
                             max_fleet_hourly_usd=0.90)
    first.provision([0, 1])
    second = flt.TrainerFleet(FakeClient(), "job-abc", _spec(hourly_usd=0.45),
                              max_fleet_hourly_usd=0.90)
    assert sorted(second.pods) == [0, 1]
    assert second.provision([2, 3]) == [], "already at the ceiling on adoption"


def test_provisioning_twice_does_not_rent_twice_with_dense_ids():
    """The regression dense ids introduced, pinned.

    `provision` used to skip an id the fleet already held, which made it
    idempotent for free. Once ids stopped coming from the caller that guard
    meant nothing, and a second call rented the whole list again -- sixteen
    extra pods on a sixteen-pod job. It fills TO a size now.
    """
    c = FakeClient()
    f = _fleet(c)
    assert len(f.provision([0, 1, 2])) == 3
    assert f.provision([0, 1, 2]) == []
    assert len(c.created) == 3
    # ...and asking for more than it holds rents only the shortfall.
    assert len(f.provision(list(range(5)))) == 2
    assert len(c.created) == 5


def test_a_topped_up_fleet_asks_for_a_size_not_for_the_gaps():
    """`top_up(8)` on a fleet of 3 must rent 5, not 8."""
    c = FakeClient()
    f = _fleet(c)
    f.provision(list(range(3)))
    assert len(c.created) == 3
    f.top_up(8, cap=8)
    assert len(c.created) == 8, c.created
    assert sorted(f.pods) == list(range(8)), "ids are not dense"


def test_shard_ids_are_dense_even_when_capacity_is_patchy():
    """A four-pod fleet out of a 32-shard plan reported shards 17, 20, 24, 25.

    Legitimate plan indices, but unreadable and impossible to match against a
    pod list -- and safe to renumber now that an id names a pod rather than
    selecting its slice.
    """
    c = FakeClient()
    f = _fleet(c)
    f.provision(list(range(4)))
    assert sorted(f.pods) == [0, 1, 2, 3]
