# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""A fleet whose owner died must not bill until somebody restarts the box.

`sweep_orphans` runs once, at startup, and its safety argument is explicit:
a record on disk must be abandoned because no job is running yet in this
process. That is true at startup and nowhere else -- so a fleet orphaned
while the container kept running had nothing coming for it. Measured once:
`job-53f626ce4dab` left 16 pods at roughly $9.50/hr, and the only thing that
collected them was the next boot's sweep, hours later.

The fix is a heartbeat. A live run stamps its record continuously, so age is
the signal and a periodic sweep can act without waiting for a restart. The
case these tests care most about is the dangerous one: a RUNNING fleet must
never be swept, however long its round takes.
"""

import json
import os
import time

import pytest

from codai.api import text_lora_fleet as flt


@pytest.fixture()
def home(tmp_path, monkeypatch):
    from codai.api import loras
    d = tmp_path / "loras"
    d.mkdir()
    monkeypatch.setattr(loras, "_loras_dir", lambda: str(d))
    return d


class FakeClient:
    def __init__(self, fail=()):
        self.terminated = []
        self.fail = set(fail)

    def terminate_pod(self, pod_id):
        if pod_id in self.fail:
            raise RuntimeError(f"RunPod said no to {pod_id}")
        self.terminated.append(pod_id)


def _record(job, pods, *, age_s=0.0):
    """Write a fleet record aged `age_s` seconds into the past."""
    rows = [{"pod_id": p, "shard": i, "hourly_usd": 0.59,
             "started_at": time.time()} for i, p in enumerate(pods)]
    flt._write_state(job, rows)
    if age_s:
        path = flt._state_path(job)
        with open(path, encoding="utf-8") as fh:
            env = json.load(fh)
        env["at"] = time.time() - age_s
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(env, fh)
    return rows


# --------------------------------------------------------------------- #
# the dangerous direction first
# --------------------------------------------------------------------- #

def test_a_live_fleet_is_never_swept(home):
    """The record was stamped a moment ago, so it has an owner."""
    _record("job-live", ["p1", "p2"])
    client = FakeClient()

    out = flt.sweep_abandoned(client=client)

    assert client.terminated == [], "a reporting fleet must be left alone"
    assert out["pods"] == 0
    assert os.path.isfile(flt._state_path("job-live"))


def test_a_long_round_does_not_look_abandoned(home):
    """Whatever the round costs in wall clock, the heartbeat is what counts."""
    _record("job-slow", ["p1"], age_s=flt.ABANDONED_AFTER_S * 4)
    client = FakeClient()

    # The owner is alive and says so, exactly as the run's wait loops do.
    flt.touch("job-slow")
    assert flt.abandoned() == []
    assert flt.sweep_abandoned(client=client)["pods"] == 0
    assert client.terminated == []


def test_the_heartbeat_moves_the_clock_forward(home):
    _record("job-beat", ["p1"], age_s=flt.ABANDONED_AFTER_S + 60)
    assert len(flt.abandoned()) == 1, "stale before the beat"
    flt.touch("job-beat")
    assert flt.abandoned() == [], "and not stale after it"


def test_a_heartbeat_on_an_empty_record_writes_nothing(home):
    """Nothing to keep alive, and no file conjured for a job with no pods."""
    flt.touch("job-nothing")
    assert not os.path.isfile(flt._state_path("job-nothing"))


# --------------------------------------------------------------------- #
# and now the leak itself
# --------------------------------------------------------------------- #

def test_an_unreported_fleet_is_terminated(home):
    _record("job-dead", ["p1", "p2", "p3"],
            age_s=flt.ABANDONED_AFTER_S + 1)
    client = FakeClient()

    out = flt.sweep_abandoned(client=client)

    assert sorted(client.terminated) == ["p1", "p2", "p3"]
    assert out["pods"] == 3 and out["jobs"] == 1
    assert not os.path.isfile(flt._state_path("job-dead")), \
        "the record goes when the last pod in it does"


def test_only_the_stale_record_is_touched(home):
    _record("job-live", ["alive1", "alive2"])
    _record("job-dead", ["dead1"], age_s=flt.ABANDONED_AFTER_S + 1)
    client = FakeClient()

    flt.sweep_abandoned(client=client)

    assert client.terminated == ["dead1"]
    assert os.path.isfile(flt._state_path("job-live"))


def test_a_record_with_no_heartbeat_at_all_is_stale(home):
    """Written by a process that predates the heartbeat, so certainly gone."""
    flt._write_state("job-old", [{"pod_id": "p9", "shard": 0,
                                  "hourly_usd": 0.59}])
    path = flt._state_path("job-old")
    with open(path, encoding="utf-8") as fh:
        env = json.load(fh)
    env.pop("at", None)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(env, fh)

    assert [r["pod_id"] for r in flt.abandoned()] == ["p9"]


def test_a_pod_that_will_not_die_stays_in_the_record(home):
    """Dropping it would make it invisible AND billing."""
    _record("job-dead", ["p1", "stubborn"],
            age_s=flt.ABANDONED_AFTER_S + 1)
    client = FakeClient(fail=["stubborn"])

    out = flt.sweep_abandoned(client=client)

    assert out["failed"] == 1 and out["pods"] == 1
    left = [r["pod_id"] for r in flt._read_state("job-dead")]
    assert left == ["stubborn"], "kept for the next sweep to try again"


def test_a_pending_pod_is_not_mistaken_for_a_real_one(home):
    flt._write_state("job-dead", [{"pod_id": "pending-xyz", "shard": 0,
                                   "hourly_usd": 0.0}])
    path = flt._state_path("job-dead")
    with open(path, encoding="utf-8") as fh:
        env = json.load(fh)
    env["at"] = time.time() - flt.ABANDONED_AFTER_S - 1
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(env, fh)
    client = FakeClient()

    flt.sweep_abandoned(client=client)
    assert client.terminated == [], "there is no pod id to terminate yet"


def test_no_runpod_client_leaves_the_record_alone(home):
    """Better the reaper's grace than a forgotten pod."""
    _record("job-dead", ["p1"], age_s=flt.ABANDONED_AFTER_S + 1)

    def _boom():
        raise RuntimeError("no account")

    import codai.models.manager as mgr
    orig = mgr.get_active_runpod_config
    mgr.get_active_runpod_config = _boom
    try:
        out = flt.sweep_abandoned()
    finally:
        mgr.get_active_runpod_config = orig

    assert out["pods"] == 0
    assert flt._read_state("job-dead"), "still recorded, so still findable"


# --------------------------------------------------------------------- #
# the startup sweep keeps its own, harsher rule
# --------------------------------------------------------------------- #

def test_the_startup_sweep_still_ignores_the_heartbeat(home):
    """At startup nothing is running, so age is irrelevant."""
    _record("job-fresh", ["p1"])
    client = FakeClient()

    out = flt.sweep_orphans(client=client)

    assert client.terminated == ["p1"], \
        "a record that survived a restart is abandoned whatever its age"
    assert out["pods"] == 1


# --------------------------------------------------------------------- #
# the wiring
# --------------------------------------------------------------------- #

def test_the_run_reports_while_it_waits_and_while_it_trains():
    import inspect
    from codai.api import loras
    src = inspect.getsource(loras._text_train_distributed)
    assert src.count("fleet.heartbeat()") >= 2, (
        "the boot wait and the round loop must both report: pods bill "
        "during the boot too, and that wait is minutes long")


def test_the_sweep_runs_on_a_timer():
    import inspect
    from codai.api import runpod_worker as rw
    src = inspect.getsource(rw._scaler_loop)
    assert "sweep_abandoned" in src, \
        "a startup-only sweep is what let the leak run until a restart"


def test_the_fleet_can_say_it_is_alive():
    assert hasattr(flt.TrainerFleet, "heartbeat")


# --------------------------------------------------------------------- #
# the records a finished fleet leaves behind
# --------------------------------------------------------------------- #

def test_an_empty_stale_record_is_forgotten(home):
    """83 of these had piled up in a day. No pods, but no bound either."""
    flt._write_state("job-emptied", [])
    path = flt._state_path("job-emptied")
    with open(path, encoding="utf-8") as fh:
        env = json.load(fh)
    env["at"] = time.time() - flt.ABANDONED_AFTER_S - 1
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(env, fh)

    out = flt.sweep_abandoned(client=FakeClient())

    assert out["forgotten"] == 1
    assert not os.path.isfile(path)


def test_forgetting_records_is_reported(home, capsys):
    """A silent cleanup cannot be told from one that never ran."""
    for job in ("job-a", "job-b"):
        flt._write_state(job, [])
        path = flt._state_path(job)
        with open(path, encoding="utf-8") as fh:
            env = json.load(fh)
        env["at"] = time.time() - flt.ABANDONED_AFTER_S - 1
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(env, fh)

    flt.sweep_abandoned(client=FakeClient())

    said = capsys.readouterr().out
    assert "forgot 2 empty" in said, said


def test_a_quiet_sweep_says_nothing(home, capsys):
    """Every two minutes forever: it must only speak when it acted."""
    flt._write_state("job-live", [])
    flt.sweep_abandoned(client=FakeClient())
    assert "forgot" not in capsys.readouterr().out


def test_a_fresh_empty_record_is_left_alone(home):
    """A job between its first record and its first rental is empty."""
    flt._write_state("job-starting", [])
    out = flt.sweep_abandoned(client=FakeClient())
    assert out["forgotten"] == 0
    assert os.path.isfile(flt._state_path("job-starting"))


def test_forgetting_empties_does_not_touch_a_record_with_pods(home):
    _record("job-dead", ["p1"], age_s=flt.ABANDONED_AFTER_S + 1)
    client = FakeClient(fail=["p1"])

    flt.sweep_abandoned(client=client)

    assert flt._read_state("job-dead"), \
        "a pod that would not die must not be forgotten as an empty record"
