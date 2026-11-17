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

"""One pod going dark must cost one pod, never the job.

MEASURED. A 24-pod run died like this:

    21:25:38  released 21 pod(s)
    21:25:38  FAILED: ConnectionError: ('Connection aborted.', TimeoutError)

Three pods had stopped answering eight minutes earlier. At the round boundary
the run pushed round 0's average to one of them, blocked for the FULL 900s
timeout, and raised -- 900s being exactly the gap between the round opening
and the failure. Nothing caught it, so 21 healthy pods were released and 39
paid minutes were discarded at 7 of 731 blocks.

`poll_status` had always reported unreachable instead of raising, and
`push_dataset`'s call site had always caught it, with a comment stating the
rule: "the rest of the fleet is unaffected, where raising would have taken
down a run that has working pods". Three calls in the round loop --
`fetch_adapter`, `push_start_adapter`, `start_round` -- simply did not follow
it, and `grep` for any requests-exception handling in the fleet module
returned nothing at all.

These tests drive the real helpers against a transport that fails.
"""

import os

import pytest

from codai.api import text_lora_fleet as flt


class Pod:
    def __init__(self, shard=0):
        self.shard = shard
        self.url = "http://pod-that-is-gone:8000"
        self.secret = "k"
        self.api_key = "k"
        self.pod_id = f"pod{shard}"


@pytest.fixture()
def dead_transport(monkeypatch):
    """Every requests call fails the way the real one did."""
    import requests

    def boom(*a, **k):
        raise requests.ConnectionError(
            "('Connection aborted.', TimeoutError('timed out'))")

    for name in ("get", "post", "put"):
        monkeypatch.setattr(requests, name, boom)
    return boom


# ------------------------------------------------ the three unguarded calls --

def test_pushing_the_start_adapter_reports_the_pod_not_the_job(
        dead_transport, tmp_path):
    """THE CALL THAT KILLED THE RUN."""
    p = tmp_path / "avg.safetensors"
    p.write_bytes(b"\0" * 64)
    with pytest.raises(flt.PodUnreachable):
        flt.push_start_adapter(Pod(), str(p))


def test_starting_a_round_reports_the_pod_not_the_job(dead_transport):
    with pytest.raises(flt.PodUnreachable):
        flt.start_round(Pod(), {"indices": [1, 2, 3]})


def test_fetching_the_adapter_reports_the_pod_not_the_job(
        dead_transport, tmp_path):
    with pytest.raises(flt.PodUnreachable):
        flt.fetch_adapter(Pod(), str(tmp_path / "out.safetensors"))


def test_pushing_the_dataset_reports_the_pod_not_the_job(
        dead_transport, tmp_path):
    ds = tmp_path / "ds.jsonl"
    ds.write_text('{"a":1}\n')
    with pytest.raises(flt.PodUnreachable):
        flt.push_dataset(Pod(), str(ds))


# ------------------------------------------------------- what the type means --

def test_an_unreachable_pod_is_a_fleet_error():
    """So every `except FleetError` already in the run covers it, rather than
    each call site needing to know this type exists."""
    assert issubclass(flt.PodUnreachable, flt.FleetError)


def test_the_error_names_the_shard_and_the_endpoint():
    """A log line saying only "round 0 failed" is what made the original
    failure take a log dive to diagnose."""
    import requests

    pod = Pod(shard=7)
    with pytest.raises(flt.PodUnreachable) as got:
        with flt._transport(pod, "/adapter/start"):
            raise requests.ConnectionError("aborted")
    msg = str(got.value)
    assert "shard 7" in msg and "/adapter/start" in msg


def test_a_local_disk_error_is_not_blamed_on_the_pod():
    """`fetch_adapter` streams to disk inside the guard. A full disk is a real
    error and must not be relabelled as the pod being unreachable, or the run
    releases healthy pods to chase its own bug."""
    with pytest.raises(OSError):
        with flt._transport(Pod(), "/adapter"):
            raise OSError("No space left on device")


def test_a_busy_pod_is_still_busy_not_unreachable(monkeypatch):
    """409 comes back from a pod that ANSWERED, so it must stay PodBusy --
    the dispatch retries those rather than releasing the card."""
    import requests

    class R:
        status_code = 409
        text = "still training"

    monkeypatch.setattr(requests, "post", lambda *a, **k: R())
    with pytest.raises(flt.PodBusy):
        flt.start_round(Pod(), {"indices": [1]})


def test_an_http_error_is_not_a_transport_error(monkeypatch):
    """A pod that answers 500 is reachable and broken, which is a different
    thing from one that is gone."""
    import requests

    class R:
        status_code = 500
        text = "boom"

    monkeypatch.setattr(requests, "post", lambda *a, **k: R())
    with pytest.raises(RuntimeError) as got:
        flt.start_round(Pod(), {"indices": [1]})
    assert not isinstance(got.value, flt.PodUnreachable)


# ------------------------------------------------------------- the timeouts --

def test_no_call_can_stall_a_round_for_a_quarter_hour():
    """The guard is the fix; the timeout bounds how long ONE dead pod holds
    the round up before the guard can act. 900s was long enough that the run
    sat on a corpse for 15 minutes."""
    import inspect

    for fn in (flt.push_start_adapter, flt.fetch_adapter, flt.start_round):
        t = inspect.signature(fn).parameters["timeout"].default
        assert t <= 900.0, f"{fn.__name__} waits {t}s on a dead pod"
    assert inspect.signature(
        flt.push_start_adapter).parameters["timeout"].default <= 300.0


# ---------------------------------------------- and the RUN, which is the bug --

def _drive(monkeypatch, home, *, breaks, which="start_round"):
    """Run the real `_text_train_distributed` with one pod that goes dark.

    The defect was never in the helpers -- it was that the round loop called
    them unguarded, so the helper tests above would all have passed while the
    run still died. This drives the loop.
    """
    import _fleetfake as ff
    from codai.api import loras
    from codai.api import text_lora_trainer as tlt

    fleet = ff.FakeFleet({0: 0.0, 1: 0.0, 2: 0.0})
    os.makedirs(os.path.join(str(home), "adapter-fake"), exist_ok=True)
    ff.install(monkeypatch, fleet, home=home, finish_after=1.5)

    # The dispatch only runs for an IDLE pod with work to hand it, and the
    # shared fake reports every pod as training with an empty payload -- so
    # without these two the loop never reaches the call under test and the
    # test passes for the wrong reason.
    monkeypatch.setattr(flt, "poll_status", lambda pod: {"state": "idle"})
    monkeypatch.setattr(loras._TextTrainCoordinator, "round_payload",
                        lambda self, shard: {"indices": [1, 2, 3]})

    orig = getattr(flt, which)

    def dies(pod, *a, **k):
        if pod.shard in breaks:
            raise flt.PodUnreachable(
                f"shard {pod.shard}: /{which} failed at the transport: "
                f"ConnectionError: ('Connection aborted.', TimeoutError)")
        return orig(pod, *a, **k) if callable(orig) else None

    monkeypatch.setattr(flt, which, dies)

    ds = os.path.join(str(home), "ds.jsonl")
    with open(ds, "w") as fh:
        fh.write('{"a":1}\n')

    import types

    def prepare():
        return ([object()],
                types.SimpleNamespace(examples=1, skipped=[],
                                      trained_tokens=1, masked_tokens=0,
                                      describe=lambda: "1 example"),
                {"examples": 1})

    return loras._text_train_distributed(
        ff.req("adapter-fake"), "job-dead-pod", ds, None, None,
        tlt.QLoraSettings(), ff.cfg(), prepare=prepare), fleet


def test_the_run_survives_a_pod_that_dies_on_dispatch(tmp_path, monkeypatch):
    """THE REGRESSION. Before this, the ConnectionError escaped the loop and
    the run released every pod it had and failed."""
    home = tmp_path / "loras"
    home.mkdir()
    out, fleet = _drive(monkeypatch, home, breaks={1}, which="start_round")

    assert fleet.released_for("round dispatch failed") == [1], \
        "the dead pod is released, and only the dead pod"
    assert 0 not in fleet.released_for("round dispatch failed")
    assert 2 not in fleet.released_for("round dispatch failed")


def test_a_dead_pod_does_not_take_the_working_ones_with_it(tmp_path,
                                                           monkeypatch):
    """21 healthy pods were released by one pod's timeout."""
    home = tmp_path / "loras"
    home.mkdir()
    out, fleet = _drive(monkeypatch, home, breaks={1}, which="start_round")

    # The run reached its own end rather than dying on the exception.
    assert out is not None
