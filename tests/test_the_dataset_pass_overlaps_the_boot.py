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

"""The dataset is encoded WHILE the pods boot, not before they are asked for.

Measured on the real dataset: the pass is about three minutes of CPU, and a
pod takes six to ten minutes to publish a port. Run in sequence that is nine
minutes to the first trained example, three of them for nothing --

    20:48:49  request accepted
    20:51:46  7308 examples, 2868079 trained tokens      <- the pass ended
    20:51:47  shard 0: pod jor2t961z3wu5d rented         <- only now

-- and the pass is what the client watches as "preparing" while no card has
even been requested.

Nothing about the fleet depends on the pass. The pod count comes from the
config; the pods are sent the raw dataset FILE and encode it themselves; and
this pass exists to refuse a bad dataset, not to produce anything a pod
needs. The only consumer is the round planner, which needs the example count
-- and that is not needed until the first round is handed out.

The cost of the overlap is bounded and stated: a dataset that fails
validation now does so a few pod-minutes in rather than for free. Pods are
released by the same `finally` that covers every other failure.
"""

import threading
import time

import pytest

import _fleetfake as ff
from codai.api import loras
from codai.api import text_lora_fleet as flt


def test_the_rental_happens_before_the_pass_finishes(monkeypatch, tmp_path):
    """The whole claim, timed.

    `prepare` takes 0.5s; the pod is addressable at once. If the pass ran
    first, the rental would be stamped after it.
    """
    marks = {}
    fleet = ff.FakeFleet({0: 0.0})

    _real_provision = fleet.provision

    def _timed_provision(shards):
        marks["rented"] = time.time()
        return _real_provision(shards)
    fleet.provision = _timed_provision

    def _prepare():
        time.sleep(0.5)
        marks["prepared"] = time.time()
        return [object()] * 32, _stats(), {"examples": 32}

    ff.run(monkeypatch, fleet=fleet, home=tmp_path / "loras",
           job="job-ovl", name="adapter-ovl",
           request=ff.req("adapter-ovl", pods=1),
           config=ff.cfg(max_pods=1, pods_start=1),
           finish_after=1.2, prepare=_prepare)

    assert "rented" in marks and "prepared" in marks
    assert marks["rented"] < marks["prepared"], (
        "the pods were asked for only after the dataset pass finished")


def test_the_pass_still_gates_the_first_round(monkeypatch, tmp_path):
    """Overlapped, not skipped: the round planner needs the example count, so
    nothing is handed out until the pass has landed."""
    order = []
    fleet = ff.FakeFleet({0: 0.0})

    def _prepare():
        time.sleep(0.4)
        order.append("prepared")
        return [object()] * 8, _stats(), {"examples": 8}

    def _start_round(pod, payload):
        order.append("round")
    monkeypatch.setattr(flt, "start_round", _start_round)

    ff.run(monkeypatch, fleet=fleet, home=tmp_path / "loras",
           job="job-gate", name="adapter-gate",
           request=ff.req("adapter-gate", pods=1),
           config=ff.cfg(max_pods=1, pods_start=1),
           finish_after=1.0, prepare=_prepare)

    assert order and order[0] == "prepared", order


def test_a_dataset_that_fails_validation_still_fails_the_job(monkeypatch,
                                                             tmp_path):
    """And the pods go. The overlap must not turn a refusal into a leak."""
    fleet = ff.FakeFleet({0: 0.0})

    def _prepare():
        time.sleep(0.2)
        raise RuntimeError("the dataset has 0 usable examples")

    with pytest.raises(RuntimeError) as exc:
        ff.run(monkeypatch, fleet=fleet, home=tmp_path / "loras",
               job="job-bad", name="adapter-bad",
               request=ff.req("adapter-bad", pods=1),
               config=ff.cfg(max_pods=1, pods_start=1),
               finish_after=5.0, prepare=_prepare)
    assert "0 usable examples" in str(exc.value)
    # released_all ran: nothing is still held.
    assert fleet.pods == {}, fleet.pods


def test_a_slow_pass_is_waited_for_and_said_so(monkeypatch, tmp_path, capsys):
    """When the dataset is the slow side, the run says it waited -- that is
    the number that tells an operator which half to attack."""
    fleet = ff.FakeFleet({0: 0.0})

    def _prepare():
        time.sleep(1.3)
        return [object()] * 8, _stats(), {"examples": 8}

    ff.run(monkeypatch, fleet=fleet, home=tmp_path / "loras",
           job="job-slow", name="adapter-slow",
           request=ff.req("adapter-slow", pods=1),
           config=ff.cfg(max_pods=1, pods_start=1),
           finish_after=1.8, prepare=_prepare)

    out = capsys.readouterr().out
    assert "dataset ready" in out
    assert "waited" in out


def test_a_dry_run_rents_nothing(monkeypatch, tmp_path):
    """It is the dataset pass and nothing else, so it must not reach a fleet.

    Executed, because the overlap moved the pass behind a callable and the
    risk is precisely that a dry run now falls through to the renting path.
    """
    import types
    import transformers

    monkeypatch.setattr(loras, "_jobs", {})
    monkeypatch.setattr(loras, "_jobs_lock", threading.Lock())
    monkeypatch.setattr(loras, "_update_job", lambda *a, **k: None)
    monkeypatch.setattr(loras, "_resolve_text_base", lambda b: "stub-base")
    monkeypatch.setattr(flt, "config_for_model",
                        lambda m: flt.LoraTrainConfig(
                            enabled=True, pods_start=8, max_pods=16, rounds=16,
                            base_model="stub-base"))

    class _Tok:
        @staticmethod
        def from_pretrained(*a, **k):
            return object()
    monkeypatch.setattr(transformers, "AutoTokenizer", _Tok)
    monkeypatch.setattr(loras, "_text_dataset_pass",
                        lambda req, path, tok: ([object()] * 7308, _stats()))

    def _no(*a, **k):
        raise AssertionError("a dry run must not build a fleet")
    monkeypatch.setattr(flt, "TrainerFleet", _no)
    monkeypatch.setattr(flt, "eligible_cards", _no)

    req = types.SimpleNamespace(
        name="adapter-dry", base_model="qwen38-train", pods=8, min_pods=1,
        max_pods=16, rounds=16, test_run=False, distributed=True,
        dataset_id="", field_map=None, dry_run=True, rank=16, alpha=32,
        dropout=0.05, learning_rate=1e-4, max_len=8192,
        accumulation_steps=8, enable_thinking=False, seed=42,
        test_examples=8, max_pod_hourly_usd=None, max_fleet_hourly_usd=None)

    got = loras._text_train_blocking(req, "job-dry", str(tmp_path / "d"))
    assert got["dry_run"] is True
    assert got["examples"] == 8


def _stats():
    import types
    return types.SimpleNamespace(examples=8, skipped=[], trained_tokens=1,
                                 masked_tokens=0,
                                 describe=lambda: "8 examples")
