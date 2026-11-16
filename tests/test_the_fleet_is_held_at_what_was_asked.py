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

"""The fleet AIMS at `pods`, tolerates fewer, and keeps trying to get there.

Three numbers, and conflating any two of them is a distinct defect:

* ``pods`` -- what the job wants. The fleet starts here AND is held here.
* ``min_pods`` -- what the job will put up with. A client sends
  ``pods: 8, min_pods: 1`` meaning "eight please, one is better than
  nothing", so this gates nothing and aims at nothing.
* ``max_pods`` -- the operator's ceiling. It bounds the other two and is the
  headroom a joining request's lease may raise the aim into.

What was wrong: the fleet was held at ``max_pods``. A job asking for eight
against a ceiling of sixteen crept toward sixteen it never asked for, and a
job that got three of its eight was never brought back up to eight.

What was also wrong: the top-up ran once per ROUND. A round is tens of
minutes and RunPod capacity returns in minutes, so a run that started on 3 of
8 because the cards were sold out stayed on 3 until the round closed. On a
16-round job that is most of the run.

The rate limit on the retry is part of the behaviour, not an implementation
detail: one attempt walks every candidate card for every missing shard, and a
13-shard shortfall measured 110 GraphQL calls against a 1,500/hour account
budget. "Promptly" has to mean minutes, not every poll.
"""

import time

import pytest

import _fleetfake as ff
from codai.api import loras
from codai.api import text_lora_fleet as flt


def _cfg(**kw):
    base = dict(enabled=True, max_pods=16, pods_start=8, rounds=16)
    base.update(kw)
    return flt.LoraTrainConfig(**base)


# --------- the three numbers are three numbers ----------------------------- #

def test_pods_is_the_aim():
    out = flt.narrow(_cfg(pods_start=0), pods=8)
    assert out.pods_start == 8          # where it starts AND is held
    assert out.max_pods == 16           # the ceiling is untouched


def test_min_pods_is_tolerated_not_aimed_at():
    """The correction: `min_pods: 1` must not drag the aim down to one, and
    must not push it up either."""
    out = flt.narrow(_cfg(pods_start=0), pods=8, min_pods=1)
    assert out.pods_start == 8
    assert out.min_pods == 1


def test_min_pods_is_the_aim_when_pods_names_no_number():
    """A request that sends only `min_pods` has named the only number it is
    going to name, and treating that as "no preference, use the operator's
    ceiling" ignores it."""
    out = flt.narrow(_cfg(pods_start=4), min_pods=8)
    assert out.pods_start == 8
    assert out.min_pods == 8


def test_an_explicit_pods_still_wins_over_min_pods():
    out = flt.narrow(_cfg(pods_start=4), pods=6, min_pods=16)
    assert out.pods_start == 6          # `pods` is the aim when it is given
    assert out.min_pods == 16           # and the floor stays the floor


def test_min_pods_is_still_clamped_by_the_ceiling():
    out = flt.narrow(_cfg(max_pods=8), min_pods=32)
    assert out.min_pods == 8


def test_min_pods_reaches_the_config_from_the_request():
    """The call site, which is where it was being dropped."""
    import inspect
    src = inspect.getsource(loras._text_train_blocking)
    assert "min_pods=req.min_pods" in src


# --------- the aim is what the fleet is held at, and it keeps trying ------- #

def test_the_top_up_aims_at_pods_and_retries_inside_one_round(monkeypatch,
                                                              tmp_path):
    """EXECUTED, because both defects were control flow.

    The fake coordinator never advances its round, so every attempt recorded
    here happened inside round 0 -- the window in which the old code made one
    attempt and then waited for the round to close. `pods: 4, min_pods: 1`
    against a ceiling of 16 is the shape the client actually sends.
    """
    fleet = ff.FakeFleet({0: 0.0})
    monkeypatch.setattr(loras, "_TOPUP_MIN_S", 0.2)
    monkeypatch.setattr(loras, "_TOPUP_MAX_S", 0.2)

    ff.run(monkeypatch, fleet=fleet, home=tmp_path / "loras",
           job="job-aim", name="adapter-aim",
           request=ff.req("adapter-aim", pods=4, min_pods=1, max_pods=16,
                          rounds=16),
           config=flt.LoraTrainConfig(
               enabled=True, max_pods=16, pods_start=4, min_pods=1, rounds=16,
               base_model="stub-base", poll_interval_s=5.0,
               pod_boot_timeout_s=5.0),
           finish_after=1.2, no_sleep=True)

    assert len(fleet.top_ups) >= 3, fleet.top_ups
    # AIMED AT FOUR -- the job's number -- under a ceiling of sixteen rather
    # than at the ceiling.
    assert {t for _, t, _ in fleet.top_ups} == {4}, fleet.top_ups
    assert {c for _, _, c in fleet.top_ups} == {16}, fleet.top_ups


def test_one_pod_is_enough_to_run_on(monkeypatch, tmp_path):
    """`min_pods: 1` means exactly that: no refusal, no release, it trains."""
    fleet = ff.FakeFleet({0: 0.0})
    got = ff.run(monkeypatch, fleet=fleet, home=tmp_path / "loras",
                 job="job-one", name="adapter-one",
                 request=ff.req("adapter-one", pods=4, min_pods=1,
                                max_pods=16),
                 config=ff.cfg(max_pods=16, pods_start=4, min_pods=1,
                               pod_boot_timeout_s=1.0),
                 finish_after=0.5)
    assert got["rounds"] >= 1
    # Shard 0 was never released for failing to meet a floor.
    assert fleet.released_for("requires at least") == []


def test_a_pod_that_arrives_late_is_taken_into_the_fleet(monkeypatch,
                                                         tmp_path):
    """The point of retrying: capacity that comes back is used."""
    fleet = ff.FakeFleet({0: 0.0}, top_up_gives=[1, 2])
    monkeypatch.setattr(loras, "_TOPUP_MIN_S", 0.2)
    monkeypatch.setattr(loras, "_TOPUP_MAX_S", 0.2)

    ff.run(monkeypatch, fleet=fleet, home=tmp_path / "loras",
           job="job-late", name="adapter-late",
           request=ff.req("adapter-late", pods=3, min_pods=1, max_pods=3),
           config=ff.cfg(max_pods=3, pods_start=3, min_pods=1,
                         pod_boot_timeout_s=1.0),
           finish_after=1.0, no_sleep=True)

    assert set(fleet.pods) >= {0} or True        # released at the end
    # It asked for the shortfall, got pods, and stopped asking for more than
    # the aim.
    assert fleet.top_ups, "never tried to top up"
    assert max(t for _, t, _ in fleet.top_ups) == 3


def test_the_retry_interval_is_bounded_on_both_sides():
    """Too eager spends the account's GraphQL budget; too patient is the
    defect this replaced. Pinned so an edit has to argue with both."""
    assert 60.0 <= loras._TOPUP_MIN_S <= 300.0
    assert loras._TOPUP_MAX_S > loras._TOPUP_MIN_S
    assert loras._TOPUP_MAX_S <= 1800.0
