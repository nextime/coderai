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

"""A pod that finishes its block takes another, not a seat.

The exchange waits for the slowest contributor, and the fleet is deliberately
unequal -- A40s, L40S, A6000s, whatever had capacity under the ceiling. So the
first pod to finish used to idle, billing, until the last one caught up: the
round loop skipped any pod that had already submitted.

It takes another block instead. That is sound for the same reason a late
arrival is: every contribution in an exchange starts from the same pushed
adapter and covers examples no one else has, and the average is weighted by
examples trained, so a pod that got through four blocks counts four times as
much as one that managed one -- which is exactly right.

Three things have to hold, and each was broken in a different way:

* no block is handed out twice in one exchange. The block counter wraps by
  epoch so a long fleet keeps finding work, and a wrap INSIDE one round would
  reintroduce the duplication this design was built to remove;
* readiness is judged on SHARDS. One fast pod with four submissions must not
  make a four-pod round look complete;
* the files must not collide. `r{rnd}-s{shard}.safetensors` named one file per
  pod per round, so a second contribution landed on top of the first --
  averaging one adapter twice and losing the other's work.
"""

import types

import pytest

from codai.api import loras
from codai.api import text_lora_rounds as tlr
from codai.api import text_lora_trainer as tlt


def _coord(n=400, n_shards=4, rounds=4):
    plan = tlr.RoundPlan(n_shards=n_shards, seed=42)
    t = tlr.RoundTracker(plan, min_fraction=0.75, deadline_s=0)
    c = loras._TextTrainCoordinator(
        "j", plan=plan, tracker=t, fleet=types.SimpleNamespace(pods={}),
        encoded=[0] * n, out_dir="/tmp", settings=tlt.QLoraSettings(),
        dataset_id="", base="b", field_map=None, total_rounds=rounds)
    return c, t


def _finish(c, t, shard, rnd=0):
    ix = c.assignment(shard)["indices"]
    t.submit(tlr.Submission(shard=shard, round=rnd, path=f"/{shard}-{len(ix)}",
                            examples=len(ix)))
    c.block_done(shard, rnd)
    return ix


def test_a_fast_pod_gets_more_work_and_it_is_all_distinct():
    c, t = _coord()
    c.cohort(0, [0, 1], grow=True)
    got = [_finish(c, t, 0) for _ in range(4)]
    got.append(c.assignment(1)["indices"])
    flat = [i for b in got for i in b]
    assert len(set(flat)) == len(flat), "a pod was given examples twice"
    assert all(b for b in got), "a handout was empty while work remained"


def test_a_held_block_is_handed_back_unchanged_on_a_retry():
    """A PodBusy or a failed push must not spend work."""
    c, t = _coord()
    c.cohort(0, [1], grow=True)
    first = c.assignment(1)["indices"]
    assert c.assignment(1)["indices"] == first
    assert c.block_for(1, 0) == c.block_for(1, 0)


def test_one_fast_pod_cannot_close_a_round_on_its_own():
    c, t = _coord()
    c.cohort(0, [0, 1, 2, 3], grow=True)
    for _ in range(3):
        _finish(c, t, 0)
    assert t.contributors(0) == 1
    assert t.ready() is False, "three submissions from one pod closed a round"
    _finish(c, t, 1)
    _finish(c, t, 2)
    assert t.ready() is True          # 3 of 4 shards, min_fraction 0.75
    assert t.missing() == [3]


def test_every_contribution_is_averaged_with_its_own_weight():
    c, t = _coord()
    c.cohort(0, [0, 1], grow=True)
    fast = [_finish(c, t, 0) for _ in range(3)]
    slow = _finish(c, t, 1)
    subs = t.have(0)
    assert len(subs) == 4
    weights = t.weights_for(subs)
    assert sum(weights) == sum(len(b) for b in fast) + len(slow)
    # the fast pod's share is three blocks' worth, not one
    assert sum(weights) > 3 * len(slow)


def test_an_exhausted_exchange_hands_out_nothing():
    """Better an idle pod than the same examples twice in one average."""
    c, t = _coord(n=100, n_shards=4, rounds=4)
    c.cohort(0, [0], grow=True)
    n = 0
    while c.blocks_left(0):
        _finish(c, t, 0)
        n += 1
    assert n > 0
    assert c.assignment(0)["indices"] == []
    assert c.blocks_left(0) == 0


def test_a_recycled_block_can_be_issued_again_in_the_same_exchange():
    """Its pod died before training it, so it is not 'already covered'."""
    c, t = _coord()
    c.cohort(0, [0, 1], grow=True)
    lost = c.assignment(1)["indices"]
    assert c.recycle([1], rnd=0)
    c.cohort(0, [0, 1, 2], grow=True)
    assert c.assignment(2)["indices"] == lost


def test_the_submission_file_is_named_per_block():
    import inspect
    src = inspect.getsource(loras._text_train_distributed)
    assert 'f"r{rnd}-s{shard}-"' in src
    assert "b{coord.block_for(shard, rnd)}.safetensors" in src


def test_the_loop_no_longer_skips_a_pod_that_submitted():
    import inspect
    src = inspect.getsource(loras._text_train_distributed)
    assert "any(s.shard == shard\n                                      for s in coord.tracker.have(rnd))" \
        not in src
    assert "A POD THAT ALREADY SUBMITTED IS NOT SKIPPED" in src
