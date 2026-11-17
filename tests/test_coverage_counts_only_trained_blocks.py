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

"""Coverage must mean examples TRAINED, not blocks handed out.

Measured on the first distributed run that ever reached the end
(16 pods, 10 exchanges, 3h04m): it reported 252/252 blocks and
7308/7308 examples at 100%, while the averaging weights -- which are
the examples each pod actually trained -- summed to 859. One eighth of
one epoch, reported as a finished epoch, and the adapter shipped.

Two independent ways that happened, both here:

* a pod that has submitted still reports ``done`` for the same round,
  because nothing has started it on anything else yet. The poll loop
  came round, built the destination filename by calling ``blocks_for``
  -- which is a HANDOUT, not a query -- and so consumed a fresh block,
  re-fetched the identical adapter, submitted it a second time and
  marked that new block covered. Nobody ever trained it.
* a submission carrying zero examples marked its block covered anyway.

Both move `_covered` without a gradient step behind it, and `_covered`
is what ends the run.
"""

import types

import pytest

from codai.api import loras
from codai.api import text_lora_rounds as tlr
from codai.api.text_lora_rounds import RoundPlan, RoundTracker
from codai.api import text_lora_trainer as tlt


def _coord(n_examples=400, n_shards=4, rounds=4):
    plan = RoundPlan(n_shards=n_shards, seed=42)
    t = RoundTracker(plan, min_fraction=0.75, deadline_s=0)
    c = loras._TextTrainCoordinator(
        "j", plan=plan, tracker=t, fleet=types.SimpleNamespace(pods={}),
        encoded=[0] * n_examples, out_dir="/tmp",
        settings=tlt.QLoraSettings(), dataset_id="", base="b",
        field_map=None, total_rounds=rounds)
    return c, t


def _per(c):
    return c.plan.block_size(len(c.encoded), c.total_rounds, c.planned_pods)


# --------------------------------------------------------------------- #
# the stale `done` that burned a block per poll
# --------------------------------------------------------------------- #

def test_a_pod_still_reporting_done_does_not_consume_another_block():
    """The poll loop comes round again on the SAME pod status."""
    c, t = _coord()
    c.cohort(0, [0], grow=True)
    t.open_round(0)

    held = c.blocks_for(0, 0)
    assert held, "the pod should get a block to train"
    per = _per(c)
    t.submit(tlr.Submission(shard=0, round=0, path="/a", examples=per))
    c.block_done(0, 0, examples=per)
    covered_once, _ = c.coverage()
    assert covered_once == 1

    # The pod has not retrained and has not been started on anything: its
    # status still reads done@0. Asking what it holds must not hand it work.
    assert c.blocks_held(0, 0) == [], \
        "a pod that already submitted holds nothing to collect again"
    left_before = c.blocks_left(0)
    assert c.blocks_held(0, 0) == []
    assert c.blocks_left(0) == left_before, \
        "querying what a pod holds must not consume a block"

    covered_twice, _ = c.coverage()
    assert covered_twice == covered_once, \
        "a repeated status must not advance coverage"


def test_what_a_pod_holds_is_a_query_not_a_handout():
    """`blocks_held` never mutates; `blocks_for` is the only handout."""
    c, _ = _coord()
    c.cohort(0, [0], grow=True)

    before = c.blocks_left(0)
    for _ in range(5):
        assert c.blocks_held(0, 0) == []
    assert c.blocks_left(0) == before

    got = c.blocks_for(0, 0)
    assert c.blocks_left(0) == before - 1
    # While it still holds it, both agree and neither consumes more.
    assert c.blocks_held(0, 0) == got
    assert c.blocks_for(0, 0) == got
    assert c.blocks_left(0) == before - 1


# --------------------------------------------------------------------- #
# zero work is not coverage
# --------------------------------------------------------------------- #

def test_a_submission_that_trained_nothing_covers_nothing():
    c, t = _coord()
    c.cohort(0, [0], grow=True)
    t.open_round(0)

    c.blocks_for(0, 0)
    t.submit(tlr.Submission(shard=0, round=0, path="/a", examples=0))
    c.block_done(0, 0, examples=0)

    done, _ = c.coverage()
    assert done == 0, "nothing was trained, so nothing is covered"


def test_an_untrained_block_goes_back_on_the_queue():
    """It must be trainable by somebody, not silently lost."""
    c, t = _coord()
    c.cohort(0, [0, 1], grow=True)
    t.open_round(0)

    mine = c.blocks_for(0, 0)
    c.block_done(0, 0, examples=0)
    # The next handout in a LATER exchange can pick it up again.
    t.round = 1
    c.cohort(1, [0, 1], grow=True)
    again = c.blocks_for(1, 1)
    assert again == mine, \
        "a block nobody trained is the first thing to hand out next"


def test_a_partial_block_counts_as_covered():
    """An OOM-skipped example or two must not reopen the whole block."""
    c, t = _coord()
    c.cohort(0, [0], grow=True)
    t.open_round(0)

    per = _per(c)
    c.blocks_for(0, 0)
    t.submit(tlr.Submission(shard=0, round=0, path="/a", examples=per - 2))
    c.block_done(0, 0, examples=per - 2)

    done, _ = c.coverage()
    assert done == 1


# --------------------------------------------------------------------- #
# the accounting the client is shown
# --------------------------------------------------------------------- #

def test_a_run_is_not_finished_on_blocks_nobody_trained():
    """The defect end to end: every block issued, nothing trained."""
    c, t = _coord(n_examples=120, n_shards=2, rounds=2)
    c.cohort(0, [0, 1], grow=True)
    t.open_round(0)

    for _ in range(c.total_blocks() * 4):
        for sh in (0, 1):
            if c.blocks_for(sh, 0):
                c.block_done(sh, 0, examples=0)

    assert not c.finished(), \
        "a run where no pod trained anything is not a finished epoch"
    done, _ = c.coverage()
    assert done == 0


# --------------------------------------------------------------------- #
# a fleet that trains nothing must not spin
# --------------------------------------------------------------------- #

def test_an_exchange_that_trained_nothing_ends_the_run(tmp_path):
    """Recycling untrained blocks must not become an unbounded rental.

    Coverage no longer advances on a bare reply, so the thing that used to
    end such a run -- falsely covering the set -- is gone. What stops it is
    the averaging itself: a weight vector that sums to zero has no mean, and
    the run fails there with the pods released by its `finally`. That is the
    honest end, and it is why no "barren fleet" counter is needed.
    """
    from _fleetfake import _write_tiny_adapter
    c, t = _coord(n_examples=120, n_shards=2, rounds=2)
    c.out_dir = str(tmp_path)
    c.cohort(0, [0, 1], grow=True)
    t.open_round(0)
    for sh in (0, 1):
        c.blocks_for(sh, 0)
        f = _write_tiny_adapter(str(tmp_path / f"s{sh}.safetensors"))
        t.submit(tlr.Submission(shard=sh, round=0, path=f, examples=0))

    with pytest.raises(tlr.RoundError, match="no shard did any work"):
        c.close_round_if_ready()

    done, _ = c.coverage()
    assert done == 0, "and nothing was counted as covered on the way out"


def test_one_pod_doing_work_still_closes_the_exchange(tmp_path):
    """A single productive pod keeps the run alive and advancing."""
    from _fleetfake import _write_tiny_adapter
    c, t = _coord(n_examples=120, n_shards=2, rounds=2)
    c.out_dir = str(tmp_path)
    per = _per(c)
    c.cohort(0, [0, 1], grow=True)
    t.open_round(0)
    for sh, ex in ((0, per), (1, 0)):
        c.blocks_for(sh, 0)
        f = _write_tiny_adapter(str(tmp_path / f"s{sh}.safetensors"))
        t.submit(tlr.Submission(shard=sh, round=0, path=f, examples=ex))
        c.block_done(sh, 0, examples=ex)

    assert c.close_round_if_ready() == 0
    done, _ = c.coverage()
    assert done == 1, "the one pod that worked is the one block covered"
