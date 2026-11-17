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

"""A pod that trains a fraction of its handout has not covered the block.

`block_done` asked `examples > 0`, so a pod that trained 4 of the 29 examples
it was handed closed the block as if it had trained all 29. Two consecutive
16-pod runs therefore covered every block they were given and reported
success: the second reached `blocks_done 139/252, examples_done 4031` -- and
4031 is exactly 139 * 29, because `examples_done` was PUBLISHED as
`blocks_done * block_size`. The one number a client could have checked was
computed from the thing that was wrong, so it could not disagree.

Three things are tested here, and the third is the one that keeps the fix from
being worse than the bug: recycling a short block is right for a pod having a
bad time and ruinous as a steady state, because the same examples would be
handed out, returned short and recycled for as long as the run was allowed to
burn money. A fleet that cannot train what it is given must FAIL.
"""

import types

import pytest

from codai.api import loras
from codai.api import text_lora_rounds as tlr


def _coord(tmp_path, *, n=290, pods=10, rounds=1):
    """A coordinator over `n` fake examples. block_size = n/(rounds*pods).

    Built the way the run builds it, so the plan doing the arithmetic is the
    real one: the asked-count is taken from it and a fake would decide the
    answer to the test.
    """
    plan = tlr.RoundPlan(n_shards=pods, seed=42)
    tracker = tlr.RoundTracker(plan, min_fraction=0.75, deadline_s=600.0)
    return loras._TextTrainCoordinator(
        "job-t", plan=plan, tracker=tracker, fleet=None,
        out_dir=str(tmp_path), base="b",
        encoded=[object()] * n, settings=types.SimpleNamespace(
            alpha=32, dropout=0.0, rank=16),
        total_rounds=rounds, planned_pods=pods, epochs=1.0,
        field_map=None)


# --------------------------------------------------------------------- #
# what a block costs, and what closing one requires
# --------------------------------------------------------------------- #

def test_the_asked_count_comes_from_the_plan(tmp_path):
    """Not from the pod: the orchestrator chose the slice."""
    c = _coord(tmp_path)
    assert c.examples_in([0]) == 29
    assert c.examples_in([0, 1]) == 58
    assert c.examples_in([]) == 0


def test_a_block_trained_in_full_is_covered(tmp_path):
    c = _coord(tmp_path)
    c.blocks_for(0, 0)
    c.block_done(0, 0, examples=29, asked=29)
    assert c.coverage()[0] == 1


def test_a_block_mostly_trained_is_covered(tmp_path):
    """The OOM guard legitimately skips an example here and there."""
    c = _coord(tmp_path)
    c.blocks_for(0, 0)
    c.block_done(0, 0, examples=27, asked=29)
    assert c.coverage()[0] == 1, "27 of 29 is a round that did its work"


def test_the_four_of_twentynine_block_is_not_covered(tmp_path):
    """The measured case. It used to count, and that is the whole defect."""
    c = _coord(tmp_path)
    held = c.blocks_for(0, 0)
    c.block_done(0, 0, examples=4, asked=29)
    assert c.coverage()[0] == 0, "4 of 29 is not a trained block"
    # ...and it is available again, to a pod that might do better.
    assert c.blocks_for(1, 0) == held, "the block goes back on the queue"


def test_a_block_trained_not_at_all_is_still_not_covered(tmp_path):
    c = _coord(tmp_path)
    c.blocks_for(0, 0)
    c.block_done(0, 0, examples=0, asked=29)
    assert c.coverage()[0] == 0


def test_the_threshold_is_a_fraction_of_what_was_asked(tmp_path):
    """So a short last block is judged against its own size, not the full one."""
    c = _coord(tmp_path, n=5, pods=1, rounds=1)
    # One block of 5. Four of five is above the fraction; one is not.
    c.blocks_for(0, 0)
    c.block_done(0, 0, examples=4, asked=5)
    assert c.coverage()[0] == 1


def test_an_unknown_count_is_trusted(tmp_path):
    """`examples=None` means the caller cannot say, and keeps the old contract."""
    c = _coord(tmp_path)
    c.blocks_for(0, 0)
    c.block_done(0, 0)
    assert c.coverage()[0] == 1


def test_asked_defaults_to_the_plan_rather_than_skipping_the_check(tmp_path):
    """A caller that forgets `asked` must not silently disable the threshold."""
    c = _coord(tmp_path)
    c.blocks_for(0, 0)
    c.block_done(0, 0, examples=4)
    assert c.coverage()[0] == 0, "the plan knows the block was 29 examples"


# --------------------------------------------------------------------- #
# the published numbers are measured
# --------------------------------------------------------------------- #

def test_examples_trained_is_counted_not_derived(tmp_path):
    c = _coord(tmp_path)
    for shard, got in ((0, 29), (1, 27), (2, 4)):
        c.blocks_for(shard, 0)
        c.block_done(shard, 0, examples=got, asked=29)
    assert c.examples_trained() == 60, "29 + 27 + 4, as reported by the pods"
    assert c.examples_asked() == 87, "three blocks of 29 were handed out"
    # And the derived number it replaces would have said two blocks * 29 = 58,
    # agreeing with itself and hiding the shard that trained 4.
    assert c.coverage()[0] == 2


def test_a_short_submission_still_counts_its_work(tmp_path):
    """The gradient steps happened; they are averaged in and must be reported."""
    c = _coord(tmp_path)
    c.blocks_for(0, 0)
    c.block_done(0, 0, examples=4, asked=29)
    assert c.examples_trained() == 4
    assert c.examples_asked() == 29


# --------------------------------------------------------------------- #
# and the guard against recycling forever
# --------------------------------------------------------------------- #

def test_a_healthy_fleet_is_not_starved(tmp_path):
    c = _coord(tmp_path, n=2900, pods=100)
    for shard in range(20):
        c.blocks_for(shard, 0)
        c.block_done(shard, 0, examples=29, asked=29)
    assert c.starved() is None


def test_one_bad_pod_does_not_abort_a_working_fleet(tmp_path):
    c = _coord(tmp_path, n=2900, pods=100)
    for shard in range(20):
        c.blocks_for(shard, 0)
        c.block_done(shard, 0, examples=(4 if shard == 3 else 29), asked=29)
    assert c.starved() is None, "19 of 20 rounds did their work"


def test_a_fleet_that_cannot_train_its_handouts_is_starved(tmp_path):
    """Otherwise the blocks recycle forever at the fleet's hourly rate."""
    c = _coord(tmp_path, n=2900, pods=100)
    for shard in range(20):
        c.blocks_for(shard, 0)
        c.block_done(shard, 0, examples=4, asked=29)
    why = c.starved()
    assert why, "20 short submissions in a row is not a slow run"
    assert "20 of 20" in why, why
    assert "80 trained of 580 asked" in why, why


def test_the_first_few_short_rounds_are_not_enough_to_abort(tmp_path):
    """A fleet mid-boot can return a couple of odd rounds; that is not proof."""
    c = _coord(tmp_path, n=2900, pods=100)
    for shard in range(3):
        c.blocks_for(shard, 0)
        c.block_done(shard, 0, examples=0, asked=29)
    assert c.starved() is None, "three submissions is too small a sample"


def test_the_starvation_message_carries_the_numbers(tmp_path):
    """It ends a run, so it has to say why without anyone reading the code."""
    c = _coord(tmp_path, n=2900, pods=100)
    for shard in range(10):
        c.blocks_for(shard, 0)
        c.block_done(shard, 0, examples=4, asked=29)
    why = c.starved() or ""
    assert "75%" in why, why
    assert "14%" in why, why


# --------------------------------------------------------------------- #
# the wiring: the run must act on it
# --------------------------------------------------------------------- #

def test_the_run_publishes_the_measured_count(tmp_path):
    import inspect
    src = inspect.getsource(loras._text_train_distributed)
    assert "examples_done=coord.examples_trained()" in src, (
        "`examples_done=_done_b * _per` restated blocks_done, so the two "
        "numbers agreed by construction and hid the undertraining")
    assert "examples_done=_done_b * _per" not in src, \
        "the derived form must be gone, not merely joined by the measured one"


def test_the_run_stops_a_starving_fleet(tmp_path):
    import inspect
    src = inspect.getsource(loras._text_train_distributed)
    assert "coord.starved()" in src, (
        "recycling short blocks without this spins and bills forever")
