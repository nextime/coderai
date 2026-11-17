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


def _coord(n=400, n_shards=4, rounds=4, epochs=1.0):
    plan = tlr.RoundPlan(n_shards=n_shards, seed=42)
    t = tlr.RoundTracker(plan, min_fraction=0.75, deadline_s=0)
    c = loras._TextTrainCoordinator(
        "j", plan=plan, tracker=t, fleet=types.SimpleNamespace(pods={}),
        encoded=[0] * n, out_dir="/tmp", settings=tlt.QLoraSettings(),
        dataset_id="", base="b", field_map=None, total_rounds=rounds,
        planned_pods=n_shards, epochs=epochs)
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
    assert c.blocks_for(1, 0) == c.blocks_for(1, 0)


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
    """Better an idle pod than the same examples twice in one average.

    `epochs=2`, deliberately: with one epoch, consuming every block in an
    exchange also COMPLETES the run, and then the empty handout proves the
    run ended rather than that the exchange was full. Two epochs separates
    the two conditions.
    """
    c, t = _coord(n=100, n_shards=4, rounds=4, epochs=2.0)
    c.cohort(0, [0], grow=True)
    n = 0
    while c.blocks_left(0):
        _finish(c, t, 0)
        n += 1
    assert n > 0
    assert c.finished() is False, "the run ended; this tests the exchange"
    assert c.assignment(0)["indices"] == []
    assert c.blocks_left(0) == 0


def test_a_finished_run_still_answers_with_an_empty_slice():
    """`round_payload` reads `indices` by subscript, so a run that finished
    between the loop's check and this call used to raise KeyError in the
    money path -- likelier now that coverage can complete at any moment
    rather than only at a round boundary."""
    c, t = _coord(n=58, n_shards=1, rounds=1)
    c.cohort(0, [0], grow=True)
    while not c.finished():
        _finish(c, t, 0)
    a = c.assignment(0)
    assert a["finished"] is True
    assert a["indices"] == []
    assert c.round_payload(0)["indices"] == []


def test_a_recycled_block_can_be_issued_again_in_the_same_exchange():
    """Its pod died before training it, so it is not 'already covered'."""
    c, t = _coord()
    c.cohort(0, [0, 1], grow=True)
    lost = c.assignment(1)["indices"]
    assert c.recycle([1], rnd=0)
    c.cohort(0, [0, 1, 2], grow=True)
    assert c.assignment(2)["indices"] == lost


def test_the_submission_file_is_named_per_block():
    """Two blocks from one pod in one exchange must not overwrite each other.

    This used to assert the exact source line, INCLUDING the
    `coord.blocks_for(...)` call that named the file through the handout --
    so the string test passed while every poll of an already-submitted pod
    consumed a fresh block and marked it covered untrained. The property is
    that the name follows the block, and that reading the name costs nothing.
    """
    import inspect
    src = inspect.getsource(loras._text_train_distributed)
    assert "b{_held[0]}" in src, \
        "the adapter is named after the block the pod actually holds"
    assert "blocks_for(shard, rnd)" not in src, \
        "naming a file must never go through the handout"

    # And the numbers it is named after really are distinct per block.
    c, _t = _coord()
    c.cohort(0, [0], grow=True)
    first = c.blocks_for(0, 0)
    c.block_done(0, 0, examples=29)
    second = c.blocks_for(0, 0)
    assert first and second and first != second, \
        "a pod taking another block gets a different one"


def test_the_loop_no_longer_skips_a_pod_that_submitted():
    import inspect
    src = inspect.getsource(loras._text_train_distributed)
    assert "any(s.shard == shard\n                                      for s in coord.tracker.have(rnd))" \
        not in src
    assert "A POD THAT ALREADY SUBMITTED IS NOT SKIPPED" in src


def test_every_pod_gets_the_same_size_unit_of_work():
    """The straggler bug, pinned.

    A handout used to be `ceil(planned / live)` blocks, and `live` was the
    cohort AT THAT MOMENT -- so the first pod into an exchange got sixteen
    blocks and the sixteenth got one. On the first exchange that ever closed,
    six pods were still training when the deadline fired and their adapters
    were refused: twenty minutes of six cards, discarded, and the average
    weighted by 297 examples of the ~2000 handed out.

    Equal units, and a fast pod simply takes more of them.
    """
    c, t = _coord(n=7308, n_shards=16, rounds=16)
    # a pod arriving when it is alone must not be handed the whole exchange
    c.cohort(0, [0], grow=True)
    alone = c.assignment(0)["indices"]
    _finish(c, t, 0)
    # ...and one arriving into a full cohort gets the same amount
    c.cohort(0, list(range(16)), grow=True)
    crowded = c.assignment(9)["indices"]
    assert len(alone) == len(crowded), (len(alone), len(crowded))
    # which is ONE block
    per = c.plan.block_size(len(c.encoded), c.total_rounds, c.planned_pods)
    assert len(alone) == per


def test_a_deadline_costs_at_most_one_block_per_pod():
    """The consequence that makes equal units the right call: whatever a pod
    is holding when an exchange closes is what gets discarded."""
    c, t = _coord(n=7308, n_shards=16, rounds=16)
    c.cohort(0, list(range(16)), grow=True)
    per = c.plan.block_size(len(c.encoded), c.total_rounds, c.planned_pods)
    for sh in range(16):
        held = c.assignment(sh)["indices"]
        assert len(held) == per, "a pod holds more than one block"


def test_the_averaging_line_separates_submissions_from_pods():
    """"averaged from 22 shard(s)" was 22 SUBMISSIONS across fewer pods, which
    reads as a fleet larger than the one that exists."""
    import inspect
    src = inspect.getsource(loras._TextTrainCoordinator.close_round_if_ready)
    assert "submission(s) across" in src
    assert "{_pods} pod(s)" in src
