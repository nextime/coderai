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

"""A pod that finishes a round after it closed must not cost its examples.

Two separate claims, and only the first was ever true:

* its ADAPTER is refused. It started from the previous average while everyone
  else has moved on, so averaging it in moves the whole fleet backwards. That
  refusal is correct and stays.

* its EXAMPLES are not refused. They were trained, they were paid for, and
  nobody else has them.

The second was not handled at all. The submit branch requires the pod's round
to BE the open one, so a straggler fell through to "idle, give it the current
round" -- the finished adapter dropped with no log line, and under the work
queue its block consumed and its examples never trained by anyone. Under the
old fixed partition they came round again next epoch, which is why this did
not show up until work became a queue.
"""

import types

import pytest

from codai.api import loras
from codai.api.text_lora_rounds import RoundPlan, RoundTracker
from codai.api import text_lora_trainer as tlt


def _coord(n_examples=400, n_shards=4, rounds=4):
    plan = RoundPlan(n_shards=n_shards, seed=42)
    t = RoundTracker(plan, min_fraction=0.75, deadline_s=0)
    return loras._TextTrainCoordinator(
        "j", plan=plan, tracker=t, fleet=types.SimpleNamespace(pods={}),
        encoded=[0] * n_examples, out_dir="/tmp",
        settings=tlt.QLoraSettings(), dataset_id="", base="b",
        field_map=None, total_rounds=rounds), t


def test_a_stale_adapter_is_still_refused():
    """The claim that was already right."""
    import codai.api.text_lora_rounds as tlr
    c, t = _coord()
    c.cohort(0, [0, 1], grow=True)
    t.open_round(0)
    t.round = 1                      # round 0 has been averaged and closed
    with pytest.raises(tlr.RoundError) as exc:
        t.submit(tlr.Submission(shard=1, round=0, path="/x", examples=29))
    assert "stale" in str(exc.value)


def test_the_late_pods_block_goes_back_on_the_queue():
    c, t = _coord()
    c.cohort(0, [0, 1], grow=True)
    kept = c.assignment(0)["indices"]
    straggler = c.assignment(1)["indices"]
    assert set(kept).isdisjoint(straggler)

    # round 0 closes without shard 1
    back = c.recycle([1], rnd=0)
    assert back, "the block was not reclaimed"

    # whoever asks next gets exactly those examples, from the CURRENT average
    t.round = 1
    c.cohort(1, [0, 1, 2], grow=True)
    nxt = [c.assignment(sh)["indices"] for sh in (0, 1, 2)]
    assert straggler in nxt, "the late pod's examples were never re-issued"


def test_recycling_names_the_round_so_the_open_one_is_untouched():
    """It must reclaim the CLOSED round's block, not the pod's current one."""
    c, t = _coord()
    c.cohort(0, [0], grow=True)
    old = c.blocks_for(0, 0)
    t.round = 1
    c.cohort(1, [0], grow=True)
    now = c.blocks_for(0, 1)
    assert now != old
    assert c.recycle([0], rnd=0) == old
    assert c.blocks_for(0, 1) == now, "the open round's assignment moved"


def test_the_loop_has_a_branch_for_it_at_all():
    """The absence was the bug, so its presence is worth pinning.

    A `done` report for an EARLIER round used to match no branch and fall
    through to the "give it the current round" path at the bottom.
    """
    import inspect
    src = inspect.getsource(loras._text_train_distributed)
    assert 'state == "done" and 0 <= int(status.get("round", -1)) < rnd' in src
    i = src.index('state == "done" and 0 <=')
    block = src[i:i + 1800]
    assert "coord.recycle([shard], rnd=_late)" in block
    assert "a round behind" in block
    # and it must NOT fetch or submit that adapter
    assert "fetch_adapter" not in block
    assert "coord.submit" not in block
