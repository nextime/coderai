"""Losing pods must cost rounds, not the run.

Reported from the digesta seat: eight pods assembled, round 0 training, then
one shard stopped checking in, its replacement could not be rented, and the
job ended -- eight cards paid for and not one round brought home.

Two separate things, and only one of them was real by the time it was
reported:

* a SINGLE lost shard never ended the run. `reap_the_dead` releases it and the
  loop continues; since per-round cohorts (v0.6.62) the next round simply runs
  on the survivors. The reported failure predates that build.
* losing the WHOLE fleet did throw away completed rounds. Every round is a
  checkpoint, so a fleet lost at round 9 of 16 is nine rounds of averaging
  already paid for, and raising handed back nothing.

And the 422 underneath it was reported as "unavailable" -- a card shortage --
for every candidate, so a rejected request body was indistinguishable from an
empty market.
"""

import inspect

import pytest

from codai.api import loras
from codai.api import runpod_client as rc
from codai.api import runpod_worker as rw


DIST = inspect.getsource(loras._text_train_distributed)


# ------------------------------------------- one lost shard is survivable ---

def test_only_an_empty_fleet_ends_the_run():
    """Not one lost shard, and not a failed replacement."""
    i = DIST.index("gone = coord.reap_the_dead()")
    block = DIST[i:i + 600]
    assert "if not fleet.pods:" in block, "the guard is on the whole fleet"


def test_a_released_shard_is_announced_not_fatal():
    i = DIST.index("gone = coord.reap_the_dead()")
    block = DIST[i:i + 400]
    assert "released silent shard(s)" in block
    assert "raise" not in block.split("if not fleet.pods:")[0]


def test_a_lost_shard_costs_its_share_of_one_round_at_most():
    """And now not even that: its examples go back on the queue.

    This used to assert a sentence of `reap_the_dead`'s docstring. The claim
    is unchanged -- losing a pod costs its share of one exchange, not the run
    -- but it got STRONGER when work became a queue, and the way to show that
    is to run it: under a partition a lost slice came round again next epoch,
    and under a queue it would simply never be issued unless the block is
    recycled.
    """
    import types
    from codai.api.text_lora_rounds import RoundPlan, RoundTracker
    from codai.api import text_lora_trainer as tlt

    plan = RoundPlan(n_shards=4, seed=42)
    t = RoundTracker(plan, min_fraction=0.75, deadline_s=0)
    reaped = []

    class _Fleet:
        pods = {}

        def to_reap(self):
            return [types.SimpleNamespace(shard=1)]

        def release(self, shard, reason=""):
            reaped.append((shard, reason))
            return True

    c = loras._TextTrainCoordinator(
        "j", plan=plan, tracker=t, fleet=_Fleet(), encoded=[0] * 400,
        out_dir="/tmp", settings=tlt.QLoraSettings(), dataset_id="",
        base="b", field_map=None, total_rounds=4)
    c.cohort(0, [0, 1], grow=True)
    kept = c.assignment(0)["indices"]
    lost = c.assignment(1)["indices"]
    assert lost and set(lost).isdisjoint(kept)

    gone = c.reap_the_dead()
    assert gone == [1]
    assert reaped and "stopped checking in" in reaped[0][1]

    # The run continues on shard 0, and the lost examples are handed to the
    # next pod that asks rather than dropped.
    c.cohort(0, [0, 2], grow=True)
    assert c.assignment(2)["indices"] == lost
    assert c.assignment(0)["indices"] == kept, "a survivor was re-tasked"


# --------------------------------- a lost fleet keeps what it has earned ----

def test_a_lost_fleet_finishes_with_the_last_average():
    i = DIST.index("if not fleet.pods:")
    block = DIST[i:i + 1400]
    assert "coord.average_path(_done - 1)" in block
    assert "rather than discarding" in block
    assert "break" in block


def test_the_record_says_which_round_it_kept():
    i = DIST.index("if not fleet.pods:")
    block = DIST[i:i + 1400]
    assert "kept the " in block and "round " in block


def test_the_final_adapter_is_the_last_average_actually_written():
    """REPLACED. `rounds = _done` corrected a count that no longer decides.

    The run used to read its result from `average_path(rounds - 1)`, so a
    fleet lost at round 9 of 16 had to rewrite `rounds` or the lookup would
    miss. Now that a run ends on dataset COVERAGE, the number of exchanges is
    emergent -- more than `rounds` on a small fleet, fewer on a large one --
    so indexing by any planned count is wrong in both directions. It searches
    back from the last round for the newest average that exists.
    """
    i = DIST.index("final = \"\"")
    block = DIST[i:i + 700]
    assert "for _r in range(coord.tracker.round, -1, -1)" in block
    assert "os.path.isfile(_p)" in block
    assert "no round produced an average" in block


def test_a_fleet_lost_before_any_round_still_fails():
    """A target is not a licence to hand back an untrained adapter."""
    i = DIST.index("if not fleet.pods:")
    block = DIST[i:i + 1400]
    assert "nothing to keep" in block
    assert "raise RuntimeError" in block


# --------------------------------- a 422 is not a card shortage -------------

def test_the_whole_422_body_reaches_the_log():
    """300 characters cut the `errors` array off exactly where the field name
    is, and the reason was unrecoverable once the log rotated."""
    src = inspect.getsource(rc.RunpodClient._rest)
    assert "REJECTED THE BODY" in src
    assert "body[:2000]" in src


def test_a_422_is_reported_as_validation_not_capacity():
    src = inspect.getsource(rw)
    assert "the REQUEST BODY was " in src
    assert "not a capacity problem" in src


def test_a_422_says_the_next_card_will_not_help():
    """Four rentals walked the candidate list on a body the API rejects, which
    reads from outside as a market with nothing in it."""
    src = inspect.getsource(rw)
    assert "will not help until the " in src


def test_a_real_capacity_miss_is_still_reported_as_one():
    src = inspect.getsource(rw)
    assert "no longer any instances" in src
    assert "unavailable — " in src
