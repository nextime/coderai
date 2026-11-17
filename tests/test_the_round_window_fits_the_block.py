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

"""A round must wait long enough for the block it handed out.

MEASURED, on a 16-pod job training 7308 examples in blocks of 29:

    card                 per block
    RTX 6000 Ada         13.2 min
    RTX A6000            16.8 min
    L40                  17.3 min
    A40                  18.7 - 22.0 min

against a deadline of `max(60, poll_interval_s * 60)` = **20.0 min**, which
is the MEDIAN of that distribution. So the deadline was not a backstop for a
dead pod -- it was how rounds closed, and it cut the fleet in half every
time. One exchange averaged 3 submissions from 13 pods; the other ten
finished minutes later, were refused as a round behind, and their blocks went
back on the queue after the GPU had already been paid for. The run reported
5.2% of one epoch after 63 minutes at $10.25/hr.

The fix is to stop guessing: time the blocks, size the deadline off them, and
keep the configured value only as a floor and as round 0's bootstrap (round 0
pays the base load inside its first block).

These tests drive the tracker rather than reading it. The defect they exist
for was invisible in the source -- every line was doing what it said.
"""

import pytest

from codai.api import text_lora_rounds as tlr


FLOOR = 1200.0          # what poll_interval_s=20 produces today


def tracker(*, floor=FLOOR, shards=13, min_fraction=0.75):
    t = tlr.RoundTracker(tlr.RoundPlan(n_shards=shards),
                         min_fraction=min_fraction, deadline_s=floor)
    t.expect(0, shards)
    return t


def feed(t, durations, *, rnd=0, shard_base=0):
    """Time `durations` (seconds) worth of blocks into the tracker."""
    clock = 1000.0
    for i, d in enumerate(durations):
        sh = shard_base + i
        t.note_started(sh, rnd, now=clock)
        t._note_block_done(sh, rnd, now=clock + d)
    return t


# ------------------------------------------------- before anything is timed --

def test_an_unmeasured_round_gets_the_configured_floor():
    """Every round alike. Round 0 was briefly given DOUBLE the floor, to pay
    for the base load inside its first block -- but that doubles the dead-pod
    timeout to fix what is really a clock problem, and
    `test_the_clock_starts_when_a_pod_is_given_work` is the fix instead."""
    t = tracker()
    assert t.deadline_for(0) == pytest.approx(FLOOR)
    assert t.deadline_for(3) == pytest.approx(FLOOR)


def test_the_clock_starts_when_a_pod_is_given_work():
    """Round 0 opens before a pod has booted, let alone quantized 17.6GB to
    4-bit. Measured on one job, the pods were dispatched 6 to 13 minutes
    after the round opened; judged from the open, round 0 spends a third of
    its window on a cost that buys no training."""
    t = tracker(shards=2, min_fraction=0.99)
    t.open_round(0, now=0.0)
    t.note_started(0, 0, now=600.0)         # booted and loaded at last
    t.submit(tlr.Submission(shard=0, round=0, path="p", examples=29, at=700.0))

    assert not t.ready(now=FLOOR + 10), \
        "the 600s of boot must not be spent out of the straggler's window"
    assert t.ready(now=600.0 + FLOOR + 1), \
        "and the window still ends, one block after the work began"


def test_a_round_that_dispatched_nothing_falls_back_to_the_open():
    """An adopted fleet after a restart: this process saw no dispatch, so
    the open is the only stamp there is and the round must still close."""
    t = tracker(shards=4, min_fraction=0.99)
    t.open_round(0, now=1000.0)
    t.submit(tlr.Submission(shard=0, round=0, path="p", examples=29,
                            at=1010.0))
    assert not t.ready(now=1100.0)
    assert t.ready(now=1000.0 + FLOOR + 1)


def test_a_disabled_deadline_stays_disabled():
    """deadline_s=0 means "wait for the cohort", and sizing must not
    conjure a timer where the operator asked for none."""
    t = tracker(floor=0.0)
    feed(t, [1200.0] * 5)
    assert t.deadline_for() == 0.0


# ------------------------------------------------------- sized off the data --

def test_the_deadline_follows_the_measured_blocks():
    """The measured distribution, in seconds. p90 of it is 22.0 min, so the
    deadline lands above every block in the sample -- which is the entire
    point."""
    measured = [13.2, 16.8, 17.3, 18.7, 19.1, 19.4, 20.5, 20.9, 21.6, 22.0]
    t = tracker()
    feed(t, [m * 60 for m in measured])

    dl = t.deadline_for()
    assert dl > max(measured) * 60, (
        "a deadline inside the measured spread recycles the slow half of "
        "the fleet every round")
    assert dl == pytest.approx(tlr.DEADLINE_SLACK * 22.0 * 60, rel=0.01)


def test_the_floor_still_wins_when_blocks_are_fast():
    """Fast blocks must not shrink the window below what the operator set:
    the next card rented may be slower than every one measured."""
    t = tracker()
    feed(t, [60.0] * 6)
    assert t.deadline_for() == pytest.approx(FLOOR)


def test_an_absurdly_slow_fleet_is_capped():
    """Waiting is not free -- the cards bill throughout."""
    t = tracker()
    feed(t, [9 * 3600.0] * 5)
    assert t.deadline_for() == pytest.approx(tlr.DEADLINE_CEILING_S)


def test_one_measurement_is_not_enough_to_size_it():
    t = tracker()
    feed(t, [40 * 60.0])
    assert t.deadline_for(1) == pytest.approx(FLOOR), \
        "one slow block is noise, not a distribution"


def test_enough_measurements_takes_over_from_the_floor():
    t = tracker()
    slow = [40 * 60.0] * tlr.DEADLINE_MIN_SAMPLES
    feed(t, slow)
    assert t.deadline_for(1) > FLOOR


# -------------------------------------------------------- what gets measured --

def test_a_submission_times_the_block_it_ends(tmp_path):
    t = tracker()
    t.note_started(0, 0, now=500.0)
    t.submit(tlr.Submission(shard=0, round=0, path="p", examples=29,
                            at=1700.0))
    assert len(t.block_seconds()) == 1
    assert t.block_seconds()[0] > 0


def test_a_second_block_in_one_round_is_timed_on_its_own():
    """A pod takes another block the instant it hands one in, so the clock
    restarts rather than being cleared. Timed from the first start, the
    second block would read as both of them -- which is exactly how round 0
    appeared to take 43 minutes a block."""
    t = tracker()
    t.note_started(0, 0, now=0.0)
    t._note_block_done(0, 0, now=1000.0)
    t._note_block_done(0, 0, now=1800.0)
    assert t.block_seconds() == pytest.approx([800.0, 1000.0])


def test_a_block_nobody_started_is_not_timed():
    """A submission from a pod this process never dispatched -- adopted
    after a restart -- must not be measured from a start it never saw."""
    t = tracker()
    t._note_block_done(7, 0, now=5000.0)
    assert t.block_seconds() == []


def test_the_dispatch_is_what_starts_the_clock_not_the_boot():
    """Boot and base load are real but are paid once, and sizing every
    round against them would make the whole job wait out a cost it only
    has at the start."""
    t = tracker()
    t.note_started(0, 0, now=2000.0)        # dispatched, after booting
    t._note_block_done(0, 0, now=3200.0)
    assert t.block_seconds() == pytest.approx([1200.0])


# ------------------------------------------------- the behaviour that broke --

def test_the_measured_fleet_now_lands_inside_its_round():
    """THE REGRESSION. Thirteen pods, the measured block times, dispatched
    within a poll slice of the round opening. Every one of them must be
    inside the window; under the old 20.0 min deadline, ten were not."""
    blocks = [13.2, 16.8, 17.3, 17.4, 18.7, 19.0, 19.1, 19.4,
              20.5, 20.9, 21.2, 21.6, 22.0]
    t = tracker(shards=13)
    feed(t, [b * 60 for b in blocks], rnd=0)          # round 0 taught it

    t.close_round()
    t.expect(1, 13)
    t._opened_at[t.round] = 0.0
    dl = t.deadline_for(1)

    late = [b for b in blocks if b * 60 > dl]
    assert late == [], f"{len(late)} pod(s) still finish after the round: {late}"
    assert dl / 60 < 40, "and not so late that a dead pod stalls the job"


def test_a_healthy_round_still_closes_on_the_quorum_not_the_clock():
    """The longer deadline must not make a good round slower: three
    quarters arriving is still what closes it."""
    t = tracker(shards=4)
    t.open_round(0, now=0.0)
    for sh in (0, 1, 2):
        t.submit(tlr.Submission(shard=sh, round=0, path="p", examples=29,
                                at=10.0))
    assert t.ready(now=10.0), \
        "3 of 4 is the quorum; it must not wait out the deadline"


def test_a_dead_pod_still_cannot_stall_the_round():
    """The deadline grew; it did not stop existing."""
    t = tracker(shards=13)
    feed(t, [20 * 60.0] * 5)
    t.close_round()
    t.expect(1, 13)
    t._opened_at[t.round] = 0.0     # close_round stamped it with the real clock
    t.submit(tlr.Submission(shard=0, round=1, path="p", examples=29, at=1.0))

    dl = t.deadline_for(1)
    assert not t.ready(now=dl - 1), "it waits for the stragglers first"
    assert t.ready(now=dl + 1), "but it does not wait forever"


def test_a_round_nobody_answered_never_closes_on_the_clock():
    """Averaging zero adapters is not a round, however long it has been."""
    t = tracker(shards=13)
    feed(t, [20 * 60.0] * 5)
    t.close_round()
    t.expect(1, 13)
    t._opened_at[t.round] = 0.0
    assert not t.ready(now=tlr.DEADLINE_CEILING_S * 10)
