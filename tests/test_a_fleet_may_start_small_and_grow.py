"""Waiting for the whole fleet before starting made a big job impossible.

The pods that arrive first bill while the rest are hunted, and the larger the
fleet the less likely it ever assembles -- so `min_pods: 8` meant a job that
never started, while `min_pods: 1` with `pods: 8` meant a job that trained one
eighth of the data and said it had finished.

Both come from the same assumption: that the partition is fixed at the start.
It is not any more. Each round has a COHORT -- the pods with an address when
it opened -- and the round's slice is split among exactly those. The fleet
grows toward the target between rounds, where it is safe, because the open
round's cohort is already decided.
"""

import inspect

import pytest

from codai.api import loras
from codai.api import text_lora_fleet as flt
from codai.api.text_lora_rounds import RoundPlan, RoundTracker, Submission


DIST = inspect.getsource(loras._text_train_distributed)
COORD = inspect.getsource(loras._TextTrainCoordinator)


# ------------------------------------- coverage survives a changing fleet ---

@pytest.mark.parametrize("sizes", [
    [1, 3, 2, 8],        # grows, shrinks, grows
    [1, 1, 1, 1],        # never grows
    [8, 8, 8, 8],        # full from the start
    [4, 1, 7, 2],        # thrashing
])
def test_every_example_is_trained_exactly_once(sizes):
    """The property that makes scaling safe rather than merely possible. With
    the old per-shard-then-per-round slicing this failed for every sequence
    that was not constant."""
    n, rounds = 100, len(sizes)
    plan = RoundPlan(n_shards=1, seed=42)
    seen = []
    for rnd, k in enumerate(sizes):
        for pos in range(k):
            seen += plan.indices_for(pos, n, round=rnd, rounds=rounds,
                                     n_shards=k)
    assert sorted(seen) == list(range(n)), (
        f"fleet sizes {sizes} did not cover the dataset exactly once")


def test_a_rounds_slice_does_not_depend_on_the_fleet_size():
    """Round r is the same 1/rounds of the data however many pods run it --
    that is what makes the sizes above interchangeable."""
    plan = RoundPlan(n_shards=1, seed=42)
    one = set(plan.indices_for(0, 60, round=2, rounds=4, n_shards=1))
    many = set()
    for pos in range(5):
        many |= set(plan.indices_for(pos, 60, round=2, rounds=4, n_shards=5))
    assert one == many


def test_a_pods_slice_is_a_sample_not_a_contiguous_block():
    """Contiguous blocks of a shuffle are still correlated within a round; a
    round-robin stride is not."""
    body = inspect.getsource(RoundPlan.indices_for)
    assert "j % k == pos" in body


# --------------------------------------------- the cohort, per round --------

def test_the_cohort_is_fixed_once_a_round_opens():
    """A pod joining mid-round would change the divisor and make every list
    already handed out wrong."""
    assert "def cohort" in COORD
    i = COORD.index("def cohort")
    assert "have is not None" in COORD[i:i + 900]


def test_a_pod_is_placed_by_its_position_in_the_cohort():
    """With a growing fleet, shard 7 can be position 2 -- the fleet shard id
    is not the slice selector."""
    assert "def position_of" in COORD
    assert "self.position_of(int(shard), rnd)" in COORD


def test_the_slice_count_comes_from_the_cohort_not_the_plan():
    assert "n_shards=max(1, len(self.cohort(rnd))" in COORD


def test_a_round_is_complete_against_its_own_cohort():
    """Judging two pods against a planned eight means the round can never be
    complete and waits out the deadline every time."""
    plan = RoundPlan(n_shards=8, seed=42)
    t = RoundTracker(plan, min_fraction=0.75, deadline_s=600)
    t.open_round(0)
    t.expect(0, 2)
    for sh in (0, 1):
        t.submit(Submission(shard=sh, round=0, path="/x", examples=10))
    assert t.ready() is True


def test_without_a_cohort_it_falls_back_to_the_planned_shape():
    plan = RoundPlan(n_shards=8, seed=42)
    t = RoundTracker(plan, min_fraction=0.75, deadline_s=600)
    t.open_round(0)
    t.submit(Submission(shard=0, round=0, path="/x", examples=10))
    assert t.ready() is False


# ------------------------------------------------------------ the growth ----

def test_provisioning_tries_every_shard():
    """Capacity is per card: shard 3 failing on an A40 says nothing about
    shard 4 getting an A6000. It used to stop at the first refusal."""
    src = inspect.getsource(flt.TrainerFleet.provision)
    assert "NOT a break" in src
    assert "continue" in src


def test_provisioning_reports_a_partial_fleet():
    src = inspect.getsource(flt.TrainerFleet.provision)
    assert "retried between" in src


def test_top_up_rents_only_what_is_missing(monkeypatch):
    # MONKEYPATCH, not a bare assignment: a module-level assignment here
    # leaked a no-op register_pod into every test that ran afterwards and
    # broke an unrelated file -- order-dependently, which is the worst kind.
    import codai.api.runpod_worker as rw
    monkeypatch.setattr(rw, "register_pod", lambda *a, **k: None)
    spec = flt.TrainerPodSpec(image="i", gpu_types=["A40"], hourly_usd=0.5)

    class Client:
        def __init__(self):
            self.made = 0

        def create_pod(self, **kw):
            self.made += 1
            return f"pod-{self.made}"

    c = Client()
    f = flt.TrainerFleet(c, "job-g", spec)
    f._flush = lambda: None
    f.provision([0, 1])
    assert c.made == 2
    f.top_up(4)
    assert c.made == 4, "it re-rented pods it already had"
    assert sorted(f.pods) == [0, 1, 2, 3]


def test_top_up_respects_the_operator_cap(monkeypatch):
    import codai.api.runpod_worker as rw
    monkeypatch.setattr(rw, "register_pod", lambda *a, **k: None)
    spec = flt.TrainerPodSpec(image="i", gpu_types=["A40"], hourly_usd=0.5)

    class Client:
        def create_pod(self, **kw):
            return "pod-x"

    f = flt.TrainerFleet(Client(), "job-c", spec)
    f._flush = lambda: None
    f.top_up(16, cap=3)
    assert len(f.pods) <= 3


def test_growth_happens_at_a_round_boundary_only():
    """The open round's cohort is already fixed, so a pod arriving now joins
    the next one -- growing mid-round would invalidate handed-out slices."""
    assert "fleet.top_up(n_shards" in DIST
    assert "_topped_up[0] = rnd" in DIST


def test_growth_is_bounded_by_the_configured_ceiling():
    assert "cap=int(cfg.max_pods or 0)" in DIST


def test_a_failed_growth_does_not_end_the_run():
    i = DIST.index("fleet.top_up(n_shards")
    block = DIST[max(0, i - 400):i + 900]
    assert "could not grow the fleet" in block
    assert "continuing with" in block


def test_the_cohort_size_is_reported_when_it_changes():
    assert "runs on " in DIST and "_last_cohort" in DIST
