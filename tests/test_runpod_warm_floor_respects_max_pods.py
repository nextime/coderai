"""max_pods counts pods that BILL, not pods that are healthy.

Observed 2026-10-18 on a pool configured max_pods=1, min_pods=1: a pod adopted
at 17:41, a restart at 17:43, a second pod rented at 17:46 -- $1.18/hr for a
configuration that asked for $0.59. The operator's explicit ceiling of one was
exceeded by the code path whose job is to hold exactly one warm.

Cause: the warm-floor top-up in maintain() computed

    in_hand = len([p for p in self.pods if p.healthy]) + self._booting

and tested BOTH the deficit and the max_pods cap against it. After a restart
the pool adopts the pod it had before; that pod is briefly not yet healthy and
was not booted by this process, so `_booting` is 0 and `in_hand` is 0 -- room
under a cap of one, beside a pod that is already running and already billing.

The two questions need two counts: the deficit is in pods that can SERVE, the
cap is in pods that are CHARGED FOR. `_should_grow` already counted
`len(self.pods) + self._booting` for its cap; this path disagreed.
"""
import pytest


class _Pod:
    def __init__(self, healthy, draining=False):
        self.healthy = healthy
        self.draining = draining


def want_to_rent(pods, booting, keep_min, max_pods, parallel=4):
    """Mirror of the corrected arithmetic in maintain()."""
    healthy_in_hand = len([p for p in pods if p.healthy]) + booting
    billing_in_hand = len(pods) + booting
    return max(0, min(keep_min - healthy_in_hand,
                      max(0, parallel - booting),
                      max(0, max_pods - billing_in_hand)))


# --- the exact regression ---

def test_an_adopted_unhealthy_pod_still_holds_its_max_pods_slot():
    """The 17:43 restart: one pod adopted, not yet healthy, nothing booting."""
    assert want_to_rent([_Pod(healthy=False)], booting=0,
                        keep_min=1, max_pods=1) == 0


def test_the_old_arithmetic_would_have_rented():
    """Guards the test itself: the bug really was reachable this way."""
    pods, booting, keep_min, max_pods = [_Pod(healthy=False)], 0, 1, 1
    old_in_hand = len([p for p in pods if p.healthy]) + booting
    old_want = min(keep_min - old_in_hand, 4 - booting, max_pods - old_in_hand)
    assert old_want == 1, "the regression must be reproducible, or this test proves nothing"
    assert want_to_rent(pods, booting, keep_min, max_pods) == 0


# --- the cap holds whatever state the existing pods are in ---

@pytest.mark.parametrize("healthy", [True, False])
def test_a_full_pool_never_rents(healthy):
    assert want_to_rent([_Pod(healthy=healthy)], 0, 1, 1) == 0


def test_a_sick_pod_does_not_free_a_slot_for_a_second_card():
    """A pod that is sick is still charged for by the minute."""
    assert want_to_rent([_Pod(healthy=True), _Pod(healthy=False)], 0,
                        keep_min=2, max_pods=2) == 0


def test_a_draining_pod_still_bills_and_holds_its_slot():
    assert want_to_rent([_Pod(healthy=True, draining=True)], 0, 1, 1) == 0


# --- and the floor is still filled when there IS room ---

def test_an_empty_pool_rents_its_floor():
    assert want_to_rent([], 0, keep_min=1, max_pods=1) == 1


def test_a_deficit_under_the_cap_is_filled():
    assert want_to_rent([_Pod(healthy=True)], 0, keep_min=4, max_pods=8) == 3


def test_a_boot_already_on_the_way_is_not_duplicated():
    assert want_to_rent([], booting=1, keep_min=1, max_pods=1) == 0


def test_the_parallel_boot_cap_still_applies():
    assert want_to_rent([], 0, keep_min=8, max_pods=8, parallel=2) == 2


def test_the_cap_binds_before_the_deficit():
    """A floor larger than max_pods cannot rent past max_pods."""
    assert want_to_rent([], 0, keep_min=10, max_pods=3) == 3


def test_an_unhealthy_pod_does_not_satisfy_the_floor_when_there_is_room():
    """The deficit still counts health: a sick pod is not a warm pod. With a
    cap of 2 there is room to rent the healthy one the floor asked for."""
    assert want_to_rent([_Pod(healthy=False)], 0, keep_min=1, max_pods=2) == 1


# --- the arithmetic above must be the arithmetic in the worker ---

def test_worker_uses_two_distinct_counts():
    src = open("codai/api/runpod_worker.py").read()
    assert "billing_in_hand = len(self.pods) + self._booting" in src
    assert "healthy_in_hand = len([p for p in self.pods if p.healthy]) + self._booting" in src
    assert "max(0, self.effective_max_pods() - billing_in_hand)" in src, \
        "the max_pods cap must be tested against pods that bill"
    assert "want = min(keep_min - healthy_in_hand," in src, \
        "the deficit must still be tested against pods that can serve"


def test_grow_and_warm_floor_agree_on_what_the_cap_counts():
    """They disagreed, and that disagreement was the bug."""
    src = open("codai/api/runpod_worker.py").read()
    assert "in_hand = len(self.pods) + self._booting" in src, \
        "_should_grow's cap count"
    assert "len([p for p in self.pods if p.healthy]) + self._booting)" not in src.replace(
        "healthy_in_hand = len([p for p in self.pods if p.healthy]) + self._booting", ""), \
        "no remaining health-filtered count feeding a max_pods test"
