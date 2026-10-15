"""The growth line must name the trigger that actually fired.

Observed 2026-10-15. A pool with one pod rented a second one second after the
first went ready, and announced:

    'datalab-to/surya-ocr-2': growing to meet 1 pod(s) of demand — 1 up, 1 booting

Read plainly that says: it needs one pod, it has one pod, and it is renting
another. It looks exactly like a double-rent bug, and it cost a real
investigation to establish that the fleet was correct — the DEMAND trigger had
not fired at all (one pod is exactly right for 16 in flight at 16 per pod); the
SATURATION trigger had, which rents ahead of the next burst. The caller printed
the demand figure whichever trigger fired.

Two triggers that mean different things must not share one sentence.
"""

import types

import pytest

from codai.api import runpod_worker as rw


def _pool(max_pods=8, threshold=16, parallel=1):
    from codai.api.runpod_worker import RunpodPodPool, parse_model_runpod

    class _Acct:
        enabled = True

    return RunpodPodPool("m", _Acct(), parse_model_runpod({
        "max_pods": max_pods,
        "scale_up_inflight_per_pod": threshold,
        "scale_up_max_parallel_boots": parallel,
    }), "m")


def _pod(inflight=0, healthy=True, draining=False):
    return types.SimpleNamespace(inflight=inflight, healthy=healthy,
                                 draining=draining, pod_id="p", url="u",
                                 last_used=0.0)


# ------------------------------------------------------------- saturation #
def test_saturation_does_not_claim_to_be_meeting_demand():
    """The exact live line. Demand is MET; this is anticipatory."""
    p = _pool(threshold=16)
    busy = _pod(inflight=16)
    p.pods = [busy]
    why = p._grow_reason([busy])
    assert why
    assert "demand" not in why.lower(), why
    assert "16-in-flight threshold" in why
    assert "next burst" in why


def test_saturation_says_it_is_not_for_queued_work():
    """The distinction that was missing: nothing is waiting for this pod."""
    p = _pool(threshold=16)
    busy = _pod(inflight=16)
    p.pods = [busy]
    assert "not for work already queued" in p._grow_reason([busy])


# ----------------------------------------------------------------- demand #
def test_demand_reports_the_load_and_the_shortfall():
    p = _pool(threshold=16)
    busy = [_pod(inflight=16), _pod(inflight=16)]
    p.pods = list(busy)
    p._waiting = 20
    why = p._grow_reason(busy)
    assert "52 page(s) in flight or waiting" in why   # 16 + 16 + 20
    assert "needs 4 pod(s)" in why
    assert "2 are in hand" in why


def test_the_first_pod_says_nothing_can_serve_yet():
    p = _pool()
    assert "nothing can serve a request yet" in p._grow_reason([])


# -------------------------------------------------------------- refusals #
def test_no_reason_is_an_empty_string_not_a_sentence():
    """`_should_grow` is `bool(reason)`, so a chatty 'no' would rent a pod."""
    p = _pool(threshold=16)
    quiet = _pod(inflight=1)
    p.pods = [quiet]
    assert p._grow_reason([quiet]) == ""


@pytest.mark.parametrize("setup,label", [
    (lambda p: setattr(p.mcfg, "max_pods", 1), "at max_pods"),
    (lambda p: setattr(p, "_booting", 1), "already booting"),
])
def test_the_guards_still_refuse(setup, label):
    p = _pool(threshold=16)
    busy = _pod(inflight=99)
    p.pods = [busy]
    setup(p)
    assert p._grow_reason([busy]) == "", label


def test_a_booting_pod_blocks_the_anticipatory_rent():
    """Single-flight: a prediction must not rent a second card while the first
    prediction is still being filled."""
    p = _pool(threshold=16, parallel=4)
    busy = _pod(inflight=16)
    p.pods = [busy]
    p._booting = 1
    assert p._grow_reason([busy]) == ""


# -------------------------------------------- the bool contract is unchanged #
def test_should_grow_still_returns_a_real_bool():
    """Callers and tests compare with `is True` / `is False`."""
    p = _pool(threshold=16)
    busy = _pod(inflight=16)
    p.pods = [busy]
    assert p._should_grow([busy]) is True
    quiet = _pod(inflight=0)
    p.pods = [quiet]
    assert p._should_grow([quiet]) is False


def test_the_caller_prints_the_reason_it_was_given():
    """It must not recompute a figure of its own — that was the bug."""
    import pathlib
    src = pathlib.Path(rw.__file__).read_text()
    blk = src.split("def acquire")[1].split("def release")[0]
    assert "growing — {why}" in blk
    assert "pod(s) of demand" not in blk
