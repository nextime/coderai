"""Asking for a single pod must not train on the local machine.

`_want_dist` required `max_pods > 1 or test_run`, so a body of
`pods: 1, min_pods: 1, rounds: 16, test_run: false` fell through to the LOCAL
path -- on an orchestrator with no GPU. Seven submissions died that way in
45 minutes, each reporting "this install has no CUDA device", having never
attempted a rental. `min_pods` was not consulted at all, though it is the
clearest statement that a caller wants a fleet.

The local fallback is the same failure already recorded elsewhere in this
module as "a test run turned into tens of gigabytes of download on an
orchestrator with no cards".
"""

import inspect

import pytest

from codai.api import loras


SRC = inspect.getsource(loras._text_train_blocking)


def _cond():
    i = SRC.index("_want_dist = (")
    return SRC[i:i + 700]


def test_pods_counts_as_asking_for_a_fleet():
    assert "(req.pods or 0) >= 1" in _cond()


def test_min_pods_counts_too():
    """The clearest signal of all, and it was ignored."""
    assert "(req.min_pods or 0) >= 1" in _cond()


def test_test_run_and_max_pods_still_count():
    c = _cond()
    assert "req.test_run" in c and "_train_cfg.max_pods > 1" in c


def test_an_explicit_distributed_false_still_wins():
    """Opting out must remain possible: it is the only way to train on a local
    card deliberately."""
    assert "req.distributed is not False" in _cond()


def test_asking_for_pods_and_still_going_local_is_refused():
    i = SRC.index("_asked_for_pods and not _want_dist")
    block = SRC[i:i + 1200]
    assert "raise RuntimeError(" in block
    assert "would train LOCALLY" in block


def test_the_refusal_shows_the_values_that_caused_it():
    """"It would run locally" without the numbers sends someone back to guess
    which field did it -- which is how the last hour went."""
    i = SRC.index("_asked_for_pods and not _want_dist")
    block = SRC[i:i + 1200]
    for field in ("pods=", "min_pods=", "max_pods=", "distributed=",
                  "test_run="):
        assert field in block


def test_the_refusal_distinguishes_the_two_causes():
    """An explicit opt-out and a block that caps at one pod need different
    answers from the reader."""
    i = SRC.index("_asked_for_pods and not _want_dist")
    block = SRC[i:i + 1200]
    assert "distributed: false` was passed explicitly" in block
    assert "max_pods=" in block
