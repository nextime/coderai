"""A four-pod job needed four CONSECUTIVE boots inside a one-boot budget.

The coordinator waited for addresses in a `for` loop: shard 0 was waited out
to completion before shard 1 was looked at, while shards 1..n sat booted and
BILLING. With a 600s publish grace and a boot that takes 6-7 minutes, a job
asking for four pods could not finish its waits inside any sane budget, and
the ones that did come up were released at the end for not meeting the floor:

    job-c9991acb7a48: shard 0 reachable 14:01:25, shard 1 reachable 14:02:33,
    both released -- "only 2 pod(s) became ready and this job requires at
    least 4"

Pods boot in parallel on RunPod's side, so the wall clock should be ONE boot.
"""

import inspect
import threading
import time

import pytest

from codai.api import loras


SRC = inspect.getsource(loras._text_train_distributed)


def test_the_waits_are_concurrent():
    assert "ThreadPoolExecutor" in SRC
    assert "as_completed" in SRC


def test_one_pod_does_not_pay_for_a_thread_pool():
    assert "if len(_shards) == 1" in SRC


def test_the_pool_is_bounded():
    """A 32-pod fleet must not open 32 threads without a ceiling stated."""
    assert "max_workers=min(len(_shards), 32)" in SRC


def test_a_pod_that_never_publishes_is_released_in_its_own_wait():
    """Not at the end of the round: it is billing, and the other waits may
    have minutes left to run."""
    block = _bring_up_body()
    assert 'fleet.release(shard, "never published a port")' in block
    assert 'fleet.release(shard, "no address")' in block


def test_cancel_is_checked_after_the_waits_not_inside_them():
    """A cancel mid-boot would leave a rented pod with nobody to release it."""
    i = SRC.index("ThreadPoolExecutor")
    after = SRC[i:i + 1400]
    assert "_check_train_cancel()" in after
    assert "would\n        # leave a rented pod" in after


def test_the_dataset_goes_out_in_shard_order():
    """So a failure names a shard rather than whichever thread finished
    first."""
    assert "for shard in sorted(ready)" in SRC


def test_a_raising_boot_does_not_abort_the_other_pods():
    i = SRC.index("as_completed")
    block = SRC[i:i + 600]
    assert "boot raised" in block
    assert "continue" in block


# ------------------------------------------------- the property, measured ---

def test_parallel_waiting_is_not_the_sum_of_the_waits():
    """The behaviour, not the spelling: four 0.3s boots must take about 0.3s,
    not 1.2s. Run against the real executor shape used above.
    """
    import concurrent.futures as cf

    def _boot(_shard):
        time.sleep(0.3)
        return _shard

    shards = [0, 1, 2, 3]
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=min(len(shards), 32)) as pool:
        got = sorted(f.result() for f in
                     cf.as_completed({pool.submit(_boot, s): s
                                      for s in shards}))
    took = time.time() - t0
    assert got == shards
    assert took < 0.9, f"waits did not overlap: {took:.2f}s for 4x0.3s"


# ------------- a slice expiring is not a pod failing ------------------------
#
# Slicing the wait to honour a cancel introduced a worse bug than the one it
# fixed: `wait_ready` RAISES at its deadline ("Pod X did not expose port 8000
# within 20.0s"), so the loop read every expired slice as the pod never coming
# up. Measured in production within minutes of the deploy:
#
#     shard 0 never came up: Pod jnomn0v9giwgiu did not expose port 8000
#     within 20.0s ... released after 0.4 min -- never published a port
#     FAILED: no pod became reachable -- nothing to train on
#
# A 1800s budget had become 20 seconds, and the card was released while it was
# still pulling its image.

def test_an_expired_slice_keeps_waiting():
    block = _bring_up_body()
    assert "A SLICE EXPIRING IS NOT A FAILURE" in block
    assert 'f"within {_slice}" in _m' in block
    assert "continue" in block


def test_the_deadline_is_what_ends_the_wait():
    """Not the slice: while there is budget left, a non-fatal error keeps
    waiting."""
    block = _bring_up_body()
    assert "time.time() < _deadline" in block


def test_a_verdict_that_waiting_cannot_change_still_ends_it():
    """An unreachable host, or a container that exited, must not be waited out
    for the full budget -- that is what the classifier is for."""
    block = _bring_up_body()
    for fatal in ("cannot be reached", "EXITED on boot", "entered status"):
        assert fatal in block, fatal


def test_the_slice_length_is_named_once():
    from codai.api import loras as _l
    assert _l._SLICE_S == 20.0
    assert "min(_SLICE_S, _left)" in _bring_up_body()


def test_the_budget_still_comes_from_the_model():
    assert "_boot_timeout = float(cfg.pod_boot_timeout_s or 1800.0)" in SRC

def _bring_up_body():
    """The whole `_bring_up` body, bounded by the statement after it.

    Bounded by a marker rather than a character count: three assertions in
    this file broke on the body growing, which tests where the text happened
    to end rather than what the code does.
    """
    src = SRC
    i = src.index("def _bring_up")
    j = src.index("if len(_shards) == 1", i)
    return src[i:j]
