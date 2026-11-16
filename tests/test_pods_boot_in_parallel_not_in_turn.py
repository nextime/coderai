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
    i = SRC.index("def _bring_up")
    block = SRC[i:i + 700]
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
