"""Per-model placement: local / remote / burst-to-RunPod.

The queue arithmetic is what matters here — a leaked slot silently caps a model's
concurrency forever, and a phantom release hands a slot to a waiter that should
still be waiting.
"""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codai.frontproxy.reqqueue import FrontQueue, QueueFull


def run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


def test_try_acquire_takes_a_free_slot_and_refuses_when_full():
    async def go():
        q = FrontQueue()
        assert await q.try_acquire("m", 1) is True      # free
        assert await q.try_acquire("m", 1) is False     # busy -> burst instead
        await q.release("m")
        assert await q.try_acquire("m", 1) is True      # freed again
    run(go())


def test_try_acquire_respects_capacity_above_one():
    async def go():
        q = FrontQueue()
        assert [await q.try_acquire("m", 3) for _ in range(3)] == [True, True, True]
        assert await q.try_acquire("m", 3) is False
    run(go())


def test_try_acquire_never_waits_even_with_a_queue_behind_it():
    async def go():
        q = FrontQueue()
        await q.acquire("m", 1, 4)                       # slot held
        waiter = asyncio.ensure_future(q.acquire("m", 1, 4))
        await asyncio.sleep(0)                           # let it enqueue
        # The point of try_acquire: it returns rather than joining that queue.
        assert await asyncio.wait_for(q.try_acquire("m", 1), timeout=0.5) is False
        await q.release("m")                             # hands the slot to waiter
        await asyncio.wait_for(waiter, timeout=0.5)
        await q.release("m")
    run(go())


def test_a_granted_slot_is_not_double_counted():
    """try_acquire must take exactly one slot — the burst path then skips the
    blocking acquire, and releasing once must fully free the model."""
    async def go():
        q = FrontQueue()
        assert await q.try_acquire("m", 1) is True
        await q.release("m")
        # If try_acquire had counted twice, this would still be occupied.
        assert await q.try_acquire("m", 1) is True
    run(go())


def test_queue_full_still_raises_for_non_burst_models():
    async def go():
        q = FrontQueue()
        await q.acquire("m", 1, 1)
        w = asyncio.ensure_future(q.acquire("m", 1, 1))
        await asyncio.sleep(0)
        with pytest.raises(QueueFull):
            await q.acquire("m", 1, 1)
        w.cancel()
        try:
            await w
        except asyncio.CancelledError:
            pass
    run(go())


# --------------------------------------------------------------------------- #
# the front's placement decision
# --------------------------------------------------------------------------- #
class _Front:
    """Just the bits of the front the placement decision touches."""

    from codai.frontproxy.app import FrontProxy as _FP
    _spill_on_busy = _FP._spill_on_busy

    def __init__(self, info):
        self._info = info

    def _model_info(self, model):
        return self._info


def test_spill_on_busy_only_when_enabled_and_triggered():
    assert _Front({}) ._spill_on_busy("m") is False
    assert _Front({"runpod_spillover": {"enabled": True}})._spill_on_busy("m") is False
    assert _Front({"runpod_spillover": {"on_busy": True}})._spill_on_busy("m") is False
    assert _Front({"runpod_spillover": {"enabled": True, "on_busy": True}}) \
        ._spill_on_busy("m") is True
    # A model that only bursts when the QUEUE overflows must not burst on busy.
    assert _Front({"runpod_spillover": {"enabled": True, "on_concurrency_full": True}}) \
        ._spill_on_busy("m") is False


def test_model_is_renamed_for_the_remote():
    import json
    from codai.frontproxy.app import FrontProxy

    body = json.dumps({"model": "local-name", "messages": []}).encode()
    assert json.loads(FrontProxy._rewrite_model(body, "remote-name"))["model"] \
        == "remote-name"
    # No served_model configured -> body untouched, byte for byte.
    assert FrontProxy._rewrite_model(body, "") is body
    # Garbage in, same garbage out rather than an exception.
    assert FrontProxy._rewrite_model(b"not json", "x") == b"not json"
