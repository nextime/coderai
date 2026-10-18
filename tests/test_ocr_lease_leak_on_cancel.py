"""A cancelled page must not keep the pod slot it was granted.

`placement.choose` runs in a thread and raises `pod.inflight` the moment it
picks a pod. If the awaiting coroutine is cancelled after the executor has
produced the Lease but before the value is delivered, the object is dropped
with the counter already raised, and nothing releases it -- the page is gone,
so no `finally` runs.

A client that gives up on a batch cancels every page of it at once, so the leak
arrives in thousands. Measured on a live run with `max_inflight_per_pod: 64`:
the pool reported all 52 healthy pods at their cap -- 3,328 pages "in flight" --
while `ss` showed 3 to 5 open sockets per pod, the cards sat at 0.2% KV cache
with `Running: 1`, and nothing was preempted. Roughly 60 of every 64 pages did
not exist. `_pick` then refused every pod, correctly, and the run fell to
0.33 pages/s.

It had been leaking since the pod pipeline was written. With
`max_inflight_per_pod: 0` an inflated counter blocked nothing, so the only
symptom was demand -- in-flight counts of 2,700 and 4,096 that the scaler
rented pods to serve. The cap did not cause the problem; it made a silent one
fatal, which is how it was finally found.
"""

import asyncio
import threading
import time

import pytest

from codai.ocr import manager as om


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def _reset():
    om._ORPHAN_LEASES[0] = 0
    om._SAID_AT.clear()
    yield
    om._ORPHAN_LEASES[0] = 0
    om._SAID_AT.clear()


class _CountingPool:
    """Stands in for a pod pool: counts slots taken and given back."""

    def __init__(self):
        self.inflight = 0
        self.granted = 0
        self.released = 0
        self._lock = threading.Lock()

    def take(self):
        with self._lock:
            self.inflight += 1
            self.granted += 1

    def give_back(self):
        with self._lock:
            self.inflight -= 1
            self.released += 1


class _Lease:
    kind = "pod"
    base = "https://10.0.0.1:1234/v1"
    api_key = "k"

    def __init__(self, pool):
        self._pool = pool

    def release(self):
        self._pool.give_back()

    def dead(self):
        pass


def _slow_choose(pool, delay=0.2):
    """`choose` that takes a slot and then takes its time delivering it."""
    def _choose(*_a, **_k):
        pool.take()                 # the counter rises HERE, in the thread
        time.sleep(delay)
        return _Lease(pool)
    return _choose


# --------------------------------------------------------------- the leak #

@pytest.mark.anyio
async def test_a_cancelled_acquisition_gives_the_slot_back(monkeypatch):
    """THE LEAK. Cancel while the lease is being granted; the slot must return."""
    from codai.models import placement
    pool = _CountingPool()
    monkeypatch.setattr(placement, "choose", _slow_choose(pool))
    monkeypatch.setattr(placement, "spill_threshold", lambda *_a, **_k: 0)

    task = asyncio.ensure_future(om._acquire_lease("m"))
    await asyncio.sleep(0.05)          # let it reach the executor
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # The acquisition finishes in its own time; then the callback returns it.
    for _ in range(100):
        if pool.released:
            break
        await asyncio.sleep(0.02)
    assert pool.granted == 1, "the slot was never taken; test proves nothing"
    assert pool.released == 1, "the slot was NOT given back — still leaking"
    assert pool.inflight == 0


@pytest.mark.anyio
async def test_the_orphan_is_counted_and_said(monkeypatch, capsys):
    """The whole cost of this bug was that it was silent."""
    from codai.models import placement
    pool = _CountingPool()
    monkeypatch.setattr(placement, "choose", _slow_choose(pool))
    monkeypatch.setattr(placement, "spill_threshold", lambda *_a, **_k: 0)

    task = asyncio.ensure_future(om._acquire_lease("m"))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    for _ in range(100):
        if om._ORPHAN_LEASES[0]:
            break
        await asyncio.sleep(0.02)
    assert om._ORPHAN_LEASES[0] == 1
    out = capsys.readouterr().out
    assert "cancelled while its pod lease was being granted" in out
    assert "times out in bulk" in out


@pytest.mark.anyio
async def test_a_bulk_cancellation_leaks_nothing(monkeypatch):
    """A client giving up on a batch cancels every page at once."""
    from codai.models import placement
    pool = _CountingPool()
    monkeypatch.setattr(placement, "choose", _slow_choose(pool, delay=0.1))
    monkeypatch.setattr(placement, "spill_threshold", lambda *_a, **_k: 0)

    tasks = [asyncio.ensure_future(om._acquire_lease("m")) for _ in range(40)]
    await asyncio.sleep(0.05)
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    for _ in range(200):
        if pool.released >= pool.granted and pool.granted:
            break
        await asyncio.sleep(0.02)
    assert pool.granted > 0
    assert pool.inflight == 0, (
        f"{pool.granted} slots taken, {pool.released} returned — "
        f"{pool.inflight} leaked")


# ------------------------------------------------------- the normal paths #

@pytest.mark.anyio
async def test_an_uncancelled_acquisition_returns_the_lease(monkeypatch):
    from codai.models import placement
    pool = _CountingPool()
    monkeypatch.setattr(placement, "choose", _slow_choose(pool, delay=0.01))
    monkeypatch.setattr(placement, "spill_threshold", lambda *_a, **_k: 0)
    lease = await om._acquire_lease("m")
    assert lease is not None
    assert pool.inflight == 1, "it must NOT release a lease it handed over"
    lease.release()
    assert pool.inflight == 0


@pytest.mark.anyio
async def test_nothing_is_released_when_choose_raises(monkeypatch):
    """It raised instead of leasing, so there is no slot to give back."""
    from codai.models import placement
    pool = _CountingPool()

    def _boom(*_a, **_k):
        # Slow enough that the cancellation lands FIRST, which is the case
        # under test: the page is gone and the acquisition then fails.
        time.sleep(0.15)
        raise RuntimeError("no upstream")

    monkeypatch.setattr(placement, "choose", _boom)
    monkeypatch.setattr(placement, "spill_threshold", lambda *_a, **_k: 0)
    task = asyncio.ensure_future(om._acquire_lease("m"))
    await asyncio.sleep(0.03)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.3)
    assert pool.released == 0
    assert om._ORPHAN_LEASES[0] == 0


@pytest.mark.anyio
async def test_a_local_lease_is_not_counted_as_an_orphan(monkeypatch):
    """choose may answer `local`, which holds no pod slot."""
    from codai.models import placement

    class _Local:
        kind = "local"
        base = "http://127.0.0.1:8000/v1"
        api_key = ""
        released = 0

        def release(self):
            type(self).released += 1

    monkeypatch.setattr(placement, "choose", lambda *_a, **_k: _Local())
    monkeypatch.setattr(placement, "spill_threshold", lambda *_a, **_k: 0)
    lease = await om._acquire_lease("m")
    assert lease.kind == "local"
    assert om._ORPHAN_LEASES[0] == 0


def test_the_release_is_wired_through_lease_a_pod():
    """`_lease_a_pod` must use the safe acquisition, not the raw pool call."""
    src = open(om.__file__).read()
    i = src.index("async def _lease_a_pod(")
    body = src[i:i + 3000]
    assert "_acquire_lease(model_id)" in body
    assert "run_on_lease_pool(" not in body, (
        "it acquires directly again — a cancellation can lose that lease")
