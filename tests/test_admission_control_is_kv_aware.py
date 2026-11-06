"""Admission that follows the cache instead of guessing a batch width.

A serving engine admits as many sequences as it is told to without checking
whether its KV cache can hold them at the length they will REACH, then evicts
half-generated requests and recomputes them. Throughput collapses while every
counter still looks busy.

Computing the right width is not available: it is a launch flag while the cache
size is only reported after boot, it lives on a model entry while cache size is
a property of the card, and a persisted measurement-driven limit that only moves
down is how `measured_n_gpu_layers` once ratcheted 31 -> 20 and stuck.

So the controller gates on cache OCCUPANCY -- a fraction, which means the same
thing on every card -- and converges the way congestion control does. These
tests pin, in order of what getting them wrong would cost:

  1. it never raises the operator's stated maximum, and never stalls below the floor
  2. a rewinding counter is a restart, not negative loss
  3. with no reading it does NOTHING, rather than guessing
  4. the projection can only veto an increase, never license one
  5. raising the limit releases waiters, or the capacity is granted to nobody
"""
import asyncio

import pytest

from codai.api import admission
from codai.api.admission import Decision, Policy, State, decide


def _pol(**kw):
    base = dict(mode="on", ceiling=40, min_parallel=4, kv_target_pct=0.85,
                kv_ceiling_pct=0.95, backoff=0.75, recover_after=3)
    base.update(kw)
    return Policy(**base)


def _read(kv=0.5, preemptions=0, running=0, **kw):
    out = {"kv_cache_pct": kv, "preemptions": preemptions, "running": running}
    out.update(kw)
    return out


# --- 1. bounds -----------------------------------------------------------

def test_it_never_raises_the_operators_maximum():
    st = State(limit=40, last=_read())
    got = decide(st, _read(kv=0.1), _pol())
    assert got.limit == 40
    assert "maximum" in got.reason


def test_it_never_falls_below_the_floor():
    st = State(limit=5, last=_read(preemptions=0))
    got = decide(st, _read(kv=0.5, preemptions=9), _pol(min_parallel=4))
    assert got.limit == 4
    for _ in range(10):
        st = State(limit=got.limit, last=_read(preemptions=0))
        got = decide(st, _read(kv=0.5, preemptions=9), _pol(min_parallel=4))
        assert got.limit == 4


def test_a_floor_above_the_ceiling_cannot_invert_the_range():
    st = State(limit=0, last=_read())
    got = decide(st, _read(kv=0.99), _pol(ceiling=2, min_parallel=64))
    assert 1 <= got.limit <= 2


def test_a_preemption_decreases_multiplicatively():
    st = State(limit=40, last=_read(preemptions=100))
    got = decide(st, _read(kv=0.5, preemptions=101), _pol())
    assert got.limit == 30                      # 40 * 0.75
    assert got.changed is True
    assert "preemption" in got.reason


def test_a_full_cache_decreases_before_any_preemption():
    """A leading indicator: back off before paying the recompute."""
    st = State(limit=40, last=_read(preemptions=7))
    got = decide(st, _read(kv=0.97, preemptions=7), _pol())
    assert got.limit == 30
    assert "ceiling" in got.reason


def test_a_comfortable_cache_increases_by_one():
    st = State(limit=20, last=_read(preemptions=3))
    got = decide(st, _read(kv=0.40, preemptions=3), _pol())
    assert got.limit == 21


def test_a_cache_inside_target_holds():
    st = State(limit=20, last=_read(preemptions=3))
    got = decide(st, _read(kv=0.90, preemptions=3), _pol())
    assert got.limit == 20
    assert got.changed is False


def test_backoff_that_would_not_move_still_moves():
    """At small limits int(limit * 0.75) can equal limit; a decrease that does
    not decrease would make the floor unreachable from above."""
    st = State(limit=5, last=_read(preemptions=0))
    got = decide(st, _read(preemptions=1), _pol(min_parallel=1, backoff=0.99))
    assert got.limit < 5


# --- 2. a rewinding counter is a restart ---------------------------------

def test_a_rewound_preemption_counter_is_not_negative_loss():
    st = State(limit=20, last=_read(preemptions=5000))
    got = decide(st, _read(kv=0.3, preemptions=12), _pol())
    assert got.limit == 20, "a restart must not be read as a reason to grow"
    assert "restart" in got.reason


def test_a_rewound_request_counter_does_not_poison_the_estimate():
    st = State(limit=20, e_tokens=4000.0,
               last=_read(requests_done=900, prompt_tokens=9e6,
                          generation_tokens=9e6))
    got = decide(st, _read(kv=0.3, requests_done=3, prompt_tokens=30.0,
                           generation_tokens=30.0), _pol())
    assert got.e_tokens == pytest.approx(4000.0)


# --- 3. no reading, no action -------------------------------------------

def test_no_occupancy_reading_leaves_the_limit_alone():
    st = State(limit=17, last={})
    got = decide(st, {"preemptions": 0}, _pol())
    assert got.limit == 17
    assert got.changed is False
    assert "no cache-occupancy" in got.reason


def test_mode_off_decides_nothing():
    st = State(limit=11, last=_read())
    got = decide(st, _read(kv=0.1), _pol(mode="off"))
    assert got.limit == 11
    assert got.reason == "disabled"


def test_no_stated_maximum_disables_the_controller():
    """`max_parallel_requests` unset means no range to work inside."""
    st = State(limit=0, last=_read())
    got = decide(st, _read(kv=0.1), _pol(ceiling=0))
    assert got.reason == "disabled"


def test_occupancy_is_read_as_a_fraction_whichever_way_it_arrives():
    """vLLM's gauge is named `..._perc` and reports a FRACTION. Reading 0.9 as
    0.9% would make a full cache indistinguishable from an empty one -- and the
    controller would grow into a cache that is already overflowing."""
    assert admission._as_fraction(0.9) == pytest.approx(0.9)
    assert admission._as_fraction(90) == pytest.approx(0.9)
    assert admission._as_fraction(None) is None
    assert admission._as_fraction("nope") is None
    st = State(limit=20, last=_read(preemptions=0))
    assert decide(st, _read(kv=97.0, preemptions=0), _pol()).limit == 15


# --- 4. the projection is a veto, never a licence ------------------------

def test_the_projection_blocks_an_increase_the_cache_would_allow():
    """Occupancy is a snapshot of something still GROWING: 20 requests that
    have each produced 2k of an eventual 8k read as a comfortable cache and are
    already committed to overflowing it."""
    st = State(limit=20, last=_read(preemptions=0))
    r = _read(kv=0.40, preemptions=0, running=20, kv_tokens=100_000)
    got = decide(st, r, _pol(expected_tokens=8000))
    assert got.limit == 20
    assert "committed" in got.reason


def test_the_projection_allows_an_increase_that_fits():
    st = State(limit=20, last=_read(preemptions=0))
    r = _read(kv=0.40, preemptions=0, running=4, kv_tokens=1_000_000)
    got = decide(st, r, _pol(expected_tokens=8000))
    assert got.limit == 21


def test_the_projection_never_raises_a_limit_a_preemption_lowered():
    """A veto only. If it could license an increase, a wrong expected-tokens
    would build a confident version of the bug it exists to prevent."""
    st = State(limit=20, last=_read(preemptions=4))
    r = _read(kv=0.10, preemptions=5, running=0, kv_tokens=10_000_000)
    got = decide(st, r, _pol(expected_tokens=1))
    assert got.limit == 15


def test_an_unknown_cache_size_falls_back_to_observed_occupancy():
    st = State(limit=20, last=_read(preemptions=0))
    got = decide(st, _read(kv=0.4, preemptions=0, running=50), _pol())
    assert got.limit == 21


def test_expected_tokens_is_measured_from_the_counter_window():
    """`requests_done` accumulates from boot, so a lifetime mean keeps folding
    in requests from before the workload changed."""
    st = State(limit=20, e_tokens=0.0,
               last=_read(requests_done=100, prompt_tokens=200_000,
                          generation_tokens=100_000))
    got = decide(st, _read(kv=0.4, requests_done=110,
                           prompt_tokens=200_000 + 20_000,
                           generation_tokens=100_000 + 10_000), _pol())
    assert got.e_tokens == pytest.approx(3000.0)      # 30,000 / 10


def test_a_configured_expectation_wins_over_the_measurement():
    st = State(limit=20, e_tokens=50.0,
               last=_read(requests_done=1, prompt_tokens=1,
                          generation_tokens=1))
    got = decide(st, _read(kv=0.4, requests_done=2, prompt_tokens=2,
                           generation_tokens=2), _pol(expected_tokens=9999))
    assert got.e_tokens == 9999


# --- 5. the hold, and applying the result -------------------------------

def test_it_holds_before_growing_again_after_a_decrease():
    pol = _pol(recover_after=3)
    c = admission.AdmissionController(pol, queue_manager=_FakeQM())
    c.state = State(limit=40, last=_read(preemptions=0))

    async def _go():
        c._read = lambda: _answer({"kv_cache_pct": 0.5, "preemptions": 1})
        await c._tick()                      # preempted -> 30, hold set
        assert c.state.limit == 30
        assert c.state.hold == 3
        for expect_hold in (2, 1, 0):
            c._read = lambda: _answer({"kv_cache_pct": 0.1, "preemptions": 1})
            await c._tick()
            assert c.state.limit == 30, "must not grow while holding"
            assert c.state.hold == expect_hold
        c._read = lambda: _answer({"kv_cache_pct": 0.1, "preemptions": 1})
        await c._tick()
        assert c.state.limit == 31, "and then it grows again"

    asyncio.run(_go())


def _answer(d):
    async def _f():
        return d
    return _f()


class _FakeQM:
    def __init__(self):
        self.caps = []

    async def set_dynamic_cap(self, limit):
        self.caps.append(int(limit))


def test_the_decision_is_applied_to_the_gate():
    qm = _FakeQM()
    c = admission.AdmissionController(_pol(), queue_manager=qm)
    c.state = State(limit=40, last=_read(preemptions=0))

    async def _go():
        c._read = lambda: _answer({"kv_cache_pct": 0.5, "preemptions": 2})
        await c._tick()

    asyncio.run(_go())
    assert qm.caps == [30]


def test_a_failing_tick_never_stops_the_controller():
    c = admission.AdmissionController(_pol(), queue_manager=_FakeQM())

    def _boom():
        raise RuntimeError("metrics endpoint went away")

    async def _go():
        c._read = _boom
        with pytest.raises(RuntimeError):
            await c._tick()
        # ...and _run swallows it: a controller is an optimisation and must
        # never be the reason a server stops serving.
        assert "except Exception" in __import__("inspect").getsource(c._run)

    asyncio.run(_go())


def test_the_snapshot_reports_what_a_reader_needs():
    c = admission.AdmissionController(_pol(), queue_manager=_FakeQM())
    snap = c.snapshot()
    for key in ("mode", "limit", "ceiling", "floor", "reason", "ticks",
                "expected_tokens"):
        assert key in snap


# --- the gate itself ----------------------------------------------------

def test_the_gate_takes_the_smaller_of_the_two_numbers():
    from codai.queue.manager import QueueManager
    qm = QueueManager()
    qm.max_parallel_requests = 40
    assert qm.effective_parallel_limit() == 40
    qm.dynamic_cap = 12
    assert qm.effective_parallel_limit() == 12
    qm.dynamic_cap = 400
    assert qm.effective_parallel_limit() == 40, (
        "the controller must not be able to raise the operator's maximum")
    qm.max_parallel_requests = 0
    qm.dynamic_cap = 7
    assert qm.effective_parallel_limit() == 7


def test_raising_the_cap_releases_waiters():
    """`_release_locked` hands a freed slot to exactly one waiter, so a RAISED
    limit frees capacity nothing is holding -- and queued requests would sit
    there until an unrelated request finished."""
    from codai.queue.manager import QueueManager

    async def _go():
        qm = QueueManager()
        qm.max_parallel_requests = 8
        qm.dynamic_cap = 1
        qm.loaded_models = {"a", "b"}
        first = await qm.acquire("r1", "a")
        assert first is not None
        waiting = asyncio.ensure_future(qm.acquire("r2", "b"))
        await asyncio.sleep(0)
        assert not waiting.done(), "the cap of 1 must hold it"
        await qm.set_dynamic_cap(4)
        await asyncio.wait_for(waiting, timeout=1.0)
        assert waiting.done()

    asyncio.run(_go())


def test_the_gate_reports_both_numbers():
    from codai.queue.manager import QueueManager
    qm = QueueManager()
    qm.max_parallel_requests = 40
    qm.dynamic_cap = 9
    m = qm.get_metrics()
    assert m["max_parallel_requests"] == 40
    assert m["dynamic_cap"] == 9
    assert m["effective_parallel_limit"] == 9
