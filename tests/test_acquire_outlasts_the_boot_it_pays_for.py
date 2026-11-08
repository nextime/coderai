"""A caller that rents a pod must be able to receive it.

Measured in production on 2026-11-08, on a pool with `boot_timeout_s: 1800`
and `load_timeout_s: 900`:

    Error during generation: RunPod: no pod slot available for
    'pool:qwen3vl30a3b' (max_pods=15, max_inflight_per_pod=0) within 1200.0s.
    [http] 500 on POST /v1/chat/completions

`acquire()` defaulted to a flat 1200 s deadline while a single cold candidate
may legitimately spend boot_timeout_s + load_timeout_s + warmup_timeout_s --
3600 s for that pool. So the request that triggered the rental was GUARANTEED a
500, and the pod it had just paid to rent went on booting for somebody else.
The deadline now derives from the pool's own boot window at the moment the
caller commits to an inline rental.
"""
import time
import types

import pytest

from codai.api import runpod_worker as rw


def _pool(**over):
    """The least of a pool that `boot_window_s` and `acquire` touch."""
    mcfg = types.SimpleNamespace(
        boot_timeout_s=1800, load_timeout_s=900, warmup_timeout_s=900,
        warmup_on_boot=True, max_inflight_per_pod=0,
        scale_up_inflight_per_pod=1, min_pods=0, max_pods=15)
    for k, v in over.items():
        setattr(mcfg, k, v)
    return types.SimpleNamespace(
        mcfg=mcfg, model_key="pool:test",
        PORT_TIMEOUT_S=rw.RunpodPodPool.PORT_TIMEOUT_S,
        HEALTH_TIMEOUT_S=rw.RunpodPodPool.HEALTH_TIMEOUT_S,
        BOOT_DEADLINE_SLACK_S=rw.RunpodPodPool.BOOT_DEADLINE_SLACK_S)


def _window(stub):
    return rw.RunpodPodPool.boot_window_s(stub)


def test_the_window_is_the_sum_of_the_phases_a_boot_really_spends():
    # 1800 port + 900 load + 900 warm-up + 30 slack
    assert _window(_pool()) == pytest.approx(3630.0)


def test_the_production_pool_window_exceeds_the_old_flat_default():
    # The whole point: 1200 could never cover this pool, which is why the
    # requester that triggered the rental always got a 500.
    assert _window(_pool()) > 1200.0


def test_warmup_is_excluded_when_the_pool_does_not_warm_up():
    assert _window(_pool(warmup_on_boot=False)) == pytest.approx(2730.0)


def test_a_zero_or_missing_timeout_falls_back_to_the_class_defaults():
    # 0 means "unset" everywhere else in this config, so it must not shorten
    # the window to the slack alone.
    w = _window(_pool(boot_timeout_s=0, load_timeout_s=0, warmup_timeout_s=0))
    assert w == pytest.approx(rw.RunpodPodPool.PORT_TIMEOUT_S
                              + rw.RunpodPodPool.HEALTH_TIMEOUT_S + 30.0)


def test_the_slack_is_included_so_the_waiter_does_not_race_the_boot():
    tight = _pool(boot_timeout_s=10, load_timeout_s=10, warmup_on_boot=False)
    assert _window(tight) == pytest.approx(20.0 + rw.RunpodPodPool.BOOT_DEADLINE_SLACK_S)


class _FakeCV:
    """A condition variable that blocks never and can act on each wait."""

    def __init__(self, on_wait=None):
        self._on_wait = on_wait
        self.waits = 0

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def wait(self, _t=None):
        self.waits += 1
        if self._on_wait:
            self._on_wait()

    def notify_all(self):
        return None


def _clock_shim(clock):
    """A stand-in for the `time` module with a clock the test drives.

    It replaces the module REFERENCE inside runpod_worker, never an attribute
    on the real `time` module: `rw.time` IS that module, so
    `setattr(rw.time, "time", ...)` freezes the clock for every library in the
    process and for pytest itself -- which aborted a whole suite run at 79%
    with no summary before this was understood.
    """
    import time as _real
    return types.SimpleNamespace(
        time=lambda: clock["t"],
        monotonic=lambda: clock["t"],
        sleep=lambda _s=0: None,
        perf_counter=_real.perf_counter,
        strftime=_real.strftime,
        localtime=_real.localtime,
        gmtime=_real.gmtime,
    )


def _acquirable(monkeypatch, clock, on_wait=None, timeout_pool=None):
    """A pool `acquire` can actually run against, with a controllable clock."""
    monkeypatch.setattr(rw, "time", _clock_shim(clock))
    stub = timeout_pool or _pool()
    stub.pods = []
    stub.boot_window_s = lambda: rw.RunpodPodPool.boot_window_s(stub)
    stub._cv = _FakeCV(on_wait)
    stub._booting = 0
    stub._waiting = 0
    stub._follow_registry = lambda: None
    stub._pick = lambda healthy, affinity: (healthy[0] if healthy else None)
    stub._remember_affinity = lambda a, p: None
    stub._adopt_shared_pod = lambda: False
    stub._note_denied = lambda: None
    stub._grow_in_background = lambda: None
    stub.effective_max_pods = lambda: 15
    grown = {"n": 0}

    def _grow_reason(_healthy):
        # Asks to grow once; afterwards the pool is "being boot ed by us".
        grown["n"] += 1
        return "nothing up" if grown["n"] == 1 else ""

    stub._grow_reason = _grow_reason
    monkeypatch.setattr(rw, "sibling_is_provisioning", lambda k: False)
    monkeypatch.setattr(rw, "_slots_of", lambda p: [])
    return stub


def test_the_default_timeout_is_the_pools_own_boot_window(monkeypatch):
    """The fix: no caller gets a deadline a cold rental cannot land inside.

    `backends/runpod.py:271` calls `acquire()` with no arguments, so the flat
    1200 s default was the number actually in play when production answered
    "no pod slot available ... within 1200.0s" on a pool that needs 3600.
    """
    clock = {"t": 1000.0}
    stub = _acquirable(monkeypatch, clock)
    stub._grow_reason = lambda _healthy: ""
    stub._cv = _FakeCV(on_wait=lambda: clock.__setitem__("t", clock["t"] + 100.0))

    from codai.api.runpod_client import RunpodError
    with pytest.raises(RunpodError) as exc:
        rw.RunpodPodPool.acquire(stub)                 # no timeout given
    # It waited out the POOL's window, not 1200.
    assert "within 3630" in str(exc.value)
    assert clock["t"] - 1000.0 >= 3630.0


def test_an_explicit_timeout_is_honoured_exactly(monkeypatch):
    """A contract, not an accident.

    `test_the_callers_timeout_is_honoured_while_a_sibling_boots` exists
    because this method once ignored its deadline while a sibling booted.
    Lengthening a caller's own deadline -- however well meant -- reintroduces
    that hang, so an explicit number means exactly itself on every path.
    """
    from codai.api.runpod_client import RunpodError
    clock = {"t": 1000.0}
    stub = _acquirable(monkeypatch, clock)
    stub._grow_reason = lambda _healthy: ""
    stub._cv = _FakeCV(on_wait=lambda: clock.__setitem__("t", clock["t"] + 20.0))

    with pytest.raises(RunpodError, match="within 30.0s"):
        rw.RunpodPodPool.acquire(stub, timeout=30.0)
    assert clock["t"] - 1000.0 < 3630.0, "the caller's 30s was overridden"


def test_a_short_explicit_timeout_still_bails_out_while_a_sibling_boots(monkeypatch):
    """The exact hang my first attempt reintroduced, pinned here."""
    from codai.api.runpod_client import RunpodError
    clock = {"t": 1000.0}
    stub = _acquirable(monkeypatch, clock)
    # Growth stays justified for as long as the sibling holds the pool, so
    # every pass takes the sibling branch -- which is the real scenario.
    stub._grow_reason = lambda _healthy: "nothing up"
    monkeypatch.setattr(rw, "sibling_is_provisioning", lambda k: True)
    stub._cv = _FakeCV(on_wait=lambda: clock.__setitem__("t", clock["t"] + 5.0))

    with pytest.raises(RunpodError, match="still booting"):
        rw.RunpodPodPool.acquire(stub, timeout=0.3)
    assert stub._booting == 0, "the boot counter leaked on the sibling path"


def test_the_refusal_names_the_window_so_the_fix_is_obvious(monkeypatch):
    """The message has to say WHY a short deadline could never work."""
    from codai.api.runpod_client import RunpodError
    clock = {"t": 1000.0}
    stub = _acquirable(monkeypatch, clock)
    stub._grow_reason = lambda _healthy: ""
    stub._cv = _FakeCV(on_wait=lambda: clock.__setitem__("t", clock["t"] + 20.0))

    with pytest.raises(RunpodError) as exc:
        rw.RunpodPodPool.acquire(stub, timeout=10.0)
    msg = str(exc.value)
    assert "boot window" in msg
    assert "3630" in msg
    assert "boot_timeout_s" in msg


def test_a_grow_inline_false_caller_is_never_held_for_a_boot(monkeypatch):
    """The 'answer now' mode must keep answering now."""
    clock = {"t": 1000.0}
    stub = _acquirable(monkeypatch, clock)
    stub._provision_one = lambda *a, **kw: pytest.fail("must not rent inline")

    with pytest.raises(rw.NoPodRightNow):
        rw.RunpodPodPool.acquire(stub, timeout=30.0, grow_inline=False)
    assert clock["t"] == 1000.0
