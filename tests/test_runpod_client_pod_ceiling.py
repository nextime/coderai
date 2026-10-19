"""A request may say how large the fleet is allowed to BECOME, not only how
many pods to hold warm.

/v1/runpod/scale set a FLOOR and nothing else. "Give me 8 pods" meant "hold at
least 8 ready", and the autoscaler stayed free to grow the pool to the
configured max_pods for as long as work kept arriving. A client running a
staged batch -- 8 pods for the first stage, 32 for the next -- had no way to
bound a stage at all.

Observed 2026-10-18: a run that asked for 8 pods was at 13 and climbing toward
128. Each pod past the plan is ~700 s of paid boot producing nothing, which is
why the realised cost per page came out 16.5x the measured baseline while every
pod was behaving correctly. Nothing was broken; the number the client sent was
simply not the number it needed to send.

So a lease may also carry `max_pods`, under the same rule `gpu_types` follows
there: it may only ever NARROW the operator's ceiling, and it dies with the
lease. The properties these tests pin, in order of what getting them wrong
would cost:

  1. a cap is never WIDENED -- a request must not be able to raise what the
     operator budgeted, which is the whole reason this is not admin-scoped
  2. EVERY growth path honours it, because a ceiling half of them respect is a
     race, not a ceiling
  3. it expires with the lease, so a client that dies holding a cap does not
     leave the pool permanently narrowed
  4. a floor and a cap are separate questions, and asking one must not silently
     answer the other
"""

import time
import types

import pytest

from codai.api import runpod_worker as rw


def _pool(max_pods=128, min_pods=0, threshold=16, parallel=4, ttl=1800):
    class _Acct:
        enabled = True

    return rw.RunpodPodPool("m", _Acct(), rw.parse_model_runpod({
        "max_pods": max_pods,
        "min_pods": min_pods,
        "scale_up_inflight_per_pod": threshold,
        "scale_up_max_parallel_boots": parallel,
        "allow_client_scale": True,
        "client_scale_ttl_s": ttl,
    }), "m")


def _pod(inflight=0, healthy=True, draining=False):
    return types.SimpleNamespace(inflight=inflight, healthy=healthy,
                                 draining=draining, pod_id="p", url="u",
                                 last_used=0.0, gpu="NVIDIA A40")


# --- 1. a request may only ever narrow ------------------------------------

def test_a_lease_cannot_raise_the_configured_ceiling():
    """The reason this endpoint can stay unauthenticated."""
    p = _pool(max_pods=8)
    out = p.set_client_floor(2, max_pods=1024)
    assert p.effective_max_pods() == 8
    assert out["configured_max_pods"] == 8
    assert out["client_max_pods"] == 8


def test_a_narrowed_ceiling_that_was_clamped_says_so():
    """A cap silently widened would be read as a cap that held."""
    p = _pool(max_pods=8)
    out = p.set_client_floor(2, max_pods=1024)
    assert any("narrowed" in n for n in out.get("notes", [])), out


def test_a_narrowing_lease_is_obeyed():
    p = _pool(max_pods=128)
    p.set_client_floor(8, max_pods=8)
    assert p.effective_max_pods() == 8


def test_no_lease_leaves_the_operator_ceiling_alone():
    p = _pool(max_pods=128)
    assert p.effective_max_pods() == 128
    assert p.client_cap() == 0


def test_an_operator_who_configured_zero_pods_still_gets_zero():
    """A LEASE can never drive the ceiling to zero, but max_pods: 0 is the
    operator having turned pod renting off, and that is not a lease."""
    p = _pool(max_pods=1)
    # `parse_model_runpod` floors the configured value at 1, so 0 only reaches
    # the pool by direct assignment -- which is how the scale-out fixtures
    # express "this pool may not rent", and the growth paths must keep
    # answering it the same way.
    p.mcfg.max_pods = 0
    assert p.effective_max_pods() == 0
    p.set_client_floor(4, max_pods=4)
    assert p.effective_max_pods() == 0


def test_a_cap_is_never_zero():
    """'Do not grow past N' is not a request to stop serving."""
    p = _pool(max_pods=128)
    p.set_client_floor(0, max_pods=0)
    assert p.effective_max_pods() == 128
    p.set_client_floor(1, max_pods=1)
    assert p.effective_max_pods() == 1


# --- 2. every growth path honours it --------------------------------------

def test_the_grower_stops_at_the_leased_ceiling():
    """_grow_reason is the autoscaler's own question."""
    p = _pool(max_pods=128, threshold=16)
    p.set_client_floor(0, max_pods=2)
    busy = [_pod(inflight=16), _pod(inflight=16)]
    p.pods = list(busy)
    assert p._grow_reason(busy) == "", "a saturated fleet still must not grow past the cap"


def test_the_grower_would_have_grown_without_the_cap():
    """Guards the test above: the cap is what stopped it, not the state."""
    p = _pool(max_pods=128, threshold=16)
    busy = [_pod(inflight=16), _pod(inflight=16)]
    p.pods = list(busy)
    assert p._grow_reason(busy), "the trigger must really fire, or the cap proves nothing"


def test_demand_is_bounded_by_the_leased_ceiling():
    """_pods_for_demand: what a measured queue justifies renting."""
    p = _pool(max_pods=128, threshold=16)
    p.set_client_floor(0, max_pods=4)
    serving = [_pod(inflight=16)]
    p.pods = list(serving)
    p._denied = [time.time()] * 2048          # a very deep queue
    assert p._pods_for_demand(serving) <= 4


def test_scale_down_targets_the_leased_ceiling():
    """_pods_needed decides how many to KEEP, so the cap must shed too."""
    p = _pool(max_pods=128, threshold=1)
    p.set_client_floor(0, max_pods=3)
    serving = [_pod(inflight=8) for _ in range(8)]
    assert p._pods_needed(serving) == 3


def test_the_warm_floor_top_up_cannot_exceed_the_leased_ceiling():
    """The floor is a convenience; the ceiling is the budget."""
    p = _pool(max_pods=128, min_pods=16)
    p.set_client_floor(0, max_pods=2)
    # The operator's own min_pods is higher than the cap. The cap wins, because
    # the alternative is renting cards the client explicitly refused.
    assert p.effective_max_pods() == 2
    assert p.effective_min_pods() <= 16
    healthy_in_hand = 0
    billing_in_hand = 0
    want = min(p.effective_min_pods() - healthy_in_hand, 4,
               max(0, p.effective_max_pods() - billing_in_hand))
    assert want <= 2


# --- 3. the cap dies with the lease ---------------------------------------

def test_the_cap_expires():
    p = _pool(max_pods=128, ttl=60)
    p.set_client_floor(2, max_pods=2)
    assert p.effective_max_pods() == 2
    p._client_cap_until = time.time() - 1
    assert p.client_cap() == 0
    assert p.effective_max_pods() == 128


def test_releasing_the_whole_lease_releases_the_cap():
    """`pods: 0` alone is a client going away; it keeps nothing."""
    p = _pool(max_pods=128)
    p.set_client_floor(4, max_pods=4)
    p.set_client_floor(0)
    assert p.client_cap() == 0
    assert p.client_floor() == 0
    assert p.effective_max_pods() == 128


def test_a_cap_survives_a_floor_of_zero_when_the_same_call_asks_for_it():
    """'Grow no further than 8, but hold nothing warm' is a real request: a
    client with a long tail of work wants the ceiling and not the floor."""
    p = _pool(max_pods=128)
    p.set_client_floor(0, max_pods=8)
    assert p.client_floor() == 0
    assert p.client_cap() == 8
    assert p.effective_max_pods() == 8


def test_renewing_the_floor_alone_keeps_the_cap():
    """max_pods omitted means 'leave it as it is', not 'drop it'."""
    p = _pool(max_pods=128)
    p.set_client_floor(4, max_pods=8)
    p.set_client_floor(6)
    assert p.client_cap() == 8
    assert p.client_floor() == 6


# --- 4. a floor and a cap are separate questions --------------------------

def test_asking_for_warm_pods_alone_does_not_cap_the_fleet():
    """The behaviour BEFORE this existed, and it must not change: a floor says
    nothing about the ceiling, so an install that never sends max_pods keeps
    the autoscaler it had."""
    p = _pool(max_pods=128)
    p.set_client_floor(8)
    assert p.effective_max_pods() == 128


def test_a_floor_above_the_cap_is_granted_as_the_cap():
    """Incoherent, but refusing would leave the client with nothing. The
    ceiling wins: it is the number that bounds the bill."""
    p = _pool(max_pods=128)
    out = p.set_client_floor(32, max_pods=8)
    assert out["granted"] == 8
    assert p.client_floor() == 8
    assert p.effective_max_pods() == 8


def test_lowering_a_cap_under_a_live_floor_lowers_the_floor_too():
    """Otherwise the pool must grow to a floor a ceiling forbids, and
    oscillates for as long as the lease lasts."""
    p = _pool(max_pods=128)
    p.set_client_floor(16)
    assert p.client_floor() == 16
    p.set_client_floor(16, max_pods=4)
    assert p.effective_max_pods() == 4
    assert p.client_floor() == 4


# --- the endpoint carries it ----------------------------------------------

def test_the_scale_endpoint_accepts_and_forwards_max_pods():
    src = open("codai/api/app.py").read()
    assert "max_pods: Optional[int] = Field(" in src
    assert "max_pods=body.max_pods" in src


def test_set_client_pods_forwards_it_for_both_ways_of_naming_a_pool():
    """A model id and a live `pool:` key take different branches, and a cap
    honoured on only one of them is a cap that depends on how it was asked."""
    import inspect
    for fn in (rw.set_client_pods, rw._scale_live_pool):
        assert "max_pods" in inspect.signature(fn).parameters, fn.__name__
    src = inspect.getsource(rw.set_client_pods)
    assert src.count("max_pods=max_pods") >= 2, (
        "both the live-pool branch and the by-model branch must pass it")


def test_the_status_surface_reports_the_ceiling_in_force():
    """A fleet that stopped growing short of max_pods must say why."""
    src = inspect_source = open("codai/api/runpod_worker.py").read()
    assert '"effective_max_pods": pool.effective_max_pods()' in src
    assert '"client_max_pods": pool.client_cap()' in src


def test_no_growth_path_still_reads_the_raw_configured_ceiling():
    """The regression guard. `mcfg.max_pods` is the BUDGET; the growth paths
    must read `effective_max_pods()`, and a new one added later that reaches
    for the raw number would silently escape every client cap."""
    import inspect
    for name in ("_pods_for_demand", "_grow_reason", "_pods_needed"):
        src = inspect.getsource(getattr(rw.RunpodPodPool, name))
        assert "mcfg.max_pods" not in src, f"{name} reads the raw ceiling"
    src = inspect.getsource(rw.RunpodPodPool.maintain)
    assert "self.mcfg.max_pods or 0" not in src, "the warm-floor top-up reads the raw ceiling"
