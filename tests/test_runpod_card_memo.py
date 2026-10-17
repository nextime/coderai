"""The fleet grew because it could not serve, and could not serve because it had grown.

`pod_weight`, `pod_capacity` and `pod_threshold` depend only on a card's NAME
and the model config. They were recomputed per POD on every `acquire()`:
`_pick` once per free pod, `_pods_for_demand` once per serving pod, and
`_preferred_vram_gb` walking every pod again inside each of those -- all while
holding the pool lock.

At 70 pods that is ~140 recomputations per acquire, each doing substring
matches across the gpu_limits and gpu_weights tables. Measured on a live
5,000-document run: 811 pages waiting, 324 of the engine's 352 threads parked
in `futex_do_wait`, `/healthz` still answering in 1.6 ms -- so not a blocked
event loop, but every lease serialised behind the pool lock. The scaler read
the waiting pages as demand and rented another pod every 15 s, and each new pod
made the per-acquire cost higher: 69 -> 73 pods, $43/hr, zero documents read.

So the arithmetic is memoised per card. These tests pin both halves: that the
memo returns what the computation returned (a cache that is subtly wrong is
worse than a slow one), and that the per-acquire cost stops growing with the
fleet.
"""

import time
import types

import pytest

from codai.api import runpod_worker as rw


def _mcfg(**kw):
    base = dict(max_pods=128, min_pods=0, max_inflight_per_pod=0,
                scale_up_inflight_per_pod=32, pod_max_parallel_requests=64,
                pod_queue_max_size=256, scale_up_max_parallel_boots=32,
                gpu_types=["NVIDIA A100-SXM4-80GB", "NVIDIA A40"],
                gpu_weights={"NVIDIA A100-SXM4-80GB": 2.9, "NVIDIA L40S": 1.25,
                             "NVIDIA A40": 1.0, "A100 PCIe": 2.8},
                gpu_limits=None, health_path="/v1/models", env={})
    base.update(kw)
    return types.SimpleNamespace(**base)


def _pool(mcfg=None, cards=()):
    """A pool with no account work done: only the bits the memo touches."""
    pool = rw.RunpodPodPool.__new__(rw.RunpodPodPool)
    import threading
    pool.mcfg = mcfg or _mcfg()
    pool._cv = threading.Condition(threading.RLock())
    pool._card_cache = {}
    pool._card_gen = 0
    pool._booting = 0
    pool._waiting = 0
    pool._affinity = {}
    pool.model_key = "m"
    pool.pods = []
    for i, (card, vram) in enumerate(cards):
        pool.pods.append(types.SimpleNamespace(
            pod_id=f"p{i}", gpu=card, vram_gb=vram, inflight=0, healthy=True,
            draining=False, last_used=0.0, is_spot=False, hourly_usd=1.0,
            started_at=time.time(), url="https://10.0.0.1:1/v1"))
    return pool


def _pod(card, vram=48.0, inflight=0):
    return types.SimpleNamespace(gpu=card, vram_gb=vram, inflight=inflight,
                                 healthy=True, draining=False, pod_id="x")


# ------------------------------------------------------------- correctness #

def test_the_preferred_card_weighs_one():
    """Everything is relative to gpu_types[0] — an absolute 2.9 would size the
    fleet three times too large for the very card it was measured on."""
    p = _pool()
    assert p.pod_weight(_pod("NVIDIA A100-SXM4-80GB")) == pytest.approx(1.0)


def test_a_slower_card_weighs_less_than_one():
    p = _pool()
    assert p.pod_weight(_pod("NVIDIA A40")) == pytest.approx(1.0 / 2.9, rel=1e-6)


def test_an_unknown_card_is_neutral_and_keeps_serving():
    """A reporting gap must not become an outage."""
    p = _pool()
    assert p.pod_weight(_pod("Some Card Nobody Listed", vram=0.0)) == 1.0


def test_the_weight_is_clamped():
    p = _pool(_mcfg(gpu_types=["NVIDIA A40"],
                    gpu_weights={"NVIDIA A40": 1.0, "Monster": 500.0}))
    assert p.pod_weight(_pod("Monster")) == 8.0
    p2 = _pool(_mcfg(gpu_types=["Monster"],
                     gpu_weights={"NVIDIA A40": 1.0, "Monster": 500.0}))
    assert p2.pod_weight(_pod("NVIDIA A40")) == 0.25


def test_a_per_card_limits_row_wins_over_the_weights_table():
    """One source of truth: if the operator stated a depth per card, their
    ratio IS the share each should carry."""
    p = _pool(_mcfg(gpu_limits={"NVIDIA A40": {"scale_up": 16},
                                "NVIDIA A100-SXM4-80GB": {"scale_up": 64}}))
    assert p.pod_weight(_pod("NVIDIA A40")) == pytest.approx(16 / 64)


def test_unlimited_inflight_stays_unlimited():
    p = _pool()
    assert p.pod_capacity(_pod("NVIDIA A40")) == 0


def test_capacity_scales_with_the_weight():
    p = _pool(_mcfg(max_inflight_per_pod=100))
    assert p.pod_capacity(_pod("NVIDIA A100-SXM4-80GB")) == 100
    assert p.pod_capacity(_pod("NVIDIA A40")) == max(1, round(100 / 2.9))


def test_a_stated_per_card_inflight_is_taken_as_written():
    p = _pool(_mcfg(max_inflight_per_pod=100,
                    gpu_limits={"NVIDIA A40": {"inflight": 7}}))
    assert p.pod_capacity(_pod("NVIDIA A40")) == 7


def test_threshold_scales_with_the_weight():
    p = _pool()
    assert p.pod_threshold(_pod("NVIDIA A100-SXM4-80GB")) == pytest.approx(32.0)
    assert p.pod_threshold(_pod("NVIDIA A40")) == pytest.approx(32 / 2.9, rel=1e-6)


def test_a_stated_per_card_scale_up_is_taken_as_written():
    p = _pool(_mcfg(gpu_limits={"NVIDIA A40": {"scale_up": 9}}))
    assert p.pod_threshold(_pod("NVIDIA A40")) == pytest.approx(9.0)


# ------------------------------------------------------------------ memo #

def test_the_tables_are_scanned_once_per_card_not_once_per_pod(monkeypatch):
    calls = []
    real = rw.card_limits
    monkeypatch.setattr(rw, "card_limits",
                        lambda m, c: calls.append(c) or real(m, c))
    p = _pool(cards=[("NVIDIA A40", 48.0)] * 40 + [("NVIDIA L40S", 48.0)] * 30)
    for pod in p.pods:
        p.pod_weight(pod)
        p.pod_capacity(pod)
        p.pod_threshold(pod)
    # Two distinct cards. Without the memo this was ~210 scans for 70 pods.
    assert len(set(calls)) <= 4
    assert len(calls) <= 12, f"{len(calls)} table scans for 2 cards"


def test_a_new_pod_invalidates_the_memo():
    """`_preferred_vram_gb` reads the pods, so the generation must move."""
    p = _pool(cards=[("NVIDIA A40", 48.0)])
    before = p._card_gen
    with p._cv:
        p.pods.append(_pod("NVIDIA L40S"))
        p._bump_cards()
    assert p._card_gen > before


def test_the_memo_does_not_grow_without_bound():
    p = _pool()
    for i in range(300):
        p._card_cache[(f"card{i}", 0)] = {"weight": 1.0}
    with p._cv:
        p._bump_cards()
    assert len(p._card_cache) == 0


def test_the_memo_answers_the_same_as_a_cold_pool():
    """A cache that is subtly wrong is worse than a slow one."""
    cards = ["NVIDIA A40", "NVIDIA L40S", "A100 PCIe",
             "NVIDIA A100-SXM4-80GB", "Unlisted Card"]
    warm = _pool(_mcfg(max_inflight_per_pod=100))
    for c in cards:
        for _ in range(3):
            warm.pod_weight(_pod(c))
    for c in cards:
        cold = _pool(_mcfg(max_inflight_per_pod=100))
        assert warm.pod_weight(_pod(c)) == cold.pod_weight(_pod(c))
        assert warm.pod_capacity(_pod(c)) == cold.pod_capacity(_pod(c))
        assert warm.pod_threshold(_pod(c)) == cold.pod_threshold(_pod(c))


# --------------------------------------------------------------- scaling #

def _acquire_cost(n_pods: int) -> float:
    """Seconds for the per-acquire arithmetic over a fleet of n pods."""
    cards = [("NVIDIA A40", 48.0), ("NVIDIA L40S", 48.0),
             ("A100 PCIe", 80.0)]
    p = _pool(cards=[cards[i % 3] for i in range(n_pods)])
    healthy = list(p.pods)
    t0 = time.monotonic()
    for _ in range(40):
        p._pick(healthy, "")
        p._pods_for_demand(healthy)
    return time.monotonic() - t0


def test_the_per_acquire_cost_does_not_explode_with_the_fleet():
    """The runaway: each pod rented made every later acquire more expensive."""
    small = _acquire_cost(8)
    large = _acquire_cost(80)
    # 10x the pods is 10x the pods to iterate, and that part is unavoidable --
    # but it must not be the ~100x of recomputing per pod per call.
    assert large < small * 30, (
        f"8 pods: {small * 1000:.1f}ms, 80 pods: {large * 1000:.1f}ms — "
        f"{large / max(small, 1e-9):.0f}x for 10x the fleet")


# ------------------------------------------------------- saying WHY not #
# "811 pages waiting with 50 pods up" has four causes needing opposite
# responses, and the pool used to return a bare None for all of them.

def _no_pod_line(pool, healthy, capsys):
    pool._no_pod_said = 0.0
    assert pool._pick(healthy, "") is None
    return capsys.readouterr().out


def test_an_empty_pool_says_so(capsys):
    p = _pool()
    assert "holds no pods at all" in _no_pod_line(p, [], capsys)


def test_all_pods_unhealthy_says_booting_or_failing(capsys):
    p = _pool(cards=[("NVIDIA A40", 48.0)] * 50)
    for pod in p.pods:
        pod.healthy = False
    out = _no_pod_line(p, [], capsys)
    assert "none of its 50 pod(s) is healthy" in out
    assert "booting" in out


def test_all_pods_draining_says_draining(capsys):
    p = _pool(cards=[("NVIDIA A40", 48.0)] * 4)
    for pod in p.pods:
        pod.draining = True
    assert "are draining" in _no_pod_line(p, list(p.pods), capsys)


def test_all_pods_at_cap_says_the_cap(capsys):
    p = _pool(_mcfg(max_inflight_per_pod=2),
              cards=[("NVIDIA A100-SXM4-80GB", 80.0)] * 3)
    for pod in p.pods:
        pod.inflight = 2
    out = _no_pod_line(p, list(p.pods), capsys)
    assert "in-flight cap" in out
    assert "max_inflight_per_pod" in out


def test_the_counts_are_all_there_for_an_operator(capsys):
    p = _pool(cards=[("NVIDIA A40", 48.0)] * 7)
    for pod in p.pods:
        pod.healthy = False
    p._booting, p._waiting = 22, 811
    out = _no_pod_line(p, [], capsys)
    for frag in ("7 in pool", "0 healthy", "22 booting", "811 waiting"):
        assert frag in out, f"missing {frag!r} in: {out}"


def test_it_is_rate_limited_not_once_per_page(capsys):
    """811 waiting pages must not write 811 lines."""
    p = _pool()
    p._no_pod_said = 0.0
    for _ in range(200):
        p._pick([], "")
    assert capsys.readouterr().out.count("NO POD") == 1


def test_a_pool_with_a_free_pod_says_nothing(capsys):
    p = _pool(cards=[("NVIDIA A40", 48.0)])
    assert p._pick(list(p.pods), "") is not None
    assert "NO POD" not in capsys.readouterr().out
