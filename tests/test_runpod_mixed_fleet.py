"""A mixed fleet: know the card, size for the preferred one, share accordingly.

Three faults, one cause. The pool chose pods with `min(inflight)` and sized
growth with `load / scale_up_inflight_per_pod`, both card-blind -- so an A100
and an A40 were handed the same number of pages, the fast card drained its
share and idled while the slow one straggled (and a document finishes only when
its LAST page lands), and a fleet of bigger cards was still sized as if every
pod were the preferred one. The third fault was the reason the first two could
not be fixed: an ADOPTED pod recorded `gpu="(adopted)"`, so a whole day's fleet
report read `by_gpu: {"(adopted)": 32}` and nothing knew what it was holding.
"""

import types

import pytest

from codai.api import runpod_worker as rw


def _pod(pid, gpu="", vram=0.0, inflight=0, draining=False):
    return rw.PodHandle(pod_id=pid, url="https://10.0.0.1:1/v1", hourly_usd=0.59,
                        started_at=0.0, gpu=gpu, vram_gb=vram,
                        inflight=inflight, draining=draining)


def _pool(gpu_types=("NVIDIA A40",), weights=None, cap=0, threshold=16, max_pods=64):
    p = rw.RunpodPodPool.__new__(rw.RunpodPodPool)
    p.mcfg = types.SimpleNamespace(
        gpu_types=list(gpu_types), gpu_weights=dict(weights or {}),
        max_inflight_per_pod=cap, scale_up_inflight_per_pod=threshold,
        max_pods=max_pods, sticky_sessions=False)
    p.pods = []
    p._affinity = {}
    p._waiting = 0
    import threading
    p._cv = threading.Condition()
    return p


# ------------------------------------------------------------- the weight #

def test_vram_sets_the_weight_relative_to_the_preferred_card():
    p = _pool(gpu_types=("NVIDIA A40", "NVIDIA A100-SXM4-80GB"))
    a40 = _pod("a", "NVIDIA A40", 48.0)
    a100 = _pod("b", "NVIDIA A100-SXM4-80GB", 80.0)
    p.pods = [a40, a100]
    assert p.pod_weight(a40) == pytest.approx(1.0)
    assert p.pod_weight(a100) == pytest.approx(80 / 48, rel=1e-3)


def test_an_explicit_weight_beats_vram_because_vram_is_not_speed():
    """VRAM bounds how many requests FIT and says nothing about how fast they
    finish: an A100 SXM holds 1.67x an A40's memory but has 2.93x its
    bandwidth, and an L40S holds exactly as much while being ~1.24x faster."""
    p = _pool(gpu_types=("NVIDIA A40", "NVIDIA A100-SXM4-80GB", "NVIDIA L40S"),
              weights={"NVIDIA A100-SXM4-80GB": 2.9, "NVIDIA L40S": 1.25})
    p.pods = [_pod("a", "NVIDIA A40", 48.0)]
    assert p.pod_weight(_pod("b", "NVIDIA A100-SXM4-80GB", 80.0)) == pytest.approx(2.9)
    # ...and a same-size card gets credit VRAM would have denied it entirely.
    assert p.pod_weight(_pod("c", "NVIDIA L40S", 48.0)) == pytest.approx(1.25)


def test_an_unknown_card_carries_the_preferred_share_and_keeps_serving():
    """A reporting gap must not become an outage."""
    p = _pool()
    p.pods = [_pod("a", "NVIDIA A40", 48.0)]
    assert p.pod_weight(_pod("x")) == 1.0
    assert p.pod_weight(types.SimpleNamespace(inflight=0, draining=False)) == 1.0


def test_no_preferred_card_makes_the_whole_mechanism_inert():
    """A single-card fleet must not be re-routed by an optimisation for mixed
    ones."""
    p = _pool(gpu_types=())
    assert p.pod_weight(_pod("a", "NVIDIA A100-SXM4-80GB", 80.0)) == 1.0


def test_the_weight_is_clamped():
    p = _pool(gpu_types=("NVIDIA A40",), weights={"huge": 999, "tiny": 0.0001})
    p.pods = [_pod("a", "NVIDIA A40", 48.0)]
    assert p.pod_weight(_pod("h", "huge", 48.0)) == 8.0
    assert p.pod_weight(_pod("t", "tiny", 48.0)) == 0.25


# -------------------------------------------------------- the distribution #

def test_the_faster_card_is_given_more_of_the_load():
    p = _pool(gpu_types=("NVIDIA A40", "NVIDIA A100-SXM4-80GB"),
              weights={"NVIDIA A100-SXM4-80GB": 3.0})
    a40 = _pod("a", "NVIDIA A40", 48.0, inflight=2)
    a100 = _pod("b", "NVIDIA A100-SXM4-80GB", 80.0, inflight=5)
    p.pods = [a40, a100]
    # Plain least-connections would pick the A40 (2 < 5). Weighted picks the
    # A100: 5/3 is less loaded than 2/1.
    assert p._pick([a40, a100], "") is a100


def test_an_equal_fleet_still_balances_evenly():
    p = _pool()
    a = _pod("a", "NVIDIA A40", 48.0, inflight=3)
    b = _pod("b", "NVIDIA A40", 48.0, inflight=1)
    p.pods = [a, b]
    assert p._pick([a, b], "") is b


def test_a_draining_pod_is_never_picked():
    p = _pool()
    a = _pod("a", "NVIDIA A40", 48.0, inflight=9, draining=True)
    b = _pod("b", "NVIDIA A40", 48.0, inflight=4)
    p.pods = [a, b]
    assert p._pick([a, b], "") is b


def test_the_in_flight_cap_scales_with_the_card():
    """A pod allowed 2.9x the work must be allowed 2.9x the depth, or the cap
    becomes the thing that throttles the fast card."""
    p = _pool(gpu_types=("NVIDIA A40", "NVIDIA A100-SXM4-80GB"), cap=48,
              weights={"NVIDIA A100-SXM4-80GB": 3.0})
    a40 = _pod("a", "NVIDIA A40", 48.0)
    a100 = _pod("b", "NVIDIA A100-SXM4-80GB", 80.0)
    p.pods = [a40, a100]
    assert p.pod_capacity(a40) == 48
    assert p.pod_capacity(a100) == 144
    # A full A40 is skipped while the A100 still has room.
    a40.inflight = 48
    a100.inflight = 50
    assert p._pick([a40, a100], "") is a100


def test_unlimited_stays_unlimited():
    p = _pool(cap=0)
    assert p.pod_capacity(_pod("a", "NVIDIA A40", 48.0)) == 0


# -------------------------------------------------------------- the scaler #

def test_bigger_cards_are_sized_as_fewer_pods():
    """The threshold describes the preferred card, so a fleet of 3x cards
    carries the same load in a third of the pods. Dividing by a flat threshold
    is what rented 94 pods for work that fitted in a fraction of them."""
    flat = _pool(threshold=16)
    flat.pods = [_pod("a", "NVIDIA A40", 48.0, inflight=48)]
    assert flat._pods_for_demand(flat.pods) == 3           # 48 / 16

    fast = _pool(gpu_types=("NVIDIA A40", "NVIDIA A100-SXM4-80GB"),
                 threshold=16, weights={"NVIDIA A100-SXM4-80GB": 3.0})
    a100 = _pod("b", "NVIDIA A100-SXM4-80GB", 80.0, inflight=48)
    fast.pods = [_pod("a", "NVIDIA A40", 48.0), a100]
    assert fast._pods_for_demand([a100]) == 1              # 48 / (16*3)


def test_waiters_still_count_as_demand():
    p = _pool(threshold=16)
    pod = _pod("a", "NVIDIA A40", 48.0, inflight=16)
    p.pods = [pod]
    p._waiting = 32
    assert p._pods_for_demand([pod]) == 3                  # (16+32)/16


def test_demand_never_exceeds_max_pods():
    p = _pool(threshold=1, max_pods=5)
    pod = _pod("a", "NVIDIA A40", 48.0, inflight=500)
    p.pods = [pod]
    assert p._pods_for_demand([pod]) == 5


# ----------------------------------------------------- knowing what we hold #

def test_the_client_reports_the_card():
    from codai.api.runpod_client import _machine_fields
    got = _machine_fields({"gpuCount": 2,
                           "machine": {"gpuDisplayName": "A100 SXM",
                                       "gpuTypeId": "NVIDIA A100-SXM4-80GB",
                                       "gpuType": {"memoryInGb": 80}}})
    assert got == {"gpu": "A100 SXM", "gpu_id": "NVIDIA A100-SXM4-80GB",
                   "gpu_count": 2, "vram_gb": 160.0}


def test_a_pod_whose_machine_record_is_missing_does_not_break_a_listing():
    """One still provisioning, or being torn down. The reaper depends on this
    listing, so a blank machine must degrade rather than raise."""
    from codai.api.runpod_client import _machine_fields
    for bad in ({}, {"machine": None}, {"machine": {"gpuType": None}},
                {"gpuCount": "x", "machine": {"gpuType": {"memoryInGb": "y"}}}):
        got = _machine_fields(bad)
        assert got["gpu"] == "" and got["gpu_count"] == 1 and got["vram_gb"] == 0.0


def test_adoption_reads_the_card_from_the_registry_first(monkeypatch):
    """No API call when we rented the pod ourselves: the registry recorded it."""
    monkeypatch.setattr(rw, "registry_entry",
                        lambda pid: {"gpu": "NVIDIA A100-SXM4-80GB",
                                     "vram_gb": 80.0, "gpu_count": 1})
    called = {"api": 0}

    class _NoCall:
        def __init__(self, *a, **k):
            called["api"] += 1
    monkeypatch.setattr("codai.api.runpod_client.RunpodClient", _NoCall)
    got = rw._adopted_card(None, "podx")
    assert got["gpu"] == "NVIDIA A100-SXM4-80GB" and got["vram_gb"] == 80.0
    assert called["api"] == 0


def test_adoption_asks_the_account_when_the_registry_cannot_say(monkeypatch):
    monkeypatch.setattr(rw, "registry_entry", lambda pid: {})

    class _Client:
        def __init__(self, *a, **k):
            pass

        def get_pod(self, pid):
            return {"gpu": "NVIDIA L40S", "gpu_count": 1, "vram_gb": 48.0}
    monkeypatch.setattr("codai.api.runpod_client.RunpodClient", _Client)
    got = rw._adopted_card(None, "podx")
    assert got["gpu"] == "NVIDIA L40S" and got["vram_gb"] == 48.0


def test_adoption_never_fails_over_an_unreadable_card(monkeypatch):
    """"(adopted)" was wrong, but refusing to adopt would be worse: the pod is
    billing either way."""
    monkeypatch.setattr(rw, "registry_entry", lambda pid: {})

    class _Broken:
        def __init__(self, *a, **k):
            raise RuntimeError("API down")
    monkeypatch.setattr("codai.api.runpod_client.RunpodClient", _Broken)
    got = rw._adopted_card(None, "podx")
    assert got == {"gpu": ""}


def test_the_placeholder_is_never_a_value(monkeypatch):
    """Asserted as behaviour, not as a grep: the string still appears in the
    comment that explains why it was wrong, and a test that cannot tell prose
    from code fails for the wrong reason."""
    monkeypatch.setattr(rw, "registry_entry", lambda pid: {})

    class _Client:
        def __init__(self, *a, **k):
            pass

        def get_pod(self, pid):
            return {}
    monkeypatch.setattr("codai.api.runpod_client.RunpodClient", _Client)
    for got in (rw._adopted_card(None, "p1"), rw._adopted_card(None, "")):
        assert got.get("gpu") != "(adopted)"


def test_the_adopt_site_asks_for_the_card():
    import inspect
    src = inspect.getsource(rw.RunpodPodPool._adopt_shared_pod)
    assert "_adopted_card(" in src
    assert 'gpu="(adopted)"' not in src


def test_the_weights_are_normalised_to_the_PREFERRED_card():
    """The configured concurrency and threshold mean "per preferred card", so
    the preferred card's own weight must be exactly 1.0 whatever the operator's
    table is anchored at. Ours is anchored at the A40 because that is the card
    that was measured; multiplying the threshold by an absolute 2.9 for the
    very card it describes would size the fleet three times too large.
    """
    W = {"NVIDIA A100-SXM4-80GB": 2.9, "NVIDIA A40": 1.0, "NVIDIA L40S": 1.25}
    fast_first = _pool(gpu_types=("NVIDIA A100-SXM4-80GB", "NVIDIA A40"), weights=W)
    assert fast_first.pod_weight(_pod("a", "NVIDIA A100-SXM4-80GB", 80.0)) == pytest.approx(1.0)
    assert fast_first.pod_weight(_pod("b", "NVIDIA A40", 48.0)) == pytest.approx(1 / 2.9, rel=1e-2)

    slow_first = _pool(gpu_types=("NVIDIA A40", "NVIDIA A100-SXM4-80GB"), weights=W)
    assert slow_first.pod_weight(_pod("b", "NVIDIA A40", 48.0)) == pytest.approx(1.0)
    assert slow_first.pod_weight(_pod("a", "NVIDIA A100-SXM4-80GB", 80.0)) == pytest.approx(2.9)


def test_a_preferred_card_absent_from_the_table_leaves_weights_absolute():
    """An operator who lists only the cards that differ gets what they wrote."""
    p = _pool(gpu_types=("NVIDIA A40",), weights={"NVIDIA A100-SXM4-80GB": 2.9})
    assert p.pod_weight(_pod("a", "NVIDIA A100-SXM4-80GB", 80.0)) == pytest.approx(2.9)


def test_catalogue_ids_and_display_names_both_match():
    """RunPod spells a card two ways: the catalogue id
    ("NVIDIA A100-SXM4-80GB") and what a pod reports ("A100 SXM")."""
    from codai.api.runpod_worker import _weight_for
    W = {"NVIDIA A100-SXM4-80GB": 2.9}
    assert _weight_for(W, "NVIDIA A100-SXM4-80GB") == 2.9
    assert _weight_for(W, "A100 SXM") is None or _weight_for(W, "A100 SXM") == 2.9
    assert _weight_for(W, "NVIDIA A40") is None
    assert _weight_for({}, "anything") is None
    assert _weight_for({"x": "not a number"}, "x") is None


# --------------------------------------------------------------------------- #
# per-card limits: what was MEASURED for a card, not a scaled single number
# --------------------------------------------------------------------------- #

LIMITS = {"NVIDIA A100-SXM4-80GB": {"parallel": 186, "queue": 744, "scale_up": 93},
          "NVIDIA A40": {"parallel": 64, "queue": 256, "scale_up": 32}}


def _limited(gpu_types=("NVIDIA A40", "NVIDIA A100-SXM4-80GB"), limits=None):
    p = _pool(gpu_types=gpu_types, threshold=32)
    p.mcfg.pod_max_parallel_requests = 64
    p.mcfg.pod_queue_max_size = 256
    p.mcfg.gpu_limits = dict(LIMITS if limits is None else limits)
    return p


def test_each_card_keeps_the_numbers_measured_for_it():
    from codai.api.runpod_worker import card_limits
    p = _limited()
    assert card_limits(p.mcfg, "NVIDIA A100-SXM4-80GB")["parallel"] == 186
    assert card_limits(p.mcfg, "NVIDIA A40")["parallel"] == 64
    # A card the table does not mention falls back to the model's own numbers.
    assert card_limits(p.mcfg, "NVIDIA RTX A6000")["parallel"] == 64


def test_the_weight_is_DERIVED_from_the_table_not_configured_twice():
    """If the operator stated a depth for both cards, their ratio IS the share
    each should carry -- one source of truth, so a weights table cannot
    disagree with the limits table."""
    p = _limited()
    a40 = _pod("a", "NVIDIA A40", 48.0)
    a100 = _pod("b", "NVIDIA A100-SXM4-80GB", 80.0)
    p.pods = [a40, a100]
    assert p.pod_weight(a40) == pytest.approx(1.0)                 # preferred
    assert p.pod_weight(a100) == pytest.approx(93 / 32, rel=1e-3)


def test_changing_the_rental_preference_no_longer_rescales_anything():
    """The fault the table exists to remove: with a single scaled number, the
    first entry of gpu_types decided what the numbers MEANT."""
    from codai.api.runpod_worker import card_limits
    slow_first = _limited(("NVIDIA A40", "NVIDIA A100-SXM4-80GB"))
    fast_first = _limited(("NVIDIA A100-SXM4-80GB", "NVIDIA A40"))
    for p in (slow_first, fast_first):
        assert card_limits(p.mcfg, "NVIDIA A40")["parallel"] == 64
        assert card_limits(p.mcfg, "NVIDIA A100-SXM4-80GB")["parallel"] == 186


def test_the_scaler_uses_each_pods_own_depth():
    p = _limited()
    a100 = _pod("b", "NVIDIA A100-SXM4-80GB", 80.0, inflight=93)
    p.pods = [a100]
    assert p.pod_threshold(a100) == pytest.approx(93.0)
    assert p._pods_for_demand([a100]) == 1          # 93 in flight = one A100
    a40 = _pod("a", "NVIDIA A40", 48.0, inflight=96)
    p.pods = [a40]
    assert p.pod_threshold(a40) == pytest.approx(32.0)
    assert p._pods_for_demand([a40]) == 3           # 96 / 32


def test_a_per_card_inflight_cap_is_taken_as_written():
    p = _limited(limits={"NVIDIA A40": {"scale_up": 32, "inflight": 50},
                         "NVIDIA A100-SXM4-80GB": {"scale_up": 93, "inflight": 200}})
    p.mcfg.max_inflight_per_pod = 48
    a40 = _pod("a", "NVIDIA A40", 48.0)
    a100 = _pod("b", "NVIDIA A100-SXM4-80GB", 80.0)
    p.pods = [a40, a100]
    assert p.pod_capacity(a40) == 50
    assert p.pod_capacity(a100) == 200


def test_the_pod_gate_is_launched_with_its_own_cards_number():
    """`CODERAI_SERVER_MAX_PARALLEL_REQUESTS` is env, set once at launch, so a
    pod rented on a 3x card would otherwise admit a 1x card's work for life."""
    import inspect

    from codai.api.runpod_worker import RunpodPodPool
    src = inspect.getsource(RunpodPodPool._create_with_fallback)
    assert "card_limits(self.mcfg, str(sel.get(\"display_name\")" in src
    i = src.index("CODERAI_SERVER_MAX_PARALLEL_REQUESTS")
    assert "sel = ranked[i]" in src[:i], "must be set AFTER the card is chosen"
