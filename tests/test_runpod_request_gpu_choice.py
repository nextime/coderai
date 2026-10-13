"""A request may narrow which cards a lease accepts — never widen.

Picking the card is normally the operator's business: they set the preference,
the cost ceiling, the VRAM floor and the pools. But a caller knows things the
operator cannot — that this batch has to be comparable with yesterday's numbers,
that a measurement on a different card is worthless, that it would rather wait
than pay a fallback's price. Letting it say so costs nothing; letting it WIDEN
would hand it the operator's budget.

The case that prompted this: a pool configured `gpu_types: [NVIDIA A40]` with
`allow_other_gpus: true` fell back to an L40S at $1.09/hr against the A40's
$0.59 when no A40 was free. Correct per the configuration, and useless to a
caller measuring throughput. Before this the only way to get the A40 was to edit
the model's configuration — which changes it for everybody, permanently, for the
sake of one request.

Per MODEL, not per engine: every pool goes through the same resolver, and
`gpu_types`/`allow_other_gpus` are per-model settings already.
"""
import time
import types

import pytest

from codai.api import runpod_worker as rw


def _mcfg(gpu_types=None, allow_other=False, **kw):
    cfg = rw.RunpodModelConfig()
    cfg.gpu_types = list(gpu_types or [])
    cfg.allow_other_gpus = bool(allow_other)
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


# --- narrowing ---------------------------------------------------------------


def test_nothing_requested_leaves_the_operators_choice_alone():
    out = rw.narrow_gpu_choice(_mcfg(["NVIDIA A40"], allow_other=True))
    assert out["gpu_types"] == ["NVIDIA A40"]
    assert out["allow_other_gpus"] is True
    assert out["notes"] == []


def test_a_request_can_pin_one_of_the_allowed_cards():
    out = rw.narrow_gpu_choice(_mcfg(["NVIDIA A40", "NVIDIA L40S"], allow_other=True),
                               gpu_types=["NVIDIA A40"])
    assert out["gpu_types"] == ["NVIDIA A40"]


def test_a_request_can_refuse_the_fallback():
    """The actual fix for the $1.09 L40S: wait for the A40 rather than take
    whatever else fits."""
    out = rw.narrow_gpu_choice(_mcfg(["NVIDIA A40"], allow_other=True),
                               gpu_types=["NVIDIA A40"], allow_other_gpus=False)
    assert out["gpu_types"] == ["NVIDIA A40"]
    assert out["allow_other_gpus"] is False


def test_the_requests_order_is_kept_because_it_is_a_preference():
    out = rw.narrow_gpu_choice(
        _mcfg(["NVIDIA A40", "NVIDIA L40S", "NVIDIA A100"], allow_other=True),
        gpu_types=["NVIDIA A100", "NVIDIA A40"])
    assert out["gpu_types"] == ["NVIDIA A100", "NVIDIA A40"]


def test_a_comma_string_is_accepted():
    out = rw.narrow_gpu_choice(_mcfg(["NVIDIA A40", "NVIDIA L40S"]),
                               gpu_types="NVIDIA L40S, NVIDIA A40")
    assert out["gpu_types"] == ["NVIDIA L40S", "NVIDIA A40"]


def test_case_does_not_matter_but_the_operators_spelling_is_published():
    out = rw.narrow_gpu_choice(_mcfg(["NVIDIA A40"]), gpu_types=["nvidia a40"])
    assert out["gpu_types"] == ["NVIDIA A40"]


# --- what a request must NOT be able to do -----------------------------------


def test_a_card_the_operator_did_not_list_is_dropped_and_reported():
    out = rw.narrow_gpu_choice(_mcfg(["NVIDIA A40"], allow_other=True),
                               gpu_types=["NVIDIA A40", "NVIDIA H100 NVL"])
    assert out["gpu_types"] == ["NVIDIA A40"]
    assert any("H100" in n for n in out["notes"]), out["notes"]


def test_narrowing_to_nothing_allowed_is_refused_not_ignored():
    """Falling back to the operator's default would be the OPPOSITE of what the
    caller asked for, and it would do so silently."""
    with pytest.raises(ValueError) as e:
        rw.narrow_gpu_choice(_mcfg(["NVIDIA A40"]), gpu_types=["NVIDIA H100 NVL"])
    assert "A40" in str(e.value), "the error does not say what IS allowed"


def test_a_request_cannot_turn_the_fallback_on():
    out = rw.narrow_gpu_choice(_mcfg(["NVIDIA A40"], allow_other=False),
                               allow_other_gpus=True)
    assert out["allow_other_gpus"] is False
    assert any("narrow" in n for n in out["notes"]), out["notes"]


def test_with_no_operator_preference_a_request_may_still_pin_a_card():
    """No configured preference means "any card within the limits", so naming
    one is strictly more restrictive — and the ceiling still binds."""
    out = rw.narrow_gpu_choice(_mcfg([], allow_other=True),
                               gpu_types=["NVIDIA A40"])
    assert out["gpu_types"] == ["NVIDIA A40"]


@pytest.mark.parametrize("field", ["max_hourly_usd", "min_vram_gb", "max_pods",
                                   "cloud_types", "allow_spot"])
def test_the_operators_limits_are_not_reachable_from_a_request(field):
    """Only the card choice is negotiable. The cost ceiling, the VRAM floor, the
    pools and the pod cap are the operator's alone."""
    import inspect
    src = inspect.getsource(rw.narrow_gpu_choice)
    body = src.split('"""')[2]  # past the docstring
    assert field not in body, f"narrow_gpu_choice touches {field}"


# --- the lease carries it, and it dies with the lease ------------------------


def _pool(mcfg):
    pool = rw.RunpodPodPool.__new__(rw.RunpodPodPool)
    import threading
    pool._cv = threading.Condition()
    pool.mcfg = mcfg
    pool.model_key = "m"
    pool.pods = []
    pool._client_floor = 0
    pool._client_floor_until = 0.0
    pool._client_floor_logged = -1
    pool._client_gpu_types = []
    pool._client_allow_other = None
    pool._release_pending = False
    return pool


def test_the_lease_reports_what_it_granted():
    pool = _pool(_mcfg(["NVIDIA A40", "NVIDIA L40S"], allow_other=True, max_pods=4))
    out = pool.set_client_floor(1, gpu_types=["NVIDIA A40"],
                                allow_other_gpus=False)
    assert out["granted"] == 1
    assert out["gpu_types"] == ["NVIDIA A40"]
    assert out["allow_other_gpus"] is False


def test_a_dropped_card_is_reported_in_the_answer():
    """A constraint silently dropped is worse than one refused: the caller would
    read the numbers as having come from the card it asked for."""
    pool = _pool(_mcfg(["NVIDIA A40"], allow_other=True, max_pods=4))
    out = pool.set_client_floor(1, gpu_types=["NVIDIA A40", "NVIDIA H100 NVL"])
    assert out["gpu_types"] == ["NVIDIA A40"]
    assert "notes" in out and any("H100" in n for n in out["notes"])


def test_the_narrowing_applies_while_the_lease_lives():
    pool = _pool(_mcfg(["NVIDIA A40", "NVIDIA L40S"], allow_other=True, max_pods=4))
    pool.set_client_floor(1, gpu_types=["NVIDIA A40"], allow_other_gpus=False)
    assert pool.client_gpu_narrow() == {"gpu_types": ["NVIDIA A40"],
                                        "allow_other_gpus": False}


def test_the_narrowing_dies_WITH_the_lease():
    """Otherwise one request's "A40 only" quietly becomes the pool's policy."""
    pool = _pool(_mcfg(["NVIDIA A40", "NVIDIA L40S"], allow_other=True, max_pods=4))
    pool.set_client_floor(1, gpu_types=["NVIDIA A40"], allow_other_gpus=False)
    pool._client_floor_until = time.time() - 1          # lease lapsed
    assert pool.client_gpu_narrow() == {}


def test_releasing_the_lease_releases_its_constraints():
    pool = _pool(_mcfg(["NVIDIA A40", "NVIDIA L40S"], allow_other=True, max_pods=4))
    pool.set_client_floor(1, gpu_types=["NVIDIA A40"], allow_other_gpus=False)
    pool.set_client_floor(0)
    assert pool.client_gpu_narrow() == {}
    assert pool._client_gpu_types == [] and pool._client_allow_other is None


def test_a_plain_lease_constrains_nothing():
    pool = _pool(_mcfg(["NVIDIA A40"], allow_other=True, max_pods=4))
    out = pool.set_client_floor(2)
    assert pool.client_gpu_narrow() == {}
    assert "gpu_types" not in out, "it reported a constraint nobody asked for"


def test_an_impossible_narrowing_raises_from_the_lease_too():
    pool = _pool(_mcfg(["NVIDIA A40"], allow_other=True, max_pods=4))
    with pytest.raises(ValueError):
        pool.set_client_floor(1, gpu_types=["NVIDIA H100 NVL"])


# --- ranking honours it ------------------------------------------------------


CATALOG = [
    {"id": "NVIDIA A40", "display_name": "A40", "memory_gb": 48,
     "secure_price": 0.59, "community_price": 0.4, "spot_price": None},
    {"id": "NVIDIA L40S", "display_name": "L40S", "memory_gb": 48,
     "secure_price": 1.09, "community_price": 0.9, "spot_price": None},
]


def _client():
    return types.SimpleNamespace(list_gpu_types=lambda: list(CATALOG))


def _acct():
    return types.SimpleNamespace(default_gpu_type="")


def test_ranking_without_narrowing_allows_the_fallback():
    cfg = _mcfg(["NVIDIA A40"], allow_other=True, cloud_types=["SECURE"],
                max_hourly_usd=1.2, min_vram_gb=40, gpu_count=1,
                allow_spot=False, gpu_type="")
    ids = [g["gpu_type_id"] for g in rw._rank_gpus(_client(), cfg, _acct())]
    assert "NVIDIA A40" in ids and "NVIDIA L40S" in ids
    assert ids[0] == "NVIDIA A40", "the operator's preference lost its rank"


def test_ranking_with_narrowing_excludes_the_fallback():
    """The point of the whole feature: no L40S on offer, so a capacity miss
    waits for an A40 instead of silently costing twice as much."""
    cfg = _mcfg(["NVIDIA A40"], allow_other=True, cloud_types=["SECURE"],
                max_hourly_usd=1.2, min_vram_gb=40, gpu_count=1,
                allow_spot=False, gpu_type="")
    ranked = rw._rank_gpus(_client(), cfg, _acct(),
                           narrow={"gpu_types": ["NVIDIA A40"],
                                   "allow_other_gpus": False})
    assert [g["gpu_type_id"] for g in ranked] == ["NVIDIA A40"]


def test_narrowing_cannot_defeat_the_cost_ceiling():
    """A request pinning an expensive card it is allowed to name still cannot
    spend past the operator's ceiling."""
    cfg = _mcfg(["NVIDIA A40", "NVIDIA L40S"], allow_other=False,
                cloud_types=["SECURE"], max_hourly_usd=0.80, min_vram_gb=40,
                gpu_count=1, allow_spot=False, gpu_type="")
    from codai.api.runpod_client import RunpodError
    with pytest.raises(RunpodError):
        rw._rank_gpus(_client(), cfg, _acct(),
                      narrow={"gpu_types": ["NVIDIA L40S"],
                              "allow_other_gpus": False})


def test_narrowing_cannot_defeat_the_vram_floor():
    cfg = _mcfg(["NVIDIA A40"], allow_other=False, cloud_types=["SECURE"],
                max_hourly_usd=5.0, min_vram_gb=80, gpu_count=1,
                allow_spot=False, gpu_type="")
    from codai.api.runpod_client import RunpodError
    with pytest.raises(RunpodError):
        rw._rank_gpus(_client(), cfg, _acct(),
                      narrow={"gpu_types": ["NVIDIA A40"],
                              "allow_other_gpus": False})


def test_provisioning_reads_the_narrowing_per_pod():
    """Read per provision, not cached, so a lapsed lease stops constraining the
    very next pod."""
    import inspect
    src = inspect.getsource(rw.RunpodPodPool._provision_one)
    assert "narrow=self.client_gpu_narrow()" in src


# --- the HTTP surface --------------------------------------------------------


def test_the_request_model_accepts_both_fields():
    from codai.api.app import RunpodScaleRequest
    b = RunpodScaleRequest(model="m", pods=1, gpu_types=["NVIDIA A40"],
                           allow_other_gpus=False)
    assert b.gpu_types == ["NVIDIA A40"] and b.allow_other_gpus is False


def test_both_fields_are_optional_and_default_to_the_model():
    """None must mean "use the model's own setting" — False would silently turn
    every caller's fallback off."""
    from codai.api.app import RunpodScaleRequest
    b = RunpodScaleRequest(model="m", pods=1)
    assert b.gpu_types is None and b.allow_other_gpus is None


def test_a_comma_string_is_accepted_over_http_too():
    from codai.api.app import RunpodScaleRequest
    b = RunpodScaleRequest(model="m", pods=1, gpu_types="NVIDIA A40,NVIDIA L40S")
    assert isinstance(b.gpu_types, str)
    assert rw.narrow_gpu_choice(
        _mcfg(["NVIDIA A40", "NVIDIA L40S"]), gpu_types=b.gpu_types
    )["gpu_types"] == ["NVIDIA A40", "NVIDIA L40S"]


def test_an_impossible_narrowing_is_a_400_not_a_500():
    import inspect
    from codai.api import app as app_mod
    src = inspect.getsource(app_mod.runpod_scale)
    assert "except ValueError" in src
    i, j = src.index("except ValueError"), src.index("except Exception")
    assert "400" in src[i:j], "a caller's mistake is reported as a server error"


def test_the_status_page_shows_a_leases_narrowing():
    """A pod on an unexpected card should have a visible cause, not only a bill."""
    import inspect
    src = inspect.getsource(rw.pools_schedule_status)
    assert '"client_gpu_narrow": pool.client_gpu_narrow()' in src
