"""The card, the image and the money are the MODEL's configuration.

`/v1/runpod/scale` states the rule for itself: "there is deliberately no way
for a request to change a model's configuration". That is what makes a renting
route safe to expose -- a caller can start work without being able to pick a
bigger card, a different image, or a higher price than the operator set on the
Models page.

A request may NARROW what is configured: fewer pods, a lower ceiling. Asking
for less than you are allowed is always safe, and the narrowing dies with the
request. Asking for MORE is clamped and said out loud, because it is a client
that does not know the limit rather than an attack -- but it cannot take
effect.
"""

import pytest

from codai.api import loras
from codai.api import text_lora_fleet as flt


BLOCK = {"enabled": True, "gpu_type_id": "NVIDIA A40", "cloud_type": "SECURE",
         "image": "ghcr.io/nextime/coderai-trainer-text:v0.27.1",
         "pods": 4, "rounds": 6, "max_pod_hourly_usd": 0.90,
         "max_fleet_hourly_usd": 3.60, "container_disk_gb": 150}


# ------------------------------------------------------------ the parsing ----

def test_a_models_block_becomes_the_configuration():
    c = flt.parse_model_lora_train(BLOCK)
    assert c.enabled and c.gpu_type_id == "NVIDIA A40" and c.pods == 4
    assert c.max_pod_hourly_usd == 0.90 and c.container_disk_gb == 150
    assert c.image.endswith(":v0.27.1")


def test_a_model_without_the_block_is_not_trainable_on_pods():
    """Silence means off: a model nobody configured for rented training must
    not become trainable on rented GPUs by default."""
    assert flt.parse_model_lora_train(None).enabled is False
    assert flt.parse_model_lora_train({}).enabled is False


def test_an_unknown_key_is_ignored_not_fatal():
    c = flt.parse_model_lora_train({**BLOCK, "from_the_future": True})
    assert c.pods == 4


def test_a_malformed_value_keeps_the_default_rather_than_crashing():
    """An operator typo in one field must not make the model untrainable."""
    c = flt.parse_model_lora_train({**BLOCK, "pods": "four"})
    assert c.pods == 1 and c.gpu_type_id == "NVIDIA A40"


# ----------------------------------------------------------- the narrowing ---

def test_a_request_may_ask_for_fewer_pods():
    c = flt.narrow(flt.parse_model_lora_train(BLOCK), pods=2)
    assert c.pods == 2


def test_a_request_cannot_ask_for_more_pods(capsys):
    c = flt.narrow(flt.parse_model_lora_train(BLOCK), pods=99)
    assert c.pods == 4, "clamped to what the operator permits"
    assert "permits 4" in capsys.readouterr().out


def test_a_request_may_lower_the_price_ceiling():
    c = flt.narrow(flt.parse_model_lora_train(BLOCK), max_pod_hourly_usd=0.40)
    assert c.max_pod_hourly_usd == 0.40


def test_a_request_cannot_raise_the_price_ceiling(capsys):
    """The one that matters: a client must not be able to rent an H100 by
    asking nicely."""
    c = flt.narrow(flt.parse_model_lora_train(BLOCK), max_pod_hourly_usd=9.99)
    assert c.max_pod_hourly_usd == 0.90
    assert "the model allows $0.90" in capsys.readouterr().out


def test_a_ceiling_can_be_introduced_where_the_model_set_none():
    """No configured ceiling means the account caps are the only limit, so a
    request adding one is narrowing, not widening."""
    c = flt.narrow(flt.parse_model_lora_train({**BLOCK, "max_pod_hourly_usd": 0}),
                   max_pod_hourly_usd=0.50)
    assert c.max_pod_hourly_usd == 0.50


def test_rounds_are_duration_not_fleet_size():
    """More rounds costs time, not a bigger fleet or a dearer card, so a
    request may raise them."""
    c = flt.narrow(flt.parse_model_lora_train(BLOCK), rounds=12)
    assert c.rounds == 12 and c.pods == 4


def test_narrowing_does_not_mutate_the_models_configuration():
    base = flt.parse_model_lora_train(BLOCK)
    flt.narrow(base, pods=1, max_pod_hourly_usd=0.10)
    assert base.pods == 4 and base.max_pod_hourly_usd == 0.90


def test_nothing_in_the_request_can_name_a_card_an_image_or_a_url():
    """The structural guarantee: those fields do not exist on the request, so
    no amount of client creativity reaches them."""
    fields = set(loras.TextLoraTrainRequest.model_fields)
    for forbidden in ("gpu_type_id", "cloud_type", "pod_image", "image",
                      "callback_url", "network_volume_id", "container_disk_gb",
                      "pod_max_lifetime_s"):
        assert forbidden not in fields, f"{forbidden} is settable by a request"


def test_the_request_can_still_narrow_the_three_that_are_safe():
    fields = set(loras.TextLoraTrainRequest.model_fields)
    for allowed in ("pods", "rounds", "max_pod_hourly_usd",
                    "max_fleet_hourly_usd", "distributed"):
        assert allowed in fields


def test_asking_for_distributed_on_an_unconfigured_model_says_where_to_set_it():
    """The error has to name the fix, or an operator goes looking in the
    request schema for a field that deliberately is not there."""
    cfg = flt.config_for_model("a-model-with-no-block")
    assert cfg.enabled is False
