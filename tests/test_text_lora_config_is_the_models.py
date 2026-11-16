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


BLOCK = {"enabled": True, "min_vram_gb": 48, "cloud_type": "SECURE",
         "image": "ghcr.io/nextime/coderai-trainer-text:v0.27.1",
         "pods": 4, "rounds": 6, "max_pod_hourly_usd": 0.90,
         "max_fleet_hourly_usd": 3.60, "container_disk_gb": 150}


# ------------------------------------------------------------ the parsing ----

def test_a_models_block_becomes_the_configuration():
    c = flt.parse_model_lora_train(BLOCK)
    assert c.enabled and c.min_vram_gb == 48 and c.pods == 4
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
    assert c.pods == 1 and c.min_vram_gb == 48


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
    for forbidden in ("min_vram_gb", "gpu_types", "cloud_type", "image",
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


# ------------------------------------------------- the Models page edits it --
#
# "It should be a configuration in the model page like all the others" -- so it
# has to survive the round trip through that page, and the admin save path is a
# WHITELIST: a key not on it is dropped silently, which is the precise shape of
# a knob that appears to save and does not.

def test_the_admin_save_path_accepts_the_block():
    """It is a whitelist. Without the key, every field on the page would look
    like it saved, reload as empty, and no error would appear anywhere."""
    import pathlib
    src = (pathlib.Path(loras.__file__).parent.parent / "admin" / "routes.py").read_text()
    assert '"lora_train",' in src


def _models_html():
    import pathlib
    return (pathlib.Path(loras.__file__).parent.parent / "admin" / "templates"
            / "models.html").read_text()


def _lt_fields():
    """The entries of _LT_FIELDS only -- the page has other arrays of its own,
    and a regex over the whole template matches them too."""
    import re
    html = _models_html()
    block = html[html.index("const _LT_FIELDS = ["):]
    block = block[:block.index("];")]
    return re.findall(r"\['([a-z-]+)',\s*'([a-z_]+)',", block)


def test_the_models_page_can_read_every_field_it_can_write():
    """A knob the form can SET and cannot READ is indistinguishable from one
    that did not save. Both directions come from one field list."""
    import re
    html = _models_html()
    assert "_populateLoraTrain" in html and "_collectLoraTrain" in html
    assert "lora_train: _collectLoraTrain()" in html, "never written on save"
    ids = set(re.findall(r'id="cfg-lt-([a-z-]+)"', html))
    listed = {sfx for sfx, _key in _lt_fields()}
    assert listed, "the field list is empty"
    assert listed <= ids, f"no input on the page for {sorted(listed - ids)}"


def test_every_configurable_field_is_a_real_config_field():
    """A page offering a knob the parser ignores is worse than no knob."""
    import dataclasses
    keys = {key for _sfx, key in _lt_fields()}
    known = {f.name for f in dataclasses.fields(flt.LoraTrainConfig)}
    assert keys and keys <= known, f"not in LoraTrainConfig: {sorted(keys - known)}"


def test_the_money_fields_are_on_the_page():
    """The whole point of moving this out of the request: an operator can see
    and set the ceiling."""
    keys = {key for _sfx, key in _lt_fields()}
    assert {"max_pod_hourly_usd", "max_fleet_hourly_usd", "min_vram_gb",
            "image", "pods"} <= keys


def test_an_off_block_is_absent_rather_than_full_of_defaults():
    """An absent block is how a model says 'not trainable on rented GPUs'.
    Writing one full of zeros would make every model look configured."""
    assert "if (!enabled) return null;" in _models_html()


# ------------------------------------------------- choosing a card by VRAM ---

class _Catalogue:
    def list_gpu_types(self):
        return [{"id": "NVIDIA A40", "memory_gb": 48, "secure_price": 0.44,
                 "community_price": 0.35},
                {"id": "NVIDIA L40S", "memory_gb": 48, "secure_price": 0.86},
                {"id": "NVIDIA RTX 6000 Ada", "memory_gb": 48, "secure_price": 0.77},
                {"id": "NVIDIA H100 PCIe", "memory_gb": 80, "secure_price": 2.69},
                {"id": "NVIDIA RTX 4090", "memory_gb": 24, "secure_price": 0.34},
                {"id": "NVIDIA A100 40GB", "memory_gb": 40, "secure_price": 0.80},
                {"id": "NVIDIA UNPRICED", "memory_gb": 48}]


def _cards(**kw):
    cfg = flt.LoraTrainConfig(**{"min_vram_gb": 48, **kw})
    return flt.eligible_cards(_Catalogue(), cfg,
                              ceiling=kw.pop("ceiling", 0.90)
                              if "ceiling" in kw else 0.90)


def test_a_vram_floor_excludes_smaller_cards():
    ids = [c["id"] for c in _cards()]
    assert "NVIDIA RTX 4090" not in ids, "24 GB is under the floor"
    assert "NVIDIA A100 40GB" not in ids, "40 GB is under a 48 GB floor"


def test_the_price_ceiling_excludes_the_expensive_ones():
    ids = [c["id"] for c in _cards()]
    assert "NVIDIA H100 PCIe" not in ids, "$2.69 is over a $0.90 ceiling"


def test_the_cheapest_eligible_card_is_first():
    """Ranked, not picked: capacity moves, and the next-cheapest card is a
    better answer than a failed job."""
    cards = _cards()
    assert [c["id"] for c in cards] == ["NVIDIA A40", "NVIDIA RTX 6000 Ada",
                                        "NVIDIA L40S"]
    assert cards[0]["hourly_usd"] == 0.44


def test_an_unpriced_card_is_excluded_when_a_ceiling_is_set():
    """Renting a card the catalogue cannot price, under a price ceiling, is
    renting blind."""
    assert "NVIDIA UNPRICED" not in [c["id"] for c in _cards()]


def test_an_allow_list_narrows_but_cannot_widen():
    cfg = flt.LoraTrainConfig(min_vram_gb=48,
                              gpu_types="NVIDIA L40S, NVIDIA RTX 4090, NVIDIA H100 PCIe")
    ids = [c["id"] for c in flt.eligible_cards(_Catalogue(), cfg, ceiling=0.90)]
    assert ids == ["NVIDIA L40S"], "the 4090 is under the floor, the H100 over the ceiling"


def test_community_pricing_is_used_when_asked_for():
    cfg = flt.LoraTrainConfig(min_vram_gb=48, cloud_type="COMMUNITY")
    cards = flt.eligible_cards(_Catalogue(), cfg, ceiling=0.90)
    assert cards[0]["id"] == "NVIDIA A40" and cards[0]["hourly_usd"] == 0.35


def test_no_eligible_card_is_an_empty_list_not_a_guess():
    cfg = flt.LoraTrainConfig(min_vram_gb=200)
    assert flt.eligible_cards(_Catalogue(), cfg, ceiling=0.90) == []
