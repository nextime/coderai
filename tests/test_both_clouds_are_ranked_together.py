"""Renting across SECURE and COMMUNITY, ranked by what each actually costs.

A ceiling of $0.90 on >=48GB cards found no capacity twice in ten minutes:
the market was tight for secure, and community -- cheaper and more available
for the same card -- was not being looked at, because `cloud_type` was one
value reduced to a boolean.

The cloud now belongs to the CANDIDATE. The same card appears once per cloud at
that cloud's price, and the create call uses the cloud the offer was priced in:
renting a community-priced offer on secure charges the secure rate, which is
how a price ceiling is quietly exceeded.
"""

import pytest

from codai.api import text_lora_fleet as flt


class FakeCatalogue:
    def __init__(self, rows):
        self.rows = rows

    def list_gpu_types(self):
        return self.rows


CARDS = [
    {"id": "A40", "memory_gb": 48, "secure_price": 0.59, "community_price": 0.39},
    {"id": "L40S", "memory_gb": 48, "secure_price": 0.99, "community_price": 0.69},
    {"id": "RTX2000", "memory_gb": 16, "secure_price": 0.24, "community_price": 0.18},
]


# ------------------------------------------------------- which clouds apply --

@pytest.mark.parametrize("raw,want", [
    ("SECURE", ["SECURE"]),
    ("COMMUNITY", ["COMMUNITY"]),
    ("ANY", ["SECURE", "COMMUNITY"]),
    ("BOTH", ["SECURE", "COMMUNITY"]),
    ("secure,community", ["SECURE", "COMMUNITY"]),
    ("", ["SECURE"]),
])
def test_cloud_list(raw, want):
    assert flt.clouds_for(flt.LoraTrainConfig(cloud_type=raw)) == want


def test_a_single_cloud_stays_single():
    """An install that chose SECURE deliberately must not silently gain
    community hosts for a dataset of court documents."""
    cfg = flt.LoraTrainConfig(cloud_type="SECURE", min_vram_gb=48)
    cards = flt.eligible_cards(FakeCatalogue(CARDS), cfg, ceiling=0.90)
    assert {c["cloud_type"] for c in cards} == {"SECURE"}


# ------------------------------------------------------------- the ranking --

def test_both_clouds_are_offered_cheapest_first():
    cfg = flt.LoraTrainConfig(cloud_type="ANY", min_vram_gb=48)
    cards = flt.eligible_cards(FakeCatalogue(CARDS), cfg, ceiling=0.90)
    assert [(c["id"], c["cloud_type"], c["hourly_usd"]) for c in cards] == [
        ("A40", "COMMUNITY", 0.39),
        ("A40", "SECURE", 0.59),
        ("L40S", "COMMUNITY", 0.69),
    ]


def test_the_ceiling_binds_per_cloud():
    """The secure L40S at $0.99 is over the ceiling while the same card on
    community at $0.69 is not -- excluding the card entirely would throw away
    capacity, and including it would overspend."""
    cfg = flt.LoraTrainConfig(cloud_type="ANY", min_vram_gb=48)
    cards = flt.eligible_cards(FakeCatalogue(CARDS), cfg, ceiling=0.90)
    assert ("L40S", "SECURE") not in [(c["id"], c["cloud_type"]) for c in cards]
    assert ("L40S", "COMMUNITY") in [(c["id"], c["cloud_type"]) for c in cards]


def test_the_vram_floor_still_excludes_a_small_card_in_both_clouds():
    cfg = flt.LoraTrainConfig(cloud_type="ANY", min_vram_gb=48)
    cards = flt.eligible_cards(FakeCatalogue(CARDS), cfg, ceiling=0.90)
    assert "RTX2000" not in {c["id"] for c in cards}


def test_a_tie_goes_to_secure():
    rows = [{"id": "A40", "memory_gb": 48, "secure_price": 0.50,
             "community_price": 0.50}]
    cfg = flt.LoraTrainConfig(cloud_type="ANY", min_vram_gb=48)
    cards = flt.eligible_cards(FakeCatalogue(rows), cfg, ceiling=0.90)
    assert cards[0]["cloud_type"] == "SECURE"


def test_an_unpriced_card_is_excluded_under_a_ceiling():
    rows = [{"id": "H100", "memory_gb": 80}]
    cfg = flt.LoraTrainConfig(cloud_type="ANY", min_vram_gb=48)
    assert flt.eligible_cards(FakeCatalogue(rows), cfg, ceiling=0.90) == []


def test_the_description_names_the_cloud():
    """The same card at two prices is two different rentals; a line that does
    not say which cloud cannot be checked against the bill."""
    cfg = flt.LoraTrainConfig(cloud_type="ANY", min_vram_gb=48)
    cards = flt.eligible_cards(FakeCatalogue(CARDS), cfg, ceiling=0.90)
    line = flt.describe_cards(cards)
    assert "community" in line and "secure" in line


# ------------------------------------------- the cloud reaches the rental ---

def test_each_candidate_is_rented_in_its_own_cloud():
    """The whole point: the create call must use the offer's cloud, not one
    global value, or the price ceiling was checked against a different rate
    than the one charged."""
    seen = []

    class Client:
        def create_pod(self, **kw):
            seen.append((kw.get("gpu_type_id"), kw.get("cloud_type")))
            raise RuntimeError("no capacity")

    spec = flt.TrainerPodSpec(image="img", gpu_types=["A40", "A40", "L40S"],
                              gpu_clouds=["COMMUNITY", "SECURE", "COMMUNITY"],
                              cloud_type="SECURE")
    fleet = flt.TrainerFleet(Client(), "job-x", spec)
    assert fleet._rent_one(0) is None
    assert seen == [("A40", "COMMUNITY"), ("A40", "SECURE"),
                    ("L40S", "COMMUNITY")]


def test_a_missing_cloud_list_falls_back_to_the_spec():
    """Backwards compatible: an older caller passing only gpu_types still
    rents in the spec's single cloud."""
    seen = []

    class Client:
        def create_pod(self, **kw):
            seen.append(kw.get("cloud_type"))
            raise RuntimeError("nope")

    spec = flt.TrainerPodSpec(image="img", gpu_types=["A40"],
                              cloud_type="COMMUNITY")
    flt.TrainerFleet(Client(), "job-y", spec)._rent_one(0)
    assert seen == ["COMMUNITY"]
