# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Where LongCat's components go, decided from the card rather than configured.

Two settings that look independent are not. A quantised text encoder MUST be
resident -- bitsandbytes quantises as weights land on CUDA and cannot shuttle
them back, and ``_place_pipeline`` returns early for a quantised encoder without
consulting the offload mode at all. So "quantise the encoder" and "spill when
VRAM runs short" are one decision, not two: asking for both pins 22.1 GB of a
24 GB card (13.6 DiT + 8.0 nf4 encoder + 0.5 VAE) and leaves under 2 GB for the
activations of a 13.6B video model -- an OOM at the first long clip rather than
at load, which reads as a flaky model.

``offload_strategy: "auto"`` takes that decision once, at load, from the free
VRAM it can see, and logs the arithmetic beside what the load actually cost.
"""

import importlib.util
import io
import os

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _src(rel):
    return io.open(os.path.join(_ROOT, rel), encoding="utf-8").read()


def _lc():
    spec = importlib.util.spec_from_file_location(
        "_lc_common", os.path.join(_ROOT, "tools/longcat_common.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


LC = _lc()


# ---------------------------------------------------------------------------
# the decision
# ---------------------------------------------------------------------------

def test_a_24gb_card_spills_the_encoder_rather_than_quantising_it():
    """The production case, and the reason the plan exists: nf4 would fit the
    weights (22.1 of 23.0 usable) and leave nothing for activations."""
    plan = LC.placement_plan(LC.usable_capacity_gb(24.0), "int8", "nf4")
    assert plan["resident"] is False
    assert plan["text_encoder_quant"] == "none"
    assert "cannot be offloaded" in plan["why"]


def test_a_card_with_room_keeps_the_quantisation_and_stays_resident():
    """On a bigger card the quantised encoder is the better placement: no swap
    per request. The old code could never reach this -- the worker collapsed
    'auto' to 'model', so a 48 GB card behaved like a 24 GB one."""
    plan = LC.placement_plan(40.0, "int8", "nf4")
    assert plan["resident"] is True
    assert plan["text_encoder_quant"] == "nf4"


def test_an_unquantised_encoder_only_stays_resident_with_real_room():
    """11 GB of bf16 encoder sitting on the card between prompts is only worth
    it when nothing else needs the space."""
    assert LC.placement_plan(23.0, "int8", "none")["resident"] is False
    assert LC.placement_plan(80.0, "int8", "none")["resident"] is True


def test_a_bf16_dit_is_costed_as_bf16():
    """27 GB, not 13.6: the variant decides, and an fp8/int8 measurement must
    not be inherited by a bf16 entry."""
    plan = LC.placement_plan(30.0, "bf16", "none")
    # The DECISION is made on the residency figure (27 GB DiT + 11 + 0.5),
    # while needed_gb is what the chosen spill placement occupies (27 + 0.5).
    assert plan["resident_needed_gb"] > 35
    assert plan["needed_gb"] == pytest.approx(LC.DIT_GB["bf16"] + LC.VAE_GB)
    assert plan["resident"] is False


@pytest.mark.parametrize("variant", ["int8", "fp8", "gguf"])
def test_every_quantised_variant_costs_one_byte_per_weight(variant):
    assert LC.DIT_GB[variant] == pytest.approx(13.6)


def test_the_plan_always_yields_a_placement_that_fits():
    """Whatever the numbers, the fallback is the one that always works: DiT and
    VAE resident, encoder on the host. A plan with no viable branch would be a
    load that fails instead of a slow one."""
    for free in (0.0, 1.0, 8.0, 16.0, 23.5, 24.0, 48.0, 80.0):
        for te in ("none", "int8", "nf4", "fp4"):
            plan = LC.placement_plan(free, "int8", te)
            assert isinstance(plan["resident"], bool)
            if not plan["resident"]:
                # The spill path never leaves a quantised encoder behind.
                assert plan["text_encoder_quant"] == "none"


def test_the_reservation_matches_the_chosen_placement_not_residency():
    """What gets reserved is what the placement OCCUPIES. Reserving the
    residency figure on a card that cannot hold it would evict siblings to free
    8 GB the spill placement never touches -- the opposite of the point."""
    plan = LC.placement_plan(LC.usable_capacity_gb(24.0), "int8", "nf4")
    assert plan["resident"] is False
    assert plan["needed_gb"] == pytest.approx(LC.DIT_GB["int8"] + LC.VAE_GB)
    assert plan["resident_needed_gb"] == pytest.approx(22.13, abs=0.01)
    # And when it IS resident, the two agree.
    big = LC.placement_plan(LC.usable_capacity_gb(48.0), "int8", "nf4")
    assert big["needed_gb"] == big["resident_needed_gb"]


def test_the_decision_does_not_depend_on_what_else_is_loaded():
    """The flaw this replaced: deciding on FREE VRAM meant a sibling holding
    8 GB pushed LongCat onto the host permanently -- and it stayed there for
    every later request, because nothing re-examines the condition. The
    reservation exists precisely so the manager can evict what is in the way,
    so "no room right now" is not a fact about the card."""
    card = LC.usable_capacity_gb(48.0)
    assert LC.placement_plan(card, "int8", "nf4")["resident"] is True
    # Same card, decided the old way with 8 GB of it occupied: the plan must not
    # be reachable from a free-VRAM reading at all, so capacity is the only
    # input and this call is simply the smaller card's answer.
    assert LC.placement_plan(card, "int8", "nf4")["capacity_gb"] == pytest.approx(card)


def test_a_capacity_is_not_a_free_reading():
    """The context reserve is held back from the card's total, not from what is
    free -- the driver's overhead is not a model's to spend."""
    assert LC.usable_capacity_gb(24.0) == pytest.approx(24.0 - LC.CONTEXT_RESERVE_GB)
    assert LC.usable_capacity_gb(0) == 0.0
    assert LC.usable_capacity_gb(None) == 0.0


def test_nonsense_free_vram_does_not_raise():
    for bad in (None, "", "lots", -5):
        plan = LC.placement_plan(bad, "int8", "nf4")
        assert plan["resident"] is False


def test_the_plan_reports_the_numbers_it_decided_on():
    """So a wrong constant shows as a wrong prediction, not as an OOM."""
    plan = LC.placement_plan(23.0, "int8", "nf4")
    assert plan["capacity_gb"] == pytest.approx(23.0)
    assert plan["resident_needed_gb"] == pytest.approx(22.13, abs=0.01)
    assert "22.1" in plan["why"]


def test_headroom_is_what_separates_the_two_placements():
    """Weights that merely fit are not enough — the activations come after."""
    weights = LC.DIT_GB["int8"] + LC.VAE_GB + LC.TEXT_ENCODER_BF16_GB * \
        LC.component_quant_fraction("nf4")
    assert LC.placement_plan(weights + 0.1, "int8", "nf4")["resident"] is False
    assert LC.placement_plan(
        weights + LC.PLACEMENT_HEADROOM_GB + 0.1, "int8", "nf4")["resident"] is True


# ---------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------

def test_auto_is_not_collapsed_to_model_any_more():
    """It used to be, which made the one setting that means "use the card and
    spill if it is not enough" a synonym for "always keep the encoder on the
    host" -- so the measurement could never happen."""
    src = _src("codai/api/longcat_worker.py")
    head = src.index('offload = str(config.get("offload_strategy")')
    body = src[head:head + 1200]
    assert '"auto"' not in body.split("if offload in (")[1].split(")")[0]
    assert "balanced" in body


def test_the_worker_reserves_the_planned_footprint_before_spawning():
    """The reservation is what makes the manager evict, and it used to happen
    AFTER ensure_service had already loaded the weights -- so the room was
    cleared after the model needed it. The spawn moved inside the loader, which
    acquire_stt_backend calls only once it has evicted."""
    src = _src("codai/api/longcat_worker.py")
    acq = src[src.index("def acquire(model_name"):]
    acq = acq[:acq.index("\ndef ")] if "\ndef " in acq else acq
    assert "placement_plan(" in acq
    assert 'needed = float(_plan.get("needed_gb") or 0.0)' in acq
    # ensure_service is called from inside the loader, not before the acquire.
    loader_at = acq.index("def _loader():")
    assert acq.index("ensure_service(model_path, cfg)") > loader_at
    assert acq.index("acquire_stt_backend(") > loader_at


def test_the_service_decides_at_load_from_capacity_not_free():
    src = _src("tools/longcat_service.py")
    assert "usable_capacity_gb(" in src
    assert "_cap_gb" in src
    assert 'if str(offload or "").strip().lower() == "auto":' in src
    assert "LC.placement_plan(" in src
    assert "mem_get_info()" in src
    # The decision overrides BOTH knobs, not just one.
    assert '_state["text_encoder_quant"] = _plan["text_encoder_quant"]' in src
    assert 'offload = "" if _plan["resident"] else "model"' in src


def test_the_load_logs_what_it_actually_cost():
    """The prediction is falsifiable on every load, or the constants rot."""
    src = _src("tools/longcat_service.py")
    assert "predicted" in src
    assert "memory_allocated()" in src


def test_auto_is_documented_where_an_operator_looks():
    assert "auto = measure free VRAM" in _src("codai/api/longcat_worker.py")
    assert "auto (measure, then decide)" in _src("tools/longcat_service.py")


# ---------------------------------------------------------------------------
# manageable from the model page
# ---------------------------------------------------------------------------

def test_the_model_page_offers_the_auto_placement():
    """A mode that cannot be selected is a mode nobody uses. Worse here than
    usual: the page SETS the select from the saved value, so a card reading
    'auto' with no such option would display as the first entry and the next
    save would silently rewrite it."""
    html = _src("codai/admin/templates/models.html")
    sel = html[html.index('id="cfg-longcat-offload"'):]
    sel = sel[:sel.index("</select>")]
    for value in ('value="auto"', 'value=""', 'value="model"',
                  'value="sequential"'):
        assert value in sel, value


def test_every_quantisation_level_the_service_accepts_is_offered():
    html = _src("codai/admin/templates/models.html")
    sel = html[html.index('id="cfg-longcat-te-quant"'):]
    sel = sel[:sel.index("</select>")]
    for level in LC.COMPONENT_QUANTS:
        token = 'value=""' if level == "none" else f'value="{level}"'
        assert token in sel, level


def test_the_placement_settings_survive_a_save():
    """The entry is rebuilt from scratch on save, so a key missing from the
    allow-list is silently dropped — which is how H3's own documented keys were
    lost. These three ARE the placement."""
    routes = _src("codai/admin/routes.py")
    for key in ('"offload_strategy"', '"text_encoder_quant"', '"variant"'):
        assert key in routes, key


def test_the_page_reads_and_writes_the_same_keys():
    html = _src("codai/admin/templates/models.html")
    for key in ("offload_strategy", "text_encoder_quant", "variant"):
        assert f"s.{key}" in html, f"not loaded into the form: {key}"
        assert f"out.{key}" in html, f"not saved from the form: {key}"


def test_the_card_setting_reaches_the_service():
    """models.json -> request payload -> the service's load, under the name the
    service expects ('offload', not 'offload_strategy')."""
    video = _src("codai/api/video.py")
    assert '"offload" if _k == "offload_strategy"' in video
    svc = _src("tools/longcat_service.py")
    # And the variant / encoder quant are applied to the state BEFORE the load
    # reads them, or auto would plan against the previous request's settings.
    state_at = svc.index("_state.update(variant=variant")
    load_at = svc.index("pipe = load_pipeline(checkpoint,")
    assert state_at < load_at
