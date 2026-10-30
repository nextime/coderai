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

"""Host-RAM admission control.

The case these cover, measured on a 54 GB host: six cards quantized only their
text encoder and left the transformer at bf16, two video cards asked for a
host-resident offload strategy, and requesting any of them ran the host out of
memory. The kernel killed the engine four times in five hours, so the client
never saw a refusal -- it saw a dead engine, and so did every unrelated request
in flight.

So what is asserted here is not "the estimate is accurate" but the decision:
does a load that cannot fit get refused BEFORE it allocates, with numbers a
caller can act on, and does a load that can fit still proceed.
"""

import pytest

from codai.models import host_ram as H


# ---------------------------------------------------------------------------
# estimate_need
# ---------------------------------------------------------------------------

def test_a_measured_load_is_believed_over_any_estimate():
    """The loader records what a load really parked on the host. Once that
    exists there is nothing left to infer."""
    need = H.estimate_need(total_need_gb=40.0, free_vram_gb=0.0,
                           measured_ram_gb=3.5)
    assert need.basis == "measured"
    assert need.gb == pytest.approx(3.5 + H.STAGING_GB)
    assert "previous load" in need.detail


def test_a_zero_measurement_is_not_a_measurement():
    """0.0 means the model held nothing on the host OR that nothing was
    recorded, and the two are indistinguishable — so fall through to the
    estimate rather than admit a 40 GB load on the strength of a zero."""
    need = H.estimate_need(total_need_gb=40.0, free_vram_gb=0.0,
                           measured_ram_gb=0.0)
    assert need.basis == "spill"
    assert need.gb == pytest.approx(40.0 + H.STAGING_GB)


@pytest.mark.parametrize("strategy", sorted(H.HOST_RESIDENT_STRATEGIES))
def test_a_host_resident_strategy_costs_the_whole_footprint(strategy):
    """offload_strategy='model' is a request to hold the weights in host RAM.
    The cost is the model, not the part that missed the GPU — and crediting it
    with free VRAM it will not use is how LongCat-Video was admitted onto a
    host that could not hold it."""
    need = H.estimate_need(total_need_gb=27.0, free_vram_gb=24.0,
                           offload_strategy=strategy)
    assert need.basis == "host-resident"
    assert need.gb == pytest.approx(27.0 + H.STAGING_GB)
    assert strategy in need.detail


def test_a_gpu_only_load_is_charged_only_for_staging():
    """Strategy 'none' keeps every weight on the card, so a model too big for it
    fails on VRAM — loudly and recoverably — and never reaches host RAM."""
    for kwargs in ({"offload_strategy": "none"}, {"no_ram": True}):
        need = H.estimate_need(total_need_gb=60.0, free_vram_gb=1.0, **kwargs)
        assert need.basis == "gpu-only"
        assert need.gb == pytest.approx(H.STAGING_GB)


def test_the_spill_is_what_did_not_fit_on_the_card():
    need = H.estimate_need(total_need_gb=17.6, free_vram_gb=5.6)
    assert need.basis == "spill"
    assert need.gb == pytest.approx(12.0 + H.STAGING_GB)


def test_a_model_that_fits_on_the_card_spills_nothing():
    need = H.estimate_need(total_need_gb=8.0, free_vram_gb=24.0)
    assert need.gb == pytest.approx(H.STAGING_GB)


def test_an_unknown_footprint_is_never_charged_a_number_nobody_has():
    """A size that could not be determined must not become a refusal. The
    protection is lost for that model; the model still loads."""
    need = H.estimate_need(total_need_gb=0.0, free_vram_gb=0.0)
    assert need.basis == "unknown"
    assert need.gb == pytest.approx(H.STAGING_GB)


def test_free_vram_is_never_credited_as_negative_space():
    need = H.estimate_need(total_need_gb=10.0, free_vram_gb=-5.0)
    assert need.gb == pytest.approx(10.0 + H.STAGING_GB)


# ---------------------------------------------------------------------------
# budget
# ---------------------------------------------------------------------------

def test_the_kernel_binds_when_no_cap_is_configured():
    bud = H.budget(kernel_available_gb=20.0, cap_gb=None, floor_gb=4.0)
    assert bud.binding == "kernel"
    assert bud.gb == pytest.approx(16.0)


def test_the_cap_binds_when_it_is_tighter_than_the_kernel():
    """A host with free RAM is still not free to use it past the server's own
    ceiling — that ceiling exists so the sibling engines keep theirs."""
    bud = H.budget(kernel_available_gb=30.0, cap_gb=43.0, fleet_held_gb=36.0,
                   floor_gb=4.0)
    assert bud.binding == "cap"
    assert bud.gb == pytest.approx(3.0)
    assert "43 GB server RAM cap" in bud.detail


def test_the_kernel_still_binds_when_it_is_the_tighter_of_the_two():
    bud = H.budget(kernel_available_gb=6.0, cap_gb=43.0, fleet_held_gb=10.0,
                   floor_gb=4.0)
    assert bud.binding == "kernel"
    assert bud.gb == pytest.approx(2.0)


def test_a_fleet_over_its_cap_has_no_budget_rather_than_a_negative_one():
    bud = H.budget(kernel_available_gb=8.0, cap_gb=43.0, fleet_held_gb=50.0,
                   floor_gb=4.0)
    assert bud.gb == 0.0
    assert bud.binding == "floor"


def test_the_floor_is_held_back_from_whichever_bound_won():
    assert H.budget(kernel_available_gb=10.0, floor_gb=0.0).gb == pytest.approx(10.0)
    assert H.budget(kernel_available_gb=10.0, floor_gb=4.0).gb == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# verdict + the refusal itself
# ---------------------------------------------------------------------------

def test_a_load_that_fits_is_admitted():
    need = H.estimate_need(total_need_gb=8.0, free_vram_gb=24.0)
    bud = H.budget(kernel_available_gb=20.0, floor_gb=4.0)
    assert H.verdict(need, bud).ok


def test_the_production_case_is_refused_and_says_by_how_much():
    """Z-Image-Turbo bf16 (17.6 GB) onto a nearly-full card, on the fleet state
    that was live when the engine was killed."""
    need = H.estimate_need(total_need_gb=17.6, free_vram_gb=2.0)
    bud = H.budget(kernel_available_gb=12.0, cap_gb=43.0, fleet_held_gb=40.0,
                   floor_gb=4.0)
    v = H.verdict(need, bud)
    assert not v.ok
    assert v.shortfall_gb == pytest.approx(17.6)


def test_the_refusal_carries_the_numbers_not_just_prose():
    """An API layer must be able to report the figures without re-deriving
    them, and a test must be able to assert on them without matching prose."""
    need = H.estimate_need(total_need_gb=17.6, free_vram_gb=2.0)
    bud = H.budget(kernel_available_gb=12.0, cap_gb=43.0, fleet_held_gb=40.0)
    exc = H.HostRamExhausted("Tongyi-MAI/Z-Image-Turbo", need, bud)
    assert exc.model == "Tongyi-MAI/Z-Image-Turbo"
    assert exc.need.gb == pytest.approx(17.6)
    assert exc.budget.gb == 0.0
    assert exc.shortfall_gb == pytest.approx(17.6)
    assert isinstance(exc, RuntimeError)   # callers that catch broadly still see it


def test_the_refusal_explains_that_proceeding_kills_the_engine():
    """The message is the whole point: before this existed the same condition
    produced a dropped connection and a respawning engine."""
    need = H.estimate_need(total_need_gb=30.0, free_vram_gb=0.0)
    bud = H.budget(kernel_available_gb=5.0)
    msg = H.HostRamExhausted("m", need, bud).message()
    assert "host RAM" in msg and "kill the engine" in msg


# ---------------------------------------------------------------------------
# remedy_for — the fix differs by what the card actually says
# ---------------------------------------------------------------------------

def test_a_text_encoder_only_card_is_told_to_quantize_its_transformer():
    """The exact shape of the six cards that caused this: quantization present,
    covering the small half."""
    r = H.remedy_for({"component_quantization": {"text_encoder": "4bit"}}, "spill")
    assert "transformer" in r and "text_encoder" in r


def test_a_card_that_already_quantizes_its_transformer_gets_no_such_advice():
    for comp in ({"transformer": "4bit"}, {"unet": "4bit"},
                 {"transformer": "4bit", "text_encoder": "4bit"}):
        assert "add 'transformer'" not in H.remedy_for(
            {"component_quantization": comp}, "spill")


def test_a_host_resident_card_is_told_about_its_strategy():
    r = H.remedy_for({"offload_strategy": "model"}, "host-resident")
    assert "offload_strategy='model'" in r


def test_an_unquantized_card_is_told_it_is_unquantized():
    r = H.remedy_for({"offload_strategy": "auto"}, "spill")
    assert "no quantization" in r


def test_a_quantized_card_that_merely_does_not_fit_is_not_lectured():
    """Advice that does not apply is worse than none: it sends the operator to
    change a setting that is already right."""
    assert H.remedy_for({"load_in_4bit": True, "offload_strategy": "auto"},
                        "spill") == ""


def test_no_card_at_all_yields_no_advice():
    assert H.remedy_for(None) == ""
    assert H.remedy_for({}) == ""


# ---------------------------------------------------------------------------
# the manager gate
# ---------------------------------------------------------------------------

@pytest.fixture
def gate(monkeypatch):
    """The real method, on a manager whose measurements are ours."""
    from codai.models.manager import multi_model_manager as m
    monkeypatch.setattr(m, "models", {}, raising=False)
    monkeypatch.setattr(m, "config", {}, raising=False)
    monkeypatch.setattr(m, "_ram_admission_enabled", staticmethod(lambda: True))
    monkeypatch.setattr(m, "_ram_admission_floor_gb", staticmethod(lambda: 4.0))
    monkeypatch.setattr(m, "_evict_idle_on_ram_enabled", staticmethod(lambda: False))
    monkeypatch.setattr(m, "_get_free_vram_gb", lambda *a, **k: 2.0)
    monkeypatch.setattr(m, "_get_model_used_vram_gb", lambda *a, **k: 17.6)
    monkeypatch.setattr(m, "_ram_cap_gb", staticmethod(lambda: 43.0))
    monkeypatch.setattr(m, "_get_process_ram_gb", staticmethod(lambda: 40.0))
    return m


def test_the_gate_refuses_the_load_that_killed_the_engine(gate, monkeypatch):
    monkeypatch.setattr("codai.models.proc_mem.host_available_gb", lambda: 12.0)
    gate.config["image:Z"] = {"component_quantization": {"text_encoder": "4bit"}}
    with pytest.raises(H.HostRamExhausted) as ei:
        gate.ensure_host_ram_for("image:Z", "Tongyi-MAI/Z-Image-Turbo", "Z-Image-Turbo")
    # Named by what the caller ASKED for, not by the internal key.
    assert ei.value.model == "Z-Image-Turbo"
    assert "add 'transformer'" in ei.value.remedy


def test_the_gate_lets_a_load_through_when_there_is_room(gate, monkeypatch):
    monkeypatch.setattr("codai.models.proc_mem.host_available_gb", lambda: 80.0)
    monkeypatch.setattr(gate, "_ram_cap_gb", staticmethod(lambda: None))
    gate.ensure_host_ram_for("image:Z", "Tongyi-MAI/Z-Image-Turbo")


def test_a_resident_model_is_never_refused(gate, monkeypatch):
    """It is not about to allocate anything — and asking "is there room for it?"
    while it IS the memory is the self-eviction bug the VRAM path already had."""
    monkeypatch.setattr("codai.models.proc_mem.host_available_gb", lambda: 0.0)
    for key in ("image:Z", "Tongyi-MAI/Z-Image-Turbo", "Z-Image-Turbo"):
        gate.models.clear()
        gate.models[key] = object()
        gate.ensure_host_ram_for("image:Z", "Tongyi-MAI/Z-Image-Turbo", "Z-Image-Turbo")


def test_turning_admission_off_restores_the_old_behaviour(gate, monkeypatch):
    """An operator who would rather take the risk keeps the right to."""
    monkeypatch.setattr("codai.models.proc_mem.host_available_gb", lambda: 0.0)
    monkeypatch.setattr(gate, "_ram_admission_enabled", staticmethod(lambda: False))
    gate.ensure_host_ram_for("image:Z", "Tongyi-MAI/Z-Image-Turbo")


def test_an_unsizeable_model_is_admitted_rather_than_blocked(gate, monkeypatch):
    monkeypatch.setattr("codai.models.proc_mem.host_available_gb", lambda: 7.0)
    monkeypatch.setattr(gate, "_get_model_used_vram_gb", lambda *a, **k: 0.0)
    gate.ensure_host_ram_for("image:Z", "Tongyi-MAI/Z-Image-Turbo")


def test_the_gate_frees_host_ram_before_it_refuses(gate, monkeypatch):
    """Refusal is the last step, not the first: the same ladder RAM pressure
    already uses runs first, and a load that fits afterwards is admitted."""
    freed = []
    avail = [6.0]
    monkeypatch.setattr("codai.models.proc_mem.host_available_gb", lambda: avail[0])
    monkeypatch.setattr(gate, "_evict_idle_on_ram_enabled", staticmethod(lambda: True))
    monkeypatch.setattr(gate, "_ram_cap_gb", staticmethod(lambda: None))
    monkeypatch.setattr(gate, "_get_model_used_vram_gb", lambda *a, **k: 10.0)

    def _unload(idle_seconds):
        freed.append(idle_seconds)
        avail[0] = 40.0

    monkeypatch.setattr(gate, "unload_idle_models", _unload)
    gate.ensure_host_ram_for("image:Z", "Tongyi-MAI/Z-Image-Turbo")
    assert freed, "eviction was never attempted before deciding"
    assert freed[0] >= 1.0, (
        "an idle threshold near zero pulls a model that is merely between its "
        "own requests")


# ---------------------------------------------------------------------------
# wiring — the gate has to be ON the load paths, not merely importable
# ---------------------------------------------------------------------------

def _src(rel):
    import io
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return io.open(os.path.join(root, rel), encoding="utf-8").read()


def test_every_vram_admission_point_also_admits_on_host_ram():
    """The two checks answer the same question about different memory. A load
    path that reserves VRAM and not host RAM is exactly the gap that let the
    kernel decide."""
    src = _src("codai/models/manager.py")
    body = src.split("def request_model(", 1)[1] if "def request_model(" in src else src
    assert body.count("self.ensure_vram_for(") == body.count("self.ensure_host_ram_for(")


def test_the_image_path_admits_on_host_ram():
    """It does its own eviction instead of going through request_model, so it
    needs its own call — and it is the path the failing cards are on."""
    assert "ensure_host_ram_for(" in _src("codai/api/images.py")


def test_the_refusal_is_reported_as_503_and_not_a_500():
    app_src = _src("codai/api/app.py")
    assert "HostRamExhausted" in app_src
    assert "status_code=503" in app_src.split("HostRamExhausted", 1)[1]


def test_the_setting_reaches_the_running_server():
    """A config field nothing plumbs is a field that does nothing: the cap it
    guards is read live from global args."""
    assert "ram_admission" in _src("codai/config.py")
    assert "global_args.ram_admission" in _src("codai/main.py")
    assert "ram_admission" in _src("codai/admin/routes.py")


# ---------------------------------------------------------------------------
# the auto-offload sizing step that was skipping itself
# ---------------------------------------------------------------------------

def test_the_kv_types_are_read_before_the_offload_sizing_uses_them():
    """The regression that took the host down three times in five hours.

    ``_kv_quantized = bool(_ck or _cv)`` sat in the auto-offload sizing block,
    while ``_ck``/``_cv`` were bound ~110 lines LATER, where llama_kwargs is
    assembled. Reading them first raised UnboundLocalError on every load that
    reached the branch; the handler printed "(auto-offload sizing skipped: …)",
    which reads like a decision; nothing reduced n_gpu_layers; and llama.cpp
    answered a 35B that did not fit with a 43,562 MiB CPU KV buffer.

    Asserted on source ORDER rather than line numbers, so moving the block does
    not break the test -- only reintroducing the bug does.
    """
    src = _src("codai/backends/vulkan.py")
    use = src.index("_kv_quantized = bool(_ck or _cv)")
    first_bind = min(src.index("_ck = kwargs.get("),
                     src.index("_cv = kwargs.get("))
    assert first_bind < use, (
        "_ck/_cv are used by the auto-offload sizing before they are bound — "
        "UnboundLocalError, swallowed as 'sizing skipped', model loads unsized")


def test_a_failed_sizing_step_does_not_read_as_a_completed_one():
    """'skipped' invites the reader to assume nothing needed doing. The message
    has to say the load is unsized, or the next person reads past it again."""
    src = _src("codai/backends/vulkan.py")
    assert "auto-offload sizing FAILED" in src
    assert "this load is UNSIZED" in src
    # And the consequence has to be in the printed line itself, not only in a
    # comment next to it: a print is what the operator sees.
    printed = [ln for ln in src.splitlines()
               if "auto-offload sizing FAILED" in ln and not ln.strip().startswith("#")]
    assert printed, "the warning is not in a print, only in a comment"
