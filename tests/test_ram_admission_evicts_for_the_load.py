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

"""Host-RAM admission must evict toward the load, not toward the cap.

Ground truth, 2026-11-01, a 54 GB host with a 43 GB server RAM cap and five
engines, on the first style video of a 576-case run:

    23:40:36  Loading 'video:LTX-2.3-10Eros': needs ~26.2 GB host RAM (24.2 GB
              measured on host RAM on a previous load, plus 2.0 GB to stage
              it), budget is 13.3 GB (17.3 GB left under the 43 GB server RAM
              cap, with the fleet holding 25.7 GB, less 4.0 GB kept free for
              the OS and the other engines) -- freeing host RAM before deciding
    23:40:37  503

One second, and no model unloaded. The eviction target was ``cap * 0.85`` =
36.55 GB while the fleet held 25.7 GB, and ``_evict_models_for_ram`` returns at
its first guard when the reading is already under its target. So the rung of the
ladder that exists to make room could not make room in the one situation that
needs it: a model whose host footprint is large compared with the cap, which is
every model that does not fit the card. The remaining 90 style videos of that
run were abandoned because all of them are in that class.

The arithmetic that matters: a load fits under the cap when
``held <= cap - floor - need``. That is the target to evict toward, and nothing
else is.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codai.models import host_ram as H
from codai.models import manager as M
from codai.models.manager import MultiModelManager


CAP = 43.0
FLOOR = 4.0
# LTX-2.3-10Eros, as the log reports it: 24.2 GB measured on the host, and
# STAGING_GB on top, which is what made the need 26.2 GB.
MEASURED = 24.2
NEED = MEASURED + H.STAGING_GB


@pytest.fixture(autouse=True)
def _no_real_trim(monkeypatch):
    """The trim rung is not under test and gc.collect() over a test process is
    pure latency."""
    monkeypatch.setattr(M, "_trim_cpu_ram", lambda: None)


def _manager(held, *, cap=CAP, floor=FLOOR, measured=MEASURED,
             kernel=50.0, frees=0.0, evict_enabled=True):
    """A manager whose host-RAM surroundings are all dials.

    ``frees`` is how much an eviction pass actually hands back, so a test can
    model both the fleet that can be brought down and the fleet that cannot.
    """
    m = MultiModelManager.__new__(MultiModelManager)
    m.models = {}
    m.config = {"video:LTX-2.3-10Eros": {"measured_ram_gb": measured}}
    state = {"held": float(held), "targets": [], "idle": []}

    m._ram_admission_enabled = lambda: True
    m._get_model_used_vram_gb = lambda *a, **k: 26.1
    m._get_free_vram_gb = lambda **k: 0.0
    m._ram_cap_gb = lambda: cap
    m._ram_admission_floor_gb = lambda: floor
    m._evict_idle_on_ram_enabled = lambda: evict_enabled
    m._get_process_ram_gb = lambda: state["held"]
    m._host_ram_budget = lambda: H.budget(
        kernel_available_gb=kernel, cap_gb=cap,
        fleet_held_gb=state["held"], floor_gb=floor)

    def _evict(target):
        state["targets"].append(target)
        state["held"] = max(0.0, state["held"] - frees)
    m._evict_models_for_ram = _evict
    m.unload_idle_models = lambda *a: state["idle"].append(a)
    return m, state


def _ensure(m):
    return m.ensure_host_ram_for("video:LTX-2.3-10Eros")


# ---------------------------------------------------------------------------
# the target
# ---------------------------------------------------------------------------

def test_the_eviction_target_is_what_this_load_needs():
    """cap - floor - need: the figure that makes the verdict come out true."""
    m, state = _manager(25.7)
    with pytest.raises(H.HostRamExhausted):
        _ensure(m)
    assert state["targets"] == [pytest.approx(CAP - FLOOR - NEED)]
    assert state["targets"][0] == pytest.approx(12.8, abs=0.01)


def test_the_old_target_would_have_asked_for_more_than_the_fleet_held():
    """The bug, stated as arithmetic: 36.55 GB asked of a fleet holding 25.7,
    which _evict_models_for_ram answers by returning at once."""
    m, state = _manager(25.7)
    with pytest.raises(H.HostRamExhausted):
        _ensure(m)
    assert CAP * 0.85 > 25.7
    assert state["targets"][0] < 25.7, (
        "the target has to be below what the fleet holds or no model is ever "
        "unloaded")


def test_the_target_is_never_looser_than_the_cap_target_it_replaced():
    """A small load must not license a laxer sweep than the old rung did: this
    change may only evict where nothing was evicted before."""
    m, state = _manager(40.0, measured=0.1)
    with pytest.raises(H.HostRamExhausted):
        _ensure(m)
    # cap - floor - 2.1 = 36.9, looser than the old 36.55, so the old target
    # binds and the sweep is exactly the one that ran before.
    assert state["targets"][0] == pytest.approx(CAP * 0.85)


@pytest.mark.parametrize("measured,held", [(24.2, 25.7), (30.0, 20.0),
                                           (12.0, 38.0), (5.0, 41.0)])
def test_whatever_the_sizes_the_target_makes_the_load_fit_under_the_cap(
        measured, held):
    """Reaching the target has to be sufficient, not merely helpful."""
    m, state = _manager(held, measured=measured)
    with pytest.raises(H.HostRamExhausted):
        _ensure(m)
    target = state["targets"][0]
    bud = H.budget(kernel_available_gb=50.0, cap_gb=CAP,
                   fleet_held_gb=target, floor_gb=FLOOR)
    need = H.estimate_need(total_need_gb=26.1, free_vram_gb=0.0,
                           measured_ram_gb=measured)
    assert H.verdict(need, bud).ok


# ---------------------------------------------------------------------------
# what it must not do
# ---------------------------------------------------------------------------

def test_a_load_the_cap_cannot_house_evicts_nothing(capsys):
    """cap - floor < need: emptying the fleet would still refuse, so paying
    every resident model for the attempt buys nothing."""
    m, state = _manager(25.7, cap=20.0)
    with pytest.raises(H.HostRamExhausted):
        _ensure(m)
    assert state["targets"] == []
    assert "cannot house" in capsys.readouterr().out


def test_a_load_that_already_fits_is_decided_without_eviction():
    m, state = _manager(5.0)
    assert _ensure(m) is None
    assert state["targets"] == []
    assert state["idle"] == []


def test_a_resident_model_is_never_refused_and_never_evicts():
    """It is not about to allocate anything."""
    m, state = _manager(42.0)
    m.models = {"video:LTX-2.3-10Eros": object()}
    assert _ensure(m) is None
    assert state["targets"] == []


def test_an_unmeasurable_footprint_proceeds_unguarded(capsys):
    """A load nobody has a size for keeps loading; the protection is what is
    lost, not the request."""
    m, state = _manager(42.0, measured=None)
    m._get_model_used_vram_gb = lambda *a, **k: 0.0
    assert _ensure(m) is None
    assert state["targets"] == []
    assert "cannot be checked" in capsys.readouterr().out


def test_eviction_is_skipped_when_the_operator_turned_it_off():
    m, state = _manager(25.7, evict_enabled=False)
    with pytest.raises(H.HostRamExhausted):
        _ensure(m)
    assert state["targets"] == []
    assert state["idle"] == []


def test_without_a_cap_the_idle_sweep_still_runs():
    """No cap means no target to compute; the kernel figure is the only bound
    and a 30 s idle sweep is what there is to try."""
    m, state = _manager(25.7, cap=None, kernel=10.0)
    with pytest.raises(H.HostRamExhausted):
        _ensure(m)
    assert state["targets"] == []
    assert state["idle"] == [(30.0,)]


# ---------------------------------------------------------------------------
# the outcome the run needed
# ---------------------------------------------------------------------------

def test_a_fleet_that_can_come_down_admits_the_load(capsys):
    """The whole point. 25.7 GB held, 13 GB of it in idle models: evicting them
    leaves the 26.2 GB load inside the cap, and LTX-2.3-10Eros renders."""
    m, state = _manager(25.7, frees=13.0)
    assert _ensure(m) is None
    assert state["targets"] == [pytest.approx(CAP - FLOOR - NEED)]
    assert "load admitted" in capsys.readouterr().out


def test_a_fleet_that_cannot_come_down_still_refuses():
    """Eviction is a rung, not a guarantee: when the held memory is not in
    evictable models the refusal is still the honest answer."""
    m, state = _manager(25.7, frees=0.0)
    with pytest.raises(H.HostRamExhausted) as exc:
        _ensure(m)
    assert state["targets"]
    assert "LTX-2.3-10Eros" in str(exc.value)


def test_the_refusal_still_carries_the_numbers_a_caller_can_act_on():
    m, _ = _manager(25.7)
    with pytest.raises(H.HostRamExhausted) as exc:
        _ensure(m)
    msg = str(exc.value)
    assert "26.2" in msg and "43" in msg
