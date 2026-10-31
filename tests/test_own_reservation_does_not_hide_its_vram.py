"""An eviction freed 16.3 GB and the load that triggered it saw 0.0 GB.

Measured on zeiss 2026-10-30:

    23:58:05  Loading 'image:…pornmaster…': need 30.9 GB VRAM, have 2.9 GB free
              — evicting models to make room
    23:58:05  On-request VRAM eviction: unloading active model
              'image:black-forest-labs/FLUX.2-klein-4B' (measured 17.24 GB)
    23:59:23  ⚠ Eviction freed 13.4 GB (now 16.3 GB free, needed 30.9 GB)
    23:59:25  needs ~32.9 GB host RAM (30.9 GB footprint against 0.0 GB free
              VRAM spills 30.9 GB to the host, plus 2.0 GB to stage it), and
              only 27.8 GB is available — short by 5.1 GB. Refusing the load.

Two seconds after eviction reported 16.3 GB free, admission read **0.0**. No
other load intervened — the next one starts a second later.

`_get_free_vram_gb()` subtracted every in-flight load reservation, including the
reservation belonging to the very load being judged:

    _reserved = sum(self._loading_reservations.values())
    return max(0.0, total_free - _reserved)      # max(0, 16.3 - 30.9) = 0.0

A reservation exists so OTHER loads do not count VRAM already spoken for.
Subtracted from the reserving load's own view it says the room it just reserved
does not exist — so the entire footprint looks like a host spill, the host-RAM
budget refuses it, and the eviction that preceded it cost another model its
residency and bought nothing.

Note what the fix is NOT. "Check admission before evicting" would have been
worse: before the eviction there were 2.9 GB free, so it would have refused
sooner and still pointlessly. Eviction HELPS the host-RAM verdict, by shrinking
the spill — it just has to be visible to the load that paid for it.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codai.models import host_ram
from codai.models.manager import MultiModelManager


def _mgr(reservations, free_gb):
    m = MultiModelManager.__new__(MultiModelManager)
    m._loading_reservations = dict(reservations)
    m._cross_gpu_pooling_enabled = lambda: False
    import codai.models.gpu_query as gq
    gq.visible_gpu_memory = lambda: [{"free": free_gb * 1e9}]
    return m


# ── the accounting ───────────────────────────────────────────────────────────

def test_a_loads_own_reservation_is_not_charged_against_it():
    m = _mgr({"image:me": 30.9}, 16.3)
    assert MultiModelManager._get_free_vram_gb(m, exclude_key="image:me") == pytest.approx(16.3, abs=0.05)


def test_without_the_exclusion_it_sees_nothing():
    """The behaviour being fixed, pinned so it cannot come back."""
    m = _mgr({"image:me": 30.9}, 16.3)
    assert MultiModelManager._get_free_vram_gb(m) == 0.0


def test_other_loads_reservations_are_still_charged():
    """The reservation's whole purpose: a sibling load must not count VRAM this
    one has already claimed."""
    m = _mgr({"image:me": 30.9, "image:sibling": 2.0}, 16.3)
    assert MultiModelManager._get_free_vram_gb(m, exclude_key="image:me") == pytest.approx(14.3, abs=0.05)


def test_an_unknown_key_excludes_nothing():
    m = _mgr({"image:a": 5.0}, 10.0)
    assert MultiModelManager._get_free_vram_gb(m, exclude_key="image:not-loading") == pytest.approx(5.0, abs=0.05)


def test_the_default_is_unchanged_for_every_other_caller():
    """Eviction, sizing and placement all call this with no argument and must
    keep seeing the conservative figure."""
    m = _mgr({"image:a": 4.0}, 10.0)
    assert MultiModelManager._get_free_vram_gb(m) == pytest.approx(6.0, abs=0.05)


def test_it_never_goes_negative():
    m = _mgr({"image:a": 99.0}, 10.0)
    assert MultiModelManager._get_free_vram_gb(m) == 0.0


def test_no_reservations_at_all():
    m = _mgr({}, 12.5)
    assert MultiModelManager._get_free_vram_gb(m, exclude_key="anything") == pytest.approx(12.5, abs=0.05)


# ── the incident, reproduced end to end ──────────────────────────────────────

def _admission(free_vram_gb, budget_gb=27.8, footprint_gb=30.9):
    need = host_ram.estimate_need(total_need_gb=footprint_gb,
                                  free_vram_gb=free_vram_gb,
                                  measured_ram_gb=None,
                                  offload_strategy="auto", no_ram=False)
    return need, host_ram.verdict(need, host_ram.Budget(budget_gb, "cap", ""))


def test_the_old_view_reproduces_the_refusal_exactly():
    """32.9 GB needed and 5.1 GB short are the numbers the server printed. If
    this stops matching, the model of the bug is wrong."""
    need, v = _admission(0.0)
    assert need.gb == pytest.approx(32.9, abs=0.1)
    assert v.ok is False
    assert v.shortfall_gb == pytest.approx(5.1, abs=0.1)


def test_the_corrected_view_admits_the_same_load():
    need, v = _admission(16.3)
    assert need.gb < 20.0, need
    assert v.ok is True


def test_checking_admission_before_evicting_would_have_been_worse():
    """The fix that was proposed first, and why it was abandoned: with the
    pre-eviction 2.9 GB free the load is refused even harder, so moving the
    check earlier would have made the outcome worse while still wasting
    nothing — the eviction simply would not have happened, and neither would
    the load."""
    _, before = _admission(2.9)
    _, after = _admission(16.3)
    assert before.ok is False
    assert after.ok is True


def test_eviction_shrinks_the_host_need_rather_than_being_irrelevant():
    """Monotonic: more free VRAM, less spill, smaller host requirement. This is
    why eviction is worth doing even for a model larger than the card."""
    needs = [_admission(f)[0].gb for f in (0.0, 8.0, 16.3, 24.0)]
    assert needs == sorted(needs, reverse=True), needs


# ── wiring ───────────────────────────────────────────────────────────────────

def test_the_admission_path_passes_its_own_key():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(root, "codai/models/manager.py")).read()
    i = src.index("def ensure_host_ram_for")
    body = src[i:i + 4000]
    assert "self._get_free_vram_gb(exclude_key=model_key)" in body, (
        "admission is judging a load against a free-VRAM figure that excludes "
        "the VRAM the load itself reserved")


def test_the_reason_is_recorded_where_the_subtraction_happens():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(root, "codai/models/manager.py")).read()
    i = src.index("def _get_free_vram_gb")
    doc = src[i:i + 2200]
    assert "16.3" in doc and "0.0 GB free" in doc
    assert "exclude_key" in doc
