# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""`auto-borderline` is the DEFAULT, and the image path ignored it.

The offload dropdown promises "Prefer VRAM — try the card, offload only if it
really will not fit (recommended)". The video path honours that contract and so
does the CUDA backend. The image path tested only the explicit strategies::

    use_sequential_offload = bool(
        _mc.get('cpu_offload')
        or (_offload_strategy in ('cpu', 'sequential', 'model', 'disk')))

`auto` and `auto-borderline` are not in that list, so they read as "never
offload" — and since `auto-borderline` is the default, no image model ever
offloaded on overshoot, however large.

Measured 2026-11-01: Krea-2-RealDream, 31.07 GB measured against a 23.56 GB
card (6.07 GB spilled to host RAM by accelerate), loaded whole and died
mid-generation on cuBLAS's workspace allocation. Its configured strategy was
`auto-borderline` the whole time.
"""

import pytest

from codai.api import images as images_mod


SRC = open("codai/api/images.py").read()


class _FakeManager:
    def __init__(self, need, free):
        self._need, self._free = need, free

    def _get_model_used_vram_gb(self, *a, **kw):
        return self._need

    def _get_free_vram_gb(self, *a, **kw):
        return self._free


@pytest.fixture
def mgr(monkeypatch):
    def _use(need, free):
        import codai.models.manager as M
        monkeypatch.setattr(M, "multi_model_manager", _FakeManager(need, free))
    return _use


def _decide(strategy="auto-borderline"):
    return images_mod._auto_offload_strategy(
        "image:Krea-2-RealDream", "unsloth/Krea-2-Turbo",
        "Krea-2-RealDream", strategy)


# --------------------------------------------------------------------------- #
#  the measured case
# --------------------------------------------------------------------------- #

def test_krea_realdream_gets_sequential_offload(mgr):
    """31.07 GB on a 23.56 GB card: not borderline, and its transformer alone
    would not fit either, so model offload cannot help."""
    mgr(31.07, 23.56)
    strategy, why = _decide()
    assert strategy == "sequential"
    assert "largest component" in why


def test_a_model_that_fits_is_left_on_the_card(mgr):
    mgr(14.0, 23.56)
    strategy, why = _decide()
    assert strategy is None
    assert "fits" in why


def test_a_borderline_overshoot_tries_the_card_first(mgr):
    """The whole point of `auto-borderline`: the estimate is often a
    conservative round number, and exiling a model that would have fit costs a
    PCIe trip per request for the life of the install."""
    mgr(25.0, 23.56)
    strategy, why = _decide("auto-borderline")
    assert strategy is None
    assert "borderline" in why


def test_plain_auto_offloads_on_any_overshoot(mgr):
    """`auto` condemns on the estimate; only `auto-borderline` gambles."""
    mgr(25.0, 23.56)
    strategy, _ = _decide("auto")
    assert strategy == "model"


def test_a_moderate_overshoot_uses_model_offload(mgr):
    """Model offload keeps the component in use resident, so it is the fast
    choice whenever that component fits on its own."""
    mgr(26.0, 23.56)
    strategy, why = _decide("auto")
    assert strategy == "model"
    assert "stays on the" in why


# --------------------------------------------------------------------------- #
#  it must never offload on a guess
# --------------------------------------------------------------------------- #

def test_an_explicit_strategy_is_not_touched():
    for explicit in ("sequential", "model", "cpu", "disk", "none", ""):
        assert _decide(explicit) == (None, "")


def test_no_footprint_means_no_decision(mgr):
    """Offloading on a guess costs every request a PCIe trip; the ordinary load
    still falls back on a real OOM."""
    mgr(0.0, 23.56)
    strategy, why = _decide()
    assert strategy is None
    assert "no footprint" in why


def test_unknown_free_vram_means_no_decision(mgr):
    mgr(31.0, 0.0)
    strategy, why = _decide()
    assert strategy is None
    assert "free VRAM unknown" in why


def test_a_broken_manager_does_not_fail_the_load(monkeypatch):
    import codai.models.manager as M

    class _Boom:
        def _get_model_used_vram_gb(self, *a, **kw):
            raise RuntimeError("nope")

        def _get_free_vram_gb(self, *a, **kw):
            return 23.5

    monkeypatch.setattr(M, "multi_model_manager", _Boom())
    strategy, why = _decide()
    assert strategy is None
    assert "sizing unavailable" in why


# --------------------------------------------------------------------------- #
#  wiring
# --------------------------------------------------------------------------- #

def test_the_loader_consults_it():
    assert "_auto_offload_strategy(" in SRC
    i = SRC.index("use_sequential_offload = bool(")
    body = SRC[i:i + 1400]
    assert "_auto_offload_strategy(" in body, \
        "the decision has to happen where offload is decided"


def test_model_offload_escalates_to_sequential_on_the_last_attempt():
    """Retrying an identical placement that already failed just pays for the
    weights again."""
    i = SRC.index("_strategy = (_offload_strategy or '').strip().lower()")
    body = SRC[i:i + 900]
    assert "load_attempt >= 3" in body
    assert "_strategy = 'sequential'" in body
