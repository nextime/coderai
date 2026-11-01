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
    """`free` is what the card has with this load's reservation EXCLUDED.

    `reserved` is what a caller that forgets exclude_key would see subtracted
    from its own view -- the mistake this fixture exists to catch.
    """

    def __init__(self, need, free, reserved=0.0):
        self._need, self._free, self._reserved = need, free, reserved
        self.exclude_seen = []

    def _get_model_used_vram_gb(self, *a, **kw):
        return self._need

    def _get_free_vram_gb(self, exclude_key: str = "", *a, **kw):
        self.exclude_seen.append(exclude_key)
        return self._free if exclude_key else self._free - self._reserved


@pytest.fixture
def mgr(monkeypatch):
    holder = {}

    def _use(need, free, reserved=0.0):
        import codai.models.manager as M
        holder["m"] = _FakeManager(need, free, reserved)
        monkeypatch.setattr(M, "multi_model_manager", holder["m"])
        return holder["m"]
    _use.last = holder
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


# --------------------------------------------------------------------------- #
#  a load must not be judged against its own reservation
# --------------------------------------------------------------------------- #

def test_the_loads_own_reservation_does_not_count_against_it(mgr):
    """Measured 2026-11-01: "9.7 GB needed against 4.8 GB free" on a card with
    about 14.5 GB free -- 14.5 minus the 9.7 this very load had just reserved.
    A model that fitted comfortably was sent to sequential offload. The same
    mistake is recorded in _get_free_vram_gb's own docstring, which is where
    the exclude_key keyword comes from."""
    m = mgr(9.7, 14.5, reserved=9.7)
    strategy, why = _decide("auto-borderline")
    assert strategy is None, why
    assert "fits" in why
    assert m.exclude_seen and all(k for k in m.exclude_seen), \
        "free VRAM has to be asked with exclude_key set"


def test_the_excluded_key_is_the_one_being_loaded(mgr):
    m = mgr(9.7, 14.5, reserved=9.7)
    _decide()
    assert m.exclude_seen[0] == "image:Krea-2-RealDream"


# --------------------------------------------------------------------------- #
#  pre-quantized components are never sent to offload
# --------------------------------------------------------------------------- #

def test_a_quantized_pipeline_is_never_auto_offloaded(mgr):
    """bitsandbytes kernels take device pointers and an offload hook moves the
    tensors out from under them: the model loads and then fails every request.
    Measured again when this function first sent Krea-2-RealDream (4-bit text
    encoder) to sequential offload and accelerate died on
    `Params4bit.__new__() got an unexpected keyword argument
    '_is_hf_initialized'`."""
    mgr(31.0, 10.0)          # hopelessly over: would otherwise be sequential
    strategy, why = images_mod._auto_offload_strategy(
        "image:X", "repo/x", "X", "auto-borderline", quantized=True)
    assert strategy is None
    assert "pre-quantized" in why
    assert "fail every request" in why


def test_an_unquantized_pipeline_still_offloads_when_it_must(mgr):
    mgr(31.0, 10.0)
    strategy, _ = images_mod._auto_offload_strategy(
        "image:X", "repo/x", "X", "auto-borderline", quantized=False)
    assert strategy == "sequential"


def test_the_quantized_check_comes_before_any_sizing(mgr):
    """It must not depend on a footprint being known: a quantized model with no
    estimate must still not be offloaded."""
    mgr(0.0, 0.0)
    strategy, why = images_mod._auto_offload_strategy(
        "image:X", "repo/x", "X", "auto", quantized=True)
    assert strategy is None
    assert "pre-quantized" in why


def test_the_caller_passes_the_quantization_flags():
    """All three ways a model declares itself quantized."""
    src = open("codai/api/images.py").read()
    i = src.index("_auto_offload_strategy(\n")
    body = src[i:i + 500]
    for flag in ("load_in_4bit", "load_in_8bit", "component_quantization"):
        assert flag in body, flag
