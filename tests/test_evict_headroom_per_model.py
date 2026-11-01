# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""The eviction headroom pays for a peak that does not scale with the weights.

A fraction of the load is the wrong shape for it. A VAE decode of a 768x1344
image asks for about 1 GB in ONE allocation whatever the checkpoint weighs, so
on a 7.49 GB model the default tenth is 0.75 GB and not enough. Measured
2026-11-01 on CyberRealistic-Pony:

    Loading 'image:cyberdelia/CyberRealisticPony…': need 8.6 GB VRAM, have
    0.0 GB free — evicting models to make room
    On-request VRAM eviction: unloading 'pornmaster-pro-pony…' (7.49 GB)
    …
    OutOfMemoryError: CUDA out of memory. Tried to allocate 1008.00 MiB.
    GPU 0 has a total capacity of 23.56 GiB of which 110.00 MiB is free

Eviction reported success -- correctly, for the WEIGHTS -- and the decode had
110 MiB to work in. So a model may state an absolute floor.
"""

import pytest

from codai.models import manager as M


@pytest.fixture
def mgr():
    m = M.MultiModelManager.__new__(M.MultiModelManager)
    m.config = {}
    return m


def test_the_default_is_the_global_fraction(mgr):
    assert mgr._evict_headroom_gb(10.0) == pytest.approx(
        10.0 * M._EVICT_HEADROOM_FRACTION)


def test_a_model_can_set_an_absolute_floor(mgr):
    """THE case: 7.49 GB of weights, a decode that wants 1008 MiB."""
    mgr.config["image:cyber"] = {"evict_headroom_gb": 1.5}
    assert mgr._evict_headroom_gb(8.6, "image:cyber") == pytest.approx(1.5)
    # ...and the floor only ever raises, never lowers.
    assert mgr._evict_headroom_gb(100.0, "image:cyber") == pytest.approx(
        100.0 * M._EVICT_HEADROOM_FRACTION)


def test_a_model_can_set_its_own_fraction(mgr):
    mgr.config["image:big"] = {"evict_headroom_fraction": 0.25}
    assert mgr._evict_headroom_gb(20.0, "image:big") == pytest.approx(5.0)


def test_the_two_combine_as_a_maximum(mgr):
    """The fraction protects a large load from an estimate landing short; the
    floor protects a small one from a peak that does not scale with it. Neither
    should be able to cancel the other."""
    mgr.config["image:both"] = {"evict_headroom_gb": 1.5,
                                "evict_headroom_fraction": 0.5}
    assert mgr._evict_headroom_gb(2.0, "image:both") == pytest.approx(1.5)
    assert mgr._evict_headroom_gb(20.0, "image:both") == pytest.approx(10.0)


def test_it_reads_a_config_nested_under_raw_cfg(mgr):
    """The shape the loader forwards. Two readers of one config disagreeing on
    this cost an hour earlier today."""
    mgr.config["image:n"] = {"_raw_cfg": {"evict_headroom_gb": 2.0}}
    assert mgr._evict_headroom_gb(5.0, "image:n") == pytest.approx(2.0)


@pytest.mark.parametrize("bad", ["", None, "lots", {}, []])
def test_an_unreadable_value_falls_back_rather_than_raising(mgr, bad):
    """This runs on the eviction path of every load; a bad config value must
    not become the error the caller sees."""
    mgr.config["image:bad"] = {"evict_headroom_gb": bad,
                               "evict_headroom_fraction": bad}
    got = mgr._evict_headroom_gb(10.0, "image:bad")
    assert got == pytest.approx(10.0 * M._EVICT_HEADROOM_FRACTION)


def test_an_unknown_key_is_the_default(mgr):
    assert mgr._evict_headroom_gb(10.0, "image:never-heard-of-it") == \
        pytest.approx(10.0 * M._EVICT_HEADROOM_FRACTION)


@pytest.mark.parametrize("need", [0.0, -5.0, None])
def test_a_missing_need_never_yields_a_negative_headroom(mgr, need):
    assert mgr._evict_headroom_gb(need) >= 0.0


# --------------------------------------------------------------------------- #
#  wiring
# --------------------------------------------------------------------------- #

SRC = open("codai/models/manager.py").read()


def test_eviction_uses_the_resolver_and_not_the_bare_constant():
    i = SRC.index("def _evict_models_for_vram")
    body = SRC[i:SRC.index("\n    def ", i + 10)]
    # `protect` became one-key-or-many on 2026-11-01, so the single-key
    # spelling it falls back to is now _protect_one. The point of the
    # assertion is unchanged: the amount comes from the resolver, keyed by a
    # model, and never from the bare module constant.
    assert "self._evict_headroom_gb(needed_gb, for_key or _protect_one)" in body
    assert "needed_gb * (1.0 + _EVICT_HEADROOM_FRACTION)" not in body, \
        "the fraction-only form is what could not cover a decode"


def test_for_key_falls_back_to_protect():
    """Most callers already name the model they are making room for; only the
    ones that do not need the extra argument."""
    i = SRC.index("def _evict_models_for_vram")
    sig = SRC[i:i + 200]
    assert "for_key: str = \"\"" in sig


def test_the_image_path_names_the_model_it_is_making_room_for():
    img = open("codai/api/images.py").read()
    i = img.index("_mmm._evict_models_for_vram(\n")
    assert "for_key=" in img[i:i + 200]


def test_both_fields_are_savable_from_the_admin_page():
    routes = open("codai/admin/routes.py").read()
    assert '"evict_headroom_gb", "evict_headroom_fraction",' in routes
