# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

""""Is guidance on?" has two answers, and they differ at exactly 1.0.

Stable Diffusion and most of diffusers treat ``guidance_scale > 1`` as guidance
on: at 1 the formula ``uncond + 1 * (cond - uncond)`` is just ``cond``.
Z-Image's own property is ``> 0``.

Z-Image-Turbo runs at guidance 1.0, so the caller said "no CFG" and the
pipeline said "CFG". The text-encoder-offload path asked ``encode_prompt`` for
no guidance and got ``negative_prompt_embeds = []``; ``__call__`` then repeated
the latents to batch 2 and handed the transformer ONE caption for TWO images.
diffusers pairs them with ``zip``, which truncates in silence, and the request
died three frames away on

    assert len(size) == bsz
    AssertionError

raised with no message at all. Measured 2026-11-01; 500 to the client, twice.
"""

import pytest

from codai.models.utils import pipeline_wants_cfg


class _SDStyle:
    """Guidance on above 1.0 -- the diffusers majority."""

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1


class _ZImageStyle:
    """Guidance on above 0 -- Z-Image."""

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 0


class _NoProperty:
    """A pipeline that does not publish a rule."""


class _Raises:
    @property
    def do_classifier_free_guidance(self):
        raise RuntimeError("not ready")


# --------------------------------------------------------------------------- #
#  the measured failure
# --------------------------------------------------------------------------- #

def test_z_image_at_guidance_one_wants_cfg():
    """THE case. `> 1.0` answered False here and cost the request."""
    assert pipeline_wants_cfg(_ZImageStyle(), 1.0) is True


def test_sd_at_guidance_one_does_not():
    """And the same scale on an SD-style pipeline is still off, so this cannot
    be fixed by simply lowering the constant everywhere."""
    assert pipeline_wants_cfg(_SDStyle(), 1.0) is False


@pytest.mark.parametrize("gs,sd,zi", [
    (0.0, False, False),
    (0.5, False, True),
    (1.0, False, True),
    (1.5, True, True),
    (7.5, True, True),
])
def test_each_family_is_asked_its_own_rule(gs, sd, zi):
    assert pipeline_wants_cfg(_SDStyle(), gs) is sd
    assert pipeline_wants_cfg(_ZImageStyle(), gs) is zi


# --------------------------------------------------------------------------- #
#  it must not disturb the pipeline it interrogates
# --------------------------------------------------------------------------- #

def test_the_probe_leaves_no_guidance_scale_behind():
    """__call__ sets _guidance_scale itself. Leaving ours on the object would
    hand the next request a scale nobody asked for."""
    pipe = _ZImageStyle()
    pipeline_wants_cfg(pipe, 3.0)
    assert not hasattr(pipe, "_guidance_scale")


def test_an_existing_guidance_scale_is_restored():
    pipe = _ZImageStyle()
    pipe._guidance_scale = 9.0
    pipeline_wants_cfg(pipe, 1.0)
    assert pipe._guidance_scale == 9.0


# --------------------------------------------------------------------------- #
#  the fallback errs toward a surplus negative, never a missing one
# --------------------------------------------------------------------------- #

def test_a_pipeline_with_no_rule_falls_back_above_zero():
    """A surplus negative embedding is wasted work on a pipeline that will not
    use it. A missing one is the crash above. Only one of those loses the
    request, so the fallback is `> 0`, not `> 1.0`."""
    assert pipeline_wants_cfg(_NoProperty(), 1.0) is True
    assert pipeline_wants_cfg(_NoProperty(), 0.0) is False


def test_a_property_that_raises_falls_back_rather_than_propagating():
    assert pipeline_wants_cfg(_Raises(), 1.0) is True


@pytest.mark.parametrize("gs", [None, "", "abc", object()])
def test_an_unreadable_scale_is_not_guidance(gs):
    assert pipeline_wants_cfg(_ZImageStyle(), gs) is False


# --------------------------------------------------------------------------- #
#  both callers go through it
# --------------------------------------------------------------------------- #

def test_the_offload_path_asks_the_pipeline():
    src = open("codai/models/text_encoder_offload.py").read()
    assert "pipeline_wants_cfg(" in src
    assert "float(gs) > 1.0" not in src, "the SD constant is what broke this"


def test_the_generic_encode_path_asks_too():
    """Same mistake, second route: a pipeline reached through the ordinary
    encode_prompt path would disagree in exactly the same way."""
    src = open("codai/api/images.py").read()
    assert "model_utils.pipeline_wants_cfg(pipeline, cfg_scale)" in src
