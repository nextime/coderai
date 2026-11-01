# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""A 2-tuple from ``encode_prompt`` is (pos, neg) for some pipelines and
(embeds, mask) for others, and nothing about its LENGTH says which.

MOSS-Image and Kromoss return ``(embeds, mask)`` and name the parameter
``prompt_attention_mask``. The disambiguation looked for one name --
``prompt_embeds_mask``, the Krea 2 / Qwen-Image spelling -- so MOSS fell
through to the SD 1.x branch and its boolean ``[B, S]`` mask was passed as
``negative_prompt_embeds``. Every MOSS generation then failed with

    Prompt embeddings must be [B, S, C] with matching batch size.

which is true, and about an argument the caller never meant to send.
"""

import pytest

from codai.api import images as images_mod


class FakeTensor:
    """Only what the check reads: ndim, shape and dtype."""

    def __init__(self, shape, dtype="torch.bfloat16"):
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self.dtype = dtype


# --------------------------------------------------------------------------- #
#  by name, when the signature says so
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("param", ["prompt_embeds_mask", "prompt_attention_mask"])
def test_either_mask_parameter_name_is_recognised(param):
    """Krea 2 / Qwen-Image spell it one way, MOSS the other. Only the first was
    known, which is the whole bug."""
    enc = (FakeTensor([1, 77, 2048]), FakeTensor([1, 77], "torch.bool"))
    assert images_mod._second_is_a_mask(enc, {"prompt", param}) is True


def test_sd1x_two_tuple_is_a_negative_not_a_mask():
    """(prompt_embeds, negative_prompt_embeds): same rank, same width."""
    enc = (FakeTensor([1, 77, 768]), FakeTensor([1, 77, 768]))
    assert images_mod._second_is_a_mask(enc, {"prompt", "negative_prompt"}) is False


# --------------------------------------------------------------------------- #
#  by shape, when the signature does not
# --------------------------------------------------------------------------- #

def test_a_boolean_second_element_is_a_mask_whatever_it_is_called():
    """MOSS's mask is bool. A negative embedding never is."""
    enc = (FakeTensor([1, 1024, 2560]), FakeTensor([1, 1024], "torch.bool"))
    assert images_mod._second_is_a_mask(enc, {"prompt"}) is True


def test_one_rank_short_is_a_mask():
    """[B, S] against [B, S, C], with a float dtype so only the rank tells."""
    enc = (FakeTensor([1, 1024, 2560]), FakeTensor([1, 1024], "torch.float32"))
    assert images_mod._second_is_a_mask(enc, {"prompt"}) is True


def test_a_different_width_at_the_same_rank_is_a_mask():
    enc = (FakeTensor([1, 77, 2048]), FakeTensor([1, 77, 1]))
    assert images_mod._second_is_a_mask(enc, {"prompt"}) is True


def test_identical_shapes_are_a_negative():
    enc = (FakeTensor([2, 333, 4096]), FakeTensor([2, 333, 4096]))
    assert images_mod._second_is_a_mask(enc, {"prompt"}) is False


# --------------------------------------------------------------------------- #
#  it must never raise on the way past
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("enc", [
    (),
    (FakeTensor([1, 77, 768]),),
    ("not a tensor", "nor this"),
    (None, None),
])
def test_anything_unreadable_falls_through_rather_than_raising(enc):
    """This runs on the generation path of every image request; a surprise here
    must not become the error the caller sees."""
    assert images_mod._second_is_a_mask(enc, {"prompt"}) is False


def test_the_call_site_uses_the_helper_and_not_the_single_name():
    src = open("codai/api/images.py").read()
    assert "_second_is_a_mask(enc, _call_params)" in src
    assert "len(enc) == 2 and 'prompt_embeds_mask' in _call_params" not in src, \
        "the one-name check is what let MOSS through"


def test_a_mask_means_the_pipeline_encodes_for_itself():
    """The outcome for a mask pipeline is no embeds at all -- the pipeline's own
    encoder runs, as Z-Image already does. Passing embeds plus a mask we do not
    understand is how this broke."""
    src = open("codai/api/images.py").read()
    head = src.index("_second_is_a_mask(enc, _call_params)")
    body = src[head:head + 1200]
    assert "embed_kwargs = {}" in body
