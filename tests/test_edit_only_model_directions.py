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

"""What a client is told about which directions a model generates in.

Four Qwen-Image-Edit siblings were published with ``image_generation`` in their
capabilities and ``modes: null``. diffusers maps no text-to-image class for that
family, and the request path says so plainly:

    501 QwenImageEditPlusPipeline cannot generate from a prompt alone:
        diffusers offers no text-to-image pipeline for this family.

So the runtime was right and the capability list was wrong, which is the wrong
way round: a capability sweep read image_generation, posted a prompt, got the
501 and filed four working edit models as broken.
"""

import io
import os

import pytest

from codai.models.capabilities import detect_model_capabilities
from codai.models.manager import _modes_from_caps

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _src(rel):
    return io.open(os.path.join(_ROOT, rel), encoding="utf-8").read()


# ---------------------------------------------------------------------------
# edit-only vs unified
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "Qwen-Image-Edit-2511",
    "Qwen/Qwen-Image-Edit-2511",
    "Qwen-Image-Edit-Rapid-AIO-v23",
    "FLUX.1-Kontext-dev",
    "black-forest-labs/FLUX.1-Kontext-dev",
    "step1x-edit",
    "firered-image-edit",
])
def test_an_edit_only_family_does_not_claim_text_to_image(name):
    caps = detect_model_capabilities(name).to_list()
    assert "instruction_edit" in caps
    assert "image_to_image" in caps
    assert "image_generation" not in caps, name


@pytest.mark.parametrize("name", [
    "black-forest-labs/FLUX.2-klein-4B",
    "FLUX.2-dev",
    "flux2-pro",
])
def test_a_unified_family_keeps_text_to_image(name):
    """FLUX.2 is one checkpoint that both generates and edits — taking its
    text-to-image away would be the same mistake in the other direction."""
    caps = detect_model_capabilities(name).to_list()
    assert "image_generation" in caps
    assert "instruction_edit" in caps


def test_an_ordinary_generator_is_untouched():
    caps = detect_model_capabilities(
        "stabilityai/stable-diffusion-xl-base-1.0").to_list()
    assert "image_generation" in caps
    assert "instruction_edit" not in caps


# ---------------------------------------------------------------------------
# the modes a client reads
# ---------------------------------------------------------------------------

def test_an_edit_only_model_publishes_i2i_alone():
    assert _modes_from_caps(
        detect_model_capabilities("Qwen-Image-Edit-2511"), "image") == ["i2i"]


def test_a_unified_model_publishes_both_directions():
    assert _modes_from_caps(
        detect_model_capabilities("FLUX.2-klein-4B"), "image") == ["t2i", "i2i"]


def test_an_image_to_video_model_publishes_i2v():
    assert _modes_from_caps(
        detect_model_capabilities("Wan-AI/Wan2.2-I2V-A14B-Diffusers"),
        "video") == ["i2v"]


def test_every_inferred_mode_is_backed_by_a_capability():
    """Inferred, never invented: the point of this fallback is to restate what
    the model already claims, not to add a claim."""
    pairs = {"t2i": "image_generation", "i2v": "image_to_video",
             "t2v": "video_generation", "v2v": "video_to_video"}
    for name in ("Qwen-Image-Edit-2511", "FLUX.2-klein-4B",
                 "stabilityai/stable-diffusion-xl-base-1.0",
                 "Wan-AI/Wan2.2-I2V-A14B-Diffusers", "Lightricks/LTX-2.5"):
        caps = detect_model_capabilities(name)
        flags = set(caps.to_list())
        for mode in (_modes_from_caps(caps) or []):
            if mode in pairs:
                assert pairs[mode] in flags, (name, mode)
            else:
                assert mode == "i2i"
                assert flags & {"image_to_image", "instruction_edit"}


def test_a_model_with_no_generation_capability_gets_no_modes():
    assert _modes_from_caps(detect_model_capabilities("bge-m3"), "embedding") is None
    assert _modes_from_caps([], "") is None
    assert _modes_from_caps(None, "") is None


def test_the_cards_own_modes_always_win():
    """Inference is a fallback. An operator who stated the directions -- for a
    model whose name says nothing, or one deliberately restricted -- keeps
    them."""
    src = _src("codai/models/manager.py")
    head = src.index("modes=((meta.get(\"defaults\") or {}).get(\"modes\")")
    body = src[head:head + 700]
    # The card's three spellings are consulted BEFORE the inference.
    assert body.index("meta.get(\"modes\")") < body.index("_modes_from_caps(")
    assert body.index("meta.get(\"video_subtypes\")") < body.index("_modes_from_caps(")
