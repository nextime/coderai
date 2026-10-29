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

"""A turbo model must not be served at 30 steps and CFG 7.5.

Measured on zeiss, 2026-10-29. The card for `Z-Image-Turbo` carried no
``defaults`` and no ``acceleration`` block, so a request that did not name a
step count got the generic fallback: 30 steps at CFG 7.5, against a checkpoint
distilled for 8 steps at CFG 1. That is about four times the work, and the
picture also comes out over-guided -- a distilled model at CFG 7.5 looks
burnt. The client's own report said "a turbo, few-step model should take
seconds" and measured ~72 s per 768x1344 still.

The order of precedence is the point of these tests: the request wins, then a
fused distill LoRA, then the operator's card, then the model's name, and only
then the generic number. Each step down is a weaker claim about the same
model, and nothing weaker may overrule anything stronger.
"""

import ast
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codai.models.acceleration import distilled_defaults_for

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = io.open(os.path.join(_ROOT, "codai/api/images.py"), encoding="utf-8").read()


def test_the_distilled_families_are_recognised():
    assert distilled_defaults_for("Tongyi-MAI/Z-Image-Turbo")["steps"] == 8
    assert distilled_defaults_for("unsloth/Krea-2-Turbo")["steps"] == 8
    assert distilled_defaults_for("black-forest-labs/FLUX.1-schnell")["steps"] == 4
    assert distilled_defaults_for("ByteDance/SDXL-Lightning")["steps"] == 8
    lcm = distilled_defaults_for("latent-consistency/lcm-lora-sdxl")
    assert (lcm["steps"], lcm["guidance_scale"]) == (6, 1.5), (
        "LCM is the one family that wants guidance above 1")


def test_a_distilled_model_is_never_given_real_cfg():
    for name in ("Tongyi-MAI/Z-Image-Turbo", "stabilityai/sdxl-turbo",
                 "black-forest-labs/FLUX.1-schnell", "ByteDance/Hyper-SD"):
        got = distilled_defaults_for(name)
        assert got is not None, name
        assert got["guidance_scale"] <= 1.5, (
            f"{name} would be served with classifier-free guidance it was "
            f"distilled to do without")


def test_a_non_distilled_model_is_left_alone():
    """A false positive is the dangerous direction: it would serve a full
    model at 8 steps, which produces an unfinished picture."""
    for name in ("Tongyi-MAI/Z-Image", "stabilityai/stable-diffusion-xl-base-1.0",
                 "black-forest-labs/FLUX.2-klein-4B", "krea/Krea-2"):
        assert distilled_defaults_for(name) is None, name


def test_a_word_inside_another_word_does_not_count():
    assert distilled_defaults_for("John6666/turbovision-xl") is None
    assert distilled_defaults_for("some/lcmx-model") is None
    assert distilled_defaults_for("") is None
    assert distilled_defaults_for(None) is None


def test_only_the_last_path_segment_is_read():
    """An org called "turbo-labs" does not make its models distilled."""
    assert distilled_defaults_for("turbo-labs/plain-sdxl-finetune") is None


def test_the_card_and_the_request_both_outrank_the_name():
    """Checked on the source, because the precedence IS the behaviour: the
    name is a guess, and a number an operator or a caller wrote is not."""
    block = _SRC.split("_distilled = None", 1)[1][:2200]
    assert "if not _d_steps and _accel_steps is None:" in block, (
        "the name heuristic runs even when the card declared its own steps")
    # In the expression itself, the distilled value must sit below the request,
    # the fused LoRA and the card.
    expr = block.split("num_steps = ", 1)[1].split("cfg_scale = ", 1)[0]
    assert expr.index("request.steps") < expr.index("_accel_steps") < \
        expr.index("_d_steps") < expr.index('_distilled["steps"]'), (
        f"precedence is wrong in: {expr.strip()!r}")
    assert expr.index('_distilled["steps"]') < expr.index("30 if quality"), (
        "the generic 30-step fallback still wins over the distilled default")


def test_the_choice_is_logged_with_what_it_replaced():
    """An unexplained step count is indistinguishable from a bug when the
    picture comes out wrong, so the line has to name both numbers."""
    block = _SRC.split("_distilled = None", 1)[1][:2200]
    assert "distilled" in block and "30 at CFG 7.5" in block
