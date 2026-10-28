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

"""Stay in VRAM; offload only when there is no other choice.

`auto` condemns a model on its ESTIMATE: if `used_vram_gb` exceeds free VRAM it
goes straight to CPU offload without ever trying the card. That estimate is a
measurement from an earlier load in some cards and a hand-entered round number
in others -- LTX-2.3-PinkCherry declared exactly 24.0 on a 24 GiB card -- so a
conservative figure permanently exiles a model that would have fit.

The cost of being wrong is asymmetric, which is what decides the default:
guessing "fits" wrongly costs one failed attempt and a fallback, while guessing
"does not fit" wrongly costs a CPU copy of every component and a PCIe trip per
request for the life of the install. On the host that prompted this, video
renders were each holding 28-37 GB of host RAM that way, on a 54 GB box with no
cgroup limit to stop them.
"""

import io
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_the_global_default_prefers_vram():
    from codai.config import Config
    assert Config().offload.strategy == "auto-borderline", (
        "the default offload strategy no longer prefers VRAM; plain 'auto' "
        "offloads on the estimate alone and never tries the card")


def test_the_model_page_falls_back_to_the_vram_preferring_strategy():
    """A card that says nothing must not inherit "offload first"."""
    html = io.open(os.path.join(ROOT, "codai/admin/templates/models.html"),
                   encoding="utf-8").read()
    m = re.search(r"cfg-offload-strategy'\)\.value = s\.offload_strategy \|\| '([a-z-]+)'",
                  html)
    assert m, "the offload-strategy fallback could not be found"
    assert m.group(1) == "auto-borderline", (
        f"the model page falls back to {m.group(1)!r}")


def test_the_recommended_option_is_offered_first():
    """The option list is a recommendation; the safest-sounding one should not
    be the pessimistic one."""
    html = io.open(os.path.join(ROOT, "codai/admin/templates/models.html"),
                   encoding="utf-8").read()
    block = html[html.index('id="cfg-offload-strategy"'):]
    block = block[:block.index("</select>")]
    values = re.findall(r'<option value="([a-z-]+)"', block)
    assert values[0] == "auto-borderline", (
        f"the first offered strategy is {values[0]!r}, not the VRAM-preferring one")
    # Both are still available: an operator who wants the pessimistic one for a
    # genuinely over-sized model should be able to say so.
    assert "auto" in values and "model" in values


def test_both_auto_modes_reach_the_loader():
    """`auto-borderline` must be understood wherever `auto` is, or a card set
    to it would be treated as an unknown strategy."""
    for path in ("codai/api/video.py", "codai/backends/cuda.py"):
        src = io.open(os.path.join(ROOT, path), encoding="utf-8").read()
        assert "auto-borderline" in src, f"{path} does not handle auto-borderline"


def test_the_image_path_does_not_force_offload_for_the_auto_modes():
    """images.py decides offload from a list of strategies; neither auto mode
    belongs in it, or "prefer VRAM" would mean the opposite."""
    src = io.open(os.path.join(ROOT, "codai/api/images.py"), encoding="utf-8").read()
    m = re.search(r"_offload_strategy in \(([^)]*)\)", src)
    assert m, "the image offload decision could not be found"
    listed = m.group(1)
    assert "auto" not in listed, (
        f"an auto mode is listed as forcing CPU offload: {listed}")


def test_the_borderline_band_is_a_named_number():
    """How far over free VRAM still counts as "worth trying" is a judgement;
    it should be visible, not buried in an expression."""
    src = io.open(os.path.join(ROOT, "codai/api/video.py"), encoding="utf-8").read()
    assert "_BORDERLINE_GB" in src
    m = re.search(r"_BORDERLINE_GB\s*=\s*([0-9.]+)", src)
    assert m and float(m.group(1)) > 0
