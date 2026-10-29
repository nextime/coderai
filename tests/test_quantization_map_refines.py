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

"""component_quantization refines load_in_4bit; it does not cancel it.

Naming one component in the map used to switch the global flag off for every
OTHER component, which is the opposite of what setting both plainly means.

Measured 2026-10-29 on LTX-2.5, configured load_in_4bit=True with a map naming
only transformer and text_encoder: ten of its twelve components loaded at full
precision, including connectors (11.82 GB) and prompt_enhancer (9.51 GB) --
21.3 GB of a 25.3 GB card before the quantized transformer arrived. The
resident attempt then OOMed on exactly those components.

The old candidate list was also written for unet/SD-era pipelines and matched
only transformer*/unet/text_encoder*, so those two were never candidates for
the global flag either.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codai.models.hf_loading import _is_quantizable_component


def test_the_classic_heavy_components_still_qualify():
    for n in ("transformer", "transformer_2", "unet", "text_encoder"):
        assert _is_quantizable_component(n), n


def test_the_multi_gb_parts_that_were_missed_now_qualify():
    """connectors and prompt_enhancer outweigh LTX-2.5's quantized
    transformer between them."""
    for n in ("connectors", "prompt_enhancer"):
        assert _is_quantizable_component(n), (
            f"{n} is still not a candidate for the global quantization flag")


def test_small_or_conv_only_components_are_not_swept_in():
    for n in ("vae", "audio_vae", "scheduler", "tokenizer", "vocoder",
              "duration_head"):
        assert not _is_quantizable_component(n), n


def test_both_builders_apply_the_global_flag_alongside_the_map():
    """The pipeline builder and the incremental cache builder must agree, or a
    cached build quantizes differently from a direct load."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(root, "codai/models/hf_loading.py"),
               encoding="utf-8").read()
    assert src.count("if global_4 or global_8:") >= 2, (
        "one of the two builders still treats the map as replacing the "
        "global flag")
    # And neither may fall back to the old narrow list.
    assert "if _is_heavy(name) and not _bnb_incompatible(name)" not in src


def test_an_explicit_map_entry_is_not_overridden():
    """The map wins for the components it names -- including an explicit
    'none', which is how an operator opts one component out."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(root, "codai/models/hf_loading.py"),
               encoding="utf-8").read()
    assert "if name in quant_mapping or name in comp_q:" in src, (
        "the global flag can overwrite a component the operator configured")
    assert "if name in targets or name in comp_q:" in src
