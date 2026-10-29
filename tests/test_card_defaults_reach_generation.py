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

"""A card's `defaults` have to reach the generation, not just `/v1/models`.

Found while adding `defaults` to the Z-Image-Turbo card, 2026-10-29.

The manager keeps runtime kwargs FLAT and the original models.json entry under
`_raw_cfg`. Only keys that `build_runtime_kwargs` promotes exist at the top
level, and `defaults` is not one of them — so `model_defaults()`, reading the
flat dict, got nothing. `/v1/models` published the card's steps and guidance
(it reads the entry) while generation used the generic 30 steps at CFG 7.5.

A client that reads `defaults` from the listing and trusts the server to apply
them was therefore wrong twice over, and the two models whose cards carried
numbers — Z-Image-Turbo-IntoRealism at 10 steps, Krea-2-RealDream at 8 — never
had them applied.

The video path had already hit this and merged `_raw_cfg` by hand at its call
site. Same trap as the `gpu_split` and `sequential_text_encoder` promotions,
which carry the same comment in `build_runtime_kwargs`.
"""

import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codai.api.images import card_view, model_defaults

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _src(rel):
    return io.open(os.path.join(_ROOT, rel), encoding="utf-8").read()


def _runtime_cfg(entry):
    """What the manager stores: flat kwargs plus the entry under _raw_cfg."""
    from codai.main import build_runtime_kwargs
    return build_runtime_kwargs(entry, "image")


def test_the_card_defaults_survive_the_runtime_config():
    entry = {"alias": "Z-Image-Turbo-bf16", "path": "Tongyi-MAI/Z-Image-Turbo",
             "precision": "bf16",
             "defaults": {"steps": 8, "guidance_scale": 1.0,
                          "sampler": "flow_euler"}}
    cfg = _runtime_cfg(entry)
    assert "defaults" not in cfg, (
        "this test is pointless if build_runtime_kwargs starts promoting it; "
        "update it to match")
    got = model_defaults(cfg)
    assert got["steps"] == 8 and got["guidance_scale"] == 1.0, (
        f"a card with defaults still reads as {got} on the image path")


def test_call_kwargs_survive_it_too():
    """MOSS-Image's `uncond_skip_layers` lives in the same place."""
    cfg = _runtime_cfg({"path": "x", "call_kwargs": {"uncond_skip_layers": [22]}})
    assert card_view(cfg).get("call_kwargs") == {"uncond_skip_layers": [22]}


def test_a_flat_config_still_works():
    """Not every caller hands over a runtime config; some pass the entry."""
    assert model_defaults({"defaults": {"steps": 4}})["steps"] == 4
    assert model_defaults({}) == {}
    assert model_defaults(None) == {}


def test_the_runtime_kwargs_win_on_a_collision():
    """They are what the loader was actually given."""
    merged = card_view({"precision": "fp8", "_raw_cfg": {"precision": "bf16"}})
    assert merged["precision"] == "fp8"


def test_both_media_paths_read_the_card_the_same_way():
    """The video path merged by hand at its call site; the image path did not
    merge at all. One of them being right is what made this invisible."""
    img = _src("codai/api/images.py")
    assert "def card_view(" in img
    dflt = img.split("def model_defaults", 1)[1].split("\ndef ", 1)[0]
    assert "card_view(model_config)" in dflt, (
        "model_defaults reads the flat config again")
    vid = _src("codai/api/video.py")
    assert "model_defaults(" in vid


def test_the_listing_and_the_runtime_agree_on_a_real_card():
    """The regression in one line: what /v1/models publishes is what the
    generation uses. Checked against the installed catalogue when there is
    one, so it also catches a card whose defaults are malformed."""
    import json
    path = os.path.expanduser("~/.coderai/models.json")
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    for entry in (data.get("image_models") or []):
        card = entry.get("defaults")
        if not isinstance(card, dict) or not card:
            continue
        got = model_defaults(_runtime_cfg(entry))
        assert got == card, (
            f"{entry.get('alias') or entry.get('path')}: the listing says "
            f"{card} and generation would use {got}")
