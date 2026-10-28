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

"""A non-text model must be reachable by its alias.

Typed entries are keyed ``<kind>:<name>``, so every lookup form that works for
a text model (plain path, plain basename) misses for a video or image one. The
basename fallback misses as well, because ``video:LTX-2.3-PinkCherry`` contains
no slash and ``basename()`` hands back the whole prefixed string.

The consequence was silent: ``_config_for_model`` returned ``{}`` for a card
that carried the setting, so per-model video/image settings were simply not
applied and the feature reported itself "not enabled". Two sibling configs
sharing one checkpoint path are only distinguishable BY alias, so this is the
one lookup that has to work for them.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _Resolver:
    """The two methods under test, lifted onto the real class."""

    def __init__(self, config, aliases):
        self.config = config
        self.model_aliases = aliases

    @staticmethod
    def _bind():
        from codai.models.manager import MultiModelManager
        _Resolver._resolve_config_key = MultiModelManager._resolve_config_key
        _Resolver._config_for_model = MultiModelManager._config_for_model


_Resolver._bind()


def _two_siblings():
    """Two video configs sharing one checkpoint path, as models.json has them."""
    path = "diffusers/LTX-2.3-Diffusers"
    eros = {"alias": "LTX-2.3-10Eros", "path": path,
            "sequential_text_encoder": True, "marker": "eros"}
    pink = {"alias": "LTX-2.3-PinkCherry", "path": path,
            "sequential_text_encoder": True, "marker": "pink"}
    config = {
        # _store_type_config writes the path key (last writer wins) and an
        # alias key per entry.
        f"video:{path}": pink,
        "video:LTX-2.3-10Eros": eros,
        "video:LTX-2.3-PinkCherry": pink,
    }
    aliases = {"LTX-2.3-10Eros": path, "LTX-2.3-PinkCherry": path}
    return _Resolver(config, aliases)


def test_alias_resolves_to_its_own_typed_entry():
    r = _two_siblings()
    for alias, marker in (("LTX-2.3-10Eros", "eros"),
                          ("LTX-2.3-PinkCherry", "pink")):
        cfg = r._config_for_model(alias)
        assert cfg, f"{alias} resolved to nothing -- every setting is dropped"
        assert cfg.get("sequential_text_encoder") is True, \
            f"{alias} lost its per-model setting"
        assert cfg.get("marker") == marker, \
            f"{alias} resolved to the sibling's config ({cfg.get('marker')})"


def test_path_still_resolves():
    """The path key keeps its meaning for callers that know only a path."""
    r = _two_siblings()
    cfg = r._config_for_model("diffusers/LTX-2.3-Diffusers")
    assert cfg.get("marker") == "pink"  # last writer, as before


def test_unknown_name_still_returns_empty():
    r = _two_siblings()
    assert r._config_for_model("no-such-model") == {}
    assert r._config_for_model("") == {}


def test_exact_key_is_preferred_over_the_typed_scan():
    """An unprefixed entry must still win, so text models are unaffected."""
    r = _Resolver({"plain-model": {"marker": "plain"},
                   "text:plain-model": {"marker": "typed"}}, {})
    assert r._config_for_model("plain-model").get("marker") == "plain"


def test_flag_is_a_top_level_runtime_kwarg():
    """The gate reads config['sequential_text_encoder'], so it must be promoted
    out of _raw_cfg -- the same trap already documented for gpu_split."""
    from codai.main import build_runtime_kwargs
    on = build_runtime_kwargs({"sequential_text_encoder": True}, "video")
    off = build_runtime_kwargs({}, "video")
    assert on["sequential_text_encoder"] is True
    assert off["sequential_text_encoder"] is False


def test_the_image_and_tts_paths_resolve_instead_of_a_raw_get():
    """Three API paths looked their config up with a bare
    ``multi_model_manager.config.get(model_key)``. Typed entries are keyed
    "<kind>:<name>", so that get() misses for every alias and the per-model
    settings are replaced by global defaults with nothing logged. The video
    path's sequential_text_encoder reporting itself "not enabled" was this bug
    one module over, so the same lookups elsewhere are pinned here."""
    import re
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for rel in ("codai/api/images.py", "codai/api/tts.py"):
        body = open(os.path.join(root, rel), encoding="utf-8").read()
        bad = re.findall(r"config\.get\(model_key\)", body)
        assert not bad, (
            f"{rel} looks its model config up with a raw config.get(model_key); "
            f"use _config_for_model so a typed/alias key resolves")
