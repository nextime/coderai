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

"""A model registered by writing ``models[key]`` directly must still stamp the
LRU clock.

``_last_used`` is stamped in exactly one place -- the ``active_in_vram`` setter,
whose own comment calls it the "single chokepoint for 'this model was just
used'". ``add_model()`` goes through it; seven sites in video.py/images.py wrote
``multi_model_manager.models[key] = obj`` directly and did not.

Two independent consequences, both for the model that was just loaded:

* ``_lru_order()`` sorts by ``self._last_used.get(k, 0.0)``, so an unstamped key
  sorts FIRST and is evicted ahead of genuinely older models -- permanently, no
  matter how recently it was used;
* the first eviction pass skips ``key == self.active_in_vram``, which an
  unstamped key never matches, so that protection never applies to it either.

Measured on a production run, 2026-10-29. A co-located engine needed room and
released the freshly-loaded video pipeline first, ahead of a TTS model that was
genuinely older but had been registered through ``add_model``:

    [cosite-evict] releasing 'video:Wan-AI/Wan2.2-TI2V-5B-Diffusers' ...
    [cosite-evict] releasing 'tts:Kokoro-82M' ...

Both the VRAM ladder (``_evict_models_for_vram``) and the cross-engine one
(``cosite_evict``) iterate ``_lru_order()``, so the mis-ordering applies to each.
"""

import ast
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# How many lines after the registry write the stamp may appear. Wide enough for
# an explanatory comment block, narrow enough that it must be the same step.
_WINDOW = 14


def _registry_writes(rel):
    """Linenos of ``multi_model_manager.models[...] = ...`` assignments."""
    tree = ast.parse(io.open(os.path.join(_ROOT, rel), encoding="utf-8").read(), rel)
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if (isinstance(tgt, ast.Subscript)
                    and isinstance(tgt.value, ast.Attribute)
                    and tgt.value.attr == "models"
                    and isinstance(tgt.value.value, ast.Name)
                    and tgt.value.value.id == "multi_model_manager"):
                out.append(node.lineno)
    return out


def _stamp_lines(rel):
    """Linenos assigning to ``multi_model_manager.active_in_vram``."""
    tree = ast.parse(io.open(os.path.join(_ROOT, rel), encoding="utf-8").read(), rel)
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if (isinstance(tgt, ast.Attribute) and tgt.attr == "active_in_vram"
                    and isinstance(tgt.value, ast.Name)
                    and tgt.value.id == "multi_model_manager"):
                out.append(node.lineno)
    return out


def test_every_raw_registry_write_stamps_the_lru_clock():
    missing = []
    for rel in ("codai/api/video.py", "codai/api/images.py"):
        writes, stamps = _registry_writes(rel), _stamp_lines(rel)
        assert writes, f"{rel}: no models[key] writes found — test anchor is stale"
        for ln in writes:
            if not any(ln < s <= ln + _WINDOW for s in stamps):
                missing.append(f"{rel}:{ln}")
    assert not missing, (
        "these registry writes never stamp the LRU clock, so the model they just "
        "loaded sorts first in _lru_order() and is evicted before older models: "
        + ", ".join(missing))


def test_the_stamp_goes_through_the_chokepoint_not_the_backing_field():
    """Writing ``_last_used`` or ``_active_in_vram`` directly would bypass the
    setter and silently drift from it. The stamp must be the public attribute."""
    for rel in ("codai/api/video.py", "codai/api/images.py"):
        src = io.open(os.path.join(_ROOT, rel), encoding="utf-8").read()
        assert "_active_in_vram" not in src, (
            f"{rel} writes the backing field, bypassing the stamping setter")
        assert "_last_used[" not in src, (
            f"{rel} stamps _last_used by hand instead of via active_in_vram")


def test_add_model_still_stamps():
    """The chokepoint itself must keep stamping, or every caller regresses."""
    src = io.open(os.path.join(_ROOT, "codai/models/manager.py"), encoding="utf-8").read()
    setter = src.split("@active_in_vram.setter", 1)[1][:600]
    assert "_last_used[key] = time.monotonic()" in setter, (
        "the active_in_vram setter no longer stamps the LRU clock")
    add = src.split("def add_model(", 1)[1][:800]
    assert "self.active_in_vram = key" in add, (
        "add_model no longer routes through the stamping setter")


def test_evict_debug_distinguishes_a_missing_entry_from_a_pipeline():
    """``getattr(model_obj, 'backend', <default>)`` returns the default both when
    the object lacks the attribute AND when model_obj is None, so the old line
    printed ``backend=missing`` for a perfectly healthy diffusers pipeline and
    for a key that had already vanished from self.models. Only the second is a
    fault, and reading the log could not tell them apart."""
    src = io.open(os.path.join(_ROOT, "codai/models/manager.py"), encoding="utf-8").read()
    blk = src.split("evict-debug", 1)[0][-1400:]
    assert "if model_obj is None:" in blk, (
        "evict-debug cannot distinguish an absent registry entry from an object "
        "that simply has no .backend attribute")
    assert "NO-ENTRY" in blk and "n/a (not a ModelManager)" in blk
    # The sentinel must be compared by identity: `_be == '<str>'` invokes the
    # loaded object's __eq__, which for a tensor-bearing object need not return
    # a bool at all.
    assert "_be is _absent" in blk, "the sentinel is not compared by identity"
    assert "obj_type=" in src.split("evict-debug", 1)[1][:400], (
        "evict-debug does not report the object type, which is what actually "
        "says whether .backend was expected")
