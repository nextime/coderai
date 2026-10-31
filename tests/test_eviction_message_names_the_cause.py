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

"""When a load falls back to CPU, say WHY it really did.

The message printed whenever eviction cannot make room is the first thing
anyone reads when a model lands on the CPU, and it used to blame "remaining
models are busy or VRAM is held elsewhere" unconditionally.

Measured over one production day, that was wrong 10 times out of 12: the model
had asked for more than the entire card (25.4 to 40.0 GB against 25.3 GB
usable), and FOUR of those printed "freed 0.0 GB (now 25.3 GB free)" -- the
card was already completely empty. Not one of the twelve was blocked by a busy
model; "Skipping eviction of ... still busy" never appeared at all.

The wording sent the investigation at the scheduler when the real causes were
an under-declared used_vram_gb and a model bigger than the GPU.
"""

import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_SRC = io.open(os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "codai/models/manager.py"), encoding="utf-8").read()

_BLOCK = _SRC.split("⚠ Eviction freed", 1)[0][-3000:] + \
    _SRC.split("⚠ Eviction freed", 1)[1][:1500]


def test_it_no_longer_blames_busy_models_unconditionally():
    tail = _SRC.split("⚠ Eviction freed", 1)[1][:1200]
    assert "Remaining models are busy or VRAM is held elsewhere — the new" not in tail, (
        "the message still asserts a cause it has not established")


def test_a_model_bigger_than_the_card_is_named_as_such():
    """The dominant real case: eviction cannot possibly help."""
    assert "needs more than the whole card" in _SRC, (
        "the commonest cause -- a model larger than the GPU -- is not reported")


def test_the_three_cases_are_distinguished():
    before = _SRC.split("⚠ Eviction freed", 1)[0][-2000:]
    assert "if _total_gb and needed_gb > _total_gb:" in before
    assert "elif _free_after >= _free_before" in before
    assert "else:" in before


def test_it_uses_a_total_vram_accessor_that_exists():
    """A misspelled accessor would fall into the except and silently report
    the wrong case every time -- the exact failure this message is meant to
    stop."""
    assert "self._total_vram_gb()" in _SRC
    assert "self._get_total_vram_gb()" not in _SRC, (
        "that method does not exist; the call would always raise and the "
        "branch would never be taken")
    import ast
    tree = ast.parse(_SRC)
    names = {n.name for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert "_total_vram_gb" in names


def test_it_suggests_something_actionable():
    """A model larger than the card is fixed by quantization or by correcting
    a wrong declared size -- not by waiting longer in a queue.

    Every occurrence of the phrase is checked, not the first: the first one in
    the file is a COMMENT that quotes the message while explaining a different
    decision, and anchoring on it made this test fail while the message it
    guards was untouched.
    """
    windows = _SRC.split("needs more than the whole card")[1:]
    assert windows, "the whole-card case no longer says so"
    assert any("quantize" in w[:400] and "used_vram_gb" in w[:400]
               for w in windows), \
        "no whole-card message names a way out of it"
