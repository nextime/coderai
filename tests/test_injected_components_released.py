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

"""Injected components must not keep the card between load attempts.

The GGUF / cached components are PRE-BUILT objects: they live in a local dict
and are handed to each attempt through _with_quant. Freeing the pipeline
therefore does not free them -- the dict is still a strong reference, and
anything the failed attempt moved onto the GPU stays there.

Measured 2026-10-29 after a resident attempt on LTX-2.5 failed:

    [video] teardown-leak breaker: 16.9 GB of this pipeline's tensors survived
    teardown; nulled 7581 external reference(s);
    holders: ["2.01 GB dict['weight', 'bias']", ...]

Those holders are the modules' own _parameters dicts, i.e. the modules
outlived the pipeline. The breaker recovers the VRAM by stripping them field
by field, which works but guts the components the next attempt is about to
reuse -- so the leak has to be closed, not just mopped up.
"""

import ast
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "codai/api/video.py")
_SRC = io.open(_PATH, encoding="utf-8").read()


def test_both_resident_failure_exits_release_the_components():
    """Resident can fail two ways -- a component that will not fit, and an OOM
    raised during the move. Both fall through to the next attempt, so both
    must hand the card back first."""
    assert _SRC.count("_release_injected_to_cpu(") >= 3, (
        "a resident-failure exit falls through without releasing the injected "
        "components, so they keep their VRAM through the retry")
    assert "'a partial resident fit'" in _SRC
    assert "'a resident OOM'" in _SRC


def test_it_moves_rather_than_drops():
    """The next attempt reuses these components; dropping them would defeat
    the point of injecting them at all."""
    body = _SRC.split("def _release_injected_to_cpu", 1)[1][:2600]
    assert "_c.to('cpu')" in body, "components are not moved to the CPU"
    assert "del " not in body.split("def ")[0], "components are being dropped"


def test_a_component_already_off_the_gpu_is_left_alone():
    body = _SRC.split("def _release_injected_to_cpu", 1)[1][:2600]
    assert "if not _on_gpu:" in body and "continue" in body


def test_a_component_that_refuses_to_move_does_not_abort_the_retry():
    """A bnb-quantized module can refuse .to('cpu'). That must not take down
    the fallback path -- the teardown-leak breaker remains the backstop."""
    body = _SRC.split("def _release_injected_to_cpu", 1)[1][:2600]
    assert "except Exception:" in body


def test_the_helper_parses_at_the_right_scope():
    """It is nested inside the loader so it can see _gguf_components; an
    indentation slip here is a SyntaxError that breaks the whole module."""
    tree = ast.parse(_SRC)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
    assert "_release_injected_to_cpu" in names
