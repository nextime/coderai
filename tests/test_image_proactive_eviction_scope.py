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

"""The minimum-eviction block before an image load has to actually run.

Observed on zeiss, 2026-10-29, on every image load:

    Proactive eviction skipped: name 'request' is not defined

``_load_diffusers_pipeline`` has no ``request`` in scope -- it takes
``(model_name, global_args, model_config)``. The sizing call read it anyway,
so the block raised NameError before it could evict anything, and the broad
``except`` printed the NameError as a skip. Every image load then fell through
to the blunt eviction further down, which frees by a different rule and
unloads models that would have fitted alongside.

A name error inside a try/except that reports "skipped" is invisible: the
feature is simply never there. These tests fail on the shape that allows it.
"""

import ast
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = io.open(os.path.join(_ROOT, "codai/api/images.py"), encoding="utf-8").read()


def _fn(name):
    for node in ast.walk(ast.parse(_SRC)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def test_the_loader_does_not_read_a_request_it_never_received():
    """Every name the function reads must be a parameter, a local, a global or
    a builtin. `request` is none of those here."""
    fn = _fn("_load_diffusers_pipeline")
    params = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
    assert "request" not in params
    reads = [n for n in ast.walk(fn)
             if isinstance(n, ast.Name) and n.id == "request" and isinstance(n.ctx, ast.Load)]
    assert not reads, (
        f"_load_diffusers_pipeline reads `request` at line(s) "
        f"{[n.lineno for n in reads]}, which does not exist in its scope")


def test_the_requested_id_is_captured_before_the_path_rewrite():
    """Sizing needs the id the CALLER asked for: two cards can share one
    checkpoint with different quantization and offload. `model_name` is
    rewritten to the checkpoint path early on, so the id has to be kept
    before that."""
    fn = _fn("_load_diffusers_pipeline")
    stores = [n.lineno for n in ast.walk(fn)
              if isinstance(n, ast.Name) and n.id == "_requested_id"
              and isinstance(n.ctx, ast.Store)]
    loads = [n.lineno for n in ast.walk(fn)
             if isinstance(n, ast.Name) and n.id == "_requested_id"
             and isinstance(n.ctx, ast.Load)]
    assert stores and loads, "the requested id is not kept at all"
    rewrite = _SRC.index("model_name = _cfg_path")
    rewrite_line = _SRC[:rewrite].count("\n") + 1
    assert min(stores) < rewrite_line, (
        f"_requested_id is assigned at line {min(stores)}, after model_name is "
        f"rewritten to the checkpoint path at {rewrite_line} — it would hold "
        f"the path, not the id that was asked for")
    assert min(stores) < min(loads)


def test_the_sizing_call_is_still_made_with_it():
    block = _SRC.split("Proactive VRAM eviction", 1)[1][:2500]
    assert "requested_id=_requested_id" in block
    assert "_evict_models_for_vram" in block
