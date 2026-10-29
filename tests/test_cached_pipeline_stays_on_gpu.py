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

"""A cache hit must not send a model that fits to the CPU.

Measured on a production run, 2026-10-28. The ladder did everything right --
it evicted an active model and even a keep-resident one, got to 24.9 GB free,
and concluded:

    Full-GPU load looks viable (22.1 GB peak need, 24.9 GB free) — using full GPU

Two minutes later the same load reported:

    [pipeline-cache] injected cached components → forcing hook-based 'model' CPU offload
    Video load strategy: model CPU offload
    [after load] VRAM 0.4/25.3 GB (24.9 GB free)
    footprint 22.11 GB total, of which 20.47 GB was offloaded to host RAM

93% of the model ran on the CPU beside an idle card, nine times across the run.

The cause was a signalling one: the sizing expressed "full GPU" by leaving
``offload`` as ``None``, which the cached-components path could not tell from
"nobody chose", so it applied its own default -- a default written for a
dual-expert Wan2.2 that genuinely does not fit in 24 GB.

It was self-perpetuating: a pipeline loaded under accelerate offload is not
re-cached ("cache must be built from a pre-offload pipeline"), so once cached,
every later load was forced to the CPU and no clean copy could ever replace it.
"""

import ast
import io
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_SRC_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "codai/api/video.py")
_SRC = io.open(_SRC_PATH, encoding="utf-8").read()


def test_the_loader_accepts_the_sizing_verdict():
    assert "full_gpu_viable" in _SRC.split("def _load_video_pipeline", 1)[1][:400], (
        "_load_video_pipeline cannot be told the load was sized to fit")


def test_a_sized_to_fit_cache_hit_stays_resident():
    branch = _SRC.split("injected cached components", 1)[1]
    head = branch[:2500]
    assert "elif full_gpu_viable" in head, (
        "a cache hit that was sized to fit still falls through to CPU offload")
    viable_at = head.index("elif full_gpu_viable")
    resident_at = head.index("offload = 'resident'", viable_at)
    model_at = head.index("offload = 'model'", viable_at)
    assert resident_at < model_at, "the sized-to-fit branch does not choose resident"


def test_the_dual_expert_default_is_kept_for_loads_that_do_not_fit():
    """The 'model' default exists because two ~10 GB experts do not fit in
    24 GB. Keeping a model resident that cannot be resident OOMs at step 0,
    so the default must survive for the not-viable case."""
    branch = _SRC.split("injected cached components", 1)[1][:2500]
    assert "offload = 'model'" in branch, (
        "the dual-expert fallback was removed; a model that cannot fit will "
        "now OOM instead of offloading")


def test_the_explicit_opt_in_still_wins():
    branch = _SRC.split("injected cached components", 1)[1][:2500]
    assert "video_resident_experts" in branch


def _fn(name):
    tree = ast.parse(_SRC)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def test_the_verdict_is_defined_on_every_path():
    """It is read unconditionally but was set inside

        if _offload is None: -> try: -> if cuda.is_available():

    so an explicit per-model strategy, a CPU-only host, or any exception in
    that block would raise NameError and break EVERY video request -- far
    worse than the offload bug being fixed."""
    fn = _fn("video_generations")
    stores = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Name)
              and n.id == "_full_gpu_viable" and isinstance(n.ctx, ast.Store)]
    loads = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Name)
             and n.id == "_full_gpu_viable" and isinstance(n.ctx, ast.Load)]
    assert stores and loads
    assert min(stores) < min(loads), (
        f"_full_gpu_viable is read at line {min(loads)} but first assigned at "
        f"{min(stores)} — a NameError on some paths")

    first = min(stores)

    # Stronger than ordering: the block holding the first assignment must
    # DOMINATE the read, i.e. be the same block or an enclosing one. Ordering
    # alone would pass even with the assignment buried in a branch that the
    # read's path skips -- which is exactly the shape of the original bug.
    def _blocks(fn):
        """Map each statement's lineno to the chain of bodies enclosing it."""
        out = {}

        def walk(stmts, chain):
            for st in stmts:
                out.setdefault(st.lineno, chain)
                for field in ("body", "orelse", "finalbody"):
                    sub = getattr(st, field, None)
                    if isinstance(sub, list) and sub:
                        walk(sub, chain + [id(sub)])
                for h in getattr(st, "handlers", []) or []:
                    walk(h.body, chain + [id(h.body)])
        walk(fn.body, [id(fn.body)])
        return out

    blocks = _blocks(fn)

    def chain_for(line):
        best, found = None, None
        for ln, ch in blocks.items():
            if ln <= line and (best is None or ln > best):
                best, found = ln, ch
        return found or []

    a_chain = chain_for(first)
    r_chain = chain_for(min(loads))
    assert a_chain == r_chain[:len(a_chain)], (
        "the first assignment does not dominate the read: it sits in a branch "
        "the read's path can skip, so _full_gpu_viable can be undefined there")


def test_the_borderline_case_also_tries_the_card_first():
    """"Marginally over, try full GPU first" means the same thing as viable:
    attempt the card, fall back on OOM."""
    assert re.search(r"_full_gpu_viable = True\s*\n\s*print\(f\"  Peak VRAM need", _SRC), (
        "the borderline branch says it will try the card but does not record it")
