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

"""The dtype default must be chosen from a device that has been detected.

`NvidiaBackend.__init__` leaves `self.device = None`. The loader picks the
precision default as `'fp16' if self.device == 'cuda' else 'fp32'`, so while
the detection sat further down the function, the FIRST load in a fresh engine
process defaulted to fp32 -- twice the VRAM on a 24 GB card, and FlashAttention
refuses fp32, so the model answered every request with "FlashAttention only
support fp16 and bf16 data type" (and, before that was surfaced, an empty
completion). A second load in the same process inherited `self.device` from the
first and quietly got fp16, so the same card loaded at two precisions
depending on load order.
"""

import ast
import io
import os

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "codai", "backends", "cuda.py")


def _load_fn():
    tree = ast.parse(io.open(SRC, encoding="utf-8").read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "NvidiaBackend":
            for fn in node.body:
                if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    src = ast.dump(fn)
                    if "_detect_device" in src and "precision" in src:
                        return fn
    raise AssertionError("no NvidiaBackend method both detects the device "
                         "and resolves a precision default")


def _detect_lines(fn):
    """Lines assigning self.device from self._detect_device()."""
    out = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assign):
            continue
        tgts = [t for t in node.targets
                if isinstance(t, ast.Attribute) and t.attr == "device"]
        if not tgts:
            continue
        if any(isinstance(c, ast.Attribute) and c.attr == "_detect_device"
               for c in ast.walk(node.value)):
            out.append(node.lineno)
    return sorted(out)


def _device_compare_lines(fn):
    """Lines comparing self.device to the literal 'cuda'."""
    out = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Compare):
            continue
        left = node.left
        if not (isinstance(left, ast.Attribute) and left.attr == "device"):
            continue
        for comp in node.comparators:
            if isinstance(comp, ast.Constant) and comp.value == "cuda":
                out.append(node.lineno)
    return sorted(out)


def test_the_device_is_detected_before_the_dtype_default_reads_it():
    fn = _load_fn()
    detects = _detect_lines(fn)
    compares = _device_compare_lines(fn)
    assert detects, "the loader never detects the device"
    assert compares, (
        "no `self.device == 'cuda'` comparison found; if the dtype default "
        "stopped depending on the device, retarget this test")
    assert min(detects) < min(compares), (
        f"self.device is compared to 'cuda' at line {min(compares)} but only "
        f"assigned from _detect_device() at line {min(detects)}: on the first "
        f"load of a fresh backend self.device is still None, so the dtype "
        f"default silently becomes fp32 on a CUDA card")


def test_the_device_is_detected_once():
    """A second detection would re-log and invite the old ordering back."""
    fn = _load_fn()
    detects = _detect_lines(fn)
    assert len(detects) == 1, (
        f"self.device is assigned from _detect_device() at lines {detects}; "
        f"keep a single detection ahead of every use")


def test_an_unset_precision_on_cuda_resolves_to_a_flash_attention_dtype():
    """fp32 is not a dtype FlashAttention accepts, so it cannot be the default."""
    import torch
    from codai.models.hf_loading import resolve_dtype
    got = resolve_dtype({"precision": None}, default="fp16")
    assert got in (torch.float16, torch.bfloat16), (
        f"an unset precision on CUDA resolved to {got}; FlashAttention only "
        f"supports fp16/bf16, and fp32 also doubles the VRAM a card must hold")
