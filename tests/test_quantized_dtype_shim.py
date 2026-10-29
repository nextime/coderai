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

"""A fully quantized module must be able to answer "what dtype are you".

transformers defines the property as

    next(param.dtype for param in self.parameters() if param.is_floating_point())

and a bnb 4-bit module has NO floating-point parameters -- Params4bit are
uint8 -- so the generator is empty and next() raises StopIteration. diffusers'
DiffusionPipeline.to() reads module.dtype unguarded, so it escapes
enable_model_cpu_offload(); and because model loads run in a worker thread,
Python converts it to

    RuntimeError: StopIteration interacts badly with generators and cannot be
    raised into a Future

failing the entire load. Measured 2026-10-29 on LTX-2.5, whose text encoder is
4-bit: the resident attempt did not fit, the fallback to model CPU offload hit
this, and the load restarted from scratch. The RECOVERY path was broken
precisely when it was needed.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from codai.models import peft_compat


class _AllQuantized(torch.nn.Module):
    """Every parameter uint8, as a 4-bit module's are."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(
            torch.zeros(4, 4, dtype=torch.uint8), requires_grad=False)


class _Normal(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(4, 4, dtype=torch.float16))


class _Empty(torch.nn.Module):
    pass


def _dtype_of(mod):
    from transformers.modeling_utils import PreTrainedModel
    return PreTrainedModel.dtype.fget(mod)


def test_the_unshimmed_property_really_does_raise():
    """The premise: without the shim this is a StopIteration, not a value."""
    peft_compat._dtype_guarded = False
    from transformers.modeling_utils import PreTrainedModel
    original = PreTrainedModel.dtype
    try:
        with pytest.raises(StopIteration):
            original.fget(_AllQuantized())
    finally:
        PreTrainedModel.dtype = original


def test_a_fully_quantized_module_reports_its_storage_dtype():
    peft_compat._dtype_guarded = False
    peft_compat.ensure_quantized_dtype_compat()
    assert _dtype_of(_AllQuantized()) == torch.uint8


def test_an_ordinary_module_is_unchanged():
    """The shim must never alter the answer for a module that HAS floating
    point parameters."""
    peft_compat._dtype_guarded = False
    peft_compat.ensure_quantized_dtype_compat()
    assert _dtype_of(_Normal()) == torch.float16


def test_a_module_with_no_parameters_at_all_still_answers():
    peft_compat._dtype_guarded = False
    peft_compat.ensure_quantized_dtype_compat()
    assert _dtype_of(_Empty()) == torch.float32


def test_applying_it_twice_does_not_stack():
    peft_compat._dtype_guarded = False
    peft_compat.ensure_quantized_dtype_compat()
    peft_compat.ensure_quantized_dtype_compat()
    peft_compat.ensure_quantized_dtype_compat()
    assert _dtype_of(_AllQuantized()) == torch.uint8
    assert _dtype_of(_Normal()) == torch.float16


def test_every_offload_call_site_is_guarded():
    """enable_model_cpu_offload() is where this surfaces, in several modules."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for rel in ("codai/api/video.py", "codai/api/audio_gen.py",
                "codai/api/images.py"):
        src = open(os.path.join(root, rel), encoding="utf-8").read()
        calls = src.count("enable_model_cpu_offload()")
        guards = src.count("ensure_quantized_dtype_compat()")
        assert guards >= calls, (
            f"{rel}: {calls} offload call(s) but only {guards} guard(s) — an "
            f"unguarded one fails the load for any 4-bit pipeline")
