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

"""Compatibility shims for third-party version skew."""

_awq_aliased = False


def ensure_peft_awq_compat():
    """peft's LoRA AWQ dispatcher does ``from gptqmodel.nn_modules.qlinear.gemm_awq
    import AwqGEMMQuantLinear``, but gptqmodel 7.1.0 renamed that class to
    AwqGEMMLinear. peft calls dispatch_awq for ANY non-bnb target whenever
    gptqmodel is installed, so the failed import crashes EVERY add_adapter() /
    load_lora_weights() — at LoRA *training* AND at *inference* when a trained
    LoRA is applied to a pipeline. Alias the renamed class so the import succeeds;
    no-op when the name already exists or gptqmodel isn't present. Cached so the
    per-request inference path doesn't re-import gptqmodel every call."""
    global _awq_aliased
    if _awq_aliased:
        return
    try:
        import importlib
        m = importlib.import_module("gptqmodel.nn_modules.qlinear.gemm_awq")
        if not hasattr(m, "AwqGEMMQuantLinear") and hasattr(m, "AwqGEMMLinear"):
            m.AwqGEMMQuantLinear = m.AwqGEMMLinear
        _awq_aliased = True
    except Exception:
        # gptqmodel not installed, or import failed — nothing to alias. Don't
        # cache so a later call can retry once gptqmodel is importable.
        pass


_dtype_guarded = False


def ensure_quantized_dtype_compat():
    """Stop a fully-quantized module raising StopIteration from ``.dtype``.

    transformers defines the property as::

        return next(param.dtype for param in self.parameters()
                    if param.is_floating_point())

    A bnb 4-bit module has NO floating-point parameters -- Params4bit are uint8
    -- so the generator is empty and ``next()`` raises StopIteration. diffusers'
    ``DiffusionPipeline.to()`` reads ``module.dtype`` unguarded while deciding
    whether to warn about fp16-on-cpu, so the exception escapes
    ``enable_model_cpu_offload()``.

    Run in a worker thread, as model loads are, Python turns that into::

        RuntimeError: StopIteration interacts badly with generators and
        cannot be raised into a Future

    which fails the whole load. Measured 2026-10-29 on LTX-2.5, whose text
    encoder is 4-bit: the resident attempt did not fit, the fallback to model
    CPU offload hit this, and the load restarted from scratch -- so the
    *recovery* path was broken exactly when it was needed.

    The property answers "what dtype is this module" and a quantized module has
    a perfectly good answer: the dtype of its storage. Fall back to the first
    parameter of any kind, then to float32, and only when there is nothing at
    all. Never changes the answer for a module that HAS floating-point
    parameters, so an unquantized model behaves exactly as before.
    """
    global _dtype_guarded
    if _dtype_guarded:
        return
    try:
        import torch
        from transformers.modeling_utils import PreTrainedModel
    except Exception:
        return
    try:
        original = PreTrainedModel.dtype
        if not isinstance(original, property):
            return

        def _dtype(self):
            try:
                return original.fget(self)
            except StopIteration:
                try:
                    return next(p.dtype for p in self.parameters())
                except StopIteration:
                    return torch.float32

        PreTrainedModel.dtype = property(_dtype)
        _dtype_guarded = True
    except Exception:
        pass
