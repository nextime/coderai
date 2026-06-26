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

"""Compatibility shims for peft <-> gptqmodel version skew."""

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
