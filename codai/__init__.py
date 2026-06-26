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

# Canonical product version for CoderAI — single source of truth. Both the API
# metadata and the admin web UI read from here.
__version__ = "0.1.27"

# Configure the CUDA caching allocator BEFORE torch is imported anywhere.
# expandable_segments lets the allocator return freed pages to the driver even
# from partially-used segments.  Without it, a single small live tensor (e.g. a
# tied embedding weight) pins an entire large segment, so torch.cuda.empty_cache()
# cannot release the GBs of already-freed weights around it after a model is
# evicted — VRAM stays occupied and the next model can't load.  Honour any value
# the user already set.
import os as _os
_alloc_conf = _os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
if "expandable_segments" not in _alloc_conf:
    _os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        (_alloc_conf + ",") if _alloc_conf else ""
    ) + "expandable_segments:True"

# Cap CPU threads BEFORE torch / OpenMP / MKL initialise.  Loading and 4-bit
# dequantising large models is CPU-heavy; left uncapped, torch/OpenMP grab every
# core and the machine's load average spikes and it becomes sluggish.  On boxes
# with >= 8 cores, limit to HALF the cores so model loads never saturate the
# machine.  Smaller machines keep the default (don't cripple them).  Honour any
# value the user already set.
try:
    _ncpu = _os.cpu_count() or 0
    if _ncpu >= 8:
        _cap = str(max(1, _ncpu // 2))
        for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            _os.environ.setdefault(_var, _cap)
except Exception:
    pass

# Silence ONE specific upstream FutureWarning from bitsandbytes' quant kernels:
#   bitsandbytes/backends/cuda/ops.py: torch._check_is_size(blocksize)
# bitsandbytes (latest, 0.49.2) still calls the deprecated torch._check_is_size
# on bleeding-edge torch.  We don't call it ourselves and can't fix their source,
# so suppress just this message (not warnings in general) to keep logs readable.
import warnings as _warnings
_warnings.filterwarnings(
    "ignore",
    message=r".*_check_is_size will be removed.*",
    category=FutureWarning,
)
# More upstream / diagnostic-only noise we can't fix from here:
#   - huggingface_hub: diffusers/transformers pass the deprecated
#     `local_dir_use_symlinks` kwarg to hf_hub_download (not our code).
#   - torch.distributed.reduce_op: emitted while the debug leak-scanner walks
#     gc.get_objects(); unavoidable without dropping the scan.
_warnings.filterwarnings(
    "ignore",
    message=r".*local_dir_use_symlinks.*",
    category=UserWarning,
)
_warnings.filterwarnings(
    "ignore",
    message=r".*reduce_op.*is deprecated.*",
    category=FutureWarning,
)

# codai module - AI model parsing utilities
from .models.parser import (
    ModelParserDispatcher,
    BaseParser,
    QwenParser,
    DeepSeekParser,
    LlamaParser,
    MistralParser,
    ClaudeParser,
    CommandRParser,
    GemmaParser,
    GrokParser,
    PhiParser,
    ApexBig50Parser,
)

from .models.templates import AgenticTemplateManager
from .models.utils import FuzzyToolBreaker

__all__ = [
    'ModelParserDispatcher',
    'BaseParser',
    'QwenParser',
    'DeepSeekParser',
    'LlamaParser',
    'MistralParser',
    'ClaudeParser',
    'CommandRParser',
    'GemmaParser',
    'GrokParser',
    'PhiParser',
    'ApexBig50Parser',
    'AgenticTemplateManager',
    'FuzzyToolBreaker',
]