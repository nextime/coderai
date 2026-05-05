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

"""Backend detection and management module."""

from codai.backends.base import ModelBackend
from codai.backends.cuda import NvidiaBackend
from codai.backends.vulkan import VulkanBackend


def detect_available_backends():
    """Detect which backends are available."""
    backends = {'cpu': True}
    
    # Check for PyTorch/CUDA
    try:
        import torch
        if torch.cuda.is_available():
            backends['nvidia'] = True
    except ImportError:
        pass
    
    # Check for llama-cpp-python (Vulkan)
    try:
        import llama_cpp
        backends['vulkan'] = True
    except ImportError:
        pass
    
    return backends


def check_flash_attn_availability() -> bool:
    """Check if flash-attn is installed and available."""
    try:
        import flash_attn
        return True
    except ImportError:
        return False