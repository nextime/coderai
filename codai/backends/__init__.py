"""Backend detection module."""


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
