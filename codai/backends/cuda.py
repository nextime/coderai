"""CUDA backend for NVIDIA GPUs."""

from typing import Optional, List, Dict

from codai.backends.base import ModelBackend


class NvidiaBackend(ModelBackend):
    """Backend for NVIDIA GPUs using HuggingFace Transformers."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.device = None
        self.use_flash_attn = False
        self.flash_attn_available = False
        self._pending_ram_gb = None
        # Import check_flash_attn_availability from codai.backends
        from codai.backends import check_flash_attn_availability
        self._check_flash_attn_availability = check_flash_attn_availability
        
    def check_flash_attn_support(self) -> None:
        """Check and print Flash Attention availability status."""
        self.flash_attn_available = self._check_flash_attn_availability()
        if self.use_flash_attn:
            if self.flash_attn_available:
                print("Flash Attention 2: Available and enabled")
            else:
                print("Warning: Flash Attention 2 requested but not installed")
                print("Install with: pip install flash-attn --no-build-isolation")
                print("Falling back to standard attention")
                self.use_flash_attn = False
    
    def load_model(self, model_name: str, **kwargs) -> None:
        """Load the model."""
        pass
    
    def generate(self, prompt: str, max_tokens: Optional[int] = None, 
                 temperature: float = 0.7, top_p: float = 1.0,
                 stop: Optional[list] = None) -> str:
        """Generate text non-streaming."""
        pass
    
    def generate_stream(self, prompt: str, max_tokens: Optional[int] = None,
                        temperature: float = 0.7, top_p: float = 1.0,
                        stop: Optional[list] = None):
        """Generate text in streaming fashion."""
        pass
    
    def format_messages(self, messages) -> str:
        """Format messages into a prompt string."""
        pass
    
    def get_model_name(self) -> str:
        """Return the loaded model name."""
        return self.model_name
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        pass
