"""Vulkan backend using llama.cpp."""

from typing import Optional, List, Dict

from codai.backends.base import ModelBackend


class VulkanBackend(ModelBackend):
    """Backend for Vulkan GPU inference using llama.cpp."""
    
    def __init__(self, original_backend: str = None):
        self.model = None
        self.model_name = None
        self.device = None
        self.n_gpu_layers = -1
        self.n_ctx = 2048
        self.verbose = True
        self.main_gpu = 0
        self.chat_template = None
        self.hf_tokenizer = None
        self.force_cuda = original_backend in ("nvidia", "cuda")
        if self.force_cuda:
            print("DEBUG: GGUF model will use CUDA backend (forced by --backend nvidia)")
    
    def load_model(self, model_name: str, **kwargs) -> None:
        """Load the model."""
        pass
    
    def generate(self, prompt: str, max_tokens: Optional[int] = None, 
                 temperature: float = 0.7, top_p: float = 1.0,
                 stop: Optional[list] = None) -> str:
        """Generate text non-streaming."""
        pass
    
    def generate_chat(self, messages: List[Dict], max_tokens: Optional[int] = None,
                      temperature: float = 0.7, top_p: float = 1.0,
                      stop: Optional[List[str]] = None, tools: Optional[List] = None,
                      response_format: Optional[Dict] = None) -> str:
        """Generate chat completion non-streaming."""
        pass
    
    async def generate_chat_stream(self, messages: List[Dict], max_tokens: Optional[int] = None,
                                    temperature: float = 0.7, top_p: float = 1.0,
                                    stop: Optional[List[str]] = None, tools: Optional[List] = None,
                                    response_format: Optional[Dict] = None):
        """Generate chat completion streaming."""
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
