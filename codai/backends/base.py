"""Base classes for model backends."""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Optional


class ModelBackend(ABC):
    """Abstract base class for model backends."""
    
    @abstractmethod
    def load_model(self, model_name: str, **kwargs) -> None:
        """Load the model."""
        pass
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: Optional[int] = None, 
                 temperature: float = 0.7, top_p: float = 1.0,
                 stop: Optional[List[str]] = None) -> str:
        """Generate text non-streaming."""
        pass
    
    @abstractmethod
    def generate_stream(self, prompt: str, max_tokens: Optional[int] = None,
                        temperature: float = 0.7, top_p: float = 1.0,
                        stop: Optional[List[str]] = None) -> AsyncGenerator[str, None]:
        """Generate text in streaming fashion."""
        pass
    
    @abstractmethod
    def format_messages(self, messages) -> str:
        """Format messages into a prompt string."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Return the loaded model name."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources."""
        pass
