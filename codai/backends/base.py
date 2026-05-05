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
                 stop: Optional[List[str]] = None,
                 repeat_penalty: float = 1.0,
                 presence_penalty: float = 0.0,
                 frequency_penalty: float = 0.0) -> str:
        """Generate text non-streaming."""
        pass
    
    @abstractmethod
    def generate_stream(self, prompt: str, max_tokens: Optional[int] = None,
                        temperature: float = 0.7, top_p: float = 1.0,
                        stop: Optional[List[str]] = None,
                        repeat_penalty: float = 1.0,
                        presence_penalty: float = 0.0,
                        frequency_penalty: float = 0.0) -> AsyncGenerator[str, None]:
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
    
    def get_context_size(self) -> int:
        """Return the model's context window size."""
        return 2048  # Default fallback