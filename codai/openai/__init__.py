# codai.openai - OpenAI-compatible API implementations
from .litellm import (
    LiteLLMBackend,
    get_litellm_backend,
    set_litellm_backend,
    LITELLM_AVAILABLE,
)

__all__ = [
    'LiteLLMBackend',
    'get_litellm_backend',
    'set_litellm_backend',
    'LITELLM_AVAILABLE',
]
