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

# OpenAI-compatible backends
from .openai.litellm import (
    LiteLLMBackend,
    get_litellm_backend,
    set_litellm_backend,
    LITELLM_AVAILABLE,
)

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
    'LiteLLMBackend',
    'get_litellm_backend',
    'set_litellm_backend',
    'LITELLM_AVAILABLE',
]
