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

# LiteLLM backend (requires litellm package)
try:
    from .litellm_backend import (
        LiteLLMBackend,
        get_litellm_backend,
        set_litellm_backend,
        LITELLM_AVAILABLE,
    )
    _LITELLM_IMPORT_ERROR = None
except ImportError as e:
    _LITELLM_IMPORT_ERROR = str(e)
    LiteLLMBackend = None
    get_litellm_backend = None
    set_litellm_backend = None
    LITELLM_AVAILABLE = False

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
