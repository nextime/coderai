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

# codai.models - Model parsing and templates
from .manager import (
    ModelManager,
    WhisperServerManager,
    MultiModelManager,
    model_manager,
    multi_model_manager,
)
from .parser import (
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
    OpenAIFormatter,
    ToolCallParser,
    ModelParserAdapter,
    filter_repetition,
    filter_malformed_content,
    cleanup_control_tokens,
    validate_json_complete,
    format_tools_for_prompt,
)

from .templates import AgenticTemplateManager

__all__ = [
    'ModelManager',
    'WhisperServerManager',
    'MultiModelManager',
    'model_manager',
    'multi_model_manager',
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
    'OpenAIFormatter',
    'ToolCallParser',
    'ModelParserAdapter',
    'AgenticTemplateManager',
    'filter_repetition',
    'filter_malformed_content',
    'cleanup_control_tokens',
    'validate_json_complete',
    'format_tools_for_prompt',
]