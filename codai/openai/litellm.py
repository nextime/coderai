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

"""LiteLLM backend wrapper for codai.

This module wraps the litellm library so that the text endpoint can forward
requests to any model provider supported by LiteLLM (Ollama, OpenAI, Anthropic,
etc.) while still returning responses in the standard OpenAI format.
"""

import re
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

try:
    import litellm
    litellm.drop_params = True  # silently drop unsupported params
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False


class LiteLLMBackend:
    """Wraps litellm.acompletion with the interface expected by codai's text endpoint."""

    def __init__(
        self,
        model: str,
        api_key: str,
        api_base: Optional[str],
        context_window: int = 8192,
        model_manager=None,
    ):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.context_window = context_window
        self.model_manager = model_manager

    def _litellm_model(self, model: str) -> str:
        """Return the model string in the format litellm expects."""
        if model.startswith('ollama:'):
            return model  # litellm already understands "ollama/<name>"
        return model

    async def chat_completion(
        self,
        messages: List[Dict],
        model: str,
        temperature: Optional[float] = 1.0,
        top_p: Optional[float] = 1.0,
        max_tokens: Optional[int] = None,
        stop=None,
        tools=None,
        tool_choice=None,
        stream: bool = False,
        tool_parser=None,
    ):
        """Call litellm.acompletion and return either a full response dict or
        an async generator of chunk dicts (when stream=True).
        """
        if not LITELLM_AVAILABLE:
            raise RuntimeError("litellm is not installed. Run: pip install litellm")

        kwargs: Dict[str, Any] = {
            "model": self._litellm_model(model),
            "messages": messages,
            "api_key": self.api_key,
            "stream": stream,
        }
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if stop:
            kwargs["stop"] = stop if isinstance(stop, list) else [stop]
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice

        if stream:
            return self._stream(kwargs)
        else:
            response = await litellm.acompletion(**kwargs)
            return response.model_dump() if hasattr(response, 'model_dump') else dict(response)

    async def _stream(self, kwargs: Dict) -> AsyncGenerator[Dict, None]:
        response = await litellm.acompletion(**kwargs)
        async for chunk in response:
            yield chunk.model_dump() if hasattr(chunk, 'model_dump') else dict(chunk)

    def get_rate_limit_headers(self, prompt_tokens: int, completion_tokens: int) -> Dict:
        return {
            "x-ratelimit-limit-requests": "1000",
            "x-ratelimit-remaining-requests": "999",
            "x-ratelimit-limit-tokens": str(self.context_window),
            "x-ratelimit-remaining-tokens": str(
                max(0, self.context_window - prompt_tokens - completion_tokens)
            ),
        }

    # ------------------------------------------------------------------
    # Qwen-specific helpers (tool calls embedded in <tool_call>…</tool_call>)
    # ------------------------------------------------------------------

    _QWEN_TOOL_PATTERN = re.compile(
        r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.DOTALL
    )
    _QWEN_TAG_PATTERN = re.compile(
        r'<tool_call>.*?</tool_call>', re.DOTALL
    )

    def parse_qwen_tool_calls(self, content: str) -> List[Dict]:
        """Extract tool calls embedded as <tool_call>{…}</tool_call> tags."""
        import json
        calls = []
        for m in self._QWEN_TOOL_PATTERN.finditer(content):
            try:
                data = json.loads(m.group(1))
                calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": data.get("name", ""),
                        "arguments": json.dumps(data.get("arguments", {})),
                    },
                })
            except (json.JSONDecodeError, KeyError):
                continue
        return calls

    def strip_tool_tags(self, content: str) -> str:
        """Remove <tool_call>…</tool_call> blocks from content."""
        return self._QWEN_TAG_PATTERN.sub('', content).strip()


def get_litellm_backend(
    model: str,
    api_key: str,
    api_base: Optional[str] = None,
    context_window: int = 8192,
    model_manager=None,
) -> LiteLLMBackend:
    """Return a LiteLLMBackend instance for the given model."""
    return LiteLLMBackend(
        model=model,
        api_key=api_key,
        api_base=api_base,
        context_window=context_window,
        model_manager=model_manager,
    )
