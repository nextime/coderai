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

"""Pydantic models for API."""

import time
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


class ToolFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict] = None


class Tool(BaseModel):
    type: str = "function"
    function: ToolFunction


class ChatMessage(BaseModel):
    role: str
    content: Optional[Union[str, List[Dict]]] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None
    cache_control: Optional[Dict] = None  # OpenAI-style: {"type": "ephemeral"}
    
    @field_validator('content', mode='before')
    @classmethod
    def convert_content_array_to_string(cls, v):
        """Convert multipart content array to string for compatibility."""
        if v is None:
            return None
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            # Handle multipart content array format (e.g., from KiloCode)
            # Format: [{"type": "text", "text": "..."}, {"type": "text", "text": "..."}]
            parts = []
            for item in v:
                if isinstance(item, dict):
                    if item.get('type') == 'text' and 'text' in item:
                        parts.append(item['text'])
                    else:
                        # Handle other content types (image_url, etc.) by converting to placeholder
                        parts.append(f"[{item.get('type', 'unknown')} content]")
                else:
                    parts.append(str(item))
            return '\n'.join(parts)
        return str(v)


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="Text/chat model id to use.")
    messages: List[ChatMessage] = Field(..., description="Conversation messages (roles: system/user/assistant/tool). Content may include text and image parts for vision models.")
    temperature: float = Field(0.7, description="Sampling temperature; higher = more random.")
    top_p: float = Field(1.0, description="Nucleus sampling probability mass.")
    n: int = Field(1, description="Number of completions to generate.")
    max_tokens: Optional[int] = Field(None, description="Max tokens to generate (model default if omitted).")
    max_completion_tokens: Optional[int] = Field(None, description="Newer OpenAI alias for max_tokens; used when max_tokens is omitted.")
    stream: bool = Field(False, description="Stream the response as Server-Sent Events.")
    stop: Optional[Union[str, List[str]]] = Field(None, description="Stop sequence(s) that end generation.")
    presence_penalty: float = Field(0.0, description="Penalize tokens already present (encourages new topics).")
    frequency_penalty: float = Field(0.0, description="Penalize frequent tokens (reduces repetition).")
    repeat_penalty: float = Field(1.0, description="llama.cpp repetition penalty.")
    tools: Optional[List[Tool]] = Field(None, description="Tool/function definitions the model may call.")
    tool_choice: Optional[Union[str, Dict]] = Field("auto", description="Tool selection: 'auto', 'none', or a specific tool.")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility.")
    logprobs: Optional[bool] = Field(None, description="Return token log-probabilities (if supported).")
    top_logprobs: Optional[int] = Field(None, description="Number of top log-probs to return per token.")
    response_format: Optional[Dict] = Field(None, description="Structured-output format, e.g. {'type': 'json_object'}.")
    user: Optional[str] = Field(None, description="Opaque end-user identifier (passthrough).")
    enable_thinking: Optional[bool] = Field(False, description="Enable thinking/reasoning mode for models that support it.")
    reasoning: Optional[Dict] = Field(None, description="Reasoning controls. OpenRouter-style; set {'exclude': true} to drop the model's thinking from the response.")
    reasoning_effort: Optional[str] = Field(None, description="OpenAI-style reasoning effort ('low'|'medium'|'high'); 'none'/'off' suppresses reasoning output.")
    suppress_reasoning: Optional[bool] = Field(None, description="Drop the model's reasoning/thinking from the response instead of returning it as a separate field.")

    model_config = ConfigDict(extra="allow")  # Allow extra fields to prevent 422 errors

    @model_validator(mode="after")
    def _alias_max_completion_tokens(self):
        """Accept the newer OpenAI ``max_completion_tokens`` as an alias for
        ``max_tokens``. ``max_tokens`` wins when both are sent."""
        if self.max_tokens is None and self.max_completion_tokens is not None:
            self.max_tokens = self.max_completion_tokens
        return self


class CompletionRequest(BaseModel):
    model: str = Field(..., description="Text model id to use.")
    prompt: Union[str, List[str]] = Field(..., description="Prompt text (or list of prompts) to complete.")
    temperature: float = 0.7
    top_p: float = 1.0
    n: int = 1
    max_tokens: Optional[int] = None
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repeat_penalty: float = 1.0
    # Extra fields that clients may send but we ignore
    seed: Optional[int] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    best_of: Optional[int] = None
    echo: Optional[bool] = None
    user: Optional[str] = None
    
    model_config = ConfigDict(extra="allow")  # Allow extra fields to prevent 422 errors


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "huggingface"
    type: Optional[str] = None          # e.g. "text", "image", "video", "audio", "tts", "vision", "embedding"
    capabilities: Optional[List[str]] = None  # list of capability strings
    backend: Optional[str] = None
    model_path: Optional[str] = None
    server_path: Optional[str] = None
    alias: Optional[str] = None
    port: Optional[int] = None
    gpu_device: Optional[int] = None
    load_mode: Optional[str] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelInfo]
