"""Pydantic models for API."""

import time
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, ConfigDict


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
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.7
    top_p: float = 1.0
    n: int = 1
    max_tokens: Optional[int] = None
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repeat_penalty: float = 1.0
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, Dict]] = "auto"
    # Extra fields that clients may send but we ignore
    seed: Optional[int] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    response_format: Optional[Dict] = None
    user: Optional[str] = None
    # Enable thinking/reasoning mode for supported models
    enable_thinking: Optional[bool] = False
    
    model_config = ConfigDict(extra="allow")  # Allow extra fields to prevent 422 errors


class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
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


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelInfo]
