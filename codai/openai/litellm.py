"""
LiteLLM Backend - OpenAI-compatible chat completion using litellm.

This module provides a litellm-based backend for the OpenAI-compatible API,
used when --parser litellm is specified.
"""

import os
import json
import re
from typing import List, Dict, Any, Optional, AsyncGenerator, Union

try:
    import litellm
    from litellm import acompletion, completion
    from litellm.exceptions import (
        AuthenticationError,
        BadRequestError,
        RateLimitError,
        ServiceUnavailableError,
        ContextWindowExceededError,
    )
    LITELLM_AVAILABLE = True
    
    # Map litellm exceptions to OpenAI error codes
    ERROR_CODE_MAP = {
        AuthenticationError: {"code": 401, "type": "invalid_api_key"},
        BadRequestError: {"code": 400, "type": "invalid_request_error"},
        RateLimitError: {"code": 429, "type": "rate_limit_error"},
        ServiceUnavailableError: {"code": 503, "type": "service_unavailable"},
        ContextWindowExceededError: {"code": 400, "type": "context_window_exceeded"},
    }
except ImportError:
    LITELLM_AVAILABLE = False
    litellm = None
    completion = None
    acompletion = None
    ERROR_CODE_MAP = {}


def get_error_response(status_code: int, message: str, error_type: str = "internal_error") -> Dict:
    """Create an OpenAI-compatible error response."""
    return {
        "error": {
            "message": message,
            "type": error_type,
            "code": status_code,
        }
    }


class LiteLLMBackend:
    """
    LiteLLM-based backend for OpenAI-compatible chat completions.
    
    Used when --parser litellm is specified to leverage litellm's
    standardized response format and broader model support.
    """
    
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        api_base: Optional[str] = None,  # Add api_base parameter
        context_window: int = 4096,
        model_manager: Optional[Any] = None,
        **kwargs
    ):
        """
        Initialize the LiteLLM backend.
        
        Args:
            model: Model name to use (e.g., "gpt-3.5-turbo", "ollama/llama2")
            api_key: API key for the model provider
            base_url: Custom base URL for OpenAI-compatible APIs
            api_base: API base URL (alternative to base_url, e.g., "http://localhost:11434/v1")
            context_window: Maximum context window size for rate limit headers
            model_manager: Reference to MultiModelManager for resolving aliases
        """
        self.model = model
        # Use provided API key, or generate a fake one if not provided
        # This allows litellm to proceed without requiring an API key
        self.api_key = api_key if api_key else "fake-key-for-local-testing"
        self.base_url = base_url or api_base  # Use either base_url or api_base
        self.context_window = context_window
        self.model_manager = model_manager
        self.tool_parser = None  # Coderai's tool parser for post-processing
        self.tools_schema = {}  # Tools schema for coderai parser
        
        # Configure litellm
        if self.base_url:
            litellm.base_url = self.base_url
        if self.api_key:
            litellm.api_key = self.api_key
    
    def normalize_model_name(self, model: str) -> str:
        """
        Normalize model name for litellm.
        
        Always formats as: openai/{provider}/{model}
        - If provider is detected from known patterns, use it
        - If model has / (e.g. HuggingFace org/model), detect or default to huggingface
        - If provider unknown, use "coderai" as default
        
        Args:
            model: Original model name (may be an alias)
            
        Returns:
            Normalized model name: openai/{provider}/{model}
        """
        print(f"DEBUG litellm: normalize_model_name input: {model}")
        
        # First, resolve alias to actual model name if we have a model manager
        resolved_model = self._resolve_model_alias(model)
        print(f"DEBUG litellm: After alias resolution: {resolved_model}")
        
        # Known litellm providers
        known_providers = ['openai', 'anthropic', 'gemini', 'meta', 'mistral', 'cohere', 
                         'ai21', 'bedrock', 'azure', 'ollama', 'huggingface', 'deepseek', 
                         'qwen', 'sagemaker', 'vertex', 'aiplatform', 'vllm', 'tgi']
        
        # Check if there's an existing provider prefix (contains /)
        if '/' in resolved_model:
            parts = resolved_model.split('/')
            prefix = parts[0].lower()
            if prefix in known_providers:
                # Valid provider, reformat as openai/{provider}/{model}
                model_part = '/'.join(parts[1:])
                result = f"openai/{prefix}/{model_part}"
                print(f"DEBUG litellm: Known provider '{prefix}', returning: {result}")
                return result
            # Otherwise, it's likely a HuggingFace org/model path
            result = f"openai/huggingface/{resolved_model}"
            print(f"DEBUG litellm: HuggingFace org/model, returning: {result}")
            return result
        
        # No provider prefix - detect provider from model name pattern
        provider_map = {
            # OpenAI models
            'gpt-': 'openai',
            'gpt3': 'openai',
            'gpt4': 'openai',
            # Anthropic models
            'claude': 'anthropic',
            # Google models
            'gemini': 'gemini',
            'palm': 'gemini',
            # Meta/Llama models
            'llama': 'meta',
            'llama2': 'meta',
            'llama3': 'meta',
            # Mistral models
            'mistral': 'mistral',
            # AWS models
            'amazon': 'bedrock',
            # Azure models
            'azure': 'azure',
            # Cohere models
            'cohere': 'cohere',
            # AI21 models
            'ai21': 'ai21',
            # Local/Ollama models
            'ollama': 'ollama',
            # HuggingFace models
            'hf': 'huggingface',
            # DeepSeek models
            'deepseek': 'deepseek',
            # Qwen models
            'qwen': 'qwen',
        }
        
        model_lower = resolved_model.lower()
        
        # Check for known patterns
        for pattern, provider in provider_map.items():
            if model_lower.startswith(pattern):
                result = f"openai/{provider}/{resolved_model}"
                print(f"DEBUG litellm: Detected provider '{provider}', returning: {result}")
                return result
        
        # Default: use "coderai" as provider for unknown models
        result = f"openai/coderai/{resolved_model}"
        print(f"DEBUG litellm: Unknown provider, using 'coderai', returning: {result}")
        return result
    
    def _resolve_model_alias(self, model: str) -> str:
        """
        Resolve model alias to actual model name.
        
        Handles aliases like "default", "image", "audio", "tts", or custom aliases
        registered via --model-alias.
        
        Args:
            model: Model name or alias
            
        Returns:
            Resolved actual model name
        """
        if not self.model_manager:
            print(f"DEBUG litellm: No model_manager, returning model as-is: {model}")
            return model
        
        # Check if model is "default" or empty - use default_model
        if not model or model == "default":
            default_model = getattr(self.model_manager, 'default_model', None)
            print(f"DEBUG litellm: Resolving 'default' alias to: {default_model}")
            if default_model:
                return default_model
            return model
        
        # Check if model is "image" - get first image model
        if model == "image":
            image_models = getattr(self.model_manager, 'image_models', [])
            resolved = image_models[0] if image_models else model
            print(f"DEBUG litellm: Resolving 'image' alias to: {resolved}")
            return resolved
        
        # Check if model is "audio" - get first audio model
        if model == "audio":
            audio_models = getattr(self.model_manager, 'audio_models', [])
            resolved = audio_models[0] if audio_models else model
            print(f"DEBUG litellm: Resolving 'audio' alias to: {resolved}")
            return resolved
        
        # Check if model is "tts" - get tts model
        if model == "tts":
            tts_model = getattr(self.model_manager, 'tts_model', None)
            print(f"DEBUG litellm: Resolving 'tts' alias to: {tts_model}")
            if tts_model:
                return tts_model
            return model
        
        # Check custom aliases registered via --model-alias
        model_aliases = getattr(self.model_manager, 'model_aliases', {})
        if model in model_aliases:
            resolved = model_aliases[model]
            print(f"DEBUG litellm: Resolving alias '{model}' to: {resolved}")
            return resolved
        
        print(f"DEBUG litellm: Model '{model}' is not an alias, returning as-is")
        return model
            
    def _convert_messages(self, messages: List[Dict]) -> List[Dict]:
        """Convert OpenAI message format to litellm format."""
        converted = []
        for msg in messages:
            # Handle both 'content' and 'tool' role variations
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Handle tool calls
            if "tool_calls" in msg and msg["tool_calls"]:
                tool_calls = []
                for tc in msg["tool_calls"]:
                    if isinstance(tc, dict):
                        tool_calls.append({
                            "id": tc.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": tc.get("function", {}).get("name", ""),
                                "arguments": tc.get("function", {}).get("arguments", "")
                            }
                        })
                # Add the assistant message with tool calls
                converted.append({
                    "role": role,
                    "content": content,
                    "tool_calls": tool_calls
                })
            elif msg.get("tool_call_id"):
                # Tool result message
                converted.append({
                    "role": role,
                    "content": content,
                    "tool_call_id": msg.get("tool_call_id")
                })
            else:
                converted.append({
                    "role": role,
                    "content": content
                })
                
        return converted
    
    def _calculate_tokens_remaining(self, prompt_tokens: int) -> int:
        """Calculate remaining context window tokens."""
        return max(0, self.context_window - prompt_tokens)
    
    def _create_response_headers(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int
    ) -> Dict[str, str]:
        """Create rate limit headers for the response."""
        remaining = self._calculate_tokens_remaining(prompt_tokens)
        
        return {
            "x-ratelimit-limit-tokens": str(self.context_window),
            "x-ratelimit-remaining-tokens": str(remaining),
            "x-ratelimit-limit-requests": "60",  # Default, can be overridden
            "x-ratelimit-remaining-requests": "60",
            "x-ratelimit-limit-tokens-usage": str(total_tokens),
            "x-ratelimit-remaining-tokens-usage": str(completion_tokens),
            "x-ratelimit-token-usage": str(total_tokens),
        }
    
    def _parse_tool_calls(self, response: Dict) -> List[Dict]:
        """Parse tool calls from litellm response."""
        tool_calls = []
        
        # Check for tool calls in the response
        if "choices" in response and response["choices"]:
            choice = response["choices"][0]
            if "message" in choice:
                msg = choice["message"]
                if "tool_calls" in msg:
                    for tc in msg["tool_calls"]:
                        if isinstance(tc, dict):
                            tool_calls.append({
                                "id": tc.get("id", f"call_{id(tc)}"),
                                "type": "function",
                                "function": {
                                    "name": tc.get("function", {}).get("name", ""),
                                    "arguments": tc.get("function", {}).get("arguments", "{}")
                                }
                            })
                                
        return tool_calls
    
    def _extract_content(self, response: Dict) -> str:
        """Extract content from litellm response."""
        if "choices" in response and response["choices"]:
            choice = response["choices"][0]
            if "message" in choice:
                return choice["message"].get("content", "") or ""
        return ""
    
    def _create_chunk(
        self,
        content: str,
        role: str = "assistant",
        tool_calls: Optional[List[Dict]] = None,
        finish_reason: Optional[str] = None,
        index: int = 0
    ) -> Dict:
        """Create a chat completion chunk."""
        chunk = {
            "id": f"chatcmpl-{id(content)}",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": self.model,
            "choices": [{
                "index": index,
                "delta": {
                    "role": role,
                    "content": content
                },
                "finish_reason": finish_reason
            }]
        }
        
        if tool_calls:
            chunk["choices"][0]["delta"]["tool_calls"] = tool_calls
            
        return chunk
    
    async def chat_completion(
        self,
        messages: List[Dict],
        model: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[Union[str, Dict]] = "auto",
        stream: bool = False,
        tool_parser=None,  # Add coderai's tool parser for post-processing
        **kwargs
    ) -> Union[Dict, AsyncGenerator]:
        """
        Generate a chat completion using litellm.
        
        Args:
            messages: List of message dictionaries
            model: Optional model override
            temperature: Sampling temperature
            top_p: Top-p sampling
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            tools: Tool definitions
            tool_choice: Tool choice mode
            stream: Whether to stream the response
            tool_parser: Optional coderai tool parser for post-processing tool calls
            
        Returns:
            Response dict or async generator for streaming
        """
        if not LITELLM_AVAILABLE:
            raise RuntimeError("litellm is not installed. Run: pip install litellm")
        
        # Store tool_parser for post-processing
        self.tool_parser = tool_parser
        
        # Convert tools to coderai schema format if tools provided
        if tools:
            self.tools_schema = {}
            for tool in tools:
                if isinstance(tool, dict) and 'function' in tool:
                    func = tool.get('function', {})
                    self.tools_schema[func.get('name', '')] = {
                        'description': func.get('description', ''),
                        'parameters': func.get('parameters', {})
                    }
        
        # Prepare the model - normalize name for litellm
        use_model = self.normalize_model_name(model or self.model)
        
        # Convert messages to litellm format
        litellm_messages = self._convert_messages(messages)
        
        # Prepare completion arguments
        completion_args = {
            "model": use_model,
            "messages": litellm_messages,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }
        
        if max_tokens:
            completion_args["max_tokens"] = max_tokens
        if stop:
            completion_args["stop"] = stop
        if tools:
            completion_args["tools"] = tools
        if tool_choice:
            completion_args["tool_choice"] = tool_choice
            
        # Add any additional kwargs
        completion_args.update(kwargs)
        
        if stream:
            return self._stream_response(completion_args)
        else:
            return await self._get_response(completion_args)
    
    async def _get_response(self, completion_args: Dict) -> Dict:
        """Get a non-streaming response from litellm."""
        try:
            response = await acompletion(**completion_args)
            return self._process_response(response)
        except Exception as e:
            return self._handle_error(e)
    
    def _process_response(self, response: Any) -> Dict:
        """Process litellm response into OpenAI format."""
        # Convert litellm response to OpenAI format
        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": response.usage.get("prompt_tokens", 0),
                "completion_tokens": response.usage.get("completion_tokens", 0),
                "total_tokens": response.usage.get("total_tokens", 0),
            }
        
        # Extract message content
        content = ""
        tool_calls = []
        
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "message"):
                msg = choice.message
                content = msg.content or ""
                
                # Handle tool calls
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        if hasattr(tc, "function"):
                            func = tc.function
                            tool_calls.append({
                                "id": tc.id or f"call_{id(tc)}",
                                "type": "function",
                                "function": {
                                    "name": func.name,
                                    "arguments": func.arguments
                                }
                            })
        
        # Build OpenAI-compatible response
        result = {
            "id": f"chatcmpl-{id(response)}",
            "object": "chat.completion",
            "created": getattr(response, "created", 0),
            "model": getattr(response, "model", self.model),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": getattr(response.choices[0], "finish_reason", None) if hasattr(response, "choices") and response.choices else None,
            }],
            "usage": usage,
        }
        
        if tool_calls:
            result["choices"][0]["message"]["tool_calls"] = tool_calls
        
        # Use coderai's tool parser for post-processing if available
        if self.tool_parser and content:
            # Try to extract tool calls using coderai's parser
            try:
                # Convert tools to the format expected by coderai parser
                tools_schema = {}
                if hasattr(self, 'tools_schema') and self.tools_schema:
                    tools_schema = self.tools_schema
                
                # Use coderai parser to extract tool calls from content
                parsed_tool_calls = self.tool_parser.extract_tool_calls(content, tools_schema) if hasattr(self.tool_parser, 'extract_tool_calls') else None
                
                if parsed_tool_calls:
                    # Replace tool calls with coderai-parsed versions
                    result["choices"][0]["message"]["tool_calls"] = parsed_tool_calls
                    # Strip tool tags from content
                    if hasattr(self.tool_parser, 'strip_tool_calls_from_content'):
                        clean_content = self.tool_parser.strip_tool_calls_from_content(content)
                        result["choices"][0]["message"]["content"] = clean_content
            except Exception as e:
                print(f"DEBUG litellm: Coderai parser post-processing error: {e}")
        
        return result
    
    async def _stream_response(self, completion_args: Dict) -> AsyncGenerator:
        """Stream response from litellm."""
        try:
            response = await acompletion(**completion_args)
            
            async for chunk in response:
                yield self._process_stream_chunk(chunk)
                
        except Exception as e:
            error_resp = self._handle_error(e)
            yield error_resp
    
    def _process_stream_chunk(self, chunk: Any) -> Dict:
        """Process a streaming chunk from litellm."""
        content = ""
        tool_calls = []
        finish_reason = None
        
        if hasattr(chunk, "choices") and chunk.choices:
            choice = chunk.choices[0]
            if hasattr(choice, "delta"):
                delta = choice.delta
                content = delta.content or ""
                
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    for tc in delta.tool_calls:
                        if hasattr(tc, "function"):
                            func = tc.function
                            tool_calls.append({
                                "id": tc.id or f"call_{id(tc)}",
                                "type": "function", 
                                "function": {
                                    "name": func.name,
                                    "arguments": func.arguments
                                }
                            })
                            
            finish_reason = getattr(choice, "finish_reason", None)
        
        result = {
            "id": f"chatcmpl-{id(chunk)}",
            "object": "chat.completion.chunk",
            "created": getattr(chunk, "created", 0),
            "model": getattr(chunk, "model", self.model),
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason,
            }]
        }
        
        if content:
            result["choices"][0]["delta"]["content"] = content
            
        if tool_calls:
            result["choices"][0]["delta"]["tool_calls"] = tool_calls
        
        # Accumulate content for coderai parser post-processing at end of stream
        if content:
            if not hasattr(self, '_accumulated_content'):
                self._accumulated_content = ""
            self._accumulated_content += content
        
        # Use coderai's tool parser for post-processing if available and this is final chunk
        if self.tool_parser and hasattr(self, '_accumulated_content') and self._accumulated_content:
            if finish_reason == 'stop':
                try:
                    # Use coderai parser to extract tool calls from accumulated content
                    tools_schema = getattr(self, 'tools_schema', {})
                    if hasattr(self.tool_parser, 'extract_tool_calls'):
                        parsed_tool_calls = self.tool_parser.extract_tool_calls(self._accumulated_content, tools_schema)
                        if parsed_tool_calls:
                            # Add tool calls to final chunk
                            result["choices"][0]["delta"]["tool_calls"] = parsed_tool_calls
                            # Strip tool tags from content
                            if hasattr(self.tool_parser, 'strip_tool_calls_from_content'):
                                clean_content = self.tool_parser.strip_tool_calls_from_content(self._accumulated_content)
                                result["choices"][0]["delta"]["content"] = clean_content
                            # Clear accumulated content after processing
                            self._accumulated_content = ""
                except Exception as e:
                    print(f"DEBUG litellm: Coderai parser stream post-processing error: {e}")
        
        return result
    
    def _handle_error(self, exception: Exception) -> Dict:
        """Handle litellm exceptions and convert to OpenAI format."""
        error_info = ERROR_CODE_MAP.get(type(exception), {"code": 500, "type": "internal_error"})
        
        return {
            "error": {
                "message": str(exception),
                "type": error_info["type"],
                "code": error_info["code"],
            }
        }
    
    def parse_qwen_tool_calls(self, text: str) -> List[Dict]:
        """
        Parse Qwen-style tool calls from text content.
        
        Handles both <tool> and <tool_call> tags, with support for:
        - JSON format: <tool>{"name": "func", "arguments": {...}}</tool>
        - Coder style: <tool=func><parameter=key>value</parameter></tool>
        
        Returns a list of tool call dictionaries in OpenAI format.
        """
        tool_calls = []
        
        # 1. IMMEDIATE REPETITION GUARD - handle looping
        if text.count('<tool') > 1:
            parts = re.split(r'<(?:tool|tool_call)', text, flags=re.IGNORECASE)
            text = f"<tool{parts[1]}" if len(parts) > 1 else text
        
        # 2. Pre-cleaning (remove thinking tags)
        clean_text = re.sub(r'<\|.*?\|>|<(?:thought|think)>.*?((?:</(?:thought|think)>)|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # 3. MATCH BOTH <tool> AND <tool_call>
        tag_pattern = r'<(?:tool|tool_call)>(.*?)(?:</(?:tool|tool_call)>|$)'
        matches = re.findall(tag_pattern, clean_text, re.DOTALL | re.IGNORECASE)
        
        # If no tags found but text looks like JSON, try whole text
        if not matches and '{' in clean_text and '"name"' in clean_text:
            matches = [clean_text]
        
        for block in matches:
            block = block.strip()
            if not block:
                continue
            
            # Clean markdown and detect partial JSON
            json_str = re.sub(r'```(?:json)?\s*(.*?)\s*```', r'\1', block, flags=re.DOTALL).strip()
            
            # Recovery of unclosed JSON
            if json_str.startswith('{') and not json_str.endswith('}'):
                json_str += '}'
            
            try:
                data = json.loads(json_str)
                if 'name' in data:
                    tool_calls.append({
                        "id": f"call_{id(data)}",
                        "type": "function",
                        "function": {
                            "name": data['name'],
                            "arguments": json.dumps(data.get('arguments', {} or data.get('parameters', {})))
                        }
                    })
                    break  # Circuit breaker after first valid call
            except json.JSONDecodeError:
                # Fallback: try regex extraction
                name_match = re.search(r'"name":\s*"([^"]+)"', json_str)
                if name_match:
                    tool_calls.append({
                        "id": f"call_{id(name_match)}",
                        "type": "function",
                        "function": {
                            "name": name_match.group(1),
                            "arguments": "{}"
                        }
                    })
                    break
        
        # 4. CODER STYLE FALLBACK
        if not tool_calls:
            pattern = r'<(?:function|tool|call)=([^>]+)>(.*?)(?:</(?:function|tool|call|tool_call)>|$)'
            for name, body in re.findall(pattern, clean_text, re.DOTALL | re.IGNORECASE):
                params = re.findall(r'<parameter=([^>]+)>(.*?)</parameter>', body, re.DOTALL)
                args = {}
                for k, v in params:
                    val = v.strip()
                    try:
                        args[k.strip()] = json.loads(val)
                    except:
                        args[k.strip()] = val
                tool_calls.append({
                    "id": f"call_{id(args)}",
                    "type": "function",
                    "function": {
                        "name": name.strip(),
                        "arguments": json.dumps(args)
                    }
                })
                break  # Circuit breaker
        
        return tool_calls
    
    def strip_tool_tags(self, text: str) -> str:
        """Strip tool call tags from text, leaving only the content."""
        # Remove <tool>...</tool> and <tool_call>...</tool_call> blocks
        clean = re.sub(r'<tool[^>]*>.*?</tool[^>]*>', '', text, flags=re.DOTALL | re.IGNORECASE)
        clean = re.sub(r'<tool_call[^>]*>.*?</tool_call[^>]*>', '', clean, flags=re.DOTALL | re.IGNORECASE)
        clean = re.sub(r'<function[^>]*>.*?</function[^>]*>', '', clean, flags=re.DOTALL | re.IGNORECASE)
        return clean.strip()
    
    def get_rate_limit_headers(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> Dict[str, str]:
        """Get rate limit headers based on current usage."""
        total = prompt_tokens + completion_tokens
        return self._create_response_headers(prompt_tokens, completion_tokens, total)


# Default instance
default_litellm_backend: Optional[LiteLLMBackend] = None


def get_litellm_backend(
    model: str = "gpt-3.5-turbo",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    api_base: Optional[str] = None,  # Add api_base parameter
    context_window: int = 4096,
    model_manager: Optional[Any] = None,
    **kwargs
) -> LiteLLMBackend:
    """Get or create the default LiteLLM backend instance."""
    global default_litellm_backend
    
    # Always create a new instance with the provided model_manager
    # This ensures aliases are resolved correctly on each call
    default_litellm_backend = LiteLLMBackend(
        model=model,
        api_key=api_key,
        base_url=base_url,
        api_base=api_base,
        context_window=context_window,
        model_manager=model_manager,
        **kwargs
    )
        
    return default_litellm_backend


def set_litellm_backend(backend: LiteLLMBackend) -> None:
    """Set the default LiteLLM backend instance."""
    global default_litellm_backend
    default_litellm_backend = backend
