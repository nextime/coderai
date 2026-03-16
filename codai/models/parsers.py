import time
import uuid


class OpenAIFormatter:
    """Formatter for standardizing chat completion responses in OpenAI format.
    
    This class provides final sanitization of responses before sending them
    to clients. It processes the output of the internal parser and formats
    them into proper OpenAI-compatible responses.
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.id = f"chatcmpl-{uuid.uuid4()}"
    
    def format_full(self, text: str, prompt_tokens: int, completion_tokens: int, tool_calls=None) -> dict:
        """Format a standard (non-streaming) response.
        
        Args:
            text: The generated text content
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion
            tool_calls: Optional list of tool calls to include
            
        Returns:
            Dictionary representation of the response
        """
        message = {
            "role": "assistant",
            "content": text if not tool_calls else None,
        }
        if tool_calls:
            message["tool_calls"] = tool_calls
        
        choice = {
            "index": 0,
            "message": message,
            "finish_reason": "tool_calls" if tool_calls else "stop",
        }
        
        return {
            "id": self.id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [choice],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "provider": {
                "provider_name": "coderai",
                "provider_id": "coderai",
            },
        }
    
    def format_chunk(self, delta_text: str, is_final: bool = False, usage: dict = None) -> dict:
        """Format a streaming chunk response.
        
        Args:
            delta_text: The incremental text content for this chunk
            is_final: Whether this is the final chunk
            usage: Optional usage information (typically only sent on final chunk)
            
        Returns:
            Dictionary representation of the chunk
        """
        delta = {
            "content": delta_text,
            "role": "assistant",
        }
        
        choice = {
            "index": 0,
            "delta": delta,
            "finish_reason": "stop" if is_final else None,
        }
        
        chunk = {
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [choice],
        }
        
        if usage and is_final:
            chunk["usage"] = usage
            
        return chunk
    
    def format_final_chunk(self, usage: dict = None) -> dict:
        """Format the final streaming chunk with usage information.
        
        Args:
            usage: Usage statistics dictionary with prompt_tokens, completion_tokens, total_tokens
            
        Returns:
            Dictionary representation of the final chunk
        """
        delta = {
            "content": None,
            "role": "assistant",
        }
        
        choice = {
            "index": 0,
            "delta": delta,
            "finish_reason": "stop",
        }
        
        chunk = {
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [choice],
        }
        
        if usage:
            chunk["usage"] = usage
            
        return chunk
