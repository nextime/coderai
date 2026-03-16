import time
import uuid

# Try to import litellm for response formatting
# Fall back to plain dicts if litellm is not available or doesn't export these
try:
    from litellm import ModelResponse, ChatCompletionChunk, Choices, StreamingChoices, Delta, Message, Usage
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    ModelResponse = None
    ChatCompletionChunk = None
    Choices = None
    StreamingChoices = None
    Delta = None
    Message = None
    Usage = None


class OpenAIFormatter:
    def __init__(self, model_name):
        self.model_name = model_name
        self.id = f"chatcmpl-{uuid.uuid4()}"

    def format_full(self, text, prompt_tokens, completion_tokens, tool_calls=None):
        """Standard Response (Non-Streaming)"""
        if LITELLM_AVAILABLE and all([ModelResponse, Choices, Message, Usage]):
            try:
                return ModelResponse(
                    id=self.id,
                    model=self.model_name,
                    object="chat.completion",
                    created=int(time.time()),
                    choices=[Choices(
                        finish_reason="tool_calls" if tool_calls else "stop",
                        index=0,
                        message=Message(content=text if not tool_calls else None, role="assistant", tool_calls=tool_calls)
                    )],
                    usage=Usage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens
                    )
                ).model_dump()
            except Exception:
                pass
        
        # Fallback to plain dict if litellm fails
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

    def format_chunk(self, delta_text, is_final=False, usage=None):
        """Streaming Chunk (Used in a Generator)"""
        if LITELLM_AVAILABLE and all([ChatCompletionChunk, StreamingChoices, Delta, (Usage if usage else True)]):
            try:
                return ChatCompletionChunk(
                    id=self.id,
                    model=self.model_name,
                    object="chat.completion.chunk",
                    created=int(time.time()),
                    choices=[StreamingChoices(
                        finish_reason="stop" if is_final else None,
                        index=0,
                        delta=Delta(content=delta_text, role="assistant")
                    )],
                    usage=Usage(**usage) if (usage and Usage) else None
                ).model_dump()
            except Exception:
                pass
        
        # Fallback to plain dict if litellm fails
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
        """Format the final streaming chunk with usage information."""
        return self.format_chunk("", is_final=True, usage=usage)
