import time
import uuid
from litellm import ModelResponse, ChatCompletionChunk, Choices, StreamingChoices, Delta, Message, Usage


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
            Dictionary representation of ModelResponse
        """
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
    
    def format_chunk(self, delta_text: str, is_final: bool = False, usage: dict = None) -> dict:
        """Format a streaming chunk response.
        
        Args:
            delta_text: The incremental text content for this chunk
            is_final: Whether this is the final chunk
            usage: Optional usage information (typically only sent on final chunk)
            
        Returns:
            Dictionary representation of ChatCompletionChunk
        """
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
            usage=usage  # Only send usage on the final chunk
        ).model_dump()
    
    def format_final_chunk(self, usage: dict = None) -> dict:
        """Format the final streaming chunk with usage information.
        
        Args:
            usage: Usage statistics dictionary with prompt_tokens, completion_tokens, total_tokens
            
        Returns:
            Dictionary representation of the final ChatCompletionChunk
        """
        return ChatCompletionChunk(
            id=self.id,
            model=self.model_name,
            object="chat.completion.chunk",
            created=int(time.time()),
            choices=[StreamingChoices(
                finish_reason="stop",
                index=0,
                delta=Delta(content=None, role="assistant")
            )],
            usage=usage
        ).model_dump()
