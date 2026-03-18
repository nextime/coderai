"""
Model Parser Dispatcher - Multi-Model Tool Call Parsing

This module provides a comprehensive tool call parsing system that supports
multiple model types with different parsing strategies and includes validation.

Usage:
    # Initialize dispatcher
    dispatcher = ModelParserDispatcher(model_name="qwen3-7b-instruct", tools_schema=my_tools)
    
    # Parse tool calls
    raw_reply = '<tool_call>{"name": "get_weather", "arguments": {"location": "London"}}</tool_call>'
    standard_output = dispatcher.parse(raw_reply)
"""

import functools
import json
import uuid
import re
import time
from difflib import get_close_matches
from typing import Dict, List, Any, Optional, Tuple


def extract_reasoning_content(text: str, model_family: str = None) -> Tuple[str, str]:
    """Extract reasoning/thinking content from model output.
    
    Returns tuple of (reasoning_content, clean_text).
    """
    reasoning_content = ""
    clean_text = text
    
    # Define reasoning patterns for different model families
    patterns = [
        (r'<thought>(.*?)</thought>', 'qwen'),
        (r'<think>(.*?)</think>', 'qwen'),
        (r'<thought>(.*?)</thought>', 'deepseek'),
        (r'<thought>(.*?)</thought>', 'llama3'),
        (r'<thought>(.*?)</thought>', 'mistral'),
        (r'<thought>(.*?)</thought>', 'gemma'),
        (r'<\|im_start\|>assistant\n<thought>(.*?)</thought>', 'hermes'),
        (r'<thought>(.*?)</thought>', 'generic'),
    ]
    
    for pattern, _ in patterns:
        try:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            if matches:
                reasoning_content = '\n'.join([m.strip() for m in matches if m.strip()])
                clean_text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE).strip()
                break
        except:
            continue
    
    # Cleanup
    for p in [r'<thought>.*?</thought>', r'<think>.*?</think>']:
        clean_text = re.sub(p, '', clean_text, flags=re.DOTALL | re.IGNORECASE)
    
    return reasoning_content, clean_text


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


def validate_tool_output(func):
    """Decorator to validate and standardize tool call output."""
    @functools.wraps(func)
    def wrapper(self, text, *args, **kwargs):
        raw_calls = func(self, text, *args, **kwargs)
        if not raw_calls or not self.tools:
            return raw_calls

        validated = []
        tool_names = list(self.tools.keys())

        for call in raw_calls:
            name = call['function']['name']
            
            # Fuzzy Match Name
            if name not in self.tools:
                match = get_close_matches(name, tool_names, n=1, cutoff=0.7)
                name = match[0] if match else name
            
            # Ensure Arguments are a clean JSON string
            try:
                args_dict = json.loads(call['function']['arguments']) if isinstance(call['function']['arguments'], str) else call['function']['arguments']
            except:
                args_dict = {}

            # Type Casting based on Schema
            if name in self.tools:
                props = self.tools[name].get('parameters', {}).get('properties', {})
                for k, v in list(args_dict.items()):
                    if k in props:
                        t = props[k].get('type')
                        if t == "integer": 
                            args_dict[k] = int(v) if str(v).isdigit() else v
                        if t == "boolean": 
                            args_dict[k] = str(v).lower() in ["true", "1", "yes"]

            validated.append({
                "id": f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args_dict)}
            })
        return validated
    return wrapper


class BaseParser:
    """Base parser class for tool calls."""
    def __init__(self, tools: Dict[str, Any] = None):
        self.tools = tools or {}
    
    def _to_oa(self, name: str, args: Any) -> Dict:
        """Convert to OpenAI format."""
        if args is None:
            args = {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except:
                args = {}
        # Return in OpenAI format with 'function' key
        return {
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(args) if isinstance(args, dict) else args
            }
        }

    def _clean_json_string(self, text: str) -> str:
        """Clean JSON string by removing markdown code fences and extra whitespace."""
        # Remove markdown code fences
        text = re.sub(r'^```json\s*', '', text.strip())
        text = re.sub(r'^```\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        # Remove any leading/trailing whitespace
        return text.strip()

# 1. QWEN PARSER (Instruct & Coder Style)
# 1. QWEN PARSER (Instruct & Coder Style)
class QwenParser(BaseParser):
    def __init__(self, tools: Dict[str, Any] = None):
        super().__init__(tools)
        self.reasoning_content = ""
        
    @validate_tool_output
    def parse(self, text: str) -> List[Dict]:
        # 1. IMMEDIATE REPETITION GUARD
        # If the model is looping the same tag, we only care about the first one.
        if text.count('<tool') > 1:
            # Split by the start tag and take the first actual content block
            parts = re.split(r'<(?:tool|tool_call)', text, flags=re.IGNORECASE)
            # Reconstruct just the first potential call
            text = f"<tool{parts[1]}" if len(parts) > 1 else text

        results = []

        # 2. Pre-cleaning (Extract thinking/reasoning content, then remove for tool parsing)
        # Extract thinking content before removing it
        # Enhanced pattern to capture various thinking tag formats
        thinking_pattern = r'<\|.*?\|>|<(?:thought|think)>.*?((?:</(?:thought|think)>)|$)|<\|begin.*?\|><\|end.*?\|>'
        thinking_matches = re.findall(thinking_pattern, text, flags=re.DOTALL | re.IGNORECASE)
        if thinking_matches:
            # Extract the actual thinking content from matches
            thinking_content_parts = []
            for match in thinking_matches:
                if isinstance(match, tuple):
                    # Handle tuple matches from regex groups
                    if match[0]:  # First group contains the content
                        thinking_content_parts.append(match[0])
                    elif len(match) > 1 and match[1]:  # Second group might contain content
                        thinking_content_parts.append(match[1])
                else:
                    # Direct string match
                    thinking_content_parts.append(match)
            
            # Join all thinking matches with newlines, filtering out empty strings
            self.reasoning_content = '\n'.join([part.strip() for part in thinking_content_parts if part and part.strip()])
        else:
            self.reasoning_content = ""
            
        # Remove thinking tags from text for tool call parsing
        clean_text = re.sub(thinking_pattern, '', text, flags=re.DOTALL | re.IGNORECASE)

        # 3. FLEXIBLE TAG MATCHING
        # Matches <tool>, <tool_call>, or even just { "name": ... } if tags are missing
        tag_pattern = r'<(?:tool|tool_call|function_call)>(.*?)(?:</(?:tool|tool_call|function_call)>|$)'
        matches = re.findall(tag_pattern, clean_text, re.DOTALL | re.IGNORECASE)

        # If no tags found but text looks like JSON, try the whole text
        if not matches and '{' in clean_text and '"name"' in clean_text:
            matches = [clean_text]

        for block in matches:
            block = block.strip()
            if not block: continue

            # Clean Markdown & detect partial JSON
            json_str = re.sub(r'```(?:json)?\s*(.*?)\s*```', r'\1', block, flags=re.DOTALL).strip()
            
            # Attempt recovery of unclosed JSON (very common in 4-bit)
            if json_str.startswith('{') and not json_str.endswith('}'):
                json_str += '}' 

            try:
                data = json.loads(json_str)
                if 'name' in data:
                    results.append(self._to_oa(data['name'], data.get('arguments', {} or data.get('parameters', {}))))
                    break # STOP after the first valid tool call to break the loop
            except json.JSONDecodeError:
                # FAILING JSON: Attempt regex extraction for name/args
                name_match = re.search(r'"name":\s*"([^"]+)"', json_str)
                if name_match:
                    # Very basic fallback for arguments if JSON is totally mangled
                    results.append(self._to_oa(name_match.group(1), {}))
                    break

        # 4. CODER STYLE FALLBACK
        if not results:
            results = self._parse_coder_style(clean_text)

        return results

    def _parse_coder_style(self, text: str):
        found = []
        # Support <tool=name>, <function=name>, or <call=name>
        pattern = r'<(?:function|tool|call)=([^>]+)>(.*?)(?:</(?:function|tool|call|tool_call)>|$)'
        for name, body in re.findall(pattern, text, re.DOTALL | re.IGNORECASE):
            params = re.findall(r'<parameter=([^>]+)>(.*?)</parameter>', body, re.DOTALL)
            args = {k.strip(): self._relaxed_val(v) for k, v in params}
            found.append(self._to_oa(name.strip(), args))
            if found: break # Circuit breaker
        
        # NEW: Support <tool_call><tool><action>name</action><parameters>...</parameters></tool></tool_call>
        if not found:
            custom_pattern = r'<tool_call>\s*<tool>\s*<action>(.*?)</action>\s*<parameters>(.*?)</parameters>\s*</tool>\s*</tool_call>'
            for match in re.findall(custom_pattern, text, re.DOTALL | re.IGNORECASE):
                action, params_xml = match
                # Try to parse params as JSON
                try:
                    params = json.loads(params_xml.strip())
                except:
                    # Fallback: extract key-value pairs
                    params = {}
                    for prop_match in re.findall(r'<(\w+)>(.*?)</\1>', params_xml, re.DOTALL):
                        k, v = prop_match
                        params[k] = v.strip()
                if action.strip():
                    found.append(self._to_oa(action.strip(), params))
                if found: break  # Circuit breaker
        
        return found

    def _relaxed_val(self, val):
        val = val.strip()
        try: return json.loads(val)
        except: return val


# 2. DEEPSEEK PARSER
class DeepSeekParser(BaseParser):
    @validate_tool_output
    def parse(self, text: str) -> List[Dict]:
        results = []
        
        # DeepSeek-V3 uses specialized JSON prompts
        calls = re.findall(r'\{"name":\s*"(.*?)",\s*"parameters":\s*(\{.*?\})}', text)
        for name, params in calls:
            try:
                results.append(self._to_oa(name, json.loads(params)))
            except:
                continue
        
        return results


# 3. LLAMA PARSER (Markdown JSON)
class LlamaParser(BaseParser):
    @validate_tool_output
    def parse(self, text: str) -> List[Dict]:
        results = []
        
        blocks = re.findall(r'```json\s*([\[\{].*?[\]\}])\s*```', text, re.DOTALL)
        for block in blocks:
            try:
                data = json.loads(block)
                for item in (data if isinstance(data, list) else [data]):
                    results.append(self._to_oa(item.get("name"), item.get("arguments")))
            except:
                continue
        
        return results


# 4. MISTRAL PARSER
class MistralParser(BaseParser):
    @validate_tool_output
    def parse(self, text: str) -> List[Dict]:
        results = []
        
        match = re.search(r'\[TOOL_CALLS\]\s*(.*)', text)
        if match:
            try:
                calls = json.loads(match.group(1))
                for c in calls:
                    results.append(self._to_oa(c["name"], c["arguments"]))
            except:
                pass
        
        return results


# 5. CLAUDE PARSER
class ClaudeParser(BaseParser):
    @validate_tool_output
    def parse(self, text: str) -> List[Dict]:
        results = []
        
        calls = re.findall(r'<tool_use>(.*?)</tool_use>', text, re.DOTALL)
        for c in calls:
            name_match = re.search(r'<name>(.*?)</name>', c)
            args_match = re.search(r'<parameters>(.*?)</parameters>', c, re.DOTALL)
            if name_match:
                try:
                    args = json.loads(args_match.group(1)) if args_match else {}
                except:
                    args = args_match.group(1) if args_match else {}
                results.append(self._to_oa(name_match.group(1), args))
        
        return results


# 6. COMMAND R PARSER
class CommandRParser(BaseParser):
    @validate_tool_output
    def parse(self, text: str) -> List[Dict]:
        results = []
        
        action = re.search(r'Action:\s*(.*)', text)
        args = re.search(r'Action Input:\s*(\{.*\})', text)
        if action and args:
            try:
                results.append(self._to_oa(action.group(1).strip(), json.loads(args.group(1))))
            except:
                pass
        
        return results


# 7. GEMMA PARSER
class GemmaParser(BaseParser):
    @validate_tool_output
    def parse(self, text: str) -> List[Dict]:
        results = []
        
        match = re.search(r'{\s*"name":\s*".*?"\s*,\s*"parameters":\s*\{.*?\}\s*\}', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                results.append(self._to_oa(data["name"], data["parameters"]))
            except:
                pass
        
        return results


# 8. GROK PARSER
class GrokParser(BaseParser):
    @validate_tool_output
    def parse(self, text: str) -> List[Dict]:
        results = []
        
        try:
            data = json.loads(text)
            if isinstance(data, list) and len(data) > 0 and "name" in data[0]:
                for c in data:
                    results.append(self._to_oa(c["name"], c["arguments"]))
        except:
            pass
        
        return results


# 9. PHI PARSER
class PhiParser(BaseParser):
    @validate_tool_output
    def parse(self, text: str) -> List[Dict]:
        results = []
        
        # Phi-3/4 uses <|tool_call|> tags
        match = re.search(r'<\|tool_call\|>(.*?)<\|', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                results.append(self._to_oa(data["name"], data["arguments"]))
            except:
                pass
        
        # Fallback to Llama-style markdown
        if not results:
            results = LlamaParser(self.tools).parse(text)
        
        return results


# 10. APEX BIG 50 (Catch-All Parser)
# Note: Most XML parsing is delegated to ToolCallParser as fallback.
# This parser keeps unique patterns that ToolCallParser doesn't handle.
class ApexBig50Parser(BaseParser):
    @validate_tool_output
    def parse(self, text: str) -> List[Dict]:
        results = []
        
        # XML patterns - basic JSON-in-XML
        # Note: Complex nested XML patterns are handled by ToolCallParser fallback
        xml_patterns = [
            r'<(?:tool|tool_call|function_call|tool_use)>(.*?)</(?:tool|tool_call|function_call|tool_use)>',
            r'\[TOOL_CALLS\](.*?)\[/TOOL_CALLS\]'
        ]
        for p in xml_patterns:
            for match in re.findall(p, text, re.DOTALL):
                try:
                    data = json.loads(match.strip())
                    items = data if isinstance(data, list) else [data]
                    for i in items:
                        name = i.get("name") or i.get("function")
                        args = i.get("arguments") or i.get("args")
                        if name:
                            results.append(self._to_oa(name, args))
                except:
                    fn = re.search(r'<(?:function|name)=(.*?)>', match)
                    if fn:
                        params = dict(re.findall(r'<(?:parameter|arg|argument)=(.*?)>(.*?)</(?:parameter|arg|argument)>', match, re.DOTALL))
                        results.append(self._to_oa(fn.group(1).strip(), params))

        # Markdown JSON patterns (unique to ApexBig50 - ToolCallParser doesn't handle this)
        md_patterns = [
            r'```json\s*([\[\{].*?[\]\}])\s*```',
        ]
        for p in md_patterns:
            for block in re.findall(p, text, re.DOTALL):
                try:
                    data = json.loads(block)
                    for item in (data if isinstance(data, list) else [data]):
                        name = item.get("name") or item.get("function")
                        args = item.get("arguments") or item.get("parameters") or item
                        if name and isinstance(name, str):
                            results.append(self._to_oa(name, args))
                except:
                    pass

        # React pattern (unique to ApexBig50)
        react_matches = re.findall(r'Action:\s*(.*?)\nAction Input:\s*(\{.*?\})', text, re.DOTALL)
        for name, args_raw in react_matches:
            try:
                results.append(self._to_oa(name.strip(), json.loads(args_raw.strip())))
            except:
                pass

        return results


# Model Parser Dispatcher
class ModelParserDispatcher:
    """Dispatcher to select the appropriate parser based on model name."""
    
    def __init__(self, model_name: str = None, tools_schema: Dict[str, Any] = None):
        self.model_name = model_name
        self.tools = tools_schema or {}
        self.parser = self._get_parser()
    
    def _get_parser(self) -> BaseParser:
        """Get the appropriate parser based on model name."""
        if not self.model_name:
            parser = ApexBig50Parser(self.tools)
            print(f"DEBUG model_parser: model_name=None, selected parser: {type(parser).__name__}")
            return parser
        
        model_lower = self.model_name.lower()
        
        # Qwen models
        if 'qwen' in model_lower:
            parser = QwenParser(self.tools)
            print(f"DEBUG model_parser: model_name={self.model_name}, selected parser: QwenParser")
            return parser
        
        # DeepSeek models
        if 'deepseek' in model_lower:
            parser = DeepSeekParser(self.tools)
            print(f"DEBUG model_parser: model_name={self.model_name}, selected parser: DeepSeekParser")
            return parser
        
        # Llama models
        if 'llama' in model_lower:
            parser = LlamaParser(self.tools)
            print(f"DEBUG model_parser: model_name={self.model_name}, selected parser: LlamaParser")
            return parser
        
        # Mistral models
        if 'mistral' in model_lower or 'mixtral' in model_lower:
            parser = MistralParser(self.tools)
            print(f"DEBUG model_parser: model_name={self.model_name}, selected parser: MistralParser")
            return parser
        
        # Claude models
        if 'claude' in model_lower:
            parser = ClaudeParser(self.tools)
            print(f"DEBUG model_parser: model_name={self.model_name}, selected parser: ClaudeParser")
            return parser
        
        # Command R models
        if 'command' in model_lower:
            parser = CommandRParser(self.tools)
            print(f"DEBUG model_parser: model_name={self.model_name}, selected parser: CommandRParser")
            return parser
        
        # Gemma models
        if 'gemma' in model_lower:
            parser = GemmaParser(self.tools)
            print(f"DEBUG model_parser: model_name={self.model_name}, selected parser: GemmaParser")
            return parser
        
        # Grok models
        if 'grok' in model_lower:
            parser = GrokParser(self.tools)
            print(f"DEBUG model_parser: model_name={self.model_name}, selected parser: GrokParser")
            return parser
        
        # Phi models
        if 'phi' in model_lower:
            parser = PhiParser(self.tools)
            print(f"DEBUG model_parser: model_name={self.model_name}, selected parser: PhiParser")
            return parser
        
        # Default: use catch-all parser
        parser = ApexBig50Parser(self.tools)
        print(f"DEBUG model_parser: model_name={self.model_name}, selected parser: ApexBig50Parser (default)")
        return parser
    
    def parse(self, text: str) -> List[Dict]:
        """Parse tool calls from model output."""
        return self.parser.parse(text)
    
    def set_tools(self, tools: Dict[str, Any]) -> None:
        """Update the tools schema."""
        self.tools = tools
        self.parser.tools = tools


class OpenAIFormatter:
    def __init__(self, model_name):
        self.model_name = model_name
        self.id = f"chatcmpl-{uuid.uuid4()}"

    def format_full(self, text, prompt_tokens, completion_tokens, tool_calls=None, reasoning=None):
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
            except Exception as e:
                print(f"DEBUG formatter: litellm fallback failed: {e}")
        
        # Fallback to plain dict if litellm fails
        message = {
            "role": "assistant",
            "content": text if not tool_calls else None,
        }
        if reasoning:
            message["reasoning"] = reasoning
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

    # Backward compatibility methods
    def format_litellm_full(self, text: str, prompt_tokens: int, completion_tokens: int, tool_calls=None) -> dict:
        """Backward compatibility method - calls format_full."""
        return self.format_full(text, prompt_tokens, completion_tokens, tool_calls)

    def format_litellm_chunk(self, delta_text: str, is_final: bool = False, usage: dict = None) -> dict:
        """Backward compatibility method - calls format_chunk."""
        return self.format_chunk(delta_text, is_final, usage)


# =============================================================================
# Content Filtering
# =============================================================================

def filter_malformed_content(text: str) -> str:
    """Filter out malformed SEARCH/REPLACE blocks that the model might output as content."""
    import re
    
    if not text:
        return text
    
    # Remove diff-like blocks that shouldn't be in the output
    filtered = text
    
    # Remove git-style diff markers and SEARCH/REPLACE patterns
    filtered = re.sub(r'<<<<<<<\s+SEARCH.*?=======', '', filtered, flags=re.DOTALL)
    filtered = re.sub(r'=======.*?>>>>>>>\s+REPLACE', '', filtered, flags=re.DOTALL)
    filtered = re.sub(r'>>>>>>>\s+REPLACE', '', filtered)
    
    # Also remove common malformed patterns seen in outputs
    filtered = re.sub(r'<<<<<<<\s+SEARCH\s*:start_line:\d+[^<]*', '', filtered, flags=re.DOTALL)
    filtered = re.sub(r'<button>Stop Generation</button>', '', filtered)
    filtered = re.sub(r'\<\|assistant\|\>', '', filtered)
    filtered = re.sub(r'\</\|assistant\|\>', '', filtered)
    
    # Clean up excessive newlines left from removal
    filtered = re.sub(r'\n{3,}', '\n\n', filtered)
    
    # Don't strip single newlines or whitespace - they might be valid content
    return filtered


# =============================================================================
# Tool Formatting
# =============================================================================

def format_tools_for_prompt(tools, messages):
    """Format tools into the system message or add a tool description."""
    import json
    
    if not tools:
        return messages
    
    # Import here to avoid circular imports
    from codai.pydantic.textrequest import ChatMessage
    
    tool_descriptions = []
    for tool in tools:
        func = tool.function
        desc = f"Tool: {func.name}"
        if func.description:
            desc += f"\nDescription: {func.description}"
        if func.parameters:
            desc += f"\nParameters: {json.dumps(func.parameters, indent=2)}"
        tool_descriptions.append(desc)
    
    tools_text = "You have access to the following tools:\n\n" + "\n\n".join(tool_descriptions)
    tools_text += "\n\nIMPORTANT: When you need to use a tool, you MUST format your response EXACTLY as:\n"
    tools_text += '<tool>{"name": "tool_name", "arguments": {"param1": "value1", "param2": "value2"}}</tool>'
    tools_text += "\n\nRules:\n"
    tools_text += "1. The content inside <tool> tags must be valid JSON\n"
    tools_text += "2. Do NOT use nested XML tags like <name> or <arguments> - use JSON format only\n"
    tools_text += "3. The 'name' field must match one of the available tool names exactly\n"
    tools_text += "4. The 'arguments' field must be a JSON object with the required parameters\n"
    tools_text += "\nExample:\n"
    tools_text += 'User: Read the file example.txt\n'
    tools_text += 'Assistant: <tool>{"name": "read_file", "arguments": {"files": [{"path": "example.txt"}]}}</tool>'
    
    # Add or prepend to system message
    new_messages = list(messages)
    system_found = False
    
    for i, msg in enumerate(new_messages):
        if msg.role == "system":
            new_messages[i] = ChatMessage(
                role="system",
                content=f"{tools_text}\n\n{msg.content or ''}"
            )
            system_found = True
            break
    
    if not system_found:
        new_messages.insert(0, ChatMessage(role="system", content=tools_text))
    
    return new_messages


# =============================================================================
# Tool Call Parser (moved from coderai)
# =============================================================================

from typing import Dict, List, Optional
import re
import json
import uuid

# Import filter_malformed_content from same module
from codai.models.parser import filter_malformed_content

# Import Tool from pydantic - use lazy import to avoid circular dependencies
def _get_tool():
    from codai.pydantic.textrequest import Tool
    return Tool

# Import ModelParserDispatcher from same module
from codai.models.parser import ModelParserDispatcher


class ToolCallParser:
    """Parse model outputs to extract tool calls."""
    
    def __init__(self, tokenizer=None, model_name: str = None):
        self.tokenizer = tokenizer
        self.model_name = model_name
        
    def _parse_xml_style_tool_calls(self, text: str) -> List[Dict]:
        """Parse XML-style tool calls like <tool><name>...</name><arguments>...</arguments></tool>."""
        tool_calls = []
        
        # Pattern for <tool><name>...</name><arguments>...</arguments></tool>
        pattern = r'<tool>\s*<name>(.*?)</name>\s*<arguments>(.*?)</arguments>\s*</tool>'
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        
        for name, args_str in matches:
            name = name.strip()
            if not name:
                continue
                
            # Try to parse arguments as JSON
            try:
                args = json.loads(args_str.strip()) if args_str.strip() else {}
            except json.JSONDecodeError:
                # If not valid JSON, treat as empty object
                args = {}
                
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:16]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args)
                }
            })
        
        # NEW: Pattern for <tool><action>...</action><parameters>...</parameters></tool>
        # Example: <tool><action>search</action><parameters>{"query": "Apple AAPL Q4"}</parameters></tool>
        pattern_action = r'<tool>\s*<action>(.*?)</action>\s*<parameters>(.*?)</parameters>\s*</tool>'
        matches_action = re.findall(pattern_action, text, re.DOTALL | re.IGNORECASE)
        
        for name, args_str in matches_action:
            name = name.strip()
            if not name:
                continue
                
            # Try to parse arguments as JSON
            try:
                args = json.loads(args_str.strip()) if args_str.strip() else {}
            except json.JSONDecodeError:
                # If not valid JSON, try to extract key-value pairs from XML
                args = {}
                # Try to parse as key-value pairs
                try:
                    for match in re.findall(r'<(\w+)>(.*?)</\1>', args_str, re.DOTALL):
                        k, v = match
                        args[k] = v.strip()
                except:
                    pass
            
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:16]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args)
                }
            })
        
        # NEW: Pattern for <tool>name</tool><tool_call>JSON</tool_call> format
        # Example: <tool>search</tool><tool_call>{"query": "Apple AAPL Q4"}</tool_call>
        # Also handles case where both are opening tags: <tool_call>...</tool_call>
        pattern2 = r'<tool>\s*(\w+)\s*</tool>\s*<tool_call>\s*(.+?)\s*(?:</tool_call>|$)'
        matches2 = re.findall(pattern2, text, re.DOTALL | re.IGNORECASE)
        
        for name, args_str in matches2:
            name = name.strip()
            if not name:
                continue
            
            # Try to parse arguments as JSON
            try:
                args = json.loads(args_str.strip()) if args_str.strip() else {}
            except json.JSONDecodeError:
                # If not valid JSON, treat as empty object
                args = {}
            
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:16]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args)
                }
            })
        
        # NEW: Pattern for <tool_call><tool><name>...</name><arguments>...</arguments></tool></tool_call>
        # Example: <tool_call><tool><name>search</name><arguments>{"query": "test"}</arguments></tool></tool_call>
        pattern_nested = r'<tool_call>\s*<tool>\s*<name>(.*?)</name>\s*<arguments>(.*?)</arguments>\s*</tool>\s*</tool_call>'
        matches_nested = re.findall(pattern_nested, text, re.DOTALL | re.IGNORECASE)
        
        for name, args_str in matches_nested:
            name = name.strip()
            if not name:
                continue
            
            # Try to parse arguments as JSON
            try:
                args = json.loads(args_str.strip()) if args_str.strip() else {}
            except json.JSONDecodeError:
                # If not valid JSON, treat as empty object
                args = {}
            
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:16]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args)
                }
            })
        
        # NEW: Pattern for <tool_call><tool><function>...</function><parameters>...</parameters></tool></tool_call>
        # Example: <tool_call><tool><function>get_financial_data</function><parameters>{"ticker": "AAPL"}</parameters></tool></tool_call>
        pattern_function = r'<tool_call>\s*<tool>\s*<function>(.*?)</function>\s*<parameters>(.*?)</parameters>\s*</tool>\s*</tool_call>'
        matches_function = re.findall(pattern_function, text, re.DOTALL | re.IGNORECASE)
        
        for name, args_str in matches_function:
            name = name.strip()
            if not name:
                continue
            
            # Try to parse arguments as JSON
            try:
                args = json.loads(args_str.strip()) if args_str.strip() else {}
            except json.JSONDecodeError:
                # If not valid JSON, treat as empty object
                args = {}
            
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:16]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args)
                }
            })
        
        # NEW: Pattern for standalone <tool><function>...</function><parameters>...</parameters></tool> (without wrapper)
        pattern_standalone_func = r'<tool>\s*<function>(.*?)</function>\s*<parameters>(.*?)</parameters>\s*</tool>'
        matches_standalone = re.findall(pattern_standalone_func, text, re.DOTALL | re.IGNORECASE)
        
        for name, args_str in matches_standalone:
            name = name.strip()
            if not name:
                continue
            
            try:
                args = json.loads(args_str.strip()) if args_str.strip() else {}
            except json.JSONDecodeError:
                args = {}
            
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:16]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args)
                }
            })
        
        # NEW: Pattern for multiple tool calls in <tool_call> wrapper
        # Example: <tool_call><tool>...</tool><tool>...</tool></tool_call>
        pattern_multi = r'<tool_call>\s*(<tool>.*?</tool>)\s*</tool_call>'
        matches_multi = re.findall(pattern_multi, text, re.DOTALL | re.IGNORECASE)
        
        for tool_block in matches_multi:
            # Extract individual tool calls from within the tool_call wrapper
            inner_pattern = r'<tool>\s*<name>(.*?)</name>\s*<arguments>(.*?)</arguments>\s*</tool>'
            inner_matches = re.findall(inner_pattern, tool_block, re.DOTALL | re.IGNORECASE)
            
            for name, args_str in inner_matches:
                name = name.strip()
                if not name:
                    continue
                
                try:
                    args = json.loads(args_str.strip()) if args_str.strip() else {}
                except json.JSONDecodeError:
                    args = {}
                
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:16]}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args)
                    }
                })
        
        return tool_calls
    
    def set_model_name(self, model_name: str):
        """Set the model name for model-specific parsing."""
        self.model_name = model_name
    
    def _is_qwen_model(self) -> bool:
        """Check if the current model is a Qwen model."""
        if not self.model_name:
            return False
        model_lower = self.model_name.lower()
        return 'qwen' in model_lower or 'qwen2' in model_lower or 'qwen3' in model_lower
    
    def _parse_qwen_tool_calls(self, text: str) -> Optional[List[Dict]]:
        """Parse tool calls from Qwen model output."""
        # This is a placeholder - the full implementation is quite long
        # For now, return None and let the main parse method handle it
        return None
    
    def _parse_nested_xml_tool(self, xml_content: str) -> Optional[Dict]:
        """Parse nested XML tool format."""
        return None
    
    def _xml_to_dict(self, xml_content: str) -> Dict:
        """Convert simple nested XML to dictionary."""
        result = {}
        pattern = r'<(\w+)>\s*(.*?)\s*</\1>'
        matches = re.findall(pattern, xml_content, re.DOTALL)
        for tag, content in matches:
            if re.search(r'<\w+>', content):
                try:
                    result[tag] = self._xml_to_dict(content)
                except:
                    result[tag] = content
            else:
                result[tag] = content
        return result if result else xml_content
    
    def _filter_malformed_content(self, text: str) -> str:
        """Filter out malformed SEARCH/REPLACE blocks."""
        return filter_malformed_content(text)
    
    def extract_tool_calls(self, text: str, available_tools: List) -> Optional[List[Dict]]:
        """Extract tool calls from model output."""
        # Debug logging for ToolCallParser
        print(f"DEBUG ToolCallParser: Called with text (first 200 chars): {repr(text[:200] if len(text) > 200 else text)}")
        print(f"DEBUG ToolCallParser: available_tools count: {len(available_tools) if available_tools else 0}")
        
        # First filter out malformed content
        text = self._filter_malformed_content(text)
        
        # For Qwen models, try Qwen-specific parsing first
        if self._is_qwen_model():
            qwen_tool_calls = self._parse_qwen_tool_calls(text)
            if qwen_tool_calls:
                print(f"DEBUG ToolCallParser: Qwen parsing found {len(qwen_tool_calls)} tool calls")
                return qwen_tool_calls
        
        tool_calls = []
        seen_signatures = set()
        
        # Look for function calls in various formats
        tool_pattern = r'<(?:tool|function)>(.*?)</(?:tool|function)>'
        tool_matches = re.findall(tool_pattern, text, re.DOTALL)
        
        for match in tool_matches:
            try:
                tool_data = json.loads(match.strip())
                if 'name' in tool_data and 'arguments' in tool_data:
                    sig = (tool_data['name'], json.dumps(tool_data['arguments'], sort_keys=True))
                    if sig not in seen_signatures:
                        seen_signatures.add(sig)
                        tool_calls.append({
                            "id": f"call_{uuid.uuid4().hex[:16]}",
                            "type": "function",
                            "function": {
                                "name": tool_data["name"],
                                "arguments": json.dumps(tool_data["arguments"])
                            }
                        })
            except json.JSONDecodeError:
                pass
        
        # If no tool calls found yet, try XML-style parsing as fallback
        if not tool_calls:
            xml_tool_calls = self._parse_xml_style_tool_calls(text)
            if xml_tool_calls:
                print(f"DEBUG ToolCallParser: XML-style parsing found {len(xml_tool_calls)} tool calls")
                tool_calls.extend(xml_tool_calls)
        
        # Debug output for results
        if tool_calls:
            print(f"DEBUG ToolCallParser: Returning {len(tool_calls)} tool calls")
            for i, tc in enumerate(tool_calls):
                print(f"DEBUG ToolCallParser:   [{i}] {tc.get('function', {}).get('name', 'unknown')}: {tc.get('function', {}).get('arguments', {})}")
        else:
            print(f"DEBUG ToolCallParser: No tool calls found, returning None")
        
        return tool_calls if tool_calls else None

    def strip_tool_calls_from_content(self, text: str) -> str:
        """Remove tool call format from text after extracting tool calls."""
        if not text:
            return text
        
        text = re.sub(r'<tool>.*?</tool>', '', text, flags=re.DOTALL)
        text = re.sub(r'<function>.*?</function>', '', text, flags=re.DOTALL)
        text = re.sub(r'<tool>\{.*?\}</tool>', '', text, flags=re.DOTALL)
        text = re.sub(r'<tool>[\s\S]*?</tool>', '', text)
        text = re.sub(r'<function>[\s\S]*?</function>', '', text)
        
        for tool_name in ['read', 'write', 'exec', 'browser', 'message', 'web_search', 'web_fetch', 
                         'memory_search', 'memory_get', 'sessions_list', 'sessions_send', 'tts', 'canvas', 'nodes']:
            text = re.sub(rf'<{tool_name}>[\s\S]*?</{tool_name}>', '', text)
        
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text


class ModelParserAdapter:
    """Adapter class that wraps ModelParserDispatcher to provide ToolCallParser interface."""
    
    def __init__(self, model_name: str = None, tools_schema: Dict = None):
        self._model_name = model_name
        self._tools_schema = tools_schema or {}
        self._dispatcher = ModelParserDispatcher(model_name=model_name, tools_schema=self._tools_schema)
    
    def set_model_name(self, model_name: str) -> None:
        """Set the model name for model-specific parsing."""
        self._model_name = model_name
        self._dispatcher = ModelParserDispatcher(model_name=model_name, tools_schema=self._tools_schema)
    
    def extract_tool_calls(self, text: str, available_tools: List) -> Optional[List[Dict]]:
        """Extract tool calls from model output using model-specific parsing."""
        if not text:
            return None
        
        tools_dict = {}
        for tool in available_tools:
            if hasattr(tool, 'function') and tool.function:
                func = tool.function
                tools_dict[func.name] = {
                    'description': func.description or '',
                    'parameters': func.parameters or {}
                }
        
        if tools_dict != self._tools_schema:
            self._tools_schema = tools_dict
            self._dispatcher.set_tools(tools_dict)
        
        tool_calls = self._dispatcher.parse(text)
        
        # Fallback: if no tool calls found, try using ToolCallParser
        if not tool_calls:
            tool_call_parser = ToolCallParser(model_name=self._model_name)
            tool_calls = tool_call_parser.extract_tool_calls(text, available_tools)
        
        if tool_calls:
            for tc in tool_calls:
                if 'id' not in tc:
                    tc['id'] = f"call_{uuid.uuid4().hex[:16]}"
                if 'type' not in tc:
                    tc['type'] = 'function'
            return tool_calls
        
        return None
    
    def strip_tool_calls_from_content(self, text: str) -> str:
        """Remove tool call format from text after extracting tool calls."""
        if not text:
            return text
        
        # Custom XML format: <tool><action>...</action><object>...</object><properties>...</properties></tool>
        text = re.sub(r'<tool>\s*<action>.*?</action>\s*<object>.*?</object>\s*<properties>.*?</properties>\s*</tool>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<tool=[^>]+>.*?</tool_call>', '', text, flags=re.DOTALL)
        text = re.sub(r'<tool=[^>]+>.*?</tool>', '', text, flags=re.DOTALL)
        text = re.sub(r'<tool>.*?</tool>', '', text, flags=re.DOTALL)
        text = re.sub(r'<function>.*?</function>', '', text, flags=re.DOTALL)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text
