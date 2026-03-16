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
from difflib import get_close_matches
from typing import Dict, List, Any, Optional


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


# 1. QWEN PARSER (Instruct & Coder Style)
class QwenParser(BaseParser):
    @validate_tool_output
    def parse(self, text: str) -> List[Dict]:
        results = []
        
        # Clean text first
        clean_text = re.sub(r'<\|.*?\|>', '', text)
        print(f"DEBUG QwenParser: Input text length = {len(text)}")
        print(f"DEBUG QwenParser: Cleaned text: {repr(clean_text[:200])}")
        # Use raw string for regex with special tokens
        think_pattern = r'<think>.*?</think>'
        clean_text = re.sub(think_pattern, '', clean_text, flags=re.DOTALL)
        
        # INSTRUCT STYLE: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
        instruct_matches = re.findall(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', clean_text, re.DOTALL)
        for match in instruct_matches:
            try:
                data = json.loads(match.strip())
                if 'name' in data and 'arguments' in data:
                    results.append(self._to_oa(data['name'], data['arguments']))
            except:
                continue
        
        # CODER STYLE: <tool_call><function=name><parameter=key>value</parameter></function></tool_call>
        # Also handle: <tool=func_name><parameter=key>value</parameter></tool_call>
        if not results:
            # Try with <tool_call> wrapper first
            coder_blocks = re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>', clean_text, re.DOTALL)
            if not coder_blocks:
                # Try direct <tool=func_name> format (with </tool> or </tool_call> closing)
                coder_blocks = re.findall(r'(<tool=[^>]+>.*?</tool_call>)', clean_text, re.DOTALL)
            if not coder_blocks:
                coder_blocks = re.findall(r'(<tool=[^>]+>.*?</tool>)', clean_text, re.DOTALL)
            if not coder_blocks:
                # Try <function=func_name> format without wrapper
                coder_blocks = re.findall(r'(<function=.*?</function>)', clean_text, re.DOTALL)
            
            for block in coder_blocks:
                # Try to extract function name from different formats
                func_name = None
                
                # Format 1: <function=name>...</function>
                func_name_match = re.search(r'<function=([^>]+)>', block)
                if func_name_match:
                    func_name = func_name_match.group(1).strip()
                
                # Format 2: <tool=name>...</tool>
                if not func_name:
                    func_name_match = re.search(r'<tool=([^>]+)>', block)
                    if func_name_match:
                        func_name = func_name_match.group(1).strip()
                
                if func_name:
                    params = re.findall(r'<parameter=([^>]+)>(.*?)</parameter>', block, re.DOTALL)
                    arguments = {}
                    for k, v in params:
                        key = k.strip()
                        val = v.strip()
                        try:
                            arguments[key] = json.loads(val)
                        except:
                            arguments[key] = val
                    results.append(self._to_oa(func_name, arguments))
        
        return results


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
class ApexBig50Parser(BaseParser):
    @validate_tool_output
    def parse(self, text: str) -> List[Dict]:
        results = []
        
        # XML patterns
        xml_patterns = [
            r'<(?:tool_call|function_call|tool_use)>(.*?)</(?:tool_call|function_call|tool_use)>',
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

        # Markdown patterns
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

        # React pattern
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
