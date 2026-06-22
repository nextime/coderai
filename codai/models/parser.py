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
    
    Uses pre-compiled REASONING_PATTERNS and protects code blocks from
    false positive matches.
    
    Returns tuple of (reasoning_content, clean_text).
    The reasoning_content will have any tool call tags stripped out.
    """
    reasoning_content = ""
    clean_text = text
    
    # Protect code blocks from being matched as reasoning tags
    code_blocks = []
    protected_text = text
    
    # Protect ```code blocks```
    for match in re.finditer(r'```[\s\S]*?```', text):
        placeholder = f"__CODE_BLOCK_{len(code_blocks)}__"
        code_blocks.append(match.group(0))
        protected_text = protected_text.replace(match.group(0), placeholder)
    
    # Protect `inline code`
    for match in re.finditer(r'`[^`\n]+`', protected_text):
        placeholder = f"__INLINE_CODE_{len(code_blocks)}__"
        code_blocks.append(match.group(0))
        protected_text = protected_text.replace(match.group(0), placeholder)
    
    # Use pre-compiled patterns on protected text (no code blocks to false-match)
    for pattern, _ in REASONING_PATTERNS:
        try:
            matches = pattern.findall(protected_text)
            if matches:
                reasoning_content = '\n'.join([m.strip() for m in matches if m.strip()])
                clean_text = pattern.sub('', protected_text).strip()
                break
        except:
            continue

    # Bare closing tag, no opening tag. Qwen3 (and other models whose chat template
    # PRE-FILLS the opening <think> in the prompt) generate only the reasoning body
    # followed by a closing </think> — there is no opening tag in the output, so the
    # paired patterns above never match and the whole thought would leak into the
    # content. Treat everything up to the first bare close tag as reasoning, as long
    # as no matching opening tag precedes it.
    if not reasoning_content:
        for close in ('</think>', '</thinking>', '</thought>'):
            idx = protected_text.lower().find(close)
            if idx == -1:
                continue
            open_tag = close.replace('</', '<', 1)
            if open_tag.lower() in protected_text[:idx].lower():
                continue  # a real opening tag exists — leave it to the paired logic
            reasoning_content = protected_text[:idx].strip()
            clean_text = protected_text[idx + len(close):].strip()
            break

    # Cleanup with pre-compiled patterns
    for p in REASONING_CLEANUP_PATTERNS:
        clean_text = p.sub('', clean_text)
    
    # Restore code blocks in clean_text
    for i, block in enumerate(code_blocks):
        clean_text = clean_text.replace(f"__CODE_BLOCK_{i}__", block)
        clean_text = clean_text.replace(f"__INLINE_CODE_{i}__", block)
    
    # FIX: If reasoning contains tool call tags, split at the first tool tag
    # The tool call part should NOT be in reasoning - it should be left in clean_text for tool extraction
    if reasoning_content:
        tool_tag_patterns = ["<tool_call>", "<tool>", "<|tool_call>", "<|tool_call|>", "<function="]
        earliest_tool_idx = len(reasoning_content)
        earliest_tool_tag = None
        for tag in tool_tag_patterns:
            idx = reasoning_content.find(tag)
            if idx != -1 and idx < earliest_tool_idx:
                earliest_tool_idx = idx
                earliest_tool_tag = tag
        
        if earliest_tool_tag:
            # Split: everything before the tool tag is reasoning, tool part goes back to clean_text
            tool_part = reasoning_content[earliest_tool_idx:]
            reasoning_content = reasoning_content[:earliest_tool_idx].strip()
            # Prepend the tool part to clean_text so it can be extracted as a tool call
            clean_text = tool_part + " " + clean_text
            clean_text = clean_text.strip()
    
    return reasoning_content, clean_text


# =============================================================================
# Pre-compiled Regex Patterns for Performance
# =============================================================================
# These patterns are compiled once at module load time for better performance

# Reasoning extraction patterns
REASONING_PATTERNS = [
    (re.compile(r'<\|begin_of_text\|>.*?<thinking>(.*?)</thinking>', re.DOTALL | re.IGNORECASE), 'qwen'),
    (re.compile(r'<thinking>(.*?)</thinking>', re.DOTALL | re.IGNORECASE), 'qwen2'),
    (re.compile(r'<think>(.*?)</think>', re.DOTALL | re.IGNORECASE), 'deepseek'),
    (re.compile(r'<thought>(.*?)</thought>', re.DOTALL | re.IGNORECASE), 'llama3'),
    (re.compile(r'<\|im_start\|>assistant\n<thought>(.*?)</thought>', re.DOTALL | re.IGNORECASE), 'hermes'),
]

# Cleanup patterns for reasoning extraction
REASONING_CLEANUP_PATTERNS = [
    re.compile(r'<thought>.*?</thought>', re.DOTALL | re.IGNORECASE),
    re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE),
]

# Tool call XML patterns - pre-compiled for performance
# These are the core patterns used in _parse_xml_style_tool_calls()
TOOL_PATTERNS = {
    # Basic pattern: <tool><name>...</name><arguments>...</arguments></tool>
    'basic': re.compile(r'<tool>\s*<name>(.*?)</name>\s*<arguments>(.*?)</arguments>\s*</tool>', re.DOTALL | re.IGNORECASE),
    # Pattern with <action> and <parameters>
    'action': re.compile(r'<tool>\s*<action>(.*?)</action>\s*<parameters>(.*?)</parameters>\s*</tool>', re.DOTALL | re.IGNORECASE),
    # Pattern with tool_call wrapper
    'tool_call_basic': re.compile(r'<tool>\s*(\w+)\s*</tool>\s*<tool_call>\s*(.+?)\s*(?:</tool_call>|$)', re.DOTALL | re.IGNORECASE),
    # Nested in tool_call: <tool_call><tool><name>...</name><arguments>...</arguments></tool></tool_call>
    'nested': re.compile(r'<tool_call>\s*<tool>\s*<name>(.*?)</name>\s*<arguments>(.*?)</arguments>\s*</tool>\s*</tool_call>', re.DOTALL | re.IGNORECASE),
    # Nested with <function> tag
    'nested_function': re.compile(r'<tool_call>\s*<tool>\s*<function>(.*?)</function>\s*<parameters>(.*?)</parameters>\s*</tool>\s*</tool_call>', re.DOTALL | re.IGNORECASE),
    # Standalone with <function>
    'standalone_function': re.compile(r'<tool>\s*<function>(.*?)</function>\s*<parameters>(.*?)</parameters>\s*</tool>', re.DOTALL | re.IGNORECASE),
    # Standalone with <parameters> instead of <arguments>
    'standalone_params': re.compile(r'<tool>\s*<name>(.*?)</name>\s*<parameters>(.*?)</parameters>\s*</tool>', re.DOTALL | re.IGNORECASE),
    # Nested with <parameters>
    'nested_params': re.compile(r'<tool_call>\s*<tool>\s*<name>(.*?)</name>\s*<parameters>(.*?)</parameters>\s*</tool>\s*</tool_call>', re.DOTALL | re.IGNORECASE),
    # Multi-tool wrapper
    'multi': re.compile(r'<tool_call>\s*(<tool>.*?</tool>)\s*</tool_call>', re.DOTALL | re.IGNORECASE),
    # Nested tool where tool name IS the tag: <tool_call><tool><toolname>...</toolname></tool></tool_call>
    'nested_tool': re.compile(r'<tool_call>\s*<tool>\s*<(\w+)>\s*(.*?)</\1>\s*</tool>\s*</tool_call>', re.DOTALL | re.IGNORECASE),
    # Standalone nested tool: <tool><toolname>...</toolname></tool>
    'standalone_nested': re.compile(r'<tool>\s*<(\w+)>\s*(.*?)</\1>\s*</tool>', re.DOTALL | re.IGNORECASE),
    # Short format: <tool>TOOL_NAME>JSON</tool>
    'short': re.compile(r'<tool>(\w+)>(\{.*?\})</tool>', re.DOTALL),
    # JSON in tool: <tool>{"name": ...}</tool>
    'json_in_tool': re.compile(r'<tool>\s*(\{.*?\})\s*</tool>', re.DOTALL),
    # Short with tool_call wrapper
    'short2': re.compile(r'<tool_call>\s*<tool>(\w+)>\s*(\{.*?\})\s*</tool>\s*</tool_call>', re.DOTALL),
    # Multi-tools in tool_call
    'multi_tools': re.compile(r'<tool_call>\s*(<tool>.*?</tool>)\s*(?:<tool_call>\s*(<tool>.*?</tool>)\s*)?</tool_call>', re.DOTALL | re.IGNORECASE),
    # Multi-line standalone
    'multiline_standalone': re.compile(r'<tool>\s*<name>\s*(.*?)\s*</name>\s*<arguments>\s*(.*?)\s*</arguments>\s*</tool>', re.DOTALL | re.IGNORECASE),
    # Multi-line wrapper
    'multiline': re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>', re.DOTALL | re.IGNORECASE),
}

# XML to dict pattern
RE_XML_TO_DICT = re.compile(r'<(\w+)>\s*(.*?)\s*</\1>')
RE_XML_NESTED = re.compile(r'<\w+>')

# =============================================================================
# Broken Tool Call Repair Patterns
# =============================================================================
# These patterns handle common hallucinated formats the model produces

def repair_broken_tool_calls(text: str) -> str:
    """
    Repair broken tool call formats that the model hallucinates.
    
    Common broken patterns:
    - <tool><tool_name><param>value</param></tool_name></tool>
    - <tool><tool_name><param1>value1</param1><param2>value2</param2></tool_name></tool>
    - <tool><tool_name><param>value</param></tool> (missing </tool_name>)
    - <tool_call><tool_name></tool_name> (missing parameters)
    - <wrong_tag><tool_name></wrong_tag> (wrong wrapper tag)
    
    Converts to valid format:
    - <tool>{"name": "tool_name", "arguments": {"param": "value", ...}}</tool>
    """
    if not text:
        return text
    
    text_lower = text.lower()
    # Check for any tool-related tags OR known tool names used as tags
    known_tool_tags = ['<tool>', '<tool_call>', '<function', '<fetch_instructions>', '<list_files>',
                       '<read_file>', '<write_file>', '<search_files>', '<execute_command>',
                       '<ask_followup_question>', '<attempt_completion>', '<browser_action>',
                       '<new_task>', '<switch_mode>', '<update_todo_list>']
    if not any(tag in text_lower for tag in known_tool_tags):
        return text
    
    # Pattern -1: Fix <tool_call> wrapper format (convert to <tool> for consistency)
    # Example: <tool_call><list_files></list_files> -> <tool><list_files></list_files></tool>
    # Also handles: <tool_call><tool><name>x</name></tool></tool_call> style
    pattern_wrapper = re.compile(
        r'<tool_call>\s*(<[^/][^>]*>.*?</[^>]+>)\s*</tool_call>',
        re.DOTALL | re.IGNORECASE
    )
    
    def fix_wrapper(match):
        inner_content = match.group(1).strip()
        # If inner content already has <tool> wrapper, keep it as-is
        if inner_content.startswith('<tool>'):
            return inner_content
        # Otherwise wrap in <tool> tags
        return f'<tool>{inner_content}</tool>'
    
    text = pattern_wrapper.sub(fix_wrapper, text)
    
    # Pattern -2: Handle wrong wrapper tags like <fetch_instructions>, <list_files> used as wrappers
    # Example: <fetch_instructions><task>read_file</task><file_path>x</file_path></fetch_instructions>
    # This should become: <tool>{"name": "fetch_instructions", "arguments": {"task": "read_file", "file_path": "x"}}</tool>
    known_tools = ['read_file', 'write_file', 'list_files', 'search_files', 'execute_command',
                   'fetch_instructions', 'ask_followup_question', 'attempt_completion',
                   'browser_action', 'new_task', 'switch_mode', 'update_todo_list']
    
    for tool_name in known_tools:
        pattern_wrong_wrapper = re.compile(
            rf'<{tool_name}>\s*((?:<\w+>[^<]*</\w+>\s*)+)\s*</{tool_name}>',
            re.DOTALL | re.IGNORECASE
        )
        
        def fix_wrong_wrapper(match, _tool_name=tool_name):
            params_content = match.group(1)
            # Extract all key-value pairs
            params = {}
            for param_name, value in re.findall(r'<(\w+)>([^<]*)</\1>', params_content, re.DOTALL):
                try:
                    val = json.loads(value.strip())
                except:
                    val = value.strip()
                params[param_name] = val
            
            if params:
                return f'<tool>{{"name": "{_tool_name}", "arguments": {json.dumps(params)}}}</tool>'
            else:
                return f'<tool>{{"name": "{_tool_name}", "arguments": {{}}}}</tool>'
        
        text = pattern_wrong_wrapper.sub(fix_wrong_wrapper, text)
    
    # Pattern -3: Handle incomplete tool calls with missing parameters
    # Example: <tool><list_files></list_files> or <tool_call><list_files></list_files>
    # Try to infer common parameters from context
    pattern_incomplete = re.compile(
        r'<tool>\s*<(\w+)>\s*</\1>\s*</tool>',
        re.DOTALL | re.IGNORECASE
    )
    
    def fix_incomplete(match):
        tool_name = match.group(1)
        # Provide default parameters for common tools
        default_params = {
            'list_files': {'path': '.', 'recursive': False},
            'read_file': {'files': [{'path': 'README.md'}]},
            'search_files': {'path': '.', 'regex': '.*', 'file_pattern': None},
        }
        
        params = default_params.get(tool_name, {})
        return f'<tool>{{"name": "{tool_name}", "arguments": {json.dumps(params)}}}</tool>'
    
    text = pattern_incomplete.sub(fix_incomplete, text)
    
    # Pattern 0: <tool><TOOL_NAME><PARAM>value</PARAM></tool>
    # This handles the most common broken format WITHOUT closing tag for tool name
    # Example: <tool><list_files><path>.</path><recursive>true</recursive></tool>
    pattern0 = re.compile(
        r'<tool>\s*<(\w+)>\s*((?:<(?:parameter|param|arg|argument|property|key)[^>]*>[^<]*</(?:parameter|param|arg|argument|property|key)>\s*)+)\s*</tool>',
        re.DOTALL | re.IGNORECASE
    )
    
    def replacer0(match):
        tool_name = match.group(1)
        params_content = match.group(2)
        # Extract all parameter name/value pairs
        param_pattern = re.compile(r'<(?:parameter|param|arg|argument|property|key)[^>]*>([^<]*)</(?:parameter|param|arg|argument|property|key)>', re.IGNORECASE)
        params = {}
        
        # Try to find the parameter names from the tags
        param_name_pattern = re.compile(r'<((?:parameter|param|arg|argument|property|key)[^>]*)>([^<]*)</\1>', re.IGNORECASE)
        for name_match, value_match in param_name_pattern.findall(params_content):
            # Extract the actual parameter name (strip prefix like 'parameter=')
            param_name = name_match.replace('parameter=', '').replace('param=', '').replace('arg=', '').strip()
            if param_name and param_name not in ['parameter', 'param', 'arg', 'argument', 'property', 'key']:
                try:
                    val = json.loads(value_match.strip())
                except:
                    val = value_match.strip()
                params[param_name] = val
        
        if params:
            return f'<tool>{{"name": "{tool_name}", "arguments": {json.dumps(params)}}}</tool>'
        else:
            return f'<tool>{{"name": "{tool_name}", "arguments": {{}}}}</tool>'
    
    text = pattern0.sub(replacer0, text)
    
    # Pattern 0a: <tool><TOOL_NAME><PARAM_NAME>value</PARAM_NAME></TOOL_NAME></tool>
    # Format with closing tag for tool name: <tool><list_files><path>.</path></list_files></tool>
    pattern0a = re.compile(
        r'<tool>\s*<(\w+)>\s*((?:<\w+>[^<]*</\w+>\s*)+)\s*</\1>\s*</tool>',
        re.DOTALL | re.IGNORECASE
    )
    
    def replacer0a(match):
        tool_name = match.group(1)
        params_content = match.group(2)
        
        # Skip if this looks like a structural tag (not a real tool)
        if tool_name.lower() in ['name', 'arguments', 'parameters', 'function', 'action', 'tool', 'tool_call']:
            return match.group(0)
        
        # Extract all key-value pairs from simple XML tags
        simple_params = re.findall(r'<(\w+)>([^<]*)</\1>', params_content, re.DOTALL)
        params = {}
        for param_name, value in simple_params:
            if param_name.lower() in ['name', 'arguments', 'parameters', 'function', 'action', 'tool', 'tool_call']:
                continue
            try:
                val = json.loads(value.strip())
            except:
                val = value.strip()
            params[param_name] = val
        
        if params:
            return f'<tool>{{"name": "{tool_name}", "arguments": {json.dumps(params)}}}</tool>'
        else:
            return f'<tool>{{"name": "{tool_name}", "arguments": {{}}}}</tool>'
    
    text = pattern0a.sub(replacer0a, text)
    
    # Pattern 0b: <tool><TOOL_NAME><PARAM_NAME>value</PARAM_NAME></tool>
    # Even simpler format without parameter= prefix: <tool><list_files><path>.</path></tool>
    pattern0b = re.compile(
        r'<tool>\s*<(\w+)>\s*((?:<\w+>[^<]*</\w+>\s*)+)\s*</tool>',
        re.DOTALL | re.IGNORECASE
    )
    
    def replacer0b(match):
        tool_name = match.group(1)
        params_content = match.group(2)
        
        # Skip if this looks like a structural tag (not a real tool)
        if tool_name.lower() in ['name', 'arguments', 'parameters', 'function', 'action', 'tool', 'tool_call']:
            return match.group(0)
        
        # Extract all key-value pairs from simple XML tags
        simple_params = re.findall(r'<(\w+)>([^<]*)</\1>', params_content, re.DOTALL)
        params = {}
        for param_name, value in simple_params:
            # Skip structural tag names
            if param_name.lower() in ['name', 'arguments', 'parameters', 'function', 'action', 'tool', 'tool_call']:
                continue
            # Try to parse as JSON, otherwise use as string
            try:
                val = json.loads(value.strip())
            except:
                val = value.strip()
            params[param_name] = val
        
        if params:
            return f'<tool>{{"name": "{tool_name}", "arguments": {json.dumps(params)}}}</tool>'
        else:
            return f'<tool>{{"name": "{tool_name}", "arguments": {{}}}}</tool>'
    
    text = pattern0b.sub(replacer0b, text)
    
    # Pattern 1: <tool><TOOL_NAME><PARAM>value</PARAM></TOOL_NAME></tool>
    # This is another common hallucination with closing tag for tool name
    pattern1 = re.compile(
        r'<tool>\s*<(\w+)>\s*(<(?:parameter|param|arg|argument|property|key)[^>]*>([^<]*)</(?:parameter|param|arg|argument|property|key)>\s*)+</\1>\s*</tool>',
        re.DOTALL | re.IGNORECASE
    )
    
    def replacer1(match):
        tool_name = match.group(1)
        # Extract all parameter name/value pairs
        param_pattern = re.compile(r'<(?:parameter|param|arg|argument|property|key)[^>]*>([^<]*)</(?:parameter|param|arg|argument|property|key)>', re.IGNORECASE)
        params = {}
        for pmatch in param_pattern.findall(match.group(0)):
            # Try to parse as JSON, otherwise use as string
            try:
                val = json.loads(pmatch.strip())
            except:
                val = pmatch.strip()
            # Use a generic parameter name if we can't determine it
            param_idx = len(params)
            params[f"param_{param_idx}"] = val
        
        # Also try to find the parameter names from the tags
        param_name_pattern = re.compile(r'<((?:parameter|param|arg|argument|property|key)[^>]*)>([^<]*)</\1>', re.IGNORECASE)
        named_params = {}
        for name_match, value_match in param_name_pattern.findall(match.group(0)):
            # Extract the actual parameter name (strip prefix like 'parameter=')
            param_name = name_match.replace('parameter=', '').replace('param=', '').replace('arg=', '').strip()
            if param_name and param_name not in ['parameter', 'param', 'arg', 'argument', 'property', 'key']:
                try:
                    val = json.loads(value_match.strip())
                except:
                    val = value_match.strip()
                named_params[param_name] = val
        
        # Merge: named params override indexed params
        if named_params:
            params = named_params
        
        if params:
            return f'<tool>{{"name": "{tool_name}", "arguments": {json.dumps(params)}}}</tool>'
        else:
            return f'<tool>{{"name": "{tool_name}", "arguments": {{}}}}</tool>'
    
    text = pattern1.sub(replacer1, text)
    
    # Pattern 2: <tool><TOOL_NAME>value</TOOL_NAME></tool> - tool name as tag with direct value
    pattern2 = re.compile(
        r'<tool>\s*<(\w+)>\s*([^<]+)\s*</\1>\s*</tool>',
        re.DOTALL | re.IGNORECASE
    )
    
    def replacer2(match):
        tool_name = match.group(1)
        value = match.group(2).strip()
        # Try to parse value as JSON
        try:
            args = json.loads(value)
        except:
            args = {"value": value}
        return f'<tool>{{"name": "{tool_name}", "arguments": {json.dumps(args)}}}</tool>'
    
    text = pattern2.sub(replacer2, text)
    
    # Pattern 3: Fix <tool><name>TOOL_NAME</name>...<arguments>...</arguments></tool> missing closing
    # This handles incomplete tool calls that were cut off
    pattern3 = re.compile(
        r'<tool>\s*<name>\s*(\w+)\s*</name>\s*<arguments>([^<]*(?:<[^/][^>]*>[^<]*</[^>]*>[^<]*)*)</arguments>\s*</tool>',
        re.DOTALL | re.IGNORECASE
    )
    
    def replacer3(match):
        tool_name = match.group(1)
        args_str = match.group(2).strip()
        # Try to extract JSON from arguments section
        try:
            # Look for JSON-like structure
            json_match = re.search(r'\{[^{}]*\}', args_str)
            if json_match:
                args = json.loads(json_match.group(0))
            else:
                args = {}
        except:
            args = {}
        return f'<tool>{{"name": "{tool_name}", "arguments": {json.dumps(args)}}}</tool>'
    
    text = pattern3.sub(replacer3, text)
    
    # Pattern 4: <tool><function>TOOL_NAME</function><parameters>...</parameters></tool>
    # This is a common hallucination where the model uses <function> for the name
    # and <parameters> for the args, but the parameters are XML not JSON
    # The standalone_function TOOL_PATTERN already handles this for extraction,
    # but we need to ensure the parameters XML gets properly converted
    pattern4 = re.compile(
        r'<tool>\s*<function>\s*(\w+)\s*</function>\s*<parameters>\s*((?:<\w+>[^<]*</\w+>\s*)*)\s*</parameters>\s*</tool>',
        re.DOTALL | re.IGNORECASE
    )
    
    def replacer4(match):
        tool_name = match.group(1)
        params_content = match.group(2).strip()
        
        # Extract all key-value pairs from XML tags
        params = {}
        for param_name, value in re.findall(r'<(\w+)>([^<]*)</\1>', params_content, re.DOTALL):
            try:
                val = json.loads(value.strip())
            except:
                val = value.strip()
            params[param_name] = val
        
        if params:
            return f'<tool>{{"name": "{tool_name}", "arguments": {json.dumps(params)}}}</tool>'
        else:
            return f'<tool>{{"name": "{tool_name}", "arguments": {{}}}}</tool>'
    
    text = pattern4.sub(replacer4, text)
    
    # Post-processing: Fill in missing required parameters with defaults
    # This handles cases where the model produces a valid tool call but omits required params
    _default_required_params = {
        'list_files': {'path': '.'},
        'search_files': {'path': '.'},
        'read_file': {},
    }
    
    def _fill_missing_params(match):
        """Fill in missing required parameters in JSON tool calls."""
        json_str = match.group(1)
        try:
            data = json.loads(json_str)
            tool_name = data.get('name', '')
            args = data.get('arguments', {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except:
                    args = {}
            
            # Fill in missing required params
            if tool_name in _default_required_params:
                defaults = _default_required_params[tool_name]
                for key, default_val in defaults.items():
                    if key not in args:
                        args[key] = default_val
                        print(f"DEBUG repair: Added missing required param '{key}' = {default_val!r} for tool '{tool_name}'")
                data['arguments'] = args
            
            return f'<tool>{json.dumps(data)}</tool>'
        except:
            return match.group(0)
    
    # Apply to all <tool>{JSON}</tool> patterns
    text = re.sub(r'<tool>(\{[^}]*\})</tool>', _fill_missing_params, text, flags=re.DOTALL)
    # Also handle multi-line JSON
    text = re.sub(r'<tool>(\{.*?\})</tool>', _fill_missing_params, text, flags=re.DOTALL)
    
    return text

# Content filtering patterns - pre-compiled
MALFORMED_PATTERNS = [
    re.compile(r'<<<<<<<\s+SEARCH.*?=======', re.DOTALL),
    re.compile(r'=======.*?>>>>>>>\s+REPLACE', re.DOTALL),
    re.compile(r'>>>>>>>\s+REPLACE'),
    re.compile(r'<<<<<<<\s+SEARCH\s*:start_line:\d+[^<]*', re.DOTALL),
    re.compile(r'<button>Stop Generation</button>'),
    re.compile(r'\<\|assistant\|\>'),
    re.compile(r'\</\|assistant\|\>'),
    re.compile(r'\n{3,}'),
]

# Tool call stripping patterns - pre-compiled
STRIP_TOOL_PATTERNS = [
    re.compile(r'<tool>.*?</tool>', re.DOTALL),
    re.compile(r'<function>.*?</function>', re.DOTALL),
    re.compile(r'<tool>\{.*?\}</tool>', re.DOTALL),
    re.compile(r'<tool>[\s\S]*?</tool>'),
    re.compile(r'<function>[\s\S]*?</function>'),
]

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
        # REPAIR: Fix broken tool call formats that the model hallucinates
        # This handles cases like <tool><tool_name><param>value</param></tool_name></tool>
        text = repair_broken_tool_calls(text)
        
        # 0. PRE-VALIDATION: Check if text looks like reasoning output
        # If text contains thinking/reasoning tags, extract only the content after them
        # This prevents parsing partial tool calls from reasoning blocks
        thinking_pattern = r'<\|.*?\|>|<(?:thought|think)>.*?((?:</(?:thought|think)>)|$)|<\|begin.*?\|><\|end.*?\|>'
        has_thinking = re.search(thinking_pattern, text, flags=re.IGNORECASE)
        
        # If text has thinking tags, check if there's actual content after them
        if has_thinking:
            # Find the last thinking tag position
            thinking_matches = list(re.finditer(thinking_pattern, text, flags=re.DOTALL | re.IGNORECASE))
            if thinking_matches:
                last_think_end = thinking_matches[-1].end()
                content_after_thinking = text[last_think_end:].strip()
                # If there's no meaningful content after thinking, return empty
                if not content_after_thinking or len(content_after_thinking) < 5:
                    print(f"DEBUG QwenParser: Text appears to be reasoning only, no content after thinking tags")
                    return []
        
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

            # Validate JSON is complete before accepting
            if not validate_json_complete(json_str):
                print(f"DEBUG QwenParser: JSON appears incomplete, skipping: {json_str[:50]}...")
                continue
            
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
        
        # 5. Fallback: if no tool calls found, try using ToolCallParser
        if not results:
            tool_call_parser = ToolCallParser()
            fallback_calls = tool_call_parser.extract_tool_calls(text, [])
            if fallback_calls:
                print(f"DEBUG QwenParser: ToolCallParser fallback found {len(fallback_calls)} tool calls")
                results.extend(fallback_calls)

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

        # DeepSeek V4 (ds4) native DSML tool calls: <｜DSML｜invoke name="…">…
        for name, args in parse_deepseek_dsml_tool_calls(
                text, set(self.tools.keys()) if self.tools else None):
            results.append(self._to_oa(name, args))
        if results:
            return results

        # Degraded plaintext <tool>name arg: value</tool> from heavy quants (e.g.
        # the ds4 q2-imatrix), which can't reliably emit the exact DSML tokens.
        if self.tools:
            for name, args in parse_tool_tag_plaintext_calls(text, set(self.tools.keys())):
                results.append(self._to_oa(name, args))
            if results:
                return results

        # DeepSeek-V3 uses specialized JSON prompts
        calls = re.findall(r'\{"name":\s*"(.*?)",\s*"parameters":\s*(\{.*?\})}', text)
        for name, params in calls:
            try:
                results.append(self._to_oa(name, json.loads(params)))
            except:
                continue
        
        # Fallback: if no tool calls found, try using ToolCallParser
        if not results:
            tool_call_parser = ToolCallParser()
            fallback_calls = tool_call_parser.extract_tool_calls(text, [])
            if fallback_calls:
                print(f"DEBUG DeepSeekParser: ToolCallParser fallback found {len(fallback_calls)} tool calls")
                results.extend(fallback_calls)
        
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
        
        # Fallback: if no tool calls found, try using ToolCallParser
        if not results:
            tool_call_parser = ToolCallParser()
            fallback_calls = tool_call_parser.extract_tool_calls(text, [])
            if fallback_calls:
                print(f"DEBUG LlamaParser: ToolCallParser fallback found {len(fallback_calls)} tool calls")
                results.extend(fallback_calls)
        
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
        
        # Fallback: if no tool calls found, try using ToolCallParser
        if not results:
            tool_call_parser = ToolCallParser()
            fallback_calls = tool_call_parser.extract_tool_calls(text, [])
            if fallback_calls:
                print(f"DEBUG MistralParser: ToolCallParser fallback found {len(fallback_calls)} tool calls")
                results.extend(fallback_calls)
        
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
        
        # Fallback: if no tool calls found, try using ToolCallParser
        if not results:
            tool_call_parser = ToolCallParser()
            fallback_calls = tool_call_parser.extract_tool_calls(text, [])
            if fallback_calls:
                print(f"DEBUG ClaudeParser: ToolCallParser fallback found {len(fallback_calls)} tool calls")
                results.extend(fallback_calls)
        
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
        
        # Fallback: if no tool calls found, try using ToolCallParser
        if not results:
            tool_call_parser = ToolCallParser()
            fallback_calls = tool_call_parser.extract_tool_calls(text, [])
            if fallback_calls:
                print(f"DEBUG CommandRParser: ToolCallParser fallback found {len(fallback_calls)} tool calls")
                results.extend(fallback_calls)
        
        return results


def _parse_gemma_loose_value(s: str, i: int):
    """Parse one value from gemma's loose object notation starting at index i.
    Returns (python_value, next_index). Handles "strings", numbers, true/false/
    null, nested {objects} and [arrays], and bareword fallbacks."""
    n = len(s)
    while i < n and s[i] in ' \t\r\n':
        i += 1
    if i >= n:
        return None, i
    c = s[i]
    if c == '"':
        # JSON-style string with escapes.
        j = i + 1
        buf = []
        while j < n:
            if s[j] == '\\' and j + 1 < n:
                esc = s[j + 1]
                buf.append({'n': '\n', 't': '\t', 'r': '\r'}.get(esc, esc))
                j += 2
                continue
            if s[j] == '"':
                j += 1
                break
            buf.append(s[j])
            j += 1
        return ''.join(buf), j
    if c == '{':
        return _parse_gemma_loose_object(s, i)
    if c == '[':
        arr = []
        j = i + 1
        while j < n:
            while j < n and s[j] in ' \t\r\n,':
                j += 1
            if j < n and s[j] == ']':
                j += 1
                break
            prev = j
            val, j = _parse_gemma_loose_value(s, j)
            arr.append(val)
            # Malformed input (e.g. a stray '}' where ']' was expected) can leave
            # j unmoved — bail instead of spinning forever appending empties.
            if j <= prev:
                break
        return arr, j
    # Bareword / number / bool / null: read until a delimiter.
    j = i
    while j < n and s[j] not in ',}]':
        j += 1
    tok = s[i:j].strip()
    low = tok.lower()
    if low == 'true':
        return True, j
    if low == 'false':
        return False, j
    if low in ('null', 'none'):
        return None, j
    try:
        return int(tok), j
    except ValueError:
        pass
    try:
        return float(tok), j
    except ValueError:
        pass
    return tok, j


def _parse_gemma_loose_object(s: str, i: int):
    """Parse a {key:value,…} object (unquoted keys) starting at the '{' at i.
    Returns (dict, next_index)."""
    n = len(s)
    obj = {}
    assert s[i] == '{'
    j = i + 1
    while j < n:
        loop_start = j
        while j < n and s[j] in ' \t\r\n,':
            j += 1
        if j < n and s[j] == '}':
            j += 1
            break
        # Read key (bareword or "quoted").
        if s[j] == '"':
            key, j = _parse_gemma_loose_value(s, j)
        else:
            k = j
            while j < n and s[j] not in ':}':
                j += 1
            key = s[k:j].strip()
        while j < n and s[j] in ' \t\r\n':
            j += 1
        if j < n and s[j] == ':':
            j += 1
        val, j = _parse_gemma_loose_value(s, j)
        if key:
            obj[key] = val
        # Forward-progress guard: malformed input must never wedge the loop.
        if j <= loop_start:
            break
    return obj, j


def parse_gemma_native_tool_calls(text: str, tool_names=None):
    """Parse gemma-4's native tool-call format — ``call:NAME{args}`` (optionally
    wrapped in the ``<|tool_call>…<tool_call|>`` special tokens) — into a list of
    ``(name, args_dict)``. ``tool_names`` (when given) restricts matches to real
    tool names so prose containing ``call:`` isn't misread. Exact-duplicate calls
    are collapsed (a degenerate model loop emits the same call repeatedly)."""
    if not text or 'call:' not in text:
        return []
    out = []
    seen = set()
    for m in re.finditer(r'call:\s*([A-Za-z_]\w*)\s*\{', text):
        name = m.group(1)
        if tool_names and name not in tool_names:
            continue
        brace = m.end() - 1   # index of '{'
        # Some models double-wrap the args: call:NAME{{"k":"v"}}. Skip the
        # redundant outer brace so the real object is parsed instead of being
        # mangled into a single key like '{"k"'.
        j = brace + 1
        while j < len(text) and text[j] in ' \t\r\n':
            j += 1
        if j < len(text) and text[j] == '{':
            brace = j
        try:
            args, _ = _parse_gemma_loose_object(text, brace)
        except Exception:
            continue
        key = (name, json.dumps(args, sort_keys=True, default=str))
        if key in seen:
            continue
        seen.add(key)
        out.append((name, args))
    return out


def parse_xml_wrapped_tool_calls(text: str, tool_names):
    """Parse ``<NAME>…</NAME>`` tool calls where NAME is a declared tool.

    Some clients (Kilo/Cline/Roo-style) describe tools in the system prompt and
    instruct the model to emit XML-tagged calls. Models then produce e.g.
    ``<bash>{"command": "ls"}</bash>`` (JSON args) or ``<bash><command>ls</command>
    </bash>`` (nested XML params). Neither matches a model's native tool format,
    so this recovers them into ``(name, args_dict)``. Restricted to real tool
    names so ordinary tagged prose (``<thinking>`` …) isn't misread."""
    if not text or not tool_names:
        return []
    out, seen = [], set()
    for name in tool_names:
        for m in re.finditer(rf'<{re.escape(name)}\s*>(.*?)</{re.escape(name)}\s*>',
                             text, re.DOTALL):
            inner = m.group(1).strip()
            args = None
            if inner.startswith('{'):
                try:
                    args = json.loads(inner)
                except Exception:
                    args = None
            if args is None:
                params = re.findall(r'<(\w+)\s*>(.*?)</\1\s*>', inner, re.DOTALL)
                if params:
                    args = {k: v.strip() for k, v in params}
            if not isinstance(args, dict):
                continue
            key = (name, json.dumps(args, sort_keys=True, default=str))
            if key in seen:
                continue
            seen.add(key)
            out.append((name, args))
    return out


def parse_tool_tag_json_calls(text: str, tool_names=None):
    """Parse tool calls emitted as ``<tool>{"name":..,"arguments":..}`` markers.

    Some Gemma finetunes use ``<tool>`` (or ``<tool_call>``) as BOTH the opening
    and closing delimiter — i.e. no closing ``</tool>`` slash — and sometimes
    append a stray ``"`` before the closer, so the strict ``<tool>{…}</tool>``
    patterns never match. Extract the first brace-balanced object after each
    marker (tolerant of trailing junk via the loose parser) and read
    name + arguments/parameters. ``tool_names`` (when given) restricts matches to
    declared tools so prose isn't misread; exact-duplicate calls are collapsed
    (these models often repeat the same call several times)."""
    if not text or '<tool' not in text.lower():
        return []
    out, seen = [], set()
    for m in re.finditer(r'<tool(?:_call)?\s*>', text, re.IGNORECASE):
        brace = text.find('{', m.end())
        if brace == -1:
            continue
        # The object must follow the marker directly — only whitespace/quotes may
        # sit between them — so a '{' belonging to later prose isn't grabbed.
        if text[m.end():brace].strip(' \t\r\n"\'`'):
            continue
        try:
            data, _ = _parse_gemma_loose_object(text, brace)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        name = data.get('name')
        args = data.get('arguments')
        if args is None:
            args = data.get('parameters', {})
        if not name or not isinstance(name, str):
            continue
        if tool_names and name not in tool_names:
            continue
        if not isinstance(args, dict):
            args = {}
        key = (name, json.dumps(args, sort_keys=True, default=str))
        if key in seen:
            continue
        seen.add(key)
        out.append((name, args))
    return out


def parse_tool_tag_plaintext_calls(text: str, tool_names=None):
    """Parse the degraded PLAINTEXT ``<tool>`` format some heavily-quantized models
    emit instead of JSON/DSML, e.g.::

        <tool>
        read filePath: /path/to/file
        </tool>

    The block wraps a first token = tool name, then one or more ``key: value``
    argument lines (the name and its first arg may share line 1; each value is split
    on its FIRST colon, so paths/URLs in the value survive). Only blocks whose name
    is a DECLARED tool are accepted (``tool_names`` is REQUIRED) and the inner text
    must be plain (no ``<…>`` sub-tags, no leading ``{``) so JSON/XML ``<tool>``
    forms are left to their own parsers and ordinary prose isn't misread. This is a
    best-effort rescue for low-bit quants that can't reproduce the exact tool-call
    tokens — higher-quant models emit proper formats and never reach here.

    To avoid catching a ``<tool>`` *example* a model writes inside an explanatory
    reply, AND to avoid amplifying the degenerate ``<tool>name</tool><tool>name</tool>…``
    spam a too-low quant emits when it falls apart, this is deliberately strict:
      - the block(s) must be the message's trailing ACTION — after the first
        ``<tool>`` everything to end-of-text must be only ``<tool>…</tool>`` blocks
        and whitespace (prose after/between → treated as text, not calls);
      - every block must carry at least one ``key: value`` argument — a BARE
        ``<tool>name</tool>`` is the spam signature, never a real call here;
      - the whole reply may yield at most a few distinct calls — a flood of blocks
        is model degeneration, so the batch is rejected wholesale.
    Returns ``[(name, args), …]``."""
    if not text or '<tool' not in text.lower() or not tool_names:
        return []
    blocks = list(re.finditer(r'<tool\s*>(.*?)</tool\s*>', text, re.DOTALL | re.IGNORECASE))
    if not blocks:
        return []
    # A flood of <tool> tags is a model falling apart, not many real calls.
    if len(blocks) > 6:
        return []
    # Require the tag(s) to form the trailing run of the message: strip the matched
    # blocks out of the tail (from the first block on) and demand only whitespace is
    # left. Otherwise this is prose that merely mentions the <tool> syntax.
    tail = text[blocks[0].start():]
    for b in blocks:
        tail = tail.replace(b.group(0), '', 1)
    if tail.strip():
        return []
    out, seen = [], set()
    for m in blocks:
        inner = m.group(1).strip()
        if not inner or inner.startswith('{') or '<' in inner:
            continue
        lines = [ln.strip() for ln in inner.splitlines() if ln.strip()]
        if not lines:
            continue
        head = lines[0].split(None, 1)
        name = head[0].strip().strip(':')
        if name not in tool_names:
            continue
        arg_lines = ([head[1]] if len(head) > 1 else []) + lines[1:]
        args = {}
        for al in arg_lines:
            if ':' in al:
                k, v = al.split(':', 1)
                k = k.strip()
                if k:
                    args[k] = v.strip()
        # A bare <tool>name</tool> with no arguments is the degenerate-spam shape,
        # never a real call in this format — drop it.
        if not args:
            continue
        key = (name, json.dumps(args, sort_keys=True, default=str))
        if key in seen:
            continue
        seen.add(key)
        out.append((name, args))
    return out


# 7. GEMMA PARSER
# DeepSeek V4 (ds4) special-token bar ｜ (U+FF5C); tolerate an ASCII | fallback.
_DSML_INVOKE = re.compile(
    r'<[｜|]DSML[｜|]invoke\s+name="([^"]+)"\s*>(.*?)'
    r'</[｜|]DSML[｜|]invoke\s*>', re.DOTALL)
_DSML_PARAM = re.compile(
    r'<[｜|]DSML[｜|]parameter\s+name="([^"]+)"([^>]*)>(.*?)'
    r'</[｜|]DSML[｜|]parameter\s*>', re.DOTALL)


def parse_deepseek_dsml_tool_calls(text: str, tool_names=None):
    """Parse DeepSeek V4 DSML tool calls into ``(name, args_dict)``.

    ds4-server emits its native tool calls as
    ``<｜DSML｜invoke name="read"><｜DSML｜parameter name="filePath" string="true">
    /path</｜DSML｜parameter></｜DSML｜invoke>`` (the ``｜`` is U+FF5C). Our parsers
    didn't know this shape, so the whole block was streamed to the client as raw
    content. Recover it here. A ``string="false"`` parameter is decoded as JSON
    (number/bool/object); otherwise the value is kept as a string."""
    if not text or 'DSML' not in text:
        return []
    names = set(tool_names or [])
    out, seen = [], set()
    for m in _DSML_INVOKE.finditer(text):
        name = m.group(1).strip()
        if names and name not in names:
            continue
        args = {}
        for pm in _DSML_PARAM.finditer(m.group(2)):
            pname = pm.group(1).strip()
            attrs = pm.group(2) or ''
            val = pm.group(3).strip()
            if 'string="true"' not in attrs:
                try:
                    val = json.loads(val)
                except Exception:
                    pass
            args[pname] = val
        key = (name, json.dumps(args, sort_keys=True, default=str))
        if key in seen:
            continue
        seen.add(key)
        out.append((name, args))
    return out


def strip_dsml_tool_calls(text: str) -> str:
    """Remove DeepSeek V4 DSML tool-call markup from displayed content, so the
    raw ``<｜DSML｜…>`` block doesn't leak once it's been parsed into tool_calls."""
    if not text or 'DSML' not in text:
        return text
    text = re.sub(r'<[｜|]DSML[｜|]tool_calls>[\s\S]*?</[｜|]DSML[｜|]tool_calls\s*>', '', text)
    text = re.sub(r'<[｜|]DSML[｜|]invoke[\s\S]*?</[｜|]DSML[｜|]invoke\s*>', '', text)
    text = re.sub(r'</?[｜|]DSML[｜|][^>]*>', '', text)  # any residual stray tags
    return text


class GemmaParser(BaseParser):
    @validate_tool_output
    def parse(self, text: str) -> List[Dict]:
        results = []

        # gemma-4 native format: call:NAME{args} (the <|tool_call>…<tool_call|>
        # markers are stripped by skip_special_tokens during decode). Restrict to
        # declared tool names when we know them, to avoid matching prose.
        native = parse_gemma_native_tool_calls(
            text, set(self.tools.keys()) if self.tools else None)
        for name, args in native:
            results.append(self._to_oa(name, args))
        if results:
            return results

        match = re.search(r'{\s*"name":\s*".*?"\s*,\s*"parameters":\s*\{.*?\}\s*\}', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                results.append(self._to_oa(data["name"], data["parameters"]))
            except:
                pass

        # XML-tagged tool calls (<bash>{…}</bash>) emitted when the client (Kilo/
        # Cline-style) prompts for XML tools rather than the model's native format.
        if not results and self.tools:
            for name, args in parse_xml_wrapped_tool_calls(text, set(self.tools.keys())):
                results.append(self._to_oa(name, args))
            if results:
                return results

        # <tool>{"name":..,"arguments":..}<tool> — some finetunes wrap a JSON call
        # in <tool> markers with no closing slash (and stray quotes); recover it.
        if not results:
            names = set(self.tools.keys()) if self.tools else None
            for name, args in parse_tool_tag_json_calls(text, names):
                results.append(self._to_oa(name, args))
            if results:
                return results

        # Fallback: if no tool calls found, try using ToolCallParser
        if not results:
            tool_call_parser = ToolCallParser()
            fallback_calls = tool_call_parser.extract_tool_calls(text, [])
            if fallback_calls:
                print(f"DEBUG GemmaParser: ToolCallParser fallback found {len(fallback_calls)} tool calls")
                results.extend(fallback_calls)

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
        
        # Fallback: if no tool calls found, try using ToolCallParser
        if not results:
            tool_call_parser = ToolCallParser()
            fallback_calls = tool_call_parser.extract_tool_calls(text, [])
            if fallback_calls:
                print(f"DEBUG GrokParser: ToolCallParser fallback found {len(fallback_calls)} tool calls")
                results.extend(fallback_calls)
        
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
        
        # Fallback: if still no tool calls found, try using ToolCallParser
        if not results:
            tool_call_parser = ToolCallParser()
            fallback_calls = tool_call_parser.extract_tool_calls(text, [])
            if fallback_calls:
                print(f"DEBUG PhiParser: ToolCallParser fallback found {len(fallback_calls)} tool calls")
                results.extend(fallback_calls)
        
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

        # Fallback: if no tool calls found, try using ToolCallParser
        if not results:
            tool_call_parser = ToolCallParser()
            fallback_calls = tool_call_parser.extract_tool_calls(text, [])
            if fallback_calls:
                print(f"DEBUG ApexBig50Parser: ToolCallParser fallback found {len(fallback_calls)} tool calls")
                results.extend(fallback_calls)

        return results


# Model Parser Dispatcher
class ModelParserDispatcher:
    """Dispatcher to select the appropriate parser based on model name."""
    
    def __init__(self, model_name: str = None, tools_schema: Dict[str, Any] = None):
        self.model_name = model_name
        self.tools = tools_schema or {}
        # Only log parser selection when a model name is provided (actual use)
        # Skip logging during initialization with model_name=None
        self._log_selection = model_name is not None
        self.parser = self._get_parser()
        self._log_selection = True  # Enable logging for subsequent calls
    
    def _get_parser(self) -> BaseParser:
        """Get the appropriate parser based on model name."""
        if not self.model_name:
            parser = ApexBig50Parser(self.tools)
            # Only log if we're being used for parsing, not during init
            # (self._log_selection is False during __init__ when model_name=None)
            return parser
        
        model_lower = self.model_name.lower()
        
        # Qwen models
        if 'qwen' in model_lower:
            parser = QwenParser(self.tools)
            if self._log_selection:
                print(f"DEBUG model_parser: model_name={self.model_name}, selected parser: QwenParser")
            return parser
        
        # DeepSeek models
        if 'deepseek' in model_lower:
            parser = DeepSeekParser(self.tools)
            if self._log_selection:
                print(f"DEBUG model_parser: model_name={self.model_name}, selected parser: DeepSeekParser")
            return parser
        
        # Llama models
        if 'llama' in model_lower:
            parser = LlamaParser(self.tools)
            if self._log_selection:
                print(f"DEBUG model_parser: model_name={self.model_name}, selected parser: LlamaParser")
            return parser
        
        # Mistral models
        if 'mistral' in model_lower or 'mixtral' in model_lower:
            parser = MistralParser(self.tools)
            if self._log_selection:
                print(f"DEBUG model_parser: model_name={self.model_name}, selected parser: MistralParser")
            return parser
        
        # Claude models
        if 'claude' in model_lower:
            parser = ClaudeParser(self.tools)
            if self._log_selection:
                print(f"DEBUG model_parser: model_name={self.model_name}, selected parser: ClaudeParser")
            return parser
        
        # Command R models
        if 'command' in model_lower:
            parser = CommandRParser(self.tools)
            if self._log_selection:
                print(f"DEBUG model_parser: model_name={self.model_name}, selected parser: CommandRParser")
            return parser
        
        # Gemma models
        if 'gemma' in model_lower:
            parser = GemmaParser(self.tools)
            if self._log_selection:
                print(f"DEBUG model_parser: model_name={self.model_name}, selected parser: GemmaParser")
            return parser
        
        # Grok models
        if 'grok' in model_lower:
            parser = GrokParser(self.tools)
            if self._log_selection:
                print(f"DEBUG model_parser: model_name={self.model_name}, selected parser: GrokParser")
            return parser
        
        # Phi models
        if 'phi' in model_lower:
            parser = PhiParser(self.tools)
            if self._log_selection:
                print(f"DEBUG model_parser: model_name={self.model_name}, selected parser: PhiParser")
            return parser
        
        # Default: use catch-all parser
        parser = ApexBig50Parser(self.tools)
        if self._log_selection:
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

    def format_full(self, text, prompt_tokens, completion_tokens, tool_calls=None, reasoning=None, context_size=None):
        """Standard Response (Non-Streaming)"""
        if LITELLM_AVAILABLE and all([ModelResponse, Choices, Message, Usage]):
            try:
                usage_dict = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
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
                    usage=Usage(**usage_dict)
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
        
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        if context_size is not None:
            usage["context_size"] = context_size
        
        return {
            "id": self.id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [choice],
            "usage": usage,
            "provider": {
                "provider_name": "coderai",
                "provider_id": "coderai",
            },
        }

    def format_chunk(self, delta_text, is_final=False, usage=None, context_size=None):
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
            if context_size is not None:
                chunk["usage"]["context_size"] = context_size
            
        return chunk

    def format_final_chunk(self, usage: dict = None, context_size: int = None) -> dict:
        """Format the final streaming chunk with usage information."""
        return self.format_chunk("", is_final=True, usage=usage, context_size=context_size)

    # Backward compatibility methods
    def format_litellm_full(self, text: str, prompt_tokens: int, completion_tokens: int, tool_calls=None, context_size=None) -> dict:
        """Backward compatibility method - calls format_full."""
        return self.format_full(text, prompt_tokens, completion_tokens, tool_calls, context_size=context_size)

    def format_litellm_chunk(self, delta_text: str, is_final: bool = False, usage: dict = None, context_size: int = None) -> dict:
        """Backward compatibility method - calls format_chunk."""
        return self.format_chunk(delta_text, is_final, usage, context_size)


# =============================================================================
# Content Filtering
# =============================================================================

def filter_malformed_content(text: str) -> str:
    """Filter out malformed SEARCH/REPLACE blocks that the model might output as content.
    
    Uses pre-compiled MALFORMED_PATTERNS for efficiency.
    """
    if not text:
        return text
    
    # Apply repetition filtering first
    text = filter_repetition(text)
    
    # Apply all pre-compiled malformed content patterns
    filtered = text
    for pattern in MALFORMED_PATTERNS:
        # The last pattern (\n{3,}) replaces with \n\n, others replace with ''
        if pattern.pattern == r'\n{3,}':
            filtered = pattern.sub('\n\n', filtered)
        else:
            filtered = pattern.sub('', filtered)
    
    return filtered


# =============================================================================
# Control Token Cleanup
# =============================================================================

# Pre-compiled pattern for cleanup_control_tokens newline cleanup
_RE_TRIPLE_NEWLINE = re.compile(r'\n{3,}')

# Control tokens to strip - defined once at module level
_CONTROL_TOKENS = [
    '<|im_end|>',
    '<|im_start|>',
    '<|endoftext|>',
    '<|end_of_text|>',
    '<|eot_id|>',
    '<|eom_id|>',
    '<|assistant|>',
    '<|model|>',
    '<|python|>',
    '<|javascript|>',
    '<|html|>',
    '\n\nassistant',
    '\nAssistant',
    'ASSISTANT',
    'Assistant',
    'assistant',
]

# Build expanded token set for O(1) lookup (includes \n and space prefixed variants)
_CONTROL_TOKENS_START = set()
_CONTROL_TOKENS_END = set()
for _t in _CONTROL_TOKENS:
    _CONTROL_TOKENS_START.update([_t, '\n' + _t, ' ' + _t])
    _CONTROL_TOKENS_END.update([_t, '\n' + _t, ' ' + _t])
# Sort by length descending so longer tokens match first (avoids partial matches)
_CONTROL_TOKENS_START_SORTED = sorted(_CONTROL_TOKENS_START, key=len, reverse=True)
_CONTROL_TOKENS_END_SORTED = sorted(_CONTROL_TOKENS_END, key=len, reverse=True)


# gemma-4 finetunes (esp. GGUF, where the markers aren't stripped as special tokens)
# emit a CORRUPTED tool-call scheme that breaks both extraction and display:
#   <|tool_call>call:bash{command:<|"|>wc -l README.md<|"|>}<tool_call|><|tool_response>…
# Three distinct corruptions:
#   * <|"|>            — the model's stand-in for a " quote; it leaks into string args
#                        (a bash command becomes  <|"|>wc -l<|"|>  → "syntax error near |")
#   * malformed open/close markers <|tool_call> / <tool_call|> (vs canonical <|tool_call|>)
#   * a hallucinated <|tool_response>…/<turn|> tail (the model fakes the tool result)
_RE_GEMMA_QUOTE = re.compile(r'<\|"\|>')
# Only gemma's MALFORMED variants — deliberately NOT the canonical <|tool_call|> /
# <|tool_call_end|>, which Phi (and others) emit and parse normally; stripping those
# here would break their extraction.
_RE_GEMMA_TOOL_MARKERS = re.compile(
    r'<\|tool_call>|<tool_call\|>'
    r'|<\|tool_response\|?>|<tool_response\|>|<\|turn\|?>|<turn\|>',
    re.IGNORECASE)


def normalize_gemma_tool_tokens(text: str) -> str:
    """Repair gemma-4's corrupted tool-call tokens so calls parse with correct args
    and the markers never leak into user-facing content. Restores ``<|"|>`` → ``"``
    and strips the malformed/hallucinated tool-call markers. The native
    ``call:NAME{…}`` body (and any ``response:NAME{…}`` residue) is left for the tool
    parser / content stripper to handle. Safe on non-gemma text — these exact token
    fragments don't occur in normal output."""
    if not text or ('<|' not in text and '|>' not in text):
        return text
    text = _RE_GEMMA_QUOTE.sub('"', text)
    text = _RE_GEMMA_TOOL_MARKERS.sub('', text)
    return text


def cleanup_control_tokens(text: str) -> str:
    """
    Clean up leading/trailing control tokens from model output.
    
    Removes tokens like <|im_end|>, <|im_start|>, 'assistant', etc. that might
    appear at the start or end of the response after reasoning extraction.
    
    Uses module-level sorted token lists for efficient matching.
    Tokens are sorted by length (longest first) to prevent partial matches
    like 'assistant' matching before '<|assistant|>'.
    """
    if not text:
        return text

    # Repair gemma-4's corrupted tool tokens (interior <|"|> quotes + stray markers)
    # before the start/end stripping below — fixes both display and downstream tool
    # extraction (which runs on this cleaned text).
    cleaned = normalize_gemma_tool_tokens(text)

    # Strip from start - keep trying until no more tokens at start
    # Max iterations bounded by text length / min token length
    changed = True
    while changed:
        changed = False
        for token in _CONTROL_TOKENS_START_SORTED:
            if cleaned.startswith(token):
                cleaned = cleaned[len(token):]
                changed = True
                break  # Restart from longest token after each removal
    
    # Strip from end - keep trying until no more tokens at end
    changed = True
    while changed:
        changed = False
        for token in _CONTROL_TOKENS_END_SORTED:
            if cleaned.endswith(token):
                cleaned = cleaned[:-len(token)]
                changed = True
                break  # Restart from longest token after each removal
    
    # Clean up any resulting triple+ newlines (pre-compiled pattern)
    cleaned = _RE_TRIPLE_NEWLINE.sub('\n\n', cleaned)
    
    # Strip leading/trailing whitespace
    cleaned = cleaned.strip()
    
    return cleaned


def filter_repetition(text: str, min_repeat_count: int = 3, ngram_sizes: tuple = (2, 3)) -> str:
    """
    Detect and remove n-gram repetition from text.
    
    This function looks for sequences of 2-3 words that are repeated 3 or more times
    consecutively (like "does does does" or "the the the the") and removes the duplicates.
    
    Args:
        text: The input text to filter
        min_repeat_count: Minimum number of repetitions to trigger removal (default: 3)
        ngram_sizes: Tuple of n-gram sizes to check (default: (2, 3))
    
    Returns:
        Text with repetition removed
    """
    if not text or len(text) < 10:
        return text
    
    import re
    
    # Split into words while preserving whitespace for reconstruction
    # Use a regex that captures words and the whitespace between them
    parts = re.split(r'(\s+)', text)
    words = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            # Even indices are text content
            words.append(part)
        else:
            # Odd indices are whitespace - attach to previous word
            if words:
                words[-1] = words[-1] + part
    
    if not words:
        return text
    
    # Convert to list of (word, is_word) tuples to track what to keep
    result = []
    i = 0
    
    while i < len(words):
        word = words[i]
        
        # Check if this is a word (contains non-whitespace)
        is_word = bool(word.strip())
        
        if not is_word:
            # Keep whitespace as-is
            result.append(word)
            i += 1
            continue
        
        # Try each n-gram size
        found_repetition = False
        for ngram_size in ngram_sizes:
            if i + ngram_size * min_repeat_count > len(words):
                continue
            
            # Build the n-gram sequence to check
            ngram_parts = []
            valid = True
            for j in range(ngram_size):
                idx = i + j
                if idx >= len(words):
                    valid = False
                    break
                # Get the word part only (strip whitespace)
                w = words[idx].strip()
                if not w:
                    valid = False
                    break
                ngram_parts.append(w)
            
            if not valid or len(ngram_parts) != ngram_size:
                continue
            
            # Check if this n-gram repeats
            ngram_str = ' '.join(ngram_parts)
            repeat_count = 1
            
            # Count consecutive repetitions
            # FIX: Use a simple next_start pointer that advances by ngram_size
            # each time a match is found, instead of the buggy double-advance formula
            next_start = i + ngram_size
            while next_start + ngram_size <= len(words):
                # Check if the n-gram at next_start matches
                next_ngram = []
                for j in range(ngram_size):
                    w = words[next_start + j].strip()
                    if not w:
                        break
                    next_ngram.append(w)
                
                if next_ngram == ngram_parts:
                    repeat_count += 1
                    next_start += ngram_size
                else:
                    break
            
            # If we found enough repetitions, remove duplicates
            if repeat_count >= min_repeat_count:
                # Keep only the first occurrence
                for j in range(ngram_size):
                    result.append(words[i + j])
                
                # Skip all the repeated n-grams
                i += ngram_size * repeat_count
                found_repetition = True
                break
        
        if not found_repetition:
            result.append(word)
            i += 1
    
    return ''.join(result)


def validate_json_complete(json_str: str) -> bool:
    """
    Validate that a JSON string is complete (not truncated).
    
    Checks for:
    - Balanced braces and brackets
    - No unclosed strings
    - Valid structure
    
    Args:
        json_str: The JSON string to validate
    
    Returns:
        True if JSON appears complete, False if it appears truncated
    """
    if not json_str:
        return False
    
    json_str = json_str.strip()
    
    # Check if it starts with { or [
    if not (json_str.startswith('{') or json_str.startswith('[')):
        return False
    
    # Try to parse it
    try:
        json.loads(json_str)
        return True
    except json.JSONDecodeError as e:
        # Check if the error is due to truncation vs. syntax error
        error_msg = str(e)
        
        # Common truncation errors
        if 'Expecting' in error_msg and ('property name' in error_msg or 'value' in error_msg or 'string' in error_msg):
            # This is likely truncated - we got cut off in the middle
            return False
        
        # If we have a valid start but missing end, it's truncated
        if json_str.endswith(',') or json_str.endswith(':'):
            return False
        
        # Check for unclosed braces/brackets
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        open_brackets = json_str.count('[')
        close_brackets = json_str.count(']')
        
        if open_braces > close_braces or open_brackets > close_brackets:
            return False
        
        # Try again - if it still fails, it's a syntax error
        try:
            json.loads(json_str)
            return True
        except:
            return False


# =============================================================================
# Tool Formatting
# =============================================================================

def format_tools_for_prompt(tools, messages, tools_closer_prompt: bool = False):
    """Format tools into the system message or add a tool description.
    
    Args:
        tools: List of Tool objects
        messages: List of ChatMessage objects
        tools_closer_prompt: If True, place tools right before the user's latest message
                           instead of in the system prompt (prompt distillation)
    
    Returns:
        Modified list of ChatMessage objects
    """
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
    
    if tools_closer_prompt:
        # Prompt distillation: insert tools right before the LAST user message
        # Find the last user message and insert tools before it
        last_user_idx = None
        for i, msg in enumerate(new_messages):
            if msg.role == "user":
                last_user_idx = i
        
        if last_user_idx is not None:
            # Insert a tool context message before the last user message
            tools_message = ChatMessage(role="system", content=f"Available tools:\n{tools_text}")
            new_messages.insert(last_user_idx, tools_message)
        else:
            # No user message found, fall back to system message
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
    else:
        # Traditional behavior: prepend tools to system message
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
        
    @staticmethod
    def _parse_args(args_str: str) -> dict:
        """Parse arguments string as JSON, falling back to XML key-value extraction."""
        args_str = args_str.strip() if args_str else ''
        if not args_str:
            return {}
        try:
            return json.loads(args_str)
        except json.JSONDecodeError:
            # Fallback: extract key-value pairs from nested XML tags
            args = {}
            for match in RE_XML_TO_DICT.findall(args_str):
                k, v = match
                v = v.strip()
                try:
                    args[k] = json.loads(v)
                except (json.JSONDecodeError, ValueError):
                    args[k] = v
            return args

    @staticmethod
    def _make_tool_call(name: str, args: dict) -> Dict:
        """Create a tool call dict in OpenAI format."""
        return {
            "id": f"call_{uuid.uuid4().hex[:16]}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(args)
            }
        }

    def _parse_xml_style_tool_calls(self, text: str) -> List[Dict]:
        """Parse XML-style tool calls using pre-compiled TOOL_PATTERNS.
        
        Consolidated from ~20 separate pattern blocks into a single method
        that iterates over pre-compiled patterns. Eliminates duplicate patterns
        (e.g., pattern == pattern_standalone_args_xml, pattern_nested == pattern_nested_args_xml).
        """
        tool_calls = []
        
        # --- Two-group patterns: (name, args_str) ---
        # Each yields tool name and arguments string
        two_group_patterns = [
            # <tool><name>...</name><arguments>...</arguments></tool>
            TOOL_PATTERNS['basic'],
            # <tool><action>...</action><parameters>...</parameters></tool>
            TOOL_PATTERNS['action'],
            # <tool>name</tool><tool_call>JSON</tool_call>
            TOOL_PATTERNS['tool_call_basic'],
            # <tool_call><tool><name>...</name><arguments>...</arguments></tool></tool_call>
            TOOL_PATTERNS['nested'],
            # <tool_call><tool><function>...</function><parameters>...</parameters></tool></tool_call>
            TOOL_PATTERNS['nested_function'],
            # <tool><function>...</function><parameters>...</parameters></tool>
            TOOL_PATTERNS['standalone_function'],
            # <tool><name>...</name><parameters>...</parameters></tool>
            TOOL_PATTERNS['standalone_params'],
            # <tool_call><tool><name>...</name><parameters>...</parameters></tool></tool_call>
            TOOL_PATTERNS['nested_params'],
            # <tool>TOOL_NAME>JSON</tool>
            TOOL_PATTERNS['short'],
            # <tool_call><tool>TOOL_NAME>JSON</tool></tool_call>
            TOOL_PATTERNS['short2'],
        ]
        
        for pattern in two_group_patterns:
            for name, args_str in pattern.findall(text):
                name = name.strip()
                if not name:
                    continue
                args = self._parse_args(args_str)
                tool_calls.append(self._make_tool_call(name, args))
        
        # --- Nested tool name-as-tag patterns: (tool_name, args_xml) ---
        # <tool_call><tool><toolname>...</toolname></tool></tool_call>
        for tool_name, args_xml in TOOL_PATTERNS['nested_tool'].findall(text):
            tool_name = tool_name.strip()
            if not tool_name:
                continue
            args = self._parse_args(args_xml)
            tool_calls.append(self._make_tool_call(tool_name, args))
        
        # <tool><toolname>...</toolname></tool> (standalone)
        for tool_name, args_xml in TOOL_PATTERNS['standalone_nested'].findall(text):
            tool_name = tool_name.strip()
            if not tool_name:
                continue
            # Skip structural tag names to avoid duplicates
            if tool_name in ('name', 'arguments', 'parameters', 'action', 'function'):
                continue
            args = self._parse_args(args_xml)
            if args:  # Only add if we parsed some arguments
                tool_calls.append(self._make_tool_call(tool_name, args))
        
        # --- JSON-in-tool pattern: <tool>{JSON}</tool> ---
        for json_str in TOOL_PATTERNS['json_in_tool'].findall(text):
            try:
                data = json.loads(json_str.strip())
                name = data.get('name') or data.get('function')
                args = data.get('arguments') or data.get('args') or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except:
                        args = {}
                if name:
                    tool_calls.append(self._make_tool_call(name, args))
            except json.JSONDecodeError:
                pass
        
        # --- Multi-tool wrapper: <tool_call><tool>...</tool></tool_call> ---
        for tool_block in TOOL_PATTERNS['multi'].findall(text):
            for name, args_str in TOOL_PATTERNS['basic'].findall(tool_block):
                name = name.strip()
                if not name:
                    continue
                args = self._parse_args(args_str)
                tool_calls.append(self._make_tool_call(name, args))
        
        # --- Multi-tools with optional second block ---
        for tool_block1, tool_block2 in TOOL_PATTERNS['multi_tools'].findall(text):
            for block in (tool_block1, tool_block2):
                if not block:
                    continue
                for name, args_str in TOOL_PATTERNS['basic'].findall(block):
                    name = name.strip()
                    if not name:
                        continue
                    args = self._parse_args(args_str)
                    tool_calls.append(self._make_tool_call(name, args))
        
        # Deduplicate tool calls based on name and arguments
        seen = set()
        unique_tool_calls = []
        for tc in tool_calls:
            name = tc.get('function', {}).get('name', '')
            args = tc.get('function', {}).get('arguments', '')
            
            # Skip empty tool calls (no name or empty arguments)
            if not name or args == '{}':
                continue
                
            signature = (name, args)
            if signature not in seen:
                seen.add(signature)
                unique_tool_calls.append(tc)
        
        return unique_tool_calls

    def _parse_multiline_tool_calls(self, text: str) -> List[Dict]:
        """Parse multi-line tool_call format with newlines between tags.
        
        Uses pre-compiled TOOL_PATTERNS['multiline'] and TOOL_PATTERNS['multiline_standalone'].
        """
        tool_calls = []
        
        # Use pre-compiled multiline wrapper pattern
        for match in TOOL_PATTERNS['multiline'].findall(text):
            # Find each <tool> block within the tool_call
            tool_blocks = re.findall(r'<tool>\s*(.*?)\s*</tool>', match, re.DOTALL | re.IGNORECASE)
            
            for tool_block in tool_blocks:
                name_match = re.search(r'<name>\s*(.*?)\s*</name>', tool_block, re.DOTALL | re.IGNORECASE)
                if not name_match:
                    continue
                
                name = name_match.group(1).strip()
                if not name:
                    continue
                
                # Extract arguments - could be JSON or XML-style
                args_match = re.search(r'<arguments>\s*(.*?)\s*</arguments>', tool_block, re.DOTALL | re.IGNORECASE)
                if not args_match:
                    args_match = re.search(r'<parameters>\s*(.*?)\s*</parameters>', tool_block, re.DOTALL | re.IGNORECASE)
                
                if args_match:
                    args = self._parse_args(args_match.group(1))
                else:
                    args = {}
                
                tool_calls.append(self._make_tool_call(name, args))
        
        # Use pre-compiled multiline standalone pattern
        for name, args_str in TOOL_PATTERNS['multiline_standalone'].findall(text):
            name = name.strip()
            if not name:
                continue
            args = self._parse_args(args_str)
            tool_calls.append(self._make_tool_call(name, args))
        
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
        """Convert simple nested XML to dictionary. Uses pre-compiled RE_XML_TO_DICT."""
        result = {}
        matches = RE_XML_TO_DICT.findall(xml_content)
        for tag, content in matches:
            if RE_XML_NESTED.search(content):
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
        
        # Repair gemma-4's corrupted tool tokens (<|"|> quotes, malformed markers).
        text = normalize_gemma_tool_tokens(text)

        # First filter out malformed content
        text = self._filter_malformed_content(text)

        # REPAIR: Fix broken tool call formats that the model hallucinates
        # This handles cases like <tool><tool_name><param>value</param></tool_name></tool>
        text = repair_broken_tool_calls(text)

        # DeepSeek V4 (ds4) native DSML tool calls: <｜DSML｜invoke name="…">… .
        # ds4-server emits these with the ｜ (U+FF5C) special-token bar; without
        # this they were streamed to the client as raw content.
        if 'DSML' in text:
            names = [getattr(getattr(t, 'function', None), 'name', None)
                     for t in (available_tools or [])]
            names = [n for n in names if n]
            dsml = parse_deepseek_dsml_tool_calls(text, names or None)
            if dsml:
                print(f"DEBUG ToolCallParser: DSML parsing found {len(dsml)} tool calls")
                return [{
                    "id": f"call_{uuid.uuid4().hex[:16]}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                } for name, args in dsml]

        # NOTE: the degraded plaintext <tool>name arg: value</tool> form (heavy
        # quants that can't emit DSML, e.g. ds4 q2-imatrix) is handled ONLY in
        # DeepSeekParser — scoped to the DeepSeek family where it occurs — so it can
        # never misread a <tool> example in some other model's prose reply here.

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
        
        # If still no tool calls, try multi-line format parsing
        if not tool_calls:
            multiline_tool_calls = self._parse_multiline_tool_calls(text)
            if multiline_tool_calls:
                print(f"DEBUG ToolCallParser: Multi-line parsing found {len(multiline_tool_calls)} tool calls")
                tool_calls.extend(multiline_tool_calls)
        
        # Debug output for results
        if tool_calls:
            print(f"DEBUG ToolCallParser: Returning {len(tool_calls)} tool calls")
            for i, tc in enumerate(tool_calls):
                print(f"DEBUG ToolCallParser:   [{i}] {tc.get('function', {}).get('name', 'unknown')}: {tc.get('function', {}).get('arguments', {})}")
        else:
            print(f"DEBUG ToolCallParser: No tool calls found, returning None")
        
        return tool_calls if tool_calls else None

    def strip_tool_calls_from_content(self, text: str) -> str:
        """Remove tool call format from text after extracting tool calls.
        Uses pre-compiled STRIP_TOOL_PATTERNS for efficiency."""
        if not text:
            return text

        # Repair gemma-4's corrupted markers/quotes, then drop its native
        # call:/response: spans so neither the call nor a hallucinated fake result
        # leaks into the content.
        text = normalize_gemma_tool_tokens(text)
        while re.search(r'(?:call|response):\s*[A-Za-z_]\w*\s*\{', text):
            m = re.search(r'(?:call|response):\s*[A-Za-z_]\w*\s*\{', text)
            try:
                _, end = _parse_gemma_loose_object(text, m.end() - 1)
            except Exception:
                end = m.end()
            text = text[:m.start()] + text[end:]

        text = strip_dsml_tool_calls(text)
        for pattern in STRIP_TOOL_PATTERNS:
            text = pattern.sub('', text)

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
        # Defer dispatcher creation - it will be created on first use or when model_name is set
        self._dispatcher = None
    
    def _ensure_dispatcher(self) -> ModelParserDispatcher:
        """Ensure dispatcher is created with current model_name."""
        if self._dispatcher is None or (self._model_name and self._dispatcher.model_name != self._model_name):
            self._dispatcher = ModelParserDispatcher(model_name=self._model_name, tools_schema=self._tools_schema)
        return self._dispatcher
    
    def set_model_name(self, model_name: str) -> None:
        """Set the model name for model-specific parsing."""
        self._model_name = model_name
        # Force dispatcher recreation on next use
        self._dispatcher = ModelParserDispatcher(model_name=model_name, tools_schema=self._tools_schema)
    
    def extract_tool_calls(self, text: str, available_tools: List) -> Optional[List[Dict]]:
        """Extract tool calls from model output using model-specific parsing."""
        if not text:
            return None

        # Repair gemma-4's corrupted tool tokens (<|"|> quotes, malformed markers) so
        # the native call:NAME{…} args parse cleanly instead of carrying <|"|> garbage
        # into e.g. bash commands.
        text = normalize_gemma_tool_tokens(text)

        # REPAIR: Fix broken tool call formats that the model hallucinates
        # This handles cases like <tool><tool_name><param>value</param></tool_name></tool>
        text = repair_broken_tool_calls(text)
        
        tools_dict = {}
        for tool in available_tools:
            if hasattr(tool, 'function') and tool.function:
                func = tool.function
                tools_dict[func.name] = {
                    'description': func.description or '',
                    'parameters': func.parameters or {}
                }
        
        # Ensure dispatcher is created and update tools if needed
        dispatcher = self._ensure_dispatcher()
        if tools_dict != self._tools_schema:
            self._tools_schema = tools_dict
            dispatcher.set_tools(tools_dict)
        
        tool_calls = dispatcher.parse(text)
        
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

        # Repair gemma-4's corrupted markers/quotes first so the call:NAME{…} spans
        # below match and the stray <|tool_call>/<|tool_response>/<|"|> fragments are
        # gone from the content shown to the user.
        text = normalize_gemma_tool_tokens(text)

        text = strip_dsml_tool_calls(text)

        # gemma-4 native: drop every `call:NAME{…}` (the real call) and any
        # `response:NAME{…}` (a hallucinated fake tool result) span — balanced braces —
        # plus the `thought` channel residue.
        while re.search(r'(?:call|response):\s*[A-Za-z_]\w*\s*\{', text):
            m = re.search(r'(?:call|response):\s*[A-Za-z_]\w*\s*\{', text)
            try:
                _, end = _parse_gemma_loose_object(text, m.end() - 1)
            except Exception:
                end = m.end()
            text = text[:m.start()] + text[end:]
        text = re.sub(r'(?m)^\s*thought\s*$\n?', '', text)

        # Custom XML format: <tool><action>...</action><object>...</object><properties>...</properties></tool>
        text = re.sub(r'<tool>\s*<action>.*?</action>\s*<object>.*?</object>\s*<properties>.*?</properties>\s*</tool>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<tool=[^>]+>.*?</tool_call>', '', text, flags=re.DOTALL)
        text = re.sub(r'<tool=[^>]+>.*?</tool>', '', text, flags=re.DOTALL)
        text = re.sub(r'<tool>.*?</tool>', '', text, flags=re.DOTALL)
        text = re.sub(r'<function>.*?</function>', '', text, flags=re.DOTALL)
        
        # NEW: Remove nested tool format where tool name is the tag
        # Pattern: <tool_call><tool><toolname>...</toolname></tool></tool_call>
        text = re.sub(r'<tool_call>\s*<tool>\s*<\w+>.*?</\w+>\s*</tool>\s*</tool_call>', '', text, flags=re.DOTALL | re.IGNORECASE)
        # Pattern: <tool><toolname>...</toolname></tool>
        text = re.sub(r'<tool>\s*<\w+>.*?</\w+>\s*</tool>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text