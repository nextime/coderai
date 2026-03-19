"""
Text generation endpoints for the codai API.
"""

import asyncio
import json
import time
import uuid
from typing import AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request

# Import from codai modules
from codai.models.manager import ModelManager, WhisperServerManager, MultiModelManager, model_manager, multi_model_manager
from codai.queue.manager import QueueManager, queue_manager
from codai.pydantic.textrequest import ChatCompletionRequest, ToolFunction, Tool
from codai.models.parser import filter_malformed_content, filter_repetition, OpenAIFormatter, ModelParserAdapter, ToolCallParser

# Import global state from state module
from codai.api.state import (
    set_global_args as _set_global_args,
    get_global_debug,
    set_global_debug as _set_global_debug,
    set_global_system_prompt as _set_global_system_prompt,
    set_global_tools_closer_prompt as _set_global_tools_closer_prompt,
    get_grammar_guided_gen,
    set_grammar_guided_gen as _set_grammar_guided_gen,
)

# Global reference to be set by coderai
global_args = None


# =============================================================================
# Helper Functions
# =============================================================================

def set_global_args(args):
    """Set global args from coderai."""
    global global_args
    global_args = args


def set_global_debug(debug: bool):
    """Set the global debug flag (via state module)."""
    _set_global_debug(debug)


def set_global_system_prompt(prompt):
    """Set the global system prompt (via state module)."""
    _set_global_system_prompt(prompt)


def set_global_tools_closer_prompt(tools_closer: bool):
    """Set the global tools-closer-prompt flag (via state module)."""
    _set_global_tools_closer_prompt(tools_closer)


def set_grammar_guided_gen(enabled: bool):
    """Set the grammar-guided generation flag (via state module)."""
    _set_grammar_guided_gen(enabled)


# =============================================================================
# Router and Endpoints
# =============================================================================

router = APIRouter()


@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, http_request: Request = None):
    """Chat completions endpoint with streaming and tool support."""
    
    # Check if we should use litellm backend
    parser_type = getattr(global_args, 'parser', 'auto') if global_args else 'auto'
    
    if parser_type == 'litellm':
        # Use LiteLLM backend
        from codai.openai.litellm import get_litellm_backend, LITELLM_AVAILABLE
        
        if not LITELLM_AVAILABLE:
            raise HTTPException(
                status_code=500,
                detail="LiteLLM is not installed. Run: pip install litellm"
            )
        
        # Check for API key in request - litellm requires an API key
        # If not provided, use a fake key to allow the request to proceed
        api_key = None
        
        # Try to get API key from request body
        if hasattr(request, 'api_key') and request.api_key:
            api_key = request.api_key
        
        # If no API key in body, try to get from Authorization header
        if not api_key:
            auth_header = http_request.headers.get('Authorization', '') if http_request else ''
            if auth_header.startswith('Bearer '):
                api_key = auth_header[7:]  # Extract token after 'Bearer '
        
        # If still no API key, use a fake key to allow litellm to proceed
        # litellm will then fail with the actual provider error if needed
        if not api_key:
            api_key = "fake-key-for-local-testing"
            print("DEBUG: No API key provided, using fake key for litellm")
        
        # Determine the base URL for litellm to connect to
        # Use the server's host and port for local connections
        api_base = None
        
        # Check if model starts with 'ollama:' - use local Ollama
        if request.model and request.model.startswith('ollama:'):
            # Get the host from the request headers
            client_host = "127.0.0.1"
            if http_request:
                host_header = http_request.headers.get('host', '')
                if host_header:
                    # Strip port if present
                    if ':' in host_header:
                        client_host = host_header.split(':')[0]
                        if client_host.replace('.', '').isdigit():
                            # It's an IP, keep it
                            pass
                        else:
                            # It's a hostname, use localhost
                            client_host = "127.0.0.1"
                    else:
                        client_host = host_header
            
            # Get port from global_args or use default
            port = getattr(global_args, 'port', 11434) if global_args else 11434
            api_base = f"http://{client_host}:{port}/v1"
            print(f"DEBUG: Using api_base for Ollama: {api_base}")
        else:
            # For non-Ollama models, use the server's own URL as base
            # This allows LiteLLM to make requests to the local server
            if http_request:
                # Get the host from the request headers
                host_header = http_request.headers.get('host', '')
                if host_header:
                    # Strip port if present to reconstruct clean URL
                    if ':' in host_header:
                        client_host = host_header.split(':')[0]
                        # Keep the port from the request for consistency
                        server_port = host_header.split(':')[1] if len(host_header.split(':')) > 1 else str(getattr(global_args, 'port', 6745))
                    else:
                        client_host = host_header
                        server_port = str(getattr(global_args, 'port', 6745))
                else:
                    # Fallback to client host if no Host header
                    client_host = http_request.client.host if http_request.client else "127.0.0.1"
                    server_port = str(getattr(global_args, 'port', 6745))
            else:
                # Fallback if no http_request
                client_host = "127.0.0.1"
                server_port = str(getattr(global_args, 'port', 6745))
            
            # Determine protocol (http or https)
            use_https = getattr(global_args, 'https', False) or getattr(global_args, 'pubkey', None)
            protocol = "https" if use_https else "http"
            api_base = f"{protocol}://{client_host}:{server_port}/v1"
            print(f"DEBUG: Using api_base for local server: {api_base}")
        
        # Get or create litellm backend
        litellm_backend = get_litellm_backend(
            model=request.model,
            api_key=api_key,
            api_base=api_base,
            context_window=8192,  # Default, can be made configurable
            model_manager=multi_model_manager  # Pass for alias resolution
        )
        
        # Get the tool_parser from multi_model_manager for model-specific parsing
        tool_parser = multi_model_manager.tool_parser if hasattr(multi_model_manager, 'tool_parser') else None
        
        # Convert messages to dict format
        messages_dict = []
        for msg in request.messages:
            msg_dict = {"role": msg.role, "content": msg.content or ""}
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                msg_dict["tool_calls"] = msg.tool_calls
            if hasattr(msg, 'tool_call_id') and msg.tool_call_id:
                msg_dict["tool_call_id"] = msg.tool_call_id
            messages_dict.append(msg_dict)
        
        # Prepare tools if provided
        tools_dict = None
        if request.tools:
            tools_dict = request.tools
        
        # Generate response
        try:
            if request.stream:
                # Streaming response
                
                async def generate():
                    try:
                        async for chunk in await litellm_backend.chat_completion(
                            messages=messages_dict,
                            model=request.model,
                            temperature=request.temperature,
                            top_p=request.top_p,
                            max_tokens=request.max_tokens,
                            stop=request.stop,
                            tools=tools_dict,
                            tool_choice=request.tool_choice,
                            stream=True,
                            tool_parser=tool_parser,
                        ):
                            # Add rate limit headers
                            headers = {}
                            if 'usage' in chunk:
                                headers = litellm_backend.get_rate_limit_headers(
                                    prompt_tokens=chunk.get('usage', {}).get('prompt_tokens', 0),
                                    completion_tokens=chunk.get('usage', {}).get('completion_tokens', 0)
                                )
                            
                            # Handle Qwen tool calls if model is Qwen family
                            if 'qwen' in request.model.lower():
                                content = chunk.get('choices', [{}])[0].get('delta', {}).get('content', '')
                                tool_calls = chunk.get('choices', [{}])[0].get('delta', {}).get('tool_calls', [])
                                
                                if not tool_calls and content:
                                    # Try to parse tool calls from content
                                    tool_calls = litellm_backend.parse_qwen_tool_calls(content)
                                    if tool_calls:
                                        # Strip tool tags from content
                                        content = litellm_backend.strip_tool_tags(content)
                                        chunk['choices'][0]['delta']['content'] = content
                                        chunk['choices'][0]['delta']['tool_calls'] = tool_calls
                            
                            yield f"data: {json.dumps(chunk)}\n\n"
                        
                        yield "data: [DONE]\n\n"
                    except Exception as e:
                        yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'internal_error'}})}\n\n"
                
                from fastapi.responses import StreamingResponse
                return StreamingResponse(generate(), media_type="text/event-stream")
            else:
                # Non-streaming response
                response = await litellm_backend.chat_completion(
                    messages=messages_dict,
                    model=request.model,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_tokens=request.max_tokens,
                    stop=request.stop,
                    tools=tools_dict,
                    tool_choice=request.tool_choice,
                    stream=False,
                    tool_parser=tool_parser,
                )
                
                # Handle Qwen tool calls
                if 'qwen' in request.model.lower() and 'choices' in response:
                    msg = response['choices'][0].get('message', {})
                    content = msg.get('content', '')
                    tool_calls = msg.get('tool_calls', [])
                    
                    if not tool_calls and content:
                        tool_calls = litellm_backend.parse_qwen_tool_calls(content)
                        if tool_calls:
                            msg['content'] = litellm_backend.strip_tool_tags(content)
                            msg['tool_calls'] = tool_calls
                            response['choices'][0]['message'] = msg
                
                # Add rate limit headers
                headers = {}
                if 'usage' in response:
                    headers = litellm_backend.get_rate_limit_headers(
                        prompt_tokens=response.get('usage', {}).get('prompt_tokens', 0),
                        completion_tokens=response.get('usage', {}).get('completion_tokens', 0)
                    )
                
                from fastapi.responses import JSONResponse
                return JSONResponse(content=response, headers=headers)
        
        except Exception as e:
            # Handle litellm errors
            error_response = {
                "error": {
                    "message": str(e),
                    "type": "internal_error",
                    "code": 500
                }
            }
            from fastapi.responses import JSONResponse
            return JSONResponse(content=error_response, status_code=500)
    
    # Continue with original implementation for 'auto' parser
    # Get the model for this request
    requested_model = request.model
    
    # Try to get the appropriate model
    mm = multi_model_manager.get_model_for_request(requested_model)
    
    if mm is None:
        # Model not loaded - try to use default
        if model_manager.backend is not None:
            # Fallback to legacy model_manager
            current_manager = model_manager
        else:
            raise HTTPException(status_code=503, detail="Model not loaded")
    else:
        current_manager = mm
    
    # Inject system prompt if --system-prompt flag was provided
    messages = request.messages
    if global_system_prompt is not None:
        # Get the custom system prompt text
        if global_system_prompt is True:
            # Default system prompt
            system_addon = "You are a helpful assistant."
        else:
            # Custom system prompt provided as argument
            system_addon = str(global_system_prompt)
        
        # Check if there's already a system message
        system_found = False
        for i, msg in enumerate(messages):
            if msg.role == "system":
                # Chain the custom system prompt at the START of existing system message
                from codai.pydantic.textrequest import ChatMessage
                messages[i] = ChatMessage(role="system", content=system_addon + "\n\n" + msg.content)
                system_found = True
                break
        
        if not system_found:
            # No existing system message, use the custom one
            from codai.pydantic.textrequest import ChatMessage
            messages = [ChatMessage(role="system", content=system_addon)] + list(messages)
    
    # Enable thinking/reasoning mode if requested via API parameter OR CLI flag
    force_reasoning_args = getattr(global_args, 'force_reasoning', None) if global_args else None
    
    enable_thinking_api = getattr(request, 'enable_thinking', False)
    
    # Parse force_reasoning: can be list (from CLI) or string (legacy)
    if isinstance(force_reasoning_args, str):
        # Legacy: convert string to list
        if force_reasoning_args == "both":
            force_reasoning_args = ["inject", "stop"]
        elif force_reasoning_args == "stop":
            force_reasoning_args = ["stop"]
        elif force_reasoning_args == "inject":
            force_reasoning_args = ["inject"]
        elif force_reasoning_args == "all":
            # 'all' enables all reasoning methods
            force_reasoning_args = ["chat", "inject", "prompt", "mock", "raw", "twopass"]
        else:
            force_reasoning_args = []
    elif not force_reasoning_args:
        force_reasoning_args = []
    
    # Combine CLI args with API param
    # 'chat' from CLI enables API reasoning param
    reasoning_enabled = enable_thinking_api or (len(force_reasoning_args) > 0)
    
    # DEBUG: Print force_reasoning status when debug mode is enabled
    if get_global_debug():
        print(f"\n{'='*60}")
        print(f"=== REASONING MODE DEBUG ===")
        print(f"{'='*60}")
        print(f"force_reasoning CLI args: {force_reasoning_args}")
        print(f"enable_thinking API param: {enable_thinking_api}")
        # Debug stop sequences if available
        if 'raw_stop_sequences' in locals():
            print(f"stop argument for chat call: {raw_stop_sequences}")
    
    # Get model family for reasoning tokens
    from codai.models.utils import get_model_family, get_reasoning_stop_tokens, get_resolved_model_name
    model_family = get_model_family(request.model)
    
    # Check if model is qwen3 and force_reasoning is enabled
    is_qwen3 = 'qwen3' in model_family.lower() if model_family else False
    use_qwen3_penalties = is_qwen3 and force_reasoning_args
    
    # System prompt addon for qwen3 with force_reasoning
    qwen3_system_addon = ""
    if use_qwen3_penalties:
        qwen3_system_addon = "\n\nCRITICAL: Do not repeat tool calls. If a tool fails with an [ERROR], do not retry the exact same parameters. Propose a different approach or ask for clarification."
        if get_global_debug():
            print(f"QWEEN3: Adding penalties and system addon for qwen3 with force_reasoning")
    
    # Handle 'chat' - enable thinking API parameter
    # Note: This only works with compatible APIs (OpenAI-like)
    # We'll set it on the request if supported
    if "chat" in force_reasoning_args or enable_thinking_api:
        if hasattr(request, 'thinking'):
            request.thinking = {"type": "enabled"}
        if get_global_debug():
            print(f"CHAT: Reasoning API param enabled")
    
    # Handle 'inject' - system prompt injection
    # Skip for 'raw' mode since it handles everything separately
    if "raw" not in force_reasoning_args and "inject" in force_reasoning_args:
        from codai.models.templates import AgenticTemplateManager
        template_manager = AgenticTemplateManager(request.model)
        
        # Use reasoning tag (]]) when prompt is also selected for consistency
        use_reasoning_tag = "prompt" in force_reasoning_args
        
        # Get the current system prompt if exists
        system_content = None
        for msg in messages:
            if msg.role == "system":
                system_content = msg.content
                break
        if system_content:
            # Inject agentic instructions
            system_content = template_manager.get_agent_system_prompt(system_content, use_reasoning_tag=use_reasoning_tag)
        else:
            system_content = template_manager.get_agent_system_prompt("You are a helpful assistant.", use_reasoning_tag=use_reasoning_tag)
        # Update or add system message
        from codai.pydantic.textrequest import ChatMessage
        system_found = False
        for i, msg in enumerate(messages):
            if msg.role == "system":
                messages[i] = ChatMessage(role="system", content=system_content)
                system_found = True
                break
        if not system_found:
            messages = [ChatMessage(role="system", content=system_content)] + list(messages)
        
        if get_global_debug():
            print(f"INJECT: System prompt injected with agentic instructions")
            print(f"\n--- INJECTED SYSTEM PROMPT ---")
            print(system_content)
            print(f"--- END SYSTEM PROMPT ---")
    
    # Handle 'prompt' - prompt seeding (ends with thought tag)
    # Note: 'prompt' and 'raw' are mutually exclusive - raw bypasses this
    if "prompt" in force_reasoning_args and "raw" not in force_reasoning_args:
        from codai.models.templates import AgenticTemplateManager
        template_manager = AgenticTemplateManager(request.model)
        
        # Convert messages to the format expected by force_reasoning_prompt
        user_message = ""
        system_prompt = "You are a helpful assistant."
        
        # Extract system and user messages
        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            elif msg.role == "user":
                user_message = msg.content
        
        # Add qwen3 system addon if applicable
        if qwen3_system_addon:
            system_prompt = system_prompt + qwen3_system_addon
        
        # Get the seeded prompt (ends with thought tag)
        seeded_prompt = template_manager.force_reasoning_prompt(system_prompt, user_message)
        
        # Replace messages with the seeded prompt (as a single user message for raw completion)
        from codai.pydantic.textrequest import ChatMessage
        messages = [ChatMessage(role="user", content=seeded_prompt)]
        
        if get_global_debug():
            print(f"PROMPT: Prompt seeding applied (ends with thought tag)")
            print(f"\n--- SEEDED PROMPT (last 80 chars) ---")
            print(f"...{seeded_prompt[-80:]}")
            print(f"--- END SEEDED PROMPT ---")
    
    # Handle 'raw' - use template_manager.format_for_raw_completion for raw completion
    # This bypasses the chat API and uses the model's native template with reasoning seed
    # The template_manager.format_for_raw_completion will be called in the block below
    
    # Prepare stop sequences
    stop_sequences = []
    if request.stop:
        if isinstance(request.stop, str):
            stop_sequences = [request.stop]
        else:
            stop_sequences = list(request.stop)
    
    # Handle 'stop' - add reasoning stop tokens (also done for 'inject' and 'prompt')
    # Skip for 'raw' mode since it handles stop tokens separately
    if "raw" not in force_reasoning_args and ("stop" in force_reasoning_args or "inject" in force_reasoning_args or "prompt" in force_reasoning_args):
        _, _, additional_stops = get_reasoning_stop_tokens(model_family)
        for stop_token in additional_stops:
            if stop_token not in stop_sequences:
                stop_sequences.append(stop_token)
        
        # When using prompt seeding, also add ]]> to force stopping after reasoning
        if "prompt" in force_reasoning_args:
            # Add common reasoning end tags based on model family
            if "</think>" not in stop_sequences:
                stop_sequences.append("</think>\n")
        
        if get_global_debug():
            print(f"STOP: Added reasoning stop tokens: {additional_stops}")
    
    # Format messages with tools if provided - BUT SKIP for raw mode
    # (raw mode handles tools separately via format_for_raw_completion)
    # Get tools_closer_prompt from global args
    tools_closer = getattr(global_args, 'tools_closer_prompt', False) if global_args else False
    if request.tools and "raw" not in force_reasoning_args:
        messages = format_tools_for_prompt(request.tools, messages, tools_closer_prompt=tools_closer)
    
    # Get the tool_parser from the current manager
    tool_parser = current_manager.tool_parser if hasattr(current_manager, 'tool_parser') else ModelParserAdapter()
    
    # Convert messages to dict format for chat completion
    messages_dict = []
    for msg in messages:
        msg_dict = {"role": msg.role}
        # Always include content key - llama_cpp template expects it
        # Convert content to string if it's a list (multipart content)
        content = msg.content
        if content is None:
            content = ""
        elif isinstance(content, list):
            # Handle multipart content array format: [{"type": "text", "text": "..."}]
            parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get('type') == 'text' and 'text' in item:
                        parts.append(item['text'])
                    else:
                        parts.append(f"[{item.get('type', 'unknown')} content]")
                else:
                    parts.append(str(item))
            content = '\n'.join(parts)
        # Ensure content is never None - convert to string
        msg_dict["content"] = str(content) if content is not None else ""
        # Handle tool_calls - convert to proper format if present
        if msg.tool_calls:
            # tool_calls should be a list of dicts with 'id', 'type', 'function' keys
            msg_dict["tool_calls"] = msg.tool_calls
        if msg.name:
            msg_dict["name"] = msg.name
        if msg.tool_call_id:
            msg_dict["tool_call_id"] = msg.tool_call_id
        messages_dict.append(msg_dict)
    
    # Final safety check: ensure NO message has None content before passing to llama_cpp
    # Also ensure content key always exists (not just None check)
    for i, m in enumerate(messages_dict):
        # Handle missing content key entirely
        if "content" not in m:
            messages_dict[i]["content"] = ""
        # Handle None content
        elif m.get("content") is None:
            messages_dict[i]["content"] = ""
        # Handle content that's not a string (shouldn't happen but be safe)
        elif not isinstance(m["content"], str):
            messages_dict[i]["content"] = str(m["content"])
    
    # Debug: print first few messages to see their structure
    print(f"DEBUG: messages_dict[0] keys: {list(messages_dict[0].keys()) if messages_dict else 'empty'}")
    if len(messages_dict) > 1:
        print(f"DEBUG: messages_dict[1] keys: {list(messages_dict[1].keys()) if len(messages_dict) > 1 else 'empty'}")
    
    # Convert tools to dict format if present
    tools_dict = None
    if request.tools:
        tools_dict = []
        for tool in request.tools:
            tools_dict.append({
                "type": tool.type,
                "function": {
                    "name": tool.function.name,
                    "description": tool.function.description,
                    "parameters": tool.function.parameters
                }
            })
    
    # Handle raw mode - use generate() instead of generate_chat() for raw prompt completion
    # Note: These may have been set earlier in the prompt handling section
    # Initialize only if not already set
    if 'use_raw_mode' not in locals():
        use_raw_mode = False
    if 'raw_prompt_for_generation' not in locals():
        raw_prompt_for_generation = None
    if 'raw_stop_sequences' not in locals():
        raw_stop_sequences = None
    
    # Check if we need to set up raw mode (if not already done in prompt handling)
    if "raw" in force_reasoning_args and not use_raw_mode:
        # Create template_manager if not already created
        if 'template_manager' not in locals():
            from codai.models.templates import AgenticTemplateManager
            template_manager = AgenticTemplateManager(request.model)
        
        # Use template_manager.format_for_raw_completion which handles everything
        if hasattr(template_manager, 'format_for_raw_completion'):
            # Extract system and user messages
            system_prompt = "You are a helpful assistant."
            user_message = ""
            for msg in messages:
                if msg.role == "system":
                    system_prompt = msg.content
                elif msg.role == "user":
                    user_message = msg.content
            
            raw_prompt_for_generation, raw_stop_sequences = template_manager.format_for_raw_completion(
                system_prompt=system_prompt,
                user_message=user_message,
                inject_system=True,
                force_reasoning=True,
                tools=request.tools,  # Pass tools for family-specific formatting
                tools_closer_prompt=tools_closer  # Pass tools-closer-prompt flag
            )
            use_raw_mode = True
            
            if get_global_debug():
                print(f"RAW: Using template_manager.format_for_raw_completion")
                print(f"RAW: Prompt ends with: ...{raw_prompt_for_generation[-80:]}")
        else:
            if get_global_debug():
                print(f"RAW: template_manager.format_for_raw_completion not available")
    
    # Get resolved model name for response (with coderai/ prefix and proper formatting)
    # Use multi_model_manager to get the actual loaded models, not the individual model manager
    response_model_name = get_resolved_model_name(requested_model, multi_model_manager)
    print(f"DEBUG: Requested model: {requested_model}, Resolved model for response: {response_model_name}")
    
    # Handle raw mode - two pass: first capture reasoning, then get final answer
    if use_raw_mode and raw_prompt_for_generation:
        if get_global_debug():
            print(f"RAW: Starting two-pass generation")
            print(f"RAW: First pass prompt: ...{raw_prompt_for_generation[-100:]}")
        
        # Build extra params for qwen3
        extra_params = {}
        if use_qwen3_penalties:
            extra_params = {
                'repeat_penalty': 1.15,
                'presence_penalty': 1.5,
                'frequency_penalty': 0.5,
            }
        
        if request.stream:
            # For streaming, we need to handle it differently
            # First pass: generate until reasoning close tag (stream it)
            async def raw_stream_generate():
                import json  # Local import for nested function
                thought_tag, close_tag, _ = get_reasoning_stop_tokens(model_family)
                reasoning_text = ""
                
                if get_global_debug():
                    print(f"DEBUG: raw_stream_generate started, stream=True")
                
                # Use the backend's async generate if available
                if hasattr(current_manager.backend, 'generate_stream'):
                    async for chunk in current_manager.backend.generate_stream(
                        prompt=raw_prompt_for_generation,
                        max_tokens=request.max_tokens or 2048,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        stop=raw_stop_sequences,
                        **extra_params,
                    ):
                        reasoning_text += chunk
                        
                        # Debug: log first pass chunks
                        if get_global_debug():
                            print(f"DEBUG FIRST PASS: chunk length={len(chunk)}, total reasoning so far={len(reasoning_text)}")
                        
                        yield f"data: {json.dumps({'choices': [{'delta': {'content': chunk}, 'finish_reason': None}]})}\n\n"
                        
                        # Check if we hit the close tag
                        if close_tag and close_tag in reasoning_text:
                            if get_global_debug():
                                print(f"DEBUG: Close tag detected in first pass, reasoning length={len(reasoning_text)}")
                            break
                else:
                    # Fallback: non-streaming
                    if get_global_debug():
                        print(f"DEBUG: Using non-streaming fallback for first pass")
                    first_pass_result = current_manager.generate(
                        prompt=raw_prompt_for_generation,
                        max_tokens=request.max_tokens or 2048,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        stop=raw_stop_sequences,
                        **extra_params,
                    )
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': first_pass_result}, 'finish_reason': None}]})}\n\n"
                
                # After reasoning, yield the close tag and continue with final answer
                if close_tag:
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': close_tag}, 'finish_reason': None}]})}\n\n"
                
                # Second pass: get the rest
                full_prompt = raw_prompt_for_generation + reasoning_text + (close_tag or "")
                
                if get_global_debug():
                    print(f"DEBUG: raw_stream_generate second pass, full_prompt length: {len(full_prompt)}")
                
                second_pass_result = current_manager.generate(
                    prompt=full_prompt,
                    max_tokens=request.max_tokens or 2048,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stop=stop_sequences,
                    **extra_params,
                )
                
                # FIX: Apply repetition filtering to both reasoning and final text
                reasoning_text = filter_repetition(reasoning_text)
                second_pass_result = filter_repetition(second_pass_result)
                
                # FIX: If reasoning contains tool call tags, split at the first tool tag
                # The tool call part should NOT be in reasoning - it should be left for tool extraction
                tool_tag_patterns = ["<tool_call>", "<tool>", "<|tool_call|", "<function="]
                earliest_tool_idx = len(reasoning_text)
                earliest_tool_tag = None
                for tag in tool_tag_patterns:
                    idx = reasoning_text.find(tag)
                    if idx != -1 and idx < earliest_tool_idx:
                        earliest_tool_idx = idx
                        earliest_tool_tag = tag
                
                if earliest_tool_tag:
                    # Split: everything before the tool tag is reasoning, everything from the tag onwards goes to second_pass_result
                    tool_part = reasoning_text[earliest_tool_idx:]
                    reasoning_text = reasoning_text[:earliest_tool_idx].strip()
                    # Prepend the tool part to second_pass_result so it can be extracted as a tool call
                    second_pass_result = tool_part + second_pass_result
                    if get_global_debug():
                        print(f"DEBUG: Moved tool call from reasoning to second_pass_result: {tool_part[:100]}...")
                
                # In debug mode, dump the full generated text (second pass result)
                if get_global_debug():
                    print(f"\n{'='*80}")
                    print(f"=== RAW STREAM: FULL GENERATED TEXT (DEBUG) ===")
                    print(f"{'='*80}")
                    print(f"--- SECOND PASS RESULT ---")
                    print(second_pass_result)
                    print(f"--- END SECOND PASS RESULT ---")
                    print(f"{'='*80}\n")
                    
                    # Also dump the reasoning text from first pass
                    print(f"\n{'='*80}")
                    print(f"=== RAW STREAM: REASONING TEXT (DEBUG) ===")
                    print(f"{'='*80}")
                    print(reasoning_text)
                    print(f"{'='*80}\n")
                
                # Try to extract tool calls from the second pass result ONLY
                # FIX: Do NOT fall back to reasoning text - tool calls should only come from final response
                extracted_tool_calls = None
                text_for_tool_extraction = second_pass_result
                
                # CRITICAL: Only extract from second pass, never from reasoning
                # Reasoning may contain partial/incomplete tool calls that confuse the parser
                if get_global_debug():
                    print(f"DEBUG: Tool extraction - using second_pass_result only")
                    print(f"DEBUG: Second pass result length: {len(second_pass_result) if second_pass_result else 0}")
                    print(f"DEBUG: Reasoning text length: {len(reasoning_text) if reasoning_text else 0}")
                
                if request.tools and text_for_tool_extraction:
                    # Convert tools for ModelParserAdapter
                    from codai.pydantic.textrequest import Tool, ToolFunction
                    from codai.models.parser import ModelParserAdapter
                    
                    tools_list = []
                    for t in request.tools:
                        try:
                            if isinstance(t, dict):
                                func_data = t.get("function", {})
                                tool_func = ToolFunction(
                                    name=func_data.get("name", ""),
                                    description=func_data.get("description"),
                                    parameters=func_data.get("parameters")
                                )
                            else:
                                tool_func = ToolFunction(
                                    name=t.function.name if hasattr(t.function, 'name') else str(t.function),
                                    description=t.function.description if hasattr(t.function, 'description') else None,
                                    parameters=t.function.parameters if hasattr(t.function, 'parameters') else None
                                )
                            tools_list.append(Tool(type=t.get("type", "function") if isinstance(t, dict) else t.type, function=tool_func))
                        except Exception as e:
                            print(f"DEBUG: Error converting tool in raw stream: {e}")
                            continue
                    
                    if tools_list:
                        adapter = ModelParserAdapter(model_name=response_model_name)
                        extracted_tool_calls = adapter.extract_tool_calls(text_for_tool_extraction, tools_list)
                        
                        # FIX: Validate extracted tool calls have valid JSON
                        if extracted_tool_calls:
                            from codai.models.parser import validate_json_complete
                            validated_calls = []
                            for tc in extracted_tool_calls:
                                args = tc.get('function', {}).get('arguments', '{}')
                                if isinstance(args, str) and validate_json_complete(args):
                                    validated_calls.append(tc)
                                elif isinstance(args, dict):
                                    # Dict is already valid
                                    validated_calls.append(tc)
                            
                            if len(validated_calls) != len(extracted_tool_calls):
                                if get_global_debug():
                                    print(f"DEBUG: Filtered out {len(extracted_tool_calls) - len(validated_calls)} invalid tool calls")
                            extracted_tool_calls = validated_calls if validated_calls else None
                        
                        if global_debug and extracted_tool_calls:
                            print(f"\n{'='*80}")
                            print(f"=== RAW STREAM: EXTRACTED TOOL CALLS (DEBUG) ===")
                            print(f"{'='*80}")
                            print(json.dumps(extracted_tool_calls, indent=2))
                            print(f"{'='*80}\n")
                        elif get_global_debug():
                            print(f"DEBUG: No tool calls found in raw stream")
                
                if extracted_tool_calls:
                    # Yield tool calls instead of content
                    yield f"data: {json.dumps({'choices': [{'delta': {'tool_calls': extracted_tool_calls}, 'finish_reason': 'tool_calls'}]})}\n\n"
                else:
                    # No tool calls, yield the content as usual
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': second_pass_result}, 'finish_reason': 'stop'}]})}\n\n"
                yield "data: [DONE]\n\n"
            
            from fastapi.responses import StreamingResponse
            return StreamingResponse(raw_stream_generate(), media_type="text/event-stream")
        
        # Non-streaming path (already implemented above)
        # First pass: generate until reasoning close tag
        first_pass_result = current_manager.generate(
            prompt=raw_prompt_for_generation,
            max_tokens=request.max_tokens or 2048,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=raw_stop_sequences,
            **extra_params,
        )
        
        if get_global_debug():
            print(f"RAW: First pass result: ...{first_pass_result[-200:]}")
        
        # Dump first pass result if --dump is enabled
        global_dump = getattr(global_args, 'dump', False) if global_args else False
        if global_dump:
            print(f"\n{'='*80}")
            print(f"=== RAW MODE: FIRST PASS RESULT (DUMP) ===")
            print(f"{'='*80}")
            print(first_pass_result)
            print(f"{'='*80}\n")
        
        # Extract reasoning (everything up to the close tag)
        thought_tag, close_tag, _ = get_reasoning_stop_tokens(model_family)
        reasoning_text = ""
        final_text = first_pass_result
        
        # Define tool tags that indicate end of reasoning
        tool_tags = ["<tool_call>", "<tool>", "<|tool_call|>", "<|tool|>", "<function="]
        
        if close_tag and close_tag in first_pass_result:
            # Split at close tag
            parts = first_pass_result.split(close_tag, 1)
            reasoning_text = parts[0]
            final_text = parts[1] if len(parts) > 1 else ""
        else:
            # Try to find tool tags as fallback stop markers
            earliest_tool_idx = len(first_pass_result)
            earliest_tool_tag = None
            for tag in tool_tags:
                idx = first_pass_result.find(tag)
                if idx != -1 and idx < earliest_tool_idx:
                    earliest_tool_idx = idx
                    earliest_tool_tag = tag
            
            if earliest_tool_tag:
                # Split at tool tag
                if get_global_debug():
                    print(f"RAW: No close tag found, using tool tag '{earliest_tool_tag}' as fallback")
                parts = first_pass_result.split(earliest_tool_tag, 1)
                reasoning_text = parts[0]
                final_text = earliest_tool_tag + (parts[1] if len(parts) > 1 else "")
        
        if get_global_debug():
            print(f"RAW: Extracted reasoning: {reasoning_text[:100]}...")
            print(f"RAW: Final text before cleanup: {final_text[:100]}...")
        
        # Dump extraction details if --dump is enabled
        if global_dump:
            print(f"\n{'='*80}")
            print(f"=== RAW MODE: EXTRACTION (DUMP) ===")
            print(f"{'='*80}")
            print(f"Close tag used: {close_tag}")
            print(f"\n--- REASONING TEXT ---")
            print(reasoning_text)
            print(f"\n--- FINAL TEXT (before cleanup) ---")
            print(final_text)
            print(f"{'='*80}\n")
        
        # Clean up control tokens from final text
        final_text = cleanup_control_tokens(final_text)
        
        # FIX: Apply repetition filtering to reasoning and final text
        reasoning_text = filter_repetition(reasoning_text)
        final_text = filter_repetition(final_text)
        
        # FIX: If reasoning contains tool call tags, split at the first tool tag
        # The tool call part should NOT be in reasoning - it should be left for tool extraction in final_text
        tool_tag_patterns = ["<tool_call>", "<tool>", "<|tool_call|>", "<function="]
        earliest_tool_idx = len(reasoning_text)
        earliest_tool_tag = None
        for tag in tool_tag_patterns:
            idx = reasoning_text.find(tag)
            if idx != -1 and idx < earliest_tool_idx:
                earliest_tool_idx = idx
                earliest_tool_tag = tag
        
        if earliest_tool_tag:
            # Split: everything before the tool tag is reasoning, everything from the tag onwards goes to final_text
            tool_part = reasoning_text[earliest_tool_idx:]
            reasoning_text = reasoning_text[:earliest_tool_idx].strip()
            # Prepend the tool part to final_text so it can be extracted as a tool call
            final_text = tool_part + final_text
            if get_global_debug():
                print(f"RAW: Moved tool call from reasoning to final_text: {tool_part[:100]}...")
        
        if get_global_debug():
            print(f"RAW: Final text after cleanup: {final_text[:100]}...")
        
        # If we have reasoning, continue with second pass to get more complete answer
        # Build the full prompt with reasoning included
        full_prompt = raw_prompt_for_generation + reasoning_text + (close_tag or "")
        
        # Second pass: generate the rest (or just use what we have)
        # For now, just return what we have + optionally continue
        if final_text.strip():
            # We have a complete answer after reasoning
            generated_text = reasoning_text + (close_tag or "") + final_text
        else:
            # Need second pass to get answer
            second_pass_result = current_manager.generate(
                prompt=full_prompt,
                max_tokens=request.max_tokens or 2048,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=stop_sequences,
                **extra_params,
            )
            # Clean up the second pass result
            second_pass_result = cleanup_control_tokens(second_pass_result)
            generated_text = reasoning_text + (close_tag or "") + second_pass_result
        
        # Additional cleanup of the full generated text
        generated_text = cleanup_control_tokens(generated_text)
        
        if get_global_debug():
            print(f"RAW: Generated text after cleanup: {generated_text[:100]}...")
        
        # Pass through the formatter/parser (same as regular mode)
        # Pipeline: Model output -> Extract reasoning (if raw mode) -> ModelParserAdapter (extract tools) -> OpenAIFormatter (final format)
        from codai.models.parser import OpenAIFormatter, ModelParserAdapter
        
        # Convert request tools for ModelParserAdapter
        tools_list = None
        if request.tools:
            from codai.pydantic.textrequest import Tool, ToolFunction
            tools_list = []
            for t in request.tools:
                try:
                    # Handle both dict and pydantic model formats
                    if isinstance(t, dict):
                        func_data = t.get("function", {})
                        tool_func = ToolFunction(
                            name=func_data.get("name", ""),
                            description=func_data.get("description"),
                            parameters=func_data.get("parameters")
                        )
                    else:
                        # Pydantic model
                        tool_func = ToolFunction(
                            name=t.function.name if hasattr(t.function, 'name') else str(t.function),
                            description=t.function.description if hasattr(t.function, 'description') else None,
                            parameters=t.function.parameters if hasattr(t.function, 'parameters') else None
                        )
                    tools_list.append(Tool(type=t.get("type", "function") if isinstance(t, dict) else t.type, function=tool_func))
                except Exception as e:
                    print(f"DEBUG: Error converting tool in raw mode: {e}, tool type: {type(t)}")
                    continue
        
        # Step 1: Use ModelParserAdapter to extract tool calls from final_text (NOT generated_text which includes reasoning)
        # This fixes Bug 2 and Bug 3: reasoning was appearing in both content AND reasoning fields
        # because the parser was receiving the full generated_text including reasoning
        extracted_tool_calls = None
        clean_text = final_text  # Use final_text (after reasoning) instead of generated_text (which includes reasoning)
        if tools_list:
            adapter = ModelParserAdapter(model_name=response_model_name)
            # Extract tool calls from final_text only (after reasoning is done)
            extracted_tool_calls = adapter.extract_tool_calls(final_text, tools_list)
            
            # FIX: Validate extracted tool calls have valid JSON
            if extracted_tool_calls:
                from codai.models.parser import validate_json_complete
                validated_calls = []
                for tc in extracted_tool_calls:
                    args = tc.get('function', {}).get('arguments', '{}')
                    if isinstance(args, str) and validate_json_complete(args):
                        validated_calls.append(tc)
                    elif isinstance(args, dict):
                        # Dict is already valid
                        validated_calls.append(tc)
                
                if len(validated_calls) != len(extracted_tool_calls):
                    print(f"DEBUG: Filtered out {len(extracted_tool_calls) - len(validated_calls)} invalid tool calls in non-streaming")
                extracted_tool_calls = validated_calls if validated_calls else None
            
            if extracted_tool_calls:
                # Strip tool calls from the text
                clean_text = adapter.strip_tool_calls_from_content(final_text)
                if get_global_debug():
                    print(f"RAW: Extracted {len(extracted_tool_calls)} tool calls from final_text (after reasoning)")
        
        # Estimate token counts
        prompt_tokens = len(raw_prompt_for_generation.split())
        completion_tokens = len(clean_text.split()) if clean_text else 0
        
        # Step 2: Use OpenAIFormatter for final formatting
        formatter = OpenAIFormatter(response_model_name)
        try:
            formatted_response = formatter.format_full(
                text=clean_text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                tool_calls=extracted_tool_calls
            )
        except Exception as e:
            print(f"RAW: ERROR in formatter.format_full: {e}")
            formatted_response = None
        
        if get_global_debug():
            if formatted_response and isinstance(formatted_response, dict):
                try:
                    choices = formatted_response.get('choices', [])
                    if choices and len(choices) > 0:
                        message = choices[0].get('message', {}) if isinstance(choices[0], dict) else {}
                        content = message.get('content', '') if isinstance(message, dict) else ''
                        print(f"RAW: Passed through formatter, got: {str(content)[:100]}...")
                    else:
                        print(f"RAW: WARNING - formatter returned empty choices!")
                except Exception as e:
                    print(f"RAW: ERROR accessing formatter response: {e}")
            else:
                print(f"RAW: WARNING - formatter returned None or invalid response!")
        
        # Add mock reasoning stats if 'mock' is in force_reasoning_args
        # But only if we DON'T already have real reasoning from extraction
        has_real_reasoning = reasoning_text and len(reasoning_text.strip()) > 10
        
        if force_reasoning_args and "mock" in force_reasoning_args and formatted_response and not has_real_reasoning:
            # Add fake reasoning tokens to trigger VSCode plugin stats
            mock_reasoning_tokens = 50
            
            # Update usage
            if "usage" in formatted_response:
                formatted_response["usage"]["completion_tokens"] += mock_reasoning_tokens
                formatted_response["usage"]["total_tokens"] += mock_reasoning_tokens
                formatted_response["usage"]["completion_tokens_details"] = {
                    "reasoning_tokens": mock_reasoning_tokens
                }
            
            # Add reasoning to message if not present
            if "choices" in formatted_response and formatted_response["choices"]:
                choice = formatted_response["choices"][0]
                if "message" in choice and "reasoning" not in choice["message"]:
                    choice["message"]["reasoning"] = "Processing task in optimized mode..."
        elif has_real_reasoning and formatted_response:
            # We have real reasoning from extraction - add it to the message
            if "choices" in formatted_response and formatted_response["choices"]:
                choice = formatted_response["choices"][0]
                if "message" in choice:
                    choice["message"]["reasoning"] = reasoning_text.strip()
                    # Also update usage with actual reasoning tokens
                    if "usage" in formatted_response:
                        reasoning_tokens = len(reasoning_text.strip().split())
                        formatted_response["usage"]["completion_tokens_details"] = {
                            "reasoning_tokens": reasoning_tokens
                        }
        
        # Dump parsed output if enabled
        if global_dump:
            import json
            print(f"\n{'='*80}")
            print(f"=== RAW MODE PARSED OUTPUT (DUMP) ===")
            print(f"{'='*80}")
            print(json.dumps(formatted_response, indent=2))
            print(f"{'='*80}\n")
        
        # Add rate limit headers
        headers = {}
        if formatted_response and 'usage' in formatted_response:
            headers = current_manager.backend.get_rate_limit_headers(
                prompt_tokens=formatted_response.get('usage', {}).get('prompt_tokens', 0),
                completion_tokens=formatted_response.get('usage', {}).get('completion_tokens', 0)
            ) if hasattr(current_manager.backend, 'get_rate_limit_headers') else {}
        
        # Ensure we have a valid response to return
        if not formatted_response:
            # Create a minimal fallback response
            formatted_response = {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": response_model_name,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": clean_text or ""
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            }
        
        from fastapi.responses import JSONResponse
        return JSONResponse(content=formatted_response, headers=headers)
    
    if request.stream:
        from fastapi.responses import StreamingResponse
        return StreamingResponse(
            stream_chat_response(
                messages_dict,
                response_model_name,
                request.max_tokens,
                request.temperature,
                request.top_p,
                stop_sequences,
                tools_dict,
                current_manager,
                tool_parser,
                request.response_format,
            ),
            media_type="text/event-stream",
        )
    else:
        return await generate_chat_response(
            messages_dict,
            response_model_name,
            request.max_tokens,
            request.temperature,
            request.top_p,
            stop_sequences,
            tools_dict,
            current_manager,
            tool_parser,
            request.response_format,
            force_reasoning_args,
        )

async def stream_chat_response(
    messages: List[Dict],
    model_name: str,
    max_tokens: Optional[int],
    temperature: float,
    top_p: float,
    stop: List[str],
    tools: Optional[List[Dict]],
    current_manager: ModelManager,
    tool_parser: ToolCallParser,
    response_format: Optional[Dict] = None,
) -> AsyncGenerator[str, None]:
    """Stream chat completion response with queue notifications."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    request_id = f"req-{uuid.uuid4().hex[:8]}"
    
    generated_text = ""
    print(f"DEBUG: stream_chat_response started, stream=True, tools={tools is not None}")
    
    # Check if model is loaded - if not, notify waiting clients
    # The model manager exists but backend may not be loaded yet in on-demand mode
    model_loaded = False
    if current_manager is not None:
        if hasattr(current_manager, 'backend') and current_manager.backend is not None:
            # Check if backend has the model loaded
            if hasattr(current_manager.backend, 'model') and current_manager.backend.model is not None:
                model_loaded = True
        elif hasattr(current_manager, 'model') and current_manager.model is not None:
            # Alternative check for some model managers
            model_loaded = True
    
    # If model not loaded, add to queue and send waiting notifications
    if not model_loaded:
        await queue_manager.add_waiting(request_id)
        wait_interval = 2.0  # Send waiting update every 2 seconds
        last_wait_update = time.time()
        
        # Send initial waiting message
        data = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": {"content": "Waiting for model to load..."},
                "finish_reason": None,
            }],
            "x_queue_info": {
                "status": "waiting",
                "message": "Model is loading, please wait...",
            },
        }
        yield f"data: {json.dumps(data)}\n\n"
        
        # Keep sending wait updates until model is loaded
        # In a real implementation, this would check a loading status
        # For now, we'll send a few updates then proceed
        max_wait_updates = 5
        wait_count = 0
        while wait_count < max_wait_updates:
            await asyncio.sleep(wait_interval)
            wait_time = await queue_manager.get_wait_time(request_id)
            wait_count += 1
            
            queue_pos = await queue_manager.get_queue_position(request_id)
            
            data = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {"content": f""},
                    "finish_reason": None,
                }],
                "x_queue_info": {
                    "status": "waiting",
                    "message": f"Waiting for model... ({int(wait_time)}s)",
                    "queue_position": queue_pos,
                    "wait_time_seconds": int(wait_time),
                },
            }
            yield f"data: {json.dumps(data)}\n\n"
    
    # Mark as starting processing
    await queue_manager.start_processing(request_id, model_name)
    
    # Send "Model starting" message
    data = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": {"content": ""},
            "finish_reason": None,
        }],
        "x_queue_info": {
            "status": "starting",
            "message": "Model starting",
        },
    }
    yield f"data: {json.dumps(data)}\n\n"
    
    try:
        chunk_count = 0
        
        # Debug: Print what is being passed to the model
        if get_global_debug():
            print(f"\n{'='*80}")
            print(f"=== MODEL INPUT (DEBUG) ===")
            print(f"{'='*80}")
            print(f"Model: {model_name}")
            print(f"Max tokens: {max_tokens}")
            print(f"Temperature: {temperature}")
            print(f"Top P: {top_p}")
            print(f"Stop sequences: {stop}")
            print(f"Tools: {tools is not None}")
            print(f"Response format: {response_format}")
            print(f"\n--- Messages ---")
            for i, msg in enumerate(messages):
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                if content and len(content) > 500:
                    content = content[:500] + "... [truncated]"
                print(f"[{i}] {role}: {repr(content)}")
            print(f"{'='*80}\n")
        
        # Use generate_chat_stream for proper chat template handling
        async for chunk in current_manager.generate_chat_stream(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            tools=tools,
            response_format=response_format,
        ):
            chunk_count += 1
            # Always filter malformed content (regex-based, works per-chunk)
            filtered_chunk = filter_malformed_content(chunk)
            
            # NOTE: filter_repetition() and strip_tool_calls_from_content() are NOT applied
            # per-chunk because they need the full accumulated text to work correctly:
            # - filter_repetition() needs enough context (6+ words) to detect n-gram repetitions
            # - strip_tool_calls_from_content() needs complete XML tags that span multiple chunks
            # Both are applied to the complete generated_text after streaming completes.
            
            # Pass through all content including whitespace - it's essential for message composition
            generated_text += filtered_chunk
            
            data = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {"content": filtered_chunk},
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(data)}\n\n"
            # Explicitly flush to ensure data is sent immediately
            await asyncio.sleep(0)
        
        print(f"DEBUG: stream_chat_response completed, {chunk_count} chunks, generated_text length: {len(generated_text)}")
        if not generated_text.strip():
            print(f"DEBUG: Warning - no content generated!")
        
        # In debug mode, dump the full generated text
        if get_global_debug():
            print(f"\n{'='*80}")
            print(f"=== FULL GENERATED TEXT (DEBUG) ===")
            print(f"{'='*80}")
            # Show both raw (actual) content and escaped representation
            print(f"--- RAW CONTENT (actual newlines shown as lines) ---")
            print(generated_text)
            print(f"--- END RAW CONTENT ---")
            print(f"--- ESCAPED CONTENT (repr() - shows \\n for newlines) ---")
            print(repr(generated_text))
            print(f"--- END ESCAPED CONTENT ---")
            print(f"{'='*80}\n")
        
        # Check for tool calls in complete output (for API response format)
        if tools:
            # Convert tools back to Tool objects for parsing
            from typing import cast
            tool_objects = []
            for t in tools:
                try:
                    # Handle both dict and pydantic model formats
                    if isinstance(t, dict):
                        func_data = t.get("function", {})
                        tool_func = ToolFunction(
                            name=func_data.get("name", ""),
                            description=func_data.get("description"),
                            parameters=func_data.get("parameters")
                        )
                    else:
                        # Pydantic model
                        tool_func = ToolFunction(
                            name=t.function.name if hasattr(t.function, 'name') else str(t.function),
                            description=t.function.description if hasattr(t.function, 'description') else None,
                            parameters=t.function.parameters if hasattr(t.function, 'parameters') else None
                        )
                    tool_objects.append(Tool(type=t.get("type", "function") if isinstance(t, dict) else t.type, function=tool_func))
                except Exception as e:
                    print(f"DEBUG: Error converting tool: {e}, tool type: {type(t)}")
                    continue
            try:
                tool_calls = tool_parser.extract_tool_calls(generated_text, tool_objects)
                
                # FIX: Validate extracted tool calls have valid JSON (stream_chat_response)
                if tool_calls:
                    from codai.models.parser import validate_json_complete
                    validated_calls = []
                    for tc in tool_calls:
                        args = tc.get('function', {}).get('arguments', '{}')
                        if isinstance(args, str) and validate_json_complete(args):
                            validated_calls.append(tc)
                        elif isinstance(args, dict):
                            validated_calls.append(tc)
                    if len(validated_calls) != len(tool_calls):
                        print(f"DEBUG: Filtered out {len(tool_calls) - len(validated_calls)} invalid tool calls in stream_chat_response")
                    tool_calls = validated_calls if validated_calls else None
            except Exception as e:
                print(f"DEBUG: Error extracting tool calls: {e}")
                tool_calls = None
            if tool_calls:
                # In debug mode, dump tool calls
                if get_global_debug():
                    print(f"\n{'='*80}")
                    print(f"=== EXTRACTED TOOL CALLS (DEBUG) ===")
                    print(f"{'='*80}")
                    print(json.dumps(tool_calls, indent=2))
                    print(f"{'='*80}\n")
                # Tool calls were extracted and stripped from content during streaming
                # Just send the tool_calls chunk
                data = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {"tool_calls": tool_calls},
                        "finish_reason": "tool_calls",
                        "logprobs": None,
                        "native_finish_reason": "tool_calls",
                    }],
                }
                yield f"data: {json.dumps(data)}\n\n"
            else:
                # Calculate token counts for usage in final chunk
                prompt_text = "\n".join([m.get("content", "") for m in messages])
                prompt_tokens = len(prompt_text.split())
                completion_tokens = len(generated_text.split()) if generated_text else 0
                
                # Use OpenAIFormatter for final chunk sanitization
                formatter = OpenAIFormatter(model_name)
                usage_details = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }
                final_chunk = formatter.format_litellm_chunk("", is_final=True, usage=usage_details)
                yield f"data: {json.dumps(final_chunk)}\n\n"
        else:
            # Calculate token counts for usage in final chunk
            prompt_text = "\n".join([m.get("content", "") for m in messages])
            prompt_tokens = len(prompt_text.split())
            completion_tokens = len(generated_text.split()) if generated_text else 0
            
            # Build complete final chunk with all OpenAI fields
            final_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "logprobs": None,
                    "native_finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "prompt_tokens_details": {
                        "cached_tokens": 0,
                        "audio_tokens": 0,
                    },
                    "completion_tokens_details": {
                        "reasoning_tokens": 0,
                        "audio_tokens": 0,
                    },
                },
                "provider": {
                    "provider_name": "coderai",
                    "provider_id": "coderai",
                },
                "system_fingerprint": None,
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
        
        yield "data: [DONE]\n\n"
    except Exception as e:
        print(f"Error during streaming generation: {e}")
        data = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": {"content": f"\n[Generation error: {str(e)}]"},
                "finish_reason": "stop",
            }],
        }
        yield f"data: {json.dumps(data)}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        # Always clean up queue state
        await queue_manager.finish_processing()

async def generate_chat_response(
    messages: List[Dict],
    model_name: str,
    max_tokens: Optional[int],
    temperature: float,
    top_p: float,
    stop: List[str],
    tools: Optional[List[Dict]],
    current_manager: ModelManager,
    tool_parser: ToolCallParser,
    response_format: Optional[Dict] = None,
    force_reasoning_args: Optional[List[str]] = None,
) -> Dict:
    """Generate non-streaming chat completion response."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    
    # Debug: Print what is being passed to the model
    if get_global_debug():
        print(f"\n{'='*80}")
        print(f"=== MODEL INPUT (DEBUG) ===")
        print(f"{'='*80}")
        print(f"Model: {model_name}")
        print(f"Max tokens: {max_tokens}")
        print(f"Temperature: {temperature}")
        print(f"Top P: {top_p}")
        print(f"Stop sequences: {stop}")
        print(f"Tools: {tools is not None}")
        print(f"Response format: {response_format}")
        print(f"\n--- Messages ---")
        for i, msg in enumerate(messages):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            if content and len(content) > 500:
                content = content[:500] + "... [truncated]"
            print(f"[{i}] {role}: {repr(content)}")
        print(f"{'='*80}\n")
    
    try:
        # Use generate_chat for proper chat template handling
        generated_text = current_manager.generate_chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            tools=tools,
            response_format=response_format,
        )
        
        # Always filter out malformed content
        generated_text = filter_malformed_content(generated_text)
        
        # Apply repetition filtering to prevent infinite loops
        generated_text = filter_repetition(generated_text)
        
        # Dump raw output if enabled
        global_dump = getattr(global_args, 'dump', False) if global_args else False
        if global_dump:
            print(f"\n{'='*80}")
            print(f"=== RAW MODEL OUTPUT (DUMP) ===")
            print(f"{'='*80}")
            print(generated_text)
            print(f"{'='*80}\n")
        
        response_message = {
            "role": "assistant",
            "content": generated_text,
        }
        
        finish_reason = "stop"
        
        # Check for tool calls
        if tools:
            # Convert tools back to Tool objects for parsing
            tool_objects = []
            for t in tools:
                try:
                    # Handle both dict and pydantic model formats
                    if isinstance(t, dict):
                        func_data = t.get("function", {})
                        tool_func = ToolFunction(
                            name=func_data.get("name", ""),
                            description=func_data.get("description"),
                            parameters=func_data.get("parameters")
                        )
                    else:
                        # Pydantic model
                        tool_func = ToolFunction(
                            name=t.function.name if hasattr(t.function, 'name') else str(t.function),
                            description=t.function.description if hasattr(t.function, 'description') else None,
                            parameters=t.function.parameters if hasattr(t.function, 'parameters') else None
                        )
                    tool_objects.append(Tool(type=t.get("type", "function") if isinstance(t, dict) else t.type, function=tool_func))
                except Exception as e:
                    print(f"DEBUG: Error converting tool: {e}, tool type: {type(t)}")
                    continue
            try:
                tool_calls = tool_parser.extract_tool_calls(generated_text, tool_objects)
                
                # FIX: Validate extracted tool calls have valid JSON (generate_chat_response)
                if tool_calls:
                    from codai.models.parser import validate_json_complete
                    validated_calls = []
                    for tc in tool_calls:
                        args = tc.get('function', {}).get('arguments', '{}')
                        if isinstance(args, str) and validate_json_complete(args):
                            validated_calls.append(tc)
                        elif isinstance(args, dict):
                            validated_calls.append(tc)
                    if len(validated_calls) != len(tool_calls):
                        print(f"DEBUG: Filtered out {len(tool_calls) - len(validated_calls)} invalid tool calls in generate_chat_response")
                    tool_calls = validated_calls if validated_calls else None
            except Exception as e:
                print(f"DEBUG: Error extracting tool calls: {e}")
                tool_calls = None
            if tool_calls:
                # Always strip tool call format from content
                clean_content = tool_parser.strip_tool_calls_from_content(generated_text)
                response_message["content"] = clean_content if clean_content.strip() else None
                response_message["tool_calls"] = tool_calls
                finish_reason = "tool_calls"
        
        # Calculate token counts - rough estimate since we don't have direct access to tokenizer
        prompt_text = "\n".join([m.get("content", "") for m in messages])
        prompt_tokens = len(prompt_text.split())
        completion_tokens = len(generated_text.split()) if generated_text else 0
        
        # Use OpenAIFormatter for final sanitization
        formatter = OpenAIFormatter(model_name)
        formatted_response = formatter.format_litellm_full(
            text=response_message.get("content", ""),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            tool_calls=response_message.get("tool_calls")
        )
        
        # Add mock reasoning stats if 'mock' is in force_reasoning_args
        # But only if we don't already have real reasoning in the response
        # Check if reasoning already exists in the message
        existing_reasoning = None
        if "choices" in formatted_response and formatted_response["choices"]:
            choice = formatted_response["choices"][0]
            if "message" in choice:
                existing_reasoning = choice["message"].get("reasoning")
        
        if force_reasoning_args and "mock" in force_reasoning_args and formatted_response and not existing_reasoning:
            # Add fake reasoning tokens to trigger VSCode plugin stats
            mock_reasoning_tokens = 50
            
            # Update usage
            if "usage" in formatted_response:
                formatted_response["usage"]["completion_tokens"] += mock_reasoning_tokens
                formatted_response["usage"]["total_tokens"] += mock_reasoning_tokens
                formatted_response["usage"]["completion_tokens_details"] = {
                    "reasoning_tokens": mock_reasoning_tokens
                }
            
            # Add reasoning to message if not present
            if "choices" in formatted_response and formatted_response["choices"]:
                choice = formatted_response["choices"][0]
                if "message" in choice and "reasoning" not in choice["message"]:
                    choice["message"]["reasoning"] = "Processing task in optimized mode..."
        
        # Dump parsed output if enabled
        if global_dump:
            import json
            print(f"\n{'='*80}")
            print(f"=== PARSED OUTPUT (DUMP) ===")
            print(f"{'='*80}")
            print(json.dumps(formatted_response, indent=2))
            print(f"{'='*80}\n")
        
        return formatted_response
    except Exception as e:
        print(f"Error during generation: {e}")
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

# =============================================================================
# Legacy Text Completions Endpoint (/v1/completions)
# =============================================================================
# NOTE: This is a legacy endpoint for backward compatibility.
# It uses raw text completion (no chat template) instead of the modern
# /v1/chat/completions API. Consider using /v1/chat/completions instead.
# =============================================================================

from codai.pydantic.textrequest import CompletionRequest


@router.post("/v1/completions")
async def completions(request: CompletionRequest):
    """Legacy text completions endpoint (for backward compatibility)."""
    # Get the model for this request
    requested_model = request.model
    
    # Try to get the appropriate model
    mm = multi_model_manager.get_model_for_request(requested_model)
    
    if mm is None:
        # Model not loaded - try to use default
        if model_manager.backend is not None:
            # Fallback to legacy model_manager
            current_manager = model_manager
        else:
            raise HTTPException(status_code=503, detail="Model not loaded")
    else:
        current_manager = mm
    
    prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]
    stop_sequences = []
    if request.stop:
        stop_sequences = [request.stop] if isinstance(request.stop, str) else request.stop
    
    if request.stream:
        from fastapi.responses import StreamingResponse
        return StreamingResponse(
            stream_completion_response(
                prompts[0],
                request.model,
                request.max_tokens,
                request.temperature,
                request.top_p,
                stop_sequences,
                current_manager,
            ),
            media_type="text/event-stream",
        )
    else:
        return await generate_completion_response(
            prompts[0],
            request.model,
            request.max_tokens,
            request.temperature,
            request.top_p,
            stop_sequences,
            current_manager,
        )

async def stream_completion_response(
    prompt: str,
    model_name: str,
    max_tokens: Optional[int],
    temperature: float,
    top_p: float,
    stop: List[str],
    current_manager: ModelManager,
) -> AsyncGenerator[str, None]:
    """Stream legacy completion response."""
    completion_id = f"cmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    
    try:
        async for chunk in current_manager.generate_stream(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        ):
            data = {
                "id": completion_id,
                "object": "text_completion",
                "created": created,
                "model": model_name,
                "choices": [{
                    "text": chunk,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(data)}\n\n"
        
        yield f"data: {json.dumps({'choices': [{'finish_reason': 'stop'}]})}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        print(f"Error during streaming completion: {e}")
        yield f"data: {json.dumps({'choices': [{'finish_reason': 'stop'}]})}\n\n"
        yield "data: [DONE]\n\n"

async def generate_completion_response(
    prompt: str,
    model_name: str,
    max_tokens: Optional[int],
    temperature: float,
    top_p: float,
    stop: List[str],
    current_manager: ModelManager,
) -> Dict:
    """Generate non-streaming legacy completion response."""
    completion_id = f"cmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    
    try:
        generated_text = current_manager.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )
        
        # Calculate token counts if tokenizer available
        if current_manager.tokenizer:
            prompt_tokens = len(current_manager.tokenizer.encode(prompt))
            completion_tokens = len(current_manager.tokenizer.encode(generated_text))
        else:
            prompt_tokens = len(prompt.split())
            completion_tokens = len(generated_text.split())
        
        return {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": model_name,
            "choices": [{
                "text": generated_text,
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
    except Exception as e:
        print(f"Error during completion: {e}")
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")
