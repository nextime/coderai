"""Utility functions for model handling."""

from typing import Optional, Any


# Global args - can be set by the application
_global_args = None


def set_global_args(args: Any):
    """Set the global arguments object for use in utility functions."""
    global _global_args
    _global_args = args


def get_global_args():
    """Get the global arguments object."""
    global _global_args
    return _global_args


def check_hf_chat_template(model_type: str = "text", model_name: str = None) -> tuple:
    """
    Check if HuggingFace chat template should be used for the model.
    Returns a tuple (should_use, template_name) where template_name is the template to use or None for auto-detect.
    
    Args:
        model_type: The model type ('text', 'image', etc.)
        model_name: The specific model name (optional)
    
    Returns:
        Tuple of (should_use: bool, template_name: str or None)
        template_name is None means auto-detect from tokenizer
    
    Examples:
        # Auto-detect and apply to all models
        --hf-chat-template auto
        
        # Apply to all text models with auto-detect
        --hf-chat-template text
        
        # Apply to specific model with auto-detect
        --hf-chat-template text:llama-3.1
        
        # Apply to specific model with specific template
        --hf-chat-template "llama-3.1:llama3"
        --hf-chat-template "phi-3:chatml"
    """
    global_args = get_global_args()
    hf_chat_template = getattr(global_args, 'hf_chat_template', []) if global_args else []
    
    # If empty list, HF chat template is not enabled
    if not hf_chat_template:
        return (False, None)
    
    for spec in hf_chat_template:
        # Handle auto-detect - try to load HF tokenizer and auto-detect template
        if spec == 'auto' or spec == '':
            # Applies to all models when using 'auto'
            return (True, None)
        
        # Check if this spec has a template specified after the model name
        # Format: "model_name:template_name" or "type:model_name:template_name"
        parts = spec.split(':')
        
        if len(parts) == 1:
            # Just a type or single value
            spec_val = parts[0]
            if spec_val == model_type or spec_val == '*':
                return (True, None)
            # Check if it matches the model name directly (when model_type is part of the name)
            if model_name and (spec_val in model_name or model_name in spec_val):
                return (True, None)
        elif len(parts) == 2:
            # Format: "type:model_name" or "model_name:template"
            spec_type = parts[0]
            spec_model = parts[1]
            
            # Check if it's "text" or "image" type
            if spec_type in ('text', 'image', '*'):
                if spec_type == model_type or spec_type == '*':
                    # Check if model name matches
                    if spec_model == model_name or spec_model == '*':
                        return (True, None)
            else:
                # It's "model_name:template" format
                if model_name and (spec_model in model_name or model_name in spec_model):
                    return (True, spec_type)  # spec_type is actually the template!
        elif len(parts) == 3:
            # Format: "type:model_name:template"
            spec_type = parts[0]
            spec_model = parts[1]
            spec_template = parts[2]
            
            if spec_type == model_type or spec_type == '*':
                if spec_model == model_name or spec_model == '*':
                    return (True, spec_template)
    
    return (False, None)


def get_resolved_model_name(requested_model: str, current_manager = None) -> str:
    """
    Get the actual model name to return in the response.
    
    Handles aliases and ensures the correct model identifier is returned.
    """
    global_args = get_global_args()
    
    # Get model aliases from args if available
    model_aliases = getattr(global_args, 'model_aliases', {}) if global_args else {}
    
    # If it's an alias, return the resolved name
    if requested_model in model_aliases:
        return model_aliases[requested_model]
    
    # Try to get from current manager if available
    if current_manager is not None:
        # Check if the model is loaded in the manager
        if hasattr(current_manager, 'models') and current_manager.models:
            # If requested_model is "default" or empty, get the actual loaded model (not "default")
            if requested_model in ("default", "", None) or not requested_model:
                # Try default_model first
                if hasattr(current_manager, 'default_model') and current_manager.default_model and current_manager.default_model != "default":
                    return current_manager.default_model
                # Otherwise return the first model that is not "default"
                for key in current_manager.models.keys():
                    if key != "default":
                        return key
                # Fallback to first model if all are "default"
                return list(current_manager.models.keys())[0]
            # Check if the model is loaded in the manager
            for key, model in current_manager.models.items():
                if requested_model == key or requested_model in key:
                    return key
    
    # Check if it's a HuggingFace model ID - if so, return as-is
    if '/' in requested_model:
        return requested_model
    
    # Check if it's a URL - return as-is
    if requested_model.startswith('http://') or requested_model.startswith('https://'):
        return requested_model
    
    # Otherwise return as-is
    return requested_model


def get_model_family(model_name: str) -> str:
    """Detect model family from model name."""
    if not model_name:
        return 'generic'
    
    model_lower = model_name.lower()
    
    # Check for reasoning models first
    if 'reasoning' in model_lower or 'deepseek-r1' in model_lower:
        return 'deepseek'
    
    if 'qwen' in model_lower:
        if 'qwen3' in model_lower or 'qwen3.5' in model_lower:
            return 'qwen3'
        elif 'qwen2' in model_lower:
            return 'qwen2'
        return 'qwen'
    if 'llama' in model_lower:
        if 'llama4' in model_lower:
            return 'llama4'
        elif 'llama3' in model_lower:
            return 'llama3'
        return 'llama'
    if 'mistral' in model_lower or 'mixtral' in model_lower:
        return 'mistral'
    if 'deepseek' in model_lower:
        return 'deepseek'
    if 'gemma' in model_lower:
        return 'gemma'
    if 'yi' in model_lower:
        return 'yi'
    if 'hermes' in model_lower:
        return 'hermes'
    if 'phi' in model_lower:
        return 'phi'
    if 'command-r' in model_lower:
        return 'command-r'
    return 'generic'


def get_reasoning_stop_tokens(model_family: str) -> tuple:
    """Get stop tokens for reasoning mode based on model family.
    
    Returns tuple of (start_token, end_token, additional_stops)
    """
    if model_family == 'qwen3':
        return (
            "<|im_start|>assistant\n",
            "<|im_end|>",
            ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]
        )
    elif model_family == 'qwen2' or model_family == 'qwen':
        return (
            "<|im_start|>assistant\n",
            "<|im_end|>",
            ["<|im_end|>", "<|endoftext|>"]
        )
    elif model_family == 'deepseek':
        return (
            "<｜Assistant｜>",
            "<｜endofassistant｜>",
            ["<｜endofassistant｜>", "<｜User｜>", "<｜endoftext｜>"]
        )
    elif model_family == 'llama3':
        return (
            "<|start_header_id|>assistant<|end_header_id|>\n\n<thought>\n",
            "</thought>",
            ["</thought>", "<|eot_id|>", "<|end_of_text|>"]
        )
    elif model_family == 'llama':
        return (
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
            "<|eot_id|>",
            ["<|eot_id|>", "<|end_of_text|>"]
        )
    elif model_family == 'mistral':
        return (
            "[/INST] <thought>\n",
            "</thought>",
            ["</thought>", "</INST>", "[INST]"]
        )
    elif model_family == 'gemma':
        return (
            "<start_of_turn>model\n<thought>\n",
            "</thought>",
            ["</thought>", "<end_of_turn>", "<start_of_turn>"]
        )
    elif model_family == 'yi' or model_family == 'hermes':
        return (
            "<|im_start|>assistant\n",
            "<|im_end|>",
            ["<|im_end|>", "<|endoftext|>"]
        )
    elif model_family == 'phi':
        return (
            "<|assistant|>\n",
            "<|end|>",
            ["<|end|>", "<|endoftext|>", "<|user|>", "<|system|>"]
        )
    elif model_family == 'command-r':
        return (
            "<|start|>assistant\n",
            "<|end|>",
            ["<|end|>", "<|endoftext|>", "<|start|>"]
        )
    else:
        # Default fallback - try common tokens
        return (
            "<|im_start|>assistant\n",
            "<|im_end|>",
            ["<|im_end|>", "<|endoftext|>"]
        )


def get_reasoning_system_prompt(model_family: str) -> str:
    """Get system prompt injection for forcing reasoning on non-native models."""
    
    if model_family in ('qwen3', 'qwen2', 'qwen'):
        return "You must reason step-by-step inside <thought> tags before every response."
    elif model_family == 'deepseek':
        return "You must reason step-by-step inside <thought> tags before every response."
    elif model_family in ('llama3', 'llama'):
        return "You must reason step-by-step inside <thought> tags before every response."
    elif model_family == 'mistral':
        return "You must reason step-by-step inside <thought> tags before every response."
    elif model_family == 'gemma':
        return "You must reason step-by-step inside <thought> tags before every response. Use <start_of_turn>model for your response."
    elif model_family in ('yi', 'hermes'):
        return "You must reason step-by-step inside <|im_start|>assistant tags before every response."
    elif model_family == 'phi':
        return "You must reason step-by-step inside <|assistant> tags before every response."
    elif model_family == 'command-r':
        return "You must reason step-by-step before every response."
    else:
        return "You must reason step-by-step before every response."
