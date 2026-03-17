"""Utility functions for model handling."""

from typing import Optional


def check_hf_chat_template(model_type: str = "text", model_name: str = None) -> tuple:
    """Check if a model supports HF chat template."""
    return (True, "chatml")


def get_resolved_model_name(requested_model: str, current_manager = None) -> str:
    """Get the resolved model name."""
    return requested_model


def get_model_family(model_name: str) -> str:
    """Detect model family from model name."""
    model_lower = model_name.lower()
    if 'qwen' in model_lower:
        return 'qwen'
    if 'llama' in model_lower:
        return 'llama'
    if 'mistral' in model_lower:
        return 'mistral'
    if 'deepseek' in model_lower:
        return 'deepseek'
    if 'gemma' in model_lower:
        return 'gemma'
    if 'yi' in model_lower:
        return 'yi'
    if 'hermes' in model_lower:
        return 'hermes'
    return 'generic'


def get_reasoning_stop_tokens(model_family: str) -> tuple:
    """Get stop tokens for reasoning mode based on model family.
    
    Returns tuple of (start_token, end_token, additional_stops)
    """
    if model_family == 'qwen':
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
    else:
        # Default fallback - try common tokens
        return (
            "<|im_start|>assistant\n",
            "<|im_end|>",
            ["<|im_end|>", "<|endoftext|>"]
        )


def get_reasoning_system_prompt(model_family: str) -> str:
    """Get system prompt injection for forcing reasoning on non-native models."""
    
    if model_family == 'qwen':
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
    else:
        return "You must reason step-by-step before every response."
