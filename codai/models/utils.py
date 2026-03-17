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
    return 'generic'


def get_reasoning_stop_tokens(model_family: str) -> tuple:
    """Get stop tokens for reasoning mode based on model family."""
    if model_family == 'qwen':
        return ('<|im_end|>', '<|endoftext|>')
    if model_family == 'deepseek':
        return ('</Thinking>',)
    return ('<|end|>',)


def get_reasoning_system_prompt(model_family: str) -> str:
    """Get the system prompt injection for forcing reasoning on non-native models."""
    if model_family == 'qwen':
        return "Please think carefully before responding."
    return ""
