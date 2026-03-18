"""Grammar loading utilities for grammar-guided generation."""

import os
from typing import Optional

# Default grammar file path
DEFAULT_GRAMMAR_PATH = os.path.join(os.path.dirname(__file__), "tool_call_grammar.gbnf")

# Cache for the loaded grammar
_grammar_cache: Optional[str] = None


def load_tool_call_grammar(grammar_path: Optional[str] = None) -> str:
    """Load the GBNF grammar for tool calls.
    
    Args:
        grammar_path: Optional path to custom grammar file. 
                     If None, uses default tool_call_grammar.gbnf.
    
    Returns:
        The grammar string.
    """
    global _grammar_cache
    
    path = grammar_path or DEFAULT_GRAMMAR_PATH
    
    # Return cached version if available
    if _grammar_cache is not None and path == DEFAULT_GRAMMAR_PATH:
        return _grammar_cache
    
    try:
        with open(path, 'r') as f:
            grammar = f.read()
        
        # Cache the default grammar
        if path == DEFAULT_GRAMMAR_PATH:
            _grammar_cache = grammar
        
        return grammar
    except FileNotFoundError:
        print(f"Warning: Grammar file not found at {path}")
        return ""
    except Exception as e:
        print(f"Warning: Failed to load grammar from {path}: {e}")
        return ""


def get_tool_call_grammar() -> str:
    """Get the default tool call grammar.
    
    Returns:
        The GBNF grammar string for tool calls.
    """
    return load_tool_call_grammar()


def is_grammar_available() -> bool:
    """Check if the default grammar file is available.
    
    Returns:
        True if grammar is available, False otherwise.
    """
    return os.path.exists(DEFAULT_GRAMMAR_PATH)
