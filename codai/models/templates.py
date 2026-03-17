"""
Agentic Template Manager - Automates prompt injection for agentic behavior.
Supports the 'Big 10' with specific triggers for tool-calling.

Uses Prompt Seeding technique to force reasoning in LLM models:
- Ends prompt with thought tag (<think>, <thought>, Thought:) to force reasoning
- Uses raw completion instead of chat API to bypass validation
- Provides family-specific stop tokens for reasoning extraction
"""

import re
from typing import Optional, Dict, List, Tuple


class AgenticTemplateManager:
    """
    Automates prompt injection to force models into an Agentic 'Thought-Action' loop.
    Supports the 'Big 10' with specific triggers for tool-calling.
    
    Uses Prompt Seeding to force reasoning by ending prompts with thought tags.
    """
    
    # Family-specific prefixes for Prompt Seeding (force reasoning start)
    # These templates end with the thought tag to force the model to start reasoning
    REASONING_PREFIXES = {
        "qwen": "<|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n",
        "deepseek": "<|begin_of_sentence|><|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n",
        "llama3": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n<thought>\n",
        "mistral": "[INST] {sys}\n\n{user} [/INST] Thought:\n",
        "anthropic": "\n\nSystem: {sys}\n\nHuman: {user}\n\nAssistant: <thinking>\n",
        "command-r": "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{sys}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>{user}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><thought>\n",
        "gemma": "<bos><start_of_turn>user\n{sys}\n\n{user}<end_of_turn>\n<start_of_turn>model\n<thought>\n",
        "phi3": "<|system|>\n{sys}<|end|>\n<|user|>\n{user}<|end|>\n<|assistant|>\n<|thought|>\n",
        "yi": "<|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n",
        "generic": "System: {sys}\nUser: {user}\nAssistant: <think>\n"
    }
    
    # Stop tokens for each family (used to stop reasoning generation)
    REASONING_STOP_TOKENS = {
        "qwen": ["</think>", "<|im_end|>", "<|endoftext|>"],
        "deepseek": ["</think>", "<|im_end|>", "<|endoftext|>"],
        "llama3": ["</thought>", "<|eot_id|>", "<|end_of_text|>"],
        "mistral": ["\nAction:", "\nObservation:"],
        "anthropic": ["</thinking>", "\n\nHuman:"],
        "command-r": ["</thought>", "<|END_OF_TURN_TOKEN|>"],
        "gemma": ["</thought>", "<end_of_turn>"],
        "phi3": ["<|end|>", "<|assistant|>"],
        "yi": ["</think>", "<|im_end|>", "<|endoftext|>"],
        "generic": ["</think>", "</thought>", "Thought:"]
    }
    
    # Original FAMILIES config for backward compatibility
    FAMILIES = {
        "qwen": {"name": "Qwen", "prefix": "<|im_start|>", "suffix": "<|im_end|>\n", "thought_tag": "<|thought|>", "call_tag": "<tool_call>"},
        "llama3": {"name": "Llama-3", "prefix": "<|start_header_id|>", "suffix": "<|end_header_id|>\n\n", "thought_tag": "<thought>", "call_tag": "<tool_call>"},
        "deepseek": {"name": "DeepSeek", "prefix": "<｜", "suffix": "｜>\n", "thought_tag": "<think>", "call_tag": "<tool_call>"},
        "mistral": {"name": "Mistral", "user": "[INST] ", "bot": " [/INST]", "thought_tag": "Thought: ", "call_tag": "Action: "},
        "anthropic": {"name": "Claude", "user": "\n\nHuman: ", "bot": "\n\nAssistant: ", "thought_tag": "<thinking>", "call_tag": "<tool_calls>"},
        "gemma": {"name": "Gemma", "user": "<start_of_turn>user\n", "bot": "<start_of_turn>model\n", "end": "<end_of_turn>\n"},
        "phi3": {"name": "Phi-3", "prefix": "<|", "suffix": "|>\n", "end": "<|end|>\n"},
        "cohere": {"name": "Command-R", "user": "<|USER_TOKEN|>", "bot": "<|CHATBOT_TOKEN|>", "thought_tag": "<thought>"},
        "yi": {"name": "Yi", "prefix": "<|im_start|>", "suffix": "<|im_end|>\n"},
        "openai": {"name": "GPT"}
    }

    def __init__(self, model_name: str):
        self.model_name = model_name.lower()
        self.family_key = self._detect_family()
        self.config = self.FAMILIES.get(self.family_key, self.FAMILIES["qwen"])

    def _detect_family(self):
        mapping = {
            "qwen": "qwen", 
            "llama": "llama3",  # Match llama, llama3, llama-3
            "deepseek": "deepseek",
            "mistral": "mistral", "mixtral": "mistral", 
            "claude": "anthropic",
            "gemma": "gemma", 
            "phi": "phi3",  # Match phi, phi3, phi-3
            "command": "cohere", 
            "yi": "yi"
        }
        for k, v in mapping.items():
            if k in self.model_name: return v
        return "openai" if "gpt" in self.model_name else "qwen"

    def get_agent_system_prompt(self, base_prompt: str, use_reasoning_tag: bool = False) -> str:
        """Injects agentic instructions specific to the model's strengths.
        
        Args:
            base_prompt: The base system prompt
            use_reasoning_tag: If True, use the reasoning tag (</think>) for consistency with prompt seeding.
                             If False, use the model's preferred thought tag (e.g., <|thought|>).
        """
        # Use reasoning tag for consistency with prompt seeding, or model's preferred tag
        if use_reasoning_tag:
            thought_tag = self.THOUGHT_TAGS.get(self.family_key, "</think>")
        else:
            thought_tag = self.config.get('thought_tag', 'Thought:')
        
        agent_addon = (
            "\n\nCRITICAL: You are an agent with access to tools. "
            f"Use the {thought_tag} tag to reason step-by-step "
            "before providing a tool call. If you have enough info, provide the final answer."
        )
        return f"{base_prompt}{agent_addon}"

    def force_reasoning_prompt(self, system_prompt: str, user_question: str) -> str:
        """
        Constructs a raw prompt that forces the model to start in a reasoning state.
        
        Uses Prompt Seeding: ends the prompt exactly where we want the model to start -
        at the opening thought tag (<think>, <thought>, Thought:).
        
        This "token hijacking" corners the model's next token prediction to generate
        logical reasoning steps.
        
        Args:
            system_prompt: The system instructions
            user_question: The user's question/query
            
        Returns:
            Formatted prompt string ending with thought tag to force reasoning
        """
        # Get the family-specific template (fallback to generic)
        template = self.REASONING_PREFIXES.get(
            self.family_key, 
            self.REASONING_PREFIXES["generic"]
        )
        
        # Format the base template
        prompt = template.format(sys=system_prompt, user=user_question)
        
        # Get the thought tag for this family
        thought_tag = self.THOUGHT_TAGS.get(self.family_key, "<think>")
        
        # Add "The user requested" after the thought tag to guide reasoning
        prompt += f"The user requested: {user_question}\n"
        
        return prompt
    
    def get_stop_tokens(self) -> List[str]:
        """
        Get the appropriate stop tokens for this model family.
        
        These tokens are used to stop reasoning generation and can be used
        to parse the reasoning from the final response.
        
        Returns:
            List of stop token strings
        """
        return self.REASONING_STOP_TOKENS.get(
            self.family_key,
            self.REASONING_STOP_TOKENS["generic"]
        )
    
    # Map family keys to their thought tags for extraction
    THOUGHT_TAGS = {
        "qwen": "<think>",
        "deepseek": "<think>",
        "llama3": "<thought>",
        "mistral": "Thought:",
        "anthropic": "<thinking>",
        "gemma": "<thought>",
        "phi3": "<|thought|>",
        "yi": "</think>",
        "cohere": "<thought>",
        "generic": "<think>"
    }
    
    # Closing tags for each family
    CLOSE_TAGS = {
        "qwen": "</think>",
        "deepseek": "</think>",
        "llama3": "</thought>",
        "mistral": None,  # Ends at Action: or newline
        "anthropic": "</thinking>",
        "gemma": "</thought>",
        "phi3": "<|end|>",
        "yi": "</think>",
        "cohere": "</thought>",
        "generic": "</think>"
    }
    
    def extract_reasoning(self, response: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract reasoning (thought) and final answer from a response.
        
        Args:
            response: The raw model response containing reasoning and answer
            
        Returns:
            Tuple of (reasoning, final_answer). Either or both may be None.
        """
        thought_tag = self.THOUGHT_TAGS.get(self.family_key, "<think>")
        close_tag = self.CLOSE_TAGS.get(self.family_key)
        
        # Try to extract reasoning
        reasoning = None
        final_answer = response
        
        if thought_tag in response:
            start_idx = response.find(thought_tag)
            if close_tag and close_tag in response:
                end_idx = response.find(close_tag, start_idx)
                if end_idx > start_idx:
                    reasoning = response[start_idx + len(thought_tag):end_idx].strip()
                    final_answer = response[end_idx + len(close_tag):].strip()
            elif self.family_key == "mistral":
                # For Mistral-style, reasoning ends at Action: or newline
                rest = response[start_idx + len(thought_tag):]
                action_idx = rest.find("\nAction:")
                if action_idx >= 0:
                    reasoning = rest[:action_idx].strip()
                    final_answer = rest[action_idx:].strip()
                else:
                    reasoning = rest.strip()
                    final_answer = ""
            else:
                # No clear closing tag, take everything after thought tag
                reasoning = response[start_idx + len(thought_tag):].strip()
                final_answer = ""
        
        return reasoning, final_answer
    
    def format_for_inference(self, messages: list) -> str:
        """Constructs the prompt string and forces the 'Thought' start."""
        if self.family_key == "openai": return messages
        
        prompt = ""
        f = self.config
        
        for m in messages:
            role, content = m['role'], m['content']
            
            # ChatML Style (Qwen, Llama3, Yi, DeepSeek)
            if "prefix" in f:
                prompt += f"{f['prefix']}{role}{f['suffix']}{content}"
                if "end" in f: prompt += f["end"]
                else: prompt += f.get("suffix", "\n")
            
            # Instruction Style (Mistral, Gemma)
            elif self.family_key == "mistral":
                if role == "user": prompt += f"{f['user']}{content}{f['bot']}"
                elif role == "assistant": prompt += f" {content} "
        
        # THE AGENTIC PUSH: Force the assistant to start with a THOUGHT
        thought_trigger = f.get("thought_tag", "Thought: ")
        if "prefix" in f:
            prompt += f"{f['prefix']}assistant{f['suffix']}{thought_trigger}"
        else:
            prompt += f"{f.get('bot', 'Assistant: ')}{thought_trigger}"
            
        return prompt
    
    def format_for_raw_completion(self, system_prompt: str, user_message: str, 
                                inject_system: bool = True,
                                force_reasoning: bool = True) -> Tuple[str, List[str]]:
        """
        Format prompt for raw completion (bypassing chat API).
        
        Args:
            system_prompt: System instructions
            user_message: User message/query
            inject_system: If True, injects agentic system instructions
            force_reasoning: If True, seeds prompt with thought tag to force reasoning
            
        Returns:
            Tuple of (formatted_prompt, stop_tokens)
        """
        effective_system = system_prompt
        
        # Inject system prompt if requested
        if inject_system:
            effective_system = self.get_agent_system_prompt(system_prompt)
        
        if force_reasoning:
            # Use prompt seeding to force reasoning
            prompt = self.force_reasoning_prompt(effective_system, user_message)
        else:
            # Use simple concatenation without seeding
            template = self.REASONING_PREFIXES.get(
                self.family_key, 
                self.REASONING_PREFIXES["generic"]
            )
            # Remove the thought tag at the end
            template = template.replace("{sys}", "{sys}").replace("{user}", "{user}")
            prompt = template.format(sys=effective_system, user=user_message)
            # Remove trailing thought tag
            thought_tag = self.THOUGHT_TAGS.get(self.family_key, "<think>")
            if prompt.endswith(thought_tag + "\n"):
                prompt = prompt[:-len(thought_tag + "\n")]
        
        stop_tokens = self.get_stop_tokens()
        
        return prompt, stop_tokens


# Convenience function for quick prompting
def create_reasoning_prompt(model_name: str, system_prompt: str, user_question: str,
                            inject_system: bool = True,
                            force_reasoning: bool = True) -> Tuple[str, List[str]]:
    """
    Convenience function to create a forced reasoning prompt.
    
    Args:
        model_name: Name of the model (e.g., "qwen3", "llama3", "deepseek")
        system_prompt: System instructions
        user_question: User question
        inject_system: If True, injects agentic system instructions
        force_reasoning: If True, seeds prompt with thought tag to force reasoning
        
    Returns:
        Tuple of (formatted_prompt, stop_tokens)
        
    Examples:
        # Full injection + seeding (default)
        prompt, stops = create_reasoning_prompt("qwen3", "You are helpful.", "Hi!")
        
        # System prompt only, no seeding
        prompt, stops = create_reasoning_prompt("qwen3", "You are helpful.", "Hi!", 
                                               inject_system=True, force_reasoning=False)
        
        # Seeding only, no system injection
        prompt, stops = create_reasoning_prompt("qwen3", "You are helpful.", "Hi!", 
                                               inject_system=False, force_reasoning=True)
    """
    manager = AgenticTemplateManager(model_name)
    return manager.format_for_raw_completion(system_prompt, user_question, 
                                            inject_system=inject_system,
                                            force_reasoning=force_reasoning)
