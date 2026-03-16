"""
Agentic Template Manager - Automates prompt injection for agentic behavior.
Supports the 'Big 10' with specific triggers for tool-calling.
"""

import re


class AgenticTemplateManager:
    """
    Automates prompt injection to force models into an Agentic 'Thought-Action' loop.
    Supports the 'Big 10' with specific triggers for tool-calling.
    """
    
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
            "qwen": "qwen", "llama-3": "llama3", "deepseek": "deepseek",
            "mistral": "mistral", "mixtral": "mistral", "claude": "anthropic",
            "gemma": "gemma", "phi-3": "phi3", "command": "cohere", "yi": "yi"
        }
        for k, v in mapping.items():
            if k in self.model_name: return v
        return "openai" if "gpt" in self.model_name else "qwen"

    def get_agent_system_prompt(self, base_prompt: str) -> str:
        """Injects agentic instructions specific to the model's strengths."""
        agent_addon = (
            "\n\nCRITICAL: You are an agent with access to tools. "
            f"Use the {self.config.get('thought_tag', 'Thought:')} tag to reason step-by-step "
            "before providing a tool call. If you have enough info, provide the final answer."
        )
        return f"{base_prompt}{agent_addon}"

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
