"""Model capabilities module."""

from dataclasses import dataclass


@dataclass
class ModelCapabilities:
    """Represents what a model can do."""
    text_generation: bool = False  # LLM/chat completion
    image_to_text: bool = False  # Image understanding (captioning, VQA)
    image_generation: bool = False  # Text-to-image (Stable Diffusion)
    speech_to_text: bool = False  # Audio transcription
    text_to_speech: bool = False  # Speech synthesis
    
    def __str__(self):
        caps = []
        if self.text_generation:
            caps.append("text")
        if self.image_to_text:
            caps.append("image-to-text")
        if self.image_generation:
            caps.append("image")
        if self.speech_to_text:
            caps.append("speech-to-text")
        if self.text_to_speech:
            caps.append("text-to-speech")
        return ", ".join(caps) if caps else "none"


def detect_model_capabilities(model_name: str) -> ModelCapabilities:
    """
    Detect model capabilities based on model name/type.
    
    This is a heuristic detection - actual capabilities may vary.
    """
    caps = ModelCapabilities()
    
    if not model_name:
        return caps
    
    name_lower = model_name.lower()
    
    # Check for image generation models (Stable Diffusion, SDXL, etc.)
    if any(x in name_lower for x in ['stable-diffusion', 'sd15', 'sdxl', 'sd-xl', 'turbo', 'playground']):
        caps.image_generation = True
        return caps  # Usually SD models are dedicated
    
    # Check for vision models (image-to-text)
    if any(x in name_lower for x in ['vision', 'vl-', '-vl', 'llava', 'qwen2-vl', 'qwen-vl', 'phi-4-mini', 'pixtral', 'clip']):
        caps.image_to_text = True
        caps.text_generation = True  # Vision models are also LLMs
        return caps
    
    # Check for TTS models
    if any(x in name_lower for x in ['kokoro', 'tts', 'speech', 'voice']):
        caps.text_to_speech = True
        return caps
    
    # Check for whisper models (speech-to-text)
    if any(x in name_lower for x in ['whisper', 'faster-whisper', 'distil-whisper']):
        caps.speech_to_text = True
        return caps
    
    # Check for GGUF models (typically text models)
    if '.gguf' in name_lower or 'gguf' in name_lower:
        caps.text_generation = True
        return caps
    
    # Default: assume text generation (most HF models are LLMs)
    caps.text_generation = True
    return caps
