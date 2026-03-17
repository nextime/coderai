"""CUDA backend using HuggingFace Transformers."""

import os
from typing import Optional, List, Dict
from threading import Thread

from codai.backends.base import ModelBackend
from codai.models.capabilities import detect_model_capabilities, check_flash_attn_availability
from codai.pydantic.textrequest import ChatMessage


class NvidiaBackend(ModelBackend):
    """Backend for NVIDIA GPUs using HuggingFace Transformers."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.device = None
        self.use_flash_attn = False
        self.flash_attn_available = False
        
    def check_flash_attn_support(self) -> None:
        """Check and print Flash Attention availability status."""
        self.flash_attn_available = check_flash_attn_availability()
        if self.use_flash_attn:
            if self.flash_attn_available:
                print("Flash Attention 2: Available and enabled")
            else:
                print("Warning: Flash Attention 2 requested but not installed")
                print("Install with: pip install flash-attn --no-build-isolation")
                print("Falling back to standard attention")
                self.use_flash_attn = False
    
    def _detect_device(self) -> str:
        """Auto-detect available GPU or fall back to CPU."""
        import torch
        if torch.cuda.is_available():
            if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                print(f"ROCm/HIP detected: {torch.version.hip}")
                return "cuda"
            else:
                print(f"CUDA detected: {torch.version.cuda}")
                return "cuda"
        else:
            print("No GPU detected, using CPU")
            return "cpu"
    
    def _get_available_vram(self) -> int:
        """Get available VRAM in bytes."""
        import torch
        if not torch.cuda.is_available():
            return 0
        try:
            total_vram = 0
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_vram += props.total_memory
            return total_vram
        except Exception as e:
            print(f"Warning: Could not detect VRAM: {e}")
            return 0
    
    def _estimate_model_size(self, model_name: str) -> Optional[int]:
        """Estimate model size in bytes from config."""
        from transformers import AutoConfig
        try:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            if hasattr(config, 'num_parameters'):
                num_params = config.num_parameters
            elif hasattr(config, 'n_params'):
                num_params = config.n_params
            elif hasattr(config, 'num_hidden_layers') and hasattr(config, 'hidden_size'):
                layers = config.num_hidden_layers
                hidden = config.hidden_size
                vocab_size = getattr(config, 'vocab_size', 50000)
                num_params = (vocab_size * hidden_size) + (layers * 4 * hidden * hidden)
            else:
                return None
            return num_params * 2
        except Exception as e:
            print(f"Warning: Could not estimate model size: {e}")
            return None
    
    def _get_gpu_memory_map(self) -> Dict:
        """Get max_memory dict for Accelerate."""
        import torch
        import psutil
        max_memory = {}
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_vram = props.total_memory
                usable_vram = int(total_vram * 0.93)
                max_memory[i] = usable_vram
                print(f"  GPU {i}: {total_vram / 1e9:.1f}GB total, {usable_vram / 1e9:.1f}GB usable")
        
        manual_ram_gb = getattr(self, '_pending_ram_gb', None)
        if manual_ram_gb:
            max_memory['cpu'] = int(manual_ram_gb * 1e9)
            print(f"  CPU: {manual_ram_gb}GB (user specified)")
        else:
            available_ram = psutil.virtual_memory().available
            usable_ram = max(0, available_ram - int(4e9))
            max_memory['cpu'] = usable_ram
            print(f"  CPU: {usable_ram / 1e9:.1f}GB (auto-detected, 4GB reserved for system)")
        
        return max_memory
    
    def _try_load_model(self, model_name: str, load_kwargs: dict, device: str):
        """Try to load model with given settings."""
        import torch
        from transformers import AutoModelForCausalLM
        
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
            if device == "cpu" and load_kwargs.get('device_map') is None:
                model = model.to(device)
            return model
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            error_msg = str(e).lower()
            if "out of memory" in error_msg or "cuda" in error_msg or "oom" in error_msg:
                return None
            raise
        except TypeError as e:
            error_msg = str(e).lower()
            if "load_in_4bit" in error_msg or "load_in_8bit" in error_msg or "unexpected keyword argument" in error_msg:
                if 'load_in_4bit' in load_kwargs or 'load_in_8bit' in load_kwargs:
                    print(f"Warning: Model does not support bitsandbytes quantization")
                    print("Retrying without quantization...")
                    retry_kwargs = load_kwargs.copy()
                    retry_kwargs.pop('load_in_4bit', None)
                    retry_kwargs.pop('load_in_8bit', None)
                    try:
                        model = AutoModelForCausalLM.from_pretrained(model_name, **retry_kwargs)
                        if device == "cpu" and retry_kwargs.get('device_map') is None:
                            model = model.to(device)
                        print("Model loaded successfully without quantization")
                        return model
                    except (RuntimeError, torch.cuda.OutOfMemoryError) as e2:
                        error_msg2 = str(e2).lower()
                        if "out of memory" in error_msg2 or "cuda" in error_msg2 or "oom" in error_msg2:
                            return None
                        raise
                    except TypeError:
                        raise e
            raise
    
    def _is_moe_model(self, model_name: str) -> bool:
        """Check if model is a MoE model."""
        moe_indicators = ['moe', 'mixtral', 'qwen3_5_moe', 'qwen3.5_moe', 'expert', 'a3b']
        model_name_lower = model_name.lower()
        return any(indicator in model_name_lower for indicator in moe_indicators)
    
    def _get_vram_percentages_for_strategy(self, strategy: str, is_moe: bool, total_vram_gb: float) -> list:
        """Get VRAM percentage steps based on offload strategy."""
        if strategy == "conservative":
            print(f"  Using conservative offload strategy")
            if is_moe:
                return [0.70, 0.65, 0.60, 0.50, 0.40, 0.30, 0.20, 0.0]
            return [0.80, 0.75, 0.70, 0.65, 0.50, 0.40, 0.30, 0.20, 0.0]
        elif strategy == "balanced":
            print(f"  Using balanced offload strategy")
            if is_moe:
                return [0.75, 0.70, 0.65, 0.60, 0.50, 0.40, 0.30, 0.20, 0.0]
            return [0.85, 0.80, 0.75, 0.70, 0.65, 0.50, 0.40, 0.30, 0.20, 0.0]
        elif strategy == "aggressive":
            print(f"  Using aggressive offload strategy")
            if is_moe:
                return [0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.50, 0.40, 0.30, 0.20, 0.0]
            return [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.50, 0.40, 0.30, 0.20, 0.0]
        elif strategy == "sequential":
            print(f"  Using sequential offload strategy")
            if is_moe:
                return [0.80, 0.78, 0.76, 0.74, 0.72, 0.70, 0.68, 0.66, 0.64, 0.62, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.0]
            return [0.93, 0.91, 0.89, 0.87, 0.85, 0.83, 0.81, 0.79, 0.77, 0.75, 0.73, 0.71, 0.69, 0.67, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.
