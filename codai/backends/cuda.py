"""CUDA backend using HuggingFace Transformers."""

import os
from typing import Optional, List, Dict
from threading import Thread
from abc import ABC

# Import from codai modules
from codai.backends.base import ModelBackend
from codai.models.capabilities import detect_model_capabilities
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
        try:
            import flash_attn
            self.flash_attn_available = True
        except ImportError:
            self.flash_attn_available = False
        
        # Always print the status when model is loaded (for visibility)
        if self.use_flash_attn:
            if self.flash_attn_available:
                print("Flash Attention 2: Available and enabled")
            else:
                print("Warning: Flash Attention 2 requested but not installed")
                print("Install with: pip install flash-attn --no-build-isolation")
                print("Falling back to standard attention")
                self.use_flash_attn = False
        else:
            # Print availability status even when not requested (for transparency)
            if self.flash_attn_available:
                print("Flash Attention 2: Available (not enabled)")
            else:
                print("Flash Attention 2: Not available")
    
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
            return [0.93, 0.91, 0.89, 0.87, 0.85, 0.83, 0.81, 0.79, 0.77, 0.75, 0.73, 0.71, 0.69, 0.67, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.20, 0.0]
        else:
            if total_vram_gb < 3:
                print(f"  Detected small GPU ({total_vram_gb:.1f}GB), using aggressive VRAM usage")
                return [0.99, 0.95, 0.90, 0.85, 0.75, 0.65, 0.50, 0.35, 0.20, 0.0]
            elif total_vram_gb <= 8:
                print(f"  Detected medium GPU ({total_vram_gb:.1f}GB), using high VRAM usage")
                return [0.96, 0.90, 0.85, 0.75, 0.65, 0.50, 0.35, 0.20, 0.0]
            else:
                if is_moe:
                    print(f"  Detected large GPU ({total_vram_gb:.1f}GB), using MoE-safe VRAM usage")
                    return [0.80, 0.75, 0.70, 0.65, 0.60, 0.50, 0.40, 0.30, 0.20, 0.0]
                else:
                    print(f"  Detected large GPU ({total_vram_gb:.1f}GB), using conservative VRAM usage")
                    return [0.93, 0.85, 0.75, 0.65, 0.50, 0.35, 0.20, 0.0]
    
    def _get_vram_percentages_for_gpu(self, model_name: str = "", strategy: str = "auto", max_gpu_percent: float = None) -> list:
        """Get VRAM percentage steps based on GPU memory size."""
        import torch
        
        if not torch.cuda.is_available():
            return [0.0]
        
        if max_gpu_percent is not None:
            max_pct = max(0.05, min(1.0, max_gpu_percent / 100.0))
            print(f"  Using custom max GPU percent: {max_pct*100:.0f}%")
            steps = []
            current = max_pct
            while current > 0.05:
                steps.append(current)
                if current > 0.3:
                    current -= 0.05
                elif current > 0.15:
                    current -= 0.03
                else:
                    current -= 0.02
            steps.append(0.0)
            return steps
        
        total_vram_gb = 0
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_vram_gb += props.total_memory / 1e9
        
        is_moe = self._is_moe_model(model_name)
        if is_moe:
            print(f"  Detected MoE model, using extra conservative VRAM limits")
        
        return self._get_vram_percentages_for_strategy(strategy, is_moe, total_vram_gb)
    
    def load_model(self, model_name: str, **kwargs) -> None:
        """Load the model using HuggingFace Transformers with automatic OOM handling."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        offload_dir = kwargs.get('offload_dir')
        load_in_4bit = kwargs.get('load_in_4bit', False)
        load_in_8bit = kwargs.get('load_in_8bit', False)
        manual_ram_gb = kwargs.get('manual_ram_gb')
        flash_attn = kwargs.get('flash_attn', False)
        offload_strategy = kwargs.get('offload_strategy', 'auto')
        max_gpu_percent = kwargs.get('max_gpu_percent', None)
        
        self._pending_ram_gb = manual_ram_gb
        
        print(f"Loading HuggingFace model: {model_name}")
        
        self.use_flash_attn = flash_attn
        self.check_flash_attn_support()
        
        self.device = self._detect_device()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        load_kwargs = {'trust_remote_code': True}
        
        if load_in_4bit or load_in_8bit:
            if 'qwen3.5' in model_name.lower() and ('a3b' in model_name.lower() or 'moe' in model_name.lower()):
                print(f"Warning: {model_name} does not support bitsandbytes quantization")
            else:
                try:
                    import bitsandbytes as bnb
                    print(f"Using {4 if load_in_4bit else 8}-bit quantization")
                    load_kwargs['load_in_4bit'] = load_in_4bit
                    load_kwargs['load_in_8bit'] = load_in_8bit
                except ImportError:
                    print("Warning: bitsandbytes not installed. Quantization disabled.")
        
        if self.device == "cuda":
            load_kwargs['dtype'] = torch.float16
        else:
            load_kwargs['dtype'] = torch.float32
        
        if offload_dir:
            os.makedirs(offload_dir, exist_ok=True)
            load_kwargs['offload_folder'] = offload_dir
        
        if self.use_flash_attn and self.flash_attn_available:
            load_kwargs['attn_implementation'] = "flash_attention_2"
            print("Using Flash Attention 2")
        
        model = None
        vram_percentages = self._get_vram_percentages_for_gpu(model_name, offload_strategy, max_gpu_percent)
        first_vram_pct = vram_percentages[0] if vram_percentages else 0.93
        
        for vram_pct in vram_percentages:
            if self.device != "cuda":
                load_kwargs['device_map'] = None
                print("Loading model in CPU-only mode...")
                model = self._try_load_model(model_name, load_kwargs, self.device)
                if model is not None:
                    break
            
            if vram_pct > 0:
                max_memory = self._get_gpu_memory_map_with_limit(vram_pct)
                load_kwargs['max_memory'] = max_memory
                load_kwargs['device_map'] = 'auto'
                print(f"\nTrying with GPU limit: {vram_pct*100:.0f}% VRAM")
                
                model = self._try_load_model(model_name, load_kwargs, self.device)
                
                if model is not None:
                    print(f"  ✓ Model loaded successfully with {vram_pct*100:.0f}% GPU VRAM limit")
                    if vram_pct < first_vram_pct:
                        print(f"  (Reduced from {first_vram_pct*100:.0f}% due to memory constraints)")
                    break
                else:
                    print(f"  ✗ Out of memory with {vram_pct*100:.0f}% GPU VRAM, trying lower limit...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            else:
                print("\nFalling back to CPU-only mode...")
                load_kwargs['max_memory'] = {0: 0, 'cpu': int((manual_ram_gb or 48) * 1e9)}
                load_kwargs['device_map'] = 'auto'
                model = self._try_load_model(model_name, load_kwargs, "cpu")
                if model is not None:
                    print("  ✓ Model loaded successfully on CPU")
                    break
        
        if model is None:
            raise RuntimeError("Failed to load model: Out of memory even with minimum GPU usage")
        
        self.model = model
        self.model.eval()
        self.model_name = model_name
        
        print(f"\nModel loaded successfully")
        print(f"Model device: {next(self.model.parameters()).device}")
        
        caps = detect_model_capabilities(model_name)
        print(f"Model capabilities: {caps}")
    
    def _get_gpu_memory_map_with_limit(self, vram_fraction: float) -> Dict:
        """Get max_memory dict with specified VRAM fraction limit."""
        import torch
        max_memory = {}
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_vram = props.total_memory
                usable_vram = int(total_vram * vram_fraction)
                max_memory[i] = usable_vram
        
        manual_ram_gb = getattr(self, '_pending_ram_gb', None)
        if manual_ram_gb:
            max_memory['cpu'] = int(manual_ram_gb * 1e9)
        else:
            import psutil
            available_ram = psutil.virtual_memory().available
            usable_ram = max(0, available_ram - int(4e9))
            max_memory['cpu'] = usable_ram
        
        return max_memory
    
    def format_messages(self, messages: List[ChatMessage]) -> str:
        """Format messages into a prompt string."""
        formatted = []
        
        for msg in messages:
            if msg.role == "system":
                formatted.append(f"System: {msg.content}")
            elif msg.role == "user":
                formatted.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                content = msg.content or ""
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        if tc.get("function"):
                            func = tc["function"]
                            content += f'\n<tool>{{"name": "{func.get("name", "")}", "arguments": {func.get("arguments", "{}")}}}</tool>'
                formatted.append(f"Assistant: {content}")
            elif msg.role == "tool":
                formatted.append(f"Tool ({msg.name}): {msg.content}")
        
        formatted.append("Assistant:")
        return "\n\n".join(formatted)
    
    def _validate_params(self, temperature: float, top_p: float):
        """Validate generation parameters."""
        if temperature <= 0:
            temperature = 1.0
            do_sample = False
        else:
            temperature = max(0.01, min(temperature, 2.0))
            do_sample = True
        top_p = max(0.0, min(top_p, 1.0))
        return temperature, top_p, do_sample
    
    def generate(self, prompt: str, max_tokens: Optional[int] = None,
                 temperature: float = 0.7, top_p: float = 1.0,
                 stop: Optional[List[str]] = None) -> str:
        """Generate text non-streaming."""
        import torch
        from transformers import LogitsProcessor, LogitsProcessorList
        
        class InvalidLogitsProcessor(LogitsProcessor):
            def __call__(self, input_ids, scores):
                scores = torch.where(torch.isnan(scores), torch.tensor(-1e9, dtype=scores.dtype, device=scores.device), scores)
                scores = torch.where(torch.isinf(scores), torch.tensor(1e9, dtype=scores.dtype, device=scores.device), scores)
                return scores
        
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        if max_tokens is None:
            max_tokens = 512
        
        temperature, top_p, do_sample = self._validate_params(temperature, top_p)
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_tokens,
                    temperature=temperature if do_sample else None,
                    top_p=top_p if do_sample else None,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    logits_processor=LogitsProcessorList([InvalidLogitsProcessor()]),
                )
            
            generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            error_msg = str(e).lower()
            if "out of memory" in error_msg or "cuda" in error_msg or "oom" in error_msg:
                print(f"Warning: CUDA OOM during generation. Clearing cache and retrying...")
                torch.cuda.empty_cache()
                try:
                    with torch.no_grad():
                        outputs = self.model.generate(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            max_new_tokens=max(1, max_tokens // 2),
                            temperature=temperature if do_sample else None,
                            top_p=top_p if do_sample else None,
                            do_sample=do_sample,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            logits_processor=LogitsProcessorList([InvalidLogitsProcessor()]),
                        )
                    
                    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
                    return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                except Exception as e2:
                    print(f"Error: Generation failed: {e2}")
                    return "[Error: Out of memory during generation]"
            raise
    
    async def generate_stream(self, prompt: str, max_tokens: Optional[int] = None,
                              temperature: float = 0.7, top_p: float = 1.0,
                              stop: Optional[List[str]] = None):
        """Generate text in streaming fashion."""
        import torch
        from transformers import TextIteratorStreamer, LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList
        
        class InvalidLogitsProcessor(LogitsProcessor):
            def __call__(self, input_ids, scores):
                scores = torch.where(torch.isnan(scores), torch.tensor(-1e9, dtype=scores.dtype, device=scores.device), scores)
                scores = torch.where(torch.isinf(scores), torch.tensor(1e9, dtype=scores.dtype, device=scores.device), scores)
                return scores
        
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        if max_tokens is None:
            max_tokens = 512
        
        temperature, top_p, do_sample = self._validate_params(temperature, top_p)
        
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        
        generation_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
            "streamer": streamer,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "logits_processor": LogitsProcessorList([InvalidLogitsProcessor()]),
        }
        
        if stop:
            class StopOnSequence(StoppingCriteria):
                def __init__(self, stop_sequences, tokenizer):
                    self.stop_sequences = stop_sequences
                    self.tokenizer = tokenizer
                
                def __call__(self, input_ids, scores, **kwargs):
                    decoded = self.tokenizer.decode(input_ids[0][-20:], skip_special_tokens=True)
                    return any(seq in decoded for seq in self.stop_sequences)
            
            generation_kwargs["stopping_criteria"] = StoppingCriteriaList([
                StopOnSequence(stop, self.tokenizer)
            ])
        
        generation_error = None
        
        def generate_with_error_handling():
            nonlocal generation_error
            try:
                self.model.generate(**generation_kwargs)
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                error_msg = str(e).lower()
                if "out of memory" in error_msg or "cuda" in error_msg or "oom" in error_msg:
                    generation_error = "oom"
                    print(f"Warning: CUDA OOM during streaming generation...")
                    torch.cuda.empty_cache()
                else:
                    generation_error = str(e)
        
        thread = Thread(target=generate_with_error_handling)
        thread.start()
        
        try:
            for text in streamer:
                yield text
        except Exception as e:
            print(f"Error during stream iteration: {e}")
        
        thread.join()
        
        if generation_error == "oom":
            yield "\n[Warning: Generation stopped due to out-of-memory.]"
        elif generation_error:
            yield f"\n[Error during generation: {generation_error}]"
    
    def get_model_name(self) -> str:
        return self.model_name or "unknown"
    
    def cleanup(self) -> None:
        import torch
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
