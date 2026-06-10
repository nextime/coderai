# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""CUDA backend using HuggingFace Transformers."""

import os
import time as _time
from typing import Optional, List, Dict
from threading import Thread
from abc import ABC

# Import from codai modules
from codai.backends.base import ModelBackend
from codai.models.capabilities import detect_model_capabilities
from codai.pydantic.textrequest import ChatMessage

# Try to import outlines for grammar-guided generation
try:
    from outlines import models, generate
    OUTLINES_AVAILABLE = True
except ImportError:
    OUTLINES_AVAILABLE = False
    models = None
    generate = None

# Import global flag from coderai (will be None if not running as server)
try:
    import coderai
    _grammar_guided_gen = getattr(coderai, 'grammar_guided_gen', False)
except (ImportError, AttributeError):
    _grammar_guided_gen = False


def _make_thermal_criteria():
    """A StoppingCriteria that pauses generation while the CPU/GPU is too hot.

    It runs ON the generation thread (between token forward passes), so blocking
    here actually pauses GPU work — unlike the streamer consumer loop, which is
    decoupled. Returns False so it never ends generation; throttled so it doesn't
    read sensors on every token. Returns None if transformers is unavailable.
    """
    try:
        from transformers import StoppingCriteria
    except Exception:
        return None

    class _ThermalPause(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            try:
                from codai.models.thermal import checkpoint
                checkpoint(context="text-gen", throttle_seconds=2.0)
            except Exception:
                pass
            return False

    return _ThermalPause()


class NvidiaBackend(ModelBackend):
    """Backend for NVIDIA GPUs using HuggingFace Transformers."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.device = None
        self.use_flash_attn = False
        self.flash_attn_available = False
        # KV prefix cache (single-entry, keyed by formatted prefix text)
        self._kv_prefix_text: Optional[str] = None
        self._kv_past_key_values = None   # past_key_values tensor tuple
        self._kv_prefix_len: int = 0      # token count of the cached prefix
        self._kv_timestamp: float = 0.0
        self._kv_ttl: float = 300.0       # 5 min TTL
        self._last_usage: Dict = {}
        
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
    
    def _make_bnb_config(self, model_name: str, load_in_4bit: bool, load_in_8bit: bool):
        """Build a transformers BitsAndBytesConfig (the modern quant API).

        Passing load_in_4bit/load_in_8bit as direct from_pretrained kwargs is
        removed in recent transformers and raises TypeError — which previously
        forced a silent fallback to FULL-PRECISION loading (the model then no
        longer fit on the GPU, offloaded to CPU, and leaked VRAM on eviction).
        Always go through quantization_config instead.
        """
        ml = model_name.lower()
        if 'qwen3.5' in ml and ('a3b' in ml or 'moe' in ml):
            print(f"Warning: {model_name} does not support bitsandbytes quantization")
            return None
        try:
            import bitsandbytes as bnb  # noqa: F401
            import torch
            from transformers import BitsAndBytesConfig
        except ImportError:
            print("Warning: bitsandbytes not installed. Quantization disabled.")
            return None
        print(f"Using {4 if load_in_4bit else 8}-bit quantization")
        if load_in_4bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        return BitsAndBytesConfig(load_in_8bit=True)

    def _is_moe_model(self, model_name: str) -> bool:
        """Check if model is a MoE model."""
        moe_indicators = ['moe', 'mixtral', 'qwen3_5_moe', 'qwen3.5_moe', 'expert', 'a3b']
        model_name_lower = model_name.lower()
        return any(indicator in model_name_lower for indicator in moe_indicators)
    
    def _get_vram_percentages_for_strategy(self, strategy: str, is_moe: bool, total_vram_gb: float) -> list:
        """Get VRAM percentage steps based on offload strategy."""
        if strategy == "none":
            print(f"  Offload strategy 'none': disabling CPU offload and VRAM auto-detection")
            return None  # Signal to skip offloading entirely
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
        """Get VRAM percentage steps based on GPU memory size.
        
        Returns None when strategy is 'none' (no offloading).
        """
        import torch
        
        if strategy == "none":
            return None  # Signal to skip offloading entirely
        
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
    
    def _derive_cuda_device(self) -> str:
        """Derive the CUDA device string from global args.
        
        Checks --vulkan-device (reused as generic GPU device ID) to determine
        which CUDA device to target. Defaults to 'cuda:0'.
        """
        try:
            from codai.api.state import get_global_args
            _global_args = get_global_args()
            if _global_args:
                # Use vulkan-device as a generic GPU device selector
                device_id = getattr(_global_args, 'vulkan_device', 0)
                if device_id is not None and device_id != 0:
                    return f"cuda:{device_id}"
        except Exception:
            pass
        return "cuda:0"
    
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
        expected_vram_gb = kwargs.get('expected_vram_gb') or 0

        # Check for --no-ram mode
        no_ram = kwargs.get('no_ram', False)
        if not no_ram:
            try:
                from codai.api.state import get_global_args
                _global_args = get_global_args()
                if _global_args and getattr(_global_args, 'no_ram', False):
                    no_ram = True
            except Exception:
                pass

        self._pending_ram_gb = manual_ram_gb

        print(f"Loading HuggingFace model: {model_name}")

        # Flash-Attention-2 requires the ENTIRE model resident on a single CUDA
        # device.  If the model will be split across GPU+CPU (offloading), FA2
        # triggers a device-side assert that corrupts the whole CUDA context.
        # So FA2 is only safe when the model fits fully in free GPU VRAM, or the
        # user forced full-GPU residence (no_ram / offload_strategy='none').
        self._fa2_safe = True
        if flash_attn:
            _full_gpu_forced = no_ram or offload_strategy == 'none'
            if not _full_gpu_forced:
                try:
                    import torch as _t
                    if _t.cuda.is_available() and expected_vram_gb > 0:
                        _free, _ = _t.cuda.mem_get_info(0)
                        _free_gb = _free / 1e9
                        # expected_vram_gb already includes ~15% overhead; the
                        # model must fit entirely on GPU for FA2 to be safe.
                        if expected_vram_gb > _free_gb:
                            self._fa2_safe = False
                            print(f"  Flash Attention 2 disabled: model needs "
                                  f"~{expected_vram_gb:.1f} GB but only {_free_gb:.1f} GB "
                                  f"GPU free → will offload to CPU (FA2 needs full-GPU "
                                  f"residence). Using SDPA instead.")
                except Exception:
                    pass

        self.use_flash_attn = flash_attn and self._fa2_safe
        self.check_flash_attn_support()
        
        self.device = self._detect_device()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # =====================================================================
        # --no-ram mode: maximize VRAM, no CPU RAM spilling
        # =====================================================================
        if no_ram and self.device == "cuda":
            cuda_device = self._derive_cuda_device()
            print(f"--no-ram mode: loading model directly on {cuda_device}")
            print(f"  device_map={cuda_device}, low_cpu_mem_usage=True, torch_dtype=auto")
            
            load_kwargs = {
                'trust_remote_code': True,
                'device_map': cuda_device,
                'low_cpu_mem_usage': True,
                'torch_dtype': "auto",
            }
            
            if self.use_flash_attn and self.flash_attn_available:
                load_kwargs['attn_implementation'] = "flash_attention_2"
                print("  Using Flash Attention 2")
            
            # Still allow quantization in no-ram mode (reduces VRAM usage)
            if load_in_4bit or load_in_8bit:
                _qc = self._make_bnb_config(model_name, load_in_4bit, load_in_8bit)
                if _qc is not None:
                    load_kwargs['quantization_config'] = _qc
            
            try:
                model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
                self.model = model
                self.model.eval()
                self.model_name = model_name
                
                print(f"\n--no-ram: Model loaded successfully on {cuda_device}")
                print(f"Model device: {next(self.model.parameters()).device}")
                
                caps = detect_model_capabilities(model_name)
                print(f"Model capabilities: {caps}")
                return
            except Exception as e:
                print(f"--no-ram: Failed to load model on {cuda_device}: {e}")
                raise RuntimeError(
                    f"--no-ram: Failed to load model entirely on GPU ({cuda_device}). "
                    f"The model may be too large for available VRAM. Error: {e}"
                )
        
        # =====================================================================
        # Standard loading path (with OOM fallback)
        # =====================================================================
        load_kwargs = {'trust_remote_code': True}
        
        if load_in_4bit or load_in_8bit:
            _qc = self._make_bnb_config(model_name, load_in_4bit, load_in_8bit)
            if _qc is not None:
                load_kwargs['quantization_config'] = _qc

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
        else:
            # SDPA safely handles GPU+CPU split models and still uses flash
            # kernels for the GPU-resident layers — the safe default when the
            # model is offloaded (FA2 would device-side-assert here).
            load_kwargs['attn_implementation'] = "sdpa"

        model = None
        vram_percentages = self._get_vram_percentages_for_gpu(model_name, offload_strategy, max_gpu_percent)
        
        # --offload-strategy none: load directly on GPU without offloading or VRAM limits
        if vram_percentages is None:
            cuda_device = self._derive_cuda_device()
            print(f"\nOffload strategy 'none': loading model directly on {cuda_device} (no CPU offload, no VRAM limits)")
            load_kwargs['device_map'] = cuda_device
            load_kwargs['low_cpu_mem_usage'] = True
            load_kwargs['torch_dtype'] = "auto"
            # Remove dtype set earlier since torch_dtype=auto takes precedence
            load_kwargs.pop('dtype', None)
            
            try:
                model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
            except Exception as e:
                raise RuntimeError(
                    f"--offload-strategy none: Failed to load model entirely on GPU ({cuda_device}). "
                    f"The model may be too large for available VRAM. Error: {e}"
                )
        else:
            first_vram_pct = vram_percentages[0] if vram_percentages else 0.93

            for vram_pct in vram_percentages:
                if self.device != "cuda":
                    # No CUDA device — go straight to CPU+disk loading below.
                    break

                if vram_pct > 0:
                    # Build max_memory: GPU budget capped at actual FREE VRAM so
                    # we never try to allocate more than what's physically available.
                    # Excess layers overflow to CPU RAM automatically via device_map.
                    max_memory = self._get_gpu_memory_map_with_limit(vram_pct)
                    load_kwargs['max_memory'] = max_memory
                    load_kwargs['device_map'] = 'auto'
                    _gpu_gb = max_memory.get(0, 0) / 1e9
                    _cpu_gb = max_memory.get('cpu', 0) / 1e9
                    print(f"\nTrying GPU {_gpu_gb:.1f} GB + CPU {_cpu_gb:.1f} GB"
                          f" (device_map=auto, {vram_pct*100:.0f}% VRAM cap)")

                    model = self._try_load_model(model_name, load_kwargs, self.device)

                    if model is not None:
                        print(f"  ✓ Model loaded — GPU {_gpu_gb:.1f} GB / CPU {_cpu_gb:.1f} GB")
                        if vram_pct < first_vram_pct:
                            print(f"  (Reduced GPU cap from {first_vram_pct*100:.0f}%"
                                  f" due to memory constraints)")
                        break
                    else:
                        print(f"  ✗ OOM at GPU {_gpu_gb:.1f} GB, trying lower GPU cap…")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                else:
                    # vram_pct == 0: GPU (all free VRAM) + CPU RAM + disk overflow.
                    # Use every byte of GPU that's free, then spill to CPU RAM, then
                    # disk — NEVER leave GPU idle when loading this fallback level.
                    import psutil as _psutil
                    _free_vram = 0
                    if torch.cuda.is_available():
                        try:
                            _free_vram, _ = torch.cuda.mem_get_info(0)
                        except Exception:
                            pass
                    _headroom = 512 * 1024 * 1024
                    _gpu_budget = max(0, _free_vram - _headroom)
                    _free_ram = _psutil.virtual_memory().available
                    _cpu_budget = max(int(2e9), int(_free_ram * 0.80))
                    _disk_dir = offload_dir or os.path.join(
                        os.path.expanduser('~'), '.cache', 'coderai', 'offload')
                    os.makedirs(_disk_dir, exist_ok=True)
                    print(f"\nGPU {_gpu_budget/1e9:.1f} GB + CPU {_cpu_budget/1e9:.1f} GB"
                          f" + disk ({_disk_dir})")
                    _spill_kwargs = {
                        **load_kwargs,
                        'device_map': 'auto',
                        'max_memory': {0: _gpu_budget, 'cpu': _cpu_budget},
                        'offload_folder': _disk_dir,
                        'offload_buffers': True,
                    }
                    model = self._try_load_model(model_name, _spill_kwargs, self.device)
                    if model is not None:
                        print(f"  ✓ Model loaded — GPU {_gpu_budget/1e9:.1f} GB"
                              f" / CPU {_cpu_budget/1e9:.1f} GB / disk overflow")
                        break

            # Absolute last resort: pure CPU without device_map.
            # Only reached when CUDA is unavailable or all GPU+RAM+disk paths failed.
            # Uses device_map=None to avoid accelerate hooks that assume CUDA.
            if model is None:
                print("\nFalling back to pure CPU (no GPU available)…")
                cpu_kwargs = {
                    'trust_remote_code': True,
                    'torch_dtype': torch.float32,
                    'low_cpu_mem_usage': True,
                }
                if offload_dir:
                    cpu_kwargs['offload_folder'] = offload_dir
                if self.use_flash_attn and self.flash_attn_available:
                    cpu_kwargs['attn_implementation'] = "flash_attention_2"
                model = self._try_load_model(model_name, cpu_kwargs, "cpu")
                if model is not None:
                    print("  ✓ Model loaded on CPU (no GPU)")
        
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
        """Get max_memory dict for device_map='auto'.

        GPU budget = min(total × fraction, free − 512 MB headroom).
        Capping at free VRAM ensures we never ask accelerate to allocate more
        than what's physically available; layers that exceed the GPU budget
        spill to CPU RAM automatically via device_map.
        """
        import torch
        max_memory = {}

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_vram = props.total_memory
                try:
                    free_vram, _ = torch.cuda.mem_get_info(i)
                except Exception:
                    free_vram = total_vram
                headroom = 512 * 1024 * 1024  # 512 MB for CUDA driver overhead
                limit_by_fraction = int(total_vram * vram_fraction)
                limit_by_free     = max(0, free_vram - headroom)
                max_memory[i] = min(limit_by_fraction, limit_by_free)

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
                 stop: Optional[List[str]] = None,
                 grammar: Optional[str] = None,
                 repeat_penalty: float = 1.0,
                 presence_penalty: float = 0.0,
                 frequency_penalty: float = 0.0) -> str:
        """Generate text non-streaming.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            stop: Stop sequences
            grammar: Optional regex pattern for constrained generation (outlines)
            repeat_penalty: Repetition penalty (1.0 = no penalty)
            presence_penalty: Presence penalty
            frequency_penalty: Frequency penalty
        """
        import torch
        from transformers import LogitsProcessor, LogitsProcessorList
        
        # Check for grammar-guided generation using outlines
        use_grammar = grammar
        if use_grammar is None:
            # Check global flag
            try:
                import coderai
                if getattr(coderai, 'grammar_guided_gen', False):
                    if not OUTLINES_AVAILABLE:
                        print("Warning: outlines not installed. Run: pip install outlines")
                        use_grammar = None
                    else:
                        # Use a regex pattern for tool calls
                        use_grammar = r'<tool>.*?</tool>|\{.*?"name".*?"arguments".*?\}|\[.*?"name".*?"arguments".*?\]'
            except (ImportError, AttributeError):
                pass
        
        # If outlines is available and grammar is enabled, use outlines
        if use_grammar and OUTLINES_AVAILABLE:
            try:
                return self._generate_with_outlines(prompt, max_tokens, temperature, top_p, stop, use_grammar)
            except Exception as e:
                print(f"Warning: Outlines generation failed: {e}, falling back to normal generation")
                # Fall through to normal generation
        
        # Normal generation without grammar
        return self._generate_normal(prompt, max_tokens, temperature, top_p, stop, repeat_penalty, presence_penalty, frequency_penalty)
    
    def _generate_with_outlines(self, prompt: str, max_tokens: Optional[int],
                                 temperature: float, top_p: float,
                                 stop: Optional[List[str]],
                                 pattern: str) -> str:
        """Generate text using outlines library for grammar-guided generation."""
        if max_tokens is None:
            max_tokens = 512
        
        # Create outlines model from the loaded model
        model = models.Transformers(self.model, self.tokenizer)
        
        # Create regex generator
        regex_generator = generate.regex(model, pattern=pattern)
        
        # Generate with outlines
        # Note: outlines uses its own sampling parameters
        result = regex_generator(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 0.7,
        )
        
        # Extract the generated text (outlines returns the full output)
        if isinstance(result, str):
            # Remove the prompt from the result
            if result.startswith(prompt):
                result = result[len(prompt):]
            return result
        return str(result)
    
    def _generate_normal(self, prompt: str, max_tokens: Optional[int],
                        temperature: float, top_p: float,
                        stop: Optional[List[str]],
                        repeat_penalty: float = 1.0,
                        presence_penalty: float = 0.0,
                        frequency_penalty: float = 0.0) -> str:
        """Normal generation without grammar constraints."""
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
        
        # Build generation kwargs with penalty parameters
        gen_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "max_new_tokens": max_tokens,
            "temperature": temperature if do_sample else None,
            "top_p": top_p if do_sample else None,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "logits_processor": LogitsProcessorList([InvalidLogitsProcessor()]),
        }
        
        # Add repetition penalty if not 1.0
        if repeat_penalty != 1.0:
            gen_kwargs["repetition_penalty"] = repeat_penalty
        
        # Add presence and frequency penalties if not 0.0
        if presence_penalty != 0.0 or frequency_penalty != 0.0:
            # Transformers uses repetition_penalty for both presence and frequency
            # For models that support both, we use the more general repetition_penalty
            if repeat_penalty == 1.0:
                # If no repetition_penalty set, use presence_penalty as repetition_penalty
                # (this is an approximation - models may handle these differently)
                gen_kwargs["repetition_penalty"] = max(presence_penalty, frequency_penalty) if max(presence_penalty, frequency_penalty) > 1.0 else 1.0
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(**gen_kwargs)
            
            generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            error_msg = str(e).lower()
            if "out of memory" in error_msg or "cuda" in error_msg or "oom" in error_msg:
                print(f"Warning: CUDA OOM during generation. Clearing cache and retrying...")
                torch.cuda.empty_cache()
                try:
                    with torch.no_grad():
                        outputs = self.model.generate(**gen_kwargs)
                    
                    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
                    return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                except Exception as e2:
                    print(f"Error: Generation failed: {e2}")
                    return "[Error: Out of memory during generation]"
            raise
    
    async def generate_stream(self, prompt: str, max_tokens: Optional[int] = None,
                              temperature: float = 0.7, top_p: float = 1.0,
                              stop: Optional[List[str]] = None,
                              grammar: Optional[str] = None,
                              repeat_penalty: float = 1.0,
                              presence_penalty: float = 0.0,
                              frequency_penalty: float = 0.0):
        """Generate text in streaming fashion.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            stop: Stop sequences
            grammar: Optional regex pattern for constrained generation (outlines)
            repeat_penalty: Repetition penalty (1.0 = no penalty)
            presence_penalty: Presence penalty
            frequency_penalty: Frequency penalty
        """
        # Check for grammar-guided generation using outlines
        use_grammar = grammar
        if use_grammar is None:
            # Check global flag
            try:
                import coderai
                if getattr(coderai, 'grammar_guided_gen', False):
                    if not OUTLINES_AVAILABLE:
                        print("Warning: outlines not installed. Run: pip install outlines")
                        use_grammar = None
                    else:
                        # Use a regex pattern for tool calls
                        use_grammar = r'<tool>.*?</tool>|\{.*?"name".*?"arguments".*?\}|\[.*?"name".*?"arguments".*?\]'
            except (ImportError, AttributeError):
                pass
        
        # If outlines is available and grammar is enabled, use outlines
        if use_grammar and OUTLINES_AVAILABLE:
            try:
                async for chunk in self._generate_stream_outlines(prompt, max_tokens, temperature, top_p, stop, use_grammar):
                    yield chunk
                return
            except Exception as e:
                print(f"Warning: Outlines streaming generation failed: {e}, falling back to normal generation")
                # Fall through to normal generation
        
        # Normal streaming generation without grammar
        async for chunk in self._generate_stream_normal(prompt, max_tokens, temperature, top_p, stop, repeat_penalty, presence_penalty, frequency_penalty):
            yield chunk
    
    async def _generate_stream_outlines(self, prompt: str, max_tokens: Optional[int],
                                        temperature: float, top_p: float,
                                        stop: Optional[List[str]],
                                        pattern: str):
        """Generate text using outlines library in streaming mode."""
        if max_tokens is None:
            max_tokens = 512
        
        # Create outlines model from the loaded model
        model = models.Transformers(self.model, self.tokenizer)
        
        # Create regex generator
        regex_generator = generate.regex(model, pattern=pattern)
        
        # Generate with outlines (outlines doesn't support true streaming, so we yield the result)
        result = regex_generator(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 0.7,
        )
        
        # Extract the generated text (outlines returns the full output)
        if isinstance(result, str):
            # Remove the prompt from the result
            if result.startswith(prompt):
                result = result[len(prompt):]
            # Yield the entire result as a single chunk (outlines doesn't support true streaming)
            yield result
        else:
            yield str(result)
    
    async def _generate_stream_normal(self, prompt: str, max_tokens: Optional[int],
                                     temperature: float, top_p: float,
                                     stop: Optional[List[str]],
                                     repeat_penalty: float = 1.0,
                                     presence_penalty: float = 0.0,
                                     frequency_penalty: float = 0.0):
        """Normal streaming generation without grammar constraints."""
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
        
        # Add repetition penalty if not 1.0
        if repeat_penalty != 1.0:
            generation_kwargs["repetition_penalty"] = repeat_penalty
        
        # Mid-generation thermal checkpoint (runs on the generate thread).
        _criteria = []
        _therm = _make_thermal_criteria()
        if _therm is not None:
            _criteria.append(_therm)
        if stop:
            class StopOnSequence(StoppingCriteria):
                def __init__(self, stop_sequences, tokenizer):
                    self.stop_sequences = stop_sequences
                    self.tokenizer = tokenizer

                def __call__(self, input_ids, scores, **kwargs):
                    decoded = self.tokenizer.decode(input_ids[0][-20:], skip_special_tokens=True)
                    return any(seq in decoded for seq in self.stop_sequences)

            _criteria.append(StopOnSequence(stop, self.tokenizer))
        if _criteria:
            generation_kwargs["stopping_criteria"] = StoppingCriteriaList(_criteria)
        
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
    
    # ------------------------------------------------------------------
    # KV prefix cache helpers
    # ------------------------------------------------------------------

    def _kv_cache_valid(self) -> bool:
        return (
            self._kv_past_key_values is not None and
            _time.time() - self._kv_timestamp < self._kv_ttl
        )

    def _model_on_cuda(self) -> bool:
        """Return True only when the model's first parameter is actually on a CUDA device."""
        try:
            return next(self.model.parameters()).is_cuda
        except StopIteration:
            return False

    def _build_kv_prefix(self, prefix_text: str):
        """Forward-pass on prefix_text to populate the KV state."""
        import torch
        # KV prefix caching requires CUDA tensors; skip on CPU-mode models.
        if not self._model_on_cuda():
            raise RuntimeError("KV prefix cache requires CUDA; model is on CPU")
        inputs = self.tokenizer(
            prefix_text, return_tensors="pt", add_special_tokens=False
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model(**inputs, use_cache=True, return_dict=True)
        return out.past_key_values, int(inputs['input_ids'].shape[1])

    def _store_kv(self, prefix_text: str, past_kv, prefix_len: int) -> None:
        self._kv_prefix_text = prefix_text
        self._kv_past_key_values = past_kv
        self._kv_prefix_len = prefix_len
        self._kv_timestamp = _time.time()

    def invalidate_kv_cache(self) -> None:
        """Discard the cached KV state (call on model unload/swap)."""
        self._kv_prefix_text = None
        if self._kv_past_key_values is not None:
            del self._kv_past_key_values
        self._kv_past_key_values = None
        self._kv_prefix_len = 0
        self._kv_timestamp = 0.0

    # ------------------------------------------------------------------
    # Usage tracking
    # ------------------------------------------------------------------

    def get_last_usage(self) -> dict:
        return dict(self._last_usage)

    # ------------------------------------------------------------------
    # Chat-level generation (with KV prefix caching)
    # ------------------------------------------------------------------

    def _format_messages_to_str(self, messages) -> str:
        """Convert a list of message dicts to a formatted prompt string."""
        from codai.pydantic.textrequest import ChatMessage
        chat_msgs = [
            ChatMessage(**m) if isinstance(m, dict) else m
            for m in messages
        ]
        return self.format_messages(chat_msgs)

    def _eos_token_ids(self):
        """All token ids that should END generation — including the chat turn
        boundary.  Qwen's turn ends with <|im_end|>, but tokenizer.eos_token_id is
        <|endoftext|>; without im_end the model never stops and hallucinates extra
        'assistant'/'user' turns.  Returns a list (HF generate accepts a list)."""
        ids = set()
        try:
            if self.tokenizer.eos_token_id is not None:
                ids.add(int(self.tokenizer.eos_token_id))
        except Exception:
            pass
        for tok in ('<|im_end|>', '<|eot_id|>', '<|end|>', '<|endoftext|>',
                    '<|end_of_text|>', '<end_of_turn>'):
            try:
                tid = self.tokenizer.convert_tokens_to_ids(tok)
                if isinstance(tid, int) and tid >= 0 and tid != getattr(
                        self.tokenizer, 'unk_token_id', None):
                    ids.add(tid)
            except Exception:
                pass
        return list(ids) if ids else self.tokenizer.eos_token_id

    def _build_chat_prompt(self, messages, enable_thinking: bool = False,
                           add_generation_prompt: bool = True) -> str:
        """Build the prompt string using the MODEL's own chat template when it has
        one (correct special tokens + proper `enable_thinking` handling for Qwen3).
        Falls back to the legacy custom formatter when no template is available.

        `enable_thinking=True` keeps reasoning <think> blocks available for callers
        that ask for them; `False` (default) suppresses them via the template.
        """
        tmpl = getattr(self.tokenizer, 'chat_template', None)
        if tmpl:
            # Normalise to plain {role, content} dicts for apply_chat_template.
            norm = []
            for m in messages:
                if isinstance(m, dict):
                    norm.append({'role': m.get('role'), 'content': m.get('content') or ''})
                else:
                    norm.append({'role': getattr(m, 'role', None),
                                 'content': getattr(m, 'content', '') or ''})
            try:
                return self.tokenizer.apply_chat_template(
                    norm, tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                    enable_thinking=enable_thinking)
            except TypeError:
                # Tokenizer's template doesn't accept enable_thinking — use plain.
                try:
                    return self.tokenizer.apply_chat_template(
                        norm, tokenize=False,
                        add_generation_prompt=add_generation_prompt)
                except Exception:
                    pass
            except Exception:
                pass
        return self._format_messages_to_str(messages)

    def generate_chat(self, messages, max_tokens=None, temperature=0.7,
                      top_p=1.0, stop=None, tools=None, response_format=None,
                      enable_thinking=False) -> str:
        """
        Non-streaming chat generation with KV prefix caching.

        Detects when the current request shares a system-prompt / history
        prefix with the previous request and reuses the cached KV state,
        only encoding the new suffix tokens.
        """
        import torch
        if max_tokens is None:
            max_tokens = 512

        full_prompt = self._build_chat_prompt(messages, enable_thinking=enable_thinking,
                                              add_generation_prompt=True)
        total_input_ids = self.tokenizer(full_prompt, return_tensors="pt")['input_ids']
        total_prompt_len = int(total_input_ids.shape[1])

        # Build prefix text (all turns except the final user turn)
        prefix_msgs = (
            messages[:-1]
            if messages and messages[-1].get('role') == 'user'
            else []
        )

        past_kv = None
        cached_len = 0

        if prefix_msgs and self._model_on_cuda():
            prefix_text = self._build_chat_prompt(
                prefix_msgs, enable_thinking=enable_thinking, add_generation_prompt=False)
            if self._kv_cache_valid() and self._kv_prefix_text == prefix_text:
                past_kv = self._kv_past_key_values
                cached_len = self._kv_prefix_len
            else:
                try:
                    past_kv, cached_len = self._build_kv_prefix(prefix_text)
                    self._store_kv(prefix_text, past_kv, cached_len)
                except Exception as e:
                    print(f"Warning: KV prefix cache build failed: {e}")
                    past_kv, cached_len = None, 0

        temperature, top_p, do_sample = self._validate_params(temperature, top_p)
        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self._eos_token_ids(),
            use_cache=True,
        )
        # Mid-generation thermal checkpoint (runs on the generate thread, so it
        # pauses GPU work between tokens when the CPU/GPU is too hot).
        _therm = _make_thermal_criteria()
        if _therm is not None:
            from transformers import StoppingCriteriaList
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList([_therm])

        generated_text = ""
        try:
            total_input_ids = total_input_ids.to(self.model.device)
            if past_kv is not None and 0 < cached_len < total_prompt_len:
                suffix_ids = total_input_ids[:, cached_len:]
                full_attn = torch.ones(
                    1, total_prompt_len, dtype=torch.long, device=self.model.device
                )
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=suffix_ids,
                        past_key_values=past_kv,
                        attention_mask=full_attn,
                        **gen_kwargs,
                    )
                new_tokens = outputs[0][suffix_ids.shape[1]:]
            else:
                cached_len = 0
                attn_mask = torch.ones_like(total_input_ids)
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=total_input_ids,
                        attention_mask=attn_mask,
                        **gen_kwargs,
                    )
                new_tokens = outputs[0][total_prompt_len:]
            generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        except Exception as e:
            print(f"Warning: KV-cached generate_chat failed ({e}), retrying without cache")
            self.invalidate_kv_cache()
            cached_len = 0
            # Determine if the error is a CUDA device-placement issue; if so, also
            # disable the internal KV cache which accumulates mixed-device tensors.
            _is_device_error = "is_cuda" in str(e) or "device" in str(e).lower()
            _fallback_kwargs = {**gen_kwargs, 'use_cache': not _is_device_error}
            try:
                total_input_ids = self.tokenizer(
                    full_prompt, return_tensors="pt"
                )['input_ids'].to(self.model.device)
                attn_mask = torch.ones_like(total_input_ids)
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=total_input_ids,
                        attention_mask=attn_mask,
                        **_fallback_kwargs,
                    )
                new_tokens = outputs[0][total_prompt_len:]
                generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            except Exception as e2:
                print(f"Error: generate_chat fallback failed: {e2}")
                # Last resort: disable internal KV cache entirely
                if _fallback_kwargs.get('use_cache', True):
                    try:
                        no_cache_kwargs = {**gen_kwargs, 'use_cache': False}
                        total_input_ids = self.tokenizer(
                            full_prompt, return_tensors="pt"
                        )['input_ids'].to(self.model.device)
                        attn_mask = torch.ones_like(total_input_ids)
                        with torch.no_grad():
                            outputs = self.model.generate(
                                input_ids=total_input_ids,
                                attention_mask=attn_mask,
                                **no_cache_kwargs,
                            )
                        new_tokens = outputs[0][total_prompt_len:]
                        generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                        print("generate_chat: recovered with use_cache=False")
                    except Exception as e3:
                        print(f"Error: generate_chat no-cache fallback failed: {e3}")
                        generated_text = ""
                else:
                    generated_text = ""

        try:
            comp_len = len(self.tokenizer.encode(generated_text)) if generated_text else 0
        except Exception:
            comp_len = len(generated_text.split())

        self._last_usage = {
            'prompt_tokens': total_prompt_len,
            'completion_tokens': comp_len,
            'cached_tokens': cached_len,
        }
        return generated_text

    async def generate_chat_stream(self, messages, max_tokens=None,
                                   temperature=0.7, top_p=1.0, stop=None,
                                   tools=None, response_format=None,
                                   enable_thinking=False):
        """
        Streaming chat generation with KV prefix caching.
        Uses the same prefix-cache strategy as generate_chat.
        """
        import torch
        from transformers import TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
        from threading import Thread

        if max_tokens is None:
            max_tokens = 512

        full_prompt = self._build_chat_prompt(messages, enable_thinking=enable_thinking,
                                              add_generation_prompt=True)
        total_input_ids = self.tokenizer(full_prompt, return_tensors="pt")['input_ids']
        total_prompt_len = int(total_input_ids.shape[1])

        prefix_msgs = (
            messages[:-1]
            if messages and messages[-1].get('role') == 'user'
            else []
        )
        past_kv = None
        cached_len = 0

        if prefix_msgs and self._model_on_cuda():
            prefix_text = self._build_chat_prompt(
                prefix_msgs, enable_thinking=enable_thinking, add_generation_prompt=False)
            if self._kv_cache_valid() and self._kv_prefix_text == prefix_text:
                past_kv = self._kv_past_key_values
                cached_len = self._kv_prefix_len
            else:
                try:
                    past_kv, cached_len = self._build_kv_prefix(prefix_text)
                    self._store_kv(prefix_text, past_kv, cached_len)
                except Exception as e:
                    print(f"Warning: KV prefix cache build failed (stream): {e}")
                    past_kv, cached_len = None, 0

        temperature, top_p, do_sample = self._validate_params(temperature, top_p)

        total_input_ids = total_input_ids.to(self.model.device)
        if past_kv is not None and 0 < cached_len < total_prompt_len:
            gen_input_ids = total_input_ids[:, cached_len:]
            full_attn = torch.ones(
                1, total_prompt_len, dtype=torch.long, device=self.model.device
            )
            extra_gen = {'past_key_values': past_kv, 'attention_mask': full_attn}
        else:
            cached_len = 0
            gen_input_ids = total_input_ids
            extra_gen = {'attention_mask': torch.ones_like(total_input_ids)}

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        gen_kwargs = dict(
            input_ids=gen_input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            do_sample=do_sample,
            streamer=streamer,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self._eos_token_ids(),
            use_cache=True,
            **extra_gen,
        )

        # Mid-generation thermal checkpoint (runs on the generate thread).
        _criteria = []
        _therm = _make_thermal_criteria()
        if _therm is not None:
            _criteria.append(_therm)
        if stop:
            class _StopOnSeq(StoppingCriteria):
                def __init__(self, seqs, tok):
                    self.seqs = seqs
                    self.tok = tok
                def __call__(self, input_ids, scores, **kw):
                    decoded = self.tok.decode(input_ids[0][-20:], skip_special_tokens=True)
                    return any(s in decoded for s in self.seqs)
            _criteria.append(_StopOnSeq(stop, self.tokenizer))
        if _criteria:
            gen_kwargs['stopping_criteria'] = StoppingCriteriaList(_criteria)

        gen_error = [None]
        comp_tokens = [0]

        def _run():
            try:
                with torch.no_grad():
                    self.model.generate(**gen_kwargs)
            except Exception as e:
                gen_error[0] = str(e)

        thread = Thread(target=_run)
        thread.start()

        try:
            for text in streamer:
                comp_tokens[0] += 1
                yield text
        except Exception as e:
            print(f"Error during KV-cached stream iteration: {e}")
        finally:
            thread.join()
            self._last_usage = {
                'prompt_tokens': total_prompt_len,
                'completion_tokens': comp_tokens[0],
                'cached_tokens': cached_len,
            }

        if gen_error[0]:
            print(f"Warning: KV-cached stream generation error: {gen_error[0]}")
            self.invalidate_kv_cache()

    def get_model_name(self) -> str:
        return self.model_name or "unknown"
    
    def cleanup(self) -> None:
        import torch, gc
        try:
            from codai.api.state import get_global_debug
            _dbg = bool(get_global_debug())
        except Exception:
            _dbg = False

        def _vram_gb():
            try:
                if torch.cuda.is_available():
                    free, total = torch.cuda.mem_get_info()
                    return (total - free) / 1e9
            except Exception:
                pass
            return -1.0

        def _cuda_param_gb():
            tot = 0
            try:
                for p in self.model.parameters():
                    if p.data.is_cuda:
                        tot += p.data.numel() * p.data.element_size()
                for b in self.model.buffers():
                    if b.data.is_cuda:
                        tot += b.data.numel() * b.data.element_size()
            except Exception:
                pass
            return tot / 1e9

        _v0 = _vram_gb()
        self.invalidate_kv_cache()
        if self.model is not None:
            _pg0 = _cuda_param_gb() if _dbg else 0.0
            # Record the GPU storage pointers of THIS model's tensors so we can,
            # after moving them to CPU, break any lingering external references
            # (e.g. accelerate's tied_params_map, which keeps tied embedding /
            # lm_head weights alive on the GPU and fragments the allocator so
            # empty_cache() can't release the surrounding memory).  Scoped by
            # data_ptr so we never touch a different (coexisting) model.
            _orig_cuda_ptrs = set()
            try:
                for _p in self.model.parameters():
                    if _p.data.is_cuda:
                        _orig_cuda_ptrs.add(_p.data.untyped_storage().data_ptr())
                for _b in self.model.buffers():
                    if _b.data.is_cuda:
                        _orig_cuda_ptrs.add(_b.data.untyped_storage().data_ptr())
            except Exception:
                pass

            # Strip accelerate dispatch hooks AND their offload bookkeeping, which
            # hold references to the original CUDA tensors.  Must happen before we
            # move tensors, or the hooks keep the GPU copies alive.
            try:
                from accelerate.hooks import remove_hook_from_submodules
                remove_hook_from_submodules(self.model)
            except Exception:
                pass
            # Walk every submodule and move its raw _parameters/_buffers storage to
            # CPU directly.  This reaches tensors that model.parameters() may skip
            # (e.g. when wrapped by accelerate) and does NOT rely on model.to('cpu'),
            # which is a silent no-op on dispatched models.
            try:
                import torch as _t
                for _mod in self.model.modules():
                    for _d in (_mod._parameters, _mod._buffers):
                        for _name, _t_obj in list(_d.items()):
                            if _t_obj is None:
                                continue
                            try:
                                if getattr(_t_obj, 'is_cuda', False):
                                    _d[_name] = _t_obj.to('cpu')
                                # accelerate stores params as nn.Parameter; keep type
                                elif hasattr(_t_obj, 'data') and getattr(_t_obj.data, 'is_cuda', False):
                                    _t_obj.data = _t_obj.data.to('cpu')
                            except Exception:
                                pass
                    # Drop per-module accelerate hook state that pins CUDA tensors.
                    for _attr in ('_hf_hook', '_old_forward'):
                        if hasattr(_mod, _attr):
                            try:
                                delattr(_mod, _attr)
                            except Exception:
                                pass
            except Exception as e:
                print(f"  cleanup: module-walk move issue: {e}")
            for _attr in ('hf_device_map', '_hf_hook'):
                try:
                    if hasattr(self.model, _attr):
                        delattr(self.model, _attr)
                except Exception:
                    pass
            if _dbg:
                print(f"  cleanup: CUDA param bytes {_pg0:.1f} → {_cuda_param_gb():.1f} GB")
            del self.model
            self.model = None

            # Break lingering references to THIS model's original GPU tensors that
            # outlive the model (accelerate tied_params_map lists, stray caches).
            # Only tensors whose storage pointer we recorded above are touched, so
            # other models loaded alongside are never affected.
            if _orig_cuda_ptrs:
                try:
                    broken = 0
                    for obj in gc.get_objects():
                        if not (isinstance(obj, torch.Tensor) and obj.is_cuda):
                            continue
                        try:
                            if obj.untyped_storage().data_ptr() not in _orig_cuda_ptrs:
                                continue
                        except Exception:
                            continue
                        # Null this tensor out of any list/dict that still holds it.
                        for ref in gc.get_referrers(obj):
                            try:
                                if isinstance(ref, list):
                                    for i, it in enumerate(ref):
                                        if it is obj:
                                            ref[i] = None
                                            broken += 1
                                elif isinstance(ref, dict):
                                    for k, v in list(ref.items()):
                                        if v is obj:
                                            ref[k] = None
                                            broken += 1
                            except Exception:
                                pass
                    if _dbg and broken:
                        print(f"  cleanup: broke {broken} external GPU-tensor reference(s)")
                except Exception:
                    pass
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        # Force Python GC before emptying the CUDA allocator pool so that all
        # Python-held tensor references (closures, local vars, etc.) are dropped.
        for _ in range(3):
            gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        # Release the model's host-side memory back to the OS (and any swap it
        # was paged into) so RSS doesn't creep up across model swaps.
        try:
            from codai.models.manager import _trim_cpu_ram
            _trim_cpu_ram()
        except Exception:
            pass
        _v1 = _vram_gb()
        if _v0 >= 0 and _v1 >= 0:
            print(f"  cleanup: freed {_v0 - _v1:.1f} GB VRAM (now {_v1:.1f} GB used)")
            if _dbg:
                try:
                    _alloc = torch.cuda.memory_allocated() / 1e9
                    _resv = torch.cuda.memory_reserved() / 1e9
                    print(f"  cleanup: torch allocated={_alloc:.1f} GB "
                          f"reserved={_resv:.1f} GB (driver used={_v1:.1f} GB)")
                except Exception:
                    pass
                # If a large chunk is still resident, name what's holding CUDA tensors.
                if (_v0 - _v1) < 1.0 and _v1 > 2.0:
                    try:
                        biggest = []
                        total = 0.0
                        seen = set()
                        for obj in gc.get_objects():
                            try:
                                if isinstance(obj, torch.Tensor) and obj.is_cuda:
                                    if id(obj) in seen:
                                        continue
                                    seen.add(id(obj))
                                    gb = obj.numel() * obj.element_size() / 1e9
                                    total += gb
                                    if gb > 0.05:
                                        rtypes = []
                                        for r in gc.get_referrers(obj)[:4]:
                                            rt = type(r).__name__
                                            if rt == 'dict':
                                                try:
                                                    rt = f"dict{list(r.keys())[:3]}"
                                                except Exception:
                                                    pass
                                            rtypes.append(rt)
                                        biggest.append((gb, tuple(obj.shape), rtypes))
                            except Exception:
                                continue
                        biggest.sort(reverse=True)
                        print(f"  cleanup-leak: {total:.1f} GB still in CUDA tensors; top holders:")
                        for gb, shape, rtypes in biggest[:6]:
                            print(f"    {gb:.2f} GB shape={shape} referrers={rtypes}")
                    except Exception as e:
                        print(f"  cleanup-leak scan failed: {e}")

    def get_context_size(self) -> int:
        """Return the model's context window size."""
        if self.model is not None and hasattr(self.model, 'config'):
            config = self.model.config
            # Try different attribute names used by different models
            for attr in ['max_position_embeddings', 'n_positions', 'max_seq_length', 'seq_length']:
                if hasattr(config, attr):
                    return getattr(config, attr)
        return 2048  # Default fallback