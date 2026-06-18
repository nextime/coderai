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
    
    def _model_head_dim(self, model_name: str) -> Optional[int]:
        """Return the model's attention head dimension from its config.

        Prefers the explicit ``head_dim`` field (Gemma sets it directly, decoupled
        from hidden_size/num_heads); otherwise derives hidden_size // num_heads.
        Returns None when the config can't be read.
        """
        from transformers import AutoConfig
        try:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        except Exception as e:
            print(f"Warning: Could not read head dimension from config: {e}")
            return None
        # Multimodal models (Gemma, Qwen-VL) nest the real attention dims under
        # text_config/vision_config; the top level reports None. Return the max
        # head dim across all sub-configs so the FA2 limit check can't be fooled.
        dims = []
        for cfg in (config,
                    getattr(config, 'text_config', None),
                    getattr(config, 'vision_config', None)):
            if cfg is None:
                continue
            head_dim = getattr(cfg, 'head_dim', None)
            if head_dim:
                dims.append(int(head_dim))
                continue
            hidden = getattr(cfg, 'hidden_size', None)
            heads = getattr(cfg, 'num_attention_heads', None)
            if hidden and heads:
                dims.append(int(hidden) // int(heads))
        return max(dims) if dims else None

    def _estimate_kv_cache_bytes(self, model_name: str, n_ctx) -> int:
        """Estimate the KV-cache size (bytes) for an ``n_ctx``-token sequence.

        KV = 2 (key+value) × Σ(effective tokens per layer) × kv_heads × head_dim ×
        dtype_bytes. Effective tokens per layer depend on the attention type:
        full-attention layers hold the whole context; sliding-window layers (gemma)
        cap at the window; linear-attention layers (Qwen3.5/Qwen3-Next) keep only a
        small fixed recurrent state (~0 KV). The cache stays fp16/bf16 (2 bytes)
        even when weights are 4-bit; head_dim/kv_heads come from the *text* config
        (multimodal models nest them under ``text_config``). Returns 0 when
        ``n_ctx`` or the architecture can't be determined.
        """
        try:
            n_ctx = int(n_ctx)
        except (TypeError, ValueError):
            return 0
        if n_ctx <= 0 or not model_name:
            return 0
        try:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            tc = getattr(cfg, 'text_config', None) or cfg
            layers = getattr(tc, 'num_hidden_layers', None)
            kv_heads = (getattr(tc, 'num_key_value_heads', None)
                        or getattr(tc, 'num_attention_heads', None))
            head_dim = getattr(tc, 'head_dim', None)
            if not head_dim:
                hidden = getattr(tc, 'hidden_size', None)
                heads = getattr(tc, 'num_attention_heads', None)
                if hidden and heads:
                    head_dim = int(hidden) // int(heads)
            try:
                sliding = int(getattr(tc, 'sliding_window', None) or 0) or None
            except (TypeError, ValueError):
                sliding = None
            # Sum cached tokens contributed by each layer, honouring its attn type.
            layer_types = getattr(tc, 'layer_types', None)
            if layer_types:
                eff_tokens = 0
                for t in layer_types:
                    tl = str(t).lower()
                    if 'linear' in tl:
                        continue  # recurrent state — negligible KV
                    if 'sliding' in tl and sliding:
                        eff_tokens += min(n_ctx, sliding)
                    else:
                        eff_tokens += n_ctx
            elif layers:
                eff_tokens = int(layers) * n_ctx
            else:
                return 0
            if not (kv_heads and head_dim and eff_tokens > 0):
                return 0
            dtype_bytes = 2  # KV cache is fp16/bf16 regardless of weight quant
            return 2 * int(eff_tokens) * int(kv_heads) * int(head_dim) * dtype_bytes
        except Exception as e:
            print(f"Warning: could not estimate KV cache size: {e}")
            return 0

    def _kv_quant_nbits(self):
        """Decide KV-cache quantization width (2 or 4 bits) or None for fp16.

        Honours an explicit ``cache_type_k``/``cache_type_v`` request (e.g. "q4_0",
        "int4", "q2"); otherwise auto-enables 4-bit quantization when the model's
        estimated fp16 KV cache is large enough to threaten VRAM. Quantizing the
        KV cache (quanto) is what lets a long context coexist with the weights on
        a single GPU instead of forcing a heavy weight offload.
        """
        # quanto/HQQ QuantizedCache only works with plain full-attention models.
        # Both hybrid linear-attention (Qwen3.5/Qwen3-Next) and sliding-window
        # (gemma) models raise during generation, so skip quantization entirely —
        # regardless of any explicit cache_type request.
        if not self._kv_quant_compatible():
            return None
        spec = str(
            getattr(self, '_pending_cache_type_k', None)
            or getattr(self, '_pending_cache_type_v', None)
            or ''
        ).lower()
        if spec in ('', 'f16', 'fp16', 'bf16', 'f32', 'none', 'auto'):
            kv = self._estimate_kv_cache_bytes(
                getattr(self, '_pending_model_name', None),
                getattr(self, '_pending_ctx', None),
            )
            return 4 if kv > 6 * 1024 ** 3 else None
        if spec.startswith('q2') or 'int2' in spec or spec == '2':
            return 2
        return 4

    def _kv_quant_compatible(self) -> bool:
        """Whether the model supports transformers' quantized KV cache.

        Only plain full-attention models do. Hybrid linear-attention models
        (Qwen3.5/Qwen3-Next, identified by 'linear' entries in ``layer_types``)
        raise "`has_previous_state` can only be called on LinearAttention layers",
        and sliding-window/gemma models also fail — so exclude both.
        """
        try:
            cfg = getattr(self.model, 'config', None)
            if cfg is None:
                from transformers import AutoConfig
                name = getattr(self, '_pending_model_name', None)
                if not name:
                    return False
                cfg = AutoConfig.from_pretrained(name, trust_remote_code=True)
            tc = getattr(cfg, 'text_config', None) or cfg
            layer_types = getattr(tc, 'layer_types', None) or []
            if any('linear' in str(t).lower() for t in layer_types):
                return False
            if self._is_sliding_window_model():
                return False
            return True
        except Exception:
            return False

    def _is_sliding_window_model(self) -> bool:
        """True for hybrid / sliding-window-attention models (gemma family).

        Prefers the loaded model's config; falls back to AutoConfig at load time
        (before the model exists) using the pending model name.
        """
        try:
            cfg = getattr(self.model, 'config', None)
            if cfg is None:
                from transformers import AutoConfig
                name = getattr(self, '_pending_model_name', None)
                if not name:
                    return False
                cfg = AutoConfig.from_pretrained(name, trust_remote_code=True)
            tc = getattr(cfg, 'text_config', None) or cfg
            model_type = (getattr(tc, 'model_type', '') or '').lower()
            cache_impl = (getattr(tc, 'cache_implementation', '') or '').lower()
            return (
                model_type.startswith('gemma')
                or getattr(tc, 'sliding_window', None) is not None
                or cache_impl in {'hybrid', 'sliding_window'}
            )
        except Exception:
            return False

    def _kv_cache_reserve_bytes(self) -> int:
        """VRAM (bytes) to reserve for the KV cache, accounting for quantization.

        Quantized caches keep a small fp16 residual window plus group metadata, so
        we scale the fp16 estimate by nbits/16 with ~1.5× overhead rather than a
        naive 4×. Returns 0 when the size is unknown.
        """
        fp16 = self._estimate_kv_cache_bytes(
            getattr(self, '_pending_model_name', None),
            getattr(self, '_pending_ctx', None),
        )
        if fp16 <= 0:
            return 0
        nbits = self._kv_quant_nbits()
        if nbits:
            return int(fp16 * (nbits / 16.0) * 1.5)
        return fp16

    def _kv_offload_threshold_bytes(self) -> int:
        """Free VRAM (after weights) below which a large KV should live on CPU.

        Computed once per load from the actual free VRAM headroom; falls back to a
        fixed 8 GB if it can't be read.
        """
        try:
            import torch
            free, _ = torch.cuda.mem_get_info()
            # Leave ~2 GB for activations/compute; KV above that goes to CPU.
            return max(int(2 * 1024 ** 3), int(free - 2 * 1024 ** 3))
        except Exception:
            return 8 * 1024 ** 3

    def _offloaded_cache_impl(self) -> str:
        """Name of the offloaded KV cache for sliding-window / hybrid models.

        transformers >=5.12 merges the hybrid offloaded cache into
        ``offloaded_static`` (the sliding/full layer structure is inferred from the
        model config automatically); ``offloaded_hybrid`` is deprecated and removed
        in v5.13. Prefer the new name when the installed transformers exposes it, so
        we stay correct across versions without emitting the deprecation warning.
        """
        try:
            from transformers.generation.configuration_utils import ALL_CACHE_IMPLEMENTATIONS
            if 'offloaded_static' in ALL_CACHE_IMPLEMENTATIONS:
                return 'offloaded_static'
        except Exception:
            pass
        return 'offloaded_hybrid'

    def _cache_gen_kwargs(self, using_prefix: bool, plain: bool = False) -> dict:
        """generate() kwargs selecting the KV-cache strategy, or {} for default.

        Priority: (1) quantized cache for compatible large-KV models (cuts VRAM
        ~4×); (2) offloaded cache when the estimated KV won't fit in free VRAM —
        keeps weights on GPU and streams KV from CPU RAM, and works on hybrid /
        sliding-window models where the quantized cache crashes. ``plain=True`` (the
        fallback path) forces the default in-GPU DynamicCache so a request can
        always succeed even if a special cache is unsupported. Skipped entirely
        when a manually-prefilled prefix cache is in use.
        """
        if using_prefix or plain:
            return {}

        # 1. Quantized cache (full-attention models only; returns None otherwise).
        nbits = self._kv_quant_nbits()
        if nbits:
            if not getattr(self, '_cache_strategy_announced', False):
                print(f"KV cache quantization enabled: quanto int{nbits} (residual_length=128)")
                self._cache_strategy_announced = True
            return {
                'cache_implementation': 'quantized',
                'cache_config': {
                    'backend': 'quanto',
                    'nbits': nbits,
                    'q_group_size': 64,
                    'residual_length': 128,
                },
            }

        # 2. Offloaded cache when the KV is too large to fit in free VRAM.
        kv = self._estimate_kv_cache_bytes(
            getattr(self, '_pending_model_name', None),
            getattr(self, '_pending_ctx', None),
        )
        if kv > 0 and kv > self._kv_offload_threshold_bytes():
            impl = self._offloaded_cache_impl() if self._is_sliding_window_model() else 'offloaded'
            if not getattr(self, '_cache_strategy_announced', False):
                print(f"KV cache offloaded to CPU: cache_implementation={impl} "
                      f"(est ~{kv/1e9:.1f}GB exceeds free VRAM)")
                self._cache_strategy_announced = True
            return {'cache_implementation': impl}

        return {}

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
            load_kwargs = self._strip_invalid_native_quant_config(model_name, load_kwargs)
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
    
    def _prequant_method(self, model_name: str):
        """Return the checkpoint's valid embedded quantization method, or None.

        Models shipped already-quantized (FP8 / GPTQ / AWQ / compressed-tensors,
        e.g. DeepSeek-V4-Flash's FineGrainedFP8Config) carry a ``quantization_config``
        in their config.json and MUST be loaded with that native config —
        bitsandbytes cannot be layered on top (transformers raises
        "is quantized with ... but you are passing a BitsAndBytesConfig").

        Some non-transformers repositories (notably MLX checkpoints) publish a
        partial ``quantization_config`` without ``quant_method``.  Transformers
        treats that as invalid and raises during ``from_pretrained`` even if we
        don't pass our own config.  Do not treat those checkpoints as native
        transformers quantized models.
        """
        try:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            qc = getattr(cfg, 'quantization_config', None)
            if not qc:
                return None
            if isinstance(qc, dict):
                return qc.get('quant_method')
            return getattr(qc, 'quant_method', None)
        except Exception:
            return None

    def _strip_invalid_native_quant_config(self, model_name: str, load_kwargs: dict) -> dict:
        """Avoid passing malformed native quantization configs to transformers.

        If a checkpoint config has ``quantization_config`` but no
        ``quant_method``, recent transformers aborts with:
        "The model's quantization config ... has no `quant_method` attribute".
        Removing it lets normal HF/bitsandbytes loading paths proceed; MLX-only
        checkpoints will then fail with a clearer architecture/weight mismatch
        instead of entering the text endpoint retry loop for a bogus quant config.
        """
        if 'quantization_config' in load_kwargs:
            return load_kwargs
        try:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            qc = getattr(cfg, 'quantization_config', None)
            if not isinstance(qc, dict) or qc.get('quant_method'):
                return load_kwargs
            if hasattr(cfg, 'quantization_config'):
                delattr(cfg, 'quantization_config')
            patched = dict(load_kwargs)
            patched['config'] = cfg
            print("Ignoring invalid checkpoint quantization_config without quant_method; "
                  "using explicit loader quantization/settings instead.")
            return patched
        except Exception:
            return load_kwargs

    def _make_bnb_config(self, model_name: str, load_in_4bit: bool, load_in_8bit: bool):
        """Build a transformers BitsAndBytesConfig (the modern quant API).

        Passing load_in_4bit/load_in_8bit as direct from_pretrained kwargs is
        removed in recent transformers and raises TypeError — which previously
        forced a silent fallback to FULL-PRECISION loading (the model then no
        longer fit on the GPU, offloaded to CPU, and leaked VRAM on eviction).
        Always go through quantization_config instead.
        """
        ml = model_name.lower()
        # Already-quantized checkpoints must load with their own config; bnb on top
        # is rejected by transformers. Skip bnb and let from_pretrained use the
        # embedded quantization_config.
        pq = self._prequant_method(model_name)
        if pq:
            print(f"Model is pre-quantized ({pq}); skipping bitsandbytes and loading "
                  f"with its native quantization config.")
            return None
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

        # Re-evaluate KV-prefix support for the model about to be loaded.
        self._kv_prefix_ok = None

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
        self._pending_model_name = model_name
        self._pending_ctx = kwargs.get('ctx')
        self._pending_cache_type_k = kwargs.get('cache_type_k')
        self._pending_cache_type_v = kwargs.get('cache_type_v')

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

        # FlashAttention-2's forward kernel supports a head dimension of at most
        # 256. Gemma (and some other large-head-dim models) exceed this, so FA2
        # raises "FlashAttention forward only supports head dimension at most
        # 256" on EVERY forward pass — both the KV-prefix build and the actual
        # model.generate (whose error is swallowed by the streamer thread, so the
        # request silently produces no output and appears to hang). Fall back to
        # SDPA, which handles any head dimension and still uses flash kernels.
        if self.use_flash_attn:
            head_dim = self._model_head_dim(model_name)
            fa2_bad = bool(head_dim and head_dim > 256)
            reason = f"head dimension {head_dim} exceeds FA2's limit of 256" if fa2_bad else None
            if not fa2_bad:
                # Gemma reports head_dim==256 but still raises "FlashAttention
                # forward only supports head dimension at most 256" on every
                # forward (its sliding-window attention path), producing empty
                # replies. Treat the whole gemma family as FA2-incompatible.
                try:
                    from transformers import AutoConfig
                    _cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                    _tc = getattr(_cfg, 'text_config', None) or _cfg
                    _mt = (getattr(_tc, 'model_type', '') or getattr(_cfg, 'model_type', '') or '').lower()
                    if _mt.startswith('gemma'):
                        fa2_bad = True
                        reason = f"gemma family (model_type={_mt}) is incompatible with FA2"
                except Exception:
                    pass
            if fa2_bad:
                self.use_flash_attn = False
                print(f"  Flash Attention 2 disabled: {reason} → using SDPA instead.")

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
                load_kwargs = self._strip_invalid_native_quant_config(model_name, load_kwargs)
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
                load_kwargs = self._strip_invalid_native_quant_config(model_name, load_kwargs)
                model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
            except Exception as e:
                raise RuntimeError(
                    f"--offload-strategy none: Failed to load model entirely on GPU ({cuda_device}). "
                    f"The model may be too large for available VRAM. Error: {e}"
                )
        else:
            # 'auto'/'auto-borderline': honour the dropdown's documented contract —
            # "over-VRAM → straight to model offload" means that when the model's
            # peak (quantized weights + KV reserve + activations) FITS in free VRAM
            # we load it full-GPU on a single device (fast, no device_map split, no
            # CPU staging), and only fall through to the device_map=auto offload
            # ladder below when it genuinely doesn't fit. The full-GPU attempt is
            # non-fatal: on OOM it falls back to the ladder (unlike strategy 'none',
            # which hard-errors). Honours the large-context KV reserve so a model
            # that fits *with* its 64k KV stays resident.
            if (model is None and self.device == "cuda"
                    and offload_strategy in ('auto', 'auto-borderline')):
                _fits = False
                try:
                    if torch.cuda.is_available() and expected_vram_gb > 0:
                        _free, _ = torch.cuda.mem_get_info(0)
                        _free_gb = _free / 1e9
                        _kv_gb = self._kv_cache_reserve_bytes() / 1e9
                        _act_gb = 1.5 if _kv_gb > 0 else 0.0
                        _need_gb = expected_vram_gb + _kv_gb + _act_gb
                        _borderline = 3.0 if offload_strategy == 'auto-borderline' else 0.0
                        _fits = _need_gb <= (_free_gb - 0.5 + _borderline)
                        if _fits:
                            print(f"\n  Auto: peak VRAM need {_need_gb:.1f} GB "
                                  f"(weights {expected_vram_gb:.1f} + KV {_kv_gb:.1f} "
                                  f"+ act {_act_gb:.1f}) fits in {_free_gb:.1f} GB free "
                                  f"— loading full-GPU (no offload)")
                        else:
                            print(f"\n  Auto: peak VRAM need {_need_gb:.1f} GB > "
                                  f"{_free_gb:.1f} GB free — going straight to "
                                  f"device_map offload")
                except Exception:
                    _fits = False
                if _fits:
                    cuda_device = self._derive_cuda_device()
                    _fg_kwargs = dict(load_kwargs)
                    _fg_kwargs['device_map'] = cuda_device
                    _fg_kwargs['low_cpu_mem_usage'] = True
                    _fg_kwargs = self._strip_invalid_native_quant_config(model_name, _fg_kwargs)
                    model = self._try_load_model(model_name, _fg_kwargs, self.device)
                    if model is not None:
                        print(f"  ✓ Model loaded full-GPU on {cuda_device}")
                    else:
                        print("  ✗ Full-GPU load OOMed — falling back to "
                              "device_map offload ladder")

            first_vram_pct = vram_percentages[0] if vram_percentages else 0.93

            for vram_pct in vram_percentages:
                if model is not None:
                    break
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

        # Reserve VRAM for the KV cache (grows with context) plus a fixed
        # activation/compute buffer, so device_map offloads enough weight layers
        # to CPU instead of packing VRAM with weights and OOMing at generation.
        # Uses the quantization-aware reserve so an int4 KV cache doesn't force a
        # needless heavy offload.
        kv_reserve = self._kv_cache_reserve_bytes()
        activation_reserve = int(1.5 * 1024 ** 3) if kv_reserve > 0 else 0

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
                weight_budget = min(limit_by_fraction, limit_by_free)
                # Cap the reservation so a large/mis-estimated KV cache can never
                # crush the weight budget: never reserve more than 60% of the GPU
                # budget for context. If the KV genuinely doesn't fit in the
                # remaining 40%, KV quantization (see _kv_quant_nbits) is the lever,
                # not starving the weights onto CPU.
                reserved = min(kv_reserve + activation_reserve, int(weight_budget * 0.6))
                if reserved > 0:
                    new_budget = max(weight_budget - reserved, int(weight_budget * 0.4))
                    print(
                        f"  GPU {i}: reserving {reserved/1e9:.1f}GB for KV+activations "
                        f"(KV~{kv_reserve/1e9:.1f}GB, ctx={getattr(self, '_pending_ctx', None)}, "
                        f"quant={self._kv_quant_nbits()}); "
                        f"weight budget {weight_budget/1e9:.1f}→{new_budget/1e9:.1f}GB "
                        f"(rest spills to CPU)"
                    )
                    weight_budget = new_budget
                max_memory[i] = weight_budget

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

        # Quantize the KV cache when enabled (completions never use a prefix cache).
        generation_kwargs.update(self._cache_gen_kwargs(using_prefix=False))
        
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
            except Exception as e:
                # Any other failure (shape/cache mismatch, transformers API change, …)
                # must still be recorded — otherwise it is silently swallowed.
                generation_error = str(e)
                print(f"Error during streaming generation: {e}")
            finally:
                # generate() only calls streamer.end() on its success path. If it
                # raised before finishing, end the streamer here so the consumer is
                # never left blocked forever on an empty queue (which freezes the
                # whole event loop).
                streamer.end()

        thread = Thread(target=generate_with_error_handling)
        thread.start()

        # Pull each token from a worker thread so a blocking streamer.__next__
        # never runs on (and freezes) the asyncio event loop between tokens.
        import asyncio
        _SENT = object()
        _it = iter(streamer)
        def _next_token():
            try:
                return next(_it)
            except StopIteration:
                return _SENT
        try:
            while True:
                text = await asyncio.to_thread(_next_token)
                if text is _SENT:
                    break
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

    def _kv_prefix_supported(self) -> bool:
        """Whether this model can safely reuse a manually-prefilled KV cache.

        The prefix fast-path builds a cache with a plain forward pass and then
        continues it via generate(input_ids=suffix, past_key_values=cache). That
        only works for models that use a simple growing (Dynamic) cache. Models
        with a sliding-window / hybrid cache (e.g. the gemma family) build a
        different cache object during generate() and raise before the first
        token when handed our prefix — so disable the fast-path for them and let
        the full forward pass handle the request.
        """
        cached = getattr(self, "_kv_prefix_ok", None)
        if cached is not None:
            return cached
        ok = True
        try:
            cfg = getattr(self.model, "config", None)
            # Multimodal wrappers nest the LM config under text_config.
            text_cfg = getattr(cfg, "text_config", None) or cfg
            model_type = (getattr(text_cfg, "model_type", "") or "").lower()
            cache_impl = (getattr(text_cfg, "cache_implementation", "") or "").lower()
            sliding = getattr(text_cfg, "sliding_window", None)
            reason = None
            if (
                model_type.startswith("gemma")
                or cache_impl in {"hybrid", "static", "sliding_window"}
                or sliding is not None
            ):
                reason = "hybrid/sliding-window cache"
            else:
                # A large configured context means the stored prefix KV is several
                # GB and lives *alongside* the generation cache — doubling KV
                # memory and risking OOM on a single GPU. Not worth it: disable the
                # fast-path so only one KV cache is ever resident.
                kv_bytes = self._estimate_kv_cache_bytes(
                    getattr(self, '_pending_model_name', None),
                    getattr(self, '_pending_ctx', None),
                )
                if kv_bytes > 2 * 1024 ** 3:
                    reason = f"large KV cache (~{kv_bytes/1e9:.1f}GB at configured ctx)"
            if reason is not None:
                ok = False
                self._kv_prefix_off_reason = reason
        except Exception:
            # If we can't introspect the config, stay safe and skip the fast-path.
            ok = False
            self._kv_prefix_off_reason = "config introspection failed"
        if not ok:
            print(
                "KV-prefix fast-path disabled for this model "
                f"({getattr(self, '_kv_prefix_off_reason', 'unsupported')}); "
                "using full forward pass"
            )
        self._kv_prefix_ok = ok
        return ok

    def _kv_prefix_headroom_ok(self, min_free_gb: float = 1.5) -> bool:
        """Whether there is enough free VRAM to safely build/store a KV prefix.

        The prefix path runs an extra forward pass and keeps a second copy of the
        prefix KV alongside the live model. On a nearly-full card that extra
        allocation OOMs (the build is caught and we fall back, but it wastes a
        forward pass and risks fragmentation). Skip it when headroom is low and
        let normal generation — which doesn't keep a separate stored prefix —
        handle the request.
        """
        import torch
        try:
            free, _total = torch.cuda.mem_get_info()
            return free / 1e9 >= min_free_gb
        except Exception:
            return False
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
        # The model's own generation_config is authoritative for the turn-end
        # token(s) — e.g. gemma-4's turn terminator is <turn|> (id 106), which has
        # no recognisable name in the loop below, so without this the model never
        # stops after a tool call and loops to max_tokens.
        try:
            gc_eos = getattr(getattr(self.model, 'generation_config', None),
                             'eos_token_id', None)
            if isinstance(gc_eos, int):
                ids.add(gc_eos)
            elif isinstance(gc_eos, (list, tuple)):
                ids.update(int(t) for t in gc_eos if isinstance(t, int))
        except Exception:
            pass
        for tok in ('<|im_end|>', '<|eot_id|>', '<|end|>', '<|endoftext|>',
                    '<|end_of_text|>', '<end_of_turn>', '<turn|>'):
            try:
                tid = self.tokenizer.convert_tokens_to_ids(tok)
                if isinstance(tid, int) and tid >= 0 and tid != getattr(
                        self.tokenizer, 'unk_token_id', None):
                    ids.add(tid)
            except Exception:
                pass
        return list(ids) if ids else self.tokenizer.eos_token_id

    def supports_native_tools(self) -> bool:
        """True when the loaded model's chat template understands `tools=` natively
        (gemma-4, Qwen, Llama-3.1, …). For those we pass the structured tools to the
        template instead of injecting coderai's custom <tool>{…} text prompt, so the
        model is prompted in — and replies in — its own trained tool-call format."""
        tmpl = getattr(self.tokenizer, 'chat_template', None)
        return bool(tmpl) and ('tools' in tmpl or 'tool_calls' in tmpl)

    def _native_tools_payload(self, tools):
        """Normalise tools to the OpenAI [{'type','function':{…}}] dicts that chat
        templates expect. Accepts dicts or pydantic Tool objects; returns None if
        there's nothing usable."""
        if not tools:
            return None
        out = []
        for t in tools:
            if isinstance(t, dict):
                fn = t.get('function') or {}
                name = fn.get('name') if isinstance(fn, dict) else None
                if not name:
                    continue
                out.append({'type': t.get('type', 'function'),
                            'function': {'name': name,
                                         'description': fn.get('description') or '',
                                         'parameters': fn.get('parameters') or {}}})
            else:
                fn = getattr(t, 'function', None)
                name = getattr(fn, 'name', None) if fn else None
                if not name:
                    continue
                out.append({'type': getattr(t, 'type', 'function'),
                            'function': {'name': name,
                                         'description': getattr(fn, 'description', '') or '',
                                         'parameters': getattr(fn, 'parameters', {}) or {}}})
        return out or None

    def _build_native_tool_prompt(self, messages, native_tools, enable_thinking,
                                  add_generation_prompt):
        """Render the prompt via the model's template with native `tools=`, keeping
        structured `tool_calls` and `role:tool` turns intact so the template emits
        the model's own tool-call/tool-response format. Returns the string, or None
        if the template can't handle it (caller falls back)."""
        import re as _re

        def _get(m, k, default=None):
            return m.get(k, default) if isinstance(m, dict) else getattr(m, k, default)

        # Going native: strip coderai's custom <tool>{…} text instruction that
        # format_tools_for_prompt() prepends to the system prompt, so the model
        # isn't told to use two different tool formats at once (native tool
        # declarations are supplied via tools= below).
        def _strip_injected(text):
            if not text or 'You have access to the following tools:' not in text:
                return text
            return _re.sub(
                r"You have access to the following tools:.*?example\.txt.*?</tool>\s*",
                "", text, count=1, flags=_re.DOTALL).lstrip()

        norm = []
        for m in messages:
            role = _get(m, 'role')
            content = _get(m, 'content') or ''
            if isinstance(content, list):
                content = '\n'.join(
                    str(p.get('text', '')) if isinstance(p, dict) else str(p)
                    for p in content)
            if role in ('system', 'developer'):
                content = _strip_injected(content)
            entry = {'role': role, 'content': content}
            tcs = _get(m, 'tool_calls')
            if tcs:
                # Pass tool_calls through in the OpenAI shape the templates expect
                # (function.name + function.arguments as a JSON string).
                norm_tcs = []
                for tc in tcs:
                    fn = (tc.get('function') if isinstance(tc, dict)
                          else getattr(tc, 'function', None)) or {}
                    name = fn.get('name') if isinstance(fn, dict) else getattr(fn, 'name', '')
                    args = fn.get('arguments') if isinstance(fn, dict) else getattr(fn, 'arguments', '{}')
                    tcid = (tc.get('id') if isinstance(tc, dict) else getattr(tc, 'id', None)) or f"call_{len(norm_tcs)}"
                    norm_tcs.append({'id': tcid, 'type': 'function',
                                     'function': {'name': name, 'arguments': args}})
                entry['tool_calls'] = norm_tcs
            tcid = _get(m, 'tool_call_id')
            if tcid:
                entry['tool_call_id'] = tcid
            name = _get(m, 'name')
            if name:
                entry['name'] = name
            norm.append(entry)
        for kwargs in ({'tools': native_tools, 'add_generation_prompt': add_generation_prompt,
                        'enable_thinking': enable_thinking},
                       {'tools': native_tools, 'add_generation_prompt': add_generation_prompt}):
            try:
                return self.tokenizer.apply_chat_template(norm, tokenize=False, **kwargs)
            except TypeError:
                continue
            except Exception:
                return None
        return None

    def _build_chat_prompt(self, messages, enable_thinking: bool = False,
                           add_generation_prompt: bool = True, tools=None) -> str:
        """Build the prompt string using the MODEL's own chat template when it has
        one (correct special tokens + proper `enable_thinking` handling for Qwen3).
        Falls back to the legacy custom formatter when no template is available.

        `enable_thinking=True` keeps reasoning <think> blocks available for callers
        that ask for them; `False` (default) suppresses them via the template.

        When `tools` is given and the template natively supports tools, the tools
        and the structured tool_calls/tool-role turns are passed straight to the
        template (native format) — see :meth:`supports_native_tools`.
        """
        import json
        tmpl = getattr(self.tokenizer, 'chat_template', None)

        # Native-tools fast path: hand structured tools + tool turns to the model's
        # own template so it renders (and the model emits) its trained tool-call
        # format, instead of folding everything into custom <tool>{…} text.
        native_tools = self._native_tools_payload(tools) if (
            tmpl and tools and self.supports_native_tools()) else None
        if native_tools:
            prompt = self._build_native_tool_prompt(
                messages, native_tools, enable_thinking, add_generation_prompt)
            if prompt is not None:
                return prompt
            # else: native render failed — fall through to the generic path.

        if tmpl:
            # Normalise to plain {role, content} dicts for apply_chat_template.
            #
            # Most chat templates (gemma, mistral, …) only understand the
            # system/user/assistant roles and a plain `content` string — they
            # ignore `tool_calls`/`tool_call_id` and reject (or silently drop) the
            # `tool` role. If we simply stripped those, an agentic client
            # (opencode, etc.) would lose the record of the tool call it already
            # made *and* the result it got back, so the model re-issues the same
            # call every turn — an infinite tool-call loop. So we fold tool turns
            # back into `content` using the same `<tool>{…}</tool>` convention the
            # tool-injection prompt teaches, and render tool results as readable
            # text under a role the template accepts.
            def _get(m, k, default=None):
                return m.get(k, default) if isinstance(m, dict) else getattr(m, k, default)

            norm = []
            for m in messages:
                role = _get(m, 'role')
                content = _get(m, 'content') or ''
                if isinstance(content, list):
                    content = '\n'.join(
                        str(p.get('text', '')) if isinstance(p, dict) else str(p)
                        for p in content)
                if role == 'assistant':
                    tcs = _get(m, 'tool_calls') or []
                    for tc in tcs:
                        fn = (tc.get('function') if isinstance(tc, dict)
                              else getattr(tc, 'function', None)) or {}
                        name = fn.get('name', '') if isinstance(fn, dict) else getattr(fn, 'name', '')
                        args = fn.get('arguments', '{}') if isinstance(fn, dict) else getattr(fn, 'arguments', '{}')
                        if not isinstance(args, str):
                            try:
                                args = json.dumps(args)
                            except Exception:
                                args = '{}'
                        block = f'<tool>{{"name": "{name}", "arguments": {args}}}</tool>'
                        content = (content + '\n' + block) if content else block
                    norm.append({'role': 'assistant', 'content': content})
                elif role == 'tool':
                    # Templates that lack a `tool` role would error/drop this.
                    # Render the result as a user turn so the model sees it.
                    name = _get(m, 'name') or ''
                    label = f'Tool result ({name})' if name else 'Tool result'
                    text = f'{label}: {content}'
                    # Merge into the previous turn if it's also a synthesised
                    # user/tool message, to avoid consecutive same-role turns that
                    # strict templates (gemma) reject.
                    if norm and norm[-1]['role'] == 'user':
                        norm[-1]['content'] = norm[-1]['content'] + '\n' + text
                    else:
                        norm.append({'role': 'user', 'content': text})
                else:
                    norm.append({'role': role, 'content': content})
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
                                              add_generation_prompt=True, tools=tools)
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

        if prefix_msgs and self._model_on_cuda() and self._kv_prefix_supported() and self._kv_prefix_headroom_ok():
            prefix_text = self._build_chat_prompt(
                prefix_msgs, enable_thinking=enable_thinking, add_generation_prompt=False, tools=tools)
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
                        **self._cache_gen_kwargs(using_prefix=False),
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
                                              add_generation_prompt=True, tools=tools)
        total_input_ids = self.tokenizer(full_prompt, return_tensors="pt")['input_ids']
        total_prompt_len = int(total_input_ids.shape[1])

        prefix_msgs = (
            messages[:-1]
            if messages and messages[-1].get('role') == 'user'
            else []
        )
        past_kv = None
        cached_len = 0

        if prefix_msgs and self._model_on_cuda() and self._kv_prefix_supported() and self._kv_prefix_headroom_ok():
            prefix_text = self._build_chat_prompt(
                prefix_msgs, enable_thinking=enable_thinking, add_generation_prompt=False, tools=tools)
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

        # Stopping criteria (thermal checkpoint + optional stop sequences) are
        # independent of the KV-prefix path, so build them once and reuse across
        # both the cached attempt and any full-forward fallback.
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
        stopping = StoppingCriteriaList(_criteria) if _criteria else None

        import asyncio
        _SENT = object()

        def _build_attempt(use_prefix, plain_cache=False):
            """Build (streamer, gen_kwargs, used_cached_len) for one attempt.
            use_prefix=False forces a clean full forward pass (no past_key_values);
            plain_cache=True forces the default in-GPU cache (fallback path)."""
            if use_prefix and past_kv is not None and 0 < cached_len < total_prompt_len:
                gen_input_ids = total_input_ids[:, cached_len:]
                full_attn = torch.ones(
                    1, total_prompt_len, dtype=torch.long, device=self.model.device
                )
                extra_gen = {'past_key_values': past_kv, 'attention_mask': full_attn}
                used_cached = cached_len
            else:
                gen_input_ids = total_input_ids
                extra_gen = {'attention_mask': torch.ones_like(total_input_ids)}
                used_cached = 0
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
            if stopping is not None:
                gen_kwargs['stopping_criteria'] = stopping
            # Select the KV-cache strategy (quantized / offloaded) on the
            # full-forward path. Not combinable with a prefix cache; plain_cache
            # forces the default cache for the guaranteed-working fallback.
            gen_kwargs.update(self._cache_gen_kwargs(
                using_prefix=used_cached > 0, plain=plain_cache))
            return streamer, gen_kwargs, used_cached

        gen_error = [None]
        comp_tokens = [0]
        final_cached_len = [0]

        async def _attempt(use_prefix, plain_cache=False):
            """Run one generation attempt, yielding decoded text. Records any
            failure in gen_error[0] and the prefix length actually used."""
            gen_error[0] = None
            streamer, gen_kwargs, used_cached = _build_attempt(use_prefix, plain_cache)
            final_cached_len[0] = used_cached

            def _run():
                try:
                    with torch.no_grad():
                        self.model.generate(**gen_kwargs)
                except Exception as e:
                    gen_error[0] = str(e)
                    print(f"Error during {'KV-cached' if use_prefix else 'full'} stream generation: {e}")
                    # Release whatever the failed pass reserved so the next request
                    # starts from a clean allocator state (esp. on OOM).
                    if "out of memory" in str(e).lower():
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                finally:
                    # generate() only ends the streamer on success; if it raised
                    # before finishing, end it here so the consumer never deadlocks
                    # on an empty queue (which would freeze the whole event loop).
                    streamer.end()

            thread = Thread(target=_run)
            thread.start()
            # Pull each token from a worker thread so a blocking streamer.__next__
            # never runs on (and freezes) the asyncio event loop between tokens.
            _it = iter(streamer)
            def _next_token():
                try:
                    return next(_it)
                except StopIteration:
                    return _SENT
            try:
                while True:
                    text = await asyncio.to_thread(_next_token)
                    if text is _SENT:
                        break
                    comp_tokens[0] += 1
                    yield text
            finally:
                thread.join()

        try:
            # First attempt: use the cached KV prefix when one is available.
            async for text in _attempt(use_prefix=True):
                yield text
            # If the first attempt failed before emitting any token — whether from
            # a stale prefix or an unsupported quantized/offloaded cache — retry
            # once with a clean full forward pass and the default in-GPU cache,
            # which is guaranteed to work if the model works at all.
            if gen_error[0] and comp_tokens[0] == 0:
                print("generation failed before first token; retrying with plain full forward pass")
                self.invalidate_kv_cache()
                async for text in _attempt(use_prefix=False, plain_cache=True):
                    yield text
        finally:
            self._last_usage = {
                'prompt_tokens': total_prompt_len,
                'completion_tokens': comp_tokens[0],
                'cached_tokens': final_cached_len[0],
            }

        if gen_error[0]:
            print(f"Warning: stream generation error (after fallback): {gen_error[0]}")
            self.invalidate_kv_cache()
            # If we produced nothing, the client would otherwise receive an empty
            # but successful completion. Stream a visible error notice instead.
            if comp_tokens[0] == 0:
                if "out of memory" in gen_error[0].lower():
                    yield ("[error: GPU ran out of memory during generation — "
                           "try a shorter prompt/context or a smaller model]")
                else:
                    yield "[error: text generation failed — see server logs]"

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
