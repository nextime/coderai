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

# AI.PROMPT: Add Vulkan backend support for AMD GPUs using llama-cpp-python
# This backend handles GGUF models on AMD GPUs via Vulkan

import os
import json
import threading
import time
from typing import AsyncIterator, Optional, Union, List, Dict, Any
from pathlib import Path

from codai.backends.base import ModelBackend
from codai.models.utils import (
    check_hf_chat_template,
    get_model_family,
    get_reasoning_stop_tokens
)
from codai.models.cache import get_cached_model_path
from codai.models.grammar import get_tool_call_grammar, is_grammar_available

# Import global flag from coderai (will be None if not running as server)
try:
    import coderai
    _grammar_guided_gen = getattr(coderai, 'grammar_guided_gen', False)
except (ImportError, AttributeError):
    _grammar_guided_gen = False


def _make_llama_thermal_criteria():
    """A llama.cpp StoppingCriteriaList that pauses generation while too hot.

    llama-cpp-python evaluates stopping criteria synchronously per token inside
    create_(chat_)completion, so blocking here pauses the GPU forward pass —
    mid-generation thermal protection for the GGUF/Vulkan/llama.cpp backend.
    The criterion never stops generation (returns False).

    The callback fires PER TOKEN and holds the GIL, which starves the engine's
    admin handlers (Tasks/status polls) under load. So we only do real work every
    N tokens: N=50 by default, and N=100 when generation is fast (>50 tok/s), where
    a per-token GIL grab hurts responsiveness most and thermal drift between checks
    is negligible. The sensor read itself is still time-throttled in checkpoint().
    """
    try:
        from llama_cpp import StoppingCriteriaList
    except Exception:
        return None

    import time as _time
    _state = {"n": 0, "interval": 50, "t": _time.monotonic(), "last_n": 0}

    def _pause(input_ids, logits):
        _state["n"] += 1
        if _state["n"] % _state["interval"] != 0:
            return False
        # Re-estimate cadence from the measured token rate over the last window.
        now = _time.monotonic()
        dt = now - _state["t"]
        if dt > 0:
            rate = (_state["n"] - _state["last_n"]) / dt
            _state["interval"] = 100 if rate > 50 else 50
        _state["t"] = now
        _state["last_n"] = _state["n"]
        try:
            from codai.models.thermal import checkpoint
            checkpoint(context="text-gen", throttle_seconds=2.0)
        except Exception:
            pass
        return False

    try:
        return StoppingCriteriaList([_pause])
    except Exception:
        return None


_CHAT_SUPPORTS_STOPPING_CRITERIA = None


def _chat_supports_stopping_criteria() -> bool:
    """Whether this llama-cpp-python's create_chat_completion accepts
    ``stopping_criteria``. Older/newer versions differ: create_completion always
    takes it, but several create_chat_completion builds do not, raising
    'unexpected keyword argument'. Checked once via signature inspection."""
    global _CHAT_SUPPORTS_STOPPING_CRITERIA
    if _CHAT_SUPPORTS_STOPPING_CRITERIA is None:
        supported = False
        try:
            import inspect
            from llama_cpp import Llama as _L
            sig = inspect.signature(_L.create_chat_completion)
            supported = ("stopping_criteria" in sig.parameters
                         or any(p.kind == inspect.Parameter.VAR_KEYWORD
                                for p in sig.parameters.values()))
        except Exception:
            supported = False
        _CHAT_SUPPORTS_STOPPING_CRITERIA = supported
    return _CHAT_SUPPORTS_STOPPING_CRITERIA

try:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import ChatFormatterResponse
    import llama_cpp as _llama_cpp
    LLAMA_CPP_AVAILABLE = True
except Exception as _llama_import_err:
    # Catch more than ImportError: a llama-cpp-python built against CUDA raises
    # RuntimeError/OSError (e.g. "libcuda.so.1: cannot open shared object file")
    # when the NVIDIA driver libs aren't present — as in Vulkan/CPU-only runs.
    # That must NOT crash the whole server import chain; the Vulkan backend just
    # becomes unavailable and other backends (CUDA/CPU/ds4) keep working.
    LLAMA_CPP_AVAILABLE = False
    Llama = None
    ChatFormatterResponse = None
    _llama_cpp = None
    if not isinstance(_llama_import_err, ImportError):
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "llama-cpp-python present but failed to load (%s); Vulkan/GGUF "
            "backend disabled. If you expect GPU GGUF, ensure the matching GPU "
            "runtime libs are mapped into this environment.", _llama_import_err)


def _llama_accepts(param: str) -> bool:
    """True when the installed llama-cpp-python ``Llama`` constructor accepts a
    given keyword. Used to gate kwargs (e.g. ``n_seq_max``) that only some builds
    expose, so passing them never raises ``TypeError`` on older bindings."""
    if Llama is None:
        return False
    try:
        import inspect
        return param in inspect.signature(Llama.__init__).parameters
    except (TypeError, ValueError):
        return False


# Friendly KV-cache quant names → llama.cpp GGML type. q8_0 is near-lossless and
# the safe default; the q5/q4 types trade a little accuracy for ~2x less KV VRAM.
_KV_TYPE_ALIASES = {
    'f16': 'GGML_TYPE_F16', 'fp16': 'GGML_TYPE_F16', 'f32': 'GGML_TYPE_F32',
    'q8_0': 'GGML_TYPE_Q8_0', 'q8': 'GGML_TYPE_Q8_0', 'q8_1': 'GGML_TYPE_Q8_1',
    'q5_0': 'GGML_TYPE_Q5_0', 'q5_1': 'GGML_TYPE_Q5_1', 'q5': 'GGML_TYPE_Q5_1',
    'q4_0': 'GGML_TYPE_Q4_0', 'q4_1': 'GGML_TYPE_Q4_1', 'q4': 'GGML_TYPE_Q4_1',
    'iq4_nl': 'GGML_TYPE_IQ4_NL',
}

# Sub-8-bit KV types that llama.cpp can only use with flash attention enabled.
_KV_NEEDS_FLASH = {'q5_0', 'q5_1', 'q5', 'q4_0', 'q4_1', 'q4', 'iq4_nl'}

_GGUF_META_CACHE: dict = {}


def _gguf_block_count(path) -> int:
    """Layer (block) count from a GGUF header (``*.block_count``), 0 if unknown.
    Reads only the metadata KV section (no tensors). Cached per path."""
    if not path:
        return 0
    if path in _GGUF_META_CACHE:
        return _GGUF_META_CACHE[path]
    import struct
    result = 0
    try:
        with open(path, 'rb') as f:
            if f.read(4) != b'GGUF':
                _GGUF_META_CACHE[path] = 0
                return 0
            struct.unpack('<I', f.read(4))                  # version
            struct.unpack('<Q', f.read(8))                  # tensor count
            n_kv = struct.unpack('<Q', f.read(8))[0]

            def rd_str():
                ln = struct.unpack('<Q', f.read(8))[0]
                return f.read(ln).decode('utf-8', 'replace')

            def rd_val(vt):
                if vt == 0:  return struct.unpack('<B', f.read(1))[0]
                if vt == 1:  return struct.unpack('<b', f.read(1))[0]
                if vt == 2:  return struct.unpack('<H', f.read(2))[0]
                if vt == 3:  return struct.unpack('<h', f.read(2))[0]
                if vt == 4:  return struct.unpack('<I', f.read(4))[0]
                if vt == 5:  return struct.unpack('<i', f.read(4))[0]
                if vt == 6:  return struct.unpack('<f', f.read(4))[0]
                if vt == 7:  return struct.unpack('<?', f.read(1))[0]
                if vt == 8:  return rd_str()
                if vt == 10: return struct.unpack('<Q', f.read(8))[0]
                if vt == 11: return struct.unpack('<q', f.read(8))[0]
                if vt == 12: return struct.unpack('<d', f.read(8))[0]
                if vt == 9:
                    et = struct.unpack('<I', f.read(4))[0]
                    cnt = struct.unpack('<Q', f.read(8))[0]
                    return [rd_val(et) for _ in range(cnt)]
                raise ValueError(f"unknown gguf value type {vt}")

            for _ in range(n_kv):
                key = rd_str()
                val = rd_val(struct.unpack('<I', f.read(4))[0])
                if key.endswith('.block_count'):
                    try:
                        result = int(val)
                    except (TypeError, ValueError):
                        result = 0
                    break
    except Exception:
        result = 0
    _GGUF_META_CACHE[path] = result
    return result


def _amd_free_vram_gb(device: int = 0) -> float:
    """Free VRAM (GB) for an AMD GPU from amdgpu sysfs, 0.0 if unavailable.

    The Vulkan path runs on AMD cards where ``torch.cuda`` can't see VRAM, so the
    CUDA query returns 0 and the auto-offload sizing below is silently skipped —
    leaving ``n_gpu_layers=all`` and OOMing a model that doesn't fit (e.g. a 13 GB
    model on an 8 GB RX 580, while a 24 GB CUDA card masks the bug). Read free VRAM
    (total - used) straight from sysfs instead, indexing AMD cards in sorted order
    to match the Vulkan device index."""
    import glob, os, re
    try:
        idx = 0
        for card in sorted(glob.glob("/sys/class/drm/card*")):
            if not re.match(r"^card\d+$", os.path.basename(card)):
                continue
            dev = os.path.join(card, "device")

            def _read(rel):
                try:
                    with open(os.path.join(dev, rel)) as f:
                        return f.read().strip()
                except OSError:
                    return None
            if (_read("vendor") or "").lower() != "0x1002":   # AMD only
                continue
            if idx != device:
                idx += 1
                continue
            total = _read("mem_info_vram_total")
            used = _read("mem_info_vram_used")
            if total is None or used is None:
                return 0.0
            return max(0.0, (int(total) - int(used)) / (1024 ** 3))
    except Exception:
        return 0.0
    return 0.0


def _free_vram_gb(device: int = 0) -> float:
    """Free VRAM (GB) on the given GPU, 0.0 if unavailable.

    Tries CUDA first (NVIDIA); falls back to amdgpu sysfs so the Vulkan/AMD
    engine can size GPU-layer offload instead of always loading everything."""
    try:
        import torch
        free, _total = torch.cuda.mem_get_info(device)
        if free:
            return free / (1024 ** 3)
    except Exception:
        pass
    return _amd_free_vram_gb(device)


def _pooled_free_vram_gb(cross: bool = False) -> float:
    """Sum of free VRAM (GB) across the GPUs this engine can actually use.

    Always sums EVERY visible CUDA device — torch honours CUDA_VISIBLE_DEVICES, so
    this is automatically scoped to this engine's NVIDIA cards (e.g. both 3090s),
    making same-backend split accounting correct without any flag. AMD cards
    (amdgpu sysfs) are added when ``cross`` (cross-backend split) is on, OR when no
    CUDA device is visible (a Radeon/Vulkan engine) so its own card(s) are counted.
    CUDA and amdgpu sysfs never name the same physical card, so there's no double
    count (NVIDIA cards don't expose mem_info_vram_*)."""
    cuda_free = 0.0
    cuda_count = 0
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    free, _ = torch.cuda.mem_get_info(i)
                    cuda_free += free / (1024 ** 3)
                    cuda_count += 1
                except Exception:
                    pass
    except Exception:
        pass
    total = cuda_free
    if cross or cuda_count == 0:
        try:
            import glob
            for tp in sorted(glob.glob('/sys/class/drm/card*/device/mem_info_vram_total')):
                up = tp.replace('vram_total', 'vram_used')
                try:
                    with open(tp) as f:
                        _t = int(f.read().strip())
                    with open(up) as f:
                        _u = int(f.read().strip())
                    total += (_t - _u) / (1024 ** 3)
                except Exception:
                    pass
        except Exception:
            pass
    return total if total > 0 else _free_vram_gb(0)


def _per_device_free_vram_gb() -> list:
    """Free VRAM (GB) per device in llama.cpp's device order — CUDA devices first
    (by index), then Vulkan/AMD cards (amdgpu sysfs, sorted) — matching how
    tensor_split is interpreted. Used to auto-derive a VRAM-proportional split when
    the user enables cross-GPU pooling without specifying a ratio."""
    out = []
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    free, _ = torch.cuda.mem_get_info(i)
                    out.append(free / (1024 ** 3))
                except Exception:
                    out.append(0.0)
    except Exception:
        pass
    try:
        import glob
        for tp in sorted(glob.glob('/sys/class/drm/card*/device/mem_info_vram_total')):
            up = tp.replace('vram_total', 'vram_used')
            try:
                with open(tp) as f:
                    _t = int(f.read().strip())
                with open(up) as f:
                    _u = int(f.read().strip())
                out.append((_t - _u) / (1024 ** 3))
            except Exception:
                pass
    except Exception:
        pass
    return out


def _ggml_kv_type(name):
    """Map a KV-cache quant name to the llama.cpp GGML type int, or None.

    Returns None for falsy / unknown / 'none' / 'auto' values (→ keep the
    llama.cpp default, f16). Unknown names log a warning instead of failing."""
    if not name or _llama_cpp is None:
        return None
    key = str(name).strip().lower().replace('-', '_').replace(' ', '')
    if key in ('', 'none', 'auto', 'default', 'f16default'):
        return None
    const = _KV_TYPE_ALIASES.get(key)
    if const is None:
        print(f"  KV cache type '{name}' not recognized — using default (f16)")
        return None
    val = getattr(_llama_cpp, const, None)
    if val is None:
        print(f"  KV cache type '{name}' unsupported by this llama.cpp build — using f16")
    return val


def _install_layer_log_callback():
    """Replace llama.cpp's log callback with one that prints load-time layer/buffer
    messages directly to stdout.  Returns the callback object — keep a reference
    alive for the duration of the load so ctypes doesn't garbage-collect it."""
    if _llama_cpp is None:
        return None

    # Keywords that identify interesting load-phase messages
    _KEEP = (
        'llm_load_tensors', 'llm_load_print_meta',
        'offload', 'layer', 'buffer size', 'buffer type',
        'GPU', 'CUDA', 'Vulkan', 'Metal', 'ROCm', 'SYCL',
        'CPU', 'VRAM', 'n_layer', 'n_gpu_layers',
    )

    @_llama_cpp.llama_log_callback
    def _cb(level, text, user_data):
        try:
            msg = (text.decode('utf-8', errors='replace') if isinstance(text, bytes) else str(text)).rstrip()
            if msg and any(k in msg for k in _KEEP):
                print(f"  [llama.cpp] {msg}", flush=True)
        except Exception:
            pass

    try:
        _llama_cpp.llama_log_set(_cb, None)
    except Exception:
        return None
    return _cb  # caller must hold this reference


async def _aiter_blocking(sync_iter, lock=None):
    """Bridge a blocking (sync) generator onto the asyncio event loop.

    llama.cpp's create_(chat_)completion returns a *synchronous* generator whose
    first ``next()`` runs the whole prompt prefill and every subsequent ``next()``
    runs a full token forward pass. Iterating it directly inside an ``async def``
    runs that work on the event loop and freezes every other HTTP request (the
    whole web UI) for the duration of a completion.

    This pulls one item at a time from a worker thread via ``asyncio.to_thread``
    so the loop stays responsive between (and during, since llama.cpp releases the
    GIL while computing) token steps. Closing the async generator — e.g. on client
    disconnect or task cancellation — closes the underlying sync generator, which
    stops llama.cpp at the next token boundary, matching the old inline ``break``.
    """
    import asyncio
    _SENT = object()

    def _next():
        try:
            return next(sync_iter)
        except StopIteration:
            return _SENT

    # When a per-instance generation lock is supplied, hold it for the whole
    # iteration so a concurrent request on the same backend can't interleave its
    # forward passes into this one's KV cache. Acquired in a worker thread so a
    # contended lock doesn't block the event loop; released from here (a plain
    # threading.Lock permits cross-thread release).
    if lock is not None:
        await asyncio.to_thread(lock.acquire)
    try:
        while True:
            item = await asyncio.to_thread(_next)
            if item is _SENT:
                break
            yield item
    finally:
        close = getattr(sync_iter, "close", None)
        if close is not None:
            try:
                close()
            except Exception:
                pass
        if lock is not None:
            try:
                lock.release()
            except RuntimeError:
                pass


class VulkanBackend(ModelBackend):
    """Backend for Vulkan (AMD GPUs) using llama-cpp-python with GGUF models."""
    
    def __init__(self, original_backend: str = None):
        self.model = None
        self.model_name = None
        self.n_gpu_layers = -1  # Offload all layers to GPU by default
        self.n_ctx = 2048
        self.verbose = True
        self.main_gpu = 0  # Default to first GPU
        self.chat_template = None  # Detected chat template name
        self.hf_tokenizer = None  # HuggingFace tokenizer for apply_chat_template
        self.supports_vision = False  # set True when an mmproj projector is loaded
        self._no_system_cached = None  # lazy: template has no 'system' role (Gemma)
        self.force_cuda = original_backend in ("nvidia", "cuda")  # Force CUDA if original was nvidia
        if self.force_cuda:
            print("DEBUG: GGUF model will use CUDA backend (forced by --backend nvidia)")
        self._last_usage: dict = {}  # usage from the most recent completion call
        # Serializes the synchronous forward pass within this one instance. The
        # instance pool may hand the same backend to two concurrent requests when
        # all instances are busy; overlapping create_completion calls share one KV
        # cache and would corrupt each other. Distinct pool instances each have
        # their own lock, so they still run in parallel. A plain Lock (not RLock)
        # is used so the streaming path can acquire it in a worker thread and
        # release it from the event-loop thread.
        self._gen_lock = threading.Lock()
        self._detect_chat_template()
    
    def _detect_chat_template(self):
        """Detect the chat template used by the model."""
        try:
            # Try to get the chat template from the model
            # llama.cpp models have a chat_template attribute
            from llama_cpp.llama_chat_format import ChatFormatterResponse
            # We'll detect it when the model is loaded
            self.chat_template = "unknown"
            print("DEBUG: Chat template detection will happen after model load")
        except Exception as e:
            print(f"DEBUG: Could not initialize chat template detection: {e}")
            self.chat_template = None
    
    def _load_huggingface_tokenizer(self, template_name: str = None):
        """Load HuggingFace tokenizer for apply_chat_template support.
        
        Args:
            template_name: Optional specific template to use (e.g., 'llama3', 'chatml'). 
                          If None, will auto-detect from tokenizer.
        """
        if self.hf_tokenizer is not None:
            return  # Already loaded
        
        model_path = getattr(self, 'model_name', None)
        if not model_path:
            print("DEBUG: No model name available for HuggingFace tokenizer")
            return
        
        # If model_path is a URL, try to get the cached local path first
        if model_path.startswith('http://') or model_path.startswith('https://'):
            cached_path = get_cached_model_path(model_path)
            if cached_path and os.path.exists(cached_path):
                model_path = cached_path
                print(f"DEBUG: Using cached model path for HF tokenizer: {model_path}")
        
        try:
            from transformers import AutoTokenizer
            
            # If a specific template is provided, we can use it directly without loading tokenizer
            if template_name:
                self.chat_template = template_name
                print(f"DEBUG: Using specified chat template: {template_name}")
                # Still need to load tokenizer to get the actual template
                # but we can use the specified template name
            
            # Try to determine the model identifier
            # If model_path is a GGUF file, try to find the corresponding HF model
            if model_path.endswith('.gguf'):
                # Try to extract model name from path
                # Common patterns: .../models/llama-3.1-8b-instruct-q4_k_m.gguf
                model_dir = os.path.dirname(model_path)
                model_file = os.path.basename(model_path)
                
                # Try to find a tokenizer config in the model directory
                tokenizer_config_path = os.path.join(model_dir, 'tokenizer_config.json')
                if os.path.exists(tokenizer_config_path):
                    # Load from local directory
                    self.hf_tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
                    print(f"DEBUG: Loaded HuggingFace tokenizer from local: {model_dir}")
                    if not template_name:
                        self.chat_template = "hf_local"
                    return
                
                # Try to infer model name from file name
                # Common patterns: llama-3.1-8b-instruct-q4_k_m.gguf -> llama-3.1-8b-instruct
                # Also handle cached files with hash prefix: hash_modelname.gguf -> modelname
                model_base = model_file.replace('.gguf', '')
                
                # Remove hash prefix (64 hex chars for SHA-256 followed by underscore)
                if len(model_base) > 64 and model_base[:64].isalnum():
                    model_base = model_base[65:]  # Skip hash + underscore
                
                # Remove common quantization suffixes (case-insensitive)
                for suffix in ['_q4_k_m', '_q4_k', '_q5_k', '_q5_k_m', '_q8_0', '_f16', '_q4_0', '_q3_k_m', '_q2_k', '_Q4_K_M', '_Q4_K', '_Q5_K', '_Q5_K_M', '_Q8_0', '_F16', '_Q4_0', '_Q3_K_M', '_Q2_K']:
                    model_base = model_base.replace(suffix, '')
                
                # Try to load from HuggingFace hub
                # First try the cleaned model_base
                model_names_to_try = [model_base]
                
                # Generate shorter versions of the model name for fallback
                # E.g., Qwen3.5-27B-Uncensored-HauhauCS-Aggressive -> try shorter variants
                parts = model_base.split('-')
                if len(parts) > 1:
                    # Try progressively shorter names by removing parts from the end
                    for i in range(len(parts) - 1, 0, -1):
                        shorter_name = '-'.join(parts[:i])
                        if shorter_name and shorter_name != model_base:
                            model_names_to_try.append(shorter_name)
                
                # Also try with just the first part (e.g., "Qwen" from "Qwen3.5-27B...")
                if len(parts) > 1:
                    model_names_to_try.append(parts[0])
                
                tokenizer_loaded = False
                last_error = None
                for model_id in model_names_to_try:
                    try:
                        self.hf_tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                        print(f"DEBUG: Loaded HuggingFace tokenizer from hub: {model_id}")
                        if not template_name:
                            self.chat_template = "hf_hub"
                        tokenizer_loaded = True
                        break
                    except Exception as fallback_err:
                        # Silently skip failed tokenizer loads - we'll use manual formatting
                        pass
                
                if tokenizer_loaded:
                    return
                
                # If HF tokenizer loading failed, try to use known template names based on model name
                # This helps when we can't find the tokenizer but know the model family
                model_base_lower = model_base.lower()
                
                # Check if this looks like a Qwen model
                known_templates_to_try = []
                if 'qwen' in model_base_lower:
                    # Try known Qwen template names in order of specificity
                    if 'qwen3.5' in model_base_lower or 'qwen3' in model_base_lower:
                        known_templates_to_try = ['qwen3', 'qwen', None]  # None means use manual formatting
                    elif 'qwen2' in model_base_lower:
                        known_templates_to_try = ['qwen2', 'qwen', None]
                    else:
                        known_templates_to_try = ['qwen', None]
                elif 'llama' in model_base_lower:
                    known_templates_to_try = ['llama3', 'llama', None]
                elif 'phi' in model_base_lower:
                    known_templates_to_try = ['phi', None]
                elif 'mistral' in model_base_lower or 'mixtral' in model_base_lower:
                    known_templates_to_try = ['mistral', None]
                
                # Try each known template - directly use the template name without loading tokenizer
                # This is the key fix: instead of trying to load more non-existent tokenizers,
                # directly set the chat_template to the known template name
                for template_name in known_templates_to_try:
                    if template_name is None:
                        # No more templates to try, use manual formatting with generic format
                        self.chat_template = "chatml"  # Use ChatML as generic fallback
                        print(f"DEBUG: No known templates worked, using generic ChatML format")
                        break
                    # Directly use this known template name - no need to load tokenizer
                    # The manual formatting will use <|im_start|> tags which work for most models
                    self.chat_template = template_name
                    print(f"DEBUG: Using known template '{template_name}' for model family detection")
                    # Successfully set template - don't try to load tokenizer
                    break
                
                if self.chat_template:
                    return
                
                # All attempts failed - warn but continue without template
                print(f"Warning: Could not load HuggingFace tokenizer for any variant of '{model_base}'")
                print(f"Warning: Will not use apply_chat_template - model will use manual formatting")
                self.chat_template = None
            else:
                # Not a GGUF file, try to load directly
                self.hf_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                print(f"DEBUG: Loaded HuggingFace tokenizer from: {model_path}")
                if not template_name:
                    self.chat_template = "hf"
                return
                    
        except ImportError as e:
            print(f"DEBUG: transformers not installed, cannot use HuggingFace chat template: {e}")
            self.chat_template = None
        except Exception as e:
            print(f"DEBUG: Failed to load HuggingFace tokenizer: {e}")
            self.chat_template = None
    
    def _finalize_chat_template_detection(self):
        """Finalize chat template detection after model is loaded."""
        # Check if the loaded GGUF model has a built-in chat template
        if hasattr(self, 'model') and self.model is not None:
            try:
                # llama.cpp models have a chat_template attribute
                if hasattr(self.model, 'chat_template'):
                    template = self.model.chat_template
                    if template:
                        print(f"DEBUG: Using GGUF model's built-in chat template")
                        # Extract template name if possible
                        template_str = str(template)
                        if 'llama' in template_str.lower():
                            self.chat_template = 'llama3'
                        elif 'qwen' in template_str.lower():
                            self.chat_template = 'qwen3'
                        elif 'chatml' in template_str.lower():
                            self.chat_template = 'chatml'
                        elif 'mistral' in template_str.lower():
                            self.chat_template = 'mistral'
                        else:
                            self.chat_template = 'builtin'
                        print(f"DEBUG: chat_template set to: {self.chat_template}")
                        return  # Use built-in template
            except Exception as e:
                print(f"DEBUG: Could not get chat template from GGUF model: {e}")
        
        # Fallback: Try to load HuggingFace tokenizer for chat template
    
    def _apply_chat_template(self, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
        """Apply chat template to messages.
        
        Tries multiple methods in order:
        1. GGUF model's built-in chat template (via llama.cpp)
        2. HuggingFace tokenizer's apply_chat_template
        3. Manual template application based on detected template name
        4. Generic ChatML format as fallback
        
        Args:
            messages: List of message dicts or ChatMessage objects
            add_generation_prompt: Whether to add generation prompt
        """
        # Convert ChatMessage objects to dicts if needed
        if messages and hasattr(messages[0], 'model_dump'):
            # It's a Pydantic model
            messages = [msg.model_dump() if hasattr(msg, 'model_dump') else {"role": msg.role, "content": msg.content} for msg in messages]
        # First try GGUF model's built-in template
        if hasattr(self, 'model') and self.model is not None:
            try:
                if hasattr(self.model, 'create_chat_completion'):
                    # llama.cpp can handle chat templates internally
                    # We'll use it to validate the format but return our own formatted string
                    pass  # Fall through to use template directly
            except Exception as e:
                print(f"DEBUG: GGUF model template check failed: {e}")
        
        # Try HuggingFace tokenizer
        if self.hf_tokenizer:
            try:
                # Check if tokenizer has apply_chat_template
                if hasattr(self.hf_tokenizer, 'apply_chat_template'):
                    # Use the tokenizer's chat template
                    # For ChatML/Others: add_generation_prompt=True adds the assistant token
                    # For Llama3: adds the correct prefix
                    rendered = self.hf_tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=add_generation_prompt
                    )
                    print(f"DEBUG: Applied HuggingFace chat template")
                    return rendered
            except Exception as e:
                print(f"DEBUG: HuggingFace apply_chat_template failed: {e}")
        
        # Try llama.cpp's built-in chat template
        if hasattr(self, 'model') and self.model is not None:
            try:
                if hasattr(self.model, 'chat'):
                    # Get the chat template from llama.cpp
                    # This works for models that have chat templates defined
                    from llama_cpp.llama_chat_format import ChatCompletionMessage
                    
                    # Try to use the model's chat handler
                    # llama.cpp's create_chat_completion handles templates internally
                    # But we need raw text for streaming
                    pass  # Fall through to manual
            except Exception as e:
                print(f"DEBUG: llama.cpp chat handling failed: {e}")
        
        # Manual template application based on detected template
        template_name = self.chat_template or "chatml"
        
        # Format messages based on template name
        if template_name in ("llama3", "llama-3", "llama-3.1"):
            return self._format_llama3(messages, add_generation_prompt)
        elif template_name in ("qwen2", "qwen2.5", "qwen3"):
            return self._format_qwen(messages, add_generation_prompt)
        elif template_name == "chatml":
            return self._format_chatml(messages, add_generation_prompt)
        elif template_name in ("mistral", "mixtral"):
            return self._format_mistral(messages, add_generation_prompt)
        else:
            # Default to ChatML format
            return self._format_chatml(messages, add_generation_prompt)
    
    def _format_llama3(self, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
        """Format messages for Llama 3.x models."""
        result = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                result.append(f"<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>")
            elif role == "user":
                result.append(f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>")
            elif role == "assistant":
                result.append(f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>")
        
        if add_generation_prompt:
            result.append(f"<|start_header_id|>assistant<|end_header_id|>\n\n")
        
        return "".join(result)
    
    def _format_qwen(self, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
        """Format messages for Qwen models."""
        result = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                result.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                result.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                result.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        
        if add_generation_prompt:
            result.append(f"<|im_start|>assistant\n")
        
        return "".join(result)
    
    def _format_chatml(self, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
        """Format messages using ChatML format."""
        result = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                result.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                result.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                result.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        
        if add_generation_prompt:
            result.append(f"<|im_start|>assistant\n")
        
        return "".join(result)
    
    def _format_mistral(self, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
        """Format messages for Mistral/Mixtral models."""
        result = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                result.append(f"[INST] <<SYS>>\n{content}\n<</SYS>>\n\n")
            elif role == "user":
                result.append(f"[INST]{content}[/INST]")
            elif role == "assistant":
                result.append(f"{content}")
        
        # Mistral uses a different format - the last user message has [/INST]
        # and we add the assistant prefix if generation prompt is needed
        if add_generation_prompt:
            # Find the last user message and add the assistant prefix after it
            # Actually, let's reformat more carefully
            pass  # Fall through to simpler format
        
        # Simpler approach: join with newlines
        formatted = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                formatted += f"<<SYS>>\n{content}\n<</SYS>>\n\n"
            elif role == "user":
                formatted += f"[INST] {content} [/INST]"
            elif role == "assistant":
                formatted += f" {content}"
        
        if add_generation_prompt:
            formatted += " "
        
        return formatted
    
    def load_model(self, model_path: str, model_type: str = "text", **kwargs) -> None:
        """Load a GGUF model.
        
        Args:
            model_path: Path to the GGUF model file, HuggingFace model ID, or URL
            model_type: Type of model (text, image, audio)
            **kwargs: Additional parameters
        """
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError("llama-cpp-python is required for GGUF models. Install with: pip install llama-cpp-python")

        # A bare alias / basename (e.g. 'coe-…-q4_k_m', no '.gguf', not an on-disk
        # path) may still name a *local* gguf — a configured model addressed by its
        # automatic alias. Resolve it before assuming it's a HuggingFace repo to
        # download (which 404s and fails the load).
        if not model_path.endswith('.gguf') and not os.path.exists(model_path) \
                and not model_path.startswith('http'):
            try:
                from codai.models.manager import _resolve_local_gguf
                _local = _resolve_local_gguf(model_path)
                if _local:
                    print(f"  Resolved '{model_path}' to local GGUF: {_local}")
                    model_path = _local
            except Exception as _rg_e:
                print(f"  (local GGUF resolve skipped: {_rg_e})")

        # Check if this looks like a GGUF model repo (e.g., "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF")
        is_gguf_repo = not model_path.endswith('.gguf') and not os.path.exists(model_path) and not model_path.startswith('http')
        
        if is_gguf_repo:
            # Try to find GGUF files in the repository
            try:
                from huggingface_hub import list_repo_files, hf_hub_download
                from codai.models.cache import get_hf_hub_cache_dir
                print(f"DEBUG: Searching for GGUF files in {model_path}...")
                files = list(list_repo_files(model_path, repo_type="model"))
                gguf_files = [f for f in files if f.lower().endswith('.gguf')]

                if gguf_files:
                    # Prefer Q4_K_M or Q4_K quantizations, otherwise use first available
                    preferred = [f for f in gguf_files if 'q4_k_m' in f.lower() or 'q4_k' in f.lower()]
                    selected = preferred[0] if preferred else gguf_files[0]
                    print(f"DEBUG: Found GGUF files: {gguf_files}")
                    print(f"DEBUG: Selected: {selected}")
                    model_path = hf_hub_download(repo_id=model_path, filename=selected, cache_dir=kwargs.get('cache_dir') or get_hf_hub_cache_dir())
                    print(f"DEBUG: Downloaded: {model_path}")
                else:
                    print(f"Warning: No GGUF files found in {model_path}, trying direct download...")
            except Exception as e:
                print(f"Warning: Could not search HuggingFace repo: {e}")
        # If it's a HuggingFace model ID (not ending in .gguf), try to download
        elif not model_path.endswith('.gguf') and not os.path.exists(model_path):
            # Try to get from HuggingFace
            try:
                from huggingface_hub import hf_hub_download
                from codai.models.cache import get_hf_hub_cache_dir
                # Download the GGUF file
                model_path = hf_hub_download(repo_id=model_path, filename="*.gguf", cache_dir=kwargs.get('cache_dir') or get_hf_hub_cache_dir())
            except Exception as e:
                print(f"Warning: Could not download from HuggingFace: {e}")
                # Try as-is
        
        # If it's a URL, download it
        if model_path.startswith('http://') or model_path.startswith('https://'):
            cached_path = get_cached_model_path(model_path)
            if cached_path:
                model_path = cached_path
            else:
                raise ValueError(f"Could not cache model from URL: {model_path}")
        
        # Fallback: a configured .gguf path that no longer exists (e.g. the file was
        # downloaded into the GGUF cache rather than the HF-hub snapshot the entry
        # points at, or a stale snapshot hash). Look for the same filename in the
        # GGUF cache dir before giving up — the model loads without re-editing the
        # config entry.
        if model_path.endswith('.gguf') and not os.path.exists(model_path):
            try:
                from codai.models.cache import get_model_cache_dir
                _base = os.path.basename(model_path)
                _cache = get_model_cache_dir()
                _cand = os.path.join(_cache, _base)
                if not os.path.exists(_cand):
                    import glob as _glob
                    _hits = _glob.glob(os.path.join(_cache, "**", _base), recursive=True)
                    _cand = _hits[0] if _hits else _cand
                if os.path.exists(_cand):
                    print(f"  Model path missing; resolved from GGUF cache: {_cand}")
                    model_path = _cand
            except Exception:
                pass

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model_name = model_path
        
        # Determine model type
        is_image = model_type == "image" or model_path.startswith("image:")
        
        # Check for --no-ram mode from global args
        no_ram = kwargs.get('no_ram', False)
        if not no_ram:
            try:
                from codai.api.state import get_global_args
                _global_args = get_global_args()
                if _global_args and getattr(_global_args, 'no_ram', False):
                    no_ram = True
            except Exception:
                pass
        
        # Configure GPU layers
        n_gpu_layers = kwargs.get('n_gpu_layers', -1)
        if no_ram:
            # --no-ram: force all layers on GPU
            self.n_gpu_layers = -1
        elif n_gpu_layers != -1:
            self.n_gpu_layers = n_gpu_layers
        else:
            # Auto (n_gpu_layers == -1): if the whole model won't fit in free VRAM,
            # place as many layers on GPU as fit and leave the rest on CPU instead
            # of trying to load everything and OOMing ("failed to create
            # llama_context"). llama.cpp has no auto-fit, so we size it ourselves.
            try:
                _exp = kwargs.get('expected_vram_gb')
                _nlayers = _gguf_block_count(model_path)
                # Usable VRAM is the POOL across the cards this engine can use, not
                # just main_gpu: all same-backend cards always (e.g. both 3090s), plus
                # the other backend's cards when cross-split is on (3090 + RX 580).
                # Otherwise we'd needlessly offload layers to CPU thinking only one
                # card's free VRAM is available.
                _free = _pooled_free_vram_gb(cross=bool(kwargs.get('gpu_split')))
                # A vision projector (mmproj) loads ON TOP of the model on main_gpu;
                # subtract it (weights + compute margin) from the usable pool so that
                # when the model+KV+projector exceed BOTH cards combined, we reduce
                # GPU layers (spill to CPU) gracefully instead of filling both cards
                # and aborting (GGML_ASSERT(buffer) failed) when the projector can't
                # allocate.
                _mmp = kwargs.get('mmproj') or (kwargs.get('_raw_cfg') or {}).get('mmproj')
                if _mmp:
                    try:
                        _mp = os.path.expanduser(str(_mmp))
                        _msz = os.path.getsize(_mp) / (1024 ** 3) if os.path.isfile(_mp) else 1.5
                    except Exception:
                        _msz = 1.5
                    _free = max(0.0, _free - (_msz + 1.5))
                if _exp and _exp > 0 and _nlayers and _free > 0 and _exp > _free * 0.95:
                    # Scale layers on GPU by the VRAM ratio (weights + KV roughly
                    # scale per-layer). The estimate tends to undercount the KV
                    # cache at large n_ctx, and a few GB of compute/output buffers
                    # stay on GPU regardless — so reserve a context-scaled headroom
                    # and inflate the need, to err toward fitting (CPU layers are
                    # slow but a failed load is worse).
                    _headroom = 2.0 + (self.n_ctx or 0) / 12000.0   # ~2 GB + ~1 GB per 12k ctx
                    _usable = max(0.0, _free - _headroom)
                    _fit = int(_nlayers * _usable / (_exp * 1.20))
                    _fit = max(0, min(_nlayers - 1, _fit))
                    self.n_gpu_layers = _fit
                    print(f"  Auto-offload: model needs ~{_exp:.1f} GB but only "
                          f"{_free:.1f} GB free — placing {_fit}/{_nlayers} layers on "
                          f"GPU, {_nlayers - _fit} on CPU (slower). Lower n_ctx or use a "
                          f"smaller model to keep it fully on GPU.", flush=True)
            except Exception as _off_e:
                print(f"  (auto-offload sizing skipped: {_off_e})", flush=True)

        # Configure context size
        if no_ram:
            # --no-ram: ignore --n-ctx, let the model use its own default
            self.n_ctx = 0  # 0 means use model's built-in default in llama.cpp
            print("DEBUG: --no-ram mode: ignoring --n-ctx, using model default context size")
        else:
            # Accept either 'n_ctx' (models.json / GGUF) or 'ctx' (CLI / older
            # configs); the manager passes both, but be robust to either alone.
            n_ctx = kwargs.get('n_ctx')
            if n_ctx is None:
                n_ctx = kwargs.get('ctx', 2048)
            self.n_ctx = n_ctx
        
        # Set verbose
        self.verbose = kwargs.get('verbose', True)
        
        # Set main GPU
        self.main_gpu = kwargs.get('main_gpu', 0)
        
        # Build kwargs for Llama constructor
        llama_kwargs = {
            'model_path': model_path,
            'n_gpu_layers': self.n_gpu_layers,
            'n_ctx': self.n_ctx,
            'verbose': self.verbose,
            'main_gpu': self.main_gpu,
        }
        
        # --no-ram: disable mmap to prevent CPU RAM usage for memory-mapped files
        if no_ram:
            llama_kwargs['use_mmap'] = False
            print("DEBUG: --no-ram mode: use_mmap=False, n_gpu_layers=-1")
        
        # Add optional parameters
        if 'n_threads' in kwargs:
            llama_kwargs['n_threads'] = kwargs['n_threads']
        if 'n_threads_batch' in kwargs:
            llama_kwargs['n_threads_batch'] = kwargs['n_threads_batch']
        if 'rope_freq_base' in kwargs:
            llama_kwargs['rope_freq_base'] = kwargs['rope_freq_base']
        if 'rope_freq_scale' in kwargs:
            llama_kwargs['rope_freq_scale'] = kwargs['rope_freq_scale']

        # KV-cache quantization (llama.cpp type_k / type_v). Shrinks the KV cache
        # so long contexts fit in less VRAM. Read from the per-model config, with
        # the raw models.json entry as a fallback (carried in _raw_cfg).
        _raw_cfg = kwargs.get('_raw_cfg') or {}
        _ck = kwargs.get('cache_type_k', _raw_cfg.get('cache_type_k'))
        _cv = kwargs.get('cache_type_v', _raw_cfg.get('cache_type_v'))
        _flash = bool(kwargs.get('flash_attn', _raw_cfg.get('flash_attn',
                      _raw_cfg.get('flash_attention', False))))
        _tk = _ggml_kv_type(_ck)
        _tv = _ggml_kv_type(_cv)
        if _tk is not None:
            llama_kwargs['type_k'] = _tk
        if _tv is not None:
            llama_kwargs['type_v'] = _tv
        # A quantized V cache below 8 bits requires flash attention in llama.cpp;
        # auto-enable it (with a note) so the config "just works".
        _v_needs_flash = str(_cv or '').strip().lower().replace('-', '_') in _KV_NEEDS_FLASH
        if (_tk is not None or _tv is not None):
            if _v_needs_flash and not _flash:
                _flash = True
                print("  KV cache: sub-8-bit V cache needs flash attention — enabling it")
            if _flash:
                llama_kwargs['flash_attn'] = True
            print(f"  KV cache: type_k={_ck or 'f16'}  type_v={_cv or 'f16'}"
                  f"{'  (flash_attn on)' if _flash else ''}")

        # KV-cache offload target. Default (True) keeps the KV cache in VRAM. Set
        # kv_offload=false on a model to keep it in *host RAM* instead (llama.cpp
        # --no-kv-offload) — frees a lot of VRAM for big contexts (a 256k KV can be
        # several GB) at the cost of slower decode, since KV ops then cross PCIe.
        # llama.cpp has no SSD/disk KV paging, so RAM is the only off-GPU option.
        _kv_off = kwargs.get('kv_offload', _raw_cfg.get('kv_offload',
                  _raw_cfg.get('offload_kqv')))
        if _kv_off is not None and not bool(_kv_off):
            llama_kwargs['offload_kqv'] = False
            print("  KV cache: offload_kqv=False — KV held in host RAM (saves VRAM, "
                  "slower decode)")

        # Batch size (llama.cpp -b / n_batch) and physical micro-batch (-ub /
        # n_ubatch). The compute/graph buffer reserved for prompt ingestion scales
        # with the micro-batch, so lowering it shrinks a large VRAM allocation (the
        # buffer that ggml_gallocr fails to reserve when a big-context model is right
        # at the VRAM ceiling) at the cost of slower prefill. llama.cpp clamps
        # n_ubatch to <= n_batch, so setting n_batch alone also caps the micro-batch.
        _n_batch = kwargs.get('n_batch', _raw_cfg.get('n_batch'))
        if _n_batch:
            try:
                llama_kwargs['n_batch'] = int(_n_batch)
                print(f"  batch        : n_batch={int(_n_batch)} (smaller prompt-ingest buffer)")
            except (TypeError, ValueError):
                pass
        _n_ubatch = kwargs.get('n_ubatch', _raw_cfg.get('n_ubatch'))
        if _n_ubatch:
            try:
                llama_kwargs['n_ubatch'] = int(_n_ubatch)
            except (TypeError, ValueError):
                pass

        # Parallel sequence slots (llama.cpp -np / n_seq_max). Each slot reserves its
        # own share of KV + compute VRAM; keeping it at 1 avoids reserving VRAM for
        # concurrent sequences we don't serve. Only newer llama-cpp-python builds
        # accept this kwarg, so pass it only when the constructor supports it.
        _n_seq = kwargs.get('n_seq_max', _raw_cfg.get('n_seq_max'))
        if _n_seq:
            try:
                _n_seq = int(_n_seq)
            except (TypeError, ValueError):
                _n_seq = None
            if _n_seq:
                if _llama_accepts('n_seq_max'):
                    llama_kwargs['n_seq_max'] = _n_seq
                    print(f"  slots        : n_seq_max={_n_seq} (fewer parallel-slot VRAM reserves)")
                else:
                    print(f"  slots        : n_seq_max={_n_seq} requested but llama-cpp-python "
                          f"{getattr(_llama_cpp, '__version__', '?')} doesn't expose it — ignoring")

        # Cross-GPU layer split (opt-in, per-model). gpu_split=True lets llama.cpp
        # spread this model's layers across multiple visible GPU devices (e.g. an
        # NVIDIA 3090 via CUDA + a Radeon via Vulkan) for more total VRAM. The
        # engine's device visibility is set by the front (GGML_VK_VISIBLE_DEVICES /
        # CUDA_VISIBLE_DEVICES); here we only choose HOW to distribute:
        #   - gpu_split off → split_mode NONE: keep the whole model on main_gpu (one
        #     card), so a foreign card the engine can see isn't used by accident.
        #   - gpu_split on  → split_mode LAYER + optional tensor_split ratio
        #     ("0.8,0.2" in llama.cpp device order: CUDA devices first, then Vulkan).
        # By default (gpu_split off) we DON'T touch split_mode: llama.cpp's default
        # LAYER split spreads the model across the engine's visible cards — which the
        # front has already isolated to this engine's own backend (e.g. both 3090s),
        # never the foreign Radeon. When gpu_split is on AND a tensor_split ratio is
        # given, we set the per-device ratio (llama.cpp device order: CUDA first,
        # then Vulkan), e.g. "0.8,0.2" = 80% on the 3090, 20% on the RX 580.
        _gpu_split = kwargs.get('gpu_split', _raw_cfg.get('gpu_split'))
        if _gpu_split is None:
            _gpu_split = bool(kwargs.get('_global_gpu_split', False))
        _ts = kwargs.get('tensor_split', _raw_cfg.get('tensor_split'))
        if _ts is None:
            _ts = kwargs.get('_global_tensor_split')
        # llama.cpp device order: all visible CUDA devices first, then Vulkan. Count
        # each so we can tell own-backend from foreign when both are exposed (the
        # engine was spawned with cross-GPU visibility for some split model).
        _n_cuda = 0
        try:
            import torch as _t
            if _t.cuda.is_available():
                _n_cuda = _t.cuda.device_count()
        except Exception:
            _n_cuda = 0
        _vk_env = os.environ.get('GGML_VK_VISIBLE_DEVICES', '')
        _n_vk = len([x for x in _vk_env.split(',') if x.strip() != '']) if _vk_env else 0
        _own = (os.environ.get('CODERAI_ENGINE_BACKEND') or '').lower()
        if _gpu_split:
            _parsed = []
            if _ts:
                for _p in str(_ts).replace(' ', '').split(','):
                    if _p == '':
                        continue
                    try:
                        _parsed.append(float(_p))
                    except ValueError:
                        _parsed = []
                        break
            if not _parsed:
                # Auto: distribute proportionally to each device's free VRAM (a 24 GB
                # 3090 + 8 GB RX 580 → ~0.75/0.25), so the bigger card carries more.
                _free_dev = _per_device_free_vram_gb()
                # mmproj-aware headroom: a vision projector (CLIP/mmproj) ALWAYS loads
                # on main_gpu (its own buffer + compute), and the model's output/KV
                # anchor is there too. If the proportional split fills main_gpu to the
                # brim, the projector can't allocate and llama.cpp aborts
                # (GGML_ASSERT(buffer) failed). So reserve the projector's size + a
                # compute margin on main_gpu by subtracting it from that device's free
                # VRAM before computing the ratio — main_gpu then gets a smaller share.
                _mg = self.main_gpu if isinstance(self.main_gpu, int) else 0
                _reserve = 0.0
                _mmp = kwargs.get('mmproj', _raw_cfg.get('mmproj'))
                if _mmp:
                    try:
                        _mp = os.path.expanduser(str(_mmp))
                        _sz = os.path.getsize(_mp) / (1024 ** 3) if os.path.isfile(_mp) else 1.5
                    except Exception:
                        _sz = 1.5
                    _reserve = _sz + 1.5   # projector weights + its compute/KV margin
                _adj = list(_free_dev)
                if _reserve > 0 and 0 <= _mg < len(_adj):
                    _adj[_mg] = max(0.5, _adj[_mg] - _reserve)
                # Split strategy: "vram" (default) = proportional to free VRAM (max
                # capacity); "performance" = fill the FAST lead card (main_gpu) first
                # and spill only the overflow to the slower card(s), so the weak GPU
                # holds the fewest layers and gates throughput the least.
                _strategy = (kwargs.get('split_strategy', _raw_cfg.get('split_strategy'))
                             or kwargs.get('_global_split_strategy') or 'vram')
                _strategy = str(_strategy).lower()
                _exp_total = kwargs.get('expected_vram_gb')
                if (_strategy == 'performance' and len(_adj) > 1
                        and _exp_total and _exp_total > 0):
                    _w = [0.0] * len(_adj)
                    _main_cap = _adj[_mg]
                    if _exp_total <= _main_cap:
                        _w[_mg] = 1.0   # fits on the fast card alone — no real spill
                    else:
                        _overflow = _exp_total - _main_cap
                        _others = sum(_adj[i] for i in range(len(_adj)) if i != _mg)
                        _w[_mg] = _main_cap
                        for i in range(len(_adj)):
                            if i != _mg and _others > 0:
                                _w[i] = _overflow * (_adj[i] / _others)
                    _sw = sum(_w)
                    if _sw > 0:
                        _parsed = [round(x / _sw, 3) for x in _w]
                        print(f"  gpu split    : PERFORMANCE tensor_split={_parsed} "
                              f"(fast card GPU{_mg} first; free {[round(f,1) for f in _free_dev]} GB, "
                              f"model ~{_exp_total:.1f} GB)")
                if not _parsed:
                    _sum = sum(_adj)
                    if len(_adj) > 1 and _sum > 0:
                        _parsed = [round(f / _sum, 3) for f in _adj]
                        _note = (f" (reserved ~{_reserve:.1f} GB on GPU{_mg} for mmproj)"
                                 if _reserve > 0 else "")
                        print(f"  gpu split    : VRAM tensor_split={_parsed} (by free VRAM "
                              f"{[round(f,1) for f in _free_dev]} GB){_note}")
            else:
                print(f"  gpu split    : tensor_split={_parsed} (multi-GPU pool)")
            if _parsed and _llama_accepts('tensor_split'):
                llama_kwargs['tensor_split'] = _parsed
        elif _n_cuda > 0 and _n_vk > 0 and _llama_accepts('tensor_split'):
            # NOT a split model, but the engine can see BOTH backends' cards (cross
            # visibility was enabled for some other split model). Confine THIS model
            # to its own backend so it doesn't accidentally spread onto the foreign
            # card — while still letting multiple SAME-backend cards share. Zero the
            # foreign-backend devices in the per-device ratio.
            if _own.startswith('nvidia') or _own in ('cuda',):
                _conf = [1.0] * _n_cuda + [0.0] * _n_vk          # CUDA only
            else:
                _conf = [0.0] * _n_cuda + [1.0] * _n_vk          # Vulkan only
            llama_kwargs['tensor_split'] = _conf
            print(f"  gpu split    : confined to own backend ({_own or 'vulkan'}); "
                  f"tensor_split={_conf} (foreign card excluded)")

        # Multimodal projector (mmproj): pairs a CLIP/vision projector GGUF with
        # this text model so it can accept images — the llama.cpp `--mmproj`
        # equivalent, which adds vision capability (e.g. gemma). Uses llama.cpp's
        # unified mtmd handler, which auto-detects the projector type from the file.
        self.supports_vision = False
        _mmproj = kwargs.get('mmproj', _raw_cfg.get('mmproj'))
        if _mmproj:
            _mmproj_path = os.path.expanduser(str(_mmproj))
            if not os.path.isfile(_mmproj_path):
                # Bare filename / moved cache → look beside the model file.
                _cand = os.path.join(os.path.dirname(model_path),
                                     os.path.basename(_mmproj_path))
                if os.path.isfile(_cand):
                    _mmproj_path = _cand
            if os.path.isfile(_mmproj_path):
                try:
                    from llama_cpp.llama_chat_format import MTMDChatHandler
                    llama_kwargs['chat_handler'] = MTMDChatHandler(
                        clip_model_path=_mmproj_path,
                        verbose=False,
                        use_gpu=(self.n_gpu_layers != 0),
                    )
                    self.supports_vision = True
                    print(f"  mmproj       : {os.path.basename(_mmproj_path)} (vision enabled)")
                except Exception as _e:
                    print(f"  mmproj       : failed to load projector ({_e}); continuing text-only")
            else:
                print(f"  mmproj       : configured path not found ({_mmproj}); skipping")

        # Force CUDA if requested
        if self.force_cuda:
            # Set environment variable to force CUDA
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            # llama-cpp-python will use CUDA when available
        
        # Pre-load summary
        gpu_label = "all" if self.n_gpu_layers == -1 else str(self.n_gpu_layers)
        print(f"  n_gpu_layers : {gpu_label}  |  n_ctx : {self.n_ctx}  |  main_gpu : {self.main_gpu}")
        if _llama_cpp:
            gpu_supported = _llama_cpp.llama_supports_gpu_offload()
            print(f"  GPU offload  : {'supported' if gpu_supported else 'NOT supported by this build'}")

        _log_cb = _install_layer_log_callback()
        # Progress feedback during the (otherwise silent) tensor load. llama.cpp's
        # progress_callback isn't exposed by the Llama wrapper, so inject it by
        # patching the default model-params factory for the duration of construction.
        # Kept alive in a local for the whole load (avoids a ctypes use-after-free).
        _prog = {'last': -5, 't0': time.time()}

        @_llama_cpp.llama_progress_callback
        def _progress_cb(progress, user_data):
            try:
                pct = int(progress * 100)
                if pct >= _prog['last'] + 5 or pct >= 100:
                    _prog['last'] = pct
                    print(f"  Loading model into VRAM/RAM: {pct}%"
                          f" ({time.time() - _prog['t0']:.0f}s)", flush=True)
            except Exception:
                pass
            return True

        # Patch the SUBMODULE attribute — llama.py does `import llama_cpp.llama_cpp
        # as llama_cpp` and builds model_params from it, so the top-level package
        # attribute is not what it looks up.
        _params_mod = getattr(_llama_cpp, 'llama_cpp', _llama_cpp)
        _orig_params = _params_mod.llama_model_default_params
        def _params_with_progress():
            p = _orig_params()
            try:
                p.progress_callback = _progress_cb
            except Exception:
                pass
            return p
        _params_mod.llama_model_default_params = _params_with_progress
        try:
            self.model = Llama(**llama_kwargs)
        except Exception as e:
            print(f"Error loading GGUF model: {e}")
            raise
        finally:
            _params_mod.llama_model_default_params = _orig_params
            # Quiet logging after load — but DO NOT drop to NULL + GC the callback.
            # ggml keeps the log-callback pointer and may still invoke it during
            # generation (e.g. gemma's iSWA hybrid cache logs every step), so a
            # garbage-collected ctypes callback becomes a use-after-free → SIGSEGV
            # in libffi. Install a persistent no-op callback and keep a strong
            # reference on self for the model's lifetime.
            if _llama_cpp:
                try:
                    @_llama_cpp.llama_log_callback
                    def _quiet_log_cb(level, text, user_data):
                        pass
                    _llama_cpp.llama_log_set(_quiet_log_cb, None)
                    self._log_cb = _quiet_log_cb   # keep alive (prevents GC/UAF)
                except Exception:
                    self._log_cb = None
            _log_cb = None  # the verbose load-phase callback is no longer referenced

        # Post-load layer/buffer summary
        try:
            n_total = _llama_cpp.llama_model_n_layer(self.model.model)
            n_gpu_actual = n_total if self.n_gpu_layers == -1 else min(self.n_gpu_layers, n_total)
            n_cpu = n_total - n_gpu_actual
            print(f"  Layers total : {n_total}")
            print(f"  Layers → GPU : {n_gpu_actual}  |  Layers → CPU : {n_cpu}")
        except Exception:
            pass

        # Multi-slot prefix cache. llama-cpp-python's LlamaRAMCache keeps several
        # past sequences' KV states in host RAM and, on each completion, reloads
        # the one sharing the longest token prefix with the new prompt (see
        # Llama._create_completion). This lets two interleaved conversations both
        # stay "warm" instead of evicting each other — only the changed suffix is
        # re-evaluated. Bounded by a per-model byte budget so it can't grow without
        # limit (the states live in CPU RAM, copied back on a slot switch).
        # kv_cache_budget_mb honours kwargs first, then the raw models.json entry,
        # mirroring how cache_type_k / flash_attn are resolved above. <=0 disables
        # it; unset falls back to a sensible default.
        _DEFAULT_CACHE_MB = 512
        _budget_mb = kwargs.get('kv_cache_budget_mb', _raw_cfg.get('kv_cache_budget_mb'))
        self._setup_prefix_cache(_budget_mb if _budget_mb is not None else _DEFAULT_CACHE_MB)

        # Try to detect and set up chat template
        self._finalize_chat_template_detection()
        print(f"  chat_template: {self.chat_template}")

    def _setup_prefix_cache(self, budget_mb) -> None:
        """Attach a multi-slot LlamaRAMCache sized to ``budget_mb`` megabytes."""
        try:
            budget = max(0, int(budget_mb)) * 1024 * 1024
        except (TypeError, ValueError):
            budget = 512 * 1024 * 1024
        self._prefix_cache_budget = budget
        if budget <= 0:
            print("  Prefix cache : disabled (kv_cache_budget_mb=0)")
            return
        try:
            from llama_cpp import LlamaRAMCache
            self.model.set_cache(LlamaRAMCache(capacity_bytes=budget))
            print(f"  Prefix cache : multi-slot RAM cache, budget {budget // (1024*1024)} MB")
        except Exception as e:
            print(f"  Prefix cache : could not enable ({e}); relying on single-slot prefix match")

    def clear_prefix_cache(self) -> None:
        """Release the host-RAM prefix cache (called by the RAM-pressure ladder).

        Re-attaches a fresh, empty LlamaRAMCache at the same budget so caching
        keeps working after the reclaim — only the stored sequences are dropped.
        """
        budget = getattr(self, '_prefix_cache_budget', 0)
        if not budget or self.model is None:
            return
        try:
            from llama_cpp import LlamaRAMCache
            with self._gen_lock:
                self.model.set_cache(LlamaRAMCache(capacity_bytes=budget))
        except Exception:
            pass

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        repeat_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0
    ) -> str:
        """Generate text non-streaming.
        
        Args:
            prompt: Input prompt (or messages for chat)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            stop: Stop sequences
            grammar: Optional GBNF grammar string for constrained generation
            repeat_penalty: Repetition penalty (1.0 = no penalty)
            presence_penalty: Presence penalty
            frequency_penalty: Frequency penalty
            
        Returns:
            Generated text
        """
        kwargs = {}
        if max_tokens is not None:
            kwargs['max_tokens'] = max_tokens
        if temperature is not None:
            kwargs['temperature'] = temperature
        if top_p is not None:
            kwargs['top_p'] = top_p
        if stop is not None:
            kwargs['stop'] = stop
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Check if prompt looks like messages (list of dicts)
        if isinstance(prompt, list):
            prompt = self._apply_chat_template(prompt, add_generation_prompt=True)
        
        # Set defaults
        max_tokens = kwargs.get('max_tokens', 256)
        temperature = kwargs.get('temperature', 0.7)
        top_p = kwargs.get('top_p', 0.9)
        top_k = kwargs.get('top_k', 40)
        repeat_penalty = kwargs.get('repeat_penalty', 1.1)
        stream = kwargs.get('stream', False)
        
        # Get stop strings
        stop = kwargs.get('stop', None)
        if stop is None:
            # Get default stop tokens based on template
            stop = get_reasoning_stop_tokens(self.chat_template)
        
        # Check for grammar-guided generation
        use_grammar = grammar
        if use_grammar is None:
            # Check global flag
            try:
                import coderai
                if getattr(coderai, 'grammar_guided_gen', False):
                    use_grammar = get_tool_call_grammar()
            except (ImportError, AttributeError):
                pass
        
        try:
            with self._gen_lock:
                result = self.model.create_completion(
                    stopping_criteria=_make_llama_thermal_criteria(),
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repeat_penalty=repeat_penalty,
                    stop=stop,
                    grammar=use_grammar,
                )
            usage = result.get('usage', {})
            self._store_usage(usage.get('prompt_tokens', 0), usage.get('completion_tokens', 0))
            return result['choices'][0]['text']
        except Exception as e:
            # If grammar generation fails, fall back to normal generation
            if use_grammar:
                print(f"Warning: Grammar-guided generation failed: {e}, falling back to normal generation")
                try:
                    with self._gen_lock:
                        result = self.model.create_completion(
                            stopping_criteria=_make_llama_thermal_criteria(),
                            prompt=prompt,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            repeat_penalty=repeat_penalty,
                            stop=stop,
                        )
                    usage = result.get('usage', {})
                    self._store_usage(usage.get('prompt_tokens', 0), usage.get('completion_tokens', 0))
                    return result['choices'][0]['text']
                except Exception as e2:
                    print(f"Error during fallback generation: {e2}")
                    raise
            print(f"Error during generation: {e}")
            raise
    
    async def generate_stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        repeat_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0
    ) -> AsyncIterator[str]:
        """Generate text with streaming.
        
        Args:
            prompt: Input prompt (or messages for chat)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            stop: Stop sequences
            grammar: Optional GBNF grammar string
            repeat_penalty: Repetition penalty (1.0 = no penalty)
            presence_penalty: Presence penalty
            frequency_penalty: Frequency penalty
            
        Yields:
            Generated text chunks
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Check if prompt looks like messages (list of dicts)
        if isinstance(prompt, list):
            prompt = self._apply_chat_template(prompt, add_generation_prompt=True)
        
        # Set defaults
        max_tokens = max_tokens if max_tokens is not None else 256
        temperature = temperature if temperature is not None else 0.7
        top_p = top_p if top_p is not None else 0.9
        top_k = 40
        repeat_penalty = repeat_penalty if repeat_penalty is not None else 1.1
        grammar = grammar
        
        # Get stop strings
        stop = stop if stop is not None else None
        if stop is None:
            # Get default stop tokens based on template
            stop = get_reasoning_stop_tokens(self.chat_template)
        
        # Check for grammar-guided generation
        use_grammar = grammar
        if use_grammar is None:
            # Check global flag
            try:
                import coderai
                if getattr(coderai, 'grammar_guided_gen', False):
                    use_grammar = get_tool_call_grammar()
            except (ImportError, AttributeError):
                pass
        
        try:
            # For chat, we need to extract just the new text from each chunk
            # The first chunk will have the full prompt + first token
            # Subsequent chunks only have new tokens
            
            first_chunk = True
            prompt_len = len(prompt) if isinstance(prompt, str) else 0

            async for chunk in _aiter_blocking(self.model.create_completion(
                stopping_criteria=_make_llama_thermal_criteria(),
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                stop=stop,
                stream=True,
                grammar=use_grammar,
            ), lock=self._gen_lock):
                text = chunk['choices'][0].get('text', '')

                if first_chunk:
                    # Skip the prompt text on first chunk
                    # The first chunk includes the full prompt plus the first new token
                    if text and len(text) > prompt_len:
                        # Extract just the new part
                        new_text = text[prompt_len:]
                        if new_text:
                            yield new_text
                    first_chunk = False
                else:
                    # Subsequent chunks only have new tokens
                    if text:
                        yield text
                
                # Check for stop
                if chunk['choices'][0].get('finish_reason'):
                    break
                    
        except Exception as e:
            # If grammar generation fails, fall back to normal generation
            if use_grammar:
                print(f"Warning: Grammar-guided streaming generation failed: {e}, falling back to normal generation")
                try:
                    first_chunk = True
                    prompt_len = len(prompt) if isinstance(prompt, str) else 0

                    async for chunk in _aiter_blocking(self.model.create_completion(
                        stopping_criteria=_make_llama_thermal_criteria(),
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repeat_penalty=repeat_penalty,
                        stop=stop,
                        stream=True,
                    ), lock=self._gen_lock):
                        text = chunk['choices'][0].get('text', '')

                        if first_chunk:
                            if text and len(text) > prompt_len:
                                new_text = text[prompt_len:]
                                if new_text:
                                    yield new_text
                            first_chunk = False
                        else:
                            if text:
                                yield text
                        
                        if chunk['choices'][0].get('finish_reason'):
                            break
                except Exception as e2:
                    print(f"Error during fallback streaming generation: {e2}")
                    raise
            else:
                print(f"Error during streaming generation: {e}")
                raise
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Dict[str, Any]:
        """Chat completion.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Generation parameters
            
        Returns:
            Response dict with 'content' and metadata
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Apply chat template
        prompt = self._apply_chat_template(messages, add_generation_prompt=True)
        
        # Generate
        max_tokens = kwargs.get('max_tokens', 256)
        temperature = kwargs.get('temperature', 0.7)
        top_p = kwargs.get('top_p', 0.9)
        repeat_penalty = kwargs.get('repeat_penalty', 1.1)
        
        stop = kwargs.get('stop', None)
        if stop is None:
            stop = get_reasoning_stop_tokens(self.chat_template)
        
        stream = kwargs.get('stream', False)
        
        if stream:
            # Return iterator for streaming
            async def generate_stream():
                first_chunk = True
                prompt_len = len(prompt)

                async for chunk in _aiter_blocking(self.model.create_completion(
                    stopping_criteria=_make_llama_thermal_criteria(),
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repeat_penalty=repeat_penalty,
                    stop=stop,
                    stream=True,
                ), lock=self._gen_lock):
                    text = chunk['choices'][0].get('text', '')
                    
                    if first_chunk:
                        if text and len(text) > prompt_len:
                            new_text = text[prompt_len:]
                            if new_text:
                                yield new_text
                        first_chunk = False
                    else:
                        if text:
                            yield text
                    
                    if chunk['choices'][0].get('finish_reason'):
                        break
            
            return {"stream": generate_stream(), "content": ""}
        else:
            with self._gen_lock:
                result = self.model.create_completion(
                    stopping_criteria=_make_llama_thermal_criteria(),
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repeat_penalty=repeat_penalty,
                    stop=stop,
                )

            content = result['choices'][0]['text']
            
            return {
                "content": content,
                "usage": result.get('usage', {}),
                "model": self.model_name,
            }
    
    async def embed(self, text: str) -> List[float]:
        """Generate embeddings.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        try:
            result = self.model.create_embedding(text)
            return result['data'][0]['embedding']
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            raise
    
    def unload_model(self):
        """Unload the model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
            self.model_name = None
            print("DEBUG: VulkanBackend unloaded model")
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.model is None:
            return {"loaded": False}
        
        return {
            "loaded": True,
            "model_name": self.model_name,
            "chat_template": self.chat_template,
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
        }

    # ------------------------------------------------------------------
    # Usage / cache helpers
    # ------------------------------------------------------------------

    def _read_cached_tokens(self, prompt_tokens: int) -> int:
        """Extract cached token count from llama.cpp timings after a completion."""
        try:
            timings = getattr(self.model, 'timings', None)
            if timings is None:
                # Try the internal context if timings property not exposed
                ctx = getattr(self.model, '_ctx', None)
                if ctx and hasattr(ctx, 'timings'):
                    timings = ctx.timings()
            if timings is not None:
                n_p_eval = getattr(timings, 'n_p_eval', None)
                if n_p_eval is not None:
                    return max(0, prompt_tokens - int(n_p_eval))
        except Exception:
            pass
        return 0

    def _store_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        cached = self._read_cached_tokens(prompt_tokens)
        self._last_usage = {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': prompt_tokens + completion_tokens,
            'cached_tokens': cached,
        }

    def get_last_usage(self) -> dict:
        """Return usage dict from the most recent completion (includes cached_tokens)."""
        return dict(self._last_usage)

    # ------------------------------------------------------------------
    # Chat-level generation (uses llama.cpp native chat template)
    # ------------------------------------------------------------------

    @staticmethod
    def _fold_system_into_user(messages):
        """Merge system message(s) into the first user turn.

        Some chat templates (notably Gemma) reject a 'system' role and llama.cpp
        raises "System role not supported". Folding the system text into the first
        user message is Gemma's own convention and is harmless for other models.
        Preserves order and multimodal (list) content."""
        def _role(m):
            return m.get('role') if isinstance(m, dict) else getattr(m, 'role', None)
        def _content(m):
            return m.get('content') if isinstance(m, dict) else getattr(m, 'content', None)

        sys_texts, rest = [], []
        for m in messages:
            if _role(m) == 'system':
                c = _content(m)
                if isinstance(c, str):
                    sys_texts.append(c)
                elif isinstance(c, list):
                    sys_texts += [p.get('text', '') for p in c
                                  if isinstance(p, dict) and p.get('type') == 'text']
                continue
            rest.append(m)
        preamble = "\n\n".join(t for t in sys_texts if t)
        if not preamble:
            return messages

        out, injected = [], False
        for m in rest:
            if not injected and _role(m) == 'user':
                c = _content(m)
                nm = dict(m) if isinstance(m, dict) else {'role': 'user', 'content': c}
                if isinstance(c, str):
                    nm['content'] = preamble + "\n\n" + c
                elif isinstance(c, list):
                    nm['content'] = [{'type': 'text', 'text': preamble}] + c
                else:
                    nm['content'] = preamble
                out.append(nm)
                injected = True
            else:
                out.append(m)
        if not injected:
            out.insert(0, {'role': 'user', 'content': preamble})
        return out

    @staticmethod
    def _is_system_role_error(exc) -> bool:
        return 'system role' in str(exc).lower()

    def _template_rejects_system(self) -> bool:
        """True when this model's chat template has no 'system' role.

        The authoritative signal is the embedded template itself: the official
        Gemma template (and other strict ones) carry
        ``raise_exception("System role not supported")``, while some finetune/
        quant conversions ship a permissive template that accepts system. So we
        fold proactively only when the template actually rejects it — not for
        every Gemma (e.g. 'heretic' gemma4 quants handle system fine). Falls back
        to GGUF architecture when the template can't be read."""
        if self._no_system_cached is not None:
            return self._no_system_cached
        val = None
        # 1) Embedded chat template (authoritative).
        tmpl = None
        try:
            md = getattr(self.model, 'metadata', None)
            if isinstance(md, dict):
                tmpl = md.get('tokenizer.chat_template')
        except Exception:
            tmpl = None
        if not tmpl:
            tmpl = getattr(self.model, 'chat_template', None) if self.model else None
        if isinstance(tmpl, str) and tmpl:
            low = tmpl.lower()
            if 'system role' in low or ('raise_exception' in low and 'system' in low):
                val = True
            else:
                val = False   # template renders without rejecting system
        # 2) No readable template → fall back to architecture (Gemma rejects system).
        if val is None:
            try:
                from codai.models.manager import _gguf_architecture
                val = (_gguf_architecture(self.model_name) or "").lower().startswith("gemma")
            except Exception:
                val = False
        self._no_system_cached = val
        return val

    def generate_chat(self, messages, max_tokens=None, temperature=0.7, top_p=1.0,
                      stop=None, tools=None, response_format=None, enable_thinking=False,
                      repeat_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0):
        """Non-streaming chat completion using llama.cpp's native chat handler."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        if self._template_rejects_system():
            messages = self._fold_system_into_user(messages)

        kwargs = dict(
            messages=messages,
            max_tokens=max_tokens or 512,
            temperature=temperature,
            top_p=top_p,
        )
        # Forward client-supplied repetition controls (no-op at defaults).
        if repeat_penalty and repeat_penalty != 1.0:
            kwargs['repeat_penalty'] = repeat_penalty
        if presence_penalty:
            kwargs['presence_penalty'] = presence_penalty
        if frequency_penalty:
            kwargs['frequency_penalty'] = frequency_penalty
        if stop:
            kwargs['stop'] = stop
        if response_format and response_format.get('type') == 'json_object':
            kwargs['response_format'] = {'type': 'json_object'}
        _tc = _make_llama_thermal_criteria()
        if _tc is not None and _chat_supports_stopping_criteria():
            kwargs['stopping_criteria'] = _tc

        with self._gen_lock:
            try:
                result = self.model.create_chat_completion(**kwargs)
            except Exception as e:
                if not self._is_system_role_error(e):
                    raise
                kwargs['messages'] = self._fold_system_into_user(messages)
                result = self.model.create_chat_completion(**kwargs)
        usage = result.get('usage', {})
        self._store_usage(
            prompt_tokens=usage.get('prompt_tokens', 0),
            completion_tokens=usage.get('completion_tokens', 0),
        )
        content = result['choices'][0]['message'].get('content') or ''
        return content

    async def generate_chat_stream(self, messages, max_tokens=None, temperature=0.7,
                                   top_p=1.0, stop=None, tools=None, response_format=None,
                                   enable_thinking=False, repeat_penalty=1.0,
                                   presence_penalty=0.0, frequency_penalty=0.0):
        """Streaming chat completion using llama.cpp's native chat handler."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        if self._template_rejects_system():
            messages = self._fold_system_into_user(messages)

        kwargs = dict(
            messages=messages,
            max_tokens=max_tokens or 512,
            temperature=temperature,
            top_p=top_p,
            stream=True,
        )
        # Forward client-supplied repetition controls (no-op at defaults).
        if repeat_penalty and repeat_penalty != 1.0:
            kwargs['repeat_penalty'] = repeat_penalty
        if presence_penalty:
            kwargs['presence_penalty'] = presence_penalty
        if frequency_penalty:
            kwargs['frequency_penalty'] = frequency_penalty
        if stop:
            kwargs['stop'] = stop
        _tc = _make_llama_thermal_criteria()
        if _tc is not None and _chat_supports_stopping_criteria():
            kwargs['stopping_criteria'] = _tc

        # Build the completion up front so a template error (e.g. Gemma's "System
        # role not supported") is caught here and retried with the system message
        # folded into the first user turn — before any chunk is streamed.
        try:
            _completion = self.model.create_chat_completion(**kwargs)
        except Exception as e:
            if not self._is_system_role_error(e):
                raise
            kwargs['messages'] = self._fold_system_into_user(messages)
            _completion = self.model.create_chat_completion(**kwargs)

        prompt_tokens = 0
        completion_tokens = 0
        try:
            async for chunk in _aiter_blocking(_completion, lock=self._gen_lock):
                delta = chunk['choices'][0].get('delta', {})
                text = delta.get('content') or ''
                if text:
                    completion_tokens += 1
                    yield text
                # Capture usage if present in final streaming chunk
                if chunk.get('usage'):
                    u = chunk['usage']
                    prompt_tokens = u.get('prompt_tokens', 0)
                    completion_tokens = u.get('completion_tokens', completion_tokens)
                if chunk['choices'][0].get('finish_reason'):
                    break
        finally:
            # Timings are available after the stream is exhausted
            if prompt_tokens == 0:
                # Estimate from word split if llama.cpp didn't report
                prompt_tokens = sum(
                    len(str(m.get('content', '')).split())
                    for m in messages
                )
            self._store_usage(prompt_tokens, completion_tokens)

    def get_model_name(self) -> str:
        """Return the loaded model name."""
        return self.model_name or "unknown"

    def format_messages(self, messages) -> str:
        """Format messages into a prompt string using chat template."""
        if isinstance(messages, list):
            return self._apply_chat_template(messages, add_generation_prompt=True)
        return str(messages)

    def cleanup(self) -> None:
        """Cleanup resources."""
        self.unload_model()
    
    def get_context_size(self) -> int:
        """Return the model's context window size."""
        return self.n_ctx