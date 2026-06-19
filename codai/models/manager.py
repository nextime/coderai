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

"""Model manager module - contains ModelManager, WhisperServerManager, and MultiModelManager classes."""

from typing import Optional, Dict, Any, List, Set
import os
import random
import subprocess
import signal
import requests
import time
import threading
import gc

# Import needed classes from other codai modules
from codai.models.parser import ModelParserAdapter
from codai.backends import detect_available_backends
from codai.backends.cuda import NvidiaBackend
from codai.backends.vulkan import VulkanBackend
from codai.models.cache import get_cached_model_path, download_model, get_model_cache_dir, load_model
from codai.models.utils import FuzzyToolBreaker
from codai.pydantic.textrequest import ModelInfo


def get_active_ds4_config():
    """Return the active Ds4Config from the server config, or None if unavailable."""
    try:
        from codai.admin.routes import config_manager
        if config_manager is not None and config_manager.config is not None:
            return config_manager.config.ds4
    except Exception:
        pass
    return None


_GGUF_ARCH_CACHE: Dict[tuple, str] = {}


def _resolve_local_gguf(model_name: str):
    """Map a model name/alias/path/basename to a local .gguf file path, or None.

    Handles the bare ids clients use (e.g. "Foo-ds4-Q2_K" with no path/extension)
    by matching configured model entries and scanning the GGUF cache by basename.
    """
    if not model_name:
        return None
    cand = os.path.expanduser(model_name)
    if cand.lower().endswith(".gguf") and os.path.isfile(cand):
        return cand
    # The cache resolver (handles aliases / repo ids), with/without the extension.
    for nm in (model_name, model_name + ".gguf"):
        try:
            p = get_cached_model_path(nm)
            if p and str(p).lower().endswith(".gguf") and os.path.isfile(str(p)):
                return str(p)
        except Exception:
            pass
    base = os.path.basename(model_name)
    base_noext = base[:-5] if base.lower().endswith(".gguf") else base
    # Match a configured models.json entry by path / basename (with or without .gguf).
    try:
        from codai.admin.routes import config_manager as cfg_mgr
        md = getattr(cfg_mgr, "models_data", None) if cfg_mgr else None
        if isinstance(md, dict):
            for cat, lst in md.items():
                if not isinstance(lst, list):
                    continue
                for m in lst:
                    key = m if isinstance(m, str) else (m.get("path") or m.get("id") or "") if isinstance(m, dict) else ""
                    if not key or not str(key).lower().endswith(".gguf"):
                        continue
                    kb = os.path.basename(key)
                    if (key == model_name or kb == base or kb[:-5] == base_noext) and os.path.isfile(os.path.expanduser(key)):
                        return os.path.expanduser(key)
    except Exception:
        pass
    # Scan the GGUF cache dir for a basename match.
    try:
        cache_dir = get_model_cache_dir()
        if cache_dir and os.path.isdir(cache_dir):
            for f in os.listdir(cache_dir):
                if f.lower().endswith(".gguf") and (f == base or f[:-5] == base_noext):
                    fp = os.path.join(cache_dir, f)
                    if os.path.isfile(fp):
                        return fp
    except Exception:
        pass
    return None


def _gguf_architecture(path: str):
    """Read ``general.architecture`` from a GGUF header. Cached by (path,mtime,size)."""
    import struct
    try:
        st = os.stat(path)
        key = (path, st.st_mtime_ns, st.st_size)
    except OSError:
        return None
    if key in _GGUF_ARCH_CACHE:
        return _GGUF_ARCH_CACHE[key] or None
    arch = ""
    _sz = {0: 1, 1: 1, 7: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 10: 8, 11: 8, 12: 8}
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                _GGUF_ARCH_CACHE[key] = ""
                return None
            f.read(4); f.read(8)  # version, tensor_count
            kv_count = struct.unpack("<Q", f.read(8))[0]

            def _rs(fh):
                n = struct.unpack("<Q", fh.read(8))[0]
                return fh.read(n)

            for _ in range(kv_count):
                k = _rs(f)
                vtype = struct.unpack("<I", f.read(4))[0]
                if vtype == 8:  # string
                    v = _rs(f)
                    if k == b"general.architecture":
                        arch = v.decode("utf-8", "ignore")
                        break
                elif vtype in _sz:
                    f.read(_sz[vtype])
                elif vtype == 9:  # array — skip its elements
                    atype = struct.unpack("<I", f.read(4))[0]
                    alen = struct.unpack("<Q", f.read(8))[0]
                    if atype == 8:
                        for _ in range(alen):
                            _rs(f)
                    elif atype in _sz:
                        f.read(_sz[atype] * alen)
                    else:
                        break  # unknown element type — stop
                else:
                    break  # unknown value type — stop
    except Exception:
        arch = ""
    _GGUF_ARCH_CACHE[key] = arch
    return arch or None


def ds4_should_handle(model_name: str) -> bool:
    """True when ds4 is enabled and ``model_name`` is a DeepSeek-V4 (ds4) model.

    Routing is by the GGUF ARCHITECTURE, not the filename: ds4 serves only genuine
    ``deepseek4`` GGUFs (its own format). Mainline DeepSeek GGUFs (deepseek/
    deepseek2/deepseek3/deepseek32) are left to llama.cpp. The configured
    ``model_id`` alias still routes (covers the variant ds4 downloads itself, which
    has no local file yet).
    """
    if not model_name:
        return False
    cfg = get_active_ds4_config()
    if cfg is None or not getattr(cfg, "enabled", False):
        return False
    name = model_name.lower()
    short = name.split("/")[-1]
    mid = (getattr(cfg, "model_id", "") or "").lower()
    if mid and (name == mid or short == mid):
        return True
    # Definitive: read the GGUF architecture for a local file — only deepseek4.
    path = _resolve_local_gguf(model_name)
    if path:
        return (_gguf_architecture(path) or "").lower() == "deepseek4"
    # No local file (HF id / not downloaded yet): conservative name check that
    # matches ONLY the V4 marker, so mainline deepseek GGUFs aren't grabbed.
    return "deepseek-v4" in name or "deepseek4" in name


def _trim_cpu_ram() -> None:
    """Return freed CPU heap memory to the OS (and let the kernel reclaim swap).

    Python frees evicted-model tensors, but glibc's allocator keeps the pages in
    the process arena (RSS stays high; swap is not returned).  malloc_trim(0)
    releases the top of the heap back to the OS.  No-op on non-glibc platforms.
    """
    try:
        gc.collect()
        import ctypes, ctypes.util
        _libc_name = ctypes.util.find_library('c') or 'libc.so.6'
        _libc = ctypes.CDLL(_libc_name)
        if hasattr(_libc, 'malloc_trim'):
            _libc.malloc_trim(0)
    except Exception:
        pass


def _drop_prefix_caches() -> int:
    """Release reclaimable KV prefix caches across all loaded text backends.

    The GGUF LlamaRAMCache lives in host RAM and the HF KV slots in VRAM; both are
    pure caches that can be rebuilt on demand. Dropping them is a cheap rung on the
    RAM/VRAM-pressure ladder, run before evicting whole models. Returns the number
    of backends whose cache was cleared.
    """
    n = 0
    try:
        from codai.models.manager import multi_model_manager as _mm
    except Exception:
        return 0

    def _clear(obj):
        nonlocal n
        backend = getattr(obj, 'backend', None)
        fn = getattr(backend, 'clear_prefix_cache', None)
        if callable(fn):
            try:
                fn()
                n += 1
            except Exception:
                pass

    try:
        for pool in list(getattr(_mm, 'model_pools', {}).values()):
            for inst in list(getattr(pool, 'instances', [])):
                _clear(inst)
        for obj in list(getattr(_mm, 'models', {}).values()):
            _clear(obj)
    except Exception:
        pass
    return n


class ModelManager:
    """Manages the loaded model and tokenizer."""

    def __init__(self, backend=None, backend_type=None):
        self.backend = backend
        self.backend_type = backend_type
        self.tool_parser = ModelParserAdapter()
    
    def _aggressive_vram_cleanup(self, model_manager):
        """
        Aggressively cleanup VRAM when switching between different model types.
        """
        try:
            import torch
            
            # First, try to move model to CPU if it has a model attribute
            if hasattr(model_manager, 'model') and model_manager.model is not None:
                model = model_manager.model
                if hasattr(model, 'to'):
                    try:
                        model.to('cpu')
                    except:
                        pass
                del model
            
            # Also handle backend directly if it's different
            if hasattr(model_manager, 'backend') and model_manager.backend is not None:
                backend = model_manager.backend
                
                if hasattr(backend, 'model') and backend.model is not None:
                    model = backend.model
                    if hasattr(model, 'to'):
                        try:
                            model.to('cpu')
                        except:
                            pass
                    del model
                
                if hasattr(backend, 'pipeline') and backend.pipeline is not None:
                    del backend.pipeline
                
                if hasattr(backend, 'vae') and backend.vae is not None:
                    del backend.vae
                
                if hasattr(backend, 'text_encoder') and backend.text_encoder is not None:
                    del backend.text_encoder
                
                if hasattr(backend, 'tokenizer') and backend.tokenizer is not None:
                    del backend.tokenizer
            
            # Force multiple rounds of garbage collection
            for _ in range(3):
                gc.collect()
            
            # Clear PyTorch cache
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            
            # Add delay to allow Vulkan to release memory
            time.sleep(2)
            
        except Exception as e:
            print(f"Warning during aggressive VRAM cleanup: {e}")
        finally:
            # Try to cleanup the model manager itself
            try:
                if hasattr(model_manager, 'cleanup'):
                    model_manager.cleanup()
            except:
                pass
        
    def load_model(self, model_name: str, backend_type: str = "auto", **kwargs):
        """Load the model with the specified backend."""
        # DeepSeek V4 via ds4: when enabled, route matching models to the managed
        # ds4-server proxy instead of the in-process nvidia/vulkan backends.
        if ds4_should_handle(model_name):
            from codai.backends.ds4 import Ds4Backend
            print(f"Routing '{model_name}' to ds4 (DeepSeek V4) backend")
            self.backend_type = "ds4"
            self.backend = Ds4Backend(get_active_ds4_config())
            self.backend.load_model(model_name, **kwargs)
            self.tool_parser = ModelParserAdapter(model_name=model_name)
            return

        available = detect_available_backends()

        # Check if model is a GGUF file
        is_gguf = model_name.endswith('.gguf') or 'gguf' in model_name.lower()
        
        # Determine backend
        if backend_type == "auto":
            if available.get('nvidia'):
                backend_type = "nvidia"
                print("Auto-detected NVIDIA backend")
            elif available.get('vulkan'):
                backend_type = "vulkan"
                print("Auto-detected Vulkan backend")
            else:
                print("Warning: No GPU backend detected.")
                backend_type = "cpu"
        
        # If GGUF file and backend is nvidia/cuda, use llama-cpp-python with CUDA backend
        original_backend = None
        if is_gguf and backend_type in ("nvidia", "cuda"):
            original_backend = backend_type
            print(f"GGUF model detected, using llama-cpp-python (original backend: {original_backend})")
            backend_type = "vulkan"
        
        self.backend_type = backend_type
        
        # Create appropriate backend
        if backend_type == "nvidia":
            if not available.get('nvidia'):
                raise RuntimeError("NVIDIA backend requested but PyTorch/CUDA not available")
            self.backend = NvidiaBackend()
        elif backend_type == "vulkan":
            if not available.get('vulkan'):
                raise RuntimeError("Vulkan backend requested but llama-cpp-python not available")
            self.backend = VulkanBackend(original_backend=original_backend)
        else:
            raise ValueError(f"Unknown backend: {backend_type}")
        
        # Load the model
        self.backend.load_model(model_name, **kwargs)
        self.tool_parser = ModelParserAdapter(model_name=model_name)
        
    def format_messages(self, messages):
        """Format messages into a prompt string."""
        if self.backend is None:
            raise RuntimeError("No model loaded")
        return self.backend.format_messages(messages)
    
    def generate(self, prompt: str, max_tokens: Optional[int] = None,
                 temperature: float = 0.7, top_p: float = 1.0,
                 stop: Optional[List[str]] = None, **kwargs):
        """Generate text non-streaming."""
        if self.backend is None:
            raise RuntimeError("No model loaded")
        return self.backend.generate(prompt, max_tokens, temperature, top_p, stop)

    
    def generate_chat(self, messages: List[Dict], max_tokens: Optional[int] = None,
                      temperature: float = 0.7, top_p: float = 1.0,
                      stop: Optional[List[str]] = None, tools: Optional[List] = None,
                      response_format: Optional[Dict] = None, enable_thinking: bool = False):
        """Generate chat completion non-streaming."""
        if self.backend is None:
            raise RuntimeError("No model loaded")
        # Use generate_chat if available (Vulkan backend), otherwise format and use generate
        if hasattr(self.backend, 'generate_chat'):
            try:
                return self.backend.generate_chat(messages, max_tokens, temperature, top_p,
                                                  stop, tools, response_format,
                                                  enable_thinking=enable_thinking)
            except TypeError:
                # Backend doesn't accept enable_thinking (e.g. Vulkan) — call plainly.
                return self.backend.generate_chat(messages, max_tokens, temperature, top_p,
                                                  stop, tools, response_format)
        else:
            # Fallback for NVIDIA backend
            from codai.pydantic.textrequest import ChatMessage
            prompt = self.format_messages([ChatMessage(**m) for m in messages])
            return self.backend.generate(prompt, max_tokens, temperature, top_p, stop)
    
    async def generate_stream(self, prompt: str, max_tokens: Optional[int] = None,
                              temperature: float = 0.7, top_p: float = 1.0,
                              stop: Optional[List[str]] = None):
        """Generate text in streaming fashion."""
        if self.backend is None:
            raise RuntimeError("No model loaded")
        async for chunk in self.backend.generate_stream(prompt, max_tokens, temperature, top_p, stop):
            yield chunk
    
    async def generate_chat_stream(self, messages: List[Dict], max_tokens: Optional[int] = None,
                                    temperature: float = 0.7, top_p: float = 1.0,
                                    stop: Optional[List[str]] = None, tools: Optional[List] = None,
                                    response_format: Optional[Dict] = None, enable_thinking: bool = False):
        """Generate chat completion streaming."""
        if self.backend is None:
            raise RuntimeError("No model loaded")
        # Use generate_chat_stream if available (Vulkan backend), otherwise format and use generate_stream
        if hasattr(self.backend, 'generate_chat_stream'):
            try:
                _gen = self.backend.generate_chat_stream(
                    messages, max_tokens, temperature, top_p, stop, tools,
                    response_format, enable_thinking=enable_thinking)
            except TypeError:
                _gen = self.backend.generate_chat_stream(
                    messages, max_tokens, temperature, top_p, stop, tools, response_format)
            async for chunk in _gen:
                yield chunk
        else:
            # Fallback for NVIDIA backend
            from codai.pydantic.textrequest import ChatMessage
            prompt = self.format_messages([ChatMessage(**m) for m in messages])
            async for chunk in self.backend.generate_stream(prompt, max_tokens, temperature, top_p, stop):
                yield chunk
    
    @property
    def model_name(self) -> str:
        if self.backend is None:
            return "unknown"
        return self.backend.get_model_name()
    
    @property
    def model(self):
        if self.backend is None:
            return None
        return self.backend
    
    @property
    def tokenizer(self):
        # Only NVIDIA backend has a tokenizer
        if isinstance(self.backend, NvidiaBackend):
            return self.backend.tokenizer
        return None
    
    def get_context_size(self) -> int:
        """Get the model's context window size."""
        if self.backend is not None:
            return self.backend.get_context_size()
        return 2048  # Default fallback

    def get_last_usage(self) -> dict:
        """Return usage info (including cached_tokens) from the most recent call."""
        if self.backend is not None and hasattr(self.backend, 'get_last_usage'):
            return self.backend.get_last_usage()
        return {}
    
    def cleanup(self):
        if self.backend is not None:
            self.backend.cleanup()
            self.backend = None


class WhisperServerManager:
    """Manages whisper-server subprocess for audio transcription."""
    
    def __init__(self, server_path: str = None, port: int = 8744):
        self.server_path = server_path
        self.port = port
        self.process = None
        self.current_model = None
        self.base_url = f"http://127.0.0.1:{port}"
        self.lock = threading.Lock()
        self._active_requests = 0
        
        # Check if port is available
        if not self._is_port_available(port):
            for new_port in range(port + 1, port + 100):
                if self._is_port_available(new_port):
                    self.port = new_port
                    self.base_url = f"http://127.0.0.1:{new_port}"
                    print(f"Port {port} in use, using port {new_port} instead")
                    break
    
    def _is_port_available(self, port: int) -> bool:
        """Check if a port is available."""
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return True
        except OSError:
            return False
    
    def is_running(self) -> bool:
        """Check if whisper-server is running."""
        if self.process is None:
            return False
        return self.process.poll() is None
    
    def start(self, model_path: str = None, gpu_device: int = 0):
        """Start whisper-server with the specified model."""
        with self.lock:
            if self.is_running():
                self.stop()

            if not self.server_path:
                print("Error: whisper-server path not set")
                return ""

            # Handle URL models - use centralized cache loading
            actual_model_path = model_path
            if model_path and (model_path.startswith('http://') or model_path.startswith('https://')):
                print(f"Loading model: {model_path}")
                actual_model_path = load_model(model_path)
                if not actual_model_path:
                    print(f"Failed to load model: {model_path}")
                    return ""

            cmd = [self.server_path]
            if actual_model_path:
                cmd.extend(["-m", actual_model_path])
            cmd.extend(["-dev", str(gpu_device)])
            cmd.append("--convert")
            cmd.extend(["--host", "127.0.0.1"])
            cmd.extend(["--port", str(self.port)])

            print(f"Starting whisper-server: {' '.join(cmd)}")

            try:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    preexec_fn=lambda: signal.signal(signal.SIGTERM, signal.SIG_DFL)
                )
                self.current_model = actual_model_path

                if self._wait_for_server(30):
                    print(f"whisper-server started on {self.base_url}")
                    return actual_model_path
                else:
                    print("Error: whisper-server failed to start")
                    self.stop()
                    return ""
            except Exception as e:
                print(f"Error starting whisper-server: {e}")
                return ""
    
    def stop(self):
        """Stop whisper-server."""
        with self.lock:
            if self.process:
                try:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait()
                except Exception as e:
                    print(f"Error stopping whisper-server: {e}")
                self.process = None
                self.current_model = None

    def cleanup(self):
        """Free VRAM by stopping the subprocess. Lets the generic VRAM-eviction
        path (_evict_one / unload_model) treat a running whisper-server like any
        other loaded model and actually release its GPU memory."""
        self.stop()

    def transcribe(self, audio_data: bytes, language: str = None, prompt: str = None):
        """Send transcription request to whisper-server."""
        if not self.is_running():
            return {"error": "whisper-server not running"}
        with self.lock:
            self._active_requests += 1
        try:
            files = {"file": ("audio.wav", audio_data, "audio/wav")}
            data = {}
            if language:
                data["language"] = language
            if prompt:
                data["prompt"] = prompt
            
            response = requests.post(
                f"{self.base_url}/inference",
                files=files,
                data=data,
                timeout=300
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Server error: {response.status_code}", "detail": response.text}
        except Exception as e:
            return {"error": str(e)}
        finally:
            with self.lock:
                self._active_requests = max(0, self._active_requests - 1)

    def active_requests(self) -> int:
        with self.lock:
            return self._active_requests

    def is_idle(self) -> bool:
        return self.active_requests() == 0
    
    def _wait_for_server(self, timeout: int = 30) -> bool:
        """Wait for whisper-server to be ready."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.base_url}/health", timeout=2)
                if response.status_code == 200:
                    return True
            except:
                pass
            time.sleep(0.5)
        return False
    
    def get_status(self) -> dict:
        """Get whisper-server status."""
        return {
            "running": self.is_running(),
            "model": self.current_model,
            "url": self.base_url
        }

    def cleanup(self):
        """Stop the subprocess — called by the VRAM eviction/unload machinery."""
        print("whisper-server: evicted from VRAM, stopping subprocess")
        self.stop()


class ModelInstancePool:
    """
    Ref-counted pool of loaded instances for a single model key.

    When max_instances > 1 and all existing instances are handling at least one
    active request, the manager can load additional instances (up to the limit)
    to serve concurrent requests in parallel.  Each caller must call release()
    when it finishes using the instance so the ref-count stays accurate.
    """

    def __init__(self, max_instances: int = 1):
        self.instances: list = []
        self.ref_counts: list = []
        self.max_instances: int = max_instances
        self._lock = threading.Lock()
        # Session-affinity map: session_key -> instance index. Routes successive
        # requests of one conversation back to the instance that already holds its
        # warm KV prefix, instead of the purely load-based least-busy pick.
        self._affinity: dict = {}

    @property
    def count(self) -> int:
        return len(self.instances)

    @property
    def primary(self):
        return self.instances[0] if self.instances else None

    def add(self, model_obj) -> int:
        """Register a new instance; return its index."""
        with self._lock:
            idx = len(self.instances)
            self.instances.append(model_obj)
            self.ref_counts.append(0)
            return idx

    def acquire(self, session_key=None):
        """Return (idx, instance) of the chosen instance, incrementing its ref-count.

        When ``session_key`` is given, prefer the instance this session was last
        routed to (cache affinity) as long as it isn't markedly busier than the
        least-busy one; otherwise fall back to the least-busy instance and record
        the new mapping. With ``session_key=None`` the behaviour is unchanged
        (pure least-busy).
        """
        with self._lock:
            if not self.instances:
                return None
            least = min(range(len(self.instances)), key=lambda i: self.ref_counts[i])
            idx = least
            if session_key is not None:
                pinned = self._affinity.get(session_key)
                if pinned is not None and 0 <= pinned < len(self.instances):
                    # Honour affinity unless the pinned instance is busier than the
                    # least-busy one by more than one in-flight request (keeps a hot
                    # cache without letting one instance pile up under load).
                    if self.ref_counts[pinned] <= self.ref_counts[least] + 1:
                        idx = pinned
                self._affinity[session_key] = idx
                self._prune_affinity()
            self.ref_counts[idx] += 1
            return idx, self.instances[idx]

    def _prune_affinity(self) -> None:
        """Bound the affinity map so it can't grow without limit (caller holds lock)."""
        _CAP = 512
        if len(self._affinity) > _CAP:
            # Drop arbitrary oldest-inserted entries; dicts preserve insertion order.
            for k in list(self._affinity.keys())[: len(self._affinity) - _CAP]:
                self._affinity.pop(k, None)

    def release(self, idx: int) -> None:
        with self._lock:
            if 0 <= idx < len(self.ref_counts):
                self.ref_counts[idx] = max(0, self.ref_counts[idx] - 1)

    def least_busy_instance(self):
        """Return the instance with the lowest ref-count without incrementing."""
        with self._lock:
            if not self.instances:
                return None
            idx = min(range(len(self.instances)), key=lambda i: self.ref_counts[i])
            return self.instances[idx]

    def all_busy(self) -> bool:
        """True if every instance is handling at least one request."""
        with self._lock:
            return bool(self.ref_counts) and all(r > 0 for r in self.ref_counts)

    def can_add(self) -> bool:
        return len(self.instances) < self.max_instances

    def needs_new_instance(self) -> bool:
        """True when all instances are busy and we are below max_instances."""
        return self.all_busy() and self.can_add()

    def cleanup_all(self) -> None:
        with self._lock:
            for obj in self.instances:
                try:
                    if hasattr(obj, 'cleanup'):
                        obj.cleanup()
                    elif hasattr(obj, 'to'):
                        obj.to('cpu')
                except Exception:
                    pass
            self.instances.clear()
            self.ref_counts.clear()


class MultiModelManager:
    """
    Manages multiple models: main text model, audio transcription, and image generation.
    """

    def __init__(self):
        self.models: Dict[str, Any] = {}  # Can hold ModelManager, diffusers pipelines, sd.cpp models, etc.
        self.default_model: Optional[str] = None
        self.audio_models: List[str] = []
        self.tts_model: Optional[str] = None
        self.image_models: List[str] = []
        self.vision_models: List[str] = []
        self.video_models: List[str] = []       # video generation models (t2v / i2v / v2v)
        self.audio_gen_models: List[str] = []   # music / sfx generation (MusicGen, AudioLDM2…)
        self.embedding_models: List[str] = []   # text / multimodal embeddings
        self.spatial_models: List[str] = []     # depth estimation, segmentation, object detection
        self.config: Dict[str, Dict] = {}  # Store model configurations
        # In the front/engine split, the front assigns a subset of models.json to
        # this engine. When set, list_models() reports only these (so /v1/models per
        # engine reflects what it actually serves); None = report all (single-process).
        self._assigned_model_keys: Optional[set] = None
        self.model_registered_types: Dict[str, Set[str]] = {}
        self.tool_parser = ModelParserAdapter()
        self.current_model_key: Optional[str] = None
        self.load_mode: str = "ondemand"
        self._last_used: Dict[str, float] = {}  # model_key -> last-served monotonic ts (LRU)
        self._active_in_vram: Optional[str] = None  # backing field for active_in_vram property
        self.models_in_vram: set = set()  # all models currently in VRAM
        self.model_aliases: Dict[str, str] = {}
        self.whisper_server: Optional[WhisperServerManager] = None  # legacy single-instance compat
        self.whisper_servers: Dict[str, WhisperServerManager] = {}  # id -> manager
        self.whisper_aliases: Dict[str, List[str]] = {}  # alias -> [model_id, ...]
        self._whisper_alias_counters: Dict[str, int] = {}  # alias -> next round-robin index
        self.model_backend_types: Dict[str, str] = {}
        self.tool_breaker = FuzzyToolBreaker(threshold=3)  # Circuit breaker for repetitive tool calls
        self._load_lock = threading.Lock()  # Prevents duplicate on-demand model loads
        # Set when NO model switch/load is in progress; cleared during load.
        # Callers that need to wait for a load to complete can poll this event.
        self._model_ready_event = threading.Event()
        self._model_ready_event.set()   # initially ready (nothing loading)
        self.model_pools: Dict[str, ModelInstancePool] = {}  # per-key instance pools
        # Set once a CUDA device-side assert / unrecoverable CUDA error is seen.
        # The CUDA context is corrupted process-wide after such an error, so all
        # further GPU work is futile until the server is restarted.  We surface
        # this immediately instead of retrying loads dozens of times.
        self.cuda_context_poisoned: bool = False
        self.cuda_poison_reason: Optional[str] = None
        self._pending_new_instance: set = set()  # keys awaiting a second+ instance load
        self._global_max_instances: int = 1  # set from config at startup
        self._measured_vram_gb: Dict[str, float] = {}  # actual measured VRAM delta per model key
        self._last_load_errors: Dict[str, str] = {}  # model_key -> last failed load message
        # Callbacks that free VRAM held *outside* the model manager (e.g. the
        # LoRA trainer caches its SD/SDXL base model between jobs). Each returns
        # the GB it freed (or None). Invoked as a last resort during eviction.
        self._external_vram_releasers: List[Any] = []

    @property
    def active_in_vram(self) -> Optional[str]:
        """Most-recently-used model key (None when nothing is active)."""
        return self._active_in_vram

    @active_in_vram.setter
    def active_in_vram(self, key: Optional[str]) -> None:
        # Single chokepoint for "this model was just used" — stamp the LRU clock so
        # both VRAM and RAM eviction order by true recency, not dict insertion order.
        self._active_in_vram = key
        if key:
            self._last_used[key] = time.monotonic()

    def register_external_vram_releaser(self, fn) -> None:
        """Register a callback that frees VRAM held outside the manager.

        Called during on-request eviction when freeing tracked models isn't
        enough. The callback must be safe to call when there's nothing to free
        (return 0.0 / None) and must not block on in-flight work it owns.
        """
        if callable(fn) and fn not in self._external_vram_releasers:
            self._external_vram_releasers.append(fn)

    @staticmethod
    def _is_cuda_context_fatal(err: Exception) -> bool:
        """True for CUDA errors that corrupt the whole process context."""
        s = str(err).lower()
        return (
            'device-side assert' in s
            or 'cuda error' in s
            or 'cublas' in s and 'not initialized' in s
            or 'an illegal memory access' in s
            or 'misaligned address' in s
        )

    def _mark_cuda_poisoned_if_fatal(self, err: Exception) -> bool:
        """Record process-wide CUDA corruption. Returns True if poisoned."""
        if self.cuda_context_poisoned:
            return True
        if self._is_cuda_context_fatal(err):
            self.cuda_context_poisoned = True
            self.cuda_poison_reason = str(err).split('\n', 1)[0][:200]
            print("\n" + "=" * 72)
            print("FATAL: CUDA context corrupted (device-side assert / CUDA error).")
            print(f"  Reason: {self.cuda_poison_reason}")
            print("  All GPU operations will fail until coderai is RESTARTED.")
            print("  Failing fast instead of retrying. Restart the server.")
            print("=" * 72 + "\n")
            return True
        return False

    @property
    def image_model(self) -> Optional[str]:
        """Return the first image model or None."""
        return self.image_models[0] if self.image_models else None

    def get_loaded_model_keys(self) -> set:
        """Return the set of currently loaded model keys."""
        return set(self.models.keys())

    def has_loaded_model(self, model_key: str) -> bool:
        """Return True when the given model key is currently loaded."""
        return model_key in self.models
    
    def cleanup(self):
        """Cleanup all models and resources."""
        # Cleanup all model managers
        for key, manager in self.models.items():
            try:
                if hasattr(manager, 'cleanup'):
                    manager.cleanup()
            except Exception as e:
                print(f"Warning: Error cleaning up model {key}: {e}")
        self.models.clear()
        
        # Cleanup whisper server(s)
        for wsm in self.whisper_servers.values():
            try:
                wsm.stop()
            except Exception as e:
                print(f"Warning: Error cleaning up whisper server: {e}")
        if self.whisper_server and self.whisper_server not in self.whisper_servers.values():
            try:
                self.whisper_server.stop()
            except Exception:
                pass
        self.whisper_servers.clear()
        
        # Clear all model lists
        self.default_model = None
        self.audio_models.clear()
        self.image_models.clear()
        self.vision_models.clear()
        self.tts_model = None
    
    def _aggressive_vram_cleanup(self, model_manager):
        """Aggressively cleanup VRAM when switching between different model types."""
        try:
            import torch
            
            if hasattr(model_manager, 'model') and model_manager.model is not None:
                model = model_manager.model
                if hasattr(model, 'to'):
                    try:
                        model.to('cpu')
                    except:
                        pass
                del model
            
            if hasattr(model_manager, 'backend') and model_manager.backend is not None:
                backend = model_manager.backend
                
                if hasattr(backend, 'model') and backend.model is not None:
                    model = backend.model
                    if hasattr(model, 'to'):
                        try:
                            model.to('cpu')
                        except:
                            pass
                    del model
                
                if hasattr(backend, 'pipeline') and backend.pipeline is not None:
                    del backend.pipeline
                
                if hasattr(backend, 'vae') and backend.vae is not None:
                    del backend.vae
                
                if hasattr(backend, 'text_encoder') and backend.text_encoder is not None:
                    del backend.text_encoder
                
                if hasattr(backend, 'tokenizer') and backend.tokenizer is not None:
                    del backend.tokenizer
            
            for _ in range(3):
                gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            
            time.sleep(2)
            
        except Exception as e:
            print(f"Warning during aggressive VRAM cleanup: {e}")
        finally:
            try:
                if hasattr(model_manager, 'cleanup'):
                    model_manager.cleanup()
            except:
                pass
    
    def set_load_mode(self, mode: str):
        """Set the load mode: 'ondemand', 'loadall', or 'loadswap'."""
        self.load_mode = mode
    
    def set_default_model(self, model_name: str, config: Dict = None, backend_type: str = "auto"):
        """Set the default/main text model and download/cache it if needed."""
        self.default_model = model_name
        self.config[model_name] = config or {}
        self.model_backend_types[model_name] = backend_type
        self._remember_registered_type(model_name, "text")

        # Download/cache the model at startup if it's a URL or HF ID
        resolved_model = self.load_model(model_name)
        if resolved_model != model_name:
            # Model was downloaded/cached, update the stored name
            self.default_model = resolved_model
            print(f"Model '{model_name}' cached as: {resolved_model}")
    
    def _load_default_model(self):
        """Load the default model on demand (thread-safe)."""
        if not self.default_model:
            return None

        # Fast path: already loaded (checked without lock for performance)
        if self.default_model in self.models and self.default_model not in self._pending_new_instance:
            return self._get_least_busy_instance(self.default_model)

        self._model_ready_event.clear()   # signal: load in progress
        with self._load_lock:
            # Re-check inside the lock to avoid duplicate loads from concurrent requests
            if self.default_model in self.models and self.default_model not in self._pending_new_instance:
                self._model_ready_event.set()
                return self._get_least_busy_instance(self.default_model)
            self._pending_new_instance.discard(self.default_model)

            config = self._config_for_model(self.default_model)
            backend_type = self.model_backend_types.get(self.default_model, "auto")

            try:
                from codai.api.state import get_global_args
                global_args = get_global_args()
            except Exception:
                global_args = None

            model_manager = ModelManager()
            try:
                # Build kwargs: per-model config takes precedence over global_args.
                # This ensures settings configured per-model in coderai (4-bit quant,
                # flash attention, offload strategy, …) are honoured on every load,
                # including on-demand reloads after VRAM eviction.
                kwargs = {}
                _ga = global_args  # shorthand

                def _cfg_or_global(cfg_key, ga_attr, default=None):
                    if cfg_key in config and config[cfg_key] is not None:
                        return config[cfg_key]
                    if _ga and hasattr(_ga, ga_attr):
                        v = getattr(_ga, ga_attr)
                        if v is not None:
                            return v
                    return default

                # Context window. The per-model config stores it as 'n_ctx'
                # (models.json), while older configs/CLI use 'ctx'. Read either,
                # and pass BOTH kwarg names downstream: the GGUF/llama.cpp backend
                # reads 'n_ctx', the transformers backend reads 'ctx'.
                # Context window. The per-model runtime cfg stores it under 'ctx'
                # (build_runtime_kwargs maps the entry's n_ctx → 'ctx'); 'n_ctx' is
                # also accepted. The PER-MODEL value must win over the global
                # vulkan.n_ctx fallback, so check the config keys first.
                ctx = config.get('ctx')
                if ctx is None:
                    ctx = config.get('n_ctx')
                if ctx is None and _ga is not None:
                    ctx = getattr(_ga, 'n_ctx', None)
                # Coerce to a positive int: a stray list/str (e.g. an old global
                # default wrapped in a list) would otherwise reach llama.cpp and
                # raise '<' int-vs-list at load.
                if isinstance(ctx, (list, tuple)):
                    ctx = ctx[0] if ctx else None
                try:
                    ctx = int(ctx) if ctx is not None else None
                except (TypeError, ValueError):
                    ctx = None
                if ctx:
                    kwargs['n_ctx'] = ctx
                    kwargs['ctx'] = ctx
                n_gpu_layers = _cfg_or_global('n_gpu_layers', 'n_gpu_layers')
                if n_gpu_layers is not None:
                    kwargs['n_gpu_layers'] = n_gpu_layers
                offload_dir = _cfg_or_global('offload_dir', 'offload_dir')
                if offload_dir:
                    kwargs['offload_dir'] = offload_dir
                manual_ram = _cfg_or_global('manual_ram_gb', 'ram')
                if manual_ram is not None:
                    kwargs['manual_ram_gb'] = manual_ram
                # flash_attn comes from the per-model config (source of truth).
                # build_kwargs_from_config populates it from the model's
                # 'flash_attention' setting; CLI/global is NOT consulted here.
                kwargs['flash_attn'] = bool(config.get('flash_attn', False))
                # KV-cache quantization (llama.cpp type_k/type_v) — pass through
                # to the backend, with the raw models.json entry as a fallback.
                _raw = config.get('_raw_cfg') if isinstance(config.get('_raw_cfg'), dict) else {}
                for _kvk in ('cache_type_k', 'cache_type_v', 'mmproj'):
                    _kvv = config.get(_kvk)
                    if _kvv is None:
                        _kvv = _raw.get(_kvk)
                    if _kvv:
                        kwargs[_kvk] = _kvv
                if _raw and '_raw_cfg' not in kwargs:
                    kwargs['_raw_cfg'] = _raw
                no_ram = _cfg_or_global('no_ram', 'no_ram', False)
                kwargs['no_ram'] = bool(no_ram)
                offload_strategy = _cfg_or_global('offload_strategy', 'offload_strategy', 'auto')
                if offload_strategy:
                    kwargs['offload_strategy'] = offload_strategy
                load_in_4bit = _cfg_or_global('load_in_4bit', 'load_in_4bit', False)
                kwargs['load_in_4bit'] = bool(load_in_4bit)
                load_in_8bit = _cfg_or_global('load_in_8bit', 'load_in_8bit', False)
                kwargs['load_in_8bit'] = bool(load_in_8bit)
                max_gpu_pct = _cfg_or_global('max_gpu_percent', 'max_gpu_percent')
                if max_gpu_pct is not None:
                    kwargs['max_gpu_percent'] = max_gpu_pct

                print(f"Loading default model on demand: {self.default_model}")
                _snap = self.vram_before_load()
                kwargs['expected_vram_gb'] = self._get_model_used_vram_gb(self.default_model)
                from codai.tasks import loading_task
                with loading_task(self.default_model, model_type="text"):
                    model_manager.load_model(self.default_model, backend_type=backend_type, **kwargs)
                self._last_load_errors.pop(self.default_model, None)
                self.add_model(self.default_model, model_manager)
                self.record_vram_delta(self.default_model, _snap)
                self.current_model_key = self.default_model
                print(f"Model loaded successfully: {self.default_model}")
                self._model_ready_event.set()
                return model_manager
            except Exception as e:
                print(f"Error loading model {self.default_model}: {e}")
                self._last_load_errors[self.default_model] = str(e)
                self._mark_cuda_poisoned_if_fatal(e)
                self._model_ready_event.set()
                return None

    def _load_model_by_name(self, model_name: str):
        """Load a model by name on demand (thread-safe)."""
        if model_name in self.models and model_name not in self._pending_new_instance:
            return self._get_least_busy_instance(model_name)

        self._model_ready_event.clear()   # signal: load in progress
        with self._load_lock:
            # Re-check inside lock to prevent duplicate loads, but honour pending new instance.
            if model_name in self.models and model_name not in self._pending_new_instance:
                return self._get_least_busy_instance(model_name)
            self._pending_new_instance.discard(model_name)

            config = self._config_for_model(model_name)
            backend_type = self.model_backend_types.get(model_name, "auto")

            try:
                from codai.api.state import get_global_args
                global_args = get_global_args()
            except Exception:
                global_args = None

            model_manager = ModelManager()
            try:
                # Per-model config takes precedence over global_args (same logic as
                # _load_default_model so on-demand reloads respect per-model settings).
                kwargs = {}
                _ga = global_args

                def _cfg_or_global(cfg_key, ga_attr, default=None):
                    if cfg_key in config and config[cfg_key] is not None:
                        return config[cfg_key]
                    if _ga and hasattr(_ga, ga_attr):
                        v = getattr(_ga, ga_attr)
                        if v is not None:
                            return v
                    return default

                # Context window. The per-model config stores it as 'n_ctx'
                # (models.json), while older configs/CLI use 'ctx'. Read either,
                # and pass BOTH kwarg names downstream: the GGUF/llama.cpp backend
                # reads 'n_ctx', the transformers backend reads 'ctx'.
                # Context window. The per-model runtime cfg stores it under 'ctx'
                # (build_runtime_kwargs maps the entry's n_ctx → 'ctx'); 'n_ctx' is
                # also accepted. The PER-MODEL value must win over the global
                # vulkan.n_ctx fallback, so check the config keys first.
                ctx = config.get('ctx')
                if ctx is None:
                    ctx = config.get('n_ctx')
                if ctx is None and _ga is not None:
                    ctx = getattr(_ga, 'n_ctx', None)
                # Coerce to a positive int: a stray list/str (e.g. an old global
                # default wrapped in a list) would otherwise reach llama.cpp and
                # raise '<' int-vs-list at load.
                if isinstance(ctx, (list, tuple)):
                    ctx = ctx[0] if ctx else None
                try:
                    ctx = int(ctx) if ctx is not None else None
                except (TypeError, ValueError):
                    ctx = None
                if ctx:
                    kwargs['n_ctx'] = ctx
                    kwargs['ctx'] = ctx
                n_gpu_layers = _cfg_or_global('n_gpu_layers', 'n_gpu_layers')
                if n_gpu_layers is not None:
                    kwargs['n_gpu_layers'] = n_gpu_layers
                offload_dir = _cfg_or_global('offload_dir', 'offload_dir')
                if offload_dir:
                    kwargs['offload_dir'] = offload_dir
                manual_ram = _cfg_or_global('manual_ram_gb', 'ram')
                if manual_ram is not None:
                    kwargs['manual_ram_gb'] = manual_ram
                # flash_attn comes from the per-model config (source of truth).
                # build_kwargs_from_config populates it from the model's
                # 'flash_attention' setting; CLI/global is NOT consulted here.
                kwargs['flash_attn'] = bool(config.get('flash_attn', False))
                # KV-cache quantization (llama.cpp type_k/type_v) — pass through
                # to the backend, with the raw models.json entry as a fallback.
                _raw = config.get('_raw_cfg') if isinstance(config.get('_raw_cfg'), dict) else {}
                for _kvk in ('cache_type_k', 'cache_type_v', 'mmproj'):
                    _kvv = config.get(_kvk)
                    if _kvv is None:
                        _kvv = _raw.get(_kvk)
                    if _kvv:
                        kwargs[_kvk] = _kvv
                if _raw and '_raw_cfg' not in kwargs:
                    kwargs['_raw_cfg'] = _raw
                no_ram = _cfg_or_global('no_ram', 'no_ram', False)
                kwargs['no_ram'] = bool(no_ram)
                offload_strategy = _cfg_or_global('offload_strategy', 'offload_strategy', 'auto')
                if offload_strategy:
                    kwargs['offload_strategy'] = offload_strategy
                load_in_4bit = _cfg_or_global('load_in_4bit', 'load_in_4bit', False)
                kwargs['load_in_4bit'] = bool(load_in_4bit)
                load_in_8bit = _cfg_or_global('load_in_8bit', 'load_in_8bit', False)
                kwargs['load_in_8bit'] = bool(load_in_8bit)
                max_gpu_pct = _cfg_or_global('max_gpu_percent', 'max_gpu_percent')
                if max_gpu_pct is not None:
                    kwargs['max_gpu_percent'] = max_gpu_pct

                pool = self.model_pools.get(model_name)
                inst_num = pool.count + 1 if pool else 1
                print(f"Loading model on demand: {model_name}"
                      + (f" (instance {inst_num})" if inst_num > 1 else ""))
                # Evict resident models to make room before loading (idempotent —
                # a no-op when request_model already freed enough). Guards the
                # direct on-demand path, which otherwise loads on top of the
                # current model and OOMs (e.g. switching to a larger model).
                if inst_num == 1:
                    try:
                        self.ensure_vram_for(model_name)
                    except Exception as _ev_e:
                        print(f"  (ensure_vram_for warning: {_ev_e})")
                _snap = self.vram_before_load()
                # Tell the backend how much VRAM this model is expected to need so
                # it can decide whether Flash-Attention-2 is safe (FA2 requires the
                # whole model on GPU; it device-side-asserts when layers offload).
                kwargs['expected_vram_gb'] = self._get_model_used_vram_gb(model_name)
                from codai.tasks import loading_task
                with loading_task(model_name, model_type="text"):
                    model_manager.load_model(model_name, backend_type=backend_type, **kwargs)
                self._last_load_errors.pop(model_name, None)
                self.add_model(model_name, model_manager)
                self.record_vram_delta(model_name, _snap)
                self.current_model_key = model_name
                print(f"Model loaded successfully: {model_name}"
                      + (f" (instance {inst_num})" if inst_num > 1 else ""))
                self._model_ready_event.set()   # signal: ready
                return model_manager
            except Exception as e:
                print(f"Error loading model {model_name}: {e}")
                self._last_load_errors[model_name] = str(e)
                self._mark_cuda_poisoned_if_fatal(e)
                self._model_ready_event.set()   # signal: ready (even on failure)
                return None

    def set_audio_model(self, model_name: str, config: Dict = None):
        """Add an audio transcription model and download/cache it if needed."""
        if model_name not in self.audio_models:
            self.audio_models.append(model_name)
        self.config[f"audio:{model_name}"] = config or {}
        self._remember_registered_type(model_name, "audio")

        if isinstance(config, dict) and config.get("backend") == "whisper-server":
            print(f"Registered whisper-server audio model: {model_name}")
            return

        # Download/cache the model at startup if it's a URL or HF ID
        resolved_model = self.load_model(model_name)
        if resolved_model != model_name:
            # Model was downloaded/cached, update the stored name
            idx = self.audio_models.index(model_name)
            self.audio_models[idx] = resolved_model
            self.config[f"audio:{resolved_model}"] = self.config.pop(f"audio:{model_name}")
            print(f"Audio model '{model_name}' cached as: {resolved_model}")

    def register_whisper_server(self, model_id: str, server_path: str, model_path: str = None,
                                 port: int = 8744, gpu_device: int = 0, config: Dict = None,
                                 alias: str = None):
        """Register a whisper-server instance as an audio model."""
        wsm = WhisperServerManager(server_path=server_path, port=port)
        wsm._model_path = model_path
        wsm._gpu_device = gpu_device
        self.whisper_servers[model_id] = wsm
        # Keep legacy single-instance reference pointing to the first one registered
        if self.whisper_server is None:
            self.whisper_server = wsm
        # Register as allowed audio model with its config
        cfg = config or {}
        cfg.setdefault("load_mode", "on-request")
        if model_id not in self.audio_models:
            self.audio_models.append(model_id)
        self.config[f"audio:{model_id}"] = cfg
        self._remember_registered_type(model_id, "audio")
        # Register alias for round-robin routing
        if alias:
            wsm._alias = alias
            ids = self.whisper_aliases.setdefault(alias, [])
            if model_id not in ids:
                ids.append(model_id)
            self._whisper_alias_counters.setdefault(alias, 0)
        print(f"Registered whisper-server audio model: {model_id} (server: {server_path})"
              + (f" alias={alias}" if alias else ""))
        return wsm

    def _estimate_gguf_vram_gb(self, path: Optional[str]) -> float:
        """Rough VRAM footprint of a gguf from its on-disk size (weights ~= file)."""
        try:
            if path and os.path.isfile(path):
                return round(os.path.getsize(path) / 1e9 * 1.1, 2)
        except Exception:
            pass
        return 0.0

    def start_whisper_server(self, model_id: str, model_path: str = None,
                             gpu_device: int = None) -> bool:
        """Start a whisper-server RUNNER, treating it as a model load for VRAM
        accounting: estimate its footprint, evict other loaded models to make room,
        then register it in the loaded-model maps so a later load can evict IT in
        turn (and so the dashboard/eviction see its VRAM). 1:1 with its gguf."""
        wsm = self.whisper_servers.get(model_id)
        if wsm is None:
            return False
        ws_key = f"audio:{model_id}"
        if wsm.is_running():
            self.models[ws_key] = wsm
            self.models_in_vram.add(ws_key)
            return True
        mp = model_path or getattr(wsm, "_model_path", None)
        gd = gpu_device if gpu_device is not None else getattr(wsm, "_gpu_device", 0)
        needed = self._estimate_gguf_vram_gb(mp)
        if needed > 0 and self._get_free_vram_gb() < needed:
            print(f"Whisper start: need ~{needed:.1f} GB VRAM for '{model_id}' — evicting")
            self._evict_models_for_vram(needed)
        wsm.start(mp, gpu_device=int(gd or 0))
        if not wsm.is_running():
            return False
        # Register it like a loaded model so eviction/accounting can see + evict it.
        self.models[ws_key] = wsm
        self.model_pools.pop(ws_key, None)
        self.active_in_vram = ws_key
        self.current_model_key = ws_key
        self.models_in_vram.add(ws_key)
        if needed > 0:
            self._measured_vram_gb.setdefault(ws_key, needed)
        return True

    def stop_whisper_server(self, model_id: str) -> bool:
        """Stop a whisper-server runner and clear its VRAM accounting (frees VRAM)."""
        wsm = self.whisper_servers.get(model_id)
        ws_key = f"audio:{model_id}"
        was = bool(wsm and wsm.is_running())
        if wsm is not None:
            try:
                wsm.stop()
            except Exception:
                pass
        self.models.pop(ws_key, None)
        self.model_pools.pop(ws_key, None)
        self.models_in_vram.discard(ws_key)
        self._measured_vram_gb.pop(ws_key, None)
        if self.active_in_vram == ws_key:
            self.active_in_vram = None
        if self.current_model_key == ws_key:
            self.current_model_key = None
        return was

    def resolve_whisper_alias(self, name: str) -> Optional[WhisperServerManager]:
        """Return the next round-robin WhisperServerManager for an alias, or None."""
        ids = self.whisper_aliases.get(name)
        if not ids:
            return None
        idx = self._whisper_alias_counters.get(name, 0) % len(ids)
        self._whisper_alias_counters[name] = idx + 1
        return self.whisper_servers.get(ids[idx])

    def resolve_whisper_alias_model_id(self, name: str) -> Optional[str]:
        """Return an idle whisper-server model id for an alias, else a random one."""
        ids = self.whisper_aliases.get(name)
        if not ids:
            return None
        idle_ids = [mid for mid in ids if (self.whisper_servers.get(mid) and self.whisper_servers[mid].is_idle())]
        choices = idle_ids or list(ids)
        if len(choices) == 1:
            return choices[0]
        return random.choice(choices)
    
    def set_tts_model(self, model_name: str, config: Dict = None):
        """Set the text-to-speech model and download/cache it if needed."""
        self.tts_model = model_name
        self.config[f"tts:{model_name}"] = config or {}
        self._remember_registered_type(model_name, "tts")

        # Download/cache the model at startup if it's a URL or HF ID
        resolved_model = self.load_model(model_name)
        if resolved_model != model_name:
            # Model was downloaded/cached, update the stored name
            self.tts_model = resolved_model
            self.config[f"tts:{resolved_model}"] = self.config.pop(f"tts:{model_name}")
            print(f"TTS model '{model_name}' cached as: {resolved_model}")
    
    def set_image_model(self, model_name: str, config: Dict = None):
        """Add an image generation model and download/cache it if needed."""
        if model_name not in self.image_models:
            self.image_models.append(model_name)
        self.config[f"image:{model_name}"] = config or {}
        self._remember_registered_type(model_name, "image")

        # For image models, we don't download at startup since they may be large
        # and handled by different backends (diffusers vs sd.cpp)
        # The download will happen when the model is first requested
        print(f"Registered image model: {model_name}")
    
    def set_vision_model(self, model_name: str, config: Dict = None):
        """Add a vision model and download/cache it if needed."""
        if model_name not in self.vision_models:
            self.vision_models.append(model_name)
        self.config[f"vision:{model_name}"] = config or {}
        self._remember_registered_type(model_name, "vision")

        resolved_model = self.load_model(model_name)
        if resolved_model != model_name:
            idx = self.vision_models.index(model_name)
            self.vision_models[idx] = resolved_model
            self.config[f"vision:{resolved_model}"] = self.config.pop(f"vision:{model_name}")
            print(f"Vision model '{model_name}' cached as: {resolved_model}")

    def set_video_model(self, model_name: str, config: Dict = None):
        """Add a video generation model (t2v / i2v / v2v)."""
        if model_name not in self.video_models:
            self.video_models.append(model_name)
        self.config[f"video:{model_name}"] = config or {}
        self._remember_registered_type(model_name, "video")
        print(f"Registered video model: {model_name}")

    def set_audio_gen_model(self, model_name: str, config: Dict = None):
        """Add a music/audio generation model (MusicGen, AudioLDM2, …)."""
        if model_name not in self.audio_gen_models:
            self.audio_gen_models.append(model_name)
        self.config[f"audio_gen:{model_name}"] = config or {}
        self._remember_registered_type(model_name, "audio_gen")
        print(f"Registered audio-gen model: {model_name}")

    def set_embedding_model(self, model_name: str, config: Dict = None):
        """Add a text/image embedding model."""
        if model_name not in self.embedding_models:
            self.embedding_models.append(model_name)
        self.config[f"embedding:{model_name}"] = config or {}
        self._remember_registered_type(model_name, "embedding")
        print(f"Registered embedding model: {model_name}")

    def set_spatial_model(self, model_name: str, config: Dict = None):
        """Add a depth estimation / segmentation / object detection model."""
        if model_name not in self.spatial_models:
            self.spatial_models.append(model_name)
        self.config[f"spatial:{model_name}"] = config or {}
        self._remember_registered_type(model_name, "spatial")
        print(f"Registered spatial model: {model_name}")

    def set_model_alias(self, alias: str, model_name: str):
        """Register an alias for a model."""
        self.model_aliases[alias] = model_name
        for model_type in self._registered_types_for(model_name):
            self._remember_registered_type(alias, model_type)

    def _config_for_model(self, name) -> dict:
        """Per-model config dict, tolerant of the id form the caller used.

        ``self.config`` is keyed by the registration id (usually the model's full
        path), but on-demand loads often arrive as a *basename* (e.g.
        ``gemma-…​.gguf``). A bare ``self.config.get(basename)`` then misses and
        returns ``{}``, so every per-model setting (n_ctx, flash_attn, parser,
        cache quant, …) is silently dropped and global defaults are used. Resolve
        through: exact id → alias map → basename / basename-without-extension."""
        if not name:
            return {}
        cfg = self.config.get(name)
        if cfg:
            return cfg
        target = self.model_aliases.get(name)
        if target and target != name:
            cfg = self.config.get(target)
            if cfg:
                return cfg
        import os
        base = os.path.basename(str(name))
        base_noext = base[:-5] if base.endswith(".gguf") else base
        for key, kcfg in self.config.items():
            if not kcfg:
                continue
            kbase = os.path.basename(str(key))
            kbase_noext = kbase[:-5] if kbase.endswith(".gguf") else kbase
            if kbase == base or kbase_noext == base_noext:
                return kcfg
        return {}
    
    def set_assigned_models(self, keys) -> None:
        """Restrict list_models() to the front-assigned subset (route-keys: alias /
        path / id). None = no restriction."""
        self._assigned_model_keys = set(keys) if keys is not None else None

    def _entry_assigned(self, m) -> bool:
        """True if a models.json entry is assigned to this engine (or no restriction)."""
        if self._assigned_model_keys is None:
            return True
        if isinstance(m, str):
            rk = m
        elif isinstance(m, dict):
            rk = m.get("alias") or m.get("path") or m.get("id")
        else:
            return True
        return rk in self._assigned_model_keys

    def get_all_allowed_identifiers(self) -> set:
        """
        Return the set of all model names, aliases, and identifiers that are
        valid for API requests.
        """
        allowed = set()

        # Default / text model
        if self.default_model:
            allowed.add(self.default_model)
            short = self.default_model.split("/")[-1] if "/" in self.default_model else self.default_model
            allowed.add(short)
            allowed.add("default")

        # Audio models
        if self.audio_models:
            allowed.add("audio")
            for m in self.audio_models:
                allowed.add(m)
                allowed.add(f"audio:{m}")
        for alias in self.whisper_aliases:
            allowed.add(alias)
            allowed.add(f"audio:{alias}")

        # TTS model
        if self.tts_model:
            allowed.add("tts")
            allowed.add(self.tts_model)
            allowed.add(f"tts:{self.tts_model}")

        # Image models
        if self.image_models:
            allowed.add("image")
            for m in self.image_models:
                allowed.add(m)
                allowed.add(f"image:{m}")

        # Vision models
        if self.vision_models:
            allowed.add("vision")
            for m in self.vision_models:
                allowed.add(m)
                allowed.add(f"vision:{m}")

        # Video models
        if self.video_models:
            allowed.add("video")
            for m in self.video_models:
                allowed.add(m)
                allowed.add(f"video:{m}")

        # Audio generation models
        if self.audio_gen_models:
            allowed.add("audio_gen")
            for m in self.audio_gen_models:
                allowed.add(m)
                allowed.add(f"audio_gen:{m}")

        # Embedding models
        if self.embedding_models:
            allowed.add("embedding")
            for m in self.embedding_models:
                allowed.add(m)
                allowed.add(f"embedding:{m}")

        # Spatial models (depth estimation, segmentation, detection)
        if self.spatial_models:
            allowed.add("spatial")
            for m in self.spatial_models:
                allowed.add(m)
                allowed.add(f"spatial:{m}")

        # Custom aliases
        for alias in self.model_aliases:
            allowed.add(alias)

        # Also include all models from config (covers configured-but-not-yet-loaded models)
        try:
            from codai.admin.routes import config_manager
            if config_manager is not None:
                md = config_manager.models_data
                for cat in ("text_models", "image_models", "audio_models",
                            "gguf_models", "tts_models", "vision_models",
                            "video_models", "audio_gen_models", "embedding_models",
                            "spatial_models"):
                    for m in md.get(cat, []):
                        if isinstance(m, str):
                            mid = m
                        elif m.get("backend") == "whisper-server":
                            mid = m.get("id") or ""
                        else:
                            mid = m.get("alias") or m.get("path") or m.get("id") or ""
                        raw = (m if isinstance(m, str) else m.get("path") or m.get("id") or "")
                        alias = "" if isinstance(m, str) else (m.get("alias") or "")
                        vals = [mid, raw]
                        if alias:
                            vals.append(alias)
                        for val in vals:
                            if val:
                                allowed.add(val)
                                short = val.split("/")[-1] if "/" in val else val
                                allowed.add(short)
        except Exception:
            pass

        return allowed

    def get_registered_model_type(self, name: str) -> Optional[str]:
        """
        Return the type a model is registered under ("text", "image", "audio",
        "tts", "vision", "video", "audio_gen", "embedding"), or None if unknown.
        Short-name (filename) matching is used so full paths resolve correctly.
        """
        def _matches(registered: str) -> bool:
            if name == registered:
                return True
            n_short = name.split("/")[-1] if "/" in name else name
            r_short = registered.split("/")[-1] if "/" in registered else registered
            return n_short == r_short

        requested_type = self._requested_type_from_registered_types(name)
        if requested_type:
            return requested_type
        if self._registered_types_for(name):
            return None

        if self.default_model and _matches(self.default_model):
            return "text"
        for m in self.image_models:
            if _matches(m):
                return "image"
        for m in self.audio_models:
            if _matches(m):
                return "audio"
        if self.tts_model and _matches(self.tts_model):
            return "tts"
        for m in self.vision_models:
            if _matches(m):
                return "vision"
        for m in self.video_models:
            if _matches(m):
                return "video"
        for m in self.audio_gen_models:
            if _matches(m):
                return "audio_gen"
        for m in self.embedding_models:
            if _matches(m):
                return "embedding"
        for m in self.spatial_models:
            if _matches(m):
                return "spatial"
        return None

    def _remember_registered_type(self, name: str, model_type: str) -> None:
        """Remember every configured type for a model identifier and short name."""
        if not name or not model_type:
            return
        for key in {name, name.split("/")[-1] if "/" in name else name}:
            self.model_registered_types.setdefault(key, set()).add(model_type)

    def _registered_types_for(self, name: str) -> Set[str]:
        """Return all configured types for a model identifier or its short name."""
        if not name:
            return set()
        short = name.split("/")[-1] if "/" in name else name
        types = set(self.model_registered_types.get(name, set()))
        types.update(self.model_registered_types.get(short, set()))
        for key, vals in self.model_registered_types.items():
            key_short = key.split("/")[-1] if "/" in key else key
            if key == name or key_short == short:
                types.update(vals)
        # Live admin saves update models.json/config_manager immediately, but an
        # already-running manager may not have had every category re-registered.
        # Treat the saved config as authoritative so entries with model_types like
        # text+image don't get rejected just because the image registration won.
        types.update(self._registered_types_from_config(name))
        return types

    def _registered_types_from_config(self, name: str) -> Set[str]:
        """Infer all configured types for a model from config_manager.models_data."""
        cat_type = {
            "text_models": "text",
            "gguf_models": "text",
            "vision_models": "vision",
            "image_models": "image",
            "audio_models": "audio",
            "tts_models": "tts",
            "video_models": "video",
            "audio_gen_models": "audio_gen",
            "embedding_models": "embedding",
            "spatial_models": "spatial",
        }
        cfg_cat_type = {
            "text_models": "text",
            "gguf_models": "text",
            "vision_models": "vision",
            "image_models": "image",
            "audio_models": "audio",
            "tts_models": "tts",
            "video_models": "video",
            "audio_gen_models": "audio_gen",
            "embedding_models": "embedding",
            "spatial_models": "spatial",
        }
        found: Set[str] = set()
        short = name.split("/")[-1] if "/" in name else name
        try:
            from codai.admin.routes import config_manager
            md = config_manager.models_data if config_manager is not None else {}
        except Exception:
            return found
        for cat, entries in md.items():
            default_type = cat_type.get(cat)
            if not default_type:
                continue
            for entry in entries or []:
                if isinstance(entry, str):
                    vals = [entry]
                    entry_types = [default_type]
                else:
                    raw = entry.get("path") or entry.get("id") or ""
                    alias = entry.get("alias") or ""
                    vals = [raw, alias]
                    raw_types = entry.get("model_types") or [entry.get("model_type") or cat]
                    entry_types = [cfg_cat_type.get(t, default_type) for t in raw_types if cfg_cat_type.get(t, default_type)]
                for val in vals:
                    if not val:
                        continue
                    val_short = val.split("/")[-1] if "/" in val else val
                    if val == name or val_short == short:
                        found.update(entry_types)
        return found

    def _requested_type_from_registered_types(self, name: str) -> Optional[str]:
        """Return a single registered type only when the model is not multi-type."""
        types = self._registered_types_for(name)
        return next(iter(types)) if len(types) == 1 else None

    def model_supports_type(self, name: str, model_type: Optional[str]) -> bool:
        """True when a configured multi-type model supports the requested type."""
        if not model_type:
            return True
        types = self._registered_types_for(name)
        if model_type in types:
            return True
        return model_type == "text" and "vision" in types

    def _config_for_model_key(self, model_key: str) -> Dict[str, Any]:
        """Return config for a key, falling back to compatible multi-type keys."""
        cfg = self.config.get(model_key, {})
        if cfg:
            return cfg
        if ":" in model_key:
            _, bare = model_key.split(":", 1)
            return self.config.get(bare, {})
        bare = model_key
        for prefix in ("vision", "image", "audio", "tts", "video", "audio_gen", "embedding", "spatial"):
            cfg = self.config.get(f"{prefix}:{bare}", {})
            if cfg:
                return cfg
        return {}

    def is_allowed_model(self, requested_or_resolved: str, model_type: str = None) -> bool:
        """
        Check if a model name (raw request value *or* resolved name) is one of
        the models registered via the command line or their aliases.

        Args:
            requested_or_resolved: The model name/path/alias to check.
            model_type: Optional type hint ("image", "text", "audio", "tts",
                        "vision").  When provided the check is scoped to that
                        type first; if it fails it still falls back to the
                        full allowed set.

        Returns:
            True if the model is registered/allowed, False otherwise.
        """
        if not requested_or_resolved:
            return False

        # ds4-served DeepSeek V4 has no models.json entry; accept it for text when
        # the ds4 worker is enabled and the name matches.
        if model_type in (None, "text") and ds4_should_handle(requested_or_resolved):
            return True

        # If a model_type is specified, reject models registered under a
        # different type (e.g. an image GGUF requested via /v1/chat/completions).
        if model_type:
            registered_types = self._registered_types_for(requested_or_resolved)
            if registered_types and not self.model_supports_type(requested_or_resolved, model_type):
                return False
            registered_type = self.get_registered_model_type(requested_or_resolved)
            if registered_type is not None and registered_type != model_type:
                # "vision" models are acceptable for "text" endpoints (multimodal)
                if not (model_type == "text" and registered_type == "vision"):
                    return False

        # Quick check against the full set of allowed identifiers
        allowed = self.get_all_allowed_identifiers()
        if requested_or_resolved in allowed:
            return True

        # Also accept prefixed forms that the caller may have built
        if model_type and model_type != "text":
            prefixed = f"{model_type}:{requested_or_resolved}"
            if prefixed in allowed:
                return True

        # Short-name match: compare the last path component of the request
        # against all allowed identifiers
        req_short = requested_or_resolved.split("/")[-1] if "/" in requested_or_resolved else None
        if req_short and req_short in allowed:
            return True

        # Reverse short-name match: check if any allowed identifier ends with
        # the same filename as the request
        if req_short:
            for a in allowed:
                a_short = a.split("/")[-1] if "/" in a else a
                if a_short == req_short:
                    return True

        return False
    
    def get_model_for_request(self, requested_model: str):
        """Get the appropriate model manager for a request based on model name."""
        global global_args
        
        # Resolve custom aliases first
        if requested_model in self.model_aliases:
            requested_model = self.model_aliases[requested_model]
        
        # Handle empty or "default" model names
        if not requested_model or requested_model == "default":
            if self.default_model:
                if self.default_model in self.models:
                    return self._get_least_busy_instance(self.default_model)
                return self._load_default_model()
            return None

        # Handle "audio" alias
        if requested_model == "audio":
            if self.audio_models:
                key = f"audio:{self.audio_models[0]}"
                if key in self.models:
                    return self._get_least_busy_instance(key)
            return None

        # Handle "image" alias
        if requested_model == "image":
            if self.image_models:
                key = f"image:{self.image_models[0]}"
                if key in self.models:
                    return self._get_least_busy_instance(key)
            return None

        # Handle "tts" alias
        if requested_model == "tts":
            if self.tts_model:
                key = f"tts:{self.tts_model}"
                if key in self.models:
                    return self._get_least_busy_instance(key)
            return None

        # Handle prefixed models
        if requested_model.startswith("audio:"):
            key = f"audio:{requested_model[6:]}"
            if key in self.models:
                return self._get_least_busy_instance(key)
            return None

        if requested_model.startswith("tts:"):
            key = f"tts:{requested_model[4:]}"
            if key in self.models:
                return self._get_least_busy_instance(key)
            return None

        if requested_model.startswith("vision:") or requested_model.startswith("image:"):
            image_name = requested_model[7:] if requested_model.startswith("vision:") else requested_model[6:]
            key = f"image:{image_name}"
            if key in self.models:
                return self._get_least_busy_instance(key)
            return None

        # Check if it's the default model
        if self.default_model and (requested_model == self.default_model or
                                    requested_model.endswith(self.default_model.split("/")[-1])):
            if self.default_model in self.models:
                return self._get_least_busy_instance(self.default_model)
            return self._load_default_model()

        # Check if any loaded model matches
        for key in self.models:
            if requested_model in key or key.endswith(requested_model.split("/")[-1]):
                return self._get_least_busy_instance(key)
        
        # Validate the model is allowed before attempting to load it.
        # This prevents loading arbitrary models not registered via command line.
        if not self.is_allowed_model(requested_model):
            print(f"Model '{requested_model}' is not an allowed model. Rejecting request.")
            return None
        
        # Model not found but allowed - try to load it
        return self._load_model_by_name(requested_model)
    
    def resolve_model_name(self, requested_model: str) -> Optional[str]:
        """
        Resolve a model name to its canonical form.
        
        Handles:
        - Aliases ("default", "image", "audio", "tts")
        - Custom aliases from model_aliases dict
        - Prefixed models ("image:", "audio:", "tts:", "vision:")
        - Default model resolution
        
        Returns the canonical model name/path, or None if not resolvable.
        """
        # Handle None or empty
        if not requested_model:
            return self.default_model
        
        # Resolve custom aliases first
        if requested_model in self.model_aliases:
            requested_model = self.model_aliases[requested_model]
        
        # Handle "default" alias
        if requested_model == "default":
            return self.default_model
        
        # Handle "audio" alias
        if requested_model == "audio":
            return f"audio:{self.audio_models[0]}" if self.audio_models else None
        
        # Handle "image" alias
        if requested_model == "image":
            return f"image:{self.image_models[0]}" if self.image_models else None
        
        # Handle "tts" alias
        if requested_model == "tts":
            return f"tts:{self.tts_model}" if self.tts_model else None
        
        # Handle "vision" alias
        if requested_model == "vision":
            return f"image:{self.vision_models[0]}" if self.vision_models else None

        # Handle "video" alias
        if requested_model == "video":
            return f"video:{self.video_models[0]}" if self.video_models else None

        # Handle prefixed models - normalize them
        if requested_model.startswith("audio:"):
            return requested_model
        if requested_model.startswith("tts:"):
            return requested_model
        if requested_model.startswith("video:"):
            return requested_model
        if requested_model.startswith("image:") or requested_model.startswith("vision:"):
            # Normalize vision: to image:
            if requested_model.startswith("vision:"):
                return f"image:{requested_model[7:]}"
            return requested_model
        
        # Check if it matches the default model (with or without path)
        if self.default_model:
            if requested_model == self.default_model:
                return self.default_model
            # Check if it's a short name match
            if requested_model.endswith(self.default_model.split("/")[-1]) or \
               self.default_model.endswith(requested_model.split("/")[-1]):
                return self.default_model
        
        # Check if it matches any loaded model key
        for key in self.models.keys():
            if requested_model in key or key.endswith(requested_model.split("/")[-1]):
                return key
        
        # Return as-is if no resolution
        return requested_model
    
    def get_currently_loaded_model_name(self) -> Optional[str]:
        """
        Get the canonical name of the model currently loaded in VRAM.

        Returns the model key from self.models if any model is loaded,
        or None if no models are loaded.
        """
        if not self.models:
            return None

        # If we have a tracked current model, return it
        if self.current_model_key and self.current_model_key in self.models:
            return self.current_model_key

        # Otherwise return the first loaded model (there should only be one in ondemand mode)
        return list(self.models.keys())[0] if self.models else None

    def get_cached_model_path(self, model_path: str) -> Optional[str]:
        """
        Check if a model is already cached.

        This is a proxy method to the cache module function.
        Returns the cached path if the model is cached, None otherwise.
        """
        return get_cached_model_path(model_path)

    def get_model_cache_dir(self) -> str:
        """
        Get the model cache directory.

        This is a proxy method to the cache module function.
        Returns the path to the model cache directory.
        """
        return get_model_cache_dir()

    def load_model(self, model_path: str, cache_dir: Optional[str] = None, file_pattern: str = '.gguf') -> Optional[str]:
        """
        Load a model with intelligent caching and resolution.

        Handles local files, URLs, and HuggingFace model IDs.
        Returns the resolved model path or identifier.
        """
        from codai.models.cache import is_huggingface_model_id

        # 1. Check if it's a local file
        if os.path.isfile(model_path):
            print(f"Using local model: {model_path}")
            return model_path

        # 2. Check if it's a URL
        if model_path.startswith('http://') or model_path.startswith('https://'):
            print(f"Loading model from URL: {model_path}")
            return load_model(model_path, cache_dir, file_pattern)

        # 3. Check if it's a HuggingFace model ID
        if is_huggingface_model_id(model_path):
            # For diffusers models (most image models), return the identifier
            # The actual loading will be handled by the specific backend (diffusers, sd.cpp, etc.)
            print(f"Using HuggingFace model: {model_path}")
            return model_path

        # 4. Try as a generic model identifier with caching
        print(f"Resolving model: {model_path}")
        cached_path = get_cached_model_path(model_path)
        if cached_path:
            print(f"Using cached model: {cached_path}")
            return cached_path

        # 5. Try to download it
        return load_model(model_path, cache_dir, file_pattern)

    
    def _move_model_to_cpu(self, model_key: str):
        """
        Move a model from VRAM to CPU RAM (for loadswap mode).
        
        The model stays in self.models but is moved to CPU so it doesn't
        consume VRAM. It can be moved back to VRAM later.
        """
        model_obj = self.models.get(model_key)
        if model_obj is None:
            return
        
        print(f"Moving model '{model_key}' from VRAM to CPU RAM...")
        
        try:
            import torch
            
            # Case 1: ModelManager with a backend
            if isinstance(model_obj, ModelManager) and model_obj.backend is not None:
                backend = model_obj.backend
                if hasattr(backend, 'model') and backend.model is not None:
                    if hasattr(backend.model, 'to'):
                        try:
                            backend.model.to('cpu')
                            print(f"  Moved backend model to CPU")
                        except Exception as e:
                            print(f"  Warning: Could not move backend model to CPU: {e}")
                    # For llama-cpp-python models, we can't move to CPU easily
                    # They stay in memory but we track them
            
            # Case 2: Diffusers pipeline (has 'to' method)
            elif hasattr(model_obj, 'to') and callable(getattr(model_obj, 'to')):
                try:
                    model_obj.to('cpu')
                    print(f"  Moved diffusers pipeline to CPU")
                except Exception as e:
                    print(f"  Warning: Could not move pipeline to CPU: {e}")
            
            # Case 3: Object with a model attribute
            elif hasattr(model_obj, 'model') and model_obj.model is not None:
                if hasattr(model_obj.model, 'to'):
                    try:
                        model_obj.model.to('cpu')
                        print(f"  Moved inner model to CPU")
                    except Exception as e:
                        print(f"  Warning: Could not move inner model to CPU: {e}")
            
            # Clear CUDA cache after moving to CPU
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            
            gc.collect()
            
        except ImportError:
            print(f"  Warning: torch not available, cannot move model to CPU")
        except Exception as e:
            print(f"  Warning during CPU offload of '{model_key}': {e}")
    
    def _move_model_to_vram(self, model_key: str):
        """
        Move a model from CPU RAM back to VRAM (for loadswap mode).
        """
        model_obj = self.models.get(model_key)
        if model_obj is None:
            return
        
        print(f"Moving model '{model_key}' from CPU RAM to VRAM...")
        
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Case 1: ModelManager with a backend
            if isinstance(model_obj, ModelManager) and model_obj.backend is not None:
                backend = model_obj.backend
                if hasattr(backend, 'model') and backend.model is not None:
                    if hasattr(backend.model, 'to'):
                        try:
                            backend.model.to(device)
                            print(f"  Moved backend model to {device}")
                        except Exception as e:
                            print(f"  Warning: Could not move backend model to {device}: {e}")
            
            # Case 2: Diffusers pipeline
            elif hasattr(model_obj, 'to') and callable(getattr(model_obj, 'to')):
                try:
                    model_obj.to(device)
                    print(f"  Moved diffusers pipeline to {device}")
                except Exception as e:
                    print(f"  Warning: Could not move pipeline to {device}: {e}")
            
            # Case 3: Object with a model attribute
            elif hasattr(model_obj, 'model') and model_obj.model is not None:
                if hasattr(model_obj.model, 'to'):
                    try:
                        model_obj.model.to(device)
                        print(f"  Moved inner model to {device}")
                    except Exception as e:
                        print(f"  Warning: Could not move inner model to {device}: {e}")
            
        except ImportError:
            print(f"  Warning: torch not available, cannot move model to VRAM")
        except Exception as e:
            print(f"  Warning during VRAM load of '{model_key}': {e}")

    def _get_free_vram_gb(self) -> float:
        """Return estimated free VRAM in GB, or a large number if unavailable."""
        try:
            import torch
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                return free / 1e9
        except Exception:
            pass
        # AMD GPU via sysfs (covers Vulkan/ROCm on Linux)
        try:
            import glob
            for total_path in sorted(glob.glob('/sys/class/drm/card*/device/mem_info_vram_total')):
                used_path = total_path.replace('vram_total', 'vram_used')
                with open(total_path) as f:
                    total = int(f.read().strip())
                with open(used_path) as f:
                    used = int(f.read().strip())
                return (total - used) / 1e9
        except Exception:
            pass
        return 999.0  # Unknown — assume enough

    @staticmethod
    def _free_vram_snapshot() -> float:
        """Return current free VRAM in bytes (CUDA) or -1 if unavailable."""
        try:
            import torch
            if torch.cuda.is_available():
                free, _ = torch.cuda.mem_get_info()
                return float(free)
        except Exception:
            pass
        return -1.0

    def vram_before_load(self) -> float:
        """Call immediately before loading a model; returns a snapshot for delta measurement."""
        return self._free_vram_snapshot()

    def record_vram_delta(self, model_key: str, free_before: float,
                          offloaded: bool = False) -> None:
        """Call immediately after a model finishes loading to record actual VRAM consumed.

        If the measured value exceeds the stored estimate by more than 10%, the real
        value is written back into the model config and persisted to models.json so
        future eviction decisions use the accurate figure.

        ``offloaded`` MUST be True when the model was loaded with any CPU/disk
        offload strategy (model/sequential/group/balanced/disk). In that case the
        weights are not resident on the GPU, so the measured delta is a tiny,
        meaningless lower bound — recording it would clobber a real full-GPU
        measurement and make the next start under-estimate the footprint, pick a
        full-GPU load, and OOM. So we skip recording entirely and keep the prior
        estimate/measurement intact.
        """
        if free_before < 0:
            return
        if offloaded:
            return
        free_after = self._free_vram_snapshot()
        if free_after < 0:
            return
        delta_gb = (free_before - free_after) / 1e9
        if delta_gb <= 0:
            return

        # The delta is the resident WEIGHT footprint measured at load time. Add
        # the runtime reserve (KV cache / activations / VAE-decode spike) so the
        # value we cache and persist reflects the model's PEAK runtime need — not
        # just its loaded weights — and future eviction frees enough headroom.
        cfg = self._config_for_model_key(model_key)
        reserve_gb = self._runtime_reserve_gb(
            cfg if isinstance(cfg, dict) else {}, model_key, delta_gb)
        measured = round(delta_gb + reserve_gb, 3)
        # In-memory truth for the rest of this run (used by eviction estimates).
        self._measured_vram_gb[model_key] = measured
        print(f"Measured VRAM for '{model_key}': {delta_gb:.2f} GB weights "
              f"+ {reserve_gb:.2f} GB runtime reserve = {measured:.2f} GB")

        # Persist the real value into the SEPARATE, system-owned
        # `measured_vram_gb` field (never the user's `used_vram_gb`). Rules:
        #   * force_vram_update set → refresh it EVERY run, even when
        #     used_vram_gb is configured.
        #   * otherwise → only when used_vram_gb is NOT configured (a configured
        #     value is authoritative and must remain unchanged).
        force_update = bool(cfg.get("force_vram_update")) if isinstance(cfg, dict) else False
        if not force_update:
            if isinstance(cfg, dict) and cfg.get("used_vram_gb") is not None:
                return  # configured & not forced → leave it exactly as-is
            # Avoid churning models.json when the measurement hasn't meaningfully
            # changed from what's already persisted.
            prev = cfg.get("measured_vram_gb") if isinstance(cfg, dict) else None
            try:
                if prev is not None and abs(measured - float(prev)) <= max(0.5, float(prev) * 0.10):
                    return
            except (TypeError, ValueError):
                pass
        try:
            _why = "force_vram_update" if force_update else "no used_vram_gb configured"
            print(f"  Saving measured_vram_gb for '{model_key}': {measured:.2f} GB ({_why})")
            cfg = dict(cfg) if isinstance(cfg, dict) else {}
            cfg["measured_vram_gb"] = measured
            self.config[model_key] = cfg
            # Persist to models.json via config_manager (separate field; the
            # user's own keys are never touched).
            from codai.admin.routes import config_manager
            if config_manager is not None:
                bare = model_key.split(":", 1)[1] if ":" in model_key else model_key
                for cat in ("text_models", "image_models", "audio_models", "tts_models",
                            "vision_models", "video_models", "audio_gen_models",
                            "embedding_models", "spatial_models"):
                    for entry in config_manager.models_data.get(cat, []):
                        if not isinstance(entry, dict):
                            continue
                        epath = entry.get("path") or entry.get("id") or ""
                        if epath == bare or epath.split("/")[-1] == bare.split("/")[-1]:
                            entry["measured_vram_gb"] = measured
                config_manager.save_models()
        except Exception as e:
            print(f"  Warning: could not persist measured_vram_gb: {e}")

    # In-process cache: model_id -> size_bytes (never stale within a run)
    _hf_size_cache: Dict[str, int] = {}

    @staticmethod
    def _hf_cached_model_size_bytes(model_id: str) -> int:
        """Return total weight-file bytes for a HuggingFace model.

        Tries in order:
        1. In-process memory cache (no I/O).
        2. Local HuggingFace hub cache scan.
        3. HuggingFace API (network, one call per model per process lifetime).

        For LoRA adapters (repos containing adapter_config.json) the size of
        the base model is added so that VRAM requirements are not
        underestimated.

        Returns 0 on any failure.
        """
        if model_id in MultiModelManager._hf_size_cache:
            return MultiModelManager._hf_size_cache[model_id]

        weight_exts = {'.safetensors', '.bin', '.gguf', '.ggml', '.pt'}

        def _resolve_base_model(base_model_id: str) -> int:
            from codai.models.cache import is_huggingface_model_id
            if not base_model_id or base_model_id == model_id:
                return 0
            if not is_huggingface_model_id(base_model_id):
                return 0
            return MultiModelManager._hf_cached_model_size_bytes(base_model_id)

        # --- Try local HF hub cache first (no network) ---
        try:
            from huggingface_hub import scan_cache_dir
            from codai.models.cache import get_all_cache_dirs
            import json as _json
            hf_dir = get_all_cache_dirs().get("huggingface")
            if hf_dir:
                info = scan_cache_dir(hf_dir)
                for repo in info.repos:
                    if repo.repo_id != model_id:
                        continue
                    revs = sorted(repo.revisions, key=lambda r: r.last_modified, reverse=True)
                    if revs:
                        rev = revs[0]
                        # Separate GGUF files (alternative quantizations of the same model)
                        # from dense weight files (safetensors/bin shards that collectively
                        # represent one model variant).  Summing all GGUFs produces wildly
                        # inflated estimates; instead take the largest single GGUF as the
                        # worst-case for that format, and sum the dense shards separately.
                        gguf_exts = {'.gguf', '.ggml'}
                        dense_exts = {'.safetensors', '.bin', '.pt'}
                        gguf_sizes = [
                            f.size_on_disk for f in rev.files
                            if os.path.splitext(f.file_name)[1].lower() in gguf_exts
                        ]
                        dense_total = sum(
                            f.size_on_disk for f in rev.files
                            if os.path.splitext(f.file_name)[1].lower() in dense_exts
                        )
                        # Use whichever single-load format is smaller (prefer the real
                        # on-disk format that will actually be used).
                        largest_gguf = max(gguf_sizes) if gguf_sizes else 0
                        total = dense_total if dense_total > 0 else largest_gguf
                        if total == 0:
                            total = largest_gguf  # fallback
                        # LoRA adapter: add base model size
                        for f in rev.files:
                            if f.file_name == "adapter_config.json":
                                try:
                                    with open(f.file_path) as fp:
                                        adapter_cfg = _json.load(fp)
                                    base_id = (
                                        adapter_cfg.get("base_model_name_or_path")
                                        or adapter_cfg.get("base_model")
                                    )
                                    total += _resolve_base_model(base_id)
                                except Exception:
                                    pass
                                break
                        if total > 0:
                            MultiModelManager._hf_size_cache[model_id] = total
                            return total
        except Exception:
            pass

        # --- Fallback: HuggingFace API (one network call, result cached) ---
        try:
            import urllib.request, urllib.parse, json as _json
            safe_id = urllib.parse.quote(model_id, safe="/")
            url = f"https://huggingface.co/api/models/{safe_id}"
            req = urllib.request.Request(url, headers={"User-Agent": "coderai/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = _json.loads(resp.read())
            gguf_sizes_api = []
            dense_total_api = 0
            has_adapter_config = False
            for sib in data.get("siblings", []):
                name = sib.get("rfilename", "")
                if name == "adapter_config.json":
                    has_adapter_config = True
                    continue
                ext = os.path.splitext(name)[1].lower()
                if ext not in weight_exts:
                    continue
                lfs = sib.get("lfs") or {}
                size = lfs.get("size") or sib.get("size") or 0
                if ext in {'.gguf', '.ggml'}:
                    gguf_sizes_api.append(size)
                else:
                    dense_total_api += size
            largest_gguf_api = max(gguf_sizes_api) if gguf_sizes_api else 0
            total = dense_total_api if dense_total_api > 0 else largest_gguf_api
            # LoRA adapter: fetch adapter_config.json to get the base model
            if has_adapter_config:
                try:
                    cfg_url = f"https://huggingface.co/{model_id}/resolve/main/adapter_config.json"
                    cfg_req = urllib.request.Request(cfg_url, headers={"User-Agent": "coderai/1.0"})
                    with urllib.request.urlopen(cfg_req, timeout=10) as resp:
                        adapter_cfg = _json.loads(resp.read())
                    base_id = (
                        adapter_cfg.get("base_model_name_or_path")
                        or adapter_cfg.get("base_model")
                    )
                    total += _resolve_base_model(base_id)
                except Exception:
                    pass
            if total > 0:
                MultiModelManager._hf_size_cache[model_id] = total
                return total
        except Exception:
            pass

        # Cache the miss too so we don't hammer the API on every call
        MultiModelManager._hf_size_cache[model_id] = 0
        return 0

    @staticmethod
    def _quant_divisor(mode) -> float:
        """Approx. VRAM reduction factor (vs fp16) for a quantization mode.

        Accepts the strings used in coderai configs / component_quantization,
        including GGUF file paths (the Qn level is sniffed from the name).
        Returns 1.0 for full precision / unknown.
        """
        s = str(mode or "").strip().lower()
        if not s or s in ("none", "full", "fp16", "bf16", "f16", "f32", "fp32", "default"):
            return 1.0
        # GGUF file or label → infer from the Qn tag (Q2..Q8).
        if "gguf" in s or s.endswith(".gguf") or s.startswith("q"):
            if "q2" in s or "2bit" in s:
                return 6.0
            if "q3" in s or "3bit" in s:
                return 5.0
            if "q4" in s or "4bit" in s:
                return 4.0
            if "q5" in s or "5bit" in s:
                return 3.2
            if "q6" in s or "6bit" in s:
                return 2.67
            if "q8" in s or "8bit" in s:
                return 2.0
            return 3.0  # unknown GGUF level — moderate default
        norm = s.replace("_", "").replace("-", "")
        if norm in ("4bit", "int4", "nf4", "fp4"):
            return 4.0
        if norm in ("8bit", "int8", "fp8"):
            return 2.0
        if norm in ("2bit", "int2"):
            return 6.0
        if norm in ("3bit", "int3"):
            return 5.0
        if norm in ("5bit",):
            return 3.2
        if norm in ("6bit",):
            return 2.67
        return 1.0

    def _effective_quant_multiplier(self, cfg: dict,
                                    load_in_4bit: bool, load_in_8bit: bool,
                                    model_key: str = None,
                                    resolved_name: str = None) -> float:
        """Fraction of full-precision weight size that stays resident.

        When per-component quantization is configured AND the model's real
        per-component parameter shares can be scanned from disk, weight each
        component's quant divisor by its ACTUAL share of the parameters — so a
        pipeline that is e.g. 99 % 4-bit (Wan2.2: two 14B experts + text encoder
        quantized, only a 0.13B VAE dense) is estimated at ~0.28× rather than the
        blunt 0.475× a fixed 70/30 split would give (which inflated 25.8 GB →
        42.7 GB and forced needless offload). Falls back to a fixed-share
        heuristic, then to the global 4/8-bit flags.
        """
        comp_q = cfg.get("component_quantization") or {}

        def _comp_divisor(name: str) -> float:
            # VAEs are conv-only → bitsandbytes/quanto can't quantize them; they
            # stay dense regardless of what the config asks for.
            low = str(name).lower()
            if low == 'vae' or low.endswith('_vae') or low.startswith('vae'):
                return 1.0
            if name in comp_q:
                return self._quant_divisor(comp_q[name])
            if load_in_4bit:
                return 4.0
            if load_in_8bit:
                return 2.0
            return 1.0

        # --- Preferred: weight by real per-component parameter shares ---
        shares = {}
        if comp_q and model_key:
            try:
                shares = self._component_param_shares(model_key, resolved_name, cfg)
            except Exception:
                shares = {}
        if shares:
            total = sum(shares.values())
            if total > 0:
                mult = 0.0
                for comp, numel in shares.items():
                    div = _comp_divisor(comp)
                    eff = (1.0 / div) if div > 1.0 else 1.0
                    mult += (numel / total) * eff
                # bitsandbytes keeps fp32 absmax + scale tensors and fp16 compute
                # buffers alongside the packed 4-bit weights (~+12 %).
                return mult * 1.12

        divisors = [self._quant_divisor(v) for v in comp_q.values()]
        divisors = [d for d in divisors if d > 1.0]
        if divisors:
            avg_div = sum(divisors) / len(divisors)
            # No per-component scan available: assume ~70 % of the footprint is
            # shrunk by quantization and ~30 % (VAE, embeddings, fp16 compute
            # buffers, bitsandbytes scale/absmax tensors) stays dense.
            quantized_share = 0.7
            return quantized_share / avg_div + (1.0 - quantized_share)
        if load_in_4bit:
            # 4-bit nf4 keeps scale/absmax + fp16 buffers → effective ~0.3, not 0.25.
            return 0.3
        if load_in_8bit:
            return 0.55
        return 1.0

    @staticmethod
    def _model_type_of(model_key: str) -> str:
        """Extract the model type from a 'type:name' key (default 'text')."""
        return model_key.split(":", 1)[0] if ":" in model_key else "text"

    # bytes-per-element for safetensors dtype strings
    _DTYPE_BYTES = {
        'F64': 8, 'F32': 4, 'F16': 2, 'BF16': 2,
        'F8_E4M3': 1, 'F8_E5M2': 1, 'I64': 8, 'I32': 4,
        'I16': 2, 'I8': 1, 'U8': 1, 'BOOL': 1,
    }
    _storage_dtype_cache: Dict[str, float] = {}

    @classmethod
    def _safetensors_numel_bytes(cls, path: str):
        """Return (total_numel, total_weight_bytes) for a .safetensors file.

        Only the header (8-byte length + JSON) is read, never the weights. Used
        to compute a parameter-weighted average bytes/elem across a model so a
        MIXED-precision model (e.g. fp32 transformer + bf16 text encoder) isn't
        misjudged from a single file.
        """
        try:
            import json as _json
            with open(path, 'rb') as f:
                n = int.from_bytes(f.read(8), 'little')
                if n <= 0 or n > 100_000_000:
                    return 0, 0
                header = _json.loads(f.read(n))
        except Exception:
            return 0, 0
        total_numel = 0
        total_bytes = 0
        for name, meta in header.items():
            if name == '__metadata__' or not isinstance(meta, dict):
                continue
            dt = meta.get('dtype')
            b = cls._DTYPE_BYTES.get(dt)
            if not b:
                continue
            numel = 1
            for d in (meta.get('shape') or []):
                numel *= max(1, int(d))
            total_numel += numel
            total_bytes += numel * b
        return total_numel, total_bytes

    def _storage_bytes_per_elem(self, model_key: str, resolved_name: str, cfg: dict) -> float:
        """Detect the on-disk weight precision (bytes/elem) for a model.

        Computes the PARAMETER-WEIGHTED average bytes/elem across all of the
        model's safetensors files (total weight bytes / total params), so a
        mixed-precision model is handled correctly. Defaults to 2.0 (fp16) when
        it can't be determined, so the normalization is a no-op rather than a
        wrong guess. Cached per model.
        """
        ck = resolved_name or model_key
        cached = self._storage_dtype_cache.get(ck)
        if cached is not None:
            return cached

        import os
        result = 2.0  # safe default: treat unknown as fp16 (normalization no-op)
        candidates = []
        for v in (resolved_name, cfg.get('path'), cfg.get('model_path'), cfg.get('model')):
            if v and isinstance(v, str):
                candidates.append(v)

        st_files = []  # (size, path)
        try:
            # Local file or directory.
            for c in candidates:
                if os.path.isfile(c) and c.endswith('.safetensors'):
                    st_files.append((os.path.getsize(c), c))
                elif os.path.isdir(c):
                    for root, _, files in os.walk(c):
                        for fn in files:
                            if fn.endswith('.safetensors'):
                                p = os.path.join(root, fn)
                                try:
                                    st_files.append((os.path.getsize(p), p))
                                except OSError:
                                    pass
            # HuggingFace hub cache.
            if not st_files:
                from huggingface_hub import scan_cache_dir
                from codai.models.cache import get_all_cache_dirs, is_huggingface_model_id
                hf_dir = get_all_cache_dirs().get("huggingface")
                for c in candidates:
                    repo_id = c.split(":", 1)[1] if ":" in c else c
                    if not (hf_dir and is_huggingface_model_id(repo_id)):
                        continue
                    info = scan_cache_dir(hf_dir)
                    for repo in info.repos:
                        if repo.repo_id != repo_id:
                            continue
                        revs = sorted(repo.revisions, key=lambda r: r.last_modified, reverse=True)
                        if revs:
                            for fobj in revs[0].files:
                                if str(fobj.file_name).endswith('.safetensors'):
                                    st_files.append((fobj.size_on_disk, str(fobj.file_path)))
                    if st_files:
                        break
        except Exception:
            pass

        if st_files:
            # Parameter-weighted average across ALL files (mixed-precision aware).
            st_files.sort(reverse=True)  # largest first (cap the count we read)
            agg_numel = 0
            agg_bytes = 0
            for _, p in st_files[:64]:
                numel, wbytes = self._safetensors_numel_bytes(p)
                agg_numel += numel
                agg_bytes += wbytes
            if agg_numel > 0:
                result = agg_bytes / agg_numel

        self._storage_dtype_cache[ck] = result
        return result

    # component (top-level subfolder) -> total params, per model. Cached.
    _component_shares_cache: Dict[str, Dict[str, int]] = {}

    def _component_param_shares(self, model_key: str, resolved_name: str,
                                cfg: dict) -> Dict[str, int]:
        """Map each pipeline component → its parameter count (numel).

        Groups every ``.safetensors`` file by its TOP-LEVEL subfolder name
        (``transformer``, ``transformer_2``, ``text_encoder``, ``vae`` …) — i.e.
        the diffusers component layout — and sums the parameters per component.
        Used to weight the effective quantization multiplier by each component's
        real share of the model, so a pipeline that is 99 % quantized isn't
        treated as if 30 % stays dense. Returns {} when it can't be determined.
        """
        ck = resolved_name or model_key
        cached = self._component_shares_cache.get(ck)
        if cached is not None:
            return cached

        import os
        shares: Dict[str, int] = {}
        candidates = []
        for v in (resolved_name, cfg.get('path'), cfg.get('model_path'), cfg.get('model')):
            if v and isinstance(v, str):
                candidates.append(v)

        def _record(root_dir: str, file_path: str):
            rel = os.path.relpath(file_path, root_dir)
            comp = rel.split(os.sep)[0]
            # Files at the pipeline root (no subfolder) → bucket as 'root'.
            if comp.endswith('.safetensors'):
                comp = 'root'
            numel, _ = self._safetensors_numel_bytes(os.path.realpath(file_path))
            if numel > 0:
                shares[comp] = shares.get(comp, 0) + numel

        try:
            scanned = False
            for c in candidates:
                if os.path.isdir(c):
                    for r, _, files in os.walk(c):
                        for fn in files:
                            if fn.endswith('.safetensors'):
                                _record(c, os.path.join(r, fn))
                    scanned = True
                    break
            if not scanned:
                from huggingface_hub import scan_cache_dir
                from codai.models.cache import get_all_cache_dirs, is_huggingface_model_id
                hf_dir = get_all_cache_dirs().get("huggingface")
                for c in candidates:
                    repo_id = c.split(":", 1)[1] if ":" in c else c
                    if not (hf_dir and is_huggingface_model_id(repo_id)):
                        continue
                    info = scan_cache_dir(hf_dir)
                    for repo in info.repos:
                        if repo.repo_id != repo_id:
                            continue
                        revs = sorted(repo.revisions,
                                      key=lambda rv: rv.last_modified, reverse=True)
                        if revs:
                            snap = str(revs[0].snapshot_path)
                            for fobj in revs[0].files:
                                fp = str(fobj.file_path)
                                if fp.endswith('.safetensors'):
                                    _record(snap, fp)
                    if shares:
                        break
        except Exception:
            shares = {}

        self._component_shares_cache[ck] = shares
        return shares

    @staticmethod
    def _load_bytes_per_elem(cfg: dict) -> float:
        """Bytes/elem for the configured LOAD precision (default fp16 = 2)."""
        prec = str((cfg or {}).get('precision') or '').lower()
        if prec in ('f32', 'fp32', 'float32'):
            return 4.0
        if prec in ('f16', 'fp16', 'bf16', 'float16', 'bfloat16', 'half'):
            return 2.0
        return 2.0  # default load precision for HF/diffusers is half

    # LoRA adapters load into VRAM alongside the base weights; applying/fusing
    # them also briefly holds extra copies. Scale the raw adapter size to cover
    # that transient spike.
    _LORA_SPIKE_FACTOR = 1.3

    @classmethod
    def _path_or_hf_size_bytes(cls, spec: str) -> int:
        """Best-effort on-disk size for a LoRA/adapter spec (file, dir, or HF id)."""
        import os
        if not spec or not isinstance(spec, str):
            return 0
        try:
            if os.path.isfile(spec):
                return os.path.getsize(spec)
            if os.path.isdir(spec):
                total = 0
                for root, _, files in os.walk(spec):
                    for f in files:
                        if f.endswith((".safetensors", ".bin", ".pt", ".ckpt")):
                            try:
                                total += os.path.getsize(os.path.join(root, f))
                            except OSError:
                                pass
                return total
        except OSError:
            pass
        # HuggingFace repo id → scan local hub cache.
        try:
            from codai.models.cache import is_huggingface_model_id
            if is_huggingface_model_id(spec):
                return cls._hf_cached_model_size_bytes(spec)
        except Exception:
            pass
        return 0

    def _lora_vram_gb(self, specs) -> float:
        """Approx. extra VRAM (GB) for one or more LoRA adapters (× spike factor).

        `specs` may be a path/id string, a list of those, or a list of objects
        exposing a `.model` attribute (LoraConfig) / dicts with a 'model' key.
        """
        if not specs:
            return 0.0
        if isinstance(specs, (str, bytes)):
            specs = [specs]
        total_bytes = 0
        for s in specs:
            path = None
            if isinstance(s, str):
                path = s
            elif isinstance(s, dict):
                path = s.get("model") or s.get("path") or s.get("lora") or s.get("name")
            else:
                path = getattr(s, "model", None) or getattr(s, "path", None)
            if path:
                total_bytes += self._path_or_hf_size_bytes(path)
        if total_bytes <= 0:
            return 0.0
        return (total_bytes / 1e9) * self._LORA_SPIKE_FACTOR

    def _config_lora_vram_gb(self, cfg: dict) -> float:
        """Extra VRAM for LoRA(s) declared in the model config (lora_path/dir)."""
        if not isinstance(cfg, dict):
            return 0.0
        specs = []
        for key in ("lora_path", "lora_model_dir"):
            v = cfg.get(key)
            if v:
                specs.append(v)
        return self._lora_vram_gb(specs)

    @staticmethod
    def _kv_cache_size_factor(cache_type) -> float:
        """KV-cache bytes-per-element relative to f16 (=1.0), by llama.cpp cache
        type. Quantized caches are much smaller, so the runtime KV reserve should
        shrink accordingly (q4_0 ≈ 0.27× f16). Unset/unknown → f16 (1.0)."""
        t = str(cache_type or "").strip().lower().replace("-", "_")
        return {
            "": 1.0, "f16": 1.0, "f32": 2.0, "bf16": 1.0,
            "q8_0": 0.53,
            "q5_1": 0.34, "q5_0": 0.33,
            "q4_1": 0.28, "q4_0": 0.27,
        }.get(t, 1.0)

    def _runtime_reserve_gb(self, cfg: dict, model_key: str, base_gb: float) -> float:
        """Extra VRAM the model needs at RUNTIME beyond its resident weights.

        A model occupies its weights at load time, but inference allocates more:
        the KV cache (autoregressive text/vision, grows with context length),
        attention/temporal activations and the VAE-decode spike (diffusion
        image/video). The eviction estimate and the persisted measurement must
        account for this so we reserve enough headroom and don't OOM mid-run.
        Returned as GB to add on top of `base_gb` (the weight footprint).
        """
        if base_gb <= 0:
            return 0.0
        mtype = self._model_type_of(model_key)
        if mtype in ("text", "vision"):
            # KV cache + activations scale with context length.
            try:
                n_ctx = int(cfg.get("n_ctx") or cfg.get("context_size") or 2048)
            except (TypeError, ValueError):
                n_ctx = 2048
            ctx_factor = min(3.0, max(1.0, n_ctx / 4096.0))
            # The KV-cache part of the reserve scales with the cache quantization:
            # a q4_0 KV cache is ~4x smaller than f16, so reserving an f16-sized KV
            # for a quantized cache wildly over-estimates and forces needless CPU
            # offload at large n_ctx. Scale only the KV term (0.08*ctx_factor); the
            # 0.10 base (activations/compute buffers) is unaffected by KV dtype.
            kv_factor = (self._kv_cache_size_factor(cfg.get("cache_type_k"))
                         + self._kv_cache_size_factor(cfg.get("cache_type_v"))) / 2.0
            return base_gb * (0.10 + 0.08 * ctx_factor * kv_factor)   # ~0.13–0.34×
        if mtype == "video":
            return base_gb * 0.08   # temporal activations + VAE decode (tiling-capped)
        if mtype == "image":
            return base_gb * 0.08
        return base_gb * 0.08

    def _get_model_used_vram_gb(self, model_key: str, resolved_name: str = None) -> float:
        """Return VRAM requirement in GB for a model.

        Priority:
        1. Explicit ``used_vram_gb`` in the model's config entry.
        2. Measured actual VRAM delta from a previous load of this model.
        3. File size of a local weight file, adjusted for quantization/offload.
        4. HuggingFace hub cache size (dense shards or largest GGUF), adjusted.
        Returns 0 when the requirement cannot be determined.
        """
        # ds4 (DeepSeek-V4) models are served by an external ds4-server that manages
        # its own memory (SSD-streamed MoE experts), so the GGUF file size (100GB+)
        # is NOT its VRAM footprint. Reporting that would make coderai try to evict
        # ~128 GB on every request (needless churn) and mis-message CPU/disk offload.
        # Use the measured value if we have one, else a modest fixed reserve.
        try:
            if ds4_should_handle(model_key) or (resolved_name and ds4_should_handle(resolved_name)):
                measured = self._measured_vram_gb.get(model_key)
                if not measured and resolved_name:
                    measured = self._measured_vram_gb.get(resolved_name)
                return float(measured) if measured else 12.0
        except Exception:
            pass
        # Resolve by basename/alias too — a model requested by basename would
        # otherwise miss self.config (keyed by full path), return 0, and skip the
        # eviction that makes room for it (→ OOM loading on top of a resident model).
        cfg = self._config_for_model(model_key)
        if not cfg and resolved_name:
            cfg = self._config_for_model(resolved_name)
        # Unwrap a forwarded `_raw_cfg` so we see the ORIGINAL model entry the
        # same way the loaders do (build_kwargs_from_config only copies a few
        # keys to the top level — component_quantization lives ONLY in _raw_cfg).
        # Without this the per-component quant map is invisible here and the
        # estimate stays at full precision (e.g. 54 GB instead of ~25 GB).
        if isinstance(cfg, dict) and isinstance(cfg.get('_raw_cfg'), dict):
            _merged = dict(cfg)
            for _k, _v in cfg['_raw_cfg'].items():
                _merged.setdefault(_k, _v)
            cfg = _merged

        # --- Read quantization / offload settings from model config ---
        # These come from the coderai model configuration and dramatically
        # change how much VRAM the model actually occupies.  Read them FIRST so
        # they apply to every estimate path, including the explicit value below.
        load_in_4bit = cfg.get("load_in_4bit", False)
        load_in_8bit = cfg.get("load_in_8bit", False)
        offload_strategy = cfg.get("offload_strategy") or "auto"
        max_gpu_percent = cfg.get("max_gpu_percent")  # explicit cap, 0-100

        explicit = cfg.get("used_vram_gb")

        # Effective quantization multiplier (fraction of full-precision weight
        # size that remains in memory). Honours BOTH the global load_in_4/8bit
        # flags AND per-component quantization (component_quantization map), so a
        # diffusion pipeline whose transformer/text_encoder are 4-bit isn't
        # wildly over-estimated at full precision.
        quant_mult = self._effective_quant_multiplier(
            cfg, load_in_4bit, load_in_8bit, model_key, resolved_name)

        # Precision normalization: used_vram_gb / disk-scan baselines are STORAGE
        # sizes at the on-disk dtype (e.g. Wan2.2 ships fp32 → 4 bytes/elem), but
        # the quant divisors are fp16-relative and the model loads at `precision`
        # (bf16/fp16 → 2 bytes). Without this a 4-bit fp32 model is ~2× over-
        # estimated (151 GB → 49 instead of ~24). factor = load_bytes/storage_bytes.
        _storage_bpe = self._storage_bytes_per_elem(model_key, resolved_name, cfg)
        _load_bpe = self._load_bytes_per_elem(cfg)
        prec_factor = (_load_bpe / _storage_bpe) if _storage_bpe > 0 else 1.0

        # GGUF files are ALREADY quantized on disk and llama.cpp loads the baked-in
        # quantization — it ignores load_in_4bit/load_in_8bit entirely. The stored
        # used_vram_gb and file-size baselines already reflect that quantized
        # footprint, so applying the 4/8-bit quant multiplier (or a storage→load
        # precision normalization) on top would 2–3× UNDER-estimate the real
        # resident size and let the loader try to fit a model that doesn't fit.
        _gguf_path = str(cfg.get("path") or resolved_name or model_key or "")
        _is_gguf = (_gguf_path.endswith(".gguf") or "gguf" in _gguf_path.lower()
                    or cfg.get("model_type") == "gguf_models")
        if _is_gguf:
            quant_mult = 1.0
            prec_factor = 1.0
            # n_gpu_layers controls how much of a GGUF actually lands in VRAM.
            # With 0 layers on the GPU the weights live in CPU RAM / are mmap'd
            # from disk, so the GPU only needs compute/KV buffers — don't reserve
            # the whole model (which would force needless eviction of other
            # models on every load attempt). A partial positive count is left
            # conservative since the total layer count isn't known here.
            try:
                _ngl = int(cfg.get("n_gpu_layers")) if cfg.get("n_gpu_layers") is not None else -1
            except (TypeError, ValueError):
                _ngl = -1
            if _ngl == 0:
                quant_mult = 0.0

        def _dbg_est(source: str, value: float) -> float:
            try:
                from codai.api.state import get_global_debug
                if get_global_debug():
                    print(f"  [vram-est][debug] key='{model_key}' source={source} "
                          f"-> {value:.2f} GB | used_vram_gb={explicit} "
                          f"measured_cfg={cfg.get('measured_vram_gb')} "
                          f"component_quantization={cfg.get('component_quantization')} "
                          f"load_in_4bit={load_in_4bit} load_in_8bit={load_in_8bit} "
                          f"quant_mult={quant_mult:.3f} "
                          f"prec(load{_load_bpe:.0f}/disk{_storage_bpe:.0f})={prec_factor:.2f}")
            except Exception:
                pass
            return value

        def _apply_factors(raw_weights_gb: float) -> float:
            """Adjust raw weight size for quantization and GPU-offload settings."""
            # Normalize storage precision → load precision, then apply the
            # (fp16-relative) quantization multiplier.
            gpu_gb = raw_weights_gb * prec_factor * quant_mult

            # When offload is active the model is split across GPU + CPU RAM.
            # Only the GPU portion counts toward free-VRAM requirements.
            if offload_strategy and offload_strategy != "none":
                if max_gpu_percent is not None:
                    # User has set an explicit GPU cap — honour it.
                    gpu_gb = gpu_gb * max(0.0, min(1.0, float(max_gpu_percent) / 100.0))
                else:
                    # Conservative default: assume the model will use at most
                    # 93 % of whatever GPU headroom exists.  Since we don't know
                    # the exact GPU size here, cap the estimate at gpu_gb itself
                    # (no reduction) so eviction is still triggered when needed.
                    # The actual split is decided by transformers/accelerate at
                    # load time.
                    pass  # no reduction — be conservative

            # Add the runtime reserve (KV cache / activations / VAE-decode spike)
            # so the estimate reflects the model's PEAK footprint, not just its
            # resident weights — otherwise eviction frees too little and the run
            # OOMs mid-generation. Type-aware (ctx-scaled for text).
            return gpu_gb + self._runtime_reserve_gb(cfg, model_key, gpu_gb)

        # Resolve a real measurement (this run via `_measured_vram_gb`, or the
        # persisted, system-owned `measured_vram_gb` field from a prior load). It
        # already reflects quant/precision/offload, so it's used WITHOUT factors.
        measured = self._measured_vram_gb.get(model_key)
        if not measured:
            _cm = cfg.get("measured_vram_gb")
            if _cm:
                try:
                    measured = float(_cm)
                except (TypeError, ValueError):
                    measured = None

        # force_vram_update: the real measurement ALWAYS wins — even over a
        # configured used_vram_gb — and is refreshed from reality every run.
        if bool(cfg.get("force_vram_update")) and measured:
            return _dbg_est("measured(force)", float(measured))

        # used_vram_gb set in config is AUTHORITATIVE — use it as-is (with the
        # usual quant/offload factors) and never override it with a measurement.
        # Per user intent: a configured value must remain unchanged even if the
        # resulting estimate is imperfect (unless force_vram_update is set).
        # Config-declared LoRA(s) are added on top — they load into VRAM beyond
        # the base weights. (Not added to forced/measured: forced is the user's
        # final number; a measurement already reflects any load-time adapters.)
        if explicit is not None:
            return _dbg_est("used_vram_gb",
                            _apply_factors(float(explicit)) + self._config_lora_vram_gb(cfg))

        # No configured used_vram_gb → the real measurement is GROUND TRUTH.
        if measured:
            return _dbg_est("measured", float(measured))

        # Build a list of candidate paths/IDs to check
        candidates = []
        if resolved_name:
            candidates.append(resolved_name)
        candidates.append(model_key)
        if ":" in model_key:
            candidates.append(model_key.split(":", 1)[1])
        for field in ("path", "model_path", "model"):
            v = cfg.get(field)
            if v:
                candidates.append(v)

        import os
        local_exts = {'.gguf', '.ggml', '.bin', '.pt', '.safetensors'}

        # --- Pass 1: local file with a known extension ---
        for path in candidates:
            if not path:
                continue
            _, ext = os.path.splitext(path)
            if ext.lower() not in local_exts:
                continue
            try:
                size_bytes = os.path.getsize(path)
                weights_gb = size_bytes / 1e9
                # _apply_factors already adds the runtime reserve (incl. KV cache).
                return _apply_factors(weights_gb) + self._config_lora_vram_gb(cfg)
            except OSError:
                continue

        # --- Pass 2: HuggingFace repo ID → scan local HF hub cache ---
        from codai.models.cache import is_huggingface_model_id
        for candidate in candidates:
            if not candidate or not is_huggingface_model_id(candidate):
                continue
            # Strip any type prefix that survived (e.g. "image:org/repo")
            repo_id = candidate.split(":", 1)[1] if ":" in candidate else candidate
            if not is_huggingface_model_id(repo_id):
                continue
            size_bytes = self._hf_cached_model_size_bytes(repo_id)
            if size_bytes > 0:
                weights_gb = size_bytes / 1e9
                return _apply_factors(weights_gb) + self._config_lora_vram_gb(cfg)

        return 0

    def _is_key_busy(self, key: str) -> bool:
        """True if any instance for this key is currently serving a request.

        Evicting a busy model would move its tensors to CPU mid-forward-pass,
        which triggers a CUDA device-side assert in the in-flight request and
        corrupts the whole context.  Such models must NOT be evicted until idle.
        """
        pool = self.model_pools.get(key)
        if pool is not None:
            with pool._lock:
                return any(r > 0 for r in pool.ref_counts)
        return False

    def _wait_until_idle(self, key: str, timeout: float = 120.0) -> bool:
        """Block until no instance of `key` is busy, or until timeout. Returns idle?"""
        import time as _time
        deadline = _time.time() + timeout
        while self._is_key_busy(key):
            if _time.time() >= deadline:
                return False
            _time.sleep(0.25)
        return True

    def unload_model(self, key: str) -> bool:
        """Fully unload ONE model by key: drop it from the cache + instance pool and
        free its VRAM/host RAM. Used when a config change (e.g. acceleration, which
        is fused at load time) needs the next request to reload the model fresh.

        Waits briefly if the model is mid-request; returns True if a model was
        actually unloaded."""
        if key not in self.models and key not in self.model_pools:
            return False
        if self._is_key_busy(key):
            if not self._wait_until_idle(key):
                print(f"  unload_model: '{key}' still busy — not unloaded")
                return False
        model_obj = self.models.pop(key, None)
        self.models_in_vram.discard(key)
        if self.current_model_key == key:
            self.current_model_key = None
        if self.active_in_vram == key:
            self.active_in_vram = None
        pool = self.model_pools.pop(key, None)
        if pool is not None:
            try:
                pool.cleanup_all()
            except Exception as e:
                print(f"  Warning cleaning pool for '{key}': {e}")
        if model_obj is not None:
            try:
                if hasattr(model_obj, 'cleanup'):
                    model_obj.cleanup()
                elif hasattr(model_obj, 'to'):
                    model_obj.to('cpu')
            except Exception as e:
                print(f"  Warning during unload of '{key}': {e}")
        del model_obj
        for _ in range(3):
            gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:
            pass
        _trim_cpu_ram()
        return True

    def _evict_one(self, key):
        """Fully unload ONE model by key and free its VRAM + host RAM.

        Shared by VRAM- and RAM-driven eviction. Never pulls a model that is still
        mid-request; waits briefly first and skips it if it stays busy.
        """
        # Make absolutely sure nothing is mid-request on this model before we
        # pull its tensors off the GPU.
        if self._is_key_busy(key):
            if not self._wait_until_idle(key):
                print(f"  Skipping eviction of '{key}' — still busy after wait")
                return
        model_obj = self.models.pop(key, None)
        self.models_in_vram.discard(key)
        self._last_used.pop(key, None)
        # Debug-only: ground-truth the object state before cleanup so an
        # orphaned/detached backend (VRAM that won't free) is visible.
        try:
            from codai.api.state import get_global_debug
            if get_global_debug():
                _be = getattr(model_obj, 'backend', '∅')
                _has_model = (getattr(_be, 'model', None) is not None) if _be not in ('∅', None) else False
                _pool = self.model_pools.get(key)
                print(f"  evict-debug '{key}': obj_id={id(model_obj)} "
                      f"backend={'None' if _be is None else ('missing' if _be=='∅' else 'set')} "
                      f"backend.model={'set' if _has_model else 'None'} "
                      f"pool_instances={_pool.count if _pool else 0}")
        except Exception:
            pass
        # Clean every instance in the pool (not just the primary) so extra
        # instances don't leak VRAM, then drop the pool.
        pool = self.model_pools.pop(key, None)
        if pool is not None:
            try:
                pool.cleanup_all()
            except Exception as e:
                print(f"  Warning cleaning pool for '{key}': {e}")
        if model_obj is not None:
            try:
                if hasattr(model_obj, 'cleanup'):
                    model_obj.cleanup()
                elif hasattr(model_obj, 'to'):
                    # Diffusers pipeline: move all components to CPU explicitly
                    # before dropping the reference so VRAM is freed promptly.
                    model_obj.to('cpu')
            except Exception as e:
                print(f"  Warning during eviction of '{key}': {e}")
        del model_obj
        for _ in range(3):
            gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:
            pass
        # Return freed CPU heap (the evicted model's host-side copy / offloaded
        # weights) to the OS, and let the kernel reclaim any swap backing it —
        # otherwise RSS stays high across evict/load cycles and the machine
        # slowly fills RAM + swap.
        _trim_cpu_ram()

    @staticmethod
    def _get_process_ram_gb() -> float:
        """Resident-set size of the server process TREE, in GB (0.0 on failure).

        Offloaded weights and worker subprocesses count against the global cap, so
        sum the root process plus all children (mirrors thermal.read_process_tree_cpu).

        Under the front/engine split the host-RAM cap is SHARED, not split: when the
        front spawned this engine it set CODERAI_FRONT_PID, so the root is the
        *front* — every engine then measures the same fleet-wide total (front + all
        engines + their workers) and enforces the single cap against it. In
        single-process mode the root is just this process, as before."""
        try:
            import os
            import psutil
            root_pid = os.environ.get("CODERAI_FRONT_PID")
            proc = None
            if root_pid:
                try:
                    proc = psutil.Process(int(root_pid))
                except Exception:
                    proc = None
            if proc is None:
                proc = psutil.Process()
            total = proc.memory_info().rss
            for child in proc.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except Exception:
                    pass
            return total / 1e9
        except Exception:
            return 0.0

    @staticmethod
    def _get_own_ram_gb() -> float:
        """RSS of THIS engine's own process tree only (ignores the shared-fleet
        root), in GB. Used for per-engine *leak* detection so unbounded growth is
        attributed to the engine that actually has it — unlike the shared cap, which
        uses the whole fleet (:meth:`_get_process_ram_gb`)."""
        try:
            import psutil
            proc = psutil.Process()
            total = proc.memory_info().rss
            for child in proc.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except Exception:
                    pass
            return total / 1e9
        except Exception:
            return 0.0

    @staticmethod
    def _ram_cap_gb() -> Optional[float]:
        """The configured global RAM ceiling in GB, or None when no cap is set."""
        try:
            from codai.api.state import get_global_args
            ga = get_global_args()
            cap = getattr(ga, 'max_ram_gb', None) if ga else None
            return float(cap) if cap else None
        except Exception:
            return None

    @staticmethod
    def _evict_idle_on_ram_enabled() -> bool:
        """Whether idle models may be unloaded when over the RAM cap (default True)."""
        try:
            from codai.api.state import get_global_args
            ga = get_global_args()
            return bool(getattr(ga, 'evict_idle_on_ram', True)) if ga else True
        except Exception:
            return True

    def _evict_models_for_ram(self, target_gb: float):
        """Unload idle models (LRU first) until process-tree RSS drops to target_gb.

        Same safety rules as VRAM eviction: never pulls a busy model, and the active
        model is only evicted as a last resort once idle."""
        if target_gb <= 0:
            return
        if self._get_process_ram_gb() <= target_gb:
            return
        _before = self._get_process_ram_gb()
        for key in self._lru_order():
            if key == self.active_in_vram:
                continue
            if self._get_process_ram_gb() <= target_gb:
                break
            if self._is_key_busy(key):
                print(f"  '{key}' is busy serving a request — not evicting it (RAM)")
                continue
            print(f"RAM eviction: unloading '{key}' to free host RAM "
                  f"(RSS {self._get_process_ram_gb():.1f} GB > cap target {target_gb:.1f} GB)")
            self._evict_one(key)
        # Last resort: the active model, only if idle.
        if (self._get_process_ram_gb() > target_gb and self.active_in_vram
                and self.active_in_vram in self.models):
            _active = self.active_in_vram
            if not self._is_key_busy(_active) and self._wait_until_idle(_active):
                print(f"RAM eviction: unloading active model '{_active}' to free host RAM")
                self._evict_one(_active)
                self.active_in_vram = None
        _freed = _before - self._get_process_ram_gb()
        if _freed > 0.05:
            print(f"RAM eviction freed {_freed:.1f} GB (RSS now {self._get_process_ram_gb():.1f} GB)")

    def _lru_order(self):
        """Loaded model keys, least-recently-used first (for eviction)."""
        return sorted(self.models.keys(), key=lambda k: self._last_used.get(k, 0.0))

    def _evict_models_for_vram(self, needed_gb: float):
        """Unload loaded models (LRU first) until we have at least needed_gb free VRAM.

        Never evicts a model that is actively serving a request — doing so would
        move its weights off the GPU mid-generation and crash the CUDA context.
        Busy models are waited on (briefly) before eviction.
        """
        if needed_gb <= 0:
            return

        _evict_key = self._evict_one

        def _size_label(key: str) -> str:
            m = self._measured_vram_gb.get(key)
            e = self._get_model_used_vram_gb(key)
            if m:
                return f"measured {m:.2f} GB"
            if e:
                return f"estimated {e:.2f} GB"
            return "size unknown"

        _free_before = self._get_free_vram_gb()

        # First pass: evict idle non-active models in LRU order.
        for key in self._lru_order():
            if key == self.active_in_vram:
                continue
            if self._get_free_vram_gb() >= needed_gb:
                break
            if self._is_key_busy(key):
                print(f"  '{key}' is busy serving a request — not evicting it")
                continue
            print(f"On-request VRAM eviction: unloading '{key}' ({_size_label(key)}) to free VRAM")
            _evict_key(key)

        # Second pass: evict the active model if still not enough AND it is idle.
        if (self._get_free_vram_gb() < needed_gb and self.active_in_vram
                and self.active_in_vram in self.models):
            _active = self.active_in_vram
            if self._is_key_busy(_active):
                print(f"  Active model '{_active}' is busy — waiting for it to go idle "
                      f"before eviction…")
            if self._wait_until_idle(_active):
                print(f"On-request VRAM eviction: unloading active model '{_active}' "
                      f"({_size_label(_active)}) to free VRAM")
                _evict_key(_active)
                self.active_in_vram = None
            else:
                print(f"  Active model '{_active}' still busy after wait — leaving it loaded")

        # Last resort: ask external holders (e.g. the LoRA trainer's cached base
        # model) to release their VRAM. These aren't tracked models, so eviction
        # above can't see them — this is what stops a cached training base from
        # forcing the next image request into CPU/disk offload.
        if self._get_free_vram_gb() < needed_gb and self._external_vram_releasers:
            for _rel in list(self._external_vram_releasers):
                try:
                    freed = _rel(needed_gb)
                    if freed:
                        print(f"On-request VRAM eviction: external releaser freed {freed:.1f} GB")
                except Exception as e:
                    print(f"  Warning in external VRAM releaser: {e}")
                if self._get_free_vram_gb() >= needed_gb:
                    break

        # Report what we actually freed so a failure to free is visible.
        _free_after = self._get_free_vram_gb()
        if _free_after < needed_gb:
            print(f"  ⚠ Eviction freed {_free_after - _free_before:.1f} GB "
                  f"(now {_free_after:.1f} GB free, needed {needed_gb:.1f} GB). "
                  f"Remaining models are busy or VRAM is held elsewhere — the new "
                  f"model will load with CPU/disk offload.")

    def ensure_vram_for(self, model_key: str, resolved_name: str = None,
                        extra_vram_gb: float = 0.0) -> None:
        """Make room in VRAM for a model about to load — ANY type, ANY load mode.

        Universal guarantee: before loading a model that is not already resident,
        if its estimated VRAM need exceeds the free VRAM, evict other loaded
        models (LRU first, then the active one) until there is enough room.
        `_evict_models_for_vram` already frees one-or-more models as needed and
        never touches a model that is busy serving a request.

        `extra_vram_gb` is added to the model's own estimate — used to reserve
        room for per-request additions the base estimate can't know about (e.g.
        dynamically-applied LoRA adapters).
        """
        needed_gb = self._get_model_used_vram_gb(model_key, resolved_name)
        if extra_vram_gb and extra_vram_gb > 0:
            needed_gb += extra_vram_gb
        if needed_gb <= 0:
            return
        free_gb = self._get_free_vram_gb()
        if free_gb < needed_gb:
            _extra = f" (+{extra_vram_gb:.1f} GB LoRA/extra)" if extra_vram_gb else ""
            print(f"Loading '{model_key}': need {needed_gb:.1f} GB VRAM{_extra}, "
                  f"have {free_gb:.1f} GB free — evicting models to make room")
            self._evict_models_for_vram(needed_gb)

    def request_model(self, requested_model: str, model_type: str = None,
                      extra_vram_gb: float = 0.0) -> Dict[str, Any]:
        """
        Central method for API modules to request a model.
        
        Handles per-model load modes:
        
        **load**: Model is pre-loaded at startup and stays in VRAM.
        
        **on-request**: Model is loaded when first needed. Before loading,
        checks free VRAM against the model's used_vram_gb config. If not
        enough VRAM, evicts other loaded models until there is enough, then
        loads the model.
        
        Legacy global modes (ondemand/loadall/loadswap) are still supported
        for backward compatibility.
        
        Args:
            requested_model: The model name/alias from the API request
            model_type: The type of model being requested ("image", "text", "audio", "tts", "vision")
                       Used to resolve empty/None model names to the appropriate default.
        
        Returns:
            Dict with:
                - 'model_key': The key used to store/retrieve the model in self.models
                - 'model_name': The resolved model name/path/HF ID
                - 'model_object': The loaded model object if already loaded, None otherwise
                - 'config': The stored configuration for this model
                - 'already_loaded': True if the model is already loaded and ready in VRAM
        """
        # Fail fast if the CUDA context is already corrupted — loading anything
        # onto a poisoned context just re-triggers the same async assert.
        if self.cuda_context_poisoned:
            return {
                'model_key': None,
                'model_name': requested_model,
                'model_object': None,
                'config': {},
                'already_loaded': False,
                'error': ("CUDA context corrupted by an earlier device-side assert "
                          f"({self.cuda_poison_reason}). Restart coderai to recover."),
            }

        # Thermal protection: before serving a request, wait if the CPU/GPU are
        # too hot so a long sequence of heavy generations can't overheat the
        # machine and trip its power-off protection. No-op when disabled or cool.
        try:
            from codai.models.thermal import wait_until_safe
            wait_until_safe(context=f"{model_type or 'model'}:{requested_model}")
        except Exception as _therr:
            # Never let thermal monitoring failures block serving.
            print(f"[thermal] monitor unavailable ({_therr}) — serving without wait")

        from codai.api.state import get_load_mode
        mode = get_load_mode()

        # Step 1: Resolve the model name from aliases
        resolved_name = None
        model_key = None
        
        # If no model specified, use the default for the given type
        if not requested_model or requested_model == model_type:
            if model_type == "image":
                resolved_name = self.image_models[0] if self.image_models else None
            elif model_type == "audio":
                resolved_name = self.audio_models[0] if self.audio_models else None
            elif model_type == "tts":
                resolved_name = self.tts_model
            elif model_type == "vision":
                resolved_name = self.vision_models[0] if self.vision_models else None
            elif model_type == "video":
                resolved_name = self.video_models[0] if self.video_models else None
            elif model_type == "audio_gen":
                resolved_name = self.audio_gen_models[0] if self.audio_gen_models else None
            elif model_type == "embedding":
                resolved_name = self.embedding_models[0] if self.embedding_models else None
            elif model_type == "spatial":
                resolved_name = self.spatial_models[0] if self.spatial_models else None
            else:
                resolved_name = self.default_model
        else:
            # Resolve custom aliases
            if requested_model in self.model_aliases:
                requested_model = self.model_aliases[requested_model]

            # Handle "default" alias
            if requested_model == "default":
                resolved_name = self.default_model
            # Handle type-specific aliases
            elif requested_model == "image":
                resolved_name = self.image_models[0] if self.image_models else None
            elif requested_model == "audio":
                resolved_name = self.audio_models[0] if self.audio_models else None
            elif requested_model == "tts":
                resolved_name = self.tts_model
            elif requested_model == "vision":
                resolved_name = self.vision_models[0] if self.vision_models else None
            elif requested_model == "video":
                resolved_name = self.video_models[0] if self.video_models else None
            elif requested_model == "audio_gen":
                resolved_name = self.audio_gen_models[0] if self.audio_gen_models else None
            elif requested_model == "embedding":
                resolved_name = self.embedding_models[0] if self.embedding_models else None
            elif requested_model == "spatial":
                resolved_name = self.spatial_models[0] if self.spatial_models else None
            # Handle prefixed models (e.g., "image:model_name")
            elif requested_model.startswith("image:"):
                resolved_name = requested_model[6:]
            elif requested_model.startswith("audio:"):
                resolved_name = requested_model[6:]
            elif requested_model.startswith("tts:"):
                resolved_name = requested_model[4:]
            elif requested_model.startswith("vision:"):
                resolved_name = requested_model[7:]
            elif requested_model.startswith("video:"):
                resolved_name = requested_model[6:]
            elif requested_model.startswith("audio_gen:"):
                resolved_name = requested_model[10:]
            elif requested_model.startswith("embedding:"):
                resolved_name = requested_model[10:]
            elif requested_model.startswith("spatial:"):
                resolved_name = requested_model[8:]
            else:
                resolved_name = requested_model
        
        if not resolved_name:
            return {
                'model_key': None,
                'model_name': None,
                'model_object': None,
                'config': {},
                'already_loaded': False,
            }
        
        # Step 1b: Validate that the resolved model is an allowed/registered model.
        # This prevents API callers from requesting arbitrary models that were not
        # specified on the command line (or registered as aliases).
        if not self.is_allowed_model(resolved_name, model_type):
            # Check if the model exists but is registered under a different type
            registered_type = self.get_registered_model_type(resolved_name)
            if registered_type is not None and registered_type != model_type:
                endpoint_hint = {
                    "image": "POST /v1/images/generations",
                    "audio": "POST /v1/audio/transcriptions",
                    "tts": "POST /v1/audio/speech",
                    "video": "POST /v1/videos/generations",
                }.get(registered_type, f"the {registered_type} endpoint")
                print(f"Model type mismatch: '{resolved_name}' is a {registered_type} model, "
                      f"requested via {model_type} endpoint")
                return {
                    'model_key': None,
                    'model_name': None,
                    'model_object': None,
                    'config': {},
                    'already_loaded': False,
                    'error': (f"Model '{resolved_name}' is a {registered_type} model and cannot be used "
                              f"for {model_type} generation. Use {endpoint_hint} instead."),
                }
            allowed_ids = sorted(self.get_all_allowed_identifiers())
            print(f"Model validation failed: '{resolved_name}' is not an allowed model. "
                  f"Allowed models: {allowed_ids}")
            return {
                'model_key': None,
                'model_name': None,
                'model_object': None,
                'config': {},
                'already_loaded': False,
                'error': f"Model '{resolved_name}' is not available. Use one of: {', '.join(allowed_ids)}",
            }
        
        # Step 1c: Resolve short-name to canonical registered name (e.g. "Foo" → "org/Foo")
        _type_registry = {
            "image": self.image_models,
            "audio": self.audio_models,
            "tts": [self.tts_model] if self.tts_model else [],
            "vision": self.vision_models,
            "video": self.video_models,
            "audio_gen": self.audio_gen_models,
            "embedding": self.embedding_models,
            "spatial": self.spatial_models,
        }
        _candidates = _type_registry.get(model_type, []) if model_type else []
        if resolved_name not in _candidates:
            for _m in _candidates:
                if "/" in _m and _m.split("/")[-1] == resolved_name:
                    resolved_name = _m
                    break

        # Step 2: Build the model key (prefixed with type)
        if model_type and model_type != "text":
            model_key = f"{model_type}:{resolved_name}"
        else:
            model_key = resolved_name
        
        # Step 3: Check if already loaded in self.models
        existing_model = self.models.get(model_key)

        # Fuzzy fallback: if not found by exact key, check for org-prefix
        # variations (e.g. "image:Model" vs "image:org/Model").
        if existing_model is None:
            type_prefix = f"{model_type}:" if model_type and model_type != "text" else ""
            short = resolved_name.split("/")[-1]
            for k, v in self.models.items():
                k_name = k[len(type_prefix):] if type_prefix and k.startswith(type_prefix) else k
                if k_name == resolved_name or k_name.split("/")[-1] == short:
                    model_key = k
                    resolved_name = k_name
                    existing_model = v
                    break

        # =====================================================================
        # PER-MODEL LOAD MODE: Check per-model config first.
        # Per-model "load" = pre-loaded (treat as loadall for this model).
        # Per-model "on-request" = load when needed with VRAM management.
        # =====================================================================
        per_model_cfg = self._config_for_model_key(model_key)
        per_model_load_mode = per_model_cfg.get("load_mode")  # "load" | "on-request" | None

        if per_model_load_mode == "on-request":
            if existing_model is not None:
                # Already loaded — just return it
                self.current_model_key = model_key
                self.active_in_vram = model_key
                self.models_in_vram.add(model_key)
                return {
                    'model_key': model_key,
                    'model_name': resolved_name,
                    'model_object': existing_model,
                    'config': per_model_cfg,
                    'already_loaded': True,
                }
            # Not loaded — make room in VRAM (evict other models as needed).
            self.ensure_vram_for(model_key, resolved_name, extra_vram_gb)
            return {
                'model_key': model_key,
                'model_name': resolved_name,
                'model_object': None,
                'config': per_model_cfg,
                'already_loaded': False,
            }

        if per_model_load_mode == "load":
            # Pre-loaded model — just return it (or signal caller to load it)
            if existing_model is not None:
                self.current_model_key = model_key
                self.active_in_vram = model_key
                return {
                    'model_key': model_key,
                    'model_name': resolved_name,
                    'model_object': existing_model,
                    'config': per_model_cfg,
                    'already_loaded': True,
                }
            # A "load" model that is NOT currently resident was evicted to make
            # room for an on-request model (e.g. the text model bumped for a video
            # model).  Make room again so it returns to VRAM instead of CPU.
            self.ensure_vram_for(model_key, resolved_name, extra_vram_gb)
            return {
                'model_key': model_key,
                'model_name': resolved_name,
                'model_object': None,
                'config': per_model_cfg,
                'already_loaded': False,
            }

        # =====================================================================
        # LOADALL MODE: All models should be pre-loaded. Just return it.
        # =====================================================================
        if mode == "loadall":
            if existing_model is not None:
                self.current_model_key = model_key
                self.active_in_vram = model_key
                return {
                    'model_key': model_key,
                    'model_name': resolved_name,
                    'model_object': existing_model,
                    'config': self.config.get(model_key, {}),
                    'already_loaded': True,
                }
            # Model not loaded yet in loadall mode - caller needs to load it
            # (this happens for models not pre-loaded at startup, e.g., image models)
            print(f"Loadall mode: Model '{model_key}' not pre-loaded, will load now")
            self.ensure_vram_for(model_key, resolved_name, extra_vram_gb)
            return {
                'model_key': model_key,
                'model_name': resolved_name,
                'model_object': None,
                'config': self.config.get(model_key, {}),
                'already_loaded': False,
            }
        
        # =====================================================================
        # LOADSWAP MODE: Keep all models in memory. Swap active model between
        # VRAM and CPU RAM. Only the active model should be in VRAM.
        # =====================================================================
        if mode == "loadswap":
            if existing_model is not None:
                # Model is loaded (either in VRAM or CPU RAM)
                if self.active_in_vram == model_key:
                    # Already the active model in VRAM
                    self.current_model_key = model_key
                    return {
                        'model_key': model_key,
                        'model_name': resolved_name,
                        'model_object': existing_model,
                        'config': self.config.get(model_key, {}),
                        'already_loaded': True,
                    }
                else:
                    # Model is in CPU RAM - need to swap
                    # First, move current VRAM model to CPU
                    if self.active_in_vram and self.active_in_vram in self.models:
                        print(f"Loadswap: Moving '{self.active_in_vram}' from VRAM to CPU RAM")
                        self._move_model_to_cpu(self.active_in_vram)
                    
                    # Also check the legacy model_manager singleton
                    from codai.models.manager import model_manager as _legacy_mm
                    if _legacy_mm.backend is not None and self.active_in_vram is None:
                        print(f"Loadswap: Moving legacy model_manager model to CPU")
                        self._move_model_to_cpu_legacy(_legacy_mm)
                    
                    # Now move the requested model to VRAM
                    print(f"Loadswap: Moving '{model_key}' from CPU RAM to VRAM")
                    self._move_model_to_vram(model_key)
                    self.active_in_vram = model_key
                    self.current_model_key = model_key
                    
                    return {
                        'model_key': model_key,
                        'model_name': resolved_name,
                        'model_object': existing_model,
                        'config': self.config.get(model_key, {}),
                        'already_loaded': True,
                    }
            else:
                # Model not loaded at all - move current VRAM model to CPU first
                if self.active_in_vram and self.active_in_vram in self.models:
                    print(f"Loadswap: Moving '{self.active_in_vram}' from VRAM to CPU RAM")
                    self._move_model_to_cpu(self.active_in_vram)
                
                # Also check the legacy model_manager singleton
                from codai.models.manager import model_manager as _legacy_mm
                if _legacy_mm.backend is not None and self.active_in_vram is None:
                    print(f"Loadswap: Moving legacy model_manager model to CPU")
                    self._move_model_to_cpu_legacy(_legacy_mm)
                
                # Caller needs to load the model fresh (into VRAM)
                self.active_in_vram = model_key  # Will be set once loaded
                return {
                    'model_key': model_key,
                    'model_name': resolved_name,
                    'model_object': None,
                    'config': self.config.get(model_key, {}),
                    'already_loaded': False,
                }
        
        # =====================================================================
        # ONDEMAND MODE (default): One model in memory at a time — but if a model
        # has max_instances > 1 and all instances are busy, load another copy.
        # =====================================================================
        if existing_model is not None:
            pool = self.model_pools.get(model_key)
            if pool is None:
                # Pool wasn't created yet (model registered outside add_model); adopt it now.
                pool = ModelInstancePool(max_instances=self._get_max_instances(model_key))
                pool.add(existing_model)
                self.model_pools[model_key] = pool

            if pool.needs_new_instance():
                needed_gb = self._get_model_used_vram_gb(model_key, resolved_name)
                free_gb = self._get_free_vram_gb()
                if needed_gb > 0 and free_gb >= needed_gb:
                    print(f"Ondemand: all {pool.count} instance(s) of '{model_key}' busy — "
                          f"loading instance {pool.count + 1}/{pool.max_instances} "
                          f"(need {needed_gb:.1f} GB, have {free_gb:.1f} GB free)")
                    self._pending_new_instance.add(model_key)
                    return {
                        'model_key': model_key,
                        'model_name': resolved_name,
                        'model_object': None,
                        'config': self.config.get(model_key, {}),
                        'already_loaded': False,
                    }

            self.current_model_key = model_key
            self.active_in_vram = model_key
            return {
                'model_key': model_key,
                'model_name': resolved_name,
                'model_object': existing_model,
                'config': self.config.get(model_key, {}),
                'already_loaded': True,
            }
        
        # Model not loaded - need to unload whatever is currently loaded
        # Check if there's anything to unload
        has_models_in_multi = len(self.models) > 0
        
        # Also check the legacy model_manager singleton
        from codai.models.manager import model_manager as _legacy_mm
        has_legacy_model = _legacy_mm.backend is not None
        
        if has_models_in_multi or has_legacy_model:
            loaded_canonical = self.get_currently_loaded_model_name()
            if not loaded_canonical and has_legacy_model:
                loaded_canonical = "legacy_model_manager"
            
            if loaded_canonical and loaded_canonical != model_key:
                needed_gb = self._get_model_used_vram_gb(model_key, resolved_name)
                free_gb = self._get_free_vram_gb()

                # Require headroom beyond raw weight size for activation buffers
                # and generation scratch (30% of model size + 1 GB base).
                headroom_gb = max(1.0, needed_gb * 0.30)
                # ds4 (DeepSeek-V4) is served by an external ds4-server that streams
                # MoE experts and wants the WHOLE GPU for its expert cache — coderai's
                # modest VRAM estimate for it is not its real footprint. Co-residing
                # another model starves that cache (layer-0 "ffn batch encode failed").
                # So a ds4 model claims VRAM exclusively: evict everything else.
                _new_is_ds4 = (ds4_should_handle(model_key)
                               or (resolved_name and ds4_should_handle(resolved_name)))
                if _new_is_ds4:
                    print(f"Ondemand mode - ds4 model '{model_key}' needs exclusive VRAM "
                          f"— unloading all other models so ds4-server gets the full GPU")
                    self.unload_all_models()
                elif needed_gb > 0 and free_gb >= needed_gb + headroom_gb:
                    print(f"Ondemand mode - keeping '{loaded_canonical}' in VRAM alongside new model "
                          f"(need {needed_gb:.1f} GB + {headroom_gb:.1f} GB headroom, have {free_gb:.1f} GB free)")
                else:
                    print(f"Ondemand mode - model switch detected (requested '{model_key}', "
                          f"loaded '{loaded_canonical}'):")
                    # Evict one-or-more models (LRU first) until there is enough
                    # room — preserves coexistence of models that still fit rather
                    # than nuking everything.
                    self._evict_models_for_vram(needed_gb + headroom_gb)
                    # If still short, fall back to cleaning the legacy singleton.
                    if has_legacy_model and self._get_free_vram_gb() < needed_gb:
                        try:
                            print(f"  -> Cleaning up legacy model_manager…")
                            _legacy_mm.cleanup()
                        except Exception as e:
                            print(f"  Warning: Error cleaning up legacy model_manager: {e}")
        
        # Global RAM cap: before loading the new model, if host RSS is near the cap
        # and idle-eviction is enabled, unload idle LRU models to make real RAM room.
        # Whatever still doesn't fit is handled by the clamped CPU budget in
        # hf_loading (overflow spills to the disk offload folder).
        _cap = self._ram_cap_gb()
        if _cap and self._evict_idle_on_ram_enabled():
            _rss = self._get_process_ram_gb()
            if _rss > _cap * 0.9:
                print(f"Global RAM cap {_cap:.1f} GB — RSS {_rss:.1f} GB near limit; "
                      f"evicting idle models before load of '{model_key}'")
                self._evict_models_for_ram(_cap * 0.85)

        # Return info for the caller to load the model
        return {
            'model_key': model_key,
            'model_name': resolved_name,
            'model_object': None,
            'config': self.config.get(model_key, {}),
            'already_loaded': False,
        }
    
    def _move_model_to_cpu_legacy(self, legacy_mm):
        """Move the legacy model_manager's model to CPU (for loadswap mode)."""
        try:
            import torch
            if legacy_mm.backend is not None:
                if hasattr(legacy_mm.backend, 'model') and legacy_mm.backend.model is not None:
                    if hasattr(legacy_mm.backend.model, 'to'):
                        try:
                            legacy_mm.backend.model.to('cpu')
                            print(f"  Moved legacy model_manager model to CPU")
                        except Exception as e:
                            print(f"  Warning: Could not move legacy model to CPU: {e}")
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print(f"  Warning during legacy model CPU offload: {e}")
    
    def unload_all_models(self):
        """
        Fully unload ALL models from VRAM. Used in ondemand mode when switching
        between different model types (e.g., text -> image or image -> text).
        
        This handles all model types:
        - ModelManager instances (have cleanup() method)
        - Diffusers pipelines (need to be moved to CPU and deleted)
        - stable-diffusion-cpp StableDiffusion instances
        - Any other model objects
        """
        print("=== FULL VRAM CLEANUP: Unloading all models ===")
        
        for key in list(self.models.keys()):
            model_obj = self.models.get(key)
            if model_obj is None:
                continue
            
            print(f"Unloading '{key}' from VRAM...")
            try:
                # Method 1: ModelManager with cleanup()
                if hasattr(model_obj, 'cleanup') and callable(getattr(model_obj, 'cleanup')):
                    model_obj.cleanup()
                # Method 2: Diffusers pipeline (has 'to' method to move to CPU)
                elif hasattr(model_obj, 'to') and callable(getattr(model_obj, 'to')):
                    try:
                        model_obj.to('cpu')
                    except:
                        pass
                    del model_obj
                # Method 3: Object with 'model' attribute (e.g., wrapper)
                elif hasattr(model_obj, 'model') and model_obj.model is not None:
                    if hasattr(model_obj.model, 'cleanup'):
                        model_obj.model.cleanup()
                    elif hasattr(model_obj.model, 'to'):
                        try:
                            model_obj.model.to('cpu')
                        except:
                            pass
                    del model_obj
                # Method 4: Just delete it
                else:
                    del model_obj
            except Exception as e:
                print(f"Warning during cleanup of '{key}': {e}")
            
            # Remove from dict
            if key in self.models:
                del self.models[key]
        
        # Reset tracking state
        self.current_model_key = None
        self.active_in_vram = None
        self.models_in_vram = set()
        self.model_pools.clear()
        self._pending_new_instance.clear()
        
        # Force garbage collection
        for _ in range(3):
            gc.collect()
        
        # Clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                print("CUDA cache cleared")
        except:
            pass
        
        # Small delay to let GPU memory settle
        time.sleep(1)
        print("=== FULL VRAM CLEANUP: Complete ===")
    
    def _get_max_instances(self, model_key: str) -> int:
        """Per-model max_instances, falling back to the global default."""
        per_model = self.config.get(model_key, {}).get("max_instances")
        if per_model is not None:
            return int(per_model)
        return self._global_max_instances

    def _get_least_busy_instance(self, key: str):
        """Return the least-busy instance for key without incrementing its ref-count."""
        pool = self.model_pools.get(key)
        if pool and pool.count > 0:
            obj = pool.least_busy_instance()
            if obj is not None:
                self.current_model_key = key
                return obj
        obj = self.models.get(key)
        if obj is not None:
            self.current_model_key = key
        return obj

    def add_model(self, key: str, manager):
        """Add a model (ModelManager, diffusers pipeline, sd.cpp model, etc.) for a specific key."""
        pool = self.model_pools.get(key)
        if pool is None:
            pool = ModelInstancePool(max_instances=self._get_max_instances(key))
            self.model_pools[key] = pool
        pool.add(manager)
        self.models[key] = pool.primary  # backward compat: always the first instance
        self.active_in_vram = key
        self.models_in_vram.add(key)

    def acquire_model_instance(self, model_key: str, session_key=None):
        """Acquire an instance, incrementing its ref-count.

        ``session_key`` (optional) routes successive requests of one conversation
        back to the instance holding its warm KV prefix; without it the least-busy
        instance is chosen. Returns (instance_idx, model_obj) or None if no
        instance is loaded. Callers MUST call release_model_instance() when done.
        """
        pool = self.model_pools.get(model_key)
        if pool and pool.count > 0:
            return pool.acquire(session_key=session_key)
        obj = self.models.get(model_key)
        if obj is not None:
            return 0, obj
        return None

    def release_model_instance(self, model_key: str, instance_idx: int) -> None:
        """Release a previously acquired instance, decrementing its ref-count."""
        pool = self.model_pools.get(model_key)
        if pool:
            pool.release(instance_idx)
    
    def get_model(self, key: str) -> Optional[ModelManager]:
        """Get a model manager by key."""
        return self.models.get(key)
    
    def get_current_model(self) -> Optional[ModelManager]:
        """Get the currently active model."""
        if self.current_model_key:
            return self.models.get(self.current_model_key)
        if self.default_model:
            return self.models.get(self.default_model)
        return None
    
    def find_capable_model(self, *capabilities: str) -> Optional[str]:
        """Return the id of the first configured model whose capabilities include
        any of the requested ones, in the priority order given. Used to auto-pick
        a default for endpoints that have no per-type default list (e.g. video
        upscaling). Returns None when nothing configured matches."""
        try:
            infos = self.list_models()
        except Exception:
            return None
        for want in capabilities:
            for info in infos:
                caps = getattr(info, "capabilities", None) or []
                if want in caps:
                    return info.id
        return None

    def list_models(self, all_engines: bool = False) -> List[ModelInfo]:
        """List all available models (configured + runtime aliases) with type/capability metadata.

        all_engines=True bypasses the per-engine assignment filter so the admin
        config UI sees every configured model — otherwise, on a multi-engine node,
        models pinned to a secondary engine (e.g. whisper-servers on `radeon`) are
        hidden from the primary's list, which serves /admin/api/models."""
        from codai.models.capabilities import detect_model_capabilities, ModelCapabilities

        models = []
        seen_ids: set = set()

        CAT_TYPE = {
            "text_models": "text",
            "gguf_models": "text",
            "vision_models": "vision",
            "image_models": "image",
            "audio_models": "audio",
            "tts_models": "tts",
            "video_models": "video",
            "audio_gen_models": "audio_gen",
            "embedding_models": "embedding",
            "spatial_models": "spatial",
        }

        # Minimum capability guaranteed by a model's config category.
        # Applied when heuristic name detection doesn't recognise the model ID.
        TYPE_MIN_CAP = {
            "image":     "image_generation",
            "video":     "video_generation",
            "audio":     "speech_to_text",
            "tts":       "text_to_speech",
            "audio_gen": "audio_generation",
            "embedding": "embeddings",
            "spatial":   "depth_estimation",
        }

        def _add(model_id: str, model_type: str = None, meta: Dict[str, Any] = None):
            if model_id in seen_ids:
                return
            seen_ids.add(model_id)
            meta = meta or {}
            # Use saved capabilities from config when available; fall back to heuristic.
            saved_caps = meta.get("capabilities") or []
            if saved_caps:
                caps = ModelCapabilities()
                for f in saved_caps:
                    if hasattr(caps, f):
                        setattr(caps, f, True)
            else:
                caps = detect_model_capabilities(model_id)
            # Apply minimum capability for the config-declared category only when
            # detection found nothing at all (unrecognized model name). If any caps
            # were detected (e.g. image_upscaling for latent-upscaler models), trust
            # the detection result — TYPE_MIN_CAP must not override it.
            if model_type and model_type in TYPE_MIN_CAP and not caps.to_list():
                setattr(caps, TYPE_MIN_CAP[model_type], True)
            resolved_type = model_type or (caps.to_list()[0].split("_")[0] if caps.to_list() else "text")
            models.append(ModelInfo(
                id=model_id,
                type=resolved_type,
                capabilities=caps.to_list(),
                backend=meta.get("backend"),
                model_path=meta.get("model_path"),
                server_path=meta.get("server_path"),
                alias=meta.get("alias"),
                port=meta.get("port"),
                gpu_device=meta.get("gpu_device"),
                load_mode=meta.get("load_mode"),
            ))

        # --- Models from config (the authoritative source) ---
        try:
            from codai.admin.routes import config_manager
            if config_manager is not None:
                md = config_manager.models_data
                for cat in ("text_models", "vision_models", "image_models",
                            "audio_models", "tts_models", "gguf_models",
                            "video_models", "audio_gen_models", "embedding_models",
                            "spatial_models"):
                    mtype = CAT_TYPE.get(cat, "text")
                    for m in md.get(cat, []):
                        # Only list models the front assigned to THIS engine (so a
                        # per-engine /v1/models reflects what it actually serves).
                        # The admin config view passes all_engines=True to see them all.
                        if not all_engines and not self._entry_assigned(m):
                            continue
                        if isinstance(m, str):
                            mid = m
                        else:
                            raw = m.get("path") or m.get("id") or ""
                            alias = m.get("alias") or ""
                            # Auto-derive a clean alias for GGUF files that have no
                            # explicit alias so the full filesystem path isn't exposed.
                            if not alias and raw.lower().endswith(".gguf"):
                                stem = raw.split("/")[-1][:-5]  # filename without .gguf
                                alias = stem
                            # whisper-server aliases are round-robin group keys shared across
                            # multiple instances — don't expose the alias as a separate model
                            if m.get("backend") == "whisper-server":
                                mid = raw
                            else:
                                mid = alias or raw
                                if raw and raw != mid:
                                    _add(raw, mtype, m)
                                    short = raw.split("/")[-1] if "/" in raw else raw
                                    if short != raw:
                                        _add(short, mtype, m)
                        if mid:
                            _add(mid, mtype, m if isinstance(m, dict) else None)
                            short = mid.split("/")[-1] if "/" in mid else mid
                            if short != mid:
                                _add(short, mtype, m if isinstance(m, dict) else None)
        except Exception:
            pass

        # --- Fallback: runtime default_model ---
        if not models and self.default_model:
            model_id = self.default_model
            if not (model_id.startswith("http://") or model_id.startswith("https://")):
                short_name = model_id.split("/")[-1] if "/" in model_id else model_id
                if short_name != model_id:
                    _add(short_name, "text")
                _add(model_id, "text")
                _add("default", "text")

        # --- Runtime-registered non-text models ---
        if self.audio_models:
            _add("audio", "audio")
            for m in self.audio_models:
                _add(f"audio:{m}", "audio")

        if self.tts_model:
            _add("tts", "tts")
            _add(f"tts:{self.tts_model}", "tts")

        if self.image_models:
            _add("image", "image")
            for m in self.image_models:
                _add(f"image:{m}", "image")

        if self.vision_models:
            _add("vision", "vision")
            for m in self.vision_models:
                _add(f"vision:{m}", "vision")

        if self.video_models:
            _add("video", "video")
            for m in self.video_models:
                _add(f"video:{m}", "video")

        if self.audio_gen_models:
            _add("audio_gen", "audio_gen")
            for m in self.audio_gen_models:
                _add(f"audio_gen:{m}", "audio_gen")

        if self.embedding_models:
            _add("embedding", "embedding")
            for m in self.embedding_models:
                _add(f"embedding:{m}", "embedding")

        if self.spatial_models:
            _add("spatial", "spatial")
            for m in self.spatial_models:
                _add(f"spatial:{m}", "spatial")

        # --- Custom aliases ---
        for alias in self.model_aliases:
            _add(alias)

        # --- DeepSeek V4 via ds4 (no models.json entry; surfaced when enabled) ---
        ds4_cfg = get_active_ds4_config()
        if ds4_cfg is not None and getattr(ds4_cfg, "enabled", False):
            mid = getattr(ds4_cfg, "model_id", "deepseek-v4") or "deepseek-v4"
            _add(mid, "text", {"backend": "ds4"})

        return models


# =============================================================================
# Singleton Instances
# =============================================================================

# Global singleton instances for convenience
model_manager = ModelManager()
multi_model_manager = MultiModelManager()
