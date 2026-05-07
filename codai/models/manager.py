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

from typing import Optional, Dict, Any, List
import os
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
                      response_format: Optional[Dict] = None):
        """Generate chat completion non-streaming."""
        if self.backend is None:
            raise RuntimeError("No model loaded")
        # Use generate_chat if available (Vulkan backend), otherwise format and use generate
        if hasattr(self.backend, 'generate_chat'):
            return self.backend.generate_chat(messages, max_tokens, temperature, top_p, stop, tools, response_format)
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
                                    response_format: Optional[Dict] = None):
        """Generate chat completion streaming."""
        if self.backend is None:
            raise RuntimeError("No model loaded")
        # Use generate_chat_stream if available (Vulkan backend), otherwise format and use generate_stream
        if hasattr(self.backend, 'generate_chat_stream'):
            async for chunk in self.backend.generate_chat_stream(messages, max_tokens, temperature, top_p, stop, tools, response_format):
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
    
    def transcribe(self, audio_data: bytes, language: str = None, prompt: str = None):
        """Send transcription request to whisper-server."""
        if not self.is_running():
            return {"error": "whisper-server not running"}
        
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

    def acquire(self):
        """Return (idx, instance) of the least-busy instance, incrementing its ref-count."""
        with self._lock:
            if not self.instances:
                return None
            idx = min(range(len(self.instances)), key=lambda i: self.ref_counts[i])
            self.ref_counts[idx] += 1
            return idx, self.instances[idx]

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
        self.config: Dict[str, Dict] = {}  # Store model configurations
        self.tool_parser = ModelParserAdapter()
        self.current_model_key: Optional[str] = None
        self.load_mode: str = "ondemand"
        self.active_in_vram: Optional[str] = None  # most-recently-used model key
        self.models_in_vram: set = set()  # all models currently in VRAM
        self.model_aliases: Dict[str, str] = {}
        self.whisper_server: Optional[WhisperServerManager] = None  # legacy single-instance compat
        self.whisper_servers: Dict[str, WhisperServerManager] = {}  # id -> manager
        self.whisper_aliases: Dict[str, List[str]] = {}  # alias -> [model_id, ...]
        self._whisper_alias_counters: Dict[str, int] = {}  # alias -> next round-robin index
        self.model_backend_types: Dict[str, str] = {}
        self.tool_breaker = FuzzyToolBreaker(threshold=3)  # Circuit breaker for repetitive tool calls
        self._load_lock = threading.Lock()  # Prevents duplicate on-demand model loads
        self.model_pools: Dict[str, ModelInstancePool] = {}  # per-key instance pools
        self._pending_new_instance: set = set()  # keys awaiting a second+ instance load
        self._global_max_instances: int = 1  # set from config at startup
    
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

        with self._load_lock:
            # Re-check inside the lock to avoid duplicate loads from concurrent requests
            if self.default_model in self.models and self.default_model not in self._pending_new_instance:
                return self._get_least_busy_instance(self.default_model)
            self._pending_new_instance.discard(self.default_model)

            config = self.config.get(self.default_model, {})
            backend_type = self.model_backend_types.get(self.default_model, "auto")

            try:
                from codai.api.state import get_global_args
                global_args = get_global_args()
            except Exception:
                global_args = None

            model_manager = ModelManager()
            try:
                kwargs = {}
                if 'ctx' in config:
                    kwargs['ctx'] = config['ctx']
                if global_args:
                    if hasattr(global_args, 'n_gpu_layers'):
                        kwargs['n_gpu_layers'] = global_args.n_gpu_layers
                    if hasattr(global_args, 'offload_dir'):
                        kwargs['offload_dir'] = global_args.offload_dir
                    if hasattr(global_args, 'ram'):
                        kwargs['manual_ram_gb'] = global_args.ram
                    if hasattr(global_args, 'flash_attn'):
                        kwargs['flash_attn'] = global_args.flash_attn
                    if hasattr(global_args, 'no_ram'):
                        kwargs['no_ram'] = global_args.no_ram
                    if hasattr(global_args, 'offload_strategy'):
                        kwargs['offload_strategy'] = global_args.offload_strategy
                    if hasattr(global_args, 'load_in_4bit'):
                        kwargs['load_in_4bit'] = global_args.load_in_4bit
                    if hasattr(global_args, 'load_in_8bit'):
                        kwargs['load_in_8bit'] = global_args.load_in_8bit
                    if hasattr(global_args, 'max_gpu_percent'):
                        kwargs['max_gpu_percent'] = global_args.max_gpu_percent

                print(f"Loading default model on demand: {self.default_model}")
                model_manager.load_model(self.default_model, backend_type=backend_type, **kwargs)
                self.add_model(self.default_model, model_manager)
                self.current_model_key = self.default_model
                print(f"Model loaded successfully: {self.default_model}")
                return model_manager
            except Exception as e:
                print(f"Error loading model {self.default_model}: {e}")
                return None
    
    def _load_model_by_name(self, model_name: str):
        """Load a model by name on demand (thread-safe)."""
        if model_name in self.models and model_name not in self._pending_new_instance:
            return self._get_least_busy_instance(model_name)

        with self._load_lock:
            # Re-check inside lock to prevent duplicate loads, but honour pending new instance.
            if model_name in self.models and model_name not in self._pending_new_instance:
                return self._get_least_busy_instance(model_name)
            self._pending_new_instance.discard(model_name)

            config = self.config.get(model_name, {})
            backend_type = self.model_backend_types.get(model_name, "auto")

            try:
                from codai.api.state import get_global_args
                global_args = get_global_args()
            except Exception:
                global_args = None

            model_manager = ModelManager()
            try:
                kwargs = {}
                if 'ctx' in config:
                    kwargs['ctx'] = config['ctx']
                if global_args:
                    if hasattr(global_args, 'n_gpu_layers'):
                        kwargs['n_gpu_layers'] = global_args.n_gpu_layers
                    if hasattr(global_args, 'offload_dir'):
                        kwargs['offload_dir'] = global_args.offload_dir
                    if hasattr(global_args, 'ram'):
                        kwargs['manual_ram_gb'] = global_args.ram
                    if hasattr(global_args, 'flash_attn'):
                        kwargs['flash_attn'] = global_args.flash_attn
                    if hasattr(global_args, 'no_ram'):
                        kwargs['no_ram'] = global_args.no_ram
                    if hasattr(global_args, 'offload_strategy'):
                        kwargs['offload_strategy'] = global_args.offload_strategy
                    if hasattr(global_args, 'load_in_4bit'):
                        kwargs['load_in_4bit'] = global_args.load_in_4bit
                    if hasattr(global_args, 'load_in_8bit'):
                        kwargs['load_in_8bit'] = global_args.load_in_8bit
                    if hasattr(global_args, 'max_gpu_percent'):
                        kwargs['max_gpu_percent'] = global_args.max_gpu_percent

                pool = self.model_pools.get(model_name)
                inst_num = pool.count + 1 if pool else 1
                print(f"Loading model on demand: {model_name}"
                      + (f" (instance {inst_num})" if inst_num > 1 else ""))
                model_manager.load_model(model_name, backend_type=backend_type, **kwargs)
                self.add_model(model_name, model_manager)
                self.current_model_key = model_name
                print(f"Model loaded successfully: {model_name}"
                      + (f" (instance {inst_num})" if inst_num > 1 else ""))
                return model_manager
            except Exception as e:
                print(f"Error loading model {model_name}: {e}")
                return None
    
    def set_audio_model(self, model_name: str, config: Dict = None):
        """Add an audio transcription model and download/cache it if needed."""
        if model_name not in self.audio_models:
            self.audio_models.append(model_name)
        self.config[f"audio:{model_name}"] = config or {}

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

    def resolve_whisper_alias(self, name: str) -> Optional[WhisperServerManager]:
        """Return the next round-robin WhisperServerManager for an alias, or None."""
        ids = self.whisper_aliases.get(name)
        if not ids:
            return None
        idx = self._whisper_alias_counters.get(name, 0) % len(ids)
        self._whisper_alias_counters[name] = idx + 1
        return self.whisper_servers.get(ids[idx])
    
    def set_tts_model(self, model_name: str, config: Dict = None):
        """Set the text-to-speech model and download/cache it if needed."""
        self.tts_model = model_name
        self.config[f"tts:{model_name}"] = config or {}

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

        # For image models, we don't download at startup since they may be large
        # and handled by different backends (diffusers vs sd.cpp)
        # The download will happen when the model is first requested
        print(f"Registered image model: {model_name}")
    
    def set_vision_model(self, model_name: str, config: Dict = None):
        """Add a vision model and download/cache it if needed."""
        if model_name not in self.vision_models:
            self.vision_models.append(model_name)
        self.config[f"vision:{model_name}"] = config or {}

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
        print(f"Registered video model: {model_name}")

    def set_audio_gen_model(self, model_name: str, config: Dict = None):
        """Add a music/audio generation model (MusicGen, AudioLDM2, …)."""
        if model_name not in self.audio_gen_models:
            self.audio_gen_models.append(model_name)
        self.config[f"audio_gen:{model_name}"] = config or {}
        print(f"Registered audio-gen model: {model_name}")

    def set_embedding_model(self, model_name: str, config: Dict = None):
        """Add a text/image embedding model."""
        if model_name not in self.embedding_models:
            self.embedding_models.append(model_name)
        self.config[f"embedding:{model_name}"] = config or {}
        print(f"Registered embedding model: {model_name}")

    def set_model_alias(self, alias: str, model_name: str):
        """Register an alias for a model."""
        self.model_aliases[alias] = model_name
    
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
                            "video_models", "audio_gen_models", "embedding_models"):
                    for m in md.get(cat, []):
                        mid = (m if isinstance(m, str) else
                               m.get("alias") or m.get("path") or m.get("id") or "")
                        raw = (m if isinstance(m, str) else m.get("path") or m.get("id") or "")
                        for val in (mid, raw):
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
        return None

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

        # If a model_type is specified, reject models registered under a
        # different type (e.g. an image GGUF requested via /v1/chat/completions).
        if model_type:
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

    def _get_model_used_vram_gb(self, model_key: str, resolved_name: str = None) -> float:
        """Return VRAM requirement in GB for a model.

        Uses configured used_vram_gb when set; otherwise estimates from file size
        for local model files (GGUF, GGML, whisper .bin, etc.) — weights + KV cache
        + 15% buffer. Returns 0 when the requirement cannot be determined.
        """
        cfg = self.config.get(model_key, {})
        explicit = cfg.get("used_vram_gb")
        if explicit:
            return float(explicit)

        # Build a list of candidate paths to check
        candidates = []
        if resolved_name:
            candidates.append(resolved_name)
        # model_key is often the path for text GGUF models
        candidates.append(model_key)
        # Strip type prefix (e.g. "audio:/path/to/model.bin")
        if ":" in model_key:
            candidates.append(model_key.split(":", 1)[1])
        # Config may store the path explicitly
        for field in ("path", "model_path", "model"):
            v = cfg.get(field)
            if v:
                candidates.append(v)

        import os
        local_exts = {'.gguf', '.ggml', '.bin', '.pt', '.safetensors'}
        for path in candidates:
            if not path:
                continue
            _, ext = os.path.splitext(path)
            if ext.lower() not in local_exts:
                continue
            try:
                size_bytes = os.path.getsize(path)
                weights_gb = size_bytes / 1e9

                # KV cache: 2 bytes × 2 (K+V) × n_layers × n_ctx × head_dim
                # We don't know the architecture, so estimate from n_ctx alone.
                # Empirically ~0.5 MB per 1 K tokens covers most small models.
                n_ctx = cfg.get("n_ctx") or 2048
                kv_cache_gb = (n_ctx / 1024) * 0.5 / 1024  # 0.5 MB per 1 K ctx → GB

                # 15% overhead for activations, graph buffers, scratch space
                return weights_gb * 1.15 + kv_cache_gb
            except OSError:
                continue

        return 0

    def _evict_models_for_vram(self, needed_gb: float):
        """Unload loaded models (LRU first) until we have at least needed_gb free VRAM."""
        if needed_gb <= 0:
            return

        def _evict_key(key):
            model_obj = self.models.pop(key, None)
            self.models_in_vram.discard(key)
            if model_obj is not None:
                try:
                    if hasattr(model_obj, 'cleanup'):
                        model_obj.cleanup()
                    elif hasattr(model_obj, 'to'):
                        model_obj.to('cpu')
                except Exception as e:
                    print(f"  Warning during eviction of '{key}': {e}")
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        # First pass: evict non-active models in LRU order
        for key in list(self.models.keys()):
            if key == self.active_in_vram:
                continue
            if self._get_free_vram_gb() >= needed_gb:
                break
            print(f"On-request VRAM eviction: unloading '{key}' to free VRAM")
            _evict_key(key)

        # Second pass: evict active model if still not enough
        if self._get_free_vram_gb() < needed_gb and self.active_in_vram and self.active_in_vram in self.models:
            print(f"On-request VRAM eviction: unloading active model '{self.active_in_vram}' to free VRAM")
            _evict_key(self.active_in_vram)
            self.active_in_vram = None

    def request_model(self, requested_model: str, model_type: str = None) -> Dict[str, Any]:
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
        per_model_cfg = self.config.get(model_key, {})
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
            # Not loaded — check VRAM and evict if needed
            needed_gb = self._get_model_used_vram_gb(model_key, resolved_name)
            if needed_gb > 0:
                free_gb = self._get_free_vram_gb()
                if free_gb < needed_gb:
                    print(f"On-request: need {needed_gb:.1f} GB VRAM, have {free_gb:.1f} GB free — evicting models")
                    self._evict_models_for_vram(needed_gb)
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

                if needed_gb > 0 and free_gb >= needed_gb:
                    print(f"Ondemand mode - keeping '{loaded_canonical}' in VRAM alongside new model "
                          f"(need {needed_gb:.1f} GB, have {free_gb:.1f} GB free)")
                else:
                    print(f"Ondemand mode - model switch detected:")
                    print(f"  Requested: '{model_key}' (resolved: '{resolved_name}')")
                    print(f"  Currently loaded: '{loaded_canonical}'")
                    print(f"  -> Unloading current model(s) before loading new model...")
                    self.unload_all_models()

                    # Also cleanup the legacy singleton if it has a model
                    if has_legacy_model:
                        try:
                            print(f"  -> Cleaning up legacy model_manager...")
                            _legacy_mm.cleanup()
                        except Exception as e:
                            print(f"  Warning: Error cleaning up legacy model_manager: {e}")
        
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

    def acquire_model_instance(self, model_key: str):
        """Acquire the least-busy instance, incrementing its ref-count.

        Returns (instance_idx, model_obj) or None if no instance is loaded.
        Callers MUST call release_model_instance() when done.
        """
        pool = self.model_pools.get(model_key)
        if pool and pool.count > 0:
            return pool.acquire()
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
    
    def list_models(self) -> List[ModelInfo]:
        """List all available models (configured + runtime aliases) with type/capability metadata."""
        from codai.models.capabilities import detect_model_capabilities

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
        }

        def _add(model_id: str, model_type: str = None, meta: Dict[str, Any] = None):
            if model_id in seen_ids:
                return
            seen_ids.add(model_id)
            caps = detect_model_capabilities(model_id)
            # If heuristic detection missed the type (e.g. custom/vendor model IDs
            # that don't match any keyword), ensure the minimum capability for the
            # config-declared type is set so badges display correctly.
            if model_type and model_type in TYPE_MIN_CAP:
                min_cap = TYPE_MIN_CAP[model_type]
                if not getattr(caps, min_cap, False):
                    setattr(caps, min_cap, True)
            resolved_type = model_type or (caps.to_list()[0].split("_")[0] if caps.to_list() else "text")
            meta = meta or {}
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
                            "video_models", "audio_gen_models", "embedding_models"):
                    mtype = CAT_TYPE.get(cat, "text")
                    for m in md.get(cat, []):
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

        # --- Custom aliases ---
        for alias in self.model_aliases:
            _add(alias)

        return models


# =============================================================================
# Singleton Instances
# =============================================================================

# Global singleton instances for convenience
model_manager = ModelManager()
multi_model_manager = MultiModelManager()
