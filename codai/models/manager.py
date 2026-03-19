"""Model manager module - contains ModelManager, WhisperServerManager, and MultiModelManager classes."""

from typing import Optional, Dict, Any, List
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


class MultiModelManager:
    """
    Manages multiple models: main text model, audio transcription, and image generation.
    """
    
    def __init__(self):
        self.models: Dict[str, ModelManager] = {}
        self.default_model: Optional[str] = None
        self.audio_models: List[str] = []
        self.tts_model: Optional[str] = None
        self.image_models: List[str] = []
        self.vision_models: List[str] = []
        self.config: Dict[str, Dict] = {}  # Store model configurations
        self.tool_parser = ModelParserAdapter()
        self.current_model_key: Optional[str] = None
        self.config: Dict[str, Dict] = {}  # Store model configurations
        self.tool_parser = ModelParserAdapter()
        self.current_model_key: Optional[str] = None
        self.load_mode: str = "ondemand"
        self.active_in_vram: Optional[str] = None
        self.model_aliases: Dict[str, str] = {}
        self.whisper_server: Optional[WhisperServerManager] = None
        self.model_backend_types: Dict[str, str] = {}
        self.tool_breaker = FuzzyToolBreaker(threshold=3)  # Circuit breaker for repetitive tool calls
    
    @property
    def image_model(self) -> Optional[str]:
        """Return the first image model or None."""
        return self.image_models[0] if self.image_models else None
    
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
        
        # Cleanup whisper server
        if self.whisper_server:
            try:
                self.whisper_server.stop()
            except Exception as e:
                print(f"Warning: Error cleaning up whisper server: {e}")
        
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
        """Load the default model on demand."""
        if not self.default_model:
            return None
        
        # Check if already loaded
        if self.default_model in self.models:
            return self.models[self.default_model]
        
        # Get config and backend type
        config = self.config.get(self.default_model, {})
        backend_type = self.model_backend_types.get(self.default_model, "auto")
        
        # Get global args for additional parameters
        try:
            from codai.api.state import get_global_args
            global_args = get_global_args()
        except:
            global_args = None
        
        # Create new model manager and load the model
        model_manager = ModelManager()
        
        try:
            # Build kwargs from config
            kwargs = {}
            if 'ctx' in config:
                kwargs['ctx'] = config['ctx']
            if global_args:
                if hasattr(global_args, 'n_gpu_layers'):
                    kwargs['n_gpu_layers'] = global_args.n_gpu_layers
                if hasattr(global_args, 'offload_dir'):
                    kwargs['offload_dir'] = global_args.offload_dir
                if hasattr(global_args, 'ram'):
                    kwargs['ram'] = global_args.ram
            
            print(f"Loading default model on demand: {self.default_model}")
            model_manager.load_model(self.default_model, backend_type=backend_type, **kwargs)
            
            # Add to models dict
            self.models[self.default_model] = model_manager
            self.current_model_key = self.default_model
            
            print(f"Model loaded successfully: {self.default_model}")
            return model_manager
            
        except Exception as e:
            print(f"Error loading model {self.default_model}: {e}")
            return None
    
    def _load_model_by_name(self, model_name: str):
        """Load a model by name on demand."""
        # Check if already loaded
        if model_name in self.models:
            return self.models[model_name]
        
        # Check if it's registered in config
        config = self.config.get(model_name, {})
        backend_type = self.model_backend_types.get(model_name, "auto")
        
        # Get global args for additional parameters
        try:
            from codai.api.state import get_global_args
            global_args = get_global_args()
        except:
            global_args = None
        
        # Create new model manager and load the model
        model_manager = ModelManager()
        
        try:
            # Build kwargs from config
            kwargs = {}
            if 'ctx' in config:
                kwargs['ctx'] = config['ctx']
            if global_args:
                if hasattr(global_args, 'n_gpu_layers'):
                    kwargs['n_gpu_layers'] = global_args.n_gpu_layers
                if hasattr(global_args, 'offload_dir'):
                    kwargs['offload_dir'] = global_args.offload_dir
                if hasattr(global_args, 'ram'):
                    kwargs['ram'] = global_args.ram
            
            print(f"Loading model on demand: {model_name}")
            model_manager.load_model(model_name, backend_type=backend_type, **kwargs)
            
            # Add to models dict
            self.models[model_name] = model_manager
            self.current_model_key = model_name
            
            print(f"Model loaded successfully: {model_name}")
            return model_manager
            
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            return None
    
    def set_audio_model(self, model_name: str, config: Dict = None):
        """Add an audio transcription model and download/cache it if needed."""
        if model_name not in self.audio_models:
            self.audio_models.append(model_name)
        self.config[f"audio:{model_name}"] = config or {}

        # Download/cache the model at startup if it's a URL or HF ID
        resolved_model = self.load_model(model_name)
        if resolved_model != model_name:
            # Model was downloaded/cached, update the stored name
            idx = self.audio_models.index(model_name)
            self.audio_models[idx] = resolved_model
            self.config[f"audio:{resolved_model}"] = self.config.pop(f"audio:{model_name}")
            print(f"Audio model '{model_name}' cached as: {resolved_model}")
    
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

        # Download/cache the model at startup if it's a URL or HF ID
        resolved_model = self.load_model(model_name)
        if resolved_model != model_name:
            # Model was downloaded/cached, update the stored name
            idx = self.vision_models.index(model_name)
            self.vision_models[idx] = resolved_model
            self.config[f"vision:{resolved_model}"] = self.config.pop(f"vision:{model_name}")
            print(f"Vision model '{model_name}' cached as: {resolved_model}")
    
    def set_model_alias(self, alias: str, model_name: str):
        """Register an alias for a model."""
        self.model_aliases[alias] = model_name
    
    def get_model_for_request(self, requested_model: str):
        """Get the appropriate model manager for a request based on model name."""
        global global_args
        
        # Resolve custom aliases first
        if requested_model in self.model_aliases:
            requested_model = self.model_aliases[requested_model]
        
        # Handle empty or "default" model names
        if not requested_model or requested_model == "default":
            if self.default_model:
                # Check if already loaded
                if self.default_model in self.models:
                    self.current_model_key = self.default_model
                    return self.models[self.default_model]
                # Model not loaded yet - try to load it
                return self._load_default_model()
            return None
        
        # Handle "audio" alias
        if requested_model == "audio":
            if self.audio_models:
                first_audio = self.audio_models[0]
                key = f"audio:{first_audio}"
                if key in self.models:
                    self.current_model_key = key
                    return self.models[key]
            return None
        
        # Handle "image" alias
        if requested_model == "image":
            if self.image_models:
                first_image = self.image_models[0]
                key = f"image:{first_image}"
                if key in self.models:
                    self.current_model_key = key
                    return self.models[key]
            return None
        
        # Handle "tts" alias
        if requested_model == "tts":
            if self.tts_model:
                key = f"tts:{self.tts_model}"
                if key in self.models:
                    self.current_model_key = key
                    return self.models[key]
            return None
        
        # Handle prefixed models
        if requested_model.startswith("audio:"):
            audio_name = requested_model[6:]
            key = f"audio:{audio_name}"
            if key in self.models:
                self.current_model_key = key
                return self.models[key]
            return None
        
        if requested_model.startswith("tts:"):
            tts_name = requested_model[4:]
            key = f"tts:{tts_name}"
            if key in self.models:
                self.current_model_key = key
                return self.models[key]
            return None
        
        if requested_model.startswith("vision:") or requested_model.startswith("image:"):
            if requested_model.startswith("vision:"):
                image_name = requested_model[7:]
            else:
                image_name = requested_model[6:]
            key = f"image:{image_name}"
            if key in self.models:
                self.current_model_key = key
                return self.models[key]
            return None
        
        # Check if it's the default model
        if self.default_model and (requested_model == self.default_model or 
                                    requested_model.endswith(self.default_model.split("/")[-1])):
            # Check if already loaded
            if self.default_model in self.models:
                self.current_model_key = self.default_model
                return self.models[self.default_model]
            # Try to load the default model
            return self._load_default_model()
        
        # Check if any loaded model matches
        for key, model in self.models.items():
            if requested_model in key or key.endswith(requested_model.split("/")[-1]):
                self.current_model_key = key
                return model
        
        # Model not found - try to load it as a new model
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
        
        # Handle prefixed models - normalize them
        if requested_model.startswith("audio:"):
            return requested_model
        if requested_model.startswith("tts:"):
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
    
    def add_model(self, key: str, manager: ModelManager):
        """Add a model manager for a specific key."""
        self.models[key] = manager
    
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
        """List all available models."""
        models = []
        
        # Add default model(s)
        if self.default_model:
            model_id = self.default_model
            if not (model_id.startswith("http://") or model_id.startswith("https://")):
                short_name = self.default_model.split("/")[-1] if "/" in self.default_model else self.default_model
                if short_name != self.default_model:
                    models.append(ModelInfo(id=short_name))
                models.append(ModelInfo(id=model_id))
                models.append(ModelInfo(id="default"))
        
        # Add aliases for first/default models
        if self.audio_models:
            models.append(ModelInfo(id="audio"))
            for audio_id in self.audio_models:
                models.append(ModelInfo(id=f"audio:{audio_id}"))
        
        if self.tts_model:
            models.append(ModelInfo(id="tts"))
            models.append(ModelInfo(id=f"tts:{self.tts_model}"))
        
        if self.image_models:
            models.append(ModelInfo(id="image"))
            for image_id in self.image_models:
                models.append(ModelInfo(id=f"image:{image_id}"))
        
        if self.vision_models:
            models.append(ModelInfo(id="vision"))
            for vision_id in self.vision_models:
                models.append(ModelInfo(id=f"vision:{vision_id}"))
        
        # Add any custom aliases
        for alias in self.model_aliases:
            models.append(ModelInfo(id=alias))
        
        return models


# =============================================================================
# Singleton Instances
# =============================================================================

# Global singleton instances for convenience
model_manager = ModelManager()
multi_model_manager = MultiModelManager()
