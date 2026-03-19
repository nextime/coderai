"""
Image generation endpoints for the codai API.
"""

import asyncio
import base64
import io
import os
import uuid

from fastapi import APIRouter, HTTPException, Request
from PIL import Image

# Import from codai modules
from codai.models.manager import multi_model_manager
from codai.pydantic.imagerequest import ImageGenerationRequest
from codai.api.state import get_load_mode


# Global reference to be set by coderai
global_args = None
global_file_path = None

# Model semaphores for concurrency control (provided by coderai)
model_semaphores = {}
queue_flags = {}


# =============================================================================
# Helper Functions
# =============================================================================

def get_cfg_scale():
    """Get CFG scale for image generation. Auto-detect VRAM for Vulkan."""
    global global_args
    
    cfg_scale = getattr(global_args, 'image_cfg_scale', 1.0)
    
    # If using Vulkan and CLI didn't specify cfg_scale (default 1.0), check VRAM
    if cfg_scale == 1.0:  # Only auto-detect if using default
        backend = getattr(global_args, 'backend', 'auto')
        image_backend = getattr(global_args, 'image_backend', 'auto')
        
        # Check if using Vulkan (either global or image-specific)
        use_vulkan = (backend == 'vulkan') or (image_backend == 'vulkan') or (image_backend == 'auto' and backend == 'auto')
        
        if use_vulkan:
            # Try to detect VRAM
            try:
                import subprocess
                # Try vulkaninfo first
                result = subprocess.run(['vulkaninfo', '-J'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    import json
                    data = json.loads(result.stdout)
                    # Find device memory
                    for dev in data.get('devices', []):
                        mem = dev.get('deviceMemoryHeap', [{}])
                        for heap in mem:
                            if heap.get('flags', []).get('deviceLocal', False):
                                vram_mb = heap.get('size', 0) / (1024 * 1024)
                                print(f"DEBUG: Detected VRAM: {vram_mb:.0f} MB")
                                if vram_mb < 16000:  # Less than 16GB
                                    print(f"DEBUG: VRAM < 16GB, using cfg_scale=1.0 for better performance")
                                    return 1.0
                                break
            except Exception as e:
                print(f"DEBUG: Could not detect VRAM: {e}")
                # Default to 1.0 for Vulkan if detection fails
                return 1.0
    
    return cfg_scale


def save_image_response(img, request_format="base64", http_request=None):
    """
    Save image to file path if configured, return response dict.
    
    If --file-path is set and request_format is url (not base64), return only URL.
    If --file-path is set and request_format is base64, return both URL and base64.
    If --file-path is not set, return base64 as usual.
    """
    global global_file_path, global_args
    
    # Convert to PIL Image if needed
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    
    result = {}
    
    # Save to file path if configured
    if global_file_path:
        os.makedirs(global_file_path, exist_ok=True)
        # Generate unique filename
        filename = f"{uuid.uuid4().hex}.png"
        file_path = os.path.join(global_file_path, filename)
        img.save(file_path, format="PNG")
        # Add URL to response
        # Determine base URL based on --url argument
        url_setting = getattr(global_args, 'url', 'auto') if global_args else 'auto'
        print(f"DEBUG: global_args={global_args}, url_setting={url_setting}")
        if url_setting == 'auto':
            # Use server host from request headers (what client used to connect)
            if http_request:
                # Get the Host header - this is what the client used to reach the server
                # The Host header typically includes the port, e.g., "192.168.1.1:6745"
                client_host = http_request.headers.get('host', '')
                if not client_host:
                    # Fallback to client IP if no Host header
                    client_host = http_request.client.host if http_request.client else '127.0.0.1'
                
                # Strip port from host if present (Host header includes port like "192.168.1.1:6745")
                if ':' in client_host:
                    # Check if the part after : is a port number
                    parts = client_host.split(':')
                    if len(parts) == 2 and parts[1].isdigit():
                        # It's a port number, strip it
                        client_host = parts[0]
                    elif len(parts) > 2:
                        # IPv6 or other format, take last part as port check
                        last_part = parts[-1]
                        if last_part.isdigit():
                            client_host = ':'.join(parts[:-1])
                
                # Check if HTTPS is enabled
                use_https = getattr(global_args, 'https', False) or getattr(global_args, 'pubkey', None)
                protocol = "https" if use_https else "http"
                port = getattr(global_args, 'port', 8000)
                base_url = f"{protocol}://{client_host}:{port}"
                print(f"DEBUG: client_host={client_host}, port={port}, base_url={base_url}")
            else:
                base_url = "http://127.0.0.1:8000"
        else:
            # Use explicitly provided URL (strip trailing slash if present)
            base_url = url_setting.rstrip('/')
        result["url"] = f"{base_url}/v1/files/{filename}"
        
        # If client explicitly requested base64, include it
        # Otherwise, only return URL when file-path is set
        if request_format == "base64":
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_bytes = buffered.getvalue()
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            result["b64_json"] = img_base64
    else:
        # No file-path, return base64 as usual
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        result["b64_json"] = img_base64
    
    return result


def set_global_args(args):
    """Set global args from coderai."""
    global global_args
    global_args = args


def set_global_file_path(path):
    """Set global file path from coderai."""
    global global_file_path
    global_file_path = path


def set_model_semaphores(semaphores):
    """Set model semaphores from coderai for concurrency control."""
    global model_semaphores
    model_semaphores = semaphores


def set_queue_flags(flags):
    """Set queue flags from coderai."""
    global queue_flags
    queue_flags = flags


# =============================================================================
# Router and Endpoints
# =============================================================================

router = APIRouter()


@router.post("/v1/images/generations")
async def create_image_generation(request: ImageGenerationRequest, http_request: Request = None):
    """
    Image generation endpoint (OpenAI-compatible).
    
    Supports:
    - Stable Diffusion via stable-diffusion-cpp-python (sd.cpp)
    - Stable Diffusion XL (via local inference with diffusers)
    """
    global global_args, global_file_path, model_semaphores, queue_flags
    
    # Get or create semaphore for this model
    model_key = f"image:{request.model}" if request.model else "image"
    mode = get_load_mode()
    
    # Check if --image-1 is set (no queue, return 409 if busy)
    use_1_mode = queue_flags.get("image_1", False)
    
    # In loadall mode, allow 1 concurrent request per model
    # In ondemand mode, serialize all requests (use global semaphore)
    if mode == "loadall":
        if model_key not in model_semaphores:
            model_semaphores[model_key] = asyncio.Semaphore(1)
        semaphore = model_semaphores[model_key]
    else:
        # Use a global semaphore for ondemand mode
        if "global_image" not in model_semaphores:
            model_semaphores["global_image"] = asyncio.Semaphore(1)
        semaphore = model_semaphores["global_image"]
    
    # Try to acquire semaphore without blocking
    if use_1_mode:
        acquired = semaphore.locked()
        if acquired:
            raise HTTPException(
                status_code=409,
                detail="Image model is busy. Try again later."
            )
    
    async with semaphore:
        image_model = multi_model_manager.image_model
    
    # If no image model configured, try to use main --model as fallback
    if not image_model:
        # Try to get the main model from args
        main_model = getattr(global_args, 'model', None)
        if main_model and isinstance(main_model, list) and len(main_model) > 0:
            image_model = main_model[0]
        elif main_model:
            image_model = main_model
        
        # Check if main model is a GGUF file - can't use for image generation
        if image_model and ('.gguf' in image_model.lower() or 'gguf' in image_model.lower()):
            print(f"Note: Main model is a GGUF file (for text), not suitable for image generation")
            image_model = None  # Can't use GGUF for images
    
    # If still no image model configured, return an error
    if not image_model:
        raise HTTPException(
            status_code=400,
            detail="Image generation not configured. Use --image-model to specify a model."
        )
    
    # Determine model to use
    # Priority: 1) model specified in request, 2) default image model from --image-model
    model_to_use = request.model
    if not model_to_use or model_to_use == "image":
        # No model specified in request, use default
        model_to_use = image_model
    elif model_to_use.startswith("image:"):
        # Legacy format - strip prefix and use default
        model_to_use = image_model
    else:
        # Check if model_to_use is a valid model (URL, file, or known model)
        # If not, fallback to the configured image model to avoid HF resolution errors
        if image_model:
            is_url = model_to_use.startswith('http://') or model_to_use.startswith('https://')
            is_file = os.path.isfile(model_to_use) if model_to_use else False
            if not is_url and not is_file:
                # Unknown model name - use default instead of trying to resolve as HF
                print(f"Warning: Unknown model '{model_to_use}' in image generation request, using configured --image-model")
                model_to_use = image_model
    
    # Check if model is loaded
    model_key = f"image:{model_to_use}"
    pipeline = multi_model_manager.get_model(model_key)
    
    # In ondemand mode, if ANY model is loaded in VRAM and it's different from what we need,
    # fully unload it first to free VRAM
    if mode == "ondemand":
        from codai.models.manager import model_manager
        has_any_model = len(multi_model_manager.models) > 0 or model_manager.backend is not None
        
        if has_any_model:
            # Resolve both the requested image model and currently loaded model to their canonical names
            requested_canonical = multi_model_manager.resolve_model_name(f"image:{model_to_use}")
            loaded_canonical = multi_model_manager.get_currently_loaded_model_name()
            
            # Also check legacy model_manager
            if not loaded_canonical and model_manager.backend is not None:
                loaded_canonical = "legacy_model_manager"
            
            # Compare: if they're different models (even if both are image models), unload first
            already_loaded = (requested_canonical and loaded_canonical and 
                            requested_canonical == loaded_canonical)
            
            if not already_loaded:
                print(f"In ondemand mode - model switch detected:")
                print(f"  Requested: 'image:{model_to_use}' (resolved to: '{requested_canonical}')")
                print(f"  Loaded: '{loaded_canonical}'")
                print(f"  -> Fully unloading current model(s) before loading new model...")
                multi_model_manager.unload_all_models()
                if model_manager.backend is not None:
                    try:
                        model_manager.cleanup()
                    except:
                        pass
        
        # Try diffusers first
        try:
            from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, DiffusionPipeline
            import torch
            
            # Check if model is XL
            is_xl = "xl" in model_to_use.lower() or "sdxl" in model_to_use.lower()
            
            # Check if it's a GGUF model - skip diffusers for those
            is_gguf_model = (model_to_use.endswith('.gguf') or 'gguf' in model_to_use.lower() or
                            (model_to_use.startswith('http') and '.gguf' in model_to_use))
            
            if is_gguf_model:
                print(f"GGUF model detected ({model_to_use}), skipping diffusers, using stable-diffusion-cpp...")
                raise Exception("GGUF model - use stable-diffusion-cpp instead")
            
            print(f"Loading diffusers model: {model_to_use}")
            
            # Determine precision from CLI argument (--image-precision)
            precision = getattr(global_args, 'image_precision', 'f32') or 'f32'
            precision_map = {
                'bf16': torch.bfloat16,
                'f32': torch.float32,
                'f16': torch.float16,
            }
            if hasattr(torch, 'float8_e4m3fn'):
                precision_map['f8'] = torch.float8_e4m3fn
            dtype = precision_map.get(precision, torch.float32)
            print(f"Using precision: {precision} ({dtype})")
            
            # Check if CPU offload is requested via CLI
            use_sequential_offload = getattr(global_args, 'image_cpu_offload', False)
            
            # Track loading attempts for OOM handling
            load_attempt = 0
            max_attempts = 3
            
            while pipeline is None and load_attempt < max_attempts:
                try:
                    load_attempt += 1
                    print(f"Loading attempt {load_attempt}/{max_attempts}...")
                    
                    # Try to load as Stable Diffusion XL first, then generic DiffusionPipeline
                    try:
                        pipeline = StableDiffusionXLPipeline.from_pretrained(
                            model_to_use,
                            torch_dtype=dtype,
                            use_safetensors=True,
                        )
                    except Exception:
                        # Try generic diffusion pipeline (supports custom pipelines like ZImagePipeline)
                        pipeline = DiffusionPipeline.from_pretrained(
                            model_to_use,
                            torch_dtype=dtype,
                            use_safetensors=True,
                        )
                    
                    # Apply memory optimizations based on attempt
                    if torch.cuda.is_available():
                        if load_attempt >= 2:
                            # Second attempt: enable attention slicing
                            print("Enabling attention slicing for lower VRAM usage...")
                            if hasattr(pipeline, 'enable_attention_slicing'):
                                pipeline.enable_attention_slicing()
                        
                        if load_attempt >= 3 or use_sequential_offload:
                            # Third attempt or offload requested: enable sequential CPU offload
                            print("Enabling sequential CPU offload for lower VRAM usage...")
                            if hasattr(pipeline, 'enable_sequential_cpu_offload'):
                                pipeline.enable_sequential_cpu_offload()
                        else:
                            # First attempt: try regular GPU
                            pipeline = pipeline.to("cuda")
                    else:
                        pipeline = pipeline.to("cpu")
                    
                except Exception as load_error:
                    error_msg = str(load_error).lower()
                    is_oom = any(x in error_msg for x in ['out of memory', 'oom', 'cuda error', 'cudamalloc'])
                    
                    if is_oom and load_attempt < max_attempts:
                        print(f"OOM during model loading: {load_error}")
                        print(f"Retrying with more aggressive memory optimization...")
                        pipeline = None  # Reset for retry
                    else:
                        print(f"Failed to load model (attempt {load_attempt}): {load_error}")
                        if load_attempt >= max_attempts:
                            raise
                        pipeline = None
            
            # Cache the model
            if pipeline is not None:
                multi_model_manager.add_model(model_key, pipeline)
                print(f"Loaded diffusers model: {model_to_use}")
            
        except ImportError as e:
            # diffusers not installed
            diffusers_error = str(e)
            print(f"diffusers not available: {diffusers_error}")
        except Exception as e:
            import traceback
            diffusers_error = str(e)
            print(f"diffusers error: {diffusers_error}")
            print(f"Traceback: {traceback.format_exc()}")
    
    # Try diffusers if available
    if pipeline is not None:
        try:
            # Determine size
            width, height = 1024, 1024
            if request.size:
                parts = request.size.split("x")
                if len(parts) == 2:
                    try:
                        width = int(parts[0])
                        height = int(parts[1])
                    except ValueError:
                        pass
            
            # Check for nan/inf in dimensions
            if width != width or width == float('inf'):  # NaN or inf check
                width = 512
            if height != height or height == float('inf'):  # NaN or inf check
                height = 512
            
            # Import torch for generation
            import torch
            
            # Ensure model is on correct device
            backend = getattr(global_args, 'backend', 'auto')
            image_backend = getattr(global_args, 'image_backend', 'auto')
            use_vulkan = (backend == 'vulkan') or (image_backend == 'vulkan') or (image_backend == 'auto' and backend == 'auto')
            
            if use_vulkan and not torch.cuda.is_available():
                # CPU mode - try to reduce memory usage
                try:
                    if hasattr(pipeline, 'enable_attention_slicing'):
                        pipeline.enable_attention_slicing(slice_size="auto")
                    if hasattr(pipeline, 'enable_vae_slicing'):
                        pipeline.enable_vae_slicing()
                except Exception as e:
                    print(f"Warning: Could not enable memory optimizations: {e}")
            elif torch.cuda.is_available():
                # Try to enable memory optimizations for CUDA
                try:
                    if hasattr(pipeline, 'enable_attention_slicing'):
                        pipeline.enable_attention_slicing(slice_size="auto")
                    if hasattr(pipeline, 'enable_vae_slicing'):
                        pipeline.enable_vae_slicing()
                except Exception as e:
                    print(f"Warning: Could not enable CUDA memory optimizations: {e}")
            
            # Get timestamp BEFORE calling diffusers (to avoid scope issues)
            import time as time_module
            timestamp = int(time_module.time())
            
            # Generate images
            # Use request seed if provided, otherwise use CLI default seed
            seed = request.seed if request.seed is not None else getattr(global_args, 'image_seed', None)
            generator = None
            if seed is not None:
                generator = torch.Generator(device=pipeline.device).manual_seed(seed)
            
            # Quality: "standard" or "hd"
            quality = request.quality or "standard"
            
            # Use request parameters if provided, otherwise fall back to quality-based defaults
            num_steps = request.steps if request.steps else (30 if quality == "standard" else 50)
            cfg_scale = request.guidance_scale if request.guidance_scale else (
                getattr(global_args, 'image_cfg_scale', 7.5) if quality == "standard" else 9.0
            )
            
            # Generate
            result = pipeline(
                prompt=request.prompt,
                negative_prompt=None,
                num_images_per_prompt=request.n,
                height=height,
                width=width,
                generator=generator,
                guidance_scale=cfg_scale,
                num_inference_steps=num_steps,
            )
            
            # Extract images
            images = []
            try:
                result_images = result.images
            except Exception as img_err:
                print(f"Warning: Could not access result.images: {img_err}")
                # Try alternative: result might have 'image' or 'output'
                result_images = getattr(result, 'image', None) or getattr(result, 'output', None)
                if result_images is None:
                    raise Exception(f"Could not extract images from diffusers result: {img_err}")
            
            for img in result_images:
                # Convert to base64
                import numpy as np
                
                # Debug: print image type and value range
                print(f"DEBUG: Image type: {type(img)}")
                if isinstance(img, np.ndarray):
                    print(f"DEBUG: Image shape: {img.shape}, dtype: {img.dtype}, min: {img.min()}, max: {img.max()}")
                    # Handle NaN/Inf values in image data - convert to valid values
                    # Replace NaN and Inf with valid values
                    img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
                    # Clip to valid range [0, 1]
                    img = np.clip(img, 0.0, 1.0)
                    print(f"DEBUG: After NaN handling - min: {img.min()}, max: {img.max()}")
                
                # Use helper function to save and get response
                img_data = save_image_response(img, request.response_format, http_request)
                images.append(img_data)
            
            return {
                "created": timestamp,
                "data": images
            }
            
        except ImportError as e:
            # diffusers/torch not installed - record error and try sd.cpp
            diffusers_error = str(e)
            print(f"diffusers not available: {diffusers_error}, trying stable-diffusion-cpp-python...")
        except Exception as e:
            # Other error with diffusers - record and try sd.cpp
            import traceback
            diffusers_error = str(e)
            print(f"diffusers error: {diffusers_error}")
            print(f"Traceback: {traceback.format_exc()}")
            print(f"Trying stable-diffusion-cpp-python...")
    
    # Try stable-diffusion-cpp-python (sd.cpp) as fallback
    # First, check all available image models to find one loaded via sd.cpp
    # Always check for cached models - allows dynamically loaded models to be reused across requests
    sd_model = None
    for key in multi_model_manager.models:
        if key.startswith("image:"):
            potential_model = multi_model_manager.get_model(key)
            if potential_model is not None:
                # Check if it's a stable-diffusion-cpp model
                try:
                    from stable_diffusion_cpp import StableDiffusion
                    if isinstance(potential_model, StableDiffusion):
                        sd_model = potential_model
                        print(f"Found cached stable-diffusion-cpp model with key: {key}")
                        break
                except ImportError:
                    pass
    
    # If no cached image model found, need to load one - first cleanup any existing models
    if sd_model is None:
        # In ondemand mode, check if we need to unload before loading sd.cpp model
        from codai.models.manager import model_manager
        has_any_model = len(multi_model_manager.models) > 0 or model_manager.backend is not None
        
        if mode == "ondemand" and has_any_model:
            # Resolve both the requested image model and currently loaded model to their canonical names
            requested_canonical = multi_model_manager.resolve_model_name(f"image:{model_to_use}")
            loaded_canonical = multi_model_manager.get_currently_loaded_model_name()
            
            # Also check legacy model_manager
            if not loaded_canonical and model_manager.backend is not None:
                loaded_canonical = "legacy_model_manager"
            
            # Compare: if they're different models, unload first
            already_loaded = (requested_canonical and loaded_canonical and 
                            requested_canonical == loaded_canonical)
            
            if not already_loaded:
                print(f"In ondemand mode - model switch detected:")
                print(f"  Requested: 'image:{model_to_use}' (resolved to: '{requested_canonical}')")
                print(f"  Loaded: '{loaded_canonical}'")
                print(f"  -> Fully unloading current model(s) before loading sd.cpp model...")
                multi_model_manager.unload_all_models()
                if model_manager.backend is not None:
                    try:
                        model_manager.cleanup()
                    except:
                        pass
    
    if sd_model is not None:
        # Check if it's a stable-diffusion-cpp model (has generate method from sd.cpp)
        try:
            from stable_diffusion_cpp import StableDiffusion
            if isinstance(sd_model, StableDiffusion):
                print(f"Using stable-diffusion-cpp-python for image generation")
                # Use sd.cpp for generation
                # Parse size
                width, height = 512, 512
                if request.size:
                    parts = request.size.split("x")
                    if len(parts) == 2:
                        try:
                            width = int(parts[0])
                            height = int(parts[1])
                        except ValueError:
                            pass
                
                # Use default steps for Z-Image Turbo (very fast)
                steps = 4  # Default for fast generation
                
                # Generate images using sd.cpp (run in thread to not block event loop)
                # Use request seed if provided, otherwise use CLI default seed
                seed = request.seed if request.seed is not None else getattr(global_args, 'image_seed', None)
                
                result = await asyncio.to_thread(
                    sd_model.generate_image,
                    prompt=request.prompt,
                    negative_prompt='',
                    width=width,
                    height=height,
                    cfg_scale=get_cfg_scale(),
                    sample_steps=steps,
                    seed=seed if seed is not None else 42,
                    batch_count=request.n if request.n else 1,
                )
                
                # Small delay to let Vulkan driver settle after generation
                import time
                time.sleep(0.1)
                
                # Convert results to response format
                images = []
                
                for img in result:
                    # Use helper function to save and get response
                    img_data = save_image_response(img, http_request=http_request)
                    images.append(img_data)
                
                return {
                    "created": int(time.time()),
                    "data": images
                }
        except ImportError as e:
            # stable-diffusion-cpp not available
            sd_cpp_error = str(e)
            print(f"stable-diffusion-cpp-python not available: {sd_cpp_error}")
        except Exception as e:
            print(f"sd.cpp generation error: {e}")
            sd_cpp_error = str(e)
    else:
        # No sd.cpp model pre-loaded, try to load dynamically
        print("No pre-loaded sd.cpp model found, trying to load...")
        try:
            from stable_diffusion_cpp import StableDiffusion
            
            # Check if model_to_use is a URL and get cached path
            # Also handle HuggingFace model IDs that need to be resolved
            model_path = None
            if model_to_use.startswith('http://') or model_to_use.startswith('https://'):
                cached_path = multi_model_manager.get_cached_model_path(model_to_use)
                if cached_path:
                    model_path = cached_path
                    print(f"Using cached model: {model_path}")
                else:
                    # Not cached - download it
                    print(f"Downloading model: {model_to_use}")
                    cache_dir = multi_model_manager.get_model_cache_dir()
                    model_path = multi_model_manager.download_model(model_to_use, cache_dir)
                    print(f"Downloaded to: {model_path}")
            elif os.path.isfile(model_to_use):
                model_path = model_to_use
            else:
                # Try to resolve as HuggingFace model ID
                print(f"Trying to resolve as HuggingFace model ID: {model_to_use}")
                try:
                    from huggingface_hub import hf_hub_download, list_repo_files
                    
                    # Parse model name (format: "org/model" or "org/model/filename.gguf")
                    parts = model_to_use.split('/')
                    if len(parts) >= 2:
                        repo_id = f"{parts[0]}/{parts[1]}"
                        
                        # First check if there's a cached GGUF file for this model
                        # Try common GGUF file patterns
                        files = list_repo_files(repo_id)
                        gguf_files = [f for f in files if f.endswith('.gguf')]
                        
                        if gguf_files:
                            # Try to find a cached version first
                            for gguf_file in gguf_files:
                                # Construct potential URL and check cache
                                potential_url = f"https://huggingface.co/{repo_id}/resolve/main/{gguf_file}"
                                cached = multi_model_manager.get_cached_model_path(potential_url)
                                if cached:
                                    model_path = cached
                                    print(f"Using cached GGUF model: {model_path}")
                                    break
                            
                            # If not cached, download the first GGUF file
                            if not model_path:
                                print(f"Downloading GGUF model from HF: {gguf_files[0]}")
                                model_path = hf_hub_download(repo_id=repo_id, filename=gguf_files[0])
                                print(f"Downloaded to: {model_path}")
                except Exception as e:
                    print(f"Could not resolve as HuggingFace model: {e}")
            
            if model_path is None:
                print("Warning: Could not resolve sd.cpp model path via HuggingFace GGUF resolution")
                # Fallback: try to use the model name as a direct path (for local models or if HF resolution failed)
                print(f"Fallback: attempting to use '{model_to_use}' as direct model path")
                if os.path.isfile(model_to_use):
                    model_path = model_to_use
                    print(f"Using local file: {model_path}")
                else:
                    # Not a local file, check if it might be a cached model under a different name
                    cached_path = multi_model_manager.get_cached_model_path(model_to_use)
                    if cached_path:
                        model_path = cached_path
                        print(f"Using cached model: {model_path}")
                    else:
                        # Last resort: try to download it as if it were a URL
                        print(f"Attempting to download '{model_to_use}' as model URL")
                        try:
                            cache_dir = multi_model_manager.get_model_cache_dir()
                            model_path = multi_model_manager.download_model(model_to_use, cache_dir)
                            print(f"Downloaded to: {model_path}")
                        except Exception as download_error:
                            print(f"Download failed: {download_error}")
                            model_path = None
                
                if model_path is None:
                    print("Error: Could not resolve sd.cpp model path through any method")
                    sd_cpp_error = "Could not resolve model path"
                else:
                    # Load sd.cpp model (continue below)
                    pass
            
            # Load sd.cpp model if we have a valid path
            if model_path is not None:
                # Check if it's a stable-diffusion-cpp model (has generate method from sd.cpp)
                try:
                    from stable_diffusion_cpp import StableDiffusion
                    if isinstance(sd_model, StableDiffusion):
                        print(f"Using stable-diffusion-cpp-python for image generation")
                        # Use sd.cpp for generation
                        # Parse size
                        width, height = 512, 512
                        if request.size:
                            parts = request.size.split("x")
                            if len(parts) == 2:
                                try:
                                    width = int(parts[0])
                                    height = int(parts[1])
                                except ValueError:
                                    pass
                        
                        # Use default steps for Z-Image Turbo (very fast)
                        steps = 4  # Default for fast generation
                        
                        # Generate images using sd.cpp (run in thread to not block event loop)
                        # Use request seed if provided, otherwise use CLI default seed
                        seed = request.seed if request.seed is not None else getattr(global_args, 'image_seed', None)
                        
                        result = await asyncio.to_thread(
                            sd_model.generate_image,
                            prompt=request.prompt,
                            negative_prompt='',
                            width=width,
                            height=height,
                            cfg_scale=get_cfg_scale(),
                            sample_steps=steps,
                            seed=seed if seed is not None else 42,
                            batch_count=request.n if request.n else 1,
                        )
                        
                        # Small delay to let Vulkan driver settle after generation
                        import time
                        time.sleep(0.1)
                        
                        # Convert results to response format
                        images = []
                        
                        for img in result:
                            # Use helper function to save and get response
                            img_data = save_image_response(img, http_request=http_request)
                            images.append(img_data)
                        
                        return {
                            "created": int(time.time()),
                            "data": images
                        }
                except ImportError as e:
                    # stable-diffusion-cpp not available
                    sd_cpp_error = str(e)
                    print(f"stable-diffusion-cpp-python not available: {sd_cpp_error}")
                except Exception as e:
                    print(f"sd.cpp generation error: {e}")
                    sd_cpp_error = str(e)
            else:
                # model_path is None after all fallback attempts
                print("Error: Could not resolve sd.cpp model path through any method")
                sd_cpp_error = "Could not resolve model path"
        except ImportError as e:
            sd_cpp_error = str(e)
            print(f"stable-diffusion-cpp-python not available: {sd_cpp_error}")
        except Exception as e:
            sd_cpp_error = str(e)
            print(f"sd.cpp error: {sd_cpp_error}")
    
    # Both backends failed - return error with installation instructions
    raise HTTPException(
        status_code=400,
        detail=f"Model '{model_to_use}' does not support image generation"
    )
