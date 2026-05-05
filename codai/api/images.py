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


def _is_gguf_model(model_name: str) -> bool:
    """Check if a model name/path indicates a GGUF model."""
    if not model_name:
        return False
    return (model_name.endswith('.gguf') or 
            'gguf' in model_name.lower() or
            (model_name.startswith('http') and '.gguf' in model_name))


def _derive_diffusers_device(global_args) -> str:
    """Derive the CUDA device string for diffusers from global args.
    
    Checks --image-vulkan-device then --vulkan-device to determine
    which CUDA device to target. Defaults to 'cuda:0'.
    """
    if global_args:
        # Check image-specific device first
        image_device = getattr(global_args, 'image_vulkan_device', None)
        if image_device is not None:
            return f"cuda:{image_device}"
        # Fall back to general device
        device_id = getattr(global_args, 'vulkan_device', 0)
        if device_id is not None and device_id != 0:
            return f"cuda:{device_id}"
    return "cuda:0"


def _load_diffusers_pipeline(model_name: str, global_args):
    """
    Try to load a model using the diffusers library.
    
    Returns the loaded pipeline or None if diffusers can't handle this model.
    Raises Exception if loading fails for other reasons.
    """
    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, DiffusionPipeline
    import torch
    
    # Check for --no-ram mode
    no_ram = getattr(global_args, 'no_ram', False) if global_args else False
    
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
    
    # --no-ram mode: override dtype to auto-select best for GPU
    if no_ram:
        dtype = torch.float16  # Use fp16 to maximize VRAM efficiency
        print(f"--no-ram mode: Using precision fp16 for maximum VRAM efficiency")
    else:
        print(f"Using precision: {precision} ({dtype})")
    
    # Check if CPU offload is requested via CLI
    use_sequential_offload = getattr(global_args, 'image_cpu_offload', False)
    
    # --no-ram mode: never use CPU offload
    if no_ram and use_sequential_offload:
        print("--no-ram mode: ignoring --image-cpu-offload, forcing full GPU loading")
        use_sequential_offload = False
    
    # =====================================================================
    # --no-ram mode: load directly on GPU, no CPU RAM fallback
    # =====================================================================
    if no_ram and torch.cuda.is_available():
        cuda_device = _derive_diffusers_device(global_args)
        print(f"--no-ram mode: loading diffusers model directly on {cuda_device}")
        
        try:
            try:
                pipeline = StableDiffusionXLPipeline.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    use_safetensors=True,
                )
            except Exception:
                pipeline = DiffusionPipeline.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    use_safetensors=True,
                )
            
            pipeline = pipeline.to(cuda_device)
            print(f"--no-ram: Diffusers model loaded on {cuda_device}")
            return pipeline
        except Exception as e:
            raise RuntimeError(
                f"--no-ram: Failed to load diffusers model entirely on GPU ({cuda_device}). "
                f"The model may be too large for available VRAM. Error: {e}"
            )
    
    # =====================================================================
    # Standard loading path (with OOM fallback)
    # =====================================================================
    # Track loading attempts for OOM handling
    pipeline = None
    load_attempt = 0
    max_attempts = 3
    
    while pipeline is None and load_attempt < max_attempts:
        try:
            load_attempt += 1
            print(f"Loading attempt {load_attempt}/{max_attempts}...")
            
            # Try to load as Stable Diffusion XL first, then generic DiffusionPipeline
            try:
                pipeline = StableDiffusionXLPipeline.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    use_safetensors=True,
                )
            except Exception:
                # Try generic diffusion pipeline (supports custom pipelines like ZImagePipeline)
                pipeline = DiffusionPipeline.from_pretrained(
                    model_name,
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
    
    return pipeline


def _generate_with_diffusers(pipeline, request, global_args, http_request=None):
    """Generate images using a diffusers pipeline."""
    import torch
    import numpy as np
    import time as time_module
    
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
    if width != width or width == float('inf'):
        width = 512
    if height != height or height == float('inf'):
        height = 512
    
    # Enable memory optimizations
    try:
        if hasattr(pipeline, 'enable_attention_slicing'):
            pipeline.enable_attention_slicing(slice_size="auto")
        if hasattr(pipeline, 'enable_vae_slicing'):
            pipeline.enable_vae_slicing()
    except Exception as e:
        print(f"Warning: Could not enable memory optimizations: {e}")
    
    # Get timestamp BEFORE calling diffusers
    timestamp = int(time_module.time())
    
    # Generate images
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
        result_images = getattr(result, 'image', None) or getattr(result, 'output', None)
        if result_images is None:
            raise Exception(f"Could not extract images from diffusers result: {img_err}")
    
    for img in result_images:
        # Debug: print image type and value range
        print(f"DEBUG: Image type: {type(img)}")
        if isinstance(img, np.ndarray):
            print(f"DEBUG: Image shape: {img.shape}, dtype: {img.dtype}, min: {img.min()}, max: {img.max()}")
            img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
            img = np.clip(img, 0.0, 1.0)
            print(f"DEBUG: After NaN handling - min: {img.min()}, max: {img.max()}")
        
        img_data = save_image_response(img, request.response_format, http_request)
        images.append(img_data)
    
    return {
        "created": timestamp,
        "data": images
    }


async def _generate_with_sdcpp(sd_model, request, global_args, http_request=None):
    """Generate images using stable-diffusion-cpp-python."""
    import time
    
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
    
    # Use default steps for fast generation
    steps = 4
    
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
    time.sleep(0.1)
    
    # Convert results to response format
    images = []
    for img in result:
        img_data = save_image_response(img, http_request=http_request)
        images.append(img_data)
    
    return {
        "created": int(time.time()),
        "data": images
    }


def _load_sdcpp_model(model_path: str, global_args, model_config: dict = None):
    """
    Try to load a model using stable-diffusion-cpp-python.
    
    Returns the loaded StableDiffusion model or None.
    """
    from stable_diffusion_cpp import StableDiffusion

    # Check for --no-ram mode
    no_ram = getattr(global_args, 'no_ram', False) if global_args else False

    print(f"Loading sd.cpp model from: {model_path}")

    # Build sd.cpp constructor args from config
    kwargs = {
        'model_path': model_path,
        'offload_params_to_cpu': False,  # Use GPU by default
        'keep_clip_on_cpu': False,
        'keep_control_net_on_cpu': False,
        'keep_vae_on_cpu': False,
    }

    # Add optional paths from CLI args
    if global_args:
        if hasattr(global_args, 'vae_path') and global_args.vae_path:
            kwargs['vae_path'] = global_args.vae_path
        if hasattr(global_args, 'llm_path') and global_args.llm_path:
            kwargs['lora_model_dir'] = global_args.llm_path

    # If backend is explicitly cpu, offload to CPU
    backend = (model_config or {}).get('backend', 'auto') if model_config else 'auto'
    if backend == 'cpu':
        kwargs['offload_params_to_cpu'] = True
        kwargs['keep_clip_on_cpu'] = True
        kwargs['keep_vae_on_cpu'] = True

    if no_ram:
        print("--no-ram mode: sd.cpp maximizing GPU usage (no CPU offload for CLIP/VAE/ControlNet)")

    try:
        sd_model = StableDiffusion(**kwargs)
    except Exception as e:
        if 'cpu' not in str(backend) and ('memory' in str(e).lower() or 'cuda' in str(e).lower() or 'out of' in str(e).lower()):
            print(f"GPU load failed ({e}), retrying with CPU offload...")
            kwargs['offload_params_to_cpu'] = True
            kwargs['keep_clip_on_cpu'] = True
            kwargs['keep_vae_on_cpu'] = True
            sd_model = StableDiffusion(**kwargs)
        else:
            raise
    return sd_model


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
        # =====================================================================
        # Step 1: Ask the manager to resolve the model and manage VRAM
        # =====================================================================
        model_info = multi_model_manager.request_model(
            requested_model=request.model,
            model_type="image"
        )
        
        # Check if the model was rejected as not allowed
        if model_info.get('error'):
            raise HTTPException(status_code=404, detail=model_info['error'])
        
        model_name = model_info['model_name']
        model_key = model_info['model_key']
        pipeline = model_info['model_object']
        
        # If no image model configured, try to use main --model as fallback
        if not model_name:
            main_model = getattr(global_args, 'model', None)
            if main_model and isinstance(main_model, list) and len(main_model) > 0:
                model_name = main_model[0]
            elif main_model:
                model_name = main_model
            
            # Check if main model is a GGUF file - can't use for image generation
            if model_name and _is_gguf_model(model_name):
                print(f"Note: Main model is a GGUF file (for text), not suitable for image generation")
                model_name = None
            
            if model_name:
                model_key = f"image:{model_name}"
        
        # If still no image model configured, return an error
        if not model_name:
            raise HTTPException(
                status_code=400,
                detail="Image generation not configured. Use --image-model to specify a model."
            )
        
        # =====================================================================
        # Step 2: Check if model is a sd.cpp StableDiffusion instance
        # =====================================================================
        is_sdcpp = False
        if pipeline is not None:
            try:
                from stable_diffusion_cpp import StableDiffusion
                if isinstance(pipeline, StableDiffusion):
                    is_sdcpp = True
            except ImportError:
                pass
        
        # =====================================================================
        # Step 3: If already loaded, generate with appropriate backend
        # =====================================================================
        if pipeline is not None:
            if is_sdcpp:
                print(f"Using cached sd.cpp model for generation")
                return await _generate_with_sdcpp(pipeline, request, global_args, http_request)
            else:
                # Assume it's a diffusers pipeline
                print(f"Using cached diffusers pipeline for generation")
                return _generate_with_diffusers(pipeline, request, global_args, http_request)
        
        # =====================================================================
        # Step 4: Model not loaded - try to load it
        # =====================================================================
        is_gguf = _is_gguf_model(model_name)
        diffusers_error = None
        sdcpp_error = None
        
        # Try diffusers first (for non-GGUF models)
        if not is_gguf:
            try:
                print(f"Loading diffusers model: {model_name}")
                pipeline = _load_diffusers_pipeline(model_name, global_args)
                
                if pipeline is not None:
                    # Cache the loaded pipeline in the manager
                    multi_model_manager.add_model(model_key, pipeline)
                    multi_model_manager.current_model_key = model_key
                    print(f"Loaded diffusers model: {model_name}")
                    
                    return _generate_with_diffusers(pipeline, request, global_args, http_request)
                    
            except ImportError as e:
                diffusers_error = str(e)
                print(f"diffusers not available: {diffusers_error}")
            except Exception as e:
                import traceback
                diffusers_error = str(e)
                print(f"diffusers error: {diffusers_error}")
                print(f"Traceback: {traceback.format_exc()}")
        
        # Try stable-diffusion-cpp-python (for GGUF models or as fallback)
        try:
            # For GGUF models or URLs, resolve the model path through the cache
            resolved_path = model_name
            if is_gguf or model_name.startswith('http://') or model_name.startswith('https://'):
                resolved_path = multi_model_manager.load_model(model_name)
                if not resolved_path:
                    raise Exception(f"Failed to resolve model path: {model_name}")
            
            # Only use sd.cpp if we have a local file path
            if resolved_path and os.path.isfile(resolved_path):
                cfg = multi_model_manager.config.get(model_key) or multi_model_manager.config.get(model_name) or {}
                sd_model = _load_sdcpp_model(resolved_path, global_args, model_config=cfg)
                
                if sd_model is not None:
                    # Cache the loaded model in the manager
                    multi_model_manager.add_model(model_key, sd_model)
                    multi_model_manager.current_model_key = model_key
                    print(f"Loaded sd.cpp model: {model_name}")
                    
                    return await _generate_with_sdcpp(sd_model, request, global_args, http_request)
            else:
                sdcpp_error = f"Model '{model_name}' is not a local file, cannot use sd.cpp"
                print(sdcpp_error)
                
        except ImportError as e:
            sdcpp_error = str(e)
            print(f"stable-diffusion-cpp-python not available: {sdcpp_error}")
        except Exception as e:
            sdcpp_error = str(e)
            print(f"sd.cpp error: {sdcpp_error}")
        
        # =====================================================================
        # Step 5: Both backends failed - return error
        # =====================================================================
        error_details = []
        if diffusers_error:
            error_details.append(f"diffusers: {diffusers_error}")
        if sdcpp_error:
            error_details.append(f"sd.cpp: {sdcpp_error}")
        
        raise HTTPException(
            status_code=400,
            detail=f"Failed to load image model '{model_name}'. Errors: {'; '.join(error_details) if error_details else 'No compatible backend found'}"
        )
