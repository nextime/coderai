"""Main entry point for codai server."""
import sys
import os

# Import configuration from codai modules
from codai.cli import parse_args


def main():
    """Main entry point for the codai server."""
    # Suppress unraisable exceptions from LlamaModel.__del__
    original_unraisablehook = sys.unraisablehook
    def suppress_llama_del_errors(unraisable):
        if isinstance(unraisable.exc_value, AttributeError) and 'LlamaModel' in repr(unraisable.object) and 'sampler' in str(unraisable.exc_value):
            return  # Ignore this specific error
        original_unraisablehook(unraisable)
    sys.unraisablehook = suppress_llama_del_errors
    
    # Optional: set process name if procname is available
    try:
        import procname
        procname.setprocname("codai")
    except ImportError:
        pass
    
    args = parse_args()

    # Handle early exit options (before heavy imports)
    if args.list_cached_models:
        print("\n=== Listing Cached Models ===")

        # Import only what's needed for cache listing
        from codai.models.cache import list_cached_models_info, get_all_cache_dirs

        cache_info = list_cached_models_info()
        caches = get_all_cache_dirs()

        # Show CoderAI GGUF cache
        coderai_dir = caches.get('coderai')
        if coderai_dir:
            print(f"\n--- CODERAI GGUF Cache ({coderai_dir}) ---")
            if cache_info['coderai']:
                for filename, size_mb in cache_info['coderai']:
                    print(f"  {filename} ({size_mb:.1f} MB)")
            else:
                print("  No cached GGUF files.")
        else:
            print(f"\n--- CODERAI GGUF Cache ---")
            print("  (directory not found)")

        # Show HuggingFace cached models
        hf_dir = caches.get('huggingface')
        if hf_dir:
            print(f"\n--- HUGGINGFACE Models Cache ({hf_dir}) ---")
            if cache_info['huggingface']:
                for repo_id, size_gb, revision_count in cache_info['huggingface']:
                    print(f"  {repo_id} ({size_gb:.2f} GB)")
                    print(f"    └─ {revision_count} revision(s)")
            else:
                print("  No cached HuggingFace models.")
        else:
            print(f"\n--- HUGGINGFACE Models Cache ---")
            print("  (directory not found)")

        # Show summary
        print(f"\n=== Summary ===")
        print(f"Total cached models: {cache_info['total_models']}")
        print(f"Total disk usage: {cache_info['total_size_gb']:.2f} GB")
        print("\nCache locations:")
        for cache_name, cache_dir in caches.items():
            print(f"  {cache_name}: {cache_dir}")

        sys.exit(0)

    # Handle --remove-all-models early
    if args.remove_all_models:
        print("\n=== Removing All Cached Models ===")

        from codai.models.cache import remove_all_cached_models

        total_removed = remove_all_cached_models()

        print(f"\n=== Removed {total_removed} item(s) from all caches ===")
        sys.exit(0)

    # Handle --remove-model early
    if args.remove_model:
        print(f"\n=== Removing Cached Model Matching: {args.remove_model} ===")

        from codai.models.cache import remove_cached_model

        removed = remove_cached_model(args.remove_model)

        if not removed:
            print(f"No cached models found matching: {args.remove_model}")
            print(f"\nUse --list-cached-models to see available models.")
            sys.exit(0)

        total_size = sum(size for _, _, size in removed)
        print(f"\nRemoved {len(removed)} cached model file(s), freeing {total_size / (1024*1024):.1f} MB")
        sys.exit(0)

    # Handle --download-model early (before heavy imports)
    if args.download_model:
        print(f"\n=== Downloading Model: {args.download_model} ===")

        from codai.models.cache import download_model, is_huggingface_model_id

        model_id = args.download_model
        file_pattern = args.download_file_pattern
        
        try:
            # For HuggingFace model IDs, try pattern-based download first
            if is_huggingface_model_id(model_id):
                # If file pattern specified, use it
                if file_pattern:
                    print(f"File pattern: {file_pattern}")
                    cached_path = download_model(model_id, file_pattern=file_pattern)
                else:
                    # Try common patterns: GGUF first, then full repo download
                    print("Trying GGUF download first...")
                    cached_path = download_model(model_id, file_pattern='.gguf')
                    
                    if not cached_path:
                        # No GGUF files - download entire repo (for transformers/diffusers models)
                        print("No GGUF files found, downloading full HuggingFace repo...")
                        try:
                            from huggingface_hub import snapshot_download
                            cached_path = snapshot_download(model_id)
                        except Exception as e:
                            print(f"Error downloading full repo: {e}")
                            cached_path = None
            else:
                # URL or local path
                cached_path = download_model(model_id, file_pattern=file_pattern or '.gguf')

            if cached_path:
                print(f"\n=== Model downloaded successfully ===")
                print(f"Cached at: {cached_path}")
                sys.exit(0)
            else:
                print(f"\n=== Failed to download model ===")
                sys.exit(1)
        except Exception as e:
            print(f"\n=== Error downloading model: {e} ===")
            sys.exit(1)

    # Import globals from codai modules (only after early exits)
    from codai.api import app
    from codai.api.state import (
        set_global_args,
        set_global_debug,
        set_global_system_prompt,
        set_global_tools_closer_prompt,
        set_global_file_path,
        set_load_mode,
        set_grammar_guided_gen,
    )
    from codai.models.manager import ModelManager, MultiModelManager, model_manager, multi_model_manager
    from codai.backends import detect_available_backends
    from codai.models.cache import (
        get_all_cache_dirs,
        get_cached_model_path,
        get_model_cache_dir,
        download_model,
        list_cached_models_info,
    )

    # Import global setters from text module FIRST (before calling them)
    from codai.api.text import (
        set_global_args,
        set_global_debug,
        set_global_system_prompt,
        set_global_tools_closer_prompt,
    )
    from codai.api.app import set_load_mode
    
    # Store args globally for access in endpoints (both state and text.py)
    set_global_args(args)
    
    # Set global variables
    global global_system_prompt, global_tools_closer_prompt, global_debug, global_dump, global_file_path, grammar_guided_gen
    
    # Set global grammar-guided-gen flag
    from codai.api.state import set_grammar_guided_gen
    grammar_guided_gen = args.grammar_guided_gen
    if grammar_guided_gen:
        print("Grammar-guided generation enabled (--grammar-guided-gen)")
    
    # Print --no-ram mode status
    if args.no_ram:
        print("No-RAM mode enabled (--no-ram): maximizing VRAM usage, no CPU RAM spilling")
        print("  llama-cpp-python: n_gpu_layers=-1, use_mmap=False, --n-ctx ignored")
        print("  HuggingFace: device_map=cuda, low_cpu_mem_usage=True, torch_dtype=auto")
        print("  Diffusers: forced full GPU loading")
        print("  sd.cpp: maximizing GPU offload")
    
    # Set global system prompt from --system-prompt flag
    global_system_prompt = args.system_prompt
    set_global_system_prompt(global_system_prompt)
    
    # Set global tools-closer-prompt flag
    global_tools_closer_prompt = args.tools_closer_prompt
    set_global_tools_closer_prompt(global_tools_closer_prompt)
    if global_tools_closer_prompt:
        print("Tools closer prompt enabled (--tools-closer-prompt)")
    
    # Set global debug flag
    global_debug = args.debug
    set_global_debug(global_debug)
    
    # Set global dump flag (enables debug as well for litellm output)
    global_dump = args.dump
    if global_dump:
        global_debug = True
        set_global_debug(True)
    
    # Set global file path for storing generated files
    global_file_path = args.file_path
    set_global_file_path(global_file_path)
    
    # Also set file path for images module
    from codai.api.images import set_global_file_path as set_images_file_path
    set_images_file_path(global_file_path)
    
    # Also set global args for images module (it has its own global_args)
    from codai.api.images import set_global_args as set_images_global_args
    set_images_global_args(args)
    
    # Also set file path for app.py (needed for /v1/files endpoint)
    from codai.api.app import set_global_file_path_wrapper
    set_global_file_path_wrapper(global_file_path)
    
    if global_debug:
        # Print the full command line that was used to invoke codai
        import shlex
        cmd_line = ' '.join(shlex.quote(arg) for arg in sys.argv)
        print(f"\n{'='*80}")
        print(f"=== COMMAND LINE: {cmd_line}")
        print(f"{'='*80}\n")
        print("DEBUG MODE ENABLED - Full requests and replies will be dumped to stdout")
    
    # Handle --vulkan-list-devices
    if args.vulkan_list_devices:
        print("\nListing Vulkan devices...")
        try:
            import subprocess
            result = subprocess.run(['vulkaninfo', '--summary'], capture_output=True, text=True)
            if result.returncode == 0:
                print(result.stdout)
            else:
                print("Could not run vulkaninfo. Make sure vulkan-tools is installed.")
        except Exception as e:
            print(f"Error listing devices: {e}")
        sys.exit(0)
    
    # Get model names from args - support multiple models
    model_names = args.model if args.model else []
    
    # Helper function to get config value by index with fallback
    def get_ctx_by_index(ctx_list, index, default):
        """Get context value by model index, with fallback to default."""
        if ctx_list and index < len(ctx_list):
            return ctx_list[index]
        return default
    
    # Validate: must have at least one model specified
    audio_models = args.audio_model if args.audio_model else []
    image_models = args.image_model if args.image_model else []
    vision_models = args.vision_model if args.vision_model else []
    
    if not model_names and not audio_models and not image_models and not vision_models and args.tts_model is None:
        print("Error: At least one of --model, --audio-model, --image-model, --vision-model, or --tts-model must be specified.")
        print("")
        print("For NVIDIA backend (HuggingFace models):")
        print("  - microsoft/DialoGPT-medium")
        print("  - meta-llama/Llama-2-7b-chat-hf (requires auth)")
        print("  - TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        print("  - Use multiple --model flags for multiple models")
        print("")
        print("For Vulkan backend (GGUF models):")
        print("  - Local path: ./phi-3-mini-4k-instruct-q4_k_m.gguf")
        print("  - Or a HuggingFace model ID: TheBloke/Mistral-7B-Instruct-v0.2-GGUF")
        print("  - Use multiple --model flags for multiple models")
        print("")
        sys.exit(1)
    
    # Determine load mode
    # Default is ondemand: pre-load only the first model, unload/load on switch
    # --loadswap: load first in VRAM, others in CPU RAM, swap on switch
    # --loadall: try to load all models in VRAM, offload to CPU RAM if fails
    # --nopreload: skip pre-loading in any mode, load on first request
    load_mode = "ondemand"  # Default: on-demand loading
    if args.loadall:
        load_mode = "loadall"
    elif args.loadswap:
        load_mode = "loadswap"
    
    nopreload = args.nopreload
    
    set_load_mode(load_mode)
    multi_model_manager.set_load_mode(load_mode)
    
    if load_mode == "ondemand":
        print("Load mode: ondemand (pre-load first model, unload/load on switch)")
    elif load_mode == "loadswap":
        print("Load mode: loadswap (first model in VRAM, others in CPU RAM, swap on switch)")
    elif load_mode == "loadall":
        print("Load mode: loadall (load all models, offload to CPU RAM if VRAM full)")
    if nopreload:
        print("  --nopreload: models will load on first request instead of at startup")
    
    # Initialize model manager
    print("\n=== Initializing Model Manager ===")
    
    # Detect available backends
    available_backends = detect_available_backends()
    print(f"Available backends: {available_backends}")
    
    # Determine which backend to use
    backend = args.backend
    if backend == "auto":
        if "nvidia" in available_backends:
            backend = "nvidia"
        elif "vulkan" in available_backends:
            backend = "vulkan"
        elif "opencl" in available_backends:
            backend = "opencl"
        else:
            print("Error: No supported backend detected (NVIDIA CUDA, AMD Vulkan, or OpenCL)")
            sys.exit(1)
    
    print(f"Using backend: {backend}")
    
    # Set the backend for the model manager
    model_manager.backend_type = backend
    
    # Store references globally for API endpoints
    from codai.api import app as fastapi_app
    fastapi_app.state.model_manager = model_manager
    fastapi_app.state.multi_model_manager = multi_model_manager
    
    # Load main text model(s)
    if model_names:
        print(f"\nMain text model(s): {model_names}")
        
        # Register models with multi_model_manager (set_default_model also resolves/caches)
        for idx, model_name in enumerate(model_names):
            multi_model_manager.set_default_model(model_name, {
                'ctx': get_ctx_by_index(args.n_ctx, idx, 0),
            })
        
        # Pre-load models at startup (unless --nopreload)
        if nopreload:
            print(f"  --nopreload: text model(s) will load on first request")
        elif load_mode == "ondemand":
            # Ondemand: pre-load only the first model into VRAM
            try:
                print(f"Preloading first model into VRAM: {model_names[0]}...")
                mm = multi_model_manager._load_default_model()
                if mm is not None and mm.backend is not None:
                    multi_model_manager.active_in_vram = multi_model_manager.default_model
                    print(f"Model loaded successfully: {model_names[0]}")
                else:
                    print(f"Warning: Model {model_names[0]} failed to load")
            except Exception as e:
                print(f"Warning: Failed to preload model: {e}")
                print(f"Model will load on first request")
        elif load_mode == "loadswap":
            # Loadswap: load first model into VRAM, others into CPU RAM
            try:
                print(f"Preloading first model into VRAM: {model_names[0]}...")
                mm = multi_model_manager._load_default_model()
                if mm is not None and mm.backend is not None:
                    multi_model_manager.active_in_vram = multi_model_manager.default_model
                    print(f"Model loaded successfully (VRAM): {model_names[0]}")
                else:
                    print(f"Warning: Model {model_names[0]} failed to load")
            except Exception as e:
                print(f"Warning: Failed to preload model: {e}")
            
            # Load remaining text models into CPU RAM
            for idx, model_name in enumerate(model_names[1:], 1):
                try:
                    print(f"Preloading model into CPU RAM: {model_name}...")
                    mm2 = multi_model_manager._load_model_by_name(model_name)
                    if mm2 is not None:
                        # Move to CPU immediately (it was loaded into VRAM by default)
                        multi_model_manager._move_model_to_cpu(model_name)
                        print(f"Model loaded successfully (CPU RAM): {model_name}")
                    else:
                        print(f"Warning: Model {model_name} failed to load")
                except Exception as e:
                    print(f"Warning: Failed to preload model {model_name}: {e}")
        elif load_mode == "loadall":
            # Loadall: try to load all models into VRAM, offload to CPU RAM if fails
            for idx, model_name in enumerate(model_names):
                try:
                    if idx == 0:
                        print(f"Preloading model into VRAM: {model_name}...")
                        mm = multi_model_manager._load_default_model()
                    else:
                        print(f"Preloading model into VRAM: {model_name}...")
                        mm = multi_model_manager._load_model_by_name(model_name)
                    
                    if mm is not None and (not hasattr(mm, 'backend') or mm.backend is not None):
                        if idx == 0:
                            multi_model_manager.active_in_vram = multi_model_manager.default_model
                        print(f"Model loaded successfully (VRAM): {model_name}")
                    else:
                        print(f"Warning: Model {model_name} failed to load")
                except Exception as e:
                    error_msg = str(e).lower()
                    is_oom = any(x in error_msg for x in ['out of memory', 'oom', 'cuda error'])
                    if is_oom:
                        print(f"VRAM full for {model_name}, offloading to CPU RAM...")
                        try:
                            mm = multi_model_manager._load_model_by_name(model_name)
                            if mm is not None:
                                multi_model_manager._move_model_to_cpu(model_name)
                                print(f"Model loaded successfully (CPU RAM): {model_name}")
                        except Exception as e2:
                            print(f"Warning: Failed to load model {model_name} even to CPU: {e2}")
                    else:
                        print(f"Warning: Failed to preload model {model_name}: {e}")
    
    # Set up audio model if specified
    if audio_models:
        print(f"\nAudio transcription model(s): {audio_models}")
        
        for idx, audio_m in enumerate(audio_models):
            multi_model_manager.set_audio_model(audio_m, {
                'ctx': get_ctx_by_index(args.audio_ctx, idx, 0),
                'offload': args.audio_offload,
            })
    
    # Set up whisper-server if specified
    if args.whisper_server:
        print(f"\nWhisper server: {args.whisper_server}")
        print(f"  Port: {args.whisper_server_port}")
        
        # Import WhisperServerManager
        from codai.models.manager import WhisperServerManager
        
        # Check if whisper-server is already running
        if multi_model_manager.whisper_server is None:
            whisper_server_mgr = WhisperServerManager(
                server_path=args.whisper_server,
                port=args.whisper_server_port
            )
            multi_model_manager.whisper_server = whisper_server_mgr
        else:
            whisper_server_mgr = multi_model_manager.whisper_server
            print("Whisper server already running, using existing instance")
        
        # Start whisper-server if we have audio_models configured
        if audio_models:
            model_to_use = audio_models[0] if audio_models else None
            gpu_device = getattr(args, 'audio_vulkan_device', 0) or 0
            print(f"DEBUG: Starting whisper-server with gpu_device={gpu_device}")
            actual_model_path = whisper_server_mgr.start(model_path=model_to_use, gpu_device=gpu_device)
            if actual_model_path:
                # Update audio_models in multi_model_manager to store the actual path (not the URL)
                if model_to_use != actual_model_path:
                    if multi_model_manager.audio_models and multi_model_manager.audio_models[0] == model_to_use:
                        multi_model_manager.audio_models[0] = actual_model_path
                print(f"Whisper server started with model: {actual_model_path}")
            else:
                print("Warning: Failed to start whisper-server, falling back to other backends")
    
    # Set up image model if specified
    if image_models:
        print(f"\nImage generation model(s): {image_models}")
        
        for idx, img_m in enumerate(image_models):
            multi_model_manager.set_image_model(img_m, {
                'ctx': get_ctx_by_index(args.image_ctx, idx, 0),
                'offload': args.image_offload,
                'llm_path': args.llm_path,
                'vae_path': args.vae_path,
                'sample_method': args.image_sample_method,
                'steps': args.image_steps,
                'width': args.image_width,
                'height': args.image_height,
                'cfg_scale': args.image_cfg_scale,
            })
    
    # Set up vision model if specified
    if vision_models:
        print(f"\nVision model(s): {vision_models}")
        
        for idx, vision_m in enumerate(vision_models):
            multi_model_manager.set_vision_model(vision_m, {
                'ctx': get_ctx_by_index(args.n_ctx, idx, 0),
                'offload': args.image_offload,
            })
    
    # Set up TTS model if specified
    if args.tts_model:
        print(f"\nText-to-speech model: {args.tts_model}")
        multi_model_manager.set_tts_model(args.tts_model, {})
    
    # Register model aliases if specified
    if args.model_aliases:
        print(f"\nRegistering model aliases:")
        for alias, model in args.model_aliases:
            multi_model_manager.set_model_alias(alias, model)
            print(f"  {alias} -> {model}")
    
    # =========================================================================
    # Pre-load non-text models for loadall and loadswap modes
    # (Text models are already handled above)
    # =========================================================================
    if not nopreload and load_mode in ("loadall", "loadswap"):
        # Collect all non-text models that need pre-loading
        # For loadall: load all into VRAM (offload to CPU if OOM)
        # For loadswap: first model in VRAM (already done for text), rest in CPU RAM
        
        # Determine if the first text model is already in VRAM
        first_model_loaded = multi_model_manager.active_in_vram is not None
        
        # Pre-load image models
        if image_models:
            print(f"\n=== Pre-loading image model(s) ===")
            for idx, img_m in enumerate(image_models):
                model_key = f"image:{img_m}"
                if model_key in multi_model_manager.models:
                    continue  # Already loaded
                
                try:
                    from codai.api.images import _load_diffusers_pipeline, _is_gguf_model, _load_sdcpp_model
                    
                    if load_mode == "loadall":
                        # Try to load into VRAM
                        print(f"Preloading image model into VRAM: {img_m}...")
                        if _is_gguf_model(img_m):
                            resolved_path = multi_model_manager.load_model(img_m)
                            if resolved_path and os.path.isfile(resolved_path):
                                sd_model = _load_sdcpp_model(resolved_path, args)
                                if sd_model:
                                    multi_model_manager.add_model(model_key, sd_model)
                                    print(f"Image model loaded (VRAM, sd.cpp): {img_m}")
                        else:
                            try:
                                pipeline = _load_diffusers_pipeline(img_m, args)
                                if pipeline:
                                    multi_model_manager.add_model(model_key, pipeline)
                                    print(f"Image model loaded (VRAM, diffusers): {img_m}")
                            except Exception as e:
                                error_msg = str(e).lower()
                                is_oom = any(x in error_msg for x in ['out of memory', 'oom', 'cuda error'])
                                if is_oom:
                                    print(f"VRAM full for image model {img_m}, will load on demand")
                                else:
                                    print(f"Warning: Failed to preload image model {img_m}: {e}")
                    
                    elif load_mode == "loadswap":
                        # Load into VRAM then move to CPU (unless it's the first model overall)
                        if not first_model_loaded:
                            # No model in VRAM yet, load this one into VRAM
                            print(f"Preloading image model into VRAM: {img_m}...")
                            if _is_gguf_model(img_m):
                                resolved_path = multi_model_manager.load_model(img_m)
                                if resolved_path and os.path.isfile(resolved_path):
                                    sd_model = _load_sdcpp_model(resolved_path, args)
                                    if sd_model:
                                        multi_model_manager.add_model(model_key, sd_model)
                                        first_model_loaded = True
                                        print(f"Image model loaded (VRAM): {img_m}")
                            else:
                                try:
                                    pipeline = _load_diffusers_pipeline(img_m, args)
                                    if pipeline:
                                        multi_model_manager.add_model(model_key, pipeline)
                                        first_model_loaded = True
                                        print(f"Image model loaded (VRAM): {img_m}")
                                except Exception as e:
                                    print(f"Warning: Failed to preload image model {img_m}: {e}")
                        else:
                            # First model already in VRAM, load this to VRAM then move to CPU
                            print(f"Preloading image model into CPU RAM: {img_m}...")
                            # Move current VRAM model to CPU temporarily
                            current_vram = multi_model_manager.active_in_vram
                            if current_vram and current_vram in multi_model_manager.models:
                                multi_model_manager._move_model_to_cpu(current_vram)
                            
                            try:
                                if _is_gguf_model(img_m):
                                    resolved_path = multi_model_manager.load_model(img_m)
                                    if resolved_path and os.path.isfile(resolved_path):
                                        sd_model = _load_sdcpp_model(resolved_path, args)
                                        if sd_model:
                                            multi_model_manager.add_model(model_key, sd_model)
                                            multi_model_manager._move_model_to_cpu(model_key)
                                            print(f"Image model loaded (CPU RAM): {img_m}")
                                else:
                                    pipeline = _load_diffusers_pipeline(img_m, args)
                                    if pipeline:
                                        multi_model_manager.add_model(model_key, pipeline)
                                        multi_model_manager._move_model_to_cpu(model_key)
                                        print(f"Image model loaded (CPU RAM): {img_m}")
                            except Exception as e:
                                print(f"Warning: Failed to preload image model {img_m}: {e}")
                            
                            # Move original model back to VRAM
                            if current_vram and current_vram in multi_model_manager.models:
                                multi_model_manager._move_model_to_vram(current_vram)
                                multi_model_manager.active_in_vram = current_vram
                
                except ImportError as e:
                    print(f"Warning: Cannot preload image model {img_m} (missing dependency): {e}")
                except Exception as e:
                    print(f"Warning: Failed to preload image model {img_m}: {e}")
        
        # Note: Audio models (faster-whisper) and TTS models (kokoro) are loaded
        # by their respective API modules on first request, as they use specialized
        # loading mechanisms. The model files are already cached by set_audio_model()
        # and set_tts_model() above.
        if audio_models:
            print(f"\nAudio model(s) registered and cached, will load into memory on first request")
        if args.tts_model:
            print(f"TTS model registered and cached, will load into memory on first request")
    
    # Start the server
    import uvicorn
    print(f"\nStarting server on http://{args.host}:{args.port}")
    print(f"API documentation available at http://{args.host}:{args.port}/docs")
    
    if model_manager.backend is not None:
        actual_backend = model_manager.backend_type
        if hasattr(model_manager.backend, 'force_cuda') and model_manager.backend.force_cuda:
            actual_backend = "cuda (via llama-cpp-python)"
        print(f"Using backend: {actual_backend}")
    
    # Print available models
    models = multi_model_manager.list_models()
    print(f"Available models: {[m.id for m in models]}")
    
    # Run server with or without HTTPS
    if args.https:
        import ssl
        
        ssl_keyfile = None
        ssl_certfile = None
        
        if args.privkey and args.pubkey:
            ssl_keyfile = args.privkey
            ssl_certfile = args.pubkey
            print(f"Using HTTPS with custom certificates: {args.pubkey}")
        else:
            print("Generating self-signed HTTPS certificate...")
            import subprocess
            try:
                cert_path = "./cert.pem"
                key_path = "./key.pem"
                subprocess.run([
                    "openssl", "req", "-x509", "-newkey", "rsa:4096",
                    "-keyout", key_path, "-out", cert_path,
                    "-days", "365", "-nodes",
                    "-subj", "/CN=localhost"
                ], check=True, capture_output=True)
                ssl_keyfile = key_path
                ssl_certfile = cert_path
                print(f"Generated self-signed certificate: {cert_path}")
            except Exception as e:
                print(f"Warning: Could not generate certificate: {e}")
                print("Falling back to HTTP...")
                uvicorn.run(app, host=args.host, port=args.port)
                return
        
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(ssl_certfile, ssl_keyfile)
        uvicorn.run(app, host=args.host, port=args.port, ssl=ssl_context)
    else:
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
