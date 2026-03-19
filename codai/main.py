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
    
    # Import globals from codai modules
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
    )
    
    # Store args globally for access in endpoints
    set_global_args(args)
    
    # Import global setters from text module
    from codai.api.text import (
        set_global_debug,
        set_global_system_prompt,
        set_global_tools_closer_prompt,
    )
    from codai.api.app import set_load_mode
    
    # Set global variables
    global global_system_prompt, global_tools_closer_prompt, global_debug, global_dump, global_file_path, grammar_guided_gen
    
    # Set global grammar-guided-gen flag
    from codai.api.state import set_grammar_guided_gen
    grammar_guided_gen = args.grammar_guided_gen
    if grammar_guided_gen:
        print("Grammar-guided generation enabled (--grammar-guided-gen)")
    
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
    
    # Handle --list-cached-models
    if args.list_cached_models:
        print("\n=== Listing Cached Models ===")
        
        caches = get_all_cache_dirs()
        if not caches:
            print("No model cache directories found.")
            sys.exit(0)
        
        all_files = []
        for cache_name, cache_dir in caches.items():
            print(f"\n--- {cache_name.upper()} Cache ({cache_dir}) ---")
            if not os.path.exists(cache_dir):
                print(f"  (directory does not exist)")
                continue
                
            files = os.listdir(cache_dir)
            if not files:
                print(f"  No cached files.")
                continue
            
            # For diffusers and huggingface, show directory structure
            if cache_name in ('diffusers', 'huggingface'):
                for root, dirs, files in os.walk(cache_dir):
                    for f in files:
                        filepath = os.path.join(root, f)
                        rel_path = os.path.relpath(filepath, cache_dir)
                        size = os.path.getsize(filepath)
                        all_files.append((cache_name, rel_path, size))
            else:
                for f in files:
                    filepath = os.path.join(cache_dir, f)
                    if os.path.isfile(filepath):
                        size = os.path.getsize(filepath)
                        all_files.append((cache_name, f, size))
        
        if not all_files:
            print("\nNo cached models found.")
            sys.exit(0)
        
        # Calculate totals
        total_size = sum(size for _, _, size in all_files)
        
        print(f"\n=== Summary ===")
        print(f"Total: {len(all_files)} files, {total_size / (1024*1024*1024):.2f} GB")
        print("\nCache locations:")
        for cache_name, cache_dir in caches.items():
            print(f"  {cache_name}: {cache_dir}")
        
        sys.exit(0)
    
    # Handle --remove-all-models
    if args.remove_all_models:
        print("\n=== Removing All Cached Models ===")
        
        import shutil
        caches = get_all_cache_dirs()
        
        if not caches:
            print("No cache directories found.")
            sys.exit(0)
        
        total_removed = 0
        for cache_name, cache_dir in caches.items():
            if not os.path.exists(cache_dir):
                continue
                
            files = os.listdir(cache_dir)
            if not files:
                continue
            
            print(f"\nRemoving from {cache_name} cache ({cache_dir})...")
            print(f"  Found {len(files)} file(s). Deleting...")
            
            # For diffusers, remove entire directory tree
            if cache_name == 'diffusers':
                for item in os.listdir(cache_dir):
                    item_path = os.path.join(cache_dir, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
                    print(f"  Deleted: {item}")
                    total_removed += 1
            else:
                for f in files:
                    filepath = os.path.join(cache_dir, f)
                    os.remove(filepath)
                    print(f"  Deleted: {f}")
                    total_removed += 1
        
        print(f"\n=== Removed {total_removed} item(s) from all caches ===")
        sys.exit(0)
    
    # Handle --remove-model
    if args.remove_model:
        print(f"\n=== Removing Cached Model Matching: {args.remove_model} ===")
        
        import shutil
        caches = get_all_cache_dirs()
        
        if not caches:
            print("No cache directories found.")
            sys.exit(0)
        
        all_matching = []
        for cache_name, cache_dir in caches.items():
            if not os.path.exists(cache_dir):
                continue
            
            # For diffusers and huggingface, search recursively
            if cache_name in ('diffusers', 'huggingface'):
                for root, dirs, files in os.walk(cache_dir):
                    for f in files:
                        if args.remove_model.lower() in f.lower():
                            filepath = os.path.join(root, f)
                            rel_path = os.path.relpath(filepath, cache_dir)
                            size = os.path.getsize(filepath)
                            all_matching.append((cache_name, rel_path, filepath, size))
            else:
                files = os.listdir(cache_dir)
                for f in files:
                    if args.remove_model.lower() in f.lower():
                        filepath = os.path.join(cache_dir, f)
                        if os.path.isfile(filepath):
                            size = os.path.getsize(filepath)
                            all_matching.append((cache_name, f, filepath, size))
        
        if not all_matching:
            print(f"No cached models found matching: {args.remove_model}")
            print(f"\nUse --list-cached-models to see available models.")
            sys.exit(0)
        
        print(f"\nFound {len(all_matching)} matching file(s):")
        for cache_name, filename, filepath, size in all_matching:
            print(f"  [{cache_name}] {filename} ({size / (1024*1024):.1f} MB)")
        
        # Confirm before deleting
        print(f"\nDeleting {len(all_matching)} file(s)...")
        for cache_name, filename, filepath, size in all_matching:
            try:
                os.remove(filepath)
                print(f"  Deleted: [{cache_name}] {filename}")
            except Exception as e:
                print(f"  Failed to delete {filename}: {e}")
        
        print(f"\nRemoved {len(all_matching)} cached model file(s).")
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
    load_mode = None
    if args.loadall:
        load_mode = "loadall"
    elif args.loadswap:
        load_mode = "loadswap"
    elif args.nopreload:
        load_mode = "nopreload"
    
    if load_mode:
        set_load_mode(load_mode)
    
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
        print(f"\nLoading main text model(s): {model_names}")
        
        # Register models with multi_model_manager
        for idx, model_name in enumerate(model_names):
            multi_model_manager.set_model(model_name, {
                'ctx': get_ctx_by_index(args.n_ctx, idx, 0),
            })
        
        # Load first model
        try:
            mm = multi_model_manager.get_model_for_request(model_names[0])
            if mm is not None:
                print(f"Model loaded successfully: {model_names[0]}")
            else:
                print(f"Warning: Model {model_names[0]} not loaded (will load on first request)")
        except Exception as e:
            print(f"Warning: Failed to load model: {e}")
            print(f"Model will load on first request")
    
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
