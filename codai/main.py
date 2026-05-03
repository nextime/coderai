"""Main entry point for codai server."""
import sys
import os

# Import configuration from codai modules
from codai.cli import parse_args
from codai.config import ConfigManager
from codai.admin.routes import init_session_manager


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
    
    # Initialize ConfigManager
    config_dir = args.config
    config_mgr = ConfigManager(config_dir)
    config = config_mgr.load()
    
    # Initialize admin session manager
    from pathlib import Path
    init_session_manager(Path(config_dir))
    
    # Handle early exit options (before heavy imports)
    if args.list_cached_models:
        print("\n=== Listing Cached Models ===")
        from codai.models.cache import list_cached_models_info, get_all_cache_dirs
        cache_info = list_cached_models_info()
        caches = get_all_cache_dirs()
        coderai_dir = caches.get('coderai')
        if coderai_dir:
            print(f"\n--- CODERAI GGUF Cache ({coderai_dir}) ---")
            if cache_info['coderai']:
                for filename, size_mb in cache_info['coderai']:
                    print(f"  {filename} ({size_mb:.1f} MB)")
            else:
                print("  No cached GGUF files.")
        hf_dir = caches.get('huggingface')
        if hf_dir:
            print(f"\n--- HUGGINGFACE Models Cache ({hf_dir}) ---")
            if cache_info['huggingface']:
                for repo_id, size_gb, revision_count in cache_info['huggingface']:
                    print(f"  {repo_id} ({size_gb:.2f} GB)")
                    print(f"    └─ {revision_count} revision(s)")
            else:
                print("  No cached HuggingFace models.")
        print(f"\n=== Summary ===")
        print(f"Total cached models: {cache_info['total_models']}")
        print(f"Total disk usage: {cache_info['total_size_gb']:.2f} GB")
        print("\nCache locations:")
        for cache_name, cache_dir in caches.items():
            print(f"  {cache_name}: {cache_dir}")
        sys.exit(0)
    
    if args.remove_all_models:
        print("\n=== Removing All Cached Models ===")
        from codai.models.cache import remove_all_cached_models
        total_removed = remove_all_cached_models()
        print(f"\n=== Removed {total_removed} item(s) from all caches ===")
        sys.exit(0)
    
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
    
    if args.download_model:
        print(f"\n=== Downloading Model: {args.download_model} ===")
        from codai.models.cache import download_model, is_huggingface_model_id
        model_id = args.download_model
        file_pattern = args.download_file_pattern
        try:
            if is_huggingface_model_id(model_id):
                if file_pattern:
                    print(f"File pattern: {file_pattern}")
                    cached_path = download_model(model_id, file_pattern=file_pattern)
                else:
                    print("Trying GGUF download first...")
                    cached_path = download_model(model_id, file_pattern='.gguf')
                    if not cached_path:
                        print("No GGUF files found, downloading full HuggingFace repo...")
                        try:
                            from huggingface_hub import snapshot_download
                            cached_path = snapshot_download(model_id)
                        except Exception as e:
                            print(f"Error downloading full repo: {e}")
                            cached_path = None
            else:
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
    
    # Import core modules (only after early exits)
    from codai.api import app
    from codai.api.state import (
        set_global_args, set_global_debug, set_global_system_prompt,
        set_global_tools_closer_prompt, set_global_file_path, set_load_mode,
        set_grammar_guided_gen, get_global_args
    )
    from codai.models.manager import ModelManager, MultiModelManager, model_manager, multi_model_manager
    from codai.backends import detect_available_backends
    from codai.api.app import set_global_file_path_wrapper
    
    # Import after early exits
    from codai.api.text import (
        set_global_args as set_global_args_text,
        set_global_debug as set_global_debug_text,
        set_global_system_prompt as set_global_system_prompt_text,
        set_global_tools_closer_prompt as set_global_tools_closer_prompt_text,
    )
    from codai.api.app import set_load_mode as set_load_mode_app
    
    # Store config reference globally for access
    fastapi_app = app
    fastapi_app.state.config_mgr = config_mgr
    fastapi_app.state.config = config
    
    # Set global variables from config and args (args override config for now)
    global global_system_prompt, global_tools_closer_prompt, global_debug, global_dump, global_file_path, grammar_guided_gen
    
    # Debug from command line flag (overrides config)
    global_debug = args.debug
    set_global_debug(global_debug)
    set_global_debug_text(global_debug)
    
    global_dump = args.dump
    if global_dump:
        global_debug = True
        set_global_debug(True)
        set_global_debug_text(True)
    
    # System prompt (from config)
    global_system_prompt = config.system_prompt
    set_global_system_prompt(global_system_prompt)
    set_global_system_prompt_text(global_system_prompt)
    
    # Tools closer prompt
    global_tools_closer_prompt = config.tools_closer_prompt
    set_global_tools_closer_prompt(global_tools_closer_prompt)
    set_global_tools_closer_prompt_text(global_tools_closer_prompt)
    if global_tools_closer_prompt:
        print("Tools closer prompt enabled")
    
    # Grammar guided generation
    grammar_guided_gen = config.grammar_guided
    if grammar_guided_gen:
        set_grammar_guided_gen(True)
        print("Grammar-guided generation enabled")
    
    # File path
    global_file_path = config.file_path
    set_global_file_path(global_file_path)
    set_global_file_path_wrapper(global_file_path)
    from codai.api.images import set_global_file_path as set_images_file_path
    set_images_file_path(global_file_path)
    
    # Debug: print command line
    if global_debug:
        import shlex
        cmd_line = ' '.join(shlex.quote(arg) for arg in sys.argv)
        print(f"\n{'='*80}")
        print(f"=== COMMAND LINE: {cmd_line}")
        print(f"{'='*80}\n")
        print("DEBUG MODE ENABLED")
    
    # Determine load mode from config
    load_mode = config.models.default_load_mode or "ondemand"
    set_load_mode(load_mode)
    multi_model_manager.set_load_mode(load_mode)
    
    print(f"\nLoad mode: {load_mode}")
    if load_mode == "ondemand":
        print("  (pre-load first model, unload/load on switch)")
    elif load_mode == "loadswap":
        print("  (first model in VRAM, others in CPU RAM, swap on switch)")
    elif load_mode == "loadall":
        print("  (load all models into VRAM, offload to CPU RAM if full)")
    
    # Detect available backends
    available_backends = detect_available_backends()
    print(f"\nAvailable backends: {available_backends}")
    
    # Determine backend from config
    backend = config.backend.type
    if backend == "auto":
        if available_backends.get('nvidia'):
            backend = "nvidia"
        elif available_backends.get('vulkan'):
            backend = "vulkan"
        elif available_backends.get('opencl'):
            backend = "opencl"
        else:
            print("Error: No supported backend detected")
            sys.exit(1)
    
    print(f"Using backend: {backend}")
    model_manager.backend_type = backend
    
    # Store global state
    fastapi_app.state.model_manager = model_manager
    fastapi_app.state.multi_model_manager = multi_model_manager
    
    # =========================================================================
    # Load models from config
    # =========================================================================
    print(f"\n=== Loading Models from Config ===")
    
    models_config = config_mgr.models_data
    
    # Helper to find model config
    def get_model_cfg(model_type, model_id):
        key = f"{model_type}:{model_id}"
        for m in models_config.get(f"{model_type}_models", []):
            if m.get("id") == model_id:
                return m
        return {}
    
    # Helper to build kwargs from model config
    def build_kwargs_from_config(model_cfg, model_type):
        kwargs = {}
        if model_type == "text":
            kwargs['ctx'] = model_cfg.get('context_size')
            kwargs['n_gpu_layers'] = model_cfg.get('n_gpu_layers', -1)
            kwargs['load_in_4bit'] = model_cfg.get('load_in_4bit', False)
            kwargs['load_in_8bit'] = model_cfg.get('load_in_8bit', False)
            kwargs['flash_attn'] = model_cfg.get('flash_attn', False)
            kwargs['offload_strategy'] = model_cfg.get('offload_strategy', 'auto')
            kwargs['manual_ram_gb'] = model_cfg.get('manual_ram_gb')
            kwargs['max_gpu_percent'] = model_cfg.get('max_gpu_percent')
            kwargs['no_ram'] = model_cfg.get('no_ram', False)
        elif model_type == "image":
            kwargs['llm_path'] = model_cfg.get('llm_path')
            kwargs['vae_path'] = model_cfg.get('vae_path')
            kwargs['sample_method'] = model_cfg.get('sample_method', 'res_multistep')
            kwargs['steps'] = model_cfg.get('steps', 4)
            kwargs['width'] = model_cfg.get('width', 512)
            kwargs['height'] = model_cfg.get('height', 512)
            kwargs['cfg_scale'] = model_cfg.get('cfg_scale', 1.0)
            kwargs['precision'] = model_cfg.get('precision', 'f32')
            kwargs['cpu_offload'] = model_cfg.get('cpu_offload', False)
            kwargs['seed'] = model_cfg.get('seed')
            kwargs['vae_tiling'] = model_cfg.get('vae_tiling', False)
            kwargs['clip_on_cpu'] = model_cfg.get('clip_on_cpu', False)
        elif model_type == "audio":
            kwargs['ctx'] = model_cfg.get('context_ms')
            kwargs['offload'] = model_cfg.get('offload')
            kwargs['vulkan_device'] = model_cfg.get('vulkan_device', 0)
        elif model_type == "vision":
            kwargs['ctx'] = model_cfg.get('context_size')
            kwargs['offload'] = model_cfg.get('offload')
            kwargs['n_gpu_layers'] = model_cfg.get('n_gpu_layers', -1)
        return kwargs
    
    # Load text models (main LLM)
    text_models = models_config.get("text_models", [])
    text_model_names = [m["id"] for m in text_models if m.get("enabled", True)]
    
    if text_model_names:
        print(f"\nMain text model(s): {text_model_names}")
        for idx, model_name in enumerate(text_models):
            multi_model_manager.set_default_model(
                model_name["id"],
                config=build_kwargs_from_config(model_name, "text"),
                backend_type=model_name.get("backend", "auto")
            )
    
    # Load preload list
    preload_list = models_config.get("preload", [])
    loaded_list = models_config.get("loaded", [])
    
    # Determine which models to preload at startup
    # loaded: models to load into VRAM (or CPU for loadswap) immediately
    # preload: models to keep in CPU RAM for fast swapping
    nopreload = False  # Config-based loading, no CLI preload skip
    
    # Pre-load models at startup based on config
    if not nopreload and load_mode in ("loadall", "loadswap"):
        all_startup_models = loaded_list + preload_list
    elif not nopreload and load_mode == "ondemand":
        all_startup_models = loaded_list[:1] if loaded_list else []
    else:
        all_startup_models = []
    
    # Pre-load process
    if text_model_names:
        first_text = text_models[0]["id"] if text_models else None
        
        if not nopreload and load_mode == "ondemand" and first_text:
            # Preload first model into VRAM
            try:
                print(f"Preloading first model into VRAM: {first_text}...")
                mm = multi_model_manager._load_default_model()
                if mm is not None and mm.backend is not None:
                    multi_model_manager.active_in_vram = multi_model_manager.default_model
                    print(f"Model loaded successfully: {first_text}")
                else:
                    print(f"Warning: Model {first_text} failed to load")
            except Exception as e:
                print(f"Warning: Failed to preload model: {e}")
                print(f"Model will load on first request")
    
    # Load audio models (registered, load on first request)
    audio_models = models_config.get("audio_models", [])
    for audio_m in audio_models:
        if audio_m.get("enabled", True):
            multi_model_manager.set_audio_model(
                audio_m["id"],
                config=build_kwargs_from_config(audio_m, "audio")
            )
    
    # Load image models
    image_models = models_config.get("image_models", [])
    for img_m in image_models:
        if img_m.get("enabled", True):
            multi_model_manager.set_image_model(
                img_m["id"],
                config=build_kwargs_from_config(img_m, "image")
            )
    
    # Load vision models
    vision_models = models_config.get("vision_models", [])
    for vis_m in vision_models:
        if vis_m.get("enabled", True):
            multi_model_manager.set_vision_model(
                vis_m["id"],
                config=build_kwargs_from_config(vis_m, "vision")
            )
    
    # Load TTS model
    tts_model = models_config.get("tts_models", [])
    if tts_model:
        for tts_m in tts_model:
            if tts_m.get("enabled", True):
                multi_model_manager.set_tts_model(tts_m["id"], {})
    
    # Register aliases
    aliases = models_config.get("aliases", {})
    for alias, model in aliases.items():
        multi_model_manager.set_model_alias(alias, model)
    
    # Print startup summary
    print(f"\nBackend: {backend}")
    print(f"Load mode: {load_mode}")
    
    available_models = multi_model_manager.list_models()
    print(f"\nAvailable models: {[m.id for m in available_models]}")
    
    # Register custom aliases from config
    if aliases:
        print(f"\nModel aliases:")
        for alias, target in aliases.items():
            print(f"  {alias} -> {target}")
    
    # Set global args for backward compatibility with existing code
    class ArgsCompat:
        pass
    global_args = ArgsCompat()
    global_args.backend = backend
    global_args.host = config.server.host
    global_args.port = config.server.port
    global_args.url = "auto"
    global_args.https = config.server.https
    global_args.privkey = config.server.https_key_path
    global_args.pubkey = config.server.https_cert_path
    global_args.offload_dir = config.offload.directory
    global_args.ram = config.offload.manual_ram_gb
    global_args.offload_strategy = config.offload.strategy
    global_args.no_ram = config.offload.no_ram
    global_args.load_in_4bit = config.offload.load_in_4bit
    global_args.load_in_8bit = config.offload.load_in_8bit
    global_args.flash_attn = config.offload.flash_attention
    global_args.max_gpu_percent = config.offload.max_gpu_percent
    global_args.n_gpu_layers = config.vulkan.n_gpu_layers
    global_args.n_ctx = [config.vulkan.n_ctx]
    global_args.vulkan_device = config.vulkan.device_id
    global_args.vulkan_single_gpu = config.vulkan.single_gpu
    global_args.image_sample_method = config.image.sample_method
    global_args.image_steps = config.image.steps
    global_args.image_width = config.image.width
    global_args.image_height = config.image.height
    global_args.image_cfg_scale = config.image.cfg_scale
    global_args.image_precision = config.image.precision
    global_args.image_cpu_offload = config.image.cpu_offload
    global_args.image_seed = config.image.seed
    global_args.vae_tiling = config.image.vae_tiling
    global_args.clip_on_cpu = config.image.clip_on_cpu
    global_args.system_prompt = config.system_prompt
    global_args.tools_closer_prompt = config.tools_closer_prompt
    global_args.grammar_guided_gen = config.grammar_guided
    global_args.debug = global_debug
    global_args.dump = global_dump
    global_args.file_path = config.file_path
    global_args.parser = config.parser
    global_args.hf_chat_template = config.hf_chat_templates
    global_args.force_reasoning = config.reasoning_options
    global_args.model = text_model_names
    global_args.language_model = text_model_names
    global_args.image_model = [m["id"] for m in image_models if m.get("enabled")]
    global_args.audio_model = [m["id"] for m in audio_models if m.get("enabled")]
    global_args.vision_model = [m["id"] for m in vision_models if m.get("enabled")]
    global_args.tts_model = tts_model[0]["id"] if tts_model else None
    global_args.model_aliases = [(k, v) for k, v in aliases.items()]
    global_args.whisper_server = config.whisper.server_path
    global_args.whisper_server_port = config.whisper.server_port
    global_args.audio_ctx = None
    global_args.audio_offload = None
    global_args.audio_vulkan_device = 0
    global_args.image_ctx = None
    global_args.image_offload = None
    global_args.download_file_pattern = None
    global_args.list_cached_models = False
    global_args.remove_all_models = False
    global_args.remove_model = None
    global_args.download_model = None
    global_args.vulkan_list_devices = False
    global_args.loadall = False
    global_args.loadswap = False
    global_args.nopreload = nopreload
    
    set_global_args(global_args)
    set_global_args_text(global_args)
    set_load_mode_app(load_mode)
    
    # Set image module global args
    from codai.api.images import set_global_args as set_images_global_args
    set_images_global_args(global_args)
    
    # Vulkan list devices
    if args.vulkan_list_devices:
        print("\nListing Vulkan devices...")
        try:
            import subprocess
            result = subprocess.run(['vulkaninfo', '--summary'], capture_output=True, text=True)
            if result.returncode == 0:
                print(result.stdout)
            else:
                print("Could not run vulkaninfo.")
        except Exception as e:
            print(f"Error: {e}")
        sys.exit(0)
    
    # Startup: Preload configured models (non-text) for loadall/loadswap
    if not nopreload and load_mode in ("loadall", "loadswap"):
        first_loaded = multi_model_manager.active_in_vram is not None
        
        if image_models:
            print(f"\n=== Pre-loading image model(s) ===")
            for img_m in image_models:
                if not img_m.get("enabled", True):
                    continue
                model_key = f"image:{img_m['id']}"
                if model_key in multi_model_manager.models:
                    continue
                try:
                    from codai.api.images import _load_diffusers_pipeline, _is_gguf_model, _load_sdcpp_model
                    if load_mode == "loadall":
                        print(f"Preloading image model into VRAM: {img_m['id']}...")
                        if _is_gguf_model(img_m['id']):
                            resolved_path = multi_model_manager.load_model(img_m['id'])
                            if resolved_path and os.path.isfile(resolved_path):
                                sd_model = _load_sdcpp_model(resolved_path, global_args)
                                if sd_model:
                                    multi_model_manager.add_model(model_key, sd_model)
                                    print(f"Image model loaded (VRAM): {img_m['id']}")
                        else:
                            try:
                                pipeline = _load_diffusers_pipeline(img_m['id'], global_args)
                                if pipeline:
                                    multi_model_manager.add_model(model_key, pipeline)
                                    print(f"Image model loaded (VRAM): {img_m['id']}")
                            except Exception as e:
                                em = str(e).lower()
                                if any(x in em for x in ['out of memory', 'oom', 'cuda error']):
                                    print(f"VRAM full for {img_m['id']}, will load on demand")
                                else:
                                    print(f"Warning: {e}")
                    elif load_mode == "loadswap" and not first_loaded:
                        print(f"Preloading image model: {img_m['id']}...")
                        if _is_gguf_model(img_m['id']):
                            resolved_path = multi_model_manager.load_model(img_m['id'])
                            if resolved_path and os.path.isfile(resolved_path):
                                sd_model = _load_sdcpp_model(resolved_path, global_args)
                                if sd_model:
                                    multi_model_manager.add_model(model_key, sd_model)
                                    first_loaded = True
                                    print(f"Image model loaded: {img_m['id']}")
                        else:
                            try:
                                pipeline = _load_diffusers_pipeline(img_m['id'], global_args)
                                if pipeline:
                                    multi_model_manager.add_model(model_key, pipeline)
                                    first_loaded = True
                                    print(f"Image model loaded: {img_m['id']}")
                            except Exception as e:
                                print(f"Warning: {e}")
                except Exception as e:
                    print(f"Warning: {e}")
    
    # Start the server
    import uvicorn
    print(f"\nStarting server on http://{config.server.host}:{config.server.port}")
    print(f"API docs: http://{config.server.host}:{config.server.port}/docs")
    print(f"Admin UI: http://{config.server.host}:{config.server.port}/admin")
    
    if model_manager.backend is not None:
        actual_backend = model_manager.backend_type
        if hasattr(model_manager.backend, 'force_cuda') and model_manager.backend.force_cuda:
            actual_backend = "cuda (via llama-cpp-python)"
        print(f"Using backend: {actual_backend}")
    
    if config.server.https:
        import ssl
        ssl_keyfile = config.server.https_key_path
        ssl_certfile = config.server.https_cert_path
        if not (ssl_keyfile and ssl_certfile):
            print("Generating self-signed HTTPS certificate...")
            import subprocess
            cert_path = config_dir / "cert.pem"
            key_path = config_dir / "key.pem"
            try:
                subprocess.run(
                    ["openssl", "req", "-x509", "-newkey", "rsa:4096",
                     "-keyout", str(key_path), "-out", str(cert_path),
                     "-days", "365", "-nodes", "-subj", "/CN=localhost"],
                    check=True, capture_output=True
                )
                ssl_keyfile = str(key_path)
                ssl_certfile = str(cert_path)
                print(f"Generated self-signed certificate")
            except Exception as e:
                print(f"Warning: Could not generate certificate: {e}")
                print("Falling back to HTTP...")
                uvicorn.run(app, host=config.server.host, port=config.server.port)
                return
        
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(ssl_certfile, ssl_keyfile)
        uvicorn.run(app, host=config.server.host, port=config.server.port, ssl_context=ssl_context)
    else:
        uvicorn.run(app, host=config.server.host, port=config.server.port)


if __name__ == "__main__":
    main()
