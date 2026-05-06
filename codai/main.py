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

"""Main entry point for codai server."""
import sys
import os

# Import configuration from codai modules
from codai.cli import parse_args
from codai.config import ConfigManager
from codai.admin.routes import init_session_manager, set_config_manager


def main():
    """Main entry point for the codai server."""
    # Suppress unraisable exceptions from LlamaModel.__del__
    original_unraisablehook = sys.unraisablehook
    def suppress_llama_del_errors(unraisable):
        if isinstance(unraisable.exc_value, AttributeError) and 'LlamaModel' in repr(unraisable.object) and 'sampler' in str(unraisable.exc_value):
            return  # Ignore this specific error
        original_unraisablehook(unraisable)
    sys.unraisablehook = suppress_llama_del_errors
    
    # Optional: set process name if setproctitle is available
    try:
        import setproctitle
        setproctitle.setproctitle("codai")
    except ImportError:
        pass
    
    args = parse_args()
    
    # Initialize ConfigManager
    config_dir = args.config
    config_mgr = ConfigManager(config_dir)
    config = config_mgr.load()

    # Apply cache directory overrides from config before any cache module is used
    if config.models.hf_cache_dir:
        os.environ['HF_HOME'] = config.models.hf_cache_dir
        os.environ['HUGGINGFACE_HUB_CACHE'] = config.models.hf_cache_dir
    if config.models.gguf_cache_dir:
        os.environ['CODERAI_CACHE_DIR'] = config.models.gguf_cache_dir
    
    # Initialize admin session manager and expose config to admin routes
    from pathlib import Path
    init_session_manager(Path(config_dir))
    set_config_manager(config_mgr)
    
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
    
    # =========================================================================
    # Register and optionally pre-load all configured models
    # Models with load_mode == "load" are pre-loaded at startup.
    # Models with load_mode == "on-request" (default) are loaded on demand.
    # =========================================================================

    def _model_id(m):
        """Return the model path/id from a config entry (dict or str)."""
        if isinstance(m, str):
            return m
        return m.get("path") or m.get("id") or ""

    def _model_cfg(m, mtype):
        cfg = build_kwargs_from_config(m, mtype) if isinstance(m, dict) else {}
        if isinstance(m, dict):
            for k in ("load_mode", "used_vram_gb", "alias"):
                if k in m:
                    cfg[k] = m[k]
        return cfg

    # Text models
    text_models = models_config.get("text_models", [])
    # Also include legacy gguf_models entries (treated as text)
    text_models = text_models + models_config.get("gguf_models", [])
    text_model_names = [_model_id(m) for m in text_models if _model_id(m)]

    if text_model_names:
        print(f"\nText model(s): {text_model_names}")
        for i, m in enumerate(text_models):
            mid = _model_id(m)
            if not mid:
                continue
            cfg = _model_cfg(m, "text")
            if i == 0:
                # Only the first text model becomes the default
                multi_model_manager.set_default_model(
                    mid, config=cfg,
                    backend_type=m.get("backend", "auto") if isinstance(m, dict) else "auto"
                )
            else:
                # Additional text models: register config only, no default override
                multi_model_manager.config[mid] = cfg
                multi_model_manager.model_backend_types[mid] = (
                    m.get("backend", "auto") if isinstance(m, dict) else "auto"
                )

    # Audio models
    audio_models = models_config.get("audio_models", [])
    for m in audio_models:
        mid = _model_id(m)
        if not mid:
            continue
        if isinstance(m, dict) and m.get("backend") == "whisper-server":
            cfg = _model_cfg(m, "audio")
            cfg.update({
                "backend": "whisper-server",
                "server_path": m.get("server_path", ""),
                "model_path": m.get("model_path") or None,
                "port": int(m.get("port", 8744)),
                "gpu_device": int(m.get("gpu_device", 0)),
            })
            multi_model_manager.register_whisper_server(
                model_id=mid,
                server_path=m.get("server_path", ""),
                model_path=m.get("model_path") or None,
                port=int(m.get("port", 8744)),
                gpu_device=int(m.get("gpu_device", 0)),
                config=cfg,
            )
        else:
            multi_model_manager.set_audio_model(mid, config=_model_cfg(m, "audio"))

    # Image models
    image_models = models_config.get("image_models", [])
    for m in image_models:
        mid = _model_id(m)
        if mid:
            multi_model_manager.set_image_model(mid, config=_model_cfg(m, "image"))

    # Vision models
    vision_models = models_config.get("vision_models", [])
    for m in vision_models:
        mid = _model_id(m)
        if mid:
            multi_model_manager.set_vision_model(mid, config=_model_cfg(m, "vision"))

    # TTS models
    tts_models = models_config.get("tts_models", [])
    for m in tts_models:
        mid = _model_id(m)
        if mid:
            multi_model_manager.set_tts_model(mid, config=_model_cfg(m, "tts") if isinstance(m, dict) else {})

    # Video generation models
    video_models = models_config.get("video_models", [])
    for m in video_models:
        mid = _model_id(m)
        if mid:
            multi_model_manager.set_video_model(mid, config=_model_cfg(m, "video") if isinstance(m, dict) else {})

    # Audio generation models (MusicGen, AudioLDM2, …)
    audio_gen_models = models_config.get("audio_gen_models", [])
    for m in audio_gen_models:
        mid = _model_id(m)
        if mid:
            multi_model_manager.set_audio_gen_model(mid, config=_model_cfg(m, "audio_gen") if isinstance(m, dict) else {})

    # Embedding models
    embedding_models = models_config.get("embedding_models", [])
    for m in embedding_models:
        mid = _model_id(m)
        if mid:
            multi_model_manager.set_embedding_model(mid, config=_model_cfg(m, "embedding") if isinstance(m, dict) else {})

    # Register aliases
    aliases = models_config.get("aliases", {})
    for alias, model in aliases.items():
        multi_model_manager.set_model_alias(alias, model)

    # Pre-load models marked as load_mode == "load" across ALL types
    all_model_entries = (
        [("text", m) for m in text_models] +
        [("audio", m) for m in audio_models] +
        [("image", m) for m in image_models] +
        [("vision", m) for m in vision_models] +
        [("tts", m) for m in tts_models] +
        [("video", m) for m in video_models] +
        [("audio_gen", m) for m in audio_gen_models] +
        [("embedding", m) for m in embedding_models]
    )
    for mtype, m in all_model_entries:
        mid = _model_id(m)
        if not mid:
            continue
        per_load_mode = m.get("load_mode", "on-request") if isinstance(m, dict) else "on-request"
        if per_load_mode != "load":
            print(f"  '{mid}' — on-request (will load when needed)")
            continue
        print(f"  Pre-loading '{mid}' (load mode)...")
        try:
            if mtype == "text":
                mm = multi_model_manager._load_model_by_name(mid)
                if mm is not None and mm.backend is not None:
                    multi_model_manager.active_in_vram = mid
                    print(f"  Loaded: {mid}")
                else:
                    print(f"  Warning: {mid} failed to load")
            elif mtype == "audio" and mid in multi_model_manager.whisper_servers:
                wsm = multi_model_manager.whisper_servers[mid]
                result = wsm.start(wsm._model_path, gpu_device=wsm._gpu_device)
                if wsm.is_running():
                    ws_key = f"audio:{mid}"
                    multi_model_manager.models[ws_key] = wsm
                    multi_model_manager.active_in_vram = ws_key
                    multi_model_manager.models_in_vram.add(ws_key)
                    print(f"  whisper-server started: {mid}")
                else:
                    print(f"  Warning: whisper-server '{mid}' failed to start")
            # image/vision/tts pre-loading is handled by their respective
            # API modules on first request; we just log intent here.
            else:
                print(f"  Note: pre-loading for {mtype} models happens on first request")
        except Exception as e:
            print(f"  Warning: failed to pre-load '{mid}': {e}")


    
    # Print startup summary
    print(f"\nBackend: {backend}")
    available_models = multi_model_manager.list_models()
    print(f"Available models: {[m.id for m in available_models]}")
    if aliases:
        print("Model aliases:")
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
    global_args.image_model = [_model_id(m) for m in image_models]
    global_args.audio_model = [_model_id(m) for m in audio_models]
    global_args.vision_model = [_model_id(m) for m in vision_models]
    global_args.tts_model = _model_id(tts_models[0]) if tts_models else None
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
    global_args.nopreload = False

    set_global_args(global_args)
    set_global_args_text(global_args)
    set_load_mode_app(load_mode)

    # Set image module global args
    from codai.api.images import set_global_args as set_images_global_args
    set_images_global_args(global_args)

    # Set video module global args
    from codai.api.video import set_global_args as set_video_global_args, set_global_file_path as set_video_file_path
    set_video_global_args(global_args)
    if global_file_path:
        set_video_file_path(global_file_path)

    # Set audio_gen module global args
    from codai.api.audio_gen import set_global_args as set_audiogen_global_args, set_global_file_path as set_audiogen_file_path
    set_audiogen_global_args(global_args)
    if global_file_path:
        set_audiogen_file_path(global_file_path)

    # Set voice clone module global args
    from codai.api.voice_clone import set_global_args as set_vc_global_args, set_global_file_path as set_vc_file_path
    set_vc_global_args(global_args)
    if global_file_path:
        set_vc_file_path(global_file_path)

    from codai.api.voice_convert import set_global_args as set_vconv_global_args, set_global_file_path as set_vconv_file_path
    set_vconv_global_args(global_args)
    if global_file_path:
        set_vconv_file_path(global_file_path)

    # Set faceswap module global args
    from codai.api.faceswap import set_global_args as set_fs_global_args, set_global_file_path as set_fs_file_path
    set_fs_global_args(global_args)
    if global_file_path:
        set_fs_file_path(global_file_path)

    # Set character profiles module global args
    from codai.api.characters import set_global_args as set_chars_global_args
    set_chars_global_args(global_args)

    # Set embeddings module global args
    from codai.api.embeddings import set_global_args as set_embed_global_args
    set_embed_global_args(global_args)

    # Pre-load image models marked as load_mode == "load"
    for m in image_models:
        mid = _model_id(m)
        if not mid:
            continue
        per_load_mode = m.get("load_mode", "on-request") if isinstance(m, dict) else "on-request"
        if per_load_mode != "load":
            continue
        model_key = f"image:{mid}"
        if model_key in multi_model_manager.models:
            continue
        try:
            from codai.api.images import _load_diffusers_pipeline, _is_gguf_model, _load_sdcpp_model
            print(f"Pre-loading image model '{mid}' (load mode)...")
            if _is_gguf_model(mid):
                resolved_path = multi_model_manager.load_model(mid)
                if resolved_path and os.path.isfile(resolved_path):
                    sd_model = _load_sdcpp_model(resolved_path, global_args)
                    if sd_model:
                        multi_model_manager.add_model(model_key, sd_model)
                        print(f"  Image model loaded: {mid}")
            else:
                pipeline = _load_diffusers_pipeline(mid, global_args)
                if pipeline:
                    multi_model_manager.add_model(model_key, pipeline)
                    print(f"  Image model loaded: {mid}")
        except Exception as e:
            print(f"  Warning: failed to pre-load image model '{mid}': {e}")


    
    # Apply queue max size from config
    from codai.queue.manager import queue_manager
    queue_manager.max_size = config.server.queue_max_size

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
