"""Command-line argument parsing for codai server."""
import argparse


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible API server supporting NVIDIA (CUDA) and Vulkan backends"
    )
    parser.add_argument(
        "--model",
        type=str,
        action="append",
        default=None,
        help="Model name, path, or URL for text-to-text LLM. Can be specified multiple times for multiple models.",
    )
    parser.add_argument(
        "--model-alias",
        type=str,
        action="append",
        default=None,
        dest="model_aliases",
        nargs=2,
        metavar=("ALIAS", "MODEL"),
        help="Register an alias for a model. Format: --model-alias <alias_name> <actual_model>",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["auto", "nvidia", "vulkan", "opencl"],
        default="auto",
        help="Backend to use: auto (detect), nvidia (CUDA), vulkan (AMD), or opencl",
    )
    parser.add_argument(
        "--image-backend",
        type=str,
        choices=["auto", "nvidia", "vulkan", "opencl"],
        default="auto",
        help="Image generation backend: auto, nvidia (CUDA), vulkan (AMD), or opencl",
    )
    parser.add_argument(
        "--audio-backend",
        type=str,
        choices=["auto", "nvidia", "vulkan", "opencl"],
        default="auto",
        help="Audio transcription backend: auto, nvidia (CUDA), vulkan (AMD), or opencl",
    )
    parser.add_argument(
        "--tts-backend",
        type=str,
        choices=["auto", "nvidia", "vulkan", "opencl"],
        default="auto",
        help="TTS backend: auto, nvidia (CUDA), vulkan (AMD), or opencl",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="auto",
        help="Base URL for media downloads: 'auto' (use request IP) or explicit URL (e.g., http://myserver:8000)",
    )
    parser.add_argument(
        "--https",
        action="store_true",
        help="Enable HTTPS with auto-generated certificate",
    )
    parser.add_argument(
        "--privkey",
        type=str,
        default=None,
        help="Path to HTTPS private key file",
    )
    parser.add_argument(
        "--pubkey",
        type=str,
        default=None,
        help="Path to HTTPS certificate file",
    )
    parser.add_argument(
        "--offload-dir",
        type=str,
        default="./offload",
        help="Directory for disk offload (NVIDIA backend only, default: ./offload)",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model in 4-bit precision (NVIDIA backend only, requires bitsandbytes)",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load model in 8-bit precision (NVIDIA backend only, requires bitsandbytes)",
    )
    parser.add_argument(
        "--ram",
        type=float,
        default=None,
        help="Maximum CPU RAM to use for model offloading in GB (NVIDIA backend only). Auto-detected if not specified. Disk offloading only occurs after this limit is exceeded.",
    )
    parser.add_argument(
        "--flash-attn",
        action="store_true",
        help="Use Flash Attention 2 (NVIDIA backend only, requires flash-attn package)",
    )
    parser.add_argument(
        "--offload-strategy",
        type=str,
        choices=["auto", "conservative", "balanced", "aggressive", "sequential"],
        default="auto",
        help="Offload strategy for NVIDIA backend (default: auto)",
    )
    parser.add_argument(
        "--max-gpu-percent",
        type=float,
        default=None,
        help="Maximum GPU VRAM to use as percentage (0-100). Overrides offload-strategy. Lower values offload more to CPU/RAM (default: None = use offload-strategy)",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=-1,
        help="Number of layers to offload to GPU (Vulkan backend only, default: -1 = all layers)",
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        action="append",
        default=None,
        help="Context window size (Vulkan backend). Can be specified multiple times, one per --model.",
    )
    parser.add_argument(
        "--vulkan-device",
        type=int,
        default=0,
        help="Vulkan GPU device ID to use (Vulkan backend only, default: 0). Use --vulkan-list-devices to see available devices",
    )
    parser.add_argument(
        "--vulkan-single-gpu",
        action="store_true",
        help="Force Vulkan to use only the specified GPU device (prevents layer distribution across multiple GPUs)",
    )
    parser.add_argument(
        "--vulkan-list-devices",
        action="store_true",
        help="List available Vulkan GPU devices and exit",
    )
    parser.add_argument(
        "--hf-chat-template",
        action="append",
        default=[],
        help="Use HuggingFace apply_chat_template. Examples: --hf-chat-template auto (all models), --hf-chat-template text (all text), --hf-chat-template mymodel:llama3 (specific model with template). Can be repeated.",
    )
    parser.add_argument(
        "--system-prompt",
        nargs="?",
        const=True,
        default=None,
        help="Inject a system prompt at the beginning of conversations. Use without a value for a default prompt, or provide custom text.",
    )
    # Multi-model arguments
    parser.add_argument(
        "--tts-model",
        type=str,
        default=None,
        help="Model for text-to-speech (e.g., kokoro, or path/URL to Kokoro model). Can be specified multiple times.",
    )
    parser.add_argument(
        "--audio-model",
        type=str,
        action="append",
        default=None,
        help="Model for audio transcription (e.g., whisper-1, base, or path to faster-whisper model). Can be specified multiple times for multiple models.",
    )
    parser.add_argument(
        "--audio-1",
        action="store_true",
        help="Disable request queue for audio models - return 409 if model is busy",
    )
    parser.add_argument(
        "--image-model",
        type=str,
        action="append",
        default=None,
        help="Model for image generation (e.g., stable-diffusion-xl-base-1.0). Can be specified multiple times for multiple models.",
    )
    parser.add_argument(
        "--vision-model",
        type=str,
        action="append",
        default=None,
        help="Model for image/video-to-text (e.g., llava-1.5, LLaVA). Supports vulkan and cuda backends.",
    )
    parser.add_argument(
        "--image-1",
        action="store_true",
        help="Disable request queue for image models - return 409 if model is busy",
    )
    parser.add_argument(
        "--llm-path",
        type=str,
        default=None,
        help="Path to CLIP LLM model for image generation (stable-diffusion-cpp-python).",
    )
    parser.add_argument(
        "--vae-path",
        type=str,
        default=None,
        help="Path to VAE model for image generation (stable-diffusion-cpp-python).",
    )
    parser.add_argument(
        "--image-sample-method",
        type=str,
        default="res_multistep",
        help="Sample method for image generation (default: res_multistep for Z-Image Turbo).",
    )
    parser.add_argument(
        "--image-steps",
        type=int,
        default=4,
        help="Number of inference steps for image generation (default: 4 for Z-Image Turbo).",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=512,
        help="Image width for generation (default: 512).",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=512,
        help="Image height for generation (default: 512).",
    )
    parser.add_argument(
        "--image-cfg-scale",
        type=float,
        default=1.0,
        help="CFG scale for image generation (default: 1.0 for Z-Image Turbo).",
    )
    parser.add_argument(
        "--image-precision",
        type=str,
        default="f32",
        choices=["bf16", "f32", "f16", "f8"],
        help="Model precision for image generation (default: f32). bf16 recommended for modern GPUs.",
    )
    parser.add_argument(
        "--image-cpu-offload",
        action="store_true",
        help="Enable sequential CPU offload for image models (lower VRAM usage).",
    )
    parser.add_argument(
        "--image-seed",
        type=int,
        default=None,
        help="Default seed for image generation (default: random).",
    )
    parser.add_argument(
        "--vae-tiling",
        action="store_true",
        help="Enable VAE tiling for lower VRAM usage (sd.cpp only).",
    )
    parser.add_argument(
        "--clip-on-cpu",
        action="store_true",
        help="Run CLIP on CPU to save VRAM (sd.cpp only).",
    )
    parser.add_argument(
        "--loadall",
        action="store_true",
        help="Load all models at startup. Tries VRAM first, offloads to CPU RAM if VRAM is full.",
    )
    parser.add_argument(
        "--loadswap",
        action="store_true",
        help="Load first model in VRAM, others in CPU RAM. Swap active model between VRAM and CPU RAM on switch.",
    )
    parser.add_argument(
        "--nopreload",
        action="store_true",
        help="Skip model pre-loading at startup. Models will load on first request using the active mode strategy (ondemand/loadswap/loadall).",
    )
    parser.add_argument(
        "--audio-ctx",
        type=int,
        action="append",
        default=None,
        help="Audio model context size in milliseconds. Can be specified multiple times, one per --audio-model.",
    )
    parser.add_argument(
        "--audio-offload",
        type=float,
        default=None,
        help="Audio model GPU offload percentage (0-100). If not set, uses CPU",
    )
    parser.add_argument(
        "--audio-vulkan-device",
        type=int,
        default=0,
        help="Vulkan GPU device ID to use for Whisper audio transcription (default: 0). Only used when using Vulkan backend.",
    )
    parser.add_argument(
        "--image-vulkan-device",
        type=int,
        default=None,
        help="Vulkan GPU device ID to use for image generation models (default: same as --vulkan-device). Use --vulkan-list-devices to see available devices",
    )

    parser.add_argument(
        "--whisper-cpp",
        type=str,
        default=None,
        help="Path to whisper.cpp CLI executable (e.g., ~/whisper.cpp/build/bin/whisper-cli). Uses Vulkan if available.",
    )
    parser.add_argument(
        "--whisper-server",
        type=str,
        default=None,
        help="Path to whisper.cpp server executable (e.g., ~/whisper.cpp/build/bin/whisper-server). Keeps model loaded in VRAM.",
    )
    parser.add_argument(
        "--whisper-server-port",
        type=int,
        default=8744,
        help="Port for whisper-server (default: 8744).",
    )
    parser.add_argument(
        "--image-ctx",
        type=int,
        action="append",
        default=None,
        help="Image model context size. Can be specified multiple times, one per --image-model.",
    )
    parser.add_argument(
        "--image-offload",
        type=float,
        default=None,
        help="Vision model GPU offload percentage (0-100). If not set, loads fully on GPU",
    )
    parser.add_argument(
        "--list-cached-models",
        action="store_true",
        help="List all cached models in the model cache directory",
    )
    parser.add_argument(
        "--remove-all-models",
        action="store_true",
        help="Remove all cached models from the model cache directory",
    )
    parser.add_argument(
        "--remove-model",
        type=str,
        default=None,
        help="Remove a specific cached model by name or hash (partial match)",
    )
    parser.add_argument(
        "--download-model",
        type=str,
        default=None,
        help="Download a model to cache (URL or HuggingFace model ID) and exit. Example: --download-model Qwen/Qwen3-8B-Instruct-2507-Q3_K_S",
    )
    parser.add_argument(
        "--download-file-pattern",
        type=str,
        default=None,
        help="File pattern for HuggingFace model downloads (e.g., .gguf, .safetensors). Default: .gguf for text models",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode - dumps full request/response to stdout for troubleshooting",
    )
    parser.add_argument(
        "--dump",
        action="store_true",
        help="Dump model output: raw output, parsed output, and litellm debug info",
    )
    parser.add_argument(
        "--file-path",
        type=str,
        default=None,
        help="Path to store generated files (images, audio). If specified, files will be saved here and served over web.",
    )
    parser.add_argument(
        "--parser",
        type=str,
        default="auto",
        choices=["auto", "litellm"],
        help="Tool call parser to use: 'auto' for internal parser, 'litellm' for LiteLLM's parser. Default: auto",
    )
    # Custom type for comma-separated reasoning options
    def reasoning_choices(value):
        if not value:
            return []
        options = [v.strip().lower() for v in value.split(',')]
        valid = {'chat', 'stop', 'inject', 'prompt', 'all', 'twopass', 'mock', 'raw'}
        invalid = [o for o in options if o not in valid]
        if invalid:
            raise argparse.ArgumentTypeError(f"Invalid choices: {invalid}. Valid options: {valid}")
        # Expand 'all' to all options
        if 'all' in options:
            options = ['chat', 'inject', 'prompt', 'mock', 'raw', 'twopass']
        return options
    
    parser.add_argument(
        "--force-reasoning",
        type=reasoning_choices,
        default=None,
        help="Force reasoning. Options: 'chat' (API), 'stop' (tokens), 'inject' (sys prompt), 'prompt' (seeding), 'twopass' (2 calls), 'mock' (fake stats), 'raw' (raw completion), 'all' (all options). Combine: --force-reasoning chat,inject.",
    )
    parser.add_argument(
        "--grammar-guided-gen",
        "--ggg",
        action="store_true",
        default=False,
        help="Enable grammar-guided generation to reduce model hallucinations when using tools. Uses GBNF grammar for Vulkan backend and outlines for CUDA backend.",
    )
    parser.add_argument(
        "--tools-closer-prompt",
        action="store_true",
        default=False,
        help="Enable prompt distillation: place tool definitions right before the user's latest request instead of in the system prompt. This can improve tool call accuracy.",
    )
    return parser.parse_args()
