"""Command-line argument parsing for codai server."""
import argparse
import json
import os
from pathlib import Path


def load_config_file(config_dir: Path) -> dict:
    """Load the main config.json file."""
    config_path = config_dir / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}


def load_models_file(config_dir: Path) -> dict:
    """Load the models.json file."""
    models_path = config_dir / "models.json"
    if models_path.exists():
        with open(models_path, 'r') as f:
            return json.load(f)
    return {}


def load_auth_file(config_dir: Path) -> dict:
    """Load the auth.json file."""
    auth_path = config_dir / "auth.json"
    if auth_path.exists():
        with open(auth_path, 'r') as f:
            return json.load(f)
    return {}


def setup_default_config(config_dir: Path):
    """Create default configuration files if they don't exist."""
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Default config.json
    default_config = {
        "version": "1.0",
        "server": {
            "host": "0.0.0.0",
            "port": 8000,
            "https": False,
            "https_key_path": None,
            "https_cert_path": None
        },
        "backend": {
            "type": "auto",
            "image_backend": "auto",
            "audio_backend": "auto",
            "tts_backend": "auto"
        },
        "models": {
            "default_load_mode": "ondemand",
            "loaded": [],
            "preload": [],
            "unloaded": []
        },
        "offload": {
            "directory": "./offload",
            "strategy": "auto",
            "max_gpu_percent": None,
            "no_ram": False,
            "load_in_4bit": False,
            "load_in_8bit": False,
            "manual_ram_gb": None,
            "flash_attention": False
        },
        "vulkan": {
            "n_gpu_layers": -1,
            "n_ctx": 2048,
            "device_id": 0,
            "single_gpu": False
        },
        "image": {
            "llm_path": None,
            "vae_path": None,
            "sample_method": "res_multistep",
            "steps": 4,
            "width": 512,
            "height": 512,
            "cfg_scale": 1.0,
            "precision": "f32",
            "cpu_offload": False,
            "seed": None,
            "vae_tiling": False,
            "clip_on_cpu": False
        },
        "whisper": {
            "server_path": None,
            "server_port": 8744
        },
        "system_prompt": None,
        "tools_closer_prompt": False,
        "grammar_guided": False,
        "file_path": None,
        "hf_chat_templates": [],
        "reasoning_options": [],
        "parser": "auto"
    }
    
    config_path = config_dir / "config.json"
    if not config_path.exists():
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
    
    # Default models.json
    default_models = {
        "text_models": [],
        "image_models": [],
        "audio_models": [],
        "vision_models": [],
        "tts_model": None,
        "aliases": {}
    }
    models_path = config_dir / "models.json"
    if not models_path.exists():
        with open(models_path, 'w') as f:
            json.dump(default_models, f, indent=2)
    
    # Default auth.json with admin / admin
    from pathlib import Path
    import secrets
    from argon2 import PasswordHasher
    if hasattr(argon2, 'PasswordHasher'):
        ph = argon2.PasswordHasher()
        default_admin_hash = ph.hash("admin")
    else:
        default_admin_hash = "argon2id$v=19$m=65536,t=3,p=4$...admin_hash_placeholder"
    
    default_auth = {
        "users": [{
            "id": 1,
            "username": "admin",
            "password_hash": default_admin_hash,
            "role": "admin",
            "created_at": "2026-05-03T00:00:00Z",
            "must_change_password": True
        }],
        "tokens": [],
        "sessions": {}
    }
    auth_path = config_dir / "auth.json"
    if not auth_path.exists():
        with open(auth_path, 'w') as f:
            json.dump(default_auth, f, indent=2)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible API server supporting NVIDIA (CUDA) and Vulkan backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Configuration: All settings are loaded from JSON config files in the
configuration directory (--config DIR, default: ~/.coderai/). Key files:
  config.json  - Server and backend settings
  models.json  - Model registry and configurations
  auth.json    - Users, tokens, and sessions"""
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.expanduser("~/.coderai/"),
        help="Configuration directory (default: ~/.coderai/)",
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
        "--vulkan-list-devices",
        action="store_true",
        help="List available Vulkan GPU devices and exit",
    )
    return parser.parse_args()

