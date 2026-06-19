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

"""Command-line argument parsing for codai server."""
import argparse
import json
import os
from pathlib import Path

from codai.platform_paths import legacy_style_config_dir


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
    try:
        from argon2 import PasswordHasher
        ph = PasswordHasher()
        default_admin_hash = ph.hash("admin")
    except ImportError:
        from codai.admin.auth import hash_password
        default_admin_hash = hash_password("admin")
    
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
    default_config = str(legacy_style_config_dir())
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible API server supporting NVIDIA (CUDA) and Vulkan backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Configuration: All settings are loaded from JSON config files in the
configuration directory (--config DIR, default: OS-specific CoderAI directory). Key files:
  config.json  - Server and backend settings
  models.json  - Model registry and configurations
  auth.json    - Users, tokens, and sessions"""
    )
    parser.add_argument(
        "--config",
        type=str,
        default=default_config,
        help=f"Configuration directory (default: {default_config})",
    )
    parser.add_argument(
        "--tmp",
        type=str,
        default=None,
        metavar="DIR",
        help="Base directory for temporary working files (frame extraction, "
             "upscaling, interpolation). Overrides config.tmp_dir. Use a large "
             "volume when /tmp is small — 4x upscaling can exhaust a small /tmp "
             "('No space left on device').",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode - dumps full request/response to stdout for troubleshooting",
    )
    parser.add_argument(
        "--debug-ws",
        action="store_true",
        help="Enable WebSocket debug logging (websockets library). Requires --debug or standalone.",
    )
    parser.add_argument(
        "--debug-web",
        action="store_true",
        help="Enable web/HTTP access logging (uvicorn per-request lines, e.g. /v1/loras/progress polling). Off by default.",
    )
    parser.add_argument(
        "--debug-thermal",
        action="store_true",
        help="Enable thermal-protection debug logging ([thermal][debug] temperature checks). Off by default.",
    )
    parser.add_argument(
        "--debug-lora",
        action="store_true",
        help="Enable LoRA training step logging to the terminal (per-step loss/progress). Off by default.",
    )
    parser.add_argument(
        "--dump",
        action="store_true",
        help="Dump model output: raw output, parsed output, and litellm debug info",
    )
    parser.add_argument(
        "--debug-requests",
        action="store_true",
        help="Log the full request/response payloads exchanged with API clients "
             "(opencode, etc.): incoming messages + tools and the outgoing "
             "content/tool_calls. Use to diagnose agentic tool-call loops.",
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
    parser.add_argument(
        "--no-resume-jobs",
        action="store_true",
        help="Do not resume/recover interrupted LoRA training jobs on restart. "
             "Mid-flight jobs are marked 'cancelled' (checkpoints are kept, so they "
             "can still be restarted manually from the Tasks page).",
    )
    parser.add_argument(
        "--pipeline-cache",
        action="store_true",
        help="Cache quantized diffusers pipelines to disk after the first build "
             "and reload them from that cache on later starts — skipping the "
             "expensive re-download/re-quantization (e.g. the Wan2.2 A14B). The "
             "fast acceleration LoRA fuse is re-applied per load. Uses extra disk.",
    )
    parser.add_argument(
        "--rebuild-pipeline-cache",
        action="store_true",
        help="Ignore any existing pipeline cache and rebuild it from scratch this "
             "run (use after changing a model's quantization/precision config).",
    )
    # ─── Frontend/engine split ───────────────────────────────────────────────
    parser.add_argument(
        "--single-process",
        action="store_true",
        help="Run the legacy single-process server (UI/API and all model work in "
             "one process). Default boots a front proxy + supervised engine "
             "subprocess(es) so the web UI stays responsive during model work.",
    )
    parser.add_argument(
        "--engine-only",
        action="store_true",
        help="Run this process as an engine (binds an internal localhost port, no "
             "front proxy). Normally launched automatically by the front; not "
             "intended to be run by hand.",
    )
    parser.add_argument(
        "--internal-port",
        type=int,
        default=None,
        help="Internal port for --engine-only mode (the front assigns one per engine).",
    )
    parser.add_argument(
        "--debug-engine",
        action="store_true",
        help="General engine debugging in the front/engine split (engine lifecycle, "
             "spawn details, health transitions). Does NOT include the internal "
             "HTTP access log — use --debug-engine-web for that.",
    )
    parser.add_argument(
        "--debug-engine-web",
        action="store_true",
        help="Show the internal front↔engine HTTP requests in an engine's access log "
             "(proxied calls, /internal/engine-state, /healthz, …). Suppressed by "
             "default since every engine only ever serves internal front traffic.",
    )
    return parser.parse_args()
