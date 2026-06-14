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

"""Configuration management for coderai."""
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field

from codai.broker.config import BrokerConfig


@dataclass
class ServerConfig:
    """Server configuration."""
    host: str = "127.0.0.1"
    port: int = 8776
    https: bool = False
    https_key_path: Optional[str] = None
    https_cert_path: Optional[str] = None
    queue_max_size: int = 6
    max_parallel_requests: int = 2


@dataclass
class BackendConfig:
    """Backend configuration."""
    type: str = "auto"
    image_backend: str = "auto"
    audio_backend: str = "auto"
    tts_backend: str = "auto"


@dataclass
class ModelsConfig:
    """Models configuration."""
    default_load_mode: str = "ondemand"
    hf_cache_dir: Optional[str] = None
    gguf_cache_dir: Optional[str] = None
    max_model_instances: int = 1  # max concurrent instances per model (global default; overridable per-model via "max_instances")


@dataclass
class OffloadConfig:
    """Offload configuration."""
    directory: str = "./offload"
    strategy: str = "auto"
    max_gpu_percent: Optional[float] = None
    no_ram: bool = False
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    manual_ram_gb: Optional[float] = None
    flash_attention: bool = False


@dataclass
class VulkanConfig:
    """Vulkan backend configuration."""
    n_gpu_layers: int = -1
    n_ctx: int = 2048
    device_id: int = 0
    single_gpu: bool = False


@dataclass
class ImageConfig:
    """Image generation configuration."""
    llm_path: Optional[str] = None
    vae_path: Optional[str] = None
    sample_method: str = "res_multistep"
    steps: int = 4
    width: int = 512
    height: int = 512
    cfg_scale: float = 1.0
    precision: str = "f32"
    cpu_offload: bool = False
    seed: Optional[int] = None
    vae_tiling: bool = False
    clip_on_cpu: bool = False


@dataclass
class WhisperConfig:
    """Whisper ASR configuration."""
    server_path: Optional[str] = None
    server_port: int = 8744


@dataclass
class ArchiveConfig:
    """Generation archive configuration."""
    enabled: bool = True
    directory: str = ""        # empty = <config_dir>/archive; relative paths resolve from config_dir
    retention: str = "never"   # one of: 1h 1d 2d 1w 1m 3m 6m 1y never


@dataclass
class ThermalConfig:
    """Thermal-protection configuration.

    Before running a request against a loaded model, wait until CPU/GPU
    temperatures are within safe limits so a long sequence of heavy
    generations can't overheat the machine and trip its power-off protection.
    Thresholds are in degrees Celsius. CPU and GPU can be toggled separately.
    """
    cpu_enabled: bool = True
    gpu_enabled: bool = True
    cpu_high: float = 90.0      # pause when CPU reaches this temperature
    cpu_resume: float = 87.0    # resume once CPU drops back to/below this
    gpu_high: float = 90.0      # pause when GPU reaches this temperature
    gpu_resume: float = 87.0    # resume once GPU drops back to/below this
    poll_seconds: float = 5.0   # how often to re-check while cooling down
    # Proactive soft-throttle: before a hard pause, when a sensor enters the warm
    # band [soft_throttle_temp, *_high) insert a short per-step sleep (scaled by
    # how close to the pause threshold) so the temperature climbs slower and the
    # hard cooldown is rarely hit. Caps the heat-rate of a single pegged core.
    soft_throttle_enabled: bool = False
    soft_throttle_temp: float = 80.0       # engage at/above this temperature (°C)
    soft_throttle_max_sleep: float = 3.0   # max seconds to sleep/checkpoint at the limit


@dataclass
class JobsConfig:
    """Background-job (LoRA training) configuration."""
    # When True, an interrupted training job (process restart) is left
    # 'interrupted' so it can resume from its on-disk checkpoint. When False,
    # such jobs are marked 'cancelled' on startup and not auto-resumed (their
    # checkpoints are kept, so they can be restarted manually from the Tasks
    # page). The --no-resume-jobs CLI flag forces this off for one run.
    resume_on_restart: bool = True


@dataclass
class Config:
    """Main configuration class."""
    version: str = "1.0"
    server: ServerConfig = field(default_factory=ServerConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    offload: OffloadConfig = field(default_factory=OffloadConfig)
    vulkan: VulkanConfig = field(default_factory=VulkanConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    archive: ArchiveConfig = field(default_factory=ArchiveConfig)
    thermal: ThermalConfig = field(default_factory=ThermalConfig)
    jobs: JobsConfig = field(default_factory=JobsConfig)
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    system_prompt: Optional[str] = None
    tools_closer_prompt: bool = False
    grammar_guided: bool = False
    file_path: Optional[str] = None
    # Base directory for temporary working files (frame extraction, upscaling,
    # interpolation, etc.). None/empty = the OS default (usually /tmp). Point it at
    # a large-capacity volume when /tmp is small — 4× upscaling extracts many large
    # frames and can exhaust a small /tmp ("No space left on device").
    tmp_dir: Optional[str] = None
    hf_chat_templates: list = field(default_factory=list)
    reasoning_options: list = field(default_factory=list)
    parser: str = "auto"


class ConfigManager:
    """Manages configuration loading, saving, and validation."""
    
    def __init__(self, config_dir: str):
        """Initialize the configuration manager.
        
        Args:
            config_dir: Path to the configuration directory
        """
        self.config_dir = Path(config_dir).expanduser()
        self.config_path = self.config_dir / "config.json"
        self.models_path = self.config_dir / "models.json"
        self.auth_path = self.config_dir / "auth.json"
        self.pipelines_path = self.config_dir / "pipelines.json"
        
        self.config: Optional[Config] = None
        self.models_data: Dict[str, Any] = {}
        self.auth_data: Dict[str, Any] = {}
        self.pipelines_data: list = []
    
    def ensure_config_dir(self):
        """Create configuration directory if it doesn't exist."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def create_default_configs(self):
        """Create default configuration files."""
        self.ensure_config_dir()
        
        # Create default config.json
        if not self.config_path.exists():
            default_config = {
                "version": "1.0",
                "server": {
                    "host": "127.0.0.1",
                    "port": 8776,
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
                    "default_load_mode": "ondemand"
                },
                "offload": {
                    "directory": "./offload"
                },
                "broker": {
                    "enabled": False,
                    "base_url": "",
                    "scope": "user",
                    "username": "",
                    "provider_id": "",
                    "client_id": "",
                    "registration_token": "",
                    "advertised_endpoint": "",
                    "transport": "websocket",
                    "heartbeat_interval_seconds": 30,
                    "connect_timeout_seconds": 10,
                    "request_timeout_seconds": 30,
                    "reconnect_initial_delay_seconds": 1,
                    "reconnect_max_delay_seconds": 60
                },
                "system_prompt": None,
                "tools_closer_prompt": False,
                "grammar_guided": False,
                "file_path": None,
                "hf_chat_templates": [],
                "reasoning_options": [],
                "parser": "auto"
            }
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            print(f"Created default config: {self.config_path}")
        
        # Create default models.json
        if not self.models_path.exists():
            default_models = {
                "text_models": [],
                "image_models": [],
                "audio_models": [],
                "vision_models": [],
                "tts_models": [],
                "gguf_models": [],
                "loaded": [],
                "preload": [],
                "unloaded": [],
                "aliases": {}
            }
            with open(self.models_path, 'w') as f:
                json.dump(default_models, f, indent=2)
            print(f"Created default models config: {self.models_path}")
        
        # Create default auth.json
        if not self.auth_path.exists():
            from codai.admin.auth import hash_password
            default_auth = {
                "users": [{
                    "id": 1,
                    "username": "admin",
                    "password_hash": hash_password("admin"),
                    "role": "admin",
                    "created_at": "2026-05-03T00:00:00Z",
                    "must_change_password": True
                }],
                "tokens": [],
                "sessions": {}
            }
            with open(self.auth_path, 'w') as f:
                json.dump(default_auth, f, indent=2)
            print(f"Created default auth config: {self.auth_path}")
            print(f"\nDefault credentials: admin / admin")
            print("IMPORTANT: Change this password immediately after first login.\n")
    
    def load(self) -> Config:
        """Load configuration from files.
        
        Returns:
            Config object with loaded settings
        """
        # Create defaults if config directory is empty or doesn't exist
        self.create_default_configs()
        
        # Load config.json
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
            
            # Parse into Config dataclass
            self.config = Config(
                version=config_data.get("version", "1.0"),
                server=ServerConfig(**config_data.get("server", {})),
                backend=BackendConfig(**config_data.get("backend", {})),
                models=ModelsConfig(**config_data.get("models", {})),
                offload=OffloadConfig(**config_data.get("offload", {})),
                vulkan=VulkanConfig(**config_data.get("vulkan", {})),
                image=ImageConfig(**config_data.get("image", {})),
                whisper=WhisperConfig(**config_data.get("whisper", {})),
                archive=ArchiveConfig(**config_data.get("archive", {})),
                thermal=ThermalConfig(**config_data.get("thermal", {})),
                jobs=JobsConfig(**config_data.get("jobs", {})),
                broker=BrokerConfig(**config_data.get("broker", {})),
                system_prompt=config_data.get("system_prompt"),
                tools_closer_prompt=config_data.get("tools_closer_prompt", False),
                grammar_guided=config_data.get("grammar_guided", False),
                file_path=config_data.get("file_path"),
                tmp_dir=config_data.get("tmp_dir"),
                hf_chat_templates=config_data.get("hf_chat_templates", []),
                reasoning_options=config_data.get("reasoning_options", []),
                parser=config_data.get("parser", "auto")
            )
        else:
            self.config = Config()
        
        # Load models.json
        if self.models_path.exists():
            with open(self.models_path, 'r') as f:
                self.models_data = json.load(f)
        else:
            self.models_data = {
                "text_models": [],
                "image_models": [],
                "audio_models": [],
                "vision_models": [],
                "tts_models": [],
                "gguf_models": [],
                "loaded": [],
                "preload": [],
                "unloaded": [],
                "aliases": {}
            }
        
        # Load auth.json
        if self.auth_path.exists():
            with open(self.auth_path, 'r') as f:
                self.auth_data = json.load(f)
        else:
            self.auth_data = {
                "users": [],
                "tokens": [],
                "sessions": {}
            }

        # Load pipelines.json
        if self.pipelines_path.exists():
            with open(self.pipelines_path, 'r') as f:
                self.pipelines_data = json.load(f)
        else:
            self.pipelines_data = []
        
        return self.config
    
    def save_config(self):
        """Save config.json to disk."""
        config_dict = {
            "version": self.config.version,
            "server": {
                "host": self.config.server.host,
                "port": self.config.server.port,
                "https": self.config.server.https,
                "https_key_path": self.config.server.https_key_path,
                "https_cert_path": self.config.server.https_cert_path,
                "queue_max_size": self.config.server.queue_max_size,
                "max_parallel_requests": self.config.server.max_parallel_requests,
            },
            "backend": {
                "type": self.config.backend.type,
                "image_backend": self.config.backend.image_backend,
                "audio_backend": self.config.backend.audio_backend,
                "tts_backend": self.config.backend.tts_backend
            },
            "models": {
                "default_load_mode": self.config.models.default_load_mode,
                "hf_cache_dir": self.config.models.hf_cache_dir,
                "gguf_cache_dir": self.config.models.gguf_cache_dir,
            },
            "offload": {
                "directory": self.config.offload.directory,
                "strategy": self.config.offload.strategy,
                "max_gpu_percent": self.config.offload.max_gpu_percent,
                "no_ram": self.config.offload.no_ram,
                "load_in_4bit": self.config.offload.load_in_4bit,
                "load_in_8bit": self.config.offload.load_in_8bit,
                "manual_ram_gb": self.config.offload.manual_ram_gb,
                "flash_attention": self.config.offload.flash_attention
            },
            "vulkan": {
                "n_gpu_layers": self.config.vulkan.n_gpu_layers,
                "n_ctx": self.config.vulkan.n_ctx,
                "device_id": self.config.vulkan.device_id,
                "single_gpu": self.config.vulkan.single_gpu
            },
            "image": {
                "llm_path": self.config.image.llm_path,
                "vae_path": self.config.image.vae_path,
                "sample_method": self.config.image.sample_method,
                "steps": self.config.image.steps,
                "width": self.config.image.width,
                "height": self.config.image.height,
                "cfg_scale": self.config.image.cfg_scale,
                "precision": self.config.image.precision,
                "cpu_offload": self.config.image.cpu_offload,
                "seed": self.config.image.seed,
                "vae_tiling": self.config.image.vae_tiling,
                "clip_on_cpu": self.config.image.clip_on_cpu
            },
            "archive": {
                "enabled": self.config.archive.enabled,
                "directory": self.config.archive.directory,
                "retention": self.config.archive.retention,
            },
            "thermal": {
                "cpu_enabled": self.config.thermal.cpu_enabled,
                "gpu_enabled": self.config.thermal.gpu_enabled,
                "cpu_high": self.config.thermal.cpu_high,
                "cpu_resume": self.config.thermal.cpu_resume,
                "gpu_high": self.config.thermal.gpu_high,
                "gpu_resume": self.config.thermal.gpu_resume,
                "poll_seconds": self.config.thermal.poll_seconds,
                "soft_throttle_enabled": self.config.thermal.soft_throttle_enabled,
                "soft_throttle_temp": self.config.thermal.soft_throttle_temp,
                "soft_throttle_max_sleep": self.config.thermal.soft_throttle_max_sleep,
            },
            "jobs": {
                "resume_on_restart": self.config.jobs.resume_on_restart,
            },
            "broker": {
                "enabled": self.config.broker.enabled,
                "base_url": self.config.broker.base_url,
                "scope": self.config.broker.scope,
                "username": self.config.broker.username,
                "provider_id": self.config.broker.provider_id,
                "client_id": self.config.broker.client_id,
                "registration_token": self.config.broker.registration_token,
                "advertised_endpoint": self.config.broker.advertised_endpoint,
                "transport": self.config.broker.transport,
                "heartbeat_interval_seconds": self.config.broker.heartbeat_interval_seconds,
                "connect_timeout_seconds": self.config.broker.connect_timeout_seconds,
                "request_timeout_seconds": self.config.broker.request_timeout_seconds,
                "reconnect_initial_delay_seconds": self.config.broker.reconnect_initial_delay_seconds,
                "reconnect_max_delay_seconds": self.config.broker.reconnect_max_delay_seconds,
            },
            "system_prompt": self.config.system_prompt,
            "tools_closer_prompt": self.config.tools_closer_prompt,
            "grammar_guided": self.config.grammar_guided,
            "file_path": self.config.file_path,
            "tmp_dir": self.config.tmp_dir,
            "hf_chat_templates": self.config.hf_chat_templates,
            "reasoning_options": self.config.reasoning_options,
            "parser": self.config.parser
        }

        with open(self.config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def save_models(self):
        """Save models.json to disk."""
        with open(self.models_path, 'w') as f:
            json.dump(self.models_data, f, indent=2)
    
    def save_auth(self):
        """Save auth.json to disk."""
        with open(self.auth_path, 'w') as f:
            json.dump(self.auth_data, f, indent=2)

    def save_pipelines(self):
        """Save pipelines.json to disk."""
        with open(self.pipelines_path, 'w') as f:
            json.dump(self.pipelines_data, f, indent=2)
    
    def reload(self):
        """Reload all configuration files."""
        return self.load()
