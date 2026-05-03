"""Configuration management for coderai."""
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class ServerConfig:
    """Server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    https: bool = False
    https_key_path: Optional[str] = None
    https_cert_path: Optional[str] = None


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


@dataclass
class OffloadConfig:
    """Offload configuration."""
    directory: str = "./offload"


@dataclass
class Config:
    """Main configuration class."""
    version: str = "1.0"
    server: ServerConfig = field(default_factory=ServerConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    offload: OffloadConfig = field(default_factory=OffloadConfig)
    system_prompt: Optional[str] = None
    tools_closer_prompt: bool = False
    grammar_guided: bool = False
    file_path: Optional[str] = None
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
        
        self.config: Optional[Config] = None
        self.models_data: Dict[str, Any] = {}
        self.auth_data: Dict[str, Any] = {}
    
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
                    "default_load_mode": "ondemand"
                },
                "offload": {
                    "directory": "./offload"
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
            try:
                from argon2 import PasswordHasher
                ph = PasswordHasher()
                default_admin_hash = ph.hash("admin")
            except ImportError:
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
            with open(self.auth_path, 'w') as f:
                json.dump(default_auth, f, indent=2)
            print(f"Created default auth config: {self.auth_path}")
            print("\nDefault credentials: admin / admin")
            print("You will be prompted to change the password on first login.\n")
    
    def load(self) -> Config:
        """Load configuration from files.
        
        Returns:
            Config object with loaded settings
        """
        # Create defaults if config directory is empty or doesn't exist
        if not self.config_dir.exists() or not any(self.config_dir.iterdir()):
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
                system_prompt=config_data.get("system_prompt"),
                tools_closer_prompt=config_data.get("tools_closer_prompt", False),
                grammar_guided=config_data.get("grammar_guided", False),
                file_path=config_data.get("file_path"),
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
                "https_cert_path": self.config.server.https_cert_path
            },
            "backend": {
                "type": self.config.backend.type,
                "image_backend": self.config.backend.image_backend,
                "audio_backend": self.config.backend.audio_backend,
                "tts_backend": self.config.backend.tts_backend
            },
            "models": {
                "default_load_mode": self.config.models.default_load_mode
            },
            "offload": {
                "directory": self.config.offload.directory
            },
            "system_prompt": self.config.system_prompt,
            "tools_closer_prompt": self.config.tools_closer_prompt,
            "grammar_guided": self.config.grammar_guided,
            "file_path": self.config.file_path,
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
    
    def reload(self):
        """Reload all configuration files."""
        return self.load()
