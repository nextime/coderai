"""Platform-aware filesystem helpers for CoderAI."""

from __future__ import annotations

import os
from pathlib import Path


APP_NAME = "coderai"


def _home_dir() -> Path:
    return Path.home()


def _windows_dir(env_var: str, fallback: Path) -> Path:
    value = os.environ.get(env_var)
    return Path(value).expanduser() if value else fallback


def user_config_dir() -> Path:
    if os.name == "nt":
        base = _windows_dir("APPDATA", _home_dir() / "AppData" / "Roaming")
        return base / "CoderAI"
    if sys_platform() == "darwin":
        return _home_dir() / "Library" / "Application Support" / "CoderAI"
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg).expanduser() if xdg else (_home_dir() / ".config")
    return base / APP_NAME


def user_data_dir() -> Path:
    if os.name == "nt":
        base = _windows_dir("LOCALAPPDATA", _home_dir() / "AppData" / "Local")
        return base / "CoderAI"
    if sys_platform() == "darwin":
        return _home_dir() / "Library" / "Application Support" / "CoderAI"
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg).expanduser() if xdg else (_home_dir() / ".local" / "share")
    return base / APP_NAME


def user_cache_dir() -> Path:
    if os.name == "nt":
        base = _windows_dir("LOCALAPPDATA", _home_dir() / "AppData" / "Local")
        return base / "CoderAI" / "Cache"
    if sys_platform() == "darwin":
        return _home_dir() / "Library" / "Caches" / "CoderAI"
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else (_home_dir() / ".cache")
    return base / APP_NAME


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_config_dir() -> Path:
    return ensure_dir(user_config_dir())


def default_data_dir() -> Path:
    return ensure_dir(user_data_dir())


def default_cache_dir() -> Path:
    return ensure_dir(user_cache_dir())


def legacy_style_config_dir() -> Path:
    if os.name == "nt":
        return _home_dir() / ".coderai"
    return _home_dir() / ".coderai"


def legacy_style_cache_root() -> Path:
    if os.name == "nt":
        base = _windows_dir("LOCALAPPDATA", _home_dir() / "AppData" / "Local")
        return base / ".cache"
    if sys_platform() == "darwin":
        return _home_dir() / "Library" / "Caches"
    xdg = os.environ.get("XDG_CACHE_HOME")
    return Path(xdg).expanduser() if xdg else (_home_dir() / ".cache")


def default_model_cache_dir() -> Path:
    return ensure_dir(legacy_style_cache_root() / APP_NAME / "models")


def default_diffusers_cache_dir() -> Path:
    return ensure_dir(legacy_style_cache_root() / "diffusers")


def default_realesrgan_model_path() -> Path:
    return default_diffusers_cache_dir().parent / "realesrgan" / "RealESRGAN_x4plus.pth"


def default_insightface_model_path() -> Path:
    return ensure_dir(legacy_style_cache_root() / "insightface" / "models") / "inswapper_128.onnx"


def default_voices_dir() -> Path:
    return ensure_dir(legacy_style_config_dir() / "voices")


def default_characters_dir() -> Path:
    return ensure_dir(legacy_style_config_dir() / "characters")


def default_environments_dir() -> Path:
    return ensure_dir(legacy_style_config_dir() / "environments")


def default_whisper_server_path() -> str:
    if os.name == "nt":
        local = _windows_dir("LOCALAPPDATA", _home_dir() / "AppData" / "Local")
        return str(local / "Programs" / "whisper-server" / "whisper-server.exe")
    if sys_platform() == "darwin":
        return "/usr/local/bin/whisper-server"
    return "/usr/local/bin/whisper-server"


def sys_platform() -> str:
    return os.sys.platform
