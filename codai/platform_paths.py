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


def deployment_config_dir():
    """The config dir this DEPLOYMENT uses, whatever started the process.

    The container launcher sets ``CODERAI_CONFIG_DIR`` and keeps its state in
    ``$CODERAI_CONFIG_DIR/coderai``. It also exports ``XDG_CONFIG_HOME`` to
    match — but only into the processes IT starts, so anything entering the
    container another way (``podman exec``, a cron job, a diagnostic script)
    resolves ``~/.config/coderai`` instead and quietly gets its own pod
    registry and its own pod TLS CA. Proven by accident: a one-off script in
    the container printed a third registry path and created a second CA.

    Returns None when the variable is unset, so a host install keeps whatever
    it was already using.
    """
    env = os.environ.get("CODERAI_CONFIG_DIR")
    if not env:
        return None
    return Path(env).expanduser() / APP_NAME


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


def default_loras_dir() -> Path:
    return ensure_dir(legacy_style_config_dir() / "loras")


def default_fonts_dir() -> Path:
    """Where an operator drops font files for captions and overlays.

    Beside voices and characters, because it is the same kind of thing: content
    the installation supplies, not something the image can know in advance. A
    caption font is a hard requirement of a render -- libass substitutes
    silently when a family is missing, and a wider substitute pushes text out
    of frame -- so an operator must be able to add one WITHOUT rebuilding the
    image. This directory is on the config volume, so what lands here survives
    a restart and a new image.
    """
    return ensure_dir(legacy_style_config_dir() / "fonts")


def default_whisper_server_path() -> str:
    if os.name == "nt":
        local = _windows_dir("LOCALAPPDATA", _home_dir() / "AppData" / "Local")
        return str(local / "Programs" / "whisper-server" / "whisper-server.exe")
    if sys_platform() == "darwin":
        return "/usr/local/bin/whisper-server"
    return "/usr/local/bin/whisper-server"


def sys_platform() -> str:
    return os.sys.platform
