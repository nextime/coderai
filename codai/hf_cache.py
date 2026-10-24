# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Where HuggingFace downloads land: ``models.hf_cache_dir``, and nothing else.

A deployment can bake ``HF_HOME`` into its image or its service unit, and a
container usually does — it has to point somewhere before any config is read.
That value then wins over the configured one in every process that does not
know better, which is how an install ends up with two HuggingFace caches: the
configured one, and a second copy of the same models wherever the image pointed.
Measured on one install: 22 GB in the image's directory, 13 GB of it a
byte-for-byte duplicate of the configured cache, on the smaller disk.

So the configured value is applied here, as early as a Python process can do it
-- from ``codai/__init__.py``, before anything imports huggingface_hub, torch or
transformers -- rather than by whatever launched the process. Every coderai
process imports the package, so every one of them agrees without the launcher
having to be in on it.

Four variables matter, and leaving any of them behind splits the cache again:

* ``HF_HOME``               — the parent; libraries derive the rest from it
* ``HF_HUB_CACHE``          — the modern explicit hub path
* ``HUGGINGFACE_HUB_CACHE`` — its older name, still read by pinned versions
* ``TRANSFORMERS_CACHE``    — deprecated upstream, still honoured, and the
  reason 3.1 GB sat in the image's directory on that same install

Subprocesses that run in their own venv (the isolated workers) never import
this package, so they get the same four through :func:`hf_cache_env`.
"""

import json
import os
from typing import Dict, Optional

#: Set this to anything non-empty to leave the environment exactly as found —
#: for a caller that is deliberately pointing one process somewhere else.
OPT_OUT = "CODERAI_NO_HF_CACHE_ENV"

_VAR_NAMES = ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE")


def _config_json_paths(config_dir: Optional[str] = None) -> list:
    """Candidate config.json paths, nearest first. ``CODERAI_CONFIG_DIR`` is what
    the container exports; a host install keeps its own user config dir."""
    if config_dir:
        return [os.path.join(config_dir, "config.json")]
    out = []
    env = os.environ.get("CODERAI_CONFIG_DIR")
    if env:
        out.append(os.path.join(os.path.expanduser(env), "coderai", "config.json"))
    try:
        from codai.platform_paths import user_config_dir
        out.append(os.path.join(str(user_config_dir()), "config.json"))
    except Exception:
        pass
    return out


def configured_hf_cache_dir(config_dir: Optional[str] = None) -> Optional[str]:
    """``models.hf_cache_dir`` from config.json, or None when it is not set.

    Reads the file directly rather than through ConfigManager: this runs before
    the server is built, and from processes that never build one."""
    for path in _config_json_paths(config_dir):
        try:
            with open(path) as fh:
                raw = (json.load(fh).get("models") or {}).get("hf_cache_dir")
        except Exception:
            continue
        if raw and str(raw).strip():
            return os.path.expanduser(str(raw).strip())
    return None


def hf_cache_env(hf_cache_dir: str) -> Dict[str, str]:
    """The four variables a child process needs to use ``hf_cache_dir``."""
    hf_cache_dir = os.path.expanduser(str(hf_cache_dir))
    hub = os.path.join(hf_cache_dir, "hub")
    return {
        "HF_HOME": hf_cache_dir,
        "HF_HUB_CACHE": hub,
        "HUGGINGFACE_HUB_CACHE": hub,
        "TRANSFORMERS_CACHE": os.path.join(hf_cache_dir, "transformers"),
    }


def current_hf_cache_dir() -> Optional[str]:
    """The HuggingFace cache THIS process is using, however it got there.

    The environment is the live truth: :func:`apply_hf_cache_env` has already
    made it match the config where one is set, and a caller that pointed this
    one process elsewhere on purpose means it. Falls back to the configured
    value, then None (the HuggingFace default, which we do not second-guess)."""
    d = os.environ.get("HF_HOME")
    if d:
        return d
    hub = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hub:
        return os.path.dirname(hub) if os.path.basename(hub) == "hub" else hub
    return configured_hf_cache_dir()


def child_env(env: Optional[dict] = None, hf_cache_dir: Optional[str] = None) -> dict:
    """``env`` (default: a copy of ours) with all four HF cache variables set.

    For subprocesses that do not import this package — the isolated-venv
    workers. Setting only the hub pair is what left ``HF_HOME`` and
    ``TRANSFORMERS_CACHE`` pointing at the image's directory, so a worker
    fetched its model a second time into it. Returns the dict unchanged when
    nothing is configured, so an install on the HuggingFace default keeps it."""
    out = dict(os.environ if env is None else env)
    d = hf_cache_dir or current_hf_cache_dir()
    if d:
        out.update(hf_cache_env(d))
    return out


def apply_hf_cache_env(hf_cache_dir: Optional[str] = None,
                       config_dir: Optional[str] = None) -> Optional[str]:
    """Point this process at the configured HuggingFace cache. Idempotent.

    Call it before importing huggingface_hub / transformers / torch. When one of
    them is already imported its constants were computed from the environment at
    import time, so those are patched too -- the server re-applies this once the
    real config is loaded, by which point huggingface_hub usually is imported.

    Returns the directory applied, or None if nothing was configured (the
    environment is then left exactly as it was: no directory is invented)."""
    if os.environ.get(OPT_OUT):
        return None
    d = hf_cache_dir or configured_hf_cache_dir(config_dir)
    if not d:
        return None
    os.environ.update(hf_cache_env(d))
    hub = os.path.join(d, "hub")
    try:
        import sys
        _hfc = sys.modules.get("huggingface_hub.constants")
        if _hfc is not None:
            _hfc.HF_HUB_CACHE = hub
            if hasattr(_hfc, "HUGGINGFACE_HUB_CACHE"):
                _hfc.HUGGINGFACE_HUB_CACHE = hub
            if hasattr(_hfc, "HF_HOME"):
                _hfc.HF_HOME = d
    except Exception:
        pass
    return d
