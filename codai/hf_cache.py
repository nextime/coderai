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
    """The four variables a child process needs to use ``hf_cache_dir``.

    All four name the SAME hub directory. ``TRANSFORMERS_CACHE`` is deprecated
    but transformers 4.x still honours it, and it honours it as the cache_dir
    handed to ``hf_hub_download`` -- i.e. as a hub-layout cache, not as a
    second kind of store. Pointing it at ``<dir>/transformers`` therefore does
    not give transformers its own corner of one cache; it gives the machine
    TWO hub caches, and which one a process uses depends on the transformers
    version in its venv.

    Measured 2026-10-28: ``<dir>/hub`` held 122 models and
    ``<dir>/transformers`` held 2, both in ``models--org--name`` layout. An
    isolated worker running transformers 4.x looked only in the second, found
    nothing, and -- being forced offline so a missing model fails fast --
    reported "We couldn't connect to https://huggingface.co" for a model that
    was sitting in the other directory all along. The same split is what makes
    a worker re-download a model the server already has.
    """
    hf_cache_dir = os.path.expanduser(str(hf_cache_dir))
    hub = os.path.join(hf_cache_dir, "hub")
    return {
        "HF_HOME": hf_cache_dir,
        "HF_HUB_CACHE": hub,
        "HUGGINGFACE_HUB_CACHE": hub,
        "TRANSFORMERS_CACHE": hub,
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



#: Names huggingface_hub / transformers accept for the Hub token, in the order
#: the libraries themselves prefer.
_TOKEN_VAR_NAMES = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")

#: Every name a token may arrive under, and every file it may sit in. Kept the
#: same as codai.api.pyannote_worker._resolve_hf_token, which has worked for
#: gated pyannote repos for a long time -- two different answers to "where is
#: the token" is how one path authenticates and another does not.
_TOKEN_ENV_NAMES = ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def _token_files(config_dir: Optional[str] = None) -> list:
    out = []
    env_file = os.environ.get("CODERAI_HF_TOKEN_FILE")
    if env_file:
        out.append(env_file)
    if config_dir:
        out.append(os.path.join(config_dir, "hf_token"))
    out.append(os.path.expanduser("~/.coderai/hf_token"))
    out.append("/config/coderai/hf_token")
    return out


def configured_hf_token(config_dir: Optional[str] = None) -> Optional[str]:
    """The operator's HuggingFace token, or None.

    The environment first, then the protected file in the coderai config dir --
    the same resolution the diarization worker has always used for gated
    pyannote repos, so a token that works for one path works for all of them.

    It exists here because the MODEL LOADING path had no such resolution. The
    host runner forwards HF_TOKEN from the host environment, which is only set
    if whoever launched the container exported it; on an install where it was
    not, huggingface_hub went out unauthenticated for every from_pretrained.
    That is invisible until a repo is gated, and then it fails as 401 "must be
    authenticated" -- which reads exactly like terms that have not been
    accepted, even when they have. Measured 2026-10-28 on
    stabilityai/stable-audio-open-1.0: the account had access and the token on
    disk could fetch the weights, while the engine loading them could not.

    Never logged: it is a credential, and it ends up in request headers.
    """
    for name in _TOKEN_ENV_NAMES:
        tok = (os.environ.get(name) or "").strip()
        if tok:
            return tok
    for path in _token_files(config_dir):
        try:
            if path and os.path.isfile(path):
                with open(path) as fh:
                    tok = fh.read().strip()
                if tok:
                    return tok
        except Exception:
            continue
    return None


def apply_hf_token(config_dir: Optional[str] = None) -> bool:
    """Put the configured token in the environment. Idempotent.

    Returns whether a token is now present. Both canonical names are set,
    because library versions differ in which they read, and a token set only
    in the one a given version ignores is the same as no token at all.
    """
    tok = configured_hf_token(config_dir)
    if not tok:
        return False
    for name in _TOKEN_VAR_NAMES:
        os.environ[name] = tok
    return True


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
    # The token travels with the cache settings: both answer "how does this
    # process talk to the Hub", both are needed before huggingface_hub is
    # imported, and an install that configured one has configured the other in
    # the same directory.
    try:
        apply_hf_token(config_dir)
    except Exception:
        pass
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
