# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Which repositories may run their own code here.

Some published pipelines ship the pipeline class in the repository rather than
in diffusers: MOSS-Image declares ``_class_name: MOSSImagePipeline`` with its
scheduler and transformer coming from a ``pipeline_moss_image`` module that
lives in the repo, and loading it means executing that module. Kromoss, a
fine-tune of it, is the same.

So the capability is necessary. Granting it blanket is not: `trust_remote_code`
on by default would make this server run arbitrary Python from any checkpoint
an operator happened to name, including one a client asked for by path. The
list is therefore an ALLOW-LIST and starts empty, and a model configured to
want remote code without being on it is REFUSED BY NAME -- rather than loaded
quietly without the pipeline it asked for, which would serve something other
than the model.

Entries are exact repo ids, or an ``org/*`` prefix for a publisher whose
fine-tunes an install trusts as a family (``lodestones/*`` covers Kromoss and
the Kroma siblings). Matching is case-insensitive, because repo ids are
compared case-insensitively everywhere else in this codebase and an operator
typing the wrong case should not get a silent refusal.
"""

from typing import List, Optional


def allow_list(cfg) -> List[str]:
    """The configured allow-list, normalised. ``cfg`` is the whole Config."""
    models = getattr(cfg, "models", None)
    raw = getattr(models, "trust_remote_code", None) or []
    if isinstance(raw, str):
        raw = [raw]
    out = []
    for item in raw:
        item = str(item or "").strip().lower()
        if item:
            out.append(item)
    return out


def allowed(cfg, repo: str) -> bool:
    """Whether ``repo`` may execute its own code on this install."""
    want = str(repo or "").strip().lower()
    if not want:
        return False
    for entry in allow_list(cfg):
        if entry == want:
            return True
        if entry.endswith("/*") and want.startswith(entry[:-1]):
            return True
        if entry == "*":
            # A deliberate "any", for an install that has decided. Spelled out
            # rather than implied by an empty list, which means the opposite.
            return True
    return False


def require(cfg, repo: str, what: str = "its own pipeline code") -> None:
    """Raise unless ``repo`` is allowed to run its own code.

    The message names the setting and the exact value to add, because the
    operator reading it is deciding whether to trust a repository and should
    not also have to work out the syntax.
    """
    if allowed(cfg, repo):
        return
    raise PermissionError(
        f"{repo!r} needs to run {what} to load, and this install does not "
        f"allow it. Add it to models.trust_remote_code (exact id, or "
        f"'{str(repo).split('/')[0]}/*' for the whole publisher) if you trust "
        f"that repository -- loading it executes Python from it.")


def kwargs_for(cfg, repo: str, custom_pipeline: Optional[str]) -> dict:
    """``from_pretrained`` kwargs for a model that ships its own pipeline.

    Empty when the model asks for nothing, so the ordinary load is untouched.
    Raises when it asks and is not allowed.
    """
    name = str(custom_pipeline or "").strip()
    if not name:
        return {}
    require(cfg, repo, f"its own pipeline module {name!r}")
    return {"custom_pipeline": name, "trust_remote_code": True}


def declared_pipeline_class(repo: str) -> Optional[str]:
    """The ``_class_name`` the repo's ``model_index.json`` declares, or None.

    Read rather than guessed, because the name is the only thing that says a
    repository means to supply its own pipeline: MOSS-Image declares
    ``MOSSImagePipeline``, which exists in the repo and nowhere in diffusers.
    """
    import json
    import os
    try:
        if os.path.isdir(repo):
            path = os.path.join(repo, "model_index.json")
            if not os.path.isfile(path):
                return None
        else:
            from huggingface_hub import hf_hub_download
            # Fetched, not read cache-only: a repo being loaded for the first
            # time has no cache yet, and a cache-only answer to "what class does
            # this declare" is permanently wrong on a freshly-rented pod. The
            # file is a few hundred bytes and the weights are about to follow.
            path = hf_hub_download(repo, "model_index.json")
        with open(path) as fh:
            name = (json.load(fh) or {}).get("_class_name")
        return str(name) if name else None
    except Exception:
        # Not a diffusers repo, not reachable, or no index. Either way there is
        # nothing to declare, and the ordinary load must be left alone.
        return None


def kwargs_for_declared_class(cfg, repo: str) -> dict:
    """``from_pretrained`` kwargs for a repo whose pipeline class is its own.

    The companion to :func:`kwargs_for`, which only fires when a card already
    names a ``custom_pipeline``. Nothing filled that field in for MOSS-Image or
    Kromoss, so the load went ahead as an ordinary repo load and diffusers
    reported ``module diffusers has no attribute MOSSImagePipeline`` -- true,
    and no help at all: it names neither the cause (the class lives in the
    repository) nor the remedy (allow-list it). The operator reads it as a
    broken model.

    So detect the case from the repo's own declaration. Allow-listed, it loads
    with its code; not allow-listed, it is refused by name with the setting to
    change, which is this module's whole policy.

    Empty dict when the declared class is available locally -- the overwhelming
    majority -- so an ordinary load never pays for this.
    """
    cls = declared_pipeline_class(repo)
    if not cls:
        return {}
    try:
        import diffusers
        if getattr(diffusers, cls, None) is not None:
            return {}
    except Exception:
        return {}
    require(cfg, repo, f"its own pipeline class {cls!r}")
    return {"custom_pipeline": repo, "trust_remote_code": True,
            "_declared_class": cls}
