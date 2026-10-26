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
