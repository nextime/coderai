# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Assign each configured model to exactly one engine.

With multiple engines, every engine would otherwise read the shared models.json and
register *every* model — so a model would appear on several engines at once. The
front instead computes a single **owner** per model and tells each engine which
models it owns; the engine then registers only those.

Owner precedence (per model):
  1. The per-model ``engine`` pin (models.json), if that engine can run the model.
  2. The configured default engine, if it can run the model.
  3. Round-robin across the capability-compatible engines (balanced, deterministic),
     so unpinned models spread out instead of all landing on one engine.

A model whose format no engine can serve is left unassigned (it can't run anyway).
"""

import json

# models.json categories that hold servable model entries.
_CATEGORIES = (
    "text_models", "gguf_models", "vision_models", "image_models",
    "audio_models", "tts_models", "video_models", "audio_gen_models",
    "embedding_models", "spatial_models",
)


def _entry_path(entry):
    """The model's path/id — used for capability detection (e.g. is it a .gguf)."""
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        return entry.get("path") or entry.get("id")
    return None


def _route_key(entry):
    """The identifier clients address this entry by (alias > path > id).

    Keying on the alias lets two *configs* of the same model — with distinct
    aliases — be assigned to different engines; configs sharing a path with no
    distinct alias collapse to one owner (they're not separately addressable)."""
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        return entry.get("alias") or entry.get("path") or entry.get("id")
    return None


def _required_cap(entry, ds4_cfg, colibri_cfg=None):
    from codai.frontproxy.router import required_capability
    path = _entry_path(entry) or ""
    backend = entry.get("backend") if isinstance(entry, dict) else None
    return required_capability(
        path, backend=backend,
        ds4_model_id=getattr(ds4_cfg, "model_id", None) if ds4_cfg else None,
        ds4_enabled=bool(getattr(ds4_cfg, "enabled", False)) if ds4_cfg else False,
        colibri_model_id=getattr(colibri_cfg, "model_id", None) if colibri_cfg else None,
        colibri_enabled=bool(getattr(colibri_cfg, "enabled", False)) if colibri_cfg else False)


def compute_assignment(engines, models_path, default_engine=None, ds4_cfg=None,
                       colibri_cfg=None):
    """Return {engine_name: [model_identifiers]} — each model owned by one engine."""
    assignment = {e.name: [] for e in engines}
    if not engines or not models_path:
        return assignment
    try:
        with open(models_path) as f:
            data = json.load(f)
    except Exception:
        return assignment

    default_engine = (default_engine or "").strip().lower()
    rr = {}   # round-robin cursor per candidate-set signature
    seen = set()

    for cat in _CATEGORIES:
        for entry in data.get(cat, []):
            ident = _route_key(entry)
            if not ident or ident in seen:
                continue
            cap = _required_cap(entry, ds4_cfg, colibri_cfg)
            candidates = [e for e in engines if e.can_serve(cap)]
            if not candidates:
                continue   # nothing can run it — leave unassigned

            owner = None
            pin = ((entry.get("engine") if isinstance(entry, dict) else "") or "").strip().lower()
            if pin:
                owner = next((e for e in candidates
                              if e.name.lower() == pin or (e.backend or "").lower() == pin), None)
            if owner is None and default_engine:
                owner = next((e for e in candidates
                              if e.name.lower() == default_engine
                              or (e.backend or "").lower() == default_engine), None)
            if owner is None:
                key = tuple(sorted(e.name for e in candidates))
                i = rr.get(key, 0)
                owner = candidates[i % len(candidates)]
                rr[key] = i + 1

            assignment[owner.name].append(ident)
            seen.add(ident)

    return assignment
