# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Decide which engine handles a proxied request.

Policy (Plan B + multi-engine):

* **Admin / auth / config / UI / status / tasks** → the **primary** engine. These
  own per-process session and config state, so pinning them to one engine keeps
  sessions consistent without a shared session store (that's Plan C).
* **Inference** (``/v1/...`` POST carrying a ``model``) → the engine that already
  has that model resident; otherwise the least-loaded engine (which loads it on
  demand). This is what lets one model load on engine A while engine B keeps
  generating.
* **Everything else** (e.g. ``GET /v1/models``, file downloads) → primary.
"""

from typing import Optional

from codai.frontproxy.registry import Engine, EngineRegistry

# POST endpoints that carry a `model` and should be load-balanced across engines.
_INFERENCE_PATHS = {
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/embeddings",
    "/v1/images/generations",
    "/v1/images/edits",
    "/v1/audio/speech",
    "/v1/audio/transcriptions",
    "/v1/videos/generations",
}


def is_inference_path(path: str) -> bool:
    p = path.split("?", 1)[0].rstrip("/")
    return p in _INFERENCE_PATHS


def is_admin_path(path: str) -> bool:
    p = path.split("?", 1)[0]
    return (p.startswith("/admin") or p.startswith("/login") or p.startswith("/logout")
            or p == "/" or p.startswith("/static"))


_warned_pins: set = set()


def _warn_bad_pin(model, pinned, cap, engine) -> None:
    key = (model, pinned)
    if key in _warned_pins:
        return
    _warned_pins.add(key)
    if engine is None:
        reason = f"no engine named/backed '{pinned}' is declared"
    elif not engine.healthy:
        reason = f"engine '{pinned}' is not healthy"
    else:
        reason = (f"engine '{pinned}' (backend '{engine.backend}') can't serve a "
                  f"'{cap}' model (capabilities: {sorted(engine.capabilities)})")
    print(f"[front] WARNING: model '{model}' is pinned to '{pinned}' but {reason}; "
          f"falling back to a compatible engine.", flush=True)


def required_capability(model: Optional[str], path: Optional[str] = None,
                        backend: Optional[str] = None,
                        ds4_model_id: Optional[str] = None,
                        ds4_enabled: bool = False) -> Optional[str]:
    """The capability an engine must have to serve this request.

    * ``whisper``      — whisper.cpp STT (transcription endpoint or a
                         ``whisper-server`` model). Runs on CUDA or Vulkan.
    * ``ds4``          — DeepSeek V4 via the native ds4 engine. CUDA-only.
    * ``gguf``         — llama.cpp model. Runs on CUDA or Vulkan.
    * ``transformers`` — safetensors/HF model. CUDA-only.

    Signals are combined: the request path and the model's configured ``backend``
    (from models.json) take precedence over the name heuristic, so whisper works
    even when the model id isn't in the request body (multipart upload)."""
    p = (path or "").split("?", 1)[0].rstrip("/")
    if p == "/v1/audio/transcriptions" or (backend or "") == "whisper-server":
        return "whisper"
    m = (model or "").lower()
    if ds4_enabled and m:
        mid = (ds4_model_id or "").lower()
        if (mid and (m == mid or m.split("/")[-1] == mid)) or "deepseek-v4" in m:
            return "ds4"
    if m.endswith(".gguf") or "gguf" in m:
        return "gguf"
    if not model:
        return None
    return "transformers"


def pick_engine(registry: EngineRegistry, path: str, method: str,
                model: Optional[str], required_cap: Optional[str] = None,
                default_engine: Optional[str] = None,
                pinned: Optional[str] = None) -> Optional[Engine]:
    """Return the engine to proxy this request to, or None if none are ready.

    Precedence for inference: per-model pin → engine already holding the model →
    configured default engine → least-loaded compatible engine. Each candidate must
    be capability-compatible (``required_cap``) and healthy. Works even when the
    model id isn't known (e.g. a multipart transcription upload), routing purely by
    capability.
    """
    if method.upper() == "POST" and is_inference_path(path):
        cap = required_cap

        # 0. The front's precomputed assignment is authoritative — it already folds
        # in the pin, the default engine, and balanced auto-selection, and is what
        # keeps a model on exactly one engine. Honour it first when it's compatible.
        if model:
            owner = registry.engine_for_assigned(model)
            if owner is not None and owner.can_serve(cap):
                return owner

        # 1. Per-model pin (models.json "engine") — only honoured if compatible.
        if pinned:
            e = registry.by_name(pinned)
            if e and e.can_serve(cap):
                if e.healthy:
                    return e
                # Pinned engine is busy (mid-generation → failing health polls) but
                # its process is alive: route here anyway so the request queues on
                # its gen-lock, instead of duplicating a pinned model on another
                # engine (which also ignores its configured n_ctx etc.).
                if e.is_alive():
                    return e
            # Pin can't be honoured (engine down or incompatible) — say why (once
            # per model+engine) instead of silently falling back.
            _warn_bad_pin(model, pinned, cap, e)

        # 2. Engine that already has the model resident.
        if model:
            e = registry.engine_for_model(model, cap)
            if e:
                return e

        # 3. The assigned owner, busy-but-alive: prefer queueing on the engine that
        # owns this model over loading a second copy on a different one.
        if model:
            owner = registry.engine_owning(model)
            if owner is not None and owner.can_serve(cap) and owner.is_alive():
                return owner

        # 4. Configured default engine, when it can serve this request.
        if default_engine:
            e = registry.by_name(default_engine)
            if e and e.healthy and e.can_serve(cap):
                return e

        # 5. Least-loaded compatible engine; then any engine rather than 503.
        return (registry.least_loaded(cap)
                or registry.least_loaded(None)
                or registry.primary())

    # Admin/auth/config/UI and everything else → primary (consistent sessions).
    return registry.primary() or registry.least_loaded()
