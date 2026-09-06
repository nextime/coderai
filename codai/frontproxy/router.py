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
    "/v1/rerank",
    "/v1/images/generations",
    "/v1/images/edits",
    "/v1/audio/speech",
    "/v1/audio/transcriptions",
    "/v1/audio/diarization",
    "/v1/audio/speaker-embeddings",
    "/v1/audio/speakers",
    "/v1/audio/speaker-identify",
    "/v1/audio/speaker-verify",
    "/v1/video/generations",
}


def is_inference_path(path: str) -> bool:
    p = path.split("?", 1)[0].rstrip("/")
    return p in _INFERENCE_PATHS


def is_admin_path(path: str) -> bool:
    p = path.split("?", 1)[0]
    return (p.startswith("/admin") or p.startswith("/login") or p.startswith("/logout")
            or p == "/" or p.startswith("/static"))


_warned_pins: set = set()


def _warn_bad_pin(model, pinned, cap, engine, fallback: bool = False) -> None:
    key = (model, pinned, fallback)
    if key in _warned_pins:
        return
    _warned_pins.add(key)
    if engine is None:
        reason = f"no engine named/backed '{pinned}' is declared"
    elif not engine.is_alive():
        reason = f"engine '{pinned}' is down"
    elif not engine.can_serve(cap):
        reason = (f"engine '{pinned}' (backend '{engine.backend}') can't serve a "
                  f"'{cap}' model (capabilities: {sorted(engine.capabilities)})")
    else:
        reason = f"engine '{pinned}' is unavailable"
    tail = ("falling back to another compatible engine (engine_fallback)."
            if fallback else "failing the request (the pin is a hard constraint).")
    print(f"[front] WARNING: model '{model}' is pinned to '{pinned}' but {reason}; "
          f"{tail}", flush=True)


def required_capability(model: Optional[str], path: Optional[str] = None,
                        backend: Optional[str] = None,
                        ds4_model_id: Optional[str] = None,
                        ds4_enabled: bool = False,
                        colibri_model_id: Optional[str] = None,
                        colibri_enabled: bool = False,
                        k3_model_id: Optional[str] = None,
                        k3_enabled: bool = False,
                        kt_model_id: Optional[str] = None,
                        kt_enabled: bool = False,
                        vllm_model_id: Optional[str] = None,
                        vllm_enabled: bool = False) -> Optional[str]:
    """The capability an engine must have to serve this request.

    * ``whisper``      — whisper.cpp STT (transcription endpoint or a
                         ``whisper-server`` model). Runs on CUDA or Vulkan.
    * ``ds4``          — DeepSeek V4 via the native ds4 engine. CUDA-only.
    * ``colibri``      — GLM-5.2 / DeepSeek-V4 / Kimi-K3 via the native colibri C engine.
    * ``k3``           — Kimi-K3 via kimi-k3-in-c. CPU.
    * ``kt``           — many families via ktransformers/SGLang. CPU+GPU.
    * ``gguf``         — llama.cpp model. Runs on CUDA or Vulkan.
    * ``transformers`` — safetensors/HF model. CUDA-only.

    This mirrors the in-process resolver (:func:`codai.models.manager.resolve_engine_backend`):
    an explicit ``backend`` pin is authoritative, then an enabled engine's ``model_id``
    alias, then an unambiguous name marker (a Kimi name claimed by BOTH colibri and k3
    falls through — the model must pin a backend)."""
    p = (path or "").split("?", 1)[0].rstrip("/")
    if p in ("/v1/audio/transcriptions", "/v1/audio/diarization",
             "/v1/audio/speaker-embeddings", "/v1/audio/speakers",
             "/v1/audio/speaker-identify", "/v1/audio/speaker-verify") \
            or (backend or "") == "whisper-server":
        return "whisper"
    # 1. Explicit engine-backend pin wins (authoritative), like the manager resolver.
    b = (backend or "").lower()
    if b in ("colibri", "ds4", "k3", "kt", "vllm", "runpod"):
        return b
    m = (model or "").lower()

    def _alias(mid: Optional[str]) -> bool:
        mid = (mid or "").lower()
        return bool(mid) and (m == mid or m.split("/")[-1] == mid)

    # 2. An enabled engine's model_id alias.
    if vllm_enabled and _alias(vllm_model_id):
        return "vllm"
    if kt_enabled and _alias(kt_model_id):
        return "kt"
    if k3_enabled and _alias(k3_model_id):
        return "k3"
    if colibri_enabled and _alias(colibri_model_id):
        return "colibri"
    if ds4_enabled and _alias(ds4_model_id):
        return "ds4"

    # 3. Name markers (unambiguous only). GLM → colibri; DeepSeek → ds4 (default owner);
    #    Kimi → colibri XOR k3 (ambiguous when both enabled → fall through to a pin).
    if m:
        if colibri_enabled and ("glm-5.2" in m or "glm5.2" in m or "colibri" in m):
            return "colibri"
        if "kimi-k3" in m or "kimi_k3" in m or "kimik3" in m:
            claimers = [c for c, en in (("colibri", colibri_enabled), ("k3", k3_enabled)) if en]
            if len(claimers) == 1:
                return claimers[0]
        if ds4_enabled and "deepseek-v4" in m:
            return "ds4"
    if m.endswith(".gguf") or "gguf" in m:
        return "gguf"
    if not model:
        return None
    return "transformers"


def pick_engine(registry: EngineRegistry, path: str, method: str,
                model: Optional[str], required_cap: Optional[str] = None,
                default_engine: Optional[str] = None,
                pinned: Optional[str] = None,
                pin_fallback: bool = False) -> Optional[Engine]:
    """Return the engine to proxy this request to, or None if none are ready.

    Precedence for inference: per-model pin → engine already holding the model →
    configured default engine → least-loaded compatible engine. Each candidate must
    be capability-compatible (``required_cap``) and healthy. Works even when the
    model id isn't known (e.g. a multipart transcription upload), routing purely by
    capability.
    """
    if method.upper() == "POST" and is_inference_path(path):
        cap = required_cap

        # Speaker diarization (pyannote) must run on a CUDA engine to use the GPU.
        # When the request carries no model to pin it, prefer the primary (nvidia)
        # so it doesn't land on a Vulkan/CPU engine and run diarization on CPU. An
        # explicit model pin (below) still wins.
        if not pinned and not model and (path or "").split("?", 1)[0].rstrip("/") \
                in ("/v1/audio/diarization", "/v1/audio/speaker-embeddings",
                    "/v1/audio/speakers", "/v1/audio/speaker-identify",
                    "/v1/audio/speaker-verify"):
            prim = registry.primary()
            if prim is not None and prim.can_serve(cap) and prim.is_alive():
                return prim

        # 0. Per-model pin (models.json "engine") is a HARD constraint: the model
        # runs on that engine or not at all. Route there if it can serve and its
        # process is alive — even when it's busy (mid-generation → failing health
        # polls), so the request queues on its gen-lock rather than spawning a
        # duplicate elsewhere with the wrong settings. If the pinned engine is down
        # or incompatible, fail (return None → 503) instead of falling back to a
        # different engine.
        if pinned:
            e = registry.by_name(pinned)
            if e is not None and e.can_serve(cap) and e.is_alive():
                return e
            # Co-located GGUF-isolation sibling: the split runs a "<name>-gguf"
            # engine on the SAME card. A model pinned to the torch engine but needing
            # `gguf` (or pinned to the gguf engine but needing transformers) should
            # transparently use whichever sibling can serve it — they share the GPU,
            # so the pin's intent (that physical card) is still honoured. Not a 503.
            sib_name = (pinned[:-5] if pinned.endswith("-gguf") else f"{pinned}-gguf")
            sib = registry.by_name(sib_name)
            if sib is not None and sib.can_serve(cap) and sib.is_alive():
                return sib
            _warn_bad_pin(model, pinned, cap, e, fallback=pin_fallback)
            # Default: a hard pin fails rather than running elsewhere. When the
            # model opts in (engine_fallback), fall through to pick another engine.
            if not pin_fallback:
                return None

        # 1. The front's precomputed assignment is authoritative — it already folds
        # in the default engine and balanced auto-selection, and keeps a model on
        # exactly one engine. Honour it first when it's compatible.
        if model:
            owner = registry.engine_for_assigned(model)
            if owner is not None and owner.can_serve(cap):
                return owner

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

        # 5. Least-loaded compatible engine. A capability-BLIND fallback
        # (least_loaded(None)) is permitted ONLY for a request that needs no
        # specific capability — a TYPED request (`transformers` video/image,
        # `gguf`, `whisper`, …) must never run on an engine that lacks that
        # capability. Otherwise the torch/GGUF process isolation breaks: when the
        # nvidia (transformers) engine is briefly down mid-restart, a video request
        # would otherwise land on the gguf-only sibling and run a torch pipeline
        # there. Instead prefer the primary when IT can serve the cap (the request
        # queues there / the caller retries as it comes back), else 503.
        best = registry.least_loaded(cap)
        if best is not None:
            return best
        if cap is None:
            return registry.least_loaded(None) or registry.primary()
        prim = registry.primary()
        return prim if (prim is not None and prim.can_serve(cap)) else None

    # Admin/auth/config/UI and everything else → primary (consistent sessions).
    return registry.primary() or registry.least_loaded()
