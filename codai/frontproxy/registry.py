# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Front-side registry of engine subprocesses.

The front never imports torch; it knows about engines only through the small,
auth-free ``/internal/engine-state`` endpoint each engine exposes on localhost.
This module holds the shared, thread-safe view the supervisor writes and the
router/aggregator read.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


def _short_stem(key: str) -> str:
    """Short name for a routable key, with a trailing ``.gguf`` stripped.

    A gguf model's assigned/loaded key is its file path, but ``/v1/models``
    advertises it (and clients address it) by the filename *without* ``.gguf`` —
    the automatic alias. Normalizing both sides here lets that alias resolve to
    the owning engine without the user setting an explicit alias."""
    short = key.split("/")[-1].split(":")[-1]
    if short.lower().endswith(".gguf"):
        short = short[:-5]
    return short


# Default model-format capabilities implied by an engine's backend:
#   transformers — safetensors/HF models (CUDA only here)
#   gguf         — llama.cpp models (CUDA or Vulkan)
#   whisper      — whisper.cpp STT (CUDA or Vulkan)
#   ds4          — DeepSeek V4 via the native ds4 engine (CUDA-only build)
# An NVIDIA engine can do all of them; a Vulkan (e.g. Radeon) engine does GGUF and
# whisper, but not transformers and not ds4.
_DEFAULT_CAPS = {
    "nvidia": {"transformers", "gguf", "whisper", "ds4"},
    "cuda": {"transformers", "gguf", "whisper", "ds4"},
    "vulkan": {"gguf", "whisper"},
    "opencl": {"gguf", "whisper"},
    "auto": {"transformers", "gguf", "whisper", "ds4"},
}


@dataclass
class Engine:
    id: int
    gpu: Optional[int]             # device hint for logs (CUDA/Vulkan index; None = n/a)
    port: int
    primary: bool = False          # the engine that owns admin/auth/config traffic
    role: str = "engine"           # "engine" (GPU/inference) or "system" (cache/downloads worker)
    name: str = ""                 # human label for logs
    backend: str = "auto"          # nvidia | vulkan | … (forced for this engine)
    env: dict = field(default_factory=dict)        # extra env applied at spawn
    capabilities: Set[str] = field(default_factory=set)  # model formats it can serve
    assigned_models: Set[str] = field(default_factory=set)  # routable ids it owns
    url: str = ""
    healthy: bool = False
    loaded_models: Set[str] = field(default_factory=set)
    # Per-model memory info from /internal/engine-state:
    # [{model, vram_gb, ram_gb, device}, …] for the engines-card tooltip.
    loaded_info: list = field(default_factory=list)
    vram: Optional[dict] = None
    tasks: list = field(default_factory=list)   # running/queued tasks on this engine
    cooling: Optional[dict] = None  # thermal cooldown state, or None when not cooling
    loading: Optional[dict] = None  # model-load progress parsed from logs (or None);
                                    # surfaced as a synthetic task while the engine's
                                    # event loop is GIL-blocked and can't be polled
    last_ok: float = 0.0           # monotonic time of last successful poll
    proc: object = None            # subprocess.Popen (set by the supervisor)
    draining: bool = False         # restart pending: stop routing NEW requests here
                                    # and let in-flight ones finish (drain grace period)
    inflight: int = 0              # proxied requests currently streaming through
    # Front-side thermal supervision bookkeeping (written by the EngineSupervisor's
    # thermal monitor; NOT reported by the engine). therm_temp is this engine's
    # hottest owned card.
    therm_temp: Optional[float] = None
    therm_paused: bool = False     # front asked this engine to pause (cooperatively)
    therm_sigstopped: bool = False  # front escalated to an OS-level SIGSTOP
    therm_stop_checks: int = 0     # consecutive checks the engine stayed busy after a pause
    _inflight_lock: object = field(default_factory=threading.Lock, repr=False, compare=False)
    # In-flight request metadata {rid: {"model","kind","path","started_at"}} so the
    # front can synthesize Tasks-page entries for work it dispatched — visible even
    # when the engine is too GIL-busy generating to answer its own /admin/api/tasks.
    active: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        if not self.url:
            self.url = f"http://127.0.0.1:{self.port}"
        if not self.name:
            self.name = f"engine#{self.id}"
        if not self.capabilities:
            self.capabilities = set(_DEFAULT_CAPS.get(self.backend, {"transformers", "gguf"}))

    def can_serve(self, required_cap: Optional[str]) -> bool:
        # The system worker (cache/downloads) never serves inference, so it must
        # never be picked as an inference target — even for cap-less requests.
        if self.role == "system":
            return False
        return (not required_cap) or (required_cap in self.capabilities)

    def is_alive(self) -> bool:
        """Process is up (so 'unhealthy' means busy/GIL-blocked, not dead).

        An engine mid-generation can't answer the health poll and reads as
        unhealthy, but it's the right place to send a request pinned/assigned to
        it — the request queues on its gen-lock instead of duplicating the model
        elsewhere. A None proc means externally managed; assume alive.

        While draining (a restart is pending) it reports not-alive so the router
        diverts new traffic elsewhere and the existing requests can finish."""
        if self.draining:
            return False
        p = self.proc
        try:
            return p is None or p.poll() is None
        except Exception:
            return True

    def enter_request(self, meta: Optional[dict] = None) -> Optional[str]:
        """Mark a request in-flight. Returns a request id to pass to exit_request.
        ``meta`` (model/kind/path) is stored so the front can show a synthetic task
        for it while the engine itself can't answer the Tasks poll."""
        import time as _t, uuid as _u
        rid = _u.uuid4().hex[:12]
        with self._inflight_lock:
            self.inflight += 1
            if meta is not None:
                m = dict(meta)
                m["started_at"] = _t.time()
                self.active[rid] = m
        return rid

    def exit_request(self, rid: Optional[str] = None) -> None:
        with self._inflight_lock:
            if self.inflight > 0:
                self.inflight -= 1
            if rid:
                self.active.pop(rid, None)


class EngineRegistry:
    def __init__(self):
        self._engines: Dict[int, Engine] = {}
        self._lock = threading.RLock()

    def add(self, engine: Engine) -> None:
        with self._lock:
            self._engines[engine.id] = engine

    def get(self, engine_id: int) -> Optional[Engine]:
        with self._lock:
            return self._engines.get(engine_id)

    def all(self) -> List[Engine]:
        with self._lock:
            return list(self._engines.values())

    def healthy(self) -> List[Engine]:
        with self._lock:
            return [e for e in self._engines.values() if e.healthy]

    def primary(self) -> Optional[Engine]:
        """The engine that owns admin/session/config — falls back to first healthy."""
        with self._lock:
            prim = next((e for e in self._engines.values() if e.primary), None)
            if prim and prim.healthy:
                return prim
            return next((e for e in self._engines.values() if e.healthy), prim)

    def by_name(self, name: Optional[str]) -> Optional[Engine]:
        """Resolve an engine by its declared name (or, failing that, its backend).

        Used for the configured default engine and per-model pins. Prefers a healthy
        match but returns an unhealthy one too, so callers can decide."""
        if not name:
            return None
        name = name.strip().lower()
        with self._lock:
            engines = list(self._engines.values())
        match = None
        for e in engines:
            if (e.name or "").lower() == name or (e.backend or "").lower() == name:
                if e.healthy:
                    return e
                match = match or e
        return match

    def update_state(self, engine_id: int, *, healthy: bool,
                     loaded_models=None, loaded_info=None, vram=None,
                     tasks=None, cooling=False) -> None:
        with self._lock:
            e = self._engines.get(engine_id)
            if not e:
                return
            if e.draining:        # a restart is pending — stay out of rotation
                healthy = False
            e.healthy = healthy
            if healthy:
                e.last_ok = time.monotonic()
            if loaded_models is not None:
                e.loaded_models = set(loaded_models)
            if loaded_info is not None:
                e.loaded_info = list(loaded_info)
            if vram is not None:
                e.vram = vram
            if tasks is not None:
                e.tasks = list(tasks)
            elif not healthy:
                e.tasks = []
            if cooling is not False:        # explicit None clears it
                e.cooling = cooling
            elif not healthy:
                e.cooling = None

    def engine_for_model(self, model_key: str, required_cap: Optional[str] = None) -> Optional[Engine]:
        """Return a healthy, capability-compatible engine that already has the model
        resident, if any.

        Matching is forgiving: exact key, short-name, or type-prefixed variants —
        the same fuzzy spirit the manager uses, but read-only over loaded keys."""
        if not model_key:
            return None
        short = _short_stem(model_key)
        with self._lock:
            for e in self._engines.values():
                if not e.healthy or not e.can_serve(required_cap):
                    continue
                for k in e.loaded_models:
                    if k == model_key or _short_stem(k) == short \
                            or k.endswith(model_key) or model_key.endswith(k.split(":")[-1]):
                        return e
        return None

    def engine_for_assigned(self, model_key: str) -> Optional[Engine]:
        """The engine the front ASSIGNED this model to (single owner), or None.

        The assignment is the authoritative routing decision (it already encodes
        pins, the default engine, and balanced auto-selection); match leniently so a
        short-name / alias resolves to the owner."""
        if not model_key:
            return None
        short = _short_stem(model_key)
        with self._lock:
            for e in self._engines.values():
                if not e.healthy:
                    continue
                for k in e.assigned_models:
                    if (k == model_key or _short_stem(k) == short
                            or k.endswith(model_key) or model_key.endswith(k.split("/")[-1])):
                        return e
        return None

    def engine_owning(self, model_key: str) -> Optional[Engine]:
        """The engine ASSIGNED this model, regardless of current health.

        Like engine_for_assigned but without the healthy filter — so a request can
        be routed to its owner while the owner is transiently busy (mid-generation,
        failing health polls) and queue there, rather than spawning a duplicate on
        another engine. Callers should gate on ``is_alive()``."""
        if not model_key:
            return None
        short = _short_stem(model_key)
        with self._lock:
            for e in self._engines.values():
                for k in e.assigned_models:
                    if (k == model_key or _short_stem(k) == short
                            or k.endswith(model_key) or model_key.endswith(k.split("/")[-1])):
                        return e
        return None

    def least_loaded(self, required_cap: Optional[str] = None) -> Optional[Engine]:
        """Pick a healthy, capability-compatible engine to load a new model on:
        fewest resident models, then most free VRAM."""
        with self._lock:
            cands = [e for e in self._engines.values()
                     if e.healthy and e.can_serve(required_cap)]
        if not cands:
            return None

        def _free(e: Engine) -> float:
            return (e.vram or {}).get("free", 0.0) if e.vram else 0.0

        cands.sort(key=lambda e: (len(e.loaded_models), -_free(e)))
        return cands[0]
