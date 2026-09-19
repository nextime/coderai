# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Which engine can run which backend — one table, used twice.

The Models page narrows its Engine / card and Backend selects to what fits
(codai/admin/templates/models.html, ``_applyCompat``) and the save path
refuses what does not (codai/admin/routes.validate_engine_pin). Both follow
the router's notion of a *required capability*, so a pin that passes here is
one the front will actually honour instead of quietly falling back.
"""

from typing import Iterable, Optional, Tuple

#: Model backend (the Backend select) → capability an engine must offer.
#: None = decided by the path (gguf → "gguf", else "transformers");
#: "" = no engine requirement at all (placement is elsewhere).
BACKEND_CAP = {
    "auto": None, "nvidia": None, "cpu": None,
    "vulkan": "gguf", "opencl": "gguf",
    "colibri": "colibri", "ds4": "ds4", "k3": "k3", "kt": "kt", "vllm": "vllm",
    "runpod": "runpod", "host": "",
    "wav2vec2": "transformers", "whisper-hf": "transformers", "nemo": "transformers",
    "vosk": "", "whisper-server": "whisper",
}

#: Backends that name a GPU vendor: the engine's own backend must match or the
#: model silently runs on a different API than the one it asked for.
VENDOR_BACKENDS = {
    "nvidia": ("nvidia", "cuda", "auto"),
    "vulkan": ("vulkan", "auto"),
    "opencl": ("opencl", "auto"),
}


def required_cap(model_backend: str, path: str = "") -> Optional[str]:
    b = (model_backend or "auto").strip().lower()
    cap = BACKEND_CAP.get(b, None)
    if cap is None:
        p = (path or "").lower()
        return "gguf" if (p.endswith(".gguf") or "gguf" in p) else "transformers"
    return cap or None


def check(engine_backend: str, engine_caps: Iterable[str], model_backend: str,
          path: str = "") -> Tuple[bool, str]:
    """(ok, reason). ``ok`` False = the pin cannot work; a non-empty reason with
    ``ok`` True is a warning (it runs, but not the way the backend says)."""
    b = (model_backend or "auto").strip().lower()
    if b == "host":
        return False, "a host model runs on the machine named in its host block, not on an engine"
    cap = required_cap(b, path)
    caps = {str(c).lower() for c in (engine_caps or ())}
    eb = (engine_backend or "auto").lower()
    if cap and cap not in caps:
        return False, (f"needs '{cap}' but this engine ({eb}) only provides "
                       f"{sorted(caps) or '(nothing)'}")
    vend = VENDOR_BACKENDS.get(b)
    if vend and eb not in vend and not eb.startswith("nvidia"):
        return True, (f"backend '{b}' on a {eb} engine: the engine's own API "
                      f"is used instead")
    if vend and b in ("vulkan", "opencl") and eb.startswith("nvidia"):
        return True, f"backend '{b}' on the NVIDIA engine runs through CUDA"
    return True, ""
