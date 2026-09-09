# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Compatibility shims for huggingface_hub API drift.

``huggingface_hub`` 1.x dropped ``proxies`` and ``resume_download`` from the arguments
that :class:`PyTorchModelHubMixin.from_pretrained` forwards to a subclass's
``_from_pretrained``. Models that vendor an older mixin signature — notably **BigVGAN**,
which both seed-vc and F5-TTS pull in as their vocoder — declare those two as *required*
keyword-only parameters, so loading them raises::

    BigVGAN._from_pretrained() missing 2 required keyword-only arguments:
    'proxies' and 'resume_download'

Pinning huggingface_hub back below 1.0 is not an option here: transformers 5.x needs the
new one. So instead we fill the gap at the call boundary — for the specific subclass being
loaded, and only when its signature actually still requires the dropped arguments.
"""

import inspect

_LEGACY_KWARGS = {"proxies": None, "resume_download": None}
_installed = False


def install_hub_legacy_kwargs_shim() -> bool:
    """Make ``PyTorchModelHubMixin.from_pretrained`` tolerate pre-1.0 subclasses.

    Idempotent, and a no-op on hub versions that still pass the legacy arguments (the
    shim only fires for a subclass whose ``_from_pretrained`` requires something the
    installed hub does not supply). Returns True when the shim is in place.
    """
    global _installed
    if _installed:
        return True
    try:
        from huggingface_hub import PyTorchModelHubMixin
    except ImportError:
        return False

    original = PyTorchModelHubMixin.from_pretrained

    def _patch_subclass(cls) -> None:
        """Give ``cls._from_pretrained`` defaults for the arguments hub stopped sending."""
        if getattr(cls, "_codai_legacy_kwargs_patched", False):
            return
        bound = cls._from_pretrained
        try:
            params = inspect.signature(bound).parameters
        except (TypeError, ValueError):
            return
        missing = {
            name: default for name, default in _LEGACY_KWARGS.items()
            if name in params and params[name].default is inspect.Parameter.empty
        }
        if not missing:
            cls._codai_legacy_kwargs_patched = True
            return

        def _from_pretrained(**kwargs):
            for name, default in missing.items():
                kwargs.setdefault(name, default)
            return bound(**kwargs)

        cls._from_pretrained = staticmethod(_from_pretrained)
        cls._codai_legacy_kwargs_patched = True
        print(f"[hub-compat] {cls.__name__}._from_pretrained: supplied "
              f"{sorted(missing)} dropped by huggingface_hub", flush=True)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        try:
            _patch_subclass(cls)
        except Exception:
            pass          # never let the shim itself break a working load
        return original.__func__(cls, *args, **kwargs)

    PyTorchModelHubMixin.from_pretrained = from_pretrained
    _installed = True
    return True
