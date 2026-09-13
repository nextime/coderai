# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Remote OpenAI backend — serve any text model from somewhere else.

Set ``service_url`` on a model entry and coderai stops loading it locally and
proxies to that endpoint instead: another coderai, a llama.cpp ``llama-server``,
a vLLM/SGLang/TGI server, a rented pod, or any hosted OpenAI-compatible API.
Nothing is downloaded and no VRAM is used here.

This is the piece that makes the GGUF/llama.cpp catalogue remotizable. Those
models load in-process through llama-cpp-python, so there was no service to
redirect; pointing the *model* at a URL sidesteps that entirely — the far side
runs llama.cpp (or anything else) and coderai is a client.

The request machinery is RunPod's — same OpenAI wire protocol, same SSE
streaming — so this subclasses it and only replaces endpoint resolution.

Model entry fields:
  service_url     base URL. With or without a trailing /v1 (it is added if missing).
  served_model    the name the remote knows the model by (default: the model's own).
  api_key         bearer token for the remote, when it wants one.
"""

import os
from typing import Dict, Optional

from codai.backends.runpod import RunpodBackend


def model_remote_url(model_name: str) -> str:
    """The ``service_url`` configured on this model's entry, or ''."""
    if not model_name:
        return ""
    try:
        from codai.models.manager import _model_entry_for
        entry = _model_entry_for(model_name) or {}
    except Exception:
        return ""
    return str(entry.get("service_url") or "").strip()


def remote_should_handle(model_name: str) -> bool:
    """True when this model is configured to be served from somewhere else."""
    return bool(model_remote_url(model_name))


class RemoteOpenAIBackend(RunpodBackend):
    """Proxy a model to any OpenAI-compatible endpoint."""

    # Its own in-flight counter: these requests run on someone else's hardware,
    # so like RunPod they must not count against the local concurrency gate —
    # but they also must not be mistaken for RunPod spend.
    _inflight = 0
    import threading as _threading
    _inflight_lock = _threading.Lock()
    del _threading

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self._mode = "remote"
        self._pool = None

    def load_model(self, model_name: str, **kwargs) -> None:
        """Resolve the remote endpoint for ``model_name``. Downloads nothing."""
        try:
            from codai.models.manager import _model_entry_for
            entry = _model_entry_for(model_name) or {}
        except Exception:
            entry = {}
        url = (str(entry.get("service_url") or "").strip()
               or os.environ.get("CODERAI_REMOTE_SERVICE_URL", "").strip())
        if not url:
            raise RuntimeError(f"no service_url configured for model {model_name!r}")

        self._model_id = model_name or "remote"
        self._served_name = (str(entry.get("served_model") or "").strip()
                             or model_name or "remote")
        url = url.rstrip("/")
        # Accept both "http://host:8000" and "http://host:8000/v1".
        self._url = url if url.endswith("/v1") else url + "/v1"

        key = str(entry.get("api_key") or "").strip() or os.environ.get("CODERAI_REMOTE_API_KEY", "")
        self._headers: Dict[str, str] = {"Authorization": f"Bearer {key}"} if key else {}

        ctx = kwargs.get("n_ctx", kwargs.get("ctx")) or entry.get("n_ctx")
        if isinstance(ctx, (list, tuple)):
            ctx = ctx[0] if ctx else None
        try:
            ctx = int(ctx or 0)
        except (TypeError, ValueError):
            ctx = 0
        if ctx > 0:
            self._ctx = ctx

        print(f"[remote] '{model_name}' -> {self._url} (served as {self._served_name!r})",
              flush=True)

    def _acquire_endpoint(self):
        """One fixed endpoint — nothing to provision or release."""
        if not self._url:
            raise RuntimeError("remote endpoint not resolved")
        return self._url, (lambda: None)

    def health(self) -> Optional[dict]:
        import requests
        try:
            r = requests.get(self._base() + "/models", headers=self._headers, timeout=10)
            return r.json() if r.ok else None
        except Exception:
            return None

    def cleanup(self) -> None:
        self._url = None
