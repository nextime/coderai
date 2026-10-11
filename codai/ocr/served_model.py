# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
# GPLv3 - see the project LICENSE.

"""An OpenAI endpoint for a model coderai already serves, wherever it is serving it.

Surya-2 and olmOCR are VLMs: the OCR engine is a thin client and the actual work
happens in a model server. Until now an install could point an OCR engine at coderai's
OWN vLLM (``ocr.*_serve = "vllm"``, an isolated venv on this box) or at a URL typed in
by hand (``"llamacpp"``/``"server"``). Both assume the model server is somewhere the
operator can name.

On a GPU-less orchestrator neither holds. The Digesta install is the case that found
it: ``ru002848`` has no GPU at all — coderai's whole job there is to rent A40 pods on
demand — and the OCR VLM is already a first-class model in ``models.json`` with
``backend: runpod``. Asking for OCR built a local vLLM venv that could never run,
failed with ``NO_VLLM_VENV``, and the document went unread, while the pod that would
have served it in sixty seconds was never asked for.

So this module answers one question — *where is model X being served, and with what
key* — and lets whichever backend owns the model answer it: a RunPod pod pool
provisions one and reports its proxy URL, a model pinned to another coderai reports
its ``service_url``, a local vLLM model reports its own instance. The OCR engine then
attaches over HTTP exactly as it would to a hand-typed URL, and the renting, the
scaling, the budget ledger and the idle reaper stay where they belong.

The endpoint is a lease, not an address: a scale-to-zero pod is reaped when OCR goes
quiet, and the caller is expected to re-resolve rather than to trust it forever.
"""

import threading

from codai.ocr.base import OcrError

_lock = threading.Lock()
_backends: dict = {}            # model id -> a loaded backend that knows its endpoint


def _entry_backend(model_id: str):
    """A loaded backend for ``model_id``, or an OcrError saying why there is none.

    Only the backends that END UP somewhere addressable can answer: a model loaded
    into this process has no URL, and inventing a loopback to our own ``/v1`` would
    hide a misconfiguration behind an extra hop.
    """
    from codai.models import manager as mm

    try:
        from codai.backends import remote_openai
        if remote_openai.remote_should_handle(model_id):
            backend = remote_openai.RemoteOpenAIBackend()
            backend.load_model(model_id)
            return backend
    except OcrError:
        raise
    except Exception as exc:
        raise OcrError(f"model {model_id!r}: remote endpoint unusable: {exc}", status=503)

    try:
        if mm.runpod_should_handle(model_id):
            from codai.backends.runpod import RunpodBackend
            backend = RunpodBackend(mm.get_active_runpod_config())
            backend.load_model(model_id)
            return backend
    except Exception as exc:
        raise OcrError(f"model {model_id!r}: RunPod endpoint unusable: {exc}", status=503)

    if mm.resolve_engine_backend(model_id) == "vllm":
        try:
            from codai.api import runpod_worker, vllm_worker
            from codai.models.manager import get_active_vllm_config
            vcfg = get_active_vllm_config()
            path = (runpod_worker._model_path_for(model_id) or "").strip() or model_id
            base = vllm_worker.ensure_service(vcfg, model_path=path, served_name=model_id)
            return _Fixed(base.rstrip("/") + "/v1", "", model_id)
        except Exception as exc:
            raise OcrError(f"model {model_id!r}: local vLLM unusable: {exc}", status=503)

    raise OcrError(
        f"model {model_id!r} is not served over HTTP by this install, so an OCR engine "
        f"cannot attach to it. Pin the model to a backend that has an endpoint "
        f"(backend: \"runpod\", or a service_url pointing at a server that holds it), "
        f"or pick a serve mode that fits this box.",
        status=503,
    )


class _Fixed:
    """A backend-shaped answer for the cases that resolve to one fixed URL."""

    def __init__(self, url: str, key: str, served: str):
        self._url, self._key, self._served = url, key, served

    def openai_endpoint(self):
        return self._url, self._key

    @property
    def served_name(self) -> str:
        return self._served


def _served_name_of(backend, model_id: str) -> str:
    for attr in ("served_name", "_served_name"):
        name = getattr(backend, attr, "")
        if isinstance(name, str) and name.strip():
            return name.strip()
    return model_id


def endpoint_for(model_id: str):
    """``(base_url ending /v1, api_key, served_name)`` for ``model_id``.

    Blocks while a cold pool rents and boots a pod — a first OCR request after an idle
    night legitimately waits for a GPU to exist. Raises :class:`OcrError` (503) with a
    message that says what to configure, because that error is what reaches the client.
    """
    model_id = (model_id or "").strip()
    if not model_id:
        raise OcrError("no model id configured for the served OCR engine", status=400)

    with _lock:
        backend = _backends.get(model_id)
    if backend is None:
        backend = _entry_backend(model_id)
        with _lock:
            _backends.setdefault(model_id, backend)
            backend = _backends[model_id]

    try:
        base, key = backend.openai_endpoint()
    except OcrError:
        raise
    except Exception as exc:
        raise OcrError(f"model {model_id!r}: no endpoint available ({exc})", status=503)
    if not base:
        raise OcrError(f"model {model_id!r}: backend reported no endpoint", status=503)
    return base, key, _served_name_of(backend, model_id)


def forget(model_id: str = "") -> None:
    """Drop cached backends so the next resolve starts over (config changed, or a
    pool that no longer answers). A pool itself is global to the process, so this
    costs nothing but a re-``load_model``."""
    with _lock:
        if model_id:
            _backends.pop(model_id.strip(), None)
        else:
            _backends.clear()


def health_url(base: str) -> str:
    """The ``/health`` base an attached client probes: the OpenAI base without ``/v1``.

    Surya's attach path refuses a URL whose ``/health`` does not answer 200, which is
    also the cheapest way for us to find out that a leased pod has gone away.
    """
    base = (base or "").rstrip("/")
    return base[:-len("/v1")].rstrip("/") if base.endswith("/v1") else base
