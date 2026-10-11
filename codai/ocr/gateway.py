# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
# GPLv3 - see the project LICENSE.

"""One stable OpenAI address for an OCR engine, fanned out per request.

An OCR pipeline (surya-2, olmOCR) is given ONE server address and drives it
itself. That is deliberate: the engine sends multipart images, a strict
``json_schema`` and ``logprobs: true``, and every one of those was quietly lost
the last time OCR went through coderai's backend abstraction, which parses a
request and re-emits it.

This module keeps the engine's single address while letting the work spread, by
being a hop that *decides nothing about the request*: the body goes upstream as
the bytes that arrived, and the reply comes back as the bytes that were sent. The
only decision is WHERE, taken per request by :mod:`codai.ocr.dispatch`, which can
pick the local server, the least-busy cluster node, or a pod from the pool —
counting the request against it so the pool grows under real load.

The three routes are exactly what surya probes on attach (see
``surya/inference/backends/spawn.py``):

* ``GET  <root>/health``                 — 200, or it refuses to attach at all;
* ``GET  <root>/v1/models``              — ``data[0].id`` must EQUAL the engine's
  configured checkpoint, or it raises a model-mismatch error;
* ``POST <root>/v1/chat/completions``    — the work.

The lease in the path is a secret generated per engine attachment, and it is also
the bearer token the engine is given, so the route is not reachable by anything
that was not handed both.
"""

import os
import secrets
import threading
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

router = APIRouter()

#: Path the engine attaches under. The OpenAI base it is given is
#: ``<self>/v1/ocr/engine/<lease>/v1`` so that stripping ``/v1`` — which is how
#: surya derives the health URL — lands on ``<self>/v1/ocr/engine/<lease>``.
PREFIX = "/v1/ocr/engine"

#: How long one page may take upstream. A dense scan on a cold pod legitimately
#: takes minutes; a timeout here would look to the engine like a failed page.
UPSTREAM_TIMEOUT_S = 3600.0

_lock = threading.Lock()
_leases: dict = {}              # lease token -> {"model_id", "served_name"}


def issue(model_id: str, served_name: str) -> str:
    """Mint a lease for one engine attachment and return its token."""
    token = secrets.token_urlsafe(24)
    with _lock:
        _leases[token] = {"model_id": model_id, "served_name": served_name}
    return token


def revoke(model_id: str = "") -> None:
    with _lock:
        if not model_id:
            _leases.clear()
            return
        for tok in [t for t, v in _leases.items() if v["model_id"] == model_id]:
            _leases.pop(tok, None)


def lease(token: str) -> Optional[dict]:
    with _lock:
        got = _leases.get((token or "").strip())
        return dict(got) if got else None


def token_is_valid(token: str) -> bool:
    """Whether a bearer token is a live OCR lease.

    The auth middleware asks this: an attached engine has no account and no API
    token, and the lease is the credential it was given.
    """
    return lease(token) is not None


def self_base() -> str:
    """This install's own base URL, for an address we hand to a local engine.

    Set at bind time (``CODERAI_SELF_BASE``). The engine runs as a subprocess on
    this machine, so loopback is both correct and the only thing that needs no
    certificate.
    """
    base = (os.environ.get("CODERAI_SELF_BASE") or "").strip().rstrip("/")
    if base:
        return base
    port = (os.environ.get("CODERAI_PORT") or "").strip()
    return f"http://127.0.0.1:{port or '8000'}"


def engine_url(token: str) -> str:
    """The OpenAI base an engine is given for this lease (ends ``/v1``)."""
    return f"{self_base()}{PREFIX}/{token}/v1"


# --------------------------------------------------------------------- routes #
def _unknown_lease() -> JSONResponse:
    return JSONResponse(status_code=404,
                        content={"error": {"message": "unknown OCR engine lease",
                                           "type": "invalid_request_error"}})


@router.get(PREFIX + "/{token}/health", include_in_schema=False)
async def health(token: str):
    """200 while something can serve the model — without renting anything.

    Surya refuses to attach to a URL whose ``/health`` is not 200, so this must
    answer before any pod exists; a scale-to-zero pool is healthy precisely
    because it can rent on demand.
    """
    got = lease(token)
    if got is None:
        return _unknown_lease()
    from codai.models import placement
    if not placement.any_candidate(got["model_id"]):
        return JSONResponse(status_code=503,
                            content={"error": {"message": "no server for "
                                                          + got["model_id"]}})
    return Response(status_code=200, content=b"OK", media_type="text/plain")


@router.get(PREFIX + "/{token}/v1/models", include_in_schema=False)
async def models(token: str):
    """The id the engine expects, which is also what we will send upstream.

    Surya compares ``data[0].id`` with its configured checkpoint and refuses on a
    mismatch. Answering from the lease — rather than relaying whatever one
    upstream happens to call it — is also what makes several upstreams
    interchangeable to the engine.
    """
    got = lease(token)
    if got is None:
        return _unknown_lease()
    return JSONResponse(content={
        "object": "list",
        "data": [{"id": got["served_name"], "object": "model",
                  "owned_by": "coderai"}],
    })


@router.post(PREFIX + "/{token}/v1/chat/completions", include_in_schema=False)
async def chat_completions(token: str, request: Request):
    """Forward the request verbatim to whichever server is least busy.

    Verbatim is the whole point. The body is not parsed, so the multipart image
    stays an image, the ``json_schema`` stays strict, and ``logprobs`` is asked
    for and answered — the three things a request-editing proxy dropped. The one
    field that is rewritten is ``model``, and only when an upstream serves it
    under another name.
    """
    got = lease(token)
    if got is None:
        return _unknown_lease()

    import httpx

    from codai.models import placement
    body = await request.body()

    try:
        # How busy is too busy is the MODEL's own setting, the same one the pool
        # scales on — not a second, OCR-shaped copy of it.
        chosen = placement.choose(
            got["model_id"],
            spill_after=placement.spill_threshold(got["model_id"]))
    except Exception as exc:
        status = int(getattr(exc, "status", 503) or 503)
        return JSONResponse(status_code=status,
                            content={"error": {"message": str(exc),
                                               "type": "server_error"}})

    try:
        headers = {"Content-Type": request.headers.get("content-type",
                                                       "application/json")}
        if chosen.api_key:
            headers["Authorization"] = f"Bearer {chosen.api_key}"
        url = chosen.base.rstrip("/") + "/chat/completions"
        async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT_S) as client:
            reply = await client.post(url, content=body, headers=headers)
        # The reply goes back as it came: a re-serialised response is how
        # logprobs got lost in the first place.
        media = reply.headers.get("content-type", "application/json")
        return Response(status_code=reply.status_code, content=reply.content,
                        media_type=media)
    except Exception as exc:
        return JSONResponse(status_code=502,
                            content={"error": {
                                "message": f"OCR upstream {chosen.kind} failed: {exc}",
                                "type": "server_error"}})
    finally:
        chosen.release()
