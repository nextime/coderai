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
import weakref
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

router = APIRouter()

#: Path the engine attaches under. The OpenAI base it is given is
#: ``<self>/v1/ocr/attach/<lease>/v1`` so that stripping ``/v1`` — which is how
#: surya derives the health URL — lands on ``<self>/v1/ocr/attach/<lease>``.
#: Deliberately not under ``/v1/ocr/engine*``: the auth gates exempt this prefix,
#: and it must not be one keystroke from the public ``/v1/ocr/engines``.
PREFIX = "/v1/ocr/attach"

#: Failures that mean "there is nothing at this address", as opposed to a
#: server that answered slowly or badly. Only these drop the upstream: a read
#: timeout is a model taking its time over a dense page, and discarding a
#: perfectly good pod for that would be its own outage.
def _dead_upstream_errors():
    import httpx
    return (httpx.ConnectError, httpx.ConnectTimeout, ConnectionRefusedError)


#: Failures that mean "the socket we reused was already closed" — which says
#: nothing at all about the server.
#:
#: ``RemoteProtocolError("Server disconnected without sending a response")`` is
#: what httpx raises when a POOLED keep-alive connection was closed by the far
#: end between two requests. uvicorn closes an idle connection after 5s; this
#: module used to hold them for 60 (see ``_KEEPALIVE_EXPIRY_S``), so the first
#: page sent after a quiet gap raised it as a matter of course. It was in the
#: dead-upstream set, so the pool dropped the pod and TERMINATED it: five A40s
#: on 2026-10-12 between 19:14 and 20:11, every one of them mid-document and
#: logging 200s at 4664 tokens/s when it was killed. A closed socket is retried
#: on a fresh connection; only if that also fails does the address come into
#: question.
def _stale_socket_errors():
    import httpx
    return (httpx.RemoteProtocolError, ConnectionResetError)


def _is_dead_upstream(exc: BaseException) -> bool:
    """Whether this failure means "nothing is at that address".

    Written as a check rather than an exception class on purpose: ``except``
    matches on the class hierarchy at C level and never consults
    ``__instancecheck__``, so a clever proxy class silently catches nothing. A
    test caught exactly that here.
    """
    try:
        return isinstance(exc, _dead_upstream_errors())
    except Exception:
        return False


def _is_stale_socket(exc: BaseException) -> bool:
    """Whether this failure is a reused connection the server had closed."""
    try:
        return isinstance(exc, _stale_socket_errors())
    except Exception:
        return False


#: How long one page may take upstream. A dense scan on a cold pod legitimately
#: takes minutes; a timeout here would look to the engine like a failed page.
UPSTREAM_TIMEOUT_S = 3600.0

#: Upstream connection pooling. A client per REQUEST meant a fresh TCP and TLS
#: handshake for every page: with `surya_instances: 24` that is 24 streams each
#: reconnecting per page, against a pod whose round trip is already the dominant
#: cost. One client holds keep-alive connections open instead, which is the
#: difference between a handshake per page and a handshake per pod.
#:
#: Keyed by event loop: an httpx.AsyncClient binds to the loop that created it,
#: and coderai runs more than one (the OCR pool has a loop of its own), so a
#: single global client would be used from a loop it does not belong to. The key
#: is the loop OBJECT, held weakly — keying by id() would both leak an entry per
#: dead loop and, because CPython reuses addresses, risk handing a new loop a
#: client bound to the closed one that used to live there.
_MAX_CONNECTIONS = 256
_MAX_KEEPALIVE = 64
#: How long an IDLE pooled connection may be kept. This must stay BELOW the
#: upstream's own keep-alive timeout, or the pool hands out sockets the server
#: has already closed: uvicorn's default is 5s and nothing in coderai raises it,
#: so 60s here guaranteed a stale-socket error on the first page after any quiet
#: gap. Retries now cover the race (``_is_stale_socket``); this keeps it rare.
#: Reuse within a burst — the case this pooling exists for — is unaffected,
#: since only connections idle longer than this are dropped.
_KEEPALIVE_EXPIRY_S = float(os.environ.get("CODERAI_OCR_KEEPALIVE_EXPIRY_S") or 4.0)

_lock = threading.Lock()
_leases: dict = {}              # lease token -> {"model_id", "served_name"}
#: loop -> {pinned?: client}. TWO clients per loop, because the upstreams need
#: different trust: a direct-TCP pod serves OUR certificate on a bare IP, which
#: only coderai's CA can verify and whose hostname can never match, while a
#: cluster node has an ordinary certificate a public CA signed. One client cannot
#: hold both policies, and using the pinned one everywhere would reject the node.
_clients: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


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
def _pinned_ssl_context():
    """The TLS policy a direct-TCP pod needs: our CA, no hostname match.

    The same policy :mod:`codai.api.pod_http` applies for every other request to
    a pod — it is only separate here because that module is built on requests and
    this hop is async. Getting it wrong is not subtle: the pod answered
    "CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate" to every
    page the first time a real document went through.
    """
    import ssl

    from codai.api.pod_tls import ensure_ca
    ctx = ssl.create_default_context(cafile=ensure_ca())
    # The pod is addressed by an IP its certificate cannot name.
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_REQUIRED
    return ctx


def _upstream_client(url: str = ""):
    """The shared client for this event loop and this upstream's trust policy."""
    import asyncio

    import httpx

    try:
        from codai.api.pod_http import is_pinned_url
        pinned = bool(is_pinned_url(url))
    except Exception:
        pinned = False
    try:
        key = asyncio.get_running_loop()
    except RuntimeError:
        key = _upstream_client            # a stable stand-in when there is no loop
    with _lock:
        bucket = _clients.get(key)
        if bucket is None:
            bucket = {}
            _clients[key] = bucket
        got = bucket.get(pinned)
        if got is not None and not got.is_closed:
            return got
        verify = _pinned_ssl_context() if pinned else True
        got = httpx.AsyncClient(
            timeout=UPSTREAM_TIMEOUT_S, verify=verify,
            limits=httpx.Limits(max_connections=_MAX_CONNECTIONS,
                                max_keepalive_connections=_MAX_KEEPALIVE,
                                keepalive_expiry=_KEEPALIVE_EXPIRY_S))
        bucket[pinned] = got
        return got


async def _one_shot_client(url: str = ""):
    """A client of its very own: no pooling, nothing reused.

    Used for the retry after a stale socket and for the health probe. Both exist
    precisely because the pooled connection was bad, so neither may be allowed
    to draw from the same pool.
    """
    import httpx

    try:
        from codai.api.pod_http import is_pinned_url
        pinned = bool(is_pinned_url(url))
    except Exception:
        pinned = False
    verify = _pinned_ssl_context() if pinned else True
    return httpx.AsyncClient(
        timeout=UPSTREAM_TIMEOUT_S, verify=verify,
        limits=httpx.Limits(max_connections=4, max_keepalive_connections=0))


#: How long the confirming health probe may take. Short: it only has to
#: establish that something is listening and speaking HTTP, and a pod busy with
#: a dense page still answers /healthz promptly.
_PROBE_TIMEOUT_S = 10.0


async def _still_alive(base: str, api_key: str = "") -> bool:
    """Whether anything is still serving at ``base``, asked on a new connection.

    A pod is a rented GPU, and dropping it from the pool terminates it. One
    failed request is not evidence enough for that: on 2026-10-12 five A40s were
    killed mid-document, each still logging 200s, because a closed keep-alive
    socket was read as a dead address. So the address gets asked directly before
    anything irreversible happens.

    Any HTTP answer counts, 401 included — that is a server, authenticating.
    Only silence means nothing is there.
    """
    root = base.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    client = await _one_shot_client(base)
    try:
        for path in ("/healthz", "/health"):
            try:
                reply = await client.get(root + path, headers=headers,
                                         timeout=_PROBE_TIMEOUT_S)
            except Exception:
                continue            # try the other path, then give up
            if reply.status_code < 500:
                return True
        return False
    finally:
        try:
            await client.aclose()
        except Exception:
            pass


async def close_upstream_clients() -> None:
    """Close the pooled clients (process shutdown, or a test tidying up)."""
    with _lock:
        clients = [c for bucket in _clients.values() for c in bucket.values()]
        _clients.clear()
    for c in clients:
        try:
            await c.aclose()
        except Exception:
            pass


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

    from codai.models import placement
    body = await request.body()

    content_type = request.headers.get("content-type", "application/json")
    # Twice at most: a rented pod can vanish between being chosen and being
    # asked (RunPod reclaimed one mid-document today). The first attempt reports
    # the corpse so the pool drops it; the second runs against whatever replaces
    # it. A third would just be a retry loop against a real outage.
    last_dead = ""
    for attempt in (1, 2):
        try:
            # How busy is too busy is the MODEL's own setting, the same one the
            # pool scales on — not a second, OCR-shaped copy of it.
            chosen = placement.choose(
                got["model_id"],
                spill_after=placement.spill_threshold(got["model_id"]))
        except Exception as exc:
            status = int(getattr(exc, "status", 503) or 503)
            return JSONResponse(status_code=status,
                                content={"error": {"message": str(exc),
                                                   "type": "server_error"}})
        headers = {"Content-Type": content_type}
        if chosen.api_key:
            headers["Authorization"] = f"Bearer {chosen.api_key}"
        url = chosen.base.rstrip("/") + "/chat/completions"
        try:
            try:
                reply = await _upstream_client(url).post(url, content=body,
                                                         headers=headers)
            except Exception as exc:
                if not _is_stale_socket(exc):
                    raise
                # The connection we reused had already been closed. That is a
                # race, not an outage: send the page again on a socket of its
                # own. Retried here rather than by picking a new candidate,
                # because the upstream we have is almost certainly fine and
                # re-choosing would send the page to a different pod — or rent
                # one.
                print(f"[ocr] upstream {chosen.base} closed a pooled connection "
                      f"({type(exc).__name__}) — resending on a fresh one",
                      flush=True)
                client = await _one_shot_client(url)
                try:
                    reply = await client.post(url, content=body, headers=headers)
                finally:
                    try:
                        await client.aclose()
                    except Exception:
                        pass
            # The reply goes back as it came: a re-serialised response is how
            # logprobs got lost in the first place.
            media = reply.headers.get("content-type", "application/json")
            return Response(status_code=reply.status_code, content=reply.content,
                            media_type=media)
        except Exception as exc:
            if not (_is_dead_upstream(exc) or _is_stale_socket(exc)):
                return JSONResponse(status_code=502,
                                    content={"error": {
                                        "message": f"OCR upstream {chosen.kind} "
                                                   f"failed: {exc}",
                                        "type": "server_error"}})
            # Looks like nothing is there — but dropping a pod TERMINATES a
            # rented GPU, so ask the address itself first, on a new connection.
            # Skipping this cost five A40s in an hour, each killed while it was
            # still answering 200s.
            if await _still_alive(chosen.base, chosen.api_key):
                print(f"[ocr] upstream {chosen.base} failed this page "
                      f"({type(exc).__name__}: {exc}) but answers its health "
                      f"check — keeping it", flush=True)
                return JSONResponse(status_code=502,
                                    content={"error": {
                                        "message": f"OCR upstream {chosen.kind} "
                                                   f"failed: {exc}",
                                        "type": "server_error"}})
            # Nothing answered at the address at all. Report it so the pool
            # stops handing out a pod that no longer exists — left unsaid, every
            # following page picks the same one and the in-flight count climbs
            # with nothing draining.
            last_dead = f"{type(exc).__name__}: {exc}"
            print(f"[ocr] upstream {chosen.base} is not answering "
                  f"({last_dead}) — dropping it", flush=True)
            chosen.dead()
            chosen = None
            continue
        finally:
            if chosen is not None:
                chosen.release()
    return JSONResponse(status_code=503,
                        content={"error": {
                            "message": f"no OCR server answered for "
                                       f"{got['model_id']!r} ({last_dead})",
                            "type": "server_error"}})
