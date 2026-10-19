# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""`requests`, but able to reach a pod over TLS it brought itself.

A pod on the direct TCP path serves HTTPS with a certificate signed by this
install's own CA (codai/api/pod_tls.py). No public CA knows it, and the pod
is addressed by a public IP that was not known when the certificate was
issued — so the default `requests` verification fails twice over. This
module routes exactly one kind of URL differently: **https to a bare IP
address** is verified against our CA with the hostname check off. Every
other URL — the RunPod proxy, HuggingFace, a host you named — goes through
`requests` untouched.

Why that rule is safe: a certificate our CA did not sign fails the chain
check before any name is compared, and only pods we provisioned hold one.
Why no hostname check: the name would have to be the IP, which a pod does
not have until it boots; checking a placeholder proves nothing extra.

Drop-in: `pod_http.get/post/request(...)` take what `requests.*` take.
"""

import ipaddress
import ssl
import threading
from urllib.parse import urlsplit

import requests
from requests.adapters import HTTPAdapter

_lock = threading.Lock()
_session = None


#: Connections kept alive PER POD. requests' own default is 10, and one shared
#: session serves every pod, so the pool is per host either way — but 10 is a
#: number for a browser talking to a website, not for an OCR run fanning
#: `ocr.max_concurrency` pages at ONE pod.
#:
#: Past the tenth concurrent request urllib3 opens a connection anyway, uses it,
#: finds the pool full on release and discards it. Every page after the tenth
#: therefore pays a fresh TLS handshake to a machine in another country: several
#: round trips of latency each, and handshake CPU on the orchestrator at exactly
#: the moment it is busiest. Measured on a live run: 74 "Connection pool is
#: full, discarding connection" warnings in five minutes, ~14 a minute, against
#: a single pod IP.
#:
#: Idle connections cost a socket and nothing else, so sizing this to the
#: concurrency the install is configured for is the cheap side of the trade.
_DEFAULT_POOL = 64
_MAX_POOL = 256


def _pool_size() -> int:
    """How many connections to keep per pod — from the configured concurrency.

    Read at session creation, not import: the config is not loaded yet when
    this module is imported, and an install that raises `ocr.max_concurrency`
    should not have to know that a connection pool exists.
    """
    try:
        # The SHARED manager is in codai.admin.routes; codai.config has no
        # `config_manager` at all, so the old read raised ImportError on every
        # call and this silently returned the floor. See
        # codai/api/thread_pool.py for what that cost.
        from codai.api.thread_pool import _configured_concurrency
        want = _configured_concurrency()
    except Exception:
        want = 0
    return max(_DEFAULT_POOL, min(_MAX_POOL, want))


#: Ceiling on the async client's connections across the WHOLE fleet.
_MAX_FLEET_POOL = 16384


def _fleet_pool_size() -> int:
    """Connections the async client may hold across EVERY pod at once.

    NOT `_pool_size()`. That number is per pod, which is what `requests` means
    by `pool_maxsize` -- it keeps one pool per host. httpx's `max_connections`
    is a TOTAL across all hosts, and this client serves the whole fleet, so
    reusing the per-pod figure capped every page forward in the install at 256.

    Measured 2026-10-19 on a 31-pod fleet: 1,583 pages holding a pod lease and
    284 TCP connections to pods in total -- six pods with 25-49 connections,
    most with exactly one. Renting more pods added cards and no throughput,
    because the fleet shared 256 connections. Pages that waited too long for a
    connection failed with an empty-message timeout, which was reported as the
    POD being unreachable and got healthy pods quarantined.

    Sized from the page ceiling the install is configured for, because every
    page in flight to a pod holds a connection for its whole life.
    """
    try:
        from codai.api.thread_pool import _configured_concurrency
        want = _configured_concurrency()
    except Exception:
        want = 0
    if want <= 0:
        return _MAX_POOL
    return max(_MAX_POOL, min(_MAX_FLEET_POOL, want + max(256, want // 4)))


class _PinnedAdapter(HTTPAdapter):
    """An adapter whose pool trusts our CA and skips hostname matching."""

    def __init__(self, ca_file: str, **kw):
        self._ca_file = ca_file
        kw.setdefault("pool_connections", _DEFAULT_POOL)
        kw.setdefault("pool_maxsize", _DEFAULT_POOL)
        super().__init__(**kw)

    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context(cafile=self._ca_file)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_REQUIRED
        kwargs["ssl_context"] = ctx
        return super().init_poolmanager(*args, **kwargs)

    def cert_verify(self, conn, url, verify, cert):
        # urllib3 would otherwise reset the context's CA to requests' bundle
        # or, with verify=False, turn verification off; the context above is
        # the whole point, so leave it alone.
        conn.cert_reqs = "CERT_REQUIRED"
        conn.ca_certs = self._ca_file
        # urllib3 matches the hostname itself, independently of the context's
        # check_hostname; the pod is addressed by an IP its cert cannot name.
        conn.assert_hostname = False


def is_pinned_url(url: str) -> bool:
    """https:// to a bare IP address — a direct-TCP pod with our certificate."""
    try:
        parts = urlsplit(url or "")
        if parts.scheme != "https" or not parts.hostname:
            return False
        ipaddress.ip_address(parts.hostname)
        return True
    except ValueError:
        return False


def _pinned_session() -> requests.Session:
    global _session
    with _lock:
        if _session is None:
            from codai.api.pod_tls import ensure_ca
            s = requests.Session()
            size = _pool_size()
            s.mount("https://", _PinnedAdapter(ensure_ca(), pool_connections=size,
                                               pool_maxsize=size))
            _session = s
        return _session


def request(method: str, url: str, **kw):
    if is_pinned_url(url):
        kw.pop("verify", None)
        return _pinned_session().request(method, url, **kw)
    return requests.request(method, url, **kw)


# The verb helpers delegate to requests' own verbs for ordinary URLs, not to
# requests.request: callers (and tests) that patch requests.get keep working.
def get(url: str, **kw):
    if is_pinned_url(url):
        return request("GET", url, **kw)
    return requests.get(url, **kw)


def post(url: str, **kw):
    if is_pinned_url(url):
        return request("POST", url, **kw)
    return requests.post(url, **kw)


def head(url: str, **kw):
    if is_pinned_url(url):
        return request("HEAD", url, **kw)
    return requests.head(url, **kw)


# --------------------------------------------------------------------------- #
# the async path
# --------------------------------------------------------------------------- #
#
# A page forward blocks for ~60 s against a real corpus, and the sync path above
# costs one THREAD for every one of them. That is survivable at tens of pages in
# flight and not at thousands: 15 documents/second at ~3.5 pages each and ~60 s
# per page is ~3,200 forwards outstanding, which is not a number of threads any
# install should be asked to carry (see codai/api/thread_pool.py, where the pod
# pool is capped at 2048 for exactly that reason).
#
# So the hot path gets a coroutine instead. Everything the sync path learned the
# hard way is kept:
#
# * the pod serves HTTPS with a certificate signed by THIS install's CA, at an
#   ip that was not known when it was issued -- so the CA is trusted and the
#   hostname check is off, and NOT verify=False, which would accept anyone;
# * one client per event loop, with connection limits from the same
#   `_pool_size()` the sync session uses;
# * a proxied pod (http://, or https:// to a name) is an ordinary request.
#
# httpx rather than aiohttp: it is already in the stack (codai/cluster/nodes.py,
# codai/main.py) and takes an ssl.SSLContext directly, so the pinned context is
# the same object the sync path builds.

_aclients: dict = {}
_alock = threading.Lock()


def _pinned_ssl_context():
    """The context the sync adapter installs, built once per call site.

    CA from this install, hostname matching off. The two together are the whole
    trust decision for a direct-TCP pod: the certificate must be one we signed,
    and the name on it cannot be the ip we reach it at.
    """
    from codai.api.pod_tls import ensure_ca
    ctx = ssl.create_default_context(cafile=ensure_ca())
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_REQUIRED
    return ctx


def async_available() -> bool:
    """Whether the async transport can be used at all."""
    try:
        import httpx            # noqa: F401
        return True
    except Exception:
        return False


def aclient(pinned: bool):
    """The AsyncClient for this event loop, created on first use.

    Keyed by loop AND by pinned-ness: the two need different trust settings, and
    a client built for one must never serve the other. Keyed by loop because an
    httpx client belongs to the loop whose transport it opened its sockets on --
    sharing one across loops is how a test suite ends up with sockets attached
    to a closed loop.
    """
    import asyncio

    import httpx

    try:
        loop_key = id(asyncio.get_running_loop())
    except RuntimeError:
        loop_key = 0
    key = (loop_key, bool(pinned))
    with _alock:
        c = _aclients.get(key)
        if c is not None and not c.is_closed:
            return c
        size = _fleet_pool_size()
        limits = httpx.Limits(max_connections=size,
                              max_keepalive_connections=size)
        kw = {"limits": limits, "timeout": httpx.Timeout(600.0, connect=30.0)}
        if pinned:
            kw["verify"] = _pinned_ssl_context()
        c = httpx.AsyncClient(**kw)
        _aclients[key] = c
        return c


async def apost(url: str, files=None, data=None, headers=None, timeout=600):
    """POST on the event loop. Answers like the sync path: an object with
    ``status_code``, ``text`` and ``json()``, so callers need not care which
    transport ran."""
    c = aclient(is_pinned_url(url))
    return await c.post(url, files=files, data=data, headers=headers,
                        timeout=timeout)


async def aget(url: str, headers=None, timeout=15):
    c = aclient(is_pinned_url(url))
    return await c.get(url, headers=headers, timeout=timeout)


async def aclose_all() -> None:
    """Close every client (shutdown, tests). Never raises."""
    with _alock:
        clients = list(_aclients.values())
        _aclients.clear()
    for c in clients:
        try:
            await c.aclose()
        except Exception:
            pass
