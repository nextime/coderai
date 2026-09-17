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


class _PinnedAdapter(HTTPAdapter):
    """An adapter whose pool trusts our CA and skips hostname matching."""

    def __init__(self, ca_file: str, **kw):
        self._ca_file = ca_file
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
            s.mount("https://", _PinnedAdapter(ensure_ca()))
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
