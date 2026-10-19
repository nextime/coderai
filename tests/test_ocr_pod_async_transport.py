"""Forwarding a page on the event loop instead of on a thread.

The thread path costs one thread per page in flight. At ~60 s per page against
a real corpus, 15 documents/second at ~3.5 pages each is ~3,200 outstanding
forwards -- more threads than any install should carry, and above the pod
pool's 2048 ceiling. So the hot path is a coroutine.

What must NOT change in the move: a pod that ANSWERS (even 503) is a verdict
about the pod and belongs on the refusal path, nothing-came-back is
PodUnreachable, a dead pooled socket is retried on a fresh one, and a timeout
is never retried. Those are 0.4.2-0.4.6 and they are transport-independent.
"""

import asyncio
import types

import pytest

from codai.api import pod_http
from codai.ocr import manager as om
from codai.ocr.base import OcrError


class _Reply:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code = status
        self._payload = payload if payload is not None else {"pages": []}
        self.text = text

    def json(self):
        return self._payload


# ---------------------------------------------------------- which transport #

def test_the_loop_is_the_default():
    from codai.config import OcrConfig
    assert OcrConfig().pod_http_async is True
    assert om._use_async_pod_http(OcrConfig()) is True


def test_the_operator_can_put_it_back_on_threads():
    assert om._use_async_pod_http(
        types.SimpleNamespace(pod_http_async=False)) is False


def test_a_config_without_the_knob_gets_the_loop():
    """An install whose config.json predates the setting."""
    assert om._use_async_pod_http(types.SimpleNamespace()) is True


def test_without_httpx_it_falls_back_to_threads(monkeypatch):
    """The transport is an optimisation; a build lacking it must still read."""
    monkeypatch.setattr(pod_http, "async_available", lambda: False)
    from codai.config import OcrConfig
    assert om._use_async_pod_http(OcrConfig()) is False


# ----------------------------------------------------------------- the TLS #

def test_a_direct_tcp_pod_is_pinned_and_a_proxied_one_is_not():
    assert pod_http.is_pinned_url("https://10.0.0.1:22001/v1") is True
    assert pod_http.is_pinned_url("https://pod.runpod.net/v1") is False
    assert pod_http.is_pinned_url("http://10.0.0.1:22001/v1") is False


def test_the_pinned_context_trusts_our_ca_and_not_the_hostname(monkeypatch):
    """verify=False would accept anyone; the pod's certificate must be one we
    signed, at an ip its certificate cannot name."""
    import ssl
    import codai.api.pod_tls as pod_tls
    import certifi
    monkeypatch.setattr(pod_tls, "ensure_ca", lambda: certifi.where())
    ctx = pod_http._pinned_ssl_context()
    assert ctx.check_hostname is False
    assert ctx.verify_mode == ssl.CERT_REQUIRED


def test_one_client_per_loop_and_pinnedness(monkeypatch):
    import certifi
    import codai.api.pod_tls as pod_tls
    monkeypatch.setattr(pod_tls, "ensure_ca", lambda: certifi.where())

    async def go():
        a = pod_http.aclient(True)
        b = pod_http.aclient(True)
        c = pod_http.aclient(False)
        assert a is b
        assert a is not c
        await pod_http.aclose_all()

    asyncio.run(go())


def test_the_client_pool_is_sized_from_the_config(monkeypatch):
    import certifi
    import codai.admin.routes as routes
    import codai.api.pod_tls as pod_tls
    monkeypatch.setattr(pod_tls, "ensure_ca", lambda: certifi.where())

    class _M:
        config = types.SimpleNamespace(ocr=types.SimpleNamespace(max_concurrency=200))
    monkeypatch.setattr(routes, "config_manager", _M(), raising=False)

    assert pod_http._pool_size() == 200          # above the floor of 64

    import httpx
    seen = {}
    real = httpx.AsyncClient

    class _Capture(real):
        def __init__(self, *a, **kw):
            seen.update(kw)
            super().__init__(*a, **kw)
    monkeypatch.setattr(httpx, "AsyncClient", _Capture)

    async def go():
        await pod_http.aclose_all()
        pod_http.aclient(False)
        # The shared async client serves the WHOLE fleet, so its limit is the
        # fleet total (_fleet_pool_size), not the per-pod 200. Asserting 200
        # here is what kept every page forward capped at 256 fleet-wide.
        assert seen["limits"].max_connections == pod_http._fleet_pool_size()
        assert seen["limits"].max_connections > 200
        assert seen["limits"].max_keepalive_connections == seen["limits"].max_connections
        await pod_http.aclose_all()

    asyncio.run(go())


# -------------------------------------------------------- answers vs sockets #

def _apost(monkeypatch, fn):
    monkeypatch.setattr(pod_http, "apost", fn)


def test_an_answer_comes_back_as_a_dict(monkeypatch):
    async def ok(url, **kw):
        return _Reply(200, {"engine": "surya", "pages": [{"text": "hello"}]})
    _apost(monkeypatch, ok)
    got = asyncio.run(om._apost_page_to_pod("https://10.0.0.1:1/v1", "k",
                                            b"%PDF", "0.pdf", 200, {}, None))
    assert got["pages"][0]["text"] == "hello"


def test_a_pod_answering_503_is_a_refusal_not_unreachable(monkeypatch):
    """The 0.4.6 distinction, on the new transport: an ANSWER is a verdict
    about the pod's image and must reach the refusal path, or one 503 sets a
    whole pool aside for five minutes."""
    async def busy(url, **kw):
        return _Reply(503, text="no pipeline in this image")
    _apost(monkeypatch, busy)
    with pytest.raises(OcrError) as e:
        asyncio.run(om._apost_page_to_pod("https://10.0.0.1:1/v1", "k", b"%PDF",
                                          "0.pdf", 200, {}, None))
    assert not isinstance(e.value, om.PodUnreachable)
    assert e.value.status == 503


def test_a_dead_socket_is_retried_on_a_fresh_one(monkeypatch):
    calls = {"n": 0}

    async def flaky(url, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionResetError("Connection aborted: RemoteDisconnected")
        return _Reply(200, {"pages": [{"text": "second try"}]})
    _apost(monkeypatch, flaky)
    got = asyncio.run(om._apost_page_to_pod("https://10.0.0.1:1/v1", "k", b"%PDF",
                                            "0.pdf", 200, {}, None))
    assert calls["n"] == 2
    assert got["pages"][0]["text"] == "second try"


def test_a_timeout_is_never_retried(monkeypatch):
    """A page the pod is still working on must not be sent twice: that is a
    second card doing the same work and a double bill."""
    calls = {"n": 0}

    async def slow(url, **kw):
        calls["n"] += 1
        raise TimeoutError("read timed out")
    _apost(monkeypatch, slow)
    with pytest.raises(om.PodUnreachable):
        asyncio.run(om._apost_page_to_pod("https://10.0.0.1:1/v1", "k", b"%PDF",
                                          "0.pdf", 200, {}, None))
    assert calls["n"] == 1


def test_a_socket_that_never_answers_is_PodUnreachable(monkeypatch):
    async def gone(url, **kw):
        raise OSError("connection refused")
    _apost(monkeypatch, gone)
    with pytest.raises(om.PodUnreachable):
        asyncio.run(om._apost_page_to_pod("https://10.0.0.1:1/v1", "k", b"%PDF",
                                          "0.pdf", 200, {}, None))


def test_no_json_is_a_502_not_a_transport_failure(monkeypatch):
    class _Junk(_Reply):
        def json(self):
            raise ValueError("not json")
    async def junk(url, **kw):
        return _Junk(200)
    _apost(monkeypatch, junk)
    with pytest.raises(OcrError) as e:
        asyncio.run(om._apost_page_to_pod("https://10.0.0.1:1/v1", "k", b"%PDF",
                                          "0.pdf", 200, {}, None))
    assert e.value.status == 502
    assert not isinstance(e.value, om.PodUnreachable)


# ----------------------------------------------------- the request it builds #

def test_the_engine_name_travels_with_the_page(monkeypatch):
    """Hardcoding "surya" once sent a chandra page to surya, which the pod
    served with the wrong model and the text came back looking plausible."""
    sent = {}

    async def grab(url, **kw):
        sent["url"] = url
        sent.update(kw)
        return _Reply(200, {"pages": []})
    _apost(monkeypatch, grab)
    asyncio.run(om._apost_page_to_pod("https://10.0.0.1:1/v1", "tok", b"%PDF",
                                      "3.pdf", 250, {"max_tokens": 9000},
                                      "stamps", "chandra"))
    assert sent["url"] == "https://10.0.0.1:1/v1/ocr"
    assert sent["data"]["engine"] == "chandra"
    assert sent["data"]["dpi"] == "250"
    assert sent["data"]["detect"] == "stamps"
    assert sent["data"]["max_tokens"] == "9000"
    assert sent["headers"]["Authorization"] == "Bearer tok"
    assert sent["files"]["file"][0] == "3.pdf"


def test_the_url_is_built_the_same_way_as_the_sync_path():
    assert om._page_post_url("https://1.2.3.4:8/v1") == "https://1.2.3.4:8/v1/ocr"
    assert om._page_post_url("https://1.2.3.4:8") == "https://1.2.3.4:8/v1/ocr"
    assert om._page_post_url("https://1.2.3.4:8/") == "https://1.2.3.4:8/v1/ocr"


def test_the_liveness_probe_asks_healthz(monkeypatch):
    seen = {}

    async def ok(url, **kw):
        seen["url"] = url
        return _Reply(200)
    monkeypatch.setattr(pod_http, "aget", ok)
    assert asyncio.run(om._apod_still_alive("https://10.0.0.1:1/v1", "k")) is True
    assert seen["url"] == "https://10.0.0.1:1/healthz"


def test_a_probe_that_raises_means_not_alive(monkeypatch):
    async def gone(url, **kw):
        raise OSError("refused")
    monkeypatch.setattr(pod_http, "aget", gone)
    assert asyncio.run(om._apod_still_alive("https://10.0.0.1:1/v1", "k")) is False


def test_a_500_from_healthz_is_not_alive_but_a_404_is(monkeypatch):
    """A pod that answers at all is still there; what it answers is about its
    image, not about whether it exists."""
    async def code(url, **kw):
        return _Reply(int(url.rsplit("#", 1)[-1]))
    monkeypatch.setattr(pod_http, "aget",
                        lambda url, **kw: code(url + "#404", **kw))
    assert asyncio.run(om._apod_still_alive("https://10.0.0.1:1", "k")) is True
    monkeypatch.setattr(pod_http, "aget",
                        lambda url, **kw: code(url + "#500", **kw))
    assert asyncio.run(om._apod_still_alive("https://10.0.0.1:1", "k")) is False


# ------------------------------------------------------- the two twins agree #

def test_both_twins_share_one_request_builder():
    """Two copies of the field list is how an engine name, a dpi or a token
    budget comes to apply on one transport and not the other."""
    import inspect
    for fn in (om._post_page_to_pod, om._apost_page_to_pod):
        src = inspect.getsource(fn)
        assert "_page_post_fields(" in src
        assert "_page_post_url(" in src
        assert "_page_response(" in src


def test_both_twins_use_the_same_attempt_budget():
    import inspect
    for fn in (om._post_page_to_pod, om._apost_page_to_pod):
        assert "POD_PAGE_ATTEMPTS" in inspect.getsource(fn)
