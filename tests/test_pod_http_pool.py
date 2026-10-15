"""Ten connections is a browser's number, not a fan-out's.

One shared session serves every pod, and `requests`' stock adapter keeps 10
connections per host. `ocr.max_concurrency` defaults to 64 and was 128 on the
install where this showed up, so from the eleventh concurrent page onward
urllib3 opened a connection, used it, found the pool full on release and threw
it away — a full TLS handshake per page to a pod in another country, and
handshake CPU on the orchestrator exactly when it had none to spare.

Measured live 2026-10-15: 74 "Connection pool is full, discarding connection:
<pod ip>. Connection pool size: 10" in five minutes, ~14 a minute, while an OCR
run's health checks were timing out.
"""

import types

import pytest

from codai.api import pod_http


def _ca() -> str:
    """A real PEM — the adapter builds its SSL context at construction, so a
    placeholder path raises before the pool is ever sized."""
    import certifi
    return certifi.where()


def test_the_adapter_no_longer_takes_requests_default_of_ten():
    a = pod_http._PinnedAdapter(_ca())
    assert a._pool_connections >= 64
    assert a._pool_maxsize >= 64


def test_an_explicit_size_wins_over_the_default():
    a = pod_http._PinnedAdapter(_ca(), pool_connections=7, pool_maxsize=7)
    assert a._pool_maxsize == 7


# ------------------------------------------------------- sizing from config #
def _cfg(monkeypatch, max_concurrency):
    mgr = types.SimpleNamespace(
        config=types.SimpleNamespace(
            ocr=types.SimpleNamespace(max_concurrency=max_concurrency)))
    import codai.config
    monkeypatch.setattr(codai.config, "config_manager", mgr, raising=False)


def test_the_pool_follows_the_configured_concurrency(monkeypatch):
    """The install that hit this runs 128 pages at once."""
    _cfg(monkeypatch, 128)
    assert pod_http._pool_size() == 128


def test_a_small_concurrency_still_gets_a_usable_floor(monkeypatch):
    """Pod traffic is not only OCR — health probes, blob pushes, forwarded
    chat all share this session."""
    _cfg(monkeypatch, 4)
    assert pod_http._pool_size() == pod_http._DEFAULT_POOL


def test_an_absurd_concurrency_is_capped(monkeypatch):
    _cfg(monkeypatch, 100000)
    assert pod_http._pool_size() == pod_http._MAX_POOL


def test_no_config_yet_is_not_an_error(monkeypatch):
    """Read at session creation precisely because import happens first, but a
    caller before the config loads must still get a session."""
    import codai.config
    monkeypatch.setattr(codai.config, "config_manager", None, raising=False)
    assert pod_http._pool_size() == pod_http._DEFAULT_POOL


def test_a_broken_config_object_falls_back(monkeypatch):
    class _Boom:
        @property
        def config(self):
            raise RuntimeError("not loaded")
    import codai.config
    monkeypatch.setattr(codai.config, "config_manager", _Boom(), raising=False)
    assert pod_http._pool_size() == pod_http._DEFAULT_POOL


# ------------------------------------------------------------ the session #
def test_the_session_mounts_the_sized_adapter(monkeypatch, tmp_path):
    monkeypatch.setattr(pod_http, "_session", None, raising=False)
    import codai.api.pod_tls as pod_tls
    monkeypatch.setattr(pod_tls, "ensure_ca", _ca)
    _cfg(monkeypatch, 96)
    try:
        s = pod_http._pinned_session()
    finally:
        pod_http._session = None
    a = s.get_adapter("https://10.0.0.1/")
    assert a._pool_maxsize == 96


def test_routing_is_unchanged():
    """Only pods get the pinned session; everything else stays on requests, so
    callers and tests that patch requests.get keep working."""
    assert pod_http.is_pinned_url("https://194.68.245.54:22031/v1/ocr")
    assert not pod_http.is_pinned_url("https://huggingface.co/x")
    assert not pod_http.is_pinned_url("http://194.68.245.54:8000/v1/ocr")
