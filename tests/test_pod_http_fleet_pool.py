"""The async pod client is shared by the WHOLE fleet, so its connection limit
is a fleet total, not a per-pod figure.

`_pool_size()` is per pod (what `requests` means by `pool_maxsize`, which pools
per host). httpx's `max_connections` is a total across hosts. Passing the
per-pod 256 to the shared AsyncClient capped every page forward in the install
at 256. Measured 2026-10-19: 31 pods, 1,583 pages leased, 284 TCP connections
to pods -- most pods holding exactly one.
"""
import inspect

from codai.api import pod_http


def test_the_async_client_is_sized_for_the_fleet():
    src = inspect.getsource(pod_http.aclient)
    assert "_fleet_pool_size()" in src
    assert "size = _pool_size()" not in src


def test_fleet_size_follows_the_page_ceiling(monkeypatch):
    import codai.api.thread_pool as tp
    monkeypatch.setattr(tp, "_configured_concurrency", lambda: 4096)
    assert pod_http._fleet_pool_size() >= 4096


def test_fleet_size_is_never_below_the_old_per_pod_figure(monkeypatch):
    import codai.api.thread_pool as tp
    monkeypatch.setattr(tp, "_configured_concurrency", lambda: 0)
    assert pod_http._fleet_pool_size() == pod_http._MAX_POOL


def test_fleet_size_is_bounded(monkeypatch):
    import codai.api.thread_pool as tp
    monkeypatch.setattr(tp, "_configured_concurrency", lambda: 10**9)
    assert pod_http._fleet_pool_size() == pod_http._MAX_FLEET_POOL


def test_the_sync_per_host_pool_is_unchanged():
    """requests pools per host, so the per-pod figure is right there."""
    src = inspect.getsource(pod_http._pinned_session)
    assert "_pool_size()" in src
