"""The width of the pod path, which was 32 threads and nobody could see it.

A 512-page semaphore, a 48-wide gate on every pod and 31 idle A40s, and the
orchestrator still delivered 1.65 pages/s. The cause was two lines:

* every sizing helper read ``codai.config.config_manager``, a name that does
  not exist (the shared manager is in ``codai.admin.routes``), so the
  ImportError was swallowed and both pools took their FLOORS forever;
* the per-page forward used ``asyncio.to_thread``, i.e. the loop's default
  32-thread pool, rather than the pod pool written for it.

32 threads / 19.4 s per page = 1.65 pages/s. Neither the semaphore nor the pod
gate could ever have moved it. These tests fail if either half comes back.
"""

import asyncio

import pytest

from codai.api import pod_http, thread_pool


class _Ocr:
    max_concurrency = 0


class _Cfg:
    def __init__(self, n):
        self.ocr = _Ocr()
        self.ocr.max_concurrency = n


class _Mgr:
    def __init__(self, n):
        self.config = _Cfg(n)


@pytest.fixture
def holder(monkeypatch):
    """Install a config manager where the REAL one lives."""
    import codai.admin.routes as routes

    def _set(n):
        monkeypatch.setattr(routes, "config_manager", _Mgr(n), raising=False)
    return _set


# ------------------------------------------------------- reading the config #

def test_the_pools_read_the_holder_that_actually_exists(holder):
    holder(512)
    assert thread_pool._configured_concurrency() == 512
    assert thread_pool.sized_from_config() is True


def test_codai_config_has_no_config_manager():
    """The regression in one line: the old read could never have worked, on any
    install, in any version. If this name ever appears, the fallback is live
    again and the sizing should prefer it."""
    import codai.config
    assert not hasattr(codai.config, "config_manager")


def test_an_unreadable_config_is_reported_not_disguised(monkeypatch):
    import codai.admin.routes as routes
    monkeypatch.setattr(routes, "config_manager", None, raising=False)
    assert thread_pool._configured_concurrency() == 0
    assert thread_pool.sized_from_config() is False
    # ...and the floors are still what you get, because a server that cannot
    # read its config must still run.
    assert thread_pool.pool_sizes() == (thread_pool._MIN_DEFAULT,
                                        thread_pool._MIN_POD)


def test_the_configured_width_reaches_both_pools(holder):
    holder(2048)
    general, pod = thread_pool.pool_sizes()
    assert general == thread_pool._MAX_DEFAULT      # 256
    assert pod == 2048
    assert pod > thread_pool._MIN_POD * 10


def test_the_pod_ceiling_allows_a_hundred_pages_a_second():
    """~100 pages/s at ~20 s per page is ~2000 forwards outstanding. A ceiling
    below that caps the install regardless of its config."""
    assert thread_pool._MAX_POD >= 2000


def test_the_connection_pool_is_sized_the_same_way(holder):
    holder(512)
    assert pod_http._pool_size() == min(pod_http._MAX_POOL, 512)
    holder(0)
    assert pod_http._pool_size() == pod_http._DEFAULT_POOL


# ------------------------------------------------------- using the pod pool #

def test_the_page_forward_does_not_use_the_default_executor():
    """Read from the source: the call must not be `asyncio.to_thread`.

    Asserted statically because the alternative is asserting on a thread name
    at runtime, which passes just as happily when the call has been moved back
    and the pool happens to be large.
    """
    import inspect

    from codai.ocr.manager import OcrManager
    src = inspect.getsource(OcrManager._read_on_pods)
    assert "_post_page_to_pod" in src
    assert "run_on_pod_pool(" in src
    # The page forward and the liveness probe both block on a POD.
    assert "asyncio.to_thread(\n" not in src.replace(" ", "")
    for call in ("_post_page_to_pod", "_pod_still_alive"):
        i = src.index(call)
        before = src[max(0, i - 400):i]
        assert "run_on_pod_pool" in before, f"{call} is not on the pod pool"


def test_the_pod_pool_is_not_the_default_pool(holder):
    holder(512)
    thread_pool._pod_pool = None
    try:
        ex = thread_pool.pod_executor()
        assert ex._max_workers == 1024          # 512 * 2
        assert ex._max_workers > thread_pool.pool_sizes()[0]
    finally:
        thread_pool._pod_pool = None


def test_run_on_pod_pool_preserves_contextvars(holder):
    """A page that lost its token budget because it changed pools would be a
    nasty way to discover the pool changed."""
    import contextvars
    holder(64)
    var = contextvars.ContextVar("probe", default="unset")

    async def go():
        var.set("set-by-caller")
        return await thread_pool.run_on_pod_pool(var.get)

    thread_pool._pod_pool = None
    try:
        assert asyncio.run(go()) == "set-by-caller"
    finally:
        thread_pool._pod_pool = None


def test_the_pod_pool_grows_when_the_config_grows(holder):
    holder(64)
    thread_pool._pod_pool = None
    try:
        assert thread_pool.pod_executor()._max_workers == 128
        holder(512)
        assert thread_pool.resize_pod_pool() == 1024
        assert thread_pool._pod_pool._max_workers == 1024
        # ...and never shrinks: threads blocked on a page would have to go
        # somewhere, and the answer that loses no pages is "nowhere".
        holder(64)
        assert thread_pool.resize_pod_pool() == 1024
        assert thread_pool._pod_pool._max_workers == 1024
    finally:
        thread_pool._pod_pool = None


def test_configure_resizes_the_pod_pool(holder):
    """Raising max_concurrency through the admin API has to reach the threads,
    or it only widens a semaphore that then queues on a pool it never touched."""
    from codai.config import OcrConfig
    from codai.ocr.manager import OcrManager

    holder(64)
    thread_pool._pod_pool = None
    try:
        thread_pool.pod_executor()
        m = OcrManager()
        cfg = OcrConfig()
        cfg.max_concurrency = 1
        m.configure(cfg)
        holder(400)
        cfg2 = OcrConfig()
        cfg2.max_concurrency = 400          # a change, so pools are rebuilt
        m.configure(cfg2)
        assert thread_pool._pod_pool._max_workers == 800
    finally:
        thread_pool._pod_pool = None
