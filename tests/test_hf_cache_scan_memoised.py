"""The HF cache scan was walking 900 GB of blobs on the event loop.

``huggingface_hub.scan_cache_dir`` stats every blob of every repo. Measured on
this machine: **27 seconds**. It was called from a per-model VRAM estimate, a
GGUF path lookup and an engine-state poll -- the last of which is polled -- so
the loop watchdog recorded 6631 stalls over 0.5s with a worst of 13.0s, during
every one of which no request of any kind was served.
"""
import time

import pytest


def test_the_scan_is_memoised():
    import codai.models.cache as C
    calls = []

    class _Info:
        repos = ()

    def _fake(_dir):
        calls.append(_dir)
        return _Info()

    C.invalidate_hf_cache_scan()
    import huggingface_hub
    orig = huggingface_hub.scan_cache_dir
    huggingface_hub.scan_cache_dir = _fake
    try:
        a = C._scan_hf_cache("/tmp/hfcache-x")
        b = C._scan_hf_cache("/tmp/hfcache-x")
        c = C._scan_hf_cache("/tmp/hfcache-x")
    finally:
        huggingface_hub.scan_cache_dir = orig
    assert a is b is c
    assert len(calls) == 1, calls


def test_different_cache_dirs_are_memoised_separately():
    import codai.models.cache as C
    calls = []

    class _Info:
        def __init__(self, d):
            self.repos = ()
            self.d = d

    import huggingface_hub
    orig = huggingface_hub.scan_cache_dir
    huggingface_hub.scan_cache_dir = lambda d: (calls.append(d), _Info(d))[1]
    try:
        C.invalidate_hf_cache_scan()
        a = C._scan_hf_cache("/tmp/one")
        b = C._scan_hf_cache("/tmp/two")
    finally:
        huggingface_hub.scan_cache_dir = orig
    assert a is not b
    assert calls == ["/tmp/one", "/tmp/two"]


def test_invalidation_forces_a_fresh_scan():
    # A model downloaded while the server runs has to become visible.
    import codai.models.cache as C
    calls = []
    import huggingface_hub
    orig = huggingface_hub.scan_cache_dir
    huggingface_hub.scan_cache_dir = lambda d: (calls.append(d), type(
        "I", (), {"repos": ()})())[1]
    try:
        C.invalidate_hf_cache_scan()
        C._scan_hf_cache("/tmp/inv")
        C._scan_hf_cache("/tmp/inv")
        assert len(calls) == 1
        C.invalidate_hf_cache_scan()
        C._scan_hf_cache("/tmp/inv")
        assert len(calls) == 2
    finally:
        huggingface_hub.scan_cache_dir = orig


def test_the_ttl_is_short_enough_to_notice_a_download():
    # Permanent caching would hide a model that arrived after boot; the window
    # has to be far below how long any download takes and far above the poll
    # interval.
    import codai.models.cache as C
    assert 5.0 <= C._HF_SCAN_TTL <= 120.0


def test_a_failed_scan_answers_none_rather_than_raising():
    import codai.models.cache as C
    import huggingface_hub
    orig = huggingface_hub.scan_cache_dir

    def _boom(_d):
        raise OSError("permission denied")

    huggingface_hub.scan_cache_dir = _boom
    try:
        C.invalidate_hf_cache_scan()
        assert C._scan_hf_cache("/tmp/boom") is None
    finally:
        huggingface_hub.scan_cache_dir = orig


def test_every_estimator_site_uses_the_memoised_scan():
    """The four call sites in the VRAM estimator, which is the hot path.

    Asserted on the source because the alternative is a 27-second test.
    """
    import inspect

    from codai.models import manager
    src = inspect.getsource(manager)
    memoised = src.count(
        "from codai.models.cache import _scan_hf_cache as scan_cache_dir")
    direct = src.count("from huggingface_hub import scan_cache_dir")
    assert memoised == 4, memoised
    assert direct == 0, "an estimator site still walks the cache directly"
    # Each site must also treat the None answer as "not found", the way it
    # treated an exception before.
    assert src.count("if info is None:") == 4


def test_the_engine_state_poll_does_not_block_the_loop():
    import inspect

    from codai.api import app
    src = inspect.getsource(app.internal_engine_state)
    assert "asyncio.to_thread(_footprint" in src, (
        "the per-model VRAM estimate is back on the event loop")
