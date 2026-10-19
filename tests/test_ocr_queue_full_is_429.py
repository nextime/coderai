"""A full queue must say so, with 429, not answer slower and slower.

The queue in front of the pods is deliberate: when a pod finishes a page there
has to be another ready to hand it, or the fleet idles between every dispatch
and is paid for the gap. So the server carries `ocr.max_concurrency` pages in
flight PLUS `ocr.max_queued_pages` waiting.

What it did past that point was accept anyway. A client offering several times
the capacity got no error -- it got replies that took longer and longer, which
reads as "the pages are getting slower" rather than "stop sending", so it sent
more. Measured 2026-10-19: 522 pages offered at once against a fleet that could
carry ~250; mean page latency 238 s against a 19 s service time; nothing in any
reply said the server was over capacity.

  1. under the limit nothing changes -- the queue is supposed to be full
  2. past the limit the answer is 429 with an actionable Retry-After
  3. the refusal happens BEFORE the upload is read
  4. the accounting can never itself refuse work
"""
import asyncio

import pytest

from codai.ocr import manager as ocr_manager


class _Cfg:
    def __init__(self, width=8, queued=0):
        self.max_concurrency = width
        self.max_queued_pages = queued


def _mgr(width=8, queued=0, running=0, waiting=0):
    m = ocr_manager.OcrManager.__new__(ocr_manager.OcrManager)
    m._cfg = _Cfg(width, queued)
    m._pages_running = running
    m._pages_queued = waiting
    m._sem = None
    return m


# --- 1. the queue is meant to be full ------------------------------------

def test_an_idle_server_is_not_full():
    assert _mgr().admission()["full"] is False


def test_a_fleet_at_full_width_with_an_empty_queue_is_not_full():
    """Every slot busy is the server working, not the server overloaded."""
    st = _mgr(width=8, running=8, waiting=0).admission()
    assert st["full"] is False


def test_a_partly_filled_queue_is_not_full():
    """This is the state the queue exists to be in: pods never starved."""
    st = _mgr(width=8, running=8, waiting=4).admission()
    assert st["full"] is False
    assert st["queue_limit"] == 8


def test_the_queue_limit_follows_max_concurrency_when_unset():
    assert _mgr(width=64).admission()["queue_limit"] == 64


def test_an_explicit_queue_limit_is_obeyed():
    assert _mgr(width=64, queued=16).admission()["queue_limit"] == 16


# --- 2. past the limit, 429 ----------------------------------------------

def test_a_full_queue_is_full():
    st = _mgr(width=8, running=8, waiting=8).admission()
    assert st["full"] is True


def test_the_endpoint_raises_429(monkeypatch):
    from fastapi import HTTPException
    from codai.api import ocr as api_ocr
    monkeypatch.setattr(api_ocr.ocr_manager, "admission",
                        lambda: {"running": 8, "queued": 8, "width": 8,
                                 "queue_limit": 8, "full": True})
    with pytest.raises(HTTPException) as got:
        api_ocr._refuse_if_full(None)
    assert got.value.status_code == 429


def test_the_429_carries_a_retry_after(monkeypatch):
    """Without it a client can only guess, and guessing means retrying at once
    -- which makes every accepted page slower, including its own."""
    from fastapi import HTTPException
    from codai.api import ocr as api_ocr
    monkeypatch.setattr(api_ocr.ocr_manager, "admission",
                        lambda: {"running": 8, "queued": 8, "width": 8,
                                 "queue_limit": 8, "full": True})
    with pytest.raises(HTTPException) as got:
        api_ocr._refuse_if_full(None)
    hdrs = got.value.headers or {}
    assert "Retry-After" in hdrs
    assert int(hdrs["Retry-After"]) >= 1


def test_the_429_says_what_the_numbers_are(monkeypatch):
    """A client tuning its window needs the width and the depth, not a
    sentence. These are the two numbers it would otherwise have to discover by
    bisection against a production fleet."""
    from fastapi import HTTPException
    from codai.api import ocr as api_ocr
    monkeypatch.setattr(api_ocr.ocr_manager, "admission",
                        lambda: {"running": 5, "queued": 9, "width": 8,
                                 "queue_limit": 9, "full": True})
    with pytest.raises(HTTPException) as got:
        api_ocr._refuse_if_full(None)
    hdrs = got.value.headers or {}
    assert hdrs["X-OCR-Page-Width"] == "8"
    assert hdrs["X-OCR-Queue-Limit"] == "9"
    assert hdrs["X-OCR-Pages-Queued"] == "9"


def test_a_server_under_the_limit_is_not_refused(monkeypatch):
    from codai.api import ocr as api_ocr
    monkeypatch.setattr(api_ocr.ocr_manager, "admission",
                        lambda: {"running": 8, "queued": 1, "width": 8,
                                 "queue_limit": 8, "full": False})
    assert api_ocr._refuse_if_full(None) is None


# --- 3. refused before the body is read ----------------------------------

@pytest.mark.parametrize("fn", ["create_ocr", "create_ocr_batch"])
def test_both_endpoints_refuse_before_reading_the_upload(fn):
    """Twice as important as it looks: the client hears at once instead of
    after transferring a document the server will not look at, and the server
    does not buffer megabytes it is about to discard. In-flight memory here is
    concurrent requests times upload size, and that product has OOM-killed it."""
    import inspect
    from codai.api import ocr as api_ocr
    src = inspect.getsource(getattr(api_ocr, fn))
    assert "_refuse_if_full(cfg)" in src, fn
    assert src.index("_refuse_if_full(cfg)") < src.index("await f"), fn


# --- 4. the accounting must never be the thing that refuses --------------

def test_broken_accounting_does_not_refuse_work(monkeypatch):
    from codai.api import ocr as api_ocr

    def _boom():
        raise RuntimeError("counters are confused")
    monkeypatch.setattr(api_ocr.ocr_manager, "admission", _boom)
    assert api_ocr._refuse_if_full(None) is None


def test_admission_survives_a_config_that_says_nonsense():
    m = _mgr()
    m._cfg = type("C", (), {"max_concurrency": "eight",
                            "max_queued_pages": None})()
    st = m.admission()
    assert st["width"] >= 1 and st["queue_limit"] >= 1


# --- the counters are real -------------------------------------------------

def test_the_slot_counts_waiting_and_running():
    async def _run():
        m = _mgr(width=1)
        m._sem = asyncio.Semaphore(1)
        seen = {}

        async def _hold():
            async with m._page_slot():
                await asyncio.sleep(0.08)

        async def _wait():
            async with m._page_slot():
                pass

        h = asyncio.ensure_future(_hold())
        w = asyncio.ensure_future(_wait())
        # Observed from outside both, while the second is blocked on the slot
        # the first is holding: that is the state the 429 is derived from.
        await asyncio.sleep(0.03)
        seen["running"] = m._pages_running
        seen["queued"] = m._pages_queued
        await asyncio.gather(h, w)
        return seen, m

    seen, m = asyncio.run(_run())
    assert seen["running"] == 1
    assert seen["queued"] == 1, "a page waiting for a slot was not counted"
    assert m._pages_running == 0 and m._pages_queued == 0, "counters leaked"


def test_a_cancelled_page_does_not_leak_a_queue_slot():
    """A client that times out in bulk cancels every page of a batch at once.
    If a cancelled waiter stayed counted, the queue would fill permanently and
    the server would 429 everything for the rest of its life."""
    async def _run():
        m = _mgr(width=1)
        m._sem = asyncio.Semaphore(1)

        async def _hold():
            async with m._page_slot():
                await asyncio.sleep(0.2)

        async def _wait():
            async with m._page_slot():
                pass

        h = asyncio.ensure_future(_hold())
        w = asyncio.ensure_future(_wait())
        await asyncio.sleep(0.03)
        assert m._pages_queued == 1
        w.cancel()
        try:
            await w
        except asyncio.CancelledError:
            pass
        await asyncio.sleep(0.01)
        queued = m._pages_queued
        h.cancel()
        try:
            await h
        except asyncio.CancelledError:
            pass
        return queued

    assert asyncio.run(_run()) == 0


def test_the_fallback_semaphore_is_one_semaphore():
    """Caught by tests/test_ocr_pod_pipeline_gate.py when this was first
    written: `self._sem or asyncio.Semaphore(4)` built a FRESH semaphore per
    page, so an unconfigured manager bounded nothing -- forty pages of one
    document ran at once against what every line of code called a limit of
    four."""
    m = _mgr()
    m._sem = None
    m._fallback_sem = None
    assert m._slot_sem() is m._slot_sem()
    src = open("codai/ocr/manager.py").read()
    assert "self._sem or asyncio.Semaphore(4)" not in src


def test_a_configured_semaphore_wins_over_the_fallback():
    m = _mgr()
    real = asyncio.Semaphore(3)
    m._sem = real
    m._fallback_sem = asyncio.Semaphore(99)
    assert m._slot_sem() is real


def test_every_page_path_goes_through_the_counted_slot():
    """An uncounted path is a queue the server cannot see, so 429 would never
    fire for the traffic that matters."""
    src = open("codai/ocr/manager.py").read()
    assert "async with sem:" not in src
    assert src.count("async with self._page_slot():") >= 3
