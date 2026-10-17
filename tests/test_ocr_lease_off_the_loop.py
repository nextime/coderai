"""A page waiting for a pod must not stop the whole server.

`placement.choose` is synchronous and ends in a pool's `acquire()`, which waits
on a condition variable for up to its default 1200 s and may rent a pod INLINE
-- a candidate sweep over every region, then a boot wait, in whichever thread
asked for the lease. Until 0.4.16 it was called straight from `_read_on_pods`'s
coroutine, so it ran ON the event loop: one page waiting for a pod blocked every
other request in the process, including /v1/runpod/status and the client's next
document. Measured on a 5,000-document run: 116 pods rented, and zero documents
completed in ten minutes while the front answered nothing at all.

It had been on the loop since the pod pipeline was written (0.3.70), and it cost
nothing for as long as acquire() returned at once -- a pool with a free pod
answers in microseconds. It turns fatal exactly when the pool is the thing under
strain, which is when it matters most.

It is also invisible to a test that awaits one page at a time: one page blocking
the loop for 0.5 s looks exactly like one page taking 0.5 s. It shows up only
when something else needs the loop WHILE the lease waits, which is what these
tests arrange.

The same applies to the per-page PDF split: pdfium permits exactly one caller at
a time, so every document's split serialises against every other one, and on the
loop that stalls the process rather than only the document.
"""

import asyncio
import threading
import time
import types

import pytest

from codai.ocr import manager as om
from codai.ocr.base import OcrError


@pytest.fixture
def anyio_backend():
    return "asyncio"


class _PodLease:
    kind = "pod"
    base = "https://10.0.0.1:1234/v1"
    api_key = "k"

    def release(self):
        pass

    def dead(self):
        pass


def _cfg(**kw):
    from codai.config import OcrConfig
    c = OcrConfig()
    # Both twins get stubbed below, but say it anyway: a policy test that
    # silently changed transport would be testing the other path.
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def _page_read(monkeypatch):
    reply = {"engine": "surya", "pages": [{"text": "ok"}]}

    async def _aok(*_a, **_k):
        return reply

    monkeypatch.setattr(om, "_post_page_to_pod", lambda *_a, **_k: reply)
    monkeypatch.setattr(om, "_apost_page_to_pod", _aok)


def _blocking_choose(monkeypatch, seconds: float, record=None):
    """A `choose` that blocks its THREAD, the way a real acquire() does."""
    from codai.models import placement

    def _choose(*_a, **_k):
        if record is not None:
            record.append(threading.current_thread())
        time.sleep(seconds)
        return _PodLease()

    monkeypatch.setattr(placement, "choose", _choose)
    monkeypatch.setattr(placement, "spill_threshold", lambda *_a, **_k: 0)
    monkeypatch.setattr(placement, "why_no_candidates", lambda *_a, **_k: "busy")


async def _worst_gap(stop: asyncio.Event) -> float:
    """The longest the loop went WITHOUT running us, until `stop` is set.

    Counting ticks over a fixed window does not detect this: a window that
    starts before the block and ends after it simply runs its ticks late, and
    the count comes out the same. The GAP is the thing -- a loop blocked for
    half a second has one half-second gap in it and no number of ticks on
    either side hides that.
    """
    worst = 0.0
    last = time.monotonic()
    while not stop.is_set():
        await asyncio.sleep(0.01)
        now = time.monotonic()
        worst = max(worst, now - last)
        last = now
    return worst


# --------------------------------------------------------------- the lease #

@pytest.mark.anyio
async def test_a_lease_that_waits_does_not_block_the_loop(monkeypatch):
    """THE REGRESSION. A page waiting 0.5 s for a pod, and the loop still runs."""
    _blocking_choose(monkeypatch, 0.5)
    _page_read(monkeypatch)
    stop = asyncio.Event()
    ticker = asyncio.ensure_future(_worst_gap(stop))
    await asyncio.sleep(0.02)                 # let the ticker take its first tick
    lease = await om._lease_a_pod("m", "prefer", 0.0)
    stop.set()
    gap = await ticker
    assert lease is not None
    assert gap < 0.2, (f"the loop was blocked for {gap:.2f}s while one page "
                       f"waited for a pod")


@pytest.mark.anyio
async def test_the_lease_runs_off_the_main_thread(monkeypatch):
    seen: list = []
    _blocking_choose(monkeypatch, 0.0, record=seen)
    await om._lease_a_pod("m", "prefer", 0.0)
    assert seen and seen[0] is not threading.main_thread()


@pytest.mark.anyio
async def test_many_pages_lease_concurrently_not_one_after_another(monkeypatch):
    """Leases overlap: 12 pages each waiting 0.3 s take ~0.3 s, not ~3.6 s."""
    _blocking_choose(monkeypatch, 0.3)
    _page_read(monkeypatch)
    t0 = time.monotonic()
    out = await asyncio.gather(*[om._lease_a_pod("m", "prefer", 0.0)
                                 for _ in range(12)])
    took = time.monotonic() - t0
    assert all(x is not None for x in out)
    assert took < 1.5, f"12 overlapping leases took {took:.2f}s — serialised"


@pytest.mark.anyio
async def test_an_error_from_choose_still_reaches_the_caller(monkeypatch):
    """Moving it to a thread must not swallow the refusal."""
    from codai.models import placement

    def _boom(*_a, **_k):
        raise OcrError("no upstream", status=503)

    monkeypatch.setattr(placement, "choose", _boom)
    monkeypatch.setattr(placement, "spill_threshold", lambda *_a, **_k: 0)
    with pytest.raises(OcrError):
        await om._lease_a_pod("m", "require", 0.0)


@pytest.mark.anyio
async def test_a_plain_exception_from_choose_is_still_a_503(monkeypatch):
    from codai.models import placement

    def _boom(*_a, **_k):
        raise RuntimeError("the pool is gone")

    monkeypatch.setattr(placement, "choose", _boom)
    monkeypatch.setattr(placement, "spill_threshold", lambda *_a, **_k: 0)
    with pytest.raises(OcrError) as e:
        await om._lease_a_pod("m", "prefer", 0.0)
    assert int(getattr(e.value, "status", 0) or 0) == 503
    assert "the pool is gone" in str(e.value)


# ---------------------------------------------------------------- the split #

@pytest.mark.anyio
async def test_the_pdf_split_runs_off_the_main_thread(monkeypatch):
    """pdfium is one-caller-at-a-time; on the loop one split stalls everything."""
    seen: list = []

    def _split(_d):
        seen.append(threading.current_thread())
        return [b"%PDF-page0"]

    monkeypatch.setattr(om, "iter_pdf_page_pdfs", _split)
    _blocking_choose(monkeypatch, 0.0)
    _page_read(monkeypatch)
    mgr = om.OcrManager()
    await mgr._read_on_pods(b"%PDF-1.4 x", "d.pdf", "application/pdf", "m",
                            dpi=200, page_opts=None, detect=None,
                            cfg=_cfg(), engine_name="surya")
    assert seen and seen[0] is not threading.main_thread()


@pytest.mark.anyio
async def test_a_slow_split_does_not_block_the_loop(monkeypatch):
    def _split(_d):
        time.sleep(0.4)
        return [b"%PDF-page0"]

    monkeypatch.setattr(om, "iter_pdf_page_pdfs", _split)
    _blocking_choose(monkeypatch, 0.0)
    _page_read(monkeypatch)
    mgr = om.OcrManager()
    stop = asyncio.Event()
    ticker = asyncio.ensure_future(_worst_gap(stop))
    await asyncio.sleep(0.02)
    await mgr._read_on_pods(b"%PDF-1.4 x", "d.pdf", "application/pdf", "m",
                            dpi=200, page_opts=None, detect=None,
                            cfg=_cfg(), engine_name="surya")
    stop.set()
    gap = await ticker
    assert gap < 0.2, f"the loop was blocked for {gap:.2f}s by a PDF split"


@pytest.mark.anyio
async def test_a_single_image_is_still_not_split(monkeypatch):
    """Nothing to split, and no thread hop to pay for finding that out."""
    called: list = []
    monkeypatch.setattr(om, "iter_pdf_page_pdfs",
                        lambda _d: called.append(1) or [b"x"])
    _blocking_choose(monkeypatch, 0.0)
    _page_read(monkeypatch)
    mgr = om.OcrManager()
    out = await mgr._read_on_pods(b"\x89PNG\r\n", "p.png", "image/png", "m",
                                  dpi=200, page_opts=None, detect=None,
                                  cfg=_cfg(), engine_name="surya")
    assert not called
    assert out is not None


# ------------------------------------------------- coping with a BOOT wait #
# A pod takes minutes to come up: image pull, vLLM load, weights. Everything
# below is about that wait being survivable rather than being paid for in
# threads or handed back to this box.

def test_one_attempt_is_bounded():
    """An unbounded attempt parks a thread for a whole boot, per page."""
    assert 0 < om.POD_LEASE_ATTEMPT_S <= 60


def test_the_ceiling_covers_a_real_boot():
    """60 s gave a cold start's whole first wave to this box."""
    from codai.config import OcrConfig
    _, wait = om._pod_pipeline_policy(OcrConfig())
    assert wait >= 300, f"a pod boots in minutes; the ceiling is {wait}s"


def test_the_loop_is_still_the_transport():
    """The lease moved to a thread; the FORWARD must not follow it."""
    from codai.config import OcrConfig
    assert OcrConfig().pod_http_async is True
    assert om._use_async_pod_http(OcrConfig()) is True


@pytest.mark.anyio
async def test_the_forward_still_runs_on_the_loop(monkeypatch):
    """Where the page actually goes: `_apost_page_to_pod`, on this thread."""
    seen: list = []

    async def _aok(*_a, **_k):
        seen.append(threading.current_thread())
        return {"engine": "surya", "pages": [{"text": "ok"}]}

    monkeypatch.setattr(om, "_apost_page_to_pod", _aok)
    monkeypatch.setattr(om, "_post_page_to_pod",
                        lambda *_a, **_k: pytest.fail("took the thread path"))
    monkeypatch.setattr(om, "iter_pdf_page_pdfs", lambda _d: [b"%PDF-page0"])
    _blocking_choose(monkeypatch, 0.0)
    mgr = om.OcrManager()
    out = await mgr._read_on_pods(b"%PDF-1.4 x", "d.pdf", "application/pdf",
                                  "m", dpi=200, page_opts=None, detect=None,
                                  cfg=_cfg(), engine_name="surya")
    assert out is not None
    assert seen and seen[0] is threading.main_thread()


@pytest.mark.anyio
async def test_a_page_waits_through_a_boot_instead_of_going_local(monkeypatch):
    """The pool has nothing for 0.4 s, then a pod. The page must take the pod."""
    from codai.models import placement

    class _Local:
        kind = "local"
        base = "http://127.0.0.1:8000/v1"
        api_key = ""

        def release(self):
            pass

    t0 = time.monotonic()

    def _choose(*_a, **_k):
        return _PodLease() if time.monotonic() - t0 > 0.4 else _Local()

    monkeypatch.setattr(placement, "choose", _choose)
    monkeypatch.setattr(placement, "spill_threshold", lambda *_a, **_k: 0)
    monkeypatch.setattr(placement, "why_no_candidates",
                        lambda *_a, **_k: "the pods are booting")
    om._POD_WAIT_COOLDOWN.pop("m", None)
    lease = await om._lease_a_pod("m", "prefer", 30.0)
    assert lease is not None and lease.kind == "pod"


@pytest.mark.anyio
async def test_waiting_out_a_boot_does_not_park_a_thread_per_page(monkeypatch):
    """24 pages waiting for a booting pool, and the pod pool stays nearly free.

    This is what makes a long ceiling affordable: the wait is on the loop, and a
    thread is touched only for the length of one bounded attempt.
    """
    from codai.api import thread_pool
    from codai.models import placement

    class _Local:
        kind = "local"
        base = "http://127.0.0.1:8000/v1"
        api_key = ""

        def release(self):
            pass

    held = {"seconds": 0.0}
    lock = threading.Lock()
    t0 = time.monotonic()

    def _choose(*_a, **_k):
        start = time.monotonic()
        try:
            time.sleep(0.05)          # what one bounded attempt costs
            return _PodLease() if time.monotonic() - t0 > 0.5 else _Local()
        finally:
            with lock:
                held["seconds"] += time.monotonic() - start

    monkeypatch.setattr(placement, "choose", _choose)
    monkeypatch.setattr(placement, "spill_threshold", lambda *_a, **_k: 0)
    monkeypatch.setattr(placement, "why_no_candidates", lambda *_a, **_k: "booting")
    om._POD_WAIT_COOLDOWN.pop("m", None)
    out = await asyncio.gather(*[om._lease_a_pod("m", "prefer", 30.0)
                                 for _ in range(24)])
    assert all(x is not None and x.kind == "pod" for x in out)
    # 24 pages each waiting out a 0.5 s "boot" is 12 thread-seconds if the wait
    # is held in a thread. Bounded attempts with the waiting on the loop cost
    # only the attempts: a few hundredths each, a couple of attempts apiece.
    parked = 24 * 0.5
    assert held["seconds"] < parked / 2, (
        f"{held['seconds']:.2f}s of thread time held against {parked:.1f}s if "
        f"every page had parked a thread for the boot")
    assert thread_pool.pod_executor() is not None
