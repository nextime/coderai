"""Parallel PDF rasterisation, one document per process.

pdfium permits one caller at a time. 0.3.65 learned that the hard way (concurrent
access reports "PDFium: Data format error" on valid documents) and 0.3.66 fixed it
with a lock -- correct, but it made rasterisation single-threaded, a ceiling of
about one core for the whole install.

pdfium is safe one-document-per-process, so this module keeps a small pool of
worker processes (``codai/ocr/workers/raster_worker.py``) and hands each document
to one of them. Parallelism comes back, and the unsafe library is never called
concurrently inside any single process.

Design notes worth keeping:

* **A document is pinned to one worker** for its whole life. That is not a
  limitation to work around -- it is the property that makes this safe, and it
  gives exactly the parallelism that matters (documents at once, which is what a
  batch has many of) rather than the kind that does not (pages of one document,
  which the OCR engines are the bottleneck for anyway).

* **Workers are not imported, they are executed.** ``import codai.ocr`` costs
  377 MB and 1.5 s via litellm; the worker script imports pypdfium2 and PIL only,
  about 25 MB. This is why the pool is not a ``ProcessPoolExecutor``: spawn
  pickles a function by qualified name and re-imports the package to find it.

* **Every failure falls back in-process**, under the 0.3.66 lock. A pool that
  cannot start, a worker that dies mid-document, a build without the worker
  script: none of them may turn into an OCR outage. Correctness does not depend
  on the optimisation working.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import subprocess
import sys
import threading
from typing import Optional

from codai.ocr.base import OcrError, render_pdf_range
from codai.ocr.workers import page_image

_WORKER = os.path.join(os.path.dirname(__file__), "workers", "raster_worker.py")

#: Rasterising is CPU-bound and competes with the engines for cores, so the auto
#: setting takes half of them and never more than 4. More workers buy nothing once
#: the engines are the bottleneck, and each one is another ~25 MB plus its window
#: of pages.
MAX_AUTO_WORKERS = 4


def auto_workers() -> int:
    try:
        n = len(os.sched_getaffinity(0))
    except Exception:
        n = os.cpu_count() or 1
    return max(1, min(MAX_AUTO_WORKERS, n // 2 or 1))


class _Worker:
    """One rasteriser process. Driven by one caller at a time."""

    def __init__(self):
        self.proc = subprocess.Popen(
            [sys.executable, "-I", _WORKER],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            bufsize=0,
        )
        hello = self._reply()
        if not hello.get("ready"):
            self.kill()
            raise OcrError(hello.get("error") or "rasteriser worker not ready",
                           status=500)

    # -- framing ---------------------------------------------------------- #
    def _reply(self) -> dict:
        line = self.proc.stdout.readline()
        if not line:
            raise OcrError("rasteriser worker exited", status=500)
        return json.loads(line.decode("utf-8"))

    def _read_exactly(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self.proc.stdout.read(n - len(buf))
            if not chunk:
                raise OcrError("rasteriser worker closed mid-frame", status=500)
            buf.extend(chunk)
        return bytes(buf)

    def _send(self, msg: dict, payload: bytes = b"") -> None:
        self.proc.stdin.write(json.dumps(msg).encode("utf-8") + b"\n")
        if payload:
            self.proc.stdin.write(payload)
        self.proc.stdin.flush()

    # -- commands --------------------------------------------------------- #
    def open(self, data: bytes, dpi: int, policy=None) -> "tuple[int, int]":
        self._send({"cmd": "open", "dpi": int(dpi), "nbytes": len(data),
                    "page_image": (policy or page_image.DEFAULT).as_dict()}, data)
        r = self._reply()
        if not r.get("ok"):
            # A bad document is the document's problem, not the worker's: raise
            # it as the 400 it is and leave this process in the pool.
            raise OcrError(r.get("error") or "failed to rasterise PDF",
                           status=int(r.get("status", 400)))
        return int(r.get("pages") or 0), int(r.get("est") or 0)

    def render(self, start: int, count: int) -> list:
        from PIL import Image
        self._send({"cmd": "render", "start": int(start), "count": int(count)})
        r = self._reply()
        if not r.get("ok"):
            raise OcrError(r.get("error") or "failed to rasterise PDF",
                           status=int(r.get("status", 400)))
        metas = r.get("pages") or []
        out = []
        for m in metas:
            raw = self._read_exactly(int(m["nbytes"]))
            mime = m.get("mime")
            if mime:
                # The worker never rendered this page: these are the bytes the
                # PDF itself holds. Image.open is lazy, so the pixels are only
                # decoded if something downstream actually looks at them -- and
                # the bytes ride along so the engine can send them onwards
                # without an encode. Carried, not trusted: any transform drops
                # them (see page_image.attach).
                import io as _io
                img = Image.open(_io.BytesIO(raw))
                out.append(page_image.attach(img, mime, raw))
                continue
            out.append(Image.frombytes("RGB", (int(m["w"]), int(m["h"])), raw))
        return out

    def close_doc(self) -> None:
        with contextlib.suppress(Exception):
            self._send({"cmd": "close"})
            self._reply()

    def alive(self) -> bool:
        return self.proc.poll() is None

    def kill(self) -> None:
        with contextlib.suppress(Exception):
            self.proc.kill()
        with contextlib.suppress(Exception):
            self.proc.wait(timeout=5)


class RasterPool:
    """A fixed set of rasteriser processes, lent out one document at a time."""

    #: How long a document waits for a BUSY worker before giving up and
    #: rasterising in-process instead. Falling back makes progress; waiting
    #: longer just moves the queue somewhere less visible.
    WAIT_SECONDS = 30.0

    def __init__(self, size: int):
        self.size = max(1, int(size))
        self._free: "queue.Queue[_Worker]" = queue.Queue()
        self._made = 0
        self._lock = threading.Lock()
        self._closed = False
        #: Set when a worker could not be created at all (missing script, no fds).
        #: Waiting for one then is waiting forever, which is how an optimisation
        #: that cannot start becomes an outage -- the first version of this class
        #: stalled every document for 300 s in exactly that case.
        self._broken = False

    def _make(self) -> Optional[_Worker]:
        try:
            return _Worker()
        except Exception as exc:
            print(f"[ocr] rasteriser worker would not start ({exc}); "
                  f"rasterising in-process instead", flush=True)
            return None

    def acquire(self) -> Optional["_Worker"]:
        """Take a worker, or None when the pool cannot serve.

        None rather than an exception is deliberate: the caller's fallback is a
        correct in-process render, so an unavailable pool costs latency, never an
        error.
        """
        if self._broken or self._closed:
            return None
        w = None
        try:
            w = self._free.get_nowait()
        except queue.Empty:
            born = False
            with self._lock:
                if not self._closed and self._made < self.size:
                    self._made += 1
                    w = self._make()
                    if w is None:
                        self._made -= 1
                        self._broken = True
                    else:
                        born = True
            if w is None and not born and not self._closed:
                if self._broken or self._made == 0:
                    # Nothing to wait for: either no worker can be created, or
                    # none exists to come back.
                    return None
                # Every worker is busy. Wait briefly, then rasterise in-process
                # rather than queue indefinitely behind the pool.
                try:
                    w = self._free.get(timeout=self.WAIT_SECONDS)
                except queue.Empty:
                    w = None
        if w is not None and not w.alive():
            with self._lock:
                self._made = max(0, self._made - 1)
            w = None
        return w

    def release(self, w: Optional["_Worker"]) -> None:
        """Return a worker. A dead one is replaced by capacity, not by a corpse."""
        if w is None:
            return
        if w.alive() and not self._closed:
            w.close_doc()
            self._free.put(w)
        else:
            w.kill()
            with self._lock:
                self._made = max(0, self._made - 1)

    @contextlib.contextmanager
    def borrow(self):
        """acquire/release as a block, for callers that do not outlive it."""
        w = self.acquire()
        try:
            yield w
        finally:
            self.release(w)

    def close(self) -> None:
        self._closed = True
        while True:
            try:
                self._free.get_nowait().kill()
            except queue.Empty:
                return


_pool: Optional[RasterPool] = None
_pool_size = 0
_pool_lock = threading.Lock()


def get_pool(raster_workers: int = 0) -> Optional[RasterPool]:
    """The process-wide pool, or None when the operator asked for in-process.

    ``raster_workers`` of 1 means "do it here, under the lock" -- the 0.3.66
    behaviour, kept reachable because it is the one configuration with no IPC and
    no extra processes, which a small host may genuinely prefer.
    """
    global _pool, _pool_size
    want = int(raster_workers) if raster_workers else auto_workers()
    if want <= 1:
        return None
    with _pool_lock:
        if _pool is None or _pool_size != want:
            if _pool is not None:
                _pool.close()
            _pool = RasterPool(want)
            _pool_size = want
        return _pool


def reset_pool() -> None:
    """Drop the pool (tests, config reload)."""
    global _pool, _pool_size
    with _pool_lock:
        if _pool is not None:
            _pool.close()
        _pool = None
        _pool_size = 0


def probe_pdf(data: bytes, dpi: int, raster_workers: int = 0, policy=None):
    """``(pages, retained_bytes)`` for a PDF, in a worker when there is one.

    Returns None when the pool could not serve, so the caller uses its own
    in-process estimate rather than treating a missing optimisation as an error.
    """
    pool = get_pool(raster_workers)
    if pool is None:
        return None
    try:
        with pool.borrow() as w:
            if w is None:
                return None
            return w.open(data, dpi, policy)
    except OcrError:
        raise                       # a bad document, not a pool problem
    except Exception as exc:
        print(f"[ocr] rasteriser probe failed ({exc}); using the in-process "
              f"estimate", flush=True)
        return None


def iter_pdf_pages(data: bytes, dpi: int, chunk: int = 8,
                   raster_workers: int = 0, policy=None):
    """Yield RGB pages from worker processes, or None if the pool cannot serve.

    A generator that yields nothing and returns is indistinguishable from an empty
    document, so unavailability is signalled by returning None from this function
    BEFORE any generator is created -- the caller checks, then falls back.

    A worker is borrowed PER CHUNK, not per document. The caller OCRs each chunk
    between pulls, which takes far longer than rasterising it, so holding a worker
    across that would idle it for most of its tenure and cap the install at
    `size` concurrent documents -- worse than the lock this replaces. The cost of
    letting go is re-opening the document per chunk; pdfium parses a page tree in
    milliseconds, and `chunk` is the caller's whole page window, so a 40-page
    document pays it twice rather than ten times.
    """
    pool = get_pool(raster_workers)
    if pool is None:
        return None
    # Probe once, up front: this is also the "can the pool serve at all" question,
    # and answering it before any generator exists is what lets the caller fall
    # back cleanly instead of mistaking an empty generator for an empty document.
    w = pool.acquire()
    if w is None:
        return None
    try:
        n, _est = w.open(data, dpi, policy)
    finally:
        pool.release(w)

    def _gen():
        start = 0
        while start < n:
            count = max(1, chunk)
            worker = pool.acquire()
            if worker is None:
                # The pool stopped serving mid-document. Rasterise the rest here
                # rather than fail a document that is already half read.
                for img in render_pdf_range(data, dpi, start, n, policy):
                    yield img
                return
            try:
                worker.open(data, dpi, policy)
                pages = worker.render(start, count)
            finally:
                pool.release(worker)
            if not pages:
                return
            for p in pages:
                yield p
            start += len(pages)

    return _gen()

