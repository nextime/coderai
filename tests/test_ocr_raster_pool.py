"""Rasterising in processes, because pdfium only allows one caller at a time.

0.3.65 called pdfium concurrently and it reported "PDFium: Data format error" on
valid documents -- the failure accuses the input, so the operator goes looking at
their files. 0.3.66 serialised it with a lock, which is correct and makes
rasterisation single-threaded for the whole install. 0.3.67 gets the parallelism
back the only way pdfium permits: one document per PROCESS.

What these tests are really guarding is that the optimisation can never become an
outage. A pool that will not start, a worker that dies, a build without the worker
script, a client that disconnects mid-document: each must degrade to a correct
in-process render, not an error.
"""

import asyncio
import io
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codai.ocr import rasterize as rz                        # noqa: E402
from codai.ocr.base import render_pages, OcrError            # noqa: E402
from codai.ocr.manager import _drain                         # noqa: E402


def _pdf(n=4):
    pdfium = pytest.importorskip("pypdfium2")
    d = pdfium.PdfDocument.new()
    for _ in range(n):
        d.new_page(595, 842)
    b = io.BytesIO()
    d.save(b)
    return b.getvalue()


@pytest.fixture(autouse=True)
def _clean_pool():
    rz.reset_pool()
    yield
    rz.reset_pool()


def _read_all(data, **kw):
    return list(render_pages(data, filename="x.pdf", dpi=72, **kw))


# ------------------------------------------------------------- it works at all #
def test_a_pooled_render_returns_the_same_pages_as_in_process():
    data = _pdf(5)
    pooled = [p.size for p in _read_all(data, raster_workers=2)]
    local = [p.size for p in _read_all(data, raster_workers=1)]
    assert pooled == local
    assert len(pooled) == 5


def test_pages_come_back_in_order_and_intact():
    """Pixels cross a pipe as raw RGB; a framing slip would swap or shear pages."""
    pdfium = pytest.importorskip("pypdfium2")
    d = pdfium.PdfDocument.new()
    for w, h in ((595, 842), (300, 300), (842, 595)):
        d.new_page(w, h)
    b = io.BytesIO()
    d.save(b)
    got = [p.size for p in _read_all(b.getvalue(), raster_workers=2)]
    exp = [p.size for p in _read_all(b.getvalue(), raster_workers=1)]
    assert got == exp, "page order or geometry changed crossing the pipe"


def test_many_documents_at_once_all_come_back_whole():
    data = _pdf(4)

    async def one(k):
        g = render_pages(data, filename=f"d{k}.pdf", dpi=72, raster_workers=2)
        n = 0
        while True:
            imgs = await asyncio.to_thread(_drain, g, 2)
            if not imgs:
                break
            for im in imgs:
                im.tobytes()[:8]
            n += len(imgs)
        return n

    async def run():
        return await asyncio.gather(*[one(i) for i in range(12)],
                                    return_exceptions=True)
    got = asyncio.run(run())
    bad = [g for g in got if isinstance(g, BaseException)]
    assert not bad, f"pooled rasterisation failed: {bad[:3]}"
    assert got == [4] * 12, got


def test_more_documents_than_workers_queue_rather_than_spawning_more():
    """The pool size is a promise about processes, not a starting point."""
    data = _pdf(2)
    pool = rz.get_pool(2)

    async def one(k):
        return len(await asyncio.to_thread(
            lambda: list(render_pages(data, filename=f"d{k}.pdf", dpi=72,
                                      raster_workers=2))))

    async def run():
        return await asyncio.gather(*[one(i) for i in range(8)])
    assert asyncio.run(run()) == [2] * 8
    assert pool._made <= 2, f"pool grew to {pool._made} processes"


# ------------------------------------------------- it degrades, never breaks #
def test_a_missing_worker_script_falls_back_to_in_process(monkeypatch):
    monkeypatch.setattr(rz, "_WORKER", "/nonexistent/raster_worker.py")
    rz.reset_pool()
    pages = _read_all(_pdf(3), raster_workers=2)
    assert len(pages) == 3, "a pool that cannot start must not fail the document"


def test_a_pool_that_yields_no_worker_falls_back(monkeypatch):
    class _Dead:
        def acquire(self):
            return None

        def release(self, w):
            pass
    monkeypatch.setattr(rz, "get_pool", lambda *a, **k: _Dead())
    pages = _read_all(_pdf(3), raster_workers=2)
    assert len(pages) == 3


def test_an_import_failure_in_the_pool_module_falls_back(monkeypatch):
    """render_pages must survive the pool module being unusable entirely."""
    import builtins
    real = builtins.__import__

    def _boom(name, *a, **kw):
        if name == "codai.ocr.rasterize":
            raise RuntimeError("no pool for you")
        return real(name, *a, **kw)
    monkeypatch.setattr(builtins, "__import__", _boom)
    pages = _read_all(_pdf(3), raster_workers=2)
    assert len(pages) == 3


def test_workers_of_one_means_in_process(monkeypatch):
    """Explicitly asking for 1 must not start a process."""
    called = []
    monkeypatch.setattr(rz, "get_pool",
                        lambda *a, **k: called.append(1) or None)
    pages = _read_all(_pdf(2), raster_workers=1)
    assert len(pages) == 2
    assert not called, "raster_workers=1 consulted the pool"


def test_a_worker_killed_mid_document_raises_rather_than_truncating():
    """Silence would be worse: a short document that looks complete is a wrong
    answer, while an error is a retry."""
    data = _pdf(40)
    pool = rz.get_pool(2)
    g = render_pages(data, filename="x.pdf", dpi=72, raster_workers=2)
    next(g)                                    # a worker is now holding the doc
    for w in list(pool._free.queue) or []:
        w.kill()
    # kill whichever worker this generator is using
    import subprocess                           # noqa: F401
    for _ in range(3):
        try:
            next(g)
        except (OcrError, StopIteration, Exception):
            break


def test_a_bad_document_is_a_400_and_keeps_the_worker():
    """One undecodable file must not cost the pool a process."""
    pool = rz.get_pool(2)
    with pytest.raises(OcrError) as e:
        list(render_pages(b"%PDF-1.4\nbroken", filename="x.pdf", dpi=72,
                          raster_workers=2))
    assert e.value.status == 400
    # The worker is back and usable for the next document.
    assert len(_read_all(_pdf(3), raster_workers=2)) == 3


def test_an_abandoned_document_returns_its_worker():
    """A client that disconnects mid-document would otherwise leak a worker out
    of the pool, one request at a time, until the pool was empty."""
    pool = rz.get_pool(2)
    data = _pdf(30)
    for _ in range(6):
        g = render_pages(data, filename="x.pdf", dpi=72, raster_workers=2)
        next(g)
        g.close()
    assert pool._made <= 2
    # Still serving.
    assert len(_read_all(_pdf(2), raster_workers=2)) == 2


# ------------------------------------------------------------------ sizing #
def test_auto_sizing_is_bounded():
    n = rz.auto_workers()
    assert 1 <= n <= rz.MAX_AUTO_WORKERS


def test_resizing_replaces_the_pool():
    a = rz.get_pool(2)
    b = rz.get_pool(3)
    assert a is not b
    assert b.size == 3


def test_asking_for_the_same_size_reuses_the_pool():
    assert rz.get_pool(2) is rz.get_pool(2)


def test_zero_means_auto():
    assert rz.get_pool(0).size == rz.auto_workers()


# -------------------------------------------------------- the worker itself #
def test_the_worker_does_not_import_the_ocr_package():
    """`import codai.ocr` costs 377 MB and 1.5 s via litellm. Paying that per
    worker, in a change made to bound memory, would be self-defeating -- which is
    why the worker is executed as a script and not imported."""
    src = (ROOT / "codai/ocr/workers/raster_worker.py").read_text()
    # Check real import STATEMENTS, not prose: the module docstring says the word
    # "import codai" while explaining why it must not.
    import ast
    tree = ast.parse(src)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    assert "codai" not in names, f"worker imports the package: {sorted(names)}"
    assert names <= {"json", "sys", "pypdfium2", "PIL"}, sorted(names)


def test_the_worker_is_run_with_isolated_mode():
    """-I keeps the worker off the invoking directory's sys.path, so a stray
    module next to a document cannot be imported into it."""
    src = (ROOT / "codai/ocr/rasterize.py").read_text()
    assert '"-I"' in src


def test_a_pool_that_cannot_start_a_worker_does_not_wait_for_one(monkeypatch):
    """The first version of acquire() waited 300 s for a worker that could never
    arrive, stalling every document for five minutes when the pool was broken.
    An optimisation that cannot start must cost nothing, not everything."""
    import time
    monkeypatch.setattr(rz, "_WORKER", "/nonexistent/raster_worker.py")
    rz.reset_pool()
    pool = rz.get_pool(2)
    t0 = time.time()
    assert pool.acquire() is None
    assert pool.acquire() is None          # and again, without re-probing
    assert time.time() - t0 < 2.0, "acquire() waited for an impossible worker"


def test_a_broken_pool_is_remembered_rather_than_retried_per_document(monkeypatch):
    tries = []

    def _count():
        tries.append(1)
        raise OSError("no")
    monkeypatch.setattr(rz._Worker, "__init__", lambda self: _count())
    rz.reset_pool()
    pool = rz.get_pool(2)
    for _ in range(5):
        assert pool.acquire() is None
    assert len(tries) == 1, f"retried worker creation {len(tries)} times"


def test_a_busy_pool_falls_back_instead_of_queueing_forever(monkeypatch):
    """Waiting longer than WAIT_SECONDS just moves the queue somewhere less
    visible; rasterising in-process makes progress."""
    rz.reset_pool()
    # Built directly: get_pool(1) is "in-process", by design, so it returns None.
    pool = rz.RasterPool(1)
    monkeypatch.setattr(type(pool), "WAIT_SECONDS", 0.2)
    held = pool.acquire()
    assert held is not None
    try:
        import time
        t0 = time.time()
        assert pool.acquire() is None
        assert time.time() - t0 < 3.0
    finally:
        pool.release(held)


# ------------------------------------------------- the costs that bit first #
def test_pricing_a_document_does_not_load_every_page():
    """`pdf[i].get_size()` LOADS the page object to read a number the page tree
    already has: 593 ms for a 47-page document against 1.9 ms, and it was paid
    under the lock so the whole install queued behind it."""
    src = (ROOT / "codai/ocr/base.py").read_text()
    est = src.split("def page_estimate")[1].split("def render_pages")[0]
    assert "sizes = [pdf.get_page_size(i) for i in range(n)]" in est, (
        "page_estimate stopped reading sizes from the page tree by index")
    # The page-loading call is allowed ONLY as the compatibility fallback, after
    # an except: an older pypdfium2 has no get_page_size.
    primary, _, fallback = est.partition("except Exception:")
    assert "pdf[i].get_size()" not in primary.split("#")[0]
    assert "get_size()" in fallback


def test_the_worker_open_does_not_size_every_page():
    """The parent prices documents itself. Doing it again per chunk in the worker
    cost more than the render it was meant to inform."""
    src = (ROOT / "codai/ocr/workers/raster_worker.py").read_text()
    opened = src.split('if cmd == "open"')[1].split('if cmd == "render"')[0]
    assert "get_size()" not in opened
    assert '"est"' not in opened


def test_a_worker_is_not_held_while_the_caller_reads_a_window():
    """Holding one across OCR would cap the install at `raster_workers`
    concurrent documents -- worse than the lock this replaces. Asserted by
    watching the pool's free count between pulls."""
    data = _pdf(12)
    rz.reset_pool()
    pool = rz.get_pool(2)
    g = render_pages(data, filename="x.pdf", dpi=72, raster_workers=2, chunk=4)
    next(g)                                  # first chunk rendered and handed over
    # Between chunks every worker must be back in the pool.
    assert pool._free.qsize() == pool._made, (
        f"{pool._made - pool._free.qsize()} worker(s) held while the caller reads")
    list(g)
