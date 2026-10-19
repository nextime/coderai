"""A page being read must not hold a DEFAULT-executor thread.

The default pool is capped at 256 and shared with model lookups and the admin
pages. On a rented A40 the engine had 512 lanes and vLLM sat at 9% KV cache,
so 256 page waits -- not the GPU -- would have been the pod's ceiling, and past
it the model lookup every page's chat request needs would queue behind them.
"""
import asyncio
import inspect

from codai.ocr import manager as om


def test_the_pool_reads_pages_on_the_pod_pool():
    src = inspect.getsource(om._Pool.recognize)
    assert "run_on_pod_pool(eng.recognize_image" in src
    assert "asyncio.to_thread(eng.recognize_image" not in src


def test_a_read_runs_on_the_pod_pool_and_not_the_default(monkeypatch):
    seen = {}

    async def fake_pod_pool(func, *a, **k):
        seen["func"] = func
        return func(*a, **k)

    monkeypatch.setattr(om, "run_on_pod_pool", fake_pod_pool)

    class _Eng:
        def recognize_image(self, image):
            return "page"

    pool = om._Pool.__new__(om._Pool)

    async def _built():
        return None

    async def go():
        pool._q = asyncio.Queue()
        await pool._q.put(_Eng())
        pool._ensure_built = _built
        pool._enter_page = lambda: None
        pool._leave_page = lambda: None
        return await pool.recognize("img")

    assert asyncio.run(go()) == "page"
    assert seen["func"].__name__ == "recognize_image"
