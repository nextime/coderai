"""An OCR run must not make the server unable to answer questions about itself.

`asyncio.to_thread` uses the loop's default executor, and a loop that was never
told otherwise builds one with min(32, cpu_count + 4) — 20 threads on a 16-core
host. That is a number for incidental background work. coderai puts calls that
block for MINUTES on it: `pool.acquire()` waiting out a pod boot, and a per-page
OCR forward.

Measured live 2026-10-16 with 64 pages in flight ("engine 'surya': 8 instance(s)
ready, 8 lane(s) each — 64 pages at once"):

    /healthz            200 in 0.002s     no thread needed
    /v1/models          200 in 0.006s     no thread needed
    /metrics            200 in 0.18s      no thread needed
    /v1/runpod/status   no answer in 30s  needs a thread

The admin page's fetch timed out, nginx returned its HTML error page, and the
page showed `Unexpected token '<', "<!doctype "... is not valid JSON`. Nothing
was broken — the server could not get a thread to answer on.
"""

import asyncio
import contextvars
import types

import pytest

from codai.api import thread_pool


def _cfg(monkeypatch, max_concurrency):
    mgr = types.SimpleNamespace(config=types.SimpleNamespace(
        ocr=types.SimpleNamespace(max_concurrency=max_concurrency)))
    import codai.config
    monkeypatch.setattr(codai.config, "config_manager", mgr, raising=False)


# ------------------------------------------------------------------ sizing #
def test_the_pools_are_bigger_than_pythons_default(monkeypatch):
    """Python's default is the thing that was too small."""
    import os
    _cfg(monkeypatch, 64)
    general, pod = thread_pool.pool_sizes()
    assert general > min(32, (os.cpu_count() or 1) + 4)
    assert pod > general


def test_the_pod_pool_holds_every_page_at_once(monkeypatch):
    """Each in-flight page may sit on a pod call for minutes; if they cannot
    all have a thread, the ones that cannot are just waiting for a thread
    instead of for a pod, which is the bug."""
    _cfg(monkeypatch, 128)
    general, pod = thread_pool.pool_sizes()
    assert pod >= 128
    assert general >= 128


def test_a_small_install_still_gets_a_floor(monkeypatch):
    _cfg(monkeypatch, 2)
    general, pod = thread_pool.pool_sizes()
    assert general == thread_pool._MIN_DEFAULT
    assert pod == thread_pool._MIN_POD


def test_an_absurd_config_cannot_ask_for_ten_thousand_stacks(monkeypatch):
    _cfg(monkeypatch, 100000)
    general, pod = thread_pool.pool_sizes()
    assert general == thread_pool._MAX_DEFAULT
    assert pod == thread_pool._MAX_POD


def test_no_config_is_not_an_error(monkeypatch):
    import codai.config
    monkeypatch.setattr(codai.config, "config_manager", None, raising=False)
    general, pod = thread_pool.pool_sizes()
    assert general == thread_pool._MIN_DEFAULT
    assert pod == thread_pool._MIN_POD


# ------------------------------------------------------------- the install #
def test_installing_replaces_the_loops_executor(monkeypatch):
    monkeypatch.setattr(thread_pool, "_installed", False, raising=False)
    _cfg(monkeypatch, 64)
    seen = {}

    async def _main():
        loop = asyncio.get_running_loop()
        loop.set_default_executor = lambda ex: seen.setdefault("ex", ex)
        thread_pool.install_default_executor()

    asyncio.run(_main())
    assert seen.get("ex") is not None
    assert seen["ex"]._max_workers >= 64


def test_installing_twice_is_a_no_op(monkeypatch):
    """The front and the engine are separate processes, but a process that
    calls it twice must not swap the pool out from under running work."""
    monkeypatch.setattr(thread_pool, "_installed", False, raising=False)
    _cfg(monkeypatch, 64)
    calls = []

    async def _main():
        loop = asyncio.get_running_loop()
        loop.set_default_executor = lambda ex: calls.append(ex)
        thread_pool.install_default_executor()
        thread_pool.install_default_executor()

    asyncio.run(_main())
    assert len(calls) == 1


def test_installing_outside_a_loop_does_not_raise(monkeypatch):
    monkeypatch.setattr(thread_pool, "_installed", False, raising=False)
    assert thread_pool.install_default_executor() is None


# --------------------------------------------------------- the separation #
def test_the_pod_pool_is_not_the_default_pool(monkeypatch):
    """The whole point: a page read cannot consume the threads the control
    plane needs."""
    monkeypatch.setattr(thread_pool, "_pod_pool", None, raising=False)
    monkeypatch.setattr(thread_pool, "_installed", False, raising=False)
    _cfg(monkeypatch, 64)
    seen = {}

    async def _main():
        loop = asyncio.get_running_loop()
        real = loop.set_default_executor
        loop.set_default_executor = lambda ex: (seen.setdefault("default", ex),
                                                real(ex))
        thread_pool.install_default_executor()
        return thread_pool.pod_executor()

    pod = asyncio.run(_main())
    assert pod is not seen["default"]


def test_a_control_plane_call_gets_a_thread_while_the_pod_pool_is_full(monkeypatch):
    """The live failure, reproduced: saturate the pod pool and check that work
    on the default pool still runs."""
    import threading
    monkeypatch.setattr(thread_pool, "_pod_pool", None, raising=False)
    _cfg(monkeypatch, 2)          # pod pool = the floor
    hold = threading.Event()
    lock = threading.Lock()
    started = set()

    def _block(i):
        with lock:
            started.add(i)
        hold.wait(10)
        return i

    async def _main():
        size = thread_pool.pool_sizes()[1]
        tasks = [asyncio.ensure_future(thread_pool.run_on_pod_pool(_block, i))
                 for i in range(size)]
        # Yield, rather than block: waiting for them on the loop thread would
        # stop the loop ever starting them. (It did, the first time.)
        for _ in range(1000):
            await asyncio.sleep(0.01)
            with lock:
                if len(started) >= size:
                    break
        with lock:
            assert len(started) >= size, "the pod pool never filled"
        # …and the control plane still answers, on the other pool.
        answered = await asyncio.wait_for(
            asyncio.to_thread(lambda: "fleet status"), timeout=5)
        hold.set()
        await asyncio.gather(*tasks)
        return answered

    assert asyncio.run(_main()) == "fleet status"


# -------------------------------------------------------------- the context #
def test_the_pod_pool_carries_contextvars_like_to_thread():
    """`run_in_executor` does not copy the context by itself. The OCR per-page
    options and the token trace are ContextVars, so a page losing its budget
    because it moved pools would be a nasty way to discover that."""
    var = contextvars.ContextVar("page_budget", default=0)

    async def _main():
        var.set(3072)
        return await thread_pool.run_on_pod_pool(var.get)

    assert asyncio.run(_main()) == 3072


def test_the_pod_pool_propagates_exceptions():
    async def _main():
        with pytest.raises(ValueError):
            await thread_pool.run_on_pod_pool(
                lambda: (_ for _ in ()).throw(ValueError("boom")))

    asyncio.run(_main())


def test_arguments_reach_the_callable():
    async def _main():
        return await thread_pool.run_on_pod_pool(lambda a, b=0: a + b, 2, b=3)

    assert asyncio.run(_main()) == 5


# ------------------------------------------------------- the call sites moved #
def test_the_blocking_pod_waits_use_the_pod_pool():
    """Both are minutes-long; on the shared pool they were the starvation."""
    import pathlib
    from codai import frontproxy
    from codai.api import remote_gateway

    front = pathlib.Path(frontproxy.app.__file__).read_text()
    assert "run_on_pod_pool(pool.acquire)" in front
    assert "_asyncio.to_thread(pool.acquire)" not in front

    gw = pathlib.Path(remote_gateway.__file__).read_text()
    assert "run_on_pod_pool(_call)" in gw


def test_the_fleet_snapshot_is_off_the_event_loop():
    """pods_status() takes the pool locks; on the loop a contended lock stalls
    every request rather than one."""
    import pathlib
    from codai.api import app as api_app
    src = pathlib.Path(api_app.__file__).read_text()
    blk = src.split("async def runpod_spend")[1].split("@app.get")[0]
    # And on the CONTROL pool, each source with its own deadline: one thread
    # from the general executor was still a thread the request path could take,
    # and one function meant the slowest source decided the whole answer.
    assert "run_on_control_pool(" in blk
    assert "asyncio.to_thread" not in blk
    assert "asyncio.wait_for(" in blk and "asyncio.gather(" in blk
