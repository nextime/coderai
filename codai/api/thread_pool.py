# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Thread pools sized for THIS server's work, not for Python's default guess.

``asyncio.to_thread`` hands work to the event loop's default executor, and a
loop that was never told otherwise builds one with ``min(32, cpu_count + 4)``
threads — 20 on a 16-core host. That number is for incidental background work:
a file read, a hash, something measured in milliseconds.

coderai puts two very different kinds of call on it:

* **Short and bounded** — a stat, a config read, a GPU probe.
* **Long and blocking** — ``pool.acquire()`` waiting for a RunPod pod to boot
  (MINUTES), and a per-page OCR forward to a pod (30-300 seconds each).

With ``ocr.max_concurrency`` at 64, the second kind alone asks for more threads
than the default pool has. Every one of them sits in a blocking HTTP call, the
rest queue, and then an unrelated request that merely wants a thread — the
admin page asking what the fleet is doing — waits behind a page of OCR.

Observed on a live orchestrator, mid-run, 64 pages in flight:

    /healthz            200 in 0.002s     (no thread needed)
    /v1/models          200 in 0.006s     (no thread needed)
    /metrics            200 in 0.18s      (no thread needed)
    /v1/runpod/status   no answer in 30s  (needs a thread)

The admin page's fetch timed out, nginx answered with its HTML error page, and
the page showed `Unexpected token '<'`. Nothing was broken; the server was
simply unable to get a thread to answer on.

Two pools, because the two kinds of work must not compete:

* the DEFAULT executor, sized for the install rather than the CPU count, for
  everything ``asyncio.to_thread`` is used for today;
* a separate POD executor for the calls that block for minutes, so that however
  many pages are in flight, a control-plane request still finds a free thread.

A thread waiting on a socket costs a stack and nothing else. The asymmetry is
the point: being too small costs availability, being too large costs memory we
have.
"""

import asyncio
import contextvars
import functools
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

#: Floor for the general pool. Above Python's default, because the default is
#: the thing that was too small, and below it nothing is gained by sizing.
_MIN_DEFAULT = 32
#: Floor for the pod pool. Every in-flight page may hold one for minutes.
_MIN_POD = 64
#: Ceilings. A blocked thread is cheap but not free, and a runaway config
#: should not be able to ask for ten thousand stacks.
_MAX_DEFAULT = 256
_MAX_POD = 512

_lock = threading.Lock()
_pod_pool: Optional[ThreadPoolExecutor] = None
_installed = False


def _configured_concurrency() -> int:
    """The widest fan-out this install is configured for.

    Read from the config rather than from the CPU count: these threads are
    waiting on other machines, so the number that matters is how many remote
    calls may be outstanding, not how many cores can run Python.
    """
    want = 0
    try:
        from codai.config import config_manager
        cfg = config_manager.config if config_manager else None
        if cfg is not None:
            want = max(want, int(getattr(cfg.ocr, "max_concurrency", 0) or 0))
    except Exception:
        pass
    return want


def pool_sizes() -> tuple:
    """(default_workers, pod_workers) for this install."""
    want = _configured_concurrency()
    general = max(_MIN_DEFAULT, min(_MAX_DEFAULT, want))
    # Room for every concurrent page to be waiting on a pod at once, plus the
    # pool's own boots and probes.
    pod = max(_MIN_POD, min(_MAX_POD, want * 2))
    return general, pod


def install_default_executor(loop=None) -> Optional[ThreadPoolExecutor]:
    """Give the running loop an executor sized for this server.

    Idempotent and safe to call from either entry point — the front and the
    engine are separate processes and each needs its own.
    """
    global _installed
    with _lock:
        if _installed:
            return None
        try:
            loop = loop or asyncio.get_running_loop()
        except RuntimeError:
            return None
        general, _pod = pool_sizes()
        ex = ThreadPoolExecutor(max_workers=general,
                                thread_name_prefix="coderai")
        loop.set_default_executor(ex)
        _installed = True
        print(f"[threads] default pool {general} workers, pod pool {_pod} "
              f"(python's default here would be "
              f"{min(32, (os.cpu_count() or 1) + 4)})", flush=True)
        return ex


def pod_executor() -> ThreadPoolExecutor:
    """The pool for calls that block on a POD rather than on this machine.

    Separate from the default one on purpose: a page read that takes five
    minutes must not be able to make the server unable to answer a question
    about itself.
    """
    global _pod_pool
    with _lock:
        if _pod_pool is None:
            _pod_pool = ThreadPoolExecutor(max_workers=pool_sizes()[1],
                                           thread_name_prefix="coderai-pod")
        return _pod_pool


async def run_on_pod_pool(func, /, *args, **kwargs):
    """``asyncio.to_thread``, but on the pod pool.

    Copies the context the same way ``to_thread`` does, so anything reading a
    ContextVar — the OCR per-page options, the token trace — behaves
    identically whichever pool it lands on. ``run_in_executor`` does NOT do
    this by itself, and a page losing its budget because it was moved to
    another pool would be a nasty way to find that out.
    """
    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()
    call = functools.partial(ctx.run, func, *args, **kwargs)
    return await loop.run_in_executor(pod_executor(), call)
