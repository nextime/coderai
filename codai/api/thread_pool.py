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
#: Raised from 512 once the sizing above started working: a run that wants
#: 100 pages/s at ~20 s per page needs ~2000 forwards outstanding, and every
#: one of them is a thread blocked on a socket. Beyond this an async client is
#: the right answer, not more stacks.
_MAX_POD = 2048

_lock = threading.Lock()
_pod_pool: Optional[ThreadPoolExecutor] = None
_installed = False


def _configured_concurrency() -> int:
    """The widest fan-out this install is configured for.

    Read from the config rather than from the CPU count: these threads are
    waiting on other machines, so the number that matters is how many remote
    calls may be outstanding, not how many cores can run Python.

    THE SHARED MANAGER LIVES IN ``codai.admin.routes``, not in ``codai.config``.
    This read used to be ``from codai.config import config_manager``, which has
    no such name -- so it raised ImportError on every call, the except swallowed
    it, and this returned 0 forever. Both pools then took their FLOORS (32 and
    64) on every install that has ever run, whatever ``ocr.max_concurrency``
    said. Measured on a live orchestrator: 32 page forwards in flight against a
    512-page semaphore and 31 idle pods, a ceiling of 1.65 pages/s
    (32 / 19.4 s per page) that moved for neither a 4x semaphore nor a 24x pod
    gate, because neither was ever the limit.

    Hence BOTH holders, and no bare except: a sizing function that silently
    answers "unconfigured" is indistinguishable from one that answers
    truthfully, and the difference was a tenfold throughput ceiling nobody
    could see. ``sized_from_config()`` reports which it was.
    """
    want = 0
    for mod, attr in (("codai.admin.routes", "config_manager"),
                      ("codai.config", "config_manager")):
        try:
            import importlib
            mgr = getattr(importlib.import_module(mod), attr, None)
            cfg = getattr(mgr, "config", None) if mgr is not None else None
            ocr = getattr(cfg, "ocr", None) if cfg is not None else None
            if ocr is not None:
                want = max(want, int(getattr(ocr, "max_concurrency", 0) or 0))
        except Exception:
            continue
    return want


def sized_from_config() -> bool:
    """Whether the pools could read the config at all.

    For the boot line: "32 workers" means something entirely different when it
    is the configured number than when it is the floor taken because nothing
    could be read.
    """
    return _configured_concurrency() > 0


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
        how = ("from ocr.max_concurrency" if sized_from_config()
               else "FLOORS — the config was not readable from here yet, so "
                    "ocr.max_concurrency did NOT size these")
        print(f"[threads] default pool {general} workers, pod pool {_pod} "
              f"({how}; python's default here would be "
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


def resize_pod_pool() -> int:
    """Grow the pod pool to the configured width. Returns its new size.

    Called when the config changes: the pool is built on first use, so an
    install that raises ``ocr.max_concurrency`` mid-run would otherwise keep
    whatever width it was born with until a restart. Growing only -- shrinking
    would have to decide what to do with the threads currently blocked on a
    page, and the answer that does not lose pages is "nothing".
    """
    global _pod_pool
    want = pool_sizes()[1]
    with _lock:
        if _pod_pool is None:
            return want                      # built at the right size on demand
        cur = getattr(_pod_pool, "_max_workers", 0)
        if want > cur:
            _pod_pool._max_workers = want
            print(f"[threads] pod pool {cur} -> {want} workers "
                  f"(ocr.max_concurrency changed)", flush=True)
        return max(cur, want)


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
