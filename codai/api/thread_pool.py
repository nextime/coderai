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
#: Raised from 512, then from 2048, as the sizing below proved able to defend
#: itself: a run that wants 100 pages/s at ~20 s per page needs ~2000 forwards
#: outstanding, and every one of them is a thread blocked on a socket.
#:
#: 2048 was a silent FLEET ceiling, which is a worse failure than a slow one. A
#: page on the pod path holds a pod-pool thread for its whole life, so this
#: constant -- not ocr.max_concurrency, not max_pods, not the cgroup -- decided
#: how many pods an orchestrator could feed: 2048 threads / 64 pages per pod =
#: 32 pods, whatever max_pods said. An operator raising max_pods to 128 got 96
#: pods' worth of nothing, and no line anywhere said why.
#:
#: Nothing here removes a real limit. `pool_sizes()` still clamps against
#: `task_budget()` -- the cgroup's pids.max minus what is already running minus
#: headroom -- and still gives the pod pool away first so the general pool can
#: keep answering /healthz. This only stops a CONSTANT from being the binding
#: limit on a box that has the tasks to spare: measured here, pids.max 16384
#: with 79 in use.
#:
#: Past this an async client really is the right answer rather than more
#: stacks; 8192 covers 128 pods at 64 pages each, which is a fleet large enough
#: that the next bottleneck is somewhere else entirely.
_MAX_POD = 8192

#: Width of the LEASE pool. Deliberately small and NOT derived from
#: `max_concurrency`: a lease is a bounded wait (`POD_LEASE_ATTEMPT_S`) that
#: either hands back a pod at once or comes back empty, so a few hundred
#: threads can serve thousands of pages by cycling. Sizing it per in-flight
#: page is what made the pod pool ask for 2048 threads and exhaust a 2048-task
#: cgroup; the page concurrency an install wants should not decide how many
#: thread stacks it needs.
_LEASE_WORKERS = 256

#: Width of the CONTROL pool: the threads that answer questions ABOUT the
#: server rather than doing its work.
#:
#: The general pool was sized for the install and then shared with the request
#: path, which makes "can an operator see what is happening" depend on how busy
#: the install is -- and the moment it is worth asking is the moment it is
#: busiest. Measured on a live orchestrator during a boot wave:
#:
#:     /healthz            200 in   2.9 ms   (no thread needed)
#:     /v1/runpod/status   no answer in 25 s (needs a thread)
#:     /v1/runpod/scale    no answer in 25 s (needs a thread)
#:
#: Both of the endpoints that stopped answering are the two that hand work to
#: a thread, and nothing was wrong with either of them. A client reasonably
#: read that as a dead server; it also meant a client lease could not be
#: RELEASED while the fleet was busy, so pods kept billing.
#:
#: Tiny and fixed, because this pool must never be the thing that is full:
#: nothing in the request path may use it, and a control-plane answer is a
#: snapshot measured in milliseconds once it has a thread to run on.
_CONTROL_WORKERS = 16

#: Width of the PDFIUM pool. pdfium permits exactly ONE caller at a time (see
#: `codai.ocr.base._PDFIUM_LOCK`), so threads beyond the second buy no
#: parallelism at all -- they only decide WHERE the queue forms. On the general
#: executor the queue formed in the pool every control-plane request also needs.
#:
#: Two rather than one so a split's non-locked work (allocating the output
#: buffer, handing the bytes back) can overlap the next split's locked work.
_PDFIUM_WORKERS = 2

_lock = threading.Lock()
_pod_pool: Optional[ThreadPoolExecutor] = None
_lease_pool: Optional[ThreadPoolExecutor] = None
_control_pool: Optional[ThreadPoolExecutor] = None
_pdfium_pool: Optional[ThreadPoolExecutor] = None
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
    if want <= 0:
        # A RENTED POD HAS NO CONFIG FILE. Its settings arrive as environment
        # variables, and the pools are built before the config manager has
        # read them -- so on every pod this returned 0 and the default pool
        # took its floor of 32. Every page on a pod runs in that pool, which
        # capped each pod at 32 pages at once while surya was ready for 512,
        # vLLM reported room for 72x, and GPU KV cache sat at 4-10%. Measured
        # 2026-10-19: 64 pages dispatched per pod, `Running: 32` on every card.
        try:
            want = int(os.environ.get("CODERAI_OCR_MAX_CONCURRENCY", "") or 0)
        except ValueError:
            want = 0
    return want


def sized_from_config() -> bool:
    """Whether the pools could read the config at all.

    For the boot line: "32 workers" means something entirely different when it
    is the configured number than when it is the floor taken because nothing
    could be read.
    """
    return _configured_concurrency() > 0


def task_budget() -> int:
    """How many threads this PROCESS is actually allowed, or 0 if unbounded.

    A pool sized from a config number is a pool sized from a wish. The ceiling
    that decides whether those threads can exist is the process's task budget,
    and in a container that is not the kernel's -- podman defaults
    `--pids-limit` to 2048, and systemd units carry `TasksMax`. Both appear as
    cgroup v2 `pids.max`, which is the number to read.

    Measured on a live orchestrator: `ocr.max_concurrency: 4096` asked the pod
    pool for its 2048 ceiling inside a cgroup whose TOTAL budget was 2048. The
    pool reached 1992 threads, the 2049th raised "can't start new thread", and
    the OCR manager read the pages on a GPU-less box instead -- the one outcome
    the whole pod pipeline exists to avoid. Nothing reported a limit, because
    nothing had looked for one.

    Returns the tightest of cgroup v2 pids.max and RLIMIT_NPROC, minus what is
    already running, or 0 when neither bounds us.
    """
    import os
    budget = 0

    def _tighten(n):
        nonlocal budget
        if n and n > 0:
            budget = n if budget <= 0 else min(budget, n)

    # cgroup v2: this process's own cgroup, then its parents -- a limit on any
    # ancestor binds us just as hard as one on our own leaf.
    try:
        with open("/proc/self/cgroup") as fh:
            rel = fh.read().strip().split(":")[-1].lstrip("/")
        node = os.path.join("/sys/fs/cgroup", rel)
        current = 0
        while True:
            try:
                with open(os.path.join(node, "pids.max")) as fh:
                    raw = fh.read().strip()
                if raw and raw != "max":
                    _tighten(int(raw))
            except Exception:
                pass
            try:
                with open(os.path.join(node, "pids.current")) as fh:
                    current = max(current, int(fh.read().strip()))
            except Exception:
                pass
            parent = os.path.dirname(node)
            if parent == node or len(parent) <= len("/sys/fs/cgroup"):
                break
            node = parent
        if budget > 0 and current > 0:
            # What is left, not what the limit says: the other processes in this
            # cgroup (the front, the system engine, every subprocess worker)
            # are already spending it.
            budget = max(0, budget - current)
    except Exception:
        pass

    try:
        import resource
        soft, _hard = resource.getrlimit(resource.RLIMIT_NPROC)
        if soft not in (resource.RLIM_INFINITY, -1):
            _tighten(int(soft))
    except Exception:
        pass
    return budget


#: Leave this many tasks unspent. Threads are not the only thing that needs to
#: fork: a rasteriser worker, an isolated-venv engine, a `podman exec` to look
#: at the box while it is busy. A pool that spends the last task takes the
#: operator's ability to diagnose it away with it.
_TASK_HEADROOM = 256


def pool_sizes() -> tuple:
    """(default_workers, pod_workers) for this install, within its budget."""
    want = _configured_concurrency()
    general = max(_MIN_DEFAULT, min(_MAX_DEFAULT, want))
    # Room for every concurrent page to be waiting on a pod at once, plus the
    # pool's own boots and probes.
    pod = max(_MIN_POD, min(_MAX_POD, want * 2))

    budget = task_budget()
    if budget > 0:
        # The fixed pools are spent before the sized ones. They are small and
        # they are not optional: an install that cannot answer a question about
        # itself cannot be fixed, and a split that cannot run produces no pages.
        spendable = budget - _TASK_HEADROOM - _CONTROL_WORKERS - _PDFIUM_WORKERS
        if spendable < _MIN_DEFAULT + _MIN_POD:
            # Too tight to honour both floors: keep them, and let the operator
            # see why from the boot line. Refusing to start would be worse --
            # an install that serves slowly beats one that does not serve.
            return _MIN_DEFAULT, _MIN_POD
        if general + pod > spendable:
            # The pod pool gives first: the general pool is what answers
            # /healthz and the admin pages, and an install that cannot be
            # looked at is an install that cannot be fixed.
            general = min(general, max(_MIN_DEFAULT, spendable // 4))
            pod = max(_MIN_POD, spendable - general)
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


def lease_executor() -> ThreadPoolExecutor:
    """The pool that pod LEASES wait in, created on first use.

    Separate from the pod pool on purpose. Both block, but on different things
    and for different lengths: a forward waits on a socket for as long as a
    page takes to read, while a lease waits for a pod to be free or to boot.
    Sharing one pool let a cold start fill every thread with leases waiting for
    pods, leaving none to carry the pages those pods were for -- and sizing
    that one pool per in-flight page is what exhausted the task budget.
    """
    global _lease_pool
    with _lock:
        if _lease_pool is None:
            budget = task_budget()
            n = _LEASE_WORKERS
            if budget > 0:
                n = max(16, min(n, max(16, (budget - _TASK_HEADROOM) // 8)))
            _lease_pool = ThreadPoolExecutor(
                max_workers=n, thread_name_prefix="coderai-lease")
            print(f"[threads] lease pool: {n} workers (a lease is a bounded "
                  f"wait, so this serves far more pages than it has threads)",
                  flush=True)
        return _lease_pool


async def run_on_lease_pool(func, /, *args, **kwargs):
    """``run_on_pod_pool``, on the lease pool."""
    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()
    call = functools.partial(ctx.run, func, *args, **kwargs)
    return await loop.run_in_executor(lease_executor(), call)


def control_executor() -> ThreadPoolExecutor:
    """The pool that answers questions ABOUT the server, created on first use.

    Reserved for the control plane -- the fleet snapshot, a scale request, a
    pod teardown -- and deliberately not used by anything in the request path.
    Its whole value is that it is empty when everything else is full.
    """
    global _control_pool
    with _lock:
        if _control_pool is None:
            _control_pool = ThreadPoolExecutor(
                max_workers=_CONTROL_WORKERS,
                thread_name_prefix="coderai-control")
            print(f"[threads] control pool: {_CONTROL_WORKERS} workers "
                  f"(reserved for status/scale/teardown, so a busy server can "
                  f"still say what it is busy with)", flush=True)
        return _control_pool


def pdfium_executor() -> ThreadPoolExecutor:
    """The pool for pdfium work, created on first use.

    pdfium is single-caller, so this is a queue with a tiny amount of overlap
    rather than a parallel pool. Splitting a document used to go to the default
    executor, where hundreds of concurrent documents each took a thread and
    then waited on the same global lock -- the serialisation was unchanged, but
    the threads they held were the ones an operator needed to look at the
    server. Queueing here costs nothing it was not already paying.
    """
    global _pdfium_pool
    with _lock:
        if _pdfium_pool is None:
            _pdfium_pool = ThreadPoolExecutor(
                max_workers=_PDFIUM_WORKERS,
                thread_name_prefix="coderai-pdfium")
        return _pdfium_pool


async def run_on_control_pool(func, /, *args, **kwargs):
    """``run_on_pod_pool``, on the control pool."""
    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()
    call = functools.partial(ctx.run, func, *args, **kwargs)
    return await loop.run_in_executor(control_executor(), call)


async def run_on_pdfium_pool(func, /, *args, **kwargs):
    """``run_on_pod_pool``, on the pdfium pool."""
    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()
    call = functools.partial(ctx.run, func, *args, **kwargs)
    return await loop.run_in_executor(pdfium_executor(), call)


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
