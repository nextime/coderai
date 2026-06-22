# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Front-managed request queue (admission control) for text generation.

Architecture: the engine handles only generation; the front owns the queue. The
front sizes a per-model concurrency gate to the model's ``max_instances`` and
never dispatches more than that many concurrent generations to the engine — so
the engine's own queue effectively never fills, while ordering, queue depth,
queue position and the "queued" Tasks entries are all owned and reported here.

Each model key gets ``capacity`` slots. ``acquire()`` grants a slot immediately
when one is free, otherwise enqueues an asyncio waiter (FIFO) and blocks until a
slot frees. Beyond ``max_waiting`` queued requests it raises :class:`QueueFull`
(the caller returns HTTP 503). A client that disconnects while queued cancels its
``acquire()`` await, which drops it from the queue with no slot leak.
"""
import asyncio


class QueueFull(Exception):
    """Raised by FrontQueue.acquire when the per-model wait queue is at capacity."""

    def __init__(self, depth: int):
        super().__init__(f"queue full ({depth} waiting)")
        self.depth = depth


class _Waiter:
    __slots__ = ("rid", "model", "engine", "enqueued_at", "event")

    def __init__(self, rid, model, engine, enqueued_at):
        self.rid = rid
        self.model = model
        self.engine = engine
        self.enqueued_at = enqueued_at
        self.event = asyncio.Event()


class FrontQueue:
    """Per-model concurrency gate with a bounded FIFO wait queue."""

    def __init__(self):
        self._lock = asyncio.Lock()
        self._active: dict = {}     # model_key -> slots currently in use
        self._waiting: dict = {}    # model_key -> list[_Waiter] (FIFO)

    async def acquire(self, key: str, capacity: int, max_waiting: int,
                      rid: str = "", model: str = "", engine: str = "") -> None:
        """Acquire a generation slot for ``key``.

        Returns once a slot is held (immediately, or after waiting). Raises
        :class:`QueueFull` if ``max_waiting`` requests are already queued. The
        caller MUST pair every successful acquire with exactly one
        :meth:`release` for the same ``key``.
        """
        key = key or ""
        capacity = max(1, int(capacity or 1))
        async with self._lock:
            active = self._active.get(key, 0)
            if active < capacity:
                self._active[key] = active + 1
                return
            wq = self._waiting.setdefault(key, [])
            if len(wq) >= max(0, int(max_waiting)):
                raise QueueFull(len(wq))
            import time as _t
            waiter = _Waiter(rid=rid, model=model, engine=engine, enqueued_at=_t.time())
            wq.append(waiter)

        # Wait outside the lock. On grant, the releasing task has handed us its
        # slot (the active count was kept constant), so we must NOT re-increment.
        try:
            await waiter.event.wait()
        except BaseException:
            # Cancelled (e.g. client disconnected) or errored while queued.
            async with self._lock:
                wq = self._waiting.get(key) or []
                if waiter in wq:
                    wq.remove(waiter)            # still queued — never held a slot
                    if not wq:
                        self._waiting.pop(key, None)
                else:
                    # Already granted a slot in the race with cancellation — give
                    # it back so the next waiter (or the count) is correct.
                    await self._release_locked(key)
            raise

    async def release(self, key: str) -> None:
        key = key or ""
        async with self._lock:
            await self._release_locked(key)

    async def _release_locked(self, key: str) -> None:
        wq = self._waiting.get(key) or []
        if wq:
            # Hand the freed slot directly to the next waiter: active count stays
            # constant (one finishes, one starts).
            waiter = wq.pop(0)
            if not wq:
                self._waiting.pop(key, None)
            waiter.event.set()
        else:
            n = self._active.get(key, 0) - 1
            if n > 0:
                self._active[key] = n
            else:
                self._active.pop(key, None)

    def snapshot(self) -> list:
        """Best-effort list of currently-queued (not yet running) requests, for the
        Tasks page. Each entry: {rid, model, engine, position, enqueued_at}."""
        out = []
        for key, wq in list(self._waiting.items()):
            for i, w in enumerate(list(wq)):
                out.append({"rid": w.rid, "model": w.model, "engine": w.engine,
                            "position": i + 1, "enqueued_at": w.enqueued_at})
        return out
