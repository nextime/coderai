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


class _SwapWaiter:
    __slots__ = ("key", "event", "granted", "enqueued_at")

    def __init__(self, key):
        import time as _t
        self.key = key
        self.event = asyncio.Event()
        self.granted = False
        self.enqueued_at = _t.time()


class GpuSwapGate:
    """Serialize model 'ownership' of one shared GPU across co-located engines.

    On the GGUF-isolation split a torch (image/video) engine and a gguf (text)
    engine share a single NVIDIA card and cannot hold both big models at once. This
    gate makes at most ONE model own the GPU at a time — so two model forwards never
    run concurrently and contend for VRAM (the OOM-then-disk-thrash failure) — while
    keeping it efficient:

      * Requests for the model that currently OWNS the GPU run immediately (a swap
        isn't needed), concurrency still capped downstream by the per-model queue.
      * A request for a DIFFERENT model queues. The owner keeps being served — up to
        `cap` requests while another model is waiting — then, once the owner is fully
        idle (its in-flight requests finished; we never swap mid-request), the GPU
        SWAPS to the waiting model (which evicts + loads), serves it, and later swaps
        BACK if the original model has requests queued. Round-robin with a per-turn
        batch cap: no thrash (a lone request doesn't force a swap) and no starvation
        (a busy model yields after `cap`).

    `acquire(key)`/`release(key)` bracket each GPU-inference request; `key` is the
    model identity that determines residency. Cancelling a pending acquire (client
    disconnect) drops the waiter with no slot leak."""

    def __init__(self, cap: int = 10):
        self._lock = asyncio.Lock()
        self.cap = max(1, int(cap))
        self._owner = None      # model key currently allowed to run GPU work
        self._running = 0       # in-flight granted requests (all for _owner)
        self._served = 0        # grants since _owner took the GPU (batch-cap counter)
        self._waiters = []      # FIFO list[_SwapWaiter]

    def _other_waiting(self) -> bool:
        return any(w.key != self._owner for w in self._waiters)

    async def acquire(self, key) -> None:
        async with self._lock:
            if self._owner is None:
                self._owner = key
                self._served = 0
            # Fast path: the resident model, unless its batch cap is spent AND a
            # different model is waiting (then it must yield — fall through to queue).
            if key == self._owner and (self._served < self.cap
                                       or not self._other_waiting()):
                self._running += 1
                self._served += 1
                return
            w = _SwapWaiter(key)
            self._waiters.append(w)
            # If the GPU is idle right now nothing will pump this waiter later, so
            # process it immediately (may swap the owner to `key`).
            if self._running == 0:
                self._pump()
        try:
            await w.event.wait()
        except BaseException:
            # Cancelled/errored while queued OR just after being granted.
            async with self._lock:
                if w in self._waiters:
                    self._waiters.remove(w)             # never held a slot
                elif w.granted:
                    self._running -= 1                  # granted as we cancelled
                    if self._running <= 0:
                        self._running = 0
                        self._pump()
            raise

    async def release(self, key) -> None:
        async with self._lock:
            self._running -= 1
            if self._running <= 0:
                self._running = 0
                self._pump()

    def _grant(self, w: "_SwapWaiter") -> None:
        self._waiters.remove(w)
        self._running += 1
        self._served += 1
        w.granted = True
        w.event.set()

    def _swap_to(self, key) -> None:
        self._owner = key
        self._served = 0
        for w in [x for x in self._waiters if x.key == key]:
            self._grant(w)

    def _pump(self) -> None:
        """Owner is idle (_running == 0): decide who runs next. Called under lock."""
        if not self._waiters:
            return  # keep _owner as the last-resident model so a repeat runs free
        owner_w = [w for w in self._waiters if w.key == self._owner]
        other_w = [w for w in self._waiters if w.key != self._owner]
        if owner_w and self._served < self.cap:
            # Keep serving the resident model. When another model is waiting, grant
            # only up to the remaining cap so it eventually yields; otherwise all.
            room = (self.cap - self._served) if other_w else len(owner_w)
            granted = 0
            for w in list(owner_w):
                if granted >= room:
                    break
                self._grant(w)
                granted += 1
            if granted == 0 and other_w:
                self._swap_to(other_w[0].key)   # cap spent → swap
            return
        if other_w:
            self._swap_to(other_w[0].key)       # cap spent or owner drained → swap
            return
        # Only the owner is waiting (cap spent, nobody else): reset the turn.
        self._served = 0
        for w in list(owner_w):
            self._grant(w)

    def snapshot(self) -> dict:
        return {"owner": self._owner, "running": self._running,
                "served": self._served, "cap": self.cap,
                "waiting": [{"key": w.key, "enqueued_at": w.enqueued_at}
                            for w in self._waiters]}
