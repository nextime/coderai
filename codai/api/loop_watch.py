"""How long the event loop went without running — measured from inside it.

A coroutine that asks to sleep 0.5 s and wakes at 0.9 s has been told something
exact: the loop was busy elsewhere for 0.4 s and could not service ANY request
in that window. That overshoot is the whole measurement, it costs one timer per
interval, and it needs no cooperation from the code that blocks.

Why it exists. A synchronous call on the loop -- a lock, a socket read, a
`requests.get`, a pool's `acquire()` -- stops the entire process, not one
request. From outside, that is indistinguishable from an overloaded upstream: a
5,000-document run rented 127 pods and completed nothing for ten minutes while
the real fault was a 1200 s `acquire()` called from a coroutine, and the first
hours of diagnosis went to the pods, which were idle and healthy. Nothing in the
process said "the loop was blocked for four minutes", though it had been.

What it is NOT. Not a profiler and not a watchdog that intervenes: it cannot say
WHICH call blocked, only that something did and for how long. That is enough to
turn "the server is slow" into "the server was not running", which is a
different investigation. Pair it with a stack dump if the culprit needs naming.

`/healthz` latency measures the same thing from outside and is the right probe
for a load test; this is the record kept when nobody is probing.
"""

import asyncio
import os
import time

#: Report a stall at or above this many seconds. Below ~100 ms an ordinary
#: handler doing ordinary work is indistinguishable from a blocking call, and a
#: log line per GC pause teaches an operator to ignore the message.
STALL_SECONDS = float(os.environ.get("CODERAI_LOOP_STALL_SECONDS", "0.5") or 0.5)

#: How often to take the measurement. The sampler can only see a stall that
#: overlaps one of its sleeps, so this bounds the shortest stall it can catch,
#: not the longest -- a 4-minute block is caught by any interval at all.
INTERVAL_SECONDS = float(os.environ.get("CODERAI_LOOP_WATCH_INTERVAL", "0.5") or 0.5)

#: Rolling record, read by /metrics and by the admin pages.
_state = {"worst": 0.0, "worst_at": 0.0, "stalls": 0, "blocked_seconds": 0.0,
          "last": 0.0, "since": 0.0}


def stats() -> dict:
    """A copy of the record. Safe to call from anywhere."""
    return dict(_state)


def reset() -> None:
    _state.update({"worst": 0.0, "worst_at": 0.0, "stalls": 0,
                   "blocked_seconds": 0.0, "last": 0.0, "since": time.time()})


async def _watch(interval: float, threshold: float) -> None:
    _state["since"] = time.time()
    while True:
        before = time.monotonic()
        await asyncio.sleep(interval)
        # Everything beyond the interval is time the loop owed us and spent
        # elsewhere. A timer may fire a few ms late on a busy but healthy loop,
        # which is why the threshold is not zero.
        lag = (time.monotonic() - before) - interval
        if lag < 0:
            lag = 0.0
        _state["last"] = lag
        if lag > _state["worst"]:
            _state["worst"] = lag
            _state["worst_at"] = time.time()
        if lag >= threshold:
            _state["stalls"] += 1
            _state["blocked_seconds"] += lag
            # One line per stall, with the number in it. An operator reading
            # "blocked for 247.1s" has the answer; "the server is slow" does not
            # even say which half of the system to look at.
            print(f"[loop] the event loop was blocked for {lag:.1f}s — no "
                  f"request of any kind was served in that window "
                  f"(worst so far {_state['worst']:.1f}s, {_state['stalls']} "
                  f"stall(s) over {threshold:.1f}s)", flush=True)


def install(threshold: float = None, interval: float = None):
    """Start the sampler on the running loop. Returns the task, or None.

    Never raises: an install that cannot measure must not be able to stop a
    process from serving.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None
    thr = STALL_SECONDS if threshold is None else float(threshold)
    itv = INTERVAL_SECONDS if interval is None else float(interval)
    if itv <= 0 or thr <= 0:
        return None
    try:
        task = loop.create_task(_watch(itv, thr))
    except Exception:
        return None
    print(f"[loop] watching for event-loop stalls over {thr:.1f}s "
          f"(sampled every {itv:.1f}s)", flush=True)
    return task
