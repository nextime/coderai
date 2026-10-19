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

What it is NOT. Not a profiler and not a watchdog that intervenes. The sampler
alone cannot say WHICH call blocked, only that something did and for how long.
So a second, plain THREAD watches the sampler's heartbeat, and when the loop has
been silent for DUMP_AFTER_SECONDS it prints the loop thread's stack while the
stall is still happening. The thread is outside the loop, so it runs exactly
when the loop cannot. Why: rented OCR pods froze for 108-784 s under load,
reported only as "blocked for N s", with no shell on the pod to attach py-spy.
A stack whose top frame is the selector means the loop was NOT running Python;
it was starved of the GIL by other threads, which is a different fix from a
blocking call.

`/healthz` latency measures the same thing from outside and is the right probe
for a load test; this is the record kept when nobody is probing.
"""

import asyncio
import os
import sys
import threading
import time
import traceback

#: Report a stall at or above this many seconds. Below ~100 ms an ordinary
#: handler doing ordinary work is indistinguishable from a blocking call, and a
#: log line per GC pause teaches an operator to ignore the message.
STALL_SECONDS = float(os.environ.get("CODERAI_LOOP_STALL_SECONDS", "0.5") or 0.5)

#: How often to take the measurement. The sampler can only see a stall that
#: overlaps one of its sleeps, so this bounds the shortest stall it can catch,
#: not the longest -- a 4-minute block is caught by any interval at all.
INTERVAL_SECONDS = float(os.environ.get("CODERAI_LOOP_WATCH_INTERVAL", "0.5") or 0.5)

#: Dump the loop thread's stack once the loop has been silent this long. 0 = off.
DUMP_AFTER_SECONDS = float(os.environ.get("CODERAI_LOOP_DUMP_AFTER", "3") or 0)

#: A long stall is dumped again every this many seconds while it lasts (the
#: frame that started it may not be the one that keeps it going), and no more
#: than DUMP_MAX_PER_HOUR dumps are printed in any hour.
DUMP_REPEAT_SECONDS = 30.0
DUMP_MAX_PER_HOUR = 20

#: Rolling record, read by /metrics and by the admin pages.
_state = {"worst": 0.0, "worst_at": 0.0, "stalls": 0, "blocked_seconds": 0.0,
          "last": 0.0, "since": 0.0, "beat": 0.0, "dumps": 0}


def _format_stack(thread_id: int, limit: int = 30) -> str:
    frame = sys._current_frames().get(thread_id)
    if frame is None:
        return "    (the loop thread has no Python frame)\n"
    return "".join(traceback.format_stack(frame, limit=limit))


def _dumper(loop, loop_thread_id: int, after: float) -> None:
    """Runs in its own thread: names the call the loop is stuck in.

    Exits when its loop closes: a finished loop has stopped beating, and that is
    not a stall."""
    dumped_at = 0.0          # monotonic time of the last dump in this stall
    stall_beat = None        # heartbeat value the current stall started from
    hour_start, in_hour = time.monotonic(), 0
    while True:
        time.sleep(min(1.0, after / 2))
        if loop.is_closed():
            return
        now = time.monotonic()
        beat = _state["beat"]
        if not beat or now - beat < after:
            stall_beat = None
            continue
        if now - hour_start > 3600:
            hour_start, in_hour = now, 0
        if stall_beat == beat and now - dumped_at < DUMP_REPEAT_SECONDS:
            continue
        if in_hour >= DUMP_MAX_PER_HOUR:
            continue
        stall_beat, dumped_at = beat, now
        in_hour += 1
        _state["dumps"] += 1
        try:
            others = [t.name for t in threading.enumerate()
                      if t.ident != loop_thread_id]
            print(f"[loop] STALL IN PROGRESS: the event loop has not run for "
                  f"{now - beat:.1f}s. Its thread is here (most recent call "
                  f"last); {len(others)} other thread(s) alive:\n"
                  f"{_format_stack(loop_thread_id)}[loop] end of stack",
                  flush=True)
        except Exception:
            pass


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
        _state["beat"] = before
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
    if DUMP_AFTER_SECONDS > 0:
        try:
            threading.Thread(target=_dumper,
                             args=(loop, threading.get_ident(),
                                   DUMP_AFTER_SECONDS),
                             daemon=True, name="loop-stall-dumper").start()
        except Exception:
            pass
    print(f"[loop] watching for event-loop stalls over {thr:.1f}s "
          f"(sampled every {itv:.1f}s; stack dump after "
          f"{DUMP_AFTER_SECONDS:.0f}s)", flush=True)
    return task
