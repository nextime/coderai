# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
# GPLv3 - see the project LICENSE.

"""What a pod was doing, readable over HTTP and after it is dead.

A rented pod is the one machine in the system whose logs cannot be read. RunPod
publishes no pod-log API — the REST spec lists 23 routes and none of them serve
logs, which is why every attempt comes back HTTP 400 — so the console is the only
outside view, and it is a human surface. ``boot.sh`` writes its phases to a file
and then ``exec``s uvicorn, whose output goes to the container's stdout and
nowhere else. ``/boot`` therefore covers the way up and nothing after it.

That gap costs real time. Three pods were investigated tonight and two were
already gone by the time there was a question to ask: a pod reclaimed mid-request
took its evidence with it, and "the serving hangs" stayed a hypothesis for an
hour because the only machine that knew was unreachable and then absent.

So the pod keeps a bounded ring of its own output and serves it at ``/logs``
with a cursor, and the renting install drains that into a file it keeps AFTER the
pod dies (see ``RunpodPodPool._drain_pod_log``). A ring rather than a file
because a pod's disk is rented too, and bounded because this must never be a
reason a pod falls over.
"""

import collections
import io
import sys
import threading

#: Lines kept. At ~120 bytes a line this is a few megabytes, which is more than
#: any boot or inference failure needs and small enough to be free.
MAX_LINES = 20000

_lock = threading.Lock()
_lines: "collections.deque" = collections.deque(maxlen=MAX_LINES)
_seq = 0                      # monotonic cursor; survives eviction from the ring
_installed = False


def record(line: str) -> None:
    """Add one line to the ring. Never raises — it is called from a write path."""
    global _seq
    try:
        text = line.rstrip("\n")
        if not text:
            return
        with _lock:
            _seq += 1
            _lines.append((_seq, text))
    except Exception:
        pass


class _Tee(io.TextIOBase):
    """Writes through to the real stream and keeps a copy in the ring.

    Deliberately a thin wrapper: the process's stdout is how everything else in
    coderai reports itself, and a logging aid that can break printing would be
    worse than no logging aid at all. Every failure path falls back to the real
    stream.
    """

    def __init__(self, real):
        self._real = real
        self._buf = ""

    # -- the parts anything printing actually uses ------------------------- #
    def write(self, s):
        try:
            self._real.write(s)
        except Exception:
            pass
        try:
            self._buf += s
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                record(line)
            if len(self._buf) > 8192:          # a writer that never newlines
                record(self._buf)
                self._buf = ""
        except Exception:
            self._buf = ""
        return len(s)

    def flush(self):
        try:
            self._real.flush()
        except Exception:
            pass

    def isatty(self):
        try:
            return self._real.isatty()
        except Exception:
            return False

    def fileno(self):
        return self._real.fileno()

    @property
    def encoding(self):
        return getattr(self._real, "encoding", "utf-8")

    def writable(self):
        return True


def install() -> bool:
    """Start copying this process's stdout and stderr into the ring.

    Idempotent, and a no-op if the streams have no ``write`` (a frozen or
    redirected environment). Returns whether it is now active.
    """
    global _installed
    with _lock:
        if _installed:
            return True
    try:
        sys.stdout = _Tee(sys.stdout)
        sys.stderr = _Tee(sys.stderr)
    except Exception:
        return False
    with _lock:
        _installed = True
    return True


def tail(since: int = 0, limit: int = 2000) -> dict:
    """Lines after ``since``, with the cursor to ask from next time.

    ``dropped`` says how many lines were evicted from the ring before the caller
    got to them — silence and "nothing happened" must not look the same.
    """
    with _lock:
        rows = [(n, t) for (n, t) in _lines if n > int(since or 0)]
        oldest = _lines[0][0] if _lines else 0
        cursor = _seq
    dropped = max(0, (oldest - 1) - int(since or 0)) if since and oldest else 0
    if limit and len(rows) > limit:
        rows = rows[-int(limit):]
    return {"cursor": cursor, "dropped": dropped,
            "lines": [t for (_n, t) in rows]}
