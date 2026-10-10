"""Guards so a runaway test cannot take the host down with it.

On 2026-10-10 a full `pytest tests/` run grew to 46 GB RSS plus 20 GB of swap on a
54 GB machine: a broker test mocked the reconnect delay to zero and the loop it was
driving swallowed the cancellation meant to stop it, so it spun at full speed
allocating ~78 MB/s (AsyncMock records every call it receives). Swap filled, the OOM
killer took the coderai server — which was serving at the time — and the box needed a
hard reset. The kernel log for the final forty minutes was lost with it.

The leak itself is fixed (codai/broker/client.py) and that test now brakes itself. But
"a test can consume the machine" is a property of the test RUN, not of one test, and
the next one will be written by someone who has never heard of this incident. So:

  RSS ceiling   — a watchdog thread samples this process's own RSS and, above the
                  ceiling, dumps every thread's stack and exits 97. A test that wants
                  several GB is a bug; dying with a traceback that names it beats an
                  OOM kill that names whatever happened to be fattest.
  stall ceiling — a hung test is how a slow leak gets the time to become a big one,
                  and it blocks every test behind it. faulthandler records where it
                  is stuck and ends the run rather than waiting forever.

Both write their report to a FILE as well as to stderr, because the process exits
hard: pytest captures stderr per test and discards the buffer when the process does
not unwind. The path is printed on the way out and defaults to
``$TMPDIR/coderai-test-abort-<pid>.log``.

Both are on by default and tunable from the environment:

  CODERAI_TEST_RSS_LIMIT_MB    ceiling in MB (default 6144; 0 disables)
  CODERAI_TEST_STALL_SECONDS   per-test stall limit (default 180; 0 disables)
  CODERAI_TEST_ABORT_LOG       where to write the report

A test that legitimately needs more should raise its own ceiling explicitly rather
than lift the whole suite's.
"""

import faulthandler
import os
import sys
import tempfile
import threading

_PAGE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096

RSS_LIMIT_MB = int(os.environ.get("CODERAI_TEST_RSS_LIMIT_MB") or 6144)
STALL_SECONDS = int(os.environ.get("CODERAI_TEST_STALL_SECONDS") or 180)
ABORT_LOG = os.environ.get("CODERAI_TEST_ABORT_LOG") or os.path.join(
    tempfile.gettempdir(), f"coderai-test-abort-{os.getpid()}.log")

# Sampled often enough that the ceiling is overshot by little: the incident ran at
# 78 MB/s, and even a pathological allocator only gets a fraction of a second.
_SAMPLE_SECONDS = 0.25

_current_test = ["(collection)"]
_log_handle = None


def _rss_mb():
    """This process's resident size in MB, or None where /proc is unavailable."""
    try:
        with open("/proc/self/statm") as handle:
            return int(handle.read().split()[1]) * _PAGE / 1048576
    except Exception:
        return None


def _open_log():
    """The report file, kept open: a watchdog must not need to allocate much to report.

    Opened once up front for the same reason — by the time the ceiling is hit, opening
    a file may itself be what fails.
    """
    global _log_handle
    if _log_handle is None:
        try:
            _log_handle = open(ABORT_LOG, "w", buffering=1)
        except Exception:
            _log_handle = sys.__stderr__
    return _log_handle


def _shout(message):
    """Say it everywhere it might survive the process exiting hard."""
    for stream in (_open_log(), sys.__stderr__):
        try:
            stream.write(message)
            stream.flush()
        except Exception:
            pass
    try:
        os.write(2, message.encode())
    except Exception:
        pass


def _watch_rss(limit_mb):
    while True:
        rss = _rss_mb()
        if rss is None:
            return          # not Linux: nothing to watch with
        if rss > limit_mb:
            _shout(
                f"\n\n*** test run aborted: {rss:.0f} MB resident, over the "
                f"{limit_mb} MB ceiling, during {_current_test[0]}\n"
                f"*** a test allocating this much is a leak, not a workload — the "
                f"stacks in {ABORT_LOG} say where.\n"
                f"*** raise CODERAI_TEST_RSS_LIMIT_MB if this one genuinely needs "
                f"the memory. Exit code 97.\n\n")
            try:
                faulthandler.dump_traceback(file=_open_log(), all_threads=True)
                _open_log().flush()
            except Exception:
                pass
            os._exit(97)
        threading.Event().wait(_SAMPLE_SECONDS)


def pytest_configure(config):
    if RSS_LIMIT_MB > 0 and _rss_mb() is not None:
        threading.Thread(target=_watch_rss, args=(RSS_LIMIT_MB,),
                         name="rss-watchdog", daemon=True).start()


def pytest_runtest_logstart(nodeid, location):
    _current_test[0] = nodeid
    if STALL_SECONDS > 0:
        # exit=True: the stack is written and the run ends. A hung test cannot be
        # waited out — it blocks every test after it, and in CI it blocks forever.
        faulthandler.dump_traceback_later(STALL_SECONDS, exit=True,
                                          file=_open_log())


def pytest_runtest_logfinish(nodeid, location):
    if STALL_SECONDS > 0:
        faulthandler.cancel_dump_traceback_later()


def pytest_sessionfinish(session, exitstatus):
    """Nothing went wrong: drop the empty report rather than leave litter behind."""
    global _log_handle
    if STALL_SECONDS > 0:
        faulthandler.cancel_dump_traceback_later()
    handle, _log_handle = _log_handle, None
    if handle is None or handle is sys.__stderr__:
        return
    try:
        empty = handle.tell() == 0
        handle.close()
        if empty:
            os.unlink(ABORT_LOG)
    except Exception:
        pass
