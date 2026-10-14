"""A ceiling on the bytes that in-flight OCR documents may hold at once.

``ocr.max_concurrency`` bounds how many pages are BEING READ. It says nothing about
how many are RESIDENT, and until 0.3.65 those were different numbers by a factor of
the document length: ``ocr_document`` rasterised every page of every document up
front and held the list until the response was written, so a burst of multi-page
PDFs could pin tens of gigabytes while the semaphore cheerfully reported that only
``max_concurrency`` pages were in flight.

It is not a hypothetical. On 2026-10-14 a production install was OOM-killed twice
inside forty minutes, at 90.7 GB and 92.7 GB on a 125 GB box; lowering
``max_concurrency`` eightfold between the two made it half a gigabyte WORSE,
because the semaphore was never the thing holding the memory.

So the bound has to be on bytes, and it has to be taken BEFORE the pages exist —
once a document is rasterised the memory is already spent. :func:`page_estimate`
prices a document from its page tree without rendering it; this module is the
gate that charges for it.

Two rules keep the gate from being worse than the problem it solves:

* **A document larger than the whole budget still runs** — clamped to the budget,
  alone. A gate that can refuse forever is a deadlock with a polite error message.
* **Reservations are always returned**, including on cancellation, which is the
  case a naive implementation misses: a client that disconnects mid-document would
  otherwise leak its reservation and shrink the budget permanently.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from typing import Optional

#: Never derive an auto budget smaller than this — below it even one ordinary
#: document would serialise the whole subsystem.
MIN_BUDGET_BYTES = 512 * 1024 * 1024

#: Share of usable memory an auto budget claims. OCR is one tenant among several
#: (engines, models, the front), so a quarter is deliberately modest: the cost of
#: being wrong high is a kill, the cost of being wrong low is queueing.
AUTO_FRACTION = 0.25


#: (path, literal meaning "no limit") pairs, cgroup v2 first. A parameter rather
#: than a constant so the sentinel handling is testable without faking `open`.
CGROUP_LIMIT_FILES = (("/sys/fs/cgroup/memory.max", "max"),
                      ("/sys/fs/cgroup/memory/memory.limit_in_bytes", None))


def _cgroup_limit_bytes(files=None) -> Optional[int]:
    """The container's memory ceiling, or None when unlimited/not containerised.

    Checked BEFORE host RAM: a 2 GB container on a 128 GB host that sizes itself
    from the host is the same bug as having no budget at all.
    """
    for path, unlimited in (files or CGROUP_LIMIT_FILES):
        try:
            with open(path) as fh:
                raw = fh.read().strip()
        except Exception:
            continue
        if unlimited is not None and raw == unlimited:
            return None
        try:
            val = int(raw)
        except ValueError:
            continue
        # cgroup v1 writes a sentinel near 2**63 for "no limit".
        if val <= 0 or val >= (1 << 62):
            return None
        return val
    return None


def _total_memory_bytes() -> Optional[int]:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return None


def auto_budget_bytes() -> int:
    """A resident-page budget derived from what this process may actually use.

    The smaller of the cgroup limit and host RAM, times :data:`AUTO_FRACTION`,
    floored at :data:`MIN_BUDGET_BYTES`. Used when ``ocr.max_resident_mb`` is 0,
    which is the shipped default — an install that never heard of this setting
    still gets a ceiling, because the installs that most need one are exactly the
    ones that will not set it.
    """
    limits = [v for v in (_cgroup_limit_bytes(), _total_memory_bytes()) if v]
    if not limits:
        return MIN_BUDGET_BYTES
    return max(MIN_BUDGET_BYTES, int(min(limits) * AUTO_FRACTION))


class DocumentBudget:
    """Admits documents while their estimated resident bytes fit the ceiling.

    Not a semaphore over a count: documents differ by three orders of magnitude
    (a one-page scan against a 400-page filing), and any count that admits the
    big ones safely throttles the small ones pointlessly.
    """

    def __init__(self, total_bytes: int):
        self._total = max(1, int(total_bytes))
        self._used = 0
        self._cond = asyncio.Condition()
        #: Peak simultaneous reservation, for reporting what the ceiling is doing.
        self.high_water = 0
        #: How many documents had to wait. A gate nobody waits at costs nothing;
        #: one everybody waits at is mis-sized, and silence hides both.
        self.waits = 0

    @property
    def total_bytes(self) -> int:
        return self._total

    @property
    def used_bytes(self) -> int:
        return self._used

    def resize(self, total_bytes: int) -> None:
        """Change the ceiling in place (config reload) without dropping reservations."""
        self._total = max(1, int(total_bytes))

    @contextlib.asynccontextmanager
    async def reserve(self, nbytes: int):
        """Hold ``nbytes`` of the budget for the duration of the block.

        A request larger than the entire budget is clamped to it and runs alone
        rather than waiting for room that can never exist.
        """
        want = max(0, int(nbytes))
        want = min(want, self._total)
        waited = False
        async with self._cond:
            while self._used + want > self._total and self._used > 0:
                waited = True
                await self._cond.wait()
            self._used += want
            if waited:
                self.waits += 1
            self.high_water = max(self.high_water, self._used)
        try:
            yield
        finally:
            # Must run on cancellation too, or a disconnected client permanently
            # shrinks the budget. asyncio.shield keeps the notify from being
            # swallowed by the cancellation that brought us here.
            async with self._cond:
                self._used = max(0, self._used - want)
                self._cond.notify_all()


_budget: Optional[DocumentBudget] = None
_budget_loop = None


def get_budget(max_resident_mb: int = 0) -> DocumentBudget:
    """The process-wide budget, created on first use and resized on config change.

    Rebuilt when the running event loop changes: an ``asyncio.Condition`` is bound
    to the loop it was created on, and a budget carried across loops (tests, a
    worker restart) would wait on a loop that is no longer running.
    """
    global _budget, _budget_loop
    want = int(max_resident_mb) * 1024 * 1024 if max_resident_mb else auto_budget_bytes()
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if _budget is None or _budget_loop is not loop:
        _budget = DocumentBudget(want)
        _budget_loop = loop
    elif _budget.total_bytes != want:
        _budget.resize(want)
    return _budget


def reset_budget() -> None:
    """Drop the process-wide budget (tests)."""
    global _budget, _budget_loop
    _budget = None
    _budget_loop = None
