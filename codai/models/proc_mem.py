# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""How much host memory the server fleet actually holds.

The host-RAM cap used to be enforced against the sum of RSS over the front, the
engines and every worker. That number cannot be driven down by evicting models,
for two reasons:

* **Shared pages are counted once per process.** The interpreter, torch, the
  CUDA runtime and libc are mapped into every engine. Summing RSS adds each of
  those pages as many times as there are processes.
* **File-backed pages count as RSS.** Walking a mmap'd GGUF or safetensors file
  maps its pages into the process; they are page cache, reclaimed by the kernel
  on demand, and dropping a model does not release them.

Measured on a production host, 2026-10-29. The mitigation ladder ran every
~15 s for hours with the figure never moving:

    RAM watch: RSS 40.0/43.0 GB (93%) -- mitigation [gc+trim+evict_idle] -> 39.1 GB
    RAM watch: RSS 39.1/43.0 GB (91%) -- mitigation [gc+trim+evict_idle] -> 39.1 GB
    RAM watch: RSS 39.1/43.0 GB (91%) -- mitigation [gc+trim+evict_idle] -> 39.1 GB

Before and after are the same value, 291 identical repetitions for one figure
alone, while the host itself had 14 GB available and 11 GB of reclaimable
cache. It was not merely noisy -- it evicted working models to chase it:

    RAM eviction: unloading 'video:Lightricks/LTX-2.5-Diffusers' to free host
    RAM (RSS 40.0 GB > cap target 36.5 GB)

The fix is to measure proportional set size instead. ``Pss`` divides each
shared page by the number of processes sharing it, so summing PSS across a tree
counts every page exactly once, and the ``Pss_Anon``/``Pss_File`` split
separates anonymous memory (which only freeing an allocation releases) from
page cache (which the kernel reclaims by itself). A cap should chase the
former.
"""

from __future__ import annotations

import os

#: Fields of ``/proc/<pid>/smaps_rollup`` we care about, in kB.
_WANTED = ("Pss", "Pss_Anon", "Pss_File", "Pss_Shmem", "Rss")


def read_rollup(pid) -> dict:
    """Parse ``/proc/<pid>/smaps_rollup`` into a dict of bytes, or ``{}``.

    Cheap: the kernel aggregates the whole address space into one record, so
    this is a single short read regardless of how many mappings a process has
    (unlike ``smaps``, which emits a stanza per mapping and is what makes
    naive memory accounting expensive).
    """
    try:
        with open(f"/proc/{int(pid)}/smaps_rollup") as fh:
            out = {}
            for line in fh:
                name, _, rest = line.partition(":")
                if name in _WANTED:
                    try:
                        out[name] = int(rest.split()[0]) * 1024
                    except (IndexError, ValueError):
                        continue
            return out
    except (OSError, ValueError, TypeError):
        return {}


def process_held_bytes(pid) -> int:
    """Memory this one process holds that only it can release.

    ``Pss_Anon`` is anonymous memory -- allocations, model weights that were
    copied rather than mapped -- shared pages already divided by their sharers.
    ``Pss_Shmem`` is shared-memory segments, which are not backed by a file on
    disk and so are not reclaimable either. ``Pss_File`` is deliberately
    excluded: those are mapped file pages, i.e. page cache.

    Falls back to ``Pss`` and then ``Rss`` on a kernel or permission setup that
    does not expose the breakdown, so the caller always gets a number -- a
    coarse one, which is still better than none.
    """
    r = read_rollup(pid)
    if not r:
        return 0
    if "Pss_Anon" in r:
        return r.get("Pss_Anon", 0) + r.get("Pss_Shmem", 0)
    if "Pss" in r:
        return r["Pss"]
    return r.get("Rss", 0)


def tree_pids(root_pid=None) -> list:
    """The root process and all its descendants."""
    try:
        import psutil
    except Exception:
        return [int(root_pid)] if root_pid else [os.getpid()]
    try:
        proc = psutil.Process(int(root_pid)) if root_pid else psutil.Process()
    except Exception:
        try:
            proc = psutil.Process()
        except Exception:
            return []
    pids = [proc.pid]
    try:
        pids += [c.pid for c in proc.children(recursive=True)]
    except Exception:
        pass
    return pids


def tree_held_gb(root_pid=None) -> float:
    """Unreclaimable host memory held by a process tree, in GB.

    Summing PSS across the tree is sound precisely because PSS has already
    divided shared pages by their sharers: a page mapped into five engines
    contributes a fifth from each, totalling one.
    """
    total = 0
    for pid in tree_pids(root_pid):
        total += process_held_bytes(pid)
    return total / 1e9


def tree_rss_gb(root_pid=None) -> float:
    """Summed RSS of the tree, in GB -- the old reading, kept for comparison.

    Reported alongside the PSS figure so the gap between them is visible in
    diagnostics rather than inferred: on a multi-engine host it is large, and
    it is entirely double-counted shared pages plus page cache.
    """
    total = 0
    for pid in tree_pids(root_pid):
        r = read_rollup(pid)
        total += r.get("Rss", 0)
    return total / 1e9


def host_available_gb() -> float:
    """What the kernel says is actually available, including reclaimable cache.

    ``MemAvailable`` is the kernel's own estimate of what can be allocated
    without swapping, and it already accounts for the page cache it is willing
    to drop. A cap that disagrees with this is measuring the wrong thing.
    """
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 ** 2)
    except OSError:
        pass
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
        return 0.0
