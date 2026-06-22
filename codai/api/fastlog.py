# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Fast, low-overhead stdout for engine logs.

The engine emits a lot of debug output from the generation hot path. Python's
``print()`` goes through a buffered, lock-guarded text stream — extra overhead per
call and a lock that can serialize against other threads. ``fast_print`` writes the
formatted line with a single ``os.write(1)`` syscall instead: unbuffered, no
Python-level lock, and atomic for pipe writes up to PIPE_BUF (so engine log lines
don't interleave). Import it as ``print`` in a hot module:

    from codai.api.fastlog import fast_print as print

It mirrors ``print``'s signature and falls back to the real ``print`` for non-stdout
targets (e.g. ``file=sys.stderr``) or if the raw write fails.
"""
import builtins
import os
import sys


def fast_print(*args, sep=" ", end="\n", file=None, flush=False):
    if file is not None and file is not sys.stdout:
        return builtins.print(*args, sep=sep, end=end, file=file, flush=flush)
    try:
        data = (sep.join(str(a) for a in args) + end).encode("utf-8", "replace")
        os.write(1, data)
    except Exception:
        try:
            builtins.print(*args, sep=sep, end=end, flush=flush)
        except Exception:
            pass
