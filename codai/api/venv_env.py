# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""The environment for spawning a DIFFERENT Python than this process's.

Every isolated-venv worker runs an interpreter that is not the server's —
3.10 for LongCat and the lip-sync tools, 3.12 for ACE-Step, against a server
on 3.13. Anything in the environment that points a Python at another
interpreter's files makes that child die at startup, before it can say
anything useful.

This is not hypothetical and it is not only about venvs the server builds:
in the shipped container the per-engine launchers set
``PYTHONHOME=/opt/coderai/python`` (the standalone 3.13), while PID 1 and an
interactive ``docker exec`` do not. A worker spawned from an engine process
therefore inherits it and a 3.12 interpreter cannot find its own stdlib,
while the same command run by hand in the same container works perfectly.
The observed symptom was a probe failing in milliseconds and the caller
reporting something else entirely.
"""

import os
import sys

#: Variables that aim a Python at another installation's files. PYTHONHOME is
#: the one that is actually set in the container; the rest are stripped because
#: the failure they cause is fatal, silent in the parent, and costs nothing to
#: prevent.
_LEAKY = ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONEXECUTABLE",
          "VIRTUAL_ENV", "PYTHONNOUSERSITE", "PYTHONUSERBASE",
          "__PYVENV_LAUNCHER__")


#: Extra library directories that belong to THIS interpreter even though they
#: are not under its prefix. The container launcher puts the ldd-discovered
#: natives here, and they are built against this python's ABI.
#: `/opt/coderai/python` is named explicitly as well: it is the standalone
#: interpreter the image ships, and nothing on a different python may load from
#: it. (Only foreign interpreters go through this helper — the 3.13 venvs
#: pyannote/canary/crisperwhisper run on build their own env, and they do need
#: that directory for libpython.)
_OWN_LIB_DIRS = ("/opt/coderai/local-libs", "/opt/coderai/python")


def _is_our_lib_dir(entry: str) -> bool:
    """Is this LD_LIBRARY_PATH entry part of the server's own install?"""
    try:
        real = os.path.realpath(entry)
    except Exception:
        return False
    roots = [os.path.realpath(p) for p in (sys.prefix, sys.base_prefix)
             if p] + [os.path.realpath(d) for d in _OWN_LIB_DIRS]
    return any(real == r or real.startswith(r + os.sep) for r in roots)


def clean_py_env(base: dict = None) -> dict:
    """Return ``base`` (default: this process's environment) without them.

    ``LD_LIBRARY_PATH`` is filtered rather than dropped: the container launcher
    exports ``/opt/coderai/python/lib:/opt/coderai/local-libs:<the server
    venv's nvidia wheels>``, and a foreign interpreter that inherits it loads
    THIS python's extension modules. Measured: SadTalker's python 3.10 picked up
    the 3.13 build and died with

        ImportError: /opt/coderai/local-libs/libtorch_python.so:
        undefined symbol: PyObject_GetOptionalAttrString

    -- a CPython 3.13 API. Entries outside the server's install (a system CUDA,
    anything the operator added) are left alone: an isolated venv ships its own
    torch and its own CUDA wheels, and those are the ones it must find.
    """
    env = dict(os.environ if base is None else base)
    for var in _LEAKY:
        env.pop(var, None)
    libs = env.get("LD_LIBRARY_PATH")
    if libs:
        keep = [e for e in libs.split(os.pathsep) if e and not _is_our_lib_dir(e)]
        if keep:
            env["LD_LIBRARY_PATH"] = os.pathsep.join(keep)
        else:
            env.pop("LD_LIBRARY_PATH", None)
    return env
