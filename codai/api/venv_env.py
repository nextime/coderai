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

#: Variables that aim a Python at another installation's files. PYTHONHOME is
#: the one that is actually set in the container; the rest are stripped because
#: the failure they cause is fatal, silent in the parent, and costs nothing to
#: prevent.
_LEAKY = ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONEXECUTABLE",
          "VIRTUAL_ENV", "PYTHONNOUSERSITE", "PYTHONUSERBASE",
          "__PYVENV_LAUNCHER__")


def clean_py_env(base: dict = None) -> dict:
    """Return ``base`` (default: this process's environment) without them."""
    env = dict(os.environ if base is None else base)
    for var in _LEAKY:
        env.pop(var, None)
    return env
