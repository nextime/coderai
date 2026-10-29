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

"""A venv built by a different interpreter has to be rebuilt, not reused.

An isolated-worker venv keeps ``bin/python`` as a symlink to the interpreter
that created it. Create it with one python and run with another -- built on the
host and used from the container, or built before the image moved its
interpreter -- and that link dangles.

The failure is not a clean "missing": ``Path.exists()`` follows symlinks and
answers False, so the code decides the venv is absent and runs ``python -m
venv`` over the top, which trips on the very same dangling ``bin/python3``:

    Error: [Errno 2] No such file or directory: '.../parler_venv/bin/python3'

Reported from production as "the venv can't be created", on a host where the
venv was fully populated and 7 GB of packages were sitting in it.
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


def _make_foreign_venv(root):
    """A venv whose interpreter symlink points at a path that is not there."""
    (root / "bin").mkdir(parents=True)
    (root / "lib" / "python3.13" / "site-packages").mkdir(parents=True)
    os.symlink("python3.13", root / "bin" / "python")
    os.symlink("/nonexistent/python3.13", root / "bin" / "python3.13")
    (root / "pyvenv.cfg").write_text(
        "home = /nonexistent/bin\ninclude-system-site-packages = false\n")
    return root


def test_the_dangling_interpreter_is_not_mistaken_for_a_working_one(tmp_path):
    from codai.api import parler_worker as P
    v = _make_foreign_venv(tmp_path / "parler_venv")
    py = v / "bin" / "python"
    assert py.is_symlink()
    assert not py.exists(), "precondition: the link dangles"
    assert P._venv_runs(py) is False


def test_a_real_interpreter_is_accepted():
    from codai.api import parler_worker as P
    assert P._venv_runs(sys.executable) is True


def test_a_foreign_venv_is_cleared_before_recreating(tmp_path, monkeypatch):
    """Without the clear, `python -m venv` dies on the dangling link and the
    engine reports "venv can't be created" on every request, for ever."""
    from codai.api import parler_worker as P

    v = _make_foreign_venv(tmp_path / "parler_venv")
    monkeypatch.setattr(P, "_VENV_DIR", v)
    monkeypatch.setattr(P, "_bootstrapped", False)
    monkeypatch.setattr(P, "_pip_ok", lambda py: True)

    created = []

    def _fake_run(cmd, **kw):
        cmd = [str(c) for c in cmd]
        # The dangling interpreter cannot run anything -- without this the
        # stub would answer "it works" and the clear would never be reached.
        if cmd[0].endswith(("bin/python", "bin/python3.13")) \
                and not os.path.exists(os.path.realpath(cmd[0])):
            return subprocess.CompletedProcess(cmd, 1)
        if "venv" in cmd:
            # Mimic the real failure when the target still holds a dangling
            # bin/python3, and success once it has been cleared.
            if (v / "bin").exists():
                raise subprocess.CalledProcessError(1, cmd)
            created.append(cmd)
            (v / "bin").mkdir(parents=True, exist_ok=True)
            (v / "bin" / "python").write_text("#!/bin/sh\n")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(P.subprocess, "run", _fake_run)
    P._bootstrap_venv()
    assert created, "the venv was never recreated after the foreign one"


def test_both_workers_check_that_the_interpreter_runs():
    """The trap is identical in each isolated worker, so neither may rely on
    a bare exists()/is_file() to decide a venv is usable."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for rel in ("codai/api/parler_worker.py", "codai/api/xtts_worker.py"):
        src = open(os.path.join(root, rel), encoding="utf-8").read()
        assert "_venv_runs(" in src, f"{rel} does not verify the interpreter runs"
        assert "rmtree" in src, f"{rel} cannot clear a foreign venv"
