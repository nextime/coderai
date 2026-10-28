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

"""An ffmpeg we started must not outlive us.

``render.run`` kills its child on cancel and on timeout, but both need this
process to still be alive to act. If it is SIGKILLed -- OOM, container stop, an
interrupted test run -- no finally block runs, and the child is reparented to
init and left going. An abandoned ffmpeg is not idle: a filter graph that never
reaches EOF spins at 100% CPU indefinitely. Two such orphans from interrupted
test runs had burned 15 h and 11 h of CPU on a dev host before anyone noticed
them.

PR_SET_PDEATHSIG moves that from "unlikely" to "cannot happen".
"""

import os
import subprocess
import sys
import textwrap
import time

import pytest

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"),
                                reason="PR_SET_PDEATHSIG is Linux-only")


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def test_a_child_started_through_run_dies_when_its_parent_is_killed(tmp_path):
    """Kill the parent with SIGKILL -- the harshest case, no cleanup possible."""
    # A parent that starts a long sleep with render's own preexec hook, prints
    # the child's pid, then waits to be killed.
    script = textwrap.dedent(f"""
        import sys, subprocess, time
        sys.path.insert(0, {os.getcwd()!r})
        from codai.compose.render import _die_with_parent
        p = subprocess.Popen(["sleep", "300"], preexec_fn=_die_with_parent)
        print(p.pid, flush=True)
        time.sleep(300)
    """)
    sf = tmp_path / "parent.py"
    sf.write_text(script)

    parent = subprocess.Popen([sys.executable, str(sf)], stdout=subprocess.PIPE)
    try:
        line = parent.stdout.readline().strip()
        assert line, "the parent never reported its child's pid"
        child = int(line)
        assert _alive(child), "the child should be running before we start"

        parent.kill()            # SIGKILL: nothing in the parent can clean up
        parent.wait(timeout=10)

        # The kernel delivers PDEATHSIG as the parent dies; allow for scheduling.
        for _ in range(100):
            if not _alive(child):
                break
            time.sleep(0.05)

        assert not _alive(child), (
            f"child {child} survived its parent being killed; an ffmpeg orphaned "
            f"this way can spin at 100% CPU for ever with nothing owning it")
    finally:
        if parent.poll() is None:
            parent.kill()


def test_run_installs_the_death_signal_on_every_ffmpeg_it_starts():
    """The hook has to be wired into Popen, not merely defined."""
    import ast
    import io
    src = io.open("codai/compose/render.py", encoding="utf-8").read()
    tree = ast.parse(src)

    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "run")
    popens = [n for n in ast.walk(fn)
              if isinstance(n, ast.Call)
              and isinstance(n.func, ast.Attribute)
              and n.func.attr == "Popen"]
    assert popens, "render.run no longer starts a subprocess; retarget this test"
    for call in popens:
        kw = {k.arg for k in call.keywords}
        assert "preexec_fn" in kw, (
            "render.run starts ffmpeg without preexec_fn, so the child can "
            "outlive this process and spin unattended")
