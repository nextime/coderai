"""The image's launcher must not kill a running server to answer a question.

`coderai` on PATH inside the image IS packaging/linux/launcher/coderai-oci, so
`podman exec <container> coderai --admin-session` ran its stale-tree sweep and
SIGKILLed the front that was serving clients. Found on a live orchestrator: six
kills in fifteen minutes, ~20 s of 502s each, nothing in the app log to show why.
"""
import os
import shutil
import stat
import subprocess
import tempfile

import pytest

LAUNCHER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "packaging", "linux", "launcher", "coderai-oci")


def _sandbox(tmp):
    """A PATH where pkill records its calls and python3 records its argv."""
    bin_dir = os.path.join(tmp, "bin")
    os.makedirs(bin_dir)
    log = os.path.join(tmp, "calls.log")
    for name in ("pkill", "python3"):
        path = os.path.join(bin_dir, name)
        with open(path, "w") as fh:
            fh.write(f'#!/bin/sh\necho "{name} $*" >> "{log}"\nexit 0\n')
        os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    # The launcher execs a fixed interpreter path; stub that too.
    fake_py = os.path.join(tmp, "opt", "coderai", "python", "bin")
    os.makedirs(fake_py)
    shutil.copy(os.path.join(bin_dir, "python3"), os.path.join(fake_py, "python3"))
    return bin_dir, log, fake_py


def _run(args, tmp):
    bin_dir, log, fake_py = _sandbox(tmp)
    # Skip the launcher's first-run config write: it calls a hard-coded
    # interpreter path that only exists in the image, and `set -e` would abort
    # the script before it ever reaches the sweep -- making every assertion here
    # pass for the wrong reason.
    cfg = os.path.join(tmp, "config", "coderai")
    os.makedirs(cfg, exist_ok=True)
    with open(os.path.join(cfg, "config.json"), "w") as fh:
        fh.write('{"server": {"host": "127.0.0.1", "port": 8776}}\n')
    env = dict(os.environ)
    env.update({
        "PATH": bin_dir + ":" + "/usr/bin:/bin",
        "CODERAI_CONFIG_DIR": os.path.join(tmp, "config"),
        "CODERAI_MODELS_DIR": os.path.join(tmp, "models"),
        "CODERAI_CACHE_DIR": os.path.join(tmp, "cache"),
    })
    # Point the hard-coded interpreter at the stub by running the script with the
    # sandbox grafted in: the launcher calls /opt/coderai/python/bin/python3 only
    # to write a first-run config.json, which pkill-recording does not need.
    subprocess.run(["sh", LAUNCHER] + args, env=env, capture_output=True, timeout=60)
    try:
        with open(log) as fh:
            return fh.read()
    except OSError:
        return ""


@pytest.mark.parametrize("flag", ["--admin-session", "--version", "--help"])
def test_a_one_shot_flag_reaps_nothing(flag):
    with tempfile.TemporaryDirectory() as tmp:
        calls = _run([flag], tmp)
    assert "pkill" not in calls, (
        f"{flag} swept the process tree: a flag that prints and exits must never "
        f"kill a running server. Calls: {calls!r}")


def test_a_server_start_still_reaps_a_survivor():
    with tempfile.TemporaryDirectory() as tmp:
        calls = _run([], tmp)
    assert "pkill" in calls and "coderai-front" in calls, (
        "the stale-tree sweep is what keeps two fronts from fighting over one GPU; "
        f"a normal start must still run it. Calls: {calls!r}")


def test_the_guard_names_the_mint_flag():
    body = open(LAUNCHER).read()
    guard = body.split("_oneshot=\"\"", 1)
    assert len(guard) == 2, "no one-shot guard in the launcher"
    assert "--admin-session" in guard[1].split("fi", 1)[0], \
        "--admin-session is the flag that caused this; it must be in the guard"
