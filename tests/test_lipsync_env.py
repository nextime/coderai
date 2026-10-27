"""The environment a lip-sync engine is spawned into.

Measured, not theorised: a presenter request on the live server answered

    sadtalker failed (rc=1): Fatal Python error: init_fs_encoding: failed to
    get the Python codec of the filesystem encoding
    ModuleNotFoundError: No module named 'encodings'

because the shims on PATH run the bundled **python 3.10** venv while the
process that spawns them exports ``PYTHONHOME=/opt/coderai/python`` — the
standalone 3.13. The child dies before executing a byte, the job falls back to
the scene's visuals, and the warning blames the engine.

Same root cause as the ACE-Step worker's "requirements missing"; see
``codai/api/venv_env.py``.
"""
import os

import pytest

from codai.api import lipsync
from codai.compose import presenter


class _FakeProc:
    returncode = 0

    def __init__(self):
        self.stdout, self.stderr = "", ""

    def communicate(self, timeout=None):
        return b"", b""

    def poll(self):
        return 0


def test_an_engine_is_not_handed_this_interpreters_pythonhome(monkeypatch):
    seen = {}

    def _popen(cmd, **kw):
        seen.update(kw)
        return _FakeProc()

    monkeypatch.setenv("PYTHONHOME", "/opt/coderai/python")
    monkeypatch.setenv("PYTHONPATH", "/opt/coderai/app")
    monkeypatch.setattr(presenter.subprocess, "Popen", _popen)
    presenter._run(["sadtalker", "--driven_audio", "a.wav"])
    env = seen["env"]
    assert env is not None, "no env= at all means the child inherits PYTHONHOME"
    assert "PYTHONHOME" not in env and "PYTHONPATH" not in env


def test_a_repo_engine_still_gets_the_path_it_needs_and_nothing_else(monkeypatch):
    """The managed checkouts import their own package as a top-level module, so
    PYTHONPATH is set deliberately — to the checkout, not to the server's app
    directory it would otherwise inherit."""
    seen = {}
    monkeypatch.setenv("PYTHONHOME", "/opt/coderai/python")
    monkeypatch.setenv("PYTHONPATH", "/opt/coderai/app")
    monkeypatch.setattr(presenter.subprocess, "Popen",
                        lambda cmd, **kw: (seen.update(kw), _FakeProc())[1])
    presenter._run(["python", "scripts/inference.py"], pythonpath="/cache/lipsync/MuseTalk")
    assert seen["env"]["PYTHONPATH"] == "/cache/lipsync/MuseTalk"
    assert "PYTHONHOME" not in seen["env"]


def test_wav2lips_runner_cleans_the_environment_too(monkeypatch, tmp_path):
    """run_wav2lip is the fallback when no wav2lip shim is on PATH, and it runs
    the repo's inference.py the same way."""
    seen = {}
    (tmp_path / "inference.py").write_text("")
    monkeypatch.setattr(lipsync, "ensure_wav2lip",
                        lambda: (tmp_path, tmp_path / "ckpt.pth"))
    monkeypatch.setenv("PYTHONHOME", "/opt/coderai/python")

    def _run(cmd, **kw):
        seen.update(kw)
        (tmp_path / "out.mp4").write_bytes(b"x")
        return type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(lipsync.subprocess, "run", _run)
    lipsync.run_wav2lip(str(tmp_path / "face.mp4"), str(tmp_path / "a.wav"),
                        str(tmp_path / "out.mp4"))
    assert "PYTHONHOME" not in seen["env"]
    assert seen["env"]["PYTHONPATH"] == str(tmp_path)


def test_the_helper_is_the_shared_one(monkeypatch):
    """One implementation, so a new worker cannot reinvent a leaky one."""
    import inspect

    src = inspect.getsource(presenter._run) + inspect.getsource(lipsync.run_wav2lip)
    assert src.count("clean_py_env()") == 2
    assert "dict(os.environ)" not in src
