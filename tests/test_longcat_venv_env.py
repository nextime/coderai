"""Spawning LongCat's Python 3.10 from coderai's 3.13.

Pointing a Python at another interpreter's files kills it during startup:

    Fatal Python error: init_fs_encoding: failed to get the Python codec of the
    filesystem encoding
    ModuleNotFoundError: No module named 'encodings'

which is what the first venv build produced. The parent reported only
"returned non-zero exit status 1" — the reason reached the log by accident, not
by design. Both halves of that are fixed here: the environment is stripped of
anything interpreter-specific, and a failed step raises with what the child said.
"""
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
lw = pytest.importorskip("codai.api.longcat_worker")


@pytest.mark.parametrize("var", [
    "PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONEXECUTABLE",
    "VIRTUAL_ENV", "PYTHONNOUSERSITE", "PYTHONUSERBASE", "__PYVENV_LAUNCHER__",
])
def test_interpreter_variables_are_stripped(monkeypatch, var):
    monkeypatch.setenv(var, "/opt/coderai/python")
    assert var not in lw._clean_py_env()


def test_everything_else_is_kept(monkeypatch):
    """Only the interpreter-specific ones go — HF cache paths, CUDA_VISIBLE_DEVICES
    and the rest still have to reach the child."""
    monkeypatch.setenv("HF_HOME", "/AI/hfcache/huggingface")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    env = lw._clean_py_env()
    assert env["HF_HOME"] == "/AI/hfcache/huggingface"
    assert env["CUDA_VISIBLE_DEVICES"] == "0"


def test_a_failing_step_raises_with_the_childs_output():
    """The whole point: an exit code alone is not a diagnosis."""
    import sys
    with pytest.raises(RuntimeError) as err:
        lw._run_py310([sys.executable, "-c",
                       "import sys; sys.stderr.write('THE REAL REASON\\n'); sys.exit(3)"],
                      "a failing step")
    msg = str(err.value)
    assert "THE REAL REASON" in msg
    assert "exit 3" in msg
    assert "a failing step" in msg


def test_a_successful_step_is_silent():
    import sys
    lw._run_py310([sys.executable, "-c", "pass"], "a fine step")


def test_every_py310_spawn_uses_the_clean_environment():
    src = (ROOT / "codai" / "api" / "longcat_worker.py").read_text(encoding="utf-8")
    # no raw inheritance left on any subprocess that runs the 3.10 side
    assert "env = dict(os.environ)" not in src
    assert src.count("_clean_py_env()") >= 4
    # and no step left reporting only an exit status. Matched as a CALL argument
    # (", check=True)"), so the prose in _run_py310's docstring explaining the old
    # behaviour does not trip it.
    assert ", check=True)" not in src


# ── which interpreter builds which venv ─────────────────────────────────────
# The two venvs want OPPOSITE things and conflating them is what produced
#   ERROR: No matching distribution found for simpletuner
# — every release filtered out by Requires-Python under 3.10, which reads exactly
# like the package does not exist on PyPI. It does; it needs >=3.12.

import subprocess
import sys as _sys


def test_the_training_venv_uses_a_modern_interpreter():
    found = lw._find_train_python()
    assert found, "no interpreter found for SimpleTuner"
    out = subprocess.run([found, "-V"], capture_output=True, text=True).stdout
    major, minor = (int(x) for x in out.split()[1].split(".")[:2])
    assert (3, 12) <= (major, minor) < (3, 15), f"SimpleTuner cannot install into {out}"


def test_this_processs_own_interpreter_is_used_when_it_qualifies():
    """coderai runs on 3.13, so the training venv needs nothing bundled."""
    if (3, 12) <= _sys.version_info[:2] < (3, 15):
        assert lw._find_train_python() == _sys.executable


def test_an_explicit_override_wins(tmp_path, monkeypatch):
    fake = tmp_path / "python3.12"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    monkeypatch.setenv("CODERAI_LONGCAT_TRAIN_PYTHON", str(fake))
    assert lw._find_train_python() == str(fake)


def test_the_two_venvs_do_not_share_an_interpreter_finder():
    """The inference venv pins 3.10 for torch 2.6 + transformers 4.41; the training
    venv needs 3.12+. One function answering both is the bug."""
    src = (ROOT / "codai" / "api" / "longcat_worker.py").read_text(encoding="utf-8")
    body = src[src.index("def ensure_train_built("):]
    body = body[:body.index("\ndef ")]
    assert "_find_train_python(" in body
    assert "_find_python310(" not in body, "the training venv must not use the 3.10"


def test_the_manual_instructions_name_the_right_version():
    src = (ROOT / "codai" / "api" / "longcat_worker.py").read_text(encoding="utf-8")
    body = src[src.index("def ensure_train_built("):]
    body = body[:body.index("\ndef ")]
    assert "python3.12+" in body and "python3.10> -m venv" not in body


# ── a venv built by the wrong interpreter ───────────────────────────────────

def _fake_venv(tmp_path, version):
    (tmp_path / "bin").mkdir(parents=True)
    (tmp_path / "pyvenv.cfg").write_text(
        f"home = /somewhere/bin\ninclude-system-site-packages = false\n"
        f"version = {version}\n")
    return tmp_path / "bin" / "python"


@pytest.mark.parametrize("version,expected", [
    ("3.10.18", (3, 10)), ("3.13.5", (3, 13)), ("3.12.1", (3, 12)),
])
def test_a_venvs_version_is_read_from_its_config(tmp_path, version, expected):
    """Read, not executed: the interpreter that built it may be gone, and a failed
    exec would look the same as an incomplete venv."""
    assert lw._venv_python_version(_fake_venv(tmp_path, version)) == expected


def test_an_unreadable_venv_reports_nothing_rather_than_guessing(tmp_path):
    (tmp_path / "bin").mkdir(parents=True)
    assert lw._venv_python_version(tmp_path / "bin" / "python") == ()
    (tmp_path / "pyvenv.cfg").write_text("home = /x\n")       # no version line
    assert lw._venv_python_version(tmp_path / "bin" / "python") == ()


def test_the_stale_venv_left_by_the_310_attempt_is_detected(tmp_path):
    """The exact leftover: the failed 3.10 run created the directory, so creation
    would be skipped and pip would install into 3.10 again, forever."""
    v = lw._venv_python_version(_fake_venv(tmp_path, "3.10.18"))
    assert v and not ((3, 12) <= v < (3, 15)), "this must be seen as unusable"


def test_a_good_venv_is_left_alone(tmp_path):
    v = lw._venv_python_version(_fake_venv(tmp_path, "3.13.5"))
    assert (3, 12) <= v < (3, 15)


def test_the_builder_rebuilds_an_unusable_training_venv():
    src = (ROOT / "codai" / "api" / "longcat_worker.py").read_text(encoding="utf-8")
    body = src[src.index("def ensure_train_built("):]
    body = body[:body.index("\ndef ")]
    assert "_venv_python_version(py)" in body
    assert "rmtree" in body, "an unusable venv must be replaced, not reused"
