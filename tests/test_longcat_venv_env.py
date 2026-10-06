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
    assert var not in lw._py310_env()


def test_everything_else_is_kept(monkeypatch):
    """Only the interpreter-specific ones go — HF cache paths, CUDA_VISIBLE_DEVICES
    and the rest still have to reach the child."""
    monkeypatch.setenv("HF_HOME", "/AI/hfcache/huggingface")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    env = lw._py310_env()
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
    assert src.count("_py310_env()") >= 4
    # and no step left reporting only an exit status. Matched as a CALL argument
    # (", check=True)"), so the prose in _run_py310's docstring explaining the old
    # behaviour does not trip it.
    assert ", check=True)" not in src
