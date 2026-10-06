"""models.json stores a repo id; everything downstream opens files by path.

"meituan-longcat/LongCat-Video" is what the model page registers, and it was handed
to the service unchanged — which failed every load with "checkpoint directory does not
exist: meituan-longcat/LongCat-Video" before a byte was read.
"""
import importlib.util
import os
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
COMMON = ROOT / "tools" / "longcat_common.py"


@pytest.fixture
def LC():
    spec = importlib.util.spec_from_file_location("lc_common_resolve", COMMON)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_cache(tmp_path, repo="meituan-longcat/LongCat-Video", revision="abc123"):
    snap = (tmp_path / "hub" / ("models--" + repo.replace("/", "--"))
            / "snapshots" / revision)
    snap.mkdir(parents=True)
    return snap


def test_a_repo_id_resolves_to_its_snapshot(LC, tmp_path, monkeypatch):
    snap = _make_cache(tmp_path)
    monkeypatch.setenv("HF_HOME", str(tmp_path))

    assert LC.resolve_checkpoint("meituan-longcat/LongCat-Video") == str(snap)


def test_an_existing_directory_is_left_alone(LC, tmp_path):
    assert LC.resolve_checkpoint(str(tmp_path)) == str(tmp_path)


def test_an_absolute_path_is_never_treated_as_a_repo_id(LC, tmp_path, monkeypatch):
    """A path with a slash in it is still a path."""
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    missing = str(tmp_path / "not-there")
    assert LC.resolve_checkpoint(missing) == missing


def test_an_unknown_repo_comes_back_unchanged(LC, tmp_path, monkeypatch):
    """So the existing "does not exist" message names what was asked for, rather than
    some invented cache path the user never mentioned."""
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert LC.resolve_checkpoint("nobody/nothing") == "nobody/nothing"


def test_it_never_downloads(LC):
    """The model page owns downloads. A resolver that falls back to the hub would turn
    a misconfigured entry into a silent 74 GB pull."""
    import ast
    tree = ast.parse(COMMON.read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "resolve_checkpoint")
    # The prose may well mention the hub; what matters is that nothing imports or
    # calls it.
    imported = {a.name.split(".")[0] for n in ast.walk(fn)
                if isinstance(n, ast.Import) for a in n.names}
    imported |= {(n.module or "").split(".")[0] for n in ast.walk(fn)
                 if isinstance(n, ast.ImportFrom)}
    assert "huggingface_hub" not in imported, f"imports {imported}"
    called = {n.func.id for n in ast.walk(fn)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "snapshot_download" not in called


@pytest.mark.parametrize("var", ["HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"])
def test_the_hub_cache_variables_are_honoured(LC, tmp_path, monkeypatch, var):
    """The container sets these, and the service's venv must land on the same store as
    the server's or it will load a different checkpoint than the one that was quantised.
    """
    snap = _make_cache(tmp_path)
    for clear in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        monkeypatch.delenv(clear, raising=False)
    monkeypatch.setenv(var, str(tmp_path / "hub"))

    assert LC.resolve_checkpoint("meituan-longcat/LongCat-Video") == str(snap)


def test_the_newest_revision_wins(LC, tmp_path, monkeypatch):
    """A re-pull leaves the old snapshot behind; loading it would serve weights that
    do not match what was just downloaded."""
    old = _make_cache(tmp_path, revision="old")
    new = _make_cache(tmp_path, revision="new")
    os.utime(old, (1000, 1000))
    os.utime(new, (2000, 2000))
    monkeypatch.setenv("HF_HOME", str(tmp_path))

    assert LC.resolve_checkpoint("meituan-longcat/LongCat-Video") == str(new)


def test_the_service_resolves_before_it_validates(LC):
    """checkpoint_problems() is what produced the error, so resolution has to happen
    first or the message is about a repo id that was never going to be a directory."""
    src = (ROOT / "tools" / "longcat_service.py").read_text(encoding="utf-8")
    assert "LC.resolve_checkpoint(body.get(\"model\")" in src
