"""The shipped image has no python3.11 and no uv, so it could install neither engine.

LatentSync and MuseTalk each pin a stack the server's 3.13 cannot host, so each
runs in its own venv on its own interpreter. Making that venv had two routes:
``python3.11 -m venv`` and uv. The container coderai ships carries the bundled
3.13, the distribution's 3.12 and no uv at all, so the admin page's Install
button failed at the first step with

    could not create a python 3.11 venv for latentsync at …:
    no python3.11 on PATH; uv not found

i.e. two engines advertised as installable were permanently uninstallable from
the image that advertises them. What is asserted here: a standalone interpreter
the image carries or an earlier install provisioned is found and used, one that
does not run is treated as absent, and the fetch of last resort is aimed at a
real asset for every version an engine pins — without downloading anything.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def le():
    from codai.api import lipsync_engines
    return lipsync_engines


def _fake_python_tree(root: Path, name: str, target: str) -> Path:
    """A standalone-python-shaped directory whose bin/python3 is `target`."""
    bindir = root / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    py = bindir / name
    py.symlink_to(target)
    return py


# ── the fetch URL ────────────────────────────────────────────────────────────

def test_every_version_an_engine_pins_has_a_build_recorded(le):
    """A missing entry would be a 404 at the end of an install, not a message."""
    for sp in le.ENGINES.values():
        assert sp.py_version in le._PBS_PATCH, sp.name
        url = le._pbs_url(sp.py_version)
        assert url.startswith("https://github.com/astral-sh/python-build-standalone/")
        assert f"cpython-{le._PBS_PATCH[sp.py_version]}+{le._PBS_RELEASE}" in url
        assert url.endswith("-install_only.tar.gz")


def test_the_patch_version_matches_the_minor_it_is_filed_under(le):
    for minor, patch in le._PBS_PATCH.items():
        assert patch.startswith(minor + ".")


def test_an_unknown_minor_names_the_file_to_edit(le):
    with pytest.raises(RuntimeError) as e:
        le._pbs_url("3.7")
    assert "_PBS_PATCH" in str(e.value)


def test_an_unsupported_architecture_says_so_rather_than_404ing(le, monkeypatch):
    monkeypatch.setattr("platform.machine", lambda: "mips64")
    with pytest.raises(RuntimeError) as e:
        le._pbs_url("3.11")
    assert "mips64" in str(e.value)


# ── finding an interpreter that is not on PATH ───────────────────────────────

def test_a_provisioned_interpreter_is_found(le, tmp_path, monkeypatch):
    mine = f"{sys.version_info[0]}.{sys.version_info[1]}"
    root = tmp_path / ("py" + mine.replace(".", ""))
    _fake_python_tree(root, "python3", sys.executable)
    monkeypatch.setattr(le, "_provisioned_python_dir", lambda v: root)
    assert le._standalone_python(mine) == root / "bin" / "python3"


def test_an_interpreter_that_does_not_run_counts_as_absent(le, tmp_path, monkeypatch):
    """Which is the normal state of a venv built on a host whose python is gone:
    bin/python is a symlink to an interpreter this filesystem has not got, so
    the file 'exists' by name and cannot execute a byte."""
    root = tmp_path / "py311"
    _fake_python_tree(root, "python3.11", "/nowhere/python3.11")
    monkeypatch.setattr(le, "_provisioned_python_dir", lambda v: root)
    assert le._standalone_python("3.11") is None
    assert le._python_minor(root / "bin" / "python3.11") is None


def test_an_interpreter_of_the_wrong_minor_is_not_accepted(le, tmp_path, monkeypatch):
    mine = f"{sys.version_info[0]}.{sys.version_info[1]}"
    other = "3.11" if mine != "3.11" else "3.10"
    root = tmp_path / ("py" + other.replace(".", ""))
    _fake_python_tree(root, f"python{other}", sys.executable)
    monkeypatch.setattr(le, "_provisioned_python_dir", lambda v: root)
    assert le._standalone_python(other) is None


def test_the_provisioned_directory_is_under_the_config_root(le):
    """Not /opt and not a temp dir: the config root is what a deployment mounts
    and keeps, so the interpreter is still there, at the same path, for the
    server that runs the engine."""
    d = le._provisioned_python_dir("3.11")
    assert d.name == "py311"
    from codai.platform_paths import legacy_style_config_dir
    assert d.parent == legacy_style_config_dir()


# ── what _create_venv does with it ───────────────────────────────────────────

def test_the_standalone_interpreter_is_used_when_path_has_none(le, tmp_path, monkeypatch):
    """The shipped image's case, end to end: no python3.11 on PATH, no uv, and a
    bundled interpreter — and a real venv comes out without a download."""
    mine = f"{sys.version_info[0]}.{sys.version_info[1]}"
    sp = le.EngineSpec(name="probe", vendor="x", py_version=mine,
                       requirements="r.txt", entrypoint="e", weights=(),
                       ready_markers=(), probe_imports=())
    monkeypatch.setitem(le.ENGINES, "probe", sp)
    root = tmp_path / ("py" + mine.replace(".", ""))
    _fake_python_tree(root, "python3", sys.executable)
    monkeypatch.setattr(le, "_provisioned_python_dir", lambda v: root)
    monkeypatch.setattr(le.shutil, "which", lambda n: None)
    monkeypatch.setattr(le, "_uv", lambda: None)
    monkeypatch.setattr(le, "_fetch_standalone_python",
                        lambda *a, **k: pytest.fail("fetched despite a bundled python"))
    venv = tmp_path / "venv"
    le._create_venv("probe", venv)
    assert le._venv_python(venv).is_file()
    assert le._python_minor(le._venv_python(venv)) == mine


def test_a_dead_python_on_path_does_not_consume_the_attempt(le, tmp_path, monkeypatch):
    """A PATH entry named python3.11 that cannot run is what a container that
    inherited a host's PATH looks like. It must not be preferred over the
    interpreter that works."""
    mine = f"{sys.version_info[0]}.{sys.version_info[1]}"
    sp = le.EngineSpec(name="probe", vendor="x", py_version=mine,
                       requirements="r.txt", entrypoint="e", weights=(),
                       ready_markers=(), probe_imports=())
    monkeypatch.setitem(le.ENGINES, "probe", sp)
    dead = tmp_path / "bin" / f"python{mine}"
    dead.parent.mkdir(parents=True)
    dead.symlink_to("/nowhere/python")
    monkeypatch.setattr(le.shutil, "which", lambda n: str(dead))
    root = tmp_path / ("py" + mine.replace(".", ""))
    _fake_python_tree(root, "python3", sys.executable)
    monkeypatch.setattr(le, "_provisioned_python_dir", lambda v: root)
    monkeypatch.setattr(le, "_uv", lambda: None)
    venv = tmp_path / "venv"
    le._create_venv("probe", venv)
    assert le._python_minor(le._venv_python(venv)) == mine


def test_the_error_names_every_route_that_failed(le, tmp_path, monkeypatch):
    sp = le.EngineSpec(name="probe", vendor="x", py_version="3.11",
                       requirements="r.txt", entrypoint="e", weights=(),
                       ready_markers=(), probe_imports=())
    monkeypatch.setitem(le.ENGINES, "probe", sp)
    monkeypatch.setattr(le.shutil, "which", lambda n: None)
    monkeypatch.setattr(le, "_standalone_python", lambda v: None)
    monkeypatch.setattr(le, "_uv", lambda: None)
    def _boom(v, log=None):
        raise RuntimeError("no network")
    monkeypatch.setattr(le, "_fetch_standalone_python", _boom)
    with pytest.raises(RuntimeError) as e:
        le._create_venv("probe", tmp_path / "venv")
    msg = str(e.value)
    assert "no usable python3.11" in msg
    assert "uv not found" in msg
    assert "no network" in msg


def test_fetching_is_the_last_route_not_the_first(le, tmp_path, monkeypatch):
    """Where uv is present it already manages interpreters; two copies of python
    3.11 on one box is not an improvement."""
    sp = le.EngineSpec(name="probe", vendor="x", py_version="3.11",
                       requirements="r.txt", entrypoint="e", weights=(),
                       ready_markers=(), probe_imports=())
    monkeypatch.setitem(le.ENGINES, "probe", sp)
    monkeypatch.setattr(le.shutil, "which", lambda n: None)
    monkeypatch.setattr(le, "_standalone_python", lambda v: None)
    seen = []
    fake_uv = tmp_path / "uv"
    fake_uv.write_text("#!/bin/sh\nexit 0\n")
    fake_uv.chmod(0o755)
    monkeypatch.setattr(le, "_uv", lambda: str(fake_uv))
    monkeypatch.setattr(le, "_fetch_standalone_python",
                        lambda *a, **k: seen.append("fetch") or pytest.fail("x"))
    le._create_venv("probe", tmp_path / "venv")
    assert seen == []


# ── repairing a venv built somewhere else ────────────────────────────────────

def test_a_venv_whose_interpreter_is_gone_is_repaired_not_abandoned(le, tmp_path, monkeypatch):
    """The host-then-container case: ~/.coderai/<engine>_venv is on a mount both
    see, but its bin/python points at the host's /usr/bin/python3.11, which the
    container has not got. `python -m venv` into that directory fails on the
    dangling symlink, so the engine could never be brought up without throwing
    away several GB of correctly installed wheels."""
    mine = f"{sys.version_info[0]}.{sys.version_info[1]}"
    sp = le.EngineSpec(name="probe", vendor="x", py_version=mine,
                       requirements="r.txt", entrypoint="e", weights=(),
                       ready_markers=(), probe_imports=())
    monkeypatch.setitem(le.ENGINES, "probe", sp)
    root = tmp_path / ("py" + mine.replace(".", ""))
    _fake_python_tree(root, "python3", sys.executable)
    monkeypatch.setattr(le, "_provisioned_python_dir", lambda v: root)
    monkeypatch.setattr(le.shutil, "which", lambda n: None)
    monkeypatch.setattr(le, "_uv", lambda: None)

    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / f"bin/python{mine}").symlink_to("/nowhere/python")
    (venv / "bin/python").symlink_to(f"python{mine}")
    site = venv / "lib" / f"python{mine}" / "site-packages"
    site.mkdir(parents=True)
    (site / "already_installed.py").write_text("# six gigabytes, pretend\n")

    le._create_venv("probe", venv)
    assert le._python_minor(le._venv_python(venv)) == mine
    assert (site / "already_installed.py").is_file(), \
        "the packages were thrown away with the dead symlinks"


def test_an_unrepairable_venv_is_rebuilt_from_scratch(le, tmp_path, monkeypatch):
    """A directory no interpreter will adopt is a corpse. Rebuilding costs a
    download; leaving it costs an engine that cannot run."""
    mine = f"{sys.version_info[0]}.{sys.version_info[1]}"
    sp = le.EngineSpec(name="probe", vendor="x", py_version=mine,
                       requirements="r.txt", entrypoint="e", weights=(),
                       ready_markers=(), probe_imports=())
    monkeypatch.setitem(le.ENGINES, "probe", sp)
    root = tmp_path / ("py" + mine.replace(".", ""))
    _fake_python_tree(root, "python3", sys.executable)
    monkeypatch.setattr(le, "_provisioned_python_dir", lambda v: root)
    monkeypatch.setattr(le.shutil, "which", lambda n: None)
    monkeypatch.setattr(le, "_uv", lambda: None)

    # No network in a test: the fetch route must not be the one that saves it.
    monkeypatch.setattr(le, "_fetch_standalone_python",
                        lambda *a, **k: pytest.fail("fetched despite a bundled python"))
    venv = tmp_path / "venv"
    venv.mkdir()
    # A file where venv wants a directory: creation fails whatever interpreter
    # is used, and pruning links cannot help.
    (venv / "bin").write_text("not a directory\n")

    le._create_venv("probe", venv)
    assert le._python_minor(le._venv_python(venv)) == mine


def test_the_rebuild_is_attempted_once(le, tmp_path, monkeypatch):
    """A second failure is about the host, not about the directory."""
    sp = le.EngineSpec(name="probe", vendor="x", py_version="3.11",
                       requirements="r.txt", entrypoint="e", weights=(),
                       ready_markers=(), probe_imports=())
    monkeypatch.setitem(le.ENGINES, "probe", sp)
    monkeypatch.setattr(le.shutil, "which", lambda n: None)
    monkeypatch.setattr(le, "_standalone_python", lambda v: None)
    monkeypatch.setattr(le, "_uv", lambda: None)
    tries = []

    def _boom(v, log=None):
        tries.append(v)
        raise RuntimeError("no network")
    monkeypatch.setattr(le, "_fetch_standalone_python", _boom)
    venv = tmp_path / "venv"
    venv.mkdir()
    with pytest.raises(RuntimeError):
        le._create_venv("probe", venv)
    assert len(tries) == 2, "the rebuild should be tried exactly once"


# ── what a render must not have to download ──────────────────────────────────

def test_the_engine_torch_cache_lives_with_the_engine(le):
    """Left at its default it is ~/.cache/torch — inside a container's writable
    layer, so the 86 MB detector MuseTalk pulls on import comes down again every
    time the container is recreated, and comes down during a render."""
    sp = le.spec("musetalk")
    env = le._engine_env(sp, Path("/engines/musetalk"))
    assert env["TORCH_HOME"] == "/engines/musetalk/.torch"


def test_everything_an_engine_fetches_by_url_is_warmed(le):
    """MuseTalk's s3fd is the one weight no WeightSet can place, because its
    vendored face_detection asks torch for it by URL at module import."""
    warm = {w.marker: w for w in le.spec("musetalk").warm}
    assert ".torch/hub/checkpoints/s3fd-619a316812.pth" in warm
    code = warm[".torch/hub/checkpoints/s3fd-619a316812.pth"].code
    assert "FaceAlignment" in code
    assert "cpu" in code, "a warm-up that needs the card is not a warm-up"


def test_latentsyncs_face_detector_is_installed_not_downloaded_at_render(le):
    """insightface downloads buffalo_l into checkpoints/auxiliary unless the
    directory is already there, and does it inside FaceAnalysis.__init__ — i.e.
    during a render."""
    sp = le.spec("latentsync")
    got = [p for ws in sp.weights for p in ws.patterns if "buffalo_l" in p]
    assert got, "buffalo_l is not part of the install"
    assert any("buffalo_l" in m for m in sp.ready_markers), \
        "an install without the detector would still count as complete"


def test_a_warm_engine_is_not_warmed_again(le, tmp_path, monkeypatch):
    sp = le.EngineSpec(name="probe", vendor="x", py_version="3.11",
                       requirements="r.txt", entrypoint="e", weights=(),
                       ready_markers=(), probe_imports=(),
                       warm=(le.WarmUp(what="a file", marker="w/f.pth",
                                       code="raise SystemExit(1)"),))
    monkeypatch.setitem(le.ENGINES, "probe", sp)
    monkeypatch.setattr(le, "install_dir", lambda n: tmp_path)
    (tmp_path / "w").mkdir()
    (tmp_path / "w" / "f.pth").write_text("already here")
    le._warm("probe", Path(sys.executable))


def test_a_warm_up_that_fetched_nothing_is_an_error_not_a_pass(le, tmp_path, monkeypatch):
    """Otherwise the failure is handed to the first render instead: a timeout in
    the middle of a compose pipeline rather than a message on the install."""
    sp = le.EngineSpec(name="probe", vendor="x", py_version="3.11",
                       requirements="r.txt", entrypoint="e", weights=(),
                       ready_markers=(), probe_imports=(),
                       warm=(le.WarmUp(what="a file", marker="w/f.pth",
                                       code="pass"),))
    monkeypatch.setitem(le.ENGINES, "probe", sp)
    monkeypatch.setattr(le, "install_dir", lambda n: tmp_path)
    with pytest.raises(RuntimeError) as e:
        le._warm("probe", Path(sys.executable))
    assert "would still download it" in str(e.value)


def test_a_warm_up_that_crashed_says_why(le, tmp_path, monkeypatch):
    sp = le.EngineSpec(name="probe", vendor="x", py_version="3.11",
                       requirements="r.txt", entrypoint="e", weights=(),
                       ready_markers=(), probe_imports=(),
                       warm=(le.WarmUp(what="a file", marker="w/f.pth",
                                       code="raise RuntimeError('no route to host')"),))
    monkeypatch.setitem(le.ENGINES, "probe", sp)
    monkeypatch.setattr(le, "install_dir", lambda n: tmp_path)
    with pytest.raises(RuntimeError) as e:
        le._warm("probe", Path(sys.executable))
    assert "no route to host" in str(e.value)
