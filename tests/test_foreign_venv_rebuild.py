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

"""A venv that cannot be used has to be rebuilt, not reused.

An isolated-worker venv keeps ``bin/python`` as a symlink to the interpreter
that created it. Create it with one python and run it with another -- built on
the host and used from the container, or built before the image moved its
interpreter -- and that link dangles.

The failure is not a clean "missing": ``Path.exists()`` follows symlinks and
answers False, so the code decides the venv is absent and runs ``python -m
venv`` over the top, which trips on the very same dangling ``bin/python3``:

    Error: [Errno 2] No such file or directory: '.../parler_venv/bin/python3'

Reported from production as "the venv can't be created", on a host where the
venv was fully populated and 7 GB of packages were sitting in it.

A second shape of the same thing, measured on zeiss 2026-10-29: an
``xtts_venv`` whose ``bin/python3`` pointed at ``/usr/bin/python3``. In the
container that is 3.12, while the venv's 875 MB of packages were installed
under ``lib/python3.13/site-packages``. The interpreter runs perfectly and
imports none of them, so "does it run" is not the whole question.

Both decisions live in one place now -- :mod:`codai.api.venv_env` -- because
every isolated worker had its own copy and they did not agree.
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from codai.api import venv_env


def _make_foreign_venv(root, pkg_version="3.13"):
    """A venv whose interpreter symlink points at a path that is not there."""
    (root / "bin").mkdir(parents=True)
    (root / "lib" / f"python{pkg_version}" / "site-packages").mkdir(parents=True)
    os.symlink("python3.13", root / "bin" / "python")
    os.symlink("/nonexistent/python3.13", root / "bin" / "python3.13")
    (root / "pyvenv.cfg").write_text(
        "home = /nonexistent/bin\ninclude-system-site-packages = false\n")
    return root


def test_the_dangling_interpreter_is_not_mistaken_for_a_working_one(tmp_path):
    v = _make_foreign_venv(tmp_path / "parler_venv")
    py = venv_env.venv_python(v)
    assert py.is_symlink()
    assert not py.exists(), "precondition: the link dangles"
    ok, why = venv_env.venv_usable(v)
    assert ok is False and "does not run" in why


def test_a_real_venv_is_accepted(tmp_path):
    """Built here with the running interpreter, so it is the real thing."""
    v = tmp_path / "ok_venv"
    venv_env.create_venv(v)
    ok, why = venv_env.venv_usable(v)
    assert ok is True, why


def test_a_version_mismatch_is_rejected_even_though_python_runs(tmp_path):
    """The xtts_venv case: the interpreter works and its packages are
    invisible to it, which no "does it run" check can see."""
    v = tmp_path / "mixed_venv"
    venv_env.create_venv(v)
    import shutil
    real = [p for p in (v / "lib").glob("python*")][0]
    shutil.move(str(real), str(v / "lib" / "python2.7"))
    ok, why = venv_env.venv_usable(v)
    assert ok is False
    assert "packages are installed for" in why, why


def test_a_system_site_venv_is_rejected(tmp_path):
    """It defeats the only reason these venvs exist: the server's
    incompatible stack leaks straight back in."""
    v = tmp_path / "shared_venv"
    venv_env.create_venv(v, system_site=True)
    ok, why = venv_env.venv_usable(v)
    assert ok is False and "system-site-packages" in why


def test_a_foreign_venv_is_cleared_before_recreating(tmp_path):
    """Without the clear, `python -m venv` dies on the dangling link and the
    engine reports "venv can't be created" on every request, for ever."""
    v = _make_foreign_venv(tmp_path / "parler_venv")
    py = venv_env.create_venv(v)
    assert py.is_file() and py.exists()
    ok, why = venv_env.venv_usable(v)
    assert ok is True, why


def test_the_child_environment_drops_what_redirects_an_interpreter(monkeypatch):
    """PYTHONHOME is exported by the engine process. Inherited by a venv
    python it takes stdlib and site-packages from the SERVER's installation,
    so the venv's own packages are not on sys.path at all."""
    monkeypatch.setenv("PYTHONHOME", "/opt/coderai/python")
    monkeypatch.setenv("PYTHONPATH", "/opt/coderai/app")
    monkeypatch.setenv("VIRTUAL_ENV", "/somewhere/else")
    monkeypatch.setenv("KEEP_ME", "yes")
    env = venv_env.clean_py_env()
    assert "PYTHONHOME" not in env
    assert "PYTHONPATH" not in env
    assert "VIRTUAL_ENV" not in env
    assert env["KEEP_ME"] == "yes", "unrelated variables must survive"


def test_the_servers_own_libraries_come_off_the_library_path(monkeypatch):
    """A worker resolving libraries through the server's directories picks up
    the wrong torch/cudnn — the CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED shape."""
    monkeypatch.setenv("LD_LIBRARY_PATH",
                       "/opt/coderai/local-libs:/opt/nvidia/lib:/opt/coderai/python/lib")
    env = venv_env.clean_py_env()
    assert env["LD_LIBRARY_PATH"] == "/opt/nvidia/lib", (
        "the driver's path must stay and coderai's must go")


def test_a_system_prefix_does_not_make_the_whole_system_ours(monkeypatch):
    """Running from a distribution python (a pip install rather than the
    image) puts sys.prefix at /usr. Treating that as the server's install
    strips /usr/local/nvidia/lib from every worker and takes the GPU away."""
    monkeypatch.setattr(venv_env.sys, "prefix", "/usr")
    monkeypatch.setattr(venv_env.sys, "base_prefix", "/usr")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/local/nvidia/lib:/opt/coderai/local-libs")
    assert venv_env.clean_py_env()["LD_LIBRARY_PATH"] == "/usr/local/nvidia/lib"


def test_a_caller_can_still_set_what_it_needs(monkeypatch):
    monkeypatch.setenv("PYTHONHOME", "/opt/coderai/python")
    env = venv_env.clean_py_env_with({"PYTHONUNBUFFERED": "1",
                                      "CUDA_VISIBLE_DEVICES": 1})
    assert env["PYTHONUNBUFFERED"] == "1"
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert "PYTHONHOME" not in env


def test_every_isolated_worker_goes_through_the_shared_helper():
    """The trap is identical in each one, and each had its own copy of the
    decision. A worker that builds its own environment dict is a worker that
    will inherit PYTHONHOME again."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    workers = ("xtts_worker", "parler_worker", "crisperwhisper_worker",
               "melotts_worker", "canary_worker", "pyannote_worker",
               "h3_worker", "audiocraft_worker")
    for name in workers:
        src = open(os.path.join(root, "codai/api", name + ".py"),
                   encoding="utf-8").read()
        assert "venv_env" in src, f"{name} does not use the shared helper"
        assert "dict(os.environ)" not in src, (
            f"{name} builds its own environment, so it inherits PYTHONHOME")
        assert "{**os.environ" not in src, (
            f"{name} builds its own environment, so it inherits PYTHONHOME")


def test_no_isolated_worker_launches_its_venv_with_a_bare_subprocess():
    """Every call INTO a venv needs the clean environment: the import probe,
    `python -m venv`, each pip install and the service itself. A clean service
    with a poisoned probe still reports the venv as broken, and a poisoned pip
    installs into the wrong tree."""
    import re
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    workers = ("xtts_worker", "parler_worker", "crisperwhisper_worker",
               "melotts_worker", "canary_worker", "pyannote_worker",
               "h3_worker", "audiocraft_worker")
    offenders = []
    for name in workers:
        path = os.path.join(root, "codai/api", name + ".py")
        src = open(path, encoding="utf-8").read()
        for m in re.finditer(r"subprocess\.(run|Popen)\(", src):
            # Find the call's argument text, up to the matching blank line, and
            # require that an env was passed.
            chunk = src[m.start():m.start() + 900]
            if "env=" not in chunk.split("\n\n")[0]:
                line = src[:m.start()].count("\n") + 1
                offenders.append(f"{name}.py:{line}")
    assert not offenders, (
        "these calls into a venv inherit the engine's environment: "
        + ", ".join(offenders))


# --------------------------------------------------------------------------
# Importable is not the same as working
# --------------------------------------------------------------------------
#
# Measured on zeiss, 2026-10-29, right after the environment fix above: the
# parler venv built, the service started, the model loaded, and the first
# synthesis died in the decoder's conv1d with
# CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED -- a CUDA-13 torch that pip had
# resolved on its own, with every cudnn sublibrary present and unloadable.
# `import parler_tts, soundfile` answered True for that venv for ever, so
# nothing ever rebuilt it.


def test_the_parler_venv_is_probed_on_the_gpu_not_just_imported():
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(root, "codai/api/parler_worker.py"),
               encoding="utf-8").read()
    assert "_gpu_ok(" in src, "the venv is trusted on an import alone"
    boot = src.split("def _bootstrap_venv", 1)[1]
    assert boot.index("_gpu_ok(py)") < boot.index("rmtree"), (
        "the probe does not drive the rebuild")
    assert "requirements-parler.txt" in src or "_REQUIREMENTS" in boot, (
        "the rebuild installs whatever pip resolves today, which is the bug")


def test_the_probe_runs_the_op_that_fails():
    """The decoder is convolutional, and a conv1d is what died with
    CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED. A probe that only imports torch
    would have passed on that venv."""
    probe = venv_env._GPU_PROBE
    assert "conv1d" in probe


def test_a_cpu_only_host_passes_the_probe():
    """There is no GPU stack to be broken, so the probe must not fail the
    bootstrap on a CPU-only machine."""
    probe = venv_env._GPU_PROBE
    assert "is_available" in probe and "SystemExit(0)" in probe


def test_the_checks_are_shared_not_copied():
    """Two workers had their own "is this venv healthy" and they did not agree
    -- which is how one of them trusted an import alone."""
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ("parler_worker", "xtts_worker"):
        src = open(os.path.join(root, "codai/api", name + ".py"),
                   encoding="utf-8").read()
        assert "venv_env.import_ok(" in src, f"{name} has its own import check"
        assert "venv_env.gpu_conv_ok(" in src, f"{name} has no GPU probe"


def test_the_parler_requirements_pin_torch():
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    req = open(os.path.join(root, "requirements-parler.txt"), encoding="utf-8").read()
    assert "torch==" in req, "torch is unpinned, so pip picks the stack again"
    assert "parler-tts" in req


def test_the_service_survives_a_cudnn_that_cannot_load():
    """Turning cudnn off costs a little speed on a decoder that is a fraction
    of the work. The alternative is no speech at all."""
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(root, "tools/parler_tts_service.py"),
               encoding="utf-8").read()
    body = src.split("def speak", 1)[1]
    assert "cudnn.enabled = False" in body
    assert "not torch.backends.cudnn.enabled" in body, (
        "a second failure would retry for ever instead of raising")


def test_the_inline_pins_match_the_requirements_file():
    """Two copies of a pin drift. The file is the source of truth; the inline
    list exists only for an install where that file is not present, and it has
    to say the same thing."""
    import os
    import re
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    from codai.api import parler_worker as P

    wanted = []
    for line in open(os.path.join(root, "requirements-parler.txt"),
                     encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        wanted.extend(line.split(" ", 1) if line.startswith("--") else [line])
    assert list(P._PINS) == wanted, (
        f"the inline pins and requirements-parler.txt disagree:\n"
        f"  file:   {wanted}\n  inline: {list(P._PINS)}")


def test_a_missing_requirements_file_does_not_block_a_rebuild():
    """The container mounts only codai/, tools/ and third_party/ over the
    image, so a repo-root file added since the image was built is absent until
    the container is recreated. A self-healing rebuild must not stop for a
    reason that has nothing to do with the venv -- which is exactly what
    happened on the first live run, and what ACE-Step's "requirements missing"
    report turned out to be."""
    from pathlib import Path
    from codai.api import parler_worker as P

    args = venv_env.pip_install_args(Path("/nope/requirements-parler.txt"), P._PINS)
    assert "-r" not in args
    assert any(a.startswith("torch==") for a in args)

    here = Path(__file__).resolve().parents[1] / "requirements-parler.txt"
    assert venv_env.pip_install_args(here, P._PINS) == ["-r", str(here)], (
        "the file must be preferred whenever it is there")


def test_the_worker_no_longer_refuses_without_the_file():
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(root, "codai/api/parler_worker.py"),
               encoding="utf-8").read()
    assert "requirements file missing" not in src
    assert "pip_install_args(" in src
