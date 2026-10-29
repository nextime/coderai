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

"""XTTS builds its own venv, like parler does.

Both engines need an isolated interpreter for the same reason (a transformers
the server cannot have). parler bootstrapped its venv on first use; XTTS only
LOOKED for one and raised a 501 telling the operator to build it by hand. Same
situation, two different behaviours -- reported from production as "speech
engines listed but not installed".

Nothing here touches the network: the subprocess calls are stubbed, and what is
asserted is the decision-making around them.
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from codai.api import xtts_worker as X


@pytest.fixture(autouse=True)
def _no_real_subprocess(monkeypatch):
    calls = []

    def _fake_run(cmd, **kw):
        calls.append([str(c) for c in cmd])
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(X.subprocess, "run", _fake_run)
    monkeypatch.setattr(X, "_bootstrapped", False)
    return calls


def test_an_existing_venv_is_not_rebuilt(monkeypatch, _no_real_subprocess):
    monkeypatch.setattr(X, "available", lambda: True)
    assert X.bootstrap() is True
    assert _no_real_subprocess == [], "it reinstalled over a working venv"


def test_a_missing_venv_is_created_and_coqui_installed(monkeypatch, tmp_path,
                                                       _no_real_subprocess):
    seen = {"n": 0}

    def _avail():
        # Unavailable on the way in, available once the install has run.
        seen["n"] += 1
        return seen["n"] > 1 and len(_no_real_subprocess) >= 1

    monkeypatch.setattr(X, "available", _avail)
    # bootstrap finishes on health(), which asks the venv itself; there is no
    # venv here, so it is stubbed alongside available().
    monkeypatch.setattr(X, "health", lambda: (_avail(), ""))
    monkeypatch.setattr(X, "venv_dir", lambda: tmp_path / "xtts_venv")
    monkeypatch.setattr(X, "venv_python",
                        lambda: tmp_path / "xtts_venv" / "bin" / "python")
    assert X.bootstrap() is True
    flat = " ".join(" ".join(c) for c in _no_real_subprocess)
    assert "-m venv" in flat or "venv" in flat
    assert ("requirements-xtts.txt" in flat or "coqui-tts" in flat), (
        f"coqui-tts was never installed: {flat}")


def test_a_failed_install_does_not_raise(monkeypatch, tmp_path):
    """The caller's fallback is the same 501 it would have raised anyway, and
    one engine's bad install must not break a request for another."""
    def _boom(cmd, **kw):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(X.subprocess, "run", _boom)
    monkeypatch.setattr(X, "available", lambda: False)
    monkeypatch.setattr(X, "venv_dir", lambda: tmp_path / "v")
    monkeypatch.setattr(X, "venv_python", lambda: tmp_path / "v" / "bin" / "python")
    assert X.bootstrap() is False          # must not raise


def test_an_install_that_yields_no_import_reports_failure(monkeypatch, tmp_path,
                                                          _no_real_subprocess):
    """A partially built venv is worse than none: an interrupted install on a
    sibling worker left 42 packages and no torch, and the engine reported
    "venv can't be created" from then on."""
    monkeypatch.setattr(X, "available", lambda: False)
    monkeypatch.setattr(X, "venv_dir", lambda: tmp_path / "v")
    monkeypatch.setattr(X, "venv_python", lambda: tmp_path / "v" / "bin" / "python")
    assert X.bootstrap() is False


def test_the_backend_asks_for_a_bootstrap_and_can_be_opted_out():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(root, "codai/api/tts_backends.py"),
               encoding="utf-8").read()
    assert "xtts_worker.bootstrap()" in src, (
        "the coqui backend no longer builds the venv on first use")
    assert "CODERAI_XTTS_AUTOINSTALL" in src, (
        "a host with no network cannot opt out of the auto-install")


# --------------------------------------------------------------------------
# Installed is not usable
# --------------------------------------------------------------------------
#
# Measured on zeiss, 2026-10-29: the venv was rebuilt, `pip install coqui-tts
# soundfile requests` installed coqui-tts 0.28.0 and its ninety dependencies,
# and NO torch at all. The install reported success, `import TTS` failed, and
# the engine said "coqui-tts installed but still not importable" — naming
# neither the module nor the reason.


def test_the_install_is_pinned_not_resolved():
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(root, "codai/api/xtts_worker.py"),
               encoding="utf-8").read()
    assert "pip_install_args(" in src, "the install is whatever pip resolves"
    assert any(p.startswith("torch==") for p in X._PINS), (
        "torch is not pinned, and a coqui-tts install brings none")
    req = open(os.path.join(root, "requirements-xtts.txt"), encoding="utf-8").read()
    assert "torch==" in req and "torchaudio==" in req, (
        "the pair has to be pinned together")


def test_the_inline_pins_match_the_requirements_file():
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wanted = []
    for line in open(os.path.join(root, "requirements-xtts.txt"), encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        wanted.extend(line.split(" ", 1) if line.startswith("--") else [line])
    assert list(X._PINS) == wanted, (
        f"the inline pins and requirements-xtts.txt disagree:\n"
        f"  file:   {wanted}\n  inline: {list(X._PINS)}")


def test_health_answers_with_a_reason():
    ok, why = X.health()
    assert isinstance(ok, bool)
    if not ok:
        assert why, "an unusable venv must say what is wrong with it"


def test_the_engine_error_carries_that_reason():
    """It used to say "no venv containing it was found" over a venv that was
    there and populated, which sends the operator looking in the wrong place."""
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(root, "codai/api/tts_backends.py"),
               encoding="utf-8").read()
    assert "xtts_worker.health()" in src, (
        "the engine does not ask the worker what is wrong")
    err = src.split("raise MissingEngineError", 1)[1][:600]
    assert "{_why}" in err, "the reason is not carried into the error"
