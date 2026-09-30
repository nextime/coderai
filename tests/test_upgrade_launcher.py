"""The in-image upgrader (`packaging/linux/launcher/coderai-upgrade`).

`coderai-docker --upgrade` is how a running install gets new code without a
rebuild, and it broke for everyone when one git host's certificate expired: the
default remote was unreachable and the script simply died, although the same
branch sits on a public mirror. These tests run the real script against local
git repositories — no network — and pin the behaviour that matters:

* a healthy default remote is used and nothing else is contacted;
* an unreachable default falls back to the mirror and says so;
* a remote the operator named explicitly is never silently replaced;
* a ref that does not exist is an error, not something to paper over with the
  mirror (that would upgrade to the wrong code).
"""

import os
import shutil
import subprocess
import time
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "packaging" / "linux" / "launcher" / "coderai-upgrade"

pytestmark = pytest.mark.skipif(shutil.which("git") is None or not SCRIPT.is_file(),
                                reason="needs git and the upgrader script")


def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=str(cwd), check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _repo(path: Path, version: str, branch: str = "production") -> str:
    """A minimal coderai tree as a git repo, on `branch`."""
    path.mkdir(parents=True, exist_ok=True)
    _git("init", "-q", "-b", branch, cwd=path)
    _git("config", "user.email", "t@t", cwd=path)
    _git("config", "user.name", "t", cwd=path)
    pkg = path / "codai"
    pkg.mkdir(exist_ok=True)
    (pkg / "__init__.py").write_text(f'__version__ = "{version}"\n')
    (path / "requirements.txt").write_text("fastapi\n")
    _git("add", "-A", cwd=path)
    _git("commit", "-q", "-m", "x", cwd=path)
    return str(path)


def _app(path: Path, version: str) -> Path:
    """A fake installed app tree the script upgrades in place.

    The files are back-dated a day: the upgrade replaces the tree with rsync,
    whose quick check is size + mtime, and two version strings of the same length
    written in the same second look identical to it. In a real image the app tree
    carries the build's timestamp, which is what this reproduces."""
    (path / "codai").mkdir(parents=True, exist_ok=True)
    (path / "codai" / "__init__.py").write_text(f'__version__ = "{version}"\n')
    (path / "requirements.txt").write_text("fastapi\n")
    old = time.time() - 86400
    for f in (path, path / "codai", path / "codai" / "__init__.py",
              path / "requirements.txt"):
        os.utime(f, (old, old))
    return path


def _run(tmp_path, app: Path, *, default_repo: str, mirror: str,
         explicit: str = "", ref: str = "production", ssh_key: str = ""):
    """Run the script with its two defaults rewritten to local repos."""
    src = SCRIPT.read_text()
    src = src.replace('APP_DIR="/opt/coderai/app"', f'APP_DIR="{app}"')
    src = src.replace('DEFAULT_HTTPS_REPO="https://git.nexlab.net/nexlab/coderai.git"',
                      f'DEFAULT_HTTPS_REPO="{default_repo}"')
    src = src.replace('MIRROR_REPO="https://github.com/nextime/coderai.git"',
                      f'MIRROR_REPO="{mirror}"')
    src = src.replace('PYBIN="/opt/coderai/python/bin/python3"',
                      f'PYBIN="{sys.executable}"')
    script = tmp_path / "upgrade.sh"
    script.write_text(src)
    script.chmod(0o755)
    env = dict(os.environ, CODERAI_UPGRADE_REF=ref, CODERAI_UPGRADE_SKIP_PIP="1")
    if explicit:
        env["CODERAI_UPGRADE_REPO"] = explicit
    if ssh_key:
        env["CODERAI_UPGRADE_SSH_KEY"] = ssh_key
    return subprocess.run([str(script)], capture_output=True, text=True, env=env,
                          timeout=180)


def test_the_default_remote_is_used_when_it_works(tmp_path):
    app = _app(tmp_path / "app", "0.2.21")
    good = _repo(tmp_path / "default", "0.2.22")
    mirror = _repo(tmp_path / "mirror", "9.9.9")        # must NOT be used
    r = _run(tmp_path, app, default_repo=good, mirror=mirror)
    assert r.returncode == 0, r.stderr
    assert "source used:       " + good in r.stderr
    assert "mirror" not in r.stderr
    assert '"0.2.22"' in (app / "codai" / "__init__.py").read_text()


def test_an_unreachable_default_falls_back_to_the_mirror(tmp_path):
    app = _app(tmp_path / "app", "0.2.21")
    mirror = _repo(tmp_path / "mirror", "0.2.22")
    # A remote that fails the way an expired certificate fails: cannot connect.
    dead = "https://127.0.0.1:1/nexlab/coderai.git"
    r = _run(tmp_path, app, default_repo=dead, mirror=mirror)
    assert r.returncode == 0, r.stderr
    assert "cannot reach " + dead in r.stderr
    assert "falling back to the public mirror" in r.stderr
    assert "source used:       " + mirror in r.stderr
    assert '"0.2.22"' in (app / "codai" / "__init__.py").read_text()


def test_a_repo_named_by_the_operator_is_never_replaced(tmp_path):
    app = _app(tmp_path / "app", "0.2.21")
    mirror = _repo(tmp_path / "mirror", "0.2.22")
    r = _run(tmp_path, app, default_repo=_repo(tmp_path / "d", "0.2.22"),
             mirror=mirror, explicit="https://127.0.0.1:1/mine.git")
    assert r.returncode != 0
    assert "falling back" not in r.stderr
    assert "clone of https://127.0.0.1:1/mine.git failed" in r.stderr
    assert '"0.2.21"' in (app / "codai" / "__init__.py").read_text()   # untouched


def test_a_missing_ref_is_an_error_not_a_reason_to_try_the_mirror(tmp_path):
    app = _app(tmp_path / "app", "0.2.21")
    good = _repo(tmp_path / "default", "0.2.22")
    mirror = _repo(tmp_path / "mirror", "0.2.22")
    r = _run(tmp_path, app, default_repo=good, mirror=mirror, ref="no-such-branch")
    assert r.returncode != 0
    assert "falling back" not in r.stderr
    assert "no-such-branch" in r.stderr
    assert '"0.2.21"' in (app / "codai" / "__init__.py").read_text()


def test_an_up_to_date_install_exits_10_and_changes_nothing(tmp_path):
    app = _app(tmp_path / "app", "0.2.22")
    good = _repo(tmp_path / "default", "0.2.22")
    r = _run(tmp_path, app, default_repo=good, mirror=_repo(tmp_path / "m", "0.2.22"))
    assert r.returncode == 10 and "already up to date" in r.stderr
