"""An admin act must be reachable without a browser.

`/admin/api/system/reload` re-reads config.json and models.json in place and
takes an admin SESSION -- correctly, because re-reading the server's
configuration is an admin act. But a session could only be minted by the
browser login form, so on 2026-11-09 a change to `allow_client_scale` sat
unapplied on disk while the client collected 403s, and the only alternative was
restarting the orchestrator: 16 queued extraction requests dropped and the pod
booting for them stranded.

`--admin-session` mints one locally. It needs write access to auth.json, and
anyone with that can already add themselves an admin user, so it grants no
privilege that was not already there.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _auth(tmp_path, users):
    (tmp_path / "auth.json").write_text(json.dumps(
        {"users": users, "tokens": [], "sessions": {}}))
    return tmp_path


def _mint(config_dir, user=""):
    from codai.main import _mint_admin_session
    return _mint_admin_session(str(config_dir), user)


def test_the_cookie_it_prints_validates_as_that_admin(tmp_path, capsys):
    cfg = _auth(tmp_path, [{"id": 1, "username": "admin", "role": "admin",
                            "password_hash": "x"}])
    assert _mint(cfg) == 0
    cookie = capsys.readouterr().out.strip()
    assert cookie, "the cookie must go to stdout so $(...) can capture it"

    from codai.admin.auth import SessionManager
    assert SessionManager(Path(cfg)).validate_session(cookie) == "admin"


def test_only_the_cookie_goes_to_stdout(tmp_path, capsys):
    """Everything explanatory on stderr, so the value can be captured clean."""
    cfg = _auth(tmp_path, [{"id": 1, "username": "admin", "role": "admin",
                            "password_hash": "x"}])
    _mint(cfg)
    cap = capsys.readouterr()
    assert len(cap.out.strip().splitlines()) == 1
    assert "admin session for" in cap.err


def test_a_named_admin_can_be_picked(tmp_path, capsys):
    cfg = _auth(tmp_path, [
        {"id": 1, "username": "first", "role": "admin", "password_hash": "x"},
        {"id": 2, "username": "second", "role": "admin", "password_hash": "x"},
    ])
    assert _mint(cfg, "second") == 0
    cookie = capsys.readouterr().out.strip()
    from codai.admin.auth import SessionManager
    assert SessionManager(Path(cfg)).validate_session(cookie) == "second"


def test_a_non_admin_user_is_refused(tmp_path, capsys):
    """And the message distinguishes 'not an admin' from 'no such user',
    because the operator's next step differs."""
    cfg = _auth(tmp_path, [{"id": 1, "username": "bob", "role": "user",
                            "password_hash": "x"}])
    assert _mint(cfg, "bob") == 3
    assert "not an admin" in capsys.readouterr().err

    assert _mint(cfg, "nobody") == 3
    assert "not in auth.json" in capsys.readouterr().err


def test_an_install_with_no_admin_is_refused(tmp_path, capsys):
    cfg = _auth(tmp_path, [{"id": 1, "username": "bob", "role": "user",
                            "password_hash": "x"}])
    assert _mint(cfg) == 3
    assert "no admin user" in capsys.readouterr().err


def test_the_wrong_config_dir_says_so(tmp_path, capsys):
    """The two-config trap: naming the wrong dir must not look like a crash."""
    assert _mint(tmp_path / "nowhere") == 2
    assert "no auth.json" in capsys.readouterr().err


def test_it_never_starts_a_server(tmp_path):
    """A one-shot utility: handled before anything heavy is initialised."""
    src = (ROOT / "codai/main.py").read_text()
    head = src.split("def main():")[1]
    before = head.split("_mint_admin_session")[0]
    assert "ConfigManager(config_dir)" not in before, (
        "the mint must be handled before the config manager is built")


def test_the_flag_is_declared_with_an_optional_user(tmp_path):
    src = (ROOT / "codai/cli.py").read_text()
    assert '"--admin-session"' in src
    body = src.split('"--admin-session"')[1][:400]
    assert 'nargs="?"' in body, "a bare --admin-session must work"
    assert 'default=None' in body, "absent must be distinguishable from bare"


# --------------------------------------------------------------------------- #
# The endpoint the cookie exists to reach (added after it 500'd in production)
# --------------------------------------------------------------------------- #
# Measured 2026-11-09 against the live install, with a freshly minted cookie:
#
#   POST /admin/api/system/reload -> HTTP 500
#   AttributeError: 'ModelsConfig' object has no attribute 'loaded'
#
# The reload itself had already succeeded -- `mgr.reload()` runs before the
# response is built -- so the config WAS applied and the caller was told 500.
# Verified by the behaviour change it carried: POST /v1/runpod/scale for
# digesta-extraction went from 403 to 200 across that same failed call.

def test_the_reload_response_reads_only_fields_that_exist():
    """The second time this endpoint shipped broken, so pin the shape.

    `loaded` and `preload` were never on ModelsConfig. A dataclass field that
    is renamed must not be able to 500 a reload that has already happened.
    """
    from dataclasses import fields
    from codai.config import ModelsConfig

    src = (ROOT / "codai/admin/routes.py").read_text()
    body = src.split('@router.post("/admin/api/system/reload"')[1]
    body = body.split("@router.")[0]
    # CODE only: the comment there quotes the broken names on purpose, and a
    # test that cannot tell prose from code would forbid documenting the bug.
    code = "\n".join(l for l in body.splitlines()
                     if not l.lstrip().startswith("#"))
    assert "cfg.models.loaded" not in code
    assert "cfg.models.preload" not in code

    names = {f.name for f in fields(ModelsConfig)}
    assert "default_load_mode" in names
    assert "loaded" not in names and "preload" not in names
    # every ModelsConfig attribute the response names must be a real field
    import re
    for attr in set(re.findall(r'getattr\(cfg\.models,\s*"([a-z_]+)"', code)):
        assert attr in names, "%s is not a ModelsConfig field" % attr


def test_the_reload_cannot_raise_after_reloading(monkeypatch):
    """Calling it with a stub manager must return, not raise.

    The failure mode that matters is exactly this: the work is done, then the
    reply explodes and the operator concludes the mechanism is broken.
    """
    import asyncio
    import types as _t
    from codai.admin import routes as r
    from codai.config import ModelsConfig

    cfg = _t.SimpleNamespace(models=ModelsConfig())
    reloaded = []
    mgr = _t.SimpleNamespace(
        reload=lambda: (reloaded.append(1), cfg)[1],
        models_data={"text_models": [{"id": "a"}, {"id": "b"}],
                     "vision_models": [{"id": "c"}]})
    monkeypatch.setattr(r, "config_manager", mgr, raising=False)
    req = _t.SimpleNamespace(app=_t.SimpleNamespace(state=_t.SimpleNamespace()))

    out = asyncio.run(r.api_reload_config.__wrapped__(req, username="admin")
                      if hasattr(r.api_reload_config, "__wrapped__")
                      else r.api_reload_config(req, username="admin"))
    assert reloaded == [1], "it must actually reload"
    assert out["success"] is True
    assert out["models_reloaded"] == 3
    assert out["config"]["load_mode"] == ModelsConfig().default_load_mode
