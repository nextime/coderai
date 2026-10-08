"""Trusted forward-auth for the admin GUI (SSO), per docs/admin-forward-auth.md.

The security contract is the whole point: a trusted header is a forged-identity
hole the moment it is honoured from an untrusted source. Every test here is
about refusing, except the two that prove it works when it should.
"""
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
UI = (ROOT / "codai/frontproxy/ui_pages.py").read_text()


def _cfg(**over):
    from codai.config import AdminForwardAuthConfig
    c = AdminForwardAuthConfig()
    for k, v in over.items():
        setattr(c, k, v)
    return c


# ------------------------------------------------------------------ the switch
def test_off_by_default():
    c = _cfg()
    assert c.enabled is False
    assert c.keep_local_login is False
    assert c.admin_without_groups is False     # never guess admin
    assert c.auto_create_users is True
    assert c.trusted_proxies == ["127.0.0.1", "::1"]
    assert c.shared_secret == ""


def test_enabled_with_no_secret_stays_off():
    """Enabled without a secret is not a configuration, it is an open door."""
    blk = UI.split("def _fa_cfg")[1].split("def _fa_trusted")[0]
    assert "if not secret:" in blk
    tail = blk.split("if not secret:")[1]
    assert "staying OFF" in tail
    assert "return None" in tail, "it must refuse to enable, not carry on"


def test_an_env_switch_exists_for_the_container():
    blk = UI.split("def _fa_cfg")[1].split("def _fa_trusted")[0]
    assert "CODERAI_ADMIN_FORWARD_AUTH" in blk
    assert "CODERAI_ADMIN_FORWARD_AUTH_SECRET" in blk
    # Env must be able to turn it OFF as well as on, or a unit file cannot
    # disable a feature the mounted config enables.
    assert '"1", "true", "yes", "on"' in blk


def test_the_config_round_trips(tmp_path):
    import json
    from codai.config import ConfigManager
    (tmp_path / "config.json").write_text(json.dumps(
        {"admin": {"forward_auth": {"enabled": True, "shared_secret": "s3cr3t",
                                    "trusted_proxies": ["10.0.0.9"],
                                    "keep_local_login": True}}}))
    cm = ConfigManager(str(tmp_path)); cm.load()
    fa = cm.config.admin.forward_auth
    assert fa.enabled and fa.shared_secret == "s3cr3t"
    assert fa.trusted_proxies == ["10.0.0.9"] and fa.keep_local_login is True
    cm.save_config()
    back = json.loads((tmp_path / "config.json").read_text())["admin"]["forward_auth"]
    assert back["enabled"] is True and back["shared_secret"] == "s3cr3t"


# --------------------------------------------------------- the trust contract
def test_both_checks_are_required_not_either():
    blk = UI.split("def _fa_trusted")[1].split("def _fa_identity")[0]
    assert "peer not in trusted" in blk
    assert "compare_digest" in blk, "the secret must not be compared with =="
    # The peer check and the secret check are separate early returns, so neither
    # alone can authenticate. Each refusal goes through _fa_reject, which returns
    # False after saying why.
    assert blk.count("_fa_reject(") >= 2


def test_the_header_is_ignored_entirely_when_untrusted():
    """A client sending X-Forwarded-User: admin directly must get nothing."""
    blk = UI.split("def _fa_identity")[1].split("def _fa_login")[0]
    i = blk.index("_fa_trusted")
    assert "return None, False" in blk[i:i + 160], \
        "the identity must be read only AFTER the trust check passes"
    assert blk.index("_fa_trusted") < blk.index("user_header"), \
        "trust is checked before the username is even looked at"


def test_a_username_from_a_header_is_validated():
    blk = UI.split("def _fa_identity")[1].split("def _fa_login")[0]
    assert '"/" in user' in blk and "len(user) > 128" in blk


def test_admin_is_not_granted_without_an_explicit_decision():
    blk = UI.split("def _fa_identity")[1].split("def _fa_login")[0]
    assert "admin_without_groups" in blk
    assert "admin_group" in blk


def test_an_unknown_user_is_refused_when_auto_create_is_off():
    blk = UI.split("def _fa_login")[1].split("def _user")[0]
    assert "auto_create_users" in blk
    assert "refusing unknown user" in blk


def test_the_created_account_has_an_unusable_password():
    """The account exists only to be reached through the proxy."""
    blk = UI.split("def _fa_login")[1].split("def _user")[0]
    assert "secrets.token_urlsafe(32)" in blk


# ------------------------------------------------------------ what it touches
def test_the_api_is_never_affected():
    """GUI-only. /v1/* keeps its bearer tokens, whatever the proxy asserts."""
    blk = UI.split("async def _forward_auth_session")[1].split("# ---")[0]
    assert 'path.startswith("/v1/")' in blk
    assert "return await call_next(request)" in blk.split('path.startswith("/v1/")')[1][:200]


def test_probes_are_not_touched():
    blk = UI.split("async def _forward_auth_session")[1].split("# ---")[0]
    assert '"/healthz", "/health"' in blk


def test_it_mints_a_real_session_so_the_dashboards_data_calls_work():
    """The pages' own /admin/api/* calls authenticate by session cookie — a
    rendered page with 401ing AJAX is not a working GUI."""
    blk = UI.split("async def _forward_auth_session")[1].split("# ---")[0]
    assert "sm.create_session(user)" in blk
    assert 'request.scope["headers"] = raw' in blk, "the cookie must reach this request"
    assert "response.set_cookie" in blk, "and the browser must keep it"
    assert "MUST_CHANGE" in blk, "an SSO account must not be asked to change a password"


def test_keep_local_login_false_refuses_the_password_form():
    blk = UI.split("async def _login_page")[1].split("@app.get")[0]
    assert "keep_local_login" in blk and "status_code=403" in blk


def test_logout_goes_upstream_when_configured():
    """Returning to our own /login would be met by the proxy's headers and log
    the user straight back in."""
    src = (ROOT / "codai/admin/routes.py").read_text()
    blk = src.split("async def logout")[1].split("@router.post")[0]
    assert "post_logout_url" in blk
    assert 'response.delete_cookie("session")' in blk


# ------------------------------------------------------- switchable from the UI
def test_the_settings_api_round_trips_it():
    from codai.admin.routes import build_settings_dict
    from codai.config import Config
    d = build_settings_dict(Config(), [])
    fa = d["admin"]["forward_auth"]
    assert fa["enabled"] is False
    # The secret is reported as a boolean, never echoed to every browser that
    # opens Settings.
    assert "shared_secret" not in fa
    assert fa["shared_secret_set"] is False


def test_saving_it_warns_instead_of_silently_doing_nothing():
    src = (ROOT / "codai/admin/routes.py").read_text()
    blk = src.split('if "admin" in data')[1].split('if "server" in data')[0]
    assert "_settings_warnings.append" in blk
    assert "no shared secret" in blk and "no trusted proxies" in blk
    # A blank secret in the payload must not wipe the stored one.
    assert 'fa_in["shared_secret"].strip()' in blk


@pytest.mark.parametrize("field", [
    "s-fa-enabled", "s-fa-user-header", "s-fa-groups-header", "s-fa-admin-group",
    "s-fa-secret-header", "s-fa-secret", "s-fa-trusted", "s-fa-autocreate",
    "s-fa-keep-local", "s-fa-admin-nogroups", "s-fa-post-logout",
])
def test_every_sso_setting_is_on_the_settings_page(field):
    """Markup, loader and serialiser — a field missing from any one of the three
    is a setting that silently will not stick."""
    html = (ROOT / "codai/admin/templates/settings.html").read_text()
    assert html.count(field) >= 3, f"{field}: {html.count(field)} occurrences"
