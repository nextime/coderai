from datetime import UTC, datetime, timedelta
import importlib.util
import json
import sys


AUTH_MODULE_PATH = "/storage/coderai/codai/admin/auth.py"
spec = importlib.util.spec_from_file_location("test_admin_auth_module", AUTH_MODULE_PATH)
auth_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = auth_module
spec.loader.exec_module(auth_module)

SessionManager = auth_module.SessionManager


def _write_auth_file(config_dir, auth_data):
    (config_dir / "auth.json").write_text(json.dumps(auth_data))


def _read_auth_file(config_dir):
    return json.loads((config_dir / "auth.json").read_text())


def test_session_manager_defaults_to_thirty_day_sliding_timeout(tmp_path):
    manager = SessionManager(tmp_path)

    assert manager.session_timeout == timedelta(days=30)


def test_create_session_stores_timezone_aware_utc_timestamps(tmp_path):
    manager = SessionManager(tmp_path)
    cookie_value = manager.create_session("admin")
    session_id = cookie_value.rsplit('.', 1)[0]

    session = _read_auth_file(tmp_path)["sessions"][session_id]
    created_at = datetime.fromisoformat(session["created_at"])
    expires_at = datetime.fromisoformat(session["expires_at"])

    assert created_at.tzinfo == UTC
    assert expires_at.tzinfo == UTC


def test_validate_session_refreshes_expiration_by_thirty_days(tmp_path):
    manager = SessionManager(tmp_path)
    cookie_value = manager.create_session("admin")
    session_id = cookie_value.rsplit('.', 1)[0]

    before_validation = _read_auth_file(tmp_path)["sessions"][session_id]["expires_at"]
    before_expiry = datetime.fromisoformat(before_validation)

    username = manager.validate_session(cookie_value)

    after_validation = _read_auth_file(tmp_path)["sessions"][session_id]["expires_at"]
    after_expiry = datetime.fromisoformat(after_validation)

    assert username == "admin"
    assert after_expiry > before_expiry
    assert after_expiry - datetime.now(UTC) > timedelta(days=29, hours=23)


def test_validate_session_rejects_and_cleans_up_expired_session(tmp_path):
    manager = SessionManager(tmp_path)
    cookie_value = manager.create_session("admin")
    session_id = cookie_value.rsplit('.', 1)[0]

    auth_data = _read_auth_file(tmp_path)
    auth_data["sessions"][session_id]["expires_at"] = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    _write_auth_file(tmp_path, auth_data)

    username = manager.validate_session(cookie_value)

    stored_auth_data = _read_auth_file(tmp_path)
    assert username is None
    assert session_id not in stored_auth_data["sessions"]


def test_validate_session_accepts_legacy_naive_expiration_timestamp(tmp_path):
    manager = SessionManager(tmp_path)
    cookie_value = manager.create_session("admin")
    session_id = cookie_value.rsplit('.', 1)[0]

    auth_data = _read_auth_file(tmp_path)
    auth_data["sessions"][session_id]["expires_at"] = (
        datetime.now(UTC).replace(tzinfo=None) + timedelta(days=7)
    ).isoformat()
    _write_auth_file(tmp_path, auth_data)

    username = manager.validate_session(cookie_value)
    stored_session = _read_auth_file(tmp_path)["sessions"][session_id]
    refreshed_expiry = datetime.fromisoformat(stored_session["expires_at"])

    assert username == "admin"
    assert refreshed_expiry.tzinfo == UTC
