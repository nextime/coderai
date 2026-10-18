"""/admin/api/system/reload must actually re-read the files.

The endpoint existed for a long time and could never have worked: it built a
bare ``Request({})`` to reach ``app.state``, which raises before anything is
read. An operator who edited a pod pool's ``max_pods`` by hand and called
reload got a 500 and, worse, no reload -- so the only way to apply a config
edit was a restart.

These tests pin the two things that matter: the call succeeds without an HTTP
request object carrying app state, and ``models_data`` is re-read, because a
pod pool takes its ceilings from that dict on every acquisition.
"""
import json
import asyncio
import pytest

from codai.admin import routes


class _Mgr:
    """Minimal ConfigManager stand-in: reload() re-reads the file."""

    def __init__(self, path):
        self.path = path
        self.models_data = {}
        self.reloads = 0

        class _M:
            loaded = []
            preload = []
            default_load_mode = "on-request"

        class _C:
            models = _M()

        self.config = _C()

    def reload(self):
        self.reloads += 1
        self.models_data = json.load(open(self.path))
        return self.config


def _call(mgr_in_state=None):
    """Invoke the handler the way FastAPI would, with an injected request."""
    class _State:
        pass

    class _App:
        pass

    class _Req:
        pass

    req = _Req()
    app = _App()
    st = _State()
    if mgr_in_state is not None:
        st.config_mgr = mgr_in_state
    app.state = st
    req.app = app
    return asyncio.run(routes.api_reload_config(req, username="admin"))


@pytest.fixture
def models_file(tmp_path):
    p = tmp_path / "models.json"
    p.write_text(json.dumps({"text_models": [{"id": "a", "runpod": {"max_pods": 128}}]}))
    return p


def test_reload_succeeds_from_app_state(models_file):
    mgr = _Mgr(models_file)
    out = _call(mgr_in_state=mgr)
    assert out["success"] is True
    assert mgr.reloads == 1


def test_reload_falls_back_to_the_module_manager(models_file, monkeypatch):
    """The front proxy does not populate app.state.config_mgr."""
    mgr = _Mgr(models_file)
    monkeypatch.setattr(routes, "config_manager", mgr)
    out = _call(mgr_in_state=None)
    assert out["success"] is True
    assert mgr.reloads == 1


def test_models_json_is_re_read_not_just_config(models_file):
    """The regression that mattered: a hand-edited pool ceiling must apply."""
    mgr = _Mgr(models_file)
    _call(mgr_in_state=mgr)
    assert mgr.models_data["text_models"][0]["runpod"]["max_pods"] == 128

    models_file.write_text(json.dumps(
        {"text_models": [{"id": "a", "runpod": {"max_pods": 1}}]}))
    _call(mgr_in_state=mgr)
    assert mgr.models_data["text_models"][0]["runpod"]["max_pods"] == 1


def test_reload_reports_how_many_models_it_read(models_file):
    mgr = _Mgr(models_file)
    out = _call(mgr_in_state=mgr)
    assert out["models_reloaded"] == 1


def test_no_manager_is_503_not_500(monkeypatch):
    from fastapi import HTTPException
    monkeypatch.setattr(routes, "config_manager", None)
    with pytest.raises(HTTPException) as e:
        _call(mgr_in_state=None)
    assert e.value.status_code == 503


def test_a_failing_reload_is_reported_not_swallowed(tmp_path):
    from fastapi import HTTPException
    bad = tmp_path / "models.json"
    bad.write_text("{not json")
    mgr = _Mgr(bad)
    with pytest.raises(HTTPException) as e:
        _call(mgr_in_state=mgr)
    assert e.value.status_code == 500
    assert "reload failed" in str(e.value.detail)
