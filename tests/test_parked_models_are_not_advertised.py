"""A model parked as `enabled: false` must not be advertised to clients.

Measured on a running install (2026-10-26): `/v1/models` offered `LTX-2.5`,
`Krea-2-Turbo` and `Krea-2-Edit` -- three entries configured ahead of a
diffusers version that does not exist yet, whose weights are downloaded and
whose settings are written, and which answer 500 if anyone asks for them.

The first attempt hid them behind `all_engines`, which was the wrong seam: the
node-wide list the system worker serves -- the one the front hands clients --
MUST pass all_engines=True, because that worker has no engine of its own and
would otherwise list nothing. So the filter was off on the only path that
mattered. Hence a separate `include_disabled`, which only the admin config view
asks for, because that is where a parked model gets re-enabled.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codai.models.manager import multi_model_manager


CATALOGUE = {
    "image_models": [
        {"id": None, "alias": "Z-Image", "path": "Tongyi-MAI/Z-Image-Turbo",
         "enabled": True},
        {"id": None, "alias": "Krea-2-Turbo", "path": "unsloth/Krea-2-Turbo",
         "enabled": False},
    ],
    "video_models": [
        {"id": None, "alias": "LTX-2.3-10Eros",
         "path": "diffusers/LTX-2.3-Diffusers", "enabled": True},
        {"id": None, "alias": "LTX-2.5",
         "path": "Lightricks/LTX-2.5-Diffusers", "enabled": False},
    ],
}


@pytest.fixture
def catalogue(monkeypatch):
    from codai.admin import routes as admin_routes

    class _ConfigManager:
        models_data = CATALOGUE
        config = None

    monkeypatch.setattr(admin_routes, "config_manager", _ConfigManager(),
                        raising=False)
    # No per-engine restriction, so the assignment filter is not what is being
    # measured here.
    monkeypatch.setattr(multi_model_manager, "_assigned_model_keys", None,
                        raising=False)
    return CATALOGUE


def _ids(**kw):
    return {m.id for m in multi_model_manager.list_models(**kw)}


def test_the_node_wide_list_clients_read_hides_a_parked_model(catalogue):
    """all_engines=True is what the system worker passes. Clients see this."""
    ids = _ids(all_engines=True)
    assert "Z-Image" in ids and "LTX-2.3-10Eros" in ids
    for parked in ("Krea-2-Turbo", "LTX-2.5"):
        assert parked not in ids, f"{parked} is parked and must not be offered"
    # Nor under its path or its basename, which are advertised as aliases too.
    assert "unsloth/Krea-2-Turbo" not in ids
    assert "Lightricks/LTX-2.5-Diffusers" not in ids
    assert "LTX-2.5-Diffusers" not in ids


def test_the_per_engine_list_hides_it_too(catalogue):
    assert "LTX-2.5" not in _ids()


def test_the_admin_view_still_shows_it(catalogue):
    """It is the page where a parked model gets re-enabled."""
    ids = _ids(all_engines=True, include_disabled=True)
    assert {"Krea-2-Turbo", "LTX-2.5"} <= ids


def test_the_admin_route_asks_for_the_parked_ones(catalogue):
    """The caller, not just the switch: /admin/api/models must pass both."""
    import inspect
    from codai.admin import routes
    src = inspect.getsource(routes.api_list_models)
    assert "include_disabled=True" in src
    assert "all_engines=True" in src


def test_the_system_worker_does_not(catalogue):
    """The one list clients read must not ask for parked models."""
    import inspect
    import codai.system_app as system_app
    src = inspect.getsource(system_app)
    assert "list_models(all_engines=True)" in src
    assert "include_disabled" not in src
