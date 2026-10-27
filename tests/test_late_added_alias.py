"""A model added to models.json while the server runs must be loadable.

`/v1/models` re-reads models.json on every call, but the loader's alias map is
built ONCE at startup. So a newly added entry was advertised immediately and
could not be loaded at all: the alias resolved to nothing, the raw alias was
handed to diffusers as a repo id, and the caller got

    Repository Not Found for url https://huggingface.co/api/models/<alias>

-- a 401 against HuggingFace for a model sitting on the local disk. Measured
2026-10-27 with a freshly registered Wan2.2-TI2V-5B whose weights were cached
and whose `path` was correct; requesting the same model by its PATH loaded it
immediately, which is what pinned the cause to the alias map.
"""
import types

import pytest

from codai.models.manager import MultiModelManager


def _mgr(models_data):
    m = object.__new__(MultiModelManager)
    m.model_aliases = {}
    m._registered_types = {}
    import codai.admin.routes as R
    R.config_manager = types.SimpleNamespace(models_data=models_data)
    return m


def test_an_alias_added_after_startup_resolves_to_its_path():
    m = _mgr({"video_models": [
        {"alias": "Wan2.2-TI2V-5B", "path": "Wan-AI/Wan2.2-TI2V-5B-Diffusers"}]})
    assert m._alias_from_config("Wan2.2-TI2V-5B") == "Wan-AI/Wan2.2-TI2V-5B-Diffusers"


def test_the_lookup_is_case_insensitive_and_trims():
    m = _mgr({"image_models": [{"alias": "Z-Image", "path": "Tongyi-MAI/Z-Image"}]})
    assert m._alias_from_config("  z-image ") == "Tongyi-MAI/Z-Image"


def test_a_parked_entry_is_not_resolved():
    """Parking is how an entry waits on a blocker; resolving it would defeat
    that and hand clients a model that answers 500."""
    m = _mgr({"video_models": [
        {"alias": "LTX-2.5", "path": "x/y", "enabled": False}]})
    assert m._alias_from_config("LTX-2.5") is None


def test_an_entry_whose_alias_equals_its_path_is_not_a_loop():
    m = _mgr({"video_models": [{"alias": "same", "path": "same"}]})
    assert m._alias_from_config("same") is None


@pytest.mark.parametrize("alias", ["", None, "   ", "nosuchmodel"])
def test_nothing_matching_returns_none(alias):
    m = _mgr({"video_models": [{"alias": "a", "path": "b"}]})
    assert m._alias_from_config(alias) is None


def test_model_path_is_accepted_as_well_as_path():
    m = _mgr({"audio_models": [
        {"alias": "whisper-x", "model_path": "/AI/guffcache/whisper-x.gguf"}]})
    assert m._alias_from_config("whisper-x") == "/AI/guffcache/whisper-x.gguf"


def test_a_broken_config_does_not_raise():
    import codai.admin.routes as R
    m = object.__new__(MultiModelManager)
    m.model_aliases = {}
    R.config_manager = None
    assert m._alias_from_config("anything") is None
    R.config_manager = types.SimpleNamespace(models_data={"video_models": "not a list"})
    assert m._alias_from_config("anything") is None


def test_request_model_consults_the_config_before_giving_up():
    # Asserted on the source: exercising request_model end to end would need a
    # whole loaded manager, and what matters is that the fallback sits in the
    # alias branch and caches what it finds.
    import inspect
    src = inspect.getsource(MultiModelManager.request_model)
    i = src.index("if requested_model in self.model_aliases:")
    window = src[i:i + 1800]
    assert "_alias_from_config" in window
    # Cached, so the config is not re-walked on every subsequent request.
    assert "set_model_alias" in window
