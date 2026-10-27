"""Two entries sharing one weights path must not be confused for each other.

LTX-2.3-10Eros and LTX-2.3-PinkCherry both point at diffusers/LTX-2.3-Diffusers
and differ only in which GGUF is their transformer. A request for 10Eros
collapsed to the shared path and loaded PinkCherry's 15.9 GB checkpoint --
the wrong model, silently, because both are "valid" configs for that path.

The VRAM estimator already disambiguated this way for lisa vs lisa-32k (same
weights file, 178000 ctx vs 32768). The config used to LOAD did not.
"""
import pytest

from codai.models.manager import MultiModelManager


def _mgr():
    m = object.__new__(MultiModelManager)
    m.config = {
        "video:diffusers/LTX-2.3-Diffusers": {
            "alias": "LTX-2.3-PinkCherry",
            "component_quantization": {"transformer": "/g/PinkCherry.gguf"},
            "used_vram_gb": 22.0,
        },
        "video:LTX-2.3-10Eros": {
            "alias": "LTX-2.3-10Eros",
            "component_quantization": {"transformer": "/g/10Eros.gguf"},
            "used_vram_gb": 21.0,
        },
    }
    return m


def test_the_requested_id_picks_its_own_entry():
    m = _mgr()
    cfg = m._config_for_model_key("video:diffusers/LTX-2.3-Diffusers",
                                  "LTX-2.3-10Eros")
    assert cfg["component_quantization"]["transformer"] == "/g/10Eros.gguf"
    assert cfg["used_vram_gb"] == 21.0


def test_the_other_sibling_gets_its_own_too():
    m = _mgr()
    cfg = m._config_for_model_key("video:diffusers/LTX-2.3-Diffusers",
                                  "LTX-2.3-PinkCherry")
    assert cfg["component_quantization"]["transformer"] == "/g/PinkCherry.gguf"
    assert cfg["used_vram_gb"] == 22.0


def test_without_a_requested_id_the_old_behaviour_is_unchanged():
    # Every existing caller passes nothing, and must keep getting the
    # key-based lookup exactly as before.
    m = _mgr()
    cfg = m._config_for_model_key("video:diffusers/LTX-2.3-Diffusers")
    assert cfg["alias"] == "LTX-2.3-PinkCherry"


def test_an_id_matching_no_alias_falls_back_to_the_key():
    m = _mgr()
    cfg = m._config_for_model_key("video:diffusers/LTX-2.3-Diffusers",
                                  "something-else")
    assert cfg["alias"] == "LTX-2.3-PinkCherry"


def test_a_model_with_no_sibling_is_unaffected():
    m = object.__new__(MultiModelManager)
    m.config = {"video:solo": {"alias": "solo", "used_vram_gb": 5.0}}
    assert m._config_for_model_key("video:solo", "solo")["used_vram_gb"] == 5.0
    assert m._config_for_model_key("video:solo")["used_vram_gb"] == 5.0


def test_request_model_passes_the_requested_id_through():
    import inspect
    src = inspect.getsource(MultiModelManager.request_model)
    assert "_config_for_model_key(model_key, _requested_id)" in src, (
        "the loading config must disambiguate by the id the caller asked for")


# ---------------------------------------------------------------- registration
def _fresh():
    m = object.__new__(MultiModelManager)
    m.config = {}
    m.video_models = []
    m._registered_types = {}
    m._remember_registered_type = lambda *a, **k: None
    return m


def test_two_entries_sharing_a_path_both_survive_registration():
    """The collision that caused the wrong model to load.

    self.config is keyed <kind>:<path>, so a second entry with the same path
    overwrote the first. LTX-2.3-10Eros and -PinkCherry share
    diffusers/LTX-2.3-Diffusers and differ only in their GGUF; PinkCherry came
    second, so a request for 10Eros loaded PinkCherry's 15.9 GB transformer.
    """
    m = _fresh()
    eros = {"alias": "LTX-2.3-10Eros",
            "component_quantization": {"transformer": "/g/10Eros.gguf"}}
    pink = {"alias": "LTX-2.3-PinkCherry",
            "component_quantization": {"transformer": "/g/PinkCherry.gguf"}}
    m.set_video_model("diffusers/LTX-2.3-Diffusers", eros)
    m.set_video_model("diffusers/LTX-2.3-Diffusers", pink)

    # Both reachable by their own alias…
    assert m.config["video:LTX-2.3-10Eros"] is eros
    assert m.config["video:LTX-2.3-PinkCherry"] is pink
    # …and the lookup picks the right one.
    got = m._config_for_model_key("video:diffusers/LTX-2.3-Diffusers",
                                  "LTX-2.3-10Eros")
    assert got["component_quantization"]["transformer"] == "/g/10Eros.gguf"


def test_the_path_key_keeps_its_old_meaning():
    # Callers that know only a path still get the last writer, unchanged.
    m = _fresh()
    m.set_video_model("p", {"alias": "a", "v": 1})
    m.set_video_model("p", {"alias": "b", "v": 2})
    assert m.config["video:p"]["v"] == 2


def test_an_entry_without_an_alias_adds_no_extra_key():
    m = _fresh()
    m.set_video_model("p", {"v": 1})
    assert set(m.config) == {"video:p"}


def test_an_alias_equal_to_the_model_name_adds_no_duplicate():
    m = _fresh()
    m.set_video_model("p", {"alias": "p"})
    assert set(m.config) == {"video:p"}


def test_every_type_setter_uses_the_shared_store():
    # A fix applied to one model type and not the others is how this bug
    # survived: it is the same collision for audio, image, embedding…
    import inspect
    src = inspect.getsource(MultiModelManager)
    import re
    leftovers = re.findall(r'self\.config\[f"[a-z_]+:\{model_name\}"\] = config',
                           src)
    assert not leftovers, leftovers
    assert src.count("_store_type_config(") >= 9


# ------------------------------------------------------- the loaded-model key
def _sibs():
    m = object.__new__(MultiModelManager)
    m.config = {
        "video:diffusers/LTX-2.3-Diffusers": {
            "alias": "LTX-2.3-PinkCherry", "path": "diffusers/LTX-2.3-Diffusers"},
        "video:LTX-2.3-PinkCherry": {
            "alias": "LTX-2.3-PinkCherry", "path": "diffusers/LTX-2.3-Diffusers"},
        "video:LTX-2.3-10Eros": {
            "alias": "LTX-2.3-10Eros", "path": "diffusers/LTX-2.3-Diffusers"},
        "video:solo": {"alias": "solo", "path": "org/solo"},
    }
    return m


def test_siblings_get_their_own_cache_key():
    """The silent one: one key meant one resident pipeline for two models.

    PinkCherry requested while 10Eros was resident never loaded its own
    transformer -- it generated from 10Eros's weights and answered 200.
    """
    m = _sibs()
    a = m._sibling_model_key("video", "diffusers/LTX-2.3-Diffusers",
                             "LTX-2.3-10Eros")
    b = m._sibling_model_key("video", "diffusers/LTX-2.3-Diffusers",
                             "LTX-2.3-PinkCherry")
    assert a == "video:LTX-2.3-10Eros"
    assert b == "video:LTX-2.3-PinkCherry"
    assert a != b, "two models must not share one self.models slot"


def test_a_lone_model_keeps_the_path_key():
    # Splitting every aliased model would undo the deliberate collapse of all
    # name forms onto one key (one instance pool, no duplicate loads).
    m = _sibs()
    assert m._sibling_model_key("video", "org/solo", "solo") is None


def test_a_request_naming_the_path_is_not_split():
    m = _sibs()
    assert m._sibling_model_key("video", "diffusers/LTX-2.3-Diffusers",
                                "diffusers/LTX-2.3-Diffusers") is None


def test_an_unknown_id_is_not_split():
    m = _sibs()
    assert m._sibling_model_key("video", "diffusers/LTX-2.3-Diffusers",
                                "something-else") is None


@pytest.mark.parametrize("rid", [None, "", "   "])
def test_no_requested_id_means_no_split(rid):
    m = _sibs()
    assert m._sibling_model_key("video", "diffusers/LTX-2.3-Diffusers", rid) is None


def test_model_path_counts_as_a_shared_target():
    m = object.__new__(MultiModelManager)
    m.config = {
        "audio:a": {"alias": "a", "model_path": "/g/shared.gguf"},
        "audio:b": {"alias": "b", "model_path": "/g/shared.gguf"},
    }
    assert m._sibling_model_key("audio", "/g/shared.gguf", "a") == "audio:a"


def test_request_model_consults_the_sibling_key_first():
    import inspect
    src = inspect.getsource(MultiModelManager.request_model)
    i = src.index("# Step 2: Build the model key")
    window = src[i:i + 1400]
    assert "_sibling_model_key" in window
    # and it must come BEFORE the ordinary path key, or it never applies
    assert window.index("_sibling_model_key") < window.index(
        'model_key = f"{model_type}:{resolved_name}"')
