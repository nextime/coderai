"""Which names one vLLM engine answers to, across model categories.

A model's TYPE is the models.json SECTION it is written in (main.py's
_CATEGORY_TYPE_PREFIX), so the same weights served both as text and as vision
are two entries in two different lists. `_served_names` collected names from
`text_models` only, so the vision name was never passed at launch: the engine
did not answer to it, and `_ensure_service_once` relaunches an engine that does
not answer -- which for a 27B sharing a card is an out-of-memory crash, not a
slowdown. One engine serves a set of WEIGHTS; which list the operator wrote a
name in says nothing about that.
"""

import pytest

from codai.api import vllm_worker as vw


REPO = "cyankiwi/Qwen3.8-27B-AWQ-INT4"


def _entry(mid, alias=None, repo=REPO, loras=None):
    e = {"id": mid, "runpod": {"hf_repo": repo}}
    if alias:
        e["alias"] = alias
    if loras:
        e["loras"] = loras
    return e


@pytest.fixture
def catalogue(monkeypatch):
    import codai.admin.routes as routes

    def _set(md):
        monkeypatch.setattr(routes, "config_manager",
                            type("M", (), {"models_data": md})(), raising=False)
    return _set


def test_a_vision_entry_on_the_same_weights_is_served(catalogue):
    """The regression: the vision name must reach the launch."""
    base = _entry("qwen38-awq", "qwen38",
                  loras=[{"name": "diogene-alpha", "path": "sha256:aa"}])
    catalogue({"text_models": [base],
               "vision_models": [_entry("qwen38-vision", "qwen38v")]})
    names = vw._served_names(base, "qwen38-awq", "qwen38")
    assert "qwen38-vision" in names
    assert "qwen38v" in names
    assert "qwen38-awq" in names


def test_an_adapter_name_is_never_served_as_a_model(catalogue):
    """vLLM would then hold a model and an adapter under one name."""
    base = _entry("qwen38-awq", "qwen38",
                  loras=[{"name": "diogene-alpha", "path": "sha256:aa"}])
    catalogue({"text_models": [base],
               "vision_models": [_entry("qwen38-vision", "qwen38v")]})
    names = vw._served_names(base, "qwen38-awq", "qwen38")
    assert "diogene-alpha" not in names


def test_different_weights_are_not_collected(catalogue):
    """Only the names of THESE weights; another model is another engine."""
    base = _entry("qwen38-awq", "qwen38",
                  loras=[{"name": "diogene-alpha", "path": "sha256:aa"}])
    catalogue({"text_models": [base],
               "vision_models": [_entry("other-vlm", "other",
                                        repo="someone/Other-7B")]})
    names = vw._served_names(base, "qwen38-awq", "qwen38")
    assert "other-vlm" not in names and "other" not in names


def test_a_model_with_no_adapters_is_named_exactly_as_before(catalogue):
    """Nothing is ambiguous about a model served alone, so nothing is added."""
    base = _entry("solo")
    catalogue({"text_models": [base], "vision_models": [_entry("solo-vision")]})
    assert vw._served_names(base, "solo", "solo") == ["solo"]


def test_every_category_is_scanned(catalogue):
    """Not a list of blessed sections: any category may name these weights."""
    base = _entry("base-id", "base-alias",
                  loras=[{"name": "ad", "path": "sha256:bb"}])
    catalogue({"text_models": [base],
               "vision_models": [_entry("as-vision")],
               "embedding_models": [_entry("as-embedding")],
               "audio_models": [_entry("as-audio")],
               "not_a_list": "ignored"})
    names = vw._served_names(base, "base-id", "base-alias")
    for n in ("as-vision", "as-embedding", "as-audio"):
        assert n in names, n


def test_names_are_unique_and_ordered_from_the_asking_name(catalogue):
    base = _entry("qwen38-awq", "qwen38",
                  loras=[{"name": "ad", "path": "sha256:cc"}])
    catalogue({"text_models": [base, _entry("qwen38-awq", "qwen38")],
               "vision_models": [_entry("qwen38-vision")]})
    names = vw._served_names(base, "qwen38-awq", "qwen38")
    assert names[0] == "qwen38-awq"
    assert len(names) == len(set(names))


def test_a_broken_catalogue_does_not_break_the_launch(monkeypatch):
    """A name set that raises would stop the engine starting at all."""
    import codai.admin.routes as routes
    monkeypatch.setattr(routes, "config_manager", None, raising=False)
    base = _entry("x", loras=[{"name": "ad", "path": "sha256:dd"}])
    assert vw._served_names(base, "x", "x")[0] == "x"
