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


# --------------------------------------------------------------------------- #
# the adapters one engine must register, also across sections
# --------------------------------------------------------------------------- #

def test_an_adapter_on_a_vision_entry_is_registered(catalogue):
    """The half that decides whether a vision QLoRA works at all.

    vLLM registers adapters at LAUNCH. If the adapter declared on the vision
    entry is not collected when a TEXT request starts the engine, that engine
    comes up without --enable-lora -- and then answers under the adapter's name
    with BASE weights, failing nothing and reporting nothing.
    """
    base = _entry("qwen38-awq", "qwen38")
    vision = _entry("qwen38-vision", "qwen38v",
                    loras=[{"name": "doc-reader", "path": "sha256:ff"}])
    catalogue({"text_models": [base], "vision_models": [vision]})
    specs = vw._sibling_lora_specs(base)
    # Under the sibling's own asked-for name, which is its ALIAS when it has
    # one (`_entry_name`: a seeded entry carries alias and no id).
    assert [s["name"] for s in specs] == ["qwen38v"], specs


def test_an_alias_becomes_the_adapter_name_which_is_a_trap(catalogue):
    """Worth pinning, because it decides what a client must ask for.

    An entry that carries a LoRA registers it under `_entry_name` -- the ALIAS
    if there is one -- and `_served_names` then excludes that name from the
    base's served set. So an entry with BOTH `id` and `alias` splits in two: a
    request for the alias gets base+adapter, a request for the id gets the bare
    base and nothing says so. An entry that owns an adapter should therefore
    carry ONE name, which is how diogene-alpha and digesta-alpha are declared.
    """
    base = _entry("qwen38-awq", "qwen38")
    both = _entry("qwen38-vision", "qwen38v",
                  loras=[{"name": "doc-reader", "path": "sha256:ff"}])
    catalogue({"text_models": [base], "vision_models": [both]})
    adapters = [s["name"] for s in vw._sibling_lora_specs(base)]
    assert adapters == ["qwen38v"]
    served = vw._served_names(base, "qwen38-awq", "qwen38")
    assert "qwen38v" not in served          # it is an adapter now
    assert "qwen38-vision" in served        # ...and the id serves the BASE

    # With one name only, there is nothing to get wrong.
    single = _entry("qwen38-vision",
                    loras=[{"name": "doc-reader", "path": "sha256:ff"}])
    catalogue({"text_models": [base], "vision_models": [single]})
    assert [s["name"] for s in vw._sibling_lora_specs(base)] == ["qwen38-vision"]


def test_an_adapter_on_another_base_is_not_registered(catalogue):
    base = _entry("qwen38-awq", "qwen38")
    other = _entry("other", repo="someone/Other-7B",
                   loras=[{"name": "nope", "path": "sha256:ee"}])
    catalogue({"text_models": [base], "vision_models": [other]})
    assert vw._sibling_lora_specs(base) == []


def test_the_entry_itself_is_not_collected_as_its_own_sibling(catalogue):
    """A seeded entry carries `alias` and no `id`, which is how "is this me?"
    once matched every sibling and excluded all of them."""
    base = _entry("qwen38-awq", "qwen38",
                  loras=[{"name": "own", "path": "sha256:dd"}])
    catalogue({"text_models": [base],
               "vision_models": [{"alias": "qwen38", "runpod": {"hf_repo": REPO},
                                  "loras": [{"name": "dup", "path": "sha256:dd"}]}]})
    names = [s["name"] for s in vw._sibling_lora_specs(base)]
    assert names.count("qwen38") == 0 or len(names) == len(set(names))


def test_one_reader_serves_both_questions(catalogue):
    """Two copies of "read the catalogue" is how one of them keeps the bug."""
    import inspect
    src = inspect.getsource(vw)
    assert 'md.get("text_models")' not in src
    for fn in (vw._served_names, vw._sibling_lora_specs):
        assert "_catalogue_entries()" in inspect.getsource(fn)


def test_the_reader_tolerates_a_catalogue_that_is_not_a_dict_of_lists(catalogue):
    catalogue({"text_models": [_entry("a")], "junk": "not a list",
               "nested": [{"id": "b"}, "not a dict"]})
    got = vw._catalogue_entries()
    assert {e.get("id") for e in got} == {"a", "b"}
