"""B3: two of the caller's characters in one frame.

`character_profiles` already took a list, but every reference went into one
IP-Adapter set -- and IP-Adapter conditions on the MEAN of its references, so
two characters produced one person who looked like neither. Keeping the
reference sets apart, and handing them to a model that takes several, is what
makes two people two people.
"""
import types

import pytest


class _Req:
    def __init__(self, **kw):
        self.characters = kw.get("characters")
        self.character_profiles = kw.get("character_profiles")
        self.character_references = kw.get("character_references")
        self.character_names = kw.get("character_names")


@pytest.fixture
def saved(monkeypatch):
    """Two saved profiles, with two reference images each."""
    store = {"alice": ["A1", "A2"], "bob": ["B1", "B2"]}
    import codai.api.characters as C
    monkeypatch.setattr(
        C, "resolve_character_profiles",
        lambda names: [i for n in names for i in store.get(str(n).lower(), [])])
    return store


# ------------------------------------------------------------------ grouping
def test_bare_names_become_one_group_each(saved):
    from codai.api.characters import character_groups
    g = character_groups(_Req(characters=["alice", "bob"]))
    assert [x["name"] for x in g] == ["alice", "bob"]
    assert g[0]["images"] == ["A1", "A2"]
    assert g[1]["images"] == ["B1", "B2"]


def test_slots_work_and_a_named_slot_without_images_is_resolved(saved):
    from codai.api.characters import character_groups
    g = character_groups(_Req(characters=[
        {"name": "carol", "images": ["C1"]},
        {"name": "alice"},
    ]))
    assert g[0]["images"] == ["C1"]
    assert g[1]["images"] == ["A1", "A2"]


def test_the_long_standing_field_still_groups_per_character(saved):
    from codai.api.characters import character_groups
    g = character_groups(_Req(character_profiles=["alice", "bob"]))
    assert len(g) == 2
    # This is the case that silently blended: two profiles, one flat list.
    assert g[0]["images"] != g[1]["images"]


def test_a_character_named_twice_is_one_character(saved):
    from codai.api.characters import character_groups
    g = character_groups(_Req(characters=["alice"], character_profiles=["Alice"]))
    assert len(g) == 1


def test_one_character_yields_one_group(saved):
    from codai.api.characters import character_groups
    assert len(character_groups(_Req(characters=["alice"]))) == 1
    assert character_groups(_Req()) == []


def test_inline_references_split_only_when_names_align(saved):
    from codai.api.characters import character_groups
    g = character_groups(_Req(character_references=["X1", "X2"],
                              character_names=["alice", "bob"]))
    assert len(g) == 2
    # Without aligned names there is no way to know which image is whose, so
    # they stay one group rather than being guessed at.
    g2 = character_groups(_Req(character_references=["X1", "X2"]))
    assert len(g2) == 1 and g2[0]["images"] == ["X1", "X2"]


def test_an_empty_profile_is_not_a_character(saved):
    from codai.api.characters import character_groups
    assert character_groups(_Req(characters=["nobody"])) == []


# --------------------------------------------------- the video request shapes
def test_video_resolves_bare_names_slots_and_legacy(saved):
    from codai.api.video import _resolve_character_inputs
    r = types.SimpleNamespace(
        character_profiles=None, characters=["alice", {"name": "bob"}],
        character_references=None, character_names=None)
    images, names = _resolve_character_inputs(r)
    assert names == ["alice", "bob"]
    assert images == ["A1", "A2", "B1", "B2"]


def test_a_bare_name_does_not_crash_the_minimax_path(saved):
    # The widened type is only safe if every consumer stopped calling .get()
    # on it. This is the one that would have raised AttributeError.
    import inspect

    from codai.api import video
    src = inspect.getsource(video)
    i = src.index('for entry in (request.characters or []):')
    window = src[i:i + 900]
    assert "isinstance(entry, str)" in window
    assert "isinstance(entry, dict)" in window


def test_the_keyframe_bridge_forwards_characters():
    import inspect

    from codai.api import video
    src = inspect.getsource(video)
    assert 'payload["characters"]' in src, (
        "a two-character clip would render its identity keyframe from neither")


# ------------------------------------------------- the image routing decision
def test_multi_character_needs_a_multi_reference_pipeline():
    # The decision rests on _is_instruction_edit, which is (image yes,
    # strength no) -- true for FLUX.2 and Qwen-Image-Edit, false for SDXL.
    from codai.api.images import _is_instruction_edit

    class _Edit:
        def __call__(self, image=None, prompt=None):
            pass

    class _Img2Img:
        def __call__(self, image=None, strength=None, prompt=None):
            pass

    assert _is_instruction_edit(_Edit()) is True
    assert _is_instruction_edit(_Img2Img()) is False


def test_the_routing_is_present_and_clears_the_blended_set():
    import inspect

    from codai.api import images
    src = inspect.getsource(images._generate_with_diffusers)
    assert "_multi = [g for g in char_groups" in src
    i = src.index("_multi = [g for g in char_groups")
    window = src[i:i + 1600]
    # Passed as a list of references …
    assert 'call_kwargs["image"] = _refs' in window
    # … and NOT also blended through IP-Adapter, which would undo it.
    assert "char_images = []" in window
    # A model that cannot do it says so rather than quietly blending.
    assert "_log.warning" in window
