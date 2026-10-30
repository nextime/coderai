# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Telling a client which published ids are the same model.

Publishing several ids per model is deliberate: an alias can stand for a
round-robin group, and a typed id coexists with the checkpoint it names. What
was missing is any way to tell which ids are one model. Measured on a live
install: 206 published ids over 119 cards (136 aliases, 20 absolute paths whose
basenames were also listed, 50 repo ids), and a capability sweep reported
seventy "new models" that were names it had already tested.

The rule that matters most here is the second one: two cards sharing a
checkpoint but differing in CONFIGURATION are not duplicates. They are separate
models with separate settings, and a client must be able to see both.
"""

import io
import os

import pytest

from codai.models.manager import _group_ids
from codai.pydantic.textrequest import ModelInfo

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _src(rel):
    return io.open(os.path.join(_ROOT, rel), encoding="utf-8").read()


def _models(*ids):
    return [ModelInfo(id=i, type="text") for i in ids]


def _by_id(models):
    return {m.id: m for m in models}


# ---------------------------------------------------------------------------
# grouping
# ---------------------------------------------------------------------------

def test_the_three_names_of_one_gguf_collapse_to_one_model():
    """Path, basename and alias: the shape that produced 206 ids for 119 cards."""
    models = _models("/AI/guffcache/qwen38.gguf", "qwen38.gguf", "qwen38")
    _group_ids(models, {i: "qwen38" for i in (
        "/AI/guffcache/qwen38.gguf", "qwen38.gguf", "qwen38")})
    got = _by_id(models)
    assert {m.canonical_id for m in models} == {"qwen38"}
    assert got["qwen38"].alias_of is None
    assert got["qwen38"].aliases == ["/AI/guffcache/qwen38.gguf", "qwen38.gguf"]
    assert got["qwen38.gguf"].alias_of == "qwen38"
    assert got["/AI/guffcache/qwen38.gguf"].alias_of == "qwen38"


def test_skipping_the_entries_with_alias_of_tests_each_model_once():
    """The property a capability sweep actually needs."""
    ids = ["/AI/x.gguf", "x.gguf", "x", "other"]
    models = _models(*ids)
    _group_ids(models, {"/AI/x.gguf": "x", "x.gguf": "x", "x": "x",
                        "other": "other"})
    assert sorted(m.id for m in models if not m.alias_of) == ["other", "x"]


def test_two_cards_on_one_checkpoint_are_not_duplicates():
    """Different configuration means a different model, whatever they share on
    disk. Grouping these would hide one of two deliberately different setups."""
    models = _models("Tongyi-MAI/Z-Image-Turbo", "z-image-turbo-bf16")
    _group_ids(models, {"Tongyi-MAI/Z-Image-Turbo": "Tongyi-MAI/Z-Image-Turbo",
                        "z-image-turbo-bf16": "z-image-turbo-bf16"})
    for m in models:
        assert m.alias_of is None
        assert m.canonical_id == m.id
    assert len({m.canonical_id for m in models}) == 2


def test_a_model_published_under_one_id_says_so_plainly():
    models = _models("lonely")
    _group_ids(models, {})
    assert models[0].canonical_id == "lonely"
    assert models[0].alias_of is None
    assert models[0].aliases is None


def test_an_unpublished_canonical_name_is_never_pointed_at():
    """The canonical entry can be filtered out of this listing -- parked, or
    assigned to another engine. Pointing a client at a name this server does
    not offer is worse than leaving the id ungrouped."""
    models = _models("/AI/x.gguf", "x.gguf")
    _group_ids(models, {"/AI/x.gguf": "x", "x.gguf": "x"})   # "x" not published
    for m in models:
        assert m.alias_of is None
        assert m.canonical_id == m.id


def test_grouping_never_invents_an_alias_that_was_not_listed():
    models = _models("x", "x.gguf")
    _group_ids(models, {"x": "x", "x.gguf": "x", "/AI/never/listed.gguf": "x"})
    assert _by_id(models)["x"].aliases == ["x.gguf"]


def test_an_id_with_no_canonical_recorded_stands_on_its_own():
    models = _models("a", "b")
    _group_ids(models, {})
    assert [m.canonical_id for m in models] == ["a", "b"]
    assert not any(m.alias_of for m in models)


def test_grouping_is_idempotent():
    """The listing is rebuilt on every call; running twice must not accumulate."""
    models = _models("/AI/x.gguf", "x")
    canon = {"/AI/x.gguf": "x", "x": "x"}
    _group_ids(models, canon)
    _group_ids(models, canon)
    assert _by_id(models)["x"].aliases == ["/AI/x.gguf"]
    assert _by_id(models)["/AI/x.gguf"].alias_of == "x"


# ---------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------

def test_the_fields_are_published():
    src = _src("codai/pydantic/textrequest.py")
    for field in ("canonical_id", "alias_of", "aliases"):
        assert f"{field}: Optional" in src


def test_the_listing_records_a_canonical_for_every_id_it_adds():
    src = _src("codai/models/manager.py")
    # list_models is the last method in the class, so the slice runs to the end
    # of the file rather than to a following "    def " that does not exist.
    body = src[src.index("def list_models(self"):]
    assert "_canon[model_id] = canonical or model_id" in body
    assert "_group_ids(models, _canon)" in body
    # Every id published for a configured card carries its card's name.
    assert body.count("canonical=mid") == 4


def test_the_helper_is_not_inside_the_class():
    """It was first added between two methods, which ends the class body: 126
    methods became module functions and list_models stopped being a method."""
    import ast
    tree = ast.parse(_src("codai/models/manager.py"))
    top = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert "_group_ids" in top
    cls = [n for n in tree.body
           if isinstance(n, ast.ClassDef) and n.name == "MultiModelManager"][0]
    methods = {m.name for m in cls.body if isinstance(m, ast.FunctionDef)}
    assert "list_models" in methods
