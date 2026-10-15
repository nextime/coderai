"""A pod registers a seeded model as a TEXT model unless told otherwise.

In models.json a model's kind is the dict KEY it sits under — `embedding_models`,
`image_models`, … — not a field on the entry. Only the ENTRY travels to a pod
(`CODERAI_SEED_MODELS`), and the pod's seeder reads `model_type` with

    section = e.get("model_type") or "text_models"

So every seeded model landed in `text_models` whatever it actually was.

Measured on a real A40 running coderai-embeddings: the pod booted, served
/v1/models, listed `BAAI/bge-m3` — and answered

    Model 'BAAI/bge-m3' is a text model and cannot be used for embedding
    generation. Use the text endpoint instead.

That is the expensive part. Nothing looks broken from outside: the routing is
right, the image is right, the catalogue is right, and the error blames the
caller for using the wrong endpoint.

Text-serving pods were unaffected, which is why this survived: a vLLM pod
serves text, and OCR engines are picked by name rather than registered as
models at all.
"""

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codai.api import runpod_worker as rw                      # noqa: E402


class _CM:
    def __init__(self, data):
        self.models_data = data


BGE = {"id": "BAAI/bge-m3", "alias": "bge-m3", "path": "BAAI/bge-m3",
       "backend": "runpod", "runpod": {"engine": "coderai"}}
SDXL = {"id": "sdxl", "path": "stabilityai/sdxl-turbo"}
LLM = {"id": "qwen", "path": "Qwen/Qwen3-8B"}


@pytest.fixture
def models(monkeypatch):
    data = {"text_models": [LLM], "image_models": [SDXL],
            "embedding_models": [BGE]}
    import codai.admin.routes as routes
    monkeypatch.setattr(routes, "config_manager", _CM(data), raising=False)
    return data


# ------------------------------------------------------- finding the section #
def test_the_section_is_found_by_identity(models):
    assert rw.entry_model_type(BGE) == "embedding_models"
    assert rw.entry_model_type(SDXL) == "image_models"
    assert rw.entry_model_type(LLM) == "text_models"


def test_a_copied_entry_still_resolves_by_id(models):
    """The entry that reaches the seeder is not always the loader's own dict."""
    assert rw.entry_model_type(dict(BGE)) == "embedding_models"


def test_an_unknown_entry_resolves_to_nothing(models):
    assert rw.entry_model_type({"id": "nope", "path": "nope"}) == ""
    assert rw.entry_model_type({}) == ""


def test_it_never_raises(monkeypatch):
    """It feeds a seeding path. A guess is not worth an exception there."""
    import codai.admin.routes as routes
    monkeypatch.setattr(routes, "config_manager", None, raising=False)
    assert rw.entry_model_type(BGE) == ""
    monkeypatch.setattr(routes, "config_manager", _CM("not a dict"), raising=False)
    assert rw.entry_model_type(BGE) == ""
    monkeypatch.setattr(routes, "config_manager",
                        _CM({"embedding_models": "not a list"}), raising=False)
    assert rw.entry_model_type(BGE) == ""


# -------------------------------------------------- what the pod is told #
def test_a_per_model_pod_seed_carries_its_section(models):
    """THE regression: without model_type the pod files bge-m3 under text."""
    got = json.loads(rw.seed_model_env(BGE, served="BAAI/bge-m3"))
    entries = got if isinstance(got, list) else [got]
    assert entries[0]["model_type"] == "embedding_models"


def test_an_explicit_model_type_on_the_entry_wins(models):
    """An entry that already says what it is is not second-guessed."""
    e = dict(BGE, model_type="image_models")
    got = json.loads(rw.seed_model_env(e, served="BAAI/bge-m3"))
    entries = got if isinstance(got, list) else [got]
    assert entries[0]["model_type"] == "image_models"


def test_a_seed_for_an_unknown_entry_is_still_produced(models):
    """Not knowing the section must not cost the pod its catalogue entry —
    that would turn a wrong kind into no model at all."""
    e = {"id": "stray", "path": "some-org/some-model"}
    got = json.loads(rw.seed_model_env(e, served="some-org/some-model"))
    entries = got if isinstance(got, list) else [got]
    assert entries[0]["path"] == "some-org/some-model"


def test_a_capability_pod_seed_carries_its_section(models):
    """The capability path knows the section already — it iterates them — and
    was dropping it on the floor at `out.append(entry)`."""
    got = rw.capability_seed_entries("embeddings")
    assert got and all(e.get("model_type") == "embedding_models" for e in got)


def test_a_capability_seed_does_not_mutate_models_json(models):
    """The stamp goes on a COPY. Writing it into the live entry would persist
    a field this install never had."""
    rw.capability_seed_entries("embeddings")
    assert "model_type" not in BGE


def test_the_pod_seeder_defaults_to_text_which_is_why_this_matters():
    """Pin the far side of the contract: if the fallback ever changes, the
    reason these tests exist changes with it."""
    src = (ROOT / "codai/config.py").read_text()
    assert 'e.get("model_type") or "text_models"' in src
