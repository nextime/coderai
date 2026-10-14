"""An embedding model on a capability pod resolved to NO capability.

Switching a model's `runpod.engine` from `coderai-vllm` to `coderai` -- the
correct setting for a capability pod -- removed an IMPLICIT capability:
`_CODERAI_SERVER_ENGINES["coderai-vllm"]` had been supplying the classification.
With plain `coderai` and no `capability` field, `model_capability()` returned ""
and:

  * the front could not classify the request, so bge-m3 was loaded LOCALLY on a
    GPU-less orchestrator;
  * `default_capability_image("")` is "", so a blank image field would raise
    "set `image` on the runpod block" at provisioning time.

It failed as "tries to load on the wrong machine", which is a long way from
"this entry has no capability". An entry in `embedding_models` IS an embeddings
model; nothing had told model_capability that, because it receives a bare dict
and `_model_entry_for()` discards the section key it matched in.
"""

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codai.api import runpod_worker as rw                      # noqa: E402


class _CM:
    def __init__(self, data):
        self.models_data = data


def _wire(monkeypatch, data):
    """Point the capability lookup at a models_data we control."""
    import types
    mod = types.ModuleType("codai.admin.routes")
    mod.config_manager = _CM(data)
    monkeypatch.setitem(sys.modules, "codai.admin.routes", mod)


def test_an_embedding_entry_gets_the_embeddings_capability(monkeypatch):
    entry = {"id": "BAAI/bge-m3", "runpod": {"engine": "coderai"}}
    _wire(monkeypatch, {"embedding_models": [entry]})
    assert rw.model_capability(entry, include_text=True) == "embeddings"


def test_it_resolves_by_identity(monkeypatch):
    """The same dict object the loader holds, even with no id at all."""
    entry = {"path": "/models/whatever", "runpod": {}}
    _wire(monkeypatch, {"embedding_models": [entry]})
    assert rw._in_embedding_section(entry) is True


def test_it_resolves_a_copy_by_id(monkeypatch):
    """An entry rebuilt or copied must still resolve, or the inference would
    depend on object identity surviving a round trip."""
    entry = {"id": "BAAI/bge-m3", "runpod": {}}
    _wire(monkeypatch, {"embedding_models": [dict(entry)]})
    assert rw._in_embedding_section(entry) is True


def test_a_text_model_is_not_claimed(monkeypatch):
    entry = {"id": "qwen38-awq", "runpod": {"engine": "coderai-vllm"}}
    _wire(monkeypatch, {"text_models": [entry], "embedding_models": []})
    assert rw._in_embedding_section(entry) is False


def test_an_explicit_capability_still_wins(monkeypatch):
    """A model that serves something its section does not imply must keep its
    override -- that is what the field is for."""
    entry = {"id": "x", "capability": "rerank"}
    _wire(monkeypatch, {"embedding_models": [entry]})
    assert rw.model_capability(entry, include_text=True) == "rerank"


def test_the_capability_now_resolves_an_image(monkeypatch):
    """The second half of the bug: with no capability, a blank image field
    raises at provisioning time instead of defaulting."""
    entry = {"id": "BAAI/bge-m3", "runpod": {"engine": "coderai"}}
    _wire(monkeypatch, {"embedding_models": [entry]})
    cap = rw.model_capability(entry, include_text=True)
    assert rw.default_capability_image(cap).endswith("coderai-embeddings:latest")


@pytest.mark.parametrize("data", [
    {},
    {"embedding_models": None},
    {"embedding_models": [None, "junk", 3]},
])
def test_a_broken_models_data_is_false_not_an_exception(monkeypatch, data):
    """A capability GUESS is never worth an exception in a placement path."""
    _wire(monkeypatch, data)
    assert rw._in_embedding_section({"id": "x"}) is False


def test_no_config_manager_at_all_is_false(monkeypatch):
    import types
    mod = types.ModuleType("codai.admin.routes")
    mod.config_manager = None
    monkeypatch.setitem(sys.modules, "codai.admin.routes", mod)
    assert rw._in_embedding_section({"id": "x"}) is False


def test_only_the_embedding_section_is_inferred():
    """Remapping every section to fix one would change placement for every
    model kind. The others work today; this stays narrow on purpose."""
    src = (ROOT / "codai/api/runpod_worker.py").read_text()
    fn = src.split("def _in_embedding_section")[1].split("\ndef ")[0]
    for other in ("text_models", "image_models", "video_models", "tts_models"):
        assert other not in fn, f"the inference reaches into {other}"
