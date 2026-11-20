"""A LoRA is named for what it DOES, not for what it is.

Reported from a chat client: requests naming `diogene-alpha` came back with an
EMPTY reply and a 200, while `qwen38` — the same weights, same pod, no adapter
— answered normally.

Parser selection reads the model name:

    if 'qwen' in model_lower:  -> QwenParser

`qwen38-awq` matches. `diogene-alpha` matches nothing, so it fell to the
generic parser and the model's `<think>…</think>` block was never separated
from the answer. With a modest `max_tokens` the whole budget goes into
reasoning that never closes, so after the client strips it there is nothing
left — an empty reply from a successful request, which is the hardest kind of
wrong to see.

An adapter is the same family as the weights it adapts.
"""

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codai.models import parser as mp                          # noqa: E402


def _entry(monkeypatch, entry):
    import codai.models.manager as mgr
    monkeypatch.setattr(mgr, "_model_entry_for", lambda n: entry, raising=False)


def test_an_adapter_is_resolved_to_the_family_it_adapts(monkeypatch):
    _entry(monkeypatch, {"id": "diogene-alpha",
                         "loras": [{"name": "diogene-alpha"}],
                         "runpod": {"hf_repo": "cyankiwi/Qwen3.8-27B-AWQ-INT4"}})
    assert "qwen" in mp._family_name("diogene-alpha").lower()


def test_the_adapter_gets_the_base_parser(monkeypatch):
    _entry(monkeypatch, {"id": "diogene-alpha",
                         "loras": [{"name": "diogene-alpha"}],
                         "runpod": {"hf_repo": "cyankiwi/Qwen3.8-27B-AWQ-INT4"}})
    chosen = mp.ModelParserDispatcher(model_name="diogene-alpha")._get_parser()
    expected = mp.ModelParserDispatcher(model_name="qwen38-awq")._get_parser()
    assert type(chosen) is type(expected)


def test_a_name_that_already_says_its_family_is_not_looked_up(monkeypatch):
    """No catalogue read for the common case, and no surprise if an entry is
    missing."""
    import codai.models.manager as mgr
    monkeypatch.setattr(mgr, "_model_entry_for",
                        lambda n: pytest.fail("should not be consulted"),
                        raising=False)
    assert mp._family_name("qwen38-awq") == "qwen38-awq"


def test_a_plain_model_with_an_odd_name_is_left_alone(monkeypatch):
    """Only an ADAPTER borrows an identity. A base model with an unusual name
    keeps the generic parser, which is the honest default."""
    _entry(monkeypatch, {"id": "mystery-7b", "path": "vendor/mystery-7b"})
    assert mp._family_name("mystery-7b") == "mystery-7b"


def test_an_unknown_model_does_not_raise(monkeypatch):
    import codai.models.manager as mgr
    monkeypatch.setattr(mgr, "_model_entry_for", lambda n: None, raising=False)
    assert mp._family_name("who-knows") == "who-knows"


def test_the_base_path_is_used_when_there_is_no_runpod_block(monkeypatch):
    _entry(monkeypatch, {"id": "tuned", "path": "Qwen/Qwen3-8B",
                         "loras": [{"name": "tuned"}]})
    assert mp._family_name("tuned") == "Qwen/Qwen3-8B"
