# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""A pod must be able to say why it has nowhere to send a page.

A rented pod answered

    model 'datalab-to/surya-ocr-2' is not served over HTTP by this install

for a model it was serving fine over its own loopback. The sentence blames
configuration; the cause was a lookup. Nothing could tell the two apart,
because `_local_candidate` ended in `except Exception: return None` and the
pod's log is not reachable from the machine that receives its 503.

So: every decline is recorded with a reason, and the ERROR carries it.
"""

import os

import pytest

from codai.models import manager as mm
from codai.models import placement
from codai.ocr.base import OcrError


SEED = '[{"backend": "vllm", "path": "datalab-to/surya-ocr-2"}]'
MID = "datalab-to/surya-ocr-2"


@pytest.fixture(autouse=True)
def _clean():
    placement._declines.clear()
    placement.forget()
    mm._SEED_CACHE = None
    yield
    placement._declines.clear()
    placement.forget()
    mm._SEED_CACHE = None


# ------------------------------------------------- the seeded catalogue #
def test_a_seeded_model_is_found_without_the_admin_global(monkeypatch):
    """THE pod bug: the entry lives in CODERAI_SEED_MODELS, not models.json."""
    monkeypatch.setenv("CODERAI_SEED_MODELS", SEED)
    mm._SEED_CACHE = None
    assert mm._model_entry_for(MID) is not None
    assert mm._pinned_backend(MID) == "vllm"


def test_the_seed_is_matched_by_basename_too(monkeypatch):
    monkeypatch.setenv("CODERAI_SEED_MODELS", SEED)
    mm._SEED_CACHE = None
    assert mm._model_entry_for("surya-ocr-2") is not None


def test_an_unseeded_name_is_still_unknown(monkeypatch):
    monkeypatch.setenv("CODERAI_SEED_MODELS", SEED)
    mm._SEED_CACHE = None
    assert mm._model_entry_for("someone/else") is None


def test_no_seed_in_the_environment_changes_nothing(monkeypatch):
    monkeypatch.delenv("CODERAI_SEED_MODELS", raising=False)
    mm._SEED_CACHE = None
    assert mm._env_seed_entries() == []


def test_a_malformed_seed_is_not_an_exception(monkeypatch):
    """A pod must boot and say so, not crash in a lookup."""
    monkeypatch.setenv("CODERAI_SEED_MODELS", "{not json")
    mm._SEED_CACHE = None
    assert mm._env_seed_entries() == []
    assert mm._model_entry_for(MID) is None


def test_a_real_catalogue_entry_wins_over_the_seed(monkeypatch):
    """An operator's edited entry must not be shadowed by the launch seed."""
    monkeypatch.setenv("CODERAI_SEED_MODELS", SEED)
    mm._SEED_CACHE = None

    class _CM:
        models_data = {"text_models": [{"path": MID, "backend": "ds4",
                                        "marker": "real"}]}

    import codai.admin.routes as ar
    monkeypatch.setattr(ar, "config_manager", _CM, raising=False)
    assert mm._model_entry_for(MID).get("marker") == "real"


# ------------------------------------------------------- the declines #
def test_a_local_decline_names_the_three_things_that_decide_it(monkeypatch):
    monkeypatch.setattr(mm, "resolve_engine_backend", lambda _m: None)
    monkeypatch.setattr(mm, "_model_entry_for", lambda _m: None)
    monkeypatch.setattr(mm, "_pinned_backend", lambda _m: "")
    monkeypatch.setattr(mm, "_engine_enabled", lambda _e: False)
    assert placement._local_candidate(MID) is None
    why = placement.why_no_candidates(MID)
    assert "NOT FOUND in this process" in why
    assert "vllm enabled=False" in why


def test_a_raised_exception_is_recorded_not_swallowed(monkeypatch):
    def _boom(_m):
        raise RuntimeError("vllm venv is missing")

    monkeypatch.setattr(mm, "resolve_engine_backend", _boom)
    assert placement._local_candidate(MID) is None
    assert "vllm venv is missing" in placement.why_no_candidates(MID)


def test_the_decline_is_said_once_per_reason(monkeypatch, capsys):
    monkeypatch.setattr(mm, "resolve_engine_backend", lambda _m: None)
    monkeypatch.setattr(mm, "_model_entry_for", lambda _m: None)
    monkeypatch.setattr(mm, "_pinned_backend", lambda _m: "")
    monkeypatch.setattr(mm, "_engine_enabled", lambda _e: False)
    for _ in range(20):
        placement._local_candidate(MID)
    assert capsys.readouterr().out.count("no local candidate") == 1


def test_a_changed_reason_is_said_again(monkeypatch, capsys):
    monkeypatch.setattr(mm, "resolve_engine_backend", lambda _m: None)
    monkeypatch.setattr(mm, "_model_entry_for", lambda _m: None)
    monkeypatch.setattr(mm, "_pinned_backend", lambda _m: "")
    monkeypatch.setattr(mm, "_engine_enabled", lambda _e: False)
    placement._local_candidate(MID)
    monkeypatch.setattr(mm, "_engine_enabled", lambda _e: True)
    placement._local_candidate(MID)
    assert capsys.readouterr().out.count("no local candidate") == 2


def test_the_error_carries_the_reasons(monkeypatch):
    """The whole point: the 503 travels to a machine the log does not."""
    monkeypatch.setattr(placement, "candidates", lambda _m: [])
    placement._decline(MID, "local", "catalogue entry NOT FOUND in this process")
    with pytest.raises(OcrError) as e:
        placement.choose(MID)
    assert "[why: " in str(e.value)
    assert "NOT FOUND in this process" in str(e.value)


def test_the_error_is_unchanged_when_nothing_was_recorded(monkeypatch):
    monkeypatch.setattr(placement, "candidates", lambda _m: [])
    with pytest.raises(OcrError) as e:
        placement.choose(MID)
    assert "[why: " not in str(e.value)


def test_a_working_local_candidate_clears_an_old_decline(monkeypatch):
    """A fixed install must stop reporting the fault it no longer has."""
    placement._decline(MID, "local", "something that was true once")
    monkeypatch.setattr(mm, "resolve_engine_backend", lambda _m: "vllm")
    from codai.api import runpod_worker, vllm_worker
    monkeypatch.setattr(mm, "get_active_vllm_config", lambda: object(),
                        raising=False)
    monkeypatch.setattr(runpod_worker, "_model_path_for", lambda _m: MID)
    monkeypatch.setattr(vllm_worker, "ensure_service",
                        lambda *_a, **_k: "http://127.0.0.1:8000")
    got = placement._local_candidate(MID)
    assert got is not None and got.kind == "local"
    assert placement.why_no_candidates(MID) == ""
