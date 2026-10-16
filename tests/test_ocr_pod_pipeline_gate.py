# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""The pod-side OCR pipeline gate must read the image a LAUNCH would use.

`runpod.image` is optional and normally absent: `engine: coderai-vllm-ocr`
is how the image is chosen, and `pod_plan` resolves it when a pod is rented.
A gate that read the raw field saw "" for exactly that configuration, decided
the pods could not run the pipeline, and rasterised every page on the
orchestrator while the pods it had rented carried the rasteriser and served
only the model. Nothing failed, nothing was logged, and the box sat at 47 GB.

So these tests pin two things: the resolver answers what a launch would launch,
and the gate says out loud which side of the line it landed on.
"""

import types

import pytest

from codai.api.runpod_worker import (image_runs_ocr_pipeline,
                                     pod_image_for_entry)
import codai.ocr.manager as om


OCR_IMAGE = "ghcr.io/nextime/coderai-vllm-ocr:latest"
PLAIN_IMAGE = "ghcr.io/nextime/coderai-vllm:latest"


def _entry(**runpod):
    rp = {"mode": "pods", "served_model": "datalab-to/surya-ocr-2"}
    rp.update(runpod)
    return {"id": "datalab-to/surya-ocr-2", "backend": "runpod",
            "enabled": True, "runpod": rp}


# --------------------------------------------------------------------------
# pod_image_for_entry
# --------------------------------------------------------------------------

def test_the_engine_name_alone_resolves_the_ocr_image():
    """The production shape: `engine` set, `image` absent."""
    got = pod_image_for_entry(_entry(engine="coderai-vllm-ocr"),
                              model_key="datalab-to/surya-ocr-2")
    assert got == OCR_IMAGE
    assert image_runs_ocr_pipeline(got)


def test_the_plain_vllm_engine_resolves_to_an_image_without_the_pipeline():
    got = pod_image_for_entry(_entry(engine="coderai-vllm"),
                              model_key="datalab-to/surya-ocr-2")
    assert got == PLAIN_IMAGE
    assert not image_runs_ocr_pipeline(got)


def test_an_explicit_image_beats_the_engine():
    """An operator who names an image gets that image, whatever `engine` says."""
    got = pod_image_for_entry(
        _entry(engine="coderai-vllm", image="registry.example/coderai-vllm-ocr:0.4.0"),
        model_key="datalab-to/surya-ocr-2")
    assert got == "registry.example/coderai-vllm-ocr:0.4.0"
    # A private mirror of the pipeline image still reads as one: the hint is a
    # substring of the repository, not an exact match on ours.
    assert image_runs_ocr_pipeline(got)


def test_a_bare_block_still_answers_something_launchable():
    """No engine, no image: whatever pod_plan would default to, never a crash."""
    got = pod_image_for_entry(_entry(), model_key="datalab-to/surya-ocr-2")
    assert got
    assert not image_runs_ocr_pipeline(got)


def test_it_takes_an_empty_entry_without_raising():
    assert pod_image_for_entry({}) is not None


def test_pod_plan_and_the_gate_cannot_disagree():
    """The two readings of the one switch come from the same function.

    The defect was never a wrong rule -- it was two places applying the rule to
    different inputs. Pin that `pod_plan` resolves the image through this
    resolver, so a future change to one cannot leave the other behind.
    """
    import inspect
    from codai.api import runpod_worker as rw
    src = inspect.getsource(rw.pod_plan)
    assert "pod_image_for_entry(" in src


# --------------------------------------------------------------------------
# pod_pipeline_model
# --------------------------------------------------------------------------

def _cfg(**kw):
    base = {"default_engine": "surya", "surya_serve": "model",
            "surya_model_id": "datalab-to/surya-ocr-2"}
    base.update(kw)
    return types.SimpleNamespace(**base)


@pytest.fixture(autouse=True)
def _forget_what_was_said():
    om._PIPELINE_GATE_SAID.clear()
    yield
    om._PIPELINE_GATE_SAID.clear()


def _with_entry(monkeypatch, entry):
    from codai.api import runpod_worker as rw
    monkeypatch.setattr(rw, "_model_entry", lambda _mid: entry)


def test_the_gate_opens_for_an_engine_named_image(monkeypatch):
    """THE regression: the configuration in production, which answered ''."""
    _with_entry(monkeypatch, _entry(engine="coderai-vllm-ocr"))
    assert om.pod_pipeline_model(_cfg(), "surya") == "datalab-to/surya-ocr-2"


def test_the_gate_stays_shut_for_a_model_only_image(monkeypatch):
    _with_entry(monkeypatch, _entry(engine="coderai-vllm"))
    assert om.pod_pipeline_model(_cfg(), "surya") == ""


def test_opening_the_gate_is_announced_with_the_image(monkeypatch, capsys):
    _with_entry(monkeypatch, _entry(engine="coderai-vllm-ocr"))
    om.pod_pipeline_model(_cfg(), "surya")
    out = capsys.readouterr().out
    assert OCR_IMAGE in out
    assert "never rasterises" in out


def test_keeping_it_shut_says_so_and_says_how_to_change_it(monkeypatch, capsys):
    """Silence here is what cost two days: the local path looks like success."""
    _with_entry(monkeypatch, _entry(engine="coderai-vllm"))
    om.pod_pipeline_model(_cfg(), "surya")
    out = capsys.readouterr().out
    assert "THIS BOX rasterises" in out
    assert "coderai-vllm-ocr" in out


def test_it_says_it_once_not_once_per_document(monkeypatch, capsys):
    _with_entry(monkeypatch, _entry(engine="coderai-vllm-ocr"))
    for _ in range(25):
        om.pod_pipeline_model(_cfg(), "surya")
    assert capsys.readouterr().out.count("never rasterises") == 1


def test_a_changed_image_is_announced_again(monkeypatch, capsys):
    """Said once per ANSWER, so flipping the switch is visible without a restart."""
    _with_entry(monkeypatch, _entry(engine="coderai-vllm"))
    om.pod_pipeline_model(_cfg(), "surya")
    _with_entry(monkeypatch, _entry(engine="coderai-vllm-ocr"))
    om.pod_pipeline_model(_cfg(), "surya")
    out = capsys.readouterr().out
    assert "THIS BOX rasterises" in out
    assert "never rasterises" in out


def test_a_broken_lookup_reports_itself_instead_of_answering_no(monkeypatch, capsys):
    from codai.api import runpod_worker as rw

    def _boom(_mid):
        raise RuntimeError("models.json is unreadable")

    monkeypatch.setattr(rw, "_model_entry", _boom)
    assert om.pod_pipeline_model(_cfg(), "surya") == ""
    out = capsys.readouterr().out
    assert "could not decide" in out
    assert "models.json is unreadable" in out


def test_chandra_and_olmocr_go_through_the_same_gate(monkeypatch):
    """The gate is per engine; all three VLM engines can run pod-side."""
    _with_entry(monkeypatch, _entry(engine="coderai-vllm-ocr"))
    cfg = _cfg(chandra_serve="model", chandra_model_id="datalab-to/chandra-ocr-2",
               olmocr_serve="model", olmocr_model_id="allenai/olmOCR-2-7B-1025-FP8")
    assert om.pod_pipeline_model(cfg, "chandra") == "datalab-to/chandra-ocr-2"
    assert om.pod_pipeline_model(cfg, "olmocr") == "allenai/olmOCR-2-7B-1025-FP8"


def test_a_non_vlm_engine_never_goes_to_a_pod(monkeypatch):
    _with_entry(monkeypatch, _entry(engine="coderai-vllm-ocr"))
    assert om.pod_pipeline_model(_cfg(), "doctr") == ""


def test_local_serve_mode_never_goes_to_a_pod(monkeypatch):
    _with_entry(monkeypatch, _entry(engine="coderai-vllm-ocr"))
    assert om.pod_pipeline_model(_cfg(surya_serve="local"), "surya") == ""
