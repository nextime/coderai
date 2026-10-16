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


@pytest.fixture
def anyio_backend():
    return "asyncio"


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


# --------------------------------------------------------------------------
# A pod that cannot run the pipeline must cost latency, not the document
# --------------------------------------------------------------------------

class _Lease:
    kind = "pod"
    base = "https://10.0.0.1:1234/v1"
    api_key = "k"

    def release(self):
        pass


@pytest.fixture(autouse=True)
def _forget_set_aside_pools():
    om._POD_PIPELINE_UNUSABLE.clear()
    yield
    om._POD_PIPELINE_UNUSABLE.clear()


def _two_page_pdf_split(monkeypatch, pages=2):
    monkeypatch.setattr(om, "iter_pdf_page_pdfs",
                        lambda _d: [b"%%PDF-page" + str(i).encode() for i in range(pages)])


def _pods_answer(monkeypatch, fn):
    from codai.models import placement
    monkeypatch.setattr(placement, "choose", lambda *_a, **_k: _Lease())
    monkeypatch.setattr(placement, "spill_threshold", lambda *_a, **_k: 0)
    _stub_page_post(monkeypatch, fn)


def _stub_page_post(monkeypatch, fn):
    """Stub BOTH page-forward twins with one callable.

    The forward has two transports -- a coroutine on the event loop and the
    sync one on the pod thread pool -- and a test that stubs only one silently
    stops testing its own stub when the default changes. That is not
    hypothetical: when the async path landed, these tests started dialling
    10.0.0.1 for real and hung for 300 s instead of failing. Whatever `fn`
    does, including raising, it now does on either path.
    """
    async def _async(*a, **k):
        return fn(*a, **k)
    monkeypatch.setattr(om, "_post_page_to_pod", fn)
    monkeypatch.setattr(om, "_apost_page_to_pod", _async)


async def _read(mgr, cfg):
    return await mgr._read_on_pods(b"%PDF-1.4 x", "f.pdf", "application/pdf",
                                   "datalab-to/surya-ocr-2", dpi=200,
                                   page_opts=None, detect=None, cfg=cfg,
                                   engine_name="surya")


@pytest.mark.anyio
async def test_a_pod_that_refuses_the_pipeline_falls_back_instead_of_failing(
        monkeypatch, capsys):
    """THE outage: a 503 from the pod turned an optimisation into a failure."""
    _two_page_pdf_split(monkeypatch)

    def _refuse(*_a, **_k):
        raise om.OcrError("pod OCR at https://10.0.0.1:1234/v1/ocr answered "
                          "503: model is not served over HTTP by this install",
                          status=503)

    _pods_answer(monkeypatch, _refuse)
    mgr = om.OcrManager()
    assert await _read(mgr, types.SimpleNamespace()) is None
    out = capsys.readouterr().out
    assert "REFUSE to run it" in out
    assert "not served over HTTP" in out          # the POD's own words


@pytest.mark.anyio
async def test_a_refusal_stops_the_rest_of_the_document(monkeypatch):
    """One failed round trip, not one per page of an 1800-page filing."""
    _two_page_pdf_split(monkeypatch, pages=40)
    calls = []

    def _refuse(*_a, **_k):
        calls.append(1)
        raise om.OcrError("pod OCR answered 503: nope", status=503)

    _pods_answer(monkeypatch, _refuse)
    mgr = om.OcrManager()
    assert await _read(mgr, types.SimpleNamespace()) is None
    assert len(calls) < 40


@pytest.mark.anyio
async def test_the_pool_is_set_aside_so_the_next_document_does_not_retry(
        monkeypatch):
    _two_page_pdf_split(monkeypatch)
    calls = []

    def _refuse(*_a, **_k):
        calls.append(1)
        raise om.OcrError("pod OCR answered 503: nope", status=503)

    _pods_answer(monkeypatch, _refuse)
    mgr = om.OcrManager()
    await _read(mgr, types.SimpleNamespace())
    before = len(calls)
    await _read(mgr, types.SimpleNamespace())
    assert len(calls) == before, "second document re-asked a pool already set aside"


@pytest.mark.anyio
async def test_the_pool_is_tried_again_once_the_cooldown_passes(monkeypatch):
    """A rebuilt image must take effect without restarting the orchestrator."""
    _two_page_pdf_split(monkeypatch)
    calls = []

    def _refuse(*_a, **_k):
        calls.append(1)
        raise om.OcrError("pod OCR answered 503: nope", status=503)

    _pods_answer(monkeypatch, _refuse)
    mgr = om.OcrManager()
    await _read(mgr, types.SimpleNamespace())
    om._POD_PIPELINE_UNUSABLE["datalab-to/surya-ocr-2"] = (0.0, "stale")
    before = len(calls)
    await _read(mgr, types.SimpleNamespace())
    assert len(calls) > before


@pytest.mark.anyio
async def test_a_real_page_failure_still_fails(monkeypatch):
    """Falling back on ANY error would hide a broken page behind a slow one."""
    _two_page_pdf_split(monkeypatch)

    def _boom(*_a, **_k):
        raise om.OcrError("pod OCR answered 400: that is not a PDF", status=400)

    _pods_answer(monkeypatch, _boom)
    mgr = om.OcrManager()
    with pytest.raises(om.OcrError):
        await _read(mgr, types.SimpleNamespace())


@pytest.mark.anyio
async def test_a_working_pod_is_untouched(monkeypatch):
    _two_page_pdf_split(monkeypatch)
    _pods_answer(monkeypatch,
                 lambda *_a, **_k: {"engine": "surya",
                                    "pages": [{"text": "ok", "index": 0}]})
    mgr = om.OcrManager()
    got = await _read(mgr, types.SimpleNamespace())
    assert got is not None
    assert got[0] == "surya"
