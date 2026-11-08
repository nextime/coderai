# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""A rented pod whose MODEL CANNOT LOAD must never be kept, let alone billed.

The boot teardown has always handled a pod that fails to come up. The gap was
one step later: the pod's HTTP front answers, the warm-up inference comes back
503 `Model 'x' failed to load: …`, and `_warmup` forgave it — "serving anyway"
— because most warm-up failures are harmless shape mismatches. The pool then
took a pod with no engine behind its address.

Measured 2026-11-08. A 32B FP8 vision model was rented an L40S, which reports
~44.4 GiB where an RTX 6000 Ada of the same nominal 48 GB reports 47.5:

    Available KV cache memory: 1.94 GiB
    ValueError: To serve at least one request with the model's max seq len
    (16384), 2.0 GiB KV cache is needed … estimated maximum model length 15888
    RuntimeError: Engine core initialization failed
    [http] 503 on POST /v1/chat/completions: "Model 'qwen3vl32-fp8' failed to
    load: vLLM exited (code 1) before becoming ready."

The pod billed $0.79/hr for over an hour. Nothing reaped it: the idle timer
does not run on a pod the pool believes in, the boot teardown was behind it,
and the engineless check waits a full boot window (1800s here) before acting.

So: a load failure raises, which puts the pod on the boot-failure path that
already terminates it and retries on the next candidate machine — possibly the
bigger card that fits. Everything else still serves.
"""
import types

import pytest

from codai.api import runpod_worker as rw


# --- the discriminator ------------------------------------------------------

@pytest.mark.parametrize("body", [
    "Model 'qwen3vl32-fp8' failed to load: vLLM exited (code 1) before "
    "becoming ready. Last output: ValueError: To serve at least one request",
    "Model 'x' could not be loaded after waiting. Another model may be using "
    "all available VRAM.",
    "Error loading model Qwen/Qwen3-VL-32B-Instruct-FP8",
    "Model 'diogene-alpha' failed to load: Unknown backend: cpu",
    "OCR engine 'chandra' failed to load: no such file",
])
def test_a_dead_engine_is_recognised(body):
    assert rw.warmup_fatal_reason(503, body)
    assert rw.warmup_fatal_reason(500, body)


@pytest.mark.parametrize("status,body", [
    # The shape mismatches this function exists NOT to kill pods over.
    (404, "Model 'BAAI/bge-m3' is a embedding model and cannot be used for "
          "text generation. Use the embedding endpoint instead."),
    (400, "max_tokens must be a positive integer"),
    (422, "field required: messages"),
    # A 503 that is NOT a load failure: an admission queue momentarily full.
    (503, "queue is full, retry"),
    (503, "Service temporarily unavailable"),
    (500, "internal error while sampling"),
])
def test_a_benign_failure_keeps_the_pod(status, body):
    assert rw.warmup_fatal_reason(status, body) == ""


def test_a_load_failure_below_500_is_not_grounds_to_kill():
    # Only the pod's own 5xx says the engine is gone. A 404 naming a model
    # that "failed to load" is the catalogue talking about something else.
    assert rw.warmup_fatal_reason(404, "model failed to load") == ""


def test_the_reason_is_carried_but_bounded():
    reason = rw.warmup_fatal_reason(503, "Model 'x' failed to load: " + "y" * 900)
    assert reason.startswith("Model 'x' failed to load:")
    assert len(reason) <= 300


def test_the_reason_is_one_line():
    # It goes into an exception message and a log line; a pasted vLLM
    # traceback must not break either into pieces.
    reason = rw.warmup_fatal_reason(
        503, "Model 'x' failed to load:\n  vLLM exited\n\n  (code 1)\n")
    assert "\n" not in reason
    assert reason == "Model 'x' failed to load: vLLM exited (code 1)"


# --- what _warmup does with it ---------------------------------------------

class _Reply:
    def __init__(self, status, text):
        self.status_code, self.text = status, text


def _pool(monkeypatch, reply):
    """The least of a pool `_warmup` touches, plus a stubbed pod HTTP."""
    sent = []

    def _post(url, json=None, headers=None, timeout=None):
        sent.append(url)
        return reply

    monkeypatch.setattr(rw.pod_http, "post", _post)
    stub = types.SimpleNamespace(
        mcfg=types.SimpleNamespace(warmup_on_boot=True, warmup_timeout_s=900),
        served="qwen3vl32-fp8", model_key="qwen3vl32-fp8", entry={},
        api_key="", pod_key=lambda _p: "", _pod_by_id=lambda _i: None)
    return stub, sent


def test_a_pod_that_cannot_load_the_model_raises(monkeypatch):
    # Verbatim from the pod that billed for an hour.
    body = ("Model 'qwen3vl32-fp8' failed to load: vLLM exited (code 1) "
            "before becoming ready. Last output: ValueError: To serve at "
            "least one request with the model's max seq len (16384), 2.0 GiB "
            "KV cache is needed, which is larger than the available KV cache "
            "memory (1.94 GiB).")
    stub, sent = _pool(monkeypatch, _Reply(503, body))
    from codai.api.runpod_client import RunpodError
    with pytest.raises(RunpodError) as got:
        rw.RunpodPodPool._warmup(stub, "7slhvnpohf5f8i", "http://pod:8000")
    # The message has to name the pod and say why, because this is what the
    # operator reads when a boot retries on another machine.
    assert "7slhvnpohf5f8i" in str(got.value)
    assert "cannot load" in str(got.value)
    assert "1.94 GiB" in str(got.value)
    assert sent, "it must have actually asked the pod"


def test_a_shape_mismatch_still_serves(monkeypatch):
    stub, _ = _pool(monkeypatch, _Reply(
        404, "Model 'BAAI/bge-m3' is a embedding model and cannot be used "
             "for text generation."))
    assert rw.RunpodPodPool._warmup(stub, "p", "http://pod:8000") is False


def test_a_transient_503_still_serves(monkeypatch):
    stub, _ = _pool(monkeypatch, _Reply(503, "queue is full, retry"))
    assert rw.RunpodPodPool._warmup(stub, "p", "http://pod:8000") is False


def test_a_good_warmup_is_unchanged(monkeypatch):
    stub, _ = _pool(monkeypatch, _Reply(200, '{"choices":[]}'))
    assert rw.RunpodPodPool._warmup(stub, "p", "http://pod:8000") is True


def test_an_unreachable_pod_is_still_not_killed_here(monkeypatch):
    # Silence is the quarantine path's business: a refused connection is as
    # likely to be a busy pod as a dead one, and killing on it once destroyed
    # eight healthy A40s.
    def _boom(*a, **k):
        raise OSError("connection refused")

    monkeypatch.setattr(rw.pod_http, "post", _boom)
    stub = types.SimpleNamespace(
        mcfg=types.SimpleNamespace(warmup_on_boot=True, warmup_timeout_s=900),
        served="x", model_key="x", entry={}, api_key="",
        pod_key=lambda _p: "", _pod_by_id=lambda _i: None)
    assert rw.RunpodPodPool._warmup(stub, "p", "http://pod:8000") is False
