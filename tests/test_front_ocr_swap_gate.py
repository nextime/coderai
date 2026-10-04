"""OCR takes its turn at the shared-GPU swap gate like any other GPU request — unless
the work is computed somewhere else (a cluster node, a RunPod pod, an external API).

Until now /v1/ocr bypassed the gate entirely: it is not in the router's _INFERENCE_PATHS
(it has no `model` field and none of the front's per-model queue/pin machinery applies),
so an OCR request could start a surya vLLM boot while an LLM was mid-generation on the
same card, and vice versa. The gate is what makes them alternate instead of contend.

The methods are exercised against a stub `self`: constructing a real FrontProxy would
need engines, config files and sockets, and none of the logic under test touches them.
"""

import asyncio
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codai.config import OcrConfig
from codai.frontproxy import router as _router
from codai.frontproxy.app import FrontProxy
from codai.frontproxy.reqqueue import GpuSwapGate


# ------------------------------------------------------------------ paths
def test_ocr_is_gpu_work_but_not_a_model_inference_path():
    for p in ("/v1/ocr", "/v1/ocr/batch", "/v1/ocr/"):
        assert _router.is_gpu_inference_path(p), p
        assert _router.is_ocr_path(p), p
        # Deliberately NOT an inference path: that predicate drives the per-model
        # queue, pins, spillover and keepalive, none of which fit OCR.
        assert not _router.is_inference_path(p), p


def test_the_model_paths_are_still_gpu_inference_paths():
    for p in ("/v1/chat/completions", "/v1/embeddings", "/v1/images/generations",
              "/v1/video/generations", "/v1/audio/transcriptions"):
        assert _router.is_inference_path(p) and _router.is_gpu_inference_path(p), p
        assert not _router.is_ocr_path(p), p


def test_schema_endpoints_are_not_gpu_work():
    for p in ("/v1/ocr/schemas", "/v1/ocr/schemas/foo", "/v1/models", "/admin"):
        assert not _router.is_ocr_path(p), p
        assert not _router.is_gpu_inference_path(p), p


# ------------------------------------------------------------------ owner key
def _stub(ocr=None, model_info=None):
    s = types.SimpleNamespace()
    s.config = types.SimpleNamespace(ocr=ocr if ocr is not None else OcrConfig())
    s._model_info = lambda m: (model_info or {})
    s._queue_key = lambda m: (m or "").lower()
    s._ocr_is_external = lambda: FrontProxy._ocr_is_external(s)
    return s


def test_all_ocr_requests_share_one_owner_on_the_card():
    """paddle and surya are one tenant as far as "who owns the GPU" goes — the OCR
    manager does its own eviction between its pools."""
    s = _stub()
    eng = types.SimpleNamespace(name="nvidia", remote=False)
    assert FrontProxy._swap_owner_key(s, eng, None, "/v1/ocr") == "ocr"
    assert FrontProxy._swap_owner_key(s, eng, "surya", "/v1/ocr/batch") == "ocr"
    # A model request still keys on the model, as before.
    assert FrontProxy._swap_owner_key(s, eng, "Qwen/Qwen3.5-9B", "/v1/chat/completions") \
        == "qwen/qwen3.5-9b"
    # No path given (older call site) → unchanged behaviour.
    assert FrontProxy._swap_owner_key(s, eng, "lisa", None) == "lisa"


# ------------------------------------------------------------------ off-box work
def test_a_cluster_node_computes_it_on_its_own_card():
    s = _stub()
    node = types.SimpleNamespace(name="node-b", remote=True)
    assert FrontProxy._computed_off_this_gpu(s, node, "lisa", "/v1/chat/completions")
    assert FrontProxy._computed_off_this_gpu(s, node, None, "/v1/ocr")


def test_a_runpod_model_uses_no_local_vram():
    s = _stub(model_info={"backend": "runpod"})
    eng = types.SimpleNamespace(name="nvidia", remote=False)
    assert FrontProxy._computed_off_this_gpu(s, eng, "big-model", "/v1/chat/completions")


def test_local_work_is_gated():
    s = _stub(model_info={"backend": "nvidia"})
    eng = types.SimpleNamespace(name="nvidia", remote=False)
    assert not FrontProxy._computed_off_this_gpu(s, eng, "lisa", "/v1/chat/completions")
    assert not FrontProxy._computed_off_this_gpu(s, eng, None, "/v1/ocr")


def test_ocr_served_entirely_by_external_endpoints_is_not_gated():
    """Every enabled engine points off-box, so no OCR request can touch this card."""
    o = OcrConfig(
        enabled=True, paddle_enabled=False, doctr_enabled=False,
        surya_enabled=True, surya_accept_license=True,
        surya_serve="llamacpp", surya_server_url="http://other-box:8080/v1",
        olmocr_enabled=True, olmocr_serve="server",
        olmocr_server_url="https://api.example/v1")
    s = _stub(ocr=o)
    eng = types.SimpleNamespace(name="nvidia", remote=False)
    assert FrontProxy._ocr_is_external(s)
    assert FrontProxy._computed_off_this_gpu(s, eng, None, "/v1/ocr")


def test_one_local_engine_is_enough_to_keep_ocr_gated():
    o = OcrConfig(
        enabled=True, paddle_enabled=True,       # local
        surya_enabled=True, surya_accept_license=True,
        surya_serve="llamacpp", surya_server_url="http://other-box:8080/v1")
    s = _stub(ocr=o)
    eng = types.SimpleNamespace(name="nvidia", remote=False)
    assert not FrontProxy._ocr_is_external(s)
    assert not FrontProxy._computed_off_this_gpu(s, eng, None, "/v1/ocr")


def test_surya_on_the_local_vllm_is_local_work():
    """vllm mode boots an instance ON THIS CARD — the case the whole fix is about."""
    o = OcrConfig(enabled=True, paddle_enabled=False, doctr_enabled=False,
                  surya_enabled=True, surya_accept_license=True, surya_serve="vllm")
    s = _stub(ocr=o)
    assert not FrontProxy._ocr_is_external(s)


def test_olmocr_in_model_mode_is_local_work():
    o = OcrConfig(enabled=True, paddle_enabled=False, doctr_enabled=False,
                  olmocr_enabled=True, olmocr_serve="model")
    s = _stub(ocr=o)
    assert not FrontProxy._ocr_is_external(s)


def test_no_enabled_engine_at_all_is_not_claimed_external():
    s = _stub(ocr=OcrConfig(enabled=True, paddle_enabled=False, doctr_enabled=False))
    assert not FrontProxy._ocr_is_external(s)


def test_a_license_gated_surya_does_not_count_as_a_local_engine():
    o = OcrConfig(enabled=True, paddle_enabled=False, doctr_enabled=False,
                  surya_enabled=True, surya_accept_license=False,  # not usable
                  olmocr_enabled=True, olmocr_serve="server",
                  olmocr_server_url="https://api.example/v1")
    s = _stub(ocr=o)
    assert FrontProxy._ocr_is_external(s)


# ------------------------------------------------------------------ alternation
def test_ocr_and_a_model_alternate_on_the_card():
    """The behaviour the gate buys: OCR waits for the model's batch, takes the card,
    and the model gets it back — turns, not contention."""
    gate = GpuSwapGate(cap=2)
    order = []

    async def _run(key, n):
        for _ in range(n):
            await gate.acquire(key)
            order.append(key)
            await asyncio.sleep(0)
            gate.release(key)

    async def _go():
        await asyncio.gather(_run("lisa", 4), _run("ocr", 4))

    asyncio.run(_go())
    assert len(order) == 8
    assert set(order) == {"lisa", "ocr"}
    # Neither side was starved: both finished all four turns, and the owner yielded
    # rather than running all four of its own first.
    assert order.count("lisa") == 4 and order.count("ocr") == 4
    assert order[0] != order[-1] or len(set(order[:4])) == 2


def test_the_owner_keeps_the_card_when_nobody_else_wants_it():
    gate = GpuSwapGate(cap=2)

    async def _go():
        for _ in range(5):          # past the cap, with no other waiter
            await gate.acquire("ocr")
            gate.release("ocr")
        return True

    assert asyncio.run(_go())
