"""A3/A4/A5: an advertised capability, an inflated estimate, a silent failure.

The three share a shape: the server held a belief about a model that nothing
checked against the model itself, and the client paid for it.
"""
import json
import os
import types

import pytest


# ============================================================ A3: FLUX.2 edit
def test_flux2_reports_instruction_edit():
    from codai.models.capabilities import detect_model_capabilities
    for mid in ("FLUX.2-klein-4B", "black-forest-labs/FLUX.2-klein-4B",
                "FLUX.2-dev", "flux2-klein", "FLUX-2-pro"):
        caps = detect_model_capabilities(mid).to_list()
        assert "instruction_edit" in caps, (mid, caps)
        assert "image_generation" in caps, (mid, caps)
        # It is not a denoising img2img model: advertising `strength` would make
        # clients show a slider for an argument the pipeline has not got.
        assert "inpainting" not in caps, (mid, caps)


def test_flux1_is_not_swept_up():
    from codai.models.capabilities import detect_model_capabilities
    caps = detect_model_capabilities("FLUX.1-dev").to_list()
    assert "instruction_edit" not in caps, caps
    # Kontext is the FLUX.1 that does edit by instruction, and still does.
    assert "instruction_edit" in detect_model_capabilities(
        "FLUX.1-Kontext-dev").to_list()


def test_the_pipeline_signature_agrees_with_the_advertised_capability():
    # The claim is only worth making if the runtime discriminator agrees:
    # images._is_instruction_edit decides on (image yes, strength no).
    diffusers = pytest.importorskip("diffusers")
    from codai.api.images import _is_instruction_edit
    for name in ("Flux2Pipeline", "Flux2KleinPipeline"):
        cls = getattr(diffusers, name, None)
        if cls is None:
            pytest.skip(f"{name} not in this diffusers")
        assert _is_instruction_edit(cls) is True, name


# ================================================= A4: pre-quantized estimate
def _mgr():
    from codai.models.manager import MultiModelManager
    m = object.__new__(MultiModelManager)
    m._prequantized_cache = {}
    m._storage_dtype_cache = {}
    return m


def test_a_bnb_checkpoint_is_recognised_from_its_config(tmp_path):
    d = tmp_path / "zimage"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({
        "_class_name": "ZImagePipeline",
        "quantization_config": {"quant_method": "bitsandbytes_4bit"},
    }))
    m = _mgr()
    assert m._is_prequantized_checkpoint("image:x", str(d), {"path": str(d)}) is True


def test_a_quantized_pipeline_component_counts(tmp_path):
    d = tmp_path / "pipe"
    (d / "transformer").mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"_class_name": "P"}))
    (d / "transformer" / "config.json").write_text(json.dumps({
        "quantization_config": {"quant_method": "gptq"}}))
    m = _mgr()
    assert m._is_prequantized_checkpoint("image:x", str(d), {"path": str(d)}) is True


def test_a_plain_checkpoint_is_not_flagged(tmp_path):
    d = tmp_path / "plain"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"_class_name": "P",
                                               "torch_dtype": "bfloat16"}))
    m = _mgr()
    assert m._is_prequantized_checkpoint("image:x", str(d), {"path": str(d)}) is False


@pytest.mark.parametrize("repo,expect", [
    ("unsloth/Z-Image-Turbo-unsloth-bnb-4bit", True),
    ("TheBloke/Model-AWQ", True),
    ("org/Model-GPTQ", True),
    ("org/Model-int8", True),
    ("org/Model-fp8", True),
    ("Qwen/Qwen3-8B", False),
    ("black-forest-labs/FLUX.2-klein-4B", False),   # "4B" is a size, not 4-bit
])
def test_the_name_is_the_fallback_when_no_config_is_reachable(repo, expect):
    m = _mgr()
    # A repo id that is not in any local cache: no config.json is reachable, so
    # the name decides.
    got = m._is_prequantized_checkpoint("text:" + repo, repo, {"path": repo})
    assert got is expect, (repo, got)


def test_a_prequantized_checkpoint_is_not_inflated_to_load_precision():
    """The A4 regression itself, at the arithmetic.

    16.3 GB of 4-bit weights (~1.17 bytes/elem on disk) were normalised to a
    2-byte load precision -- a 1.70x factor -- and estimated at 30.03 GB
    against a 24 GB card. Nothing can free 30 GB of 24, so every still was
    generated with the pipeline partly on the CPU: ~72 s for one image from a
    few-step turbo model.
    """
    from codai.models.manager import MultiModelManager
    m = object.__new__(MultiModelManager)
    m._prequantized_cache = {}
    m._storage_dtype_cache = {}
    m._measured_vram_gb = {}
    m.config = {}

    key = "image:unsloth/Z-Image-Turbo-unsloth-bnb-4bit"
    cfg = {"path": "unsloth/Z-Image-Turbo-unsloth-bnb-4bit",
           "used_vram_gb": 16.316, "model_type": "image_models"}

    m._config_for_model = types.MethodType(lambda self, k: cfg, m)
    m._resolve_config_key = types.MethodType(lambda self, k: key, m)
    m._storage_bytes_per_elem = types.MethodType(lambda self, *a: 1.1736, m)
    m._quantized_cache_gb = types.MethodType(lambda self, *a: 0.0, m)
    m._config_lora_vram_gb = types.MethodType(lambda self, c: 0.0, m)
    m._runtime_reserve_gb = types.MethodType(lambda self, c, k, g: 2.0, m)
    m._longcat_resident_gb = types.MethodType(lambda self, *a: 0.0, m)
    m._effective_quant_multiplier = types.MethodType(lambda self, *a: 1.0, m)

    est = m._get_model_used_vram_gb(key)
    # 16.316 + 2.0 reserve, NOT 16.316 * 1.70 + reserve = 30.03.
    assert 17.0 <= est <= 20.0, est
    assert est < 24.0, f"still does not fit a 24 GB card: {est}"


# ============================================== A5: the silent empty answer
def test_a_failed_generation_raises_instead_of_returning_empty():
    from codai.backends.cuda import GenerationFailed, _generation_failed
    exc = _generation_failed(
        RuntimeError("Pointer argument (at 0) cannot be accessed from Triton "
                     "(cpu tensor?)"))
    assert isinstance(exc, GenerationFailed)
    assert isinstance(exc, RuntimeError)   # so existing handlers still see it
    msg = str(exc)
    assert "Triton" in msg
    # The hint names the cause, not the symptom: an operator reading "Pointer
    # argument" alone has nowhere to go.
    assert "resident on the CPU" in msg, msg
    assert "used_vram_gb" in msg, msg


def test_the_first_error_leads_and_duplicates_collapse():
    from codai.backends.cuda import _generation_failed
    msg = str(_generation_failed(RuntimeError("first"), RuntimeError("second"),
                                 RuntimeError("second")))
    assert msg.index("first") < msg.index("second")
    assert msg.count("second") == 1


def test_an_oom_gets_its_own_hint():
    from codai.backends.cuda import _generation_failed
    assert "ran out of memory" in str(
        _generation_failed(RuntimeError("CUDA out of memory")))


def test_the_empty_answer_is_no_longer_reachable_from_the_ladder():
    # The three-rung ladder is still there -- it is real recovery -- but the
    # bottom rung no longer assigns "" and returns. Asserted on the source so
    # the guarantee survives a refactor of the ladder itself.
    import inspect
    from codai.backends import cuda
    src = inspect.getsource(cuda.NvidiaBackend.generate_chat)
    assert "no-cache fallback failed" in src
    tail = src[src.index("no-cache fallback failed"):]
    assert "_generation_failed" in tail
    assert 'generated_text = ""' not in tail


class _FakeQuantModel:
    def __init__(self, device_map):
        self.hf_device_map = device_map
        self.config = types.SimpleNamespace(quantization_config={"q": 1})


def test_a_quantized_model_offloaded_to_cpu_is_refused():
    from codai.backends.cuda import NvidiaBackend
    b = object.__new__(NvidiaBackend)
    model = _FakeQuantModel({"layers.0": 0, "layers.1": "cpu", "layers.2": "disk"})
    with pytest.raises(RuntimeError) as e:
        b._reject_offloaded_quantized(model, "Qwen/Qwen3.5-9B",
                                      {"quantization_config": object()})
    msg = str(e.value)
    assert "quantized" in msg
    assert "2 of its 3 modules" in msg, msg
    assert "fail on every request" in msg


def test_a_quantized_model_fully_on_the_gpu_is_accepted():
    from codai.backends.cuda import NvidiaBackend
    b = object.__new__(NvidiaBackend)
    model = _FakeQuantModel({"layers.0": 0, "layers.1": 0})
    b._reject_offloaded_quantized(model, "m", {"quantization_config": object()})


def test_an_unquantized_model_may_still_offload():
    # Offload is a legitimate, merely slower path for dense weights; only
    # quantized kernels cannot read them.
    from codai.backends.cuda import NvidiaBackend
    b = object.__new__(NvidiaBackend)
    model = types.SimpleNamespace(
        hf_device_map={"layers.0": 0, "layers.1": "cpu"},
        config=types.SimpleNamespace(quantization_config=None))
    b._reject_offloaded_quantized(model, "m", {})
