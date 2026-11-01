"""An f32 default filled a 24 GB card, and a decode OOM was called a load failure.

Measured on zeiss, 2026-10-31, mid test run:

* ``pornmaster-pro-pony`` (SDXL, no ``precision`` on its entry) loaded at
  **f32**, completed 30/30 denoising steps, and then died on the VAE decode --
  "Tried to allocate 1008.00 MiB ... 704.69 MiB is free", with 22.12 GiB of a
  23.56 GiB card held by its own weights. The same architecture at half width
  measures 7.49 GB.
* ``Krea-2-Edit`` (GGUF Q4_K_M transformer + 4-bit text encoder, bf16) loaded
  correctly, completed 8/8 steps, and died the same way on 190 MiB.

Two things were wrong.

1. The bf16 default added for an undeclared precision was unreachable: the
   config layer always supplied ``image.precision = "f32"``, so there was no
   such thing as "undeclared" by the time the loader looked. The default is now
   ``"auto"`` and resolves from the card -- half on a card of 32 GB or less,
   f32 above it, since a card big enough for both loses nothing by upcasting.

2. Both failures were reported as ``Failed to load image model 'X'. Errors:
   diffusers: CUDA out of memory`` -- because the generate call sat inside the
   load's ``except Exception``, which also fell through to the sd.cpp backend.
   The models had in fact loaded and run. Two separate readers concluded from
   that wording that VRAM eviction was leaking; it was not, and the real causes
   went unexamined for it.
"""
import ast

import pytest

from codai.api import images as images_mod
from codai.config import ImageConfig


SRC = open("codai/api/images.py").read()


# --------------------------------------------------------------------------- #
#  the default is a decision, not a constant
# --------------------------------------------------------------------------- #

def test_the_global_image_precision_default_is_auto_not_f32():
    # Not cosmetic: while this said "f32" the loader's own "card states none"
    # branch could never run, because the config layer filled the key in.
    assert ImageConfig().precision == "auto"


@pytest.mark.parametrize("vram_gb,expected", [
    (8.0, "bf16"),      # RX 580 / small
    (23.56, "bf16"),    # the 3090 this was measured on
    (32.0, "bf16"),     # boundary: "32 or less" includes 32
    (32.1, "f32"),      # just over
    (48.0, "f32"),      # A6000
    (80.0, "f32"),      # A100
])
def test_auto_precision_follows_the_card_size(monkeypatch, vram_gb, expected):
    monkeypatch.setattr(images_mod, "_largest_visible_vram_gb", lambda: vram_gb)
    precision, why = images_mod._auto_precision()
    assert precision == expected
    assert why, "the choice has to explain itself in the log"


def test_an_unknown_card_is_treated_as_small():
    # Guessing half on a big card wastes nothing; guessing f32 on a small one
    # is the failure this whole module is about.
    import unittest.mock as mock
    with mock.patch.object(images_mod, "_largest_visible_vram_gb", lambda: 0.0):
        assert images_mod._auto_precision()[0] == "bf16"


def test_the_half_type_is_bf16_because_fp16_blacked_out_z_image():
    import torch
    import unittest.mock as mock
    with mock.patch.object(images_mod, "_largest_visible_vram_gb", lambda: 24.0):
        assert images_mod._auto_precision()[0] == "bf16"
    # Same size, four orders of magnitude more range. See
    # tests/test_black_image_and_precision.py for what fp16 cost.
    assert torch.finfo(torch.bfloat16).bits == torch.finfo(torch.float16).bits
    assert torch.finfo(torch.bfloat16).max > torch.finfo(torch.float16).max * 1000


def test_largest_visible_vram_is_the_biggest_card_not_the_sum(monkeypatch):
    # A pipeline loads onto ONE card unless explicitly sharded, so two 24 GB
    # cards must not read as a 48 GB one and tip the policy over to f32.
    monkeypatch.setattr(
        "codai.models.gpu_query.visible_gpu_memory",
        lambda: [{"total": 24e9}, {"total": 24e9}])
    assert images_mod._largest_visible_vram_gb() == pytest.approx(24.0, abs=0.1)


def test_an_explicit_precision_is_still_honoured():
    # The policy changes the DEFAULT. An operator who wrote f32 gets f32.
    body = SRC[SRC.index("_precision_cfg = str(_cfg('precision')"):][:400]
    assert "else:" in body and "precision = _precision_cfg" in body


# --------------------------------------------------------------------------- #
#  a decode failure is not a load failure
# --------------------------------------------------------------------------- #

def test_generation_runs_outside_the_load_exception_handler():
    # The structural fix. If the generate call is inside the try that catches
    # load errors, a decode OOM is filed as a load error again.
    tree = ast.parse(SRC)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef))
              and n.name == "create_image_generation")

    def generate_calls(node):
        return [c for c in ast.walk(node)
                if isinstance(c, ast.Call)
                and getattr(c.func, "id", None) == "_generate_with_diffusers"]

    for handler_owner in ast.walk(fn):
        if not isinstance(handler_owner, ast.Try):
            continue
        names = {getattr(h.type, "id", None) for h in handler_owner.handlers}
        if "ImportError" not in names:
            continue        # not the load try
        for call in generate_calls(handler_owner):
            pytest.fail("_generate_with_diffusers is back inside the load try; "
                        "a VAE-decode OOM will be reported as "
                        "'Failed to load image model' again")


def test_a_generation_failure_says_the_model_loaded():
    assert "the model loaded and ran, so this is not a " in SRC
    assert "load failure" in SRC


def test_a_decode_oom_is_503_and_names_the_decode():
    body = SRC[SRC.index("if _loaded_diffusers is not None:"):][:2000]
    assert "503 if _oom else 500" in body
    assert "VAE decode" in body


@pytest.mark.parametrize("exc,expected", [
    (RuntimeError("CUDA out of memory. Tried to allocate 1008.00 MiB"), True),
    (RuntimeError("Out Of Memory"), True),
    # Measured on Krea-2-RealDream: a 31 GB pipeline on a 24 GB card, where the
    # card filled before cuBLAS could take its workspace. The allocator never
    # got to say "out of memory", so this was answered 500 -- a server fault --
    # and described as something other than a full card.
    (RuntimeError("CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling "
                  "`cublasCreate(handle)`"), True),
    (RuntimeError("cuDNN error: CUDNN_STATUS_ALLOC_FAILED"), True),
    (RuntimeError("cudaErrorMemoryAllocation"), True),
    (RuntimeError("failed to allocate 2.00 GiB on device 0"), True),
    (ValueError("size must be a multiple of 8"), False),
    (RuntimeError("CUDA error: device-side assert triggered"), False),
    # Not every cuBLAS complaint is a full card, and a mis-set compute
    # capability must not be answered "come back later".
    (RuntimeError("CUBLAS_STATUS_EXECUTION_FAILED"), False),
])
def test_is_cuda_oom_matches_the_wordings_this_actually_arrives_as(exc, expected):
    assert images_mod._is_cuda_oom(exc) is expected


def test_an_allocator_failure_from_any_library_is_still_a_full_card():
    """The markers are a declared list and not a chain of ifs, so the next
    library that reports its own allocation failure is one line."""
    assert "alloc_failed" in images_mod._CUDA_OOM_MARKERS
    assert "out of memory" in images_mod._CUDA_OOM_MARKERS


def test_the_oom_detail_names_cublas_as_well_as_the_decode():
    """Both are real points of failure and they are not the same place; a
    message naming only the decode sent the last one looking in the VAE."""
    body = SRC[SRC.index("if _loaded_diffusers is not None:"):][:2000]
    assert "cuBLAS" in body and "VAE decode" in body


def test_is_cuda_oom_matches_the_torch_type_too():
    torch = pytest.importorskip("torch")
    oom = getattr(torch, "OutOfMemoryError", None)
    if oom is None:
        pytest.skip("this torch has no OutOfMemoryError type")
    assert images_mod._is_cuda_oom(oom("ran out")) is True


# --------------------------------------------------------------------------- #
#  tiling is what bounds a single large decode
# --------------------------------------------------------------------------- #

def test_vae_tiling_is_honoured_from_the_model_config():
    body = SRC[SRC.index("# Enable memory optimizations"):][:2200]
    assert "(model_config or {}).get('vae_tiling')" in body, \
        "tiling has to be readable from the model's own config"
    assert "enable_tiling" in body and "enable_vae_tiling" in body, \
        "both the vae-level and pipeline-level diffusers APIs"


def test_slicing_alone_is_not_enough_and_the_code_says_why():
    body = SRC[SRC.index("# Enable memory optimizations"):][:2200]
    # Slicing splits a BATCH; it does nothing for n=1, which is the case that
    # OOMed. Whoever next trims these comments should know that.
    assert "n=1" in body
