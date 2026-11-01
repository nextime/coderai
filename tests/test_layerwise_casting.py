# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""An fp8 file that fits the card on disk should fit it in VRAM.

A community fine-tune is usually published in fp8: one safetensors file of
about 13 GB for a 12B transformer. Loaded with ``torch_dtype=bfloat16`` it
becomes 26 GB, because that is what upcasting means. Measured 2026-11-01:

* Krea-2-RealDream (``civitai:3396368:3304483``, 12.9 GB on disk) measured
  31.07 GB and died mid-generation on cuBLAS's workspace;
* Z-Image-Turbo-IntoRealism (fp8, 5.9 GB on disk) is upcast the same way.

There is no GGUF of either to fall back to -- searched, and the only RealDream
Krea-2 repository publishes fp8 and nvfp4 safetensors, not GGUF. So store
narrow and compute wide instead: diffusers' layerwise casting keeps the weights
at fp8 and upcasts each module on its own forward. On Ampere that is the only
thing fp8 can be, since the hardware has no fp8 matmul at all.
"""

import pytest

torch = pytest.importorskip("torch")

from codai.models import layerwise


_HAS_F8 = hasattr(torch, "float8_e4m3fn")
requires_f8 = pytest.mark.skipif(not _HAS_F8, reason="this torch has no fp8")


# --------------------------------------------------------------------------- #
#  reading the config
# --------------------------------------------------------------------------- #

def test_a_model_that_asks_for_nothing_is_untouched():
    for cfg in ({}, None, {"layerwise_casting": None},
                {"layerwise_casting": False}, {"layerwise_casting": ""},
                {"layerwise_casting": {}}):
        assert layerwise.spec(cfg) == ({}, "")


@requires_f8
@pytest.mark.parametrize("value", ["f8", "fp8", "F8", "float8_e4m3fn",
                                   "e4m3", True])
def test_the_usual_spellings_all_mean_e4m3(value):
    got, desc = layerwise.spec({"layerwise_casting": value})
    assert got
    assert set(got.values()) == {torch.float8_e4m3fn}
    assert "float8_e4m3fn" in desc


@requires_f8
def test_the_other_fp8_flavour_is_available_too():
    got, _ = layerwise.spec({"layerwise_casting": "f8_e5m2"})
    assert set(got.values()) == {torch.float8_e5m2}


def test_a_dtype_nobody_has_is_not_guessed_at():
    assert layerwise.spec({"layerwise_casting": "int3"}) == ({}, "")


@requires_f8
def test_the_heavy_components_are_the_default_set():
    got, _ = layerwise.spec({"layerwise_casting": "f8"})
    assert "transformer" in got and "unet" in got
    assert "text_encoder" in got
    assert "vae" not in got, \
        "the VAE is the last thing to touch the latents; a narrow decode shows"


@requires_f8
def test_one_component_can_be_named_on_its_own():
    got, _ = layerwise.spec({"layerwise_casting": {"transformer": "f8"}})
    assert list(got) == ["transformer"]
    assert layerwise.storage_dtype_for(
        {"layerwise_casting": {"transformer": "f8"}}, "transformer") \
        is torch.float8_e4m3fn
    assert layerwise.storage_dtype_for(
        {"layerwise_casting": {"transformer": "f8"}}, "vae") is None


# --------------------------------------------------------------------------- #
#  applying it
# --------------------------------------------------------------------------- #

class _Comp:
    def __init__(self, fails=False):
        self.calls = []
        self._fails = fails

    def enable_layerwise_casting(self, storage_dtype=None, compute_dtype=None):
        if self._fails:
            raise RuntimeError("no hooks here")
        self.calls.append((storage_dtype, compute_dtype))


class _Pipe:
    def __init__(self, **components):
        for k, v in components.items():
            setattr(self, k, v)


@requires_f8
def test_it_casts_the_configured_component_and_says_so():
    t = _Comp()
    pipe = _Pipe(transformer=t, vae=_Comp())
    note = layerwise.apply(pipe, {"layerwise_casting": {"transformer": "f8"}},
                           torch.bfloat16)
    assert t.calls == [(torch.float8_e4m3fn, torch.bfloat16)]
    assert "transformer" in note and "bfloat16" in note


@requires_f8
def test_a_component_the_pipeline_does_not_have_is_simply_skipped():
    pipe = _Pipe(transformer=_Comp())
    note = layerwise.apply(pipe, {"layerwise_casting": "f8"}, torch.bfloat16)
    assert "transformer" in note
    # unet / text_encoder are absent from this pipeline and must not appear as
    # failures -- a pipeline is not broken for lacking a component.
    assert "unet" not in note


@requires_f8
def test_a_component_that_cannot_cast_is_reported_not_raised():
    """A VRAM optimisation must never be why a generation fails."""
    pipe = _Pipe(transformer=_Comp(fails=True))
    note = layerwise.apply(pipe, {"layerwise_casting": "f8"}, torch.bfloat16)
    assert "not applied to" in note
    assert "no hooks here" in note


@requires_f8
def test_an_old_diffusers_without_the_method_is_reported():
    class _Old:
        pass

    pipe = _Pipe(transformer=_Old())
    note = layerwise.apply(pipe, {"layerwise_casting": "f8"}, torch.bfloat16)
    assert "cannot" in note


def test_nothing_configured_means_no_note_and_no_calls():
    t = _Comp()
    assert layerwise.apply(_Pipe(transformer=t), {}, torch.bfloat16) == ""
    assert t.calls == []


# --------------------------------------------------------------------------- #
#  wiring
# --------------------------------------------------------------------------- #

def test_the_component_loader_reads_the_storage_dtype():
    """Casting AFTER a bf16 load would save the resident cost and still need
    the bf16 peak to get there -- which for a 12.9 GB file on a 24 GB card is
    the entire problem."""
    src = open("codai/models/hf_loading.py").read()
    assert "layerwise.storage_dtype_for" in src or \
        "_lw.storage_dtype_for" in src
    i = src.index("storage_dtype_for")
    body = src[max(0, i - 900):i + 600]
    assert '"torch_dtype": _store_dtype or dtype' in body


def test_the_loader_applies_it_before_placement():
    """The device move should carry the narrow tensors, not the wide ones."""
    src = open("codai/api/images.py").read()
    apply_at = src.index("_layerwise.apply(pipeline")
    place_at = src.index("# Apply memory optimizations based on attempt")
    assert apply_at < place_at


def test_it_is_savable_from_the_admin_page():
    assert '"layerwise_casting",' in open("codai/admin/routes.py").read()


# --------------------------------------------------------------------------- #
#  the shape the loader actually passes
# --------------------------------------------------------------------------- #

NESTED = {
    "precision": "bf16",
    "_raw_cfg": {
        "layerwise_casting": {"transformer": "f8"},
        "component_quantization": {"text_encoder": "4bit"},
    },
}


@requires_f8
def test_it_reads_a_config_nested_under_raw_cfg():
    """A model's config reaches the image loader as TRANSLATED kwargs with the
    original entry under ``_raw_cfg``; hf_loading._norm exists to unwrap that
    and every other reader uses it. This module had its own _norm that did not,
    so `layerwise_casting` was invisible and NO casting hooks were installed --
    while the storage dtype, read through the component loader's already
    unwrapped copy, applied. Measured 2026-11-01: the weights loaded as fp8 and
    nothing upcast them for the matmul, which on Ampere is a dtype the hardware
    cannot multiply at all.
    """
    got, desc = layerwise.spec(NESTED)
    assert got, "the nested config has to be seen"
    assert "transformer" in got
    assert layerwise.storage_dtype_for(NESTED, "transformer") is torch.float8_e4m3fn


@requires_f8
def test_a_flat_config_still_works():
    flat = {"layerwise_casting": {"transformer": "f8"}}
    assert layerwise.storage_dtype_for(flat, "transformer") is torch.float8_e4m3fn


@requires_f8
def test_the_hooks_are_installed_for_a_nested_config():
    """The consequence, not just the parse: apply() had returned "" and
    installed nothing."""
    t = _Comp()
    note = layerwise.apply(_Pipe(transformer=t), NESTED, torch.bfloat16)
    assert t.calls == [(torch.float8_e4m3fn, torch.bfloat16)]
    assert "transformer" in note


def test_the_unwrap_is_shared_and_not_reimplemented():
    src = open("codai/models/layerwise.py").read()
    assert "hf_loading import _norm" in src, \
        "a second copy of the unwrap is how the two readers disagreed"
