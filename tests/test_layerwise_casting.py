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

class _Pipe:
    def __init__(self, **components):
        for k, v in components.items():
            setattr(self, k, v)


def _f8_net():
    """A module with one genuinely narrow parameter."""
    net = torch.nn.Module()
    net.add_module("big", torch.nn.Linear(8, 8, bias=False))
    with torch.no_grad():
        net.big.weight.data = net.big.weight.data.to(torch.float8_e4m3fn)
    return net


@requires_f8
def test_it_hooks_the_configured_component_and_says_so():
    net = _f8_net()
    note = layerwise.apply(_Pipe(transformer=net, vae=_f8_net()),
                           {"layerwise_casting": {"transformer": "f8"}},
                           torch.bfloat16)
    assert "transformer" in note and "bfloat16" in note
    assert net.big.weight.dtype is torch.float8_e4m3fn


@requires_f8
def test_a_component_the_pipeline_does_not_have_is_simply_skipped():
    note = layerwise.apply(_Pipe(transformer=_f8_net()),
                           {"layerwise_casting": "f8"}, torch.bfloat16)
    assert "transformer" in note
    # unet / text_encoder are absent from this pipeline and must not appear as
    # failures -- a pipeline is not broken for lacking a component.
    assert "unet" not in note


@requires_f8
def test_something_that_is_not_a_module_is_reported_not_raised():
    """A VRAM optimisation must never be why a generation fails."""
    note = layerwise.apply(_Pipe(transformer=object()),
                           {"layerwise_casting": "f8"}, torch.bfloat16)
    assert "not a module" in note


@requires_f8
def test_a_failure_inside_the_hook_is_reported_not_raised(monkeypatch):
    def _boom(*a, **kw):
        raise RuntimeError("no hooks here")
    monkeypatch.setattr(layerwise, "_hook_one", _boom)
    note = layerwise.apply(_Pipe(transformer=_f8_net()),
                           {"layerwise_casting": "f8"}, torch.bfloat16)
    assert "not applied to" in note and "no hooks here" in note


def test_nothing_configured_means_no_note():
    assert layerwise.apply(_Pipe(transformer=object()), {}, None) == ""


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
    net = _f8_net()
    note = layerwise.apply(_Pipe(transformer=net), NESTED, torch.bfloat16)
    assert "transformer" in note
    assert net.big.weight.dtype is torch.float8_e4m3fn
    assert net.big(torch.ones(1, 8, dtype=torch.bfloat16)).dtype is torch.bfloat16


def test_the_unwrap_is_shared_and_not_reimplemented():
    src = open("codai/models/layerwise.py").read()
    assert "hf_loading import _norm" in src, \
        "a second copy of the unwrap is how the two readers disagreed"


# --------------------------------------------------------------------------- #
#  a checkpoint is not uniform
# --------------------------------------------------------------------------- #

@requires_f8
def _mixed_net():
    """The real shape of Real Dream's fp8 build: 256 fp8 linears, F16
    time-embedding projections, F32 norms."""
    net = torch.nn.Module()
    net.add_module("big", torch.nn.Linear(8, 8, bias=False))
    net.add_module("time_embed", torch.nn.Linear(4, 4, bias=True))
    net.add_module("norm", torch.nn.LayerNorm(4))
    with torch.no_grad():
        net.big.weight.data = net.big.weight.data.to(torch.float8_e4m3fn)
        net.time_embed.weight.data = net.time_embed.weight.data.to(torch.float16)
        net.time_embed.bias.data = net.time_embed.bias.data.to(torch.float16)
        net.norm.weight.data = net.norm.weight.data.to(torch.float32)
        net.norm.bias.data = net.norm.bias.data.to(torch.float32)
    return net


@requires_f8
def test_only_the_narrow_modules_are_hooked_and_the_rest_are_recast():
    """Casting the WHOLE component to fp8 is what broke this: diffusers'
    default skip patterns then excluded the precision-critical modules from the
    hooks, so they kept an fp8 weight nothing upcast and the first matmul said

        mat1 and mat2 must have the same dtype, but got BFloat16 and
        Float8_e4m3fn

    naming time_embed.linear_1 -- F16 on disk. Measured 2026-11-01.
    """
    net = _mixed_net()
    pipe = _Pipe(transformer=net)
    note = layerwise.apply(pipe, {"layerwise_casting": {"transformer": "f8"}},
                           torch.bfloat16)
    assert net.big.weight.dtype is torch.float8_e4m3fn, \
        "the big linear must stay narrow at rest -- that is the whole point"
    assert net.time_embed.weight.dtype is torch.bfloat16
    assert net.time_embed.bias.dtype is torch.bfloat16
    assert net.norm.weight.dtype is torch.bfloat16
    assert "1 module(s) kept at float8_e4m3fn" in note
    assert "brought to bfloat16" in note


@requires_f8
def test_every_forward_agrees_on_dtype_afterwards():
    """The failure was a matmul, so the test is a matmul."""
    net = _mixed_net()
    layerwise.apply(_Pipe(transformer=net),
                    {"layerwise_casting": {"transformer": "f8"}}, torch.bfloat16)
    assert net.big(torch.ones(1, 8, dtype=torch.bfloat16)).dtype is torch.bfloat16
    assert net.time_embed(torch.ones(1, 4, dtype=torch.bfloat16)).dtype \
        is torch.bfloat16


@requires_f8
def test_a_component_with_nothing_narrow_is_reported_not_silently_hooked():
    """If the file turned out to hold no narrow tensors at all, saying so beats
    reporting a saving that did not happen."""
    net = torch.nn.Module()
    net.add_module("lin", torch.nn.Linear(4, 4))
    note = layerwise.apply(_Pipe(transformer=net),
                           {"layerwise_casting": {"transformer": "f8"}},
                           torch.bfloat16)
    assert "nothing stored at float8_e4m3fn" in note


def test_the_loader_keeps_the_files_own_widths():
    """torch_dtype=None, not the storage dtype: diffusers casts the WHOLE
    component to whatever torch_dtype says."""
    src = open("codai/models/hf_loading.py").read()
    # Anchored on _store_dtype: `kwargs: Dict[str, Any] = {` occurs more than
    # once in this file and the first hit is a different function.
    assert '"torch_dtype": None if _store_dtype is not None else dtype' in src


def test_skip_patterns_are_not_relied_on():
    """diffusers appends the model's own _skip_layerwise_casting_patterns even
    to an empty tuple, so a pattern-based approach can still exclude a module
    behind our back. The per-module hook has no pattern matching."""
    src = open("codai/models/layerwise.py").read()
    assert "apply_layerwise_casting_hook" in src
    assert "skip_modules_pattern" not in src.split('"""')[-1]


# --------------------------------------------------------------------------- #
#  the pipeline casts a component it is handed
# --------------------------------------------------------------------------- #

@requires_f8
def _mixed_for_pipeline():
    net = torch.nn.Module()
    net.add_module("big", torch.nn.Linear(8, 8, bias=False))
    net.add_module("time_embed", torch.nn.Linear(4, 4))
    with torch.no_grad():
        net.big.weight.data = net.big.weight.data.to(torch.float8_e4m3fn)
        net.time_embed.weight.data = net.time_embed.weight.data.to(torch.float16)
        net.time_embed.bias.data = net.time_embed.bias.data.to(torch.float16)
    return net


@requires_f8
def test_the_narrow_modules_are_remembered_before_the_pipeline_sees_them():
    net = _mixed_for_pipeline()
    assert layerwise.remember_narrow(net, torch.float8_e4m3fn) == 1


@requires_f8
def test_they_are_restored_after_the_pipeline_has_cast_everything():
    """DiffusionPipeline.from_pretrained casts a component it is handed to its
    own torch_dtype, so layerwise casting afterwards found nothing narrow and
    installed nothing. Measured 2026-11-01: "not applied to transformer
    (nothing stored at float8_e4m3fn)" and a 31.07 GB footprint where the same
    model had measured 17.22 GB."""
    net = _mixed_for_pipeline()
    layerwise.remember_narrow(net, torch.float8_e4m3fn)
    net.to(torch.bfloat16)                       # what the pipeline does
    assert net.big.weight.dtype is torch.bfloat16

    note = layerwise.apply(_Pipe(transformer=net), NESTED, torch.bfloat16)
    assert net.big.weight.dtype is torch.float8_e4m3fn, "put back to the file's width"
    assert net.time_embed.weight.dtype is torch.bfloat16, "and only those"
    assert "1 module(s) kept at float8_e4m3fn" in note
    assert net.big(torch.ones(1, 8, dtype=torch.bfloat16)).dtype is torch.bfloat16


@requires_f8
def test_the_round_trip_through_bfloat16_is_exact():
    """Which is what makes restoring them sound rather than a second loss:
    bf16 carries the same exponent range as float8_e4m3fn and four more
    mantissa bits."""
    a = torch.randn(256).to(torch.float8_e4m3fn)
    back = a.to(torch.bfloat16).to(torch.float8_e4m3fn)
    assert torch.equal(back.float(), a.float())


@requires_f8
def test_only_the_recorded_modules_are_narrowed():
    """A module the file stored at F16 or F32 must never be pushed down to fp8;
    that was the bug before this -- casting the whole component."""
    net = _mixed_for_pipeline()
    layerwise.remember_narrow(net, torch.float8_e4m3fn)
    net.to(torch.bfloat16)
    layerwise.apply(_Pipe(transformer=net), NESTED, torch.bfloat16)
    assert net.time_embed.weight.dtype is torch.bfloat16
    assert net.time_embed.bias.dtype is torch.bfloat16


@requires_f8
def test_without_a_record_it_falls_back_to_asking_each_module():
    """A component loaded some other way still works, just without the
    restore step."""
    net = _mixed_for_pipeline()          # no remember_narrow call
    note = layerwise.apply(_Pipe(transformer=net), NESTED, torch.bfloat16)
    assert "1 module(s) kept at float8_e4m3fn" in note
    assert net.big.weight.dtype is torch.float8_e4m3fn


@requires_f8
def test_a_record_for_a_different_dtype_is_ignored():
    net = _mixed_for_pipeline()
    layerwise.remember_narrow(net, torch.float16)   # not what the config asks
    net.to(torch.bfloat16)
    note = layerwise.apply(_Pipe(transformer=net), NESTED, torch.bfloat16)
    assert "nothing stored at float8_e4m3fn" in note


def test_the_loader_records_them():
    src = open("codai/models/hf_loading.py").read()
    assert "remember_narrow(components[name], _store_dtype)" in src
    i = src.index("remember_narrow(")
    load = src.index("cls.from_single_file(")
    assert load < i, "recorded after the file is read, before the pipeline"
