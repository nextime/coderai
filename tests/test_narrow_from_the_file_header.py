# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Which tensors are narrow is a fact about the FILE, not the loaded module.

Three attempts failed on the difference, all on Krea-2-RealDream, 2026-11-01:

* ``torch_dtype=<storage>`` casts the WHOLE component, so the F16 time
  embeddings and F32 norms become fp8 too. diffusers' default skip patterns
  then leave exactly those out of the casting hooks and the first matmul says
  "mat1 and mat2 must have the same dtype, but got BFloat16 and Float8_e4m3fn",
  naming ``time_embed.linear_1`` -- F16 on disk.

* ``torch_dtype=None`` preserves nothing either. The meta loader ends at
  accelerate's ``set_module_tensor_to_device``:

      if dtype is None:
          value = value.to(old_value.dtype)

  so every incoming fp8 tensor is cast to whatever the freshly-built model
  already holds. Recording "which modules hold fp8" afterwards found ZERO and
  the footprint stayed 31.07 GB against the 17.22 GB the same model measures
  loaded as fp8.

* Comparing safetensors' dtype names to torch's by string. safetensors says
  ``F8_E4M3`` where torch says ``float8_e4m3fn``; stripping punctuation and
  folding case never makes those equal, and it reported "0 module(s) are
  float8_e4m3fn" for a file whose 256 big linears are precisely that.

So: load narrow, read the header for the truth, and put the wider tensors back
from the file at their own precision.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from safetensors.torch import save_file

from codai.models import layerwise as L
from codai.models import scaled_fp8 as S

_HAS_F8 = hasattr(torch, "float8_e4m3fn")
requires_f8 = pytest.mark.skipif(not _HAS_F8, reason="this torch has no fp8")


# --------------------------------------------------------------------------- #
#  the two vocabularies
# --------------------------------------------------------------------------- #

@requires_f8
@pytest.mark.parametrize("st_name,attr", [
    ("F8_E4M3", "float8_e4m3fn"),
    ("F8_E5M2", "float8_e5m2"),
    ("BF16", "bfloat16"),
    ("F16", "float16"),
    ("F32", "float32"),
])
def test_safetensors_dtype_names_map_to_torch_dtypes(st_name, attr):
    assert S.torch_dtype_of(st_name) is getattr(torch, attr)


def test_an_unknown_name_is_not_guessed_at():
    for bad in ("I8", "U8", "BOOL", "", None, "F8_E3M4"):
        assert S.torch_dtype_of(bad) is None


def test_the_names_are_not_compared_as_strings():
    """"F8_E4M3" and "float8_e4m3fn" are not each other under any case folding
    or punctuation stripping, which is how this reported zero narrow modules
    for a file full of them."""
    src = open("codai/models/layerwise.py").read()
    assert "_sfp8.torch_dtype_of(dtype_name) is storage_dtype" in src


# --------------------------------------------------------------------------- #
#  reading the header, restoring from the file
# --------------------------------------------------------------------------- #

@pytest.fixture
def mixed_file(tmp_path):
    """A checkpoint shaped like the real one: fp8 linears, F16 modulation, F32
    norms, and a bf16 scale beside the fp8 weight."""
    p = tmp_path / "mini.safetensors"
    save_file({
        "blocks.0.attn.wq.weight": torch.randn(8, 8).to(torch.float8_e4m3fn),
        "blocks.0.attn.wq.weight_scale": torch.tensor(0.0042, dtype=torch.bfloat16),
        "blocks.0.mod.lin": torch.randn(6, 4).to(torch.float16),
        "blocks.0.prenorm.scale": torch.randn(4).to(torch.float32),
    }, str(p))
    return str(p)


@requires_f8
def test_the_header_reports_every_stored_dtype(mixed_file):
    got = S.dtypes_in_file(mixed_file)
    assert got["blocks.0.attn.wq.weight"] == "F8_E4M3"
    assert got["blocks.0.mod.lin"] == "F16"
    assert got["blocks.0.prenorm.scale"] == "F32"


@pytest.fixture
def renamer(monkeypatch):
    """diffusers' converter, standing in with the real Krea2 rules."""
    class Krea2Transformer2DModel:
        pass

    def conv(ck, **kw):
        pre = {"blocks.": "transformer_blocks."}
        blk = {".attn.wq.": ".attn.to_q.", ".mod.lin": ".scale_shift_table",
               ".prenorm.scale": ".norm1.weight"}
        out = {}
        for k in list(ck):
            nk = k
            for o, n in pre.items():
                if nk.startswith(o):
                    nk = n + nk[len(o):]
                    break
            for o, n in blk.items():
                nk = nk.replace(o, n)
            out[nk] = ck.pop(k)
        return out

    import diffusers.loaders.single_file_model as sfm
    monkeypatch.setitem(sfm.SINGLE_FILE_LOADABLE_CLASSES,
                        "Krea2Transformer2DModel",
                        {"checkpoint_mapping_fn": conv})
    return Krea2Transformer2DModel


def _net():
    net = torch.nn.Module()
    net.add_module("transformer_blocks", torch.nn.ModuleList([torch.nn.Module()]))
    b = net.transformer_blocks[0]
    b.add_module("attn", torch.nn.Module())
    b.attn.add_module("to_q", torch.nn.Linear(8, 8, bias=False))
    b.scale_shift_table = torch.nn.Parameter(torch.zeros(6, 4))
    b.add_module("norm1", torch.nn.Module())
    b.norm1.weight = torch.nn.Parameter(torch.zeros(4))
    return net


@requires_f8
def test_the_narrow_module_is_found_and_the_rest_restored(mixed_file, renamer):
    net = _net().to(torch.float8_e4m3fn)     # what loading narrow does
    n = L.remember_narrow(net, torch.float8_e4m3fn, path=mixed_file,
                          component_cls=renamer)
    b = net.transformer_blocks[0]
    assert n == 1, "the fp8 linear"
    assert b.attn.to_q.weight.dtype is torch.float8_e4m3fn, "stays narrow"
    assert b.scale_shift_table.dtype is torch.float16, "restored from the file"
    assert b.norm1.weight.dtype is torch.float32, "restored from the file"


@requires_f8
def test_the_restored_values_come_from_the_file_not_a_round_trip(mixed_file, renamer):
    """The point of restoring rather than upcasting: nothing that was not fp8
    on disk passes through fp8."""
    from safetensors import safe_open
    with safe_open(mixed_file, framework="pt") as fh:
        want = fh.get_tensor("blocks.0.prenorm.scale")
    net = _net().to(torch.float8_e4m3fn)
    L.remember_narrow(net, torch.float8_e4m3fn, path=mixed_file,
                      component_cls=renamer)
    assert torch.equal(net.transformer_blocks[0].norm1.weight.data, want)


@requires_f8
def test_then_every_forward_agrees_on_dtype(mixed_file, renamer):
    """The failure was a matmul, so the test is a matmul."""
    class _Pipe:
        pass

    net = _net().to(torch.float8_e4m3fn)
    L.remember_narrow(net, torch.float8_e4m3fn, path=mixed_file,
                      component_cls=renamer)
    pipe = _Pipe()
    pipe.transformer = net
    note = L.apply(pipe, {"_raw_cfg": {"layerwise_casting": {"transformer": "f8"}}},
                   torch.bfloat16)
    b = net.transformer_blocks[0]
    assert "1 module(s) kept at float8_e4m3fn" in note
    assert b.attn.to_q.weight.dtype is torch.float8_e4m3fn
    assert b.attn.to_q(torch.ones(1, 8, dtype=torch.bfloat16)).dtype is torch.bfloat16


@requires_f8
def test_a_scale_key_is_not_mistaken_for_a_parameter(mixed_file, renamer):
    """`...to_q.weight_scale` renames onto a module that has no such
    attribute; it must be skipped, not assigned."""
    net = _net().to(torch.float8_e4m3fn)
    L.remember_narrow(net, torch.float8_e4m3fn, path=mixed_file,
                      component_cls=renamer)
    assert not hasattr(net.transformer_blocks[0].attn.to_q, "weight_scale")


@requires_f8
def test_without_a_file_it_falls_back_to_asking_the_module():
    """A component not read from a single file still works -- and this is why
    the file path exists at all. Asked of a component that was loaded narrow,
    the fallback says EVERY module is narrow (here all three), because that is
    true of the loaded object and says nothing about what the file stored. Only
    the header distinguishes the fp8 linear from the F16 and F32 tensors that
    were downcast on the way in."""
    net = _net().to(torch.float8_e4m3fn)
    assert L.remember_narrow(net, torch.float8_e4m3fn) == 3


@requires_f8
def test_the_file_distinguishes_what_the_module_cannot(mixed_file, renamer):
    """The same component, the same moment: 3 by asking the object, 1 by asking
    the file. The second is the truth."""
    by_object = L.remember_narrow(_net().to(torch.float8_e4m3fn),
                                  torch.float8_e4m3fn)
    by_file = L.remember_narrow(_net().to(torch.float8_e4m3fn),
                                torch.float8_e4m3fn, path=mixed_file,
                                component_cls=renamer)
    assert (by_object, by_file) == (3, 1)


def test_the_loader_loads_narrow_and_passes_the_file():
    src = open("codai/models/hf_loading.py").read()
    assert '"torch_dtype": _store_dtype or dtype' in src, \
        "torch_dtype=None preserves nothing; accelerate casts to the old dtype"
    assert "path=_path, component_cls=cls" in src
