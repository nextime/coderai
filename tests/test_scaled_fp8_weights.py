# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Weights with the scale kept beside them, not folded in.

A large part of the community publishes fine-tunes this way::

    blocks.0.attn.wq.weight         F8_E4M3  [6144, 6144]
    blocks.0.attn.wq.weight_scale   BF16     []            a scalar

``weight_scale`` appears nowhere in diffusers' single_file_utils.py or
single_file_model.py. Its converter renames every key and passes it through, so
the scale lands as ``transformer_blocks.0.attn.to_q.weight_scale``, which the
model does not have, and it is dropped as an unexpected key while the weight is
used UNSCALED.

Measured on Real Dream's Krea 2 V3 fp8 build, 2026-11-01: 256 scales, min
0.000839, median 0.00249, max 0.0125, not one within 1% of 1.0. Dropping them
leaves every weight about four hundred times too large -- noise, not a slightly
worse image. All three of that version's builds (int8, fp8, nvfp4) carry them.

The scale is a scalar, so for a linear layer

    y = (W * s) @ x  ==  s * (W @ x)

and applying it to the output leaves the weights at the width they were stored
at. Dequantizing would be correct too and would cost 25 GB of bf16 for a
12.9 GB file.
"""

import json
import struct

import pytest

from codai.models import scaled_fp8 as S


# --------------------------------------------------------------------------- #
#  reading them out of a file
# --------------------------------------------------------------------------- #

def _write_safetensors(path, entries):
    """entries: {name: (dtype, shape, raw_bytes)} -> a minimal safetensors file."""
    header, blob, off = {}, b"", 0
    for name, (dtype, shape, raw) in entries.items():
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [off, off + len(raw)]}
        blob += raw
        off += len(raw)
    hb = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(hb)) + hb + blob)
    return str(path)


def _bf16(value):
    """`value` as the top two bytes of its float32 representation."""
    u = struct.unpack("<I", struct.pack("<f", value))[0]
    return struct.pack("<H", u >> 16)


def test_scales_are_read_without_touching_the_weights(tmp_path):
    f = _write_safetensors(tmp_path / "m.safetensors", {
        "blocks.0.attn.wq.weight": ("F8_E4M3", [4, 4], b"\x00" * 16),
        "blocks.0.attn.wq.weight_scale": ("BF16", [], _bf16(0.00418)),
        "blocks.0.attn.wk.weight_scale": ("F32", [],
                                          struct.pack("<f", 0.00769)),
    })
    got = S.scales_in_file(f)
    assert set(got) == {"blocks.0.attn.wq.weight_scale",
                        "blocks.0.attn.wk.weight_scale"}
    assert got["blocks.0.attn.wq.weight_scale"] == pytest.approx(0.00418, rel=1e-2)
    assert got["blocks.0.attn.wk.weight_scale"] == pytest.approx(0.00769, rel=1e-3)


def test_a_file_with_no_scales_reads_as_none(tmp_path):
    """The ordinary case: nothing here may change how a plain checkpoint loads."""
    f = _write_safetensors(tmp_path / "plain.safetensors", {
        "blocks.0.attn.wq.weight": ("BF16", [2, 2], b"\x00" * 8),
    })
    assert S.scales_in_file(f) == {}


def test_a_per_channel_scale_is_refused_rather_than_misapplied(tmp_path):
    """A vector scale needs applying along a dimension; treating it as a number
    would be wrong in a way no error would report."""
    f = _write_safetensors(tmp_path / "pc.safetensors", {
        "blocks.0.attn.wq.weight_scale": ("F32", [4], b"\x00" * 16),
    })
    with pytest.raises(ValueError) as ei:
        S.scales_in_file(f)
    assert "per channel" in str(ei.value)


# --------------------------------------------------------------------------- #
#  applying them
# --------------------------------------------------------------------------- #

torch = pytest.importorskip("torch")


class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.lin(x)


def test_the_output_is_scaled_and_the_weights_are_not():
    """The whole point: exact arithmetic, weights untouched at their stored
    width."""
    net = _Net()
    with torch.no_grad():
        net.lin.weight.copy_(torch.eye(4))
    before = net.lin.weight.clone()

    applied, missing = S.install(net, {"lin": 0.0025})
    assert (applied, missing) == (1, [])
    assert torch.equal(net.lin.weight, before), "the weights must not be rewritten"

    x = torch.ones(1, 4)
    assert torch.allclose(net(x), torch.full((1, 4), 0.0025), atol=1e-7)


def test_scaling_matches_multiplying_the_weights():
    """y = (W*s) @ x == s * (W @ x), which is why the output trick is exact."""
    torch.manual_seed(0)
    a, b = _Net(), _Net()
    with torch.no_grad():
        b.lin.weight.copy_(a.lin.weight)
    s = 0.00249
    S.install(a, {"lin": s})
    with torch.no_grad():
        b.lin.weight.mul_(s)          # the dequantize-into-the-weights way
    x = torch.randn(3, 4)
    assert torch.allclose(a(x), b(x), atol=1e-6)


def test_a_tuple_output_keeps_its_other_elements():
    class _Tup(torch.nn.Module):
        def forward(self, x):
            return x, "metadata"

    net = torch.nn.Module()
    net.add_module("m", _Tup())
    S.install(net, {"m": 2.0})
    out = net.m(torch.ones(2))
    assert torch.allclose(out[0], torch.full((2,), 2.0))
    assert out[1] == "metadata"


def test_nothing_configured_installs_nothing():
    net = _Net()
    assert S.install(net, {}) == (0, [])


# --------------------------------------------------------------------------- #
#  it refuses rather than half-applying
# --------------------------------------------------------------------------- #

def test_a_scale_with_no_module_is_reported_as_missing():
    net = _Net()
    applied, missing = S.install(net, {"lin": 1.0, "nope.deeper": 1.0})
    assert applied == 1
    assert missing == ["nope.deeper"]


def test_apply_to_component_refuses_when_a_scale_cannot_be_placed(tmp_path, monkeypatch):
    """Half-applied scales answer 200 with a plausible image, which is the
    worst of the three outcomes."""
    f = _write_safetensors(tmp_path / "m.safetensors", {
        "whatever.weight_scale": ("BF16", [], _bf16(0.002)),
    })
    monkeypatch.setattr(S, "module_scales",
                        lambda cls, raw: {"not.a.module": 0.002})

    class _Cls:
        __name__ = "Krea2Transformer2DModel"

    with pytest.raises(RuntimeError) as ei:
        S.apply_to_component(_Net(), _Cls, f)
    msg = str(ei.value)
    assert "did not match a module" in msg
    assert "four hundred times" in msg


def test_apply_to_component_is_a_no_op_for_a_plain_file(tmp_path):
    f = _write_safetensors(tmp_path / "plain.safetensors", {
        "blocks.0.attn.wq.weight": ("BF16", [2, 2], b"\x00" * 8),
    })
    assert S.apply_to_component(_Net(), object, f) == ""


# --------------------------------------------------------------------------- #
#  renaming is diffusers' job, not ours
# --------------------------------------------------------------------------- #

def test_the_rename_uses_diffusers_own_converter():
    """Copying their rename table would drift away from how the WEIGHTS are
    renamed, and the scales would then attach to the wrong modules."""
    src = open("codai/models/scaled_fp8.py").read()
    assert "SINGLE_FILE_LOADABLE_CLASSES" in src
    assert "checkpoint_mapping_fn" in src
    assert "blocks." not in src.split('"""', 2)[2], \
        "no rename rules of our own outside the docstring"


def test_a_class_diffusers_cannot_load_single_file_is_reported():
    class _Unknown:
        __name__ = "NotAThingModel"

    with pytest.raises(RuntimeError) as ei:
        S.module_scales(_Unknown, {"x.weight_scale": 1.0})
    assert "no single-file converter" in str(ei.value)


def test_the_loader_applies_this_after_a_single_file_component_load():
    src = open("codai/models/hf_loading.py").read()
    load = src.index("cls.from_single_file(")
    note = src.index("_sfp8.apply_to_component(")
    assert load < note, "the scales go on after the component exists"
