"""Loading the INT8 LongCat DiT must not cost what the bf16 model costs.

The straightforward way to load a quantised checkpoint — instantiate the model,
swap the Linears, read every shard into one dict, copy it in — needs the full
bf16 model (~27 GB for this 13.6B DiT) plus the weights twice before it is done.
On a 54 GB host that is an out-of-memory kill, which is exactly what happened.

So the contract these tests hold is about HOW it loads, not only that it loads:
the skeleton is built on the meta device and each shard is streamed in and
assigned. A fake DiT records the device it was constructed on, so an
implementation that allocates for real fails here instead of on the host.
"""
import importlib.util
import json
import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
from safetensors.torch import save_file

ROOT = pathlib.Path(__file__).resolve().parents[1]
SERVICE = ROOT / "tools" / "longcat_service.py"

HIDDEN = 8
DEPTH = 2


class FakeDiT(nn.Module):
    """Shaped like the real thing where it matters: a skipped final_layer.linear,
    Linears inside numbered blocks, and plain parameters that are not quantised."""

    construction_devices: list = []

    def __init__(self, hidden_size=HIDDEN, depth=DEPTH, **kwargs):
        super().__init__()
        FakeDiT.construction_devices.append(torch.empty(1).device.type)
        self.kwargs_seen = kwargs
        self.blocks = nn.ModuleList(
            nn.ModuleDict({"attn": nn.Linear(hidden_size, hidden_size, bias=False)})
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.final_layer = nn.ModuleDict(
            {"linear": nn.Linear(hidden_size, hidden_size, bias=True)}
        )


class FakeQuantizedLinear(nn.Module):
    """Upstream's layout: int8 weights and a per-channel scale, both buffers."""

    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.register_buffer("weight_int8",
                             torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("weight_scale",
                             torch.zeros(out_features, dtype=torch.float32))
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.bfloat16))


@pytest.fixture
def service(monkeypatch):
    """tools/longcat_service.py with a fake longcat_video package behind it."""
    pkg = type(sys)("longcat_video")
    pkg.__path__ = []
    modules = type(sys)("longcat_video.modules")
    modules.__path__ = []
    quant = type(sys)("longcat_video.modules.quantization")
    quant.QuantizedLinear = FakeQuantizedLinear
    quant.DEFAULT_SKIP_PATTERNS = {"final_layer.linear"}
    dit = type(sys)("longcat_video.modules.longcat_video_dit")
    dit.LongCatVideoTransformer3DModel = FakeDiT
    for name, mod in (("longcat_video", pkg),
                      ("longcat_video.modules", modules),
                      ("longcat_video.modules.quantization", quant),
                      ("longcat_video.modules.longcat_video_dit", dit)):
        monkeypatch.setitem(sys.modules, name, mod)

    spec = importlib.util.spec_from_file_location("lc_service_under_test", SERVICE)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, mod)
    spec.loader.exec_module(mod)
    FakeDiT.construction_devices.clear()
    return mod


def _expected_tensors():
    """What the quantiser writes for a FakeDiT: int8 pairs for the two block
    Linears, bf16 for the skipped final layer, plain params for the norm."""
    out = {}
    for i in range(DEPTH):
        out[f"blocks.{i}.attn.weight_int8"] = torch.full(
            (HIDDEN, HIDDEN), i + 1, dtype=torch.int8)
        out[f"blocks.{i}.attn.weight_scale"] = torch.full(
            (HIDDEN,), 0.5 * (i + 1), dtype=torch.float32)
    out["norm.weight"] = torch.ones(HIDDEN)
    out["norm.bias"] = torch.zeros(HIDDEN)
    out["final_layer.linear.weight"] = torch.full((HIDDEN, HIDDEN), 3.0)
    out["final_layer.linear.bias"] = torch.full((HIDDEN,), 4.0)
    return out


def _write_checkpoint(qdir, tensors, shards=2):
    qdir.mkdir(parents=True, exist_ok=True)
    (qdir / "config.json").write_text(json.dumps(
        {"hidden_size": HIDDEN, "depth": DEPTH, "_class_name": "ignored"}))
    keys = list(tensors)
    weight_map = {}
    per = (len(keys) + shards - 1) // shards
    for n in range(shards):
        chunk = keys[n * per:(n + 1) * per]
        if not chunk:
            continue
        name = f"quantized_model-{n + 1:05d}-of-{shards:05d}.safetensors"
        save_file({k: tensors[k].contiguous() for k in chunk}, str(qdir / name))
        weight_map.update({k: name for k in chunk})
    (qdir / "quantized_model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}))


def test_the_skeleton_is_built_on_the_meta_device(service, tmp_path):
    """The load that took the host down built the bf16 model for real first."""
    _write_checkpoint(tmp_path / "base_model_int8", _expected_tensors())

    service._load_quantized_base_dit(str(tmp_path), "base_model_int8")

    assert FakeDiT.construction_devices == ["meta"], (
        "the DiT was allocated for real before any INT8 weight was read — that is "
        "~27 GB of bf16 on the real model")


def test_every_tensor_ends_up_real_and_correct(service, tmp_path):
    """A meta skeleton is only useful if the streamed shards actually land in it."""
    tensors = _expected_tensors()
    _write_checkpoint(tmp_path / "base_model_int8", tensors)

    model = service._load_quantized_base_dit(str(tmp_path), "base_model_int8")

    state = dict(model.named_parameters())
    state.update(model.named_buffers())
    assert not [n for n, t in state.items() if t.is_meta]
    assert set(state) == set(tensors)
    for name, want in tensors.items():
        assert torch.equal(state[name], want), name


def test_the_skipped_layer_keeps_its_bf16_linear(service, tmp_path):
    """final_layer.linear is in DEFAULT_SKIP_PATTERNS: quantising it wrecks output."""
    _write_checkpoint(tmp_path / "base_model_int8", _expected_tensors())

    model = service._load_quantized_base_dit(str(tmp_path), "base_model_int8")

    assert isinstance(model.final_layer["linear"], nn.Linear)
    assert all(isinstance(b["attn"], FakeQuantizedLinear) for b in model.blocks)


def test_a_checkpoint_missing_tensors_is_an_error_not_a_meta_model(service, tmp_path):
    """Returning a model with tensors still on meta hands the caller something that
    fails much later, somewhere unrelated. Say so here."""
    tensors = _expected_tensors()
    del tensors["norm.weight"]
    _write_checkpoint(tmp_path / "base_model_int8", tensors)

    with pytest.raises(RuntimeError, match="meta device"):
        service._load_quantized_base_dit(str(tmp_path), "base_model_int8")


def test_extra_tensors_are_tolerated(service, tmp_path):
    """A newer quantiser writing something this config has no slot for should warn,
    not abort a load that is otherwise complete."""
    tensors = _expected_tensors()
    tensors["blocks.0.attn.weight_absmax"] = torch.ones(HIDDEN)
    _write_checkpoint(tmp_path / "base_model_int8", tensors)

    model = service._load_quantized_base_dit(str(tmp_path), "base_model_int8")

    assert not [n for n, t in model.named_buffers() if t.is_meta]


def test_shards_are_streamed_one_at_a_time(service, tmp_path):
    """Accumulating every shard into one dict before loading doubles the peak."""
    src = SERVICE.read_text(encoding="utf-8")
    body = src[src.index("def _load_quantized_base_dit"):
               src.index("def _load_audio_encoder")]
    assert "safe_open" in body, "whole-shard load_file holds a shard in memory twice"
    assert "assign=True" in body, "without assign= the meta skeleton is copied into"
    assert "load_file" not in body
