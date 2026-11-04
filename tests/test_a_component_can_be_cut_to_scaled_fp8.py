"""Writing an fp8 build of a component has to reproduce the original's output.

coderai could already READ a scaled-fp8 checkpoint -- weights at F8_E4M3 with a
scalar <module>.weight_scale beside each one, the scale installed as a forward
hook on the module output so the weights stay narrow (codai.models.scaled_fp8).
It could not WRITE one, so a pipeline shipping 35 GB of bf16 shards had two
options: do not serve it at all on a 24 GB card, or quantize all 35 GB on every
cold start, because the per-component cache only covers bitsandbytes.

The one thing that makes this exact rather than approximately right:

    scaled_fp8.install hooks the module's WHOLE output, giving s * (W'x + b').
    For that to equal Wx + b the file must store W' = W/s AND b' = b/s.

Measured here, on a three-layer stack with small weights and a bias: scaling the
bias with its weight gives 0.14% relative error against the bf16 original;
leaving the bias alone gives 100% -- the output is unrelated to the input. It is
not a subtle degradation, because the scales on a real checkpoint are around
1e-3 to 1e-6, so an unscaled bias is thousands of times too large relative to
the weights it is added to.

Everything else is copied byte-for-byte, and which keys get touched is decided
by asking the class what each module IS (nn.Linear, on the meta device) rather
than by the shape of the key -- output scaling is valid for a linear map and not
for something that applies a nonlinearity to the result.
"""
import json
import os
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

import torch.nn as nn                                          # noqa: E402
from safetensors.torch import load_file, save_file              # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools.quantize_component_to_scaled_fp8 as Q              # noqa: E402
from codai.models import scaled_fp8 as S                        # noqa: E402

N = 64


class _Stack(nn.Module):
    def __init__(self, n=3):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(N, N) for _ in range(n)])
        self.norm = nn.LayerNorm(N)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


@pytest.fixture
def built(tmp_path, monkeypatch):
    """A tiny bf16 component folder, quantized by the tool."""
    torch.manual_seed(0)
    src = tmp_path / "comp"
    src.mkdir()
    sd, ref = {}, _Stack()
    with torch.no_grad():
        for i, layer in enumerate(ref.layers):
            # Small weights on purpose: cast straight to e4m3 these land in its
            # subnormals, which is what the per-tensor scale exists to avoid.
            layer.weight.mul_(0.002)
            layer.bias.mul_(0.05)
            sd[f"layers.{i}.weight"] = layer.weight.data.to(torch.bfloat16)
            sd[f"layers.{i}.bias"] = layer.bias.data.to(torch.bfloat16)
    sd["norm.weight"] = torch.ones(N, dtype=torch.bfloat16)
    save_file(sd, str(src / "model.safetensors"))
    (src / "config.json").write_text(json.dumps({"_class_name": "_Stack"}))

    monkeypatch.setattr(
        Q, "_linear_weights",
        lambda _s: ({f"layers.{i}.weight": True for i in range(3)}, "_Stack"))
    out = tmp_path / "fp8.safetensors"
    monkeypatch.setattr(sys, "argv",
                        ["q", "--src", str(src), "--out", str(out)])
    assert Q.main() == 0
    return sd, load_file(str(out)), ref


def _loaded(weights, scales):
    m = _Stack()
    with torch.no_grad():
        for i in range(3):
            m.layers[i].weight.copy_(weights[f"layers.{i}.weight"].to(torch.float32))
            m.layers[i].bias.copy_(weights[f"layers.{i}.bias"].to(torch.float32))
    applied, missing = S.install(m, scales)
    assert not missing and applied == 3
    return m


def _scales(got):
    return {k[: -len(S.SCALE_SUFFIX)]: float(v)
            for k, v in got.items() if k.endswith(S.SCALE_SUFFIX)}


# ── the shape of the file ────────────────────────────────────────────────────

def test_the_file_is_the_shape_scaled_fp8_reads(built):
    _, got, _ = built
    for i in range(3):
        assert got[f"layers.{i}.weight"].dtype == torch.float8_e4m3fn
        assert f"layers.{i}{S.SCALE_SUFFIX}" in got
        assert got[f"layers.{i}{S.SCALE_SUFFIX}"].shape == torch.Size([])


def test_a_bias_is_not_emitted_twice(built):
    """".bias" sorts before ".weight", so an owned bias is reachable from both
    the plain branch and its module's group. A duplicate key overwrites its own
    header entry and safetensors refuses the file: "invalid offset for tensor".
    That the file loaded at all is the assertion; this names why."""
    _, got, _ = built
    assert len([k for k in got if k.endswith(".bias")]) == 3
    assert len(got) == 3 * 3 + 1          # weight+scale+bias each, plus norm


def test_everything_that_is_not_a_linear_weight_is_copied_untouched(built):
    sd, got, _ = built
    assert got["norm.weight"].dtype == torch.bfloat16
    assert torch.equal(got["norm.weight"], sd["norm.weight"])


def test_the_scale_uses_the_fp8_range(built):
    """s = amax/448 maps the tensor's peak onto the largest finite e4m3 value;
    a scale near 1.0 would mean the range is being wasted."""
    sd, got, _ = built
    for i in range(3):
        amax = float(sd[f"layers.{i}.weight"].abs().to(torch.float32).max())
        assert float(got[f"layers.{i}{S.SCALE_SUFFIX}"]) == pytest.approx(
            amax / Q.E4M3_MAX, rel=1e-3)


# ── the arithmetic ───────────────────────────────────────────────────────────

def test_the_output_matches_the_bf16_original(built):
    sd, got, ref = built
    x = torch.randn(8, N)
    with torch.no_grad():
        want = ref(x)
        have = _loaded(got, _scales(got))(x)
    rel = float((want - have).norm() / want.norm())
    assert rel < 0.02, f"fp8 round trip is {rel:.4f} off the original"


def test_leaving_the_bias_unscaled_would_destroy_the_output(built):
    """The assertion that justifies the bias handling. Not a degradation --
    the result is unrelated to the input."""
    sd, got, ref = built
    unscaled = dict(got)
    for i in range(3):
        unscaled[f"layers.{i}.bias"] = sd[f"layers.{i}.bias"]
    x = torch.randn(8, N)
    with torch.no_grad():
        want = ref(x)
        have = _loaded(unscaled, _scales(got))(x)
    rel = float((want - have).norm() / want.norm())
    assert rel > 0.5, (
        f"the unscaled-bias control only differs by {rel:.4f}, so this test is "
        f"no longer demonstrating anything")


def test_a_zero_weight_does_not_divide_by_zero(built, tmp_path, monkeypatch):
    """An all-zero tensor has no peak to map; its scale must be 1.0, not a nan."""
    src = tmp_path / "z"
    src.mkdir()
    save_file({"layers.0.weight": torch.zeros(N, N, dtype=torch.bfloat16)},
              str(src / "model.safetensors"))
    (src / "config.json").write_text(json.dumps({"_class_name": "_Stack"}))
    monkeypatch.setattr(Q, "_linear_weights",
                        lambda _s: ({"layers.0.weight": True}, "_Stack"))
    out = tmp_path / "z.safetensors"
    monkeypatch.setattr(sys, "argv", ["q", "--src", str(src), "--out", str(out)])
    assert Q.main() == 0
    got = load_file(str(out))
    s = float(got[f"layers.0{S.SCALE_SUFFIX}"])
    assert s == 1.0
    assert not torch.isnan(got["layers.0.weight"].to(torch.float32)).any()


def test_the_promised_size_is_checked_against_what_was_written(tmp_path):
    """The header is computed from shapes before any data is written, so a
    mismatch means the two disagree -- and a truncated or over-long file would
    load as garbage. The tool removes it instead."""
    import inspect
    body = inspect.getsource(Q.main)
    assert 'header promised' in body
    assert 'os.unlink(tmp)' in body
    assert '.partial' in body, "the output must not be written in place"
