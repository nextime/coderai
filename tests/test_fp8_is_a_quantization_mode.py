"""`component_quantization: {"transformer": "fp8"}` has to mean something.

Until now it did not. The map understood `4bit`, `8bit`, `2bit` and a `.gguf`
path; anything else fell through `_normalize_quant_mode` as None, printed a
warning and loaded the component at FULL PRECISION from the base repo. For a
37.1 GB LTX-2.3 transformer on a 24 GB card that is the stall the whole setting
exists to avoid, and fp8 is the format community fine-tunes are actually
published in -- so the one spelling an operator was most likely to reach for was
the one that quietly did nothing.

fp8 and 8bit are deliberately not synonyms. `8bit` is bitsandbytes int8:
integers with a per-channel scale, precision concentrated near zero. `fp8` is
`float8_e4m3fn`: an exponent plus three mantissa bits, so it keeps dynamic range
instead. For a diffusion transformer, whose activations span orders of
magnitude, that is usually the better trade. Spelling either as the other would
load something different from what was asked for.

Weight-only, and the comment in the code says so, because "fp8" reads like a
speed setting: the weights are stored as float8 and upcast for the matmul, so on
Ampere -- the 3090 this mostly runs on, which has no fp8 matmul -- the win is
VRAM and nothing else. It is also NOT a way to save room against a GGUF: ~18.6 GB
for an LTX-2.3 transformer against 14.3-15.9 GB for a Q4/Q5 GGUF of the same
weights, because a GGUF component stays quantized in VRAM under
GGUFQuantizationConfig and dequantizes per forward. fp8 buys fidelity over a
K-quant, not room.
"""
import inspect
import pathlib

import pytest

from codai.models import hf_loading
from codai.models.hf_loading import _normalize_quant_mode as mode


# ── the spellings ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw", ["fp8", "FP8", "f8", "float8", "e4m3",
                                 "fp8_e4m3", "fp8-e4m3", "float8_e4m3fn",
                                 "float8wo", "fp8wo", " fp8 "])
def test_every_reasonable_spelling_of_fp8_is_accepted(raw):
    assert mode(raw) == "fp8", raw


@pytest.mark.parametrize("raw,want", [("8bit", "8bit"), ("int8", "8bit"),
                                      ("bnb8", "8bit"), ("4bit", "4bit"),
                                      ("nf4", "4bit"), ("2bit", "2bit")])
def test_the_existing_modes_are_untouched(raw, want):
    assert mode(raw) == want


def test_fp8_and_int8_are_not_the_same_thing():
    """float8_e4m3 keeps dynamic range; bitsandbytes int8 keeps precision near
    zero. Conflating them would load a different format than was asked for."""
    assert mode("fp8") != mode("8bit")
    assert mode("int8") == "8bit", "int8 must not be swallowed by the fp8 branch"
    assert mode("float8") == "fp8"


@pytest.mark.parametrize("raw", ["bf16", "fp16", "f16", "fp32", "none", "off",
                                 "", None, False, "/AI/x/y.gguf"])
def test_a_precision_or_a_file_is_still_not_a_quant_mode(raw):
    """`precision` and `component_models` are different settings; a value that
    belongs to one of those must keep falling through to the warning."""
    assert mode(raw) is None


# ── the config it builds ─────────────────────────────────────────────────────

def test_the_builder_has_an_fp8_branch_before_the_bitsandbytes_one():
    src = inspect.getsource(hf_loading.build_pipeline_quant_config)
    assert "_fp8_cfg" in src
    assert src.index("if mode == 'fp8':") < src.index("BnB = TfBnb")


def test_it_passes_a_config_OBJECT_not_the_string_spelling():
    """torchao >= 0.14 rejects the string outright — measured on torchao 0.17.0
    in the shipped image: diffusers raises "quant_type must be an AOBaseConfig
    instance, got str". That exception is caught, so the string form would not
    crash: it would warn and load 37 GB of bf16, the one outcome this setting
    exists to prevent."""
    src = inspect.getsource(hf_loading.build_pipeline_quant_config)
    i = src.index("def _fp8_cfg")
    body = src[i:i + 2600]
    # Anchored on the two RETURN statements, not on the names: the comment
    # above them explains the string form and mentions it first.
    obj = body.index("return TAC(Float8WeightOnlyConfig())")
    txt = body.index("return TAC('float8wo_e4m3')")
    assert obj < txt, (
        "the string spelling must only be the fallback for an older torchao")
    assert "except ImportError" in body[obj:txt], (
        "the fallback must be reached only when Float8WeightOnlyConfig is "
        "absent, not on any error from building the config")


def test_both_libraries_get_their_own_config_class():
    """A component owned by transformers takes transformers' TorchAoConfig; one
    owned by diffusers takes diffusers'. Crossing them raises."""
    src = inspect.getsource(hf_loading.build_pipeline_quant_config)
    body = src[src.index("def _fp8_cfg"):][:2600]
    assert "from transformers import TorchAoConfig" in body
    assert "from diffusers import TorchAoConfig" in body


def test_a_missing_torchao_degrades_loudly(capsys, monkeypatch):
    """The failure mode that matters: no torchao means the component loads at
    full precision, and that has to be said as loudly as an unknown mode."""
    import importlib.util as iu
    real = iu.find_spec

    def _no_torchao(name, *a, **k):
        if name == "torchao":
            return None
        return real(name, *a, **k)

    monkeypatch.setattr(iu, "find_spec", _no_torchao)
    qc, desc = hf_loading.build_pipeline_quant_config(
        "diffusers/LTX-2.3-Diffusers",
        {"component_quantization": {"transformer": "fp8"}}, None)
    out = capsys.readouterr().out
    assert "FULL PRECISION" in out, out
    assert "torchao" in out
    assert "transformer" not in (desc or ""), (
        "a component that could not be quantized must not be described as if "
        "it had been")


def test_the_unknown_mode_warning_now_offers_fp8(capsys):
    """An operator who typed something close needs to be told the real list."""
    hf_loading.build_pipeline_quant_config(
        "diffusers/LTX-2.3-Diffusers",
        {"component_quantization": {"transformer": "fp9"}}, None)
    out = capsys.readouterr().out
    assert "fp8" in out and "FULL PRECISION" in out


# ── what it must not quietly break ───────────────────────────────────────────

def test_the_conv_only_skip_still_covers_fp8():
    """torchao, like bitsandbytes and quanto, only touches nn.Linear. Applying
    it to the conv-only VAE is a hard error, not a smaller VAE."""
    src = inspect.getsource(hf_loading.build_pipeline_quant_config)
    i = src.index("def _bnb_incompatible")
    assert "torchao" in src[i - 400:i + 200].lower()


def test_switching_a_component_to_fp8_invalidates_the_vram_measurement():
    """`measured_vram_gb` is reused as a cap and the learned GPU-layer split
    with it, so a measurement taken under another quantization must not carry
    over. `_quant_signature` hashes component_quantization — this pins that it
    keeps doing so now that fp8 is one of its values."""
    from codai.models.manager import multi_model_manager as m
    a = m._quant_signature({"component_quantization": {"transformer": "fp8"}})
    b = m._quant_signature({"component_quantization":
                            {"transformer": "/AI/guffcache/x-Q5_K_M.gguf"}})
    c = m._quant_signature({"component_quantization": {"transformer": "8bit"}})
    assert len({a, b, c}) == 3, "these are three different residencies"
