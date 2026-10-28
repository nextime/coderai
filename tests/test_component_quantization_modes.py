"""A value `component_quantization` does not understand must not vanish.

The map takes `4bit`, `8bit`, `2bit` or a `.gguf` path. Anything else used to
be skipped silently — and because one unusable entry can leave the mapping
empty, the global load_in_4bit went with it and the component loaded at full
precision from the base repo.

Measured on LTX-2.5: `text_encoder` named a bf16 Gemma-4 12B safetensors, the
quantization config came back None, the 12B loaded unquantized, and every
generation ended with the card full — 23.53 GiB of 23.56 GiB in use, unable
to allocate 30 MiB. Loading a component from a file is `component_models`;
this map is only about how to quantize it.
"""
from codai.models import hf_loading


def test_a_file_path_that_is_not_a_gguf_is_called_out(capsys):
    cfg = {"component_quantization": {"transformer": "/x/y.gguf",
                                      "text_encoder": "/a/gemma-bf16.safetensors"},
           "load_in_4bit": True}
    hf_loading.build_pipeline_quant_config("m", cfg, "bfloat16")
    out = capsys.readouterr().out
    assert "text_encoder" in out and "FULL PRECISION" in out
    assert "component_models" in out          # names the key that does what they meant


def test_a_gguf_path_stays_silent(capsys):
    """GGUF components are handled elsewhere; they are not a mistake."""
    cfg = {"component_quantization": {"transformer": "/x/y.gguf"}}
    hf_loading.build_pipeline_quant_config("m", cfg, "bfloat16")
    assert "FULL PRECISION" not in capsys.readouterr().out


def test_an_explicit_none_stays_silent(capsys):
    """"none" is how a card says "leave this component alone".

    Asserted on the COMPONENT, not on the phrase: the sibling asks for 4bit,
    and on a build without bitsandbytes that legitimately reports its own
    full-precision warning. Matching the bare phrase made this test depend on
    whether bitsandbytes happened to be installed.
    """
    cfg = {"component_quantization": {"vae": "none", "text_encoder": "4bit"}}
    hf_loading.build_pipeline_quant_config("m", cfg, "bfloat16")
    out = capsys.readouterr().out
    offending = [ln for ln in out.splitlines()
                 if "FULL PRECISION" in ln and "vae" in ln.lower()]
    assert not offending, f"'none' was called out as a mistake: {offending}"


def test_a_bnb_mode_without_bitsandbytes_is_reported_not_raised(capsys, monkeypatch):
    """Constructing a BitsAndBytesConfig without the package raises.

    The import succeeds either way; only construction consults
    importlib.metadata, so a Vulkan-only or CPU install got
    PackageNotFoundError("No package metadata was found for bitsandbytes") --
    an error naming neither the model nor the setting that asked for it. It is
    a configuration to report, and reported as loudly as an unrecognised mode,
    because the consequence is the same: full precision.
    """
    import codai.models.hf_loading as H

    class _Boom:
        def __init__(self, *a, **kw):
            raise RuntimeError("No package metadata was found for bitsandbytes")

    monkeypatch.setattr("diffusers.BitsAndBytesConfig", _Boom, raising=False)
    monkeypatch.setattr("transformers.BitsAndBytesConfig", _Boom, raising=False)
    qc, desc = H.build_pipeline_quant_config(
        "m", {"component_quantization": {"text_encoder": "4bit"}}, "bfloat16")
    out = capsys.readouterr().out
    assert "bitsandbytes" in out and "FULL PRECISION" in out, out
    # And it must not pretend the component was quantized.
    assert "text_encoder" not in (desc or "")
