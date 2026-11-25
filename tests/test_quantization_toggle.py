"""A training run can opt out of 4-bit and train a full-dtype LoRA.

The default stays 4-bit QLoRA (``quantization="nf4"``), the historical
behaviour. ``quantization="none"`` (and its aliases) trains a LoRA on the full
``compute_dtype`` base instead -- the path for a base the installed
bitsandbytes cannot 4-bit, e.g. spark2_5 under transformers 4.57.1, where the
4-bit quant_state is never consumed and raw uint8 weights reach the matmul.
``is_4bit`` is the single predicate that decides this, and
``describe_loaded_base`` must not cry "failure" when an unquantized base is
exactly what was asked for.
"""
import torch

from codai.api import text_lora_trainer as tlt
from codai.api.text_lora_trainer import QLoraSettings, is_4bit


def test_default_is_four_bit_qlora():
    assert is_4bit(QLoraSettings()) is True
    assert is_4bit(QLoraSettings(quantization="nf4")) is True
    assert is_4bit(QLoraSettings(quantization="fp4")) is True


def test_none_and_aliases_mean_unquantized():
    for v in ("none", "None", "off", "no", "false", "bf16", "fp16", "full"):
        assert is_4bit(QLoraSettings(quantization=v)) is False, v


def test_blank_falls_back_to_the_four_bit_default():
    # A blank/None value is "use the default", not "unquantize" -- unquantizing
    # must always be explicit, never what a missing field silently does.
    assert is_4bit(QLoraSettings(quantization="")) is True


class _Tiny(torch.nn.Module):
    """A 2-D weight in bf16 -- what an unquantized base looks like to the report
    (bnb would instead leave uint8 blocks, which the report keys on)."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(4, 4, dtype=torch.bfloat16))


def test_report_does_not_cry_failure_for_an_intended_unquantized_run():
    m = _Tiny()
    expected = tlt.describe_loaded_base(m, expected_4bit=False)
    assert expected["quantized_4bit"] is False
    assert expected["expected_4bit"] is False
    assert "NOT QUANTIZED" not in expected["line"]
    assert "as requested" in expected["line"]


def test_report_still_shouts_when_four_bit_silently_failed():
    # 4-bit was asked for but the base came back full-dtype -- the silent
    # failure the report exists to catch.
    m = _Tiny()
    got = tlt.describe_loaded_base(m, expected_4bit=True)
    assert got["quantized_4bit"] is False
    assert "NO - THE BASE IS NOT QUANTIZED" in got["line"]
