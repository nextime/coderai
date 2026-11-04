"""A distill LoRA that failed to load must not leave the entry's step count.

`_generate_video` already had a guard for this, and the guard did not work.
It declines to apply the acceleration PRESET's step count when
`_coderai_accel_fused` is False, and prints "using safe defaults (25 steps)".
But the entry's own `defaults` are applied two blocks further down with
`setdefault`, and by then nothing has been set -- so an entry that carries
`steps: 8` *because* it configures a distill LoRA still ran 8 steps after that
LoRA failed, at the `guidance_scale: 1.0` the same entry declares, which is no
classifier-free guidance either. The exact failure the guard exists to prevent,
arriving one block later, under a log line claiming it had been prevented.

Measured 2026-11-04 on both LTX-2.3 entries: neither had a working distill LoRA
(see test_a_distill_lora_is_scaled_as_the_file_declares) and both ran 8 steps at
CFG 1.0 through a client's 576-run sweep, scoring 1.4/5 and 2.9/5.

The request still wins. A client that asks for 8 steps gets 8 steps -- this only
refuses to let a model CARD impose a step count that is survivable only with an
adapter the model does not have.
"""
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = (ROOT / "codai" / "api" / "video.py").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def guard():
    """The clamp, which must sit AFTER the last setdefault that can set steps."""
    # rfind, not find: the earlier `elif _accel and _accel_fused is False:`
    # (which only skips the PRESET's step count) contains this as a substring,
    # and anchoring on it would test the guard that did not work.
    i = SRC.rfind("if _accel and _accel_fused is False:")
    j = SRC.index("kw.setdefault('num_inference_steps', 25)")
    assert i > j, ("the clamp runs before the generic defaults, so it reads a "
                   "step count that has not been decided yet")
    return SRC[i:i + 1600]


def test_the_clamp_exists_after_every_default(guard):
    assert "kw['num_inference_steps'] = 25" in guard
    assert "kw['guidance_scale'] = 7.5" in guard


def test_a_low_step_count_from_the_card_is_raised(guard):
    """8 steps un-distilled is the reported failure; the threshold has to be
    above it and below the generic 25."""
    m = re.search(r"int\(kw\['num_inference_steps'\]\) < (\d+)", guard)
    assert m, "the clamp does not compare the resolved step count"
    assert 8 < int(m.group(1)) <= 25


def test_cfg_1_is_raised_because_it_disables_guidance(guard):
    """CFG 1.0 means no classifier-free guidance at all — a distilled-model
    setting that makes an un-distilled model washed out."""
    assert re.search(r"float\(kw\['guidance_scale'\]\) <= 1(\.0)?", guard)


def test_the_clients_own_numbers_are_never_overridden(guard):
    """Snapshotted before any default is applied, because everything below uses
    setdefault — that is the only point where "asked for" and "defaulted" can
    still be told apart."""
    assert "_steps_from_request" in guard and "_cfg_from_request" in guard
    assert "not _steps_from_request" in guard
    assert "not _cfg_from_request" in guard
    snap = SRC.index("_steps_from_request = 'num_inference_steps' in kw")
    assert snap < SRC.index("if _accel and _accel_fused is not False:"), (
        "the snapshot is taken after the preset defaults, so a preset value "
        "would be mistaken for the client's own")
    assert snap > SRC.index("kw = _build_call_kwargs(request, model_name, model_cfg)")


def test_it_only_fires_when_acceleration_was_configured_and_failed(guard):
    """A model with no acceleration at all keeps its card's step count: a
    legitimately 8-step checkpoint must not be forced to 25."""
    assert guard.startswith("if _accel and _accel_fused is False:")


def test_both_clamps_say_why_in_the_log(guard):
    """The old guard's log line claimed an outcome it did not produce. These
    name the number they are overriding and the reason."""
    assert "failed to load" in guard
    assert "classifier-free guidance" in guard
    assert guard.count("flush=True") >= 2
