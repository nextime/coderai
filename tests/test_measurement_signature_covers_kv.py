"""A learned GPU-layer split must not outlive the KV settings it was learned under.

`measured_n_gpu_layers` is reused as a cap on later loads, so the model jumps
straight to the split that fit instead of rediscovering it through OOM retries.
`measured_quant_sig` is what stops a measurement outliving the configuration it
describes -- and it covered quantization only.

Measured 2026-10-30: setting swa_full=False on a gemma card cut its KV
reservation by ~19 GB (its 25 sliding layers stop reserving the full 178k
context). The learned split of 18 of 30 layers, measured with the old full-SWA
KV, was then far too conservative -- and the signature did not change, so it
would have survived and pinned the model to CPU on a card that could now hold
nearly all of it. That is the layer-ratchet bug arriving through a setting
nothing was watching.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codai.models.manager import MultiModelManager


@pytest.fixture
def sig():
    m = MultiModelManager.__new__(MultiModelManager)
    return m._quant_signature


BASE = {"precision": "bf16", "n_ctx": 32768, "cache_type_k": "q4_0",
        "cache_type_v": "q4_0", "swa_full": False}


def test_identical_config_gives_the_same_signature(sig):
    assert sig(dict(BASE)) == sig(dict(BASE))


@pytest.mark.parametrize("field,other", [
    ("swa_full", True),
    ("n_ctx", 178000),
    ("cache_type_k", "q8_0"),
    ("cache_type_v", "f16"),
    ("kv_offload", False),
])
def test_changing_a_kv_setting_invalidates_the_measurement(sig, field, other):
    """Each of these changes how much of the card the KV takes, and therefore
    how many layers fit."""
    changed = dict(BASE)
    changed[field] = other
    assert sig(changed) != sig(BASE), f"{field} does not invalidate the measurement"


def test_the_quantization_fields_still_count(sig):
    """The original purpose must survive the extension."""
    for field, other in (("precision", "f32"), ("load_in_4bit", True),
                         ("load_in_8bit", True),
                         ("component_quantization", {"transformer": "nf4"})):
        changed = dict(BASE)
        changed[field] = other
        assert sig(changed) != sig(BASE), field


def test_offload_kqv_is_read_as_a_spelling_of_kv_offload(sig):
    """The loader accepts either name, so the signature has to too -- otherwise
    a card written one way keeps a measurement taken the other way."""
    a = dict(BASE); a["kv_offload"] = False
    b = dict(BASE); b["offload_kqv"] = False
    assert sig(a) == sig(b)


def test_an_unrelated_field_does_not_invalidate_it(sig):
    """Over-invalidating throws away every measurement and restores the OOM
    ladder on every load, which is the cost this whole mechanism avoids."""
    for field, val in (("alias", "lisa"), ("load_mode", "on-request"),
                       ("n_batch", 256), ("measured_n_gpu_layers", 18),
                       ("used_vram_gb", 17.6), ("flash_attention", True)):
        changed = dict(BASE)
        changed[field] = val
        assert sig(changed) == sig(BASE), f"{field} should not invalidate"


def test_a_missing_field_is_stable(sig):
    """An absent setting and an explicitly-null one are the same thing, so a
    card that never set swa_full keeps its measurement when nothing changed."""
    a = {"precision": "bf16", "n_ctx": 32768}
    b = dict(a); b["swa_full"] = None
    assert sig(a) == sig(b)


def test_it_survives_nonsense_without_raising(sig):
    """It runs on the load path; a bad config must cost the measurement, not
    the load."""
    assert isinstance(sig(None), str)
    assert isinstance(sig({}), str)
    assert isinstance(sig({"n_ctx": object()}), str)


def test_the_signature_is_recorded_and_compared_against_itself():
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'codai', 'models', 'manager.py')).read()
    assert '_persist("measured_quant_sig", self._quant_signature(cfgd))' in src
    assert 'cfgd.get("measured_quant_sig") == self._quant_signature(cfgd)' in src
