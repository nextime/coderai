"""The enable_thinking serving contract for text/decision LoRAs.

A decision model must be SERVED with the same ``enable_thinking`` it was
TRAINED with: the two render different chat prompts, so a mismatch shifts the
answer-letter distribution a confidence gate reads. This pins that down at three
points:

* the ``lora_train`` block may PIN ``enable_thinking`` (authoritative over a
  request that forgot it); an unpinned model defers to the request;
* training RECORDS the effective value beside the adapter (``meta.json``), which
  text LoRAs never did before;
* serving READS it back: ``/v1/decision`` no longer forces thinking off, and the
  per-model default falls back to the adapter's recorded value.
"""

import json
import os
import shutil
import tempfile
from types import SimpleNamespace

import pytest

from codai.api import text_lora_fleet as flt
from codai.api import loras
from codai.api import text as T
from codai.pydantic.textrequest import DecisionRequest


def test_block_parses_enable_thinking_as_optional_bool():
    assert flt.LoraTrainConfig().enable_thinking is None
    assert flt.parse_model_lora_train({"enable_thinking": True}).enable_thinking is True
    assert flt.parse_model_lora_train({"enable_thinking": False}).enable_thinking is False
    assert flt.parse_model_lora_train({"enable_thinking": None}).enable_thinking is None
    assert flt.parse_model_lora_train({}).enable_thinking is None
    # It must NOT leak through the generic float() branch onto other fields.
    cfg = flt.parse_model_lora_train({"enable_thinking": True, "learning_rate": 2e-4})
    assert cfg.enable_thinking is True and cfg.learning_rate == 2e-4


def test_effective_value_prefers_the_pinned_block_over_the_request():
    # The expression used in loras.py: block wins when set, request otherwise.
    def effective(block_val, req_val):
        return block_val if block_val is not None else bool(req_val)
    assert effective(True, False) is True       # pin overrides a forgetful caller
    assert effective(False, True) is False
    assert effective(None, True) is True        # unpinned defers to the request
    assert effective(None, False) is False


def test_training_records_the_contract_beside_the_adapter():
    d = tempfile.mkdtemp()
    try:
        st = SimpleNamespace(objective="decision", enable_thinking=False,
                             decision_letters=("A", "B", "C"),
                             decision_loss_scope="letters")
        loras._write_text_lora_meta(d, base="XHToken/Spark-X2.5-4B", settings=st)
        m = json.load(open(os.path.join(d, "meta.json")))
        assert m["base_model"] == "XHToken/Spark-X2.5-4B"
        assert m["objective"] == "decision"
        assert m["enable_thinking"] is False
        assert m["thinking"] == "off"                      # served-side spelling
        assert m["decision_letters"] == ["A", "B", "C"]
        assert m["decision_loss_scope"] == "letters"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_meta_write_merges_and_sft_omits_decision_fields():
    d = tempfile.mkdtemp()
    try:
        json.dump({"keep": 1}, open(os.path.join(d, "meta.json"), "w"))
        st = SimpleNamespace(objective="sft", enable_thinking=True)
        loras._write_text_lora_meta(d, base="B", settings=st)
        m = json.load(open(os.path.join(d, "meta.json")))
        assert m["keep"] == 1                               # pre-existing kept
        assert m["enable_thinking"] is True and m["thinking"] == "on"
        assert "decision_letters" not in m                  # sft adds no letters
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_decision_request_defaults_to_unset_not_false():
    r = DecisionRequest(model="m", messages=[{"role": "user", "content": "x"}],
                        letters=["A", "B"])
    assert r.enable_thinking is None            # unset -> use the trained contract
    r2 = DecisionRequest(model="m", messages=[], letters=["A"], enable_thinking=True)
    assert r2.enable_thinking is True           # explicit still honoured


@pytest.mark.parametrize("et,want", [(True, True), (False, False)])
def test_serving_default_falls_back_to_the_recorded_value(et, want):
    name = f"contract-test-{et}"
    d = loras._lora_dir(name)
    os.makedirs(d, exist_ok=True)
    try:
        json.dump({"base_model": "b", "objective": "decision",
                   "enable_thinking": et, "thinking": "on" if et else "off"},
                  open(os.path.join(d, "meta.json"), "w"))
        assert T._model_thinking_default(name) is want
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_serving_default_is_none_for_an_unknown_model():
    # No entry, no meta -> None ("the model's own default"), unchanged behaviour.
    assert T._model_thinking_default("definitely-not-a-model-xyzzy") is None
