# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""Setting an OCR knob to 0 must reach the pod, not be read as "unset".

A rented pod has no config.json: environment is the only channel. The seeder
dropped every zero, on the reading that `0` means "I did not configure this".
For a field whose own default is non-zero that is backwards -- 0 is the
operator switching the thing OFF -- and the drop handed the pod the default
they were trying to remove.

Measured on the Digesta orchestrator, 2026-11-06: `ocr.chandra_max_tokens`
was lowered to 4096 and `ocr.chandra_max_continuations` set to 0 to stop a
page spending its budget several times over. The cap travelled; the zero did
not, so every pod rented afterwards would still have read pages with 8
continuations -- up to 8x the budget the operator had just set, with nothing
anywhere saying the setting had been discarded.

The same loop already carries a comment about this exact trap for booleans
("that filter drops any value equal to 0 and False == 0 in Python, so 'off'
sent through there would be discarded and the pod would run with the feature
on, which is exactly what made it impossible to A/B"). The integer budgets have
the identical property and were left in the loop that eats zeros.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codai.api import runpod_worker as rw
from codai.config import OcrConfig


@pytest.fixture
def ocr_cfg(monkeypatch):
    holder = {}

    def _set(**kw):
        holder["cfg"] = OcrConfig(**kw)
        return holder["cfg"]

    monkeypatch.setattr(rw, "_ocr_cfg", lambda: holder.get("cfg"))
    monkeypatch.setattr(rw, "_surya_accepted", lambda: True)
    return _set


class _Mcfg:
    def __getattr__(self, _name):
        return None


def _seed(engine: str, image: str = "coderai-vllm-ocr") -> dict:
    """The env for a pod that runs the OCR PIPELINE itself -- the image that
    makes these budgets the pod's own business rather than the orchestrator's."""
    return rw._plan(engine="", image=image, args={}, mcfg=_Mcfg(),
                    api_key="k", entry={"path": engine}, served=engine,
                    capability="ocr").get("env", {})


# --- the discriminator ----------------------------------------------------

def test_zero_is_a_choice_where_the_default_is_not_zero():
    o = OcrConfig()
    assert rw._deliberate_zero(o, "chandra_max_continuations") is True
    assert rw._deliberate_zero(o, "chandra_max_retries") is True
    assert rw._deliberate_zero(o, "chandra_max_tokens") is True


def test_zero_is_not_a_choice_where_the_default_is_already_zero():
    """`*_max_tokens_floor` defaults to 0 and 0 means off for everyone, so
    sending it would be noise rather than a setting."""
    assert rw._deliberate_zero(OcrConfig(), "chandra_max_tokens_floor") is False


def test_an_unknown_field_is_never_called_deliberate():
    assert rw._deliberate_zero(OcrConfig(), "no_such_field_here") is False


def test_a_config_that_is_not_a_dataclass_says_no():
    """Never guess from a stand-in object: not knowing the default means not
    knowing whether the zero was meant."""
    class _Plain:
        chandra_max_continuations = 0
    assert rw._deliberate_zero(_Plain(), "chandra_max_continuations") is False


# --- what the pod is actually told ----------------------------------------

def test_continuations_off_reaches_the_pod(ocr_cfg):
    """The regression itself."""
    ocr_cfg(chandra_enabled=True, chandra_serve="model",
            chandra_model_id="datalab-to/chandra-ocr-2",
            chandra_max_tokens=4096, chandra_max_continuations=0)
    env = _seed("chandra")
    assert env["CODERAI_OCR_CHANDRA_MAX_TOKENS"] == "4096"
    assert env["CODERAI_OCR_CHANDRA_MAX_CONTINUATIONS"] == "0"


def test_retries_off_reaches_the_pod(ocr_cfg):
    """`chandra_max_retries` defaults to 2, so 0 is "do not re-run a page"."""
    ocr_cfg(chandra_enabled=True, chandra_serve="model",
            chandra_model_id="datalab-to/chandra-ocr-2",
            chandra_max_retries=0)
    assert _seed("chandra")["CODERAI_OCR_CHANDRA_MAX_RETRIES"] == "0"


def test_a_default_zero_is_still_left_out(ocr_cfg):
    """The old behaviour, where it was right: an unset floor stays unset."""
    ocr_cfg(chandra_enabled=True, chandra_serve="model",
            chandra_model_id="datalab-to/chandra-ocr-2",
            chandra_max_tokens_floor=0)
    assert "CODERAI_OCR_CHANDRA_MAX_TOKENS_FLOOR" not in _seed("chandra")


def test_non_zero_values_are_unaffected(ocr_cfg):
    """The fix must not change the path every existing pod was rented on."""
    ocr_cfg(chandra_enabled=True, chandra_serve="model",
            chandra_model_id="datalab-to/chandra-ocr-2",
            chandra_max_tokens=12384, chandra_max_continuations=8,
            chandra_instances=64)
    env = _seed("chandra")
    assert env["CODERAI_OCR_CHANDRA_MAX_TOKENS"] == "12384"
    assert env["CODERAI_OCR_CHANDRA_MAX_CONTINUATIONS"] == "8"
    assert env["CODERAI_OCR_CHANDRA_INSTANCES"] == "64"


def test_every_engine_gets_the_same_treatment(ocr_cfg):
    """A per-name fix is how surya ended up the only engine that worked."""
    ocr_cfg(surya_enabled=True, surya_serve="model",
            surya_model_id="datalab-to/surya-ocr-2", surya_accept_license=True,
            surya_max_continuations=0)
    assert _seed("surya")["CODERAI_OCR_SURYA_MAX_CONTINUATIONS"] == "0"

    ocr_cfg(olmocr_enabled=True, olmocr_serve="model",
            olmocr_model_id="allenai/olmOCR-2-7B-1025-FP8",
            olmocr_max_continuations=0)
    assert _seed("olmocr")["CODERAI_OCR_OLMOCR_MAX_CONTINUATIONS"] == "0"


def test_the_pod_reads_back_what_it_is_sent():
    """A sent-and-never-read setting reverts to a default while the UI shows
    the operator's value -- the sibling failure this file's cousin covers."""
    src = open("codai/config.py").read()
    for name in ("CODERAI_OCR_CHANDRA_MAX_CONTINUATIONS",
                 "CODERAI_OCR_CHANDRA_MAX_RETRIES"):
        suffix = name.replace("CODERAI_OCR_CHANDRA_", "")
        assert (name in src
                or f'f"CODERAI_OCR_{{' in src and suffix.lower() in src.lower())
