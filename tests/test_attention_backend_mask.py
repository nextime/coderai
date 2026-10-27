"""SageAttention cannot take an attention mask, so `auto` must not pick it.

LTX-2 masks its text conditioning. The model loaded cleanly and then failed at
the first denoising step with

    ValueError: `attn_mask` is not supported for sage attention

which makes sage not a slower path for that family but a broken one. The
backend resolver already gates on "can diffusers actually dispatch to sage";
this is the same gate, extended to the model.
"""
import types

import pytest

import codai.api.video as V


class _Comp:
    def __init__(self):
        self.backend = None

    def set_attention_backend(self, name):
        self.backend = name


def _pipe(cls_name):
    ns = {"transformer": _Comp(), "transformer_2": None}
    return type(cls_name, (), ns)()


@pytest.mark.parametrize("cls,expect", [
    ("LTX2Pipeline", True),
    ("LTX2ImageToVideoPipeline", True),
    ("WanPipeline", False),
    ("LTXPipeline", False),          # the 0.9 family is a different thing
    ("CogVideoXPipeline", False),
])
def test_only_the_masking_families_are_recognised(cls, expect):
    assert V._pipeline_uses_attn_mask(_pipe(cls)) is expect


@pytest.mark.parametrize("chosen", ["sage", "flash"])
def test_neither_fast_backend_is_given_to_a_masking_pipeline(monkeypatch, chosen):
    """Falling back from sage to flash is NOT a fix.

    Both raise on an attention mask, with the same sentence and a different
    name. A first version of this guard moved LTX-2 from sage to flash and it
    failed identically, so the only correct answer is the default SDPA.
    """
    monkeypatch.setattr(V, "_resolve_attn_backend", lambda cfg: chosen)
    p = _pipe("LTX2Pipeline")
    V._apply_attention_backend(p, {})
    assert p.transformer.backend is None, (
        f"{chosen} cannot take an attention mask either")


def test_a_non_masking_pipeline_still_gets_sage(monkeypatch):
    # The whole point of choosing sage is speed; this must not regress Wan.
    monkeypatch.setattr(V, "_resolve_attn_backend", lambda cfg: "sage")
    p = _pipe("WanPipeline")
    V._apply_attention_backend(p, {})
    assert p.transformer.backend == "sage"


def test_flash_still_reaches_a_non_masking_pipeline(monkeypatch):
    monkeypatch.setattr(V, "_resolve_attn_backend", lambda cfg: "flash")
    p = _pipe("WanPipeline")
    V._apply_attention_backend(p, {})
    assert p.transformer.backend == "flash"


def test_an_operator_who_names_sage_explicitly_is_obeyed(monkeypatch):
    """Their call to make, and they get the error.

    The override exists so an operator can force a backend; silently ignoring
    it would be a different kind of wrong from silently choosing a broken one.
    """
    monkeypatch.setattr(V, "_resolve_attn_backend", lambda cfg: "sage")
    p = _pipe("LTX2Pipeline")
    V._apply_attention_backend(p, {"video_attention_backend": "sage"})
    assert p.transformer.backend == "sage"


def test_nothing_configured_leaves_the_pipeline_alone(monkeypatch):
    monkeypatch.setattr(V, "_resolve_attn_backend", lambda cfg: None)
    p = _pipe("LTX2Pipeline")
    V._apply_attention_backend(p, {})
    assert p.transformer.backend is None
