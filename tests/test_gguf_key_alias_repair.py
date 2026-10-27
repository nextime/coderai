"""A converter and its model class disagreeing about a name is not a missing weight.

diffusers 0.38 converts an LTX-2 checkpoint's prompt-adaln weights to
`prompt_adaln_single.*` / `audio_prompt_adaln_single.*`, while
LTX2VideoTransformer3DModel expects `prompt_adaln.*` / `audio_prompt_adaln.*`.
Twelve names, one-to-one, the same tensors. The load failed with "Cannot copy
out of meta tensor" for parameters that were present all along -- and the
tempting "fix" was to initialise them, which would have invented weights to
paper over a rename.

Measured identically on LTX-2.3-10Eros v1.5 and PinkCherry v1.8: after the
rename, 0 parameters missing and 0 unused.
"""
import pytest

from codai.models.hf_loading import _rename_unexpected_keys


def test_a_suffix_only_mismatch_is_renamed():
    state = {"prompt_adaln_single.linear.weight": 1,
             "prompt_adaln_single.linear.bias": 2,
             "blocks.0.attn.to_q.weight": 3}
    expected = {"prompt_adaln.linear.weight", "prompt_adaln.linear.bias",
                "blocks.0.attn.to_q.weight"}
    out, moved = _rename_unexpected_keys(state, expected)
    assert len(moved) == 2
    assert set(out) == expected
    assert out["prompt_adaln.linear.weight"] == 1


def test_a_key_the_class_already_wants_is_left_alone():
    state = {"prompt_adaln.linear.weight": 1}
    expected = {"prompt_adaln.linear.weight"}
    out, moved = _rename_unexpected_keys(state, expected)
    assert moved == []
    assert out == state


def test_nothing_is_invented_for_a_genuinely_absent_parameter():
    """The property that matters.

    A rename moves a tensor the checkpoint HAS. If the class wants something
    the checkpoint simply lacks, this must leave it absent so the load still
    fails loudly -- that is the difference between repairing a name and
    fabricating a weight.
    """
    state = {"blocks.0.attn.to_q.weight": 1}
    expected = {"blocks.0.attn.to_q.weight", "audio_proj_in.weight"}
    out, moved = _rename_unexpected_keys(state, expected)
    assert moved == []
    assert "audio_proj_in.weight" not in out


def test_it_never_overwrites_an_existing_target():
    # If both spellings are present, the real one wins and nothing is clobbered.
    state = {"prompt_adaln.linear.weight": "real",
             "prompt_adaln_single.linear.weight": "other"}
    expected = {"prompt_adaln.linear.weight"}
    out, moved = _rename_unexpected_keys(state, expected)
    assert out["prompt_adaln.linear.weight"] == "real"
    assert moved == []


def test_the_suffix_is_matched_per_segment_not_as_a_substring():
    # "single" inside a longer word must not be stripped, and only a whole
    # path segment ending in the suffix is a candidate.
    state = {"my_single_block.weight": 1}
    expected = {"my_block.weight"}
    out, moved = _rename_unexpected_keys(state, expected)
    assert moved == []


def test_an_empty_or_unrelated_state_is_a_no_op():
    assert _rename_unexpected_keys({}, set()) == ({}, [])
    s = {"a.b": 1}
    assert _rename_unexpected_keys(s, {"a.b"}) == (s, [])


def test_the_patch_restores_the_converter_it_wrapped():
    # It wraps a THIRD-PARTY global for the duration of one load; leaving it
    # wrapped would change every later load in the process.
    import inspect

    from codai.models import hf_loading as H
    src = inspect.getsource(H._patch_converter_key_aliases)
    assert "_restore" in src
    assert 'entry["checkpoint_mapping_fn"] = original' in src
    caller = inspect.getsource(H.build_gguf_pipeline_components)
    assert "finally:" in caller and "_undo()" in caller


def test_it_returns_none_when_there_is_nothing_to_reconcile():
    from codai.models import hf_loading as H
    # An unknown class has no registered converter, so no patch is installed.
    assert H._patch_converter_key_aliases("NoSuchModelClass", None, {}, "x") is None


@pytest.mark.parametrize("real_gguf", [
    "/AI/hfcache/guffcache/10Eros_v1.5-Q4_K_M.gguf",
])
def test_the_real_checkpoint_maps_exactly_after_the_rename(real_gguf):
    """End to end on the actual file, when it is present."""
    import json
    import os
    import glob
    if not os.path.isfile(real_gguf):
        pytest.skip("checkpoint not on this machine")
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    gguf = pytest.importorskip("gguf")
    if not hasattr(diffusers, "LTX2VideoTransformer3DModel"):
        pytest.skip("this diffusers has no LTX-2 transformer")
    from diffusers.loaders.single_file_utils import (
        convert_ltx2_transformer_to_diffusers)
    conf = glob.glob("/AI/hfcache/huggingface/hub/"
                     "models--diffusers--LTX-2.3-Diffusers/snapshots/*/"
                     "transformer/config.json")
    if not conf:
        pytest.skip("LTX-2.3 base not on this machine")
    c = json.load(open(conf[0]))
    c.pop("_class_name", None)
    c.pop("_diffusers_version", None)
    with torch.device("meta"):
        m = diffusers.LTX2VideoTransformer3DModel.from_config(c)
    expected = set(dict(m.named_parameters())) | set(dict(m.named_buffers()))
    sd = {t.name: torch.zeros(1) for t in gguf.GGUFReader(real_gguf).tensors}
    converted = convert_ltx2_transformer_to_diffusers(sd, config={})
    before_missing = expected - set(converted)
    fixed, moved = _rename_unexpected_keys(converted, expected)
    after_missing = expected - set(fixed)

    # Version-independent invariants. The exact count (12, against diffusers
    # 0.38) depends on which diffusers is installed -- 0.37 ignores several of
    # this config's attributes and so builds a different module set -- so what
    # is asserted here is the PROPERTY, not the number.
    if not moved:
        # Nothing to reconcile on THIS diffusers. The mismatch is specific to
        # versions whose converter and class disagree (0.38 does; 0.37 has no
        # prompt_adaln modules at all, so the question does not arise). The
        # repair being a no-op is the correct outcome there, not a failure.
        pytest.skip(f"diffusers {diffusers.__version__}: converter and class "
                    f"already agree on every key")
    for old_key, new_key in moved:
        assert new_key in expected, new_key          # renamed onto a real slot
        assert old_key not in expected               # and away from a dead one
        assert fixed[new_key] is converted[old_key]  # the same tensor, moved
    assert after_missing < before_missing, "the rename must reduce the gap"
    assert not (after_missing - before_missing), "nothing new went missing"
    # And nothing was conjured: every key present afterwards came from the
    # converter, none were created.
    assert set(fixed) <= set(converted) | {n for _, n in moved}


# --------------------------------------------------------------------------- #
# LTX-2.5: the same disagreement, spelled the other way round
# --------------------------------------------------------------------------- #

def test_attention_norms_are_reconciled_both_ways():
    """LTX-2.5's GGUF names the attention norms `q_norm` / `k_norm` while
    LTX2VideoTransformer3DModel expects `norm_q` / `norm_k`: 724 parameters,
    one to one, the same tensors. Unreconciled they stay on the meta device
    and the load dies with "Cannot copy out of meta tensor" -- about weights
    the file carries under the other spelling.

    A suffix strip cannot express this one, so the repair also takes whole
    segments, and it takes them in both directions: nothing says which
    spelling the next exporter will pick."""
    state = {"transformer_blocks.0.attn2.q_norm.weight": "a",
             "transformer_blocks.0.attn2.k_norm.weight": "b",
             "transformer_blocks.0.audio_attn1.q_norm.weight": "c"}
    expected = {"transformer_blocks.0.attn2.norm_q.weight",
                "transformer_blocks.0.attn2.norm_k.weight",
                "transformer_blocks.0.audio_attn1.norm_q.weight"}
    out, moved = _rename_unexpected_keys(state, expected)
    assert len(moved) == 3
    assert set(out) == expected
    assert out["transformer_blocks.0.attn2.norm_q.weight"] == "a"

    # And a checkpoint spelled the diffusers way, against a class wanting the
    # other, is repaired too.
    back, moved_back = _rename_unexpected_keys(
        {"attn1.norm_q.weight": "x"}, {"attn1.q_norm.weight"})
    assert moved_back == [("attn1.norm_q.weight", "attn1.q_norm.weight")]
    assert back == {"attn1.q_norm.weight": "x"}


def test_both_repairs_apply_in_one_pass():
    """A checkpoint can need the suffix strip AND a segment rename; an
    `elif` between them would fix one family and leave the other on meta."""
    state = {"prompt_adaln_single.linear.weight": "p",
             "transformer_blocks.0.attn2.q_norm.weight": "q"}
    expected = {"prompt_adaln.linear.weight",
                "transformer_blocks.0.attn2.norm_q.weight"}
    out, moved = _rename_unexpected_keys(state, expected)
    assert len(moved) == 2 and set(out) == expected


def test_a_segment_rename_still_refuses_to_overwrite_or_invent():
    """Same guard as the suffix path: only when the class does not want the
    current name, does want the renamed one, and nothing holds the target."""
    # Target already occupied by a real weight: leave both alone.
    state = {"attn1.q_norm.weight": "new", "attn1.norm_q.weight": "real"}
    expected = {"attn1.norm_q.weight"}
    out, moved = _rename_unexpected_keys(state, expected)
    assert moved == [] and out["attn1.norm_q.weight"] == "real"

    # Class wants neither spelling: nothing is fabricated.
    out2, moved2 = _rename_unexpected_keys({"attn1.q_norm.weight": "x"}, set())
    assert moved2 == [] and out2 == {"attn1.q_norm.weight": "x"}
