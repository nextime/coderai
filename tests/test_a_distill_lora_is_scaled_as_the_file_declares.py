"""A distillation LoRA must be applied at the strength its checkpoint declares.

diffusers infers ONE ``lora_alpha`` for a whole adapter, from whichever module
happens to be first in the state dict::

    r = lora_alpha = list(rank_dict.values())[0]        # peft_utils.py

``r`` is then recomputed to the most common rank and the odd ones go into
``rank_pattern``, but ``lora_alpha`` is never revisited, and ``alpha_pattern``
is filled only from a ``network_alphas`` argument that LTX-2's loader hard-codes
to ``None``. peft scales each module by ``lora_alpha / r``, so on any LoRA whose
modules do not all share one rank, every module whose rank differs from the
first one's is scaled wrongly.

Measured 2026-11-04 on the two LTX-2.3 entries. Both checkpoints have the same
1660 modules at ranks 384/256/128/32, and 10Eros declares ``alpha == rank`` for
every one of them -- a scale of exactly 1.0. What they got instead:

    body (1360 modules, rank 384)   0.667x   <- PinkCherry, two thirds strength
    proj_in / proj_out  (rank 128)  2x
    *.to_gate_logits    (rank 32)   8x       <- a saturated attention gate

10Eros escaped the first row only because its most common rank happens to equal
the rank of the module diffusers reads the alpha from. It did not escape the
other two, and it had a worse problem besides: its 1660 ``<module>.alpha``
scalars survive LTX-2's converter (Wan's and Qwen's consume theirs), and
``load_lora_weights`` checks ``all("lora" in key)`` and refuses the whole
checkpoint. So its LoRA never loaded at all.

Scaling is corrected by baking ``declared_alpha / inferred_alpha`` into each
module's up-projection, which needs no cooperation from the loader and survives
``set_adapters``. An absent ``.alpha`` means ``alpha == rank``: the peft default,
and what the file that does declare them agrees on.
"""
import pytest

torch = pytest.importorskip("torch")

from codai.models.acceleration import (_normalise_lora_scaling,  # noqa: E402
                                       _prepared_lora)


def _mod(sd, name, rank, dim=64, alpha=None, bias=False):
    """One LoRA module, as a converted diffusers state dict carries it."""
    sd[f"{name}.lora_A.weight"] = torch.ones(rank, dim)
    sd[f"{name}.lora_B.weight"] = torch.ones(dim, rank)
    if bias:
        sd[f"{name}.lora_B.bias"] = torch.ones(dim)
    if alpha is not None:
        sd[f"{name}.alpha"] = torch.tensor(float(alpha))
    return sd


def _effective(orig, fixed):
    """{module: the scale peft will really apply, relative to the declared one}.

    Reproduces peft's own arithmetic: one inferred alpha per component group,
    divided by the module's rank, times whatever was baked into ``lora_B``.
    """
    ranks, declared, groups = {}, {}, {}
    for k, v in orig.items():
        if k.endswith(".alpha"):
            declared[k[: -len(".alpha")]] = float(v)
        elif "lora_B" in k and v.ndim > 1:
            ranks[k.split(".lora_B.")[0]] = int(v.shape[1])
    for m, r in ranks.items():
        groups.setdefault(m.split(".", 1)[0], float(r))
    out = {}
    for m, r in ranks.items():
        want = declared.get(m, float(r)) / r
        peft = groups[m.split(".", 1)[0]] / r
        a = orig[f"{m}.lora_B.weight"].float().norm()
        b = fixed[f"{m}.lora_B.weight"].float().norm()
        baked = float(b / a) if a > 0 else 1.0
        out[m] = round(peft * baked / want, 6)
    return out


# ── the keys diffusers refuses the whole checkpoint over ─────────────────────

def test_the_alpha_scalars_come_out_or_nothing_loads_at_all():
    """`load_lora_weights` checks `all("lora" in key)`. One `.alpha` and the
    ValueError takes the entire adapter with it."""
    sd = _mod({}, "transformer.blocks.0.attn.to_q", 64, alpha=64)
    assert not all("lora" in k for k in sd), "fixture does not reproduce it"
    fixed, note = _normalise_lora_scaling(sd)
    assert all("lora" in k for k in fixed)
    assert "transformer.blocks.0.attn.to_q.alpha" not in fixed
    assert "non-peft" in (note or ""), "a dropped key must be reported"


def test_a_uniform_rank_lora_is_left_exactly_alone():
    """The common case — one rank, alpha == rank. Touching it would be a
    regression risk for every Lightning checkpoint that works today."""
    sd = {}
    for i in range(4):
        _mod(sd, f"transformer.blocks.{i}.attn.to_q", 64)
    fixed, note = _normalise_lora_scaling(sd)
    assert note is None, f"nothing needed changing, but got: {note}"
    for k, v in sd.items():
        assert torch.equal(fixed[k], v), k


# ── the scaling itself ───────────────────────────────────────────────────────

def test_mixed_ranks_all_end_up_at_the_declared_strength():
    """The LTX-2.3 shape: a rank-256 module first, a rank-384 body behind it,
    and small rank-32 gates. Un-corrected those come out 0.667x and 8x."""
    sd = {}
    _mod(sd, "transformer.time_embed.linear_1", 256)       # first => alpha=256
    for i in range(6):
        _mod(sd, f"transformer.blocks.{i}.attn.to_q", 384)
    for i in range(3):
        _mod(sd, f"transformer.blocks.{i}.attn.to_gate_logits", 32)
    fixed, note = _normalise_lora_scaling(sd)
    got = _effective(sd, fixed)
    assert set(got.values()) == {1.0}, f"not at declared strength: {got}"
    assert "corrected the scaling" in (note or "")
    # And the damage it prevented, stated explicitly.
    before = _effective(sd, {k: v for k, v in sd.items() if "lora" in k})
    assert before["transformer.blocks.0.attn.to_q"] == pytest.approx(0.6667, abs=1e-3)
    assert before["transformer.blocks.0.attn.to_gate_logits"] == 8.0


def test_a_declared_alpha_that_is_not_the_rank_is_honoured():
    """alpha != rank is a legitimate choice; the file decides, not the loader."""
    sd = {}
    _mod(sd, "transformer.a", 64, alpha=64)
    _mod(sd, "transformer.b", 128, alpha=32)      # deliberately a quarter
    fixed, _ = _normalise_lora_scaling(sd)
    assert set(_effective(sd, fixed).values()) == {1.0}


def test_each_component_group_gets_its_own_inferred_alpha():
    """`load_lora_weights` splits the dict by prefix and configures each group
    separately, so one global correction would be wrong for one of them."""
    sd = {}
    _mod(sd, "transformer.first", 256)
    _mod(sd, "transformer.body", 384)
    _mod(sd, "connectors.first", 32)
    _mod(sd, "connectors.body", 128)
    fixed, _ = _normalise_lora_scaling(sd)
    assert set(_effective(sd, fixed).values()) == {1.0}


def test_the_up_projection_bias_is_scaled_with_its_weight():
    """peft applies its scale to the whole up-projection; a bias left behind
    would be applied at the wrong strength."""
    sd = {}
    _mod(sd, "transformer.first", 256)
    _mod(sd, "transformer.body", 384, bias=True)
    fixed, _ = _normalise_lora_scaling(sd)
    w = fixed["transformer.body.lora_B.weight"].float().mean()
    b = fixed["transformer.body.lora_B.bias"].float().mean()
    assert float(w) == pytest.approx(float(b), rel=1e-5)
    assert float(b) == pytest.approx(384 / 256, rel=1e-4)


def test_the_weights_keep_their_dtype():
    """A bf16 adapter must not come back as fp32 and double its footprint."""
    sd = {}
    _mod(sd, "transformer.first", 256)
    _mod(sd, "transformer.body", 384)
    for k in list(sd):
        sd[k] = sd[k].to(torch.bfloat16)
    fixed, _ = _normalise_lora_scaling(sd)
    assert all(v.dtype == torch.bfloat16 for v in fixed.values())


def test_a_dict_with_no_lora_b_at_all_is_not_a_crash():
    """Defensive: an unexpected layout must degrade, not raise — the caller's
    fallback is to hand the reference to diffusers unchanged."""
    fixed, note = _normalise_lora_scaling({"transformer.x.lora_magnitude": torch.ones(2)})
    assert "lora" in next(iter(fixed))
    assert _normalise_lora_scaling({}) == ({}, None)


# ── the fallback ─────────────────────────────────────────────────────────────

def test_a_pipeline_that_cannot_be_pre_read_falls_back(monkeypatch):
    """A LoRA we cannot read ahead of time must still get its chance to load
    the old way, or this fix becomes a new way for acceleration to fail."""
    class _NoLoader:
        pass

    assert _prepared_lora(_NoLoader(), "some/repo", None) == (None, None)

    class _Raises:
        @classmethod
        def lora_state_dict(cls, *a, **k):
            raise OSError("offline")

    assert _prepared_lora(_Raises(), "some/repo", None) == (None, None)

    class _Empty:
        @classmethod
        def lora_state_dict(cls, *a, **k):
            return {}

    assert _prepared_lora(_Empty(), "some/repo", None) == (None, None)


def test_a_tuple_returning_loader_is_unwrapped():
    """`lora_state_dict` returns (sd, metadata) when asked for metadata, and a
    plain dict otherwise; both have to work."""
    sd = _mod({}, "transformer.a", 64, alpha=64)

    class _Tuple:
        @classmethod
        def lora_state_dict(cls, *a, **k):
            return sd, None

    got, note = _prepared_lora(_Tuple(), "some/repo", None)
    assert got is not None and all("lora" in k for k in got)


def test_the_weight_name_is_passed_through():
    """A `repo:file.safetensors` ref names one file in a multi-LoRA repo — the
    Wan2.2 dual-expert entries depend on it."""
    seen = {}

    sd = _mod({}, "transformer.a", 64)

    class _Pipe:
        @classmethod
        def lora_state_dict(cls, repo, **kw):
            seen.update(repo=repo, **kw)
            return sd

    _prepared_lora(_Pipe(), "lightx2v/Wan2.2-Lightning", "high_noise_model.safetensors")
    assert seen["weight_name"] == "high_noise_model.safetensors"
    assert seen["repo"] == "lightx2v/Wan2.2-Lightning"


def test_an_owned_dict_is_released_as_it_is_copied():
    """These adapters are large — PinkCherry's is 7.25 GB, kept as a RUNTIME
    adapter rather than fused, so it stays resident beside the transformer.
    Holding the original and the rescaled copy at once would transiently double
    that on a box with a host-RAM ceiling."""
    sd = {}
    _mod(sd, "transformer.first", 256)
    _mod(sd, "transformer.body", 384)
    fixed, _ = _normalise_lora_scaling(sd, owned=True)
    assert "transformer.body.lora_B.weight" not in sd, (
        "the original is still referenced after being copied")
    assert "transformer.body.lora_B.weight" in fixed
    # Untouched modules are not copied at all, so there is nothing to release.
    assert "transformer.first.lora_B.weight" in sd
    # The rescaling itself is unaffected by who owns the input: 384/256 = 1.5.
    assert float(fixed["transformer.body.lora_B.weight"].mean()) == pytest.approx(
        384 / 256, rel=1e-5)
    assert float(fixed["transformer.first.lora_B.weight"].mean()) == 1.0


def test_the_default_leaves_the_callers_dict_alone():
    """Only _prepared_lora owns its dict; anyone else keeps theirs."""
    sd = {}
    _mod(sd, "transformer.first", 256)
    _mod(sd, "transformer.body", 384)
    before = dict(sd)
    _normalise_lora_scaling(sd)
    assert set(sd) == set(before)
    assert all(sd[k] is before[k] for k in before)
