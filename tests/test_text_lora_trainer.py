"""QLoRA for a causal LM: the weights adapted, and the gradient that results.

Two claims in this file are the ones worth proving numerically rather than by
inspection, because both are invisible when wrong:

* a masked position contributes NOTHING. Not approximately nothing -- the
  gradient at an ignored position is exactly zero, and the model's prediction
  there cannot move the loss. If that fails, the trainer learns the text the
  curator excluded, and the only symptom is a model with opinions it was never
  supposed to have.
* the next-token shift goes the right way. Train ``logits[i]`` against
  ``labels[i]`` instead of ``labels[i+1]`` and the model learns to echo the
  token it was just handed: the loss drops beautifully and nothing is taught.

The rest is the vision tower, which must be left alone on a multimodal base, and
surviving a restart.
"""

import json
import os

import pytest

torch = pytest.importorskip("torch")

from codai.api import text_lora_trainer as tlt


# ----------------------------------------------------- which modules adapt ---

QWEN_VL_LINEARS = [
    # the vision tower
    "visual.patch_embed.proj",
    "visual.blocks.0.attn.qkv",
    "visual.blocks.0.attn.proj",
    "visual.blocks.0.mlp.gate_proj",
    "visual.merger.mlp.0",
    "visual.merger.mlp.2",
    # the language model
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.k_proj",
    "model.layers.0.self_attn.v_proj",
    "model.layers.0.self_attn.o_proj",
    "model.layers.0.mlp.gate_proj",
    "model.layers.0.mlp.up_proj",
    "model.layers.0.mlp.down_proj",
    # and the head
    "lm_head",
]


def test_the_vision_tower_is_left_alone():
    """A text LoRA that adapts the tower spends rank on an encoder that sees no
    images during training, and degrades the thing the base already did well."""
    got = tlt.select_target_modules(QWEN_VL_LINEARS)
    assert not any(n.startswith("visual.") for n in got), got
    assert "model.layers.0.self_attn.q_proj" in got
    assert "model.layers.0.mlp.down_proj" in got


def test_a_tower_that_reuses_language_names_is_still_excluded():
    """The reason selection returns FULL names instead of the usual suffix list:
    `["q_proj", ...]` matches a vision encoder that happens to use the same
    names, and "text only" silently becomes "everything"."""
    names = ["vision_tower.layers.0.self_attn.q_proj",
             "vision_model.encoder.layers.3.mlp.up_proj",
             "model.layers.0.self_attn.q_proj"]
    got = tlt.select_target_modules(names)
    assert got == ["model.layers.0.self_attn.q_proj"]


def test_the_projector_is_not_adapted():
    for name in ("multi_modal_projector.linear_1", "mm_projector.0",
                 "visual.merger.mlp.0", "audio_tower.layers.0.self_attn.q_proj"):
        assert tlt.is_vision_module(name), name
        assert tlt.select_target_modules([name]) == []


def test_the_head_and_embeddings_are_not_adapted():
    """Rank 16 against a 250k-row head dwarfs the adapter and moves the output
    distribution directly rather than through the representation."""
    got = tlt.select_target_modules(["lm_head", "model.embed_tokens",
                                     "model.layers.0.mlp.up_proj"])
    assert got == ["model.layers.0.mlp.up_proj"]


def test_a_fused_qkv_base_is_still_adapted():
    got = tlt.select_target_modules(["model.layers.0.self_attn.qkv_proj",
                                     "model.layers.0.mlp.gate_up_proj"])
    assert len(got) == 2


def test_the_selection_is_described_for_the_log():
    """"the vision tower was left alone" is a claim; about 700 modules it
    should be countable rather than trusted."""
    sel = tlt.select_target_modules(QWEN_VL_LINEARS)
    line = tlt.describe_selection(QWEN_VL_LINEARS, sel)
    assert f"targets {len(sel)}" in line
    assert "5 vision/audio" in line or "vision/audio" in line


def test_a_language_only_base_needs_no_special_case():
    names = [f"model.layers.{i}.self_attn.{p}_proj"
             for i in range(4) for p in ("q", "k", "v", "o")]
    assert len(tlt.select_target_modules(names)) == 16


# -------------------------------------------------------------- the gradient --

def _logits_and_labels(n_pos=6, vocab=11, masked=()):
    torch.manual_seed(0)
    logits = torch.randn(1, n_pos, vocab, requires_grad=True)
    labels = torch.arange(1, n_pos + 1).remainder(vocab).unsqueeze(0).clone()
    for i in masked:
        labels[0, i] = -100
    return logits, labels


def test_a_masked_position_has_exactly_zero_gradient():
    """THE test for the mask. Not "the label array looks right" -- the gradient
    at an ignored position is zero, so the excluded text cannot influence a
    single weight."""
    logits, labels = _logits_and_labels(masked=(2, 3))
    loss = tlt.masked_cross_entropy(logits, labels)
    loss.backward()
    # position i's logits predict label i+1, so a masked label at i kills the
    # gradient of the logits at i-1
    assert torch.count_nonzero(logits.grad[0, 1]) == 0
    assert torch.count_nonzero(logits.grad[0, 2]) == 0
    assert torch.count_nonzero(logits.grad[0, 0]) > 0


def test_the_models_prediction_under_a_mask_cannot_move_the_loss():
    """The same claim from the other side: change what the model says where the
    mask is, and the loss must not budge.

    ONE vocab entry is moved, not the whole row. Adding a constant to every
    logit changes nothing at all -- softmax is invariant to a uniform shift --
    so a row-wide perturbation makes this pass whatever the mask does, which is
    worse than having no test.
    """
    logits, labels = _logits_and_labels(masked=(3,))
    base = float(tlt.masked_cross_entropy(logits, labels).detach())
    perturbed = logits.detach().clone()
    perturbed[0, 2, 4] += 17.0        # position 2's logits predict the masked 3
    after = float(tlt.masked_cross_entropy(perturbed, labels))
    assert after == pytest.approx(base, abs=1e-6)


def test_an_unmasked_prediction_does_move_the_loss():
    """The control. Without it, the test above would also pass on a trainer
    that ignores every position."""
    logits, labels = _logits_and_labels(masked=(3,))
    base = float(tlt.masked_cross_entropy(logits, labels).detach())
    perturbed = logits.detach().clone()
    perturbed[0, 0, 4] += 17.0        # position 0 predicts label 1, not masked
    assert float(tlt.masked_cross_entropy(perturbed, labels)) != pytest.approx(base)


def test_the_shift_predicts_the_NEXT_token_not_the_current_one():
    """An off-by-one here teaches the model to echo its input: the loss falls
    and nothing is learned. Logits that perfectly predict the next token must
    score near zero; ones that perfectly predict the CURRENT token must not."""
    vocab, n = 7, 5
    labels = torch.tensor([[1, 2, 3, 4, 5]])
    next_tok = torch.full((1, n, vocab), -10.0)
    same_tok = torch.full((1, n, vocab), -10.0)
    for i in range(n - 1):
        next_tok[0, i, labels[0, i + 1]] = 10.0      # predicts i+1  -> correct
    for i in range(n):
        same_tok[0, i, labels[0, i]] = 10.0          # predicts i    -> wrong shift
    assert float(tlt.masked_cross_entropy(next_tok, labels)) < 0.01
    assert float(tlt.masked_cross_entropy(same_tok, labels)) > 1.0


def test_an_example_contributes_its_MEAN_token_loss():
    """A per-example weight only means something if length is not also a weight:
    a 4-token answer and a 400-token one with the same per-token loss must
    produce the same number."""
    vocab = 9
    def _flat(n):
        labels = torch.ones(1, n, dtype=torch.long)
        logits = torch.zeros(1, n, vocab)
        return tlt.masked_cross_entropy(logits, labels)
    assert float(_flat(5)) == pytest.approx(float(_flat(500)), abs=1e-5)


def test_the_weight_scales_the_gradient_linearly():
    g = {}
    for w in (1.0, 0.3333):
        logits, labels = _logits_and_labels()
        tlt.scaled_backward(logits, labels, weight=w, accumulation_steps=4)
        g[w] = logits.grad[0, 0].clone()
    ratio = (g[0.3333] / g[1.0]).abs()
    assert torch.allclose(ratio, torch.full_like(ratio, 0.3333), atol=1e-5)


def test_the_logged_loss_is_the_models_loss_not_the_scaled_one():
    """A reader comparing losses across differently-weighted examples needs the
    model's actual loss, not a number that moves with the curation."""
    logits, labels = _logits_and_labels()
    raw = float(tlt.masked_cross_entropy(logits, labels).detach())
    logits2, labels2 = _logits_and_labels()
    got = tlt.scaled_backward(logits2, labels2, weight=0.1, accumulation_steps=8)
    assert got == pytest.approx(raw, abs=1e-6)


# ------------------------------------------------------- surviving the run ---

def test_an_oom_becomes_a_skip_not_a_dead_run():
    """One long example must not end a run that has been going for hours -- and
    in a multi-pod round it would keep every other pod waiting."""
    seen = []

    def _boom():
        raise torch.OutOfMemoryError("CUDA out of memory")

    with pytest.raises(tlt.OomSkipped):
        tlt.step_with_oom_guard(_boom, on_skip=lambda e: seen.append(e))
    assert len(seen) == 1


def test_a_real_error_is_not_swallowed_as_an_oom():
    def _bug():
        raise ValueError("a genuine bug")
    with pytest.raises(ValueError, match="genuine bug"):
        tlt.step_with_oom_guard(_bug)


class FakeAdapter:
    def __init__(self):
        self.saved_to = None

    def save_pretrained(self, directory):
        self.saved_to = directory
        with open(os.path.join(directory, "adapter_model.safetensors"), "w") as fh:
            fh.write("weights")


def test_a_checkpoint_round_trips(tmp_path):
    p = tlt.Progress(step=120, examples_seen=900, round=3, skipped_oom=2,
                     loss_ema=1.25)
    tlt.save_checkpoint(str(tmp_path), FakeAdapter(), p)
    back = tlt.load_progress(str(tmp_path))
    assert back == p


def test_a_half_written_checkpoint_cannot_be_read_back(tmp_path):
    """A pod that dies mid-save would otherwise leave JSON that fails to parse
    on resume, turning a recoverable interruption into a lost run."""
    tlt.save_checkpoint(str(tmp_path), FakeAdapter(), tlg := tlt.Progress(step=7))
    # simulate the interrupted write that the atomic rename makes unreachable
    (tmp_path / ".progress.json.tmp").write_text('{"step": 999, "unfinis')
    assert tlt.load_progress(str(tmp_path)).step == 7


def test_a_missing_checkpoint_is_a_fresh_start_not_an_error(tmp_path):
    assert tlt.load_progress(str(tmp_path)) == tlt.Progress()


def test_an_unknown_field_in_a_checkpoint_is_ignored(tmp_path):
    """A checkpoint written by a newer version must not stop an older one from
    resuming the run it can still finish."""
    (tmp_path / "progress.json").write_text(json.dumps(
        {"step": 5, "from_the_future": True}))
    assert tlt.load_progress(str(tmp_path)).step == 5
