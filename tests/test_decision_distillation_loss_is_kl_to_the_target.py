"""The decision loss is a soft cross-entropy, read only at the last position.

Two properties make it correct:

* it equals ``-sum(p_i * log_softmax(logits[-1])[id_i])`` -- a cross-entropy
  from the target distribution to the model's, taken over the FULL-vocab
  softmax and gathered at the letter ids (not renormalised over the letters, a
  quantity the serving side never computes); so minimising it drives the
  model's answer-letter probabilities toward the target, and it bottoms out at
  the target's own entropy when the model matches it;
* it reads ONLY the final position: every earlier position's logits must get no
  gradient, or the "decision at the end of the prompt" framing is a fiction.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from codai.api.text_lora_trainer import decision_distillation_loss


def test_it_equals_the_soft_cross_entropy_formula():
    vocab, seq = 10, 4
    torch.manual_seed(0)
    logits = torch.randn(1, seq, vocab)
    ids = [2, 5, 7]
    p = [0.5, 0.3, 0.2]

    got = decision_distillation_loss(logits, ids, p).item()

    last = logits[0, -1, :]
    logp = torch.log_softmax(last, dim=-1)
    want = -sum(pi * logp[i].item() for pi, i in zip(p, ids))
    assert got == pytest.approx(want, rel=1e-5)


def test_unnormalised_targets_are_normalised():
    logits = torch.randn(1, 3, 8)
    ids = [1, 4]
    a = decision_distillation_loss(logits, ids, [3.0, 1.0]).item()
    b = decision_distillation_loss(logits, ids, [0.75, 0.25]).item()
    assert a == pytest.approx(b, rel=1e-6)


def test_the_minimum_is_the_target_entropy_when_the_model_matches():
    """When the model puts all its mass on the letters in the target ratio, the
    loss equals the target's entropy -- the floor of a cross-entropy."""
    vocab = 6
    ids = [0, 1]
    p = [0.7, 0.3]
    # Build logits whose softmax is ~ [0.7, 0.3, 0, 0, 0, 0] at the last pos.
    big = 30.0
    last = torch.full((vocab,), -big)
    last[0] = math.log(0.7) + big
    last[1] = math.log(0.3) + big
    logits = torch.zeros(1, 2, vocab)
    logits[0, -1, :] = last
    got = decision_distillation_loss(logits, ids, p).item()
    entropy = -sum(pi * math.log(pi) for pi in p)
    assert got == pytest.approx(entropy, abs=1e-3)


def test_only_the_last_position_carries_gradient():
    vocab, seq = 12, 5
    logits = torch.randn(1, seq, vocab, requires_grad=True)
    loss = decision_distillation_loss(logits, [3, 8], [0.6, 0.4])
    loss.backward()
    grad = logits.grad[0]
    # earlier positions: exactly zero
    assert torch.count_nonzero(grad[:-1]) == 0
    # last position: non-zero
    assert torch.count_nonzero(grad[-1]) > 0


def test_raising_a_letters_logit_lowers_the_loss():
    """The gradient points the right way: more mass on a target letter helps."""
    vocab = 8
    ids = [2, 5]
    p = [0.9, 0.1]
    base = torch.zeros(1, 1, vocab)
    loss0 = decision_distillation_loss(base, ids, p).item()
    up = base.clone()
    up[0, -1, 2] += 2.0                      # push the dominant letter up
    loss1 = decision_distillation_loss(up, ids, p).item()
    assert loss1 < loss0
