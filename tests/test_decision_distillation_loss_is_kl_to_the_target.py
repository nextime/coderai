"""The decision loss is a soft cross-entropy over the answer letters, read only
at the last position.

The quantity ``/v1/decision`` reads back is the softmax over the letter logits
ALONE -- it takes each letter's probability and renormalises over the letters it
found, and renormalising a softmax over a subset is identically the softmax
restricted to that subset. So the loss trains that same quantity by default
(``scope="letters"``), and its defining property is clean: when the model's
letter softmax matches the target, the loss equals the target's ENTROPY --
whatever the off-menu mass is, because the off-menu logits are never in the
softmax. The older full-vocabulary behaviour (``scope="vocab"``) is kept for
reproducing an adapter trained before the letters scope existed; it also drives
off-menu mass down, so it sits strictly above the target entropy while any such
mass remains. Either way the loss reads ONLY the final position -- every earlier
position's logits must get no gradient, or "a decision at the end of the prompt"
is a fiction.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from codai.api.text_lora_trainer import decision_distillation_loss


def _logits_with(letter_ids, letter_vals, vocab, off=0.0, seq=1):
    """Last-position logits with the given values on the letters and ``off``
    everywhere else."""
    last = torch.full((vocab,), float(off))
    for i, v in zip(letter_ids, letter_vals):
        last[i] = float(v)
    logits = torch.zeros(1, seq, vocab)
    logits[0, -1, :] = last
    return logits


def test_letters_scope_equals_letter_only_cross_entropy():
    vocab, seq = 10, 4
    torch.manual_seed(0)
    logits = torch.randn(1, seq, vocab)
    ids = [2, 5, 7]
    p = [0.5, 0.3, 0.2]

    got = decision_distillation_loss(logits, ids, p).item()   # default: letters

    last = logits[0, -1, :]
    logp = torch.log_softmax(last[ids], dim=-1)               # letters alone
    want = -sum(pi * logp[k].item() for k, pi in enumerate(p))
    assert got == pytest.approx(want, rel=1e-5)


def test_vocab_scope_equals_full_vocab_cross_entropy():
    vocab, seq = 10, 4
    torch.manual_seed(1)
    logits = torch.randn(1, seq, vocab)
    ids = [2, 5, 7]
    p = [0.5, 0.3, 0.2]

    got = decision_distillation_loss(logits, ids, p, scope="vocab").item()

    last = logits[0, -1, :]
    logp = torch.log_softmax(last, dim=-1)
    want = -sum(pi * logp[i].item() for pi, i in zip(p, ids))
    assert got == pytest.approx(want, rel=1e-5)


def test_unnormalised_targets_are_normalised():
    logits = torch.randn(1, 3, 8)
    ids = [1, 4]
    for scope in ("letters", "vocab"):
        a = decision_distillation_loss(logits, ids, [3.0, 1.0], scope=scope).item()
        b = decision_distillation_loss(logits, ids, [0.75, 0.25], scope=scope).item()
        assert a == pytest.approx(b, rel=1e-6)


def test_letters_scope_floor_is_target_entropy_regardless_of_offmenu():
    """digesta's correctness criterion for fix (1): with the letter logits in the
    target ratio, the letters-scope loss equals H(target) no matter the off-menu
    logits -- the clean 'aligned' signal."""
    vocab = 6
    ids = [0, 1]
    p = [0.7, 0.3]
    entropy = -sum(pi * math.log(pi) for pi in p)
    for off in (-5.0, 0.0, 5.0, 50.0):
        logits = _logits_with(ids, [math.log(0.7), math.log(0.3)], vocab, off=off)
        got = decision_distillation_loss(logits, ids, p).item()
        assert got == pytest.approx(entropy, abs=1e-5)


def test_vocab_scope_sits_above_entropy_and_grows_with_offmenu():
    """The full-vocab scope cannot reach H(target) while any off-menu mass
    remains, and rises as that mass grows -- the offset digesta described."""
    vocab = 6
    ids = [0, 1]
    p = [0.7, 0.3]
    entropy = -sum(pi * math.log(pi) for pi in p)
    prev = None
    for off in (-10.0, -2.0, 0.0, 1.0):
        logits = _logits_with(ids, [math.log(0.7), math.log(0.3)], vocab, off=off)
        got = decision_distillation_loss(logits, ids, p, scope="vocab").item()
        assert got > entropy
        if prev is not None:
            assert got > prev          # more off-menu mass -> strictly higher
        prev = got


def test_the_two_scopes_agree_when_offmenu_mass_vanishes():
    """At the optimum the two formulations coincide; they differ only off it."""
    vocab = 6
    ids = [0, 1]
    p = [0.7, 0.3]
    big = 40.0
    logits = _logits_with(ids, [math.log(0.7) + big, math.log(0.3) + big],
                          vocab, off=-big)
    a = decision_distillation_loss(logits, ids, p, scope="letters").item()
    b = decision_distillation_loss(logits, ids, p, scope="vocab").item()
    assert a == pytest.approx(b, abs=1e-3)


def test_only_the_last_position_carries_gradient():
    vocab, seq = 12, 5
    for scope in ("letters", "vocab"):
        logits = torch.randn(1, seq, vocab, requires_grad=True)
        loss = decision_distillation_loss(logits, [3, 8], [0.6, 0.4], scope=scope)
        loss.backward()
        grad = logits.grad[0]
        assert torch.count_nonzero(grad[:-1]) == 0     # earlier positions: zero
        assert torch.count_nonzero(grad[-1]) > 0       # last position: non-zero


def test_raising_a_letters_logit_lowers_the_loss():
    """The gradient points the right way under both scopes: more mass on the
    dominant target letter helps."""
    vocab = 8
    ids = [2, 5]
    p = [0.9, 0.1]
    for scope in ("letters", "vocab"):
        base = torch.zeros(1, 1, vocab)
        loss0 = decision_distillation_loss(base, ids, p, scope=scope).item()
        up = base.clone()
        up[0, -1, 2] += 2.0                      # push the dominant letter up
        loss1 = decision_distillation_loss(up, ids, p, scope=scope).item()
        assert loss1 < loss0
