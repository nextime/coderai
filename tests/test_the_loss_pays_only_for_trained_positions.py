# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""The loss paid for every position and trained a fraction of them.

`masked_cross_entropy` handed the whole `[1, seq, vocab]` logits tensor to
`F.cross_entropy(..., ignore_index=-100)`, after a `.contiguous()` copy of it.
The arithmetic was right -- a mean over the un-ignored positions -- but it was
charged for all of them, and cross_entropy upcasts to fp32 internally and
keeps that for the backward.

On the workload that found this the prompt is masked and only the answer is
trained: ~392 trained positions in a ~5,000-token sequence, over a 151,936
token vocabulary. That is a 1.5 GB copy plus roughly 9 GB of fp32 log-softmax,
saved-for-backward and gradient, to produce a loss over 7% of the positions.
13 of 16 examples raised OutOfMemoryError on a 48 GB card; the training loop
SKIPS an OOM, so the pod answered `done` having trained 3 and the run reported
100%.

These tests pin the two things that matter: the value must not move, and the
work must scale with the TRAINED positions rather than the sequence.
"""

import pytest

torch = pytest.importorskip("torch")

import torch.nn.functional as F                                 # noqa: E402

from codai.api import text_lora_trainer as tlt                  # noqa: E402


def _old_masked_cross_entropy(logits, labels):
    """Verbatim the implementation this replaced, as the reference."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1), ignore_index=-100, reduction="mean")


def _case(seq, vocab, trained, *, seed=0, batch=1):
    """Logits and labels with exactly `trained` un-masked positions."""
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(batch, seq, vocab, generator=g, dtype=torch.float32)
    labels = torch.full((batch, seq), -100, dtype=torch.long)
    # The trained positions sit at the END, as a masked prompt leaves them.
    tgt = torch.randint(0, vocab, (batch, trained), generator=g)
    labels[:, seq - trained:] = tgt
    return logits, labels


# --------------------------------------------------------------------- #
# the value must not move
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("seq,vocab,trained", [
    (64, 128, 8),
    (64, 128, 1),
    (64, 128, 63),
    (257, 512, 40),
    (16, 32, 15),
])
def test_it_agrees_with_the_implementation_it_replaced(seq, vocab, trained):
    logits, labels = _case(seq, vocab, trained)
    got = tlt.masked_cross_entropy(logits, labels)
    want = _old_masked_cross_entropy(logits, labels)
    assert torch.allclose(got, want, rtol=1e-5, atol=1e-6), (got, want)


def test_it_agrees_with_a_scattered_mask():
    """A mask with holes in the middle, not just a masked prefix."""
    logits, labels = _case(128, 256, 128)
    labels[0, ::3] = -100
    got = tlt.masked_cross_entropy(logits, labels)
    want = _old_masked_cross_entropy(logits, labels)
    assert torch.allclose(got, want, rtol=1e-5, atol=1e-6)


def test_it_agrees_across_a_batch():
    """Batch is 1 in this trainer, but the function must not silently depend
    on that -- a wrong answer for batch>1 would be found by nobody."""
    logits, labels = _case(48, 96, 10, batch=4)
    got = tlt.masked_cross_entropy(logits, labels)
    want = _old_masked_cross_entropy(logits, labels)
    assert torch.allclose(got, want, rtol=1e-5, atol=1e-6)


def test_the_mean_is_over_positions_not_the_sequence():
    """Two identical trained positions in sequences of very different length
    give the same loss: that is what `reduction="mean"` over kept positions
    means, and it is what the per-example weight assumes."""
    g = torch.Generator().manual_seed(7)
    vocab = 64
    row = torch.randn(1, vocab, generator=g)
    out = []
    for seq in (8, 400):
        logits = torch.zeros(1, seq, vocab)
        logits[0, seq - 2] = row
        labels = torch.full((1, seq), -100, dtype=torch.long)
        labels[0, seq - 1] = 3
        out.append(tlt.masked_cross_entropy(logits, labels))
    assert torch.allclose(out[0], out[1], rtol=1e-6, atol=1e-7)


# --------------------------------------------------------------------- #
# gradients still reach the logits
# --------------------------------------------------------------------- #

def test_the_gradient_matches_the_old_form():
    """A memory fix that changed the gradient would train a different model."""
    logits, labels = _case(96, 192, 12, seed=3)

    a = logits.clone().requires_grad_(True)
    tlt.masked_cross_entropy(a, labels).backward()
    b = logits.clone().requires_grad_(True)
    _old_masked_cross_entropy(b, labels).backward()

    assert a.grad is not None and b.grad is not None
    assert torch.allclose(a.grad, b.grad, rtol=1e-5, atol=1e-7)


def test_the_untrained_positions_get_no_gradient():
    """They never contributed; a gradient there would be a leak of the mask."""
    logits, labels = _case(64, 128, 5, seed=4)
    x = logits.clone().requires_grad_(True)
    tlt.masked_cross_entropy(x, labels).backward()
    # Position i predicts i+1, so the kept targets at [59:64] are produced by
    # logits at [58:63]. Everything before that contributed nothing.
    assert torch.count_nonzero(x.grad[0, :58]) == 0
    assert torch.count_nonzero(x.grad[0, 58:63]) > 0


def test_an_all_masked_example_keeps_the_graph():
    """A plain zero would detach it and make backward() raise."""
    logits = torch.randn(1, 16, 32, requires_grad=True)
    labels = torch.full((1, 16), -100, dtype=torch.long)
    loss = tlt.masked_cross_entropy(logits, labels)
    assert float(loss.detach()) == 0.0
    loss.backward()                      # must not raise
    assert logits.grad is not None


# --------------------------------------------------------------------- #
# and the point of the change: the cost follows the trained positions
# --------------------------------------------------------------------- #

def test_the_work_scales_with_trained_positions_not_sequence_length():
    """The regression guard. Peak allocation is what OOM-ed; here it is
    measured as the size of the tensor cross_entropy is actually given."""
    seen = {}
    real = F.cross_entropy

    def _spy(inp, tgt, *a, **kw):
        seen["rows"] = int(inp.shape[0])
        return real(inp, tgt, *a, **kw)

    import codai.api.text_lora_trainer as mod
    orig = mod.F if hasattr(mod, "F") else None
    assert orig is None, "the module imports F inside the function"

    logits, labels = _case(2048, 256, 32)
    monkey = pytest.MonkeyPatch()
    monkey.setattr(F, "cross_entropy", _spy)
    try:
        tlt.masked_cross_entropy(logits, labels)
    finally:
        monkey.undo()

    assert seen["rows"] == 32, (
        f"cross_entropy was given {seen['rows']} rows for 32 trained "
        f"positions: the full-width form is what exhausted the card")


def test_no_full_width_tensor_is_made_contiguous():
    """`.contiguous()` on the whole logits tensor was a second full-size
    allocation -- 1.5 GB at this vocabulary, before cross_entropy had been
    given anything. Counted rather than read out of the source, because the
    source test for this matched the comment explaining it."""
    big = []
    real = torch.Tensor.contiguous

    def _spy(self, *a, **kw):
        if self.numel() > 100_000:
            big.append(tuple(self.shape))
        return real(self, *a, **kw)

    logits, labels = _case(2048, 256, 32)       # 2048*256 = 524,288 elements
    monkey = pytest.MonkeyPatch()
    monkey.setattr(torch.Tensor, "contiguous", _spy, raising=True)
    try:
        tlt.masked_cross_entropy(logits, labels)
    finally:
        monkey.undo()

    assert big == [], f"a full-width copy was still made: {big}"
