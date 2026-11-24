"""The LR schedule spans the whole run, not each distributed block.

rizzo's recipe is a linear warmup over ~3% of the steps, then cosine decay to
zero. In coderai's distributed trainer the run is cut into blocks trained a
round at a time, so the schedule must be built over the TOTAL optimizer steps
across every round, positioned at this block's offset -- otherwise the cosine
resets to the peak at the start of each block (a sawtooth that never really
decays), the one implementation trap digesta flagged. ``build_lr_schedule`` is
the single place that decides this, so it is what we pin. No GPU: a dummy
one-parameter optimizer is enough to read the lr curve the scheduler drives.
"""
import math

import torch

from codai.api import text_lora_trainer as tlt


def _curve(settings, *, examples, accum, resumed_step=0, steps=None):
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))],
                          lr=settings.learning_rate)
    sched = tlt.build_lr_schedule(opt, settings, examples=examples, accum=accum,
                                  resumed_step=resumed_step)
    if sched is None:
        return None
    n = steps if steps is not None else math.ceil(examples / accum)
    lrs = []
    for _ in range(n):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    return lrs


def test_constant_no_warmup_is_a_flat_lr():
    s = tlt.QLoraSettings(learning_rate=5e-5, lr_scheduler_type="constant",
                          warmup_ratio=0.0)
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=5e-5)
    # None means the optimizer keeps its flat lr -- the historical behaviour.
    assert tlt.build_lr_schedule(opt, s, examples=1000, accum=16) is None


def test_cosine_warms_up_then_decays_to_zero_over_the_whole_run():
    # digesta's config: 22,128 examples / 16 = 1383 steps, 3% warmup ~= 41.
    s = tlt.QLoraSettings(learning_rate=5e-5, lr_scheduler_type="cosine",
                          warmup_ratio=0.03)
    lrs = _curve(s, examples=22128, accum=16)
    assert len(lrs) == 1383                                # ceil(22128/16)
    peak_at = max(range(len(lrs)), key=lambda i: lrs[i])
    assert 38 <= peak_at <= 44                             # ~41 warmup steps
    assert abs(lrs[peak_at] - 5e-5) < 2e-6                 # reaches the peak lr
    assert lrs[0] < 1e-6                                   # warms UP from ~0
    after = lrs[peak_at:]
    assert all(after[i] + 1e-12 >= after[i + 1]            # monotone decay
               for i in range(len(after) - 1))
    assert lrs[-1] < 2e-6                                  # ...to ~0 at the end


def test_distributed_offset_continues_the_curve_instead_of_a_sawtooth():
    # The reference: the true global curve over the whole 1383-step run.
    whole = tlt.QLoraSettings(learning_rate=5e-5, lr_scheduler_type="cosine",
                              warmup_ratio=0.03)
    ref = _curve(whole, examples=22128, accum=16)
    # A later block of the SAME run: horizon is the whole run (1383), this block
    # starts at global step 700. Its first lr must be the mid-decay value at
    # 700, NOT reset to the peak -- that reset is the sawtooth bug.
    block = tlt.QLoraSettings(learning_rate=5e-5, lr_scheduler_type="cosine",
                              warmup_ratio=0.03, lr_total_steps=1383,
                              lr_step_offset=700)
    lrs = _curve(block, examples=5000, accum=16, steps=10)
    assert abs(lrs[0] - ref[700]) < 1e-9                   # continues the curve
    assert lrs[0] < ref[41]                                # far below the peak


def test_weight_decay_default_preserved_and_overridable():
    assert tlt.QLoraSettings().weight_decay == 0.01        # torch's / historical
    assert tlt.QLoraSettings(weight_decay=0.0).weight_decay == 0.0
