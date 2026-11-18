"""A test run's `step 0/1` read as a broken real run.

A job that asked for `rounds: 16` and showed `step=0/1` looked like the
request being ignored, and the rounds path was investigated on that basis.
It was not a bug: `test_run` forces one round and truncates the set by
design, and the two jobs that asked for 16 without `test_run` got
`step=0/16`. The record simply did not say which kind of run it was, so the
number had no stated source.

`narrow()` passes rounds straight through -- a request may RAISE it, because
rounds change duration rather than fleet size or price -- so "1" is only ever
explained by test_run or by the block.
"""

import inspect

from codai.api import loras


SRC = inspect.getsource(loras._text_train_blocking)
DIST = inspect.getsource(loras._text_train_distributed)


def test_a_test_run_records_that_it_is_one():
    i = SRC.index("TEST RUN")
    block = SRC[i:i + 900]
    assert "_update_job(job_id, test_run=True, rounds=1" in block


def test_the_test_run_note_says_the_request_was_overridden():
    i = SRC.index("TEST RUN")
    block = SRC[i:i + 900]
    assert "whatever the request asked for" in block


def test_a_real_run_records_its_rounds_too():
    """Otherwise the two cannot be compared and the absence of a flag means
    nothing in particular."""
    assert "_update_job(job_id, test_run=False, rounds=rounds" in DIST


def test_the_real_run_note_explains_what_a_round_buys():
    """The operator's own reason for wanting 16, as the record now states it.

    This asserted `"each one is a "`, which by then matched only the COMMENT
    recording that the note had once overclaimed -- so it passed on a
    re-wrapped comment and said nothing about what a client is told. It now
    asserts the two things the note itself must carry: what one round's
    failure costs, and that a stopped run can be continued, which it could
    not when the overclaim was first corrected.
    """
    assert "1/{rounds} of the work" in DIST
    assert "resume_from=" in DIST


def test_rounds_is_not_clamped_by_narrowing():
    """The policy that was suspected and is not the cause: a request may raise
    rounds. Only test_run forces 1."""
    from codai.api import text_lora_fleet as flt
    cfg = flt.LoraTrainConfig(rounds=1)
    assert flt.narrow(cfg, rounds=16).rounds == 16


def test_narrowing_still_clamps_what_costs_money():
    """Rounds are duration; pods and price are not, and those must still only
    narrow -- or a client raises the operator's ceiling by asking."""
    from codai.api import text_lora_fleet as flt
    cfg = flt.LoraTrainConfig(max_pods=2, max_pod_hourly_usd=0.9)
    out = flt.narrow(cfg, pods=99, max_pods=99, max_pod_hourly_usd=99.0)
    assert out.max_pods == 2
    assert out.max_pod_hourly_usd == 0.9
