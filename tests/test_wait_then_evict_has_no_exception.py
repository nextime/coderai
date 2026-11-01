# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""A busy model is waited for and then evicted. There is no exception.

Measured on zeiss 2026-11-01: eviction freed 12.2 GB of the 18.0 GB needed and
stopped 1.5 GB short, "held by a busy model or another process", and
Z-Image-Turbo loaded with CPU/disk offload. The room it needed was not
unavailable -- it was in use, by a request that was about to finish.

The rule was already the house rule for the active model and the keep-resident
set; ordinary LRU models were the exception, and three deadlines elsewhere were
exceptions in a different dress. A deadline here does not avert a hang: it
converts "this load waits a few seconds" into "this load runs on the CPU for
its whole life", silently.
"""

import inspect
import os

from codai.models import manager as M


_SRC = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "codai/models/manager.py"), encoding="utf-8").read()


def _func(name):
    """The whole method, bounded by the next one -- never by a character count.
    A fixed window stops covering the code it was written for the moment
    anything is added inside it."""
    i = _SRC.index(f"def {name}(")
    return _SRC[i:_SRC.index("\n    def ", i + 10)]


def _code(name):
    """The method WITHOUT its docstring.

    These tests assert that certain sentences no longer appear, and the
    docstrings quote those very sentences to record what changed -- so a check
    against the whole method matches its own documentation and fails for the
    wrong reason.
    """
    body = _func(name)
    first = body.find('"""')
    if first == -1:
        return body
    end = body.find('"""', first + 3)
    return body[:first] + body[end + 3:] if end != -1 else body[:first]


# --------------------------------------------------------------------------- #
#  the wait itself
# --------------------------------------------------------------------------- #

def test_waiting_for_idle_has_no_deadline_by_default():
    sig = inspect.signature(M.MultiModelManager._wait_until_idle)
    assert sig.parameters["timeout"].default is None, \
        "a default deadline is the exception, reintroduced"


def test_the_wait_ends_if_the_model_is_unloaded_under_us():
    """One of the two things that keep an unbounded wait from being a hang."""
    body = _code("_wait_until_idle")
    assert "key not in self.models" in body
    assert "return True" in body


def test_a_long_wait_is_never_silent():
    """The other: a hung generation must read AS a hung generation, not as an
    eviction that stopped responding."""
    body = _code("_wait_until_idle")
    assert "_WAIT_IDLE_SAY_EVERY_S" in body
    assert "still waiting for" in body
    assert isinstance(M._WAIT_IDLE_SAY_EVERY_S, (int, float))
    assert 0 < M._WAIT_IDLE_SAY_EVERY_S <= 60


# --------------------------------------------------------------------------- #
#  no pass declines to evict a busy model
# --------------------------------------------------------------------------- #

def test_a_busy_lru_model_is_waited_for_and_not_passed_over():
    body = _code("_evict_models_for_vram")
    assert "not evicting it" not in body, \
        "this was the first pass abandoning a busy model"
    assert "_busy_candidates" in body
    assert "for key in _busy_candidates:" in body


def test_no_pass_reports_leaving_a_busy_model_loaded():
    """Every one of these strings was an exception to the rule."""
    body = _code("_evict_models_for_vram")
    for phrase in ("leaving it loaded",
                   "still busy after wait",
                   "wait budget spent",
                   "wait budget exhausted"):
        assert phrase not in body, f"still declines to evict: {phrase!r}"


def test_the_busy_sweep_runs_after_the_free_ones():
    """Idle models cost nothing, so they go first: a cheap eviction must never
    queue behind a wait that may turn out to be unnecessary."""
    body = _code("_evict_models_for_vram")
    collect = body.index("_busy_candidates.append(key)")
    sweep = body.index("for key in _busy_candidates:")
    assert collect < sweep


def test_evict_one_waits_rather_than_skipping():
    body = _code("_evict_one")
    assert "Skipping eviction" not in body
    assert "_wait_until_idle(key)" in body


# --------------------------------------------------------------------------- #
#  the two bounded waits that remain are bounded on purpose
# --------------------------------------------------------------------------- #

def test_unload_model_stays_bounded_and_says_why():
    """An operator action with a caller waiting on an answer, where "not
    unloaded" IS an answer. Eviction has no such option."""
    body = _func("unload_model")
    assert "_wait_until_idle(key, timeout=120.0)" in body
    assert "on purpose" in body.lower() or "ON PURPOSE" in body


def test_the_host_ram_evictor_is_left_alone_deliberately():
    """Waiting there risks the kernel's OOM killer taking the whole engine --
    four times in five hours the night that code was written. Different
    resource, different failure mode; changing it is a separate decision."""
    assert "not evicting it (RAM)" in _SRC
