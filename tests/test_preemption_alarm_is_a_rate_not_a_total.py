"""A cumulative counter must not be shown as a present-tense alarm.

``preemptions`` comes from vLLM, is listed in ``serving_load._CUMULATIVE``, and
is deliberately carried across an engine restart -- so it only ever goes up. The
RunPod page painted any non-zero total red with "past the KV ceiling", present
tense. The effect, observed on a nine-pod chandra fleet on 2026-11-06: every pod
carried a permanent red "16 preempted — past the KV ceiling" while the pods were
in fact healthy -- caches at 50-76%, the on-pod controller reporting "no
preemptions" and still handing out slots. An alarm that cannot turn off is an
alarm an operator learns to ignore, which costs more than showing nothing.

serving_load's own note already said which number is the signal: "A RISING count
is the clearest 'we are past the ceiling' signal there is." These tests hold the
page to that -- the total stays visible as history, and red is reserved for a
count that has actually moved.
"""

import pathlib
import re

import pytest

from codai.api import pod_telemetry

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "codai/admin/templates/runpod.html"


@pytest.fixture(autouse=True)
def _clean():
    pod_telemetry.forget()
    yield
    pod_telemetry.forget()


def _note(pod_id, total):
    d = {"preemptions": total}
    pod_telemetry._note_preemptions(pod_id, d)
    return d


# ------------------------------------------------------- the first sighting #
def test_a_total_inherited_from_before_we_were_watching_is_not_an_alarm():
    """The bug, directly: 16 preemptions already on the clock are history."""
    d = _note("pod-a", 16)
    assert d["preemptions"] == 16          # still reported
    assert d["preemptions_recent"] == 0    # but nothing happened just now
    assert d["preemptions_age_s"] is None  # and we never saw it rise


def test_a_pod_that_never_preempted_reports_a_quiet_zero():
    d = _note("pod-a", 0)
    assert d["preemptions_recent"] == 0
    assert d["preemptions_age_s"] is None


# ------------------------------------------------------------- staying quiet #
def test_an_unchanged_total_stays_quiet_however_large_it_is():
    _note("pod-a", 16)
    for _ in range(5):
        d = _note("pod-a", 16)
        assert d["preemptions_recent"] == 0, d


# -------------------------------------------------------------- the signal #
def test_a_rising_count_is_reported_as_the_rise():
    _note("pod-a", 16)
    d = _note("pod-a", 19)
    assert d["preemptions_recent"] == 3
    assert d["preemptions_age_s"] is not None
    assert d["preemptions_age_s"] < 5


def test_the_age_is_remembered_after_the_rise_stops():
    """So the page can keep the condition visible for a window rather than
    flashing it on the single poll the counter moved."""
    _note("pod-a", 16)
    _note("pod-a", 17)
    d = _note("pod-a", 17)
    assert d["preemptions_recent"] == 0
    assert d["preemptions_age_s"] is not None


# ------------------------------------------------------- a counter restart #
def test_a_counter_that_went_backwards_never_reports_a_negative():
    _note("pod-a", 40)
    d = _note("pod-a", 2)
    assert d["preemptions_recent"] == 0


def test_a_restarted_counter_rebaselines_instead_of_swallowing_the_next_rise():
    """Keeping the old high-water mark would hide every real preemption until
    the fresh counter climbed past it."""
    _note("pod-a", 40)
    _note("pod-a", 2)                      # accumulator lost, counting again
    d = _note("pod-a", 5)
    assert d["preemptions_recent"] == 3, d


# ------------------------------------------------------------- bookkeeping #
def test_each_pod_is_tracked_on_its_own():
    _note("pod-a", 10)
    _note("pod-b", 99)
    assert _note("pod-a", 12)["preemptions_recent"] == 2
    assert _note("pod-b", 99)["preemptions_recent"] == 0


def test_forgetting_a_pod_drops_its_baseline():
    _note("pod-a", 10)
    pod_telemetry.forget("pod-a")
    d = _note("pod-a", 10)
    assert d["preemptions_age_s"] is None


def test_a_pod_reporting_no_counter_at_all_gets_no_fields():
    """A pod that is booting or unreachable reports nothing, and a 0 invented
    for it would read as a measurement."""
    d = {}
    pod_telemetry._note_preemptions("pod-a", d)
    assert "preemptions_recent" not in d
    assert "preemptions_age_s" not in d


def test_the_tracker_does_not_grow_without_bound():
    for i in range(700):
        _note(f"pod-{i}", 1)
    assert len(pod_telemetry._PREEMPT) <= 512


# ----------------------------------------------------------------- the page #
def test_the_page_no_longer_paints_the_bare_total_red():
    html = TEMPLATE.read_text(encoding="utf-8")
    assert "past the KV ceiling" in html, "the badge should still exist"
    # The old form keyed the red div off p.preemptions alone.
    bad = re.search(r"p\.preemptions \?[^\n]*#ef4444", html)
    assert bad is None, f"still alarming on the total: {bad.group(0)!r}"


def test_the_page_decides_red_from_the_rate():
    html = TEMPLATE.read_text(encoding="utf-8")
    assert "preemptions_recent" in html
    assert "preemptions_age_s" in html


def test_the_page_still_shows_a_historical_total():
    """Hiding it would lose the information; the fix is the colour and the
    tense, not the number."""
    html = TEMPLATE.read_text(encoding="utf-8")
    assert "preempted earlier" in html
