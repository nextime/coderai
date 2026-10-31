"""A pool rented a pod, booted it for fourteen minutes, and killed it on arrival.

Measured on Saturday 2026-10-31, against a pool whose warm-pod window is
weekdays only (so the warm floor was 0 and the idle timeout is treated as one
second, by design -- "do not wait out idle_timeout_s before releasing a pod that
only existed to be warm"):

    09:44:53  pod w9u0tgs1reu5dn created for 'pool:digesta-qwen38' — booting
    09:52:06  port open after 433s (image pull + container start)
    09:58:39  weights resident after 392s (warm-up inference)
    09:58:39  pod w9u0tgs1reu5dn ready for 'pool:digesta-qwen38'
    09:58:54  pod w9u0tgs1reu5dn terminated (idle); billed ~$0.002

and the same cycle at 08:54, 09:05, 09:15, 09:34, 09:42: about $0.14 of A40 per
cycle, four to six cycles an hour, for a pod that never answered one request.

Two things made it. The pod was rented by the ANTICIPATORY trigger -- "renting
one more for the next burst, not for work already queued" -- into a pool that
reaps anything idle, so the thing it was rented to do was not a thing it was
allowed to do. And the ledger recorded fourteen minutes of billing as fifteen
seconds, because the handle was stamped after the image pull and the warm-up,
so the one place built to show this waste reported $0.002.
"""

import types

import pytest

from codai.api import runpod_worker as rw


def _pool(min_pods=1, threshold=4, max_pods=64):
    from codai.api.runpod_worker import RunpodPodPool, parse_model_runpod

    class _Acct:
        enabled = True

    return RunpodPodPool("m", _Acct(), parse_model_runpod({
        "min_pods": min_pods,
        "max_pods": max_pods,
        "scale_up_inflight_per_pod": threshold,
    }), "m")


def _pod(inflight=0):
    return types.SimpleNamespace(inflight=inflight, healthy=True,
                                 draining=False, pod_id="p", url="u",
                                 last_used=0.0)


# ── when a fresh pod would be reaped ─────────────────────────────────────────

def test_a_closed_warm_window_is_reported_as_reaping_on_arrival():
    p = _pool(min_pods=1)
    p.effective_min_pods = lambda: 0
    why = p._reaps_on_arrival()
    assert why and "window" in why


def test_a_requested_release_is_too():
    p = _pool(min_pods=1)
    p.effective_min_pods = lambda: 1
    p._release_pending = True
    assert p._reaps_on_arrival()


def test_an_ordinary_pool_is_not():
    p = _pool(min_pods=1)
    p.effective_min_pods = lambda: 1
    p._release_pending = False
    assert p._reaps_on_arrival() == ""


def test_min_pods_zero_is_not_a_closed_window():
    """A pool configured to hold nothing is not a pool whose floor just
    dropped: min_pods 0 with effective 0 holds exactly what it is meant to."""
    p = _pool(min_pods=0)
    p.effective_min_pods = lambda: 0
    p._release_pending = False
    assert p._reaps_on_arrival() == ""


# ── what that does to the two growth triggers ────────────────────────────────

def test_it_does_not_rent_ahead_of_a_burst_it_will_not_keep_the_pod_for():
    p = _pool(min_pods=1, threshold=4)
    p.effective_min_pods = lambda: 0          # window closed
    busy = _pod(inflight=4)
    p.pods = [busy]
    assert p._grow_reason([busy]) == ""


def test_work_that_exists_still_grows_the_fleet():
    """The gate is only on the prediction. A pod that takes queued work is not
    idle, so nothing reaps it -- and refusing to grow here would leave real
    pages waiting."""
    p = _pool(min_pods=1, threshold=4)
    p.effective_min_pods = lambda: 0          # window closed
    busy = _pod(inflight=16)                  # four pods' worth at 4 per pod
    p.pods = [busy]
    why = p._grow_reason([busy])
    assert why and "in flight or waiting" in why


def test_the_anticipatory_trigger_still_fires_inside_the_window():
    p = _pool(min_pods=1, threshold=4)
    p.effective_min_pods = lambda: 1
    p._release_pending = False
    busy = _pod(inflight=4)
    p.pods = [busy]
    why = p._grow_reason([busy])
    assert "next burst" in why


def test_the_refusal_is_logged_once_not_every_pass(capsys):
    p = _pool(min_pods=1, threshold=4)
    p.effective_min_pods = lambda: 0
    busy = _pod(inflight=4)
    p.pods = [busy]
    for _ in range(5):
        p._grow_reason([busy])
    out = capsys.readouterr().out
    assert out.count("not renting ahead of the next burst") == 1, out


def test_the_refusal_says_why(capsys):
    p = _pool(min_pods=1, threshold=4)
    p.effective_min_pods = lambda: 0
    busy = _pod(inflight=4)
    p.pods = [busy]
    p._grow_reason([busy])
    out = capsys.readouterr().out
    assert "warm-pod window is closed" in out
    assert "terminated as soon as it finished booting" in out


# ── the bill that hid it ─────────────────────────────────────────────────────

def test_the_handle_is_stamped_when_the_pod_was_RENTED():
    """RunPod bills from creation, so the ledger has to start there. Stamped
    after the image pull and the warm-up, a fourteen-minute pod was recorded as
    fifteen seconds."""
    src = rw.__file__
    text = open(src, encoding='utf-8').read()
    rent = text.index('_rented_at = time.time()')
    wait = text.index('client.wait_ready(', rent)
    stamp = text.index('started_at=_rented_at', rent)
    assert rent < wait < stamp, (
        "the rent timestamp must be taken before the readiness wait, and used "
        "for the handle after it")


def test_cost_is_measured_from_that_stamp():
    text = open(rw.__file__, encoding='utf-8').read()
    assert text.count('(time.time() - pod.started_at) / 3600.0 * pod.hourly_usd') >= 2
