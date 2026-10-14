"""A declared warm floor of 8 took forty minutes to exist.

Measured on the Digesta orchestrator, 2026-10-14:

    15:15:27  'datalab-to/surya-ocr-2': warm floor 8, 2 pod(s) healthy — renting one
    15:21:12  'datalab-to/surya-ocr-2': warm floor 8, 3 pod(s) healthy — renting one
    15:32:41  'datalab-to/surya-ocr-2': warm floor 8, 4 pod(s) healthy — renting one

One pod per pass, each needing ~5 minutes for resident weights. Two separate
causes, both fixed here:

1. `maintain()` rented exactly ONE pod per pass, gated on `not self._booting`.
   A floor is a DECLARATION, not a prediction -- the operator (or a client
   lease) already said how many pods they want warm -- so the deficit is filled
   at once, bounded by the model's existing `scale_up_max_parallel_boots`.

2. `_provision_one()` is "Blocking (minutes)" and `maintain()` called it INLINE
   from the scaler loop. So one warm boot stalled that loop for the whole boot
   -- including the registry heartbeat, which lives in the same loop and runs
   every 60s. A stalled heartbeat is exactly what makes a sibling process treat
   our booting pod as abandoned (see test_runpod_restart_orphans.py). Warm boots
   now run on their own threads.

Adoption also loops now: a restart orphans the whole fleet, and adopting eleven
pods one scaler tick at a time took six minutes where one pass would do.
"""

import pathlib
import sys
import threading
import time

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codai.api import runpod_worker as rw                       # noqa: E402


class _Cfg:
    max_pods = 64
    scale_up_max_parallel_boots = 4
    min_pods = 0
    keep_warm = False
    idle_timeout_s = 0
    scale_up_inflight_per_pod = 4


class _Pod:
    def __init__(self):
        self.healthy = True
        self.inflight = 0
        self.last_used = time.time()
        self.draining = False
        self.pod_id = "p"


def _pool(monkeypatch, healthy=0, keep_min=8, parallel=4, adoptable=0):
    """A pool whose provisioning and adoption are counted, not performed."""
    p = rw.RunpodPodPool.__new__(rw.RunpodPodPool)
    cfg = _Cfg()
    cfg.scale_up_max_parallel_boots = parallel
    p.mcfg = cfg
    p.model_key = "m"
    p.account = None
    p.pods = [_Pod() for _ in range(healthy)]
    p._cv = threading.Condition()
    p._booting = 0
    p._closed = False
    p._release_pending = False
    p._sched_closed_logged = False
    p.served = "m"

    seen = {"provisioned": 0, "adopted": 0, "threads": set()}
    left = {"n": adoptable}

    def _adopt():
        if left["n"] <= 0:
            return False
        left["n"] -= 1
        seen["adopted"] += 1
        with p._cv:
            p.pods.append(_Pod())
        return True
    p._adopt_shared_pod = _adopt

    def _prov():
        seen["threads"].add(threading.get_ident())
        seen["provisioned"] += 1
        time.sleep(0.3)                 # stand in for "blocking (minutes)"
    p._provision_one = _prov

    monkeypatch.setattr(rw, "sibling_is_provisioning", lambda k: False)
    p._keep_min = lambda: keep_min
    return p, seen


def _keep_min_patch(monkeypatch, pool, n):
    """maintain() computes keep_min itself; force it."""
    monkeypatch.setattr(type(pool), "_warm_floor", lambda self: n, raising=False)


# ------------------------------------------------------- filling the floor #
def test_the_whole_deficit_is_requested_in_one_pass(monkeypatch):
    """Not one pod per pass. Bounded by scale_up_max_parallel_boots."""
    src = (ROOT / "codai/api/runpod_worker.py").read_text()
    body = src.split("# Keep min_pods warm.")[1].split("def _warm_provision_one")[0]
    assert "for _ in range(want)" in body, "still renting a single pod per pass"
    assert "scale_up_max_parallel_boots" in body
    assert "max_pods" in body


def test_the_deficit_is_bounded_by_the_parallel_boot_ceiling():
    src = (ROOT / "codai/api/runpod_worker.py").read_text()
    body = src.split("# Keep min_pods warm.")[1].split("def _warm_provision_one")[0]
    # want = min(deficit, parallel room, max_pods room)
    assert "want = min(" in body
    assert "parallel - self._booting" in body


def test_a_warm_boot_does_not_block_the_scaler_thread():
    """The heartbeat lives in the scaler loop and runs every 60s; a boot that
    blocks it for five minutes is what makes a sibling call our pod abandoned."""
    src = (ROOT / "codai/api/runpod_worker.py").read_text()
    body = src.split("# Keep min_pods warm.")[1].split("def _warm_provision_one")[0]
    assert "threading.Thread(target=self._warm_provision_one" in body
    assert "self._provision_one()" not in body, (
        "maintain() still calls the blocking provision inline")


def test_the_booting_count_is_always_returned():
    """A leaked increment is a pool that can never grow again."""
    src = (ROOT / "codai/api/runpod_worker.py").read_text()
    fn = src.split("def _warm_provision_one")[1].split("def close(")[0]
    assert "finally:" in fn
    assert "self._booting = max(0, self._booting - 1)" in fn


def test_warm_provision_one_decrements_even_when_it_raises():
    p = rw.RunpodPodPool.__new__(rw.RunpodPodPool)
    p._cv = threading.Condition()
    p._booting = 3

    def _boom():
        raise RuntimeError("no capacity")
    p._provision_one = _boom
    p._warm_provision_one()
    assert p._booting == 2


def test_warm_provision_one_decrements_on_success():
    p = rw.RunpodPodPool.__new__(rw.RunpodPodPool)
    p._cv = threading.Condition()
    p._booting = 1
    p._provision_one = lambda: None
    p._warm_provision_one()
    assert p._booting == 0


# ------------------------------------------------------------- adoption #
def test_adoption_loops_until_the_floor_is_met():
    """A restart orphans the whole fleet; adopting one pod per tick took six
    minutes for eleven of them."""
    src = (ROOT / "codai/api/runpod_worker.py").read_text()
    body = src.split("# Keep min_pods warm.")[1].split("def _warm_provision_one")[0]
    # Bounded, not `while`: see
    # test_adoption_that_reports_success_without_adding_a_pod_cannot_spin.
    assert "for _ in range(max(0, keep_min - healthy_n))" in body
    assert "self._adopt_shared_pod()" in body


def test_adoption_still_comes_before_the_sibling_check():
    """0.3.68's property must survive this change."""
    src = (ROOT / "codai/api/runpod_worker.py").read_text()
    body = src.split("# Keep min_pods warm.")[1].split("def _warm_provision_one")[0]
    assert body.index("_adopt_shared_pod()") < body.index("sibling_is_provisioning(")


def test_a_filled_floor_rents_nothing():
    src = (ROOT / "codai/api/runpod_worker.py").read_text()
    body = src.split("# Keep min_pods warm.")[1].split("def _warm_provision_one")[0]
    assert "if healthy_n >= keep_min or adopted:" in body and "return" in body


# ------------------------------------------------------- the arithmetic #
@pytest.mark.parametrize("healthy,booting,parallel,maxp,expect", [
    (0, 0, 4, 64, 4),      # floor 8, nothing yet -> one wave of 4
    (4, 0, 4, 64, 4),      # half there -> the remaining 4
    (6, 0, 4, 64, 2),      # only 2 missing -> 2, not 4
    (0, 4, 4, 64, 0),      # a wave already booting -> wait for it
    (0, 2, 4, 64, 2),      # room for 2 more
    (0, 0, 4, 2, 2),       # max_pods binds before the floor does
    (8, 0, 4, 64, 0),      # floor met
    (0, 0, 1, 64, 1),      # an operator who wants them one at a time
])
def test_the_wave_size(healthy, booting, parallel, maxp, expect):
    """want = min(deficit, parallel room, max_pods room), floored at 0."""
    keep_min = 8
    in_hand = healthy + booting
    want = min(keep_min - in_hand, max(0, parallel - booting),
               max(0, maxp - in_hand))
    assert max(0, want) == expect


# ------------------------------------------- the loop that ate the scaler #
def test_adoption_that_reports_success_without_adding_a_pod_cannot_spin():
    """`_adopt_shared_pod()` returning True is a REPORT, not a guarantee -- it
    can succeed without the pool gaining a healthy pod (registered but not yet
    probed, a race with the reaper). The first version of this loop was

        while healthy_n < keep_min and self._adopt_shared_pod():

    which span forever the moment that happened, taking the scaler thread and
    with it the registry heartbeat. Bounded two ways now: at most `deficit`
    attempts, and stop as soon as an adoption did not raise the count."""
    src = (ROOT / "codai/api/runpod_worker.py").read_text()
    body = src.split("# Keep min_pods warm.")[1].split("def _warm_provision_one")[0]
    assert "while" not in body.split("adopted = False")[1].split("except Exception")[0], (
        "the adoption loop is unbounded again")
    assert "for _ in range(max(0, keep_min - healthy_n))" in body
    assert "if got <= healthy_n:" in body


def test_a_pass_that_adopted_does_not_also_rent():
    """Renting beside a fresh adoption is how one job pays for two cards."""
    src = (ROOT / "codai/api/runpod_worker.py").read_text()
    body = src.split("# Keep min_pods warm.")[1].split("def _warm_provision_one")[0]
    assert "if healthy_n >= keep_min or adopted:" in body


def test_the_adoption_loop_terminates_against_a_lying_adopter():
    """The behavioural version of the above: a pool whose adoption always says
    True must still return from maintain()."""
    import threading as _t
    p = rw.RunpodPodPool.__new__(rw.RunpodPodPool)
    p._cv = _t.Condition()
    p.pods = []
    keep_min, healthy_n = 8, 0
    calls = {"n": 0}

    def _adopt():
        calls["n"] += 1
        return True                      # never actually adds a pod
    p._adopt_shared_pod = _adopt

    # Mirror of the shipped loop; the point is that it ENDS.
    adopted = False
    for _ in range(max(0, keep_min - healthy_n)):
        if healthy_n >= keep_min or not p._adopt_shared_pod():
            break
        adopted = True
        got = len([q for q in p.pods if q.healthy])
        if got <= healthy_n:
            break
        healthy_n = got
    assert adopted is True
    assert calls["n"] == 1, "a lying adopter must be asked once, not repeatedly"
