"""One controller owns the RunPod fleet; N OCR worker processes forward pages.

Why: every page went through two Python processes, each limited to one core.
Measured 2026-10-19 on a 35-pod fleet: front 99% CPU, engine 101%, 14 cores
idle, pod GPUs at 0-7% KV cache, ~16 pages/s whatever the fleet size -- 128
pods need ~85.

What these tests pin, in order of what getting it wrong would cost:

  1. a WORKER can never rent, adopt, reap, scale or terminate. N processes
     each able to destroy the shared fleet is the failure to rule out first.
  2. every pod belongs to exactly one worker, and never moves while pages are
     on it, so its in-flight count lives in one place.
  3. the CONTROLLER sees the workers' pages. Its own in-flight is ~0 now, and
     without the reports a pod full of work reads idle and is reaped under it.
"""
import json
import os
import threading
import time

import pytest

from codai.api import runpod_fleet_share as share
from codai.api import runpod_worker as rw


@pytest.fixture
def registry(tmp_path, monkeypatch):
    path = tmp_path / "runpod-pods.json"
    monkeypatch.setattr(rw, "_pod_registry_path", lambda: str(path))
    return path


def _write_registry(path, entries):
    path.write_text(json.dumps(entries))


def _pool(model="m", max_pods=128):
    p = rw.RunpodPodPool.__new__(rw.RunpodPodPool)
    p._cv = threading.Condition()
    p.model_key = model
    p._closed = False
    p._booting = 0
    p._waiting = 0
    p._surplus_since = 0.0
    p._denied = []
    p._affinity = {}
    p.pods = []
    p.api_key = ""
    cfg = rw.RunpodModelConfig()
    cfg.max_pods = max_pods
    p.mcfg = cfg
    p._bump_cards = lambda: None
    p._say_no_pod = lambda *a, **k: None
    return p


def _pod(pid, inflight=0):
    return rw.PodHandle(pod_id=pid, url="https://10.0.0.1:1/v1", hourly_usd=0.59,
                        started_at=time.time(), gpu="A40", inflight=inflight)


# --- 0. who is a worker --------------------------------------------------

@pytest.mark.parametrize("raw,want", [
    ("0/8", (0, 8)), ("7/8", (7, 8)), ("0/1", (0, 1)),
    ("8/8", None), ("-1/8", None), ("x/8", None), ("3", None), ("", None),
    ("0/0", None)])
def test_worker_slot_parsing_is_strict(monkeypatch, raw, want):
    """Anything malformed is NOT a worker -- but that means the process may
    rent, so the launcher is the only thing that sets it and the parse refuses
    anything it does not fully understand."""
    monkeypatch.setenv("CODERAI_OCR_WORKER", raw)
    assert share.worker_slot() == want


def test_no_env_means_controller(monkeypatch):
    monkeypatch.delenv("CODERAI_OCR_WORKER", raising=False)
    assert share.is_follower() is False


# --- 1. a worker can never act on the fleet -------------------------------

@pytest.fixture
def worker(monkeypatch):
    monkeypatch.setenv("CODERAI_OCR_WORKER", "0/2")


def test_a_worker_never_grows(worker):
    p = _pool()
    busy = [_pod("a", inflight=999)]
    p.pods = busy
    assert p._grow_reason(busy) == ""


def test_a_worker_never_provisions(worker):
    from codai.api.runpod_client import RunpodError
    with pytest.raises(RunpodError):
        _pool()._provision_one()


def test_a_worker_never_adopts(worker):
    assert _pool()._adopt_shared_pod() is False


def test_a_worker_never_reaps(worker):
    assert rw.reap_orphans() == 0


def test_a_worker_never_terminates(worker, monkeypatch):
    called = []
    import codai.api.runpod_client as rc
    monkeypatch.setattr(rc.RunpodClient, "terminate_pod",
                        lambda self, pid: called.append(pid))
    p = _pool()
    pod = _pod("a")
    p.pods = [pod]
    p._terminate(pod, "anything")
    assert called == []
    assert pod not in p.pods, "dropped from its own view only"


def test_a_worker_exiting_leaves_the_fleet_alone(worker, monkeypatch):
    """close() runs at process exit and used to terminate every pod."""
    called = []
    monkeypatch.setattr(rw.RunpodPodPool, "_terminate",
                        lambda self, pod, why: called.append(pod.pod_id))
    p = _pool()
    p.pods = [_pod("a"), _pod("b")]
    p.close()
    assert called == []


def test_a_worker_does_not_terminate_after_failed_probes(worker, monkeypatch):
    """It reports the pod suspect; the controller re-probes and decides."""
    p = _pool()
    pod = _pod("a", inflight=3)
    p.pods = [pod]
    p._pod_reachable = lambda _p: False
    pod.quarantined_until = time.time() - 1
    terminated = []
    monkeypatch.setattr(rw.RunpodPodPool, "_terminate",
                        lambda self, pod, why: terminated.append(pod.pod_id))
    for _ in range(rw.QUARANTINE_STRIKES + 2):
        pod.quarantined_until = time.time() - 1
        p._recheck_quarantined(time.time())
    assert terminated == []
    assert pod.quarantined_until > 0, "still set aside, so still reported suspect"


# --- 2. every pod in exactly one slice, and it never moves ----------------

def test_every_pod_belongs_to_exactly_one_worker():
    pods = [f"pod{i:04d}" for i in range(500)]
    for n in (1, 2, 3, 8, 16):
        for pid in pods:
            owners = [i for i in range(n) if share.owns(pid, (i, n))]
            assert len(owners) == 1, (pid, n, owners)


def test_the_hash_alone_would_be_badly_uneven():
    """Guards the design choice: a plain hash split 128 pods 8-29 over 8
    workers, which would saturate the busiest exactly like one process did."""
    pods = [f"p{i}" for i in range(128)]
    sizes = [sum(share.owns(p, (i, 8)) for p in pods) for i in range(8)]
    assert max(sizes) - min(sizes) > 6


def test_assignment_is_balanced(registry):
    got = share.assign("m", [f"p{i}" for i in range(128)], 8)
    sizes = [list(got.values()).count(i) for i in range(8)]
    assert sizes == [16] * 8


def test_assignment_is_stable(registry):
    a = share.assign("m", [f"p{i}" for i in range(20)], 4)
    b = share.assign("m", [f"p{i}" for i in range(20)] + ["new"], 4)
    assert all(b[k] == v for k, v in a.items()), "no existing pod moved"


def test_a_new_pod_goes_to_the_emptiest_worker(registry):
    share.assign("m", ["a", "b", "c"], 2)                # 2 + 1
    got = share.assign("m", ["a", "b", "c", "d"], 2)
    assert list(got.values()).count(0) == list(got.values()).count(1) == 2


def test_a_departed_pod_frees_its_slot(registry):
    share.assign("m", ["a", "b", "c", "d"], 2)
    got = share.assign("m", ["a", "b", "c"], 2)
    assert "d" not in got


def test_two_workers_agree_on_the_assignment(registry):
    ids = [f"p{i}" for i in range(30)]
    assert share.assign("m", ids, 3) == share.assign("m", list(reversed(ids)), 3)


def test_lowering_n_reassigns_only_what_no_longer_exists(registry):
    share.assign("m", [f"p{i}" for i in range(8)], 4)
    got = share.assign("m", [f"p{i}" for i in range(8)], 2)
    assert all(0 <= w < 2 for w in got.values())


def test_a_pod_never_moves_when_others_come_and_go():
    """A list assignment would move pods on every change -- and a pod that
    moves while pages are on it gets a second worker's 64 on top."""
    before = {p: [i for i in range(8) if share.owns(p, (i, 8))][0]
              for p in (f"x{i}" for i in range(50))}
    after = {p: [i for i in range(8) if share.owns(p, (i, 8))][0]
             for p in before}
    assert before == after


def test_a_worker_follows_only_its_slice(worker, registry):
    entries = {f"p{i}": {"pool": "m", "url": f"https://10.0.0.{i}:1/v1",
                         "key": "tok", "created": 1.0} for i in range(40)}
    entries["other"] = {"pool": "different-model", "url": "https://x/v1"}
    entries["booting"] = {"pool": "m", "url": "", "key": "tok"}
    _write_registry(registry, entries)
    p = _pool()
    p._follow_registry(force=True)
    got = {x.pod_id for x in p.pods}
    mapping = share.assign("m", [k for k in entries if k.startswith("p")], 2)
    want = {k for k, w in mapping.items() if w == 0}
    assert got == want and len(got) == 20
    assert "other" not in got and "booting" not in got
    # The key travels WITH EACH POD now (v0.4.59): a majority vote over the
    # registry's keys gave every pod the commonest one, so a pod carrying a
    # different token 401'd every page sent to it.
    assert {p.api_key for p in p.pods} == {"tok"}
    assert p.pod_key(p.pods[0]) == "tok"


def test_following_keeps_the_handle_that_holds_in_flight_pages(worker, registry):
    pid = "solo"                          # the only pod: assigned to worker 0
    _write_registry(registry, {pid: {"pool": "m", "url": "https://a/v1", "key": "t"}})
    p = _pool()
    p._follow_registry(force=True)
    handle = p.pods[0]
    handle.inflight = 37
    _write_registry(registry, {pid: {"pool": "m", "url": "https://b/v1", "key": "t"}})
    p._follow_registry(force=True)
    assert p.pods[0] is handle
    assert handle.inflight == 37
    assert handle.url == "https://b/v1", "the address is refreshed"


def test_a_pod_gone_from_the_registry_is_dropped(worker, registry):
    pid = "solo"
    _write_registry(registry, {pid: {"pool": "m", "url": "https://a/v1"}})
    p = _pool()
    p._follow_registry(force=True)
    assert p.pods
    _write_registry(registry, {})
    p._follow_registry(force=True)
    assert p.pods == []


def test_a_draining_flag_stops_new_pages(worker, registry):
    pid = "solo"
    _write_registry(registry, {pid: {"pool": "m", "url": "https://a/v1"}})
    share.set_flag(pid, draining=True)
    p = _pool()
    p._follow_registry(force=True)
    assert p.pods[0].draining is True
    assert p._pick(p.pods, "") is None


# --- 3. the controller sees the workers' pages ----------------------------

def test_reports_round_trip(registry):
    share.write_reports("w0:m", {"a": {"inflight": 12, "used": 5.0, "suspect": False}})
    got = share.read_reports()
    assert got["a"]["inflight"] == 12


def test_a_report_for_a_pod_the_worker_dropped_is_withdrawn(registry):
    share.write_reports("w0:m", {"a": {"inflight": 1}, "b": {"inflight": 2}})
    share.write_reports("w0:m", {"a": {"inflight": 1}})
    assert "b" not in share.read_reports()


def test_reports_from_two_workers_do_not_erase_each_other(registry):
    share.write_reports("w0:m", {"a": {"inflight": 1}})
    share.write_reports("w1:m", {"b": {"inflight": 2}})
    got = share.read_reports()
    assert set(got) == {"a", "b"}


def test_stale_reports_are_ignored(registry):
    share.write_reports("w0:m", {"a": {"inflight": 9}})
    assert share.read_reports(max_age=-1) == {}


def test_the_controller_counts_the_workers_pages(registry, monkeypatch):
    monkeypatch.delenv("CODERAI_OCR_WORKER", raising=False)
    share.write_reports("w0:m", {"a": {"inflight": 40, "used": time.time()}})
    p = _pool()
    pod = _pod("a")
    p.pods = [pod]
    p._merge_worker_reports(time.time())
    assert pod.worker_inflight == 40
    assert rw._load(pod) == 40


def test_a_busy_pod_is_not_reaped_as_idle(registry, monkeypatch):
    """The failure this whole mechanism exists to prevent."""
    src = __import__("inspect").getsource(rw.RunpodPodPool.maintain)
    assert "if _load(p) == 0 and idle_to > 0" in src
    assert src.index("_merge_worker_reports") < src.index("if _load(p) == 0 and idle_to")


def test_a_suspect_report_puts_the_pod_up_for_the_controllers_own_probe(registry, monkeypatch):
    monkeypatch.delenv("CODERAI_OCR_WORKER", raising=False)
    share.write_reports("w0:m", {"a": {"inflight": 0, "suspect": True}})
    p = _pool()
    pod = _pod("a")
    p.pods = [pod]
    p._merge_worker_reports(time.time())
    assert pod.healthy is False and pod.quarantined_until > 0


def test_draining_is_published_to_the_workers(registry, monkeypatch):
    monkeypatch.delenv("CODERAI_OCR_WORKER", raising=False)
    share.write_reports("w0:m", {"a": {"inflight": 5}})
    p = _pool()
    pod = _pod("a")
    pod.draining = True
    p.pods = [pod]
    p._merge_worker_reports(time.time())
    assert share.read_flags().get("a", {}).get("draining") is True
    pod.draining = False
    p._merge_worker_reports(time.time())
    assert "a" not in share.read_flags()


def test_controller_decisions_read_the_combined_load():
    import inspect
    for name in ("_pods_needed", "_scale_down_victim", "shed_now", "_scale_down",
                 "_pick"):
        src = inspect.getsource(getattr(rw.RunpodPodPool, name))
        assert "_load(" in src, name
