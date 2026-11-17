"""Orphans were systematic: every restart and every failure left pods rented.

Two independent causes, both structural rather than accidental.

1. A JOB FLEET's records were never swept at startup. `orphans()` could always
   find them -- its own docstring calls itself "the answer to the orchestrator
   restarted mid-training" -- and nothing called it, so the pods survived until
   the GENERIC reaper noticed 1200 seconds later. About $0.20 a pod, every
   restart, with fleets of four to eight and many restarts in a day.

2. Adoption reset the IDLE CLOCK. Both adoption paths set `last_used` to now,
   and adoption runs on every restart, so an idle pod got a fresh
   idle_timeout_s lease each time and never retired. Measured on this install:
   six idle embedding pods, one 611 minutes old against a 900s timeout,
   $2.24/hr between them.
"""

import inspect

import pytest

from codai.api import runpod_worker as rw
from codai.api import text_lora_fleet as flt


# ------------------------------------------- the fleet sweep, at startup ----

def test_the_sweep_runs_at_startup_before_the_reaper():
    src = inspect.getsource(rw._scaler_loop)
    assert "sweep_orphans()" in src
    assert src.index("sweep_orphans()") < src.index("reap_orphans()")


def test_the_sweep_needs_no_grace_and_says_why():
    doc = flt.sweep_orphans.__doc__ or ""
    assert "No grace is needed" in doc
    assert "definitively gone" in doc


def test_a_failed_terminate_keeps_the_record(tmp_path, monkeypatch):
    """Dropping it is how a pod becomes invisible AND billing.

    Was a source match on `sweep_orphans`, which broke the moment the
    terminate body moved into the helper the periodic sweep shares -- while
    the behaviour was unchanged. Driven instead.
    """
    from codai.api import loras
    d = tmp_path / "loras"
    d.mkdir()
    monkeypatch.setattr(loras, "_loras_dir", lambda: str(d))
    flt._write_state("job-x", [
        {"pod_id": "ok", "shard": 0, "hourly_usd": 0.59},
        {"pod_id": "stubborn", "shard": 1, "hourly_usd": 0.59},
    ])

    class _C:
        def __init__(self):
            self.killed = []

        def terminate_pod(self, pid):
            if pid == "stubborn":
                raise RuntimeError("RunPod said no")
            self.killed.append(pid)

    client = _C()
    out = flt.sweep_orphans(client=client)

    assert client.killed == ["ok"]
    assert out["failed"] == 1
    assert [r["pod_id"] for r in flt._read_state("job-x")] == ["stubborn"], \
        "the one that would not die is kept for the next sweep"


def test_the_sweep_terminates_and_forgets(tmp_path, monkeypatch):
    monkeypatch.setenv("CODERAI_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(flt, "_state_dir", lambda: str(tmp_path / "fleet"))
    (tmp_path / "fleet").mkdir()
    flt._write_state("job-dead", [
        {"pod_id": "pod-a", "shard": 0, "hourly_usd": 0.59},
        {"pod_id": "pod-b", "shard": 1, "hourly_usd": 0.59},
    ])

    killed = []

    class Client:
        def terminate_pod(self, pid):
            killed.append(pid)

    got = flt.sweep_orphans(Client())
    assert sorted(killed) == ["pod-a", "pod-b"]
    assert got["pods"] == 2 and got["jobs"] == 1
    assert flt.orphans() == [], "the record survived its pods"


def test_a_pod_that_will_not_die_is_kept_for_the_next_sweep(tmp_path, monkeypatch):
    monkeypatch.setattr(flt, "_state_dir", lambda: str(tmp_path / "fleet"))
    (tmp_path / "fleet").mkdir()
    flt._write_state("job-stuck", [{"pod_id": "pod-x", "shard": 0}])

    class Stubborn:
        def terminate_pod(self, pid):
            raise RuntimeError("RunPod says no")

    got = flt.sweep_orphans(Stubborn())
    assert got["failed"] == 1 and got["pods"] == 0
    assert [o["pod_id"] for o in flt.orphans()] == ["pod-x"]


def test_a_placeholder_is_not_terminated(tmp_path, monkeypatch):
    """`pending-*` is written BEFORE the rental, so it names no pod."""
    monkeypatch.setattr(flt, "_state_dir", lambda: str(tmp_path / "fleet"))
    (tmp_path / "fleet").mkdir()
    flt._write_state("job-p", [{"pod_id": "pending-job-p-0", "shard": 0}])
    killed = []

    class Client:
        def terminate_pod(self, pid):
            killed.append(pid)

    flt.sweep_orphans(Client())
    assert killed == []


def test_no_runpod_account_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(flt, "_state_dir", lambda: str(tmp_path / "fleet"))
    (tmp_path / "fleet").mkdir()
    flt._write_state("job-q", [{"pod_id": "pod-z", "shard": 0}])
    import codai.models.manager as mgr
    monkeypatch.setattr(mgr, "get_active_runpod_config", lambda: None)
    assert flt.sweep_orphans()["pods"] == 0


# --------------------------------------------- the idle clock survives ------

def test_adoption_keeps_the_recorded_idle_clock():
    src = inspect.getsource(rw)
    assert 'last_used=float(info.get("used") or 0.0) or now' in src
    assert "last_used=_adopted_last_used(shared_id)" in src


def test_the_registry_is_the_source_of_the_clock():
    src = inspect.getsource(rw._adopted_last_used)
    assert '.get("used")' in src


def test_an_unknown_clock_falls_back_to_now_not_zero():
    """Zero means "never used" to the engineless sweep, which would retire a
    freshly adopted pod for having no engine yet."""
    src = inspect.getsource(rw._adopted_last_used)
    assert "return time.time()" in src
    assert "never used" in src


def test_a_freshly_rented_pod_still_gets_a_full_idle_window():
    """Only ADOPTION was wrong. A pod just rented has not been used, but
    giving it `now` is right: renting and then shedding immediately is worse
    than one idle window."""
    src = inspect.getsource(rw)
    assert 'is_spot=sel["is_spot"], healthy=True, last_used=time.time()' in src
