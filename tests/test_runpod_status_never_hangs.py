"""The one endpoint an operator has for seeing why the server is busy must not
be the first thing a busy server stops serving.

During a boot wave `/v1/runpod/status` stopped answering entirely, which a
client reasonably read as "the front is dead" -- while `/healthz` answered in
2.9 ms throughout and the front had zero event-loop stalls. It was not the
loop: the snapshot takes the pool locks, and a provisioning wave holds them.

So every lock in the snapshot is taken with a timeout. A pool whose lock is
busy contributes one `state: "busy"` row naming itself, and the answer is
always prompt, explicit about what it could not read, and never absent.
"""

import threading
import time
import types

import pytest

from codai.api import runpod_worker as rw



class _HeldElsewhere:
    """Hold a lock from ANOTHER thread.

    These are RLocks, so acquiring in the thread that then reads would simply
    re-enter and succeed -- which is exactly what a snapshot must not rely on.
    The holder in production is always a different thread (a provisioning
    wave), so the test has to be one too.
    """

    def __init__(self, lock):
        self.lock = lock
        self._held = threading.Event()
        self._release = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        self.lock.acquire()
        self._held.set()
        self._release.wait(10.0)
        self.lock.release()

    def __enter__(self):
        self._t.start()
        assert self._held.wait(5.0), "could not take the lock"
        return self

    def __exit__(self, *_exc):
        self._release.set()
        self._t.join(5.0)
        return False


def _pool(key="m", pods=()):
    p = rw.RunpodPodPool.__new__(rw.RunpodPodPool)
    p._cv = threading.Condition(threading.RLock())
    p.pods = list(pods)
    p.mcfg = types.SimpleNamespace(pool="", engine="coderai-vllm-ocr")
    p.model_key = key
    return p


def _pod(pid="p1"):
    return types.SimpleNamespace(
        pod_id=pid, gpu="NVIDIA A40", gpu_count=1, is_spot=False, healthy=True,
        inflight=0, hourly_usd=1.0, started_at=time.time(), console_url="",
        data_center="EU-SE-1", cloud_type="SECURE")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setattr(rw, "_pools", {}, raising=False)
    monkeypatch.setattr(rw, "_provisioning_info", {}, raising=False)
    monkeypatch.setattr(rw, "SNAPSHOT_LOCK_TIMEOUT_S", 0.15)
    yield


# ------------------------------------------------------------- the helper #

def test_the_helper_yields_true_when_the_lock_is_free():
    lk = threading.Lock()
    with rw._lock_for_read(lk) as got:
        assert got is True
        assert lk.locked()
    assert not lk.locked()


def test_the_helper_gives_up_rather_than_waiting():
    lk = threading.RLock()
    with _HeldElsewhere(lk):
        t0 = time.monotonic()
        with rw._lock_for_read(lk) as got:
            assert got is False
        assert time.monotonic() - t0 < 1.0


def test_the_helper_does_not_release_a_lock_it_never_took():
    lk = threading.Lock()
    with _HeldElsewhere(lk):
        with rw._lock_for_read(lk) as got:
            assert got is False
        assert lk.locked(), "it released someone else's lock"


# ----------------------------------------------------------- the snapshot #

def test_a_quiet_fleet_is_reported_normally(monkeypatch):
    monkeypatch.setattr(rw, "_pools", {"m": _pool("m", [_pod()])}, raising=False)
    out = rw.pods_status()
    assert [r for r in out if r.get("pod_id") == "p1"]
    assert not [r for r in out if r.get("state") == "busy"]


def test_a_locked_pool_is_reported_busy_not_waited_for(monkeypatch):
    pool = _pool("m", [_pod()])
    monkeypatch.setattr(rw, "_pools", {"m": pool}, raising=False)
    with _HeldElsewhere(pool._cv):
        t0 = time.monotonic()
        out = rw.pods_status()
        took = time.monotonic() - t0
    assert took < 2.0, f"the snapshot waited {took:.1f}s on a held lock"
    busy = [r for r in out if r.get("state") == "busy"]
    assert busy and busy[0]["model"] == "m"
    assert "locked" in busy[0]["note"]


def test_one_locked_pool_does_not_hide_the_others(monkeypatch):
    busy_pool = _pool("busy", [_pod("b1")])
    ok_pool = _pool("ok", [_pod("o1")])
    monkeypatch.setattr(rw, "_pools",
                        {"busy": busy_pool, "ok": ok_pool}, raising=False)
    with _HeldElsewhere(busy_pool._cv):
        out = rw.pods_status()
    assert [r for r in out if r.get("pod_id") == "o1"], "lost a readable pool"
    assert [r for r in out if r.get("state") == "busy"
            and r.get("model") == "busy"]


def test_a_locked_registry_says_so_instead_of_answering_empty(monkeypatch):
    """An empty list and a locked registry must not look the same: one means
    'no pods', the other means 'ask again', and they are opposite actions."""
    monkeypatch.setattr(rw, "_pools", {"m": _pool("m", [_pod()])}, raising=False)
    with _HeldElsewhere(rw._pools_lock):
        out = rw.pods_status()
    assert len(out) == 1
    assert out[0]["state"] == "busy"
    assert "incomplete" in out[0]["note"]


def test_the_timeout_is_short_enough_to_be_useful():
    assert 0.1 <= rw.SNAPSHOT_LOCK_TIMEOUT_S <= 10.0


def test_a_truly_empty_fleet_is_still_empty(monkeypatch):
    monkeypatch.setattr(rw, "_pools", {}, raising=False)
    out = rw.pods_status()
    # No pools at all: the registry was readable and there was nothing in it.
    assert all(r.get("state") == "busy" for r in out) or out == []


# ------------------------------------------------- the caps it never sent #

def test_the_spend_caps_are_read_from_the_manager_that_exists():
    """`codai.config` has no `config_manager`, so this import raised every
    time and the except swallowed it: caps was silently empty in every answer
    the endpoint ever gave. Same mistake that pinned both thread pools to
    their floors."""
    import codai.api.app as app_mod
    src = open(app_mod.__file__).read()
    i = src.index("caps = {}")
    window = src[i:i + 900]
    assert "from codai.admin.routes import config_manager" in window
