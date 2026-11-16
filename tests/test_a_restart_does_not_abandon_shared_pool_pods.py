"""A restart left shared-pool pods owned by a process that no longer exists.

`reclaim_registered_pods` took this deployment's pods back at startup, but
skipped any whose pool key is shared (`pool:…`), on the reasoning that such a
pool's shape cannot be rebuilt from the registry and the on-demand path would
handle it. The on-demand path cannot: `find_shared_pod` skips our own registry
entries, because a pool handles its own. A pod owned by a DEAD process is
therefore invisible to both -- neither claims it, and it bills until the
reaper's grace expires.

Measured on this install: three pods across two restarts, all
`pool:digesta-vision`, ~$2.25/hr of a $8.28/hr fleet -- 27% of the bill doing
nothing. Verified by hand before fixing it: offering two such pods for
adoption got both claimed by their pool within minutes, while a third left
alone was still unmanaged two hours later.

The shape problem is untouched -- this does not build the pool. It only clears
the dead owner, which is the single condition adoption requires.
"""

import inspect

import pytest

from codai.api import runpod_worker as rw


OTHER = {"pid": 999999, "proc": "dead-process", "at": 0.0}


@pytest.fixture
def registry(monkeypatch):
    """A registry whose updates are applied in memory, as the file would."""
    state = {}

    def _update(fn):
        fn(state)
        return dict(state)

    monkeypatch.setattr(rw, "_registry_update", _update)
    monkeypatch.setattr(rw, "forget_account_listing", lambda: None)
    return state


def _pod(pool, **over):
    d = dict(OTHER); d["pool"] = pool; d.update(over); return d


def test_a_dead_owner_is_cleared_so_a_pool_may_adopt(registry):
    data = {"p1": _pod("pool:digesta-vision"),
            "p2": _pod("pool:digesta-vision")}
    registry.update({k: dict(v) for k, v in data.items()})
    assert rw._offer_shared_for_adoption("pool:digesta-vision", data) == 2
    for k in ("p1", "p2"):
        assert registry[k]["pid"] == 0
        assert registry[k]["proc"] == "offered-for-adoption"


def test_only_that_pools_pods_are_offered(registry):
    data = {"p1": _pod("pool:digesta-vision"), "p2": _pod("pool:other")}
    registry.update({k: dict(v) for k, v in data.items()})
    assert rw._offer_shared_for_adoption("pool:digesta-vision", data) == 1
    assert registry["p2"]["pid"] == 999999, "another pool's pod was disturbed"


def test_this_processs_own_pods_are_left_alone(registry, monkeypatch):
    """Disowning a live pod would invite a sibling's reaper to take a card
    this process is actively using -- the documented mutual-kill loop."""
    monkeypatch.setattr(rw, "_is_this_process", lambda info: info.get("mine"))
    data = {"mine": _pod("pool:v", mine=True), "theirs": _pod("pool:v")}
    registry.update({k: dict(v) for k, v in data.items()})
    assert rw._offer_shared_for_adoption("pool:v", data) == 1
    assert registry["mine"]["pid"] == 999999


def test_nothing_to_offer_is_not_an_error(registry):
    assert rw._offer_shared_for_adoption("pool:empty", {}) == 0


def test_a_booting_pod_is_offered_too(registry):
    """No health probe on purpose: a restart during a 6-14 minute image pull
    must not lose the pod whose boot has already been paid for."""
    src = inspect.getsource(rw._offer_shared_for_adoption)
    assert "NOT a health probe" in src
    data = {"booting": _pod("pool:v", healthy=False)}
    registry.update({k: dict(v) for k, v in data.items()})
    assert rw._offer_shared_for_adoption("pool:v", data) == 1


def test_the_entry_survives_so_the_reaper_shield_still_applies(registry):
    """Deleting the entry would drop the reaper's "known" shield immediately
    and kill a pod a pool was about to adopt."""
    data = {"p1": _pod("pool:v")}
    registry.update({k: dict(v) for k, v in data.items()})
    rw._offer_shared_for_adoption("pool:v", data)
    assert "p1" in registry and registry["p1"]["pool"] == "pool:v"


def test_a_registry_write_failure_is_reported_not_swallowed(registry, monkeypatch, capsys):
    def _boom(fn):
        raise OSError("read-only file system")

    monkeypatch.setattr(rw, "_registry_update", _boom)
    assert rw._offer_shared_for_adoption("pool:v", {"p1": _pod("pool:v")}) == 0
    out = capsys.readouterr().out
    assert "could not offer" in out and "reaper will collect" in out


def test_startup_calls_it_instead_of_skipping(registry):
    """The whole defect: the branch printed "leaving shared pool … to the
    on-demand path" and returned, handing the pod to a path that cannot see
    it."""
    src = inspect.getsource(rw.reclaim_registered_pods)
    assert "_offer_shared_for_adoption(key, data)" in src
    assert "leaving shared pool" not in src


def test_the_log_line_says_what_it_prevents(registry):
    src = inspect.getsource(rw.reclaim_registered_pods)
    assert "OFFERED FOR ADOPTION" in src
    assert "instead of renting beside" in src
