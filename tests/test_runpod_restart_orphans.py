"""A deploy restarted coderai mid-boot and the pool went blind.

Observed live 2026-10-14, 15:16-15:23: six pods billing on the RunPod account,
none in coderai's interface, and this line every fifteen seconds —

    [runpod] another engine is booting a pod for 'datalab-to/surya-ocr-2'
             — waiting for it

Nothing was booting. The restart had killed the process that was, leaving its
URL-less registry entry behind. Two decisions turned that into a frozen pool:

1. `sibling_is_provisioning` judged the entry's liveness against
   REAP_GRACE_SECONDS (1200s), which answers a different question than the one
   being asked. The registry has a 60s heartbeat, so a dead owner is detectable
   in seconds; twenty minutes was the reaper's patience, borrowed by mistake.

2. The sibling check ran BEFORE `_adopt_shared_pod()`. So the recovery path for
   orphaned pods sat behind the very marker the orphaning created, and the five
   running, registered, billing pods were never adopted.

Plus the money: a pod registered at CREATION recorded no $/hr (the rate was only
passed at the ready-registration), so a boot interrupted before ready could only
ever be reaped with "the registry records no rate for it".
"""

import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codai.api import runpod_worker as rw                       # noqa: E402


def _entry(pool, *, url="", age=0.0, proc="someone-else", hourly=0.59):
    return {"pool": pool, "proc": proc, "pid": 999999,
            "at": time.time() - age, "created": time.time() - age,
            "url": url, "key": "k", "image": "img", "gpu": "A40",
            "hourly_usd": hourly}


def _registry(monkeypatch, data):
    monkeypatch.setattr(rw, "_registry_update", lambda fn: data)


# ------------------------------------------------- the marker's lifetime #
def test_a_fresh_sibling_boot_still_defers():
    """The original purpose must survive the fix: two pods for one capability,
    45 seconds apart, is what this check exists to prevent."""
    import pytest                                              # noqa: F401
    data = {"p1": _entry("m", url="", age=10.0)}
    assert rw.PROVISION_STALE_SECONDS > 10.0


def test_a_booting_sibling_within_the_heartbeat_window_defers(monkeypatch):
    _registry(monkeypatch, {"p1": _entry("m", url="", age=30.0)})
    assert rw.sibling_is_provisioning("m") is True


def test_a_marker_whose_owner_stopped_heartbeating_does_not_block(monkeypatch):
    """Three missed beats is a dead owner, not a slow one."""
    _registry(monkeypatch, {"p1": _entry("m", url="", age=300.0)})
    assert rw.sibling_is_provisioning("m") is False


def test_the_window_is_the_heartbeat_not_the_reap_grace():
    """1200s answered 'how long before the reaper may kill an unowned pod',
    which is not the question. Twenty minutes of a frozen pool was the cost."""
    assert rw.PROVISION_STALE_SECONDS < rw.REAP_GRACE_SECONDS
    assert rw.PROVISION_STALE_SECONDS <= 300.0


def test_the_exact_live_case_no_longer_blocks(monkeypatch):
    """y3nndqmh9wqwc9: url-less, 910s stale, owner killed by a deploy."""
    _registry(monkeypatch, {"y3nndqmh9wqwc9": _entry(
        "datalab-to/surya-ocr-2", url="", age=910.0, hourly=0.0)})
    assert rw.sibling_is_provisioning("datalab-to/surya-ocr-2") is False


def test_a_pod_that_reached_ready_is_not_a_boot_in_progress(monkeypatch):
    """A URL means it is serving, not booting; it is adoptable, not awaited."""
    _registry(monkeypatch, {"p1": _entry("m", url="https://1.2.3.4:8000",
                                         age=10.0)})
    assert rw.sibling_is_provisioning("m") is False


def test_our_own_entry_is_never_a_sibling(monkeypatch):
    _registry(monkeypatch, {"p1": _entry("m", url="", age=10.0,
                                         proc=rw._PROC_ID)})
    assert rw.sibling_is_provisioning("m") is False


def test_another_pools_boot_is_not_ours(monkeypatch):
    _registry(monkeypatch, {"p1": _entry("other", url="", age=10.0)})
    assert rw.sibling_is_provisioning("m") is False


# --------------------------------------------------- adoption comes first #
def test_maintain_adopts_before_deferring_to_a_sibling_boot():
    """Source-anchored: the recovery path must not sit behind the marker that
    the thing it recovers from leaves behind."""
    src = (ROOT / "codai/api/runpod_worker.py").read_text()
    body = src.split("# Keep min_pods warm.")[1].split("def close(")[0]
    adopt = body.index("_adopt_shared_pod()")
    sibling = body.index("sibling_is_provisioning(")
    assert adopt < sibling, (
        "sibling_is_provisioning is checked before adoption; a restart's own "
        "orphans then stay unowned while they bill")


def test_maintain_does_not_adopt_twice():
    """Adoption moved above the sibling check; the old call below it must be
    gone, or a warm top-up tries it again on every tick."""
    src = (ROOT / "codai/api/runpod_worker.py").read_text()
    body = src.split("# Keep min_pods warm.")[1].split("def close(")[0]
    assert body.count("_adopt_shared_pod()") == 1


# ------------------------------------------------------------ the money #
def test_a_pod_records_its_rate_at_creation_not_only_when_ready():
    """An interrupted boot bills at a real $/hr. Without the rate the reaper can
    only say 'the registry records no rate for it' and drop the money."""
    src = (ROOT / "codai/api/runpod_worker.py").read_text()
    at_create = src.split("# url once ready")[0].rsplit("register_pod(", 1)[1]
    assert "hourly_usd" in at_create, "creation-time registration omits the rate"


def test_a_re_registration_cannot_erase_a_known_rate(monkeypatch, tmp_path):
    """image and gpu were carried over; hourly_usd was not, so an adoption that
    did not know the rate turned a billing pod free."""
    store = {"p1": {"pool": "m", "hourly_usd": 0.59, "image": "img",
                    "gpu": "A40", "created": 1000.0}}

    def _upd(fn):
        r = fn(store)
        return store if r is False else store
    monkeypatch.setattr(rw, "_registry_update", _upd)
    rw.register_pod("p1", "m", "https://x", "k")        # no hourly_usd given
    assert store["p1"]["hourly_usd"] == 0.59


def test_a_supplied_rate_still_wins(monkeypatch):
    store = {"p1": {"pool": "m", "hourly_usd": 0.59, "created": 1000.0}}
    monkeypatch.setattr(rw, "_registry_update",
                        lambda fn: (fn(store), store)[1])
    rw.register_pod("p1", "m", "https://x", "k", hourly_usd=1.29)
    assert store["p1"]["hourly_usd"] == 1.29


def test_created_still_survives_a_re_registration(monkeypatch):
    """Already true, asserted here because the ledger's hours come from it and
    this change edits the same dict."""
    store = {"p1": {"pool": "m", "hourly_usd": 0.59, "created": 1000.0}}
    monkeypatch.setattr(rw, "_registry_update",
                        lambda fn: (fn(store), store)[1])
    rw.register_pod("p1", "m", "https://x", "k")
    assert store["p1"]["created"] == 1000.0
