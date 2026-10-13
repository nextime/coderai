"""A registry entry must stop vouching for a pod once its owner is gone.

Live incident, 2026-10-13 on the Digesta orchestrator. An unattended upgrade
restarted coderai at 21:17. A qwen38 pool pod (A40, $0.59/hr) survived the
restart, and then nothing touched it:

  * its pool's warm floor was 0 — outside its 08:00-20:00 schedule window — so
    nothing provisioned, and the adoption path (`find_shared_pod`) only runs
    when a pool actually needs a pod. No pool tracked it, so no pool applied
    `idle_timeout_s` to it either;
  * the stale-pod reaper skips anything in `_known_pod_ids()`, and the DEAD
    process's registry entry was still in there — `registered_pod_ids()`
    returned every key in the file regardless of who wrote it or whether that
    process still existed.

So the pod was simultaneously unmanaged and shielded, and billed on. `at` was
written once at registration and never updated, so it could not tell the two
cases apart. It is now a heartbeat: an owner refreshes it every scaler pass,
and silence past REAP_GRACE_SECONDS lets the pod be reaped like any other
leftover.
"""

import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codai.api import runpod_worker as rw                          # noqa: E402


def _registry(tmp_path, entries):
    path = tmp_path / "runpod-pods.json"
    path.write_text(json.dumps(entries))
    return path


def _patch_registry(monkeypatch, path):
    monkeypatch.setattr(rw, "_pod_registry_path", lambda: str(path))


# --- the heartbeat ----------------------------------------------------------

def test_registering_records_creation_separately_from_last_seen(monkeypatch, tmp_path):
    path = _registry(tmp_path, {})
    _patch_registry(monkeypatch, path)
    rw.register_pod("p1", pool_key="pool:x", url="https://h", hourly_usd=0.59)
    info = json.loads(path.read_text())["p1"]
    # "at" moves with the heartbeat, so the hours a reaped pod billed can only
    # be computed from a timestamp that does not move.
    assert "created" in info and info["created"] <= info["at"] + 1


def test_the_heartbeat_refreshes_only_our_own_entries(monkeypatch, tmp_path):
    old = time.time() - 5000
    path = _registry(tmp_path, {
        "mine": {"pool": "p", "proc": rw._PROC_ID, "at": old, "url": "u"},
        "theirs": {"pool": "p", "proc": "999-deadbeef", "at": old, "url": "u"},
    })
    _patch_registry(monkeypatch, path)
    assert rw.touch_registered_pods() == 1
    data = json.loads(path.read_text())
    assert data["mine"]["at"] > old + 4000
    # Touching a sibling's entry would be claiming to be alive on its behalf.
    assert data["theirs"]["at"] == old


def test_the_heartbeat_survives_an_empty_registry(monkeypatch, tmp_path):
    _patch_registry(monkeypatch, _registry(tmp_path, {}))
    assert rw.touch_registered_pods() == 0


# --- what still counts as known --------------------------------------------

def test_our_own_entry_counts_however_old(monkeypatch, tmp_path):
    # This process manages its own pods directly; the file is a courtesy to
    # siblings, not the source of truth for us.
    _patch_registry(monkeypatch, _registry(tmp_path, {
        "mine": {"pool": "p", "proc": rw._PROC_ID, "at": time.time() - 99999}}))
    assert rw.live_registered_pod_ids() == {"mine"}


def test_a_siblings_fresh_entry_counts(monkeypatch, tmp_path):
    _patch_registry(monkeypatch, _registry(tmp_path, {
        "theirs": {"pool": "p", "proc": "999-x", "at": time.time() - 30}}))
    assert rw.live_registered_pod_ids() == {"theirs"}


def test_an_entry_whose_owner_went_quiet_stops_counting(monkeypatch, tmp_path):
    # The incident: the owner was a process that no longer exists.
    _patch_registry(monkeypatch, _registry(tmp_path, {
        "orphan": {"pool": "p", "proc": "41-06956b37e2e9",
                   "at": time.time() - (rw.REAP_GRACE_SECONDS + 60)}}))
    assert rw.live_registered_pod_ids() == set()
    # …while the raw view still has it, which is what the reaper used before.
    assert rw.registered_pod_ids() == {"orphan"}


def test_the_lapse_is_the_reap_grace_not_something_shorter(monkeypatch, tmp_path):
    """A sibling whose scaler is between passes must not lose its pods.

    The heartbeat runs every fourth 15 s pass, so the margin here is sixty
    seconds against twenty minutes.
    """
    _patch_registry(monkeypatch, _registry(tmp_path, {
        "theirs": {"pool": "p", "proc": "999-x",
                   "at": time.time() - (rw.REAP_GRACE_SECONDS - 120)}}))
    assert rw.live_registered_pod_ids() == {"theirs"}


def test_a_malformed_entry_is_ignored_not_fatal(monkeypatch, tmp_path):
    _patch_registry(monkeypatch, _registry(tmp_path, {"junk": "not a dict"}))
    assert rw.live_registered_pod_ids() == set()


def test_known_ids_uses_the_live_view(monkeypatch, tmp_path):
    _patch_registry(monkeypatch, _registry(tmp_path, {
        "orphan": {"pool": "p", "proc": "dead-1",
                   "at": time.time() - (rw.REAP_GRACE_SECONDS + 60)}}))
    # The whole bug in one assertion: this used to contain "orphan", and the
    # reaper skips everything it contains.
    assert "orphan" not in rw._known_pod_ids()


# --- the entry the reaper reads to attribute the spend ----------------------

def test_registry_entry_reads_one_entry(monkeypatch, tmp_path):
    _patch_registry(monkeypatch, _registry(tmp_path, {
        "p1": {"pool": "pool:x", "hourly_usd": 0.59, "created": 1000.0}}))
    info = rw.registry_entry("p1")
    assert info["pool"] == "pool:x" and info["hourly_usd"] == 0.59
    assert rw.registry_entry("nope") == {}


def test_the_reaper_attributes_and_forgets_what_it_kills(monkeypatch, tmp_path):
    began = time.time() - 7200          # two hours
    _patch_registry(monkeypatch, _registry(tmp_path, {
        "orphan": {"pool": "pool:x", "proc": "dead-1", "hourly_usd": 0.59,
                   "created": began,
                   "at": began}}))

    killed, spend = [], []

    class _Client:
        def list_pods(self):
            return [{"id": "orphan", "name": "coderai-dep-model-abc",
                     "status": "RUNNING", "uptime_s": 7200}]

        def terminate_pod(self, pid):
            killed.append(pid)

    # reap_orphans imports these inside the function, so patch them at source.
    import codai.api.runpod_client as rc
    monkeypatch.setattr(rc, "RunpodClient", lambda *a, **k: _Client())
    monkeypatch.setattr(rw, "_deploy_tag", lambda *a, **k: "dep")

    import codai.api.runpod_ledger as ledger
    monkeypatch.setattr(ledger, "record_spend",
                        lambda key, cost, meta: spend.append((key, cost, meta)))

    class _Acct:
        enabled = True
        api_key = "k"
        deployment_id = "dep"

    import codai.models.manager as mm
    monkeypatch.setattr(mm, "get_active_runpod_config", lambda: _Acct())

    rw.reap_orphans()

    assert killed == ["orphan"]
    # An orphan's hours used to be invisible: only _terminate recorded spend,
    # and that runs for a pod a pool was tracking.
    assert spend and spend[0][0] == "pool:x"
    assert 1.1 < spend[0][1] < 1.2        # 2h at $0.59
    assert spend[0][2]["reaped"] is True
    # And it must not be reaped twice on the next sweep.
    assert "orphan" not in json.loads((tmp_path / "runpod-pods.json").read_text())
