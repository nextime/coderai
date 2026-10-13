"""Hours nobody ledgered: a pod that went away without a pool noticing.

A pool closes its own pods' bills — `_terminate` for one it retires, `forget`
for one RunPod says is gone. Neither runs for a pod only the REGISTRY knew
about: one orphaned by a restart, or terminated from the console or straight
through the RunPod API. Those hours were billed and recorded as nothing.

Observed on 2026-10-13: pod lgdgfs749chkib was killed by hand through the RunPod
API and never appeared in the ledger, while two pods that went through coderai
did. A cost that is invisible cannot be budgeted against, and the surprise
arrives later, on the invoice.

The second half: `record_spend` kept a fixed set of fields and dropped `reaped`,
which was being passed — so in the file a pod the reaper killed looked exactly
like one a pool retired on purpose.
"""
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codai.api import runpod_ledger as ledger                      # noqa: E402
from codai.api import runpod_worker as rw                          # noqa: E402


def _registry(monkeypatch, tmp_path, entries):
    path = tmp_path / "runpod-pods.json"
    path.write_text(json.dumps(entries))
    monkeypatch.setattr(rw, "_pod_registry_path", lambda: str(path))
    return path


def _spend_sink(monkeypatch):
    seen = []
    monkeypatch.setattr(ledger, "record_spend",
                        lambda key, usd, meta=None: seen.append((key, usd, meta)))
    return seen


# --- the fields the ledger keeps -------------------------------------------

def test_the_ledger_keeps_the_reaped_flag(monkeypatch, tmp_path):
    monkeypatch.setattr(ledger, "_PATH", str(tmp_path / "led.json"), raising=False)
    monkeypatch.setattr(ledger, "_ledger_path", lambda: str(tmp_path / "led.json"),
                        raising=False)
    kept = {}
    monkeypatch.setattr(ledger, "_load", lambda: {})
    monkeypatch.setattr(ledger, "_save", lambda d: kept.update(d))
    ledger.record_spend("m", 0.5, {"pod_id": "p1", "gpu": "(orphan)",
                                   "reaped": True, "reason": "stale",
                                   "ignored": "x"})
    rec = kept["m"][0]
    assert rec["reaped"] is True and rec["reason"] == "stale"
    assert rec["pod_id"] == "p1" and rec["gpu"] == "(orphan)"
    # Still an allow-list: the ledger is persisted for every pod, so it does not
    # grow a field because a caller passed one.
    assert "ignored" not in rec


# --- closing a bill from the registry alone --------------------------------

def test_a_pods_hours_are_computed_from_when_it_was_created(monkeypatch, tmp_path):
    born = time.time() - 3600
    _registry(monkeypatch, tmp_path,
              {"p1": {"pool": "m", "proc": "dead", "created": born, "at": born,
                      "hourly_usd": 0.59, "url": "u", "key": "k"}})
    seen = _spend_sink(monkeypatch)
    cost = rw.close_bill_from_registry("p1", "gone from the account listing")
    assert 0.58 < cost < 0.60                  # an hour at $0.59
    key, usd, meta = seen[0]
    assert key == "m" and abs(usd - cost) < 1e-9
    assert meta["pod_id"] == "p1" and meta["reaped"] is True
    assert meta["reason"] == "gone from the account listing"


def test_an_entry_with_no_rate_says_so_instead_of_recording_zero(monkeypatch,
                                                                 tmp_path, capsys):
    _registry(monkeypatch, tmp_path,
              {"p1": {"pool": "m", "created": time.time() - 60}})
    seen = _spend_sink(monkeypatch)
    assert rw.close_bill_from_registry("p1", "reaped") == 0.0
    assert seen == []
    # Silence would leave an unexplained gap between the invoice and the ledger.
    assert "cannot attribute p1's spend" in capsys.readouterr().out


def test_an_unknown_pod_is_not_invented(monkeypatch, tmp_path):
    _registry(monkeypatch, tmp_path, {})
    seen = _spend_sink(monkeypatch)
    assert rw.close_bill_from_registry("nope", "reaped") == 0.0 and seen == []


def test_a_pod_with_no_pool_recorded_is_attributed_to_orphan(monkeypatch, tmp_path):
    born = time.time() - 1800
    _registry(monkeypatch, tmp_path,
              {"p1": {"created": born, "at": born, "hourly_usd": 0.59}})
    seen = _spend_sink(monkeypatch)
    rw.close_bill_from_registry("p1", "reaped as a stale pod")
    assert seen[0][0] == "(orphan)"


# --- the reaper uses it, once, for both of its paths ----------------------

def test_the_reaper_attributes_a_pod_that_vanished(monkeypatch, tmp_path):
    """RunPod reclaimed it, or someone killed it from the console: no pool had
    it, so nothing closed the bill."""
    born = time.time() - 7200
    _registry(monkeypatch, tmp_path,
              {"p1": {"pool": "m", "proc": "dead", "created": born, "at": born,
                      "hourly_usd": 0.59, "url": "u", "key": "k"}})
    seen = _spend_sink(monkeypatch)

    class _Client:
        def list_pods(self):
            return []                      # the account no longer has it

    import codai.api.runpod_client as rc
    monkeypatch.setattr(rc, "RunpodClient", lambda *a, **k: _Client())
    monkeypatch.setattr(rw, "_deploy_tag", lambda *a, **k: "dep")

    class _Acct:
        enabled = True
        api_key = "k"
        deployment_id = "dep"

    import codai.models.manager as mm
    monkeypatch.setattr(mm, "get_active_runpod_config", lambda: _Acct())
    # Absent from TWO listings in a row before anything acts — one API page is
    # never evidence (the rule reconcile_pools exists to enforce).
    rw.reap_orphans()
    assert seen == []
    rw.reap_orphans()
    assert seen and seen[0][2]["reason"] == "gone from the account listing"


def test_a_pod_a_pool_still_holds_is_not_counted_twice(monkeypatch, tmp_path):
    """`forget` closes that one's bill; attributing it here as well would
    double-count every pod RunPod reclaims."""
    born = time.time() - 600
    _registry(monkeypatch, tmp_path,
              {"p1": {"pool": "m", "proc": rw._PROC_ID, "created": born,
                      "at": born, "hourly_usd": 0.59, "url": "u", "key": "k"}})
    calls = []
    monkeypatch.setattr(rw, "close_bill_from_registry",
                        lambda pid, reason: calls.append(pid))
    monkeypatch.setattr(rw, "reconcile_pools", lambda live, known: {"p1"})
    monkeypatch.setattr(rw, "unregister_pod", lambda pid: None)

    class _Pool:
        model_key = "m"
        pods = [rw.PodHandle(pod_id="p1", url="u", hourly_usd=0.59,
                             started_at=born)]

        def __init__(self):
            import threading
            self._cv = threading.Condition()

    class _Client:
        def list_pods(self):
            return [{"id": "zzz", "name": "coderai-dep-other", "status": "RUNNING",
                     "uptime_s": 10}]

    import codai.api.runpod_client as rc
    monkeypatch.setattr(rc, "RunpodClient", lambda *a, **k: _Client())
    monkeypatch.setattr(rw, "_deploy_tag", lambda *a, **k: "dep")

    class _Acct:
        enabled = True
        api_key = "k"
        deployment_id = "dep"

    import codai.models.manager as mm
    monkeypatch.setattr(mm, "get_active_runpod_config", lambda: _Acct())
    with rw._pools_lock:
        rw._pools["m"] = _Pool()
    try:
        rw.reap_orphans()
    finally:
        with rw._pools_lock:
            rw._pools.pop("m", None)
    assert calls == [], "a pool-held pod was ledgered twice"


def test_both_reaper_paths_share_one_implementation():
    src = (ROOT / "codai/api/runpod_worker.py").read_text()
    assert src.count("close_bill_from_registry(") >= 3    # def + two call sites
    # The old inline copy must not come back.
    assert 'runpod_ledger.record_spend(\n                        str(info.get("pool")' not in src


def test_a_registry_entry_for_a_pod_that_no_longer_exists_is_cleaned_up(
        monkeypatch, tmp_path):
    """Found by the test above, and caused by v0.3.56.

    Making a stale entry stop counting as "known" (so the reaper could kill the
    pod it named) also stopped it being RECONCILED, because the reconcile was
    only asked about known ids. An entry for a pod that no longer exists was
    then never removed — it would sit in the registry for the life of the
    install, and id() reuse aside, it is exactly the kind of record that later
    gets believed.
    """
    born = time.time() - 7200
    path = _registry(monkeypatch, tmp_path,
                     {"p1": {"pool": "m", "proc": "dead", "created": born,
                             "at": born, "hourly_usd": 0.59, "url": "u",
                             "key": "k"}})
    _spend_sink(monkeypatch)

    class _Client:
        def list_pods(self):
            return []

    import codai.api.runpod_client as rc
    monkeypatch.setattr(rc, "RunpodClient", lambda *a, **k: _Client())
    monkeypatch.setattr(rw, "_deploy_tag", lambda *a, **k: "dep")

    class _Acct:
        enabled = True
        api_key = "k"
        deployment_id = "dep"

    import codai.models.manager as mm
    monkeypatch.setattr(mm, "get_active_runpod_config", lambda: _Acct())
    rw.reap_orphans()
    rw.reap_orphans()                       # the two-sweep rule
    assert json.loads(path.read_text()) == {}
