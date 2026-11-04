"""The pods page has to show a pod the ACCOUNT is billing, not just ours.

Every source `pods_status` has is local: the pools, the provisioning list, THIS
process's registry file, and the parked set. That is the whole fleet only if one
orchestrator rents on the account, and that is not the arrangement -- a second
host rents on the same account and keeps its own registry file, so its pods were
invisible here. So was a pod orphaned by a process whose registry entry was
lost, and that one is a leak: it bills until somebody opens the RunPod console.

The registry source was added for this exact symptom once already -- "the
account billed four pods while this page showed two" -- and it closed only the
half where the pod was ours. The authoritative answer is the account listing,
which is what `reconcile_pools` and the reaper already decide from, and never a
probe: a pod that does not answer is still billing.

Two things this must not do. It must not spend the API quota that renting and
boot-polling need, because a 429 there stops the fleet scaling -- so the listing
is cached and the page's 10s poll (per open tab) cannot reach past the cache.
And it must not fold another orchestrator's spend into this host's rate, which
would make the two hosts' pages disagree about one account.
"""
import asyncio
import inspect
import pathlib
import re

import pytest

from codai.api import runpod_worker as W

PAGE = (pathlib.Path(__file__).resolve().parents[1]
        / "codai/admin/templates/runpod.html").read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
def _clear_cache():
    W._account_cache["pods"] = None
    W._account_cache["at"] = 0.0
    yield
    W._account_cache["pods"] = None
    W._account_cache["at"] = 0.0


ACCOUNT = [
    {"id": "known1", "name": "qwen38-a", "status": "RUNNING",
     "cost_per_hr": 0.74, "gpu": "A40", "uptime_s": 3600.0,
     "image": "ghcr.io/nextime/coderai-vllm:latest"},
    {"id": "strange", "name": "qwen38-b", "status": "RUNNING",
     "cost_per_hr": 0.82, "gpu": "RTX A6000", "uptime_s": 1800.0,
     "image": "ghcr.io/nextime/coderai-vllm:latest"},
    {"id": "stopped", "name": "old", "status": "EXITED",
     "cost_per_hr": 0.50, "gpu": "A40", "uptime_s": 99.0},
]


@pytest.fixture
def listing(monkeypatch):
    monkeypatch.setattr(W, "account_listing", lambda force=False: list(ACCOUNT))


def test_a_pod_no_local_source_knows_about_gets_a_row(listing):
    rows = W.account_only_pod_rows({"known1"})
    assert [r["pod_id"] for r in rows] == ["strange"]
    r = rows[0]
    assert r["state"] == "unmanaged"
    assert r["hourly_usd"] == 0.82
    assert r["gpu"] == "RTX A6000"
    assert r["live_cost_usd"] == pytest.approx(0.82 * 0.5, rel=1e-3)


def test_a_pod_we_already_show_is_not_duplicated(listing):
    assert W.account_only_pod_rows({"known1", "strange"}) == []


def test_a_stopped_pod_is_not_listed_as_live(listing):
    """A stopped pod bills storage, not compute; showing it among running pods
    would be its own kind of wrong."""
    assert "stopped" not in {r["pod_id"] for r in W.account_only_pod_rows(set())}


def test_the_note_says_where_it_might_belong(listing):
    note = W.account_only_pod_rows({"known1"})[0]["note"]
    assert "billing" in note
    assert "leak" in note, "an operator needs to know this may be nobody's pod"
    assert "qwen38-b" in note, "the pod's own name is the first clue to its owner"
    assert "coderai-vllm:latest" in note


def test_it_offers_a_console_link(listing):
    """The always-available fallback, and the only tool for a pod this process
    genuinely does not own."""
    assert W.account_only_pod_rows({"known1"})[0]["console_url"]


# ── the quota, which renting depends on ──────────────────────────────────────

def test_the_listing_is_cached(monkeypatch):
    """The page polls every 10s and every open tab polls separately. One REST
    read per tab per 10s out of the quota renting needs is how a fleet loses
    the ability to scale."""
    calls = []

    class _Client:
        def __init__(self, acct):
            pass

        def list_pods(self):
            calls.append(1)
            return list(ACCOUNT)

    monkeypatch.setattr("codai.api.runpod_client.RunpodClient", _Client)
    monkeypatch.setattr("codai.models.manager.get_active_runpod_config",
                        lambda: type("A", (), {"enabled": True})())
    for _ in range(5):
        assert len(W.account_listing()) == 3
    assert len(calls) == 1, f"{len(calls)} API reads for 5 page polls"
    assert W.ACCOUNT_LISTING_TTL_S >= 60


def test_runpod_disabled_asks_nothing(monkeypatch):
    monkeypatch.setattr("codai.models.manager.get_active_runpod_config",
                        lambda: None)

    def _boom(*a, **k):
        raise AssertionError("asked RunPod with the integration disabled")

    monkeypatch.setattr("codai.api.runpod_client.RunpodClient", _boom)
    assert W.account_listing() == []


def test_an_api_failure_does_not_take_the_page_with_it(monkeypatch, capsys):
    """The spend figures beside this must stay right even when reads are
    being refused."""
    class _Client:
        def __init__(self, acct):
            pass

        def list_pods(self):
            raise RuntimeError("429 Too Many Requests")

    monkeypatch.setattr("codai.api.runpod_client.RunpodClient", _Client)
    monkeypatch.setattr("codai.models.manager.get_active_runpod_config",
                        lambda: type("A", (), {"enabled": True})())
    assert W.account_listing() == []
    assert W.account_only_pod_rows(set()) == []
    assert "account listing unavailable" in capsys.readouterr().out


def test_a_stale_listing_beats_no_listing(monkeypatch):
    """Once a pod has been seen on the account, a transient API failure must not
    make it vanish again -- disappearing is what made it invisible."""
    W._account_cache["pods"] = list(ACCOUNT)
    W._account_cache["at"] = 0.0          # expired

    class _Client:
        def __init__(self, acct):
            pass

        def list_pods(self):
            raise RuntimeError("502")

    monkeypatch.setattr("codai.api.runpod_client.RunpodClient", _Client)
    monkeypatch.setattr("codai.models.manager.get_active_runpod_config",
                        lambda: type("A", (), {"enabled": True})())
    assert len(W.account_listing()) == 3


# ── the endpoint and the page ────────────────────────────────────────────────

def test_the_rate_counts_them_but_the_ledger_figure_does_not():
    """The meter and the ledger are different questions.

    `live_hourly_usd` exists because it once read $0.59/hr against an account
    billing $4.72 and a client concluded nothing was being spent. A pod bills
    whoever rented it, so filtering these out of the RATE would recreate that
    understatement by a new route -- see
    tests/test_live_hourly_counts_booting_pods.py, which pins the shape.

    `live_uncommitted_usd` is "running, not yet billed to the ledger", and this
    host's ledger will never bill another orchestrator's pod. So that one is
    ours alone, and the attribution is reported as its own figure.
    """
    from codai.admin import routes
    src = inspect.getsource(routes.api_runpod_stats)
    assert "account_only_pod_rows" in src
    flat = " ".join(src.split())
    assert ('live_hourly = sum((p.get("hourly_usd") or 0.0) for p in pods)'
            in flat), "the rate must stay account-wide"
    i = flat.index("live_cost = sum(")
    assert 'p.get("state") != "unmanaged"' in flat[i:i + 200], (
        "the uncommitted/ledger figure must exclude pods we will never bill")
    assert "unmanaged_hourly_usd" in src, "their cost has to be attributable"


def test_the_endpoint_does_not_block_the_loop_on_it():
    """A RunPod read on the event loop is how the front wedges."""
    from codai.admin import routes
    src = inspect.getsource(routes.api_runpod_stats)
    i = src.index("account_only_pod_rows")
    assert "await _part(" in src[max(0, i - 300):i], (
        "the account listing must go through the bounded off-loop helper")


def test_the_page_renders_the_state_and_offers_the_button():
    assert "p.state === 'unmanaged'" in PAGE
    i = PAGE.index("const loud =")
    assert "unmanaged" in PAGE[i:i + 160], (
        "a billing pod nothing manages is exactly the urgent case")
    assert "unmanaged_hourly_usd" in PAGE


def test_the_pod_count_tile_keeps_them_separate():
    """Counted as healthy it would look like capacity; counted as unhealthy it
    would look like our problem to restart. It is neither."""
    # rindex: the first "rp-podcount" is the tile's own markup up in the HTML,
    # not the JS that fills it.
    i = PAGE.rindex("rp-podcount")
    body = PAGE[max(0, i - 900):i + 400]
    assert "unmanaged" in body
