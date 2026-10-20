"""A pool that adopts a running pod's token must not lose the pages in flight.

Live incident, 2026-10-13 22:27 on the Digesta orchestrator. A surya run started,
and a pod for the same pool was already running (ours, from before a restart), so
the pool adopted it instead of renting a second A40:

  22:27:15 [runpod] adopting pod sncrmkqdnd7c57's bearer token
  22:27:15 [runpod] reusing the pod already running … instead of renting another
  22:27:15 Inference error: Error code: 401 - Invalid API key.

Every one of the 64 pages in flight failed with 401 and was lost; the same lease
answered 200 seven minutes later, and the pod answered 200 to its recorded token
throughout. The cause was the ORDER of two reads: `RunpodBackend.load_model`
snapshotted `pool.api_key` into `self._headers`, and `_acquire_endpoint()` — the
call that triggers the adoption and therefore CHANGES `pool.api_key` — runs
after it. A request that crossed that moment carried the token from before it.

A pod only ever answers to the token it was launched with, so this is not
recoverable downstream: the page is thrown away for a credential that was
already correct by the time the error arrived.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from codai.backends.runpod import RunpodBackend                    # noqa: E402


class _Pool:
    def __init__(self, key):
        self.api_key = key


def _backend(pool, snapshot):
    b = RunpodBackend.__new__(RunpodBackend)
    b._pool = pool
    b._headers = {"Authorization": f"Bearer {snapshot}"} if snapshot else {}
    return b


# --- the live accessor ------------------------------------------------------

def test_the_header_is_read_from_the_pool_not_the_snapshot():
    pool = _Pool("adopted-token")
    b = _backend(pool, "stale-token")
    assert b._auth() == {"Authorization": "Bearer adopted-token"}


def test_a_token_adopted_after_load_is_used():
    # Exactly the incident: load_model snapshots, then the acquire adopts.
    pool = _Pool("own-generated")
    b = _backend(pool, "own-generated")
    pool.api_key = "the-running-pods-token"       # _adopt_shared_pod does this
    assert b._auth()["Authorization"] == "Bearer the-running-pods-token"


def test_a_pool_with_no_token_sends_no_authorization_header():
    # Some servers reject an empty bearer outright, so the header must be absent
    # rather than present and blank.
    assert _backend(_Pool(""), "") == {} or _backend(_Pool(""), "")._auth() == {}


def test_serverless_keeps_its_snapshot():
    """Serverless has no pool and no adoption event; its header is the account's."""
    b = RunpodBackend.__new__(RunpodBackend)
    b._pool = None
    b._headers = {"Authorization": "Bearer account-key"}
    assert b._auth() == {"Authorization": "Bearer account-key"}


def test_the_accessor_is_what_every_send_uses():
    """Pinned against the source: three sites used the snapshot, and any one of
    them left behind would reproduce the incident on that path.

    Since v0.4.59 the accessor is given the bearer of the pod the acquire just
    handed out (`_auth(bearer)`), because one pool can hold pods carrying
    DIFFERENT tokens -- a pool-wide key 401s whichever pod does not share it.
    Reading the pool is still the fallback."""
    src = (ROOT / "codai/backends/runpod.py").read_text()
    assert "headers=self._auth(bearer)" in src           # chat completions
    assert "headers = self._auth(bearer)" in src         # streaming
    assert "(self._auth(bearer) or {}).get(\"Authorization\"" in src  # openai_endpoint
    assert "def _auth(self, bearer: str = \"\")" in src
    # …and the placement candidate takes it FROM the acquire, since reading it
    # afterwards can pick up another concurrent request's pod.
    pl = (ROOT / "codai/models/placement.py").read_text()
    after = pl.split("base, release, bearer = backend._acquire_endpoint()")[1][:500]
    assert "backend._auth(bearer)" in after


# --- the gateway's last line of defence ------------------------------------

def test_a_401_is_retried_once_against_a_freshly_chosen_upstream(monkeypatch):
    import asyncio
    import httpx
    from codai.ocr import gateway
    from test_ocr_fanout import (_CapturingClient, _Req, _cand, _no_node_load,
                                 _fixed_candidates, _reset, SURYA_BODY)

    _reset()
    calls = []

    class _RefusesOnce(_CapturingClient):
        # The gateway reuses its pooled client on the second attempt and asks
        # whether it is still open; the shared stub is only ever used once.
        is_closed = False

        async def post(self, url, content=None, headers=None):
            calls.append(headers.get("Authorization"))

            class _R:
                status_code = 401 if len(calls) == 1 else 200
                headers = {"content-type": "application/json"}
                content = b'{"ok": true}'
            return _R()

    monkeypatch.setattr(httpx, "AsyncClient", _RefusesOnce)
    _no_node_load(monkeypatch)
    c = _cand("node:a", base="http://a/v1")
    c.api_key = "k"
    _fixed_candidates(monkeypatch, [c])
    token = gateway.issue("m", "m")
    reply = asyncio.run(gateway.chat_completions(token, _Req(SURYA_BODY)))
    # The page is served, not lost.
    assert reply.status_code == 200
    assert len(calls) == 2


def test_a_second_401_is_reported_rather_than_retried_forever(monkeypatch):
    import asyncio
    import httpx
    from codai.ocr import gateway
    from test_ocr_fanout import (_CapturingClient, _Req, _cand, _no_node_load,
                                 _fixed_candidates, _reset, SURYA_BODY)

    _reset()
    calls = []

    class _AlwaysRefuses(_CapturingClient):
        is_closed = False

        async def post(self, url, content=None, headers=None):
            calls.append(1)

            class _R:
                status_code = 401
                headers = {"content-type": "application/json"}
                content = b'{"error": "nope"}'
            return _R()

    monkeypatch.setattr(httpx, "AsyncClient", _AlwaysRefuses)
    _no_node_load(monkeypatch)
    c = _cand("node:a", base="http://a/v1")
    c.api_key = "k"
    _fixed_candidates(monkeypatch, [c])
    token = gateway.issue("m", "m")
    reply = asyncio.run(gateway.chat_completions(token, _Req(SURYA_BODY)))
    # A real credential problem must surface as itself, not as a retry loop.
    assert reply.status_code == 401
    assert len(calls) == 2


# --- the registry must not forget a pod's age when it changes hands ---------

def test_adoption_keeps_the_original_created_timestamp(monkeypatch, tmp_path):
    import json
    import time

    from codai.api import runpod_worker as rw

    path = tmp_path / "runpod-pods.json"
    born = time.time() - 3600
    path.write_text(json.dumps({"p1": {"pool": "m", "proc": "old-1",
                                       "created": born, "at": born,
                                       "url": "u", "key": "k"}}))
    monkeypatch.setattr(rw, "_pod_registry_path", lambda: str(path))
    rw.register_pod("p1", pool_key="m", url="u", api_key="k", hourly_usd=0.59)
    info = json.loads(path.read_text())["p1"]
    # Re-stamping it would make an hourly-adopted pod look newly born every
    # hour, and the reaper's ledger attribution is computed from this.
    assert abs(info["created"] - born) < 1
    assert info["proc"] == rw._PROC_ID
    assert info["at"] > born + 3000


def test_a_pod_registered_for_the_first_time_gets_a_created_stamp(monkeypatch, tmp_path):
    import json
    import time

    from codai.api import runpod_worker as rw

    path = tmp_path / "runpod-pods.json"
    path.write_text("{}")
    monkeypatch.setattr(rw, "_pod_registry_path", lambda: str(path))
    rw.register_pod("p2", pool_key="m", url="u", api_key="k")
    info = json.loads(path.read_text())["p2"]
    assert abs(info["created"] - time.time()) < 5


def test_an_adopted_pod_is_billed_from_when_it_was_created(monkeypatch, tmp_path):
    """Its hours before the adoption were charged to nobody.

    Live: a pod created 22:27 and adopted 23:10 was ledgered for 16 minutes
    when it had been running for an hour — `_terminate` computes the spend from
    `started_at`, which was set to the moment of adoption. The registry already
    records `created`, and only cost and uptime reporting read `started_at`, so
    backdating it corrects the bill and nothing else.
    """
    import json
    import time

    from codai.api import runpod_worker as rw

    born = time.time() - 3600
    path = tmp_path / "runpod-pods.json"
    path.write_text(json.dumps({"p1": {"pool": "m", "proc": "old-1",
                                       "created": born, "at": born,
                                       "url": "https://h/v1", "key": "k",
                                       "hourly_usd": 0.59}}))
    monkeypatch.setattr(rw, "_pod_registry_path", lambda: str(path))

    src = (ROOT / "codai/api/runpod_worker.py").read_text()
    blk = src.split("reusing the pod already running")[1][:1400]
    assert 'registry_entry(shared_id).get("created")' in blk

    # …and the helper really returns it, so the handle gets an hour-old start.
    assert abs(rw.registry_entry("p1")["created"] - born) < 1
