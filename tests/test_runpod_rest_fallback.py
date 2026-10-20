"""RunPod REST API v2: its own quota, and the high-volume reads moved to it.

Measured 2026-10-20 with live headers: GraphQL's daily quota (43,200) was spent
-- x-ratelimit-remaining-day: 0, Retry-After 3939 -- while REST v2 reported
"day";r=86399 of q=86400 and terminated a pod (204). REST v1 is deprecated
(retired 2026-11-15) and its writes failed through that ban.
"""
import time

import pytest

from codai.api import runpod_client as rc

V2_POD = {"id": "p1", "name": "coderai-x", "status": "RUNNING", "cost": 0.59,
          "image": "img", "createdAt": "2026-10-19T18:38:07.482Z",
          "gpu": {"id": "NVIDIA A40", "count": 1},
          "dataCenterId": "EU-SE-1",
          "runtime": {"uptime": 120, "memory": {"util": 71}, "cpu": {"util": 40},
                      "ports": [{"private": 8000, "public": 22187, "type": "tcp",
                                 "ip": "1.2.3.4"}]}}


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    rc._GQL_BACKOFF_UNTIL[0] = 0.0
    rc._V2_BACKOFF_UNTIL[0] = 0.0
    monkeypatch.setattr(rc, "_catalog_path", lambda: str(tmp_path / "cat.json"))
    monkeypatch.setattr(rc, "_GPU_CATALOG", (time.time() + 600, [
        {"id": "NVIDIA A40", "display_name": "A40", "memory_gb": 48, "secure_price": 0.59}]))
    yield
    rc._GQL_BACKOFF_UNTIL[0] = 0.0
    rc._V2_BACKOFF_UNTIL[0] = 0.0


def _client(monkeypatch, handler, gql_ok=False):
    calls = []

    def _rest(self, method, path, body=None, timeout=30.0, params=None):
        calls.append((method, path, body))
        return handler(method, path, body)
    monkeypatch.setattr(rc.RunpodClient, "_rest", _rest)
    if not gql_ok:
        monkeypatch.setattr(rc.RunpodClient, "_gql",
                            lambda *a, **k: pytest.fail("GraphQL was called"))
    c = rc.RunpodClient.__new__(rc.RunpodClient)
    c._api_key, c._cfg = "k" * 20, None
    return c, calls


def test_get_pod_is_v2_and_has_the_graphql_shape(monkeypatch):
    c, calls = _client(monkeypatch, lambda m, p, b: V2_POD)
    got = c.get_pod("p1")
    assert calls[0][:2] == ("GET", "/pods/p1")
    assert got["status"] == "RUNNING" and got["ready"] is True
    assert got["ports"] == [{"ip": "1.2.3.4", "isIpPublic": True, "privatePort": 8000,
                             "publicPort": 22187, "type": "tcp"}]
    assert got["uptime_s"] == 120 and got["memory_util"] == 71
    assert got["gpu"] == "A40" and got["vram_gb"] == 48


def test_a_provisioning_pod_is_not_ready_and_not_failed(monkeypatch):
    pod = dict(V2_POD, status="PROVISIONING", runtime={})
    c, _ = _client(monkeypatch, lambda m, p, b: pod)
    got = c.get_pod("p1")
    assert got["status"] == "RUNNING" and got["phase"] == "PROVISIONING"
    assert got["ready"] is False


def test_an_errored_pod_reads_as_failed(monkeypatch):
    c, _ = _client(monkeypatch, lambda m, p, b: dict(V2_POD, status="ERROR"))
    assert c.get_pod("p1")["status"] == "FAILED"


def test_listing_is_v2_with_pagination(monkeypatch):
    pages = iter([{"pods": [V2_POD], "pagination": {"hasNextPage": True, "nextCursor": "c2"}},
                  {"pods": [dict(V2_POD, id="p2")], "pagination": {"hasNextPage": False}}])
    c, calls = _client(monkeypatch, lambda m, p, b: next(pages))
    rows = c.list_pods()
    assert [r["id"] for r in rows] == ["p1", "p2"]
    assert rows[0]["createdAt"] == "2026-10-19T18:38:07.482Z"   # the reaper's age


def test_terminate_is_v2_and_a_404_is_gone(monkeypatch):
    def h(m, p, b):
        raise rc.RunpodError("RunPod REST v2 DELETE /pods/p1 HTTP 404: not found")
    c, calls = _client(monkeypatch, h)
    c.terminate_pod("p1")
    assert calls[0][:2] == ("DELETE", "/pods/p1")


def test_create_goes_to_v2_during_a_graphql_ban(monkeypatch):
    rc._GQL_BACKOFF_UNTIL[0] = time.time() + 3600
    c, calls = _client(monkeypatch, lambda m, p, b: {"id": "new1"})
    pid = c.create_pod(name="n", image="img", gpu_type_id="NVIDIA A40", port=8000,
                       env={"A": "1"}, direct_tcp=True, data_center_id="EU-SE-1",
                       network_volume_id="vol1")
    assert pid == "new1"
    m, path, body = calls[0]
    assert (m, path) == ("POST", "/pods")
    assert body["gpu"] == {"id": "NVIDIA A40", "count": 1}
    assert body["ports"] == ["8000/tcp"] and body["cloud"] == "SECURE"
    assert body["dataCenterIds"] == ["EU-SE-1"]
    assert body["mounts"]["network"] == [{"volumeId": "vol1", "path": "/workspace"}]


def test_v2_will_not_rent_spot(monkeypatch):
    rc._GQL_BACKOFF_UNTIL[0] = time.time() + 3600
    c, _ = _client(monkeypatch, lambda *a: {"id": "x"})
    with pytest.raises(rc.RunpodError):
        c.create_pod(name="n", image="i", gpu_type_id="g", port=8000,
                     is_spot=True, bid_per_gpu=0.2)


def test_the_catalogue_comes_from_v2_during_a_ban(monkeypatch):
    rc._GQL_BACKOFF_UNTIL[0] = time.time() + 3600
    monkeypatch.setattr(rc, "_GPU_CATALOG", (0.0, None))
    cat = {"gpus": [{"id": "NVIDIA A40", "name": "A40", "memory": 48, "secure": True,
                     "community": False, "price": {"secure": 0.59, "community": 0.4}}]}
    c, _ = _client(monkeypatch, lambda m, p, b: cat)
    rows = c.list_gpu_types()
    assert rows[0]["secure_price"] == 0.59 and rows[0]["community_price"] is None
    assert rows[0]["spot_price"] is None


def test_the_saved_catalogue_is_the_last_resort(monkeypatch):
    rows = [{"id": "NVIDIA A40", "display_name": "A40", "memory_gb": 48, "secure_price": 0.59}]
    rc._save_catalog(rows)
    rc._GQL_BACKOFF_UNTIL[0] = time.time() + 3600
    monkeypatch.setattr(rc, "_GPU_CATALOG", (0.0, None))

    def down(m, p, b):
        raise rc.RunpodTransient("v2 down")
    c, _ = _client(monkeypatch, down)
    assert c.list_gpu_types() == rows


def test_boot_polling_is_v2_plus_a_real_connect(monkeypatch):
    import socket
    c, _ = _client(monkeypatch, lambda m, p, b: V2_POD)
    monkeypatch.setattr(socket, "create_connection",
                        lambda *a, **k: (_ for _ in ()).throw(ConnectionRefusedError()))
    assert c._rest_boot_state("p1", 8000)["ready"] is False

    class _S:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: _S())
    assert c._rest_boot_state("p1", 8000)["ready"] is True


def test_v2_429_is_honoured(monkeypatch):
    import requests

    class _R:
        status_code = 429
        content = b"x"
        text = "slow down"
        headers = {"Retry-After": "12", "RateLimit": '"minute";r=0;t=12'}
    sent = []
    monkeypatch.setattr(requests, "request", lambda *a, **k: sent.append(1) or _R())
    c = rc.RunpodClient.__new__(rc.RunpodClient)
    c._api_key, c._cfg = "k" * 20, None
    with pytest.raises(rc.RunpodTransient):
        c._rest("GET", "/pods")
    with pytest.raises(rc.RunpodTransient):
        c._rest("GET", "/pods")
    assert len(sent) == 1
    assert rc.v2_quota()["text"].startswith('"minute";r=0')


def test_graphql_calls_are_counted_by_operation():
    rc._GQL_COUNTS.clear()
    rc._count_gql("query Pod($input: PodFilter!) { pod(input: $input) { id } }")
    rc._count_gql("mutation { podTerminate(input: {podId: \"x\"}) }")
    got = rc.gql_call_counts()
    assert got.get("pod") == 1 and got.get("podTerminate") == 1


# ------------------------------------------- staying under the daily quota #

def test_over_the_hourly_budget_routes_to_v2(monkeypatch):
    """RunPod's GraphQL day quota (43,200) averages 1,800/h; spending it ends
    in a ban until midnight UTC. Past our own budget, v2 does the work."""
    import collections
    monkeypatch.setattr(rc, "GQL_HOURLY_BUDGET", 5)
    monkeypatch.setattr(rc, "_GQL_RECENT", collections.deque([time.time()] * 5))
    assert rc._gql_banned() is True
    c, calls = _client(monkeypatch, lambda m, p, b: {"id": "new1"})
    assert c.create_pod(name="n", image="i", gpu_type_id="NVIDIA A40", port=8000,
                        direct_tcp=True) == "new1"
    assert calls[0][:2] == ("POST", "/pods")


def test_old_calls_fall_out_of_the_window(monkeypatch):
    import collections
    monkeypatch.setattr(rc, "GQL_HOURLY_BUDGET", 5)
    monkeypatch.setattr(rc, "_GQL_RECENT", collections.deque([time.time() - 4000] * 50))
    assert rc._gql_banned() is False
    assert rc._gql_calls_last_hour() == 0


def test_the_budget_does_not_block_what_only_graphql_can_do(monkeypatch):
    """Spot rentals have no v2 equivalent: the budget reroutes, it never
    refuses a GraphQL call on its own -- only RunPod's 429 does that."""
    import collections, requests
    monkeypatch.setattr(rc, "GQL_HOURLY_BUDGET", 1)
    monkeypatch.setattr(rc, "_GQL_RECENT", collections.deque([time.time()] * 3))

    class _R:
        status_code = 200
        text = ""

        def json(self):
            return {"data": {"podRentInterruptable": {"id": "spot1"}}}
    monkeypatch.setattr(requests, "post", lambda *a, **k: _R())
    c = rc.RunpodClient.__new__(rc.RunpodClient)
    c._api_key, c._gql_base, c._cfg = "k" * 20, "https://api.runpod.io/graphql", None
    assert c._gql("mutation { podRentInterruptable(input: {}) { id } }") == {
        "podRentInterruptable": {"id": "spot1"}}


def test_zero_disables_the_budget(monkeypatch):
    import collections
    monkeypatch.setattr(rc, "GQL_HOURLY_BUDGET", 0)
    monkeypatch.setattr(rc, "_GQL_RECENT", collections.deque([time.time()] * 99999))
    assert rc._gql_over_budget() is False
