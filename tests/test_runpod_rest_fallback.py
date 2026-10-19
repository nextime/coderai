"""While RunPod's GraphQL is rate-limited, the pool keeps working over REST.

The 2026-10-19 ban lasted ~3.3 h and REST answered throughout. Without these
fallbacks a ban stops everything: renting (create + the catalogue that prices
cards), watching a pod boot (get_pod), the reaper (list_pods), and teardown.
"""
import json
import time

import pytest

from codai.api import runpod_client as rc

REST_POD = {"id": "p1", "name": "coderai-x", "desiredStatus": "RUNNING",
            "costPerHr": 0.59, "imageName": "img", "publicIp": "1.2.3.4",
            "portMappings": {"8000": 22187}, "lastStatusChange": "Rented by User",
            "lastStartedAt": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time() - 120))
            + ".477 +0000 UTC"}


@pytest.fixture
def banned(monkeypatch, tmp_path):
    rc._GQL_BACKOFF_UNTIL[0] = time.time() + 3600
    monkeypatch.setattr(rc, "_catalog_path", lambda: str(tmp_path / "cat.json"))
    yield tmp_path
    rc._GQL_BACKOFF_UNTIL[0] = 0.0


def _client(monkeypatch, handler):
    calls = []

    def _rest(self, method, path, body=None, timeout=30.0):
        calls.append((method, path, body))
        return handler(method, path, body)
    monkeypatch.setattr(rc.RunpodClient, "_rest", _rest)
    monkeypatch.setattr(rc.RunpodClient, "_gql",
                        lambda *a, **k: pytest.fail("GraphQL called during a ban"))
    c = rc.RunpodClient.__new__(rc.RunpodClient)
    c._api_key, c._cfg = "k" * 20, None
    return c, calls


def test_get_pod_over_rest_has_the_same_shape(monkeypatch, banned):
    pod = dict(REST_POD, lastStartedAt=time.strftime(
        "%Y-%m-%d %H:%M:%S", time.gmtime(time.time() - 120)) + ".477 +0000 UTC")
    c, _ = _client(monkeypatch, lambda m, p, b: pod)
    got = c.get_pod("p1")
    assert got["ready"] is True and got["status"] == "RUNNING"
    assert got["ports"] == [{"ip": "1.2.3.4", "isIpPublic": True,
                             "privatePort": 8000, "publicPort": 22187, "type": "tcp"}]
    assert 100 < got["uptime_s"] < 200


def test_list_pods_over_rest(monkeypatch, banned):
    c, _ = _client(monkeypatch, lambda m, p, b: [REST_POD])
    rows = c.list_pods()
    assert rows[0]["id"] == "p1" and rows[0]["cost_per_hr"] == 0.59
    assert "uptime_s" in rows[0]


def test_terminate_over_rest_and_a_404_is_gone(monkeypatch, banned):
    def h(m, p, b):
        raise rc.RunpodError("RunPod REST DELETE /pods/p1 HTTP 404: not found")
    c, calls = _client(monkeypatch, h)
    c.terminate_pod("p1")                       # no raise: already gone
    assert calls[0][:2] == ("DELETE", "/pods/p1")


def test_create_goes_to_rest_during_a_ban(monkeypatch, banned):
    import requests
    sent = {}

    class _R:
        status_code = 200
        content = b"x"
        text = ""

        def json(self):
            return {"id": "new1"}

    def post(url, json=None, headers=None, timeout=None):
        sent.update(url=url, body=json)
        return _R()
    monkeypatch.setattr(requests, "post", post)
    c = rc.RunpodClient.__new__(rc.RunpodClient)
    c._api_key, c._cfg = "k" * 20, None
    pid = c.create_pod(name="n", image="img", gpu_type_id="NVIDIA A40", port=8000,
                       env={"A": "1"}, direct_tcp=True)
    assert pid == "new1"
    assert sent["url"].endswith("/pods")
    assert sent["body"]["ports"] == ["8000/tcp"]
    assert sent["body"]["gpuTypeIds"] == ["NVIDIA A40"]


def test_the_catalogue_is_saved_and_used_during_a_ban(monkeypatch, banned):
    rows = [{"id": "NVIDIA A40", "display_name": "A40", "memory_gb": 48,
             "secure_price": 0.59}]
    rc._GQL_BACKOFF_UNTIL[0] = 0.0
    rc._save_catalog(rows)
    rc._GQL_BACKOFF_UNTIL[0] = time.time() + 3600
    monkeypatch.setattr(rc, "_GPU_CATALOG", (0.0, None))
    c, _ = _client(monkeypatch, lambda *a: {})
    assert c.list_gpu_types() == rows


# ------------------------------------------------- fewer GraphQL calls at all #

def test_listing_uses_rest_even_without_a_ban(monkeypatch):
    """The reaper and the survivor sweep list every minute."""
    rc._GQL_BACKOFF_UNTIL[0] = 0.0
    c, calls = _client(monkeypatch, lambda m, p, b: [dict(REST_POD, machine={"gpuTypeId": "NVIDIA A40"})])
    rows = c.list_pods()
    assert calls and calls[0][1].startswith("/pods?includeMachine=true")
    assert rows[0]["id"] == "p1"


def test_boot_polling_is_rest_plus_a_real_connect(monkeypatch):
    """GraphQL lists the port when the container STARTS; RunPod refuses a TCP
    connect until the server inside LISTENS. The connect is the signal."""
    import socket
    rc._GQL_BACKOFF_UNTIL[0] = 0.0
    c, _ = _client(monkeypatch, lambda m, p, b: REST_POD)
    monkeypatch.setattr(socket, "create_connection",
                        lambda *a, **k: (_ for _ in ()).throw(ConnectionRefusedError()))
    assert c._rest_boot_state("p1", 8000)["ready"] is False

    class _S:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: _S())
    st = c._rest_boot_state("p1", 8000)
    assert st["ready"] is True and st["ports"][0]["publicPort"] == 22187


def test_graphql_calls_are_counted_by_operation():
    rc._GQL_COUNTS.clear()
    rc._count_gql("query Pod($input: PodFilter!) { pod(input: $input) { id } }")
    rc._count_gql("mutation { podTerminate(input: {podId: \"x\"}) }")
    got = rc.gql_call_counts()
    assert got.get("pod") == 1 and got.get("podTerminate") == 1


def test_a_rest_write_during_a_ban_says_why(monkeypatch):
    import requests
    rc._GQL_BACKOFF_UNTIL[0] = time.time() + 600

    class _R:
        status_code = 500
        content = b"x"
        text = '{"error":"delete pod: unexpected end of JSON input","status":500}'
    monkeypatch.setattr(requests, "request", lambda *a, **k: _R())
    c = rc.RunpodClient.__new__(rc.RunpodClient)
    c._api_key, c._cfg = "k" * 20, None
    with pytest.raises(rc.RunpodTransient) as e:
        c._rest("DELETE", "/pods/p1")
    assert "share the GraphQL rate limit" in str(e.value)
    rc._GQL_BACKOFF_UNTIL[0] = 0.0
