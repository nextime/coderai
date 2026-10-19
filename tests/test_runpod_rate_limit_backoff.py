"""A RunPod 429 is honoured: no GraphQL call until its retryAfter has passed.

retryAfter is in SECONDS (observed falling by 20 every 20 s from 11,871 --
over three hours). Retrying every tick inside it only re-confirms the ban.
"""
import types

import pytest

from codai.api import runpod_client as rc


class _R:
    def __init__(self, code, body):
        self.status_code = code
        self._b = body
        self.text = str(body)
        self.headers = {}

    def json(self):
        return self._b


@pytest.fixture(autouse=True)
def _clear():
    rc._GQL_BACKOFF_UNTIL[0] = 0.0
    yield
    rc._GQL_BACKOFF_UNTIL[0] = 0.0


def _client(monkeypatch, replies):
    import requests
    calls = []

    def post(*a, **k):
        calls.append(1)
        return replies.pop(0)
    monkeypatch.setattr(requests, "post", post)
    c = rc.RunpodClient.__new__(rc.RunpodClient)
    c._api_key = "k" * 20
    c._gql_base = "https://api.runpod.io/graphql"
    return c, calls


def test_a_429_stops_further_calls_until_retry_after(monkeypatch):
    c, calls = _client(monkeypatch, [_R(429, {"error": "Too Many Requests",
                                              "retryAfter": 11871})])
    with pytest.raises(rc.RunpodTransient):
        c._gql("query { myself { id } }")
    with pytest.raises(rc.RunpodTransient) as e:
        c._gql("query { myself { id } }")
    assert len(calls) == 1, "called RunPod again inside the ban"
    assert "rate-limited" in str(e.value)
    assert 11000 < rc._GQL_BACKOFF_UNTIL[0] - __import__("time").time() <= 11871


def test_the_ban_is_capped(monkeypatch):
    c, _ = _client(monkeypatch, [_R(429, {"retryAfter": 10 ** 9})])
    with pytest.raises(rc.RunpodTransient):
        c._gql("q")
    left = rc._GQL_BACKOFF_UNTIL[0] - __import__("time").time()
    assert left <= rc._GQL_BACKOFF_MAX_S


def test_calls_resume_after_the_window(monkeypatch):
    c, calls = _client(monkeypatch, [_R(200, {"data": {"myself": {"id": "x"}}})])
    rc._GQL_BACKOFF_UNTIL[0] = 1.0          # long past
    assert c._gql("q") == {"myself": {"id": "x"}}
    assert len(calls) == 1
