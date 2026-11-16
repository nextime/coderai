"""A direct-TCP pod must be rented where a public ip can be asked for.

GraphQL now rejects `supportPublicIp` on PodFindAndDeployOnDemandInput. A
direct-TCP pod without a public ip cannot be reached by anything, so every
such rental was born unreachable and billed until the boot timeout gave up:
four trainer rentals in a row at ~$0.45 each, and a vision pool re-renting in
a circle every 17 seconds for the same reason.

coderai was already PRINTING this ("renting without it, so ... may be
unreachable") and renting anyway, on the reasoning that a pod without the hint
beats no pod. For direct TCP that is backwards -- the rental's only possible
outcome is paying for a boot that cannot succeed, and refusing is free.
"""

import inspect

import pytest

from codai.api.runpod_client import (PUBLIC_IP_FIELD, RunpodClient,  # noqa
                                     RunpodError)


class Client(RunpodClient):
    """A client with the network mocked out, to see which path a create takes."""

    def __init__(self):
        self.rest_calls = []
        self.gql_calls = []
        self._cfg = type("C", (), {"api_key": "k", "rest_base": ""})()

    def _rest(self, method, path, body=None, timeout=0.0):
        self.rest_calls.append((method, path, body))
        return {"id": "pod-rest"}

    def _gql(self, q, variables=None, timeout=0.0):
        self.gql_calls.append((q, variables))
        # Keyed on the mutation the query actually uses: a spot rental is
        # podRentInterruptable, and answering with the on-demand key makes the
        # client report "no pod id" -- a fake that fails for the wrong reason.
        name = ("podRentInterruptable" if "podRentInterruptable" in q
                else "podFindAndDeployOnDemand")
        return {name: {"id": "pod-gql"}}


def _create(c, **kw):
    base = dict(name="n", image="i", gpu_type_id="A40", port=8000)
    base.update(kw)
    return c.create_pod(**base)


# ------------------------------------------------------------- the routing --

def test_a_direct_tcp_pod_is_created_through_rest():
    c = Client()
    assert _create(c, direct_tcp=True) == "pod-rest"
    assert c.gql_calls == [], "a direct-TCP pod must not be rented on GraphQL"
    assert c.rest_calls and c.rest_calls[0][1] == "/pods"


def test_the_rest_body_asks_for_a_public_ip_and_a_tcp_port():
    """The two things the GraphQL path could not express."""
    c = Client()
    _create(c, direct_tcp=True)
    body = c.rest_calls[0][2]
    assert body[PUBLIC_IP_FIELD] is True
    assert body["ports"] == ["8000/tcp"]


def test_a_proxied_pod_still_uses_graphql():
    """Only direct TCP needs the public ip; a proxied pod must not be moved
    off a path that works, nor spend REST's separate quota."""
    c = Client()
    assert _create(c, direct_tcp=False) == "pod-gql"
    assert c.rest_calls == []


def test_the_source_routes_on_direct_tcp():
    src = inspect.getsource(RunpodClient.create_pod)
    assert "(direct_tcp and not is_spot)" in src


# ------------------------------------------------------------- the refusal --

def test_graphql_still_rents_a_spot_pod_without_the_field(monkeypatch):
    """NOT a refusal. I made this refuse and it was wrong: the field is a hint
    and most hosts publish a public ip anyway. What REST buys for non-spot
    pods is better odds, not the difference between working and not."""
    src = inspect.getsource(RunpodClient.create_pod)
    i = src.index("rejects {PUBLIC_IP_FIELD!r}")
    block = src[max(0, i - 1200):i + 400]
    assert "raise RunpodError(" not in block
    assert "hint, not" in block


def test_a_proxied_pod_keeps_the_lenient_behaviour():
    """It reaches its port through RunPod's proxy, so a missing public ip is
    genuinely not fatal: that rental must still go through."""
    src = inspect.getsource(RunpodClient.create_pod)
    assert "needs no public ip" in src


def test_a_direct_tcp_spot_pod_stays_on_graphql():
    """REST v2 has no interruptible bidding, so routing a spot request there
    would refuse it outright and never reach the GraphQL refusal that explains
    why a reachable direct-TCP spot pod cannot be rented at all right now.
    REST is taken only where REST can serve the request."""
    c = Client()
    assert _create(c, direct_tcp=True, is_spot=True,
                   bid_per_gpu=0.2) == "pod-gql"
    assert c.rest_calls == []


def test_a_spot_rental_still_asks_for_the_public_ip_on_graphql():
    """It is refused by today's schema, but asking is what makes the refusal
    detectable -- and a schema that accepts it rents a reachable pod."""
    c = Client()
    _create(c, direct_tcp=True, is_spot=True, bid_per_gpu=0.2)
    sent = c.gql_calls[0][1]["input"]
    assert sent[PUBLIC_IP_FIELD] is True
