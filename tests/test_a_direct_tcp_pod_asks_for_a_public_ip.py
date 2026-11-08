"""A direct-TCP pod must ask RunPod for a host that can publish a public IP.

The evidence this exists for, measured on 2026-11-08 against a vision pool
restricted to five native-FP8 cards:

    [runpod] pod d3cn0bw39o7cd6 boot FAILED: RunPod published 19123/http but
    NOT the declared port 8000, and the container has been running 132s --
    this host does not expose it.

Ten pods were rented in half an hour, nine were terminated, $1.21 was billed
for pods that could never be reached, and the client got
"no pod slot available ... within 1200.0s". 19123/http is RunPod's own agent
port, on a 100.64/10 carrier-grade-NAT address; the declared 8000/tcp was
never published at all, because those hosts cannot publish one and nothing in
the create request had asked them to. Pods that DID work -- three of them on
community hosts -- all showed (8000, 'tcp', isIpPublic=True).

So the create request now carries `supportPublicIp: true` whenever the pod is
reached directly, on both the GraphQL and the REST v2 path. A proxied pod must
NOT carry it: it is reached through proxy.runpod.net, needs no public address,
and asking would discard offers for nothing.
"""
import types

import pytest

from codai.api import runpod_client as rc


class _Cfg:
    api_key = "test-key-not-a-real-one"
    api_base = "https://api.runpod.io/graphql"


def _client(monkeypatch, *, gql=None, rest=None):
    """A client whose transport records what it was asked to send."""
    seen = {"gql": [], "rest": []}

    def _gql(self, query, variables=None, timeout=30.0):
        seen["gql"].append({"query": query, "variables": variables})
        if gql is not None:
            return gql(query, variables)
        name = ("podRentInterruptable" if "podRentInterruptable" in query
                else "podFindAndDeployOnDemand")
        return {name: {"id": "pod-1"}}

    def _rest(self, method, path, body=None, timeout=30.0):
        seen["rest"].append({"method": method, "path": path, "body": body})
        if rest is not None:
            return rest(method, path, body)
        return {"id": "pod-1"}

    monkeypatch.setattr(rc.RunpodClient, "_gql", _gql)
    monkeypatch.setattr(rc.RunpodClient, "_rest", _rest)
    monkeypatch.setattr(rc, "_gql_banned", lambda: False)
    return rc.RunpodClient(_Cfg()), seen


def _args(**over):
    base = dict(name="p", image="img:latest", gpu_type_id="NVIDIA L40S",
                port=8000, cloud_type="COMMUNITY")
    base.update(over)
    return base


def test_the_field_name_is_the_one_runpod_documents():
    # RunPod spells it this way on PodFindAndDeployOnDemandInput (GraphQL) and
    # on PodCreateInput (REST v2). A typo here is a silently ignored field.
    assert rc.PUBLIC_IP_FIELD == "supportPublicIp"


def test_a_direct_tcp_pod_asks_for_a_public_ip(monkeypatch):
    cli, seen = _client(monkeypatch)
    assert cli.create_pod(direct_tcp=True, **_args()) == "pod-1"
    sent = seen["gql"][0]["variables"]["input"]
    assert sent[rc.PUBLIC_IP_FIELD] is True
    assert sent["ports"] == "8000/tcp"


def test_a_proxied_pod_does_not(monkeypatch):
    # The proxy needs no public address, and asking for one would throw away
    # offers that serve a proxied pod perfectly well.
    cli, seen = _client(monkeypatch)
    cli.create_pod(direct_tcp=False, **_args())
    sent = seen["gql"][0]["variables"]["input"]
    assert rc.PUBLIC_IP_FIELD not in sent
    assert sent["ports"] == "8000/http"


def test_the_rest_path_asks_too(monkeypatch):
    # REST is taken for entrypoint/cmd overrides and whenever GraphQL is
    # rate-limited -- the flag cannot live on one path only.
    cli, seen = _client(monkeypatch)
    cli.create_pod(direct_tcp=True, entrypoint=["/bin/sh"], **_args())
    assert not seen["gql"]
    body = seen["rest"][0]["body"]
    assert body[rc.PUBLIC_IP_FIELD] is True
    assert body["ports"] == ["8000/tcp"]


def test_the_rest_path_leaves_it_out_when_proxied(monkeypatch):
    cli, seen = _client(monkeypatch)
    cli.create_pod(direct_tcp=False, entrypoint=["/bin/sh"], **_args())
    assert rc.PUBLIC_IP_FIELD not in seen["rest"][0]["body"]


def test_a_graphql_schema_that_refuses_the_field_still_rents(monkeypatch):
    """The spot input type could not be read from RunPod's published schema.

    If it has no such field, GraphQL answers "is not defined by type" -- and
    the pod must still be rented, without the hint, rather than the create
    failing outright.
    """
    calls = []

    def _gql(query, variables):
        calls.append(dict(variables["input"]))
        if len(calls) == 1:
            raise rc.RunpodError(
                'Field "supportPublicIp" is not defined by type '
                '"PodRentInterruptableInput".')
        return {"podRentInterruptable": {"id": "pod-spot"}}

    cli, seen = _client(monkeypatch, gql=_gql)
    assert cli.create_pod(direct_tcp=True, is_spot=True, bid_per_gpu=0.3,
                          **_args()) == "pod-spot"
    assert calls[0][rc.PUBLIC_IP_FIELD] is True   # asked once
    assert rc.PUBLIC_IP_FIELD not in calls[1]     # then dropped, and rented


def test_an_unrelated_graphql_error_is_not_swallowed(monkeypatch):
    # The retry exists for ONE thing. A capacity error must still surface, or
    # a create failure turns into a silent second attempt.
    def _gql(query, variables):
        raise rc.RunpodError("There are no longer any instances available")

    cli, _seen = _client(monkeypatch, gql=_gql)
    with pytest.raises(rc.RunpodError, match="no longer any instances"):
        cli.create_pod(direct_tcp=True, **_args())


def test_the_refusal_guard_needs_the_field_named(monkeypatch):
    # _field_refused_gql keys on the field NAME: an error about some other
    # field must not drop this one.
    assert rc._field_refused_gql(
        'Field "supportPublicIp" is not defined by type "X".',
        rc.PUBLIC_IP_FIELD)
    assert not rc._field_refused_gql(
        'Field "globalVolumeId" is not defined by type "X".',
        rc.PUBLIC_IP_FIELD)
    assert not rc._field_refused_gql("capacity", rc.PUBLIC_IP_FIELD)


def test_the_global_volume_guard_still_works(monkeypatch):
    # The two guards share a helper now; the older one must behave as before.
    calls = []

    def _gql(query, variables):
        calls.append(dict(variables["input"]))
        if len(calls) == 1:
            raise rc.RunpodError(
                f'Field "{rc.GLOBAL_VOLUME_FIELD}" is not defined by type "X".')
        return {"podFindAndDeployOnDemand": {"id": "pod-2"}}

    cli, _seen = _client(monkeypatch, gql=_gql)
    assert cli.create_pod(direct_tcp=True, global_volume_id="vol-9",
                          **_args()) == "pod-2"
    assert rc.GLOBAL_VOLUME_FIELD not in calls[1]
    # ...and the public-ip ask survives a global-volume refusal.
    assert calls[1][rc.PUBLIC_IP_FIELD] is True
