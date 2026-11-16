"""The REST create body was invented, and the API rejected all of it.

`_create_pod_rest` sent `image`, `gpu: {id, count}`, `cloud`, `disk`,
`registry` and a `mounts` tree. RunPod's published PodCreateInput names are
`imageName`, `gpuTypeIds`, `gpuCount`, `cloudType`, `containerDiskInGb`,
`containerRegistryAuthId`, `networkVolumeId` and `volumeInGb`. Measured: HTTP
400 on a probe with the old body, and in production a 422 "additional
properties".

It went unseen because this path was taken only for entrypoint/cmd overrides
and while GraphQL was rate-limited -- a default that worked hid a fallback
that could not. Moving direct-TCP rentals here is what exposed it.

Field names are asserted literally on purpose: this is a wire contract with
someone else's validator, and a rename is exactly the failure being prevented.
"""

import json

import pytest

from codai.api import runpod_client as rc


class Client(rc.RunpodClient):
    def __init__(self):
        self.body = None
        self.err = None
        self._cfg = type("C", (), {"api_key": "k", "rest_base": ""})()

    def _rest(self, method, path, body=None, timeout=0.0):
        self.body = body
        assert method == "POST" and path == "/pods"
        if self.err:
            raise self.err
        return {"id": "pod-1"}


def _body(**over):
    """A create that takes the REST path, the way production reaches it.

    `entrypoint` is what routes here: GraphQL has no entrypoint override, so
    this is one of the two real REST callers (the other is a GraphQL
    rate-limit). direct_tcp alone does NOT route here -- routing it here broke
    every new rental on the install and was reverted.
    """
    c = Client()
    args = dict(name="n", image="ghcr.io/x/y:1", gpu_type_id="NVIDIA A40",
                port=8000, direct_tcp=True, entrypoint=["/bin/sh"])
    args.update(over)
    c.create_pod(**args)
    return c.body


# ------------------------------------------------------ the documented names --

@pytest.mark.parametrize("field", [
    "name", "imageName", "gpuTypeIds", "gpuCount", "cloudType",
    "containerDiskInGb", "ports", "env",
])
def test_the_required_shape_uses_the_documented_name(field):
    assert field in _body()


@pytest.mark.parametrize("gone", ["image", "gpu", "cloud", "disk", "registry",
                                  "mounts", "cmd"])
def test_no_invented_name_survives(gone):
    assert gone not in _body()


def test_the_image_and_card_land_in_the_right_fields():
    b = _body()
    assert b["imageName"] == "ghcr.io/x/y:1"
    assert b["gpuTypeIds"] == ["NVIDIA A40"]   # an ARRAY, per the schema
    assert b["gpuCount"] == 1


def test_ports_is_an_array_of_strings():
    """The schema declares items of type string; its own `default` is written
    as one comma-joined string, which contradicts the declared type. The
    declared type is what this sends.

    The proxied case needs `entrypoint` to stay on REST at all: a proxied pod
    is otherwise rented on GraphQL, which is the point of the routing change.
    """
    assert _body(direct_tcp=True)["ports"] == ["8000/tcp"]
    assert _body(direct_tcp=False)["ports"] == ["8000/http"]


def test_the_public_ip_field_is_sent_for_a_direct_tcp_pod():
    """It DOES exist on this endpoint, contrary to the report that it does not:
    documented as needed on COMMUNITY, implied on SECURE."""
    assert _body(direct_tcp=True)[rc.PUBLIC_IP_FIELD] is True
    assert rc.PUBLIC_IP_FIELD not in _body(direct_tcp=False)


def test_a_start_command_is_an_array_not_a_string():
    """A bare string would be accepted as a single argv element and the
    container would try to exec the whole command line as one program."""
    b = _body(docker_args="python3 /opt/trainer/runner.py --flag v")
    assert b["dockerStartCmd"] == ["python3", "/opt/trainer/runner.py",
                                   "--flag", "v"]


def test_an_explicit_start_cmd_and_entrypoint_use_the_docker_names():
    b = _body(entrypoint=["/bin/sh", "-c"], start_cmd=["echo", "hi"])
    assert b["dockerEntrypoint"] == ["/bin/sh", "-c"]
    assert b["dockerStartCmd"] == ["echo", "hi"]


def test_volumes_and_registry_use_the_documented_names():
    b = _body(volume_gb=20, network_volume_id="vol-1",
              registry_auth_id="auth-1", data_center_id="EU-SE-1")
    assert b["volumeInGb"] == 20 and b["volumeMountPath"] == "/workspace"
    assert b["networkVolumeId"] == "vol-1"
    assert b["containerRegistryAuthId"] == "auth-1"
    assert b["dataCenterIds"] == ["EU-SE-1"]


def test_the_body_is_json_serialisable():
    """It is sent as JSON; a stray non-serialisable value fails at the wire."""
    json.dumps(_body(env={"A": 1}, volume_gb=10))


def test_env_values_are_strings():
    assert _body(env={"N": 5})["env"] == {"N": "5"}


# --------------------------------------------- the global-volume guard ------

def test_a_refused_global_volume_is_dropped_and_the_pod_still_rents():
    c = Client()
    calls = []
    real = c._rest

    def _rest(method, path, body=None, timeout=0.0):
        calls.append(dict(body or {}))
        if len(calls) == 1:
            raise rc.RunpodError(
                f'Request validation failed: additional properties '
                f'"{rc.GLOBAL_VOLUME_FIELD}" not allowed')
        return {"id": "pod-1"}

    c._rest = _rest
    assert c.create_pod(name="n", image="i", gpu_type_id="g", port=8000,
                        direct_tcp=True, entrypoint=["/bin/sh"],
                        global_volume_id="gv-1") == "pod-1"
    assert rc.GLOBAL_VOLUME_FIELD in calls[0]
    assert rc.GLOBAL_VOLUME_FIELD not in calls[1]


def test_an_unrelated_rest_failure_is_not_retried_as_a_volume_problem():
    """Keying on the bare word "global" would swallow a capacity error that
    merely mentions it, turning a real failure into a silent retry."""
    c = Client()
    c.err = rc.RunpodError("no instances available in any global region")
    with pytest.raises(rc.RunpodError, match="no instances"):
        c.create_pod(name="n", image="i", gpu_type_id="g", port=8000,
                     direct_tcp=True, entrypoint=["/bin/sh"],
                     global_volume_id="gv-1")


@pytest.mark.parametrize("msg,hit", [
    ('additional properties "%s" not allowed' % rc.GLOBAL_VOLUME_FIELD, True),
    ('%s is not a recognised field' % rc.GLOBAL_VOLUME_FIELD, True),
    ("no capacity in this global region", False),
    ("rate limited", False),
])
def test_the_volume_refusal_keys_on_the_field_name(msg, hit):
    """Built from the constant, not a hand-typed name: the field is
    `globalNetworkVolumeId`, and a test that spells it `globalVolumeId` passes
    or fails for reasons unrelated to the code."""
    assert rc._global_field_refused_rest(msg) is hit
