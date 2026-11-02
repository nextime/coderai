# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

""""Still booting" and "can never be reached" are not the same wait.

Measured on a production account, 2026-11-02. An RTX PRO 6000 pod was rented
for the OCR pool at $1.69/hr and reported "booting" for 29 minutes, then was
terminated for not exposing its port, billed $0.846 and retried -- which rented
another one that did the same thing. RunPod's own answer for it, at 209s of
container uptime:

    runtime ports: [{"ip": "100.65.33.207", "private": 19123,
                     "public": 60595, "type": "http"}]

That is RunPod's internal agent port, on 100.64.0.0/10 -- carrier-grade NAT.
The pod declared `8000/tcp` and RunPod never published it AT ALL. Meanwhile
three pods on other hosts (A40, L40, RTX 6000 Ada -- one of them COMMUNITY too,
so this is the host and not the cloud type) each had 8000/tcp on a globally
routable address and accepted connections:

    8000/tcp -> 12640 107.150.186.62 ROUTABLE  TCP-ACCEPTS

So there was nothing to connect to and no timeout could have changed it. Two
defects made that invisible:

* the boot loop printed "image pull in progress" every 60s for ANY status that
  was not EXITED -- including a pod whose own `uptime` proved the pull had
  finished eight minutes earlier. The operator was told the wrong phase was
  slow, which is worse than being told nothing;
* `_rest_boot_state` cleared `ports` whenever the pod was not ready, throwing
  away the one piece of evidence that separates the two cases.

Fixing the timeout would have made it worse: a longer wait buys more of a wait
that cannot succeed. So the states are classified instead, and the one that
cannot recover fails in seconds.
"""

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codai.api import runpod_client as rc
from codai.api.runpod_client import RunpodClient, RunpodError, classify_boot


#: Exactly what RunPod answered for the pod that cost $0.846 twice.
_PRO6000 = {"status": "RUNNING", "uptime_s": 209.0, "published_ports": [
    {"ip": "100.65.33.207", "privatePort": 19123, "publicPort": 60595,
     "type": "http"}]}

#: ...and for one that worked, on the same COMMUNITY cloud.
_WORKING = {"status": "RUNNING", "uptime_s": 2296.0, "published_ports": [
    {"ip": "107.150.186.62", "privatePort": 8000, "publicPort": 12640,
     "type": "tcp", "isIpPublic": True},
    {"ip": "100.65.11.42", "privatePort": 19123, "publicPort": 60191,
     "type": "http", "isIpPublic": True}]}


# ---------------------------------------------------------------------------
# routability
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ip", ["107.150.186.62", "81.27.69.179", "194.68.245.200"])
def test_a_public_address_is_routable(ip):
    assert rc._is_routable(ip) is True


@pytest.mark.parametrize("ip", [
    "100.65.33.207",    # carrier-grade NAT: the one that actually bit
    "100.65.11.42",
    "10.0.0.1", "192.168.1.1", "172.16.0.1",
    "127.0.0.1", "169.254.1.1", "", None, "not-an-ip",
])
def test_an_unreachable_address_is_not_routable(ip):
    assert rc._is_routable(ip) is False


# ---------------------------------------------------------------------------
# the three states
# ---------------------------------------------------------------------------

def test_the_pro6000_host_is_called_unreachable():
    verdict, why = classify_boot(_PRO6000, 8000)
    assert verdict == "unreachable"
    assert "8000" in why and "19123/http" in why


def test_a_working_host_is_not_called_unreachable():
    """The same check must not condemn the pods that do work -- one of them on
    the same COMMUNITY cloud as the pod that did not."""
    verdict, _why = classify_boot(_WORKING, 8000)
    assert verdict == "starting"


def test_a_cold_machine_pulling_its_image_is_left_alone():
    """uptime 0 is the container not started: a multi-GB pull is minutes of
    exactly this, and it is the normal case after every image release."""
    verdict, why = classify_boot(
        {"status": "RUNNING", "uptime_s": 0.0, "published_ports": []}, 8000)
    assert verdict == "pulling"
    assert "pull" in why


def test_the_grace_covers_a_mapping_that_arrives_just_after_the_container():
    """A host that is going to publish the port does it at container start, but
    not necessarily in the same instant -- and condemning a pod for being a few
    seconds behind would throw away good machines."""
    early = dict(_PRO6000, uptime_s=5.0)
    assert classify_boot(early, 8000)[0] == "pulling"
    late = dict(_PRO6000, uptime_s=rc.PORT_PUBLISH_GRACE_S + 1)
    assert classify_boot(late, 8000)[0] == "unreachable"


def test_no_mapping_at_all_past_the_grace_is_unreachable():
    verdict, why = classify_boot(
        {"status": "RUNNING", "uptime_s": 600.0, "published_ports": []}, 8000)
    assert verdict == "unreachable"
    assert "no port mapping at all" in why


def test_the_declared_port_on_a_cgnat_address_is_unreachable():
    """The other shape of the same failure: the port IS published, just
    somewhere nothing can reach. A direct-TCP pod there is a dead end."""
    verdict, why = classify_boot(
        {"status": "RUNNING", "uptime_s": 600.0, "published_ports": [
            {"ip": "100.65.4.4", "privatePort": 8000, "publicPort": 41000,
             "type": "tcp"}]}, 8000)
    assert verdict == "unreachable"
    assert "routable" in why


def test_a_proxied_pod_does_not_need_a_routable_address():
    """A non-direct-TCP pod is reached through RunPod's own proxy, so the
    mapping's address is none of our business -- condemning it for being
    private would break every proxied pod."""
    info = {"status": "RUNNING", "uptime_s": 600.0, "published_ports": [
        {"ip": "100.65.4.4", "privatePort": 8000, "publicPort": 41000,
         "type": "http"}]}
    assert classify_boot(info, 8000, direct_tcp=False)[0] == "starting"
    assert classify_boot(info, 8000, direct_tcp=True)[0] == "unreachable"


def test_a_ready_pod_is_ready():
    assert classify_boot({"ready": True}, 8000) == ("ready", "")


def test_an_unreadable_uptime_is_treated_as_not_started():
    """Never condemn a pod on a field that could not be parsed."""
    for bad in (None, "", "soon", object()):
        assert classify_boot(
            {"status": "RUNNING", "uptime_s": bad, "published_ports": []},
            8000)[0] == "pulling"


def test_it_falls_back_to_ports_when_published_ports_is_absent():
    """The GraphQL path and `get_pod` carry `ports` and no `published_ports`."""
    verdict, _ = classify_boot(
        {"status": "RUNNING", "uptime_s": 600.0, "ports": [
            {"ip": "1.2.3.4", "privatePort": 8000, "publicPort": 9}]}, 8000)
    assert verdict == "starting"


# ---------------------------------------------------------------------------
# the evidence has to survive the readiness check
# ---------------------------------------------------------------------------

def test_the_boot_state_keeps_the_published_ports_even_when_not_ready():
    """`ports` is still cleared, so no caller can build a URL to a port that
    does not accept yet -- but clearing it was also destroying the only
    evidence that tells the two waits apart."""
    c = RunpodClient.__new__(RunpodClient)
    c._rest = lambda *a, **k: {
        "id": "p1", "status": "RUNNING", "gpu": {"id": "NVIDIA A40", "count": 1},
        "runtime": {"uptime": 300, "ports": [
            {"ip": "100.65.1.1", "private": 19123, "public": 5000,
             "type": "http"}]}}
    row = c._rest_boot_state("p1", 8000)
    assert row["ready"] is False
    assert row["ports"] == [], "unready pods must not hand out a URL"
    assert row["published_ports"], "but the evidence must survive"
    assert classify_boot(row, 8000)[0] == "unreachable"


# ---------------------------------------------------------------------------
# the boot loop acts on it
# ---------------------------------------------------------------------------

def _waiting_client(rows):
    c = RunpodClient.__new__(RunpodClient)
    seq = list(rows)
    c._rest_boot_state = lambda pod_id, port: (
        seq.pop(0) if len(seq) > 1 else seq[0])
    c.get_pod = lambda pod_id: seq[0]
    return c


def test_wait_ready_gives_up_at_once_on_an_unreachable_host(monkeypatch):
    """The whole point: seconds, not the full boot_timeout_s. The real run cost
    1800s and $0.846 per attempt, three attempts per provision, in a loop."""
    monkeypatch.setattr(rc.time, "sleep", lambda s: None)
    c = _waiting_client([dict(_PRO6000)])
    t0 = time.time()
    with pytest.raises(RunpodError) as exc:
        c.wait_ready("p1", 8000, ready_timeout=1800.0, direct_tcp=True,
                     poll_every=0.01)
    assert time.time() - t0 < 5.0, "it must not wait out the budget"
    msg = str(exc.value)
    assert "cannot be reached" in msg
    assert "MACHINE" in msg, "it has to say what to do about it"


def test_wait_ready_keeps_waiting_while_the_image_pulls(monkeypatch):
    """A pull must still be given the full budget, or every cold machine is
    thrown away."""
    monkeypatch.setattr(rc.time, "sleep", lambda s: None)
    pulling = {"status": "RUNNING", "uptime_s": 0.0, "published_ports": []}
    c = _waiting_client([pulling])
    with pytest.raises(RunpodError) as exc:
        c.wait_ready("p1", 8000, ready_timeout=0.2, direct_tcp=True,
                     poll_every=0.01)
    assert "did not expose port" in str(exc.value), str(exc.value)


def test_wait_ready_returns_the_url_once_the_port_accepts(monkeypatch):
    monkeypatch.setattr(rc.time, "sleep", lambda s: None)
    ready = dict(_WORKING, ready=True, ports=_WORKING["published_ports"])
    c = _waiting_client([ready])
    url = c.wait_ready("p1", 8000, ready_timeout=30.0, direct_tcp=True,
                       tls=True, poll_every=0.01)
    assert "107.150.186.62" in url and "12640" in url


def test_the_timeout_message_no_longer_only_blames_the_image_pull(monkeypatch):
    """It told the operator to raise boot_timeout_s, which for this failure
    buys a longer wait for something that cannot happen."""
    monkeypatch.setattr(rc.time, "sleep", lambda s: None)
    pulling = {"status": "RUNNING", "uptime_s": 0.0, "published_ports": []}
    c = _waiting_client([pulling])
    with pytest.raises(RunpodError) as exc:
        c.wait_ready("p1", 8000, ready_timeout=0.2, direct_tcp=True,
                     poll_every=0.01)
    assert "does NOT help a host that never publishes the port" in str(exc.value)


def test_an_exited_container_is_still_reported_as_exited(monkeypatch):
    """The pre-existing fast path for a process that died on boot must not be
    swallowed by the new classification."""
    monkeypatch.setattr(rc.time, "sleep", lambda s: None)
    dead = {"status": "EXITED", "uptime_s": 0.0, "published_ports": [],
            "last_change": "OOM"}
    c = _waiting_client([dead])
    monkeypatch.setattr(rc.time, "time",
                        lambda _c=[0.0]: (_c.__setitem__(0, _c[0] + 20.0)
                                          or _c[0]))
    with pytest.raises(RunpodError) as exc:
        c.wait_ready("p1", 8000, ready_timeout=600.0, direct_tcp=True,
                     poll_every=0.01)
    assert "EXITED" in str(exc.value)
