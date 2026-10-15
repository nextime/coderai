"""A control plane that could not be ASKED is not a pod that failed.

Observed live 2026-10-15. A request for an embedding model rented a pod, which
began pulling a freshly pushed 9.65 GB image — a five-minute cold pull, the
normal case for the first pod after an image release. Two minutes in, RunPod's
API answered HTTP 503 once. Three things then went wrong in a row:

  1. the boot wait raised, and the handler reads any exception as "this pod
     failed to boot" — so a healthy pod, mid-pull, was condemned;
  2. the teardown went out through the SAME unavailable API, failed, and was
     swallowed by `except Exception: pass`;
  3. the stale-pod reaper could never clean up after it, because pod age came
     from `runtime.uptimeInSeconds` — the CURRENT run's uptime, which resets on
     every restart. The pod reported 234s while it was 11 minutes old and could
     never cross the 20-minute grace.

Net: an L40S billing at $1.09/hr that nothing tracked and no sweep would ever
have found, and a 502 to the caller whose pod was still coming up.
"""

import time
import types

import pytest

from codai.api import runpod_worker
from codai.api.runpod_client import RunpodClient, RunpodError, RunpodTransient


# ------------------------------------------------------- transient vs final #
class _Resp:
    def __init__(self, status, text='{"message":"nope"}'):
        self.status_code, self.text = status, text

    def json(self):
        import json
        return json.loads(self.text)


def _client(monkeypatch, responses):
    """A client whose transport returns the given responses in order."""
    cfg = types.SimpleNamespace(api_key="k", api_base="", rest_base="")
    c = RunpodClient(cfg)
    seq = list(responses)

    def _post(*a, **k):
        r = seq.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    import requests
    monkeypatch.setattr(requests, "post", _post)
    return c


@pytest.mark.parametrize("status", [500, 502, 503, 504, 408, 429])
def test_a_server_side_status_is_transient(monkeypatch, status):
    c = _client(monkeypatch, [_Resp(status)])
    with pytest.raises(RunpodTransient):
        c.get_pod("p1")


@pytest.mark.parametrize("status", [400, 403, 404, 422])
def test_a_client_side_status_is_final(monkeypatch, status):
    """Retrying a bad id or a malformed query only burns the deadline."""
    c = _client(monkeypatch, [_Resp(status)])
    with pytest.raises(RunpodError) as e:
        c.get_pod("p1")
    assert not isinstance(e.value, RunpodTransient)


def test_an_unreachable_api_is_transient(monkeypatch):
    import requests
    c = _client(monkeypatch, [requests.ConnectionError("no route")])
    with pytest.raises(RunpodTransient):
        c.get_pod("p1")


def test_a_rejected_key_is_not_transient(monkeypatch):
    """401 will say 401 again in five seconds; retrying hides a real problem."""
    c = _client(monkeypatch, [_Resp(401)])
    with pytest.raises(RunpodError) as e:
        c.get_pod("p1")
    assert not isinstance(e.value, RunpodTransient)


def test_transient_is_still_a_runpod_error():
    """Every existing `except RunpodError` must keep catching it."""
    assert issubclass(RunpodTransient, RunpodError)


# ------------------------------------------------------------- the boot wait #
def _ready_pod():
    return {"id": "p1", "status": "RUNNING", "uptime_s": 10, "ready": True,
            "ports": [{"ip": "1.2.3.4", "isIpPublic": True, "privatePort": 8000,
                       "publicPort": 22031, "type": "http"}]}


def test_the_boot_wait_rides_out_a_transient_api_failure(monkeypatch):
    """The pod was fine; only the question went unanswered."""
    cfg = types.SimpleNamespace(api_key="k", api_base="", rest_base="")
    c = RunpodClient(cfg)
    calls = []

    def _get_pod(pod_id):
        calls.append(pod_id)
        if len(calls) <= 3:
            raise RunpodTransient("RunPod API HTTP 503")
        return _ready_pod()

    monkeypatch.setattr(c, "get_pod", _get_pod)
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    url = c.wait_ready("p1", 8000, ready_timeout=60, poll_every=0.01)
    assert url
    assert len(calls) == 4, "it must keep asking, not give up on the first 503"


def test_a_wait_that_never_got_an_answer_says_so(monkeypatch):
    """The caller tears down on a boot failure, so the distinction decides
    whether a healthy pod is destroyed."""
    cfg = types.SimpleNamespace(api_key="k", api_base="", rest_base="")
    c = RunpodClient(cfg)
    monkeypatch.setattr(c, "get_pod",
                        lambda *_: (_ for _ in ()).throw(RunpodTransient("503")))
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    with pytest.raises(RunpodTransient) as e:
        c.wait_ready("p1", 8000, ready_timeout=0.05, poll_every=0.01)
    assert "not known-bad" in str(e.value)


def test_a_pod_that_never_opens_its_port_is_still_a_plain_failure(monkeypatch):
    """When the API ANSWERS and the pod simply never comes up, that is the
    pod's fault and the existing behaviour must not change."""
    cfg = types.SimpleNamespace(api_key="k", api_base="", rest_base="")
    c = RunpodClient(cfg)
    monkeypatch.setattr(c, "get_pod", lambda *_: {
        "id": "p1", "status": "RUNNING", "uptime_s": 5, "ready": False, "ports": []})
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    with pytest.raises(RunpodError) as e:
        c.wait_ready("p1", 8000, ready_timeout=0.05, poll_every=0.01)
    assert not isinstance(e.value, RunpodTransient)


# -------------------------------------------------------------- the teardown #
def test_terminate_retries_while_the_api_is_down(monkeypatch):
    """The teardown runs through the API that just failed, so the attempt that
    matters most is the one most likely to fail."""
    cfg = types.SimpleNamespace(api_key="k", api_base="", rest_base="")
    c = RunpodClient(cfg)
    tries = []

    def _gql(q, v=None, timeout=30.0):
        tries.append(1)
        if len(tries) < 3:
            raise RunpodTransient("RunPod API HTTP 503")
        return {}

    monkeypatch.setattr(c, "_gql", _gql)
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    c.terminate_pod("p1")
    assert len(tries) == 3


def test_terminate_eventually_raises_rather_than_pretending(monkeypatch):
    """A caller that swallows this is choosing to leak a pod; it must at least
    be given the chance to say so."""
    cfg = types.SimpleNamespace(api_key="k", api_base="", rest_base="")
    c = RunpodClient(cfg)
    monkeypatch.setattr(c, "_gql",
                        lambda *a, **k: (_ for _ in ()).throw(RunpodTransient("503")))
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    with pytest.raises(RunpodTransient):
        c.terminate_pod("p1", attempts=2)


def test_a_final_error_is_not_retried(monkeypatch):
    cfg = types.SimpleNamespace(api_key="k", api_base="", rest_base="")
    c = RunpodClient(cfg)
    tries = []

    def _gql(*a, **k):
        tries.append(1)
        raise RunpodError("RunPod API HTTP 404: no such pod")

    monkeypatch.setattr(c, "_gql", _gql)
    with pytest.raises(RunpodError):
        c.terminate_pod("p1", attempts=4)
    assert len(tries) == 1


# ---------------------------------------------------------------- the age #
RENTED = "Rented by User: Thu Oct 15 2026 19:36:58 GMT+0000 (Coordinated Universal Time)"


def test_age_comes_from_the_rental_not_the_current_run():
    """The live failure in one assertion: 11 minutes old, uptime 234s."""
    import datetime
    rented = (datetime.datetime.now(datetime.timezone.utc)
              - datetime.timedelta(minutes=11))
    stamp = rented.strftime("Rented by User: %a %b %d %Y %H:%M:%S GMT+0000 (x)")
    age = runpod_worker._pod_age_seconds({"last_change": stamp, "uptime_s": 234})
    assert age is not None
    assert age > 600, "a restart-looping pod must not look 234 seconds old"


def test_a_restart_looping_pod_crosses_the_reap_grace():
    """The whole point: the pod that most needs reaping was the one pod the
    reaper could never see."""
    import datetime
    rented = (datetime.datetime.now(datetime.timezone.utc)
              - datetime.timedelta(seconds=runpod_worker.REAP_GRACE_SECONDS + 300))
    stamp = rented.strftime("Rented by User: %a %b %d %Y %H:%M:%S GMT+0000 (x)")
    age = runpod_worker._pod_age_seconds({"last_change": stamp, "uptime_s": 60})
    assert age >= runpod_worker.REAP_GRACE_SECONDS


def test_uptime_still_works_when_there_is_no_rental_stamp():
    assert runpod_worker._pod_age_seconds({"uptime_s": 900}) == 900


def test_an_unparsable_stamp_falls_back_rather_than_guessing():
    age = runpod_worker._pod_age_seconds({"last_change": "Rented by User: sometime",
                                          "uptime_s": 120})
    assert age == 120


def test_no_age_at_all_is_still_none():
    """The reaper fails CLOSED on None, so this must not become 0."""
    assert runpod_worker._pod_age_seconds({"status": "RUNNING"}) is None


def test_a_status_change_that_is_not_a_rental_is_not_an_age():
    """'Stopped by user' dates the stop, not the pod."""
    assert runpod_worker._rental_age_seconds(
        {"last_change": "Stopped by User: Thu Oct 15 2026 19:36:58 GMT+0000 (x)"}) is None


def test_a_future_rental_stamp_is_refused():
    """A host clock behind RunPod's would otherwise make every pod ancient."""
    import datetime
    later = (datetime.datetime.now(datetime.timezone.utc)
             + datetime.timedelta(hours=2))
    stamp = later.strftime("Rented by User: %a %b %d %Y %H:%M:%S GMT+0000 (x)")
    assert runpod_worker._rental_age_seconds({"last_change": stamp}) is None


def test_the_real_timestamp_runpod_sends_parses():
    """Copied verbatim from the account listing, sentence and all."""
    assert runpod_worker._rental_age_seconds({"last_change": RENTED}) is not None
