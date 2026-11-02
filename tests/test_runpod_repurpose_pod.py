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

"""A card you are already holding beats the right image.

Measured on a production account, 2026-11-02 17:20: an OCR pool logged "no
candidate GPU could be deployed (capacity)" 301 times in one hour trying for an
A40 or an L40, while an A40 belonging to another pool on the same account sat
at 0.0% GPU with 45.2 of its 48.3 GB merely reserved. The ordinary way to move
it -- give it back, rent the one you want -- joins the queue that has already
been lost 301 times, because RunPod documents `stop` as "releasing GPU/CPU
compute while keeping its disk". `restart` is documented as restarting a
RUNNING pod's container IN PLACE, and that is the difference the whole feature
rests on.

Three things had to exist for that to be survivable, and each is pinned here:

* **parking.** A pod its pool still holds does not go quietly while its
  container restarts. The first failed request quarantines it for
  QUARANTINE_S, and QUARANTINE_STRIKES probes later the pool TERMINATES it:
  45s x 3 is ~135 seconds, against a 6-14 minute boot. "Just press restart"
  ends with the pool killing the card the restart existed to keep. A parked pod
  is in no pool (so nothing routes to it and no adoption takes it) and still
  carries this process's pid (so no reaper touches it through the boot).
* **the environment.** A pod keeps what it BOOTED with -- served model, bearer
  token, admission gate. An image swap that leaves the old env boots the new
  image configured as the old pool: the wrong model, and a bearer the new pool
  does not hold, which presents as a pod that 401s everything for the rest of
  its paid life.
* **verification.** RunPod does not promise in writing that the machine
  survives. So the pod id and the card are read before and after, and a pod
  that moved is NOT handed to a pool -- it is left parked and said out loud,
  because a half-done move discovered an hour later is worse than a refusal.

And a parked pod is rented, billing, and in no pool, so it is reported on the
page as its own state. Every expensive surprise in this file's history has the
same shape: a pod that bills while being invisible.
"""

import inspect
import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codai.api import runpod_worker as rw
from codai.api.runpod_client import RunpodClient, RunpodError


# ---------------------------------------------------------------------------
# scaffolding
# ---------------------------------------------------------------------------

class _Mcfg:
    port = 8000
    pool = ""
    engine = "coderai"


def _pool(key="digesta-qwen38"):
    p = rw.RunpodPodPool.__new__(rw.RunpodPodPool)
    p._cv = threading.Condition()
    p.model_key = key
    p.pods = []
    p.mcfg = _Mcfg()
    p.api_key = "pool-key"
    p.health_path = "/v1/models"
    p._bump_cards = lambda: None
    p._pod_keys = {}
    return p


def _handle(pod_id="pod1", inflight=0, hourly=0.59):
    return rw.PodHandle(pod_id=pod_id, url="https://old/v1", hourly_usd=hourly,
                        started_at=time.time() - 3600, gpu="A40",
                        api_key="pod-key", healthy=True, inflight=inflight,
                        gpu_count=1, vram_gb=48.0)


class _Registry:
    """A stand-in for the on-disk registry, with the same read-modify-write."""

    def __init__(self, data=None):
        self.data = dict(data or {})

    def update(self, fn):
        fn(self.data)
        return dict(self.data)


def _install(monkeypatch, pool=None, registry=None):
    reg = registry if registry is not None else _Registry()
    monkeypatch.setattr(rw, "_registry_update", reg.update)
    pools = {pool.model_key: pool} if pool is not None else {}
    monkeypatch.setattr(rw, "_pools", pools)
    monkeypatch.setattr(rw, "_load", lambda p: int(p.inflight))
    return reg


# ---------------------------------------------------------------------------
# the client calls
# ---------------------------------------------------------------------------

def _client(seen):
    c = RunpodClient.__new__(RunpodClient)

    def _rest(method, path, body=None, timeout=30.0, params=None):
        seen.append((method, path, body))
        return {"id": "pod1", "status": "RUNNING",
                "gpu": {"id": "NVIDIA A40", "count": 1}}
    c._rest = _rest
    return c


def test_edit_pod_patches_image_and_env():
    seen = []
    _client(seen).edit_pod("pod1", image="ghcr.io/x/ocr:latest",
                           env={"A": 1, "B": "two"})
    method, path, body = seen[0]
    assert (method, path) == ("PATCH", "/pods/pod1")
    assert body["image"] == "ghcr.io/x/ocr:latest"
    # Every value a string: the API's env is a map of strings, and an int here
    # is a 422 that reads like an auth problem.
    assert body["env"] == {"A": "1", "B": "two"}


def test_edit_pod_omits_what_it_was_not_given():
    """A PATCH leaves omitted fields untouched, so sending `env: {}` because
    the caller passed nothing would CLEAR the pod's environment -- `env`
    replaces rather than merges."""
    seen = []
    _client(seen).edit_pod("pod1", image="img")
    assert "env" not in seen[0][2]


def test_edit_pod_refuses_to_do_nothing():
    with pytest.raises(RunpodError):
        _client([]).edit_pod("pod1")
    with pytest.raises(RunpodError):
        _client([]).edit_pod("", image="img")


def test_pod_action_posts_the_action():
    seen = []
    _client(seen).pod_action("pod1", "restart")
    assert seen[0] == ("POST", "/pods/pod1/action", {"action": "restart"})


def test_pod_action_refuses_an_action_runpod_does_not_have():
    with pytest.raises(RunpodError):
        _client([]).pod_action("pod1", "reboot")


def test_the_repurpose_restarts_and_never_stops():
    """`stop` releases the compute. A repurpose that reached for it would hand
    back the very card it exists to keep, and the two words are one letter
    apart in the call."""
    src = inspect.getsource(rw.repurpose_pod)
    assert '"restart"' in src
    assert '"stop"' not in src


# ---------------------------------------------------------------------------
# parking
# ---------------------------------------------------------------------------

def test_park_drops_the_handle_so_nothing_routes_to_it(monkeypatch):
    pool = _pool()
    pool.pods.append(_handle())
    _install(monkeypatch, pool)
    out = rw.park_pod("pod1", reason="test")
    assert out["parked"] is True
    assert out["was_pool"] == "digesta-qwen38"
    assert pool.pods == []


def test_park_files_it_under_a_pool_no_pool_matches(monkeypatch):
    pool = _pool()
    pool.pods.append(_handle())
    reg = _install(monkeypatch, pool, _Registry({
        "pod1": {"pool": "digesta-qwen38", "pid": 1, "url": "https://old/v1",
                 "key": "pod-key", "image": "vllm:latest", "gpu": "A40",
                 "created": 100.0, "hourly_usd": 0.59}}))
    rw.park_pod("pod1")
    assert reg.data["pod1"]["pool"] == rw.PARKED_POOL
    assert rw.PARKED_POOL not in ("", None)
    assert reg.data["pod1"]["parked"]["from"] == "digesta-qwen38"
    assert reg.data["pod1"]["parked"]["image"] == "vllm:latest"


def test_a_parked_pod_keeps_our_pid_so_no_reaper_kills_it(monkeypatch):
    """The reaper's grace is 1200s and these images boot in 6-14 minutes, so
    the margin is thin even when it works. An entry owned by this process is
    counted unconditionally, which removes the race entirely."""
    pool = _pool()
    pool.pods.append(_handle())
    reg = _install(monkeypatch, pool)
    rw.park_pod("pod1")
    assert reg.data["pod1"]["pid"] == os.getpid()
    assert reg.data["pod1"]["proc"] == rw._PROC_ID
    assert rw._is_this_process(reg.data["pod1"]) is True
    assert "pod1" in rw.live_registered_pod_ids()


def test_no_pool_adopts_a_parked_pod(monkeypatch):
    """Adoption is keyed on the pool name, and the parked sentinel is not a
    pool key -- so a pod being re-imaged cannot be taken mid-flight by the
    pool it is on its way to, nor taken back by the one it left."""
    reg = _Registry({"pod1": {"pool": rw.PARKED_POOL, "pid": 999,
                              "url": "https://old/v1", "key": "k",
                              "image": "img", "gpu": "A40",
                              "parked": {"from": "digesta-qwen38"}}})
    monkeypatch.setattr(rw, "_registry_update", reg.update)
    monkeypatch.setattr(rw, "_is_this_process", lambda info: False)
    monkeypatch.setattr(rw, "_pod_health_ok", lambda *a, **k: True)
    for key in ("digesta-qwen38", "datalab-to/chandra-ocr-2", rw.PARKED_POOL):
        pid, url, _k, _r = rw.find_shared_pod(key, "/v1/models", "",
                                              image="img", gpus=["A40"])
        if key == rw.PARKED_POOL:
            continue            # nothing names itself that; not a real pool
        assert pid is None and not url, key


def test_park_refuses_a_pod_with_work_on_it(monkeypatch):
    pool = _pool()
    pool.pods.append(_handle(inflight=7))
    _install(monkeypatch, pool)
    with pytest.raises(RuntimeError) as exc:
        rw.park_pod("pod1")
    assert "7" in str(exc.value)
    assert pool.pods, "the pod stays in its pool when the park is refused"


def test_park_can_be_forced_past_in_flight_work(monkeypatch):
    pool = _pool()
    pool.pods.append(_handle(inflight=7))
    _install(monkeypatch, pool)
    out = rw.park_pod("pod1", force=True)
    assert out["inflight"] == 7 and out["parked"] is True


def test_park_bills_the_hours_to_the_pool_that_rented_them(monkeypatch):
    """The money belongs to the work the pod did, not to whoever gets it next.
    A cost recorded nowhere is the one that turns up on the invoice."""
    pool = _pool()
    pool.pods.append(_handle())
    _install(monkeypatch, pool)
    spends = []
    import codai.api.runpod_ledger as ledger
    monkeypatch.setattr(ledger, "record_spend",
                        lambda key, amt, meta: spends.append((key, amt, meta)))
    out = rw.park_pod("pod1", reason="moving")
    assert spends and spends[0][0] == "digesta-qwen38"
    assert spends[0][1] == pytest.approx(0.59, rel=0.05)    # one hour at 0.59
    assert out["billed_usd"] == pytest.approx(0.59, rel=0.05)


def test_park_records_the_port_so_the_address_can_be_found_again(monkeypatch):
    """A restarted container is published on a NEW public port, so the url in
    the registry is stale the moment it goes down. _recover_pod_url needs the
    PRIVATE port to match, and register_pod never wrote it."""
    pool = _pool()
    pool.mcfg.port = 9123
    pool.pods.append(_handle())
    reg = _install(monkeypatch, pool)
    rw.park_pod("pod1")
    assert reg.data["pod1"]["port"] == 9123


def test_park_refuses_a_pod_this_process_knows_nothing_about(monkeypatch):
    _install(monkeypatch, _pool())
    with pytest.raises(LookupError):
        rw.park_pod("ghost")


# ---------------------------------------------------------------------------
# a parked pod must be impossible to lose
# ---------------------------------------------------------------------------

def test_parked_pods_are_listed(monkeypatch):
    _install(monkeypatch, None, _Registry({
        "pod1": {"pool": rw.PARKED_POOL, "gpu": "A40", "hourly_usd": 0.59,
                 "image": "vllm:latest", "created": 10.0,
                 "parked": {"from": "digesta-qwen38", "reason": "moving",
                            "at": 20.0}},
        "pod2": {"pool": "other", "gpu": "L40"}}))
    rows = rw.parked_pods()
    assert [r["pod_id"] for r in rows] == ["pod1"]
    assert rows[0]["from_pool"] == "digesta-qwen38"
    assert rows[0]["hourly_usd"] == 0.59


def test_a_parked_pod_appears_on_the_status_page(monkeypatch):
    """It is in no pool, so without this it is reported nowhere at all while
    billing by the hour."""
    _install(monkeypatch, None, _Registry({
        "pod1": {"pool": rw.PARKED_POOL, "gpu": "A40", "hourly_usd": 0.59,
                 "created": time.time() - 7200,
                 "parked": {"from": "digesta-qwen38", "reason": "moving"}}}))
    rows = [r for r in rw.pods_status() if r.get("state") == "parked"]
    assert len(rows) == 1
    row = rows[0]
    assert row["pod_id"] == "pod1"
    assert row["hourly_usd"] == 0.59
    assert row["live_cost_usd"] == pytest.approx(1.18, rel=0.05)   # 2h at 0.59
    assert "billing" in row["note"]


# ---------------------------------------------------------------------------
# unparking
# ---------------------------------------------------------------------------

def test_unpark_refuses_without_a_destination(monkeypatch):
    _install(monkeypatch, None, _Registry({
        "pod1": {"pool": rw.PARKED_POOL, "parked": {}}}))
    with pytest.raises(LookupError):
        rw.unpark_pod("pod1")


def test_unpark_refuses_a_pool_that_is_not_running(monkeypatch):
    _install(monkeypatch, None, _Registry({
        "pod1": {"pool": rw.PARKED_POOL, "parked": {"from": "gone"}}}))
    with pytest.raises(LookupError) as exc:
        rw.unpark_pod("pod1")
    assert "gone" in str(exc.value)


def test_unpark_resolves_the_address_again_rather_than_trusting_the_old_one(monkeypatch):
    pool = _pool("datalab-to/chandra-ocr-2")
    _install(monkeypatch, pool, _Registry({
        "pod1": {"pool": rw.PARKED_POOL, "url": "https://stale/v1",
                 "key": "k", "gpu": "A40", "hourly_usd": 0.59,
                 "created": 1000.0, "vram_gb": 48.0, "gpu_count": 1,
                 "parked": {"from": "digesta-qwen38"}}}))
    monkeypatch.setattr(rw, "_recover_pod_url", lambda pid, info: "https://fresh/v1")
    monkeypatch.setattr(rw, "_pod_health_ok", lambda *a, **k: True)
    monkeypatch.setattr(rw, "register_pod", lambda *a, **k: None)
    out = rw.unpark_pod("pod1", "datalab-to/chandra-ocr-2", wait_s=5)
    assert out["adopted"] is True
    assert out["url"] == "https://fresh/v1"
    assert pool.pods[0].url == "https://fresh/v1"


def test_an_unparked_pod_keeps_billing_from_when_it_was_CREATED(monkeypatch):
    """A restarted pod is not a new pod: RunPod bills from creation. Resetting
    the clock would make the page understate what the account is paying."""
    pool = _pool("p")
    _install(monkeypatch, pool, _Registry({
        "pod1": {"pool": rw.PARKED_POOL, "key": "k", "gpu": "A40",
                 "hourly_usd": 0.59, "created": 1234.0,
                 "parked": {"from": "p"}}}))
    monkeypatch.setattr(rw, "_recover_pod_url", lambda pid, info: "https://x/v1")
    monkeypatch.setattr(rw, "_pod_health_ok", lambda *a, **k: True)
    monkeypatch.setattr(rw, "register_pod", lambda *a, **k: None)
    rw.unpark_pod("pod1", "p", wait_s=5)
    assert pool.pods[0].started_at == 1234.0


def test_unpark_clears_the_park_marker(monkeypatch):
    """Otherwise the leak list -- whose whole job is to be empty -- still names
    a pod that has been serving for hours."""
    pool = _pool("p")
    reg = _install(monkeypatch, pool, _Registry({
        "pod1": {"pool": rw.PARKED_POOL, "key": "k", "gpu": "A40",
                 "created": 1.0, "parked": {"from": "p"}}}))
    monkeypatch.setattr(rw, "_recover_pod_url", lambda pid, info: "https://x/v1")
    monkeypatch.setattr(rw, "_pod_health_ok", lambda *a, **k: True)
    monkeypatch.setattr(rw, "register_pod", lambda *a, **k: None)
    rw.unpark_pod("pod1", "p", wait_s=5)
    assert "parked" not in reg.data["pod1"]
    assert rw.parked_pods() == []


def test_a_pod_that_never_answers_stays_parked(monkeypatch):
    """Adding a pod that cannot serve to a pool means routing work to it. The
    honest outcome is a pod still parked and a line saying it is billing."""
    pool = _pool("p")
    _install(monkeypatch, pool, _Registry({
        "pod1": {"pool": rw.PARKED_POOL, "key": "k", "created": 1.0,
                 "parked": {"from": "p"}}}))
    monkeypatch.setattr(rw, "_recover_pod_url", lambda pid, info: "")
    monkeypatch.setattr(rw.time, "sleep", lambda s: None)
    out = rw.unpark_pod("pod1", "p", wait_s=1)
    assert out["adopted"] is False
    assert out["still_parked"] is True
    assert pool.pods == []


# ---------------------------------------------------------------------------
# the environment travels with the pool, not with the pod
# ---------------------------------------------------------------------------

def test_pool_launch_env_refuses_a_pool_that_is_not_running(monkeypatch):
    monkeypatch.setattr(rw, "_pools", {})
    with pytest.raises(LookupError):
        rw.pool_launch_env("datalab-to/chandra-ocr-2")


def test_pool_launch_env_is_the_pools_own_plan(monkeypatch):
    pool = _pool("p")
    pool.served, pool.entry, pool.seed_entries = "served", {}, []
    monkeypatch.setattr(rw, "_pools", {"p": pool})
    monkeypatch.setattr(rw, "_model_path_for", lambda k: "")
    monkeypatch.setattr(rw, "pod_plan", lambda *a, **k: {
        "env": {"CODERAI_API_TOKEN": "t",
                "CODERAI_SERVER_MAX_PARALLEL_REQUESTS": "192"}})
    env = rw.pool_launch_env("p")
    assert env["CODERAI_SERVER_MAX_PARALLEL_REQUESTS"] == "192"


# ---------------------------------------------------------------------------
# the whole move
# ---------------------------------------------------------------------------

class _FakeClient:
    def __init__(self, before, after, fail=""):
        self._rows = [before, after]
        self.calls = []
        self._fail = fail

    def get_pod(self, pod_id):
        self.calls.append(("get", pod_id))
        return self._rows[0] if len(self.calls) <= 1 else self._rows[1]

    def edit_pod(self, pod_id, image="", env=None):
        self.calls.append(("edit", image, dict(env or {})))
        if self._fail == "edit":
            raise RuntimeError("422 unprocessable")
        return self._rows[1]

    def pod_action(self, pod_id, action):
        self.calls.append(("action", action))
        if self._fail == "action":
            raise RuntimeError("409 conflict")
        return self._rows[1]


def _repurpose_env(monkeypatch, before, after, fail="", pool=None):
    pool = pool or _pool()
    pool.pods.append(_handle())
    reg = _install(monkeypatch, pool, _Registry({
        "pod1": {"pool": pool.model_key, "pid": os.getpid(),
                 "url": "https://old/v1", "key": "pod-key",
                 "image": "vllm:latest", "gpu": "A40", "created": 100.0,
                 "hourly_usd": 0.59}}))
    client = _FakeClient(before, after, fail)
    import codai.api.runpod_client as rc
    monkeypatch.setattr(rc, "RunpodClient", lambda acct: client)
    import codai.models.manager as mm
    monkeypatch.setattr(mm, "get_active_runpod_config",
                        lambda: type("A", (), {"enabled": True, "api_key": "k"})())
    monkeypatch.setattr(rw.time, "sleep", lambda s: None)
    return pool, reg, client


_RUNNING_A40 = {"id": "pod1", "status": "RUNNING", "gpu": "A40",
                "gpu_id": "NVIDIA A40"}


def test_the_dry_run_restarts_and_changes_nothing_else(monkeypatch):
    """No image and no env: the one move that proves restart-in-place keeps the
    machine without risking a configuration mistake at the same time."""
    pool, reg, client = _repurpose_env(monkeypatch, _RUNNING_A40, _RUNNING_A40)
    out = rw.repurpose_pod("pod1", background=False, wait_s=0.1)
    assert out["dry_run"] is True
    assert out["edited"] is False
    assert out["restarted"] is True
    assert ("action", "restart") in client.calls
    assert not any(c[0] == "edit" for c in client.calls)
    assert out["kept_the_machine"] is True


def test_a_card_that_changed_is_not_handed_to_a_pool(monkeypatch):
    """The one risk of the operation. A pod that came back on a different card
    is a different machine, and discovering that from a throughput graph an
    hour later is the outcome this check exists to prevent."""
    moved = dict(_RUNNING_A40, gpu="RTX 4090", gpu_id="NVIDIA GeForce RTX 4090")
    pool, reg, client = _repurpose_env(monkeypatch, _RUNNING_A40, moved)
    out = rw.repurpose_pod("pod1", background=False)
    assert out["kept_the_machine"] is False
    assert "did NOT keep the machine" in out["error"]
    assert pool.pods == [], "it stays parked, out of every pool"
    assert reg.data["pod1"]["parked"]


def test_a_pod_the_restart_destroyed_is_reported_as_such(monkeypatch):
    gone = {"id": "pod1", "status": "TERMINATED", "gpu": "", "gpu_id": ""}
    pool, reg, client = _repurpose_env(monkeypatch, _RUNNING_A40, gone)
    out = rw.repurpose_pod("pod1", background=False)
    assert out["kept_the_machine"] is False
    assert "TERMINATED" in out["error"]


def test_a_blank_card_mid_restart_is_not_read_as_a_changed_card(monkeypatch):
    """RunPod reports the card on the MACHINE, and a pod mid-restart can have
    no machine record. "Not told" must not be mistaken for "moved", or a
    perfectly good pod is abandoned parked."""
    quiet = {"id": "pod1", "status": "RUNNING", "gpu": "", "gpu_id": ""}
    pool, reg, client = _repurpose_env(monkeypatch, _RUNNING_A40, quiet)
    monkeypatch.setattr(rw, "_recover_pod_url", lambda pid, info: "https://x/v1")
    monkeypatch.setattr(rw, "_pod_health_ok", lambda *a, **k: True)
    monkeypatch.setattr(rw, "register_pod", lambda *a, **k: None)
    out = rw.repurpose_pod("pod1", background=False, wait_s=1)
    assert out["kept_the_machine"] is True


def test_the_image_and_env_are_written_before_the_restart(monkeypatch):
    """Both land on the container's next start, so the order is the whole
    point: restarting first would boot the old image once for nothing and need
    a second restart nobody would remember to do."""
    pool, reg, client = _repurpose_env(monkeypatch, _RUNNING_A40, _RUNNING_A40)
    monkeypatch.setattr(rw, "_recover_pod_url", lambda pid, info: "https://x/v1")
    monkeypatch.setattr(rw, "_pod_health_ok", lambda *a, **k: True)
    monkeypatch.setattr(rw, "register_pod", lambda *a, **k: None)
    rw.repurpose_pod("pod1", "digesta-qwen38", image="ocr:latest",
                     env={"CODERAI_API_TOKEN": "new"}, background=False,
                     wait_s=1)
    kinds = [c[0] for c in client.calls]
    assert kinds.index("edit") < kinds.index("action")
    edit = next(c for c in client.calls if c[0] == "edit")
    assert edit[1] == "ocr:latest"
    assert edit[2] == {"CODERAI_API_TOKEN": "new"}


def test_a_failed_runpod_call_leaves_the_pod_parked_and_says_so(monkeypatch):
    """Not back in its old pool: the image may or may not have been written,
    and putting a pod of unknown configuration back into rotation is worse
    than leaving it out of one and naming it."""
    pool, reg, client = _repurpose_env(monkeypatch, _RUNNING_A40,
                                       _RUNNING_A40, fail="action")
    out = rw.repurpose_pod("pod1", image="ocr:latest", background=False)
    assert "409" in out["error"]
    assert out["restarted"] is False
    assert pool.pods == []
    assert reg.data["pod1"]["parked"]


def test_it_refuses_a_pod_that_is_already_gone(monkeypatch):
    gone = {"id": "pod1", "status": "EXITED", "gpu": "", "gpu_id": ""}
    pool, reg, client = _repurpose_env(monkeypatch, gone, gone)
    with pytest.raises(RuntimeError):
        rw.repurpose_pod("pod1", background=False)
    assert pool.pods, "nothing was parked: the pod was never movable"


def test_the_destination_defaults_to_the_pool_it_came_from(monkeypatch):
    pool, reg, client = _repurpose_env(monkeypatch, _RUNNING_A40, _RUNNING_A40)
    out = rw.repurpose_pod("pod1", background=False, wait_s=0.1)
    assert out["to_pool"] == "digesta-qwen38"


# ---------------------------------------------------------------------------
# a restart must not cost the pod
# ---------------------------------------------------------------------------

def test_a_park_orphaned_by_a_restart_is_re_owned_and_sent_home(monkeypatch):
    """The collision that WILL happen: the unattended upgrade restarts this
    process every hour, and these images boot in 6-14 minutes. A parked entry
    whose pid is gone matches no pool's adoption (its pool key is the
    sentinel), so nothing would take it and the reaper would terminate the card
    the repurpose existed to keep."""
    reg = _Registry({
        "pod1": {"pool": rw.PARKED_POOL, "pid": 999999, "proc": "dead",
                 "gpu": "A40", "hourly_usd": 0.59, "created": 1.0,
                 "parked": {"from": "pool:digesta-qwen38"}}})
    monkeypatch.setattr(rw, "_registry_update", reg.update)
    monkeypatch.setattr(rw, "_is_this_process", lambda info: False)
    sent = []
    monkeypatch.setattr(rw, "_unpark_quietly",
                        lambda pid, key: sent.append((pid, key)))
    def _inline(target=None, args=(), **kw):
        return type("T", (), {"start": lambda self: target(*args)})()
    monkeypatch.setattr(threading, "Thread", _inline)
    import codai.models.manager as mm
    monkeypatch.setattr(mm, "get_active_runpod_config",
                        lambda: type("A", (), {"enabled": True})())
    rw.reclaim_registered_pods()
    assert reg.data["pod1"]["pid"] == os.getpid(), "the reaper's deadline is off it"
    assert reg.data["pod1"]["proc"] == rw._PROC_ID
    assert sent == [("pod1", "pool:digesta-qwen38")]


def test_an_orphaned_park_with_nowhere_to_go_is_still_re_owned(monkeypatch):
    """Keeping it is the point: an unpark that cannot be worked out is an
    operator's problem, not a reason to let the pod be terminated."""
    reg = _Registry({
        "pod1": {"pool": rw.PARKED_POOL, "pid": 999999, "proc": "dead",
                 "created": 1.0, "parked": {}}})
    monkeypatch.setattr(rw, "_registry_update", reg.update)
    monkeypatch.setattr(rw, "_is_this_process", lambda info: False)
    sent = []
    monkeypatch.setattr(rw, "_unpark_quietly",
                        lambda pid, key: sent.append((pid, key)))
    import codai.models.manager as mm
    monkeypatch.setattr(mm, "get_active_runpod_config",
                        lambda: type("A", (), {"enabled": True})())
    rw.reclaim_registered_pods()
    assert reg.data["pod1"]["pid"] == os.getpid()
    assert sent == []


def test_a_parked_pod_is_not_counted_as_a_pool_to_reclaim(monkeypatch):
    """Its pool key is the sentinel; feeding that to the ordinary reclaim would
    have it look for a model named '(parked)'."""
    reg = _Registry({
        "pod1": {"pool": rw.PARKED_POOL, "pid": 999999, "proc": "dead",
                 "created": 1.0, "parked": {"from": "p"}}})
    monkeypatch.setattr(rw, "_registry_update", reg.update)
    monkeypatch.setattr(rw, "_is_this_process", lambda info: False)
    monkeypatch.setattr(rw, "_unpark_quietly", lambda pid, key: None)
    looked = []
    monkeypatch.setattr(rw, "_model_entry", lambda k: looked.append(k) or {})
    import codai.models.manager as mm
    monkeypatch.setattr(mm, "get_active_runpod_config",
                        lambda: type("A", (), {"enabled": True})())
    assert rw.reclaim_registered_pods() == 0
    assert rw.PARKED_POOL not in looked


# ---------------------------------------------------------------------------
# a pod that bills must never be invisible
# ---------------------------------------------------------------------------

def test_a_pod_no_pool_holds_is_still_reported(monkeypatch):
    """Live, 2026-11-02 19:22: the account billed FOUR pods while this page
    showed two. `pods_status` walked the pools and the provisioning list, and a
    pod that outlived the process that rented it is in neither -- so a healthy
    RTX 6000 Ada at $0.74/hr serving nothing, and a booting RTX PRO 5000 at
    $0.82/hr, appeared nowhere at all. That is the most expensive row on the
    page precisely because nothing else mentions it."""
    _install(monkeypatch, None, _Registry({
        "orphan1": {"pool": "datalab-to/chandra-ocr-2", "pid": os.getpid(),
                    "proc": rw._PROC_ID, "at": time.time(),
                    "url": "https://107.150.186.62:12640/v1",
                    "gpu": "RTX 6000 Ada", "hourly_usd": 0.74,
                    "created": time.time() - 3600}}))
    rows = [r for r in rw.pods_status() if r.get("state") == "unowned"]
    assert len(rows) == 1
    row = rows[0]
    assert row["pod_id"] == "orphan1"
    assert row["gpu"] == "RTX 6000 Ada"
    assert row["live_cost_usd"] == pytest.approx(0.74, rel=0.05)
    assert "billing" in row["note"]
    assert "nothing routes to it" in row["note"]


def test_an_orphan_with_no_address_says_it_is_still_booting(monkeypatch):
    """The two orphans have different stories and an operator acts on them
    differently: one is serving nothing it could serve, the other may yet come
    up."""
    _install(monkeypatch, None, _Registry({
        "orphan2": {"pool": "p", "pid": os.getpid(), "proc": rw._PROC_ID,
                    "at": time.time(), "url": "", "gpu": "RTX PRO 5000",
                    "hourly_usd": 0.82, "created": time.time() - 300}}))
    row = [r for r in rw.pods_status() if r.get("state") == "unowned"][0]
    assert "still booting" in row["note"]


def test_a_pod_a_pool_here_holds_is_not_also_listed_as_unowned(monkeypatch):
    """Otherwise every healthy pod is reported twice, and the fleet's cost is
    doubled on the page built to show it."""
    pool = _pool("p")
    pool.pods.append(_handle("pod1"))
    pool.pod_capacity_now = lambda p: 32
    _install(monkeypatch, pool, _Registry({
        "pod1": {"pool": "p", "pid": os.getpid(), "proc": rw._PROC_ID,
                 "at": time.time(), "url": "https://x/v1", "gpu": "A40",
                 "hourly_usd": 0.59, "created": time.time() - 60}}))
    rows = rw.pods_status()
    assert [r["state"] for r in rows] == ["ready"]


def test_a_live_siblings_pods_are_left_to_the_sibling(monkeypatch):
    """coderai runs one engine per GPU, each with its own pools. Listing a
    sibling's managed pods would double-count the fleet on this page."""
    _install(monkeypatch, None, _Registry({
        "theirs": {"pool": "p", "pid": 999999, "proc": "another",
                   "at": time.time(), "url": "https://x/v1", "gpu": "A40",
                   "hourly_usd": 0.59, "created": time.time() - 60}}))
    monkeypatch.setattr(rw, "_is_this_process", lambda info: False)
    assert [r for r in rw.pods_status() if r.get("state") == "unowned"] == []


def test_a_dead_siblings_pod_is_reported(monkeypatch):
    """Its heartbeat stopped, so nothing is managing it — which is exactly the
    pod that has to be seen."""
    _install(monkeypatch, None, _Registry({
        "stale": {"pool": "p", "pid": 999999, "proc": "another",
                  "at": time.time() - rw.REAP_GRACE_SECONDS - 10,
                  "url": "https://x/v1", "gpu": "A40", "hourly_usd": 0.59,
                  "created": time.time() - 7200}}))
    monkeypatch.setattr(rw, "_is_this_process", lambda info: False)
    rows = [r for r in rw.pods_status() if r.get("state") == "unowned"]
    assert [r["pod_id"] for r in rows] == ["stale"]


def test_a_parked_pod_is_not_listed_twice(monkeypatch):
    """It has its own row already, with its own story."""
    _install(monkeypatch, None, _Registry({
        "pod1": {"pool": rw.PARKED_POOL, "pid": os.getpid(),
                 "proc": rw._PROC_ID, "at": time.time(), "gpu": "A40",
                 "hourly_usd": 0.59, "created": time.time() - 60,
                 "parked": {"from": "p", "reason": "moving"}}}))
    states = [r.get("state") for r in rw.pods_status()]
    assert states.count("parked") == 1
    assert "unowned" not in states
