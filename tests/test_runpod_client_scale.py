"""A client may ask for warm pods, up to the model's configured ceiling.

Digesta knows things the orchestrator cannot: that a batch of a thousand
documents is about to start. The autoscaler would discover that one cold start
at a time, so a client that knows is allowed to say so — bounded three ways,
because a client that can raise the floor can raise the bill:

* per model (``allow_client_scale``, off by default),
* clamped to ``max_pods``,
* and as a LEASE that expires, so a client that dies holding four A40s stops
  paying for them without anyone noticing.

The lease is the part worth testing hardest: the other two fail visibly, that
one fails as a bill.
"""
import time

from codai.api import runpod_worker as rw


class _Cfg:
    """Just enough of RunpodModelConfig for the floor arithmetic."""

    def __init__(self, **kw):
        self.min_pods = kw.get("min_pods", 0)
        self.max_pods = kw.get("max_pods", 1)
        self.schedule_enabled = kw.get("schedule_enabled", False)
        self.schedule_days = kw.get("schedule_days", [])
        self.schedule_start = kw.get("schedule_start", "")
        self.schedule_end = kw.get("schedule_end", "")
        self.schedule_tz = kw.get("schedule_tz", "")
        self.allow_client_scale = kw.get("allow_client_scale", True)
        self.client_scale_ttl_s = kw.get("client_scale_ttl_s", 1800)
        self.idle_timeout_s = kw.get("idle_timeout_s", 300)
        self.data_center = kw.get("data_center", "EU-SE-1")
        self.api_key = "cra-test"
        self.allow_open_pod = False
        self.health_path = ""


def _pool(**kw):
    """A pool object without touching RunPod or the model list."""
    p = rw.RunpodPodPool.__new__(rw.RunpodPodPool)
    import threading
    p.model_key = kw.pop("model_key", "qwen38-awq")
    p.mcfg = _Cfg(**kw)
    p.pods = []
    p._cv = threading.Condition(threading.RLock())
    p._client_floor = 0
    p._client_floor_until = 0.0
    p._client_floor_logged = -1
    p._sched_closed_logged = False
    p._closed = False
    return p


# ------------------------------------------------------------------ the clamp

def test_a_client_gets_what_it_asks_for_within_the_ceiling():
    p = _pool(max_pods=4)
    out = p.set_client_floor(3)
    assert out["granted"] == 3 and out["clamped"] is False
    assert p.effective_min_pods() == 3


def test_asking_for_more_than_max_pods_gets_the_ceiling_not_an_error():
    """It asked for "as many as you can"; refusing would leave it with none."""
    p = _pool(max_pods=2)
    out = p.set_client_floor(99)
    assert out["granted"] == 2 and out["clamped"] is True
    assert out["max_pods"] == 2
    assert p.effective_min_pods() == 2


def test_zero_releases_the_lease():
    p = _pool(max_pods=4)
    p.set_client_floor(3)
    out = p.set_client_floor(0)
    assert out["granted"] == 0 and out["expires_in_s"] == 0
    assert p.client_floor() == 0
    assert p.effective_min_pods() == 0


def test_a_negative_number_is_a_release_not_a_negative_floor():
    p = _pool(max_pods=4)
    p.set_client_floor(2)
    assert p.set_client_floor(-5)["granted"] == 0
    assert p.effective_min_pods() == 0


# ------------------------------------------------------------------ the lease

def test_the_lease_expires_on_its_own():
    """The failure that costs money: a client asks and never comes back."""
    p = _pool(max_pods=4, client_scale_ttl_s=60)
    p.set_client_floor(3)
    assert p.client_floor() == 3
    p._client_floor_until = time.time() - 1          # as if 60s had passed
    assert p.client_floor() == 0
    assert p.effective_min_pods() == 0


def test_renewing_extends_the_lease():
    p = _pool(max_pods=4, client_scale_ttl_s=120)
    p.set_client_floor(2)
    p._client_floor_until = time.time() + 5
    p.set_client_floor(2)
    assert p._client_floor_until - time.time() > 100


def test_the_configured_ttl_is_honoured():
    p = _pool(max_pods=2, client_scale_ttl_s=300)
    assert p.set_client_floor(1)["expires_in_s"] == 300


def test_a_ttl_of_zero_does_not_mean_forever():
    """An unexpiring lease is the one thing a lease must not be, so a 0 in the
    config falls back to the default instead of being honoured."""
    cfg = rw.parse_model_runpod({"client_scale_ttl_s": 0, "allow_client_scale": True})
    assert cfg.client_scale_ttl_s == 1800
    cfg = rw.parse_model_runpod({"client_scale_ttl_s": 5})
    assert cfg.client_scale_ttl_s == 60      # floored, never seconds-short


# ------------------------------------------------- how it meets the schedule

def test_the_lease_and_the_schedule_are_maxed_not_summed():
    """Both mean "hold this many ready". Summing would rent twice the pods."""
    p = _pool(min_pods=1, max_pods=4)
    p.set_client_floor(2)
    assert p.effective_min_pods() == 2


def test_a_client_cannot_lower_the_configured_floor():
    """qwen38 is warm during office hours by configuration; a client asking for
    1 must not be able to switch that off, nor asking for 0 to tear it down."""
    p = _pool(min_pods=2, max_pods=4)
    p.set_client_floor(1)
    assert p.effective_min_pods() == 2
    p.set_client_floor(0)
    assert p.effective_min_pods() == 2


def test_a_lease_works_outside_the_warm_window():
    """Out of hours the configured floor is 0 — that is where a client asking
    for pods is most useful, so the lease must survive the closed window."""
    p = _pool(min_pods=1, max_pods=3, schedule_enabled=True,
              schedule_start="08:00", schedule_end="20:00",
              schedule_days=[0, 1, 2, 3, 4], schedule_tz="Europe/Rome")
    # Pretend we are outside: assert on both branches rather than on the clock.
    out_of_window = 0 if not rw.schedule_state(p.mcfg)["in_window"] else None
    p.set_client_floor(3)
    assert p.effective_min_pods() == 3
    if out_of_window is not None:
        p.set_client_floor(0)
        assert p.effective_min_pods() == 0


def test_the_ceiling_still_holds_when_a_schedule_is_configured():
    p = _pool(min_pods=1, max_pods=2, schedule_enabled=True,
              schedule_start="08:00", schedule_end="20:00")
    p.set_client_floor(10)
    assert p.effective_min_pods() == 2


# ----------------------------------------------------------- the opt-in gate

def test_client_scaling_is_off_by_default():
    """It spends money, so it cannot be something a model has by accident."""
    assert rw.parse_model_runpod({}).allow_client_scale is False


def test_the_flag_is_read_from_the_block():
    assert rw.parse_model_runpod({"allow_client_scale": True}).allow_client_scale is True


def test_a_model_that_did_not_opt_in_is_refused_as_forbidden():
    """403, not 404: the model exists, the permission does not — a different
    problem for whoever is integrating."""
    import pytest
    saved = rw.model_runpod_block
    rw.model_runpod_block = lambda name: {"mode": "pods", "max_pods": 2}
    try:
        with pytest.raises(PermissionError):
            rw.set_client_pods("qwen38-awq", 2)
    finally:
        rw.model_runpod_block = saved


def test_a_model_with_no_runpod_block_is_not_found():
    import pytest
    saved = rw.model_runpod_block
    rw.model_runpod_block = lambda name: {}
    try:
        with pytest.raises(LookupError):
            rw.set_client_pods("not-a-runpod-model", 1)
    finally:
        rw.model_runpod_block = saved


def test_an_empty_model_name_is_not_found():
    import pytest
    with pytest.raises(LookupError):
        rw.set_client_pods("", 1)


def test_serverless_is_refused_because_runpod_owns_its_scaling():
    import pytest
    saved = rw.model_runpod_block
    rw.model_runpod_block = lambda name: {"mode": "serverless", "endpoint_id": "e1",
                                          "allow_client_scale": True}
    try:
        with pytest.raises(PermissionError):
            rw.set_client_pods("llama-3.3-70b", 2)
    finally:
        rw.model_runpod_block = saved


# -------------------------------------------------------------- the reporting

def test_the_lease_is_visible_in_the_schedule_status():
    """A bill nobody expected has to have a visible cause."""
    p = _pool(max_pods=3, model_key="qwen38-awq")
    p.set_client_floor(2)
    saved = dict(rw._pools)
    rw._pools.clear()
    rw._pools["qwen38-awq"] = p
    try:
        row = rw.pools_schedule_status()[0]
    finally:
        rw._pools.clear()
        rw._pools.update(saved)
    assert row["client_pods"] == 2
    assert row["client_scale_allowed"] is True
    assert row["max_pods"] == 3
    assert row["client_lease_expires_in_s"] > 0
    assert row["effective_min_pods"] == 2


def test_an_expired_lease_reports_zero_not_a_stale_number():
    p = _pool(max_pods=3)
    p.set_client_floor(2)
    p._client_floor_until = time.time() - 1
    saved = dict(rw._pools)
    rw._pools.clear()
    rw._pools["qwen38-awq"] = p
    try:
        row = rw.pools_schedule_status()[0]
    finally:
        rw._pools.clear()
        rw._pools.update(saved)
    assert row["client_pods"] == 0
    assert row["client_lease_expires_in_s"] == 0


# ------------------------------------------------------------ the HTTP surface

def test_the_endpoint_exists_and_is_a_post():
    from codai.api.app import app
    routes = {getattr(r, "path", ""): getattr(r, "methods", set()) for r in app.routes}
    assert "/v1/runpod/scale" in routes
    assert "POST" in routes["/v1/runpod/scale"]


def test_the_endpoint_maps_the_two_errors_to_different_statuses():
    """404 for "no such model", 403 for "not allowed" — so a client can tell a
    typo from a permission it has to be granted."""
    src = (__import__("pathlib").Path(rw.__file__).parents[1]
           / "api" / "app.py").read_text()
    blk = src.split("async def runpod_scale")[1].split("@app.")[0]
    assert "LookupError" in blk and "status_code=404" in blk
    assert "PermissionError" in blk and "status_code=403" in blk


def test_the_endpoint_rejects_a_negative_body_before_it_reaches_the_pool():
    """ge=0 on the field, so pydantic answers 422 rather than the pool guessing."""
    from codai.api.app import RunpodScaleRequest
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        RunpodScaleRequest(model="qwen38-awq", pods=-1)
    assert RunpodScaleRequest(model="qwen38-awq", pods=0).pods == 0


# ------------------------------------- warming at boot, which pays real money

def test_a_disabled_model_is_never_warmed():
    """A catalogue entry parked for later with keep_warm still set would have
    rented a GPU at every boot. Nothing else here consults `enabled`."""
    src = __import__("pathlib").Path(rw.__file__).read_text()
    blk = src.split("def warm_configured_pools")[1].split("def _all_pools_hourly_rate")[0]
    assert 'entry.get("enabled") is False' in blk


def test_a_runpod_only_entry_is_found_by_its_id():
    """Such an entry has no local `path` — there are no weights on this machine
    — and an `alias` only when a second name is wanted. Keyed on alias-or-path
    alone it warmed under "" and the pool was never the one serving requests."""
    src = __import__("pathlib").Path(rw.__file__).read_text()
    blk = src.split("def warm_configured_pools")[1].split("def _all_pools_hourly_rate")[0]
    assert 'entry.get("id")' in blk
    assert 'print("[runpod] a keep_warm entry has no id/alias/path' in blk


def test_boot_warming_respects_the_schedule():
    """ensure_ready() is the "a request is waiting" path and provisions
    unconditionally. At boot nothing is waiting, so a restart at 02:00 must not
    rent the pod the schedule exists to avoid."""
    src = __import__("pathlib").Path(rw.__file__).read_text()
    blk = src.split("def warm_configured_pools")[1].split("def _all_pools_hourly_rate")[0]
    i_guard = blk.find("effective_min_pods() < 1")
    i_ready = blk.find("pool.ensure_ready()")
    assert i_guard != -1 and i_ready != -1 and i_guard < i_ready


def test_a_client_lease_can_warm_at_boot_even_out_of_hours():
    """The guard reads effective_min_pods, which already maxes the lease in, so
    a lease that outlives a restart is honoured rather than dropped."""
    p = _pool(min_pods=1, max_pods=3, schedule_enabled=True,
              schedule_start="08:00", schedule_end="20:00")
    p.set_client_floor(2)
    assert p.effective_min_pods() >= 2


# ------------------------------------- not renting two pods for a floor of one

def test_ensure_ready_marks_the_pool_as_provisioning():
    """A pod is not in self.pods until it answers its health probe, minutes
    later. Without this flag the 15 s scaler reads "0 healthy, floor 1" and
    rents a second one — which is exactly what happened on the first live
    boot: two A40s where the configuration asked for one."""
    src = __import__("pathlib").Path(rw.__file__).read_text()
    blk = src.split("def ensure_ready")[1].split("def _pick")[0]
    assert "self._provisioning = True" in blk
    assert "finally:" in blk and "self._provisioning = False" in blk
    assert blk.index("if self._provisioning:") < blk.index("self._provisioning = True")


def test_the_warm_top_up_does_not_stack_on_a_booting_pod():
    src = __import__("pathlib").Path(rw.__file__).read_text()
    blk = src.split("def maintain")[1].split("def close")[0]
    assert "not self._provisioning" in blk
    assert "sibling_is_provisioning" in blk


def test_the_demand_path_already_had_the_guard_and_keeps_it():
    """_should_grow has checked it all along; this pins that it still does, so
    the two paths cannot drift apart again."""
    src = __import__("pathlib").Path(rw.__file__).read_text()
    blk = src.split("def _should_grow")[1].split("def _grow_in_background")[0]
    assert "or self._provisioning" in blk


def test_the_answer_names_the_model_the_client_asked_about():
    """Models sharing a pool answer with the pool key otherwise, which the
    caller never sent and cannot look up."""
    src = __import__("pathlib").Path(rw.__file__).read_text()
    blk = src.split("def set_client_pods")[1]
    assert 'out["model"] = key' in blk
    assert 'out["pool"]' in blk


def test_a_pod_is_recorded_even_when_the_region_is_unset():
    """The leak, reproduced at the level it happened: build the handle fields
    the way _provision_one does, with nothing configured anywhere. If the
    region expression raises, the pool never records the pod it just rented."""
    class _Acct:
        data_center = ""

    pool = rw.RunpodPodPool.__new__(rw.RunpodPodPool)
    pool.account = _Acct()
    pool.mcfg = rw.parse_model_runpod({})
    pool._volume_dc = None
    assert pool._data_center() == ""        # no exception, and a usable value


# ------------------------------- a restart must not rent beside its own pod

def test_the_warm_path_adopts_before_it_rents():
    """A warm floor is re-established at EVERY boot. Without adoption an hourly
    unattended upgrade rents a fresh A40 each time and leaves the previous one
    billing until the reaper's 20-minute grace expires — observed three times
    on the production orchestrator in one afternoon."""
    src = __import__("pathlib").Path(rw.__file__).read_text()
    ready = src.split("def ensure_ready")[1].split("def _pick")[0]
    assert "_adopt_shared_pod()" in ready
    assert ready.index("_adopt_shared_pod()") < ready.index("self._provision_one()")
    warm = src.split("def maintain")[1].split("def close")[0]
    assert "if not self._adopt_shared_pod():" in warm


def test_adoption_is_one_implementation_for_both_callers():
    """It lived inside acquire() only, which is why the warm path never had it."""
    src = __input = __import__("pathlib").Path(rw.__file__).read_text()
    assert src.count("def _adopt_shared_pod") == 1
    assert src.count("_adopt_shared_pod()") >= 3   # acquire + ensure_ready + maintain


def test_an_adopted_pod_keeps_its_price():
    """It used to be recorded at $0/hr as "billed by its owner". After OUR
    restart there is no other owner: reporting it free hides a real A40 from
    the $/hr cap and from the spend page."""
    src = __import__("pathlib").Path(rw.__file__).read_text()
    blk = src.split("def _adopt_shared_pod")[1].split("def _data_center")[0]
    assert "hourly_usd=float(shared_rate or 0.0)" in blk
    assert "hourly_usd=0.0" not in blk


def test_the_registry_records_the_rate_so_adoption_can_read_it():
    import json
    import tempfile
    d = tempfile.mkdtemp()
    saved = rw._pod_registry_path
    rw._pod_registry_path = lambda: d + "/pods.json"
    try:
        rw.register_pod("p1", "pool:x", "http://pod:8000", "tok", hourly_usd=0.59)
        rec = json.load(open(d + "/pods.json"))["p1"]
        assert rec["hourly_usd"] == 0.59
        assert rec["url"] == "http://pod:8000" and rec["key"] == "tok"
        # An older entry with no rate must not break adoption.
        rec.pop("hourly_usd")
        json.dump({"p1": rec}, open(d + "/pods.json", "w"))
        saved_health = rw._pod_health_ok
        rw._pod_health_ok = lambda *a, **k: True
        try:
            assert rw.find_shared_pod("pool:x")[3] == 0.0
        finally:
            rw._pod_health_ok = saved_health
    finally:
        rw._pod_registry_path = saved


def test_the_warm_path_waits_for_a_pod_that_is_still_booting():
    """A booting pod is registered with no URL, so there is nothing to adopt
    yet — but it is one pod on the way, not none. An upgrade landing during the
    ~6 minutes an image pull takes would otherwise rent a second one, and the
    upgrade runs hourly."""
    src = __import__("pathlib").Path(rw.__file__).read_text()
    ready = src.split("def ensure_ready")[1].split("def _pick")[0]
    assert "sibling_is_provisioning" in ready
    assert ready.index("sibling_is_provisioning") < ready.index("self._provision_one()")


def test_a_booting_pod_from_a_previous_process_counts():
    """The whole point after a restart: the pid differs, so the entry is not
    'ours', and it has no url yet."""
    import json
    import os
    import tempfile
    d = tempfile.mkdtemp()
    saved = rw._pod_registry_path
    rw._pod_registry_path = lambda: d + "/pods.json"
    try:
        json.dump({"p1": {"pool": "pool:x", "pid": os.getpid() + 1,
                          "at": rw.time.time(), "url": "", "key": "k"}},
                  open(d + "/pods.json", "w"))
        assert rw.sibling_is_provisioning("pool:x") is True
        # Stale beyond the grace must not block provisioning forever.
        json.dump({"p1": {"pool": "pool:x", "pid": os.getpid() + 1,
                          "at": rw.time.time() - rw.REAP_GRACE_SECONDS - 10,
                          "url": "", "key": "k"}},
                  open(d + "/pods.json", "w"))
        assert rw.sibling_is_provisioning("pool:x") is False
    finally:
        rw._pod_registry_path = saved


def test_the_reaper_prunes_entries_for_pods_that_are_gone():
    """A terminated-but-registered id stays 'known', so it is shielded from
    reaping and accumulates for the life of the install."""
    src = __import__("pathlib").Path(rw.__file__).read_text()
    blk = src.split("def reap_orphans")[1].split("def _scaler_loop")[0]
    assert "known - live" in blk and "unregister_pod(gone)" in blk
