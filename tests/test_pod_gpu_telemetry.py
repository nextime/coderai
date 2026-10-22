"""Live GPU/VRAM for rented pods on the RunPod page, read FROM THE PODS.

A pod serves /internal/engine-state, which already carries a `vram` block
(used/free/total per card, cached 4s there). I first built this against
/admin/api/gpu-stats and the pods answered 404: a pod runs the SLIM app, which
serves /healthz, /v1/models and /internal/engine-state and no admin API at all.
Probe the pod before choosing its endpoint.

Asking the pod is cheaper and fresher than asking RunPod, whose utilisation
lags by tens of seconds and whose every read spends the same quota that renting
and boot-polling need -- telemetry must never be able to cost a fleet its
ability to rent.
"""
import asyncio
import types

import pytest

from codai.api import pod_telemetry as pt


def setup_function():
    pt.forget()


# --------------------------------------------------------------- summarising #

def test_utilisation_is_the_busiest_card_not_the_average():
    """On a multi-card pod the saturated card decides whether it can take more
    work; averaging it with an idle sibling hides exactly that."""
    out = pt.summarise([{"util": 98, "mem_used": 40, "mem_total": 48, "temp": 81},
                        {"util": 2, "mem_used": 1, "mem_total": 48, "temp": 44}])
    assert out["gpu_util"] == 98.0
    assert out["gpu_temp_c"] == 81.0
    assert out["vram_used_gb"] == 41.0      # summed: one pool to load into
    assert out["vram_total_gb"] == 96.0
    assert out["gpu_cards"] == 2


def test_one_card_reads_straight_through():
    out = pt.summarise([{"util": 37.5, "mem_used": 12.25, "mem_total": 48, "temp": 60}])
    assert out == {"gpu_cards": 1, "gpu_util": 37.5, "gpu_temp_c": 60.0,
                   "vram_used_gb": 12.25, "vram_total_gb": 48.0}


def test_no_cards_means_no_numbers():
    """Not zeroes: a zero draws a bar that reads as "idle"."""
    assert pt.summarise([]) == {}
    assert pt.summarise(None) == {}


def test_a_partial_card_contributes_what_it_has():
    out = pt.summarise([{"util": 50}])
    assert out["gpu_util"] == 50.0
    assert "vram_used_gb" not in out


def test_junk_rows_are_ignored():
    assert pt.summarise(["nope", None, {"util": 10}])["gpu_util"] == 10.0


# ------------------------------------------------------------------ the url #

@pytest.mark.parametrize("given,want", [
    ("https://1.2.3.4:22145", "https://1.2.3.4:22145"),
    ("https://1.2.3.4:22145/", "https://1.2.3.4:22145"),
    ("https://1.2.3.4:22145/v1", "https://1.2.3.4:22145"),
    ("https://1.2.3.4:22145/v1/", "https://1.2.3.4:22145"),
    ("", ""),
])
def test_the_admin_path_hangs_off_the_root(given, want):
    assert pt._root(given) == want


# ------------------------------------------------------------- the fetching #

def _state(devices, status=200):
    """A pod's /internal/engine-state answer."""
    r = types.SimpleNamespace()
    r.status_code = status
    r.json = lambda: {"ok": True, "vram": {"devices": devices,
                                           "device_count": len(devices)}}
    return r


def _answer(cards, status=200):
    """Cards in summarise() shape, expressed as a pod would report them:
    the pod reports FREE, not used."""
    devs = [{"total": c.get("mem_total", 48),
             "free": c.get("mem_total", 48) - c.get("mem_used", 0),
             **{k: c[k] for k in ("util", "temp") if k in c}}
            for c in cards]
    return _state(devs, status)


def test_a_pod_is_asked_with_its_own_bearer(monkeypatch):
    seen = {}

    def _get(url, headers=None, timeout=None):
        seen.update(url=url, headers=headers or {}, timeout=timeout)
        return _answer([{"util": 55, "mem_used": 20, "mem_total": 48}])
    monkeypatch.setattr("codai.api.pod_http.get", _get)
    out = asyncio.run(pt.collect([("p1", "https://pod:1/v1", "its-own-key")]))
    assert out["p1"]["gpu_util"] == 55.0
    assert seen["url"] == "https://pod:1/internal/engine-state"
    assert seen["headers"]["Authorization"] == "Bearer its-own-key"
    assert seen["timeout"] == pt.TIMEOUT_S


def test_several_pods_are_asked_at_once(monkeypatch):
    def _get(url, headers=None, timeout=None):
        return _answer([{"util": 10 * int(url.split("//pod")[1][0]),
                         "mem_used": 1, "mem_total": 48}])
    monkeypatch.setattr("codai.api.pod_http.get", _get)
    out = asyncio.run(pt.collect([("a", "https://pod1:1", "k"),
                                  ("b", "https://pod2:1", "k")]))
    assert out["a"]["gpu_util"] == 10.0 and out["b"]["gpu_util"] == 20.0


def test_a_pod_that_does_not_answer_has_no_numbers(monkeypatch):
    """Booting, saturated or gone: the page still renders, and its health is
    reported beside it already."""
    def _boom(*a, **k):
        raise OSError("connection refused")
    monkeypatch.setattr("codai.api.pod_http.get", _boom)
    assert asyncio.run(pt.collect([("p1", "https://pod:1", "k")])) == {}


def test_a_non_200_is_not_telemetry(monkeypatch):
    monkeypatch.setattr("codai.api.pod_http.get",
                        lambda *a, **k: _answer([], status=401))
    assert asyncio.run(pt.collect([("p1", "https://pod:1", "k")])) == {}


def test_one_slow_pod_does_not_hide_the_others(monkeypatch):
    import time

    def _get(url, headers=None, timeout=None):
        if "slow" in url:
            time.sleep(5)
        return _answer([{"util": 42, "mem_used": 1, "mem_total": 48}])
    monkeypatch.setattr("codai.api.pod_http.get", _get)
    out = asyncio.run(pt.collect([("slow", "https://slow:1", "k"),
                                  ("fast", "https://fast:1", "k")],
                                 overall_timeout=1.0))
    assert "fast" in out and "slow" not in out


def test_a_pod_with_no_address_is_skipped(monkeypatch):
    called = []
    monkeypatch.setattr("codai.api.pod_http.get",
                        lambda *a, **k: called.append(1) or _answer([]))
    assert asyncio.run(pt.collect([("booting", "", "k")])) == {}
    assert not called


# ----------------------------------------------------------------- the cache #

def test_a_fresh_answer_is_reused(monkeypatch):
    """Several open tabs must collapse onto one round of requests."""
    calls = []

    def _get(url, headers=None, timeout=None):
        calls.append(url)
        return _answer([{"util": 5, "mem_used": 1, "mem_total": 48}])
    monkeypatch.setattr("codai.api.pod_http.get", _get)
    eps = [("p1", "https://pod:1", "k")]
    asyncio.run(pt.collect(eps))
    asyncio.run(pt.collect(eps))
    assert len(calls) == 1


def test_a_stale_answer_is_re_asked(monkeypatch):
    calls = []

    def _get(url, headers=None, timeout=None):
        calls.append(url)
        return _answer([{"util": 5, "mem_used": 1, "mem_total": 48}])
    monkeypatch.setattr("codai.api.pod_http.get", _get)
    eps = [("p1", "https://pod:1", "k")]
    asyncio.run(pt.collect(eps))
    monkeypatch.setattr(pt, "TTL_S", -1.0)
    asyncio.run(pt.collect(eps))
    assert len(calls) == 2


def test_the_cache_is_bounded(monkeypatch):
    for i in range(700):
        pt._store(f"p{i}", {"gpu_util": 1.0})
    assert len(pt._CACHE) <= 512


# ------------------------------------------------- the addresses stay inside #

def test_the_pod_bearer_never_reaches_the_page():
    """held_pod_endpoints is deliberately separate from pods_status: the
    snapshot is serialised into an admin page, the addresses are not."""
    import inspect
    from codai.api import runpod_worker as rw
    src = inspect.getsource(rw.pods_status)
    assert "pod_key" not in src and "api_key" not in src


def test_endpoints_carry_each_pods_own_key():
    import threading
    from codai.api import runpod_worker as rw
    pool = rw.RunpodPodPool.__new__(rw.RunpodPodPool)
    pool.api_key = "pool-key"
    pool._pod_keys = {}
    pool._cv = threading.Condition(threading.RLock())
    pool.pods = [rw.PodHandle(pod_id="a", url="https://a:1", hourly_usd=0.5,
                              started_at=0.0, api_key="own-key"),
                 rw.PodHandle(pod_id="b", url="https://b:1", hourly_usd=0.5,
                              started_at=0.0)]
    saved = dict(rw._pools)
    try:
        rw._pools.clear()
        rw._pools["pool:x"] = pool
        got = dict((pid, key) for pid, _url, key in rw.held_pod_endpoints())
    finally:
        rw._pools.clear()
        rw._pools.update(saved)
    assert got == {"a": "own-key", "b": "pool-key"}


# --------------------------- the pod reports FREE, so used is the difference #

def test_used_is_derived_from_free_because_that_is_what_the_pod_reports():
    cards = pt.cards_from_engine_state(
        {"vram": {"devices": [{"index": 0, "name": "NVIDIA A40",
                               "free": 4.5, "total": 48.31}]}})
    assert cards == [{"mem_total": 48.31, "mem_used": pytest.approx(43.81)}]


def test_util_and_temp_ride_along_when_the_pod_reports_them():
    cards = pt.cards_from_engine_state(
        {"vram": {"devices": [{"free": 4.5, "total": 48.31,
                               "util": 97, "temp": 74}]}})
    assert cards[0]["util"] == 97.0 and cards[0]["temp"] == 74.0


def test_an_older_pod_reports_vram_only():
    """It serves engine-state without util/temp. VRAM must still show, and the
    page shows a dash for utilisation rather than a zeroed bar."""
    out = pt.summarise(pt.cards_from_engine_state(
        {"vram": {"devices": [{"free": 4.5, "total": 48.31}]}}))
    assert out["vram_used_gb"] == pytest.approx(43.81)
    assert "gpu_util" not in out and "gpu_temp_c" not in out


def test_a_pod_with_no_card_reports_nothing():
    assert pt.cards_from_engine_state({"vram": None}) == []
    assert pt.cards_from_engine_state({}) == []
    assert pt.cards_from_engine_state({"vram": {"devices": []}}) == []


def test_totals_without_a_device_list_are_still_used():
    cards = pt.cards_from_engine_state({"vram": {"used": 43.8, "free": 4.5,
                                                 "total": 48.31}})
    assert cards[0]["mem_total"] == 48.31


def test_the_real_shape_a_live_pod_returned():
    """Copied from pod 997ags1r9l6im3, 2026-10-22."""
    out = pt.summarise(pt.cards_from_engine_state({
        "ok": True, "vram": {"used": 43.8, "free": 4.5, "total": 48.31,
                             "gpu": "NVIDIA A40",
                             "devices": [{"index": 0, "name": "NVIDIA A40",
                                          "free": 4.5, "total": 48.31}],
                             "device_count": 1}}))
    assert out["vram_used_gb"] == pytest.approx(43.81)
    assert out["vram_total_gb"] == 48.31
    assert out["gpu_cards"] == 1


def test_what_the_pod_has_served_comes_through(monkeypatch):
    """The counter an operator reads as "is this rented pod earning its hour?".
    It travels in the engine-state `load` block beside the limits, and only the
    numeric keys the page draws are copied -- a pod on an older image simply
    does not send them, and the page shows a dash rather than a zero."""
    r = types.SimpleNamespace()
    r.status_code = 200
    r.json = lambda: {"ok": True,
                      "vram": {"devices": [{"total": 48, "free": 8}]},
                      "load": {"requests_done": 4012, "prompt_tokens": 1250000,
                               "generation_tokens": 412000, "running": 3,
                               "ram_used_gb": 19.4, "ram_total_gb": 60.0,
                               "kv_cache_pct": 26.5, "preemptions": 4,
                               "ram_source": "cgroup"}}
    monkeypatch.setattr("codai.api.pod_http.get",
                        lambda *a, **k: r)
    out = asyncio.run(pt.collect([("p1", "https://pod:1", "k")]))["p1"]
    assert out["requests_done"] == 4012
    assert out["prompt_tokens"] == 1250000
    assert out["generation_tokens"] == 412000
    assert out["kv_cache_pct"] == 26.5
    assert "ram_source" not in out          # a string, not a figure to draw


def test_an_older_pod_sends_no_served_counts(monkeypatch):
    monkeypatch.setattr("codai.api.pod_http.get",
                        lambda *a, **k: _answer([{"mem_used": 40, "mem_total": 48}]))
    out = asyncio.run(pt.collect([("p9", "https://pod:1", "k")]))["p9"]
    assert "requests_done" not in out
    assert out["vram_used_gb"] == 40.0
