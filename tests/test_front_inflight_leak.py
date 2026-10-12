"""The in-flight count is load-bearing, so a leak in it freezes the install.

`coderai_engine_inflight` / `coderai_requests_active` are not decoration. The hourly
autoupgrade's idle guard reads them to decide whether a restart is safe, and with
`MAX_DEFERRALS=0` a non-zero count means it SKIPS — for ever. So a single request that
is counted in and never out stops the host taking upgrades at all, silently, and the
only symptom is a version that stops moving.

That happened on the Digesta orchestrator on 2026-10-12: the gauge read 13 with two
real requests in the log, the 09:01 tick logged "SKIP: 7 piece(s) of work in progress"
and left the image alone, and nothing short of a manual restart would have cleared it.

Three defences, one test each:

* the streaming path releases from BOTH the relay generator and the background task,
  because Starlette runs the latter only after a successful send;
* `exit_request` is idempotent per rid, so releasing twice cannot decrement someone
  else's request;
* a reaper releases anything held implausibly long, so an unknown fourth leak cannot
  wedge the install either.
"""

import pathlib
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codai.frontproxy.registry import Engine                   # noqa: E402


def _engine():
    return Engine(id=1, name="e1", backend="gguf", gpu="", port=9001)


# ------------------------------------------------------------- the accounting #
def test_a_request_is_counted_in_and_out():
    e = _engine()
    rid = e.enter_request()
    assert e.inflight == 1
    e.exit_request(rid)
    assert e.inflight == 0


def test_releasing_twice_does_not_double_decrement():
    """The streaming path deliberately releases from two places. If the second
    call decremented again it would cancel an unrelated in-flight request, and the
    count would under-report real work — which is worse than over-reporting it,
    because then a restart happens DURING a request."""
    e = _engine()
    a = e.enter_request()
    b = e.enter_request()
    assert e.inflight == 2
    e.exit_request(a)
    e.exit_request(a)          # the second path firing for the same request
    e.exit_request(a)
    assert e.inflight == 1     # b is still genuinely in flight
    e.exit_request(b)
    assert e.inflight == 0


def test_the_count_never_goes_negative():
    e = _engine()
    e.exit_request("never-existed")
    e.exit_request(None)
    assert e.inflight == 0


def test_every_request_is_timestamped_not_just_inference_ones():
    """`active` only holds requests that carried metadata (the inference ones), so
    a leaked non-inference request used to be invisible: the count said 13 and the
    Tasks page showed nothing. Timestamping all of them is what makes a leak both
    visible and reapable."""
    e = _engine()
    plain = e.enter_request()                       # no meta
    infer = e.enter_request({"model": "m", "kind": "chat", "path": "/v1/chat"})
    assert set(e.inflight_since) == {plain, infer}
    assert set(e.active) == {infer}


# ----------------------------------------------------------------- the reaper #
def test_a_stuck_request_is_released():
    e = _engine()
    rid = e.enter_request({"path": "/v1/chat/completions"})
    e.inflight_since[rid] = time.time() - 7200       # two hours ago
    assert e.reap_stale_requests(3600) == 1
    assert e.inflight == 0
    assert rid not in e.inflight_since
    assert rid not in e.active


def test_a_young_request_is_left_alone():
    """A cold RunPod pod can make one OCR page take 14 minutes. Reaping a live
    request would be a far worse bug than the one being fixed."""
    e = _engine()
    rid = e.enter_request()
    assert e.reap_stale_requests(3600) == 0
    assert e.inflight == 1
    e.exit_request(rid)


def test_the_reaper_can_be_switched_off():
    e = _engine()
    rid = e.enter_request()
    e.inflight_since[rid] = time.time() - 99999
    assert e.reap_stale_requests(0) == 0
    assert e.inflight == 1


def test_reaping_is_selective():
    e = _engine()
    old = e.enter_request()
    new = e.enter_request()
    e.inflight_since[old] = time.time() - 7200
    assert e.reap_stale_requests(3600) == 1
    assert e.inflight == 1
    assert new in e.inflight_since


def test_a_reaped_request_releasing_late_is_harmless():
    """The request may finish after being reaped; its release must be a no-op
    rather than a decrement of whatever is in flight by then."""
    e = _engine()
    rid = e.enter_request()
    e.inflight_since[rid] = time.time() - 7200
    e.reap_stale_requests(3600)
    other = e.enter_request()
    e.exit_request(rid)                  # the late release
    assert e.inflight == 1               # `other` survived
    e.exit_request(other)
    assert e.inflight == 0


def test_the_accounting_is_thread_safe():
    """The front releases from a background task and an async generator; the
    supervisor reaps from its own thread."""
    e = _engine()
    rids = []
    lock = threading.Lock()

    def _work():
        for _ in range(200):
            rid = e.enter_request()
            with lock:
                rids.append(rid)

    threads = [threading.Thread(target=_work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert e.inflight == 800
    for rid in rids:
        e.exit_request(rid)
    assert e.inflight == 0


# --------------------------------------------------- the structural fix itself #
def test_the_streaming_path_releases_from_both_places():
    """Starlette awaits a background task only AFTER the body is sent, so a client
    that disconnects mid-stream never triggers it. The relay generator's finally is
    the half that covers that case; the background task covers the non-streaming
    return. Asserted on the source because the alternative is standing up a real
    ASGI server and a disconnecting client to prove a two-line invariant."""
    import inspect
    from codai.frontproxy.app import FrontProxy
    src = inspect.getsource(FrontProxy.proxy)
    assert "_released = False" in src, "the release must be idempotent"
    assert "BackgroundTask(_release)" in src, "the non-streaming half"
    # ...and the generator half: a finally that awaits the release.
    assert src.count("await _release()") >= 1, "the disconnect half"


def test_the_metrics_scrape_reaps_before_reporting():
    """The scrape the autoupgrade reads is exactly the moment a stale count does
    damage, so it is the right moment to correct it."""
    import inspect
    from codai.frontproxy import metrics
    src = inspect.getsource(metrics)
    assert "reap_stale_requests" in src
    assert src.index("reap_stale_requests") < src.index("coderai_requests_active")


def test_the_drain_wait_reaps_first():
    """Otherwise a restart waits out the whole grace period for a request that
    finished an hour ago."""
    import inspect
    from codai.frontproxy import engine_supervisor
    src = inspect.getsource(engine_supervisor)
    assert "reap_stale_requests" in src


def test_the_ceiling_is_configurable_and_generous():
    from codai.config import ServerConfig
    assert ServerConfig().inflight_max_age_s == 3600.0
