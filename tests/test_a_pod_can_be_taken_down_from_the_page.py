"""The RunPod page can stop a pod, and the rows that needed it most had no way.

Every other control on that page scales a POOL. That is fine for a pod under a
floor, and useless for the two states the page itself prints in red:

* `unowned` -- in the registry, held by no pool, billing. The table says so in
  the loudest colour on the page precisely because nothing else would have
  mentioned it (the account billed four pods while the table showed two), and
  then offered the operator a link to the logs.
* `parked` -- deliberately out of every pool so an in-place restart can survive
  its own pool's quarantine. Rented, billing, routed nothing. It is meant to
  last minutes, and a parked row still there an hour later is money going
  nowhere.

Neither has a floor to lower, so no scale-down will ever reach either one, and
the only remaining way to stop them was the RunPod console. `release_pod` has
done this from the API since `/v1/runpod/release`; this is the same call with a
button on it.

Forced on purpose. `shed_now` is the graceful path (drain, release when the last
request finishes); a button an operator reaches for while a pod is wedged must
not be refused by the same in-flight counter that is wedged with it. What it
owes the operator instead is to say what it cost -- so the reply carries the
in-flight count the server saw at the moment the pod went, and the confirm says
beforehand that the work will be discarded and that a floored pod will simply be
re-rented at a fresh boot.
"""
import asyncio
import inspect
import json
import pathlib
import re
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAGE = ROOT / "codai/admin/templates/runpod.html"
ENDPOINT = "/admin/api/runpod/pod-release"


@pytest.fixture(scope="module")
def html():
    return PAGE.read_text(encoding="utf-8")


# ── the endpoint ─────────────────────────────────────────────────────────────

def test_the_endpoint_is_admin_gated_like_every_other_pod_action():
    from codai.admin import routes
    src = inspect.getsource(routes)
    assert ENDPOINT in src, "the take-down endpoint is gone"
    body = src[src.index(ENDPOINT):][:1800]
    assert "Depends(require_admin)" in body, (
        "terminating a named pod is not reversible and must not be reachable "
        "without an admin session")


def test_it_terminates_the_pod_and_reports_what_that_cost(monkeypatch):
    """The reply is the worker's own answer, in-flight count included."""
    from codai.admin import routes
    from codai.api import runpod_worker

    seen = {}

    def _fake_release(pod_id, reason="x"):
        seen["pod_id"], seen["reason"] = pod_id, reason
        return {"pod_id": pod_id, "found_in_pool": "qwen3-8b", "inflight": 3,
                "terminated": True, "billed_usd": 1.2345}

    monkeypatch.setattr(runpod_worker, "release_pod", _fake_release)
    out = asyncio.run(routes.api_runpod_pod_release(
        payload={"pod_id": " abc123 "}, username="root"))
    assert seen["pod_id"] == "abc123", "the id is not trimmed"
    assert out["success"] is True
    assert out["terminated"] is True
    assert out["inflight"] == 3, "a discarded request must be reported"
    assert out["billed_usd"] == 1.2345


def test_the_reason_names_the_operator_who_pressed_it():
    """A terminated pod leaves a ledger entry and a log line; both are read
    later by somebody asking who stopped it."""
    from codai.admin import routes
    from codai.api import runpod_worker
    import pytest as _p
    seen = {}

    def _fake_release(pod_id, reason="x"):
        seen["reason"] = reason
        return {"pod_id": pod_id, "terminated": True}

    mp = _p.MonkeyPatch()
    mp.setattr(runpod_worker, "release_pod", _fake_release)
    try:
        asyncio.run(routes.api_runpod_pod_release(
            payload={"pod_id": "p1"}, username="alice"))
        assert "alice" in seen["reason"]
        asyncio.run(routes.api_runpod_pod_release(
            payload={"pod_id": "p1", "reason": "draining the fleet"},
            username="alice"))
        assert seen["reason"] == "draining the fleet", (
            "a caller-supplied reason must win")
    finally:
        mp.undo()


def test_a_missing_pod_id_is_refused_before_anything_is_terminated(monkeypatch):
    from fastapi import HTTPException
    from codai.admin import routes
    from codai.api import runpod_worker

    def _boom(*a, **k):
        raise AssertionError("release_pod was called with no pod id")

    monkeypatch.setattr(runpod_worker, "release_pod", _boom)
    for payload in ({}, {"pod_id": ""}, {"pod_id": "   "}, None):
        with pytest.raises(HTTPException) as got:
            asyncio.run(routes.api_runpod_pod_release(payload=payload,
                                                      username="root"))
        assert got.value.status_code == 400


def test_a_pod_nobody_has_heard_of_is_a_404_not_a_500(monkeypatch):
    """`release_pod` raises LookupError for an id it cannot make sense of; that
    is the client's mistake, and a 500 would send an operator to the logs."""
    from fastapi import HTTPException
    from codai.admin import routes
    from codai.api import runpod_worker

    def _missing(pod_id, reason="x"):
        raise LookupError("no pod id given")

    monkeypatch.setattr(runpod_worker, "release_pod", _missing)
    with pytest.raises(HTTPException) as got:
        asyncio.run(routes.api_runpod_pod_release(payload={"pod_id": "zz"},
                                                  username="root"))
    assert got.value.status_code == 404


def test_a_half_done_termination_is_surfaced_not_swallowed(monkeypatch):
    """A pod dropped from its pool but still billing is the worst state of the
    three, and the only way an operator can act on it is to be told."""
    from fastapi import HTTPException
    from codai.admin import routes
    from codai.api import runpod_worker

    def _half(pod_id, reason="x"):
        raise RuntimeError("pod p1 was removed from pool but RunPod did not "
                           "confirm the termination")

    monkeypatch.setattr(runpod_worker, "release_pod", _half)
    with pytest.raises(HTTPException) as got:
        asyncio.run(routes.api_runpod_pod_release(payload={"pod_id": "p1"},
                                                  username="root"))
    assert got.value.status_code == 500
    assert "did not confirm" in str(got.value.detail)


def test_the_dead_pods_telemetry_is_dropped_with_it(monkeypatch):
    """Telemetry is cached for 4s. Left behind, the next refresh draws GPU and
    VRAM bars for a pod that no longer exists."""
    from codai.admin import routes
    from codai.api import runpod_worker, pod_telemetry

    monkeypatch.setattr(runpod_worker, "release_pod",
                        lambda pod_id, reason="x": {"terminated": True})
    pod_telemetry._store("abc123", {"vram_used_gb": 43.2, "vram_total_gb": 48.0})
    assert pod_telemetry.cached("abc123"), "fixture did not take"
    asyncio.run(routes.api_runpod_pod_release(payload={"pod_id": "abc123"},
                                              username="root"))
    assert not pod_telemetry.cached("abc123"), (
        "a terminated pod is still drawing bars")


# ── the button ───────────────────────────────────────────────────────────────

needs_node = pytest.mark.skipif(not shutil.which("node"),
                                reason="node is not installed")


def _fn(html, name):
    m = re.search(r"^function %s\(.*?^}" % re.escape(name), html, re.S | re.M)
    assert m, f"{name}() is gone from the page"
    return m.group(0)


def _kill(html, tmp_path, pod):
    js = tmp_path / "kill.js"
    js.write_text("const _rpRows = {};\n" + _fn(html, "_killCell")
                  + "\nconsole.log(_killCell(%s));" % json.dumps(pod),
                  encoding="utf-8")
    proc = subprocess.run(["node", str(js)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr.strip()[:800]
    return proc.stdout


@needs_node
def test_every_pod_with_an_id_gets_a_button(html, tmp_path):
    out = _kill(html, tmp_path, {"pod_id": "abc123", "model": "qwen3-8b",
                                 "state": "ok"})
    assert "rpRelease('abc123')" in out
    assert "Take down" in out


@needs_node
def test_the_rows_that_no_pool_can_reach_get_the_loud_button(html, tmp_path):
    """`unowned` and `parked` are the states this button exists for."""
    for state in ("unowned", "parked"):
        out = _kill(html, tmp_path, {"pod_id": "p1", "state": state})
        assert "#ef4444" in out, f"{state} is not marked as the urgent case"
    plain = _kill(html, tmp_path, {"pod_id": "p1", "state": "ok"})
    assert "#ef4444" not in plain, "a healthy pod must not be dressed as urgent"


@needs_node
def test_a_degraded_row_with_no_pod_id_gets_no_button(html, tmp_path):
    """`pods_status` emits placeholder rows when a pool lock is held. A button
    on one of those could only ever 400."""
    assert _kill(html, tmp_path, {"pod_id": "", "model": "qwen3-8b"}).strip() == ""


@needs_node
def test_an_apostrophe_in_a_model_name_cannot_break_the_button(html, tmp_path):
    """Model names are operator-supplied text; the handler takes the pod id
    only, and the row is looked up from the last snapshot."""
    out = _kill(html, tmp_path, {"pod_id": "p1", "model": "l'ours", "state": "ok"})
    assert "rpRelease('p1')" in out
    assert "l'ours" not in out, "the model name is being pasted into the handler"


def test_the_confirm_says_what_cannot_be_undone(html):
    """Three things make this expensive and none of them is visible in a
    button: it is irreversible, it discards in-flight work, and a floored pod
    is re-rented at a fresh boot."""
    body = html[html.index("async function rpRelease("):][:2000]
    assert "cannot be undone" in body
    assert "discarded" in body
    assert "keep_warm" in body and "boot again" in body
    assert "confirm(" in body, "it must ask first"


def test_a_pod_that_was_already_gone_is_reported_as_success(html):
    """Reaped for idle a second before the click: nothing left to terminate,
    and sending the operator to the console to check would be wrong."""
    body = html[html.index("async function rpRelease("):][:2000]
    assert "already_gone" in body


def test_the_remembered_rows_do_not_outlive_the_snapshot(html):
    """A released pod must stop being addressable, and a tab left open all day
    must not accumulate rows."""
    body = html[html.index("async function rpRefresh("):]
    assert "delete _rpRows[k]" in body, "the row map is never cleared"
    assert body.index("delete _rpRows[k]") < body.index("_killCell"), (
        "the map is cleared after the rows are drawn, which erases them")


def test_the_table_has_a_column_for_it(html):
    """A cell added without widening the header silently shifts every column,
    and the empty-state colspan stops spanning the table."""
    head = re.search(r"<thead><tr[^>]*>(.*?)</tr></thead>", html, re.S).group(1)
    cols = len(re.findall(r"<th", head))
    assert cols == 12, f"the pods table header has {cols} columns"
    spans = set(re.findall(r'colspan="(\d+)" class="dim">No pods running',
                           html))
    assert spans == {"12"}, f"empty-state colspan(s) {spans} != header columns"
