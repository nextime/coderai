"""A pod that is booting is a pod that is billing.

RunPod charges for a pod from the moment it is created. A pod spends ~12
minutes becoming useful -- pulling the image, then loading weights -- and is
not healthy for any of it. `live_hourly_usd` summed only the healthy ones, so
a fleet entirely in paid boot was reported as free.

Measured 2026-10-18 against the account's own billing:

    reported  $0.59/hr      actual  $4.72/hr      (8x understated)
    reported  $0.00/hr      actual  $7.67/hr      (13 pods, 8 serving)

This is the most expensive shape a wrong number can have: a client polling it
concluded that nothing was being spent and kept feeding work. The realised
cost per page for that run came out 16.5x the measured baseline, and the field
whose job was to say so said zero.

The healthy subset is still reported, under a name that says what it is. The
ratio of the two is how much of the bill is boot -- the figure that tells an
operator a ramp is too steep -- and it did not exist before.
"""
import pytest


def rates(pods: list) -> dict:
    """Mirror of the arithmetic in runpod_spend / api_runpod_stats."""
    live = sum((p.get("hourly_usd") or 0.0) for p in pods)
    serving = sum((p.get("hourly_usd") or 0.0) for p in pods if p.get("healthy"))
    return {"live_hourly_usd": round(live, 4),
            "live_hourly_serving_usd": round(serving, 4),
            "live_hourly_booting_usd": round(live - serving, 4)}


def _pod(usd, healthy, state="ready"):
    return {"hourly_usd": usd, "healthy": healthy, "state": state}


# --- the exact regression --------------------------------------------------

def test_a_fleet_entirely_in_boot_is_not_free():
    pods = [_pod(0.59, False, "provisioning") for _ in range(8)]
    out = rates(pods)
    assert out["live_hourly_usd"] == pytest.approx(4.72)
    assert out["live_hourly_serving_usd"] == 0.0
    assert out["live_hourly_booting_usd"] == pytest.approx(4.72)


def test_the_old_arithmetic_would_have_said_zero():
    """Guards the test: the understatement really was this shape."""
    pods = [_pod(0.59, False, "provisioning") for _ in range(8)]
    old = sum(p["hourly_usd"] for p in pods if p["healthy"])
    assert old == 0.0, "the regression must be reproducible"
    assert rates(pods)["live_hourly_usd"] == pytest.approx(4.72)


def test_the_13_pod_sample():
    """8 of 13 serving: the reported figure was $0.00 and the bill $7.67."""
    pods = ([_pod(0.59, True) for _ in range(8)]
            + [_pod(0.59, False, "provisioning") for _ in range(5)])
    out = rates(pods)
    assert out["live_hourly_usd"] == pytest.approx(7.67)
    assert out["live_hourly_serving_usd"] == pytest.approx(4.72)
    assert out["live_hourly_booting_usd"] == pytest.approx(2.95)


# --- the boot share is what makes a steep ramp visible ---------------------

def test_the_boot_share_is_reported_separately():
    pods = [_pod(1.0, True), _pod(1.0, False, "provisioning")]
    out = rates(pods)
    assert out["live_hourly_booting_usd"] == 1.0
    assert out["live_hourly_serving_usd"] == 1.0


def test_a_settled_fleet_reports_no_boot_cost():
    pods = [_pod(0.59, True) for _ in range(4)]
    out = rates(pods)
    assert out["live_hourly_usd"] == out["live_hourly_serving_usd"]
    assert out["live_hourly_booting_usd"] == 0.0


def test_an_empty_fleet_is_zero_everywhere():
    assert rates([]) == {"live_hourly_usd": 0.0,
                         "live_hourly_serving_usd": 0.0,
                         "live_hourly_booting_usd": 0.0}


def test_an_unhealthy_READY_pod_still_bills():
    """Sick is not free either. It is the same card on the same meter."""
    out = rates([_pod(0.59, False, "ready")])
    assert out["live_hourly_usd"] == pytest.approx(0.59)


def test_a_missing_rate_does_not_raise():
    """A status row is assembled from several sources; one of them being
    absent must not take the whole figure down."""
    out = rates([{"healthy": False}, _pod(0.59, True)])
    assert out["live_hourly_usd"] == pytest.approx(0.59)


# --- both endpoints that report it, and the pages that read it ------------

@pytest.mark.parametrize("path,marker", [
    ("codai/api/app.py", "async def runpod_spend"),
    ("codai/admin/routes.py", "async def api_runpod_stats"),
])
def test_neither_endpoint_filters_the_rate_by_health(path, marker):
    src = open(path).read()
    body = src[src.index(marker):]
    body = body[:body.index("live_uncommitted_usd")]
    flat = " ".join(body.split())          # the sums wrap; the shape is the point
    assert 'live_hourly = sum((p.get("hourly_usd") or 0.0) for p in pods)' in flat, path
    assert 'serving_hourly = sum((p.get("hourly_usd") or 0.0) for p in pods if p.get("healthy"))' in flat, \
        f"{path}: the serving subset must still be reported"


@pytest.mark.parametrize("path", [
    "codai/api/app.py", "codai/admin/routes.py",
])
def test_both_endpoints_report_the_split(path):
    src = open(path).read()
    assert '"live_hourly_serving_usd"' in src
    assert '"live_hourly_booting_usd"' in src


def test_the_admin_pages_say_how_much_of_the_rate_is_boot():
    """The number alone is an increase nobody can attribute; the point of the
    split is that a too-steep ramp becomes visible on the page."""
    rp = open("codai/admin/templates/runpod.html").read()
    assert "live_hourly_booting_usd" in rp
    assert "still booting" in rp
    tasks = open("codai/admin/templates/tasks.html").read()
    # "0 pod(s) · $7.67/hr" was the live reading, and it is self-contradictory.
    assert "pod(s) serving" in tasks
