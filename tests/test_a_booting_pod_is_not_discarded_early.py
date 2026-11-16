"""A normal boot must outlive the port-publish grace.

The grace was 120s after the container starts, so a rental was declared
unreachable at roughly 360s (pull, start, grace, give up) -- while the only
pod on this install that ever became reachable took 397.9s. The threshold sat
BELOW the single successful observation, so ordinary boots were discarded
about a minute before they were ready, and every discarded pod was already
being billed.

Raising it costs nothing on success: the card bills while it boots either way.
The grace still earns its keep against a host that publishes only RunPod's own
agent port -- that one is abandoned at ~10 minutes rather than waiting out a
1800s budget (measured 2026-11-08: nine such pods, $1.21).
"""

import pytest

from codai.api import runpod_client as rc


MEASURED_GOOD_BOOT_S = 397.9      # the one that worked
MEASURED_EARLY_ABORT_S = 360.0    # when the old grace gave up


def _info(uptime, ports=(), status="RUNNING"):
    return {"status": status, "uptime_s": uptime, "ports": list(ports),
            "ready": False}


AGENT_PORT_ONLY = [{"ip": "100.65.0.9", "isIpPublic": False,
                    "privatePort": 19123, "publicPort": 19123, "type": "http"}]


def test_the_grace_exceeds_the_only_successful_boot():
    """The property that was violated, stated as a number."""
    assert rc.PORT_PUBLISH_GRACE_S > MEASURED_GOOD_BOOT_S


def test_the_grace_exceeds_the_old_abort_point():
    assert rc.PORT_PUBLISH_GRACE_S > MEASURED_EARLY_ABORT_S


def test_a_pod_at_the_old_abort_point_is_still_waited_for():
    verdict, _why = rc.classify_boot(_info(MEASURED_EARLY_ABORT_S,
                                           AGENT_PORT_ONLY), 8000, True)
    assert verdict == "pulling", "a boot in progress was called unreachable"


def test_the_pod_that_worked_would_not_have_been_discarded():
    verdict, _why = rc.classify_boot(_info(MEASURED_GOOD_BOOT_S,
                                           AGENT_PORT_ONLY), 8000, True)
    assert verdict != "unreachable"


def test_a_host_that_never_publishes_is_still_given_up_on():
    """The grace is longer, not gone: this is what it is for."""
    verdict, why = rc.classify_boot(_info(rc.PORT_PUBLISH_GRACE_S + 60,
                                          AGENT_PORT_ONLY), 8000, True)
    assert verdict == "unreachable"
    assert "19123" in why


def test_it_gives_up_well_before_the_boot_budget():
    """Otherwise a dud costs a full 1800s of billing per attempt."""
    assert rc.PORT_PUBLISH_GRACE_S < 1800.0


def test_the_grace_is_still_overridable():
    import inspect
    src = inspect.getsource(rc)
    assert "CODERAI_RUNPOD_PORT_GRACE" in src


def test_an_explicit_grace_still_wins():
    verdict, _why = rc.classify_boot(_info(200, AGENT_PORT_ONLY), 8000, True,
                                     grace_s=100)
    assert verdict == "unreachable"


def test_a_published_port_is_not_delayed_by_the_grace():
    """A pod that IS ready must be used immediately, whatever the grace."""
    good = [{"ip": "194.68.245.178", "isIpPublic": True, "privatePort": 8000,
             "publicPort": 41234, "type": "tcp"}]
    verdict, _why = rc.classify_boot(_info(30, good), 8000, True)
    assert verdict != "unreachable" and verdict != "pulling"


# ------------------------------------- configurable from the web interface --
#
# A timeout that decides whether a paid boot is kept or binned belongs in the
# interface, not in a module constant or an env var. Three places have to
# agree or the knob is a lie: the dataclass, the form (input + getter +
# setter) and the hand-listed save merge. A field missing from the merge saves
# in the form and comes back as the default on reload.

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_the_account_config_carries_it():
    from codai.config import RunpodConfig
    assert RunpodConfig().port_publish_grace_s == 600.0


def test_the_client_reads_it_from_the_config():
    import inspect
    from codai.api.runpod_client import RunpodClient
    src = inspect.getsource(RunpodClient.wait_ready)
    assert "port_publish_grace_s" in src
    assert "grace_s=" in src


def test_zero_means_the_built_in_default():
    """So clearing the box does not mean "give up instantly"."""
    import inspect
    from codai.api.runpod_client import RunpodClient
    src = inspect.getsource(RunpodClient.wait_ready)
    assert "if _grace > 0" in src


def test_the_form_has_an_input_a_getter_and_a_setter():
    html = (ROOT / "codai/admin/templates/settings.html").read_text()
    assert html.count("s-runpod-port-grace") == 3, (
        "an input without a getter shows blank; without a setter it never saves")


def test_the_save_merge_keeps_it():
    """The merge is hand-listed: a field missing here is a knob that appears to
    save and silently reverts."""
    src = (ROOT / "codai/admin/routes.py").read_text()
    # Anchored on `rp = c.runpod`, not on `if "runpod" in data:` -- that
    # string appears more than once in this module and the first hit is 96k
    # characters from the merge.
    i = src.index("rp = c.runpod")
    block = src[i:i + 4000]
    assert '"port_publish_grace_s" in d' in block
    assert "rp.port_publish_grace_s" in block


def test_the_hint_tells_the_operator_what_it_costs():
    html = (ROOT / "codai/admin/templates/settings.html").read_text()
    i = html.index("s-runpod-port-grace")
    hint = html[i:i + 700]
    assert "already paid for" in hint


def test_the_settings_getter_returns_it():
    """The SETTER alone is not enough, and I shipped only the setter once.

    Both the GET payload and the save merge are hand-listed in routes.py. A
    field in the merge but not the payload is a knob that appears to save and
    comes back blank on reload -- and a blank numeric input then saves 0 over
    the real value on the next submit.
    """
    src = (ROOT / "codai/admin/routes.py").read_text()
    i = src.index('"registry_auth_id": c.runpod.registry_auth_id')
    block = src[i:i + 900]
    assert '"port_publish_grace_s": c.runpod.port_publish_grace_s' in block


def test_both_halves_exist_for_this_field():
    """Stated as one assertion so neither half can be removed alone."""
    src = (ROOT / "codai/admin/routes.py").read_text()
    assert '"port_publish_grace_s": c.runpod.port_publish_grace_s' in src   # GET
    assert '"port_publish_grace_s" in d' in src                             # POST
