"""The unattended upgrade has to recover a host that came back with no server.

Two failure modes, both found the hard way on 2026-10-10 after zeiss was hard-reset:

1. **A server that is DOWN read as "cannot tell".** Both idle probes are HTTP, so a
   stopped container is indistinguishable from a wrong token: inflight() said
   "unknown", the fail-closed guard refused to act, and every hourly run skipped —
   leaving the host on old code and, worse, still down. Nothing in the loop would
   ever have recovered it. A container that is not running cannot be serving, and
   the restart that follows an upgrade is also what brings it back.

2. **The conf silently beat the environment.** Every knob reads
   `VAR="${VAR:-default}"`, which looks like "the environment may override" — but
   the conf is sourced first and assigns unconditionally, so
   `MAX_DEFERRALS=1 coderai-autoupgrade` did nothing at all, with no line in the log
   to say why. An operator passing a variable for one run is overriding the
   unattended policy on purpose.

Unlike its sibling file, this one RUNS the script: the env-restore is eval-based and
the container check branches on an exit code, and neither is something reading the
source can confirm. The container engine, the host runner, the restart command and
the health endpoint are all stubbed, so nothing here touches a real install.
"""

import os
import pathlib
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "packaging/linux/launcher/coderai-autoupgrade"
SH = SCRIPT.read_text()

STUB_ENGINE = """#!/usr/bin/env bash
# A fake container engine.
#   STUB_RUNNING  space-separated names of "running" containers
#   STUB_BROKEN=1 the engine cannot be asked at all (daemon down, wrong socket)
case "${1:-}" in
    ps)
        [ "${STUB_BROKEN:-0}" = 1 ] && { echo "cannot connect to the daemon" >&2; exit 1; }
        case "${2:-}" in
            --quiet) exit 0 ;;                 # no ancestor match, ever
            *) for n in ${STUB_RUNNING:-}; do echo "$n"; done; exit 0 ;;
        esac ;;
    run)   echo "__version__ = \\"$(cat "$STUB_VERSION_FILE" 2>/dev/null || echo 0.0.0)\\""
           exit 0 ;;
    tag|image) exit 0 ;;
esac
exit 0
"""

# The upgrade, observable: it writes the new version the engine then reports.
STUB_RUNNER = '#!/bin/sh\necho "$STUB_NEW_VERSION" > "$STUB_VERSION_FILE"\nexit 0\n'
STUB_RESTART = '#!/bin/sh\necho RESTARTED >> "$STUB_RESTART_LOG"\nexit 0\n'


@pytest.fixture
def install(tmp_path):
    """A throwaway install: stub engine/runner/restart, a conf, and a health file."""

    def _write(name, body, mode=0o755):
        path = tmp_path / name
        path.write_text(body)
        path.chmod(mode)
        return path

    engine = _write("engine", STUB_ENGINE)
    runner = _write("runner", STUB_RUNNER)
    restart = _write("restart", STUB_RESTART)
    health = tmp_path / "health.json"
    health.write_text('{"ok": true}\n')      # curl reads file:// happily
    version = tmp_path / "version"
    version.write_text("0.3.3\n")
    conf = tmp_path / "autoupgrade.conf"
    # What both production installs carry: never interrupt work, fail closed.
    conf.write_text('MAX_DEFERRALS="0"\nREQUIRE_IDLE_CHECK="1"\nADMIN_TOKEN=""\n')

    def run(**env):
        base = {
            "HOME": str(tmp_path / "home"),
            "PATH": "/usr/bin:/bin",
            "CODERAI_AUTOUPGRADE_CONF": str(conf),
            "ENGINE": str(engine),
            "RUNNER": str(runner),
            "IMAGE_TAG": "testimg:latest",
            "STATE_DIR": str(tmp_path / "state"),
            "LOG": str(tmp_path / "state" / "log"),
            "RESTART_CMD": str(restart),
            "HEALTH_URL": f"file://{health}",
            "STUB_VERSION_FILE": str(version),
            "STUB_NEW_VERSION": "0.3.5",
            "STUB_RESTART_LOG": str(tmp_path / "restarts"),
        }
        base.update({k: str(v) for k, v in env.items()})
        proc = subprocess.run(["bash", str(SCRIPT)], env=base, text=True,
                              capture_output=True, timeout=120)
        return proc.stdout + proc.stderr

    run.tmp = tmp_path
    run.restarts = tmp_path / "restarts"
    run.version = version
    return run


def _restarted(install):
    return install.restarts.exists() and "RESTARTED" in install.restarts.read_text()


# ------------------------------------------------- a server that is not running
def test_a_stopped_container_is_idle_not_unknown():
    """The source side of it: the check has to come BEFORE the token check, because
    a host whose server is down is in the same position as one with no token."""
    blk = SH.split("inflight(){")[1].split("guard_idle()")[0]
    assert blk.index("container_state") < blk.index("no ADMIN_TOKEN configured")


def test_a_host_that_came_back_with_no_server_upgrades_and_starts_it(install):
    """The whole point: unattended recovery, with no human noticing anything."""
    out = install(STUB_RUNNING="")
    assert "idle (no 'coderai' container running" in out
    assert "image upgraded 0.3.3 -> 0.3.5" in out
    assert "OK: running 0.3.5 and healthy" in out
    assert _restarted(install), "the restart is what brings the server back"


def test_a_running_server_with_unreadable_probes_still_fails_closed(install):
    """The protection that must survive the fix: a live request is worth more than
    an up-to-date image, and an unreadable probe usually means a wrong token."""
    out = install(STUB_RUNNING="coderai")
    assert "SKIP: cannot tell whether CoderAI is serving" in out
    assert "image upgraded" not in out
    assert not _restarted(install)


def test_an_engine_that_cannot_be_asked_is_not_assumed_idle(install):
    """Only a POSITIVE "not there" counts. A daemon that is down tells us nothing
    about the server, so the fail-closed guard still applies."""
    out = install(STUB_BROKEN="1", STUB_RUNNING="coderai")
    assert "SKIP: cannot tell whether CoderAI is serving" in out
    assert not _restarted(install)


def test_an_install_that_names_its_container_differently_is_covered(install):
    out = install(CONTAINER_NAME="digesta-coderai", STUB_RUNNING="something-else")
    assert "container=digesta-coderai" in out
    assert "idle (no 'digesta-coderai' container running" in out


def test_the_container_it_looks_for_is_in_the_log(install):
    """A name mismatch would otherwise look exactly like a stopped server."""
    assert "container=coderai" in install(STUB_RUNNING="coderai")


# --------------------------------------------------------- environment vs config
def test_the_environment_overrides_the_config_file(install):
    """`MAX_DEFERRALS=5 coderai-autoupgrade` used to do nothing whatsoever."""
    assert "max-deferrals=5" in install(MAX_DEFERRALS=5, STUB_RUNNING="coderai")


def test_the_config_file_still_wins_when_the_environment_is_silent(install):
    """The unattended path is the normal one; this fix must not invert it."""
    assert "max-deferrals=0" in install(STUB_RUNNING="coderai")


def test_the_idle_policy_can_be_lifted_from_the_environment(install):
    """The escape hatch an operator reaches for when the probe is broken but they
    can see for themselves that nothing is running."""
    out = install(REQUIRE_IDLE_CHECK=0, STUB_BROKEN="1", STUB_RUNNING="coderai")
    assert "require-idle=0" in out
    assert "WARN: idle check unavailable" in out
    assert "image upgraded 0.3.3 -> 0.3.5" in out


def test_both_decision_knobs_are_logged_every_run(install):
    """"It skipped again" and "it was told to skip" used to look identical."""
    out = install(STUB_RUNNING="coderai")
    assert "require-idle=" in out and "max-deferrals=" in out


def test_the_snapshot_covers_every_knob_the_script_resolves():
    """A knob added to the defaults but not to the snapshot list is a knob the conf
    keeps silently overriding — the bug this fixed, one variable at a time."""
    import re

    resolved = set(re.findall(r'^([A-Z][A-Z0-9_]+)="\$\{\1:-', SH, re.M))
    listed = set(SH.split("AUTOUPGRADE_TUNABLES=\"")[1].split('"')[0].split())
    # ENGINE is resolved by a command-v probe rather than the ${VAR:-} idiom, and
    # is in the list; the reverse direction is what matters here.
    missing = resolved - listed
    assert not missing, f"not overridable from the environment: {sorted(missing)}"


def test_the_token_is_never_echoed_by_the_new_lines():
    for line in SH.splitlines():
        if "say " in line:
            assert "$ADMIN_TOKEN" not in line, line
