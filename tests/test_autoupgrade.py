"""The unattended upgrade for a production orchestrator.

Two production installs now follow this repo: zeiss (nexlab) and the Digesta
host in Aruba. The Aruba one upgrades itself hourly from `production`, with no
one logged in -- so the failure modes are the ones nobody is watching for:

- restarting on every tick, because the host runner exits 0 BOTH when it
  upgraded and when the image was already current (run_oci.sh: the rc==10 branch
  prints "no upgrade applied" and then falls through to the same `exit 0`). The
  script must therefore decide by reading the version baked into the image
  before and after, never by the exit code.
- upgrading while CoderAI is serving. A request being served by a rented RunPod
  pod is the worst case: killing it loses the work AND keeps paying for the pod
  that was doing it. The check covers local engines and pods, and fails CLOSED.
- a mangled config. The file is SOURCED, so an unquoted
  `RESTART_CMD=systemctl --user restart coderai.service` assigns "systemctl" and
  runs the rest -- restarting nothing, failing the health check, and rolling back
  a perfectly good upgrade. Caught in testing; hence the echoed config line.
- a timer that never fires, because a user unit needs `loginctl enable-linger`.
"""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
SH = (ROOT / "packaging/linux/launcher/coderai-autoupgrade").read_text()
SERVICE = (ROOT / "packaging/linux/systemd/coderai-autoupgrade.service").read_text()
TIMER = (ROOT / "packaging/linux/systemd/coderai-autoupgrade.timer").read_text()
CONF = (ROOT / "packaging/linux/systemd/autoupgrade.conf.example").read_text()
RUNNER = (ROOT / "packaging/linux/run_oci.sh").read_text()


# ------------------------------------------------- why the exit code is not usable
def test_the_runner_really_does_collapse_both_outcomes():
    """The premise of the whole design. If this ever changes, the script could be
    simplified -- and if it changes silently, this test says so."""
    blk = RUNNER.split('if [[ "$UPGRADE" -eq 1 ]]; then')[1]
    assert 'elif [[ "$rc" -eq 10 ]]; then' in blk
    nothing = blk.split('elif [[ "$rc" -eq 10 ]]; then')[1].split("else")[0]
    assert "no upgrade applied" in nothing
    assert "exit" not in nothing, "a distinct exit code here would be observable"


def test_the_decision_comes_from_the_image_not_the_exit_code():
    assert "baked_version()" in SH
    assert 'BEFORE="$(baked_version)"' in SH
    assert 'AFTER="$(baked_version)"' in SH
    assert 'if [ "$AFTER" = "$BEFORE" ]' in SH


def test_nothing_to_do_means_no_restart():
    tail = SH.split('if [ "$AFTER" = "$BEFORE" ]; then')[1].split("\n    fi")[0]
    assert "already current" in tail
    assert "exit 0" in tail
    # The message may SAY "nothing to restart"; what must not appear is any way
    # of actually restarting.
    assert "restart_and_verify" not in tail
    assert "$RESTART_CMD" not in tail


# --------------------------------------------------------------- the idle guard
def test_the_upgrade_is_gated_on_idle_not_only_the_restart():
    """A busy host must never even reach the state where new code sits in the
    image waiting for a window."""
    body = SH.split("# ------------------------------------------------------------------- the upgrade")[1]
    gate = body.index("guard_idle || exit 0")
    upgrade = body.index('"$RUNNER" --upgrade')
    assert gate < upgrade, "the idle check must come before the upgrade runs"


def test_runpod_serving_counts_even_when_the_engines_are_idle():
    """The case the requirement names: all models remote, so the only in-flight
    work is on the pods."""
    assert "_probe_runpod()" in SH
    assert '"inflight"' in SH
    blk = SH.split("inflight(){")[1].split("guard_idle()")[0]
    assert '[ "$r" -gt "$best" ] && best="$r"' in blk, "the higher of the two must win"


def test_local_serving_counts_too():
    assert "coderai_engine_inflight" in SH


def test_an_unreadable_probe_skips_rather_than_upgrades():
    """Fail closed: an unreadable probe usually means a wrong token or URL, not
    an idle server."""
    blk = SH.split("guard_idle(){")[1].split("\n}")[0]
    assert 'if [ "$count" = unknown ]' in blk
    first = blk.split('if [ "$count" = unknown ]')[1]
    assert "SKIP" in first and "return 1" in first


def test_the_fail_closed_default_is_on():
    assert 'REQUIRE_IDLE_CHECK="${REQUIRE_IDLE_CHECK:-1}"' in SH


def test_a_missing_token_is_not_treated_as_idle():
    blk = SH.split("inflight(){")[1].split("guard_idle()")[0]
    assert 'echo "unknown no ADMIN_TOKEN configured"' in blk


def test_idle_is_re_checked_immediately_before_the_restart():
    """The upgrade takes a minute; a request can arrive while it runs, and the
    restart is the part that would kill it."""
    after = SH.split('printf \'%s -> %s\\n\' "$BEFORE" "$AFTER" > "$PENDING"')[1]
    assert "inflight)" in after
    assert after.index("inflight)") < after.index("restart_and_verify")


def test_by_default_a_live_request_is_never_interrupted():
    assert 'MAX_DEFERRALS="${MAX_DEFERRALS:-0}"' in SH
    blk = SH.split('if [ "$MAX_DEFERRALS" -le 0 ]; then')[1].split("fi")[0]
    assert "exit 0" in blk, "it must wait, not restart"


# ------------------------------------------------------------- safety on failure
def test_a_failed_upgrade_rolls_back():
    assert "ROLLBACK_TAG" in SH
    assert '"$ENGINE" tag "$ROLLBACK_TAG" "$IMAGE_TAG"' in SH


def test_the_rollback_point_is_taken_before_the_upgrade():
    body = SH.split("# ------------------------------------------------------------------- the upgrade")[1]
    assert body.index('"$ENGINE" tag "$IMAGE_TAG" "$ROLLBACK_TAG"') < body.index('"$RUNNER" --upgrade')


def test_health_is_verified_after_the_restart():
    assert "healthy()" in SH
    assert "HEALTH_TIMEOUT" in SH
    blk = SH.split("restart_and_verify(){")[1].split("\n}")[0]
    assert "if healthy; then return 0; fi" in blk


def test_a_service_left_down_is_said_plainly_and_exits_nonzero():
    tail = SH.split("# ------------------------------------------------------------------- the rollback")[1]
    assert "Needs a human" in tail
    assert "exit 1" in tail


def test_concurrent_runs_cannot_overlap():
    assert "flock -n 9" in SH
    blk = SH.split("flock -n 9")[1].split("fi")[0]
    assert "exit 0" in blk, "an overlapping run is a no-op, not a queue"


# --------------------------------------------------------- the config-file trap
def test_every_multi_word_value_in_the_example_is_quoted():
    """Unquoted, a sourced `KEY=two words` assigns one word and runs the rest."""
    for line in CONF.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        if " " in val.strip():
            assert val.strip().startswith('"') and val.strip().endswith('"'), line


def test_the_example_warns_about_the_quoting_trap():
    assert "QUOTE any value containing spaces" in CONF


def test_the_resolved_config_is_logged_every_run():
    """So a mangled value is visible in the log instead of causing a rollback."""
    assert 'say "config: restart=[$RESTART_CMD]' in SH


# ------------------------------------------------------------------ the schedule
def test_the_timer_runs_hourly():
    assert re.search(r"^OnCalendar=hourly$", TIMER, re.M)


def test_a_missed_check_is_caught_up_after_a_reboot():
    assert re.search(r"^Persistent=true$", TIMER, re.M)


def test_a_fleet_does_not_all_restart_on_the_same_minute():
    assert re.search(r"^RandomizedDelaySec=", TIMER, re.M)


def test_the_unit_is_a_oneshot_with_room_to_finish():
    assert re.search(r"^Type=oneshot$", SERVICE, re.M)
    m = re.search(r"^TimeoutStartSec=(\d+)$", SERVICE, re.M)
    assert m and int(m.group(1)) >= 1800, "a pip sync plus a health wait needs room"


def test_the_linger_requirement_is_documented_where_it_bites():
    """A user timer is silently inert after a reboot without it -- the one
    failure mode of this setup that leaves no trace anywhere."""
    assert "enable-linger" in SERVICE


# --------------------------------------------------------------------- hygiene
def test_an_hourly_job_does_not_grow_a_log_for_ever():
    assert "LOG_MAX_LINES" in SH
    assert "tail -n" in SH


def test_the_token_is_never_written_to_the_log():
    for line in SH.splitlines():
        if "say " in line or "tee -a" in line:
            assert "$ADMIN_TOKEN" not in line, line


def test_the_runner_is_told_which_engine_to_use():
    """run_oci.sh defaults to docker (ENGINE="${CONTAINER_ENGINE:-docker}"). On the
    rootless-podman host this script exists for, not passing it would make every
    tick look for a docker socket that is not there."""
    assert 'CONTAINER_ENGINE="$ENGINE"' in SH
    blk = SH.split('"$RUNNER" --upgrade')[0]
    assert blk.rstrip().endswith('CODERAI_UPGRADE_REF="$UPGRADE_REF" \\') or \
        'CONTAINER_ENGINE="$ENGINE" CODERAI_UPGRADE_REF' in SH


def test_the_runner_default_really_is_docker():
    """The premise above, asserted against the runner itself."""
    assert 'ENGINE="${CONTAINER_ENGINE:-docker}"' in RUNNER
