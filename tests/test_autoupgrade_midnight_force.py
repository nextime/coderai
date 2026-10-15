"""An install that is busy every hour never upgraded at all.

The timer's own comment promises "a restart only happens on a real upgrade, and
only when the fleet is idle (or after MAX_DEFERRALS hours of never being idle)".
The second half could not happen:

    guard_idle || exit 0          # exits here on a busy host
    ...upgrade...
    N=$(cat "$DEFERRALS" ...)     # the counter only ever incremented HERE

So on a busy host the script left before the image was even pulled, the counter
the force depends on never advanced, and `max-deferrals` was unreachable.
Observed on the Digesta orchestrator 2026-10-14: 24 skips in one day, two
versions behind, `deferrals` still 0, and 0.3.68 only landed because the host
happened to be idle at 17:01.

Now a busy skip counts, and at MAX_DEFERRALS the script BOOKS a forced upgrade
for the next midnight -- the hourly timer already ticks at 00:00, so there is no
second timer to install and nothing to clean up after a reboot.
"""

import os
import pathlib
import subprocess
import sys
import time

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "packaging/linux/launcher/coderai-autoupgrade"


def _harness(state_dir, body, max_deferrals=3, now_env=None):
    """Run `body` against the script's real helper functions.

    The helpers are extracted rather than reimplemented: a test that restates
    the logic proves only that the test agrees with itself.
    """
    src = SCRIPT.read_text()
    funcs = []
    for name in ("arm_midnight_force", "force_due"):
        start = src.index(f"{name}(){{")
        # to the first line that is exactly "}" at column 0
        end = src.index("\n}\n", start) + len("\n}\n")
        funcs.append(src[start:end])
    script = "\n".join([
        "set -u",
        f'STATE_DIR="{state_dir}"',
        'FORCE_AT="$STATE_DIR/force-upgrade-at"',
        'DEFERRALS="$STATE_DIR/deferrals"',
        f'MAX_DEFERRALS={max_deferrals}',
        'say(){ printf "%s\\n" "$*"; }',
        *funcs,
        body,
    ])
    env = dict(os.environ)
    if now_env:
        env.update(now_env)
    return subprocess.run(["bash", "-c", script], capture_output=True,
                          text=True, env=env)


def test_the_script_is_valid_shell():
    r = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# ------------------------------------------------------------ the booking #
def test_arming_books_the_next_midnight(tmp_path):
    r = _harness(tmp_path, 'arm_midnight_force 3')
    assert r.returncode == 0, r.stderr
    at = int((tmp_path / "force-upgrade-at").read_text().strip())
    # Midnight local, in the future, within the next 24h.
    now = time.time()
    assert now < at <= now + 86400 + 60
    lt = time.localtime(at)
    assert (lt.tm_hour, lt.tm_min, lt.tm_sec) == (0, 0, 0), time.ctime(at)


def test_arming_says_when_and_how_to_cancel(tmp_path):
    """An operator must be able to see a forced restart coming, and stop it."""
    r = _harness(tmp_path, 'arm_midnight_force 3')
    out = r.stdout
    assert "FORCED upgrade" in out
    assert "--cancel-force" in out
    assert "restart the service even with work in flight" in out


def test_a_booking_is_not_due_before_its_time(tmp_path):
    (tmp_path / "force-upgrade-at").write_text(str(int(time.time()) + 3600))
    r = _harness(tmp_path, 'force_due && echo DUE || echo NOT_DUE')
    assert "NOT_DUE" in r.stdout


def test_a_booking_is_due_once_its_time_has_passed(tmp_path):
    (tmp_path / "force-upgrade-at").write_text(str(int(time.time()) - 5))
    r = _harness(tmp_path, 'force_due && echo DUE || echo NOT_DUE')
    assert "DUE" in r.stdout


def test_no_booking_is_never_due(tmp_path):
    r = _harness(tmp_path, 'force_due && echo DUE || echo NOT_DUE')
    assert "NOT_DUE" in r.stdout


def test_a_corrupt_booking_is_discarded_not_obeyed(tmp_path):
    """Garbage must not read as "the deadline passed" and force a restart, nor
    sit there forever blocking a fresh booking."""
    f = tmp_path / "force-upgrade-at"
    f.write_text("not-a-timestamp")
    r = _harness(tmp_path, 'force_due && echo DUE || echo NOT_DUE')
    assert "NOT_DUE" in r.stdout
    assert not f.exists(), "a corrupt booking was left in place"


# ------------------------------------------------- the counter that was dead #
def test_a_busy_skip_now_counts():
    """The bug itself: this path used to return without touching the counter."""
    src = SCRIPT.read_text()
    block = src.split('say "SKIP: $count piece(s) of work in progress')[1] \
               .split("return 1")[0]
    assert 'echo "$n" > "$DEFERRALS"' in block, "a busy skip still does not count"
    assert "arm_midnight_force" in block


def test_the_counter_survives_a_junk_file():
    src = SCRIPT.read_text()
    block = src.split('say "SKIP: $count piece(s) of work in progress')[1] \
               .split("return 1")[0]
    assert "*[!0-9]*" in block, "a non-numeric deferrals file would break arithmetic"


def test_arming_happens_once_not_every_tick():
    src = SCRIPT.read_text()
    block = src.split('say "SKIP: $count piece(s) of work in progress')[1] \
               .split("return 1")[0]
    assert '[ ! -f "$FORCE_AT" ]' in block, (
        "every busy tick would re-book, pushing the deadline forward forever")


def test_zero_max_deferrals_never_books():
    """0 keeps the old meaning -- wait for an idle moment, however long -- so an
    install that did not opt in is unchanged."""
    src = SCRIPT.read_text()
    block = src.split('say "SKIP: $count piece(s) of work in progress')[1] \
               .split("return 1")[0]
    assert '[ "$MAX_DEFERRALS" -gt 0 ]' in block


# --------------------------------------------------------------- the firing #
def test_a_due_booking_skips_the_idle_guard():
    src = SCRIPT.read_text()
    flow = src.split("if [ -f \"$PENDING\" ]; then")[1]
    due = flow.index("if force_due; then")
    guard = flow.index("guard_idle || exit 0")
    assert due < guard, "the idle guard still runs before the booking is checked"


def test_a_forced_run_does_not_defer_the_restart():
    """Deferring there would put us back where the booking was made."""
    src = SCRIPT.read_text()
    assert 'if [ "${FORCED:-0}" = 1 ]; then' in src
    tail = src.split('if [ "${FORCED:-0}" = 1 ]; then')[1]
    assert "forced: restarting with work in flight" in tail


def test_a_landed_upgrade_clears_the_booking():
    """So the next busy stretch counts from zero instead of inheriting a
    deadline from the last one."""
    src = SCRIPT.read_text()
    assert 'rm -f "$PENDING" "$DEFERRALS" "$FORCE_AT"' in src


def test_the_banner_announces_a_pending_booking():
    src = SCRIPT.read_text()
    assert "a FORCED upgrade is booked for" in src


def test_the_force_file_is_named_in_the_state_dir():
    src = SCRIPT.read_text()
    assert 'FORCE_AT="$STATE_DIR/force-upgrade-at"' in src
