"""An operator has to be able to say "upgrade now", without editing state files.

The unattended policy waits for an idle moment and, on a host that never has
one, books a forced upgrade for the next midnight. Neither branch is an operator
asking for the upgrade NOW. Before these options the only way to ask was to
write an epoch into $STATE_DIR/force-upgrade-at by hand: undocumented, easy to
get wrong, and it leaves a booking behind that forces every LATER run too --
which is exactly what happened on a production orchestrator, where a hand-made
`1` turned every hourly tick into a forced one for a day.

These tests drive the real script. --help, --force-at and --cancel-force all
return before the runner check and before anything touches a container engine,
so they are safe to run anywhere.
"""

import os
import pathlib
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "packaging/linux/launcher/coderai-autoupgrade"


def _run(state_dir, *args):
    env = dict(os.environ)
    env["CODERAI_AUTOUPGRADE_CONF"] = str(state_dir / "no-such.conf")
    env["STATE_DIR"] = str(state_dir)
    return subprocess.run([str(SCRIPT), *args], capture_output=True, text=True,
                          env=env, timeout=60)


def _booking(state_dir):
    f = state_dir / "force-upgrade-at"
    return int(f.read_text().strip()) if f.exists() else None


# ------------------------------------------------------------------- help #
def test_help_lists_every_force_option():
    r = subprocess.run([str(SCRIPT), "--help"], capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0, r.stderr
    for flag in ("--force", "--force-at", "--cancel-force"):
        assert flag in r.stdout, flag


def test_an_unknown_option_is_refused_rather_than_ignored():
    """Silently ignoring it would run the UNATTENDED policy under a typo."""
    r = subprocess.run([str(SCRIPT), "--forse"], capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 2
    assert "--forse" in r.stderr


# ---------------------------------------------------------------- booking #
def test_force_at_books_the_time_it_was_given(tmp_path):
    r = _run(tmp_path, "--force-at", "tomorrow 02:00")
    assert r.returncode == 0, r.stderr
    at = _booking(tmp_path)
    assert at is not None
    lt = time.localtime(at)
    assert (lt.tm_hour, lt.tm_min) == (2, 0), time.ctime(at)
    # "tomorrow 02:00" is between 2 and 26 hours away depending on the hour the
    # suite runs. The first version of this asserted "within 24h" and went red
    # at 00:22 — the test was wrong about the clock, not the code about the
    # booking. What matters is that it is in the future and is tomorrow.
    assert time.time() < at <= time.time() + 2 * 86400
    assert lt.tm_mday == time.localtime(time.time() + 86400).tm_mday


def test_force_at_takes_a_relative_time(tmp_path):
    r = _run(tmp_path, "--force-at", "+2 hours")
    assert r.returncode == 0, r.stderr
    at = _booking(tmp_path)
    assert abs(at - (time.time() + 7200)) < 120


def test_force_at_refuses_a_time_it_cannot_parse(tmp_path):
    """Writing garbage would book a deadline nothing can read -- and force_due
    silently DROPS an unparsable booking, so the upgrade would never land and
    nothing would say why."""
    r = _run(tmp_path, "--force-at", "two tuesdays hence")
    assert r.returncode == 1
    assert _booking(tmp_path) is None
    assert "not a time" in r.stdout


def test_force_at_needs_an_argument(tmp_path):
    r = _run(tmp_path, "--force-at")
    assert r.returncode == 2
    assert _booking(tmp_path) is None


def test_force_at_accepts_the_equals_form(tmp_path):
    assert _run(tmp_path, "--force-at=+1 hour").returncode == 0
    assert _booking(tmp_path) is not None


# ----------------------------------------------------------------- cancel #
def test_cancel_force_removes_a_booking(tmp_path):
    _run(tmp_path, "--force-at", "+3 hours")
    assert _booking(tmp_path) is not None
    r = _run(tmp_path, "--cancel-force")
    assert r.returncode == 0, r.stderr
    assert _booking(tmp_path) is None
    assert "CANCELLED" in r.stdout


def test_cancel_force_on_a_host_with_no_booking_is_not_an_error(tmp_path):
    """An operator clearing a booking they are not sure exists must not get a
    failure -- a timer wrapper would report it as a broken upgrade."""
    r = _run(tmp_path, "--cancel-force")
    assert r.returncode == 0
    assert "nothing to cancel" in r.stdout


def test_cancel_force_clears_a_booking_this_script_made(tmp_path):
    """The midnight booking and an operator's booking are the same file, so one
    command cancels either."""
    (tmp_path / "force-upgrade-at").write_text(str(int(time.time()) + 600))
    assert _run(tmp_path, "--cancel-force").returncode == 0
    assert _booking(tmp_path) is None


# ---------------------------------------------- the booking edits are logged #
def test_a_schedule_change_is_written_to_the_upgrade_log(tmp_path):
    """The log is the history of what this host did. A booking made or cancelled
    by hand belongs in the same place as the runs it changes."""
    _run(tmp_path, "--force-at", "+4 hours")
    _run(tmp_path, "--cancel-force")
    log = (tmp_path / "autoupgrade.log").read_text()
    assert "booked" in log
    assert "CANCELLED" in log


# --------------------------------------------------- what --force wires into #
def test_force_sets_the_same_flag_the_booking_sets():
    """--force must reach the RESTART step too, not only the upgrade step. A
    host whose upgrade has sat in the image behind deferrals is the case the
    operator types --force for, and that path skips the upgrade block entirely.
    """
    src = SCRIPT.read_text()
    assert 'FORCED="$FORCE_NOW"' in src


def test_force_leaves_no_booking_behind():
    """Taking the upgrade now settles any deadline made for the same upgrade;
    leaving it would force the next run as well, with nobody having asked."""
    src = SCRIPT.read_text()
    i = src.index('FORCED UPGRADE (--force)')
    # the clear sits just above the announcement, inside the same branch
    assert 'rm -f "$FORCE_AT"' in src[i - 400:i]
