"""Who the kernel kills when the host runs out of memory.

On 2026-10-10 a runaway test process took zeiss to 46 GB RSS plus 20 GB of swap.
The OOM killer then picked `coderai-nvidia` — not because it was at fault, but
because `oom_score` is essentially "how much memory does this task hold", and an
engine with a model loaded is always near the top of that list. The leak survived;
the server died; the box needed a hard reset.

The fix states the order explicitly, in three places that have to agree:

* the container gets a protected base (`--oom-score-adj`, set by the engine daemon
  because nothing inside the container has CAP_SYS_RESOURCE);
* engines step one tier ABOVE the front, so shedding an engine (which the front
  notices and respawns) always comes before shedding the front (which loses
  everything, including the ability to respawn anything);
* a training run — hours long, tens of GB, restartable — goes to the top.

The direction matters and is not symmetric: raising is unprivileged, lowering is
not. Every helper here can only ever make a process MORE killable, so a missing
privilege can never silently turn protection into exposure.
"""

import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codai.util import oom

RUNNER = (ROOT / "packaging/linux/run_oci.sh").read_text()
SUPERVISOR = (ROOT / "codai/frontproxy/engine_supervisor.py").read_text()
LORAS = (ROOT / "codai/api/loras.py").read_text()

HAS_PROC = pathlib.Path("/proc/self/oom_score_adj").exists()
needs_proc = pytest.mark.skipif(not HAS_PROC, reason="no /proc/self/oom_score_adj")


# --------------------------------------------------------------------- the order
def test_the_front_is_the_most_protected_tier():
    """It is the one process whose death loses everything."""
    assert oom.TIERS["front"] == 0
    assert oom.TIERS["engine"] > oom.TIERS["front"]
    assert oom.TIERS["worker"] > oom.TIERS["engine"]
    assert oom.TIERS["training"] >= oom.TIERS["worker"]


def test_the_tiers_stay_inside_the_kernels_range():
    """A base of -500 plus the largest tier must still be a legal value."""
    assert oom.ADJ_MIN == -1000 and oom.ADJ_MAX == 1000
    assert -500 + max(oom.TIERS.values()) <= oom.ADJ_MAX


# ------------------------------------------------------------- only ever upwards
@needs_proc
def test_a_tier_is_relative_to_whatever_base_the_container_was_given():
    """The base is -500 under docker and 0 under rootless podman, so a tier has to
    keep its DISTANCE from the front rather than name an absolute number."""
    script = """
import sys; sys.path.insert(0, %r)
from codai.util import oom
print(oom.current_adj()); oom.mark("engine"); print(oom.current_adj())
""" % str(ROOT)
    out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                         text=True, timeout=60).stdout.split()
    base, after = int(out[0]), int(out[1])
    assert after - base == oom.TIERS["engine"]


@needs_proc
def test_lowering_is_refused_rather_than_attempted():
    """Lowering needs CAP_SYS_RESOURCE. Asking and failing would leave a process
    that believes it is protected and is not."""
    assert oom.nudge(-100) is False
    assert oom.nudge(0) is False


@needs_proc
def test_a_nudge_that_would_pass_the_ceiling_is_clamped_not_an_error():
    script = """
import sys; sys.path.insert(0, %r)
from codai.util import oom
oom.nudge(900); oom.nudge(900); print(oom.current_adj())
""" % str(ROOT)
    out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                         text=True, timeout=60).stdout.strip()
    assert int(out) == oom.ADJ_MAX


@needs_proc
def test_the_tier_actually_reaches_the_kernel():
    """Not just bookkeeping: the number has to land in /proc and move the score."""
    script = """
import sys; sys.path.insert(0, %r)
from codai.util import oom
before = oom.current_score()
oom.mark("training")
print(oom.current_adj(), before, oom.current_score())
""" % str(ROOT)
    adj, before, after = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True,
        timeout=60).stdout.split()
    assert int(adj) == oom.TIERS["training"]
    assert int(after) > int(before), "a raised adj must raise the kernel's badness"


@needs_proc
def test_a_child_can_be_marked_from_the_parent():
    """How the training job is marked: a preexec_fn would run Python between fork
    and exec in a threaded server, which is the documented way to deadlock."""
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert oom.mark("worker", child.pid) is True
        assert oom.current_adj(child.pid) == oom.TIERS["worker"]
    finally:
        child.kill()
        child.wait(timeout=10)


def test_an_unreadable_proc_is_not_fatal():
    """Not Linux, or a hardened /proc: the server still has to start."""
    assert oom.current_adj(pid=2**30) is None
    assert oom.nudge(100, pid=2**30) is False
    assert "unavailable" in oom.describe(pid=2**30)


@needs_proc
def test_describe_says_which_side_of_neutral_a_process_is_on():
    assert "neutral" in oom.describe() or "expendable" in oom.describe()


# ----------------------------------------------------------- where it is applied
def test_the_container_gets_a_protected_base():
    """It has to be set by the ENGINE: the container has no CAP_SYS_RESOURCE, so
    nothing inside it could ever lower its own score."""
    assert 'OOM_SCORE_ADJ="${CODERAI_OOM_SCORE_ADJ:--500}"' in RUNNER
    assert 'args+=(--oom-score-adj "$OOM_SCORE_ADJ")' in RUNNER


def test_the_protection_can_be_turned_off():
    block = RUNNER.split('OOM_SCORE_ADJ="${CODERAI_OOM_SCORE_ADJ:--500}"')[1][:200]
    assert 'if [[ "$OOM_SCORE_ADJ" != "0" ]]' in block


def test_the_flag_is_on_the_run_not_the_upgrade():
    """The upgrade runs a throwaway container; protecting it would be meaningless."""
    assert "up_args+=(--oom-score-adj" not in RUNNER


def test_engines_are_marked_one_tier_above_the_front():
    """In the preexec hook, so the engine AND everything it spawns inherit it —
    the isolated model workers come from the engine, not from the front."""
    hook = SUPERVISOR.split("def _engine_preexec():")[1].split("\ndef ")[0]
    assert 'oom.mark("engine")' in hook


def test_the_engine_mark_cannot_break_a_spawn():
    """This runs between fork and exec. Anything that raises here fails the spawn,
    so the whole fleet would stop starting over a /proc write."""
    hook = SUPERVISOR.split("def _engine_preexec():")[1].split("\ndef ")[0]
    before, after = hook.split('oom.mark("engine")')
    assert "try:" in before and "except" not in before.rsplit("try:", 1)[1]
    assert after.lstrip().startswith("except Exception:")


def test_training_is_the_first_choice_of_victim():
    blk = LORAS.split('subprocess.Popen([py, script, "--job", job_path]')[1][:900]
    assert 'oom.mark("training", proc.pid)' in blk


def test_the_incident_is_recorded_where_the_number_is_chosen():
    """-500 is a judgement call; the next person deserves to know what it is for."""
    assert re.search(r"oom_score is essentially", RUNNER)
    assert "coderai-nvidia" in RUNNER
