"""Who the kernel should kill first when the host runs out of memory.

On 2026-10-10 a runaway test process on zeiss took the host to 46 GB RSS plus 20 GB
of swap. The kernel's OOM killer then picked `coderai-nvidia` — not because it was
at fault, but because `oom_score` is essentially "how much memory does this task
hold", and an engine holding a model is always near the top of that list. The thing
that leaked survived; the thing that was serving died, and the box needed a hard
reset.

So the ordering has to be stated explicitly. Linux offers exactly one lever,
``/proc/<pid>/oom_score_adj`` (-1000..1000), added to the score the kernel computes:

    front / supervisor   protected   the one process whose death loses everything
    engines              neutral     losing one loses its in-flight work; the front
                                     notices and respawns it
    model workers,       expendable  minutes of work at most, restartable, and the
    training                         usual suspects for holding tens of GB

Two kernel rules shape what is possible, and both were verified on the two
production hosts rather than assumed:

* **Raising is free, lowering is privileged.** Any process may make itself a more
  attractive victim; moving the other way needs CAP_SYS_RESOURCE. A container
  started with ``--oom-score-adj`` inherits the protected value from the engine
  daemon, and in-container processes can then only step UP from it — which is
  exactly the direction this module needs.
* **Rootless cannot protect at all.** Under rootless podman a negative value is
  clamped to 0. 0 still beats the +200 a user service inherits from systemd's user
  manager, so the flag is worth passing there too, but the real protection on such
  a host comes from making the expendable processes expendable.

Everything here is best-effort and silent on failure: a process that cannot adjust
its score must still start and serve.
"""

from __future__ import annotations

import os

# Steps, relative to whatever the container was given. Deliberately coarse: the
# point is a strict order between the three tiers, not a tuned number. Engines sit
# one step above the front so that shedding an engine (recoverable) always comes
# before shedding the front (not recoverable).
TIERS = {
    "front": 0,
    "engine": 100,
    "worker": 300,
    "training": 500,
}

ADJ_MIN, ADJ_MAX = -1000, 1000


def _path(pid=None) -> str:
    return f"/proc/{'self' if pid is None else int(pid)}/oom_score_adj"


def current_adj(pid=None):
    """This (or another) process's oom_score_adj, or None where /proc has none."""
    try:
        with open(_path(pid)) as handle:
            return int(handle.read().strip())
    except Exception:
        return None


def current_score(pid=None):
    """The kernel's own badness figure, for logging. Higher = killed sooner."""
    try:
        with open(f"/proc/{'self' if pid is None else int(pid)}/oom_score") as handle:
            return int(handle.read().strip())
    except Exception:
        return None


def nudge(delta: int, pid=None) -> bool:
    """Make a process MORE killable by ``delta``, relative to where it is now.

    Relative, because the base is set outside this process (``--oom-score-adj`` on
    the container) and a tier should keep its distance from the front whatever that
    base turns out to be: -500 on a docker host, 0 on a rootless one.

    Never lowers. A negative ``delta`` would need a privilege the server does not
    have and should not want, and silently failing to protect something is worse
    than not trying: returns False instead.
    """
    if delta <= 0:
        return False
    base = current_adj(pid)
    if base is None:
        return False
    want = max(ADJ_MIN, min(ADJ_MAX, base + int(delta)))
    if want <= base:
        return False                      # already at the ceiling
    try:
        with open(_path(pid), "w") as handle:
            handle.write(f"{want}\n")
    except Exception:
        return False
    return current_adj(pid) == want


def mark(tier: str, pid=None) -> bool:
    """Put a process in one of the TIERS, relative to the front's base."""
    return nudge(TIERS.get(tier, 0), pid)


def preexec(tier: str):
    """A ``preexec_fn`` that marks the CHILD, after fork and before exec.

    Done in the child so the parent's own score is untouched, and so the value is
    already in place for whatever the child goes on to spawn: workers inherit it
    from the engine that started them, which is how one call here covers a tree.
    """
    delta = TIERS.get(tier, 0)

    def _apply():
        if delta > 0:
            try:
                nudge(delta)
            except Exception:
                pass

    return _apply


def describe(pid=None) -> str:
    """One line for a log: where this process sits and how exposed it is."""
    adj, score = current_adj(pid), current_score(pid)
    if adj is None:
        return "oom: /proc/self/oom_score_adj unavailable"
    where = "protected" if adj < 0 else "neutral" if adj == 0 else "expendable"
    return f"oom: adj={adj:+d} ({where}), kernel score={score}"
