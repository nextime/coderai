# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Picking a distributed text-LoRA run up where it stopped.

A distributed run already wrote the one expensive thing it produces -- the
per-round average, `round{N}.safetensors` -- and then could not use it. The
weights were on disk; the record of WHICH EXAMPLES had been trained was an
in-memory set, so a resubmit reshuffled and started from block 0. On a job
that had spent four hours that is four hours bought twice.

So the state this module persists is deliberately small. Two things make a
resume, and they are not the same thing:

* **the adapter** -- the last round's average, which is the training itself;
* **the coverage** -- which blocks of the seeded order have been trained,
  without which the fleet retrains ground it has already paid for.

A checkpoint carrying only the first is still worth having, and is what a run
interrupted before its first round close can offer. A checkpoint carrying
neither is not a resume and this module says so rather than starting a fresh
run under the name of a continuation.

**Why the identity block is as long as it is.** A block id means nothing by
itself: block 7 is `shuffle(seed)[7*per:(7+1)*per]`, so the examples it names
depend on the seed, on how many examples were encoded, and on the block size
-- which is itself `ceil(n / (rounds * planned_pods))`. Resume a checkpoint
against a different dataset, seed, round count or pod count and `covered`
still loads, still looks plausible, and skips examples that were never
trained while retraining others. Nothing downstream would notice. So every
input that block identity depends on is recorded and compared, and a
mismatch is refused with the field that differs named.

The same argument applies to the adapter: averaging a continuation on top of
weights trained at a different rank, or for another customer's session, is an
adapter that loads and serves and is not what anybody trained. Those are
ownership checks, in the sense :mod:`codai.api.loras`' diffusion paths already
use the word.
"""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
import time
from typing import Any, Dict, List, Optional

#: The checkpoint's filename inside the run's output directory. Beside the
#: averages it describes, because the pair only means anything together.
RESUME_FILE = "resume.json"

#: Bumped when a field's MEANING changes, not when one is added. An older
#: checkpoint with fewer fields resumes; one whose numbers would be read
#: differently must not.
RESUME_VERSION = 1

#: Fields that must match for a checkpoint to be resumable at all, mapped to
#: how to say it in a refusal. Order is the order a human would want them
#: checked in: what was trained, then what it was trained into, then the
#: arithmetic that gives block ids their meaning.
_IDENTITY: List[tuple] = [
    ("dataset_id", "the dataset"),
    ("base", "the base model"),
    ("target", "the target"),
    ("session", "the session"),
    ("rank", "the LoRA rank"),
    ("alpha", "the LoRA alpha"),
    ("dropout", "the LoRA dropout"),
    ("max_len", "the sequence limit"),
    ("seed", "the shuffle seed"),
    ("n_examples", "the encoded example count"),
    ("block_size", "the block size"),
]


class ResumeError(RuntimeError):
    """A checkpoint cannot be resumed, with what differs or is missing."""


@dataclasses.dataclass
class ResumeState:
    """What a stopped run leaves behind that its successor can use.

    The identity half is compared before anything is adopted; the progress
    half is what is adopted.
    """

    # -- identity: every input block ids and weights depend on ------------ #
    dataset_id: str = ""
    base: str = ""
    target: str = "text"
    session: str = ""
    rank: int = 0
    alpha: int = 0
    dropout: float = 0.0
    max_len: int = 0
    seed: int = 0
    n_examples: int = 0
    block_size: int = 0

    # -- provenance: for the operator, never compared --------------------- #
    version: int = RESUME_VERSION
    job_id: str = ""
    name: str = ""
    written_at: float = 0.0
    rounds: int = 0
    planned_pods: int = 0
    epochs: float = 1.0

    # -- progress: what is adopted ---------------------------------------- #
    #: The round to OPEN next -- one past the round the average belongs to.
    round: int = 0
    #: Blocks whose adapters were averaged in. The resume point.
    covered: List[int] = dataclasses.field(default_factory=list)
    #: The handout counter and the blocks freed by pods that died, so a
    #: resumed queue does not re-walk ground the old one had moved past.
    next_block: int = 0
    recycle: List[int] = dataclasses.field(default_factory=list)
    #: The measured counters, carried so the published progress of a resumed
    #: run is the whole run's and not just this leg's.
    trained: int = 0
    asked: int = 0
    subs: int = 0
    short: int = 0
    #: The average to start the pods from, as a BASENAME inside the run's
    #: directory. Not an absolute path: the directory moves between hosts and
    #: a stale absolute path is a resume that silently starts from the bare
    #: base.
    average: str = ""

    def to_json(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, raw: Dict[str, Any]) -> "ResumeState":
        """Build from a parsed checkpoint, ignoring fields we do not know.

        Unknown keys are dropped rather than refused so a checkpoint written
        by a newer build is still resumable by an older one as far as it
        understands it -- the version guards the cases where that is wrong.
        """
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in (raw or {}).items() if k in fields})

    # -- what it is worth ------------------------------------------------- #

    @property
    def blocks_done(self) -> int:
        return len(self.covered)

    def describe(self) -> str:
        """One line an operator can act on, without reading the file."""
        what = (f"round {self.round}, {self.blocks_done} block(s) trained"
                f" ({self.trained} example(s))")
        if self.average:
            return f"{what}, starting from {self.average}"
        return f"{what}, NO average on disk — coverage only"


def path_for(out_dir: str) -> str:
    return os.path.join(out_dir, RESUME_FILE)


def save(out_dir: str, state: ResumeState) -> str:
    """Write ``state`` into ``out_dir`` atomically. Returns the path.

    Atomic because this is written while pods are training and read after
    something went wrong: a half-written checkpoint is indistinguishable from
    a corrupt one, and the moment it is most likely to be truncated is a
    process dying -- exactly the case it exists for.
    """
    os.makedirs(out_dir, exist_ok=True)
    state.written_at = time.time()
    state.version = RESUME_VERSION
    dest = path_for(out_dir)
    fd, tmp = tempfile.mkstemp(dir=out_dir, prefix=".resume-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(state.to_json(), fh, indent=1, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, dest)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return dest


def load(out_dir: str) -> Optional[ResumeState]:
    """The checkpoint in ``out_dir``, or None if there is none.

    A checkpoint that cannot be parsed raises: a run asked to resume from a
    corrupt file must not quietly become a run that starts from nothing,
    which is the whole failure this module exists to end.
    """
    p = path_for(out_dir)
    if not os.path.isfile(p):
        return None
    try:
        with open(p) as fh:
            raw = json.load(fh)
    except Exception as exc:
        raise ResumeError(f"{p} cannot be read as a checkpoint: {exc}") from exc
    if not isinstance(raw, dict):
        raise ResumeError(f"{p} is not a checkpoint object")
    st = ResumeState.from_json(raw)
    if int(st.version) > RESUME_VERSION:
        raise ResumeError(
            f"{p} was written by a newer build (checkpoint version "
            f"{st.version}, this build reads {RESUME_VERSION}) — its fields "
            f"may not mean what this build would read them as")
    return st


def average_file(out_dir: str, state: ResumeState) -> str:
    """The absolute path of the adapter to start from, or "" if there is none.

    Resolved against ``out_dir`` at READ time and checked to exist, because a
    checkpoint naming an average that was cleaned up is a resume that would
    start every pod from the bare base while reporting a continuation.
    """
    if not state.average:
        return ""
    p = os.path.join(out_dir, os.path.basename(state.average))
    return p if os.path.isfile(p) else ""


def mismatches(state: ResumeState, want: Dict[str, Any]) -> List[str]:
    """Which identity fields disagree, said in full sentences.

    ``want`` carries the new run's values. A field absent from ``want`` is not
    compared -- the caller decides what it knows -- but a field present and
    different is always a refusal, never a warning: see the module docstring
    on why a mismatched block id is undetectable downstream.
    """
    out: List[str] = []
    for field, label in _IDENTITY:
        if field not in want:
            continue
        mine = getattr(state, field)
        theirs = want[field]
        if isinstance(mine, float) or isinstance(theirs, float):
            same = abs(float(mine or 0) - float(theirs or 0)) < 1e-9
        else:
            same = str(mine) == str(theirs)
        if not same:
            out.append(f"{label} changed ({mine!r} -> {theirs!r})")
    return out


def check(state: ResumeState, want: Dict[str, Any], *, where: str = "") -> None:
    """Raise :class:`ResumeError` unless ``state`` may be resumed as ``want``.

    The message names every field that differs rather than the first, because
    a caller correcting a resume request wants the whole list -- usually they
    changed `rounds` or `pods`, which moves `block_size`, which is the one
    that actually matters and the least obvious.
    """
    bad = mismatches(state, want)
    if not bad:
        return
    head = f"the checkpoint in {where} " if where else "the checkpoint "
    extra = ""
    if any(b.startswith("the block size") for b in bad):
        extra = (" The block size is ceil(examples / (rounds * pods)), so "
                 "changing `rounds` or `pods` changes which examples every "
                 "recorded block id refers to; resume with the original "
                 "values, or start a fresh run.")
    raise ResumeError(
        head + "was written for a different run and cannot be resumed: "
        + "; ".join(bad) + "." + extra)


def nothing_to_resume(state: Optional[ResumeState]) -> Optional[str]:
    """Why this checkpoint is not worth resuming, or None if it is.

    A checkpoint with no coverage AND no average is a run that died before
    its first round closed. Resuming it is identical to starting it, and
    saying so is better than a "resumed" run that begins from nothing.
    """
    if state is None:
        return ("there is no checkpoint — the run was interrupted before it "
                "closed its first round, or it predates resume support")
    if not state.covered and not state.average:
        return ("the checkpoint records no trained blocks and names no "
                "average: the run stopped before its first round closed, so "
                "a resume would start from the beginning anyway")
    return None
