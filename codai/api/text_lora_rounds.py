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

"""Training one adapter on several pods that cannot reach each other.

The usual multi-machine trainer in this repo (:mod:`codai.cluster.ddp`) averages
gradients after EVERY backward -- an all-reduce per step, which is fine on a LAN
and impossible on rented pods: pods in different datacenters cannot address one
another, so there is no ring to reduce around.

So the exchange is periodic instead. Each pod trains its own shard for a round,
sends the ADAPTER it ended the round with, and starts the next round from the
average of everyone's. Local SGD rather than synchronous SGD: more compute
between exchanges, far less communication, and -- the part that matters here --
**no pod ever has to be addressable**. Every message is a pod calling the
orchestrator, which is the only direction that works and happens to be the only
direction the rest of the RunPod subsystem uses.

Averaging an adapter is ~160 MB per pod per round rather than gradients per
step, so the round length is chosen from measured throughput: long enough that
the transfer is a rounding error, short enough that the shards do not diverge.

Two properties are worth stating because getting either wrong is silent:

* **the average is weighted by work done.** A pod that managed 40 examples in a
  round must not pull as hard as one that managed 400, or a slow card quietly
  gets a bigger vote;
* **the accumulation is float32.** Adapters are bf16, which has 8 mantissa bits;
  summing eight of them in bf16 loses low-order bits on every add, and the
  result is a slightly lobotomised adapter that still loads and still answers.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple


class RoundError(RuntimeError):
    """A shard's submission cannot be averaged, with what differs."""


def average_adapters(state_dicts: Sequence[Dict[str, Any]],
                     weights: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    """Weighted mean of several adapter state dicts.

    ``weights`` is the work each shard did in the round (examples trained);
    omitted means equal. They are normalised here, so a caller passes counts
    rather than fractions and cannot make them not sum to one.

    Accumulated in float32 and cast back to each tensor's own dtype: see the
    module docstring on why bf16 accumulation is a quiet loss.
    """
    import torch

    if not state_dicts:
        raise RoundError("nothing to average")
    if weights is None:
        weights = [1.0] * len(state_dicts)
    if len(weights) != len(state_dicts):
        raise RoundError(f"{len(state_dicts)} adapters but {len(weights)} weights")
    total = float(sum(weights))
    if total <= 0:
        raise RoundError("the weights sum to zero: no shard did any work")

    first = state_dicts[0]
    keys = set(first)
    for i, sd in enumerate(state_dicts[1:], 1):
        if set(sd) != keys:
            missing = sorted(keys - set(sd))[:3]
            extra = sorted(set(sd) - keys)[:3]
            # Refused rather than intersected: a shard with different keys
            # trained a different target set, and averaging the overlap would
            # produce an adapter that loads, serves, and is not what any shard
            # trained.
            raise RoundError(
                f"shard {i} has a different parameter set than shard 0 "
                f"(missing {missing}, extra {extra}) -- it was trained with a "
                f"different target_modules or rank")
        for k in keys:
            if sd[k].shape != first[k].shape:
                raise RoundError(
                    f"shard {i} key {k!r} has shape {tuple(sd[k].shape)}, "
                    f"shard 0 has {tuple(first[k].shape)} -- different rank")

    out: Dict[str, Any] = {}
    for k in first:
        acc = torch.zeros(first[k].shape, dtype=torch.float32,
                          device=first[k].device)
        for sd, w in zip(state_dicts, weights):
            acc += sd[k].to(torch.float32) * (float(w) / total)
        out[k] = acc.to(first[k].dtype)
    return out


def average_adapter_files(paths: Sequence[str], weights: Sequence[float],
                          out_path: str) -> Dict[str, Any]:
    """:func:`average_adapters` over safetensors files, written to ``out_path``.

    One shard is loaded at a time... deliberately NOT: the mean needs every
    tensor, and loading eight 160 MB adapters is 1.3 GB of host RAM, which the
    orchestrator has and a streaming implementation would complicate for no
    gain. If adapters ever reach gigabytes this is the function to revisit.
    """
    from safetensors.torch import load_file, save_file
    sds = [load_file(p) for p in paths]
    avg = average_adapters(sds, weights)
    save_file(avg, out_path)
    return avg


@dataclasses.dataclass
class Submission:
    """One shard's result for one round."""

    shard: int
    round: int
    path: str
    examples: int
    at: float = dataclasses.field(default_factory=time.time)


@dataclasses.dataclass
class RoundPlan:
    """How the work is split, and what a pod is told when it asks.

    Assignment is ``shard = index % n_shards`` over the SEEDED shuffle the
    trainer uses, so it is a pure function of (seed, n_shards, index): a pod
    that dies and is replaced gets exactly the shard its predecessor had, and a
    resumed job reproduces the same split instead of reshuffling the dataset
    under a half-trained adapter.
    """

    n_shards: int
    seed: int = 42
    round_examples: int = 0      # 0 = the whole shard each round

    def shard_of(self, index: int) -> int:
        return index % max(1, int(self.n_shards))

    # -- work as a QUEUE, not as a per-round partition ------------------- #
    #
    # The partition below (`indices_for`) cuts a round's chunk into one slice
    # per pod in that round's cohort, which forces two things that cost real
    # money: the cohort has to be fixed before any slice is handed out, and a
    # pod arriving mid-round has no slice at all -- so it idles until the
    # round closes, or (as happened) gets handed position 0's slice and
    # duplicates another pod's work.
    #
    # Neither is necessary. What local SGD actually requires is that everyone
    # whose adapter goes into one average STARTED FROM THE SAME ADAPTER and
    # trained DISJOINT data. Which data, and how much of it each pod got, is
    # free -- the average is weighted by examples trained, so a pod that
    # managed one block and a pod that managed three both count correctly.
    #
    # So the dataset is cut into fixed-size BLOCKS of the seeded order and
    # handed out on demand. A pod that becomes ready halfway through an
    # exchange takes the next unclaimed block and its result joins that same
    # exchange. More pods therefore means more of the dataset covered per
    # exchange rather than the same work divided more finely, and no number
    # anywhere depends on how many pods happened to be present when the
    # exchange opened.

    def block_size(self, n_examples: int, rounds: int,
                   n_shards: Optional[int] = None) -> int:
        """How many examples one handout is.

        Chosen so that the PLANNED fleet over the PLANNED rounds covers the
        set exactly once: `n / (rounds * n_shards)`. That keeps the totals a
        caller already reasoned about when the fleet is the size it asked for,
        and simply covers more ground when more pods turn up.
        """
        k = max(1, int(n_shards if n_shards is not None else self.n_shards))
        r = max(1, int(rounds))
        return max(1, -(-int(n_examples) // (r * k)))

    def block_at(self, n_examples: int, block: int, per: int) -> List[int]:
        """Block number ``block`` of the seeded order, wrapping by epoch.

        Wrapping rather than running out: a fleet larger than planned consumes
        blocks faster, and the honest answer to "we have covered the set" is
        to start a second pass over the same shuffle, not to hand a pod an
        empty slice it would report as zero work.
        """
        import random
        n = max(0, int(n_examples))
        if n == 0:
            return []
        per = max(1, int(per))
        order = list(range(n))
        random.Random(self.seed).shuffle(order)
        blocks = max(1, -(-n // per))
        b = int(block) % blocks
        return order[b * per:(b + 1) * per]

    def indices_for(self, shard: int, n_examples: int, *,
                    round: int = -1, rounds: int = 1,
                    n_shards: Optional[int] = None) -> List[int]:
        """One position's examples for one round.

        ROUND FIRST, then the split across positions. The order used to be the
        other way -- the shard's own list cut into ``rounds`` chunks -- which
        gives each pod its whole shard over a full cycle, but ONLY while the
        number of shards never changes. The moment a fleet grows or shrinks
        between rounds, the union across rounds both repeats some examples and
        misses others, silently: coverage is a property of a fixed partition,
        and the fleet is not fixed.

        Slicing the global order by round first makes every round's content a
        function of (seed, round, rounds) alone -- the same 1/rounds of the
        dataset whatever the fleet size -- and then splits THAT chunk among
        however many pods are present. So a run can start on one pod and reach
        eight without training some examples twice and others never, which is
        what makes scaling up safe rather than merely possible.

        ``n_shards`` overrides the plan's own count, because the number that
        matters is how many pods are in THIS round's cohort.
        """
        import random
        k = int(n_shards if n_shards is not None else self.n_shards)
        k = max(1, k)
        pos = int(shard) % k
        order = list(range(n_examples))
        random.Random(self.seed).shuffle(order)
        if round is None or int(round) < 0 or int(rounds) <= 1:
            chunk = order
        else:
            r = int(round) % max(1, int(rounds))
            per = -(-len(order) // int(rounds))   # ceil, so nothing is dropped
            chunk = order[r * per:(r + 1) * per]
        # Round-robin within the round's chunk: contiguous blocks would hand a
        # pod a run of adjacent examples from the shuffle, and with a
        # fleet-sized block that is a correlated slice rather than a sample.
        return [idx for j, idx in enumerate(chunk) if j % k == pos]


class RoundTracker:
    """Who has submitted what, and when a round may be averaged.

    The orchestrator's half of the protocol. It holds no network code on
    purpose: every transfer is a pod calling in, so this only has to decide
    whether a round is complete, which stragglers to wait for, and which shard
    to hand to a pod that arrives.
    """

    def __init__(self, plan: RoundPlan, *, min_fraction: float = 0.75,
                 deadline_s: float = 0.0):
        self.plan = plan
        self.min_fraction = float(min_fraction)
        self.deadline_s = float(deadline_s)
        self.round = 0
        #: round -> shard -> every submission that shard made for it. A LIST
        #: because a pod that finishes its block while the exchange is still
        #: open takes another one: each contribution starts from the same
        #: pushed adapter and covers examples no one else has, so they are all
        #: legitimate operands of the same average. Keyed by shard only, it
        #: was a dict, and a second contribution silently REPLACED the first.
        self._subs: Dict[int, Dict[int, List[Submission]]] = {}
        #: Submissions for a round that has not opened yet, held rather than
        #: refused.
        self._future: Dict[int, Dict[int, List[Submission]]] = {}
        self._opened_at: Dict[int, float] = {}
        self._missed: Dict[int, int] = {}
        #: How many pods a given round actually HAS. `plan.n_shards` is the
        #: shape the job was planned at; a fleet that starts on one pod and
        #: grows to eight has a different number in every round, and judging
        #: "enough arrived" against the planned number means a round with two
        #: pods in it can never be complete -- it waits out the deadline every
        #: time, which is how a growing fleet crawls.
        self._expected: Dict[int, int] = {}
        #: The actual shard ids in a round, when the caller knows them.
        self._members: Dict[int, List[int]] = {}

    # -- submissions ------------------------------------------------------ #

    def expect(self, rnd: int, cohort: int) -> None:
        """Record how many pods are in round ``rnd``."""
        self._expected[int(rnd)] = max(1, int(cohort))

    def expect_shards(self, rnd: int, shards: Sequence[int]) -> None:
        """Record WHICH pods are in round ``rnd``, not just how many.

        :meth:`missing` used to answer `range(expected())`, which is only the
        right set when the shard ids happen to be 0..n-1. A fleet that grows
        and loses pods has ids like {0, 1, 15}, so a three-pod round reported
        shards 0, 1 and 2 as its members: shard 2 did not exist and accrued
        misses, while shard 15 could never accrue any and so could never be
        declared dead.
        """
        ids = sorted({int(x) for x in shards})
        if ids:
            self._members[int(rnd)] = ids
            self._expected[int(rnd)] = len(ids)

    def expected(self, rnd: Optional[int] = None) -> int:
        """The cohort size for a round, falling back to the planned shape."""
        r = self.round if rnd is None else int(rnd)
        got = self._expected.get(r)
        return int(got) if got else max(1, int(self.plan.n_shards))

    def open_round(self, rnd: int, now: Optional[float] = None) -> None:
        self._opened_at.setdefault(rnd, now if now is not None else time.time())

    def submit(self, sub: Submission) -> None:
        """Record a shard's adapter for a round.

        A submission for a round that is already averaged is REJECTED, not
        folded into the current one: a straggler's adapter is two rounds of
        training behind, and quietly mixing it in moves the average backwards
        for everyone.
        """
        if sub.round < self.round:
            raise RoundError(
                f"shard {sub.shard} submitted for round {sub.round} but round "
                f"{self.round} is already open -- the result is stale and is "
                f"not averaged in")
        if sub.round > self.round:
            # HELD, not refused. A result for a round that has not opened is
            # not wrong, only early -- it was trained from an adapter this
            # round will start from -- so it waits for its round instead of
            # being thrown away. Refusing it was the one case where correct
            # work was discarded for being fast.
            self._future.setdefault(sub.round, {}).setdefault(
                int(sub.shard), []).append(sub)
            return
        self.open_round(sub.round)
        self._subs.setdefault(sub.round, {}).setdefault(
            int(sub.shard), []).append(sub)

    def have(self, rnd: Optional[int] = None) -> List[Submission]:
        """Every submission for a round, from every shard that made one."""
        r = self.round if rnd is None else rnd
        got = self._subs.get(r, {})
        return [sub for k in sorted(got) for sub in got[k]]

    def contributors(self, rnd: Optional[int] = None) -> int:
        """How many SHARDS submitted, which is what readiness is judged on.

        Not the number of submissions: one pod that got through four blocks
        while three pods are still on their first must not make the round
        look four-sevenths complete.
        """
        r = self.round if rnd is None else rnd
        return len(self._subs.get(r, {}))

    def take_future(self, rnd: Optional[int] = None) -> int:
        """Move held early submissions into a round once it is open."""
        r = self.round if rnd is None else int(rnd)
        early = self._future.pop(r, None)
        if not early:
            return 0
        dest = self._subs.setdefault(r, {})
        n = 0
        for sh, subs in early.items():
            dest.setdefault(int(sh), []).extend(subs)
            n += len(subs)
        self.open_round(r)
        return n

    # -- completion ------------------------------------------------------- #

    def ready(self, now: Optional[float] = None) -> bool:
        """True when this round can be averaged.

        Either enough shards arrived, or the deadline passed and at least one
        did. Waiting for every shard is what turned one dead pod into a stalled
        fleet: the round proceeds with the survivors and the absent shard is
        reassigned.
        """
        got = self.contributors()
        if got == 0:
            return False
        want = self.expected()
        if got >= want:
            return True
        if got / float(want) >= self.min_fraction:
            return True
        if self.deadline_s > 0:
            opened = self._opened_at.get(self.round)
            now = now if now is not None else time.time()
            if opened is not None and (now - opened) >= self.deadline_s:
                return True
        return False

    def members(self, rnd: Optional[int] = None) -> List[int]:
        """The shard ids in a round, falling back to 0..expected-1."""
        r = self.round if rnd is None else int(rnd)
        got = self._members.get(r)
        if got:
            return list(got)
        return list(range(self.expected(r)))

    def missing(self) -> List[int]:
        present = set(self._subs.get(self.round, {}))
        return [s for s in self.members() if s not in present]

    def close_round(self) -> Tuple[List[Submission], List[int]]:
        """Finish the round: returns what will be averaged and who was absent.

        An absent shard's miss count is what decides a pod is gone. The round
        upload IS the heartbeat -- a training pod serves nothing and answers no
        health probe, so its only honest liveness signal is that it keeps
        finishing rounds.
        """
        subs = self.have()
        absent = self.missing()
        for s in absent:
            self._missed[s] = self._missed.get(s, 0) + 1
        for s in set(self._subs.get(self.round, {})):
            self._missed[s] = 0
        self.round += 1
        self.open_round(self.round)
        # Anything that arrived early for THIS round now counts as arrived.
        self.take_future(self.round)
        return subs, absent

    def dead_shards(self, after_misses: int = 2) -> List[int]:
        """Shards whose pod has missed enough rounds to be considered gone."""
        return sorted(s for s, n in self._missed.items() if n >= after_misses)

    def weights_for(self, subs: Sequence[Submission]) -> List[float]:
        """The averaging weights: examples actually trained this round."""
        return [float(max(0, s.examples)) for s in subs]
