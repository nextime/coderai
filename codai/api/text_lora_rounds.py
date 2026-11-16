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

    def indices_for(self, shard: int, n_examples: int, *,
                    round: int = -1, rounds: int = 1) -> List[int]:
        """This shard's examples, or just this ROUND's slice of them.

        With ``round`` given, the shard's own list is cut into ``rounds`` equal
        chunks and one is returned, so over a full cycle a pod sees its whole
        shard exactly once. That is what lets the round LENGTH be chosen from
        measured throughput -- short rounds exchange adapters more often, long
        rounds spend less time transferring -- without changing what any pod
        eventually trains on.
        """
        import random
        order = list(range(n_examples))
        random.Random(self.seed).shuffle(order)
        mine = [idx for i, idx in enumerate(order)
                if self.shard_of(i) == int(shard)]
        if round is None or int(round) < 0 or int(rounds) <= 1:
            return mine
        r = int(round) % max(1, int(rounds))
        per = -(-len(mine) // int(rounds))        # ceil, so nothing is dropped
        return mine[r * per:(r + 1) * per]


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
        self._subs: Dict[int, Dict[int, Submission]] = {}
        self._opened_at: Dict[int, float] = {}
        self._missed: Dict[int, int] = {}

    # -- submissions ------------------------------------------------------ #

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
            raise RoundError(
                f"shard {sub.shard} submitted for round {sub.round}, ahead of "
                f"the open round {self.round}")
        self.open_round(sub.round)
        self._subs.setdefault(sub.round, {})[int(sub.shard)] = sub

    def have(self, rnd: Optional[int] = None) -> List[Submission]:
        r = self.round if rnd is None else rnd
        return [self._subs.get(r, {})[k] for k in sorted(self._subs.get(r, {}))]

    # -- completion ------------------------------------------------------- #

    def ready(self, now: Optional[float] = None) -> bool:
        """True when this round can be averaged.

        Either enough shards arrived, or the deadline passed and at least one
        did. Waiting for every shard is what turned one dead pod into a stalled
        fleet: the round proceeds with the survivors and the absent shard is
        reassigned.
        """
        got = len(self._subs.get(self.round, {}))
        if got == 0:
            return False
        if got >= self.plan.n_shards:
            return True
        if got / float(self.plan.n_shards) >= self.min_fraction:
            return True
        if self.deadline_s > 0:
            opened = self._opened_at.get(self.round)
            now = now if now is not None else time.time()
            if opened is not None and (now - opened) >= self.deadline_s:
                return True
        return False

    def missing(self) -> List[int]:
        present = set(self._subs.get(self.round, {}))
        return [s for s in range(self.plan.n_shards) if s not in present]

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
        return subs, absent

    def dead_shards(self, after_misses: int = 2) -> List[int]:
        """Shards whose pod has missed enough rounds to be considered gone."""
        return sorted(s for s, n in self._missed.items() if n >= after_misses)

    def weights_for(self, subs: Sequence[Submission]) -> List[float]:
        """The averaging weights: examples actually trained this round."""
        return [float(max(0, s.examples)) for s in subs]
