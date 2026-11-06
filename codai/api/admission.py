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

"""KV-aware admission control: how many requests this server may hold at once.

A serving engine admits as many sequences as it is told to, without checking
whether its KV cache can hold them at the length they will reach. It discovers
the shortfall during DECODE, and its only recourse is to evict half-generated
requests and recompute them. Throughput collapses while every counter still
looks busy, and it is invisible in utilisation and in VRAM -- vLLM claims its
``gpu_memory_utilization`` share at boot, so a pod reads ~90% VRAM from the
moment it starts whether it is serving one request or sixty-four.

The fix is not to compute the right batch width. That cannot be done:

* the width is a launch flag, and the cache size is only reported after boot;
* the width lives on a model entry and applies to every pod of that model,
  while cache size is a property of the CARD -- and one entry may accept half a
  dozen GPU types across three architectures;
* same VRAM is not same cache, because the engine picks a different attention
  backend per architecture and reserves a different workspace;
* and a persisted, measurement-driven limit that only ever moves down is how
  `measured_n_gpu_layers` once ratcheted 31 -> 20 under transient contention and
  stuck there, spilling to CPU with every number looking healthy.

So this measures nothing about the hardware and remembers nothing between pods.
It gates on CACHE OCCUPANCY, which is a fraction and therefore means the same
thing on every card, and it converges the way congestion control does:
additive increase while the cache is comfortable and nothing has been preempted,
multiplicative decrease the moment it has. The costs are asymmetric -- one slot
too few wastes a little throughput, one too many throws away work already done
-- which is exactly what AIMD is for.

Three properties worth stating because they are load-bearing:

**The operator's figure is a CEILING, not a setpoint.** `max_parallel_requests`
already means "how hard this server may be driven", so the controller only ever
runs at or below it. No existing install can become more aggressive, and with
the controller off the behaviour is byte-for-byte what it was.

**Nothing is persisted.** The controller re-converges from scratch in every
process. It therefore cannot ratchet down and stick: the worst case is one
server running below capacity for its own lifetime.

**Absent a reading it does nothing.** An engine that reports no cache occupancy
(llama.cpp, or an unreachable metrics endpoint) leaves the configured value
exactly as it was. Degrading to today's behaviour is correct; degrading to a
guess is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import asyncio

#: Weight of a new sample in the exponential moving average of tokens-per-request.
#: Low, because the estimate feeds a veto: reacting fast to a brief run of long
#: pages would clamp admission for the pages after them.
_E_ALPHA = 0.3

#: What an un-configured tokens-per-request estimate is multiplied by. The
#: measurement available from counters is a MEAN, and preemption is driven by the
#: tail, so the mean alone would under-state the worst case it is meant to bound.
_E_SAFETY = 1.5


@dataclass
class Policy:
    """What the controller is allowed to do. Fixed for a process's lifetime."""
    mode: str = "auto"              # auto | on | off
    kv_target_pct: float = 0.85
    kv_ceiling_pct: float = 0.95
    min_parallel: int = 4
    #: The operator's stated maximum. 0 means nothing was stated, and then there
    #: is no range to work inside -- the controller stays out of the way.
    ceiling: int = 0
    backoff: float = 0.75
    recover_after: int = 3
    #: Tokens one request is expected to occupy at its longest. 0 = derive it
    #: from the engine's own counters. Configured wins, because a bounded
    #: workload KNOWS its worst case (an image's tokens plus a generation cap)
    #: while a measurement can only average what has already happened.
    expected_tokens: int = 0
    interval_s: float = 5.0


@dataclass
class Decision:
    limit: int
    reason: str
    #: The tokens-per-request estimate this decision used, so the caller can
    #: carry it forward and a reader can see what the veto was computed from.
    e_tokens: float = 0.0
    changed: bool = False


@dataclass
class State:
    """Everything the controller carries between ticks. Deliberately small."""
    limit: int = 0
    #: Ticks still to wait before increasing again, after a decrease.
    hold: int = 0
    #: The previous cumulative reading, for deltas.
    last: dict = field(default_factory=dict)
    e_tokens: float = 0.0
    #: Why the last tick decided what it did, for `/internal/engine-state`.
    reason: str = "not started"


def _as_fraction(v) -> Optional[float]:
    """Cache occupancy as 0..1, whichever way the engine expressed it.

    vLLM's gauge is named ``..._perc`` but reports a FRACTION. Reading it as a
    percentage would make 0.9 look like 0.9% -- a full cache indistinguishable
    from an empty one, and the controller would raise the limit into a cache
    that is already overflowing. So the shape is detected rather than assumed.
    """
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f < 0:
        return None
    return f / 100.0 if f > 1.0 else f


def _expected_tokens(state: State, reading: dict, policy: Policy) -> float:
    """Tokens one request is expected to hold, or 0 when it cannot be known.

    Configured first. Otherwise measured from the engine's own cumulative
    counters, as a WINDOW rather than a lifetime mean: `requests_done`
    accumulates from boot, so a lifetime average keeps folding in requests from
    before the workload changed.
    """
    if policy.expected_tokens > 0:
        return float(policy.expected_tokens)
    prev = state.last or {}

    def _d(key):
        now, before = reading.get(key), prev.get(key)
        if not isinstance(now, (int, float)) or not isinstance(before, (int, float)):
            return None
        return now - before

    d_done = _d("requests_done")
    d_prompt = _d("prompt_tokens")
    d_gen = _d("generation_tokens")
    if d_done is None or d_prompt is None or d_gen is None:
        return state.e_tokens
    # A counter that went BACKWARDS means the engine restarted; the window is
    # meaningless rather than negative, so keep what we had.
    if d_done <= 0 or d_prompt < 0 or d_gen < 0:
        return state.e_tokens
    sample = (d_prompt + d_gen) / float(d_done)
    if sample <= 0:
        return state.e_tokens
    if state.e_tokens <= 0:
        return sample
    return (1.0 - _E_ALPHA) * state.e_tokens + _E_ALPHA * sample


def _projected_over_target(reading: dict, e_tokens: float,
                           policy: Policy) -> Optional[bool]:
    """Would one MORE request push the committed set past the target?

    This is the part current occupancy cannot answer. Occupancy is a snapshot of
    a quantity that is still growing: every running request will keep appending
    to the cache until it finishes, so a gate that admits at 70% can find itself
    far past 100% once those requests reach their full length -- and by then the
    preemption was already decided at admission time.

    So the question asked is not "how full is the cache" but "if everything now
    running ran to its expected length, what fraction would it need". Current
    occupancy drops out of it entirely.

    Returns None when it cannot be computed, and the caller then falls back to
    observed occupancy alone. A VETO only: see `decide`.
    """
    kv_tokens = reading.get("kv_tokens")
    running = reading.get("running")
    if not isinstance(kv_tokens, (int, float)) or kv_tokens <= 0:
        return None
    if not isinstance(running, (int, float)) or running < 0:
        return None
    if e_tokens <= 0:
        return None
    e = e_tokens if policy.expected_tokens > 0 else e_tokens * _E_SAFETY
    projected = ((float(running) + 1.0) * e) / float(kv_tokens)
    return projected > policy.kv_target_pct


def decide(state: State, reading: dict, policy: Policy) -> Decision:
    """One control tick. Pure: no I/O, no clock, no globals.

    `reading` is the engine's load as `codai.api.serving_load` reports it, plus
    `kv_tokens` (the cache's absolute size, measured once at engine boot) when
    it is known.
    """
    limit = int(state.limit or 0)
    e_tokens = _expected_tokens(state, reading, policy)

    if policy.mode == "off" or policy.ceiling <= 0:
        return Decision(limit=limit, reason="disabled", e_tokens=e_tokens)

    floor = max(1, min(int(policy.min_parallel), int(policy.ceiling)))
    if limit <= 0:
        limit = int(policy.ceiling)

    kv = _as_fraction(reading.get("kv_cache_pct"))
    if kv is None:
        # No occupancy to steer by. In `auto` this is an engine that does not
        # report it at all, which is not a failure -- it is a different engine.
        return Decision(limit=limit, e_tokens=e_tokens,
                        reason="no cache-occupancy reading; leaving the "
                               "configured limit alone")

    prev = state.last or {}
    p_now, p_prev = reading.get("preemptions"), prev.get("preemptions")
    delta = None
    restarted = False
    if isinstance(p_now, (int, float)) and isinstance(p_prev, (int, float)):
        if p_now < p_prev:
            # Cumulative counters rewind when the engine restarts. Reading that
            # as negative loss would hand the controller a reason to raise the
            # limit on the evidence of a restart.
            restarted = True
        else:
            delta = p_now - p_prev

    def _down(reason):
        new = max(floor, int(limit * policy.backoff))
        if new >= limit:
            new = max(floor, limit - 1)
        return Decision(limit=new, reason=reason, e_tokens=e_tokens,
                        changed=new != limit)

    if restarted:
        return Decision(limit=limit, e_tokens=e_tokens,
                        reason="the engine restarted (preemption counter "
                               "rewound); re-baselining before deciding")
    if delta is not None and delta > 0:
        return _down(f"{int(delta)} preemption(s) since the last tick: the "
                     f"cache could not hold what was admitted")
    if kv > policy.kv_ceiling_pct:
        return _down(f"cache at {kv * 100:.0f}% is over the "
                     f"{policy.kv_ceiling_pct * 100:.0f}% ceiling; backing off "
                     f"before the recompute rather than after it")
    if state.hold > 0:
        return Decision(limit=limit, e_tokens=e_tokens,
                        reason=f"holding for {state.hold} more tick(s) after a "
                               f"decrease")
    if kv < policy.kv_target_pct:
        over = _projected_over_target(reading, e_tokens, policy)
        if over:
            return Decision(limit=limit, e_tokens=e_tokens,
                            reason=f"cache is at {kv * 100:.0f}% but the "
                                   f"requests already running are committed to "
                                   f"more than the "
                                   f"{policy.kv_target_pct * 100:.0f}% target; "
                                   f"not admitting more")
        if limit >= policy.ceiling:
            return Decision(limit=limit, e_tokens=e_tokens,
                            reason=f"at the configured maximum of "
                                   f"{policy.ceiling}")
        new = min(int(policy.ceiling), limit + 1)
        return Decision(limit=new, reason=f"cache at {kv * 100:.0f}%, no "
                                          f"preemptions; one more slot",
                        e_tokens=e_tokens, changed=new != limit)
    return Decision(limit=limit, e_tokens=e_tokens,
                    reason=f"cache at {kv * 100:.0f}%, within target")


def policy_from_config(cfg) -> Policy:
    """Build a Policy from a `ServerConfig`-shaped object."""
    def _f(name, default):
        try:
            v = float(getattr(cfg, name, default))
        except (TypeError, ValueError):
            return default
        return v

    def _i(name, default):
        try:
            return int(getattr(cfg, name, default))
        except (TypeError, ValueError):
            return default

    return Policy(
        mode=str(getattr(cfg, "admission_control", "auto") or "auto").strip().lower(),
        kv_target_pct=_f("admission_kv_target_pct", 0.85),
        kv_ceiling_pct=_f("admission_kv_ceiling_pct", 0.95),
        min_parallel=_i("admission_min_parallel", 4),
        ceiling=_i("max_parallel_requests", 0),
        backoff=_f("admission_backoff", 0.75),
        recover_after=_i("admission_recover_after", 3),
        expected_tokens=_i("admission_expected_tokens", 0),
        interval_s=_f("admission_interval_s", 5.0),
    )


class AdmissionController:
    """Runs `decide` on a timer and applies the result to the queue gate.

    Lives in the process that SERVES, because that is where the engine's metrics
    and its measured cache size already are. Nothing has to travel and nothing
    has to be seeded: a rented pod runs this against its own engine, on whatever
    card it was given.
    """

    def __init__(self, policy: Policy, queue_manager=None):
        self.policy = policy
        self.state = State(limit=int(policy.ceiling or 0))
        self._qm = queue_manager
        self._task = None
        self._ticks = 0

    # -- reporting ---------------------------------------------------------

    def snapshot(self) -> dict:
        """What this controller is doing, for `/internal/engine-state`."""
        return {
            "mode": self.policy.mode,
            "limit": int(self.state.limit or 0),
            "ceiling": int(self.policy.ceiling or 0),
            "floor": int(self.policy.min_parallel or 0),
            "expected_tokens": round(float(self.state.e_tokens or 0.0), 1),
            "reason": self.state.reason,
            "ticks": self._ticks,
        }

    # -- the loop ----------------------------------------------------------

    def start(self) -> None:
        if self.policy.mode == "off" or self.policy.ceiling <= 0:
            return
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    async def _read(self) -> dict:
        """The engine's load plus its measured cache size.

        Off the event loop: `serving_load` reaches the metrics endpoint with a
        BLOCKING client, and a synchronous call made from a coroutine stops every
        request in the process -- which from outside looks like an overloaded
        upstream rather than a server that is not running.
        """
        from codai.api import serving_load, vllm_worker
        urls = vllm_worker.running_service_urls()
        if not urls:
            return {}
        out = dict(await asyncio.to_thread(serving_load.engine_load, urls[0]))
        try:
            kv = await asyncio.to_thread(vllm_worker.kv_capacity)
        except Exception:
            kv = {}
        for k in ("kv_tokens", "kv_gib"):
            if isinstance((kv or {}).get(k), (int, float)):
                out[k] = kv[k]
        return out

    async def _apply(self, limit: int) -> None:
        qm = self._qm
        if qm is None:
            from codai.queue.manager import queue_manager as qm
        await qm.set_dynamic_cap(int(limit))

    async def _tick(self) -> None:
        reading = await self._read()
        decision = decide(self.state, reading, self.policy)
        self._ticks += 1
        if decision.changed:
            print(f"[admission] {self.state.limit} -> {decision.limit} in "
                  f"flight: {decision.reason}", flush=True)
            if decision.limit < self.state.limit:
                self.state.hold = max(0, int(self.policy.recover_after))
        elif self.state.hold > 0:
            self.state.hold -= 1
        self.state.limit = decision.limit
        self.state.reason = decision.reason
        self.state.e_tokens = decision.e_tokens
        if reading:
            self.state.last = dict(reading)
        await self._apply(decision.limit)

    async def _run(self) -> None:
        interval = max(1.0, float(self.policy.interval_s or 5.0))
        print(f"[admission] KV-aware admission control is on: between "
              f"{self.policy.min_parallel} and {self.policy.ceiling} requests in "
              f"flight, target {self.policy.kv_target_pct * 100:.0f}% cache, "
              f"every {interval:.0f}s", flush=True)
        while True:
            try:
                await asyncio.sleep(interval)
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # A controller is an optimisation. It must never be the reason a
                # server stops serving, so a bad tick is reported and skipped.
                print(f"[admission] tick skipped: {exc}", flush=True)


#: The process's controller, once started. None = never started (and then
#: nothing anywhere should behave differently from before this existed).
controller: Optional[AdmissionController] = None


def start(cfg, queue_manager=None) -> Optional[AdmissionController]:
    """Build and start the process controller. Idempotent."""
    global controller
    if controller is not None:
        return controller
    policy = policy_from_config(cfg)
    if policy.mode == "off":
        return None
    if policy.ceiling <= 0:
        # Nothing stated means no range to work inside. Say so once: an operator
        # who turned this on is entitled to know why it did nothing.
        if policy.mode == "on":
            print("[admission] admission_control is on but "
                  "server.max_parallel_requests is unset, so there is no "
                  "maximum to work below and no gate to move; set it to the "
                  "most this server may ever hold in flight.", flush=True)
        return None
    controller = AdmissionController(policy, queue_manager=queue_manager)
    controller.start()
    return controller


def snapshot() -> dict:
    """The process controller's state, or {} when it never started."""
    return controller.snapshot() if controller is not None else {}
