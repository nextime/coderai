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

"""Rented GPUs that run a TRAINING command, and are never left billing.

The serving pools in :mod:`codai.api.runpod_worker` rent a pod, wait for an
OpenAI-compatible server to answer, and terminate it after N failed health
probes. A training pod serves nothing and answers no probe, so that machinery
would rent it and then kill it as unreachable -- which is why this is a separate
fleet rather than a pool with a flag.

The orchestrator DRIVES: it pushes the dataset once, pushes each round's work
and starting adapter, polls the pod's ``/status``, and pulls the finished
adapter. coderai already addresses pods -- that is how it proxies inference to
them -- and ``wait_ready(..., direct_tcp=True)`` returns an ``ip:port`` that
bypasses the RunPod proxy, so a 160 MB transfer meets neither the Cloudflare
idle limit nor its body cap.

Liveness is still the WORK, not a ping: ``/status`` reports the round and the
step, so a pod stuck at step 40 for an hour is distinguishable from one that is
gone. A plain health probe cannot tell those apart, and silence cannot either.

THE RULE THIS MODULE EXISTS TO KEEP: nothing is left billing with nobody owning
it. Every pod id is written to disk BEFORE it is rented and removed only once it
is terminated, so a crashed orchestrator, a restarted container or a new process
can still find and kill what the last one started. :func:`orphans` is that list.
A fleet also enforces a maximum lifetime, because the failure that costs real
money is not a crash -- it is a training loop that keeps checking in forever.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

#: How long a pod may live with no check-in before it is assumed dead. Two round
#: deadlines, give or take: long enough that a slow shard is not killed mid-round.
DEFAULT_MAX_SILENCE_S = 1800.0

#: A hard ceiling on a pod's life regardless of check-ins. The expensive failure
#: is not a crash but a loop that keeps reporting progress forever.
DEFAULT_MAX_LIFETIME_S = 86400.0

#: How many pods may be rented and reaped WITHOUT EVER ANSWERING before the
#: top-up stops growing the fleet. Three rather than one: a single dead
#: machine is luck, and a fleet that stops growing on the first one would
#: never assemble on a busy day.
DOA_STREAK_CAP = 3


class FleetError(RuntimeError):
    """Provisioning refused, with the reason."""


@dataclasses.dataclass
class TrainerPodSpec:
    """What a training pod is: an image, a command, and the money it may cost."""

    image: str
    #: Ranked candidates, cheapest first. A list because RunPod capacity moves:
    #: the card that was free when the catalogue was read is often gone by the
    #: time the pod is requested, and a single choice turns that into a failed
    #: job instead of the next-cheapest card.
    gpu_types: List[str] = dataclasses.field(default_factory=list)
    #: The cloud to rent each of ``gpu_types`` in, same order. A card eligible
    #: in both clouds appears twice at two prices, so the cloud belongs to the
    #: CANDIDATE: renting a community-priced offer on secure charges the secure
    #: rate, which is how a price ceiling gets quietly exceeded. Empty means
    #: ``cloud_type`` for every candidate, which is the old behaviour.
    gpu_clouds: List[str] = dataclasses.field(default_factory=list)
    #: Each candidate's hourly price, same order. ``hourly_usd`` below is one
    #: ESTIMATE for the whole list, which is what made a pod report $0.33/hr
    #: while RunPod charged $0.59: the page, the ledger and the fleet ceiling
    #: were all reasoning about a number no card actually cost.
    gpu_prices: List[float] = dataclasses.field(default_factory=list)
    start_cmd: Optional[List[str]] = None
    entrypoint: Optional[List[str]] = None
    env: Dict[str, str] = dataclasses.field(default_factory=dict)
    cloud_type: str = "SECURE"
    container_disk_gb: int = 80
    volume_gb: int = 0
    network_volume_id: str = ""
    data_center_id: str = ""
    registry_auth_id: str = ""
    hourly_usd: float = 0.0          # estimate, for the cap arithmetic
    #: Refuse any single pod priced above this. A per-POD ceiling: it is what
    #: stops a capacity search from "helpfully" landing on an H100 at six times
    #: the rate because the A40s were full.
    max_pod_hourly_usd: float = 0.0  # 0 = no per-pod ceiling
    max_silence_s: float = DEFAULT_MAX_SILENCE_S
    max_lifetime_s: float = DEFAULT_MAX_LIFETIME_S
    #: How long to wait for a pod to publish its port. A cold machine pulling
    #: a 20 GB image is minutes; one that never publishes is released.
    pod_boot_timeout_s: float = 1800.0


@dataclasses.dataclass
class TrainerPod:
    """One rented pod and what we know about it without being able to call it."""

    pod_id: str
    shard: int
    started_at: float
    hourly_usd: float = 0.0
    last_checkin: float = 0.0
    rounds_done: int = 0
    #: Where to reach it. The orchestrator PULLS -- coderai already addresses
    #: pods to proxy inference, and a direct TCP mapping bypasses the RunPod
    #: proxy, so a 160 MB adapter meets no Cloudflare idle limit or body cap.
    #: Empty until the runtime publishes the mapping.
    url: str = ""
    secret: str = ""
    #: The card this pod actually got, for the page and the record. A fleet
    #: tries candidates in order, so which one it landed on is not knowable
    #: from the spec afterwards.
    gpu: str = ""

    def silent_for(self, now: Optional[float] = None) -> float:
        now = time.time() if now is None else now
        return now - (self.last_checkin or self.started_at)

    def age(self, now: Optional[float] = None) -> float:
        return (time.time() if now is None else now) - self.started_at

    def cost_so_far(self, now: Optional[float] = None) -> float:
        return self.age(now) / 3600.0 * float(self.hourly_usd or 0.0)


# --------------------------------------------------------------------------- #
# the durable record
# --------------------------------------------------------------------------- #
#
# Written before the pod exists and cleared after it is gone, so the window in
# which a pod is running and unrecorded is empty. The alternative -- record it
# after a successful create -- loses exactly the pods whose create succeeded and
# whose caller died before writing, which is the case that bills all night.

def _state_dir() -> str:
    from codai.api.loras import _loras_dir
    d = os.path.join(_loras_dir(), "_trainer_fleet")
    os.makedirs(d, exist_ok=True)
    return d


def _state_path(job_id: str) -> str:
    safe = "".join(c if (c.isalnum() or c in "-_") else "-" for c in job_id)
    return os.path.join(_state_dir(), f"{safe}.json")


_file_lock = threading.Lock()


def _write_state(job_id: str, rows: List[dict]) -> None:
    path = _state_path(job_id)
    tmp = path + ".tmp"
    with _file_lock:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"job": job_id, "pods": rows, "at": time.time()}, fh)
        os.replace(tmp, path)


def _read_state(job_id: str) -> List[dict]:
    try:
        with open(_state_path(job_id), "r", encoding="utf-8") as fh:
            return list((json.load(fh) or {}).get("pods") or [])
    except Exception:
        return []


def _read_envelope(job_id: str) -> dict:
    """The whole record, `at` included. `_read_state` drops everything else."""
    try:
        with open(_state_path(job_id), "r", encoding="utf-8") as fh:
            return dict(json.load(fh) or {})
    except Exception:
        return {}


#: How long a fleet record may go untouched before its pods are considered
#: abandoned. The owner stamps `at` every poll slice (~5s) and on every
#: rental, so this is three orders of magnitude of slack; what it really has
#: to survive is the gap between the last pod arriving and the first slice of
#: the round loop, which is the dataset pass -- measured at ~4 minutes for
#: 7308 examples.
ABANDONED_AFTER_S = 900.0


def touch(job_id: str) -> None:
    """Say the owner of this record is still alive.

    THE HEARTBEAT THE LEAK NEEDED. `sweep_orphans` is startup-only, and
    deliberately so: it argues that a record on disk must be abandoned
    because no job is running yet in this process. That argument holds at
    startup and nowhere else, so a fleet whose owner died -- a run that threw
    where its `finally` could not release, a job whose pods outlived it --
    kept billing until somebody restarted the container. Measured once:
    16 pods, ~$9.50/hr, swept only by the next boot.

    With `at` refreshed while the run lives, staleness is the signal, and the
    sweep no longer has to wait for a restart to be sure.
    """
    rows = _read_state(job_id)
    if not rows:
        return
    _write_state(job_id, rows)


def abandoned(ttl_s: float = ABANDONED_AFTER_S) -> List[dict]:
    """Recorded pods whose owner has stopped saying it is alive.

    Unlike :func:`orphans` this is safe to call WHILE jobs are running: a
    live run stamps its record continuously, so anything older than the TTL
    has no owner. A record with no `at` at all predates the heartbeat and is
    treated as stale, which is the conservative reading -- it was written by
    a process that is certainly gone.
    """
    cut = time.time() - max(1.0, float(ttl_s))
    out: List[dict] = []
    try:
        names = sorted(os.listdir(_state_dir()))
    except OSError:
        return out
    for name in names:
        if not name.endswith(".json"):
            continue
        job = name[:-5]
        env = _read_envelope(job)
        at = float(env.get("at") or 0.0)
        if at > cut:
            continue
        for row in (env.get("pods") or []):
            out.append({**row, "job": job, "at": at})
    return out


def sweep_orphans(client=None) -> dict:
    """Terminate every pod a dead fleet left behind. Call this at STARTUP.

    `orphans()` has always been able to find these -- its docstring calls
    itself "the answer to the orchestrator restarted mid-training" -- and
    nothing called it. So a restart or a crash left the fleet's pods rented
    with nobody to release them, and they survived until the GENERIC reaper
    noticed, 1200 seconds later: about $0.20 a pod, every restart, every
    failure. With several restarts in a day and fleets of four to eight, that
    is the systematic bleed, not an accident.

    No grace is needed here, unlike the generic reaper. That one must wait,
    because it cannot tell a sibling process's live pod from an abandoned one.
    This can: a fleet record only exists while a job owns it, `forget()`
    removes it when the fleet empties, and at startup no job is running in
    this process. A record that is still on disk is therefore a pod whose
    owner is definitively gone.

    Returns {"pods": n, "jobs": n, "failed": n}. Terminate failures leave the
    record alone so the next sweep tries again -- dropping it would be how a
    pod becomes invisible AND billing.
    """
    return _terminate_recorded(orphans(), client=client, why="sweep")


def sweep_abandoned(client=None, ttl_s: float = ABANDONED_AFTER_S) -> dict:
    """Terminate pods whose owner stopped reporting. Call this PERIODICALLY.

    The companion to :func:`sweep_orphans`, which only runs at startup
    because its safety argument -- "a record on disk must be abandoned,
    because no job is running yet" -- is only true then. This one is safe on
    a timer, because it reads the heartbeat instead: a live run refreshes its
    record every poll slice, so a record older than `ttl_s` has nobody.

    Without it the bleed had no upper bound but the next restart. With it the
    worst case is `ttl_s`.
    """
    out = _terminate_recorded(abandoned(ttl_s), client=client,
                              why="abandoned sweep")
    out["forgotten"] = _forget_stale_empty(ttl_s)
    return out


def _forget_stale_empty(ttl_s: float = ABANDONED_AFTER_S) -> int:
    """Drop records that hold no pods and have not been touched.

    A fleet that released everything leaves `{"pods": []}` behind, and
    nothing removed it: 83 such files had accumulated in a day. Harmless --
    there is no pod in them to bill -- but they grow without bound and every
    sweep reads all of them. Only the STALE ones go, because a job between
    its first record and its first rental is legitimately empty.
    """
    cut = time.time() - max(1.0, float(ttl_s))
    gone = 0
    try:
        names = sorted(os.listdir(_state_dir()))
    except OSError:
        return 0
    for name in names:
        if not name.endswith(".json"):
            continue
        job = name[:-5]
        env = _read_envelope(job)
        if env.get("pods"):
            continue
        if float(env.get("at") or 0.0) > cut:
            continue
        forget(job)
        gone += 1
    if gone:
        # SAID, because a silent cleanup is indistinguishable from one that
        # never ran: the first production sweep cleared 83 records and left
        # no trace of having done so, which cost ten minutes of wondering
        # whether the timer fired at all.
        print(f"[text-lora-fleet] abandoned sweep: forgot {gone} empty "
              f"fleet record(s) whose pods were all released", flush=True)
    return gone


def _terminate_recorded(rows: List[dict], *, client=None,
                        why: str = "sweep") -> dict:
    """Terminate every pod in `rows` and clear the records that empty out."""
    out = {"pods": 0, "jobs": 0, "failed": 0}
    if not rows:
        return out
    if client is None:
        try:
            from codai.models.manager import get_active_runpod_config
            from codai.api.runpod_client import RunpodClient
            acct = get_active_runpod_config()
            if acct is None or not getattr(acct, "enabled", False):
                return out
            client = RunpodClient(acct)
        except Exception as exc:
            print(f"[text-lora-fleet] {why}: no RunPod client ({exc}); "
                  f"{len(rows)} recorded pod(s) left for the reaper",
                  flush=True)
            return out
    jobs = {}
    for row in rows:
        jobs.setdefault(str(row.get("job") or ""), []).append(row)
    for job, pods in jobs.items():
        kept = []
        for row in pods:
            pid = str(row.get("pod_id") or "")
            if not pid or pid.startswith("pending-"):
                continue
            try:
                client.terminate_pod(pid)
            except Exception as exc:
                out["failed"] += 1
                kept.append(row)
                print(f"[text-lora-fleet] {why}: could not terminate {pid} "
                      f"from {job} ({exc}) — LEFT IN THE RECORD for the next "
                      f"sweep", flush=True)
                continue
            out["pods"] += 1
            rate = float(row.get("hourly_usd") or 0.0)
            _idle = ""
            if row.get("at"):
                _idle = (f", unreported for "
                         f"{int(time.time() - float(row['at']))}s")
            print(f"[text-lora-fleet] {why}: terminated {pid}, left by "
                  f"{job} (~${rate:.2f}/hr{_idle})", flush=True)
            try:
                from codai.api import runpod_worker as rw
                rw.unregister_pod(pid)
            except Exception:
                pass
        if kept:
            _write_state(job, kept)
        else:
            forget(job)
            out["jobs"] += 1
    if out["pods"] or out["failed"]:
        print(f"[text-lora-fleet] {why}: {out['pods']} pod(s) from "
              f"{out['jobs']} abandoned job(s) terminated"
              + (f", {out['failed']} could not be" if out["failed"] else ""),
              flush=True)
    return out


def orphans() -> List[dict]:
    """Every pod any fleet recorded and did not release.

    What a new process reads to find what the last one left running. This is the
    answer to "the orchestrator restarted mid-training": the pods are still
    rented, nothing in memory knows about them, and this does.
    """
    out: List[dict] = []
    try:
        names = sorted(os.listdir(_state_dir()))
    except OSError:
        return out
    for name in names:
        if not name.endswith(".json"):
            continue
        job = name[:-5]
        for row in _read_state(job):
            out.append({**row, "job": job})
    return out


def forget(job_id: str) -> None:
    """Drop a job's record once every pod in it is gone."""
    try:
        os.unlink(_state_path(job_id))
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# the fleet
# --------------------------------------------------------------------------- #

class TrainerFleet:
    """Rents, watches and releases the pods of one training job."""

    def __init__(self, client, job_id: str, spec: TrainerPodSpec, *,
                 pool_name: str = "",
                 ledger_key: str = "text-lora", port: int = 8000,
                 max_fleet_hourly_usd: float = 0.0,
                 budget_check: Optional[Callable[[float], str]] = None,
                 on_event: Optional[Callable[[str], None]] = None):
        """``max_fleet_hourly_usd`` caps the WHOLE fleet's rate; the spec's
        ``max_pod_hourly_usd`` caps each pod. They are different limits and a
        run usually wants both: one bounds the bill, the other bounds the
        surprise. Zero means no ceiling of that kind, and ``budget_check``
        (the account-wide caps) is consulted on top of both."""
        self.client = client
        self.job_id = job_id
        #: What the fleet's pods are registered UNDER. The job id changes on
        #: every resubmission, so keying the pool to it meant each attempt
        #: rented a brand-new fleet while the previous one kept billing --
        #: measured: eleven pools, 35 pods, $21.25/hr for a job using five.
        #: A caller passes something that SURVIVES a resubmission (the adapter
        #: name), and then an attempt can adopt what the last one left.
        self.pool_name = str(pool_name or job_id)
        self.spec = spec
        self.ledger_key = ledger_key
        self.port = int(port)
        self._budget_check = budget_check
        self.max_fleet_hourly_usd = float(max_fleet_hourly_usd or 0.0)
        self._on_event = on_event
        self.pods: Dict[int, TrainerPod] = {}
        self._lock = threading.RLock()
        #: Pods rented and reaped WITHOUT EVER ANSWERING, in total and in a
        #: row. A pod that worked and died is a pod to replace; a pod that
        #: never once checked in is capacity handing back dead machines, and
        #: the top-up could not tell the two apart -- it re-rented the slot
        #: immediately and for ever. Measured on one 15-hour run: 87 such
        #: releases, about $27 of A40s that trained nothing, five slots
        #: cycling rent-reap-rent on a ~30 minute period for the whole run.
        self._doa = 0
        self._doa_streak = 0
        self._doa_said = False
        # Anything a previous process left behind for THIS job is adopted rather
        # than abandoned: it is already rented and already being paid for.
        for row in _read_state(job_id):
            try:
                p = TrainerPod(pod_id=str(row["pod_id"]), shard=int(row["shard"]),
                               started_at=float(row.get("started_at") or time.time()),
                               hourly_usd=float(row.get("hourly_usd") or 0.0),
                               last_checkin=float(row.get("last_checkin") or 0.0))
                self.pods[p.shard] = p
                self._say(f"adopted pod {p.pod_id} for shard {p.shard} from a "
                          f"previous process")
            except Exception:
                continue

    # -- bookkeeping ------------------------------------------------------ #

    def _say(self, msg: str) -> None:
        print(f"[text-lora-fleet] {self.job_id}: {msg}", flush=True)
        if self._on_event is not None:
            try:
                self._on_event(msg)
            except Exception:
                pass

    def _flush(self) -> None:
        with self._lock:
            rows = [dataclasses.asdict(p) for p in self.pods.values()]
        _write_state(self.job_id, rows)

    def heartbeat(self) -> None:
        """Say this fleet still has an owner.

        Called from every wait the run does -- the boot, the dataset pass,
        each round slice -- because what the periodic sweep reads is the
        record's age. A fleet that stops calling this has lost its owner and
        its pods are terminated within `ABANDONED_AFTER_S`, instead of
        billing until somebody restarts the container.
        """
        with self._lock:
            if not self.pods:
                return
        try:
            self._flush()
        except Exception:
            pass        # a missed beat costs one sweep interval, never a crash

    def hourly_rate(self) -> float:
        with self._lock:
            return sum(float(p.hourly_usd or 0.0) for p in self.pods.values())

    # -- renting ---------------------------------------------------------- #

    def provision(self, shards: Sequence[int]) -> List[TrainerPod]:
        """Bring the fleet UP TO ``len(shards)`` pods, trying every shortfall.

        A partial fleet is a usable fleet -- the exchange averages what
        arrives -- so a budget ceiling reached halfway is reported and the job
        continues with fewer pods rather than failing or, worse, renting past
        the cap.

        FILL-TO-SIZE, not one-per-id. It used to rent a pod for each id in the
        list that the fleet did not already hold, which made it idempotent:
        `provision([0, 1])` twice rented two pods, not four. Dense ids removed
        that guard -- ids no longer come from the caller, so "already have
        this id" stopped meaning anything -- and a second call rented the
        whole list again. That is a money bug and the suite caught it: what
        the callers actually want is a fleet of a given size, which is also
        what `top_up` wants, so the contract says that instead.
        """
        want = len({int(x) for x in shards})
        with self._lock:
            need = max(0, want - len(self.pods))
        started: List[TrainerPod] = []
        for _ in range(need):
            # DENSE IDS, allocated here rather than taken from the caller's
            # list. The caller passes plan indices -- `range(n_shards)` -- and
            # when most of them have no capacity the survivors keep whatever
            # indices happened to succeed: a four-pod fleet reported shards
            # 17, 20, 24 and 25 out of a 32-shard plan, which reads as a bug
            # in the numbering and is impossible to match against a pod list.
            #
            # Safe because a shard id is now only a handle. It used to select
            # the slice -- position in the round's cohort -- so renumbering
            # would have changed what a pod trained; work comes off a queue
            # now and is claimed on demand, so the id names a pod and nothing
            # more.
            with self._lock:
                shard = next(i for i in range(10_000) if i not in self.pods)
            reason = self._refusal_for_one_more()
            if not reason and self._budget_check is not None:
                reason = self._budget_check(float(self.spec.hourly_usd or 0.0))
            if reason:
                self._say(f"stopping at {len(started)} pod(s): {reason}")
                break
            pod = self._rent_one(shard)
            if pod is None:
                # NOT a break. Capacity is per CARD: shard 3 failing to find
                # an A40 says nothing about shard 4 getting an A6000, and
                # stopping here abandoned every remaining shard on one
                # refusal -- so a job that asked for eight often got two and
                # never tried for the rest.
                continue
            started.append(pod)
        if started and len(started) < need:
            self._say(f"{len(started)} of {need} pod(s) rented; "
                      f"the rest had no capacity and will be retried between "
                      f"rounds")
        return started

    def adopt_existing(self, *, probe: bool = True) -> List[TrainerPod]:
        """Take over pods a previous attempt at THIS pool left running.

        The point of a pool name that outlives the job id: an attempt that
        failed, or a process that restarted, leaves rented pods behind, and
        the next attempt should use them rather than rent beside them. This is
        what the serving pools have always done -- "a surviving pod on a
        fallback card is one this pool rented and is already paying for".

        Shard numbers are assigned in order rather than restored. Since the
        partition is decided per ROUND from the cohort's positions, a shard id
        is only a handle; what must be preserved is the pod's url and its own
        bearer, both of which are in the registry.

        ``probe`` asks each pod whether it is alive before adopting it. A
        registry entry can outlive its pod, and adopting a corpse would fill a
        shard slot with something that never submits -- worse than renting,
        because it looks like progress.
        """
        try:
            from codai.api import runpod_worker as rw
            data = rw._registry_update(lambda d: False) or {}
        except Exception as exc:
            print(f"[text-lora-fleet] {self.job_id}: cannot read the registry "
                  f"to adopt ({exc})", flush=True)
            return []
        want_key = f"{_fleet_pool_prefix()}{self.pool_name}"
        with self._lock:
            have = {p.pod_id for p in self.pods.values()}
            used = set(self.pods)
        found = []
        for pod_id, info in (data.items() if isinstance(data, dict) else []):
            if not isinstance(info, dict) or str(pod_id) in have:
                continue
            if str(info.get("pool") or "") != want_key:
                continue
            url = str(info.get("url") or "").strip()
            if not url:
                continue          # never became reachable; nothing to adopt
            found.append((str(pod_id), info, url))
        if not found:
            return []
        taken: List[TrainerPod] = []
        for pod_id, info, url in sorted(found, key=lambda x: x[0]):
            shard = next(i for i in range(10_000) if i not in used)
            pod = TrainerPod(
                pod_id=pod_id, shard=shard, started_at=float(
                    info.get("created") or time.time()),
                hourly_usd=float(info.get("hourly_usd") or 0.0),
                url=url, secret=str(info.get("api_key") or ""),
                gpu=str(info.get("gpu") or ""))
            if probe:
                try:
                    st = poll_status(pod)
                    if not isinstance(st, dict) or not st:
                        raise RuntimeError("no status")
                except Exception as exc:
                    print(f"[text-lora-fleet] {self.job_id}: not adopting "
                          f"{pod_id} — it does not answer ({exc}); leaving it "
                          f"to the sweep", flush=True)
                    continue
            used.add(shard)
            with self._lock:
                self.pods[shard] = pod
            pod.last_checkin = time.time()
            taken.append(pod)
            self._register(pod)
            self._say(f"adopted pod {pod_id} as shard {shard} from an earlier "
                      f"attempt at {self.pool_name!r} (~${pod.hourly_usd:.2f}"
                      f"/hr, already paid for)")
        if taken:
            self._flush()
        return taken

    def top_up(self, want: int, *, cap: int = 0) -> List[TrainerPod]:
        """Try to reach ``want`` pods, renting only what is missing.

        Called between rounds. Waiting for a full fleet before starting is
        what made a large job impossible: the pods that arrive first bill
        while the others are hunted, and the bigger the fleet the less likely
        it ever assembles. Starting small and growing costs nothing extra --
        every pod is paid by the second either way -- and the round protocol
        re-partitions per round, so a pod that joins at round 5 simply takes a
        share of round 5.
        """
        with self._lock:
            have = set(self.pods)
        want = max(0, int(want))
        if cap > 0:
            want = min(want, int(cap))
        if len(have) >= want:
            return []
        # DEAD CAPACITY IS NOT WORTH RE-RENTING. Every reap used to be
        # followed straight back here, which is right for a pod that worked
        # and died and wrong for one that never answered: the slot is re-let
        # within the minute, the replacement is as dead as the last, and the
        # job pays the boot of each. After this many consecutive pods that
        # never once checked in, the fleet holds the size it has instead of
        # growing -- it does NOT shrink, and the first pod that answers
        # resets the streak, so a transient shortage costs nothing. On the
        # run that produced this, it would have stopped about $27 of rentals
        # that trained nothing without touching the pods that were working.
        if self._doa_streak >= DOA_STREAK_CAP and have:
            if not self._doa_said:
                self._doa_said = True
                self._say(
                    f"holding at {len(have)} pod(s) instead of {want}: the "
                    f"last {self._doa_streak} pod(s) rented never checked in "
                    f"({self._doa} so far this run). Capacity is handing back "
                    f"machines that do not answer; the fleet will grow again "
                    f"as soon as one does")
            return []
        self._doa_said = False
        # The SIZE, not the gaps: `provision` fills to the size it is given,
        # so handing it a list of missing ids would ask for that many MORE.
        return self.provision(list(range(want)))

    def _refusal_for_one_more(self) -> str:
        """Why one more pod may not be rented, or ''.

        Checked BEFORE the create call, because a cap discovered afterwards has
        already been exceeded and the pod is already billing.
        """
        rate = float(self.spec.hourly_usd or 0.0)
        pod_cap = float(self.spec.max_pod_hourly_usd or 0.0)
        if pod_cap > 0 and rate > pod_cap + 1e-9:
            return (f"a pod at ${rate:.2f}/hr is over the ${pod_cap:.2f}/hr "
                    f"per-pod ceiling")
        fleet_cap = self.max_fleet_hourly_usd
        if fleet_cap > 0:
            after = self.hourly_rate() + rate
            if after > fleet_cap + 1e-9:
                have = len(self.pods)
                return (f"{have} pod(s) at ${self.hourly_rate():.2f}/hr; one more "
                        f"would reach ${after:.2f}/hr, over the ${fleet_cap:.2f}/hr "
                        f"fleet ceiling")
        return ""

    def _rent_one(self, shard: int) -> Optional[TrainerPod]:
        import secrets as _secrets
        # A credential PER POD, minted here rather than taken from the caller.
        # Each pod's API is then reachable only with its own bearer, so one
        # leaked pod is one pod -- and nothing upstream has to remember to
        # supply it, which is the kind of omission that ships an open port.
        secret = _secrets.token_urlsafe(32)
        env = dict(self.spec.env or {})
        env.update({"CODERAI_TRAIN_JOB": self.job_id,
                    "CODERAI_TRAIN_SHARD": str(shard),
                    "CODERAI_TRAIN_SECRET": secret})
        # Recorded BEFORE the create call: a create that succeeds and a process
        # that dies before writing is the one case that bills unnoticed, and the
        # id is predictable only afterwards -- so a placeholder goes down first
        # and is replaced with the real id.
        placeholder = TrainerPod(pod_id=f"pending-{self.job_id}-{shard}",
                                 shard=shard, started_at=time.time(),
                                 hourly_usd=float(self.spec.hourly_usd or 0.0))
        with self._lock:
            self.pods[shard] = placeholder
        self._flush()
        pod_id = None
        last_error = ""
        _cands = list(self.spec.gpu_types or [""])
        _clouds = list(self.spec.gpu_clouds or [])
        _prices = list(self.spec.gpu_prices or [])
        won_gpu, won_price = "", float(self.spec.hourly_usd or 0.0)
        for _i, gpu in enumerate(_cands):
            cloud = (_clouds[_i] if _i < len(_clouds) else "") \
                or self.spec.cloud_type
            try:
                pod_id = self._create(shard, gpu, env, cloud)
                won_gpu = str(gpu)
                if _i < len(_prices) and float(_prices[_i]) > 0:
                    won_price = float(_prices[_i])
                break
            except Exception as exc:
                last_error = f"{gpu} [{cloud.lower()}]: {exc}"
                self._say(f"shard {shard}: {gpu} unavailable in "
                          f"{cloud.lower()} ({exc}); trying the next card")
        if pod_id is None:
            with self._lock:
                self.pods.pop(shard, None)
            self._flush()
            self._say(f"shard {shard}: no candidate card could be rented "
                      f"({last_error})")
            return None
        with self._lock:
            placeholder.pod_id = str(pod_id)
            placeholder.secret = secret
            # The CARD's price, not the list's estimate: the page, the ledger
            # and the fleet ceiling are only as honest as this number.
            placeholder.hourly_usd = won_price
            placeholder.gpu = won_gpu
            pod = placeholder
        self._flush()
        # INTO THE DEPLOYMENT REGISTRY, not only this fleet's own record file.
        # A trainer pod used to be invisible to every cleanup path: it is named
        # `coderai-train-job-…`, so the reaper's `coderai-<deployment>-` prefix
        # does not match it, and it was absent from the registry that startup
        # reclaim reads. While the job lived nothing noticed; when the process
        # died -- a crash, or a forced upgrade -- the pod billed on with no
        # owner and nothing to collect it.
        #
        # Registering fixes both ends with one mechanism: the entry has a live
        # owner, so `_known_pod_ids` counts it and the admin page stops
        # reporting a working training pod as unmanaged; and when the owner
        # stops heartbeating, the same entry stops vouching after the reap
        # grace and the pod is collected like any other leftover.
        self._register(pod)
        self._say(f"shard {shard}: pod {pod_id} rented — {won_gpu or '?'} "
                  f"at ${won_price:.2f}/hr")
        return pod

    def _register(self, pod: "TrainerPod") -> None:
        """Record this pod as the deployment's, owned by this job."""
        try:
            from codai.api import runpod_worker as rw
            rw.register_pod(
                str(pod.pod_id),
                # Namespaced by JOB, so the page can say which run owns it and
                # a stale entry names the job that died.
                pool_key=f"{_fleet_pool_prefix()}{self.pool_name}",
                url=str(pod.url or ""),
                api_key=str(pod.secret or ""),
                hourly_usd=float(pod.hourly_usd or self.spec.hourly_usd or 0.0),
                gpu=str(pod.gpu or ""),
                image=str(self.spec.image or ""))
        except Exception as exc:
            # Never fail a rental over bookkeeping -- but say so, because the
            # consequence is a pod that leaks if this process dies.
            print(f"[text-lora-fleet] {self.job_id}: could not register pod "
                  f"{pod.pod_id} ({exc}); if this process dies the pod will "
                  f"not be reclaimed or reaped automatically", flush=True)

    def _unregister(self, pod_id: str) -> None:
        try:
            from codai.api import runpod_worker as rw
            rw.unregister_pod(str(pod_id))
        except Exception:
            pass

    def _create(self, shard: int, gpu: str, env: Dict[str, str],
                cloud_type: str = "") -> str:
        """One create call for one candidate card, in that candidate's cloud."""
        if True:
            return self.client.create_pod(
                name=f"coderai-train-{self.job_id}-s{shard}",
                image=self.spec.image, gpu_type_id=gpu,
                port=self.port,
                cloud_type=(cloud_type or self.spec.cloud_type),
                container_disk_gb=self.spec.container_disk_gb,
                volume_gb=self.spec.volume_gb,
                network_volume_id=self.spec.network_volume_id,
                data_center_id=self.spec.data_center_id,
                registry_auth_id=self.spec.registry_auth_id,
                env=env, entrypoint=self.spec.entrypoint,
                start_cmd=self.spec.start_cmd,
                # Direct TCP: the orchestrator pulls a 160 MB adapter every
                # round, and the proxy hostname is behind Cloudflare with a
                # 100 s idle limit and a body cap.
                direct_tcp=True)

    def address(self, shard: int, *, timeout: float = 1800.0) -> str:
        """Wait for this pod to publish its ip:port, and remember it.

        Separate from renting because it is the slow half -- a cold machine
        pulling a 20 GB image is minutes -- and because a pod that never
        publishes a port has to be released rather than waited on forever.
        """
        with self._lock:
            pod = self.pods.get(int(shard))
        if pod is None:
            return ""
        if pod.url:
            return pod.url
        url = self.client.wait_ready(pod.pod_id, self.port,
                                     ready_timeout=timeout, direct_tcp=True)
        with self._lock:
            pod.url = str(url or "")
        self._flush()
        if pod.url:
            # Re-registered WITH the address. A pool's pod becomes reachable in
            # the registry the same way, which is how one row renderer serves
            # both without knowing what a trainer is.
            self._register(pod)
            self._say(f"shard {shard}: pod {pod.pod_id} reachable")
        return pod.url

    # -- liveness --------------------------------------------------------- #

    def note_checkin(self, shard: int, rounds_done: Optional[int] = None) -> None:
        """A shard reported in. THIS is the health signal, not a probe."""
        with self._lock:
            pod = self.pods.get(int(shard))
            if pod is None:
                return
            pod.last_checkin = time.time()
            if rounds_done is not None:
                pod.rounds_done = int(rounds_done)
        self._flush()

    def to_reap(self, now: Optional[float] = None) -> List[TrainerPod]:
        """Pods that have gone quiet, or outlived the hard ceiling."""
        now = time.time() if now is None else now
        out = []
        with self._lock:
            for pod in self.pods.values():
                if pod.pod_id.startswith("pending-"):
                    continue
                if pod.age(now) >= float(self.spec.max_lifetime_s):
                    out.append(pod)
                elif pod.silent_for(now) >= float(self.spec.max_silence_s):
                    out.append(pod)
        return out

    # -- releasing -------------------------------------------------------- #

    def release(self, shard: int, reason: str = "") -> bool:
        """Terminate one pod, record what it cost, forget it."""
        with self._lock:
            pod = self.pods.get(int(shard))
        if pod is None:
            return False
        ok = True
        if not pod.pod_id.startswith("pending-"):
            try:
                self.client.terminate_pod(pod.pod_id)
            except Exception as exc:
                # Kept in the record on purpose: a terminate that failed means
                # the pod may still be running, and dropping it here is exactly
                # how a pod ends up billing with nobody owning it.
                self._say(f"shard {shard}: could not terminate {pod.pod_id}: "
                          f"{exc} — LEFT IN THE RECORD for a later sweep")
                return False
        # Counted on the way out, where the pod's whole life is known: a url
        # means it answered at least once, so it was a pod that worked and
        # died. No url and it never answered at all.
        if str(getattr(pod, "url", "") or ""):
            self._doa_streak = 0
        else:
            self._doa += 1
            self._doa_streak += 1
        cost = pod.cost_so_far()
        try:
            from codai.api import runpod_ledger
            runpod_ledger.record_spend(self.ledger_key, cost,
                                       {"job": self.job_id, "shard": shard,
                                        "pod_id": pod.pod_id, "reason": reason})
        except Exception:
            pass
        # AFTER the terminate succeeded, and only then: a failed terminate
        # returns above with the pod still in the record, and dropping its
        # registry entry there would hide a pod that is still running from the
        # one sweep meant to find it.
        self._unregister(pod.pod_id)
        with self._lock:
            self.pods.pop(int(shard), None)
            empty = not self.pods
        self._flush()
        if empty:
            forget(self.job_id)
        self._say(f"shard {shard}: pod {pod.pod_id} released after "
                  f"{pod.age() / 60:.1f} min (~${cost:.2f})"
                  + (f" — {reason}" if reason else ""))
        return ok

    def release_all(self, reason: str = "job finished") -> int:
        """Terminate every pod. Call this from a finally: it is the guarantee.

        Returns how many were released. A pod that refuses to terminate stays in
        the on-disk record, so :func:`orphans` still names it afterwards.
        """
        with self._lock:
            shards = sorted(self.pods)
        n = 0
        for shard in shards:
            if self.release(shard, reason):
                n += 1
        return n


# --------------------------------------------------------------------------- #
# the per-model configuration
# --------------------------------------------------------------------------- #
#
# Where the GPU, the image and the money live: the model's own entry in
# models.json, edited on the Models page, exactly like its `runpod` block.
#
# NOT in the request. That is the same rule `/v1/runpod/scale` states for
# itself -- "there is deliberately no way for a request to change a model's
# configuration" -- and it is what makes the route safe to expose: a caller can
# start a training run without being able to pick a bigger card, a different
# image, or a higher price than the operator configured.
#
# A request may still NARROW what is configured (fewer pods, a lower ceiling),
# because asking for less than you are allowed is always safe. The narrowing
# dies with the request; it cannot raise anything.

@dataclasses.dataclass
class LoraTrainConfig:
    """A model's `lora_train` block: how THIS model is trained on rented GPUs."""

    enabled: bool = False
    #: What to TRAIN, when that is not what this entry SERVES. A model served
    #: from a pod as AWQ, or with no local path at all, cannot be the base of a
    #: QLoRA: bitsandbytes quantizes an unquantized checkpoint, and there is
    #: nothing on disk here to quantize. Same idea as the image trainer's
    #: `lora_train_base_model` -- train against this, serve with that. Blank
    #: falls back to the entry's own path or id.
    base_model: str = ""
    #: The floor, in GB. Any card with at least this much VRAM is acceptable,
    #: and the cheapest eligible one is tried first. A fleet of unequal cards
    #: is fine here precisely because of how rounds end: a round closes on its
    #: deadline with whoever arrived, and the average is weighted by the
    #: examples each pod actually trained -- so a slower card contributes less
    #: rather than holding up everyone else.
    min_vram_gb: float = 0.0
    #: An optional allow-list, as RunPod names them, comma-separated. Empty
    #: means "any card that meets the floor". Narrowing only: a card named here
    #: that is under the floor or over the ceiling is still refused.
    gpu_types: str = ""
    cloud_type: str = "SECURE"
    image: str = "ghcr.io/nextime/coderai-trainer-text:latest"
    #: The operator's ceiling on how many pods one job may rent. Named like the
    #: serving block's `max_pods` because it means the same thing and a request
    #: may only ever narrow it.
    max_pods: int = 1
    #: How many pods to rent AT THE START. `max_pods` is the ceiling the fleet
    #: may be maintained at; this is where it begins. They were one field, so
    #: `pods: 8, max_pods: 16` set the ceiling to 8 and the pool could never
    #: reach 16 however much capacity appeared -- the request's own starting
    #: number silently became its limit. 0 = start at the ceiling.
    pods_start: int = 0
    #: The fewest pods a job will tolerate, from a request's `min_pods`. It is
    #: NOT a target and NOT a gate -- a run that can only get one pod runs on
    #: one, because every round trains the whole dataset whatever the cohort
    #: size. Recorded so the run can say how far below the aim it started.
    min_pods: int = 0
    rounds: int = 1
    container_disk_gb: int = 120
    network_volume_id: str = ""
    #: The ceilings. Checked against the GPU catalogue's real price before
    #: anything is rented, so they bind on what RunPod charges rather than on
    #: what a request claimed.
    max_pod_hourly_usd: float = 0.0
    max_fleet_hourly_usd: float = 0.0
    pod_max_silence_s: float = 1800.0
    pod_max_lifetime_s: float = 86400.0
    poll_interval_s: float = 20.0
    pod_boot_timeout_s: float = 1800.0
    #: Training defaults for this model; a request may override the ones that
    #: cost nothing to change (rank, lr, length) but not the ones that cost
    #: money (pods, rounds, cards, ceilings).
    rank: int = 16
    alpha: int = 0                    # 0 = 2x rank
    dropout: float = 0.05
    learning_rate: float = 1e-4
    max_len: int = 8192
    accumulation_steps: int = 8
    #: A Hugging Face token for the pods to pull the base with. Empty means the
    #: orchestrator's own environment is used instead (see hf_token_for_pods),
    #: and empty in both places means every pod pulls anonymously -- which is a
    #: gated-repo failure for some bases and a rate-limited one for the rest,
    #: minutes into a paid boot. It is a SECRET: it reaches the pod env and is
    #: never logged, never echoed in a job message, and never in a config dump.
    hf_token: str = ""


def parse_model_lora_train(block: Optional[dict]) -> LoraTrainConfig:
    """A models.json `lora_train` block to a config, unknown keys ignored."""
    cfg = LoraTrainConfig()
    if not isinstance(block, dict):
        return cfg
    for f in dataclasses.fields(cfg):
        if f.name not in block:
            continue
        raw = block[f.name]
        try:
            if f.type is bool or f.name == "enabled":
                setattr(cfg, f.name, bool(raw))
            elif f.name in ("gpu_types", "cloud_type", "image",
                            "network_volume_id", "base_model"):
                setattr(cfg, f.name, str(raw or "").strip())
            elif f.name in ("max_pods", "rounds", "container_disk_gb", "rank",
                            "alpha", "accumulation_steps", "max_len"):
                setattr(cfg, f.name, int(raw))
            else:
                setattr(cfg, f.name, float(raw))
        except (TypeError, ValueError):
            continue        # a malformed value keeps the default, loudly below
    return cfg


def config_for_model(model_key: str) -> LoraTrainConfig:
    """The `lora_train` block for this key, by entry name OR by training base.

    Two names mean the same model to a human and different keys to code: the
    catalogue entry is ``qwen38`` while the thing actually trained is
    ``Qwen/Qwen3.8-27B``, and a request may reasonably say either. Looking only
    at the entry's own names meant the second one found no block, defaulted to
    disabled, and the job fell through to training locally -- on an
    orchestrator with no GPU, which is tens of gigabytes of download and then
    a CUDA error that names nothing.

    So: the entry's names first, then any entry whose ``lora_train.base_model``
    is this key. An ambiguous match is REFUSED rather than guessed, because
    silently picking one of two configurations is how a run uses the wrong
    ceiling.
    """
    try:
        from codai.models.manager import _model_entry_for
        entry = _model_entry_for(model_key) or {}
        if entry.get("lora_train"):
            return parse_model_lora_train(entry["lora_train"])
    except Exception:
        pass
    # Not found in memory -- which does not mean not configured. Both that
    # helper and the scan below read the catalogue a PROCESS happens to hold
    # (codai.admin.routes.config_manager.models_data), and an engine process
    # where the admin module was never initialised holds nothing. The block is
    # then invisible while being right there in models.json, and the honest
    # report of that is "I could not see the catalogue", not "there is no
    # block". So the file is the fallback, and the file is the truth.
    for name_match in _entry_from_disk(model_key):
        if name_match.get("lora_train"):
            return parse_model_lora_train(name_match["lora_train"])
    want = (model_key or "").strip().lower()
    if not want:
        return LoraTrainConfig()
    matches = []
    try:
        # Memory merged with the FILE: a parked training entry exists only in
        # the latter, and it is exactly the kind of entry this scan is for.
        sources = [_catalogue(), _file_catalogue()]
        seen_ids = set()
        for lst in [l for src in sources for l in src.values()]:
            if not isinstance(lst, list):
                continue
            for m in lst:
                if not isinstance(m, dict):
                    continue
                blk = m.get("lora_train")
                if not isinstance(blk, dict):
                    continue
                if str(blk.get("base_model") or "").strip().lower() == want:
                    key = str(m.get("alias") or m.get("id") or id(m))
                    if key in seen_ids:
                        continue      # the same entry from both sources
                    seen_ids.add(key)
                    matches.append((key, blk))
    except Exception:
        return LoraTrainConfig()
    if len(matches) > 1:
        names = ", ".join(n for n, _ in matches)
        raise ValueError(
            f"{model_key!r} is the training base of more than one model "
            f"({names}) — name the catalogue entry instead, so the ceilings and "
            f"pod count are not picked for you")
    if matches:
        return parse_model_lora_train(matches[0][1])
    return LoraTrainConfig()


def narrow(cfg: LoraTrainConfig, *, pods: Optional[int] = None,
           min_pods: Optional[int] = None,
           max_pods: Optional[int] = None,
           rounds: Optional[int] = None,
           max_pod_hourly_usd: Optional[float] = None,
           max_fleet_hourly_usd: Optional[float] = None) -> LoraTrainConfig:
    """Apply a request's narrowing. It can only ever ask for LESS.

    ``pods`` is how many this job wants, ``max_pods`` a ceiling it will accept;
    both are clamped by the operator's ``max_pods``, the same way a scale
    request may only narrow a model's pool. Asking for more is not worth
    failing a job over -- it is a client that does not know the limit -- so it
    is clamped and said out loud. What matters is that it cannot take effect.

    ``min_pods`` is the floor the job TOLERATES, not a size it asks for: a
    request for `pods: 8, min_pods: 1` wants eight and accepts one. It is
    recorded and gates nothing. What the fleet is held at is ``pods``.
    """
    out = dataclasses.replace(cfg)
    ceiling = int(cfg.max_pods or 1)
    if max_pods is not None and int(max_pods) > 0:
        want = int(max_pods)
        ceiling = min(want, ceiling)
        if want > cfg.max_pods:
            print(f"[text-lora] request allowed up to {want} pods; the model "
                  f"permits {cfg.max_pods} — using {ceiling}", flush=True)
    out.max_pods = ceiling
    if pods is not None and int(pods) > 0:
        want = int(pods)
        # The STARTING size, not the ceiling. Writing it to max_pods made a
        # request for "8 now, up to 16" mean "8, ever".
        out.pods_start = min(want, ceiling)
        if want > ceiling:
            print(f"[text-lora] request asked to start on {want} pods; the "
                  f"ceiling is {ceiling} — starting on {out.pods_start}",
                  flush=True)
    if min_pods is not None and int(min_pods) > 0:
        # RECORDED, and the aim ONLY WHEN `pods` names no number. `min_pods`
        # is how few this job will put up with, not how many it wants: a
        # client sending `pods: 8, min_pods: 1` is saying "eight please, and
        # one is better than nothing", so it must not drag the aim down to
        # one. But a request that asks for `min_pods: 4` and nothing else has
        # said the only number it is going to say, and treating that as "no
        # preference, use the operator's ceiling" ignores it.
        out.min_pods = min(int(min_pods), ceiling)
        if not (pods is not None and int(pods or 0) > 0):
            # It is then the only number this request named, and it stands in
            # for `pods` exactly as `pods` would -- including raising the
            # block's own starting size, which a request may do up to the
            # ceiling.
            out.pods_start = out.min_pods
    if rounds is not None and int(rounds) > 0:
        out.rounds = int(rounds)        # rounds change duration, not the fleet
    for name, want in (("max_pod_hourly_usd", max_pod_hourly_usd),
                       ("max_fleet_hourly_usd", max_fleet_hourly_usd)):
        if want is None or float(want) <= 0:
            continue
        have = float(getattr(cfg, name) or 0.0)
        new = float(want) if have <= 0 else min(float(want), have)
        if have > 0 and float(want) > have:
            print(f"[text-lora] request asked for {name}=${float(want):.2f}; "
                  f"the model allows ${have:.2f} — using ${new:.2f}", flush=True)
        setattr(out, name, new)
    return out


# --------------------------------------------------------------------------- #
# driving a pod
# --------------------------------------------------------------------------- #
#
# The orchestrator's side of the pull: push the dataset once, push a round,
# poll, fetch the adapter. Plain HTTP over the direct TCP mapping, with the
# pod's own bearer.
#
# Timeouts are per call and generous on the bodies: a 119 MB dataset and a
# 160 MB adapter over a datacenter link are tens of seconds, and a timeout
# shorter than the transfer turns a working pod into a reaped one.

class PodUnreachable(FleetError):
    """A per-pod HTTP call failed at the transport. NEVER fatal to the job.

    These helpers let `requests` exceptions straight through, and three of
    them are called from the round loop with no guard, so one pod going dark
    at a round boundary killed the whole run. Measured: a `push_start_adapter`
    to a pod that had stopped answering blocked for its full 900s timeout and
    then raised ConnectionError, which released 21 HEALTHY pods and discarded
    39 minutes of a paid fleet at 7/731 blocks.

    `push_dataset`'s call site already had the right answer, and said so --
    "the rest of the fleet is unaffected, where raising would have taken down
    a run that has working pods". This makes that reachable for the rest: a
    dead pod is a routine event in a rented fleet, so it is reported as one
    and the pod is replaced.
    """


@contextlib.contextmanager
def _transport(pod: "TrainerPod", what: str):
    """Turn a transport failure on `pod` into :class:`PodUnreachable`.

    Only `requests.RequestException` -- an OSError from the local disk while
    streaming an adapter is a real error and must not be relabelled as the
    pod being unreachable.
    """
    import requests
    try:
        yield
    except requests.RequestException as exc:
        raise PodUnreachable(
            f"shard {pod.shard}: {what} failed at the transport: "
            f"{type(exc).__name__}: {exc}") from exc


class PodBusy(RuntimeError):
    """The pod is already working; a second round was refused."""


def push_dataset(pod: TrainerPod, path: str, *, timeout: float = 900.0) -> int:
    """Upload the training set. Once per pod, not once per round."""
    import requests
    with open(path, "rb") as fh:
        data = fh.read()
    with _transport(pod, "/dataset"):
        r = requests.post(pod.url.rstrip("/") + "/dataset", data=data,
                      headers={"Authorization": f"Bearer {pod.secret}",
                               "Content-Type": "application/x-ndjson"},
                      timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"shard {pod.shard}: /dataset answered "
                           f"{r.status_code}: {(r.text or '')[:200]}")
    return len(data)


def push_start_adapter(pod: TrainerPod, path: str, *,
                       timeout: float = 300.0) -> int:
    """Send the previous round's average for this pod to resume from."""
    import requests
    with open(path, "rb") as fh:
        data = fh.read()
    with _transport(pod, "/adapter/start"):
        r = requests.put(pod.url.rstrip("/") + "/adapter/start", data=data,
                         headers={"Authorization": f"Bearer {pod.secret}"},
                         timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"shard {pod.shard}: /adapter/start answered "
                           f"{r.status_code}: {(r.text or '')[:200]}")
    return len(data)


def start_round(pod: TrainerPod, payload: Dict[str, Any], *,
                timeout: float = 60.0) -> None:
    """Tell the pod to train one round. 409 means it is still on the last one."""
    import requests
    with _transport(pod, "/round"):
        r = requests.post(pod.url.rstrip("/") + "/round", json=payload,
                          headers={"Authorization": f"Bearer {pod.secret}"},
                          timeout=timeout)
    if r.status_code == 409:
        raise PodBusy(f"shard {pod.shard} is still working: "
                      f"{(r.text or '')[:200]}")
    if r.status_code not in (200, 202):
        raise RuntimeError(f"shard {pod.shard}: /round answered "
                           f"{r.status_code}: {(r.text or '')[:200]}")


def poll_status(pod: TrainerPod, *, timeout: float = 30.0) -> Dict[str, Any]:
    """What the pod is doing. The liveness signal, and more honest than a ping.

    Unreachable is reported rather than raised: one failed poll on a busy
    machine is not a dead pod, and the fleet's silence window is what decides
    that.
    """
    import requests
    try:
        r = requests.get(pod.url.rstrip("/") + "/status",
                         headers={"Authorization": f"Bearer {pod.secret}"},
                         timeout=timeout)
        if r.status_code != 200:
            return {"state": "unreachable", "http": r.status_code}
        return dict(r.json() or {})
    except Exception as exc:
        return {"state": "unreachable", "error": str(exc)[:200]}


def fetch_adapter(pod: TrainerPod, out_path: str, *,
                  timeout: float = 900.0) -> str:
    """Pull the finished adapter, streamed, written atomically.

    Streamed because it is ~160 MB and reading it into memory per pod per round
    is pointless; renamed at the end so a transfer cut halfway cannot be
    averaged as though it were an adapter.
    """
    import requests
    tmp = out_path + ".part"
    with _transport(pod, "/adapter"), \
            requests.get(pod.url.rstrip("/") + "/adapter", stream=True,
                      headers={"Authorization": f"Bearer {pod.secret}"},
                      timeout=timeout) as r:
        if r.status_code != 200:
            raise RuntimeError(f"shard {pod.shard}: /adapter answered "
                               f"{r.status_code}: {(r.text or '')[:200]}")
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                if chunk:
                    fh.write(chunk)
    os.replace(tmp, out_path)
    return out_path


# --------------------------------------------------------------------------- #
# a staged timeline
# --------------------------------------------------------------------------- #

class Stages:
    """What happened, in order, with how long each part took.

    A distributed training run has eight or nine places it can fail and most of
    them look the same from outside: the job errors, minutes in, having spent
    money. A timeline turns "it failed" into "it rented in 4 s, became
    reachable in 390 s, and then the tokenizer load timed out" -- which names
    the cause and, as often, names the SLOW part that is not a failure at all.
    """

    def __init__(self):
        self._t0 = time.time()
        self._mark = self._t0
        self.rows: List[Dict[str, Any]] = []

    def done(self, name: str, **detail) -> None:
        now = time.time()
        self.rows.append({"stage": name, "seconds": round(now - self._mark, 1),
                          "at": round(now - self._t0, 1), "ok": True, **detail})
        self._mark = now
        print(f"[stage] {name}: {self.rows[-1]['seconds']}s", flush=True)

    def failed(self, name: str, error: str) -> None:
        now = time.time()
        self.rows.append({"stage": name, "seconds": round(now - self._mark, 1),
                          "at": round(now - self._t0, 1), "ok": False,
                          "error": str(error)[:300]})
        self._mark = now
        print(f"[stage] {name}: FAILED after {self.rows[-1]['seconds']}s — "
              f"{error}", flush=True)

    @property
    def total(self) -> float:
        return round(time.time() - self._t0, 1)

    def as_dict(self) -> Dict[str, Any]:
        bad = [r for r in self.rows if not r.get("ok")]
        return {"stages": self.rows, "total_seconds": self.total,
                "failed_at": bad[0]["stage"] if bad else None,
                # The slowest stage, because on a first run the answer to "why
                # did that take twenty minutes" is almost always one stage and
                # almost never the one anyone guessed.
                "slowest": (max(self.rows, key=lambda r: r["seconds"])["stage"]
                            if self.rows else None)}


def eligible_cards(client, cfg: LoraTrainConfig, *,
                   ceiling: float = 0.0) -> List[Dict[str, Any]]:
    """Cards that meet the VRAM floor and the price ceiling, cheapest first.

    Read from RunPod's own catalogue, so the ceiling binds on what is actually
    charged rather than on a number someone wrote down. Returned ranked because
    capacity moves: the cheapest eligible card is frequently sold out, and the
    next one down is a better answer than a failed job.

    A card the catalogue does not price is EXCLUDED when a ceiling is set --
    renting an unpriced card under a price ceiling is renting blind.
    """
    # THE POOL'S OWN RANKER, not a second implementation of it. This used to
    # re-derive "which card, which cloud, under what ceiling" in parallel with
    # `_rank_gpus`, which is how a trainer ended up with its own answers to
    # questions the serving fleets had already settled: preference versus hard
    # filter, multi-GPU pods counted per pod rather than per card, spot, the
    # cheaper/faster criteria, and the per-cloud pricing this file had to grow
    # by hand. One ranker means the trainer inherits all of it, and the next
    # improvement to it, instead of needing the same fix twice.
    #
    # The shape returned here is unchanged, because callers and the UI read it:
    # {id, vram_gb, hourly_usd, cloud_type}.
    try:
        from codai.api import runpod_worker as rw
    except Exception as exc:                      # pragma: no cover
        raise FleetError(f"cannot rank cards: {exc}") from exc
    mcfg = training_mcfg(cfg, ceiling=ceiling)
    try:
        ranked = rw._rank_gpus(client, mcfg, getattr(client, "_cfg", None))
    except Exception as exc:
        # The ranker raises when nothing fits, with the constraints spelled
        # out. That message is better than an empty list, but an empty list is
        # what this function has always returned, and `eligible_cards` is
        # called where "none" is a decision rather than an error.
        print(f"[text-lora] no eligible card: {exc}", flush=True)
        return []
    out: List[Dict[str, Any]] = []
    for c in ranked:
        out.append({"id": str(c.get("gpu_type_id") or ""),
                    "vram_gb": float(c.get("memory_gb") or 0.0),
                    "hourly_usd": float(c.get("price") or 0.0),
                    "cloud_type": str(c.get("cloud_type") or ""),
                    "display_name": str(c.get("display_name") or ""),
                    "gpu_count": int(c.get("gpu_count") or 1),
                    "is_spot": bool(c.get("is_spot"))})
    return out


def training_mcfg(cfg: "LoraTrainConfig", ceiling: float = 0.0):
    """A ``RunpodModelConfig`` describing what this training job may rent.

    The adapter between a `lora_train` block and the config every serving pool
    is already described by, so one ranker can answer for both. Nothing here
    is new policy -- it is the same floor, ceiling and clouds the block always
    meant, said in the vocabulary the rest of the RunPod code speaks.
    """
    from codai.api.runpod_worker import RunpodModelConfig
    m = RunpodModelConfig()
    m.cloud_types = list(clouds_for(cfg))
    m.min_vram_gb = float(cfg.min_vram_gb or 0.0)
    m.max_hourly_usd = float(ceiling or cfg.max_pod_hourly_usd or 0.0)
    # An ALLOW-LIST when the block names cards, not a preference. For a pool,
    # `gpu_types` is a preference and `allow_other_gpus` opens the rest -- the
    # right default there, because a named card being busy should not fail a
    # request. Here the list is the OPERATOR's, on a route a client can call,
    # so it has the same standing as max_pods and the price ceiling: a request
    # may narrow it and must not widen it. Treating it as a preference would
    # have let a job rent cards the operator never listed, which is why the
    # test for it is called "narrows but cannot widen".
    #
    # With no list, anything meeting the floor and the ceiling is fair game.
    m.gpu_types = [c.strip() for c in (cfg.gpu_types or "").split(",")
                   if c.strip()]
    m.allow_other_gpus = not m.gpu_types
    # Never spot for training: an interrupted pod loses the round, and a round
    # is the unit of progress here.
    m.allow_spot = False
    m.selection_criteria = "cheaper"
    m.gpu_count = 1
    return m


#: What `cloud_type` may say to mean "try both". A single value still means
#: only that one, so an install that chose SECURE deliberately keeps it.
_ANY_CLOUD = ("ANY", "BOTH", "SECURE,COMMUNITY", "COMMUNITY,SECURE", "*")


def clouds_for(cfg: "LoraTrainConfig") -> List[str]:
    """The clouds to consider, in preference order.

    SECURE first when both are allowed: for a tie in price it is the better
    place to run, and for a dataset of court documents that preference is not
    a detail. Community is reached when it is cheaper or when secure has no
    capacity, which is the whole reason for allowing both.
    """
    raw = (getattr(cfg, "cloud_type", "") or "SECURE").strip().upper()
    if raw in _ANY_CLOUD:
        return ["SECURE", "COMMUNITY"]
    return ["COMMUNITY"] if raw == "COMMUNITY" else ["SECURE"]


def describe_cards(cards: Sequence[Dict[str, Any]]) -> str:
    """One line naming what was chosen and what it costs."""
    if not cards:
        return "no eligible card"
    def _one(c):
        # The cloud is part of the identity of an offer, not decoration: the
        # same card at two prices is two different rentals.
        tag = str(c.get("cloud_type") or "")
        return (f"{c['id']} ({c['vram_gb']:.0f}GB ${c['hourly_usd']:.2f}/hr"
                + (f" {tag.lower()}" if tag else "") + ")")

    return ", ".join(_one(c) for c in cards[:6]) + ("…" if len(cards) > 6 else "")


def _models_json_path() -> str:
    """models.json, from the environment the launcher sets or the default."""
    import os as _os
    env = (_os.environ.get("CODERAI_CONFIG_DIR") or "").strip()
    if env:
        return _os.path.join(env, "models.json")
    try:
        from codai.platform_paths import default_config_dir
        return str(default_config_dir() / "models.json")
    except Exception:
        return _os.path.expanduser("~/.coderai/models.json")


def _catalogue() -> Dict[str, Any]:
    """The model catalogue: in-memory if a process has it, else from disk.

    Two sources because they fail differently. The in-memory one is current
    but only exists where the admin module was initialised; the file is always
    there but may trail an unsaved edit. Preferring memory and falling back to
    the file means a lookup answers correctly in both the front and an engine.
    """
    try:
        from codai.admin.routes import config_manager as cfg_mgr
        md = getattr(cfg_mgr, "models_data", None)
        if isinstance(md, dict) and md:
            return md
    except Exception:
        pass
    try:
        with open(_models_json_path(), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _file_catalogue() -> Dict[str, Any]:
    """The catalogue as models.json has it -- every entry, parked or not."""
    try:
        with open(_models_json_path(), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _entry_from_disk(model_key: str) -> List[Dict[str, Any]]:
    """Entries matching this key by alias, id, path or basename, FROM THE FILE.

    From the file unconditionally, which is the whole point and was the bug:
    this used to read the merged catalogue, which prefers the in-memory one --
    and the in-memory catalogue EXCLUDES parked entries. A training-only entry
    is naturally parked (nothing should serve it), so a populated-but-filtered
    memory view answered "no such entry" and the file was never consulted. A
    non-empty catalogue is not the same as a complete one.
    """
    import os as _os
    want = (model_key or "").strip().lower()
    if not want:
        return []
    out = []
    for lst in _file_catalogue().values():
        if not isinstance(lst, list):
            continue
        for m in lst:
            if not isinstance(m, dict):
                continue
            path = str(m.get("path") or "")
            names = {path.lower(), _os.path.basename(path.rstrip("/")).lower(),
                     str(m.get("alias") or "").lower(),
                     str(m.get("id") or "").lower()}
            if want in names:
                out.append(m)
    return out


def catalogue_source() -> str:
    """Where a lookup read the catalogue, for an error message.

    Names BOTH, because naming one was misleading: the message said "the
    in-memory catalogue" while the file held the answer, which is true of what
    the lookup did and useless for working out why.
    """
    import os as _os
    parts = []
    try:
        from codai.admin.routes import config_manager as cfg_mgr
        md = getattr(cfg_mgr, "models_data", None)
        if isinstance(md, dict) and md:
            n = sum(len(v) for v in md.values() if isinstance(v, list))
            parts.append(f"the in-memory catalogue ({n} entries, parked ones "
                         f"excluded)")
        else:
            parts.append("the in-memory catalogue (empty in this process)")
    except Exception:
        parts.append("the in-memory catalogue (unavailable)")
    path = _models_json_path()
    if _os.path.isfile(path):
        n = sum(len(v) for v in _file_catalogue().values()
                if isinstance(v, list))
        parts.append(f"{path} ({n} entries)")
    else:
        parts.append(f"{path} — WHICH DOES NOT EXIST")
    return " and ".join(parts)


#: Fallback names only, for the case where the install's own resolver cannot
#: be imported. The resolver is the source of record: it reads these AND the
#: token files, which is where this install keeps it.
_HF_TOKEN_ENV = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN")


def hf_token_for_pods(cfg: Optional["LoraTrainConfig"] = None) -> str:
    """The token to give the pods: the block's, else the orchestrator's own.

    There was no path for this at all -- a pod was created with three
    CODERAI_TRAIN_* variables and nothing else, so every pod pulled the base
    anonymously. That is a 404 on a gated repo and a rate-limit on an ungated
    one, and either way it happens minutes into a boot that is already being
    billed.

    The block wins over the environment so one model can train from a repo the
    rest of the install has no access to. Returns "" when neither has one; the
    caller decides whether to warn or refuse, because an ungated base genuinely
    does not need it.
    """
    tok = str(getattr(cfg, "hf_token", "") or "").strip()
    if tok:
        return tok
    # The install's OWN resolver, not a second scan of the environment. It
    # checks the same names and also the token FILES this install actually
    # keeps the token in, and its docstring says why that matters: "two
    # different answers to 'where is the token' is how one path authenticates
    # and another does not". A private copy here would have been exactly that.
    try:
        from codai.hf_cache import configured_hf_token
        return (configured_hf_token() or "").strip()
    except Exception:
        import os as _os
        for name in _HF_TOKEN_ENV:
            tok = (_os.environ.get(name) or "").strip()
            if tok:
                return tok
        return ""


def redact(text: str, *secrets: str) -> str:
    """Replace each secret with a marker. For log lines built from a spec.

    Cheap insurance: the pod env now carries a credential, and the fleet prints
    what it is doing. One formatted line that happens to include the env is all
    it takes to put a token in a log that gets pasted into a chat.
    """
    out = str(text or "")
    for sec in secrets:
        sec = str(sec or "").strip()
        if len(sec) >= 8:
            out = out.replace(sec, "<redacted>")
    return out


def _fleet_pool_prefix() -> str:
    """The registry prefix marking a pod as owned by a job's fleet.

    Read from runpod_worker so the writer and the three readers cannot drift:
    a prefix typed twice is a pod labelled "unowned" while a job is using it.
    """
    try:
        from codai.api.runpod_worker import FLEET_POOL_PREFIX
        return str(FLEET_POOL_PREFIX)
    except Exception:
        return "text-lora:"
