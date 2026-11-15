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

What replaces the health probe is the work itself: a training pod CALLS IN at
the end of every round to hand over its adapter (see
:mod:`codai.api.text_lora_rounds`). That check-in is the liveness signal, and it
is a better one than a probe -- a pod whose process is wedged still answers
HTTP, while one that has stopped finishing rounds has stopped being useful by
definition. It also means no pod ever has to be addressable, which is what makes
this work across datacenters at all.

THE RULE THIS MODULE EXISTS TO KEEP: nothing is left billing with nobody owning
it. Every pod id is written to disk BEFORE it is rented and removed only once it
is terminated, so a crashed orchestrator, a restarted container or a new process
can still find and kill what the last one started. :func:`orphans` is that list.
A fleet also enforces a maximum lifetime, because the failure that costs real
money is not a crash -- it is a training loop that keeps checking in forever.
"""

from __future__ import annotations

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


class FleetError(RuntimeError):
    """Provisioning refused, with the reason."""


@dataclasses.dataclass
class TrainerPodSpec:
    """What a training pod is: an image, a command, and the money it may cost."""

    image: str
    gpu_type_id: str
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
    max_silence_s: float = DEFAULT_MAX_SILENCE_S
    max_lifetime_s: float = DEFAULT_MAX_LIFETIME_S


@dataclasses.dataclass
class TrainerPod:
    """One rented pod and what we know about it without being able to call it."""

    pod_id: str
    shard: int
    started_at: float
    hourly_usd: float = 0.0
    last_checkin: float = 0.0
    rounds_done: int = 0

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
                 ledger_key: str = "text-lora", port: int = 8000,
                 budget_check: Optional[Callable[[float], str]] = None,
                 on_event: Optional[Callable[[str], None]] = None):
        self.client = client
        self.job_id = job_id
        self.spec = spec
        self.ledger_key = ledger_key
        self.port = int(port)
        self._budget_check = budget_check
        self._on_event = on_event
        self.pods: Dict[int, TrainerPod] = {}
        self._lock = threading.RLock()
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

    def hourly_rate(self) -> float:
        with self._lock:
            return sum(float(p.hourly_usd or 0.0) for p in self.pods.values())

    # -- renting ---------------------------------------------------------- #

    def provision(self, shards: Sequence[int]) -> List[TrainerPod]:
        """Rent one pod per shard, stopping at the first refusal.

        A partial fleet is a usable fleet -- the round protocol averages what
        arrives -- so a budget ceiling reached halfway is reported and the job
        continues with fewer shards rather than failing or, worse, renting past
        the cap.
        """
        started: List[TrainerPod] = []
        for shard in shards:
            with self._lock:
                if shard in self.pods:
                    continue
            if self._budget_check is not None:
                reason = self._budget_check(float(self.spec.hourly_usd or 0.0))
                if reason:
                    self._say(f"stopping at {len(started)} pod(s): {reason}")
                    break
            pod = self._rent_one(shard)
            if pod is None:
                break
            started.append(pod)
        return started

    def _rent_one(self, shard: int) -> Optional[TrainerPod]:
        env = dict(self.spec.env or {})
        env.update({"CODERAI_TRAIN_JOB": self.job_id,
                    "CODERAI_TRAIN_SHARD": str(shard)})
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
        try:
            pod_id = self.client.create_pod(
                name=f"coderai-train-{self.job_id}-s{shard}",
                image=self.spec.image, gpu_type_id=self.spec.gpu_type_id,
                port=self.port, cloud_type=self.spec.cloud_type,
                container_disk_gb=self.spec.container_disk_gb,
                volume_gb=self.spec.volume_gb,
                network_volume_id=self.spec.network_volume_id,
                data_center_id=self.spec.data_center_id,
                registry_auth_id=self.spec.registry_auth_id,
                env=env, entrypoint=self.spec.entrypoint,
                start_cmd=self.spec.start_cmd)
        except Exception as exc:
            with self._lock:
                self.pods.pop(shard, None)
            self._flush()
            self._say(f"shard {shard}: RunPod refused the pod: {exc}")
            return None
        with self._lock:
            placeholder.pod_id = str(pod_id)
            pod = placeholder
        self._flush()
        self._say(f"shard {shard}: pod {pod_id} rented "
                  f"(~${self.spec.hourly_usd:.2f}/hr)")
        return pod

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
        cost = pod.cost_so_far()
        try:
            from codai.api import runpod_ledger
            runpod_ledger.record_spend(self.ledger_key, cost,
                                       {"job": self.job_id, "shard": shard,
                                        "pod_id": pod.pod_id, "reason": reason})
        except Exception:
            pass
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
