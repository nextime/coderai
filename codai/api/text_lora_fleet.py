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
    #: Ranked candidates, cheapest first. A list because RunPod capacity moves:
    #: the card that was free when the catalogue was read is often gone by the
    #: time the pod is requested, and a single choice turns that into a failed
    #: job instead of the next-cheapest card.
    gpu_types: List[str] = dataclasses.field(default_factory=list)
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
        self.spec = spec
        self.ledger_key = ledger_key
        self.port = int(port)
        self._budget_check = budget_check
        self.max_fleet_hourly_usd = float(max_fleet_hourly_usd or 0.0)
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
            reason = self._refusal_for_one_more()
            if not reason and self._budget_check is not None:
                reason = self._budget_check(float(self.spec.hourly_usd or 0.0))
            if reason:
                self._say(f"stopping at {len(started)} pod(s): {reason}")
                break
            pod = self._rent_one(shard)
            if pod is None:
                break
            started.append(pod)
        return started

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
        for gpu in (self.spec.gpu_types or [""]):
            try:
                pod_id = self._create(shard, gpu, env)
                break
            except Exception as exc:
                last_error = f"{gpu}: {exc}"
                self._say(f"shard {shard}: {gpu} unavailable ({exc}); "
                          f"trying the next card")
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
            pod = placeholder
        self._flush()
        self._say(f"shard {shard}: pod {pod_id} rented "
                  f"(~${self.spec.hourly_usd:.2f}/hr)")
        return pod

    def _create(self, shard: int, gpu: str, env: Dict[str, str]) -> str:
        """One create call for one candidate card."""
        if True:
            return self.client.create_pod(
                name=f"coderai-train-{self.job_id}-s{shard}",
                image=self.spec.image, gpu_type_id=gpu,
                port=self.port, cloud_type=self.spec.cloud_type,
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
    want = (model_key or "").strip().lower()
    if not want:
        return LoraTrainConfig()
    matches = []
    try:
        from codai.admin.routes import config_manager as cfg_mgr
        md = getattr(cfg_mgr, "models_data", None) or {}
        for lst in md.values():
            if not isinstance(lst, list):
                continue
            for m in lst:
                if not isinstance(m, dict):
                    continue
                blk = m.get("lora_train")
                if not isinstance(blk, dict):
                    continue
                if str(blk.get("base_model") or "").strip().lower() == want:
                    matches.append((str(m.get("alias") or m.get("id") or "?"), blk))
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
        out.max_pods = min(want, ceiling)
        if want > ceiling:
            print(f"[text-lora] request asked for {want} pods; the ceiling is "
                  f"{ceiling} — using {out.max_pods}", flush=True)
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

class PodBusy(RuntimeError):
    """The pod is already working; a second round was refused."""


def push_dataset(pod: TrainerPod, path: str, *, timeout: float = 900.0) -> int:
    """Upload the training set. Once per pod, not once per round."""
    import requests
    with open(path, "rb") as fh:
        data = fh.read()
    r = requests.post(pod.url.rstrip("/") + "/dataset", data=data,
                      headers={"Authorization": f"Bearer {pod.secret}",
                               "Content-Type": "application/x-ndjson"},
                      timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"shard {pod.shard}: /dataset answered "
                           f"{r.status_code}: {(r.text or '')[:200]}")
    return len(data)


def push_start_adapter(pod: TrainerPod, path: str, *,
                       timeout: float = 900.0) -> int:
    """Send the previous round's average for this pod to resume from."""
    import requests
    with open(path, "rb") as fh:
        data = fh.read()
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
                  timeout: float = 1800.0) -> str:
    """Pull the finished adapter, streamed, written atomically.

    Streamed because it is ~160 MB and reading it into memory per pod per round
    is pointless; renamed at the end so a transfer cut halfway cannot be
    averaged as though it were an adapter.
    """
    import requests
    tmp = out_path + ".part"
    with requests.get(pod.url.rstrip("/") + "/adapter", stream=True,
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
    want_secure = (cfg.cloud_type or "SECURE").strip().upper() != "COMMUNITY"
    allow = {c.strip().lower() for c in (cfg.gpu_types or "").split(",")
             if c.strip()}
    floor = float(cfg.min_vram_gb or 0.0)
    out: List[Dict[str, Any]] = []
    for g in (client.list_gpu_types() or []):
        gid = str(g.get("id") or "").strip()
        if not gid:
            continue
        if allow and gid.lower() not in allow:
            continue
        try:
            vram = float(g.get("memory_gb") or 0)
        except (TypeError, ValueError):
            vram = 0.0
        if floor > 0 and vram < floor:
            continue
        price = g.get("secure_price") if want_secure else g.get("community_price")
        if price is None:
            price = g.get("on_demand_price")
        if price is None:
            if ceiling > 0:
                continue
            price = 0.0
        if ceiling > 0 and float(price) > ceiling + 1e-9:
            continue
        out.append({"id": gid, "vram_gb": vram, "hourly_usd": float(price)})
    out.sort(key=lambda c: (c["hourly_usd"], -c["vram_gb"]))
    return out


def describe_cards(cards: Sequence[Dict[str, Any]]) -> str:
    """One line naming what was chosen and what it costs."""
    if not cards:
        return "no eligible card"
    return ", ".join(f"{c['id']} ({c['vram_gb']:.0f}GB ${c['hourly_usd']:.2f}/hr)"
                     for c in cards[:6]) + ("…" if len(cards) > 6 else "")
