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

"""A fake trainer fleet, so the distributed run can be EXECUTED in a test.

Not a test module: shared by the files that drive
:func:`codai.api.loras._text_train_distributed` for real rather than reading
its source. The reason it exists is that the defects worth catching here are
control flow -- a `with ThreadPoolExecutor(...)` block whose exit joins every
future, a top-up gated on a round number -- and source assertions cannot see
either. Several of those tests were string matches, and they all passed while
production could not get past round 1.

`FakeFleet` makes each shard addressable at a time the test chooses, so "pod 0
is up and pod 2 never will be" is expressible, which is the case the real
scheduling has to handle.
"""

from __future__ import annotations

import os
import time
import types
from typing import Dict, List, Optional

from codai.api import text_lora_fleet as flt


class Pod:
    """Enough of a :class:`TrainerPod` for the run to drive it."""

    def __init__(self, shard: int):
        self.shard = shard
        self.url = ""
        self.api_key = "k"
        self.pod_id = f"pod{shard}"

    def cost_so_far(self) -> float:
        return 0.0


class FakeFleet:
    """`ready_after` maps shard -> seconds from construction until it answers.

    A shard absent from the mapping NEVER becomes addressable: that is the
    dead-pod case, and it is the one that used to hold up the whole fleet.
    """

    def __init__(self, ready_after: Dict[int, float],
                 *, top_up_gives: Optional[List[int]] = None):
        self.ready_after = dict(ready_after)
        self.born = time.time()
        self.pods: Dict[int, Pod] = {}
        self.released: List[tuple] = []
        self.top_ups: List[tuple] = []
        self._top_up_gives = list(top_up_gives or [])

    # -- the surface the run uses ----------------------------------------- #
    def adopt_existing(self):
        return []

    def provision(self, shards):
        # ONLY WHAT COULD BE RENTED. The real fleet registers a pod when
        # RunPod gives it one; a shard with no capacity is simply absent, and
        # a fake that registered all of them would hide every shortfall --
        # which is exactly what the top-up is for.
        for sh in shards:
            if int(sh) in self.ready_after:
                self.pods.setdefault(int(sh), Pod(int(sh)))
        return list(self.pods.values())

    def address(self, shard, timeout=0.0):
        due = self.ready_after.get(int(shard))
        if due is None:
            # Exactly how the real one reports a slice running out, because
            # the caller classifies on that text.
            time.sleep(min(float(timeout), 0.05))
            raise flt.FleetError(
                f"Pod pod{shard} did not expose port 8000 within {timeout}")
        deadline = time.time() + max(0.0, float(timeout))
        while True:
            if time.time() - self.born >= due:
                self.pods[int(shard)].url = f"http://pod{shard}:8000"
                return self.pods[int(shard)].url
            if time.time() >= deadline:
                raise flt.FleetError(
                    f"Pod pod{shard} did not expose port 8000 within {timeout}")
            time.sleep(0.01)

    def note_checkin(self, shard, rounds_done=None):
        pass

    def release(self, shard, reason=""):
        self.released.append((int(shard), str(reason)))
        self.pods.pop(int(shard), None)

    def release_all(self, reason=""):
        n = len(self.pods)
        self.pods.clear()
        return n

    def top_up(self, target, cap=0):
        self.top_ups.append((time.time() - self.born, int(target), int(cap)))
        made = []
        for sh in list(self._top_up_gives):
            self._top_up_gives.remove(sh)
            self.pods.setdefault(int(sh), Pod(int(sh)))
            self.ready_after.setdefault(int(sh), time.time() - self.born)
            made.append(self.pods[int(sh)])
        return made

    # -- what a test asks afterwards -------------------------------------- #
    def released_for(self, reason_fragment):
        return sorted(sh for sh, why in self.released
                      if reason_fragment in why)


def install(monkeypatch, fleet, *, home, pushes=None, finish_after=0.0,
            cancelled=False, no_sleep=False, push_fails=()):
    """Stub everything the run touches except the fleet itself.

    `home` is the loras directory; `finish_after` is how long the fake
    coordinator reports unfinished for, which is how a test bounds its own
    runtime. `no_sleep` removes the poll sleep while leaving the clock real --
    needed when a test watches a sub-second window against the 5s poll floor.
    """
    from codai.api import loras
    from codai.api import runpod_client as rpc
    from codai.models import manager as _mgr

    os.makedirs(str(home), exist_ok=True)
    monkeypatch.setattr(loras, "_loras_dir", lambda: str(home))
    monkeypatch.setattr(loras, "_SLICE_S", 0.2)
    monkeypatch.setattr(loras, "_update_job", lambda *a, **k: None)
    monkeypatch.setattr(loras, "_train_cancelled",
                        lambda job_id="": bool(cancelled))
    monkeypatch.setattr(loras, "_check_train_cancel", lambda *a, **k: None)
    monkeypatch.setattr(loras, "_resolve_text_base", lambda b: "stub-base")
    monkeypatch.setattr(loras, "train_lease", lambda jid: 0)
    monkeypatch.setattr(loras, "_forget_train_lease", lambda jid: None)
    if no_sleep:
        monkeypatch.setattr(loras, "time", types.SimpleNamespace(
            time=time.time, monotonic=time.monotonic, sleep=lambda _s: None))

    monkeypatch.setattr(_mgr, "get_active_runpod_config",
                        lambda: types.SimpleNamespace(api_key="x"))
    monkeypatch.setattr(rpc, "RunpodClient", lambda acct: object())
    monkeypatch.setattr(flt, "eligible_cards",
                        lambda *a, **k: [{"id": "A40", "hourly_usd": 0.59,
                                          "cloud_type": "SECURE"}])
    monkeypatch.setattr(flt, "describe_cards", lambda c: "A40")
    monkeypatch.setattr(flt, "clouds_for", lambda cfg: ["SECURE"])
    monkeypatch.setattr(flt, "hf_token_for_pods", lambda cfg: "")
    monkeypatch.setattr(flt, "orphans", lambda: [])
    monkeypatch.setattr(flt, "TrainerFleet", lambda *a, **k: fleet)

    def _push(pod, path):
        if pod.shard in push_fails:
            raise flt.FleetError(f"pod{pod.shard} refused the dataset")
        if pushes is not None:
            pushes.append((pod.shard, time.time() - fleet.born))
    monkeypatch.setattr(flt, "push_dataset", _push)
    monkeypatch.setattr(flt, "poll_status", lambda pod: {"state": "training"})
    monkeypatch.setattr(flt, "start_round", lambda pod, payload: None)
    monkeypatch.setattr(flt, "push_start_adapter", lambda pod, p: None)

    class _Coord:
        def __init__(self, *a, **k):
            self.tracker = types.SimpleNamespace(
                round=0, have=lambda rnd=None: [],
                expect=lambda r, c: None)
            self.born = time.time()

        def finished(self):
            return (time.time() - self.born) >= float(finish_after)

        def cohort(self, rnd, live=None, *, grow=False):
            # `grow=True` is how an exchange gains contributors now, so the
            # fake has to accept it or every run through here raises.
            return list(live or [])

        def round_payload(self, shard):
            return {}

        def close_round_if_ready(self):
            return None

        def reap_the_dead(self):
            return []

        def average_path(self, rnd):
            p = os.path.join(str(home), f"avg{rnd}.safetensors")
            if not os.path.isfile(p):
                with open(p, "wb") as fh:
                    fh.write(b"\0")
            return p
    monkeypatch.setattr(loras, "_TextTrainCoordinator", _Coord)


def cfg(**kw):
    base = dict(enabled=True, max_pods=3, pods_start=3, rounds=1,
                base_model="stub-base", poll_interval_s=5.0,
                pod_boot_timeout_s=30.0, min_vram_gb=48.0)
    base.update(kw)
    return flt.LoraTrainConfig(**base)


def req(name="adapter-fake", **kw):
    base = dict(name=name, base_model="qwen38-train", pods=None,
                min_pods=None, max_pods=None, rounds=1, test_run=False,
                distributed=True, dataset_id="", field_map=None,
                dry_run=False, max_pod_hourly_usd=None,
                max_fleet_hourly_usd=None)
    base.update(kw)
    return types.SimpleNamespace(**base)


def run(monkeypatch, *, fleet, home, job="job-fake", name="adapter-fake",
        request=None, config=None, dataset=None, prepare=None, **install_kw):
    """Drive the real `_text_train_distributed` against `fleet`."""
    from codai.api import loras
    from codai.api import text_lora_trainer as tlt

    os.makedirs(os.path.join(str(home), name), exist_ok=True)
    install(monkeypatch, fleet, home=home, **install_kw)
    if dataset is None:
        dataset = os.path.join(str(home), "ds.jsonl")
        with open(dataset, "w") as fh:
            fh.write('{"a":1}\n')
    if prepare is None:
        # The default stands in for the deferred pass: the run now receives a
        # callable and encodes while the pods boot, so a test that does not
        # care about that still has to give it one.
        def prepare():
            return ([object()],
                    types.SimpleNamespace(examples=1, skipped=[],
                                          trained_tokens=1, masked_tokens=0,
                                          describe=lambda: "1 example"),
                    {"examples": 1})
    return loras._text_train_distributed(
        request if request is not None else req(name),
        job, str(dataset), None, None,
        tlt.QLoraSettings(),
        config if config is not None else cfg(),
        prepare=prepare)
