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

"""Training starts on the first pod that is ready, not on the last.

A fleet of sixteen used to wait for sixteen boots before the first byte of the
dataset was sent -- and a pod that never published a port spent its ENTIRE
timeout first, so one dead shard set the start time for fifteen working ones
that were up and billing. Measured on an eight-pod run: pods reachable at
18:13:36, 18:14:19 and 18:15:50, round 0 beginning at 18:17:50, with a 600s
per-pod budget that one silent shard would have spent in full.

Run, not read: the barrier was a `with ThreadPoolExecutor(...)` block whose
exit joined every future, which is invisible unless the control flow is
actually executed. So this drives `_text_train_distributed` against a fake
fleet whose shards become addressable at different times.
"""

import threading
import time
import types

import pytest

from codai.api import loras
from codai.api import text_lora_fleet as flt


class _Pod:
    def __init__(self, shard):
        self.shard = shard
        self.url = ""
        self.api_key = "k"
        self.pod_id = f"pod{shard}"

    def cost_so_far(self):
        return 0.0


class _FakeFleet:
    """A fleet whose shards come up when the test says they do.

    `ready_after` is seconds from construction; a shard not named there never
    becomes addressable, which is the dead-shard case that used to hold up
    everyone else.
    """

    def __init__(self, ready_after):
        self.ready_after = dict(ready_after)
        self.born = time.time()
        self.pods = {}
        self.released = []

    # -- what the run calls ------------------------------------------------ #
    def adopt_existing(self):
        return []

    def provision(self, shards):
        for sh in shards:
            self.pods.setdefault(sh, _Pod(sh))
        return list(self.pods.values())

    def address(self, shard, timeout=0.0):
        due = self.ready_after.get(shard)
        if due is None:
            time.sleep(min(timeout, 0.05))
            raise flt.FleetError(f"did not expose port 8000 within {timeout}")
        deadline = time.time() + max(0.0, timeout)
        while time.time() < deadline:
            if time.time() - self.born >= due:
                self.pods[shard].url = f"http://pod{shard}:8000"
                return self.pods[shard].url
            time.sleep(0.01)
        raise flt.FleetError(f"did not expose port 8000 within {timeout}")

    def note_checkin(self, shard, rounds_done=None):
        pass

    def release(self, shard, reason=""):
        self.released.append((shard, reason))
        self.pods.pop(shard, None)

    def release_all(self, reason=""):
        n = len(self.pods)
        self.pods.clear()
        return n

    def top_up(self, target, cap=0):
        return []


def _install(monkeypatch, fleet, *, pushes, stop_after=0.0, home=None):
    """Stub everything around the fleet; record every dataset push."""
    if home is not None:
        import os
        # The run copies the final average into `<loras_dir>/<name>/`; the
        # real coordinator makes that directory and this fake does not.
        os.makedirs(os.path.join(str(home), "adapter-boot"), exist_ok=True)
        monkeypatch.setattr(loras, "_loras_dir", lambda: str(home))
    monkeypatch.setattr(loras, "_SLICE_S", 0.2)
    monkeypatch.setattr(loras, "_update_job", lambda *a, **k: None)
    monkeypatch.setattr(loras, "_train_cancelled", lambda job_id="": False)
    monkeypatch.setattr(loras, "_check_train_cancel", lambda *a, **k: None)
    monkeypatch.setattr(loras, "_resolve_text_base", lambda b: "stub-base")
    monkeypatch.setattr(loras, "train_lease", lambda jid: 0)
    monkeypatch.setattr(loras, "_forget_train_lease", lambda jid: None)

    from codai.models import manager as _mgr
    monkeypatch.setattr(_mgr, "get_active_runpod_config",
                        lambda: types.SimpleNamespace(api_key="x"))
    from codai.api import runpod_client as rpc
    monkeypatch.setattr(rpc, "RunpodClient", lambda acct: object())

    monkeypatch.setattr(flt, "eligible_cards",
                        lambda *a, **k: [{"id": "A40", "hourly_usd": 0.59,
                                          "cloud_type": "SECURE",
                                          "vram_gb": 48}])
    monkeypatch.setattr(flt, "describe_cards", lambda c: "A40")
    monkeypatch.setattr(flt, "clouds_for", lambda cfg: ["SECURE"])
    monkeypatch.setattr(flt, "hf_token_for_pods", lambda cfg: "")
    monkeypatch.setattr(flt, "orphans", lambda: [])
    monkeypatch.setattr(flt, "TrainerFleet", lambda *a, **k: fleet)

    def _push(pod, path):
        pushes.append((pod.shard, time.time() - fleet.born))
    monkeypatch.setattr(flt, "push_dataset", _push)
    monkeypatch.setattr(flt, "poll_status", lambda pod: {"state": "training"})
    monkeypatch.setattr(flt, "start_round", lambda pod, payload: None)
    monkeypatch.setattr(flt, "push_start_adapter", lambda pod, p: None)

    class _Coord:
        """Finishes on a wall clock, so the test bounds its own runtime."""

        def __init__(self, *a, **k):
            self.plan = types.SimpleNamespace(
                block_size=lambda *a, **k: 1)
            self.encoded = []
            self.total_rounds = 1
            self.planned_pods = 1
            self.tracker = types.SimpleNamespace(
                round=0, have=lambda rnd=None: [], expect=lambda r, c: None)
            self.born = time.time()

        def finished(self):
            return (time.time() - self.born) >= stop_after

        def cohort(self, rnd, live=None, *, grow=False):
            return list(live or [])

        def round_payload(self, shard):
            return {}

        def close_round_if_ready(self):
            return None

        def reap_the_dead(self):
            return []

        def coverage(self):
            # The run reports progress as blocks covered now, so a fake that
            # lacks this breaks every run through the loop.
            return (0, 1)

        @property
        def epochs(self):
            return 1.0

        #: The run derives `adapter_config.json` from the average, so the
        #: fake supplies the base and the settings it reads for that.
        base = "stub-base"

        @property
        def settings(self):
            return types.SimpleNamespace(alpha=32, dropout=0.0, rank=16)

        def blocks_held(self, shard, rnd):
            return []

        def average_path(self, rnd):
            # A real file: the run copies the last round's average into the
            # adapter and fails if there is none, and that path is not what
            # these tests are about. Adapter-SHAPED, because the run now
            # reads its keys to write the config beside it.
            import os
            from _fleetfake import _write_tiny_adapter
            d = loras._loras_dir()
            os.makedirs(d, exist_ok=True)
            p = os.path.join(d, f"avg{rnd}.safetensors")
            if not os.path.isfile(p):
                _write_tiny_adapter(p)
            return p
    monkeypatch.setattr(loras, "_TextTrainCoordinator", _Coord)


def _cfg(**kw):
    base = dict(enabled=True, pods_start=3, max_pods=3, rounds=1,
                base_model="stub-base", poll_interval_s=5.0,
                pod_boot_timeout_s=30.0, min_vram_gb=48.0)
    base.update(kw)
    return flt.LoraTrainConfig(**base)


def _req(**kw):
    base = dict(name="adapter-boot", base_model="qwen38-train", pods=3,
                min_pods=3, max_pods=3, rounds=1, test_run=False,
                distributed=True, dataset_id="", field_map=None,
                dry_run=False, max_pod_hourly_usd=None,
                max_fleet_hourly_usd=None)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_a_pod_gets_the_dataset_without_waiting_for_the_others(monkeypatch,
                                                              tmp_path):
    """Shard 0 is up at once, shard 1 after a second, shard 2 never.

    The old code pushed nothing until all three had resolved -- which, with a
    shard that never comes up, meant not until the boot timeout expired.
    """
    ds = tmp_path / "ds.jsonl"
    ds.write_text('{"a":1}\n')
    fleet = _FakeFleet({0: 0.0, 1: 1.0})
    pushes = []
    _install(monkeypatch, fleet, pushes=pushes, stop_after=2.5,
             home=tmp_path / 'loras')

    loras._text_train_distributed(
        _req(), "job-boot", str(ds), [object()], 
        types.SimpleNamespace(examples=1, skipped=[]),
        __import__("codai.api.text_lora_trainer", fromlist=["x"]
                   ).QLoraSettings(), _cfg())

    got = {sh: at for sh, at in pushes}
    assert 0 in got, pushes
    # Shard 0 was served BEFORE shard 1 could possibly have been ready...
    assert got[0] < 1.0, pushes
    # ...and shard 2, which never comes up, did not delay either of them.
    assert 2 not in got
    assert 1 in got and got[1] < 30.0, pushes


def test_the_run_begins_before_the_slowest_pod(monkeypatch, tmp_path, capsys):
    ds = tmp_path / "ds.jsonl"
    ds.write_text('{"a":1}\n')
    fleet = _FakeFleet({0: 0.0, 1: 2.0, 2: 2.0})
    pushes = []
    _install(monkeypatch, fleet, pushes=pushes, stop_after=1.0,
             home=tmp_path / 'loras')

    loras._text_train_distributed(
        _req(), "job-boot2", str(ds), [object()],
        types.SimpleNamespace(examples=1, skipped=[]),
        __import__("codai.api.text_lora_trainer", fromlist=["x"]
                   ).QLoraSettings(), _cfg())

    out = capsys.readouterr().out
    # It said so, and it stopped (stop_after=1.0) a second before shards 1 and
    # 2 were due: the run was under way with one pod.
    assert "is up with the dataset" in out
    assert "still booting" in out


def test_a_shard_that_never_comes_up_is_released(monkeypatch, tmp_path):
    """It is billing and it will never be useful, so it goes on its own
    timeout rather than at the end of the fleet's."""
    ds = tmp_path / "ds.jsonl"
    ds.write_text('{"a":1}\n')
    fleet = _FakeFleet({0: 0.0})
    pushes = []
    _install(monkeypatch, fleet, pushes=pushes, stop_after=2.0,
             home=tmp_path / 'loras')

    loras._text_train_distributed(
        _req(), "job-boot3", str(ds), [object()],
        types.SimpleNamespace(examples=1, skipped=[]),
        __import__("codai.api.text_lora_trainer", fromlist=["x"]
                   ).QLoraSettings(), _cfg(pod_boot_timeout_s=1.0))

    gone = {sh for sh, _ in fleet.released}
    assert {1, 2} <= gone, fleet.released


def test_no_pod_at_all_is_still_a_failure(monkeypatch, tmp_path):
    ds = tmp_path / "ds.jsonl"
    ds.write_text('{"a":1}\n')
    fleet = _FakeFleet({})
    _install(monkeypatch, fleet, pushes=[], stop_after=0.0,
             home=tmp_path / 'loras')

    with pytest.raises(RuntimeError) as exc:
        loras._text_train_distributed(
            _req(), "job-boot4", str(ds), [object()],
            types.SimpleNamespace(examples=1, skipped=[]),
            __import__("codai.api.text_lora_trainer", fromlist=["x"]
                       ).QLoraSettings(), _cfg(pod_boot_timeout_s=1.0))
    assert "no pod became reachable" in str(exc.value)
