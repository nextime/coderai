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

"""A distributed run that ENDED must stop looking like one that is running.

The one-run-per-adapter rule reads the job record's `status`, so a finished run
left at `training` makes its adapter permanently unavailable: every later
request for that name is folded into a job that is over, answers "carrying this
request into that run", and rents nothing at all. Measured in production --
eleven requests swallowed over 34 minutes while the fleet had been gone since
the first of them:

    job-72e7bba56216: 'digesta-extraction-a4v1' is already being trained by
    job-3f5f9b2585a5; carrying this request into that run

job-3f5f9b2585a5 had printed "finished 1 round(s)" half an hour earlier. The
local training path marked its record done; the distributed path returned
straight to its caller.

Executed, not read: the bug is a missing call, and source-text assertions
cannot see an absence they were not told to look for.
"""

import threading
import types

from codai.api import loras
from codai.api import text_lora_fleet as flt


def _stub_everything(monkeypatch, tmp_path, *, result):
    """Everything up to the distributed dispatch, stubbed."""
    import transformers

    monkeypatch.setattr(loras, "_jobs", {})
    monkeypatch.setattr(loras, "_jobs_lock", threading.Lock())
    recorded = []
    real_new = loras._new_job

    def _upd(jid, force=False, **kw):
        recorded.append(dict(kw))
        with loras._jobs_lock:
            loras._jobs.setdefault(jid, {"target": "text"}).update(kw)
    monkeypatch.setattr(loras, "_update_job", _upd)

    monkeypatch.setattr(flt, "config_for_model",
                        lambda m: flt.LoraTrainConfig(
                            enabled=True, pods_start=8, max_pods=16, rounds=16,
                            base_model="stub-base"))
    monkeypatch.setattr(loras, "_resolve_text_base", lambda b: "stub-base")

    class _Tok:
        @staticmethod
        def from_pretrained(*a, **k):
            return object()
    monkeypatch.setattr(transformers, "AutoTokenizer", _Tok)

    stats = types.SimpleNamespace(
        examples=7308, skipped=[], trained_tokens=2868079,
        masked_tokens=34542, describe=lambda: "7308 examples")
    monkeypatch.setattr(loras, "_text_dataset_pass",
                        lambda req, path, tok: ([object()] * 7308, stats))
    monkeypatch.setattr(loras, "_text_train_distributed",
                        lambda *a, **k: result)
    return recorded


def _req(name="adapter-f"):
    return types.SimpleNamespace(
        name=name, base_model="qwen38-train", pods=8, max_pods=16, min_pods=1,
        rounds=16, test_run=False, distributed=True, dataset_id="",
        field_map=None, dry_run=False, rank=16, alpha=32, dropout=0.05,
        learning_rate=1e-4, max_len=8192, accumulation_steps=8,
        enable_thinking=False, seed=42, test_examples=8,
        max_pod_hourly_usd=None, max_fleet_hourly_usd=None)


def test_a_finished_distributed_run_goes_terminal(monkeypatch, tmp_path):
    recorded = _stub_everything(
        monkeypatch, tmp_path,
        result={"rounds": 16, "shards": 8, "adapter": "/x/a.safetensors",
                "path": "/x"})

    with loras._jobs_lock:
        loras._jobs["job-dist"] = {"target": "text", "name": "adapter-f",
                                   "status": "training"}

    got = loras._text_train_blocking(_req(), "job-dist", str(tmp_path / "d"))

    assert got["rounds"] == 16
    assert any(r.get("status") == "done" for r in recorded), recorded
    final = [r for r in recorded if r.get("status")][-1]
    assert final["status"] == "done"
    assert final["active"] is False


def test_the_adapter_can_be_trained_again_afterwards(monkeypatch, tmp_path):
    """The consequence, which is the part that cost the time.

    The record left at `training` is not a cosmetic defect: it is the thing
    `_active_text_job_for` consults, so the next request joins a dead run.
    """
    _stub_everything(monkeypatch, tmp_path,
                     result={"rounds": 16, "shards": 8, "path": "/x"})
    # The record the ROUTE creates before the thread starts: `target`, `name`
    # and `status` are what `_active_text_job_for` matches on, so a stub that
    # omits them makes this test pass against the bug.
    with loras._jobs_lock:
        loras._jobs["job-first"] = {"target": "text", "name": "adapter-g",
                                    "status": "training"}

    loras._text_train_blocking(_req("adapter-g"), "job-first",
                               str(tmp_path / "d"))
    assert loras._active_text_job_for("adapter-g") is None

    # ...and a second request therefore starts its own run rather than being
    # folded into the finished one.
    got = loras._text_train_blocking(_req("adapter-g"), "job-second",
                                     str(tmp_path / "d"))
    assert "joined" not in got
    assert got["rounds"] == 16


def test_a_run_still_going_is_still_joined(monkeypatch, tmp_path):
    """The guard this must not break: a LIVE run still refuses a second fleet."""
    _stub_everything(monkeypatch, tmp_path,
                     result={"rounds": 16, "shards": 8, "path": "/x"})
    with loras._jobs_lock:
        loras._jobs["job-live"] = {"target": "text", "name": "adapter-h",
                                   "status": "training"}
    got = loras._text_train_blocking(_req("adapter-h"), "job-new",
                                     str(tmp_path / "d"))
    assert got["joined"] == "job-live"
    loras._forget_train_lease("job-live")
