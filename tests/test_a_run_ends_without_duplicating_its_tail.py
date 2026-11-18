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

"""What a 15-hour run cost, in four defects and one unloadable deliverable.

The run finished 364 of 366 blocks and had to be stopped by hand. Everything
here was measured on it rather than reasoned about:

* the handout asked only whether THIS round had issued a block, never whether
  it was already trained. The counter wraps, so once it passed the stragglers
  the tail degenerated: rounds 30, 31 and 32 covered NOTHING -- two hours and
  ~$28 of A40s for one block -- and ~30% of the dataset was trained twice;
* `epochs` could not be satisfied above 1.0, because `finished()` compared the
  SIZE OF A SET of block ids against `total_blocks * epochs`;
* five pod slots cycled rent-reap-rent for the whole run: 87 releases and ~$27
  of machines that never once answered, re-let within the minute each time;
* the adapter the run produced could not be loaded. `target_modules` recorded
  256 fully-qualified module PATHS where a loader matches leaf names, and the
  tensors were float32 at twice the size of every adapter the install serves.

The last one is the one that matters most, because nothing detected it: the
directory looked like an adapter, PEFT parsed the config, and the failure
appeared only when a pod tried to apply it and answered 500.
"""

from __future__ import annotations

import json
import os
import types

import pytest

import codai.api.loras as loras
import codai.api.text_lora_fleet as flt
import codai.api.text_lora_rounds as tlr
import codai.api.text_lora_trainer as tlt


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def coord(tmp, *, n_examples=100, rounds=2, pods=5, epochs=1.0):
    plan = tlr.RoundPlan(n_shards=pods, seed=42)
    t = tlr.RoundTracker(plan, min_fraction=0.75, deadline_s=600.0)
    c = loras._TextTrainCoordinator(
        "job-tail", plan=plan, tracker=t,
        fleet=types.SimpleNamespace(pods={}),
        encoded=[0] * n_examples, out_dir=str(tmp),
        settings=tlt.QLoraSettings(), dataset_id="sha256:abc",
        base="qwen", field_map=None, total_rounds=rounds,
        planned_pods=pods, epochs=epochs, session="sess")
    return c, t


def train(c, shard, rnd, blocks):
    """Record ``blocks`` as trained in full by ``shard`` in round ``rnd``."""
    per = c.plan.block_size(len(c.encoded), c.total_rounds, c.planned_pods)
    with c._lock:
        c._blocks.setdefault(int(rnd), {})[int(shard)] = list(blocks)
    c.block_done(shard, rnd, examples=per * len(blocks),
                 asked=per * len(blocks))


def lora_file(path, *, keys, dtype):
    import torch
    from safetensors.torch import save_file
    sd = {}
    for k in keys:
        sd[k + ".lora_A.weight"] = torch.ones((16, 32), dtype=dtype)
        sd[k + ".lora_B.weight"] = torch.ones((32, 16), dtype=dtype)
    save_file(sd, path, metadata={"format": "pt"})
    return path


# --------------------------------------------------------------------------- #
# the tail of a run trains what is missing
# --------------------------------------------------------------------------- #

def test_a_block_already_trained_is_not_handed_out_again(tmp_path):
    """THE ONE THAT COST THE MONEY.

    With every block but one covered, the next handout must be that block.
    Before the fix the walk took whatever the counter pointed at and asked
    only whether this round had issued it, so a round could hand out ten
    blocks that were trained hours earlier and cover nothing.
    """
    c, _ = coord(tmp_path, n_examples=100, rounds=2, pods=5)
    total = c.total_blocks()
    missing = 7
    for b in range(total):
        if b != missing:
            train(c, 0, 0, [b])
    # The counter is well past `missing`: this is the tail, after the walk
    # has wrapped. The next block handed to anybody must be the one block
    # nobody has trained.
    assert c.blocks_for(1, 5) == [missing]


def test_the_tail_covers_the_stragglers_within_one_round(tmp_path):
    """Three missing blocks and three pods: the round must finish the job.

    Measured on the real run, three consecutive rounds of twelve pods each
    covered nothing at all while three blocks were outstanding.
    """
    c, _ = coord(tmp_path, n_examples=100, rounds=2, pods=5)
    total = c.total_blocks()
    missing = {3, 6, 8}
    assert max(missing) < total, "the test must name blocks that exist"
    for b in range(total):
        if b not in missing:
            train(c, 0, 0, [b])
    handed = set()
    for shard in (1, 2, 3):
        got = c.blocks_for(shard, 9)
        assert got, f"shard {shard} was given no work with blocks outstanding"
        handed.update(got)
    assert handed == missing


def test_a_fleet_ahead_of_the_queue_still_gets_work(tmp_path):
    """The preference must not idle a pod once everything is covered.

    A round that has nothing UNTRAINED left should still hand out a block
    rather than return None -- otherwise the fix trades duplicate work for an
    idle fleet, which costs exactly the same per hour.
    """
    c, _ = coord(tmp_path, n_examples=100, rounds=2, pods=5)
    for b in range(c.total_blocks()):
        train(c, 0, 0, [b])
    assert c.blocks_for(1, 4) is not None


def test_no_two_pods_in_one_round_get_the_same_block(tmp_path):
    """The guard the preference must not break.

    Two pods training the same examples in one average double-counts them,
    which is the duplication this whole design replaced.
    """
    c, _ = coord(tmp_path, n_examples=100, rounds=2, pods=5)
    seen = []
    for shard in range(5):
        got = c.blocks_for(shard, 0) or []
        seen.extend(got)
    assert len(seen) == len(set(seen))


# --------------------------------------------------------------------------- #
# epochs above one can be satisfied
# --------------------------------------------------------------------------- #

def test_a_single_epoch_still_ends_exactly_where_it_did(tmp_path):
    """The regression guard: one epoch is one pass, counted as before."""
    c, _ = coord(tmp_path, n_examples=100, rounds=2, pods=5, epochs=1.0)
    total = c.total_blocks()
    for b in range(total):
        assert not c.finished()
        train(c, 0, 0, [b])
    assert c.finished()
    assert c.coverage() == (total, total)


def test_two_epochs_can_finish_at_all(tmp_path):
    """`epochs=2` was unreachable: a SET of ids against twice its own size.

    One pass must leave the run unfinished, and the second must end it. Before
    the fix the first assertion passed and the second could never hold, so the
    run wrapped the queue and retrained until it lost its fleet.
    """
    c, _ = coord(tmp_path, n_examples=100, rounds=2, pods=5, epochs=2.0)
    total = c.total_blocks()
    for b in range(total):
        train(c, 0, 0, [b])
    assert not c.finished(), "one pass of two cannot be a finished run"
    assert c.coverage() == (total, 2 * total)
    for b in range(total):
        train(c, 1, 1, [b])
    assert c.finished()


def test_a_second_epoch_trains_every_block_before_any_block_thrice(tmp_path):
    """The preference has to hold per PASS, not just once.

    Otherwise a two-epoch run gives some blocks three passes and others one,
    which is the uneven weighting the single-epoch run already suffered.
    """
    c, _ = coord(tmp_path, n_examples=100, rounds=2, pods=5, epochs=2.0)
    total = c.total_blocks()
    for b in range(total):
        train(c, 0, 0, [b])
    handed = []
    for i in range(total):
        got = c.blocks_for(i, 7) or []
        handed.extend(got)
        for b in got:
            train(c, i, 7, [b])
    with c._lock:
        counts = dict(c._cover_counts)
    assert max(counts.values()) <= 2, counts
    assert min(counts.values()) == 2, counts


def test_progress_and_termination_cannot_disagree(tmp_path):
    """`coverage()` must publish the number `finished()` tests.

    A published figure that is not the terminating condition is how a 5.3%
    run once reported 55%.
    """
    c, _ = coord(tmp_path, n_examples=100, rounds=2, pods=5, epochs=2.0)
    for b in range(c.total_blocks()):
        train(c, 0, 0, [b])
        done, want = c.coverage()
        assert c.finished() == (done >= want)


# --------------------------------------------------------------------------- #
# capacity that never answers is not re-rented for ever
# --------------------------------------------------------------------------- #

class _Client:
    def __init__(self):
        self.created = 0
        self.terminated = []

    def terminate_pod(self, pod_id):
        self.terminated.append(pod_id)


def fleet_with(tmp_path, monkeypatch, *, pods):
    """A fleet holding ``pods`` already-rented pods, renting nothing new."""
    monkeypatch.setattr(flt, "_read_state", lambda job: [])
    monkeypatch.setattr(flt, "_write_state", lambda *a, **k: None,
                        raising=False)
    spec = flt.TrainerPodSpec(image="trainer:test")
    f = flt.TrainerFleet(_Client(), "job-doa", spec)
    f._flush = lambda: None
    f._say = lambda msg: f.said.append(msg)
    f.said = []
    for shard, url in pods.items():
        f.pods[shard] = flt.TrainerPod(pod_id=f"pod-{shard}", shard=shard,
                                       started_at=0.0, hourly_usd=0.59,
                                       url=url)
    return f


def test_a_pod_that_never_answered_is_counted_apart(tmp_path, monkeypatch):
    """A pod with no url never checked in; one with a url worked and died."""
    f = fleet_with(tmp_path, monkeypatch,
                   pods={0: "", 1: "http://pod-1"})
    f.release(0, "stopped checking in")
    assert (f._doa, f._doa_streak) == (1, 1)
    f.release(1, "stopped checking in")
    assert f._doa == 1, "a pod that answered is not dead on arrival"
    assert f._doa_streak == 0, "one working pod resets the streak"


def test_the_fleet_stops_growing_into_dead_capacity(tmp_path, monkeypatch):
    """After the streak the top-up holds its size instead of re-renting.

    On the measured run this was 87 releases and ~$27 of A40s that trained
    nothing, each slot re-let within the minute.
    """
    f = fleet_with(tmp_path, monkeypatch,
                   pods={0: "http://pod-0", 1: "", 2: "", 3: ""})
    for shard in (1, 2, 3):
        f.release(shard, "stopped checking in")
    assert f._doa_streak >= flt.DOA_STREAK_CAP
    calls = []
    f.provision = lambda shards: calls.append(list(shards)) or []
    assert f.top_up(24) == []
    assert calls == [], "the fleet re-rented capacity that never answers"
    assert any("holding at" in m for m in f.said), f.said


def test_it_says_so_once_not_every_round(tmp_path, monkeypatch):
    f = fleet_with(tmp_path, monkeypatch, pods={0: "http://pod-0", 1: ""})
    f._doa_streak = flt.DOA_STREAK_CAP
    f.provision = lambda shards: []
    for _ in range(4):
        f.top_up(24)
    assert sum("holding at" in m for m in f.said) == 1, f.said


def test_the_fleet_does_not_shrink_and_recovers(tmp_path, monkeypatch):
    """Holding is not releasing, and one answering pod lets it grow again."""
    f = fleet_with(tmp_path, monkeypatch, pods={0: "http://pod-0", 1: ""})
    f._doa_streak = flt.DOA_STREAK_CAP
    f.provision = lambda shards: []
    f.top_up(24)
    assert set(f.pods) == {0, 1}, "holding must not drop working pods"
    f.release(0, "worked then died")        # had a url -> resets the streak
    calls = []
    f.provision = lambda shards: calls.append(list(shards)) or []
    f.top_up(24)
    assert calls, "a pod that answered must let the fleet grow again"


# --------------------------------------------------------------------------- #
# the adapter the run produces is one a pod can load
# --------------------------------------------------------------------------- #

PRE = "base_model.model.model.language_model.layers."


def test_target_modules_are_leaf_names_not_paths(tmp_path):
    """THE ONE THAT MADE THE DELIVERABLE UNUSABLE.

    A loader resolves `target_modules` against the SUFFIX of the base's module
    names. The first distributed adapter recorded 256 fully-qualified paths,
    matched nothing, and answered 500 "Model not available" -- while the two
    adapters the same install already serves list seven leaf names and load.
    """
    p = lora_file(str(tmp_path / "a.safetensors"),
                  keys=[PRE + "0.mlp.down_proj", PRE + "0.self_attn.q_proj",
                        PRE + "1.mlp.down_proj"],
                  dtype=__import__("torch").bfloat16)
    cfg = tlr.adapter_config_for(p, base="Qwen/X", alpha=32, dropout=0.05)
    assert sorted(cfg["target_modules"]) == ["down_proj", "q_proj"]
    assert not any("." in m for m in cfg["target_modules"]), \
        "a module path is not a target_modules entry"
    assert cfg["r"] == 16 and cfg["lora_alpha"] == 32


def test_the_rank_is_still_read_off_the_tensors(tmp_path):
    """The leaf-name change must not disturb what was already right."""
    import torch
    p = lora_file(str(tmp_path / "a.safetensors"),
                  keys=[PRE + "0.mlp.up_proj"], dtype=torch.bfloat16)
    cfg = tlr.adapter_config_for(p, base="Qwen/X", alpha=None, dropout=None)
    assert cfg["r"] == 16
    assert cfg["lora_alpha"] == 16, "no alpha given falls back to r, not zero"


def test_a_file_with_no_lora_tensors_is_still_refused(tmp_path):
    import torch
    from safetensors.torch import save_file
    p = str(tmp_path / "nope.safetensors")
    save_file({"model.layers.0.weight": torch.ones((2, 2))}, p,
              metadata={"format": "pt"})
    with pytest.raises(ValueError):
        tlr.adapter_config_for(p, base="Qwen/X")


def test_the_served_adapter_is_cast_down_from_the_pods_float32(tmp_path):
    """The pods hand back float32; a served adapter is bf16 at half the size."""
    import torch
    src = lora_file(str(tmp_path / "avg.safetensors"),
                    keys=[PRE + "0.mlp.down_proj", PRE + "1.mlp.down_proj"],
                    dtype=torch.float32)
    dest = str(tmp_path / "adapter_model.safetensors")
    info = tlr.write_serving_adapter(src, dest)
    assert info["dtype"] == "bfloat16"
    assert info["cast"] == info["tensors"] == 4
    assert info["bytes"] < os.path.getsize(src)
    from safetensors.torch import load_file
    got = load_file(dest)
    assert {v.dtype for v in got.values()} == {torch.bfloat16}


def test_the_cast_preserves_the_weights_that_were_trained(tmp_path):
    """bf16 is lossy; it must not be WRONG.

    Measured over all 512 tensors of the real run the worst relative change
    in tensor norm was 4.7e-05, so a loose bound here still catches a cast
    that mangles or transposes anything.
    """
    import torch
    from safetensors.torch import load_file, save_file
    torch.manual_seed(0)
    sd = {PRE + "0.mlp.down_proj.lora_A.weight": torch.randn(16, 64),
          PRE + "0.mlp.down_proj.lora_B.weight": torch.randn(64, 16)}
    src = str(tmp_path / "avg.safetensors")
    save_file(sd, src, metadata={"format": "pt"})
    dest = str(tmp_path / "out.safetensors")
    tlr.write_serving_adapter(src, dest)
    got = load_file(dest)
    assert set(got) == set(sd), "the cast must not rename or drop a tensor"
    for k, v in sd.items():
        assert got[k].shape == v.shape
        a, b = v.norm().item(), got[k].float().norm().item()
        assert abs(a - b) / a < 1e-2, k


def test_the_key_layout_is_reported_because_it_cannot_be_checked(tmp_path):
    """The one packaging fault the orchestrator cannot detect for itself.

    An adapter whose keys say `model.layers.0` against a base that calls it
    `model.language_model.layers.0` is a valid file that adapts NOTHING, and
    knowing the base's module tree means loading the base. So the prefix is
    recorded, and the next mismatch is one log line rather than a day.
    """
    import torch
    src = lora_file(str(tmp_path / "avg.safetensors"),
                    keys=[PRE + "0.mlp.down_proj", PRE + "1.mlp.up_proj"],
                    dtype=torch.bfloat16)
    info = tlr.write_serving_adapter(src, str(tmp_path / "out.safetensors"))
    assert info["key_prefix"].startswith("base_model.model.model.")
    assert "language_model" in info["key_prefix"]


def test_a_config_written_beside_the_adapter_names_leaves(tmp_path):
    """End to end: what the completion path leaves on disk."""
    import torch
    src = lora_file(str(tmp_path / "avg.safetensors"),
                    keys=[PRE + "0.self_attn.k_proj"], dtype=torch.float32)
    dest = str(tmp_path / "adapter_model.safetensors")
    tlr.write_serving_adapter(src, dest)
    tlr.write_adapter_config(str(tmp_path), dest, base="Qwen/X",
                             alpha=32, dropout=0.05)
    cfg = json.load(open(os.path.join(str(tmp_path), "adapter_config.json")))
    assert cfg["target_modules"] == ["k_proj"]
    assert cfg["peft_type"] == "LORA" and cfg["task_type"] == "CAUSAL_LM"
