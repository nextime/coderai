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

"""A distributed run that stopped can be picked up where it left off.

The averages were always written to disk and read by nothing, so a cancelled
four-hour job was four hours bought twice. Two things have to hold for a
resume to be worth anything, and they fail in opposite directions:

* adopting the checkpoint must actually make the pods start from the carried
  adapter -- the dispatch pushes `average_path(round - 1)` and skips the push
  entirely at round 0, so a continuation numbered from zero reports itself as
  a resume and trains from the bare base;
* adopting it must be REFUSED whenever a recorded block id would mean
  something else, because a block id is `shuffle(seed)[b*per:(b+1)*per]` and
  nothing downstream can notice that it pointed at the wrong examples.
"""

from __future__ import annotations

import json
import os
import types

import pytest

import codai.api.loras as loras
import codai.api.text_lora_resume as tlres
import codai.api.text_lora_rounds as tlr
import codai.api.text_lora_trainer as tlt


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def state(**kw) -> tlres.ResumeState:
    """A plausible checkpoint, with every identity field populated."""
    base = dict(
        dataset_id="sha256:abc", base="qwen", target="text",
        session="sess", rank=16, alpha=32, dropout=0.05, max_len=8192,
        seed=42, n_examples=100, block_size=10,
        job_id="job-1", name="run-a", rounds=2, planned_pods=5, epochs=1.0,
        round=3, covered=[0, 1, 2], next_block=3, recycle=[],
        trained=30, asked=30, subs=3, short=0, average="round2.safetensors")
    base.update(kw)
    return tlres.ResumeState(**base)


def coord(tmp, *, n_examples=100, rounds=2, pods=5, session="sess",
          dataset_id="sha256:abc", base="qwen", settings=None):
    """A coordinator over a real directory, as the run builds one."""
    plan = tlr.RoundPlan(n_shards=pods, seed=42)
    t = tlr.RoundTracker(plan, min_fraction=0.75, deadline_s=600.0)
    c = loras._TextTrainCoordinator(
        "job-2", plan=plan, tracker=t, fleet=types.SimpleNamespace(pods={}),
        encoded=[0] * n_examples, out_dir=str(tmp),
        settings=settings or tlt.QLoraSettings(), dataset_id=dataset_id,
        base=base, field_map=None, total_rounds=rounds, planned_pods=pods,
        epochs=1.0, session=session)
    return c, t


# --------------------------------------------------------------------------- #
# the checkpoint file
# --------------------------------------------------------------------------- #

def test_a_checkpoint_round_trips(tmp_path):
    want = state()
    tlres.save(str(tmp_path), want)
    got = tlres.load(str(tmp_path))
    assert got is not None
    assert got.covered == [0, 1, 2]
    assert got.round == 3
    assert got.block_size == 10
    assert got.average == "round2.safetensors"
    assert got.trained == 30 and got.asked == 30
    assert got.written_at > 0


def test_no_checkpoint_is_not_an_error(tmp_path):
    """An older run's directory has no resume.json, and that is a None."""
    assert tlres.load(str(tmp_path)) is None


def test_a_corrupt_checkpoint_is_refused_not_ignored(tmp_path):
    """The failure this module exists to end must not come back as silence.

    A truncated checkpoint read as "nothing to resume" is a run that reports
    a continuation and starts from block 0 -- precisely the behaviour being
    fixed, reintroduced through the error path.
    """
    with open(tlres.path_for(str(tmp_path)), "w") as fh:
        fh.write('{"covered": [0, 1')
    with pytest.raises(tlres.ResumeError) as exc:
        tlres.load(str(tmp_path))
    assert "cannot be read" in str(exc.value)


def test_a_checkpoint_from_a_newer_build_is_refused(tmp_path):
    tlres.save(str(tmp_path), state())
    p = tlres.path_for(str(tmp_path))
    raw = json.load(open(p))
    raw["version"] = tlres.RESUME_VERSION + 5
    with open(p, "w") as fh:
        json.dump(raw, fh)
    with pytest.raises(tlres.ResumeError) as exc:
        tlres.load(str(tmp_path))
    assert "newer build" in str(exc.value)


def test_an_unknown_field_does_not_break_the_load(tmp_path):
    tlres.save(str(tmp_path), state())
    p = tlres.path_for(str(tmp_path))
    raw = json.load(open(p))
    raw["something_a_later_build_added"] = {"x": 1}
    with open(p, "w") as fh:
        json.dump(raw, fh)
    assert tlres.load(str(tmp_path)).covered == [0, 1, 2]


def test_the_save_leaves_no_half_written_file_behind(tmp_path):
    for _ in range(3):
        tlres.save(str(tmp_path), state())
    names = sorted(os.listdir(str(tmp_path)))
    assert names == [tlres.RESUME_FILE], names


# --------------------------------------------------------------------------- #
# what may be resumed
# --------------------------------------------------------------------------- #

def test_an_identical_run_is_resumable():
    st = state()
    want = {k: getattr(st, k) for k, _ in tlres._IDENTITY}
    assert tlres.mismatches(st, want) == []
    tlres.check(st, want)           # must not raise


def test_every_changed_field_is_named_not_just_the_first():
    """A caller fixing a refused resume wants the whole list.

    Reporting only the first sends them round the loop once per wrong field,
    at a pod rental each if the check had been late.
    """
    st = state()
    bad = tlres.mismatches(st, {"dataset_id": "sha256:other", "seed": 7,
                                "rank": 32})
    assert len(bad) == 3
    joined = " ".join(bad)
    assert "the dataset" in joined
    assert "the shuffle seed" in joined
    assert "the LoRA rank" in joined


def test_a_different_dataset_is_refused():
    with pytest.raises(tlres.ResumeError) as exc:
        tlres.check(state(), {"dataset_id": "sha256:other"}, where="/d")
    msg = str(exc.value)
    assert "/d" in msg
    assert "the dataset changed" in msg


def test_a_changed_block_size_explains_itself():
    """The mismatch a caller causes without knowing they did.

    `rounds` and `pods` are the two fields a client is most likely to tune
    between attempts, and both move the block size, which silently renumbers
    every recorded block. So this refusal has to say WHY rather than name a
    field nobody set directly.
    """
    with pytest.raises(tlres.ResumeError) as exc:
        tlres.check(state(), {"block_size": 20})
    msg = str(exc.value)
    assert "the block size changed (10 -> 20)" in msg
    assert "rounds" in msg and "pods" in msg


def test_a_field_the_caller_cannot_know_is_not_compared():
    """The early check runs before the dataset pass, so it omits the count."""
    assert tlres.mismatches(state(), {"dataset_id": "sha256:abc"}) == []


def test_a_resume_across_sessions_is_refused():
    """An ownership check: one tenant's coverage must not graft onto another.

    Averaging a continuation on top of weights from a different session gives
    an adapter that loads, serves, and is not what either run trained.
    """
    with pytest.raises(tlres.ResumeError):
        tlres.check(state(), {"session": "somebody-else"})


def test_a_run_that_never_closed_a_round_is_not_resumable():
    assert tlres.nothing_to_resume(None) is not None
    empty = state(covered=[], average="", round=0)
    why = tlres.nothing_to_resume(empty)
    assert why and "stopped before its first round closed" in why


def test_coverage_alone_is_still_worth_resuming():
    """No adapter, but the blocks those GPU hours bought are not retrained."""
    assert tlres.nothing_to_resume(state(covered=[0, 1], average="")) is None


def test_an_average_that_is_gone_resolves_to_nothing(tmp_path):
    """Named but absent must not become a path that silently fails open.

    `average_file` returning the path regardless would start every pod from a
    file that does not exist -- or, worse, be passed to a push that falls
    back to the base while the run reported a warm start.
    """
    assert tlres.average_file(str(tmp_path), state()) == ""
    open(os.path.join(str(tmp_path), "round2.safetensors"), "w").close()
    got = tlres.average_file(str(tmp_path), state())
    assert got == os.path.join(str(tmp_path), "round2.safetensors")


def test_a_stale_absolute_path_is_resolved_in_the_current_directory(tmp_path):
    """The directory moves between hosts; the basename is what is portable."""
    open(os.path.join(str(tmp_path), "round2.safetensors"), "w").close()
    st = state(average="/somewhere/that/moved/round2.safetensors")
    assert tlres.average_file(str(tmp_path), st) == os.path.join(
        str(tmp_path), "round2.safetensors")


# --------------------------------------------------------------------------- #
# the coordinator writes one
# --------------------------------------------------------------------------- #

def test_what_the_run_writes_is_what_a_resume_checks(tmp_path):
    """The two sides of the comparison are built by the same method.

    If the written identity and the checked identity could drift, every
    resume would refuse on a phantom mismatch -- which is how a `session`
    that lived on the request but not on `QLoraSettings` nearly shipped.
    """
    c, _ = coord(tmp_path)
    c.save_resume(0, os.path.join(str(tmp_path), "round0.safetensors"))
    st = tlres.load(str(tmp_path))
    assert st is not None
    tlres.check(st, c.resume_identity())        # must not raise


def test_the_checkpoint_carries_the_coverage_and_the_counters(tmp_path):
    c, _ = coord(tmp_path)
    c._covered = {0, 1, 4}
    c._next_block = 5
    c._recycle = [2]
    c._trained, c._asked, c._subs, c._short = 29, 30, 3, 1
    c.save_resume(1, os.path.join(str(tmp_path), "round1.safetensors"))
    st = tlres.load(str(tmp_path))
    assert st.covered == [0, 1, 4]
    assert st.next_block == 5
    assert st.recycle == [2]
    assert (st.trained, st.asked, st.subs, st.short) == (29, 30, 3, 1)
    # The round to OPEN, not the one just closed.
    assert st.round == 2
    assert st.average == "round1.safetensors"


def test_the_checkpoint_records_the_block_size_not_just_the_knobs(tmp_path):
    """Because the block size is what gives a recorded id its meaning."""
    c, _ = coord(tmp_path, n_examples=100, rounds=2, pods=5)
    c.save_resume(0)
    assert tlres.load(str(tmp_path)).block_size == 10


def test_an_unwritable_checkpoint_does_not_end_a_healthy_run(tmp_path, capsys):
    """Bookkeeping for the NEXT run must never fail the one that is training.

    That inversion is the mistake this whole path exists to correct.
    """
    c, _ = coord(tmp_path)
    c.out_dir = "/proc/this/cannot/be/written"
    assert c.save_resume(0) == ""
    assert "could not be written" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# the coordinator adopts one
# --------------------------------------------------------------------------- #

def test_a_resumed_run_starts_the_pods_from_the_carried_average(tmp_path):
    """THE ONE THAT MATTERS.

    The dispatch pushes `average_path(round - 1)` and skips the push entirely
    when the round is 0. So a resume that renumbered the continuation from
    zero would hand every pod the bare base while the carried average sat
    unread on disk -- a run that reports itself as resumed and trains as if
    it were not. Nothing in a log would say so.
    """
    c, t = coord(tmp_path)
    avg = os.path.join(str(tmp_path), "round2.safetensors")
    open(avg, "w").close()
    c.adopt_resume(state(round=3), avg)
    assert t.round == 3
    assert c.average_path(t.round - 1) == avg


def test_a_resumed_run_does_not_retrain_what_it_already_trained(tmp_path):
    c, t = coord(tmp_path)
    c.adopt_resume(state(round=3, covered=[0, 1, 2, 3]), "")
    done, want = c.coverage()
    assert done == 4
    assert want == 10               # 100 examples / blocks of 10
    assert not c.finished()


def test_a_resumed_run_that_had_covered_everything_is_finished(tmp_path):
    """The terminating condition is coverage, so it has to see the carried set.

    Without this a resume of a run that stopped on its last block would rent
    a fresh fleet to discover it had nothing to do.
    """
    c, _ = coord(tmp_path)
    c.adopt_resume(state(round=2, covered=list(range(10))), "")
    assert c.finished()


def test_the_carried_counters_are_published_as_the_whole_run(tmp_path):
    """`examples_done` is what a client watches; it must not restart at zero.

    A resumed leg reporting only its own examples looks exactly like a run
    that lost everything, which is the thing the client was told to watch for.
    """
    c, _ = coord(tmp_path)
    c.adopt_resume(state(trained=2900, asked=2910), "")
    assert c.examples_trained() == 2900
    assert c.examples_asked() == 2910


def test_a_resumed_queue_does_not_walk_the_covered_ground_again(tmp_path):
    """The handout counter is carried, so the first block out is a fresh one."""
    c, t = coord(tmp_path)
    c.adopt_resume(state(round=3, covered=[0, 1, 2], next_block=3), "")
    c.cohort(3, [0], grow=True)
    assert c.blocks_for(0, 3) == [3]


def test_a_recycled_block_comes_back_before_the_queue_moves_on(tmp_path):
    """A pod died holding block 1 before the stop; the resume still owes it."""
    c, _ = coord(tmp_path)
    c.adopt_resume(state(round=3, covered=[0, 2], next_block=3,
                         recycle=[1]), "")
    c.cohort(3, [0], grow=True)
    assert c.blocks_for(0, 3) == [1]


def test_a_resume_at_round_zero_files_no_average(tmp_path):
    """A checkpoint from a run that closed nothing must not invent round -1."""
    c, _ = coord(tmp_path)
    c.adopt_resume(state(round=0, covered=[], average=""), "")
    assert c.average_path(-1) == ""
    assert c._averages == {}


def test_the_identity_the_coordinator_reports_sees_a_changed_dataset(tmp_path):
    """The late check, which is the only one that can see the example count.

    A dataset that gained a row, or an example that stopped being skipped,
    renumbers every block -- and neither is visible until the pass has run.
    """
    c, _ = coord(tmp_path, n_examples=110)
    with pytest.raises(tlres.ResumeError) as exc:
        tlres.check(state(), c.resume_identity())
    msg = str(exc.value)
    assert "the encoded example count changed (100 -> 110)" in msg
    assert "the block size changed" in msg


# --------------------------------------------------------------------------- #
# the round boundary writes it for real
# --------------------------------------------------------------------------- #

def test_closing_a_round_leaves_a_resume_point_on_disk(tmp_path):
    """Driven through the real close, not by calling the writer directly.

    The ordering is the substance: the checkpoint must name an average that
    is ALREADY on disk, because one that names a file not yet written is a
    resume that starts from the bare base while reporting a continuation. A
    test that called `save_resume` itself could not see that at all.
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _fleetfake import _write_tiny_adapter

    c, t = coord(tmp_path, n_examples=100, rounds=2, pods=2)
    c.cohort(0, [0, 1], grow=True)
    t.open_round(0)
    per = c.plan.block_size(len(c.encoded), c.total_rounds, c.planned_pods)
    for sh in (0, 1):
        c.blocks_for(sh, 0)
        f = _write_tiny_adapter(str(tmp_path / f"s{sh}.safetensors"))
        t.submit(tlr.Submission(shard=sh, round=0, path=f, examples=per))
        c.block_done(sh, 0, examples=per)

    assert c.close_round_if_ready() == 0

    st = tlres.load(str(tmp_path))
    assert st is not None, "a closed round left no resume point"
    assert st.round == 1, "the checkpoint must name the round to OPEN"
    assert sorted(st.covered) == [0, 1]
    assert st.trained == 2 * per
    # The average it names has to EXIST, which is the ordering guarantee.
    assert tlres.average_file(str(tmp_path), st) != ""
    assert st.average == "round0.safetensors"
    # And it describes this very run, so a resume of it is accepted.
    tlres.check(st, c.resume_identity())


def test_the_resume_point_is_refreshed_each_round(tmp_path):
    """Round 1's checkpoint must replace round 0's, not sit behind it."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _fleetfake import _write_tiny_adapter

    c, t = coord(tmp_path, n_examples=100, rounds=2, pods=2)
    per = c.plan.block_size(len(c.encoded), c.total_rounds, c.planned_pods)
    for rnd in (0, 1):
        c.cohort(rnd, [0, 1], grow=True)
        t.open_round(rnd)
        for sh in (0, 1):
            c.blocks_for(sh, rnd)
            f = _write_tiny_adapter(str(tmp_path / f"r{rnd}s{sh}.safetensors"))
            t.submit(tlr.Submission(shard=sh, round=rnd, path=f,
                                    examples=per))
            c.block_done(sh, rnd, examples=per)
        assert c.close_round_if_ready() == rnd

    st = tlres.load(str(tmp_path))
    assert st.round == 2
    assert st.average == "round1.safetensors"
    assert len(st.covered) == 4


# --------------------------------------------------------------------------- #
# driven through the real run
# --------------------------------------------------------------------------- #

def _ck(home, name, **kw):
    """A checkpoint on disk for a previous run called ``name``."""
    d = os.path.join(str(home), name)
    os.makedirs(d, exist_ok=True)
    # The identity the fake run really has: `QLoraSettings()` defaults,
    # one encoded example, three planned pods over one round.
    st = state(dataset_id="", base="stub-base", session="", rank=16,
               alpha=32, dropout=0.05, max_len=8192, seed=42, n_examples=1,
               block_size=1, rounds=1, planned_pods=3, round=2,
               covered=[0], average="")
    for k, v in kw.items():
        setattr(st, k, v)
    tlres.save(d, st)
    return d


def test_a_resumed_run_adopts_the_checkpoint(tmp_path, monkeypatch, capsys):
    """The wiring, exercised by the run itself rather than asserted on.

    The early vet and the adopt live in `_text_train_distributed`, not in the
    coordinator, so nothing in the unit tests above touches them. Asserted on
    what the run SAYS, because the run unregisters its coordinator when it
    ends and the object is gone by the time a test could read it -- and the
    second line below is printed only after `adopt_resume` has returned.
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import _fleetfake as ff

    _ck(tmp_path, "prev-run")
    fleet = ff.FakeFleet({0: 0.0, 1: 0.0, 2: 0.0})
    ff.run(monkeypatch, fleet=fleet, home=tmp_path,
           request=ff.req("adapter-fake", resume_from="prev-run"))
    out = capsys.readouterr().out
    # The checkpoint's OWN numbers, which only appear if it was read and
    # vetted rather than defaulted.
    assert "resuming 'prev-run'" in out
    assert "round 2, 1 block(s) trained" in out
    assert "resumed at round" in out
    # It named no average, so the run must say the pods start from the base
    # instead of implying a warm start it has not got.
    assert "coverage only" in out


def test_a_mismatched_checkpoint_is_refused_before_a_pod_is_rented(
        tmp_path, monkeypatch):
    """THE POINT OF VETTING EARLY.

    A resume refused after the fleet is up is refused with the meter running:
    the previous attempts cost $9.53 and $14 discovering their own faults.
    So the check that a caller is most likely to trip -- they changed
    `rounds` or `pods` between attempts -- has to fire before a rental.
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import _fleetfake as ff

    _ck(tmp_path, "prev-run", rounds=9)
    fleet = ff.FakeFleet({0: 0.0, 1: 0.0, 2: 0.0})
    with pytest.raises(RuntimeError) as exc:
        ff.run(monkeypatch, fleet=fleet, home=tmp_path,
               request=ff.req("adapter-fake", resume_from="prev-run"))
    msg = str(exc.value)
    assert "9 round(s)" in msg
    assert "ceil(examples / (rounds * pods))" in msg
    assert fleet.pods == {}, "a pod was rented before the resume was vetted"


def test_resuming_a_run_that_does_not_exist_says_so(tmp_path, monkeypatch):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import _fleetfake as ff

    fleet = ff.FakeFleet({0: 0.0})
    with pytest.raises(RuntimeError) as exc:
        ff.run(monkeypatch, fleet=fleet, home=tmp_path,
               request=ff.req("adapter-fake", resume_from="never-ran"))
    assert "no such run" in str(exc.value)
    assert fleet.pods == {}


def test_a_run_without_resume_from_is_untouched(tmp_path, monkeypatch):
    """The default path must not acquire a dependency on any of this."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import _fleetfake as ff

    fleet = ff.FakeFleet({0: 0.0, 1: 0.0, 2: 0.0})
    ff.run(monkeypatch, fleet=fleet, home=tmp_path)
    coord = loras._train_coordinators.get("job-fake")
    assert not hasattr(coord, "adopted")
