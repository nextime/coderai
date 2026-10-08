"""The denoising progress the caller polls, and the segment count it spans.

A 304-frame request on 2026-10-08 sat at "generating 0/16 (0%)" for its whole
nine-minute first segment while the service's own log showed
``Denoising: 94%|...| 15/16``. Two separate defects:

1. ``tools/longcat_service.py`` set ``_progress["step"] = 0`` at the start of every
   segment and never again. The step count existed only inside the vendored
   pipeline's ``tqdm(total=len(timesteps), desc="Denoising")``, which takes no
   callback, so nothing reached ``GET /progress``. The consumer in
   ``codai/api/video.py`` was already polling it correctly -- it faithfully reported
   the 0 it was handed.

2. Even with a live step, the totals counted ONE segment. A four-segment render
   would have run 0->16 four times over against a denominator of 16.
"""
import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SERVICE = ROOT / "tools" / "longcat_service.py"


@pytest.fixture
def service(monkeypatch):
    spec = importlib.util.spec_from_file_location("lc_service_progress", SERVICE)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, mod)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------- the instrumented bar

def _bars(service, monkeypatch):
    """Install the patched tqdm into two stand-in pipeline modules and return it."""
    import types

    mods = {}
    for name in ("longcat_video.pipeline_longcat_video",
                 "longcat_video.pipeline_longcat_video_avatar"):
        m = types.ModuleType(name)
        m.tqdm = object()
        monkeypatch.setitem(sys.modules, name, m)
        mods[name] = m
    service._install_progress_bars()
    return mods


def test_the_pipeline_modules_bar_is_replaced(service, monkeypatch):
    """`from tqdm import tqdm` binds the name on the MODULE, so that is what has to
    be rebound -- patching tqdm.tqdm after the import would be too late."""
    mods = _bars(service, monkeypatch)
    for name, mod in mods.items():
        assert mod.tqdm.__name__ == "_ProgressTqdm", name


def test_installing_twice_does_not_stack_wrappers(service, monkeypatch):
    mods = _bars(service, monkeypatch)
    first = mods["longcat_video.pipeline_longcat_video"].tqdm
    service._install_progress_bars()
    assert mods["longcat_video.pipeline_longcat_video"].tqdm is first


def test_a_denoising_bar_publishes_its_step(service, monkeypatch):
    """The defect itself: stepping the bar must move what /progress reports."""
    mods = _bars(service, monkeypatch)
    bar_cls = mods["longcat_video.pipeline_longcat_video"].tqdm
    service._progress.update(segments=1)
    with bar_cls(total=16, desc="Denoising") as bar:
        assert service._progress["steps"] == 16
        assert service._progress["step"] == 0
        for expected in range(1, 17):
            bar.update()
            assert service._progress["step"] == expected


def test_a_bar_that_is_not_denoising_is_ignored(service, monkeypatch):
    """Checkpoint-shard bars must not move the request's progress."""
    mods = _bars(service, monkeypatch)
    bar_cls = mods["longcat_video.pipeline_longcat_video"].tqdm
    service._progress.update(step=4, steps=16)
    with bar_cls(total=5, desc="Loading checkpoint shards") as bar:
        bar.update()
    assert (service._progress["step"], service._progress["steps"]) == (4, 16)


def test_the_bar_total_sets_the_step_count_for_every_segment(service, monkeypatch):
    """`steps` is the whole stage: one pass's steps times the segment count."""
    mods = _bars(service, monkeypatch)
    bar_cls = mods["longcat_video.pipeline_longcat_video"].tqdm
    service._progress.update(segments=4)
    with bar_cls(total=16, desc="Denoising"):
        pass
    assert service._progress["steps"] == 64


def test_the_avatar_bar_is_instrumented_too(service, monkeypatch):
    """The avatar pipeline is a different module with its own denoising loops, and
    its distilled pass is 8 steps rather than 16."""
    mods = _bars(service, monkeypatch)
    bar_cls = mods["longcat_video.pipeline_longcat_video_avatar"].tqdm
    service._progress.update(segments=1)
    with bar_cls(total=8, desc="Denoising") as bar:
        bar.update()
        assert (service._progress["step"], service._progress["steps"]) == (1, 8)


# ------------------------------------------------------------------ segment offsets

def test_a_later_segment_continues_the_count(service, monkeypatch):
    """Segment 2 of 4 starts at 16, not back at 0 -- the bar restarts, the stage
    does not."""
    mods = _bars(service, monkeypatch)
    bar_cls = mods["longcat_video.pipeline_longcat_video"].tqdm
    service._progress.update(segments=4)
    service._bar["offset"] = 16
    with bar_cls(total=16, desc="Denoising") as bar:
        assert service._progress["step"] == 16
        bar.update()
        assert service._progress["step"] == 17


def test_the_offset_advances_one_pass_per_segment(service, monkeypatch):
    """_run_segments is what sets the offset, so drive it with a fake pass."""
    mods = _bars(service, monkeypatch)
    bar_cls = mods["longcat_video.pipeline_longcat_video"].tqdm
    seen = []

    def fake_pass(pipe, task, stage, ctx, cond=None, log=None):
        with bar_cls(total=16, desc="Denoising") as bar:
            for _ in range(16):
                bar.update()
        seen.append((service._progress["segment"], service._bar["offset"],
                     service._progress["step"], service._progress["steps"]))
        return [object()] * 93

    monkeypatch.setattr(service, "_one_pass", fake_pass)
    monkeypatch.setattr(service, "_frames_from_output", list)
    monkeypatch.setattr(service, "torch_gc", lambda: None)
    ctx = {"num_segments": 4, "num_cond_frames": 13, "cond_video": None,
           "num_inference_steps": None, "guidance_scale": None}
    service._progress.update(segments=4)
    service._bar["offset"] = 0
    service._bar["pass_steps"] = 0
    acc, yielded = service._run_segments(None, "t2v", "distill", ctx, log=lambda *_: None)

    assert yielded == 0
    assert [s[0] for s in seen] == [1, 2, 3, 4]
    assert [s[1] for s in seen] == [0, 16, 32, 48]
    # The step count rises monotonically across the whole stage and ends at the total.
    assert [s[2] for s in seen] == [16, 32, 48, 64]
    assert {s[3] for s in seen} == {64}


def test_the_first_segment_falls_back_to_the_stage_default(service, monkeypatch):
    """Before any bar has existed there is no measured pass length, so segment 1
    must still report a sane denominator rather than 0."""
    _bars(service, monkeypatch)
    calls = []

    def fake_pass(pipe, task, stage, ctx, cond=None, log=None):
        calls.append((service._progress["step"], service._progress["steps"]))
        return [object()] * 93

    monkeypatch.setattr(service, "_one_pass", fake_pass)
    monkeypatch.setattr(service, "_frames_from_output", list)
    monkeypatch.setattr(service, "torch_gc", lambda: None)
    service._bar["offset"] = 0
    service._bar["pass_steps"] = 0
    ctx = {"num_segments": 2, "num_cond_frames": 13, "cond_video": None,
           "num_inference_steps": None, "guidance_scale": None}
    service._run_segments(None, "t2v", "distill", ctx, log=lambda *_: None)
    # 16 is the distilled stage's default, times two segments.
    assert calls[0] == (0, 32)


def test_the_steps_honour_an_explicit_request_count(service, monkeypatch):
    _bars(service, monkeypatch)

    def fake_pass(pipe, task, stage, ctx, cond=None, log=None):
        return [object()] * 93

    monkeypatch.setattr(service, "_one_pass", fake_pass)
    monkeypatch.setattr(service, "_frames_from_output", list)
    monkeypatch.setattr(service, "torch_gc", lambda: None)
    service._bar["offset"] = 0
    service._bar["pass_steps"] = 0
    ctx = {"num_segments": 3, "num_cond_frames": 13, "cond_video": None,
           "num_inference_steps": 20, "guidance_scale": None}
    service._run_segments(None, "t2v", "distill", ctx, log=lambda *_: None)
    assert service._progress["steps"] == 60


def test_progress_reports_no_private_keys(service):
    """The offset bookkeeping is ours; /progress returns the documented contract."""
    assert set(service._progress) == {"active", "stage", "segment", "segments",
                                      "step", "steps"}


# ----------------------------------------------------- the consumer in codai/api/video.py
# Extracted from source the way tests/test_longcat_cancellable.py does: the poll loop
# lives inside a long async request handler that cannot be called in a unit test.

def _longcat_body() -> str:
    import re
    src = (ROOT / "codai" / "api" / "video.py").read_text(encoding="utf-8")
    start = src.index("async def _generate_longcat")
    nxt = re.search(r"\n(async def |def )", src[start + 10:])
    return src[start:start + 10 + nxt.start()] if nxt else src[start:]


def test_the_total_spans_every_segment():
    """A four-segment render must not be measured against one segment's steps."""
    body = _longcat_body()
    assert "_grand_total(" in body
    assert "num_segments" in body


def test_refinement_is_not_multiplied_by_the_segment_count():
    """It is a single pass over the finished video, not one per segment."""
    body = _longcat_body()
    assert '(1 if s == "refinement" else segments)' in body


def test_the_segment_count_is_corrected_from_the_service():
    """A request that asked for a duration rather than a segment count only learns
    the real count from the service's first progress report."""
    body = _longcat_body()
    assert '_vid_progress_total(' in body
    assert 'p.get("segments")' in body


def test_a_finished_stage_contributes_what_it_actually_ran():
    """lc.STAGE_DEFAULTS is one pass's steps; the stage ran it once per segment, and
    the service reports that in `steps`."""
    body = _longcat_body()
    assert '_stage_total or lc.STAGE_DEFAULTS.get(' in body
    assert 'p.get("steps")' in body


def test_the_task_page_total_is_kept_in_step():
    body = _longcat_body()
    assert 'task_registry.step(_tid, _step, total=' in body


def test_correcting_the_total_does_not_restart_the_bar():
    """_vid_progress_reset would zero `current` and the elapsed time the rate is
    derived from, which is why this is a separate setter."""
    from codai.api import video

    video._vid_progress_reset(16)
    video._vid_progress_step(7)
    started = video._vid_progress["started_at"]
    rate = video._vid_progress["it_per_s"]

    video._vid_progress_total(64)

    assert video._vid_progress["total"] == 64
    assert video._vid_progress["current"] == 7
    assert video._vid_progress["active"] is True
    assert video._vid_progress["started_at"] == started
    assert video._vid_progress["it_per_s"] == rate
