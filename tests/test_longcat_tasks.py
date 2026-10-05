"""Step 4: image-to-video, continuation, long video, and the staging.

Step 2's generate() was written against an ASSUMED pipeline interface and was wrong. The
real one, read from upstream's demos, is four separate methods —
``generate_t2v(height, width, …)``, ``generate_i2v(image, resolution, …)``,
``generate_vc(video, num_cond_frames, use_kv_cache, …)`` and
``generate_refine(stage1_video, spatial_refine_only, …)`` — each returning ``[0]``-indexed
frames normalised to [0, 1] rather than an object with ``.frames``, and the distilled pass
is a ``use_distill=True`` FLAG, not a LoRA we fuse. These tests pin that contract so it
cannot drift back.

Still no GPU, venv or weights here: the pipeline is a recording stub.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _service():
    """Import the service module without its heavy deps resolved (they load lazily)."""
    spec = importlib.util.spec_from_file_location(
        "longcat_service", ROOT / "tools" / "longcat_service.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


SVC = _service()
LC = SVC.LC


class _Pipe:
    """Records which upstream method was called, with what."""

    def __init__(self, frames=4):
        self.calls = []
        self._n = frames

    def _out(self):
        import numpy as np
        # [0]-indexed, normalised [0, 1], shape (frames, h, w, 3) — as upstream returns.
        return [np.zeros((self._n, 8, 8, 3), dtype="float32")]

    def generate_t2v(self, **kw):
        self.calls.append(("t2v", kw))
        return self._out()

    def generate_i2v(self, **kw):
        self.calls.append(("i2v", kw))
        return self._out()

    def generate_vc(self, **kw):
        self.calls.append(("vc", kw))
        return self._out()

    def generate_refine(self, **kw):
        self.calls.append(("refine", kw))
        return self._out()


def _ctx(**kw):
    base = {"prompt": "a cat", "negative_prompt": "blurry", "height": 480, "width": 832,
            "num_frames": 93, "num_cond_frames": 13, "num_segments": 1,
            "resolution": "480p", "num_inference_steps": None, "guidance_scale": None,
            "offload_kv_cache": False, "image": None, "cond_video": None,
            "generator": None}
    base.update(kw)
    return base


# ------------------------------------------------------------ the real API
def test_t2v_uses_height_and_width():
    p = _Pipe()
    SVC._one_pass(p, "t2v", "base", _ctx())
    name, kw = p.calls[0]
    assert name == "t2v"
    assert kw["height"] == 480 and kw["width"] == 832
    assert "resolution" not in kw           # that is the i2v/vc parameter


def test_i2v_uses_a_resolution_name():
    p = _Pipe()
    SVC._one_pass(p, "i2v", "base", _ctx(image="IMG"))
    name, kw = p.calls[0]
    assert name == "i2v"
    assert kw["image"] == "IMG" and kw["resolution"] == "480p"
    assert "height" not in kw


def test_continuation_passes_the_conditioning_video():
    p = _Pipe()
    SVC._one_pass(p, "vc", "base", _ctx(cond_video=["f1", "f2"]))
    name, kw = p.calls[0]
    assert name == "vc"
    assert kw["video"] == ["f1", "f2"]
    assert kw["num_cond_frames"] == 13
    assert kw["use_kv_cache"] is True


def test_the_distilled_stage_is_a_flag_not_a_lora():
    """Upstream passes use_distill=True to the SAME method; fusing cfg_step_lora by hand
    is not the interface."""
    p = _Pipe()
    SVC._one_pass(p, "t2v", "distill", _ctx())
    _name, kw = p.calls[0]
    assert kw["use_distill"] is True
    assert kw["num_inference_steps"] == 16 and kw["guidance_scale"] == 1.0
    assert not hasattr(SVC, "_apply_stage_lora")


def test_the_distilled_stage_passes_no_negative_prompt():
    """Its guidance is 1.0, so a negative prompt does nothing — upstream omits it."""
    p = _Pipe()
    SVC._one_pass(p, "t2v", "distill", _ctx())
    assert "negative_prompt" not in p.calls[0][1]
    p2 = _Pipe()
    SVC._one_pass(p2, "t2v", "base", _ctx())
    assert p2.calls[0][1]["negative_prompt"] == "blurry"


def test_frames_come_back_as_pil_from_normalised_floats():
    import numpy as np
    out = SVC._frames_from_output(np.zeros((3, 4, 4, 3), dtype="float32"))
    assert len(out) == 3
    assert out[0].mode == "RGB" and out[0].size == (4, 4)


def test_already_uint8_output_is_not_rescaled():
    import numpy as np
    arr = np.full((2, 4, 4, 3), 255, dtype="uint8")
    out = SVC._frames_from_output(arr)
    assert out[0].getpixel((0, 0)) == (255, 255, 255)


def test_the_resolution_name_follows_the_size():
    assert SVC._resolution_name(480, 832) == "480p"
    assert SVC._resolution_name(720, 1280) == "720p"


# ------------------------------------------------------------ the segment loop
def test_a_single_segment_keeps_every_frame():
    p = _Pipe(frames=10)
    frames, yielded = SVC._run_segments(p, "t2v", "base", _ctx(num_segments=1),
                                        log=lambda *_: None)
    assert len(frames) == 10 and yielded == 0
    assert [c[0] for c in p.calls] == ["t2v"]


def test_later_segments_drop_the_re_rendered_tail():
    """A call emits num_frames but its first num_cond_frames re-render the previous
    tail, so only the remainder is new. Keeping them would stutter the video."""
    p = _Pipe(frames=20)
    frames, _ = SVC._run_segments(p, "t2v", "base",
                                  _ctx(num_segments=3, num_cond_frames=5),
                                  log=lambda *_: None)
    assert len(frames) == 20 + 15 + 15
    # The first call starts the video; every later one continues it.
    assert [c[0] for c in p.calls] == ["t2v", "vc", "vc"]


def test_a_continuation_request_continues_from_the_first_call():
    p = _Pipe(frames=20)
    SVC._run_segments(p, "vc", "base",
                      _ctx(num_segments=2, cond_video=["prev"]), log=lambda *_: None)
    assert [c[0] for c in p.calls] == ["vc", "vc"]
    assert p.calls[0][1]["video"] == ["prev"]


def test_each_segment_conditions_on_the_previous_output():
    p = _Pipe(frames=20)
    SVC._run_segments(p, "t2v", "base", _ctx(num_segments=2, num_cond_frames=5),
                      log=lambda *_: None)
    # The second call's conditioning is the first call's frames, not the accumulator.
    assert len(p.calls[1][1]["video"]) == 20


def test_progress_tracks_the_segment():
    p = _Pipe(frames=20)
    SVC._run_segments(p, "t2v", "base", _ctx(num_segments=3), log=lambda *_: None)
    assert SVC._progress["segments"] == 3


# ------------------------------------------------------------ yielding the GPU
def test_a_yield_request_stops_at_a_segment_boundary():
    """A release arriving mid-request costs ONE segment instead of the whole generation,
    and the request still returns rather than being lost."""
    p = _Pipe(frames=20)
    SVC._yield_flag.set()
    try:
        frames, yielded = SVC._run_segments(p, "t2v", "base",
                                            _ctx(num_segments=5, num_cond_frames=5),
                                            log=lambda *_: None)
    finally:
        SVC._yield_flag.clear()
    assert len(p.calls) == 1            # stopped after the first segment
    assert yielded == 4 and len(frames) == 20


def test_a_single_segment_request_is_not_interrupted():
    p = _Pipe(frames=20)
    SVC._yield_flag.set()
    try:
        _frames, yielded = SVC._run_segments(p, "t2v", "base", _ctx(num_segments=1),
                                             log=lambda *_: None)
    finally:
        SVC._yield_flag.clear()
    assert yielded == 0 and len(p.calls) == 1


def test_the_worker_asks_for_a_yield_before_waiting():
    from codai.api import longcat_worker as W
    src = (ROOT / "codai/api/longcat_worker.py").read_text()
    rel = src[src.index("def release_vram"):]
    assert "_request_yield(key)" in rel
    assert callable(W._request_yield)


# ------------------------------------------------------------ refinement
def test_refinement_takes_the_earlier_frames_as_stage1_video():
    src = (ROOT / "tools/longcat_service.py").read_text()
    body = src[src.index('if stage == "refinement":'):]
    assert "stage1_video" in body
    assert "generate_refine" in body


# ------------------------------------------------------------ dispatch mapping
def test_the_modes_map_onto_the_upstream_tasks():
    src = (ROOT / "codai/api/video.py").read_text()
    seg = src[src.index('_task = {"t2v"'):src.index("if _task is None")]
    for mode, task in (("i2v", "i2v"), ("ti2v", "i2v"), ("extend", "vc"), ("v2v", "vc")):
        assert f'"{mode}": "{task}"' in seg, mode


def test_an_unsupported_mode_is_still_refused():
    src = (ROOT / "codai/api/video.py").read_text()
    fn = src[src.index("async def _generate_longcat"):]
    fn = fn[:fn.index("\nasync def ")]
    seg = fn[fn.index("if _task is None"):fn.index('payload = {')]
    assert "status_code=400" in seg


def test_continuation_reuses_the_existing_cond_frames_vocabulary():
    """A client that already chains clips through the VACE path needs no new field."""
    src = (ROOT / "codai/api/video.py").read_text()
    fn = src[src.index("async def _generate_longcat"):]
    fn = fn[:fn.index("\nasync def ")]
    assert "request.cond_frames" in fn and "cond_frames_b64" in fn


def test_the_long_video_fields_exist_and_the_server_owns_the_loop():
    from codai.pydantic.videorequest import VideoGenerationRequest as R
    r = R(model="m", prompt="p", num_segments=11, num_cond_frames=13)
    assert (r.num_segments, r.num_cond_frames) == (11, 13)
    src = (ROOT / "codai/api/video.py").read_text()
    fn = src[src.index("async def _generate_longcat"):]
    assert "num_segments" in fn and "total_frames" in fn


def test_a_shortened_video_is_reported_to_the_caller():
    """Yielding produces a shorter video; the caller must learn that from the response,
    not discover it."""
    src = (ROOT / "codai/api/video.py").read_text()
    fn = src[src.index("async def _generate_longcat"):]
    assert 'data.get("warning")' in fn and "warnings.append" in fn


def test_the_service_returns_a_tail_for_chaining():
    src = (ROOT / "tools/longcat_service.py").read_text()
    assert "tail_b64" in src
