"""A video-to-video edit on VACE must be conditioned on the clip it was given.

Measured 2026-11-03 on Wan2.2-VACE-Fun-A14B, "make it a snowy winter day": the
answer was a competent snowy scene that followed neither the input's content
nor its motion, rated 2/5. The clip never reached the pipeline -- a v2v request
fell through the VACE branch's keyframe and continuation cases into "pure t2v",
which deliberately leaves video/mask unset. The request answered 200 either
way, so nothing distinguished "edited your clip" from "wrote a new one".

Separately, every non-VACE v2v call was handed the file's BYTES as `video=`,
which no diffusers pipeline accepts.
"""
import inspect

import pytest
from PIL import Image

from codai.api import video as V


def _mask_is_white(m):
    return m.mode == 'L' and m.getextrema() == (255, 255)


# ------------------------------------------------- the control-video pairing

def test_every_frame_is_reactive_so_the_whole_clip_is_re_rendered():
    """A black (conditioning) mask would reproduce the input unchanged, which
    is not an edit; white is what makes VACE re-render while its control
    branch still encodes the frames."""
    frames = [Image.new('RGB', (64, 48), (i, i, i)) for i in range(5)]
    video, mask = V._build_vace_control(frames, 5, 64, 48)
    assert len(video) == len(mask) == 5
    assert all(_mask_is_white(m) for m in mask)
    assert video[0] is frames[0], "the caller's frames must be used as-is"


def test_a_short_clip_is_padded_and_the_count_still_matches():
    frames = [Image.new('RGB', (32, 32))] * 2
    video, mask = V._build_vace_control(frames, 5, 32, 32)
    assert len(video) == 5 and len(mask) == 5
    assert video[-1].getpixel((0, 0)) == (128, 128, 128)


def test_a_longer_clip_is_cut_to_the_frame_budget():
    frames = [Image.new('RGB', (32, 32))] * 9
    video, mask = V._build_vace_control(frames, 4, 32, 32)
    assert len(video) == 4 and len(mask) == 4


def test_conditioning_and_control_are_opposite_arrangements():
    """Guards the pair: _build_vace_conditioning KEEPS what it is given."""
    frames = [Image.new('RGB', (32, 32))]
    _v, m = V._build_vace_conditioning(frames, 3, 32, 32)
    assert m[0].getextrema() == (0, 0), "a conditioned frame is masked black"
    assert m[-1].getextrema() == (255, 255)


# ----------------------------------------------------- reading the reference

def test_frames_are_sampled_across_the_whole_clip_not_off_its_head(monkeypatch):
    """Trimming to the first N frames hands the model a fraction of the motion
    it was told to follow, at the wrong speed."""
    import numpy as np
    src = [np.full((4, 4, 3), i, dtype=np.uint8) for i in range(100)]
    monkeypatch.setattr(V, '_decode_video_pyav', lambda path: (src, 25.0))
    monkeypatch.setattr(V, '_decode_b64_or_url', lambda data: b'not-a-real-mp4')
    got = V._frames_from_video_ref("ref", 8, 8, 5)
    assert len(got) == 5
    assert got[0].size == (8, 8)
    firsts = [f.getpixel((0, 0))[0] for f in got]
    assert firsts[0] == 0 and firsts[-1] == 99, firsts
    assert firsts == sorted(firsts)


def test_a_reference_that_decodes_to_nothing_returns_nothing(monkeypatch):
    monkeypatch.setattr(V, '_decode_b64_or_url', lambda data: b'')
    assert V._frames_from_video_ref("ref", 8, 8, 5) == []


def test_the_temp_file_is_cleaned_up(monkeypatch):
    """It writes the clip to disk to decode it; a server that leaks one file
    per v2v request fills its disk."""
    import numpy as np
    seen = {}

    def _dec(path):
        seen['path'] = path
        return [np.zeros((2, 2, 3), dtype=np.uint8)], 8.0

    monkeypatch.setattr(V, '_decode_video_pyav', _dec)
    monkeypatch.setattr(V, '_decode_b64_or_url', lambda data: b'x')
    V._frames_from_video_ref("ref", 4, 4, 1)
    import os
    assert not os.path.exists(seen['path'])


# ------------------------------------------------------------- the wiring

def test_the_vace_branch_has_a_video_to_video_case():
    """Read as source: reaching it needs the weights. The assertion that
    matters is that v2v is handled BEFORE the t2v else, which is what dropped
    the clip."""
    src = inspect.getsource(V._generate_video)
    i = src.index("is_vace = ")
    j = src.index("# Pure t2v on VACE", i)
    branch = src[i:j]
    assert "mode == 'v2v'" in branch
    assert "_build_vace_control" in branch


def test_a_plain_v2v_pipeline_is_handed_frames_not_bytes():
    src = inspect.getsource(V._generate_video)
    i = src.index("elif mode == 'v2v' and request.video:")
    tail = src[i:i + 700]
    assert "_frames_from_video_ref" in tail
    assert "kw['video'] = _decode_b64_or_url" not in src
