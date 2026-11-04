"""Plain Wan video-to-video is a pipeline, not a missing feature.

`WanVideoToVideoPipeline` has been in the installed diffusers all along, over
exactly the components `WanPipeline` uses (tokenizer, text encoder,
transformer, VAE, scheduler), and it takes `video` + `strength`. Nothing
selected it: the Wan branch picked the text-to-video or image-to-video class,
neither of which takes a video, and a v2v request on a plain Wan entry was
refused for a capability that was present. A client's 16p report measured it
from the other side -- "WanImageToVideoPipeline ... unexpected keyword argument
'video'" on Wan2.2-Remix.

It also brings its own trap: the class has NO `num_frames` argument, because a
video-to-video run takes its length from the clip it was handed. Passing the
frame count anyway is the unexpected-keyword failure in a new costume.
"""
import inspect

import pytest

from codai.api import video as V


def test_a_plain_wan_v2v_request_gets_the_video_to_video_class():
    from diffusers import WanVideoToVideoPipeline
    assert V._detect_pipeline_class("Wan2.2-Remix", "v2v") is WanVideoToVideoPipeline
    assert V._detect_pipeline_class("Wan2.2-T2V-A14B", "v2v") is WanVideoToVideoPipeline


def test_vace_still_wins_for_v2v():
    """VACE conditions on a control video and must keep its own pipeline."""
    from diffusers import WanVACEPipeline
    assert V._detect_pipeline_class(
        "Wan2.2-VACE-Fun-A14B", "v2v") is WanVACEPipeline


def test_the_other_modes_are_unchanged():
    from diffusers import WanPipeline, WanImageToVideoPipeline
    assert V._detect_pipeline_class("Wan2.2-T2V-A14B", "t2v") is WanPipeline
    for mode in ("i2v", "ti2v", "interp"):
        assert V._detect_pipeline_class(
            "Wan2.2-I2V-A14B", mode) is WanImageToVideoPipeline


def test_a_v2v_request_is_no_longer_refused_up_front():
    """It was: the class chosen for it took no video."""
    assert V._mode_is_servable_by_class("Wan2.2-Remix", "v2v") is None


def test_ltx2_v2v_is_still_refused_because_the_class_does_not_exist():
    """0.37.0.dev0 ships LTX2Pipeline, LTX2ImageToVideoPipeline and
    LTX2LatentUpsamplePipeline -- no video-to-video class at all. The refusal
    names that rather than implying the model is at fault."""
    why = V._mode_is_servable_by_class("LTX-2.5", "v2v")
    assert why and "LTX-2" in why


def test_num_frames_is_dropped_for_a_pipeline_without_it():
    """The length comes from the clip; the argument does not exist."""
    from diffusers import WanVideoToVideoPipeline
    assert 'num_frames' not in inspect.signature(
        WanVideoToVideoPipeline.__call__).parameters
    src = inspect.getsource(V._generate_video)
    i = src.find("not _pipeline_takes(pipe, 'num_frames')")
    assert i > 0, "the v2v branch must drop a frame count the class cannot take"
    assert "kw.pop('num_frames')" in src[i:i + 400]


def test_strength_is_only_passed_where_it_is_accepted():
    src = inspect.getsource(V._generate_video)
    assert "_pipeline_takes(pipe, 'strength')" in src


def test_the_video_refusal_no_longer_claims_vace_is_the_only_way():
    """The sentence was wrong the moment this class was selectable."""
    src = inspect.getsource(V._require_video_conditioning)
    assert "WanVideoToVideoPipeline" in src
