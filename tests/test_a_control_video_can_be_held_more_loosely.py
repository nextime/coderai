"""How tightly a VACE run follows its control video is now the caller's choice.

With the input clip reaching VACE as a control video (fixed in v0.6.2), a
video-to-video edit follows the original's layout and motion -- which is what
was asked for, and is also why the prompt can lose. Measured 2026-11-03 on
Wan2.2-VACE-Fun-A14B, "make it a snowy winter day" over a summer street clip:
the layout, the bicycles, the people and the motion all came through, the
people were restyled into winter coats, and there was no snow on the ground or
in the air. At `conditioning_scale=1.0`, the pipeline default, the control
video's structure outvotes a prompt that asks for the scene to change.

So the scale is exposed as `control_strength`, per request or as a per-model
default, and passed only to pipelines that take one.
"""
import pytest

from codai.pydantic.videorequest import VideoGenerationRequest
from codai.api import video as V


def test_the_field_exists_and_defaults_to_the_pipeline_default():
    """Unset means unset: VACE's own 1.0, not a number coderai invented."""
    assert VideoGenerationRequest(model="m").control_strength is None
    assert VideoGenerationRequest(model="m", control_strength=0.65
                                  ).control_strength == 0.65


def test_it_reaches_the_pipeline_as_conditioning_scale():
    import inspect
    src = inspect.getsource(V._generate_video)
    i = src.find("_cs = getattr(request, 'control_strength'")
    assert i > 0, "the v2v VACE branch must read the knob"
    window = src[i:i + 700]
    assert "conditioning_scale" in window
    assert "_pipeline_takes(pipe, 'conditioning_scale')" in window, (
        "a pipeline without the argument must not be handed it — that is the "
        "unexpected-keyword failure this module keeps fixing")


def test_a_per_model_default_is_honoured():
    """An operator who tuned an entry should not need every client to ask."""
    import inspect
    src = inspect.getsource(V._generate_video)
    i = src.find("_cs = getattr(request, 'control_strength'")
    window = src[i:i + 700]
    assert "defaults" in window, (
        "models.json defaults.control_strength must be consulted when the "
        "request is silent")


def test_the_request_wins_over_the_card():
    import inspect
    src = inspect.getsource(V._generate_video)
    i = src.find("_cs = getattr(request, 'control_strength'")
    window = src[i:i + 700]
    # The request is read first and the card only consulted when it is None.
    i_req = window.find("getattr(request, 'control_strength'")
    i_cfg = window.find("defaults")
    assert 0 <= i_req < i_cfg
