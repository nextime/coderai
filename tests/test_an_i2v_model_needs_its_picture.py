"""A text-to-video request on image-to-video weights is refused, not served.

Wan ships the two directions as separate transformers and only the weights say
which one a checkpoint holds: 16 in-channels is text-to-video, 36 is
image-to-video. Asked for t2v on a 36-channel model, coderai used to rebuild
the i2v sibling pipeline and seed it with a neutral gray frame, on the theory
that an i2v model with no keyframe is still prompt-driven.

Measured 2026-11-03 over 30 clips on Wan2.2-I2V-A14B-Diffusers, -Lightning and
Wan2.2-Remix: every one answered 200 with uniform grey noise, rated 1/5 each
time. So the request is refused with a sentence that names the missing picture,
and -- because no reload can change the answer -- the generation ladder stops
instead of paying for another rung.
"""
import inspect

import pytest

from codai.api import video as V


class _T:
    def __init__(self, in_channels):
        self.config = type("cfg", (), {"in_channels": in_channels})()


class WanPipeline:                      # the name is what the code matches on
    def __init__(self, in_channels):
        self.transformer = _T(in_channels)

    def __call__(self, **kw):           # noqa: D401 - never reached in these tests
        raise AssertionError("the refusal must happen before the pipeline runs")


class WanImageToVideoPipeline(WanPipeline):
    pass


class LTX2Pipeline(WanPipeline):
    pass


# --------------------------------------------------------- what is refused

def test_a_prompt_only_request_on_i2v_weights_is_refused():
    with pytest.raises(V.VideoRequestMismatch) as err:
        V._refuse_t2v_on_an_i2v_model(WanPipeline(36), {}, 't2v')
    msg = str(err.value)
    assert "needs an input picture" in msg
    assert "init_image" in msg, "the refusal must name the field to send"


def test_the_refusal_is_a_client_error_not_a_server_error():
    """It is mapped to 400, so it must be the kind of error that means
    "change the request", not the kind that means "retry"."""
    assert issubclass(V.VideoRequestMismatch, ValueError)


# ----------------------------------------------------- what is left alone

def test_a_real_t2v_model_still_generates_from_a_prompt():
    pipe = WanPipeline(16)
    got, mode = V._refuse_t2v_on_an_i2v_model(pipe, {}, 't2v')
    assert got is pipe and mode == 't2v'


def test_i2v_weights_with_a_picture_are_exactly_the_supported_case():
    pipe = WanPipeline(36)
    kw = {"image": object()}
    got, mode = V._refuse_t2v_on_an_i2v_model(pipe, kw, 'i2v')
    assert got is pipe and mode == 'i2v'
    assert "image" in kw, "the keyframe must not be dropped"


def test_an_undetermined_channel_count_is_not_refused():
    """Reading the channel count is best-effort; a model it cannot read must
    still be allowed to try, or an unknown Wan variant becomes unservable."""
    pipe = WanPipeline(None)
    pipe.transformer = None
    got, _ = V._refuse_t2v_on_an_i2v_model(pipe, {}, 't2v')
    assert got is pipe


def test_another_family_is_not_judged_by_wan_channel_counts():
    pipe = LTX2Pipeline(36)
    got, _ = V._refuse_t2v_on_an_i2v_model(pipe, {}, 't2v')
    assert got is pipe


# ------------------------------------- the other half: an image nobody takes

def test_a_pipeline_that_takes_no_image_refuses_the_same_way():
    """16o/16p arrived as `LTX2Pipeline.__call__() got an unexpected keyword
    argument 'image'` from inside diffusers. Both halves of the disagreement
    now raise the one class the endpoint knows how to answer."""
    class _NoImage:
        def __call__(self, prompt=None, num_frames=None):
            pass
    with pytest.raises(V.VideoRequestMismatch):
        V._require_image_conditioning(_NoImage(), 'i2v')


# ------------------------------------------------- how the endpoint answers

def test_the_endpoint_answers_400_and_keeps_the_weights_loaded():
    """Read as source: reaching this line in the endpoint means loading a
    multi-GB video model.

    The teardown and the remaining ladder rungs must be skipped -- the pipeline
    is healthy and the next caller should not pay to reload it.
    """
    src = inspect.getsource(V.video_generations)
    assert "isinstance(e, VideoRequestMismatch)" in src
    i = src.index("isinstance(e, VideoRequestMismatch)")
    assert "status_code=400" in src[i:i + 300]
    assert "_teardown" not in src[i:i + 300], (
        "the refusal must be answered before the pipeline is torn down")
