"""What the pipeline class settles is settled before the load, not after it.

Measured 2026-11-03 on a client's re-test of the v0.6.2 refusals: the answers
were right and arrived far too late. An LTX-2.5 video-to-video request was
refused after **2099 s**, an LTX-2.3 fine-tune after 1439 s, and a Wan
text-to-video request on image-to-video weights after 1394 s -- every second of
it spent loading several GB for a request that could not be served by the class
chosen to serve it. Nothing in those three answers needed the weights:

  * which class a family uses for a mode is `_detect_pipeline_class`, decided
    from the name and the mode alone;
  * whether that class takes `image` or `video` is its own signature;
  * whether a Wan checkpoint is the 16- or the 36-channel transformer is a
    field in its `transformer/config.json`, a few hundred bytes next to the
    weights.

So all three are answered up front. The post-load guards stay: a checkpoint
whose class comes from its own `model_index.json` (the `DiffusionPipeline`
fallback) genuinely cannot be read from here, and must still be caught late
rather than not at all.
"""
import json
import os

import pytest

from codai.api import video as V


# --------------------------------------------------------------------------- #
# the class decides, and it decides before the load
# --------------------------------------------------------------------------- #
class _NoVideo:
    def __call__(self, prompt=None, image=None, num_frames=None):
        raise AssertionError("not reached")


class _TakesVideo:
    def __call__(self, prompt=None, video=None, mask=None, conditioning_scale=None):
        raise AssertionError("not reached")


class _NoImage:
    def __call__(self, prompt=None, num_frames=None):
        raise AssertionError("not reached")


def test_a_class_that_takes_no_video_refuses_v2v_up_front(monkeypatch):
    monkeypatch.setattr(V, "_detect_pipeline_class",
                        lambda name, mode: _NoVideo)
    why = V._mode_is_servable_by_class("LTX-2.5", "v2v")
    assert why, "a v2v request on a class that takes no video must be refused"
    assert "_NoVideo" in why and "v2v" in why
    # The sentence has to say it did not pay for the load -- that is the whole
    # point of the check, and the thing the client measured.
    assert "before loading" in why


def test_a_class_that_takes_video_is_left_alone(monkeypatch):
    monkeypatch.setattr(V, "_detect_pipeline_class",
                        lambda name, mode: _TakesVideo)
    assert V._mode_is_servable_by_class("Wan2.2-VACE-Fun-A14B", "v2v") is None


def test_a_class_that_takes_no_image_refuses_i2v_up_front(monkeypatch):
    monkeypatch.setattr(V, "_detect_pipeline_class", lambda name, mode: _NoImage)
    for mode in ("i2v", "ti2v", "interp"):
        why = V._mode_is_servable_by_class("some-model", mode)
        assert why and mode in why


def test_t2v_needs_no_conditioning_and_is_never_refused_here(monkeypatch):
    monkeypatch.setattr(V, "_detect_pipeline_class", lambda name, mode: _NoImage)
    assert V._mode_is_servable_by_class("some-model", "t2v") is None


def test_the_diffusers_fallback_is_left_to_the_post_load_guards(monkeypatch):
    """A class chosen by the checkpoint cannot be inspected from here."""
    from diffusers import DiffusionPipeline
    monkeypatch.setattr(V, "_detect_pipeline_class",
                        lambda name, mode: DiffusionPipeline)
    assert V._mode_is_servable_by_class("mystery-model", "v2v") is None


def test_interp_is_image_conditioned_like_i2v():
    """It is driven by two frames, so it needs the image-to-video class.

    Left out of that tuple, `interp` picked the text-to-video class and then
    passed `image=` to it -- the exact mismatch `_effective_mode` exists to
    prevent, one mode further along.
    """
    assert "interp" in V._IMAGE_CONDITIONED_MODES
    assert V._conditioning_kwarg_for("interp") == "image"


# --------------------------------------------------------------------------- #
# the 36-channel question, from the config file
# --------------------------------------------------------------------------- #
def _write_transformer(tmp_path, in_channels):
    d = tmp_path / "transformer"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"in_channels": in_channels}))
    return str(tmp_path)


def test_in_channels_is_read_from_the_config_not_the_weights(tmp_path):
    root = _write_transformer(tmp_path, 36)
    assert V._wan_in_channels_on_disk(root) == 36


def test_a_t2v_request_on_36_channel_weights_is_refused_before_loading(tmp_path):
    root = _write_transformer(tmp_path, 36)
    why = V._refuse_t2v_on_an_i2v_model_before_loading(
        root + "/Wan2.2-I2V-A14B", "t2v", {"path": root})
    assert why and "needs an input picture" in why
    assert "36 in-channels" in why
    assert "before the weights are loaded" in why


def test_a_real_t2v_checkpoint_is_not_refused(tmp_path):
    root = _write_transformer(tmp_path, 16)
    assert V._refuse_t2v_on_an_i2v_model_before_loading(
        root + "/Wan2.2-T2V-A14B", "t2v", {"path": root}) is None


def test_the_request_that_brings_a_picture_is_never_this_case(tmp_path):
    root = _write_transformer(tmp_path, 36)
    for mode in ("i2v", "ti2v"):
        assert V._refuse_t2v_on_an_i2v_model_before_loading(
            root + "/Wan2.2-I2V-A14B", mode, {"path": root}) is None


def test_vace_is_left_alone(tmp_path):
    """VACE serves t2v by masked conditioning, whatever its channel count."""
    root = _write_transformer(tmp_path, 36)
    assert V._refuse_t2v_on_an_i2v_model_before_loading(
        root + "/Wan2.2-VACE-Fun-A14B", "t2v", {"path": root}) is None


def test_an_unreadable_config_falls_back_to_the_post_load_guard(tmp_path):
    """Silence, not a guess: the loaded-pipeline check still decides."""
    assert V._wan_in_channels_on_disk(str(tmp_path / "nope")) is None
    assert V._refuse_t2v_on_an_i2v_model_before_loading(
        "Wan2.2-I2V-A14B", "t2v", {}) is None


def test_the_disk_peek_never_touches_the_network(monkeypatch, tmp_path):
    """It runs on the request path; a Hub round-trip there is not acceptable."""
    calls = {}

    def _snap(repo, **kw):
        calls.update(kw)
        raise OSError("not cached")

    monkeypatch.setattr("huggingface_hub.snapshot_download", _snap)
    V._wan_in_channels_on_disk("some-org/some-wan-repo")
    assert calls.get("local_files_only") is True
