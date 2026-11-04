"""One bad request must not leave a model serving nothing.

A loaded video pipeline is cached under the model key alone -- the mode is not
part of it -- so the class that is resident is whichever mode asked first. Wan
serves text-to-video and image-to-video with two different pipeline classes
over the same catalogue entry, and that is where the two facts collide.

Measured 2026-11-03, the sequence a client hit on three entries at once
(Wan2.2-I2V-A14B-Diffusers, -Lightning, Wan2.2-Remix):

  1. a t2v request loads `WanPipeline`, is refused because the transformer is
     the 36-channel image-to-video one, and leaves the pipeline warm on purpose
     -- it is healthy, and the next caller should not pay for this one's
     mistake;
  2. every `i2v` request after it reuses that warm `WanPipeline` without
     loading anything, and is refused in 0 s with "WanPipeline takes no image".

So the model could do neither direction: the picture was there and
`WanImageToVideoPipeline` was installed, but the cache lookup never asked which
mode the resident class could serve. The answer is a reload, not a refusal --
the request IS serveable, just not by what happens to be resident.
"""
import pytest

from codai.api import video as V


class WanPipeline:
    def __call__(self, prompt=None, num_frames=None):
        raise AssertionError("not reached")


class WanImageToVideoPipeline:
    def __call__(self, prompt=None, image=None, num_frames=None):
        raise AssertionError("not reached")


class WanVACEPipeline:
    def __call__(self, prompt=None, video=None, mask=None):
        raise AssertionError("not reached")


class LTX2Pipeline:
    def __call__(self, prompt=None, num_frames=None):
        raise AssertionError("not reached")


def _detect(mapping):
    return lambda name, mode: mapping.get(mode)


def test_a_resident_t2v_class_does_not_get_to_answer_i2v(monkeypatch):
    """The case the client measured: 0 s refusals on a model that can do it."""
    monkeypatch.setattr(V, "_detect_pipeline_class", _detect({
        "t2v": WanPipeline, "i2v": WanImageToVideoPipeline,
        "ti2v": WanImageToVideoPipeline}))
    warm = WanPipeline()
    for mode in ("i2v", "ti2v"):
        assert V._warm_pipe_cannot_serve(warm, "Wan2.2-I2V-A14B-Diffusers", mode)


def test_the_right_class_resident_is_reused(monkeypatch):
    """A reload here would be a multi-GB answer to a question nobody asked."""
    monkeypatch.setattr(V, "_detect_pipeline_class", _detect({
        "i2v": WanImageToVideoPipeline}))
    assert not V._warm_pipe_cannot_serve(
        WanImageToVideoPipeline(), "Wan2.2-I2V-A14B-Diffusers", "i2v")


def test_t2v_never_triggers_a_reload(monkeypatch):
    """t2v carries no conditioning, so every class can be asked for it."""
    monkeypatch.setattr(V, "_detect_pipeline_class", _detect({
        "t2v": WanImageToVideoPipeline}))
    assert not V._warm_pipe_cannot_serve(WanPipeline(), "Wan2.2-T2V", "t2v")


def test_no_sibling_means_a_refusal_not_a_reload(monkeypatch):
    """When the class for this mode cannot serve it either, reloading is futile.

    This is the LTX video-to-video case: `_detect_pipeline_class` has no
    video-to-video class to offer, so the honest answer is the refusal from
    `_mode_is_servable_by_class`, not several GB of reload that ends in the
    same sentence.
    """
    monkeypatch.setattr(V, "_detect_pipeline_class", _detect({
        "v2v": LTX2Pipeline}))
    assert not V._warm_pipe_cannot_serve(LTX2Pipeline(), "LTX-2.5", "v2v")


def test_a_v2v_request_on_a_resident_image_class_reloads_vace(monkeypatch):
    monkeypatch.setattr(V, "_detect_pipeline_class", _detect({
        "v2v": WanVACEPipeline}))
    assert V._warm_pipe_cannot_serve(
        WanImageToVideoPipeline(), "Wan2.2-VACE-Fun-A14B", "v2v")


def test_a_broken_detector_does_not_break_the_request(monkeypatch):
    def _boom(name, mode):
        raise RuntimeError("diffusers import exploded")

    monkeypatch.setattr(V, "_detect_pipeline_class", _boom)
    assert not V._warm_pipe_cannot_serve(WanPipeline(), "whatever", "i2v")


def test_the_endpoint_drops_the_stale_pipeline_before_reloading():
    """The reload has to evict, or the manager keeps handing the same object back.

    Checked as source because the surrounding handler needs a loaded model, a
    GPU and a request to run: what matters is that the branch pops the model
    key and frees the VRAM off the loop rather than merely reassigning a local.
    """
    import inspect
    src = inspect.getsource(V.video_generations)
    # The LAST mention: the first is the comment pointing at this function.
    i = src.rfind("_warm_pipe_cannot_serve")
    assert i > 0, "the endpoint must consult the warm-pipeline check"
    window = src[i:i + 900]
    assert "models.pop(model_key" in window
    assert "model_pools.pop(model_key" in window
    assert "_free_pipeline_vram" in window
    assert "run_in_executor" in window, "freeing a pipeline must leave the loop"
    assert "pipe = None" in window, "the loader must then build the right class"


def test_the_endpoint_checks_the_class_before_it_loads():
    """The pre-load refusals must sit above the `if pipe is None:` load block."""
    import inspect
    src = inspect.getsource(V.video_generations)
    i_class = src.find("_mode_is_servable_by_class")
    i_disk = src.find("_refuse_t2v_on_an_i2v_model_before_loading")
    i_load = src.find("if pipe is None:")
    assert 0 < i_class < i_load
    assert 0 < i_disk < i_load


def test_gguf_video_models_skip_the_class_check():
    """stable-diffusion.cpp models never go through a diffusers class."""
    import inspect
    src = inspect.getsource(V.video_generations)
    i = src.find("_mode_is_servable_by_class")
    assert "_is_gguf_video" in src[max(0, i - 600):i]
