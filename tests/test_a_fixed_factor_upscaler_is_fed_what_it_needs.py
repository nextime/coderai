"""An x2 request must not cost the memory of an x4.

The diffusers super-resolution pipelines have their ratio baked in -- x2 for
the latent one, x4 for the SD one -- and they denoise at the OUTPUT
resolution, so their peak is set by the picture, not by the checkpoint, and it
grows with the square of the input side.

The old path ran the model on the caller's picture at its full size and then
LANCZOS'd the oversized result back down to the requested size: four times the
memory for detail it then threw away. Measured 2026-11-03, a scale-2 request on
a 1024x1024 photo through stable-diffusion-x4-upscaler:

    500 Tried to allocate 9.00 GiB ... 5.26 GiB is free ...
        this process has 17.98 GiB memory in use

even with the VAE tiled and attention sliced -- those bound the decode, not the
UNet. Feeding the model `target / factor` instead gets the requested size out
of it directly, at a quarter of the peak, with no resampling after it at all.
"""
import pytest

from codai.api import images as I
from PIL import Image


def _img(w, h):
    return Image.new("RGB", (w, h))


def test_a_x2_request_on_the_x4_model_runs_at_a_quarter_of_the_pixels():
    """The client's exact case: 1024x1024, scale 2."""
    run, note = I._prepare_upscaler_input(_img(1024, 1024), 'diffusers', 2)
    # 512 * 4 == 2048 == 1024 * 2: the model's own output IS the answer, so
    # nothing is resampled and the peak is a quarter of what OOMed.
    assert run.size == (512, 512)
    assert note and "512x512" in note and "2048x2048" in note
    assert "capped" not in note


def test_the_model_output_needs_no_resampling_afterwards():
    src = _img(1024, 1024)
    run, _ = I._prepare_upscaler_input(src, 'diffusers', 2)
    out = _img(run.size[0] * 4, run.size[1] * 4)
    assert I._to_requested_scale(out, src, 2).size == (2048, 2048)
    assert I._to_requested_scale(out, src, 2) is out, (
        "the result is already the requested size — resampling it would only "
        "throw detail away")


def test_a_full_factor_request_is_left_alone():
    """x4 of a small picture is what the model was trained for."""
    run, note = I._prepare_upscaler_input(_img(128, 128), 'diffusers', 4)
    assert run.size == (128, 128)
    assert note is None


def test_the_latent_upscaler_knows_its_own_factor():
    assert I._native_upscale_factor('diffusers_latent') == 2
    assert I._native_upscale_factor('diffusers') == 4


def test_the_latent_upscaler_is_not_capped():
    """It ran the same 2048px output on the same card and was rated 5/5.

    Capping it would make a working path resample for no reason, so the cap is
    per backend and this one has none by default.
    """
    assert I._upscaler_input_cap('diffusers_latent') == 0
    run, note = I._prepare_upscaler_input(_img(1024, 1024), 'diffusers_latent', 2)
    assert run.size == (1024, 1024)
    assert note is None


def test_an_oversized_target_is_capped_rather_than_dying():
    """x4 of a 1024px picture still cannot run in one pass; it must not OOM."""
    run, note = I._prepare_upscaler_input(_img(1024, 1024), 'diffusers', 4)
    assert max(run.size) <= 640
    assert note and "capped" in note


def test_a_non_square_picture_keeps_its_aspect(monkeypatch):
    monkeypatch.setitem(I._UPSCALER_MAX_INPUT_PX, 'diffusers', 0)
    run, _ = I._prepare_upscaler_input(_img(1600, 900), 'diffusers', 2)
    assert run.size == (800, 450)


def test_the_cap_keeps_the_aspect_too():
    run, _ = I._prepare_upscaler_input(_img(1600, 900), 'diffusers', 2)
    assert run.size == (640, 360)


def test_the_cap_is_configurable(monkeypatch):
    monkeypatch.setitem(I._UPSCALER_MAX_INPUT_PX, 'diffusers', 256)
    run, _ = I._prepare_upscaler_input(_img(1024, 1024), 'diffusers', 2)
    assert max(run.size) <= 256


def test_real_esrgan_is_not_touched():
    """A GAN upscaler takes `outscale` and streams tiles; it has no fixed ratio."""
    import inspect
    src = inspect.getsource(I._run_upscale)
    i_gan = src.find("realesrgan")
    i_prep = src.find("_prepare_upscaler_input")
    assert 0 < i_gan < i_prep, "the ESRGAN branch must return before the prep"
