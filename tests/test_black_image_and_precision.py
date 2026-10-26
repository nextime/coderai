"""A black image is not an image, and --no-ram must not change the numerics.

Reported from the field, three edits of one photo:

* Qwen-Image-Edit-2511 -- good.
* Z-Image-Turbo -- **a black image**.
* stable-diffusion-xl-base-1.0 -- "changing way more than requested".

The middle one was ours. The Z-Image entry is configured `precision: bf16`,
and `no_ram: true` overrode it with fp16 on the way in -- "to maximize VRAM
efficiency", except fp16 and bf16 are both TWO BYTES per parameter, so the
override saved nothing and only narrowed the range. fp16 tops out near 65504,
a flow-matching DiT runs past it, the latents go NaN, and a VAE decodes NaN to
uniform black. Answered 200 OK with a plausible PNG; the only way anyone found
out was a person looking at the picture.

So: the override now applies only from f32 (where it genuinely halves
memory), and a uniformly black or white result is refused instead of returned.
"""
import pytest
from PIL import Image

from codai.api import images as images_mod


# --------------------------------------------------------------------------- #
#  --no-ram and precision
# --------------------------------------------------------------------------- #

def test_no_ram_no_longer_overrides_a_configured_half_precision():
    src = open("codai/api/images.py").read()
    head = src.index("# --no-ram mode: override dtype to auto-select best for GPU")
    body = src[head:head + 1400]
    assert "if dtype is torch.float32:" in body, \
        "the fp16 coercion has to be conditional on f32"
    assert "dtype = torch.float16  # Use fp16 to maximize VRAM efficiency" not in src, \
        "the unconditional override is what blacked out every Z-Image edit"


def test_the_two_half_types_are_the_same_size_which_is_why_the_override_was_pointless():
    import torch
    assert torch.finfo(torch.float16).bits == torch.finfo(torch.bfloat16).bits == 16
    # ...and this is what it cost: four orders of magnitude of headroom.
    assert torch.finfo(torch.bfloat16).max > torch.finfo(torch.float16).max * 1000


# --------------------------------------------------------------------------- #
#  refusing a flat image
# --------------------------------------------------------------------------- #

def _flat(colour, size=(16, 16)):
    return Image.new("RGB", size, colour)


def test_a_black_result_is_refused_rather_than_returned():
    with pytest.raises(RuntimeError, match="uniformly black"):
        images_mod._refuse_uniform_output([_flat((0, 0, 0))], "Z-Image-Turbo")


def test_a_white_result_is_refused_too():
    with pytest.raises(RuntimeError, match="uniformly white"):
        images_mod._refuse_uniform_output([_flat((255, 255, 255))], "some-model")


def test_the_message_names_the_dtype_because_that_is_what_needs_changing():
    import torch
    with pytest.raises(RuntimeError, match="bf16"):
        images_mod._refuse_uniform_output([_flat((0, 0, 0))], "m",
                                          torch.float16)


def test_the_message_says_which_image_of_how_many():
    with pytest.raises(RuntimeError, match=r"image 2 of 2"):
        images_mod._refuse_uniform_output(
            [_flat((10, 20, 30)), _flat((0, 0, 0))], "m")
    # ...and the first one, which has real content, is not what was blamed.


def test_a_flat_mid_grey_is_left_alone():
    """Only the extremes are the NaN signature. A deliberately flat render in
    some other colour is content, however odd, and refusing it would be us
    deciding what a picture may look like."""
    images_mod._refuse_uniform_output([_flat((128, 128, 128))], "m")
    images_mod._refuse_uniform_output([_flat((7, 90, 200))], "m")


def test_an_ordinary_picture_passes():
    img = Image.new("RGB", (8, 8), (0, 0, 0))
    img.putpixel((3, 3), (255, 255, 255))      # any variation at all
    images_mod._refuse_uniform_output([img], "m")


def test_nothing_to_judge_is_not_an_error():
    images_mod._refuse_uniform_output([], "m")
    images_mod._refuse_uniform_output(None, "m")
    images_mod._refuse_uniform_output(["not an image"], "m")


def test_both_result_paths_check_their_output():
    """Generation and editing both return images to a client, so a flat one
    must not get through either."""
    src = open("codai/api/images.py").read()
    assert src.count("_refuse_uniform_output(") >= 3, \
        "definition plus both call sites"
    # The edit path turns it into a 500 rather than letting it escape as one.
    edit = src[src.index("_refuse_uniform_output(result.images"):][:400]
    assert "status_code=500" in edit


# --------------------------------------------------------------------------- #
#  a component IS the model
# --------------------------------------------------------------------------- #

def test_a_component_that_cannot_load_fails_the_model(monkeypatch):
    """`component_models` is how one id serves a fine-tuned transformer on
    another pipeline's VAE. Skipping a failed one loads the BASE component, so
    the id quietly serves a different model and the load goes on to succeed.

    In production that skipped the V19 transformer, served plain
    Qwen-Image-Edit-2511 under the Rapid-AIO name, then OOMed holding it whole.
    """
    import diffusers
    from codai.models import hf_loading

    class Boom:
        @staticmethod
        def from_pretrained(source, **kw):
            raise ValueError("Blockwise 4bit quantization only supports "
                             "16/32-bit floats, but got torch.float8_e4m3fn")

    monkeypatch.setattr(hf_loading, "_discover_components",
                        lambda name: {"transformer": ["diffusers", "FakeT"]})
    monkeypatch.setattr(diffusers, "FakeT", Boom, raising=False)

    cfg = {"component_models": {"transformer": "org/some-fp8-tune"},
           "component_quantization": {"transformer": "4bit"}}
    with pytest.raises(RuntimeError) as e:
        hf_loading.build_component_models("base/model", cfg, "bf16")
    msg = str(e.value)
    assert "transformer" in msg and "org/some-fp8-tune" in msg


def test_an_fp8_source_says_what_to_do_instead():
    """The remedy is specific and worth printing: serve that component from a
    GGUF, which is what the working siblings do."""
    import diffusers
    from codai.models import hf_loading

    class Boom:
        @staticmethod
        def from_pretrained(source, **kw):
            raise ValueError("got torch.float8_e4m3fn")

    import unittest.mock as m
    with m.patch.object(hf_loading, "_discover_components",
                        lambda name: {"transformer": ["diffusers", "FakeT2"]}), \
         m.patch.object(diffusers, "FakeT2", Boom, create=True):
        with pytest.raises(RuntimeError) as e:
            hf_loading.build_component_models(
                "base/model",
                {"component_models": {"transformer": "org/fp8"},
                 "component_quantization": {"transformer": "4bit"}}, "bf16")
    assert ".gguf" in str(e.value), "the message has to name the way out"
