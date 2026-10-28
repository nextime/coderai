# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""C4: reference-image editing for Krea 2, which diffusers does not ship.

`Krea2Pipeline.__call__` takes a prompt and no image and there is no
`Krea2Img2ImgPipeline`, so an edit request fell through to text-to-image and
silently ignored the image. The edit is done IN CONTEXT: the reference is
encoded once, kept clean, and appended to the transformer's `hidden_states`
beside the canvas, which is what the family's edit LoRAs are trained against.

These test the parts that are easy to get wrong and do not need the 12B
weights: the coordinates that tell the two token groups apart, the geometry
snapping, and that the pipeline is recognised as an instruction edit (so no
`strength` slider is offered for a control it does not have).
"""

import pytest

from codai.pipelines import krea2_edit as K


# ------------------------------------------------------------- coordinates

def test_the_reference_sits_on_its_own_plane():
    """Identical coordinates would make the two groups indistinguishable.

    `position_ids` is (seq, 3): text at the origin, image tokens carrying
    (plane, h, w). The reference repeats the canvas's (h, w) so the same
    coordinate means the same place in both, and differs on the first axis so
    the rotary embedding can still tell a canvas token from the reference token
    aligned with it.
    """
    torch = pytest.importorskip("torch")
    ids = K.edit_position_ids(3, 2, 2, 2, 2, torch.device("cpu"))
    assert ids.shape == (3 + 4 + 4, 3)

    text, canvas, ref = ids[:3], ids[3:7], ids[7:]
    assert text.abs().sum() == 0, "text tokens must sit at the origin"
    assert (canvas[:, 0] == 0).all(), "the canvas must stay on plane 0"
    assert (ref[:, 0] == 1).all(), "the reference must be on its own plane"
    # Same spatial grid in both, so a coordinate means the same place.
    assert canvas[:, 1:].tolist() == ref[:, 1:].tolist()


def test_a_reference_of_a_different_shape_keeps_its_own_grid():
    torch = pytest.importorskip("torch")
    ids = K.edit_position_ids(1, 2, 3, 4, 1, torch.device("cpu"))
    assert ids.shape == (1 + 6 + 4, 3)
    ref = ids[1 + 6:]
    assert (ref[:, 0] == 1).all()
    assert ref[:, 1].max().item() == 3, "the reference lost its own height"


def test_the_coordinates_are_text_then_canvas_then_reference():
    """Order matters: the canvas is sliced off the prediction by length, so it
    has to be the first image group."""
    torch = pytest.importorskip("torch")
    ids = K.edit_position_ids(2, 1, 1, 1, 1, torch.device("cpu"))
    assert ids[2, 0].item() == 0, "the canvas must come before the reference"
    assert ids[3, 0].item() == 1


# ---------------------------------------------------------------- geometry

@pytest.mark.parametrize("size,longest,multiple,expect", [
    ((1024, 1024), 1024, 16, (1024, 1024)),
    ((2048, 1024), 1024, 16, (512, 1024)),       # landscape, scaled by longest
    ((1000, 1000), 512, 16, (512, 512)),
    ((100, 50), 1024, 16, (512, 1024)),
])
def test_a_reference_is_scaled_and_snapped_to_the_grid(size, longest, multiple,
                                                       expect):
    """Latents are patched p-square after a vae_scale_factor compression, so a
    side that does not divide by the product makes the packing reshape
    invalid."""
    class _Img:
        pass
    img = _Img()
    img.size = size
    got = K._fit_to_grid(img, longest, multiple)
    assert got == expect, f"{size} at {longest} -> {got}, expected {expect}"
    assert got[0] % multiple == 0 and got[1] % multiple == 0


def test_a_tiny_reference_never_snaps_to_zero():
    class _Img:
        size = (3, 3)
    got = K._fit_to_grid(_Img(), 8, 16)
    assert got == (16, 16), f"got {got}; a zero side cannot be packed"


def test_the_size_of_whatever_the_caller_passed():
    class _PIL:
        size = (800, 600)
    assert K._image_size(_PIL()) == (800, 600)

    class _Tensor:
        shape = (1, 3, 480, 640)
    assert K._image_size(_Tensor()) == (640, 480)

    assert K._image_size([_PIL()]) == (800, 600)
    assert K._image_size(None) == (1024, 1024)      # a usable default


# ------------------------------------------------------------- integration

def test_editing_refuses_without_an_image():
    """The one thing it cannot do is what the base pipeline already does."""
    if not hasattr(K.Krea2EditPipeline, "_edit_position_ids"):
        pytest.skip("this diffusers build has no Krea 2")
    import inspect
    params = inspect.signature(K.Krea2EditPipeline.__call__).parameters
    assert "image" in params, "the edit pipeline takes no image"
    assert params["image"].default is None
    src = inspect.getsource(K.Krea2EditPipeline.__call__)
    assert "needs an `image`" in src, "a missing image is not refused"


def test_it_is_an_instruction_edit_not_img2img():
    """`_is_instruction_edit` decides from the signature: an image and NO
    strength. A strength slider would offer control over a partial noising
    that in-context editing does not do."""
    if not hasattr(K.Krea2EditPipeline, "_edit_position_ids"):
        pytest.skip("this diffusers build has no Krea 2")
    import inspect
    params = inspect.signature(K.Krea2EditPipeline.__call__).parameters
    assert "image" in params
    assert "strength" not in params, (
        "a `strength` would make this read as img2img to _is_instruction_edit")


def test_the_reference_is_re_attached_clean_every_step():
    """Noising it would make identity decay as the canvas resolves, which is
    the failure in-context editing exists to avoid."""
    if not hasattr(K.Krea2EditPipeline, "_edit_position_ids"):
        pytest.skip("this diffusers build has no Krea 2")
    import inspect
    src = inspect.getsource(K.Krea2EditPipeline.__call__)
    assert "torch.cat([latents, ref_tokens" in src, (
        "the reference is no longer concatenated to the canvas each step")
    assert "[:, :canvas_len]" in src, (
        "the prediction must be sliced back to the canvas, or the reference's "
        "own prediction is fed to the scheduler")


def test_the_reference_is_encoded_deterministically():
    """A reference is a fixed fact about the request; sampling it would make
    the same edit of the same picture differ run to run."""
    if not hasattr(K.Krea2EditPipeline, "_encode_reference"):
        pytest.skip("this diffusers build has no Krea 2")
    import inspect
    src = inspect.getsource(K.Krea2EditPipeline._encode_reference)
    assert ".mode()" in src and ".sample(" not in src


def test_the_task_view_knows_about_krea2():
    """Without this hook an edit request reaches text-to-image and the image
    is ignored."""
    import inspect
    from codai.api import images
    src = inspect.getsource(images._task_view)
    assert "krea2_edit" in src
    assert "build_edit_view" in src


def test_a_build_without_krea2_fails_loudly_not_on_import():
    """Importing this module must never break a build that has no Krea 2."""
    assert K.Krea2EditPipeline is not None
    assert callable(K.build_edit_view)
    assert K.is_krea2_pipeline(object()) is False
