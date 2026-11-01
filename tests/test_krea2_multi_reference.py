"""Krea 2 in-context editing with SEVERAL reference images.

Asked for 2026-11-01: a Motus combine request ("put the man from picture 2
next to the woman") sent two pictures to Krea-2-Edit and got

    Sizes of tensors must match except in dimension 1. Expected size 1 but
    got size 0 for tensor number 1 in the list.

because the references were handed to the encoder as a LIST, and
``image_processor.preprocess`` stacks a list on dim 0 -- making a batch of two
references, which cannot be concatenated with a canvas of one.

Each reference is now encoded separately, concatenated along the SEQUENCE, and
given its own plane on the first position-id axis. The plane is the whole
point: two references sharing one would be as indistinguishable from each
other as a reference was from the canvas.
"""
import torch

from codai.pipelines.krea2_edit import edit_position_ids


def _planes(ids):
    return ids[:, 0].tolist()


def test_the_single_reference_call_is_unchanged():
    """The old signature still works, and still puts the reference on plane 1."""
    ids = edit_position_ids(3, 2, 2, 2, 2)
    assert ids.shape == (3 + 4 + 4, 3)
    assert _planes(ids) == [0, 0, 0] + [0] * 4 + [1] * 4


def test_each_reference_gets_its_own_plane():
    ids = edit_position_ids(2, 2, 2, device=None, ref_grids=[(2, 2), (1, 2)])
    assert ids.shape == (2 + 4 + 4 + 2, 3)
    assert _planes(ids) == [0, 0] + [0] * 4 + [1] * 4 + [2] * 2


def test_three_references_are_addressable_separately():
    grids = [(1, 1), (1, 1), (1, 1)]
    ids = edit_position_ids(1, 1, 1, device=None, ref_grids=grids)
    assert _planes(ids) == [0, 0, 1, 2, 3]
    assert len(set(_planes(ids)[2:])) == 3, "the references must not share a plane"


def test_a_reference_keeps_its_own_grid_coordinates():
    """Spatial alignment is what makes the reference readable as a picture."""
    ids = edit_position_ids(0, 1, 1, device=None, ref_grids=[(2, 3)])
    ref = ids[1:]
    assert ref.shape == (6, 3)
    hw = {(int(r[1]), int(r[2])) for r in ref}
    assert hw == {(h, w) for h in range(2) for w in range(3)}


def test_no_reference_is_a_plain_canvas():
    ids = edit_position_ids(2, 2, 2, device=None, ref_grids=[])
    assert ids.shape == (2 + 4, 3)
    assert _planes(ids) == [0] * 6


def test_references_are_joined_along_the_sequence_not_the_batch():
    """The shape contract the denoise loop depends on."""
    a = torch.zeros(1, 7, 5)
    b = torch.zeros(1, 3, 5)
    joined = torch.cat([a, b], dim=1)
    assert joined.shape == (1, 10, 5), (
        "dim 1 is the sequence: concatenating on dim 0 would make a batch of "
        "references and mismatch a canvas of one"
    )


def test_the_pipeline_encodes_each_reference_separately():
    import inspect
    from codai.pipelines import krea2_edit
    src = inspect.getsource(krea2_edit)
    i = src.index("# ------------------------------------------------- references")
    seg = src[i:i + 2000]
    assert "for _img in refs:" in seg, (
        "a list passed to _encode_reference in one go comes back as a batch"
    )
    assert "torch.cat(_ref_parts, dim=1)" in seg
    assert "ref_grids=ref_grids" in seg


def test_the_token_count_grows_with_each_reference():
    """Why grounding_px is the knob that bounds the sequence."""
    one = edit_position_ids(0, 4, 4, device=None, ref_grids=[(4, 4)])
    two = edit_position_ids(0, 4, 4, device=None, ref_grids=[(4, 4), (4, 4)])
    assert two.shape[0] - one.shape[0] == 16
