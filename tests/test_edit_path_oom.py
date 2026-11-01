"""An edit on a resident pipeline must make room, and must report an OOM as one.

Measured 2026-11-01 on zeiss:

    [http] 500 on POST /v1/images/edits: 'Image editing failed: CUDA out of
    memory. Tried to allocate 576.00 MiB. GPU 0 has a total capacity of
    23.56 GiB of which 212.00 MiB is free.'

Two defects in one line. An edit on an already-loaded pipeline loads nothing,
so every piece of admission on the load path is skipped -- no sizing, no
headroom, no eviction -- and the call runs against whatever the last model to
load left free. And the result was a bare 500, where the identical condition
on /v1/images/generations answers 503 with an explanation: a 500 reads as a
bad request, not as a resource condition worth retrying.
"""
import inspect

import codai.api.images as images


def _edit_source():
    return inspect.getsource(images.create_image_edit)


def test_the_edit_path_asks_for_headroom_before_calling():
    src = _edit_source()
    assert "_evict_headroom_gb" in src, (
        "an edit on a resident pipeline gets no admission from the load path, "
        "so it has to ask for its own working room"
    )
    i_head = src.index("_evict_headroom_gb")
    i_call = src.index("run_in_executor")
    assert i_head < i_call, "the room must be made BEFORE the call, not after"


def test_it_asks_only_for_headroom_not_for_the_weights():
    """The weights are already resident; asking for them would evict for nothing."""
    src = _edit_source()
    assert "_evict_models_for_vram(\n                    _head" in src or \
           "_evict_models_for_vram(_head" in src


def test_it_protects_the_model_it_is_making_room_for():
    src = _edit_source()
    seg = src[src.index("_evict_models_for_vram"):]
    assert "protect=" in seg[:200], (
        "without protect, the pass can evict the very pipeline the edit is "
        "about to call"
    )


def test_an_oom_is_reported_as_503_with_an_explanation():
    src = _edit_source()
    assert "_is_cuda_oom(e)" in src
    i_oom = src.index("_is_cuda_oom(e)")
    seg = src[i_oom:i_oom + 900]
    assert "status_code=503" in seg
    assert "_exc_text(e)" in seg, (
        "the message must name the exception even when str(e) is empty"
    )


def test_other_failures_are_still_500():
    src = _edit_source()
    assert 'status_code=500, detail=f"Image editing failed: {e}"' in src
