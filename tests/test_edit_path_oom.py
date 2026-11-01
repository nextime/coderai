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
    # The call became `await asyncio.to_thread(pipe, **_call_kwargs)` on
    # 2026-11-01, when the text-encoder offload was added ahead of it.
    i_call = src.index("await asyncio.to_thread(pipe,")
    assert i_head < i_call, "the room must be made BEFORE the call, not after"


def test_it_asks_only_for_headroom_not_for_the_weights():
    """The weights are already resident; asking for them would evict for nothing."""
    src = _edit_source()
    seg = src[src.index("_evict_headroom_gb"):]
    i = seg.index("_evict_models_for_vram")
    # The call now goes through asyncio.to_thread, so the amount is the first
    # argument AFTER the bound method rather than inside the parentheses.
    assert "_head," in seg[i:i + 200], seg[i:i + 200]


def test_it_protects_the_model_it_is_making_room_for():
    src = _edit_source()
    seg = src[src.index("_evict_headroom_gb"):]
    # Protection is now a SET built by identity -- protecting the requested
    # name alone evicted the sibling key the pipeline was registered under.
    # See tests/test_edit_eviction_protects_the_pipeline_in_use.py.
    assert "_protect" in seg, (
        "without protection, the pass can evict the very pipeline the edit is "
        "about to call"
    )
    assert "if v is pipe" in seg


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
