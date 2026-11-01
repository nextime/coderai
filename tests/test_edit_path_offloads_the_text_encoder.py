"""An edit must hand back the text encoder's VRAM, like generation does.

Measured 2026-11-01 on zeiss. Krea-2-RealDream generates fine -- 17.22 GB of
weights, and its generations log

    [text-encoder-offload] prompt encoded, encoder moved off the GPU
      — 3.0 GiB back for the denoiser

-- and editing with the same model died with 23.50 GiB of a 23.56 GiB card in
use: "Tried to allocate 48.00 MiB ... 46.00 MiB is free", with nothing else
resident to evict. Its card asks for the offload (`sequential_text_encoder:
true`); only the generation path did it.

An edit needs it more than a generation does: the in-context reference appends
its tokens to the canvas's, so attention activations are larger for the same
weights, while the encoder is just as dead after the first step.
"""
import inspect

import codai.api.images as images


def _edit_src():
    return inspect.getsource(images.create_image_edit)


def test_the_edit_path_offloads_when_the_card_asks():
    src = _edit_src()
    assert "sequential_text_encoder" in src, (
        "the edit path must honour the same per-model flag as generation"
    )
    assert "encode_then_free" in src


def test_it_is_opt_in_per_model():
    """Freeing it costs a host-RAM copy, so it is not done unasked."""
    src = _edit_src()
    i = src.index("sequential_text_encoder")
    assert "if bool(" in src[max(0, i - 120):i], (
        "guarded by the card's flag, not unconditional"
    )


def test_the_offload_happens_before_the_pipeline_call():
    src = _edit_src()
    i_off = src.index("encode_then_free")
    i_call = src.index("await asyncio.to_thread(pipe,")
    assert i_off < i_call, (
        "offloading after the call would free nothing the denoiser could use"
    )


def test_the_kwargs_are_built_once_and_reused():
    """encode_then_free rewrites the kwargs, so the call must use its result."""
    src = _edit_src()
    assert "_call_kwargs, _ = await asyncio.to_thread(" in src
    assert "await asyncio.to_thread(pipe, **_call_kwargs)" in src


def test_a_failed_offload_does_not_fail_the_edit():
    src = _edit_src()
    i = src.index("encode_then_free")
    seg = src[i:i + 500]
    assert "except Exception" in seg, (
        "an offload that cannot run is a missed saving, not a failed request"
    )


def test_both_paths_use_the_same_helper():
    """So the two cannot drift into offloading differently."""
    src = inspect.getsource(images)
    assert src.count(
        "from codai.models.text_encoder_offload import encode_then_free") == 2
