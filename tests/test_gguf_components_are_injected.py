"""A GGUF component is pre-built, so the placement rules must treat it as one.

`build_gguf_pipeline_components` hands `from_pretrained` finished objects, the
same as the pipeline cache does. accelerate's device_map only dispatches
components it LOADS: one it is handed keeps whatever device it is on, with no
offload hook, and the forward pass then fails with

    Expected all tensors to be on the same device, but got index is on cuda:0,
    different from other tensors on cpu

video.py knew this and said so in a comment — but only the cache path set the
flag that acts on it. LTX-2.5 therefore failed under every strategy it tried
(auto, balanced, model, disk), and `video_resident_experts`, which exists for
exactly this, was never consulted.
"""
import inspect

from codai.api import video as video_mod


def _loader_source():
    src = inspect.getsource(video_mod)
    start = src.index("_have_injected_components = False")
    return src[start:start + 12000]


def test_gguf_components_set_the_injected_flag():
    src = _loader_source()
    gguf_at = src.index("Video GGUF components:")
    after = src[gguf_at:gguf_at + 900]
    assert "_have_injected_components = True" in after


def test_the_flag_still_gates_the_placement_choice():
    """The fix is only worth anything if the flag is what forces a hook-based
    or resident placement."""
    src = _loader_source()
    assert "if _have_injected_components and offload in (None, 'auto', 'balanced', 'disk'):" in src
    gate = src[src.index("if _have_injected_components and offload in"):]
    assert "video_resident_experts" in gate[:2600]
    assert "offload = 'resident'" in gate[:2600]
    assert "offload = 'model'" in gate[:3000]


def test_an_explicit_strategy_is_still_respected():
    """A card that says offload_strategy: model keeps it — the override only
    covers the strategies that let accelerate place components itself."""
    src = _loader_source()
    cond = src[src.index("if _have_injected_components and offload in"):]
    assert "'model'" not in cond[:cond.index("\n")]
    assert "'sequential'" not in cond[:cond.index("\n")]
