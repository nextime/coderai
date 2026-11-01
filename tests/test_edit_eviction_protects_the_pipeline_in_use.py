"""The headroom pass must not evict the pipeline the edit is about to call.

Measured 2026-11-01 on zeiss, and caused by the edit-headroom block added
one commit earlier:

    Edit needs about 1.0 GB of working room and the card has less —
    evicting idle models
    On-request VRAM eviction: unloading 'image:Krea-2-Edit' (estimated
    10.02 GB) to free VRAM
      _evict_one → obj.to('cpu')
    Image editing failed: Input type (CUDABFloat16Type) and weight type
    (CPUBFloat16Type) should be the same

The edit was on Krea-2-RealDream and `protect` held that key, but the
pipeline object it was given is registered under 'image:Krea-2-Edit' -- the
sibling on the same checkpoint. Protecting a NAME cannot cover an object
reachable under several of them, so the pass moved the weights in use to host
RAM and the call ran with its input on the GPU.

The same block also ran a multi-gigabyte `.to('cpu')` inline on the event
loop, which tripped the stall watchdog every 30 s for the length of it.
"""
import inspect

import codai.api.images as images
from codai.models.manager import MultiModelManager


def _mgr(models):
    m = MultiModelManager.__new__(MultiModelManager)
    m.config = {}
    m._measured_vram_gb = {}
    m.models = dict(models)
    return m


def test_protect_accepts_a_set_of_keys():
    sig = inspect.signature(MultiModelManager._evict_models_for_vram)
    assert sig.parameters["protect"].default == "", "still defaults to no protection"
    src = inspect.getsource(MultiModelManager._evict_models_for_vram)
    assert "isinstance(protect, str)" in src, (
        "protect must take one key or many; one pipeline can be reachable "
        "under several keys"
    )
    assert "key in _protected" in src, (
        "the skip must be a membership test, not an equality that misses the "
        "other spellings"
    )
    assert "self.active_in_vram not in _protected" in src


def test_a_single_key_still_reaches_the_headroom_lookup():
    """_evict_headroom_gb reads a per-model config, so one key must survive."""
    src = inspect.getsource(MultiModelManager._evict_models_for_vram)
    assert "_protect_one" in src
    assert "for_key or _protect_one" in src


def test_the_edit_path_protects_by_identity():
    src = inspect.getsource(images.create_image_edit)
    seg = src[src.index("_evict_headroom_gb"):]
    assert "if v is pipe" in seg, (
        "the keys holding THIS pipeline object must be protected, whatever "
        "they are called"
    )
    assert "_protect" in seg


def test_the_edit_eviction_runs_off_the_event_loop():
    src = inspect.getsource(images.create_image_edit)
    seg = src[src.index("_evict_headroom_gb"):]
    i_evict = seg.index("_evict_models_for_vram")
    assert "asyncio.to_thread" in seg[:i_evict], (
        "moving pipelines to host RAM takes tens of seconds; inline it "
        "blocks every other request"
    )
