"""A sibling's eviction must not leave this pipeline's weights on the CPU.

Reported 2026-11-01 from a Motus run, stopped at 188/576. The first
Qwen-Image-Edit-2511 edit failed after 253 s and then EVERY later edit failed
in under a second -- 2511-NSFW, Rapid-AIO, Rapid-AIO-v23, edit and combine
alike -- with

    Input type (CUDABFloat16Type) and weight type (CPUBFloat16Type) should be
    the same          (autoencoder_kl_qwenimage.py, encode)

until the server was restarted. Sub-second failure means no load happened: a
CACHED pipeline was reused and failed immediately.

Those four cards share one weights file, so they share their component
MODULES, and eviction moves a pipeline with ``obj.to('cpu')``. Unloading one
sibling therefore moved the shared VAE and transformer off the card while the
others were still registered as loaded, and nothing moved them back -- which
is why the breakage outlived the request that caused it.

Two defences: eviction protects every key sharing a component (prevention),
and the device is checked before the call (cure, so no restart is needed).
"""
import inspect

import torch
import torch.nn as nn

import codai.api.images as images


class _Pipe:
    def __init__(self, **comps):
        self._comps = comps
        for k, v in comps.items():
            setattr(self, k, v)

    @property
    def components(self):
        return dict(self._comps)


def test_components_are_found_through_the_components_mapping():
    vae = nn.Linear(2, 2)
    p = _Pipe(vae=vae, transformer=nn.Linear(2, 2))
    got = images._components_of(p)
    assert got["vae"] is vae and "transformer" in got


def test_components_are_found_by_attribute_when_there_is_no_mapping():
    class _Bare:
        def __init__(self):
            self.vae = nn.Linear(2, 2)
    got = images._components_of(_Bare())
    assert "vae" in got


def test_a_cpu_component_is_brought_back():
    """Recorded rather than really moved, so it runs without a GPU."""
    class _Recording(nn.Linear):
        def __init__(self):
            super().__init__(2, 2)
            self.sent_to = None

        def to(self, *a, **kw):
            self.sent_to = a[0] if a else kw.get("device")
            return self

    vae = _Recording()
    p = _Pipe(vae=vae)
    moved = images._restore_pipeline_device(p, "cuda:0")
    assert moved == ["vae"]
    assert vae.sent_to == "cuda:0"


def test_a_component_already_on_the_target_is_not_touched():
    class _Recording(nn.Linear):
        def __init__(self):
            super().__init__(2, 2)
            self.sent_to = None

        def to(self, *a, **kw):
            self.sent_to = a[0] if a else kw.get("device")
            return self

    vae = _Recording()
    p = _Pipe(vae=vae)
    # Its parameters report cpu, and cpu is what was asked for.
    assert images._restore_pipeline_device(p, "cpu") == []
    assert vae.sent_to is None, "a no-op must not move anything"


def test_an_offloaded_component_is_left_to_its_hook():
    """Living on the CPU between calls is what an accelerate hook is for."""
    vae = nn.Linear(2, 2)
    vae._hf_hook = object()
    p = _Pipe(vae=vae)
    assert images._restore_pipeline_device(p, "cuda:0") == []


def test_a_component_that_cannot_move_does_not_raise():
    class _Stubborn(nn.Linear):
        def to(self, *a, **kw):
            raise RuntimeError("nope")
    p = _Pipe(vae=_Stubborn(2, 2))
    assert images._restore_pipeline_device(p, "cuda:0") == []


def test_a_component_without_parameters_is_skipped():
    class _NoParams:
        pass
    p = _Pipe(feature_extractor=_NoParams())
    assert images._restore_pipeline_device(p, "cuda:0") == []


def test_the_edit_path_checks_the_device_before_calling():
    src = inspect.getsource(images.create_image_edit)
    i_fix = src.index("_restore_pipeline_device")
    i_call = src.index("await asyncio.to_thread(pipe,")
    assert i_fix < i_call, "checking after the call cures nothing"


def test_eviction_protects_keys_that_share_a_component():
    src = inspect.getsource(images.create_image_edit)
    seg = src[src.index("_evict_headroom_gb"):]
    assert "_components_of(pipe)" in seg, (
        "protecting only the same OBJECT leaves a sibling free to move the "
        "modules this pipeline shares with it"
    )
    assert "id(c) in _mine" in seg
