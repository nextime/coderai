"""Eviction must not free what another loaded model is still using.

Measured 2026-11-01 across the four Qwen-Image-Edit cards, which share one
weights file and therefore the same component MODULES. Freeing a pipeline
walks ``pipe.components`` and strips their tensors, so unloading one sibling
took the VAE and transformer of the others with it while they were still
registered as loaded:

    Input type (CUDABFloat16Type) and weight type (CPUBFloat16Type) should be
    the same          (autoencoder_kl_qwenimage.py, encode)

and it stayed that way until the server was restarted.

The edit path's own headroom pass was taught to protect sharers first, but the
LOAD path evicts too, and a later run showed the cure firing for real --
"brought vae, transformer back to cuda:0" -- which is prevention missing a
path. This is the fix at the layer every eviction goes through.
"""
from codai.models.manager import MultiModelManager


class _Comp:
    def __init__(self, tag):
        self.tag = tag


class _Pipe:
    def __init__(self, **comps):
        self._comps = comps
        for k, v in comps.items():
            setattr(self, k, v)

    @property
    def components(self):
        return {k: getattr(self, k) for k in self._comps}


def _mgr(models):
    m = MultiModelManager.__new__(MultiModelManager)
    m.models = dict(models)
    return m


def test_components_held_by_another_model_are_detached_not_freed():
    vae, tr = _Comp("vae"), _Comp("tr")
    a = _Pipe(vae=vae, transformer=tr)
    b = _Pipe(vae=vae, transformer=tr)          # the sibling, same modules
    m = _mgr({"image:b": b})                    # 'a' already popped
    skip, detached = m._detach_components_shared_with_loaded("image:a", a)
    assert skip is False
    assert sorted(detached) == ["transformer", "vae"]
    assert a.vae is None and a.transformer is None, (
        "detached so the free routine cannot reach them"
    )
    assert b.vae is vae and b.transformer is tr, "the sibling keeps its own"


def test_unshared_components_are_left_attached_to_be_freed():
    a = _Pipe(vae=_Comp("va"), transformer=_Comp("ta"))
    b = _Pipe(vae=_Comp("vb"), transformer=_Comp("tb"))
    m = _mgr({"image:b": b})
    skip, detached = m._detach_components_shared_with_loaded("image:a", a)
    assert skip is False and detached == []
    assert a.vae is not None, "nothing else holds it, so it must still be freed"


def test_the_same_object_under_two_keys_frees_nothing():
    shared = _Pipe(vae=_Comp("v"))
    m = _mgr({"image:other": shared})
    skip, detached = m._detach_components_shared_with_loaded("image:a", shared)
    assert skip is True, (
        "another key holds this very pipeline; dropping the entry is the whole "
        "of the eviction"
    )
    assert detached == []
    assert shared.vae is not None


def test_no_other_models_means_a_normal_full_free():
    a = _Pipe(vae=_Comp("v"))
    m = _mgr({})
    assert m._detach_components_shared_with_loaded("image:a", a) == (False, [])
    assert a.vae is not None


def test_a_non_pipeline_object_is_not_disturbed():
    class _Plain:
        pass
    obj = _Plain()
    m = _mgr({"text:x": _Plain()})
    assert m._detach_components_shared_with_loaded("text:y", obj) == (False, [])


def test_evict_one_consults_it_before_freeing():
    import inspect
    src = inspect.getsource(MultiModelManager._evict_one)
    i_check = src.index("_detach_components_shared_with_loaded")
    i_free = src.index("_free_pipeline_vram")
    assert i_check < i_free, "deciding after the free is too late"


def test_evict_one_skips_the_free_entirely_when_told_to():
    import inspect
    src = inspect.getsource(MultiModelManager._evict_one)
    assert "_skip_free" in src
    i = src.index("if _skip_free:")
    assert "model_obj = None" in src[i:i + 400], (
        "the free must be bypassed, not merely logged"
    )
