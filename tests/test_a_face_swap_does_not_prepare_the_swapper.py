"""`prepare()` belongs to FaceAnalysis, not to INSwapper.

insightface's detection app is prepared with a context id and a detection size;
the swapper is not. INSwapper builds its ONNX session in `__init__` and is then
used straight through `get(img, target_face, source_face, paste_back=True)`.

Measured 2026-11-03, 34 s into a request, with one reference face:

    500 Face swap failed: 'INSwapper' object has no attribute 'prepare'

and it surfaced only once the download finally worked -- the earlier 503
("Failed to download inswapper model") had been hiding it. Other model_zoo
classes DO have `prepare`, so it is called where it exists rather than dropped.
"""
import pytest

from codai.api import faceswap as F


class _INSwapper:
    """No `prepare` -- insightface's real one has none either."""

    def get(self, img, target_face, source_face, paste_back=True):
        return img


class _WithPrepare(_INSwapper):
    def __init__(self):
        self.prepared = None

    def prepare(self, ctx_id=0):
        self.prepared = ctx_id


class _PrepareExplodes(_INSwapper):
    def prepare(self, ctx_id=0):
        raise RuntimeError("no CUDA here")


def _install(monkeypatch, obj):
    monkeypatch.setattr(F, "_swapper", None, raising=False)
    monkeypatch.setattr(F, "_ensure_model", lambda: None)
    import sys
    import types
    mod = types.ModuleType("insightface.model_zoo")
    mod.get_model = lambda path, download=False: obj
    pkg = types.ModuleType("insightface")
    pkg.model_zoo = mod
    monkeypatch.setitem(sys.modules, "insightface", pkg)
    monkeypatch.setitem(sys.modules, "insightface.model_zoo", mod)


def test_a_swapper_without_prepare_loads(monkeypatch):
    obj = _INSwapper()
    _install(monkeypatch, obj)
    assert F._get_swapper() is obj


def test_a_swapper_with_prepare_still_gets_it(monkeypatch):
    obj = _WithPrepare()
    _install(monkeypatch, obj)
    assert F._get_swapper() is obj
    assert obj.prepared == 0


def test_a_failing_prepare_does_not_fail_the_swap(monkeypatch):
    """The session is already built; a failed prepare is not a dead swapper."""
    obj = _PrepareExplodes()
    _install(monkeypatch, obj)
    assert F._get_swapper() is obj


def test_the_swapper_is_cached(monkeypatch):
    obj = _INSwapper()
    _install(monkeypatch, obj)
    calls = []
    import sys
    sys.modules["insightface.model_zoo"].get_model = (
        lambda path, download=False: calls.append(1) or obj)
    F._get_swapper()
    F._get_swapper()
    assert len(calls) == 1
