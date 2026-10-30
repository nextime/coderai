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

"""How many tokens a picture is worth to a GGUF vision model.

The failure these exist for, measured on Qwen3.8-27B with its own matching
projector loaded and llama.cpp reporting ``has vision encoder``:

* a 448x448 image added 16 tokens to the prompt (61 -> 77); the same image on
  the HF path added 156;
* two completely different pictures produced byte-identical descriptions at
  temperature 0, neither resembling either image;
* the same prompt with no image answered differently, so the pictures arrived
  and only their content did not.

llama.cpp had printed the reason on every load -- ``qwen3vl_merger`` needs at
least 1024 image tokens -- and ``mtmd_context_params.image_min_tokens`` was
never set. A 200 with a fluent wrong answer is the worst outcome available, so
these tests are about the floor being applied, and about not applying it where
it is not needed.
"""

import io
import os

import pytest

from codai.backends import mtmd_vision as M

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _src(rel):
    return io.open(os.path.join(_ROOT, rel), encoding="utf-8").read()


# ---------------------------------------------------------------------------
# the floor itself
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("declared", [
    "qwen3vl_merger", "QWEN3VL_MERGER", "qwen3vl-merger", "qwen2vl_merger",
])
def test_a_qwen_vl_merger_gets_the_floor_llama_cpp_asks_for(monkeypatch, declared):
    monkeypatch.setattr(M, "projector_type", lambda p: declared.lower())
    tokens, why = M.min_image_tokens("/any/mmproj.gguf")
    assert tokens == 1024
    assert "image tokens" in why


def test_a_projector_with_no_documented_floor_is_left_alone(monkeypatch):
    """Forcing an upscale costs prefill time on every request — a projector
    happy with what the image gives it must not pay for this."""
    monkeypatch.setattr(M, "projector_type", lambda p: "mlp")
    assert M.min_image_tokens("/any/mmproj.gguf") == (0, "")


def test_an_unreadable_projector_behaves_exactly_as_before(monkeypatch):
    monkeypatch.setattr(M, "projector_type", lambda p: "")
    assert M.min_image_tokens("/any/mmproj.gguf") == (0, "")


def test_the_card_always_wins(monkeypatch):
    monkeypatch.setattr(M, "projector_type", lambda p: "qwen3vl_merger")
    tokens, why = M.min_image_tokens("/any/mmproj.gguf", 2048)
    assert tokens == 2048
    assert "card" in why


def test_an_explicit_zero_on_the_card_restores_the_old_behaviour(monkeypatch):
    """An operator who would rather have the previous behaviour keeps it, and
    0 must not be read as "unset" — which is the usual way a deliberate zero
    gets ignored."""
    monkeypatch.setattr(M, "projector_type", lambda p: "qwen3vl_merger")
    assert M.min_image_tokens("/any/mmproj.gguf", 0)[0] == 0


def test_nonsense_on_the_card_does_not_raise(monkeypatch):
    monkeypatch.setattr(M, "projector_type", lambda p: "qwen3vl_merger")
    assert M.min_image_tokens("/any/mmproj.gguf", "lots")[0] == 0
    assert M.min_image_tokens("/any/mmproj.gguf", -5)[0] == 0


def test_the_projector_type_is_read_from_the_file_not_the_name():
    """An mmproj is routinely renamed to sit beside the model it serves, and
    the base model's name says nothing about which merger was exported."""
    assert M.projector_type("/does/not/exist.gguf") == ""
    src = _src("codai/backends/mtmd_vision.py")
    assert "clip.projector_type" in src
    assert "GGUFReader" in src


# ---------------------------------------------------------------------------
# the handler
# ---------------------------------------------------------------------------

def test_no_floor_means_the_plain_upstream_handler(monkeypatch):
    """Nothing of ours in the way for the ordinary case."""
    built = {}

    class _Fake:
        def __init__(self, **kw):
            built.update(kw)

    import sys
    import types
    mod = types.ModuleType("llama_cpp.llama_chat_format")
    mod.MTMDChatHandler = _Fake
    pkg = types.ModuleType("llama_cpp")
    pkg.llama_chat_format = mod
    monkeypatch.setitem(sys.modules, "llama_cpp", pkg)
    monkeypatch.setitem(sys.modules, "llama_cpp.llama_chat_format", mod)
    out = M.handler("/m.gguf", use_gpu=True, image_min_tokens=0)
    assert isinstance(out, _Fake)
    assert built == {"clip_model_path": "/m.gguf", "verbose": False,
                     "use_gpu": True}


def test_the_bounds_reach_the_params_upstream_builds(monkeypatch):
    """Upstream sets four fields on mtmd_context_params_default() and offers no
    hook in between; the token bounds are not among them. Wrapping the factory
    for the duration of that one call means the struct upstream receives
    already carries them AND everything upstream does to it still happens —
    copying the method instead would stop tracking any field upstream adds."""
    import sys
    import types

    class _Params:
        def __init__(self):
            self.use_gpu = None
            self.image_min_tokens = 0
            self.image_max_tokens = 0

    seen = {}

    class _Mtmd:
        @staticmethod
        def mtmd_context_params_default():
            return _Params()

    class _Upstream:
        def __init__(self, **kw):
            self.mtmd_ctx = None
            self._mtmd_cpp = _Mtmd()

        def _init_mtmd_context(self, llama_model):
            # Stand-in for upstream: build params, set its own field, record.
            params = self._mtmd_cpp.mtmd_context_params_default()
            params.use_gpu = True
            seen["params"] = params

    mod = types.ModuleType("llama_cpp.llama_chat_format")
    mod.MTMDChatHandler = _Upstream
    pkg = types.ModuleType("llama_cpp")
    pkg.llama_chat_format = mod
    monkeypatch.setitem(sys.modules, "llama_cpp", pkg)
    monkeypatch.setitem(sys.modules, "llama_cpp.llama_chat_format", mod)

    # Through handler(), the entry point the backend actually calls. Built via
    # _make_handler_class() directly, this test passed while the shipped path
    # raised NameError -- a probe that does not call what the code calls proves
    # nothing.
    M._HANDLER_CLASS = None
    h = M.handler("/m.gguf", use_gpu=True, image_min_tokens=1024,
                  image_max_tokens=4096)
    h._init_mtmd_context(object())
    assert seen["params"].image_min_tokens == 1024
    assert seen["params"].image_max_tokens == 4096
    # Upstream's own field survived our wrapper.
    assert seen["params"].use_gpu is True
    # And the factory is restored, so nothing else sees our values.
    assert h._mtmd_cpp.mtmd_context_params_default().image_min_tokens == 0


# ---------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------

def test_the_backend_sets_the_floor_when_it_loads_a_projector():
    src = _src("codai/backends/vulkan.py")
    head = src.index("_mmproj = kwargs.get('mmproj'")
    body = src[head:head + 2500]
    assert "mtmd_vision.min_image_tokens(" in body
    assert "mtmd_vision.handler(" in body
    # The old unconditional construction is gone.
    assert "MTMDChatHandler(\n" not in body


def test_the_floor_is_announced_in_the_log():
    """It changes how much prefill every image costs; an operator reading the
    load log should see it without having to know this file exists."""
    src = _src("codai/backends/vulkan.py")
    assert "image tokens : at least" in src


def test_the_field_is_savable_from_the_admin_page():
    assert '"image_min_tokens", "image_max_tokens",' in _src("codai/admin/routes.py")


def test_the_floor_handler_is_reachable_the_way_the_backend_reaches_it(monkeypatch):
    """The regression this file shipped once: the class was named as a global
    inside this module and built by a module-level __getattr__, which PEP 562
    consults only for attribute access from OUTSIDE the module. So handler()
    raised NameError, the except around the mmproj block reported "failed to
    load projector (name 'ImageTokenFloorHandler' is not defined)", and the
    model came up text-only -- refusing images instead of answering them
    wrongly, which is better, and still broken."""
    import sys
    import types

    class _Upstream:
        def __init__(self, **kw):
            self.kw = kw
            self.mtmd_ctx = None

    mod = types.ModuleType("llama_cpp.llama_chat_format")
    mod.MTMDChatHandler = _Upstream
    pkg = types.ModuleType("llama_cpp")
    pkg.llama_chat_format = mod
    monkeypatch.setitem(sys.modules, "llama_cpp", pkg)
    monkeypatch.setitem(sys.modules, "llama_cpp.llama_chat_format", mod)
    monkeypatch.setattr(M, "_HANDLER_CLASS", None)
    out = M.handler("/m.gguf", use_gpu=True, image_min_tokens=1024)
    assert isinstance(out, _Upstream)
    assert out.kw["clip_model_path"] == "/m.gguf"
