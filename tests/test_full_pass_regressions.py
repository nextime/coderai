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

"""Fixes from one end-to-end pass over every capability on a live install.

Each test below is a row that came back failing, reduced to the decision that
actually produced it. They are unrelated bugs with one thing in common: in every
case the server knew enough to get it right and declined to use what it knew.
"""

import io
import os

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _src(rel):
    return io.open(os.path.join(_ROOT, rel), encoding="utf-8").read()


# ---------------------------------------------------------------------------
# "PonyRealism and pornmaster don't load" -- they loaded at fp32
# ---------------------------------------------------------------------------

def test_an_image_card_without_a_precision_does_not_get_f32():
    """The two that failed measured 14.98 GB; their SDXL siblings measured 7.49.
    Exactly double, because f32 was the default and only --no-ram coerced it
    down -- so a model's width in bytes came down to an unrelated flag."""
    src = _src("codai/api/images.py")
    assert "_cfg('precision', 'f32')" not in src
    assert "precision = str(_precision_cfg or '').strip() or 'bf16'" in src


def test_an_explicit_f32_is_still_honoured():
    """Changing a default must not overrule an operator who chose."""
    src = _src("codai/api/images.py")
    head = src.index("_precision_cfg = _cfg('precision')")
    assert "_cfg('precision')" in src[head:head + 200]
    # The default is applied only when the card states nothing.
    assert "if not _precision_cfg:" in src


# ---------------------------------------------------------------------------
# "LTX x3 reject the width and height"
# ---------------------------------------------------------------------------

def test_every_family_declares_a_side_multiple():
    from codai.models.video_geometry import FAMILIES
    for name, geom in FAMILIES.items():
        assert int(geom.get("side_multiple") or 0) > 0, name


def test_the_ltx_request_that_was_refused_is_now_snapped():
    """`height` and `width` have to be divisible by 32 but are 848 and 480:
    480 was already legal, 848 is 26.5 cells of 32."""
    from codai.models import video_geometry as G
    geom = G.geometry_for("Lightricks/LTX-2.5-Diffusers", {})
    assert geom["side_multiple"] == 32
    w, h, moved = G.snap_frame_size(848, 480, geom)
    assert w % 32 == 0 and h % 32 == 0
    assert h == 480          # already legal, so untouched
    assert moved             # the width moved, and the caller is told


def test_snapping_moves_a_side_by_less_than_one_cell():
    """Rounding to NEAREST, not down: the aspect ratio barely shifts."""
    from codai.models import video_geometry as G
    geom = {"side_multiple": 64}
    for value in (100, 500, 721, 1081, 1920):
        out = G.snap_side(value, geom)
        assert out % 64 == 0
        assert abs(out - value) <= 32


def test_a_legal_size_is_left_exactly_alone():
    from codai.models import video_geometry as G
    geom = {"side_multiple": 32}
    w, h, moved = G.snap_frame_size(832, 480, geom)
    assert (w, h, moved) == (832, 480, False)


def test_a_side_never_snaps_to_zero():
    """Rounding 10 pixels down to a multiple of 64 is zero, and a zero-width
    frame is a different error further in."""
    from codai.models import video_geometry as G
    geom = {"side_multiple": 64}
    assert G.snap_side(10, geom) == 64
    assert G.snap_side(0, geom) == 64
    assert G.snap_side(None, geom) == 64
    assert G.snap_side("nonsense", geom) == 64


def test_the_request_path_actually_snaps():
    """The multiple was declared for every family and read by nothing — the
    table existed and the request path never consulted it."""
    src = _src("codai/api/video.py")
    assert "snap_frame_size(" in src
    assert src.count("None, _generate_video, pipe, request, model_name") == 2


# ---------------------------------------------------------------------------
# "Kokoro English: espeak doesn't support 'en'"
# ---------------------------------------------------------------------------

def test_a_bare_en_becomes_an_espeak_language():
    """espeak-ng names English by region and refuses the bare code, while every
    OpenAI-compatible client sends 'en'. Kokoro's Italian worked and its
    English did not, which reads as a broken model."""
    from codai.api.tts_backends import espeak_language
    assert espeak_language("en") == "en-us"
    assert espeak_language("EN") == "en-us"
    assert espeak_language("en_US") == "en-us"


def test_languages_espeak_accepts_bare_are_untouched():
    """Rewriting 'it' would break the voices that were working."""
    from codai.api.tts_backends import espeak_language
    for code in ("it", "fr", "de", "es", "en-gb", "en-us"):
        assert espeak_language(code) == code


def test_an_empty_language_still_has_a_default():
    from codai.api.tts_backends import espeak_language
    assert espeak_language("") == "en-us"
    assert espeak_language(None) == "en-us"


def test_an_unknown_language_is_a_request_error_not_a_500():
    """It reached the client as RuntimeError: language "en" is not supported by
    the espeak backend — a server fault for a request problem."""
    src = _src("codai/api/tts_backends.py")
    head = src.index("class _KokoroBackend")
    body = src[head:src.index("class _TransformersBackend")]
    assert "raise ValueError(" in body
    assert "espeak" in body


# ---------------------------------------------------------------------------
# "longcat gives a 500 error" -- the pin resolved to the wrong engine
# ---------------------------------------------------------------------------

class _FakeEngine:
    def __init__(self, name, backend, healthy=True):
        self.name = name
        self.backend = backend
        self.healthy = healthy


@pytest.fixture
def registry_with_a_split_engine():
    """The GGUF-isolation split: two engines, one card, the same backend."""
    from codai.frontproxy import registry as R
    reg = R.EngineRegistry() if hasattr(R, "EngineRegistry") else None
    if reg is None:                                  # pragma: no cover
        pytest.skip("registry class not found")
    return reg


def test_an_exact_engine_name_beats_a_backend_match(registry_with_a_split_engine):
    """Both engines on a split declare backend 'nvidia'; only one is NAMED it.
    A single pass accepting either returned whichever came first, so a
    transformers model pinned to 'nvidia' could land on the GGUF sibling and be
    refused for a capability the engine it named has."""
    reg = registry_with_a_split_engine
    gguf = _FakeEngine("nvidia-gguf", "nvidia")
    torch_engine = _FakeEngine("nvidia", "nvidia")
    # GGUF first in iteration order: the order that produced the bug.
    reg._engines = {1: gguf, 2: torch_engine}
    assert reg.by_name("nvidia") is torch_engine
    assert reg.by_name("nvidia-gguf") is gguf


def test_a_backend_match_still_works_when_nothing_bears_the_name(
        registry_with_a_split_engine):
    reg = registry_with_a_split_engine
    only = _FakeEngine("engine-0", "radeon")
    reg._engines = {1: only}
    assert reg.by_name("radeon") is only


def test_an_exact_name_wins_even_unhealthy(registry_with_a_split_engine):
    """The pin names a specific engine, and the caller checks is_alive() for
    itself — the pin path has its own sibling fallback for the down case, and it
    can only compute the sibling correctly once the name resolves correctly."""
    reg = registry_with_a_split_engine
    down = _FakeEngine("nvidia", "nvidia", healthy=False)
    up = _FakeEngine("nvidia-gguf", "nvidia", healthy=True)
    reg._engines = {1: up, 2: down}
    assert reg.by_name("nvidia") is down
