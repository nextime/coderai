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


# ---------------------------------------------------------------------------
# "Five models won't load" -- llama.cpp said why and the filter dropped it
# ---------------------------------------------------------------------------

def test_llama_cpp_log_lines_are_all_kept():
    """The callback printed only lines matching a keyword list about layers and
    buffers. An error naming the architecture or a missing tensor matches none
    of them, so three MTP checkpoints failed with no reason anywhere in the log
    and llama-cpp-python's one generic sentence reached the client."""
    from codai.backends import vulkan as V
    V._LOG_TAIL.clear()
    V._LOG_TAIL.extend([
        "llama_model_loader: loaded meta data",
        "load_tensors: layer 0 assigned to device CUDA0",
        "error loading model: missing tensor 'blk.0.nextn.weight'",
    ])
    tail = V.llama_log_tail()
    assert "missing tensor" in tail


def test_the_tail_prefers_lines_that_look_like_a_diagnosis():
    """A 60-line ring ending in routine chatter must still surface the error."""
    from codai.backends import vulkan as V
    V._LOG_TAIL.clear()
    V._LOG_TAIL.append("unknown model architecture: 'qwen35moe'")
    for i in range(40):
        V._LOG_TAIL.append(f"load_tensors: layer {i} assigned to device CUDA0")
    tail = V.llama_log_tail()
    assert "unknown model architecture" in tail


def test_the_whole_tail_is_available_when_nothing_looks_wrong():
    from codai.backends import vulkan as V
    V._LOG_TAIL.clear()
    V._LOG_TAIL.extend(["routine one", "routine two"])
    assert "routine two" in V.llama_log_tail()
    assert "routine two" in V.llama_log_tail(trouble_only=False)


def test_an_empty_log_yields_no_claim():
    from codai.backends import vulkan as V
    V._LOG_TAIL.clear()
    assert V.llama_log_tail() == ""


def test_the_ladder_stops_once_cpu_only_has_failed():
    """The retry ladder answers "does it fit?" by moving layers to the host. An
    attempt with no layers on the GPU has answered it — the three MTP models
    each burned six attempts down to n_gpu_layers=0, every one failing in a
    fraction of a second with the same generic message."""
    src = _src("codai/backends/vulkan.py")
    assert "not retrying: this attempt had every layer on" in src
    head = src.index("Error loading GGUF model ")
    body = src[head:head + 2000]
    assert "if _was == 0:" in body


def test_the_raised_error_carries_llama_cpps_words():
    src = _src("codai/backends/vulkan.py")
    assert "llama.cpp reported:" in src


def test_trouble_lines_are_printed_even_when_no_keyword_matches():
    src = _src("codai/backends/vulkan.py")
    head = src.index("def _install_layer_log_callback")
    body = src[head:src.index("async def _aiter_blocking")]
    assert "_LOG_TROUBLE" in body
    assert "_LOG_TAIL.append(msg)" in body


# ---------------------------------------------------------------------------
# "XTTS-v2 and parler don't load" -- both got an alias, not a repo id
# ---------------------------------------------------------------------------

def test_every_tts_backend_is_given_the_cards_path():
    """Kokoro already took `model_path or model_name`; coqui, bark, parler, melo
    and the transformers fallback took the request's name, which for a card
    registered under an alias is the alias."""
    src = _src("codai/api/tts_backends.py")
    head = src.index("def _load_backend(")
    body = src[head:src.index("# encoding", head)]
    assert "_ckpt = model_path or model_name" in body
    assert "_CoquiBackend(_ckpt" in body
    assert "_BarkBackend(_ckpt" in body
    assert "_KokoroBackend(_ckpt" in body
    assert "_TransformersBackend(_ckpt" in body
    assert "parler_worker.ensure_service(_ckpt)" in body
    assert "melotts_worker.ensure_service(_ckpt)" in body
    # And no backend is left taking the bare request name.
    assert "_CoquiBackend(model_name" not in body
    assert "_BarkBackend(model_name" not in body


def test_the_xtts_worker_recognises_an_org_less_alias():
    """'XTTS venv could not load XTTS-v2: not enough values to unpack
    (expected 4, got 1)' — coqui splits a name into four parts and the bare
    alias has no separator at all. The old test required a '/', which is
    exactly the case that does not apply."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_xtts_svc", os.path.join(_ROOT, "tools/xtts_service.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    four = "tts_models/multilingual/multi-dataset/xtts_v2"
    assert mod._coqui_name("XTTS-v2") == four
    assert mod._coqui_name("coqui/XTTS-v2") == four
    assert mod._coqui_name("xtts_v2") == four
    assert mod._coqui_name("xtts_v1.1").endswith("xtts_v1.1")
    assert mod._coqui_name("coqui/xtts-v1.1").endswith("xtts_v1.1")
    # Already coqui's own four-part naming, and an unrelated model: untouched.
    assert mod._coqui_name(four) == four
    assert mod._coqui_name("some/other-model") == "some/other-model"


def test_the_multimodal_path_is_visible_in_the_log():
    """Two different pictures produced byte-identical descriptions from a GGUF
    model with a configured mmproj, while the same prompt with no image
    answered differently — the images reached the model and their content did
    not. Nothing mtmd does was printed: projector init, marker substitution,
    bitmap decode and chunk eval all log under names the filter did not have."""
    src = _src("codai/backends/vulkan.py")
    head = src.index("_KEEP = (")
    keep = src[head:src.index("\n    )", head)]
    for word in ("mtmd", "clip", "bitmap", "media", "projector"):
        assert f"'{word}'" in keep, word


# ---------------------------------------------------------------------------
# "Plain whisper0-3 return no word timings"
# ---------------------------------------------------------------------------

def test_openai_shaped_whisper_server_segments_are_read():
    from codai.api.transcriptions import _whisper_server_timings
    segs, words = _whisper_server_timings({
        "text": "ciao mondo",
        "segments": [{"start": 0.0, "end": 1.5, "text": " ciao"},
                     {"start": 1.5, "end": 2.25, "text": " mondo"}],
    })
    assert [s["text"] for s in segs] == ["ciao", "mondo"]
    assert segs[1]["start"] == 1.5 and segs[1]["end"] == 2.25
    assert words == []


def test_whisper_cpps_own_shape_is_read_and_converted_from_ms():
    """whisper.cpp's native json names the list 'transcription' and counts in
    MILLISECONDS — read as seconds it would place a 2-second clip 2000 s in."""
    from codai.api.transcriptions import _whisper_server_timings
    segs, _ = _whisper_server_timings({
        "transcription": [
            {"offsets": {"from": 0, "to": 1500}, "text": " ciao"},
            {"offsets": {"from": 1500, "to": 2250}, "text": " mondo"},
        ]})
    assert segs[0]["end"] == 1.5
    assert segs[1]["start"] == 1.5 and segs[1]["end"] == 2.25


def test_word_timings_are_read_from_either_place():
    """A DTW build puts them at the top level or inside each segment."""
    from codai.api.transcriptions import _whisper_server_timings
    _, words = _whisper_server_timings({
        "words": [{"word": "ciao", "start": 0.0, "end": 0.4}]})
    assert words == [{"word": "ciao", "start": 0.0, "end": 0.4}]
    _, words = _whisper_server_timings({
        "segments": [{"start": 0, "end": 1, "text": "ciao",
                      "words": [{"text": "ciao",
                                 "offsets": {"from": 0, "to": 400}}]}]})
    assert words == [{"word": "ciao", "start": 0.0, "end": 0.4}]


def test_nothing_is_invented_when_the_server_sent_nothing():
    """The old behaviour returned [] for every whisper.cpp model; the new one
    must return [] only when the server really sent nothing."""
    from codai.api.transcriptions import _whisper_server_timings
    assert _whisper_server_timings({"text": "ciao"}) == ([], [])
    assert _whisper_server_timings({}) == ([], [])
    assert _whisper_server_timings({"segments": [None, "x"]}) == ([], [])


def test_a_malformed_timing_does_not_raise():
    from codai.api.transcriptions import _whisper_server_timings
    segs, words = _whisper_server_timings({
        "segments": [{"start": "nonsense", "end": None, "text": "ciao"}],
        "words": [{"word": "ciao", "start": "x", "end": "y"}]})
    assert segs[0]["start"] == 0.0
    assert words[0]["start"] == 0.0


def test_the_server_is_asked_for_the_verbose_reply():
    """It was never asked, so there was nothing to map."""
    src = _src("codai/models/manager.py")
    head = src.index("def transcribe(self, audio_data")
    body = src[head:head + 2500]
    assert 'data["response_format"] = "verbose_json"' in body


def test_the_route_passes_the_timings_on():
    src = _src("codai/api/transcriptions.py")
    assert "_whisper_server_timings(result)" in src
    assert "_finalize(response_format, result.get(\"text\", \"\"),\n" in src


def test_a_missing_word_timing_is_reported_not_silently_empty():
    src = _src("codai/api/transcriptions.py")
    assert "returned no word timings" in src
