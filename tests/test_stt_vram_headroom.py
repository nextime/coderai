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

"""A resident speech model still needs room for the pass it is about to run.

Measured on a production render, 2026-10-29. Every scene logged:

    word timestamps failed (500: … CrisperWhisper worker error 500:
    CUDA out of memory. Tried to allocate 86.00 MiB …)

The speech model was loaded and idle between scenes while the TTS and visual
models of the same job loaded around it and filled the card. 86 MiB is what a
resident model needs to run; nothing was holding it, because the only
reservation in the system is taken at LOAD time and this model was already
loaded. Every caption in the render lost its word timings.

Two defects, fixed together:

* the reserve for a whisper-family model was hard-coded at 3 GB in the
  in-process path, while the isolated worker serving the same checkpoint used
  8 GB and said in its own comment that 3 GB "is about a third of what a
  verbatim pass actually peaks at". Eviction freed a third of the room needed;
* nothing re-checked once a model was resident.
"""

import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from codai.api import crisperwhisper_worker as W
from codai.api import stt_backends as S

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _src(rel):
    return io.open(os.path.join(_ROOT, rel), encoding="utf-8").read()


@pytest.fixture(autouse=True)
def _clean_measurements(monkeypatch):
    monkeypatch.setattr(S, "_activation_gb", {})
    from codai.models.manager import multi_model_manager as m
    monkeypatch.setattr(m, "_measured_vram_gb", dict(m._measured_vram_gb))
    yield


def test_a_verbatim_model_is_not_sized_at_a_third_of_its_peak():
    assert S.reserve_gb("nyrahealth/CrisperWhisper", {}, "crisperwhisper") == 8.0
    assert S.reserve_gb("nyrahealth/CrisperWhisper", {}, "whisper_hf") == 8.0
    assert S.reserve_gb("facebook/wav2vec2-base-960h", {}, "wav2vec2") == 3.0


def test_the_two_ways_of_serving_one_checkpoint_agree_about_its_size():
    """The same model runs in-process or in an isolated worker. Two different
    answers means one of the two paths always evicts the wrong amount."""
    for cfg in ({}, {"used_vram_gb": 5.5}):
        assert S.reserve_gb("nyrahealth/CrisperWhisper", cfg, "crisperwhisper") == \
            W.reserved_vram_gb("nyrahealth/CrisperWhisper", cfg)


def test_the_operator_outranks_a_measurement_and_a_measurement_the_default():
    S.record_pass("some/model", 3.0, 4.0)          # weights + per-pass
    assert S.reserve_gb("some/model", {}, "whisper_hf") == 7.0
    assert S.reserve_gb("some/model", {"used_vram_gb": 4.0}, "whisper_hf") == 4.0


def test_a_measurement_is_a_high_water_mark_not_an_average():
    """A one-word clip touches a fraction of what a 30 s window does, so an
    average under-reserves for the longer audio that follows."""
    S.record_pass("some/model", 3.0, 4.0)
    S.record_pass("some/model", 3.0, 0.2)
    assert S.reserve_gb("some/model", {}, "whisper_hf") == 7.0
    assert S.activation_gb("some/model") == pytest.approx(4.0)


def test_the_per_pass_headroom_is_measured_after_the_first_pass():
    assert S.activation_gb("fresh/model") == S._FIRST_PASS_ACTIVATION_GB
    assert S._FIRST_PASS_ACTIVATION_GB > 0, (
        "a zero floor reserves nothing, which is the bug being fixed")
    S.record_pass("fresh/model", 3.2, 4.2)
    assert S.activation_gb("fresh/model") == pytest.approx(4.2)


def test_a_negative_delta_reports_no_headroom():
    """Another model freeing memory during the pass can make the delta
    negative. Reserved as a negative number it would silently disable the
    check."""
    S.record_pass("odd/model", 9.0, -4.0)
    assert S.activation_gb("odd/model") == S._FIRST_PASS_ACTIVATION_GB
    assert S.reserve_gb("odd/model", {}, "whisper_hf") == 9.0


def test_the_pass_is_charged_its_own_delta_not_the_whole_card():
    """The absolute process peak would charge a speech model for every other
    model resident beside it -- a 16 GB image model on the same card would be
    recorded as part of its footprint, and every later load would evict the
    world."""
    src = _src("codai/api/stt_backends.py")
    body = src.split("class _Wav2Vec2Backend", 1)[1].split("class _VoskBackend", 1)[0]
    assert "memory_allocated() / 1e9" in body, "nothing records what was held"
    assert "max_memory_allocated() / 1e9 - _held" in body, (
        "the pass is measured as the process peak, not as its own delta")


# --------------------------------------------------------------------------
# The manager's side
# --------------------------------------------------------------------------

def test_room_is_only_made_when_there_is_not_enough(monkeypatch):
    from codai.models.manager import multi_model_manager as m
    calls = []
    monkeypatch.setattr(m, "_evict_models_for_vram", lambda gb: calls.append(gb))

    monkeypatch.setattr(m, "_get_free_vram_gb", lambda: 10.0)
    m.ensure_free_vram(4.0, reason="a pass")
    assert calls == [], "it evicted with room to spare"

    monkeypatch.setattr(m, "_get_free_vram_gb", lambda: 0.3)
    m.ensure_free_vram(4.0, reason="a pass")
    assert calls == [4.0]


def test_asking_for_nothing_evicts_nothing(monkeypatch):
    from codai.models.manager import multi_model_manager as m
    calls = []
    monkeypatch.setattr(m, "_evict_models_for_vram", lambda gb: calls.append(gb))
    monkeypatch.setattr(m, "_get_free_vram_gb", lambda: 0.0)
    m.ensure_free_vram(0.0)
    m.ensure_free_vram(-1.0)
    assert calls == []


# --------------------------------------------------------------------------
# The wiring
# --------------------------------------------------------------------------

def test_both_transcription_paths_reserve_before_the_pass():
    src = _src("codai/api/stt_backends.py")
    in_process = src.split("class _Wav2Vec2Backend", 1)[1].split("class _VoskBackend", 1)[0]
    remote = src.split("class _RemoteCrisperWhisperBackend", 1)[1]
    for name, body in (("in-process", in_process), ("worker", remote)):
        assert "ensure_free_vram" in body, (
            f"the {name} path runs a pass without asking for room")
        assert "activation_gb(" in body, (
            f"the {name} path does not use the measured per-pass headroom")


def test_an_oom_mid_pass_is_retried_once_after_freeing():
    """A pass is seconds of work on audio that already exists. Losing the
    render's captions to save that retry is the worse trade."""
    src = _src("codai/api/stt_backends.py")
    for marker in ("out of memory", "_evict_models_for_vram"):
        assert src.count(marker) >= 2, (
            f"{marker!r} appears {src.count(marker)}x — both the in-process and "
            f"the worker path need the retry")


def test_the_entry_point_no_longer_carries_its_own_number():
    src = _src("codai/api/transcriptions.py")
    assert "stt_backends.reserve_gb(" in src
    assert "3.0 if family == 'whisper_hf'" not in src, (
        "the hard-coded third-of-the-peak reserve is back")


def test_the_measurement_is_fed_back_from_both_paths():
    src = _src("codai/api/stt_backends.py")
    assert "record_pass(" in src, "an in-process pass measures nothing"
    assert "crisperwhisper_worker.record_measurement(" in src, (
        "the worker's own peak is not recorded")
