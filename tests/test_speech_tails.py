"""Two things a synthesised clip must not end with: a spoken full stop, and
silence.

Measured 2026-11-03:

* **XTTS-v2 (Claribel Dervla), Italian** -- "…in centro." came back spoken as
  "in centro punto". Transcribing the audio heard the word, so this is XTTS's
  own Italian text normaliser expanding the symbol, not a mishearing.
* **bark, Spanish** -- "background noise and a long silence at the end"; XTTS
  and bark both pad the tail. A clip whose last second is silence is a clip
  whose duration lies: compose lines scenes up by that duration, so the pad
  becomes a gap in the middle of a conversation.
"""
import numpy as np
import pytest

from codai.api import tts_backends as T


# ------------------------------------------------------- the spoken full stop

def test_a_trailing_full_stop_is_not_handed_to_xtts():
    assert T.xtts_normalize_text("Il bar è in centro.") == "Il bar è in centro"


def test_several_and_whitespace_too():
    assert T.xtts_normalize_text("Fine...  ") == "Fine"
    assert T.xtts_normalize_text("終わり。") == "終わり"


def test_intonation_marks_are_left_alone():
    """"!" and "?" change the delivery; "." at the end of an utterance does
    not, which is the whole reason this is safe."""
    assert T.xtts_normalize_text("Davvero?") == "Davvero?"
    assert T.xtts_normalize_text("Attento!") == "Attento!"


def test_punctuation_inside_the_sentence_is_untouched():
    """XTTS phrases on it; stripping it would change the prosody of every
    sentence to fix the last character of one."""
    text = "Prima, poi... e infine la fine."
    assert T.xtts_normalize_text(text) == "Prima, poi... e infine la fine"


def test_text_that_is_only_punctuation_is_passed_through():
    """Better the engine's own handling than an empty request."""
    assert T.xtts_normalize_text("...") == "..."


def test_the_backend_uses_it():
    import inspect
    src = inspect.getsource(T._CoquiBackend.synthesize)
    assert "xtts_normalize_text(text)" in src


# ------------------------------------------------------------ the silent tail

def _clip(sr=24000, speech_s=1.0, tail_s=1.5):
    speech = np.sin(np.linspace(0, 400, int(sr * speech_s))).astype("float32")
    return np.concatenate([speech, np.zeros(int(sr * tail_s), dtype="float32")]), sr


def test_trailing_silence_is_cut_back_to_a_margin(monkeypatch):
    monkeypatch.delenv("CODERAI_TTS_TRIM_SILENCE", raising=False)
    arr, sr = _clip()
    got = T.trim_trailing_silence(arr, sr)
    assert len(got) == pytest.approx(int(sr * 1.0) + int(sr * 0.08), abs=2)


def test_a_margin_is_left_so_a_fading_consonant_survives(monkeypatch):
    monkeypatch.delenv("CODERAI_TTS_TRIM_SILENCE", raising=False)
    arr, sr = _clip(tail_s=0.02)
    assert len(T.trim_trailing_silence(arr, sr)) == len(arr), (
        "a clip that already ends tight must come back untouched")


def test_only_the_tail_is_touched(monkeypatch):
    """A pause in the middle of the sentence is the speech, not padding."""
    monkeypatch.delenv("CODERAI_TTS_TRIM_SILENCE", raising=False)
    sr = 24000
    tone = np.sin(np.linspace(0, 400, sr)).astype("float32")
    arr = np.concatenate([tone, np.zeros(sr, dtype="float32"), tone])
    got = T.trim_trailing_silence(arr, sr)
    assert len(got) == pytest.approx(len(arr) + int(sr * 0.08), abs=sr * 0.09)


def test_a_silent_clip_is_not_ours_to_judge(monkeypatch):
    monkeypatch.delenv("CODERAI_TTS_TRIM_SILENCE", raising=False)
    arr = np.zeros(1000, dtype="float32")
    assert len(T.trim_trailing_silence(arr, 24000)) == 1000


def test_an_operator_can_keep_what_the_engine_produced(monkeypatch):
    monkeypatch.setenv("CODERAI_TTS_TRIM_SILENCE", "0")
    arr, sr = _clip()
    assert len(T.trim_trailing_silence(arr, sr)) == len(arr)


def test_stereo_is_trimmed_on_the_time_axis(monkeypatch):
    monkeypatch.delenv("CODERAI_TTS_TRIM_SILENCE", raising=False)
    arr, sr = _clip()
    st = np.stack([arr, arr], axis=1)
    got = T.trim_trailing_silence(st, sr)
    assert got.ndim == 2 and got.shape[1] == 2
    assert got.shape[0] < st.shape[0]


def test_every_synthesised_clip_goes_through_it():
    import inspect
    src = inspect.getsource(T.encode_audio)
    assert "trim_trailing_silence(samples, sample_rate)" in src
