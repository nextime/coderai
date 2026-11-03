"""A Kokoro voice this install cannot phonemise is refused, not mumbled.

kokoro-onnx phonemises with espeak-ng, which has no Japanese grapheme-to-
phoneme rules and falls back to naming the characters it cannot read. Measured
2026-11-03 on Kokoro-82M / ``jf_alpha``: a five-second Japanese sentence came
back as 23 seconds of audio reading out "Chinese letter … Japanese letter …",
rated 1/5, while every other Kokoro language scored 5.

It answered 200 with a valid wav, so no client could tell. Kokoro upstream
uses misaki[ja] (pyopenjtalk) for this; until that is installed the honest
answer names the engines that do declare Japanese.
"""
import pytest

from codai.api import tts_backends as T


class _FakeKokoro:
    def __init__(self):
        self.calls = []

    def create(self, text, voice, speed, lang):
        self.calls.append((text, voice, speed, lang))
        import numpy as np
        return np.zeros(8, dtype="float32"), 24000

    def get_voices(self):
        return ["af_sarah", "jf_alpha"]


def _backend():
    be = T._KokoroBackend.__new__(T._KokoroBackend)
    be._kokoro = _FakeKokoro()
    return be


def test_a_japanese_request_is_refused_with_somewhere_to_go(monkeypatch):
    monkeypatch.delenv("CODERAI_KOKORO_ALLOW_JA", raising=False)
    be = _backend()
    with pytest.raises(ValueError) as err:
        be.synthesize("これはテストです", "jf_alpha", 1.0, "ja")
    msg = str(err.value)
    assert "misaki" in msg or "pyopenjtalk" in msg
    assert "XTTS" in msg or "bark" in msg, "name an engine that can speak it"
    assert be._kokoro.calls == [], "nothing may be synthesised"


def test_the_voice_decides_even_when_no_language_was_sent(monkeypatch):
    """/v1/audio/speech defaults `language` to "en-us", so a Japanese voice
    asked for with no language would read Japanese with English rules."""
    monkeypatch.delenv("CODERAI_KOKORO_ALLOW_JA", raising=False)
    be = _backend()
    with pytest.raises(ValueError):
        be.synthesize("これはテストです", "jf_alpha", 1.0, "en-us")


def test_every_other_language_is_untouched(monkeypatch):
    monkeypatch.delenv("CODERAI_KOKORO_ALLOW_JA", raising=False)
    be = _backend()
    be.synthesize("buongiorno", "if_sara", 1.0, "it")
    be.synthesize("hello", "af_sarah", 1.0, "en")
    assert [c[3] for c in be._kokoro.calls] == ["it", "en-us"]


def test_an_install_with_its_own_g2p_can_opt_back_in(monkeypatch):
    monkeypatch.setenv("CODERAI_KOKORO_ALLOW_JA", "1")
    be = _backend()
    be.synthesize("これはテストです", "jf_alpha", 1.0, "ja")
    assert len(be._kokoro.calls) == 1


# ------------------------------------------------- what the catalogue says

def test_japanese_is_not_advertised_while_it_cannot_be_delivered(monkeypatch):
    monkeypatch.delenv("CODERAI_KOKORO_ALLOW_JA", raising=False)
    langs = T.family_languages("kokoro/Kokoro-82M")
    assert "ja" not in langs
    assert "it" in langs and "en-us" in langs


def test_and_is_advertised_again_once_it_can(monkeypatch):
    monkeypatch.setenv("CODERAI_KOKORO_ALLOW_JA", "1")
    assert "ja" in T.family_languages("kokoro/Kokoro-82M")


def test_a_model_card_still_overrides_the_table(monkeypatch):
    monkeypatch.delenv("CODERAI_KOKORO_ALLOW_JA", raising=False)
    got = T.family_languages("kokoro/Kokoro-82M", {"languages": ["ja", "en"]})
    assert got == ["ja", "en"], "an operator who declared it knows something"


# --------------------------------------------------------- how it is answered

def test_the_endpoint_answers_a_request_problem_as_a_request_problem():
    import inspect
    from codai.api import tts as TTS
    src = inspect.getsource(TTS)
    i = src.index("except ValueError as e:")
    assert "status_code=400" in src[i:i + 500]
