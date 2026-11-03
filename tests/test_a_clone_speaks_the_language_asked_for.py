"""`auto` must not hand an Italian request to an English-only engine.

F5-TTS's ``infer`` takes no language argument: the checkpoint is the whole of
what it can pronounce (F5TTS_v1_Base is Emilia, English and Chinese). The old
`auto` rule compared the target language with the REFERENCE's language only, so
an Italian reference WITH its transcript -- the case F5 is best at, and the one
the docs recommend a transcript for -- chose F5, which then read Italian with
English phonetics.

Measured 2026-11-03: `POST /v1/audio/clone` with `language: "it"`, an Italian
Kokoro sample and its Italian transcript answered 200 with 3.3 s of audio that
does not sound Italian, rated 3/5 ("the clone speaks the wrong language").
"""
import pytest

from codai.api import voice_clone as VC


IT = "Questa frase è detta da una voce clonata."


def test_an_italian_target_does_not_go_to_f5():
    assert VC._pick_engine("auto", IT, "it", "it") == "xtts"


def test_english_with_a_transcript_still_goes_to_f5():
    """F5 is the better clone when it can speak the language, and a transcript
    is what it needs; this is the case that must not regress."""
    assert VC._pick_engine("auto", "This is the reference.", "en", "en") == "f5"


def test_chinese_is_in_the_checkpoint_too():
    assert VC._pick_engine("auto", "参考文本", "zh", "zh") == "f5"


def test_no_language_asked_for_is_not_a_claim_about_the_language():
    """A caller who names no language gets the old behaviour: F5 with a
    transcript. Refusing here would break every request that never had to say
    what language it was in."""
    assert VC._pick_engine("auto", "reference text", "", "") == "f5"


def test_a_missing_transcript_still_decides_first():
    assert VC._pick_engine("auto", "", "en", "en") == "xtts"


def test_a_target_the_reference_does_not_speak_still_goes_to_xtts():
    assert VC._pick_engine("auto", "reference", "de", "en") == "xtts"


def test_an_explicit_engine_is_always_honoured():
    """Including one that cannot speak the language: the caller said f5, and
    silently substituting another engine would be worse than a note."""
    assert VC._pick_engine("f5", IT, "it", "it") == "f5"
    assert VC._pick_engine("xtts", IT, "it", "it") == "xtts"


def test_an_unknown_engine_is_still_refused():
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as err:
        VC._pick_engine("whatever", IT, "it", "it")
    assert err.value.status_code == 400


# ------------------------------------------------- the operator's override

def test_a_multilingual_fine_tune_can_be_declared(monkeypatch):
    """An F5 fine-tune that does speak Italian is a config fact, not a code
    change: CODERAI_F5_LANGUAGES says what the installed checkpoint covers."""
    monkeypatch.setattr(VC, "F5_LANGUAGES", ("en", "zh", "it"))
    assert VC._pick_engine("auto", IT, "it", "it") == "f5"


def test_the_language_check_reads_only_the_first_two_letters():
    assert VC._f5_can_speak("en-US") and VC._f5_can_speak("EN")
    assert not VC._f5_can_speak("it-IT")


# ------------------------------------------------------ what the answer says

def test_the_answer_reports_the_engine_and_warns_when_it_cannot_pronounce():
    import inspect
    src = inspect.getsource(VC)
    i = src.index('if engine == "f5" and not _f5_can_speak(request.language)')
    assert "notes.append" in src[i:i + 400]
    j = src.index('"engine": engine')
    assert "warnings" in src[j:j + 200], (
        "a request that was served with the wrong phonetics must say so")
