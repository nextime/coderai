"""A TTS voice's regional tag must not break speech-to-text (A1).

A compose render picks a voice for its accent, so the language it carries is
``en-us`` / ``en-gb`` / ``pt-br``. Every speech backend wants something else,
and transformers' Whisper RAISES on a tag it doesn't know -- which arrived at
the client as ``500: CrisperWhisper worker error 500: Unsupported language:
en-us`` on every captioned scene.
"""
import pytest

from codai.api import lang_codes as L


# ---------------------------------------------------------------- the mapping
@pytest.mark.parametrize("tag,code,name", [
    ("en-us", "en", "english"),
    ("en-GB", "en", "english"),
    ("en_gb", "en", "english"),
    ("pt-br", "pt", "portuguese"),
    ("it", "it", "italian"),
    ("italian", "it", "italian"),
    ("Italian", "it", "italian"),
    ("ita", "it", "italian"),
    ("zh-Hans-CN", "zh", "chinese"),
    ("mandarin", "zh", "chinese"),
    ("fr-ca", "fr", "french"),
    ("es-419", "es", "spanish"),
])
def test_every_spelling_reaches_the_same_language(tag, code, name):
    assert L.to_code(tag) == code
    assert L.to_name(tag) == name


@pytest.mark.parametrize("family,expected", [
    ("crisperwhisper", "english"),      # transformers: the NAME
    ("whisper_hf", "english"),
    ("whisper-transformers", "english"),
    ("nemo", "en"),                     # Canary: the CODE
    ("whisper-server", "en"),           # whisper.cpp: the CODE
    ("faster_whisper", "en"),           # faster-whisper: the CODE, not the name
    ("wav2vec2", "en"),
    ("vosk", "en"),
    (None, "en"),
])
def test_each_backend_gets_the_spelling_its_api_wants(family, expected):
    assert L.normalize("en-us", family) == expected


@pytest.mark.parametrize("tag", ["klingon", "xx-YY", "auto", "", None, "  "])
def test_an_unmappable_tag_becomes_autodetect_not_an_error(tag):
    # None means "omit the argument", i.e. let the model detect. Never a raise:
    # refusing to transcribe because the caller named a dialect is strictly
    # worse than transcribing with detection.
    assert L.normalize(tag, "crisperwhisper") is None
    assert L.normalize(tag, "nemo") is None


def test_names_with_spaces_survive_the_subtag_split():
    # "haitian creole" must not be truncated at the first word, and a name
    # containing a hyphen must not be split either.
    assert L.to_code("haitian creole") == "ht"
    assert L.to_name("haitian creole") == "haitian creole"


# ------------------------------------------------------------- the validation
def test_declared_english_accepts_a_regional_tag():
    from codai.api.transcriptions import _validate_language
    # No raise: this is the exact combination that 400'd before.
    _validate_language("crisperwhisper", {"languages": ["english", "italian"]},
                       "en-gb", None)
    _validate_language("canary", {"languages": ["en", "it"]}, "english", None)
    _validate_language("canary", {"languages": ["en"]}, "en-us", None)


def test_a_language_the_model_really_lacks_is_still_refused():
    from fastapi import HTTPException
    from codai.api.transcriptions import _validate_language
    with pytest.raises(HTTPException) as e:
        _validate_language("wav2vec2-it", {"languages": ["italian"]}, "en-us", None)
    assert e.value.status_code == 400
    # And a model that declares nothing still accepts anything.
    _validate_language("whatever", {}, "en-us", None)


def test_aliases_are_symmetric():
    assert L.aliases("en-gb") == {"en-gb", "en", "english"}
    assert "en" in L.aliases("english")
    # An unmappable tag still matches a config that declares exactly it.
    assert L.aliases("klingon") == {"klingon"}


# --------------------------------------------------- the backends, not the map
def test_crisperwhisper_backend_sends_the_name(monkeypatch):
    import codai.api.stt_backends as sb
    sent = {}

    class _Resp:
        status_code = 200
        def json(self):
            return {"text": "hi", "words": [], "segments": []}

    class _Req:
        exceptions = __import__("requests").exceptions
        @staticmethod
        def post(url, params=None, data=None, timeout=None, headers=None):
            sent.update(params or {})
            return _Resp()

    monkeypatch.setitem(__import__("sys").modules, "requests", _Req)
    b = sb._RemoteCrisperWhisperBackend("crisperwhisper", {}, "http://x")
    import tempfile, os
    fd, path = tempfile.mkstemp()
    os.close(fd)
    try:
        b.transcribe(path, language="en-us")
    finally:
        os.unlink(path)
    assert sent.get("language") == "english", sent


def test_nemo_backend_sends_the_code(monkeypatch):
    import codai.api.stt_backends as sb
    sent = {}

    class _Resp:
        status_code = 200
        def json(self):
            return {"text": "hi"}

    class _Req:
        exceptions = __import__("requests").exceptions
        @staticmethod
        def post(url, params=None, data=None, timeout=None, headers=None):
            sent.update(params or {})
            return _Resp()

    monkeypatch.setitem(__import__("sys").modules, "requests", _Req)
    b = sb._RemoteNemoBackend("canary", {}, "http://x")
    import tempfile, os
    fd, path = tempfile.mkstemp()
    os.close(fd)
    try:
        b.transcribe(path, language="pt-br", target_language="en-us")
    finally:
        os.unlink(path)
    assert sent.get("language") == "pt", sent
    assert sent.get("target_language") == "en", sent


def test_compose_model_picker_accepts_a_regional_tag(monkeypatch):
    import codai.api.compose as C
    monkeypatch.setattr(C, "_stt_entry", lambda m: {"languages": ["english"]})
    # Before: "en-gb" not in {"english"} -> the only word-timing model was
    # disqualified and captions silently fell back to estimated timings.
    assert C._stt_speaks("crisperwhisper", ["en-gb"]) is True
    monkeypatch.setattr(C, "_stt_entry", lambda m: {"languages": ["italian"]})
    assert C._stt_speaks("wav2vec2-it", ["en-gb"]) is False
