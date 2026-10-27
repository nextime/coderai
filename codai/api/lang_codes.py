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

"""One language tag, four spellings: normalisation for the speech backends.

A language reaches the server as whatever the caller naturally has. A voice is
chosen for an accent, so a TTS language is a REGIONAL tag -- ``en-us``,
``en-gb``, ``pt-br`` -- and that is the tag a compose request carries from the
voice to the captions. The speech backends then each want something else:

* HF Whisper (``whisper_hf``, ``crisperwhisper``) wants the English NAME:
  ``english``. Handed ``en-us`` it raises
  ``Unsupported language: en-us. Language should be one of: ['english', ...]``,
  which the worker returns as a 500 -- measured on every scene of every render
  that asked for word-timed captions (2026-10-27).
* whisper.cpp (``whisper-server``) and NeMo/Canary want the ISO-639-1 CODE:
  ``en``. whisper.cpp also accepts the literal ``auto``.
* wav2vec2 is per-language (the code is advisory) and vosk picks the language
  by model directory, so for those the code is the sane pass-through.

So the tag is normalised ONCE, here, for the backend that will receive it, and
an unmappable tag becomes ``None`` -- which every backend reads as
auto-detect. Refusing a transcription because the caller named a dialect is
never the better outcome: the words are the same words.

Nothing here validates. :func:`codai.api.transcriptions._validate_language`
decides what a model accepts, and uses :func:`aliases` so a model that declares
``english`` also answers to ``en`` and ``en-gb``.
"""

from typing import Optional

# ISO-639-1 -> the English name Whisper uses. This is Whisper's own language
# table (the 99 it was trained on); the names are what transformers lists in
# the error above, so they are the spelling to send.
WHISPER_NAMES = {
    "en": "english", "zh": "chinese", "de": "german", "es": "spanish",
    "ru": "russian", "ko": "korean", "fr": "french", "ja": "japanese",
    "pt": "portuguese", "tr": "turkish", "pl": "polish", "ca": "catalan",
    "nl": "dutch", "ar": "arabic", "sv": "swedish", "it": "italian",
    "id": "indonesian", "hi": "hindi", "fi": "finnish", "vi": "vietnamese",
    "he": "hebrew", "uk": "ukrainian", "el": "greek", "ms": "malay",
    "cs": "czech", "ro": "romanian", "da": "danish", "hu": "hungarian",
    "ta": "tamil", "no": "norwegian", "th": "thai", "ur": "urdu",
    "hr": "croatian", "bg": "bulgarian", "lt": "lithuanian", "la": "latin",
    "mi": "maori", "ml": "malayalam", "cy": "welsh", "sk": "slovak",
    "te": "telugu", "fa": "persian", "lv": "latvian", "bn": "bengali",
    "sr": "serbian", "az": "azerbaijani", "sl": "slovenian", "kn": "kannada",
    "et": "estonian", "mk": "macedonian", "br": "breton", "eu": "basque",
    "is": "icelandic", "hy": "armenian", "ne": "nepali", "mn": "mongolian",
    "bs": "bosnian", "kk": "kazakh", "sq": "albanian", "sw": "swahili",
    "gl": "galician", "mr": "marathi", "pa": "punjabi", "si": "sinhala",
    "km": "khmer", "sn": "shona", "yo": "yoruba", "so": "somali",
    "af": "afrikaans", "oc": "occitan", "ka": "georgian", "be": "belarusian",
    "tg": "tajik", "sd": "sindhi", "gu": "gujarati", "am": "amharic",
    "yi": "yiddish", "lo": "lao", "uz": "uzbek", "fo": "faroese",
    "ht": "haitian creole", "ps": "pashto", "tk": "turkmen", "nn": "nynorsk",
    "mt": "maltese", "sa": "sanskrit", "lb": "luxembourgish", "my": "myanmar",
    "bo": "tibetan", "tl": "tagalog", "mg": "malagasy", "as": "assamese",
    "tt": "tatar", "haw": "hawaiian", "ln": "lingala", "ha": "hausa",
    "ba": "bashkir", "jw": "javanese", "su": "sundanese", "yue": "cantonese",
}

# The reverse direction, plus the spellings people actually type. Whisper's own
# table carries several of these aliases and they are cheap to honour.
_NAME_TO_CODE = {v: k for k, v in WHISPER_NAMES.items()}
_NAME_TO_CODE.update({
    "mandarin": "zh", "chinese (simplified)": "zh", "chinese (traditional)": "zh",
    "brazilian portuguese": "pt", "castilian": "es", "flemish": "nl",
    "burmese": "my", "valencian": "ca", "moldavian": "ro", "moldovan": "ro",
    "sinhalese": "si", "panjabi": "pa", "pushto": "ps", "farsi": "fa",
    "nepali (individual language)": "ne", "greek (modern)": "el",
    "norwegian nynorsk": "nn", "bokmal": "no", "norwegian bokmal": "no",
    "letzeburgesch": "lb", "javanese (jw)": "jw", "haitian": "ht",
})

# Three-letter codes worth understanding, because ``eng``/``ita`` turn up in
# corpora and in some TTS catalogues. Only where unambiguous.
_ISO3 = {
    "eng": "en", "ita": "it", "deu": "de", "ger": "de", "fra": "fr",
    "fre": "fr", "spa": "es", "por": "pt", "rus": "ru", "nld": "nl",
    "dut": "nl", "pol": "pl", "ces": "cs", "cze": "cs", "jpn": "ja",
    "kor": "ko", "zho": "zh", "chi": "zh", "cmn": "zh", "ara": "ar",
    "heb": "he", "hin": "hi", "tur": "tr", "swe": "sv", "dan": "da",
    "nor": "no", "fin": "fi", "ell": "el", "gre": "el", "ukr": "uk",
    "ron": "ro", "rum": "ro", "hun": "hu", "bul": "bg", "hrv": "hr",
    "srp": "sr", "slk": "sk", "slo": "sk", "slv": "sl", "cat": "ca",
    "eus": "eu", "baq": "eu", "glg": "gl", "vie": "vi", "tha": "th",
    "ind": "id", "msa": "ms", "may": "ms", "fas": "fa", "per": "fa",
    "urd": "ur", "ben": "bn", "tam": "ta", "tel": "te", "mar": "mr",
    "guj": "gu", "kan": "kn", "mal": "ml", "pan": "pa", "swa": "sw",
    "afr": "af", "isl": "is", "ice": "is", "est": "et", "lav": "lv",
    "lit": "lt", "kat": "ka", "geo": "ka", "hye": "hy", "arm": "hy",
    "aze": "az", "kaz": "kk", "uzb": "uz", "mkd": "mk", "mac": "mk",
    "sqi": "sq", "alb": "sq", "bel": "be", "mlt": "mt", "cym": "cy",
    "wel": "cy", "lat": "la", "yue": "yue", "haw": "haw",
}

# Families whose ``language`` argument is an English NAME rather than a code.
# Only the transformers pipeline is in here. faster-whisper and whisper.cpp are
# both built on Whisper too but take the CODE, so they are deliberately absent:
# the distinction is the API, not the model.
_NAME_FAMILIES = {"whisper_hf", "crisperwhisper", "whisper-hf",
                  "whisper-transformers"}


def primary(code) -> str:
    """The primary language subtag, lowercased: ``en-GB`` -> ``en``.

    Accepts the separators that turn up in the wild (``-``, ``_``) and the
    extensions that follow a tag (``zh-Hans-CN``). Returns ``""`` for nothing.
    """
    s = str(code or "").strip().lower().replace("_", "-")
    if not s:
        return ""
    return s.split("-", 1)[0].strip()


def to_code(code) -> Optional[str]:
    """Any spelling of a language -> its ISO-639-1 code, or ``None``.

    Understands a regional tag (``pt-br``), a name (``Portuguese``), a
    three-letter code (``por``) and a code that is already right (``pt``).
    """
    s = str(code or "").strip().lower().replace("_", "-")
    if not s or s in ("auto", "automatic", "detect", "none"):
        return None
    # A full name may itself contain a space or a hyphen ("haitian creole"),
    # so names are tried on the whole string BEFORE it is split on "-".
    if s in _NAME_TO_CODE:
        return _NAME_TO_CODE[s]
    head = primary(s)
    if head in WHISPER_NAMES:
        return head
    if head in _ISO3:
        return _ISO3[head]
    if head in _NAME_TO_CODE:
        return _NAME_TO_CODE[head]
    return None


def to_name(code) -> Optional[str]:
    """Any spelling -> the English name Whisper expects, or ``None``."""
    c = to_code(code)
    return WHISPER_NAMES.get(c) if c else None


def normalize(code, family=None) -> Optional[str]:
    """The spelling ``family`` wants, or ``None`` for auto-detect.

    ``None`` is a deliberate answer, not a failure: every backend treats a
    missing language as "detect it", which is a far better outcome for the
    caller than a 500 because they named a dialect this model has never heard
    of. The caller should omit the argument entirely when this returns
    ``None`` -- passing ``None`` through is equivalent for all four backends.
    """
    fam = str(family or "").strip().lower()
    if fam in _NAME_FAMILIES:
        return to_name(code)
    return to_code(code)


def aliases(code) -> set:
    """Every spelling that means this language, for membership tests.

    ``aliases("en-gb")`` -> ``{"en-gb", "en", "english"}``. Used by language
    validation so a model declaring ``["english"]`` is not made to refuse
    ``en`` -- and vice versa. The original string is always included, so an
    unmappable tag still matches a config that declares exactly it.
    """
    s = str(code or "").strip().lower().replace("_", "-")
    out = {s} if s else set()
    c = to_code(code)
    if c:
        out.add(c)
        name = WHISPER_NAMES.get(c)
        if name:
            out.add(name)
    return out
