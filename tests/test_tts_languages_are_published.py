"""A seventeen-language engine was taken for an English-and-Italian one.

Every TTS model published ``languages: null`` on /v1/models while the STT models
published theirs correctly, so a client was told nothing about what any voice
could speak and could only discover a language by trying it. Measured on a live
deployment 2026-10-30:

    coqui/XTTS-v2    languages: None   voices: 0
    hexgrad/Kokoro-82M  languages: None   voices: 0
    suno/bark        languages: None   voices: 0
    parler-...-multilingual  languages: None   voices: 0
    wav2vec2-it      languages: ['it']            <- STT, correct
    canary-1b-flash  languages: ['en','de','fr','es']

Nothing in the server restricted the languages; it simply never named them. The
sets below are each read from the engine's own files, not assumed -- the tests
that can check against those files do.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codai.api import tts_backends as T


# ── what each family declares ────────────────────────────────────────────────

def test_xtts_publishes_all_seventeen():
    """XTTS-v2's config.json lists 17; publishing 2 is the bug this fixes."""
    langs = T.family_languages("coqui/XTTS-v2")
    assert len(langs) == 17
    for expect in ("en", "it", "ar", "zh-cn", "ko", "hi", "cs", "hu"):
        assert expect in langs, f"{expect} missing from XTTS languages"


def test_bark_publishes_all_thirteen():
    """suno/bark ships a speaker bank of 13 languages."""
    langs = T.family_languages("suno/bark")
    assert sorted(langs) == ["de", "en", "es", "fr", "hi", "it", "ja", "ko",
                             "pl", "pt", "ru", "tr", "zh"]


def test_parler_multilingual_publishes_its_eight():
    langs = T.family_languages("parler-tts/parler-tts-mini-multilingual")
    assert sorted(langs) == ["de", "en", "es", "fr", "it", "nl", "pl", "pt"]


def test_kokoro_languages_come_from_its_own_voice_ids():
    """Derived from _KOKORO_ACCENTS rather than a second hand-kept list, so a
    new voice family cannot be added without its language appearing."""
    langs = T.family_languages("hexgrad/Kokoro-82M")
    assert sorted(langs) == sorted({l for l, _ in T._KOKORO_ACCENTS.values()})
    assert "it" in langs and "en-us" in langs and "ja" in langs


def test_kokoro_language_list_tracks_the_accent_map(monkeypatch):
    """The derivation is real: add an accent, the language shows up."""
    patched = dict(T._KOKORO_ACCENTS)
    patched["k"] = ("sv", "Swedish")
    monkeypatch.setattr(T, "_KOKORO_ACCENTS", patched)
    assert "sv" in T.family_languages("hexgrad/Kokoro-82M")


def test_an_unknown_engine_claims_nothing():
    """A wrong language list is worse than none: a client that believes it sends
    text the voice cannot pronounce and gets noise back, with no error."""
    assert T.family_languages("some/unknown-tts-model") == []
    assert T.family_languages("myorg/melo-whatever") == []


def test_the_card_wins():
    """An operator who restricted a model knows something the table does not."""
    assert T.family_languages("suno/bark", {"languages": ["it"]}) == ["it"]
    assert T.family_languages("coqui/XTTS-v2", {"languages": ["en", "fr"]}) == ["en", "fr"]


def test_an_empty_card_list_falls_back_rather_than_claiming_nothing():
    """`languages: []` is an unset field, not a model that speaks nothing."""
    assert T.family_languages("suno/bark", {"languages": []}) != []
    assert T.family_languages("suno/bark", {}) != []


# ── bark's voices ────────────────────────────────────────────────────────────

def test_bark_static_voices_cover_every_language(tmp_path, monkeypatch):
    """The static list used to be the ten English speakers only, which is where
    "bark is English" literally came from."""
    monkeypatch.setattr(T, "_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(T, "_VOICES_MEMO", {}, raising=False)
    v = T.family_voices("suno/bark")
    assert v["voices_source"] == "static"
    langs = {l for x in v["voices"] for l in (x["languages"] or [])}
    assert langs == set(T._FAMILY_LANGUAGES["bark"])
    assert len(v["voices"]) == 13 * 10


def test_bark_voices_start_with_english(tmp_path, monkeypatch):
    """A client taking voices[0] as its default must not suddenly speak Hindi."""
    monkeypatch.setattr(T, "_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(T, "_VOICES_MEMO", {}, raising=False)
    v = T.family_voices("suno/bark")
    assert v["voices"][0]["id"] == "v2/en_speaker_0"
    assert all(x["languages"] == ["en"] for x in v["voices"][:10])


def test_a_remembered_voice_list_still_beats_the_static_one(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(T, "_VOICES_MEMO", {}, raising=False)
    (tmp_path / "tts_voices.json").write_text(
        json.dumps({"suno/bark": ["v2/it_speaker_3"]}))
    v = T.family_voices("suno/bark")
    assert v["voices_source"] == "remembered"
    assert [x["id"] for x in v["voices"]] == ["v2/it_speaker_3"]


# ── the memo on the polled path ──────────────────────────────────────────────

def test_voices_cache_is_reread_when_it_changes(tmp_path, monkeypatch):
    """/v1/models is polled, so the file is memoised -- but a load records new
    voices and those must appear without a restart."""
    monkeypatch.setattr(T, "_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(T, "_VOICES_MEMO", {}, raising=False)
    f = tmp_path / "tts_voices.json"
    f.write_text(json.dumps({"m": ["a", "b"]}))
    assert T._remembered_voices("m") == ["a", "b"]
    assert T._remembered_voices("m") == ["a", "b"]          # from the memo
    os.utime(f, (1, 1))
    f.write_text(json.dumps({"m": ["c"]}))
    assert T._remembered_voices("m") == ["c"]


def test_a_missing_or_broken_cache_costs_only_the_voice_list(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(T, "_VOICES_MEMO", {}, raising=False)
    assert T._remembered_voices("m") == []                   # no file
    (tmp_path / "tts_voices.json").write_text("{ not json")
    assert T._remembered_voices("m") == []


# ── the listing actually publishes them ──────────────────────────────────────

MANAGER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'codai', 'models', 'manager.py')


def test_the_listing_derives_tts_languages():
    src = open(MANAGER).read()
    assert 'tts_backends.family_languages(' in src
    assert 'tts_backends.family_voices(' in src
    assert '"text_to_speech" in caps.to_list()' in src


def test_the_card_is_consulted_before_the_derivation_in_the_listing():
    src = open(MANAGER).read()
    assert 'languages=meta.get("languages") or tts_languages or None,' in src


def test_a_failure_in_the_derivation_cannot_cost_the_whole_listing():
    """One unparseable TTS entry must not empty /v1/models for every model."""
    src = open(MANAGER).read()
    i = src.index('tts_backends.family_languages(')
    tail = src[i:i + 500]
    assert 'except Exception:' in tail
    assert 'tts_languages, tts_voices = None, None' in tail


def test_model_info_carries_voices():
    from codai.pydantic.textrequest import ModelInfo
    m = ModelInfo(id="x", voices=[{"id": "v2/it_speaker_1", "languages": ["it"]}])
    assert m.voices[0]["languages"] == ["it"]
    assert ModelInfo(id="y").voices is None


def test_languages_is_documented_for_both_directions():
    """It was labelled speech-to-text only, which is why nobody populated it for
    the speaking side."""
    src = open(os.path.join(os.path.dirname(MANAGER), '..', 'pydantic',
                            'textrequest.py')).read()
    i = src.index('languages: Optional[List[str]] = None')
    head = src[max(0, i - 450):i]
    assert 'text-to-speech' in head


# ── the sets match the engines' own files, where those are on disk ───────────

HF = '/AI/huggingface/hub'


@pytest.mark.skipif(not os.path.isdir(HF), reason='no local HF cache')
def test_xtts_set_matches_the_checkpoints_own_config():
    """The one set that can be checked against the model itself, is."""
    import glob
    hits = glob.glob(os.path.join(HF, 'models--coqui--XTTS-v2',
                                  'snapshots', '*', 'config.json'))
    if not hits:
        pytest.skip('XTTS-v2 not downloaded')
    declared = json.load(open(hits[0])).get('languages')
    if not declared:
        pytest.skip('this XTTS config declares no languages')
    assert sorted(T._FAMILY_LANGUAGES['coqui']) == sorted(declared)
