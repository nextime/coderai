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


# ── kokoro's voices, named without loading it ────────────────────────────────

def _fake_voices_bin(path, names):
    """An npz whose keys are voice names — the shape kokoro-onnx selects from."""
    import numpy as np
    np.savez(str(path), **{n: np.zeros((1, 1), dtype=np.float32) for n in names})
    # np.savez appends .npz; the real file has no such suffix.
    os.replace(str(path) + '.npz', str(path))
    return path


def test_kokoro_voices_are_named_from_the_file_the_backend_selects_from(
        tmp_path, monkeypatch):
    """Not a hand-kept list: the names come out of the very file synthesis picks
    from, so the listing cannot offer a voice that then gets rejected."""
    monkeypatch.setattr(T, "_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(T, "_KOKORO_VOICE_MEMO", {}, raising=False)
    monkeypatch.setattr(T, "_VOICES_MEMO", {}, raising=False)
    _fake_voices_bin(tmp_path / 'voices-v1.0.bin',
                     ['af_sarah', 'am_adam', 'if_sara', 'jf_alpha'])
    v = T.family_voices('hexgrad/Kokoro-82M')
    assert v['voices_source'] == 'static'
    assert [x['id'] for x in v['voices']] == ['af_sarah', 'am_adam',
                                              'if_sara', 'jf_alpha']
    # And the accent map still decodes each one.
    by_id = {x['id']: x for x in v['voices']}
    assert by_id['if_sara']['languages'] == ['it']
    assert by_id['am_adam']['gender'] == 'male'


def test_a_listing_never_downloads_the_voices_file(tmp_path, monkeypatch):
    """_resolve_kokoro_files fetches a 28 MB release when nothing local matches.
    /v1/models is polled; it must not be a download trigger."""
    monkeypatch.setattr(T, "_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(T, "_KOKORO_VOICE_MEMO", {}, raising=False)
    monkeypatch.setattr(T, "_VOICES_MEMO", {}, raising=False)

    def explode(*a, **k):
        raise AssertionError('the listing tried to download a TTS asset')

    monkeypatch.setattr(T, '_download', explode)
    monkeypatch.setattr(T, '_resolve_kokoro_files', explode)
    v = T.family_voices('hexgrad/Kokoro-82M')      # no file on disk
    assert v['voices'] == []
    assert v['voices_source'] == 'none'


def test_kokoro_voice_file_is_reread_when_it_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(T, "_KOKORO_VOICE_MEMO", {}, raising=False)
    f = tmp_path / 'voices-v1.0.bin'
    _fake_voices_bin(f, ['af_sarah'])
    assert T._kokoro_static_voices() == ['af_sarah']
    assert T._kokoro_static_voices() == ['af_sarah']       # memoised
    os.utime(f, (1, 1))
    _fake_voices_bin(f, ['af_sarah', 'if_sara'])
    assert T._kokoro_static_voices() == ['af_sarah', 'if_sara']


def test_a_corrupt_voices_file_costs_only_the_voice_list(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(T, "_KOKORO_VOICE_MEMO", {}, raising=False)
    (tmp_path / 'voices-v1.0.bin').write_bytes(b'not an npz')
    assert T._kokoro_static_voices() == []


def test_a_cards_voices_path_is_honoured(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "_cache_dir", lambda: tmp_path / 'nothing-here')
    monkeypatch.setattr(T, "_KOKORO_VOICE_MEMO", {}, raising=False)
    monkeypatch.setattr(T, "_VOICES_MEMO", {}, raising=False)
    custom = tmp_path / 'my-voices.bin'
    _fake_voices_bin(custom, ['bf_emma'])
    v = T.family_voices('hexgrad/Kokoro-82M', config={'voices_path': str(custom)})
    assert [x['id'] for x in v['voices']] == ['bf_emma']


def test_a_remembered_list_still_beats_the_kokoro_file(tmp_path, monkeypatch):
    """What a load actually enumerated outranks what the file contains."""
    monkeypatch.setattr(T, "_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(T, "_KOKORO_VOICE_MEMO", {}, raising=False)
    monkeypatch.setattr(T, "_VOICES_MEMO", {}, raising=False)
    _fake_voices_bin(tmp_path / 'voices-v1.0.bin', ['af_sarah', 'am_adam'])
    (tmp_path / 'tts_voices.json').write_text(
        json.dumps({'hexgrad/Kokoro-82M': ['if_sara']}))
    v = T.family_voices('hexgrad/Kokoro-82M')
    assert v['voices_source'] == 'remembered'
    assert [x['id'] for x in v['voices']] == ['if_sara']


def test_described_engines_report_no_voices_and_say_why():
    """parler has no speaker names at all; an empty list plus selection
    'described' is the honest answer, not a bug to fill in."""
    v = T.family_voices('parler-tts/parler-tts-mini-multilingual')
    assert v['voices'] == []
    assert v['voice_selection'] == 'described'


def test_the_listing_passes_the_card_to_the_voice_lookup():
    src = open(MANAGER).read()
    assert 'tts_backends.family_voices(model_id, config=meta)' in src


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


# ── a language we advertise has to actually come out ─────────────────────────
#
# Advertising is only half the job. Probing all nine of Kokoro's languages
# through the real endpoint on 2026-10-30 found eight working and ONE failing:
# `fr` returned `espeak-ng has no such language`, because French is regioned in
# espeak (fr-fr, fr-be, fr-ch) exactly as English is. The map's own comment
# asserted the opposite -- that "it", "fr" and "de" all pass bare -- so the
# claim had never been tested. de/it/nl/pl/tr/ru/ko/ar/cs/hu/sv do pass bare;
# fr does not.

def test_french_is_regioned_like_english():
    """The regression: a bare 'fr' is what every OpenAI-compatible client
    sends, and it 500ed."""
    assert T.espeak_language("fr") == "fr-fr"


@pytest.mark.parametrize("code", ["en", "fr", "pt", "zh"])
def test_every_regioned_language_is_mapped(code):
    """These four are the ones espeak refuses bare."""
    assert T.espeak_language(code) != code


@pytest.mark.parametrize("code", ["de", "it", "nl", "pl", "tr", "ru",
                                  "ko", "ar", "cs", "hu", "sv"])
def test_a_bare_code_espeak_accepts_is_left_alone(code):
    """Measured: each of these works bare. Rewriting one would send espeak a
    region it may not have."""
    assert T.espeak_language(code) == code


@pytest.mark.parametrize("code", ["fr-fr", "fr-be", "en-gb", "pt-br"])
def test_an_explicit_region_passes_through(code):
    assert T.espeak_language(code) == code


def test_a_blank_language_defaults_to_english():
    assert T.espeak_language("") == "en-us"
    assert T.espeak_language(None) == "en-us"


def test_case_and_underscores_are_normalised():
    """Clients send en_US and EN-GB; both name a real espeak voice."""
    assert T.espeak_language("FR") == "fr-fr"
    assert T.espeak_language("en_GB") == "en-gb"


def test_every_language_kokoro_advertises_resolves_to_something_espeak_takes():
    """The listing and the synthesis path must agree. Each advertised code is
    either accepted bare (measured) or mapped to a region that is.

    This is the invariant that was broken: `fr` was advertised through the voice
    ids and rejected at synthesis.
    """
    measured_bare_ok = {"es", "hi", "it", "ja", "en-gb", "en-us", "pt-br"}
    for code in T.family_languages("hexgrad/Kokoro-82M"):
        resolved = T.espeak_language(code)
        assert resolved in measured_bare_ok or code in T._ESPEAK_REGIONLESS, (
            f"Kokoro advertises {code!r} but it resolves to {resolved!r}, "
            f"which was never measured against espeak")


# ── XTTS keys by its OWN codes, one of which is not two letters ──────────────
#
# Probing XTTS's seventeen on 2026-10-30: sixteen returned audio and `zh-cn`
# failed with `KeyError: 'zh'`. Our own call site did it -- `language=lang[:2]`
# truncated the one XTTS code longer than two characters, so the single language
# with a regioned code was the single language that could not be spoken.

def test_chinese_keeps_its_region():
    """The regression: zh-cn[:2] == 'zh', which XTTS has no table entry for."""
    assert T.xtts_language("zh-cn") == "zh-cn"


def test_a_bare_zh_reaches_the_regioned_code():
    """Clients send 'zh'; XTTS only has 'zh-cn'."""
    assert T.xtts_language("zh") == "zh-cn"


@pytest.mark.parametrize("code", ["en", "es", "fr", "de", "it", "pt", "pl",
                                  "tr", "ru", "nl", "cs", "ar", "hu", "ko",
                                  "ja", "hi"])
def test_the_two_letter_codes_are_unchanged(code):
    assert T.xtts_language(code) == code


@pytest.mark.parametrize("sent,expect", [("en-us", "en"), ("en_GB", "en"),
                                         ("pt-br", "pt"), ("ZH_CN", "zh-cn")])
def test_a_region_a_client_sends_lands_on_the_code_xtts_has(sent, expect):
    """OpenAI-compatible clients send locales; XTTS mostly wants bare codes."""
    assert T.xtts_language(sent) == expect


def test_every_language_xtts_advertises_resolves_into_its_own_table():
    """The invariant that broke: nothing may be advertised that then raises."""
    for code in T.family_languages("coqui/XTTS-v2"):
        assert T.xtts_language(code) in T._FAMILY_LANGUAGES["coqui"], code


def test_the_call_site_no_longer_truncates():
    src = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'codai', 'api', 'tts_backends.py')).read()
    head = src.index("class _CoquiBackend") if "class _CoquiBackend" in src \
        else src.index('family = "coqui"')
    body = src[head:head + 3000]
    assert '"language": xtts_language(lang)' in body
    assert '(lang or "en")[:2]' not in body


def test_the_isolated_worker_agrees_with_the_server():
    """tools/xtts_service.py runs in coqui's own venv and cannot import codai,
    so the list is duplicated there. Duplicated is fine; diverging is not --
    the worker is what actually synthesises when XTTS is run isolated."""
    import re
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'tools', 'xtts_service.py')
    src = open(path).read()
    m = re.search(r'_XTTS_LANGUAGES\s*=\s*\[(.*?)\]', src, re.S)
    assert m, 'the worker no longer declares its language list'
    worker = [x.strip().strip('"\'') for x in m.group(1).split(',') if x.strip()]
    assert worker == T._FAMILY_LANGUAGES["coqui"], (
        f'worker and server disagree:\n  worker: {worker}\n'
        f'  server: {T._FAMILY_LANGUAGES["coqui"]}')
    assert '(req.get("language") or "en")[:2]' not in src
    assert '_xtts_language(req.get("language"))' in src


def test_the_worker_normaliser_behaves_like_the_servers():
    """Same inputs, same answers -- checked by running the worker's own copy."""
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'tools', 'xtts_service.py')
    src = open(path).read()
    ns: dict = {}
    start = src.index('_XTTS_LANGUAGES = [')
    end = src.index('def _tts(', start)
    exec(compile(src[start:end], '<worker>', 'exec'), ns)
    for code in ("zh", "zh-cn", "en", "en-us", "pt-br", "it", "ja", "", None, "xx"):
        assert ns['_xtts_language'](code) == T.xtts_language(code), code
