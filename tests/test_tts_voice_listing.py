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

"""Listing a TTS model's built-in voices.

A client offers a voice picker before anything is loaded, so this has to be
answerable without loading a model -- a cheap catalogue call that loads is how
a listing route turns into a several-hundred-MB loader.

It also must not invent metadata. Kokoro encodes the accent and gender in the
id (``af_sarah`` = American female) so those are derived; a family whose ids
carry nothing reports ``None`` rather than a plausible-looking guess.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from codai.api import tts_backends as T


class _Loaded:
    def __init__(self, voices):
        self.voices = voices


def test_a_loaded_model_is_asked_directly():
    got = T.family_voices("hexgrad/Kokoro-82M", _Loaded(["af_sarah", "bm_george"]))
    assert got["voices_source"] == "loaded"
    ids = [v["id"] for v in got["voices"]]
    assert ids == ["af_sarah", "bm_george"]


def test_kokoro_ids_give_up_their_accent_and_gender():
    got = T.family_voices("hexgrad/Kokoro-82M", _Loaded(["af_sarah", "bm_george"]))
    sarah, george = got["voices"]
    assert sarah["gender"] == "female" and sarah["languages"] == ["en-us"]
    assert sarah["name"] == "Sarah"
    assert george["gender"] == "male" and george["languages"] == ["en-gb"]


def test_an_id_that_carries_nothing_reports_none_not_a_guess():
    """A wrong gender against a voice is worse than no gender."""
    got = T.family_voices("coqui/XTTS-v2", _Loaded(["Claribel Dervla"]))
    v = got["voices"][0]
    assert v["id"] == "Claribel Dervla"
    assert v["gender"] is None
    assert v["languages"] == []


def test_bark_is_listed_without_loading_anything():
    """The static list was the ten English speakers. suno/bark ships a bank of
    13 languages x speakers 0-9, and listing only the English ten is what made
    a 13-language engine look like an English one. English stays first so a
    client taking voices[0] as its default is unaffected."""
    got = T.family_voices("suno/bark", None)
    assert got["voices_source"] == "static"
    assert len(got["voices"]) == 13 * 10
    assert got["voices"][0]["languages"] == ["en"]
    assert all(v["languages"] == ["en"] for v in got["voices"][:10])
    langs = {l for v in got["voices"] for l in v["languages"]}
    assert langs == set(T._FAMILY_LANGUAGES["bark"])


def test_parler_says_it_is_described_rather_than_chosen():
    """Parler has no speaker names -- it is steered by a description. A client
    reading only an empty `voices` would wrongly conclude it cannot be
    directed at all."""
    got = T.family_voices("parler-tts/parler-tts-mini-multilingual", None)
    assert got["voices"] == []
    assert got["voice_selection"] == "described"


def test_xtts_offers_cloning_as_well_as_named_speakers():
    got = T.family_voices("coqui/XTTS-v2", None)
    assert "clone" in got["voice_selection"]


def test_an_unloaded_model_reports_none_rather_than_loading():
    got = T.family_voices("coqui/XTTS-v2", None)
    assert got["voices_source"] in ("none", "remembered")


def test_voices_are_remembered_across_a_load(tmp_path, monkeypatch):
    """So the next listing needs no load at all."""
    monkeypatch.setattr(T, "_cache_dir", lambda: tmp_path)
    T.remember_voices("coqui/XTTS-v2", _Loaded(["Ana Florence", "Damien Black"]))
    got = T.family_voices("coqui/XTTS-v2", None)
    assert got["voices_source"] == "remembered"
    assert [v["id"] for v in got["voices"]] == ["Ana Florence", "Damien Black"]


def test_remembering_never_breaks_synthesis(tmp_path, monkeypatch):
    """It runs on the load path, so every failure mode is swallowed."""
    monkeypatch.setattr(T, "_cache_dir", lambda: tmp_path)

    class _Angry:
        @property
        def voices(self):
            raise RuntimeError("cannot enumerate")

    T.remember_voices("x", _Angry())          # must not raise
    T.remember_voices("y", None)              # must not raise
    monkeypatch.setattr(T, "_cache_dir",
                        lambda: tmp_path / "nope" / "deeper")
    T.remember_voices("z", _Loaded(["a"]))    # unwritable: must not raise


def test_a_corrupt_cache_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "_cache_dir", lambda: tmp_path)
    (tmp_path / "tts_voices.json").write_text("{not json")
    assert T.family_voices("coqui/XTTS-v2", None)["voices"] == []
    T.remember_voices("coqui/XTTS-v2", _Loaded(["Ana Florence"]))
    assert T.family_voices("coqui/XTTS-v2", None)["voices_source"] == "remembered"


def test_the_capabilities_route_does_not_collide_with_voiceprints():
    """/v1/audio/speakers is enrolled speaker voiceprints (recognition). The
    built-in TTS voices must NOT be served from that path."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    import re
    tts = open(os.path.join(root, "codai/api/tts.py"), encoding="utf-8").read()
    # Route DECLARATIONS only -- the path is named in a comment here on
    # purpose, to say why it is not being used.
    routes = re.findall(r'@router\.\w+\(\s*"([^"]+)"', tts)
    assert not [r for r in routes if r.startswith("/v1/audio/speakers")], (
        f"the TTS voice list is being served from the speaker-recognition "
        f"path; routes declared here: {routes}")
    assert "/v1/audio/speech/capabilities" in routes


# --------------------------------------------------------------------------
# Whether the engine can answer at all
# --------------------------------------------------------------------------
#
# "Speech engines listed but not installed": XTTS answered 501 and parler 500,
# and a client had no way to know before it asked. Both build their venv on
# first use now, so the honest answer is not "unavailable" but "the first
# request will spend minutes building it" -- which a client can act on.


def test_an_engine_that_needs_no_venv_is_always_ready():
    for name in ("hexgrad/Kokoro-82M", "suno/bark"):
        got = T.environment_report(name)
        assert got == {"needs_venv": False, "env_ready": True, "env_detail": ""}, name


def test_a_missing_venv_is_reported_as_not_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "_family_venv_dir", lambda fam: tmp_path / "nothing_here")
    got = T.environment_report("coqui/XTTS-v2")
    assert got["needs_venv"] is True
    assert got["env_ready"] is False
    assert "does not exist" in got["env_detail"]


def test_a_venv_without_the_package_is_not_ready(tmp_path, monkeypatch):
    """The venv builds fine and the install is what failed -- the state a
    half-finished first request leaves behind."""
    from codai.api import venv_env
    venv = tmp_path / "xtts_venv"
    venv_env.create_venv(venv)
    monkeypatch.setattr(T, "_family_venv_dir", lambda fam: venv)
    got = T.environment_report("coqui/XTTS-v2")
    assert got["env_ready"] is False
    assert "TTS" in got["env_detail"] and "not" in got["env_detail"]


def test_a_populated_venv_is_ready_without_importing_anything(tmp_path, monkeypatch):
    from codai.api import venv_env
    venv = tmp_path / "xtts_venv"
    venv_env.create_venv(venv)
    site = next(iter(venv.glob("lib/python*/site-packages")))
    for pkg in T._VENV_FAMILIES["coqui"]:
        (site / pkg).mkdir()
    monkeypatch.setattr(T, "_family_venv_dir", lambda fam: venv)

    def _no_imports(*a, **k):
        raise AssertionError("the readiness check imported into the venv")

    monkeypatch.setattr(venv_env, "run", _no_imports)
    got = T.environment_report("coqui/XTTS-v2")
    assert got["env_ready"] is True


def test_the_capabilities_route_reports_it():
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(root, "codai/api/tts.py"), encoding="utf-8").read()
    route = src.split("async def speech_capabilities", 1)[1].split("@router", 1)[0]
    assert "environment_report" in route


def test_the_engine_package_alone_is_not_evidence(tmp_path, monkeypatch):
    """`pip install coqui-tts` brought the engine and its ninety dependencies
    and NO torch, so the venv looked complete, this route said ready, and the
    first synthesis failed on `import TTS` (zeiss, 2026-10-29)."""
    from codai.api import venv_env
    venv = tmp_path / "xtts_venv"
    venv_env.create_venv(venv)
    site = next(iter(venv.glob("lib/python*/site-packages")))
    (site / "TTS").mkdir()
    monkeypatch.setattr(T, "_family_venv_dir", lambda fam: venv)

    got = T.environment_report("coqui/XTTS-v2")
    assert got["env_ready"] is False
    assert "torch" in got["env_detail"], got["env_detail"]

    (site / "torch").mkdir()
    assert T.environment_report("coqui/XTTS-v2")["env_ready"] is True


def test_every_venv_family_requires_torch():
    for fam, modules in T._VENV_FAMILIES.items():
        assert "torch" in modules, f"{fam} can report ready without torch"


# ── the loaded/remembered paths had never once worked ────────────────────────
#
# Every backend in tts_backends declares `voices` as a METHOD, and both callers
# read it as a sequence:
#
#     [str(v) for v in (getattr(backend, "voices", None) or [])]
#
# A bound method is truthy, so `or []` never fired; iterating it raised
# TypeError; both callers caught that and fell back to "nothing known". So no
# voice was ever recorded or returned for any engine, and the cache file was
# never written. Kokoro and bark hid it behind their static lists; XTTS and
# parler, having none, reported zero voices for ever -- after 17 successful
# syntheses in one afternoon.
#
# These tests did not catch it because `_Loaded` sets `voices` as a LIST
# ATTRIBUTE, a shape no real backend has. The fixtures below use both.


class _LoadedMethod:
    """How every real backend actually looks."""

    def __init__(self, voices):
        self._v = voices

    def voices(self):
        return self._v


def test_a_backend_whose_voices_is_a_method_is_read():
    got = T.family_voices("coqui/XTTS-v2", _LoadedMethod(["Claribel Dervla",
                                                          "Daisy Studious"]))
    assert got["voices_source"] == "loaded"
    assert [v["id"] for v in got["voices"]] == ["Claribel Dervla", "Daisy Studious"]


def test_the_attribute_shape_still_works():
    """Belt and braces: whichever way a backend exposes them."""
    got = T.family_voices("coqui/XTTS-v2", _Loaded(["A", "B"]))
    assert [v["id"] for v in got["voices"]] == ["A", "B"]


def test_every_real_backend_exposes_voices_the_way_the_reader_expects():
    """The assumption the bug rested on, asserted against the real classes
    instead of a fixture."""
    import inspect
    classes = [obj for _, obj in inspect.getmembers(T, inspect.isclass)
               if hasattr(obj, "voices") and obj.__module__ == T.__name__]
    assert classes, "no backend classes found — this test has stopped checking"
    for cls in classes:
        assert callable(cls.voices), f"{cls.__name__}.voices is not callable"
        # And the reader handles exactly that.
        assert T._voice_ids(_LoadedMethod(["x"])) == ["x"]


def test_a_backend_that_raises_when_asked_costs_only_the_list(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(T, "_VOICES_MEMO", {}, raising=False)

    class Broken:
        def voices(self):
            raise RuntimeError("engine not ready")

    class Angry:
        @property
        def voices(self):
            raise RuntimeError("cannot even read the attribute")

    assert T._voice_ids(Broken()) == []
    assert T._voice_ids(Angry()) == []
    assert T.family_voices("coqui/XTTS-v2", Broken())["voices_source"] == "none"


def test_a_backend_with_no_voices_attribute_at_all():
    class Bare:
        pass

    assert T._voice_ids(Bare()) == []


def test_remember_voices_actually_writes_now(tmp_path, monkeypatch):
    """It had never written the file once."""
    import json
    monkeypatch.setattr(T, "_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(T, "_VOICES_MEMO", {}, raising=False)
    T.remember_voices("coqui/XTTS-v2", _LoadedMethod(["Claribel Dervla"]))
    stored = json.loads((tmp_path / "tts_voices.json").read_text())
    assert stored["coqui/XTTS-v2"] == ["Claribel Dervla"]


# ── recorded under one id, published under another ───────────────────────────

def test_voices_recorded_under_an_alias_are_found_by_the_canonical_id(
        tmp_path, monkeypatch):
    """A model answers to several ids. The one that RECORDS is whichever the
    request used; the one that ASKS is whichever the listing publishes. 58 XTTS
    voices were written under "XTTS-v2" and looked up under "coqui/XTTS-v2",
    so they read as none."""
    import json
    monkeypatch.setattr(T, "_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(T, "_VOICES_MEMO", {}, raising=False)
    (tmp_path / "tts_voices.json").write_text(
        json.dumps({"XTTS-v2": ["Claribel Dervla", "Daisy Studious"]}))
    assert len(T._remembered_voices("coqui/XTTS-v2")) == 2
    assert len(T._remembered_voices("XTTS-v2")) == 2


def test_the_match_works_in_the_other_direction_too(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(T, "_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(T, "_VOICES_MEMO", {}, raising=False)
    (tmp_path / "tts_voices.json").write_text(
        json.dumps({"coqui/XTTS-v2": ["Claribel Dervla"]}))
    assert len(T._remembered_voices("XTTS-v2")) == 1


def test_the_match_is_case_insensitive(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(T, "_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(T, "_VOICES_MEMO", {}, raising=False)
    (tmp_path / "tts_voices.json").write_text(json.dumps({"XTTS-v2": ["A"]}))
    assert len(T._remembered_voices("coqui/xtts-v2")) == 1


def test_a_different_model_is_not_matched(tmp_path, monkeypatch):
    """Matching on the last segment must not collapse distinct models."""
    import json
    monkeypatch.setattr(T, "_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(T, "_VOICES_MEMO", {}, raising=False)
    (tmp_path / "tts_voices.json").write_text(json.dumps({"XTTS-v2": ["A"]}))
    assert T._remembered_voices("suno/bark") == []
    assert T._remembered_voices("hexgrad/Kokoro-82M") == []


def test_an_exact_key_wins_over_a_segment_match(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(T, "_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(T, "_VOICES_MEMO", {}, raising=False)
    (tmp_path / "tts_voices.json").write_text(json.dumps({
        "XTTS-v2": ["alias-recorded"],
        "coqui/XTTS-v2": ["canonical-recorded"]}))
    assert T._remembered_voices("coqui/XTTS-v2") == ["canonical-recorded"]
