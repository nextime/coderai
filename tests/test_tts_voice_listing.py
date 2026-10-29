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
    got = T.family_voices("suno/bark", None)
    assert got["voices_source"] == "static"
    assert len(got["voices"]) == 10
    assert got["voices"][0]["languages"] == ["en"]


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
    assert "TTS is not installed" in got["env_detail"]


def test_a_populated_venv_is_ready_without_importing_anything(tmp_path, monkeypatch):
    from codai.api import venv_env
    venv = tmp_path / "xtts_venv"
    venv_env.create_venv(venv)
    site = next(iter(venv.glob("lib/python*/site-packages")))
    (site / "TTS").mkdir()
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
