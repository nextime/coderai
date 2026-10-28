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

"""``music.generate`` has to read the response the endpoint actually returns.

``audio_generate`` answers an ``AudioGenerationResponse`` -- a pydantic model,
not a dict -- so ``(res or {}).get("data")`` raised

    'AudioGenerationResponse' object has no attribute 'get'

and compose logged "music generation failed (...); rendered without music".
The render then SUCCEEDED, just silent, which is why this survived in
production: nothing failed loudly enough to be noticed.
"""

import base64
import os
import struct
import sys

import pytest

from codai.api import compose as C
from codai.pydantic.audiogenrequest import AudioGenerationResponse


def _wav_bytes(seconds=1, rate=8000):
    n = int(seconds * rate)
    pcm = b"\x00\x00" * n
    hdr = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt "
    hdr += struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
    hdr += b"data" + struct.pack("<I", len(pcm))
    return hdr + pcm


@pytest.fixture
def fake_audio_generate(monkeypatch):
    """Stand in for the endpoint, answering the real response TYPE."""
    calls = {}

    async def _gen(request, http_request=None):
        calls["model"] = request.model
        calls["prompt"] = request.prompt
        calls["duration"] = request.duration
        return AudioGenerationResponse(
            created=0,
            data=[{"b64_wav": base64.b64encode(_wav_bytes()).decode()}])

    import codai.api.audio_gen as ag
    monkeypatch.setattr(ag, "audio_generate", _gen)
    return calls


def test_music_generation_reads_a_pydantic_response(tmp_path, fake_audio_generate):
    out = str(tmp_path / "music-gen.wav")
    got = C._generate_music(
        None, {"model": "ACE-Step/ACE-Step-v1-3.5B",
               "prompt": "upbeat lo-fi", "duration": 12}, 30.0, out)
    assert got == out, f"returned {got!r} instead of the output path"
    assert os.path.exists(out), "no audio file was written"
    assert os.path.getsize(out) > 44, "the written file holds no PCM"
    assert fake_audio_generate["model"] == "ACE-Step/ACE-Step-v1-3.5B"
    assert fake_audio_generate["prompt"] == "upbeat lo-fi"
    assert fake_audio_generate["duration"] == 12.0


def test_a_plain_dict_response_still_works(tmp_path, monkeypatch):
    """The same call is reachable through paths that answer a dict."""
    async def _gen(request, http_request=None):
        return {"created": 0,
                "data": [{"b64_wav": base64.b64encode(_wav_bytes()).decode()}]}

    import codai.api.audio_gen as ag
    monkeypatch.setattr(ag, "audio_generate", _gen)
    out = str(tmp_path / "m.wav")
    assert C._generate_music(None, {"model": "m", "prompt": "p"}, 30.0, out) == out
    assert os.path.getsize(out) > 44


def test_an_empty_response_is_an_error_not_a_silent_pass(tmp_path, monkeypatch):
    async def _gen(request, http_request=None):
        return AudioGenerationResponse(created=0, data=[])

    import codai.api.audio_gen as ag
    monkeypatch.setattr(ag, "audio_generate", _gen)
    with pytest.raises(Exception) as ei:
        C._generate_music(None, {"model": "m", "prompt": "p"}, 30.0,
                          str(tmp_path / "m.wav"))
    assert "nothing" in str(ei.value).lower()


def test_the_requested_duration_is_bounded_but_honoured(tmp_path, fake_audio_generate):
    """No duration asked for -> derived from the video, clamped to 10..120 s."""
    C._generate_music(None, {"model": "m", "prompt": "p"}, 45.0,
                      str(tmp_path / "a.wav"))
    assert fake_audio_generate["duration"] == 45.0
    C._generate_music(None, {"model": "m", "prompt": "p"}, 4.0,
                      str(tmp_path / "b.wav"))
    assert fake_audio_generate["duration"] == 10.0, "below the floor"
    C._generate_music(None, {"model": "m", "prompt": "p"}, 600.0,
                      str(tmp_path / "c.wav"))
    assert fake_audio_generate["duration"] == 120.0, "above the ceiling"
