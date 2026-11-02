"""A whisper.cpp weights file must not be reported as a missing Hub repo.

``_run_transcription_inner`` guards the generic (faster-whisper / transformers)
path against whisper.cpp weights, because that loader cannot read them and says
so misleadingly -- it reports a FILE as a missing HuggingFace repo, or raises
the opaque "Invalid model size".

The guard matched only ``.gguf``. whisper.cpp's own historical format is
``.bin`` (ggml-large-v3.bin), which is still what most published quantisations
ship, so the commoner of the two extensions was exactly the one that fell
through: measured in production 2026-10-27,
``whisper-large-v3-turbo-q8_0.bin`` answered

    500 "... is not a local folder and is not a valid model identifier"

while its ``.gguf`` sibling answered the actionable 503. Both are whisper.cpp
files and both must reach the same guard.
"""
import asyncio

import pytest
from fastapi import HTTPException

from codai.api import transcriptions


GGUF = "/AI/guffcache/whisper-large-v3-q8_0.gguf"
BIN = "/AI/guffcache/whisper-large-v3-turbo-q8_0.bin"


class _NoWhisperServers:
    """An engine with no whisper-server registered -- the case the guard is for."""

    whisper_servers: dict = {}

    def resolve_whisper_server_key(self, model):
        return None

    def request_model(self, requested_model=None, model_type=None):
        # The manager resolved the name to a weights FILE on disk.
        return {"model_name": requested_model, "model_key": requested_model,
                "model_object": object(), "config": {}}


async def _unused_finalize(fmt, text, segments, words):     # pragma: no cover
    raise AssertionError("the guard must raise before any transcription runs")


def _call(model):
    return asyncio.run(transcriptions._run_transcription_inner(
        b"RIFF....", model, None, None, "json", 0.0, "a.wav",
        None, _unused_finalize))


@pytest.fixture(autouse=True)
def _no_whisper_servers(monkeypatch):
    monkeypatch.setattr(transcriptions, "multi_model_manager", _NoWhisperServers())


@pytest.mark.parametrize("path", [BIN, GGUF])
def test_either_whisper_cpp_suffix_is_refused_actionably(path):
    """The reported bug: .bin used to pass this guard, .gguf did not."""
    with pytest.raises(HTTPException) as err:
        _call(path)
    assert err.value.status_code == 503
    assert "whisper-server" in str(err.value.detail)


@pytest.mark.parametrize("path", [BIN, GGUF])
def test_neither_suffix_is_reported_as_a_missing_hub_repo(path):
    """What the caller actually saw, and must not see again."""
    with pytest.raises(HTTPException) as err:
        _call(path)
    detail = str(err.value.detail).lower()
    assert "is not a local folder" not in detail
    assert "model identifier" not in detail
    assert "invalid model size" not in detail


def test_the_suffix_match_is_case_insensitive():
    with pytest.raises(HTTPException) as err:
        _call("/AI/guffcache/WHISPER-LARGE-V3-TURBO-Q8_0.BIN")
    assert err.value.status_code == 503


def test_the_message_no_longer_calls_every_such_file_a_gguf():
    """A .bin reaching a message that says "GGUF" sends the reader hunting for
    a file that does not exist."""
    with pytest.raises(HTTPException) as err:
        _call(BIN)
    assert "GGUF" not in str(err.value.detail)


def test_a_real_hub_model_still_reaches_the_generic_path():
    """The guard must not swallow an ordinary transformers model: a directory
    or Hub id has neither suffix and belongs to faster-whisper."""
    with pytest.raises(HTTPException) as err:
        _call("openai/whisper-large-v3")
    # It fails later, for its own reasons (no real model behind the stub), but
    # it must NOT be refused by this guard.
    assert err.value.status_code != 503 or "whisper-server" not in str(err.value.detail)
