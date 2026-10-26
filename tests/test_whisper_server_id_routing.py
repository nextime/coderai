"""A whisper-server model is advertised under a name that routed to nothing.

Two models.json rows describe one whisper-server: a RUNNER (model_path +
server_path + port, registered under its id) and a MODEL row naming the weights
(path, no model_path), which registration skips. ``/v1/models`` takes its id
from the model row, so the catalogue advertised an id that matched no runner --
and the request fell through to the generic audio path, which reported the
whisper.cpp file as a missing HuggingFace repo.
"""
import types

import pytest

from codai.models.manager import MultiModelManager


class _WSM:
    def __init__(self, model_path, idle=True):
        self._model_path = model_path
        self._idle = idle

    def is_idle(self):
        return self._idle


def _mgr(servers, aliases=None):
    m = object.__new__(MultiModelManager)
    m.whisper_servers = dict(servers)
    m.whisper_aliases = dict(aliases or {})
    return m


GGUF = "/AI/guffcache/whisper-large-v3-q8_0.gguf"
BIN = "/AI/guffcache/whisper-large-v3-turbo-q8_0.bin"


def test_the_runners_own_id_still_resolves():
    m = _mgr({"whisper3": _WSM(BIN)})
    assert m.resolve_whisper_server_key("whisper3") == "whisper3"


def test_an_alias_still_resolves():
    m = _mgr({"whisper0": _WSM(GGUF), "whisper1": _WSM(GGUF)},
             {"whisper": ["whisper0", "whisper1"]})
    assert m.resolve_whisper_server_key("whisper") in ("whisper0", "whisper1")


def test_the_advertised_basename_resolves():
    """The reported bug, exactly: this is the id /v1/models hands a client."""
    m = _mgr({"whisper3": _WSM(BIN)})
    assert m.resolve_whisper_server_key("whisper-large-v3-turbo-q8_0.bin") == "whisper3"


def test_the_advertised_full_path_resolves():
    m = _mgr({"whisper3": _WSM(BIN)})
    assert m.resolve_whisper_server_key(BIN) == "whisper3"


def test_the_basename_without_extension_resolves():
    m = _mgr({"whisper3": _WSM(BIN)})
    assert m.resolve_whisper_server_key("whisper-large-v3-turbo-q8_0") == "whisper3"


def test_naming_the_weights_picks_an_idle_runner_of_the_several_sharing_them():
    """Three runners serve one file; naming the file should balance, not always
    land on whichever registered first."""
    m = _mgr({"w0": _WSM(GGUF, idle=False),
              "w1": _WSM(GGUF, idle=True),
              "w2": _WSM(GGUF, idle=False)})
    assert m.resolve_whisper_server_key("whisper-large-v3-q8_0.gguf") == "w1"


def test_a_different_model_does_not_resolve():
    m = _mgr({"whisper3": _WSM(BIN)})
    assert m.resolve_whisper_server_key("crisperwhisper") is None
    assert m.resolve_whisper_server_key("whisper-large-v3-q8_0.gguf") is None


def test_an_empty_name_resolves_to_nothing():
    m = _mgr({"whisper3": _WSM(BIN)})
    assert m.resolve_whisper_server_key("") is None
    assert m.resolve_whisper_server_key(None) is None


def test_a_runner_without_a_model_path_is_not_matched_by_accident():
    m = _mgr({"broken": _WSM(None)})
    assert m.resolve_whisper_server_key("whisper-large-v3-q8_0.gguf") is None


def test_transcriptions_uses_the_resolver():
    """Guard the call site: the old one asked whisper_servers directly with the
    client's string, which is what missed."""
    import inspect
    from codai.api import transcriptions
    src = inspect.getsource(transcriptions._run_transcription_inner)
    assert "resolve_whisper_server_key" in src
    assert "whisper_servers.get(model)" not in src
