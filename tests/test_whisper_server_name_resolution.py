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

"""Every name that identifies a whisper.cpp runner has to reach it.

A whisper-server model is described by TWO models.json rows:

* a RUNNER: ``model_path`` + ``server_path`` + ``port``, registered under its
  ``id``, which owns the subprocess;
* a MODEL row naming the weights (``path``, no ``model_path``), skipped at
  registration as "config, not a runner" but carrying the ``capabilities`` that
  put the weights path into ``/v1/models``.

So the advertised id is the weights path, and nothing is registered under it.
When that name resolved to nothing the request fell through to the generic
audio path, which treated ``whisper-large-v3-turbo-q8_0.bin`` as a hub repo:

    500 ... is not a local folder and is not a valid model identifier

Several runners may share one alias on purpose -- that is a round-robin pool,
and naming the weights has to spread across it the same way the alias does
rather than always landing on the runner registered first.
"""

import pytest

from codai.models.manager import multi_model_manager as M


class _FakeWSM:
    """Stands in for a WhisperServerManager without starting a subprocess."""

    def __init__(self, model_path, idle=True):
        self._model_path = model_path
        self._gpu_device = 0
        self._idle = idle

    def is_idle(self):
        return self._idle

    def is_running(self):
        return True


GGUF = "/AI/guffcache/whisper-large-v3-q8_0.gguf"
BIN = "/AI/guffcache/whisper-large-v3-turbo-q8_0.bin"


@pytest.fixture
def runners():
    """Two runners on one alias, plus a lone runner for the .bin weights."""
    saved_servers = dict(M.whisper_servers)
    saved_aliases = dict(getattr(M, "whisper_aliases", {}) or {})
    M.whisper_servers.clear()
    M.whisper_servers.update({
        "t_whisper0": _FakeWSM(GGUF),
        "t_whisper1": _FakeWSM(GGUF),
        "t_whisper3": _FakeWSM(BIN),
    })
    if not hasattr(M, "whisper_aliases") or M.whisper_aliases is None:
        M.whisper_aliases = {}
    M.whisper_aliases.clear()
    M.whisper_aliases.update({"t_whisper": ["t_whisper0", "t_whisper1"]})
    try:
        yield
    finally:
        M.whisper_servers.clear()
        M.whisper_servers.update(saved_servers)
        M.whisper_aliases.clear()
        M.whisper_aliases.update(saved_aliases)


@pytest.mark.parametrize("name", [
    GGUF,                               # the full path /v1/models advertises
    "whisper-large-v3-q8_0.gguf",       # its basename
    "whisper-large-v3-q8_0",            # basename without the extension
])
def test_the_advertised_weights_path_reaches_a_runner(runners, name):
    key = M.resolve_whisper_server_key(name)
    assert key in ("t_whisper0", "t_whisper1"), (
        f"{name!r} resolved to {key!r}; an id /v1/models advertises as "
        f"speech_to_text must reach the runner that serves it, or the request "
        f"falls through to the HuggingFace loader and reports a whisper.cpp "
        f"file as a missing hub repo")


@pytest.mark.parametrize("name", [BIN, "whisper-large-v3-turbo-q8_0.bin",
                                  "whisper-large-v3-turbo-q8_0"])
def test_a_bin_file_resolves_too_not_only_a_gguf(runners, name):
    """.bin is whisper.cpp's other extension; the guard used to miss it."""
    assert M.resolve_whisper_server_key(name) == "t_whisper3", (
        f"{name!r} did not reach its runner")


def test_a_runner_is_reachable_by_its_own_id(runners):
    assert M.resolve_whisper_server_key("t_whisper0") == "t_whisper0"


def test_an_unknown_name_resolves_to_nothing(runners):
    """It must not fall back to an arbitrary runner."""
    assert M.resolve_whisper_server_key("not-a-whisper-model") is None
    assert M.resolve_whisper_server_key("") is None


def test_naming_the_weights_spreads_across_a_shared_alias(runners):
    """Two runners share these weights, so both must get used.

    Several rows with one alias are a round-robin pool by design. Resolving the
    weights path always to the first match would quietly serialise a pool the
    operator configured for capacity.
    """
    seen = {M.resolve_whisper_server_key(GGUF) for _ in range(200)}
    assert seen <= {"t_whisper0", "t_whisper1"}, f"resolved outside the pool: {seen}"
    assert len(seen) == 2, (
        f"200 resolutions of the shared weights only ever reached {seen}; both "
        f"runners in the pool should be used")


def test_a_busy_pool_still_resolves(runners):
    """With every runner busy it must still pick one, not return None."""
    for wsm in M.whisper_servers.values():
        wsm._idle = False
    assert M.resolve_whisper_server_key(GGUF) in ("t_whisper0", "t_whisper1")


def test_an_idle_runner_is_preferred_over_a_busy_one(runners):
    M.whisper_servers["t_whisper0"]._idle = False
    seen = {M.resolve_whisper_server_key(GGUF) for _ in range(50)}
    assert seen == {"t_whisper1"}, (
        f"expected the idle runner every time, got {seen}")
