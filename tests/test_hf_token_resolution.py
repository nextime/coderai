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

"""One answer to "where is the HuggingFace token".

The diarization worker had resolved it from the environment or a protected
file in the config dir for a long time, and gated pyannote repos worked. The
MODEL LOADING path had no such resolution: the host runner forwards HF_TOKEN
from the host environment, which is only set if whoever launched the container
exported it, so on an install where it was not, huggingface_hub went out
unauthenticated for every from_pretrained.

That is invisible until a repo is gated, and then it fails as 401 "must be
authenticated" -- which reads exactly like terms that have not been accepted,
even when they have. Measured on stabilityai/stable-audio-open-1.0: the
account had access, a token on disk could fetch the weights with curl, and the
engine loading them could not.
"""

import os

import pytest

from codai import hf_cache


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN",
                 "CODERAI_HF_TOKEN_FILE"):
        monkeypatch.delenv(name, raising=False)


# ------------------------------------------------------------ the resolution

def test_the_environment_wins(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "from-env")
    assert hf_cache.configured_hf_token() == "from-env"


@pytest.mark.parametrize("name", ["HF_TOKEN", "HUGGINGFACE_TOKEN",
                                  "HUGGING_FACE_HUB_TOKEN"])
def test_every_name_the_libraries_use_is_accepted(monkeypatch, name):
    """A token under the one name a given library version ignores is the same
    as no token at all, so all three are read."""
    monkeypatch.setenv(name, "tok-" + name)
    assert hf_cache.configured_hf_token() == "tok-" + name


def test_a_token_file_is_read_when_the_environment_is_empty(tmp_path, monkeypatch):
    f = tmp_path / "hf_token"
    f.write_text("from-file\n")
    monkeypatch.setenv("CODERAI_HF_TOKEN_FILE", str(f))
    assert hf_cache.configured_hf_token() == "from-file"


def test_the_config_dir_file_is_read(tmp_path):
    (tmp_path / "hf_token").write_text("  from-config-dir  \n")
    assert hf_cache.configured_hf_token(str(tmp_path)) == "from-config-dir"


@pytest.fixture
def only_here(tmp_path, monkeypatch):
    """Search ONLY tmp_path, so "no token" can actually be tested.

    A config_dir does not restrict the search -- the fallbacks
    (~/.coderai/hf_token, /config/coderai/hf_token) are the point of the
    resolver -- and on a host that has one, a test asserting "none found"
    silently finds the real one.
    """
    monkeypatch.setattr(hf_cache, "_token_files",
                        lambda config_dir=None: [str(tmp_path / "hf_token")])
    return tmp_path


def test_an_empty_token_file_is_not_a_token(only_here):
    (only_here / "hf_token").write_text("   \n")
    assert hf_cache.configured_hf_token(str(only_here)) is None


def test_a_missing_file_is_not_an_error(monkeypatch):
    monkeypatch.setenv("CODERAI_HF_TOKEN_FILE", "/nonexistent/hf_token")
    hf_cache.configured_hf_token("/also/nonexistent")   # must not raise


# ------------------------------------------------------------- applying it

def test_applying_sets_both_canonical_names(tmp_path, monkeypatch):
    (tmp_path / "hf_token").write_text("abc123")
    assert hf_cache.apply_hf_token(str(tmp_path)) is True
    assert os.environ.get("HF_TOKEN") == "abc123"
    assert os.environ.get("HUGGING_FACE_HUB_TOKEN") == "abc123"


def test_applying_without_a_token_changes_nothing(only_here):
    assert hf_cache.apply_hf_token(str(only_here)) is False
    assert "HF_TOKEN" not in os.environ


def test_applying_is_idempotent(tmp_path):
    (tmp_path / "hf_token").write_text("abc123")
    assert hf_cache.apply_hf_token(str(tmp_path)) is True
    assert hf_cache.apply_hf_token(str(tmp_path)) is True
    assert os.environ.get("HF_TOKEN") == "abc123"


def test_the_cache_entry_point_applies_the_token_too():
    """Both answer "how does this process talk to the Hub", and both have to
    be set before huggingface_hub is imported."""
    import inspect
    src = inspect.getsource(hf_cache.apply_hf_cache_env)
    assert "apply_hf_token" in src, (
        "the token is no longer applied where the cache settings are, so a "
        "process can be pointed at the right cache and still be anonymous")


# ------------------------------------------------- one implementation, shared

def test_the_diarization_worker_uses_the_shared_resolver():
    """Two answers to "where is the token" is how one path authenticates and
    another does not."""
    import inspect
    from codai.api import pyannote_worker
    src = inspect.getsource(pyannote_worker._resolve_hf_token)
    assert "configured_hf_token" in src
    # Its own per-request override must still win over anything configured.
    assert pyannote_worker._resolve_hf_token({"hf_token": "per-request"}) == \
        "per-request"


def test_a_per_request_token_still_beats_the_configured_one(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "configured")
    from codai.api import pyannote_worker
    assert pyannote_worker._resolve_hf_token({"hf_token": "explicit"}) == "explicit"
    assert pyannote_worker._resolve_hf_token({}) == "configured"


def test_the_token_is_never_logged():
    """It ends up in request headers; it must not end up in the log too."""
    import inspect
    src = inspect.getsource(hf_cache)
    body = src[src.index("def configured_hf_token"):]
    body = body[:body.index("def apply_hf_cache_env")]
    for bad in ("print(", "_log.info", "logger.info", "logging.info"):
        assert bad not in body, f"the token resolver calls {bad}"
