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

"""The fork engine was renamed ``llamasrv_fork`` -> ``llamampere``.

Renaming an engine key is not a cosmetic change: the key is the word an
operator writes in ``config.json`` and in every ``models.json`` entry pinned to
that engine, and it is the word the image's binary variable is derived from. A
rename that only changes the code is a rename that SILENTLY unconfigures every
install that already uses it, and in each case the symptom is not an error:

* a ``config.json`` block under the old key reads as ``{}``, so the engine
  loads its defaults, reports itself disabled, and every model pinned to it
  stops being served -- with nothing to point at.
* an entry pinned ``backend: "llamasrv_fork"`` matches no engine, so
  ``resolve_engine_backend`` returns None and the model is answered by the
  ordinary in-process path instead. It still answers. On another engine.
* a per-model override block under the old key is not found, so the model is
  launched with the engine-wide template -- another model's context size and
  cache types, which looks like a working server.
* the binary variable moves with the key, so a baked image is not found and
  the engine tries to BUILD a fork at runtime, in a runtime image that has no
  CUDA toolchain.

So both spellings resolve, everywhere, and nothing has to be migrated.
"""

from __future__ import annotations

import pytest


# --------------------------------------------------------------------------- #
# the key itself

def test_the_old_key_normalises_to_the_new_one():
    from codai.models.manager import normalise_engine_key
    assert normalise_engine_key("llamasrv_fork") == "llamampere"
    assert normalise_engine_key("LLAMASRV_FORK") == "llamampere"
    assert normalise_engine_key("  llamasrv_fork  ") == "llamampere"


def test_the_new_key_is_left_alone():
    from codai.models.manager import normalise_engine_key
    assert normalise_engine_key("llamampere") == "llamampere"


def test_an_unrelated_engine_is_not_rewritten():
    """The table must not be a general-purpose rewriter."""
    from codai.models.manager import normalise_engine_key
    for other in ("llamasrv", "vllm", "ds4", "colibri", "k3", "kt", "runpod"):
        assert normalise_engine_key(other) == other


def test_the_engine_is_listed_under_its_new_key_only():
    from codai.models.manager import _ENGINE_BACKENDS
    assert "llamampere" in _ENGINE_BACKENDS
    assert "llamasrv_fork" not in _ENGINE_BACKENDS


def test_the_label_is_not_the_key():
    """THE POINT OF THE RENAME. The key has to be lowercase because a pin is
    lowercased before it is matched; the engine is still called llamAmpere."""
    from codai.config import LlamAmpereConfig
    assert LlamAmpereConfig.ENGINE_KEY == "llamampere"
    assert LlamAmpereConfig.ENGINE_LABEL == "llamAmpere"
    assert "llamasrv_fork" in LlamAmpereConfig.LEGACY_ENGINE_KEYS


# --------------------------------------------------------------------------- #
# config.json

def test_a_config_written_under_the_old_key_still_configures_the_engine():
    """THE SILENT ONE. Defaults load, the engine reports itself disabled, and
    every model pinned to it stops being served with no error anywhere."""
    from codai.config import _engine_block, LlamAmpereConfig
    old = {"llamasrv_fork": {"enabled": True, "git_ref": "v0.6.4",
                             "cache_type_k": "sjkvarn4"}}
    blk = _engine_block(old, LlamAmpereConfig)
    assert blk.get("enabled") is True
    assert blk.get("cache_type_k") == "sjkvarn4"


def test_the_new_key_wins_over_the_old_one():
    from codai.config import _engine_block, LlamAmpereConfig
    both = {"llamasrv_fork": {"git_ref": "stale"},
            "llamampere": {"git_ref": "current"}}
    assert _engine_block(both, LlamAmpereConfig).get("git_ref") == "current"


def test_an_empty_new_block_does_not_shadow_a_populated_old_one():
    """A config saved by a version that knew the key but had nothing in it must
    not hide the real settings sitting under the old name."""
    from codai.config import _engine_block, LlamAmpereConfig
    data = {"llamampere": {}, "llamasrv_fork": {"enabled": True}}
    assert _engine_block(data, LlamAmpereConfig).get("enabled") is True


def test_a_config_with_neither_key_is_empty_not_an_error():
    from codai.config import _engine_block, LlamAmpereConfig
    assert _engine_block({}, LlamAmpereConfig) == {}
    assert _engine_block({"vllm": {"enabled": True}}, LlamAmpereConfig) == {}


def test_the_stock_engine_has_no_legacy_key_to_fall_back_on():
    from codai.config import LlamaSrvConfig
    assert tuple(LlamaSrvConfig.LEGACY_ENGINE_KEYS) == ()


# --------------------------------------------------------------------------- #
# the image's binary variable

def test_both_binary_variables_are_accepted():
    from codai.api.llamasrv_worker import bin_env_vars
    from codai.config import LlamAmpereConfig
    names = bin_env_vars(LlamAmpereConfig())
    assert names[0] == "CODERAI_LLAMAMPERE_BIN"
    assert "CODERAI_LLAMASRV_FORK_BIN" in names


def test_a_binary_named_by_the_old_variable_is_found(monkeypatch):
    """An image baked before the rename exports the old variable. Missing it
    means building a fork at runtime, with no CUDA toolchain to do it with."""
    from codai.api import llamasrv_worker as w
    from codai.config import LlamAmpereConfig
    monkeypatch.delenv("CODERAI_LLAMAMPERE_BIN", raising=False)
    monkeypatch.setenv("CODERAI_LLAMASRV_FORK_BIN", "/opt/baked/llama-server")
    assert w._env_bin_for(LlamAmpereConfig()) == "/opt/baked/llama-server"


def test_the_current_variable_wins(monkeypatch):
    from codai.api import llamasrv_worker as w
    from codai.config import LlamAmpereConfig
    monkeypatch.setenv("CODERAI_LLAMAMPERE_BIN", "/new/llama-server")
    monkeypatch.setenv("CODERAI_LLAMASRV_FORK_BIN", "/old/llama-server")
    assert w._env_bin_for(LlamAmpereConfig()) == "/new/llama-server"


def test_the_stock_engine_does_not_pick_up_the_forks_variable(monkeypatch):
    """One variable for both would serve whichever was baked last to every
    pinned model, which is what the two-engine split exists to prevent."""
    from codai.api import llamasrv_worker as w
    from codai.config import LlamaSrvConfig
    monkeypatch.delenv("CODERAI_LLAMASRV_BIN", raising=False)
    monkeypatch.setenv("CODERAI_LLAMAMPERE_BIN", "/fork/llama-server")
    monkeypatch.setenv("CODERAI_LLAMASRV_FORK_BIN", "/fork/llama-server")
    assert w._env_bin_for(LlamaSrvConfig()) == ""


# --------------------------------------------------------------------------- #
# the front and the manager must agree

def test_the_front_routes_the_old_pin_to_the_same_engine():
    """Two independent spellings-to-engine mappings is how the front and the
    manager come to disagree about who serves a model."""
    from codai.frontproxy.router import required_capability
    assert required_capability("any-model", "/v1/chat/completions",
                               backend="llamasrv_fork") == "llamampere"
    assert required_capability("any-model", "/v1/chat/completions",
                               backend="llamampere") == "llamampere"


def test_the_per_model_override_block_is_read_under_either_key():
    from codai.backends.llamasrv import _LLAMASRV_FIELDS
    from codai.config import LlamAmpereConfig
    # The fields a per-model block may refine are the engine's own, so a block
    # written under the old key is applied by the same whitelist.
    for f in ("ctx", "cache_type_k", "cache_type_v", "extra_args"):
        assert f in _LLAMASRV_FIELDS
    assert "llamasrv_fork" in LlamAmpereConfig.LEGACY_ENGINE_KEYS


# --------------------------------------------------------------------------- #
# nothing kept the old spelling by accident

@pytest.mark.parametrize("path", [
    "codai/models/manager.py",
    "codai/config.py",
    "codai/frontproxy/router.py",
    "codai/frontproxy/assignment.py",
    "codai/frontproxy/engine_supervisor.py",
    "codai/backends/llamasrv.py",
    "codai/api/llamasrv_worker.py",
    "codai/admin/routes.py",
])
def test_the_old_key_survives_only_where_compatibility_needs_it(path):
    """Every remaining mention must be the compatibility path, not a missed
    rename: a half-renamed engine routes in one place and not another."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    text = (root / path).read_text()
    for n, line in enumerate(text.splitlines(), 1):
        if "llamasrv_fork" not in line:
            continue
        assert ("LEGACY_ENGINE_KEYS" in line
                or "_ENGINE_KEY_ALIASES" in line), f"{path}:{n}: {line.strip()}"


# --------------------------------------------------------------------------- #
# the pin has to resolve where it is ENFORCED, not only where it is checked

def test_a_legacy_pin_routes_to_the_engine_that_answers_to_it():
    """MEASURED ON A LIVE INSTALL, 05:35:33. required_capability agreed and the
    assignment handed qwen38cyber to 'llamampere', and then pick_engine looked
    up the raw pin, found no engine of that name, and failed the request:

        [front] WARNING: model 'qwen38cyber' is pinned to 'llamasrv_fork' but no
        engine named/backed 'llamasrv_fork' is declared; failing the request
        (the pin is a hard constraint).

    A 503 on a model the engine had already claimed. Normalising in two of the
    three places is the same as not normalising at all."""
    from codai.frontproxy.registry import Engine, EngineRegistry
    from codai.frontproxy.router import pick_engine
    reg = EngineRegistry()
    e = Engine(id=0, gpu=None, port=1, name="llamampere", backend="llamampere",
               primary=True, capabilities={"gguf"})
    reg.add(e)
    reg.update_state(0, healthy=True, loaded_models=["qwen38cyber"])
    assert pick_engine(reg, "/v1/chat/completions", "POST", "qwen38cyber",
                       "gguf", pinned="llamasrv_fork") is e
    assert pick_engine(reg, "/v1/chat/completions", "POST", "qwen38cyber",
                       "gguf", pinned="llamampere") is e


def test_a_pin_naming_nothing_still_fails():
    """The normaliser must not become a way for any pin to match something."""
    from codai.frontproxy.registry import Engine, EngineRegistry
    from codai.frontproxy.router import pick_engine
    reg = EngineRegistry()
    e = Engine(id=0, gpu=None, port=1, name="llamampere", backend="llamampere",
               primary=True, capabilities={"gguf"})
    reg.add(e)
    reg.update_state(0, healthy=True, loaded_models=["qwen38cyber"])
    assert pick_engine(reg, "/v1/chat/completions", "POST", "qwen38cyber",
                       "gguf", pinned="no_such_engine") is None
