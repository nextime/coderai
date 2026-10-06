"""Where the township tool gets its CoderAI Bearer token.

CoderAI gates every /v1 endpoint and makes no exemption for loopback, so the
bundled tools — which call 127.0.0.1 from inside the same container — need a token
exactly like a remote client. The container launches this tool WITHOUT --api-key,
so "train a video LoRA" came back as:

    POST /v1/loras/train -> 401: Invalid API key. Provide a valid Bearer token.

Fourteen call sites built their client straight from default_args.api_key, which
was None. They now go through one resolver with a defined order.
"""
import argparse
import importlib.util
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "tw_key", ROOT / "tools" / "gen_township_fighters.py")
TW = importlib.util.module_from_spec(_spec)
sys.modules["tw_key"] = TW
_spec.loader.exec_module(TW)


def _args(**kw):
    kw.setdefault("api_key", None)
    kw.setdefault("config", None)
    kw.setdefault("out_dir", None)
    return argparse.Namespace(**kw)


def test_the_running_options_win(tmp_path, monkeypatch):
    (tmp_path / "township_config.json").write_text(json.dumps({"api_key": "on-disk"}))
    monkeypatch.setenv("CODERAI_API_KEY", "in-env")
    assert TW.resolve_api_key(_args(api_key="live", out_dir=str(tmp_path))) == "live"


def test_a_saved_config_is_used_when_nothing_was_passed(tmp_path):
    """The Connection card writes here, so pasting a key and pressing Save config
    is enough — no restart, no CLI flag."""
    (tmp_path / "township_config.json").write_text(json.dumps({"api_key": "on-disk"}))
    assert TW.resolve_api_key(_args(out_dir=str(tmp_path))) == "on-disk"


def test_an_explicit_config_path_is_honoured(tmp_path):
    cfg = tmp_path / "elsewhere.json"
    cfg.write_text(json.dumps({"api_key": "explicit"}))
    assert TW.resolve_api_key(_args(config=str(cfg))) == "explicit"


def test_the_environment_is_the_last_resort(tmp_path, monkeypatch):
    """How a launcher can supply the token without editing a config."""
    monkeypatch.setenv("CODERAI_API_KEY", "in-env")
    assert TW.resolve_api_key(_args(out_dir=str(tmp_path))) == "in-env"


def test_no_key_anywhere_is_none_not_an_empty_string(tmp_path, monkeypatch):
    """An empty string would set 'Authorization: Bearer ' and fail differently."""
    monkeypatch.delenv("CODERAI_API_KEY", raising=False)
    assert TW.resolve_api_key(_args(out_dir=str(tmp_path))) is None
    assert TW.resolve_api_key(None) is None


@pytest.mark.parametrize("value", ["", "   ", None])
def test_a_blank_key_is_not_treated_as_a_key(tmp_path, monkeypatch, value):
    monkeypatch.delenv("CODERAI_API_KEY", raising=False)
    (tmp_path / "township_config.json").write_text(json.dumps({"api_key": value}))
    assert TW.resolve_api_key(_args(api_key=value, out_dir=str(tmp_path))) is None


def test_an_unreadable_config_does_not_crash(tmp_path, monkeypatch):
    monkeypatch.delenv("CODERAI_API_KEY", raising=False)
    (tmp_path / "township_config.json").write_text("{ not json")
    assert TW.resolve_api_key(_args(out_dir=str(tmp_path))) is None


def test_every_client_goes_through_the_resolver():
    """The bug was 14 call sites each reading default_args.api_key directly."""
    src = (ROOT / "tools" / "gen_township_fighters.py").read_text(encoding="utf-8")
    assert 'getattr(default_args, "api_key", None))' not in src
    assert "getattr(default_args, 'api_key', None)," not in src
    assert src.count("resolve_api_key(default_args)") >= 14


def test_a_401_says_what_to_do_about_it():
    """The raw body ("Invalid API key. Provide a valid Bearer token.") tells an
    operator nothing about WHERE to put one in this tool."""
    src = (ROOT / "tools" / "gen_township_fighters.py").read_text(encoding="utf-8")
    assert "has no API key" in src
    assert "Save config" in src
    assert "CODERAI_API_KEY" in src
