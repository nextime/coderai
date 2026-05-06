from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.manual_multimodal_test_client import build_parser, choose_mode_interactively, parse_args


def test_parse_args_accepts_direct_mode_and_global_overrides():
    args = parse_args([
        "llm",
        "--url", "http://127.0.0.1:6745",
        "--token", "secret-token",
        "--model", "text:test-model",
        "--prompt", "Say hello",
        "--output-dir", "tmp/out",
    ])

    assert args.mode == "llm"
    assert args.url == "http://127.0.0.1:6745"
    assert args.token == "secret-token"
    assert args.model == "text:test-model"
    assert args.prompt == "Say hello"
    assert args.output_dir == "tmp/out"


def test_choose_mode_interactively_maps_numeric_selection(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "2")

    mode = choose_mode_interactively()

    assert mode == "transcription"


def test_parse_args_leaves_mode_empty_for_interactive_fallback():
    args = parse_args([])

    assert args.mode is None
