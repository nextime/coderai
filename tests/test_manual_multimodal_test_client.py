from tools.manual_multimodal_test_client import (
    MODE_DEFAULTS,
    build_parser,
    choose_mode_interactively,
    parse_args,
    resolve_mode_config,
)


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


def test_resolve_mode_config_uses_mode_defaults_when_overrides_absent(tmp_path):
    args = parse_args(["audio-generation", "--output-dir", str(tmp_path)])

    config = resolve_mode_config(args, selected_mode="audio-generation")

    assert config["mode"] == "audio-generation"
    assert config["url"] == "http://127.0.0.1:6745"
    assert config["model"] == MODE_DEFAULTS["audio-generation"]["model"]
    assert config["prompt"] == MODE_DEFAULTS["audio-generation"]["prompt"]
    assert config["output_dir"] == tmp_path


def test_resolve_mode_config_prefers_explicit_model_prompt_and_url_overrides(tmp_path):
    args = parse_args([
        "video-generation",
        "--url", "http://example.invalid:9999",
        "--model", "video:custom-model",
        "--prompt", "Custom prompt",
        "--output-dir", str(tmp_path),
    ])

    config = resolve_mode_config(args, selected_mode="video-generation")

    assert config["url"] == "http://example.invalid:9999"
    assert config["model"] == "video:custom-model"
    assert config["prompt"] == "Custom prompt"
    assert config["output_dir"] == tmp_path
