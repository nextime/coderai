from tools.manual_multimodal_test_client import (
    MODE_DEFAULTS,
    build_request_spec,
    build_parser,
    choose_mode_interactively,
    execute_request,
    handle_response_payload,
    parse_args,
    resolve_mode_config,
)

import pytest


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


def test_parse_args_rejects_generic_file_argument():
    try:
        parse_args(["llm", "--file", "sample.dat"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("parse_args accepted deprecated --file argument")


def test_resolve_mode_config_uses_mode_defaults_when_overrides_absent(tmp_path):
    args = parse_args(["audio-generation", "--output-dir", str(tmp_path)])

    config = resolve_mode_config(args, selected_mode="audio-generation")

    assert config["mode"] == "audio-generation"
    assert config["url"] == "http://127.0.0.1:6745"
    assert config["model"] == MODE_DEFAULTS["audio-generation"]["model"]
    assert config["prompt"] == MODE_DEFAULTS["audio-generation"]["prompt"]
    assert config["response_format"] == MODE_DEFAULTS["audio-generation"]["response_format"]
    assert config["output_dir"] == tmp_path
    assert "file" not in config


def test_resolve_mode_config_normalizes_url_and_uses_default_input_files(tmp_path):
    args = parse_args([
        "transcription",
        "--url", "http://127.0.0.1:6745/",
        "--output-dir", str(tmp_path),
    ])

    config = resolve_mode_config(args, selected_mode="transcription")

    assert config["url"] == "http://127.0.0.1:6745"
    assert config["audio_file"] == MODE_DEFAULTS["transcription"]["audio_file"]


def test_resolve_mode_config_uses_default_video_file_for_video_doubt(tmp_path):
    args = parse_args(["video-doubt", "--output-dir", str(tmp_path)])

    config = resolve_mode_config(args, selected_mode="video-doubt")

    assert config["video_file"] == MODE_DEFAULTS["video-doubt"]["video_file"]


def test_resolve_mode_config_prefers_explicit_model_prompt_url_and_response_format_overrides(tmp_path):
    args = parse_args([
        "video-generation",
        "--url", "http://example.invalid:9999/",
        "--model", "video:custom-model",
        "--prompt", "Custom prompt",
        "--response-format", "b64_json",
        "--output-dir", str(tmp_path),
    ])

    config = resolve_mode_config(args, selected_mode="video-generation")

    assert config["url"] == "http://example.invalid:9999"
    assert config["model"] == "video:custom-model"
    assert config["prompt"] == "Custom prompt"
    assert config["response_format"] == "b64_json"
    assert config["output_dir"] == tmp_path


def test_resolve_mode_config_keeps_explicit_empty_string_overrides(tmp_path):
    args = parse_args([
        "video-generation",
        "--model", "",
        "--audio-file", "",
        "--video-file", "",
        "--response-format", "",
        "--output-dir", str(tmp_path),
    ])

    config = resolve_mode_config(args, selected_mode="video-generation")

    assert config["model"] == ""
    assert config["audio_file"] == ""
    assert config["video_file"] == ""
    assert config["response_format"] == ""


def test_build_request_spec_for_llm_uses_chat_completions_payload(tmp_path):
    config = {
        "mode": "llm",
        "url": "http://127.0.0.1:6745",
        "model": "text:test",
        "prompt": "Ping",
        "output_dir": tmp_path,
        "token": None,
        "audio_file": None,
        "video_file": None,
        "response_format": None,
    }

    spec = build_request_spec(config)

    assert spec == {
        "method": "POST",
        "url": "http://127.0.0.1:6745/v1/chat/completions",
        "headers": {"Accept": "application/json"},
        "json": {
            "model": "text:test",
            "messages": [{"role": "user", "content": "Ping"}],
        },
    }


def test_build_request_spec_for_transcription_uses_multipart_file(tmp_path):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"wav-bytes")
    config = {
        "mode": "transcription",
        "url": "http://127.0.0.1:6745",
        "model": "audio:test",
        "prompt": "Transcribe carefully",
        "output_dir": tmp_path,
        "token": None,
        "audio_file": str(audio_path),
        "video_file": None,
        "response_format": None,
    }

    spec = build_request_spec(config)

    assert spec == {
        "method": "POST",
        "url": "http://127.0.0.1:6745/v1/audio/transcriptions",
        "headers": {"Accept": "application/json"},
        "data": {
            "model": "audio:test",
            "prompt": "Transcribe carefully",
        },
        "files": {
            "file": ("sample.wav", b"wav-bytes"),
        },
    }


def test_build_request_spec_for_audio_generation_uses_json_payload(tmp_path):
    config = {
        "mode": "audio-generation",
        "url": "http://127.0.0.1:6745",
        "model": "audio_gen:test",
        "prompt": "Generate a bell sound",
        "output_dir": tmp_path,
        "token": None,
        "audio_file": None,
        "video_file": None,
        "response_format": "url",
    }

    spec = build_request_spec(config)

    assert spec == {
        "method": "POST",
        "url": "http://127.0.0.1:6745/v1/audio/generate",
        "headers": {"Accept": "application/json"},
        "json": {
            "model": "audio_gen:test",
            "prompt": "Generate a bell sound",
            "response_format": "url",
        },
    }


def test_build_request_spec_for_video_generation_uses_json_payload(tmp_path):
    config = {
        "mode": "video-generation",
        "url": "http://127.0.0.1:6745",
        "model": "video:test",
        "prompt": "Generate a short test clip",
        "output_dir": tmp_path,
        "token": None,
        "audio_file": None,
        "video_file": None,
        "response_format": "url",
    }

    spec = build_request_spec(config)

    assert spec == {
        "method": "POST",
        "url": "http://127.0.0.1:6745/v1/video/generations",
        "headers": {"Accept": "application/json"},
        "json": {
            "model": "video:test",
            "prompt": "Generate a short test clip",
            "response_format": "url",
        },
    }


def test_build_request_spec_for_video_doubt_request_builder_uses_text_endpoint_with_video_context(tmp_path):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video-bytes")
    config = {
        "mode": "video-doubt",
        "url": "http://127.0.0.1:6745",
        "model": "vision:test",
        "prompt": "What happens in this clip?",
        "output_dir": tmp_path,
        "token": None,
        "audio_file": None,
        "video_file": str(video_path),
        "response_format": None,
    }

    spec = build_request_spec(config)

    assert spec["url"].endswith("/v1/chat/completions")
    assert spec["json"]["model"] == "vision:test"
    assert str(video_path) in spec["json"]["messages"][0]["content"]
    assert "What happens in this clip?" in spec["json"]["messages"][0]["content"]
    assert "local path reference only" in spec["json"]["messages"][0]["content"]
    assert "may or may not support reasoning from that reference" in spec["json"]["messages"][0]["content"]
    assert "acknowledge that limitation" in spec["json"]["messages"][0]["content"]


def test_build_request_spec_for_music_audio_doubt_request_builder_uses_text_endpoint_with_audio_context(tmp_path):
    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"audio-bytes")
    config = {
        "mode": "music-audio-doubt",
        "url": "http://127.0.0.1:6745",
        "model": "audio:test",
        "prompt": "Describe the music.",
        "output_dir": tmp_path,
        "token": None,
        "audio_file": str(audio_path),
        "video_file": None,
        "response_format": None,
    }

    spec = build_request_spec(config)

    assert spec["url"].endswith("/v1/chat/completions")
    assert spec["json"]["model"] == "audio:test"
    assert str(audio_path) in spec["json"]["messages"][0]["content"]
    assert "Describe the music." in spec["json"]["messages"][0]["content"]
    assert "local path reference only" in spec["json"]["messages"][0]["content"]
    assert "may or may not support reasoning from that reference" in spec["json"]["messages"][0]["content"]
    assert "acknowledge that limitation" in spec["json"]["messages"][0]["content"]


def test_build_request_spec_for_video_doubt_request_builder_requires_video_file_flag(tmp_path):
    config = {
        "mode": "video-doubt",
        "url": "http://127.0.0.1:6745",
        "model": "vision:test",
        "prompt": "What happens in this clip?",
        "output_dir": tmp_path,
        "token": None,
        "audio_file": None,
        "video_file": None,
        "response_format": None,
    }

    with pytest.raises(FileNotFoundError, match=r"Missing required file\. Supply --video-file\."):
        build_request_spec(config)


def test_build_request_spec_for_music_audio_doubt_request_builder_requires_audio_file_flag(tmp_path):
    config = {
        "mode": "music-audio-doubt",
        "url": "http://127.0.0.1:6745",
        "model": "audio:test",
        "prompt": "Describe the music.",
        "output_dir": tmp_path,
        "token": None,
        "audio_file": None,
        "video_file": None,
        "response_format": None,
    }

    with pytest.raises(FileNotFoundError, match=r"Missing required file\. Supply --audio-file\."):
        build_request_spec(config)


def test_build_request_spec_for_transcription_requires_audio_file_flag(tmp_path):
    config = {
        "mode": "transcription",
        "url": "http://127.0.0.1:6745",
        "model": "audio:test",
        "prompt": "Transcribe carefully",
        "output_dir": tmp_path,
        "token": None,
        "audio_file": None,
        "video_file": None,
        "response_format": None,
    }

    with pytest.raises(FileNotFoundError, match=r"Missing required file\. Supply --audio-file\."):
        build_request_spec(config)


def test_build_request_spec_for_transcription_requires_existing_audio_file(tmp_path):
    missing_path = tmp_path / "missing.wav"
    config = {
        "mode": "transcription",
        "url": "http://127.0.0.1:6745",
        "model": "audio:test",
        "prompt": "Transcribe carefully",
        "output_dir": tmp_path,
        "token": None,
        "audio_file": str(missing_path),
        "video_file": None,
        "response_format": None,
    }

    with pytest.raises(FileNotFoundError, match=rf"File not found: {missing_path}\. Supply --audio-file\."):
        build_request_spec(config)


class DummyResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = "payload-text"

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_task5_handle_response_payload_returns_llm_text_without_artifact(tmp_path):
    payload = {
        "choices": [{"message": {"content": "hello from model"}}]
    }
    response = DummyResponse(payload)

    result = handle_response_payload("llm", response, tmp_path)

    assert result["text"] == "hello from model"
    assert result["artifact_path"] is None
    assert result["payload"] == payload


def test_task5_handle_response_payload_downloads_url_artifact(monkeypatch, tmp_path):
    payload = {
        "data": [{"url": "http://example.invalid/audio.wav", "text": "generated audio summary"}]
    }
    response = DummyResponse(payload)
    monkeypatch.setattr(
        "tools.manual_multimodal_test_client._download_artifact",
        lambda url: b"wave-bytes",
    )

    result = handle_response_payload("audio-generation", response, tmp_path)

    assert result["artifact_path"].suffix == ".wav"
    assert result["artifact_path"].read_bytes() == b"wave-bytes"
    assert result["text"] == "generated audio summary"
    assert result["payload"] == payload


def test_task5_handle_response_payload_decodes_base64_artifact(tmp_path):
    payload = {
        "data": [{"b64_json": "aGVsbG8=", "caption": "inline audio artifact"}]
    }
    response = DummyResponse(payload)

    result = handle_response_payload("audio-generation", response, tmp_path)

    assert result["artifact_path"].read_bytes() == b"hello"
    assert result["text"] == "inline audio artifact"
    assert result["payload"] == payload


def test_task5_handle_response_payload_downloads_video_generation_artifact_as_mp4(monkeypatch, tmp_path):
    payload = {
        "data": [{"url": "http://example.invalid/video.mp4", "caption": "generated video summary"}]
    }
    response = DummyResponse(payload)
    monkeypatch.setattr(
        "tools.manual_multimodal_test_client._download_artifact",
        lambda url: b"video-bytes",
    )

    result = handle_response_payload("video-generation", response, tmp_path)

    assert result["artifact_path"].suffix == ".mp4"
    assert result["artifact_path"].read_bytes() == b"video-bytes"
    assert result["text"] == "generated video summary"
    assert result["payload"] == payload


def test_task5_execute_request_forwards_method_url_timeout_and_kwargs(monkeypatch):
    captured = {}

    def fake_request(*, method, url, timeout, **kwargs):
        captured["method"] = method
        captured["url"] = url
        captured["timeout"] = timeout
        captured["kwargs"] = kwargs
        return "response-object"

    monkeypatch.setattr("tools.manual_multimodal_test_client.requests.request", fake_request)

    result = execute_request({
        "method": "POST",
        "url": "http://127.0.0.1:6745/v1/audio/generate",
        "headers": {"Accept": "application/json"},
        "json": {"prompt": "Ping"},
    })

    assert result == "response-object"
    assert captured == {
        "method": "POST",
        "url": "http://127.0.0.1:6745/v1/audio/generate",
        "timeout": 300,
        "kwargs": {
            "headers": {"Accept": "application/json"},
            "json": {"prompt": "Ping"},
        },
    }


def test_task5_execute_request_filters_method_and_url_from_forwarded_kwargs(monkeypatch):
    captured = {}

    def fake_request(*, method, url, timeout, **kwargs):
        captured["method"] = method
        captured["url"] = url
        captured["timeout"] = timeout
        captured["kwargs"] = kwargs
        return "response-object"

    monkeypatch.setattr("tools.manual_multimodal_test_client.requests.request", fake_request)

    result = execute_request({
        "method": "GET",
        "url": "http://127.0.0.1:6745/health",
        "params": {"verbose": "1"},
    })

    assert result == "response-object"
    assert captured["method"] == "GET"
    assert captured["url"] == "http://127.0.0.1:6745/health"
    assert captured["timeout"] == 300
    assert captured["kwargs"] == {"params": {"verbose": "1"}}
    assert "method" not in captured["kwargs"]
    assert "url" not in captured["kwargs"]
