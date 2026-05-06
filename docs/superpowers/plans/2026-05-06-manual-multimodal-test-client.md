# Manual Multimodal Test Client Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone Python test client that can run one manual smoke-test request at a time for LLM, transcription, audio generation, video generation, video doubt, and music audio doubt with built-in defaults and explicit overrides.

**Architecture:** Implement a single focused CLI script under `tools/` using `argparse`, a mode registry, shared request/response helpers, and predictable local artifact saving. Keep the script independent from server modules, and cover it with unit tests that validate argument resolution, request construction, artifact handling, interactive fallback behavior, and the constrained “doubt” mode payloads using mocked HTTP responses.

**Tech Stack:** Python, argparse, requests, pathlib, base64, unittest.mock/pytest

---

### Task 1: Establish script location, mode registry, and CLI skeleton

**Files:**
- Create: `tools/manual_multimodal_test_client.py`
- Create: `tests/test_manual_multimodal_test_client.py`

- [ ] **Step 1: Write the failing parser/mode tests**

Create `tests/test_manual_multimodal_test_client.py` with initial tests for direct mode parsing, interactive fallback selection, and global override parsing:

```python
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
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `/storage/coderai/venv_all/bin/python -m pytest tests/test_manual_multimodal_test_client.py -k 'parse_args or choose_mode_interactively'`
Expected: FAIL with import errors because `tools/manual_multimodal_test_client.py` does not exist yet.

- [ ] **Step 3: Write the minimal CLI skeleton**

Create `tools/manual_multimodal_test_client.py` with the mode list, parser builder, parse helper, and interactive selector:

```python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional


MODES = [
    "llm",
    "transcription",
    "audio-generation",
    "video-generation",
    "video-doubt",
    "music-audio-doubt",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manual multimodal smoke-test client")
    parser.add_argument("mode", nargs="?", choices=MODES)
    parser.add_argument("--url", default="http://127.0.0.1:6745")
    parser.add_argument("--token", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--output-dir", default="tmp/manual-client-output")
    parser.add_argument("--file", default=None)
    parser.add_argument("--audio-file", default=None)
    parser.add_argument("--video-file", default=None)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def choose_mode_interactively() -> str:
    for idx, mode in enumerate(MODES, start=1):
        print(f"{idx}. {mode}")
    raw = input("Choose mode: ").strip()
    selected = int(raw)
    if selected < 1 or selected > len(MODES):
        raise ValueError(f"Invalid mode selection: {raw}")
    return MODES[selected - 1]
```

- [ ] **Step 4: Run the focused tests to verify they pass**

Run: `/storage/coderai/venv_all/bin/python -m pytest tests/test_manual_multimodal_test_client.py -k 'parse_args or choose_mode_interactively'`
Expected: PASS.

- [ ] **Step 5: Commit the CLI skeleton**

```bash
git add tools/manual_multimodal_test_client.py tests/test_manual_multimodal_test_client.py
git commit -m "feat: scaffold manual multimodal test client"
```

### Task 2: Add per-mode defaults and resolved configuration helpers

**Files:**
- Modify: `tools/manual_multimodal_test_client.py`
- Modify: `tests/test_manual_multimodal_test_client.py`

- [ ] **Step 1: Write the failing default-resolution tests**

Extend `tests/test_manual_multimodal_test_client.py` with tests for per-mode defaults and explicit override precedence:

```python
from tools.manual_multimodal_test_client import MODE_DEFAULTS, resolve_mode_config


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
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `/storage/coderai/venv_all/bin/python -m pytest tests/test_manual_multimodal_test_client.py -k 'resolve_mode_config or MODE_DEFAULTS'`
Expected: FAIL because `MODE_DEFAULTS` and `resolve_mode_config` are not implemented yet.

- [ ] **Step 3: Implement the defaults registry and resolver**

Add to `tools/manual_multimodal_test_client.py` a central defaults mapping and a resolver that returns normalized config values:

```python
MODE_DEFAULTS = {
    "llm": {
        "model": "text",
        "prompt": "Reply with a short test acknowledgement.",
    },
    "transcription": {
        "model": "audio",
        "prompt": None,
        "audio_file": "samples/transcription.wav",
    },
    "audio-generation": {
        "model": "audio_gen",
        "prompt": "Generate a short calm ambient audio test clip.",
        "response_format": "url",
    },
    "video-generation": {
        "model": "video",
        "prompt": "Generate a short test clip of a rotating cube.",
        "response_format": "url",
    },
    "video-doubt": {
        "model": "vision",
        "prompt": "Describe the important events in this video.",
        "video_file": "samples/question-video.mp4",
    },
    "music-audio-doubt": {
        "model": "audio",
        "prompt": "Describe the music and prominent sounds in this audio.",
        "audio_file": "samples/question-audio.wav",
    },
}


def resolve_mode_config(args: argparse.Namespace, selected_mode: str) -> dict:
    defaults = MODE_DEFAULTS[selected_mode]
    return {
        "mode": selected_mode,
        "url": args.url.rstrip("/"),
        "token": args.token,
        "model": args.model or defaults.get("model"),
        "prompt": args.prompt if args.prompt is not None else defaults.get("prompt"),
        "output_dir": Path(args.output_dir),
        "file": args.file,
        "audio_file": args.audio_file or defaults.get("audio_file"),
        "video_file": args.video_file or defaults.get("video_file"),
        "response_format": defaults.get("response_format"),
    }
```

- [ ] **Step 4: Run the focused tests to verify they pass**

Run: `/storage/coderai/venv_all/bin/python -m pytest tests/test_manual_multimodal_test_client.py -k 'resolve_mode_config or MODE_DEFAULTS'`
Expected: PASS.

- [ ] **Step 5: Commit the defaults layer**

```bash
git add tools/manual_multimodal_test_client.py tests/test_manual_multimodal_test_client.py
git commit -m "feat: add multimodal client defaults"
```

### Task 3: Add request builders for llm, transcription, audio generation, and video generation

**Files:**
- Modify: `tools/manual_multimodal_test_client.py`
- Modify: `tests/test_manual_multimodal_test_client.py`

- [ ] **Step 1: Write the failing request-construction tests**

Add tests for the four directly-supported endpoint payload builders:

```python
from tools.manual_multimodal_test_client import build_request_spec


def test_build_request_spec_for_llm_uses_chat_completions_payload(tmp_path):
    config = {
        "mode": "llm",
        "url": "http://127.0.0.1:6745",
        "model": "text:test",
        "prompt": "Ping",
        "output_dir": tmp_path,
        "token": None,
        "file": None,
        "audio_file": None,
        "video_file": None,
        "response_format": None,
    }

    spec = build_request_spec(config)

    assert spec["method"] == "POST"
    assert spec["url"] == "http://127.0.0.1:6745/v1/chat/completions"
    assert spec["json"]["model"] == "text:test"
    assert spec["json"]["messages"][0]["content"] == "Ping"


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
        "file": None,
        "audio_file": str(audio_path),
        "video_file": None,
        "response_format": None,
    }

    spec = build_request_spec(config)

    assert spec["url"].endswith("/v1/audio/transcriptions")
    assert spec["data"]["model"] == "audio:test"
    assert spec["data"]["prompt"] == "Transcribe carefully"
    assert spec["files"]["file"][0] == "sample.wav"
    assert spec["files"]["file"][1] == b"wav-bytes"


def test_build_request_spec_for_audio_generation_uses_json_payload(tmp_path):
    config = {
        "mode": "audio-generation",
        "url": "http://127.0.0.1:6745",
        "model": "audio_gen:test",
        "prompt": "Generate a bell sound",
        "output_dir": tmp_path,
        "token": None,
        "file": None,
        "audio_file": None,
        "video_file": None,
        "response_format": "url",
    }

    spec = build_request_spec(config)

    assert spec["url"].endswith("/v1/audio/generate")
    assert spec["json"]["model"] == "audio_gen:test"
    assert spec["json"]["prompt"] == "Generate a bell sound"
    assert spec["json"]["response_format"] == "url"


def test_build_request_spec_for_video_generation_uses_json_payload(tmp_path):
    config = {
        "mode": "video-generation",
        "url": "http://127.0.0.1:6745",
        "model": "video:test",
        "prompt": "Generate a short test clip",
        "output_dir": tmp_path,
        "token": None,
        "file": None,
        "audio_file": None,
        "video_file": None,
        "response_format": "url",
    }

    spec = build_request_spec(config)

    assert spec["url"].endswith("/v1/video/generations")
    assert spec["json"]["model"] == "video:test"
    assert spec["json"]["prompt"] == "Generate a short test clip"
    assert spec["json"]["response_format"] == "url"
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `/storage/coderai/venv_all/bin/python -m pytest tests/test_manual_multimodal_test_client.py -k 'build_request_spec and (llm or transcription or audio_generation or video_generation)'`
Expected: FAIL because `build_request_spec` does not exist yet.

- [ ] **Step 3: Implement the request builders and required-file checks**

Add `build_request_spec()` and a helper for reading required local files into `tools/manual_multimodal_test_client.py`:

```python
def _require_file(path_value: str | None, flag_name: str) -> Path:
    if not path_value:
        raise FileNotFoundError(f"Missing required file. Supply {flag_name}.")
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}. Supply {flag_name}.")
    return path


def build_request_spec(config: dict) -> dict:
    mode = config["mode"]
    headers = {"Accept": "application/json"}
    if config.get("token"):
        headers["Authorization"] = f"Bearer {config['token']}"

    if mode == "llm":
        return {
            "method": "POST",
            "url": f"{config['url']}/v1/chat/completions",
            "headers": headers,
            "json": {
                "model": config["model"],
                "messages": [{"role": "user", "content": config["prompt"]}],
            },
        }

    if mode == "transcription":
        audio_path = _require_file(config.get("audio_file"), "--audio-file")
        return {
            "method": "POST",
            "url": f"{config['url']}/v1/audio/transcriptions",
            "headers": headers,
            "data": {
                "model": config["model"],
                "prompt": config["prompt"],
            },
            "files": {
                "file": (audio_path.name, audio_path.read_bytes()),
            },
        }

    if mode == "audio-generation":
        return {
            "method": "POST",
            "url": f"{config['url']}/v1/audio/generate",
            "headers": headers,
            "json": {
                "model": config["model"],
                "prompt": config["prompt"],
                "response_format": config["response_format"] or "url",
            },
        }

    if mode == "video-generation":
        return {
            "method": "POST",
            "url": f"{config['url']}/v1/video/generations",
            "headers": headers,
            "json": {
                "model": config["model"],
                "prompt": config["prompt"],
                "response_format": config["response_format"] or "url",
            },
        }

    raise ValueError(f"Unsupported mode for this task: {mode}")
```

- [ ] **Step 4: Run the focused tests to verify they pass**

Run: `/storage/coderai/venv_all/bin/python -m pytest tests/test_manual_multimodal_test_client.py -k 'build_request_spec and (llm or transcription or audio_generation or video_generation)'`
Expected: PASS.

- [ ] **Step 5: Commit the direct endpoint builders**

```bash
git add tools/manual_multimodal_test_client.py tests/test_manual_multimodal_test_client.py
git commit -m "feat: add multimodal client request builders"
```

### Task 4: Implement constrained video-doubt and music-audio-doubt request modes

**Files:**
- Modify: `tools/manual_multimodal_test_client.py`
- Modify: `tests/test_manual_multimodal_test_client.py`

- [ ] **Step 1: Write the failing doubt-mode payload tests**

Add tests that lock the first-version behavior to a proven text-endpoint-compatible contract using prompt plus file-path context text rather than inventing unsupported binary chat transport:

```python
def test_build_request_spec_for_video_doubt_uses_text_endpoint_with_video_context(tmp_path):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video-bytes")
    config = {
        "mode": "video-doubt",
        "url": "http://127.0.0.1:6745",
        "model": "vision:test",
        "prompt": "What happens in this clip?",
        "output_dir": tmp_path,
        "token": None,
        "file": None,
        "audio_file": None,
        "video_file": str(video_path),
        "response_format": None,
    }

    spec = build_request_spec(config)

    assert spec["url"].endswith("/v1/chat/completions")
    assert spec["json"]["model"] == "vision:test"
    assert str(video_path) in spec["json"]["messages"][0]["content"]
    assert "What happens in this clip?" in spec["json"]["messages"][0]["content"]


def test_build_request_spec_for_music_audio_doubt_uses_text_endpoint_with_audio_context(tmp_path):
    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"audio-bytes")
    config = {
        "mode": "music-audio-doubt",
        "url": "http://127.0.0.1:6745",
        "model": "audio:test",
        "prompt": "Describe the music.",
        "output_dir": tmp_path,
        "token": None,
        "file": None,
        "audio_file": str(audio_path),
        "video_file": None,
        "response_format": None,
    }

    spec = build_request_spec(config)

    assert spec["url"].endswith("/v1/chat/completions")
    assert spec["json"]["model"] == "audio:test"
    assert str(audio_path) in spec["json"]["messages"][0]["content"]
    assert "Describe the music." in spec["json"]["messages"][0]["content"]
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `/storage/coderai/venv_all/bin/python -m pytest tests/test_manual_multimodal_test_client.py -k 'video_doubt or music_audio_doubt'`
Expected: FAIL because those modes are not implemented yet.

- [ ] **Step 3: Implement the constrained doubt-mode builders**

Extend `build_request_spec()` so the two doubt modes use the chat completions endpoint with explicit file-path context text that matches the approved constraint of using only proven text-compatible request shapes:

```python
    if mode == "video-doubt":
        video_path = _require_file(config.get("video_file"), "--video-file")
        content = (
            f"Video file: {video_path}\n"
            f"Question: {config['prompt']}\n"
            "Answer based on the referenced video input if the model/backend supports it."
        )
        return {
            "method": "POST",
            "url": f"{config['url']}/v1/chat/completions",
            "headers": headers,
            "json": {
                "model": config["model"],
                "messages": [{"role": "user", "content": content}],
            },
        }

    if mode == "music-audio-doubt":
        audio_path = _require_file(config.get("audio_file"), "--audio-file")
        content = (
            f"Audio file: {audio_path}\n"
            f"Question: {config['prompt']}\n"
            "Answer based on the referenced audio input if the model/backend supports it."
        )
        return {
            "method": "POST",
            "url": f"{config['url']}/v1/chat/completions",
            "headers": headers,
            "json": {
                "model": config["model"],
                "messages": [{"role": "user", "content": content}],
            },
        }
```

- [ ] **Step 4: Run the focused tests to verify they pass**

Run: `/storage/coderai/venv_all/bin/python -m pytest tests/test_manual_multimodal_test_client.py -k 'video_doubt or music_audio_doubt'`
Expected: PASS.

- [ ] **Step 5: Commit the doubt-mode support**

```bash
git add tools/manual_multimodal_test_client.py tests/test_manual_multimodal_test_client.py
git commit -m "feat: add multimodal client doubt modes"
```

### Task 5: Add HTTP execution and response parsing with local artifact saving

**Files:**
- Modify: `tools/manual_multimodal_test_client.py`
- Modify: `tests/test_manual_multimodal_test_client.py`

- [ ] **Step 1: Write the failing response-handling tests**

Add tests for text extraction, URL artifact download, and base64 artifact decoding:

```python
from types import SimpleNamespace

from tools.manual_multimodal_test_client import handle_response_payload


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


def test_handle_response_payload_returns_llm_text_without_artifact(tmp_path):
    response = DummyResponse({
        "choices": [{"message": {"content": "hello from model"}}]
    })

    result = handle_response_payload("llm", response, tmp_path)

    assert result["text"] == "hello from model"
    assert result["artifact_path"] is None


def test_handle_response_payload_downloads_url_artifact(monkeypatch, tmp_path):
    response = DummyResponse({
        "data": [{"url": "http://example.invalid/audio.wav"}]
    })
    monkeypatch.setattr(
        "tools.manual_multimodal_test_client._download_artifact",
        lambda url: b"wave-bytes",
    )

    result = handle_response_payload("audio-generation", response, tmp_path)

    assert result["artifact_path"].suffix == ".wav"
    assert result["artifact_path"].read_bytes() == b"wave-bytes"


def test_handle_response_payload_decodes_base64_artifact(tmp_path):
    response = DummyResponse({
        "data": [{"b64_json": "aGVsbG8="}]
    })

    result = handle_response_payload("audio-generation", response, tmp_path)

    assert result["artifact_path"].read_bytes() == b"hello"
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `/storage/coderai/venv_all/bin/python -m pytest tests/test_manual_multimodal_test_client.py -k 'handle_response_payload or download_artifact'`
Expected: FAIL because response handling is not implemented yet.

- [ ] **Step 3: Implement HTTP execution, artifact persistence, and stdout-oriented result extraction**

Add to `tools/manual_multimodal_test_client.py`:

```python
import base64
import json
import time
import requests


def _download_artifact(url: str) -> bytes:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.content


def _artifact_suffix_for_mode(mode: str) -> str:
    if mode == "audio-generation":
        return ".wav"
    if mode == "video-generation":
        return ".mp4"
    return ".bin"


def _write_artifact(output_dir: Path, mode: str, payload: bytes) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / f"{mode}-{int(time.time() * 1000)}{_artifact_suffix_for_mode(mode)}"
    artifact_path.write_bytes(payload)
    return artifact_path


def handle_response_payload(mode: str, response, output_dir: Path) -> dict:
    response.raise_for_status()
    payload = response.json()

    if mode in {"llm", "video-doubt", "music-audio-doubt"}:
        text = payload["choices"][0]["message"]["content"]
        return {"text": text, "artifact_path": None, "payload": payload}

    if mode == "transcription":
        text = payload.get("text") or payload.get("transcript") or json.dumps(payload)
        return {"text": text, "artifact_path": None, "payload": payload}

    if mode in {"audio-generation", "video-generation"}:
        first = payload["data"][0]
        text = first.get("text") or first.get("caption") or ""
        if "url" in first:
            artifact_bytes = _download_artifact(first["url"])
        elif "b64_json" in first:
            artifact_bytes = base64.b64decode(first["b64_json"])
        else:
            raise ValueError(f"No artifact found in generation response for mode: {mode}")
        artifact_path = _write_artifact(output_dir, mode, artifact_bytes)
        return {"text": text, "artifact_path": artifact_path, "payload": payload}

    raise ValueError(f"Unsupported response mode: {mode}")


def execute_request(spec: dict):
    method = spec["method"]
    kwargs = {k: v for k, v in spec.items() if k not in {"method", "url"}}
    return requests.request(method, spec["url"], timeout=300, **kwargs)
```

- [ ] **Step 4: Run the focused tests to verify they pass**

Run: `/storage/coderai/venv_all/bin/python -m pytest tests/test_manual_multimodal_test_client.py -k 'handle_response_payload or download_artifact'`
Expected: PASS.

- [ ] **Step 5: Commit the response-handling layer**

```bash
git add tools/manual_multimodal_test_client.py tests/test_manual_multimodal_test_client.py
git commit -m "feat: add multimodal client artifact handling"
```

### Task 6: Wire the script entrypoint, output printing, and error handling

**Files:**
- Modify: `tools/manual_multimodal_test_client.py`
- Modify: `tests/test_manual_multimodal_test_client.py`

- [ ] **Step 1: Write the failing main-flow tests**

Add tests for selecting the interactive fallback, executing one mode, and printing text/artifact outputs:

```python
def test_main_runs_selected_mode_and_prints_text(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr("tools.manual_multimodal_test_client.choose_mode_interactively", lambda: "llm")
    monkeypatch.setattr(
        "tools.manual_multimodal_test_client.execute_mode",
        lambda args, selected_mode: {"text": "done", "artifact_path": None},
    )

    from tools.manual_multimodal_test_client import main

    exit_code = main(["--output-dir", str(tmp_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "done" in captured.out


def test_main_prints_artifact_path_when_present(monkeypatch, capsys, tmp_path):
    artifact = tmp_path / "out.wav"
    artifact.write_bytes(b"bytes")
    monkeypatch.setattr(
        "tools.manual_multimodal_test_client.execute_mode",
        lambda args, selected_mode: {"text": "summary", "artifact_path": artifact},
    )

    from tools.manual_multimodal_test_client import main

    exit_code = main(["audio-generation", "--output-dir", str(tmp_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "summary" in captured.out
    assert str(artifact) in captured.out
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `/storage/coderai/venv_all/bin/python -m pytest tests/test_manual_multimodal_test_client.py -k 'main_runs or main_prints_artifact_path'`
Expected: FAIL because `main()` and `execute_mode()` are not implemented yet.

- [ ] **Step 3: Implement the full execution path**

Add to `tools/manual_multimodal_test_client.py`:

```python
def execute_mode(args: argparse.Namespace, selected_mode: str) -> dict:
    config = resolve_mode_config(args, selected_mode)
    spec = build_request_spec(config)
    response = execute_request(spec)
    return handle_response_payload(selected_mode, response, config["output_dir"])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    selected_mode = args.mode or choose_mode_interactively()
    try:
        result = execute_mode(args, selected_mode)
    except Exception as exc:
        print(f"ERROR [{selected_mode}]: {exc}")
        return 1

    if result.get("text"):
        print(result["text"])
    if result.get("artifact_path") is not None:
        print(result["artifact_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the focused tests to verify they pass**

Run: `/storage/coderai/venv_all/bin/python -m pytest tests/test_manual_multimodal_test_client.py -k 'main_runs or main_prints_artifact_path'`
Expected: PASS.

- [ ] **Step 5: Commit the executable client flow**

```bash
git add tools/manual_multimodal_test_client.py tests/test_manual_multimodal_test_client.py
git commit -m "feat: finalize multimodal test client flow"
```

### Task 7: Run the complete client test file and verify repository state

**Files:**
- Verify: `tests/test_manual_multimodal_test_client.py`
- Verify: `tools/manual_multimodal_test_client.py`

- [ ] **Step 1: Run the full client test file**

Run: `/storage/coderai/venv_all/bin/python -m pytest tests/test_manual_multimodal_test_client.py`
Expected: PASS with all new client tests green.

- [ ] **Step 2: Run the whisper-server regression file to confirm no accidental regressions**

Run: `/storage/coderai/venv_all/bin/python -m pytest tests/test_whisper_server_local_models.py`
Expected: PASS with existing warning profile unchanged.

- [ ] **Step 3: Inspect git status for only intended changes**

Run: `git status --short`
Expected output includes only the planned new/modified files:

```text
A tools/manual_multimodal_test_client.py
A tests/test_manual_multimodal_test_client.py
```

If commits from earlier tasks were created as planned, the working tree should instead be clean. Report the actual state clearly.

- [ ] **Step 4: Commit verification-complete state if needed**

If all earlier task commits already exist and the working tree is clean, do not create another commit. Otherwise use:

```bash
git add tools/manual_multimodal_test_client.py tests/test_manual_multimodal_test_client.py
git commit -m "feat: add manual multimodal test client"
```

## Self-Review

- Spec coverage: Task 1 covers direct plus interactive invocation, Task 2 covers built-in defaults and override resolution, Tasks 3 and 4 cover all required mode request builders including the constrained doubt modes, Task 5 covers artifact saving and stdout text behavior, and Task 6 wires execution and error handling. Task 7 verifies the full new test file and confirms no regression to the whisper-server test file.
- Placeholder scan: all file paths, commands, test names, code snippets, and commit messages are explicit.
- Type consistency: the plan consistently uses `parse_args`, `choose_mode_interactively`, `resolve_mode_config`, `build_request_spec`, `handle_response_payload`, `execute_request`, `execute_mode`, and `main` across all tasks.
