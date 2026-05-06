from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path

import requests


MODES = [
    "llm",
    "transcription",
    "audio-generation",
    "video-generation",
    "video-doubt",
    "music-audio-doubt",
]


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manual multimodal smoke-test client")
    parser.add_argument("mode", nargs="?", choices=MODES)
    parser.add_argument("--url", default="http://127.0.0.1:6745")
    parser.add_argument("--token", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--response-format", default=None)
    parser.add_argument("--output-dir", default="tmp/manual-client-output")
    parser.add_argument("--audio-file", default=None)
    parser.add_argument("--video-file", default=None)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def resolve_mode_config(args: argparse.Namespace, selected_mode: str) -> dict:
    defaults = MODE_DEFAULTS[selected_mode]
    return {
        "mode": selected_mode,
        "url": args.url.rstrip("/"),
        "token": args.token,
        "model": args.model if args.model is not None else defaults.get("model"),
        "prompt": args.prompt if args.prompt is not None else defaults.get("prompt"),
        "output_dir": Path(args.output_dir),
        "audio_file": args.audio_file if args.audio_file is not None else defaults.get("audio_file"),
        "video_file": args.video_file if args.video_file is not None else defaults.get("video_file"),
        "response_format": args.response_format if args.response_format is not None else defaults.get("response_format"),
    }


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

    if mode == "video-doubt":
        video_path = _require_file(config.get("video_file"), "--video-file")
        content = (
            f"Video file: {video_path}\n"
            f"Question: {config['prompt']}\n"
            "This request provides a local path reference only; the backend/model may or may not support reasoning from that reference. "
            "If you cannot access the referenced media, acknowledge that limitation in your answer."
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
            "This request provides a local path reference only; the backend/model may or may not support reasoning from that reference. "
            "If you cannot access the referenced media, acknowledge that limitation in your answer."
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

    raise ValueError(f"Unsupported mode for this task: {mode}")


def choose_mode_interactively() -> str:
    for idx, mode in enumerate(MODES, start=1):
        print(f"{idx}. {mode}")
    raw = input("Choose mode: ").strip()
    selected = int(raw)
    if selected < 1 or selected > len(MODES):
        raise ValueError(f"Invalid mode selection: {raw}")
    return MODES[selected - 1]


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
    kwargs = {key: value for key, value in spec.items() if key not in {"method", "url"}}
    return requests.request(method=method, url=spec["url"], timeout=300, **kwargs)
