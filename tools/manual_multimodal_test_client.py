from __future__ import annotations

import argparse
from pathlib import Path


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


def choose_mode_interactively() -> str:
    for idx, mode in enumerate(MODES, start=1):
        print(f"{idx}. {mode}")
    raw = input("Choose mode: ").strip()
    selected = int(raw)
    if selected < 1 or selected > len(MODES):
        raise ValueError(f"Invalid mode selection: {raw}")
    return MODES[selected - 1]
