from __future__ import annotations

import argparse


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
