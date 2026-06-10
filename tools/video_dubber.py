#!/usr/bin/env python3
"""Dub a video/audio file through CoderAI API while preserving background audio.

The script keeps orchestration, media slicing, timing, mixing, and muxing local.
All AI work is delegated to CoderAI endpoints:
  - /v1/audio/transcriptions for dialogue detection/transcription
  - /v1/chat/completions for speaker assignment, translation, and metric fitting
  - /v1/audio/voices for voice profiles
  - /v1/audio/clone for cloned speech generation
  - /v1/audio/convert for singing/performance voice conversion when requested
  - /v1/audio/stems for optional dialogue/background separation

External tools required locally: ffmpeg and ffprobe.
Python dependency required: requests.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

try:
    import requests
except ImportError as exc:  # pragma: no cover - user environment check
    raise SystemExit("This script requires requests: pip install requests") from exc


DEFAULT_BASE_URL = os.environ.get("CODERAI_BASE_URL", "http://127.0.0.1:8000")
DEFAULT_API_KEY = os.environ.get("CODERAI_API_KEY")
SERVICE_ENV_PREFIXES = {
    "transcribe": "CODERAI_TRANSCRIBE",
    "text": "CODERAI_TEXT",
    "voice": "CODERAI_VOICE",
    "convert": "CODERAI_CONVERT",
    "stems": "CODERAI_STEMS",
}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".webm"}
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}
SRT_TIME_RE = re.compile(
    r"(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2})[,.](?P<ms>\d{1,3})\s*-->\s*"
    r"(?P<eh>\d{2}):(?P<em>\d{2}):(?P<es>\d{2})[,.](?P<ems>\d{1,3})"
)


@dataclasses.dataclass
class Segment:
    index: int
    start: float
    end: float
    text: str
    speaker: str = "speaker_01"
    translated: str = ""
    is_singing: bool = False
    voice_name: str = ""
    ref_audio: Path | None = None
    generated_audio: Path | None = None

    @property
    def duration(self) -> float:
        return max(0.05, self.end - self.start)


class CoderAIClient:
    def __init__(self, base_url: str, api_key: str | None = None, timeout: int = 7200):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"

    def _post_json(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        response = self.session.post(f"{self.base_url}{path}", json=body, timeout=self.timeout)
        if not response.ok:
            raise RuntimeError(f"POST {path} failed: {response.status_code} {response.text[:800]}")
        return response.json()

    def _post_multipart(self, path: str, data: dict[str, Any], files: dict[str, Any]) -> Any:
        response = self.session.post(f"{self.base_url}{path}", data=data, files=files, timeout=self.timeout)
        if not response.ok:
            raise RuntimeError(f"POST {path} failed: {response.status_code} {response.text[:800]}")
        return response

    def list_models(self) -> list[dict[str, Any]]:
        response = self.session.get(f"{self.base_url}/v1/models", timeout=60)
        if not response.ok:
            return []
        return response.json().get("data", [])

    def transcribe(self, audio_path: Path, model: str, language: str | None) -> list[Segment]:
        with audio_path.open("rb") as handle:
            response = self._post_multipart(
                "/v1/audio/transcriptions",
                {
                    "model": model,
                    "language": language or "",
                    "response_format": "srt",
                    "temperature": "0",
                },
                {"file": (audio_path.name, handle, "application/octet-stream")},
            )
        return parse_srt(response.text)

    def chat_json(self, model: str, system: str, user: str, max_tokens: int = 4096) -> Any:
        data = self._post_json(
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.2,
                "max_tokens": max_tokens,
            },
        )
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
        return extract_json(content)

    def create_voice(self, name: str, audio_path: Path, transcript: str, description: str) -> None:
        with audio_path.open("rb") as handle:
            response = self.session.post(
                f"{self.base_url}/v1/audio/voices",
                data={"name": name, "transcript": transcript, "description": description},
                files={"audio": (audio_path.name, handle, "audio/wav")},
                timeout=self.timeout,
            )
        if response.status_code == 400 and "already" in response.text.lower():
            return
        if not response.ok:
            raise RuntimeError(f"Create voice {name} failed: {response.status_code} {response.text[:800]}")

    def clone_voice(self, voice_name: str, text: str, speed: float, out_path: Path) -> None:
        data = self._post_json(
            "/v1/audio/clone",
            {"voice_name": voice_name, "text": text, "speed": speed, "response_format": "b64_wav"},
        )
        item = (data.get("data") or [{}])[0]
        write_api_audio_item(item, out_path, self.session)

    def convert_voice(
        self,
        source_audio: Path,
        voice_name: str | None,
        out_path: Path,
        target_voice: Path | None = None,
        f0_condition: bool = True,
        length_adjust: float = 1.0,
    ) -> None:
        body: dict[str, Any] = {
            "source_audio": file_data_uri(source_audio, "audio/wav"),
            "f0_condition": f0_condition,
            "length_adjust": length_adjust,
            "response_format": "b64_wav",
        }
        if target_voice is not None:
            body["target_voice"] = file_data_uri(target_voice, "audio/wav")
        elif voice_name:
            body["voice_name"] = voice_name
        else:
            raise RuntimeError("Voice conversion requires voice_name or target_voice")
        data = self._post_json(
            "/v1/audio/convert",
            body,
        )
        item = (data.get("data") or [{}])[0]
        write_api_audio_item(item, out_path, self.session)

    def separate_stems(self, audio_path: Path, workdir: Path, fallback: bool) -> tuple[Path, Path] | None:
        data = self._post_json(
            "/v1/audio/stems",
            {
                "audio": file_data_uri(audio_path, "audio/wav"),
                "stem_mode": "vocals-instrumental",
                "fallback_mode": fallback,
                "response_format": "b64_wav",
            },
        )
        vocals = None
        instrumental = None
        for item in data.get("data", []):
            target = workdir / f"stem_{item.get('name', uuid.uuid4().hex)}.wav"
            write_api_audio_item(item, target, self.session)
            role = (item.get("role") or item.get("name") or "").lower()
            if "vocal" in role:
                vocals = target
            if "instrument" in role or "backing" in role:
                instrumental = target
        if vocals and instrumental:
            return vocals, instrumental
        return None


@dataclasses.dataclass(frozen=True)
class CoderAIClients:
    default: CoderAIClient
    transcribe: CoderAIClient
    text: CoderAIClient
    voice: CoderAIClient
    convert: CoderAIClient
    stems: CoderAIClient


def run(cmd: list[str], *, timeout: int | None = None) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
    if proc.returncode != 0:
        rendered = " ".join(cmd)
        detail = proc.stderr.strip() or proc.stdout.strip() or "command failed"
        raise RuntimeError(f"{rendered}\n{detail[:2000]}")


def require_binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise SystemExit(f"Required binary not found: {name}")
    return path


def media_duration(path: Path) -> float:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip())
    return float(json.loads(proc.stdout)["format"]["duration"])


def is_video(path: Path) -> bool:
    if path.suffix.lower() in VIDEO_EXTS:
        return True
    if path.suffix.lower() in AUDIO_EXTS:
        return False
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return "video" in proc.stdout


def extract_audio(input_path: Path, output_path: Path) -> None:
    run(["ffmpeg", "-y", "-i", str(input_path), "-vn", "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", str(output_path)])


def slice_audio(input_path: Path, start: float, end: float, output_path: Path) -> None:
    run([
        "ffmpeg",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "22050",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ])


def adjust_audio_timing(input_path: Path, target_duration: float, output_path: Path, max_stretch: float) -> None:
    source_duration = media_duration(input_path)
    if source_duration <= 0:
        raise RuntimeError(f"Invalid generated audio duration for {input_path}")
    ratio = source_duration / target_duration
    ratio = min(max(ratio, 1.0 / max_stretch), max_stretch)
    filters = []
    if abs(ratio - 1.0) > 0.03:
        filters.append(atempo_chain(ratio))
    if source_duration / ratio < target_duration:
        filters.append(f"apad=whole_dur={target_duration:.3f}")
    filter_arg = ",".join(filters) if filters else "anull"
    run([
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-af",
        filter_arg,
        "-t",
        f"{target_duration:.3f}",
        "-ac",
        "2",
        "-ar",
        "44100",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ])


def atempo_chain(ratio: float) -> str:
    parts: list[str] = []
    remaining = ratio
    while remaining > 2.0:
        parts.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        parts.append("atempo=0.5")
        remaining /= 0.5
    parts.append(f"atempo={remaining:.6f}")
    return ",".join(parts)


def build_dub_track(segments: list[Segment], duration: float, out_path: Path, workdir: Path) -> None:
    silence = workdir / "silence.wav"
    run([
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-t",
        f"{duration:.3f}",
        "-c:a",
        "pcm_s16le",
        str(silence),
    ])
    inputs = ["-i", str(silence)]
    filter_parts = []
    mix_inputs = ["[0:a]"]
    input_index = 1
    for segment in segments:
        if not segment.generated_audio:
            continue
        inputs.extend(["-i", str(segment.generated_audio)])
        delay_ms = max(0, int(round(segment.start * 1000)))
        filter_parts.append(
            f"[{input_index}:a]adelay={delay_ms}|{delay_ms},volume=1.0[d{input_index}]"
        )
        mix_inputs.append(f"[d{input_index}]")
        input_index += 1
    if len(mix_inputs) == 1:
        run(["ffmpeg", "-y", "-i", str(silence), "-c:a", "pcm_s16le", str(out_path)])
        return
    filter_parts.append(f"{''.join(mix_inputs)}amix=inputs={len(mix_inputs)}:duration=longest:normalize=0[out]")
    run(["ffmpeg", "-y", *inputs, "-filter_complex", ";".join(filter_parts), "-map", "[out]", "-t", f"{duration:.3f}", str(out_path)])


def duck_background(original_audio: Path, segments: list[Segment], out_path: Path, workdir: Path, duck_db: float) -> None:
    volume = 10 ** (duck_db / 20.0)
    mask = workdir / "dialogue_mask.wav"
    duration = media_duration(original_audio)
    silence = workdir / "mask_silence.wav"
    run([
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=channel_layout=mono:sample_rate=44100",
        "-t",
        f"{duration:.3f}",
        "-c:a",
        "pcm_s16le",
        str(silence),
    ])
    tone_inputs = ["-i", str(silence)]
    filter_parts = []
    mix_inputs = ["[0:a]"]
    for i, segment in enumerate(segments, 1):
        tone = workdir / f"mask_{i:04d}.wav"
        run([
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "aevalsrc=1:s=44100",
            "-t",
            f"{segment.duration:.3f}",
            str(tone),
        ])
        tone_inputs.extend(["-i", str(tone)])
        delay_ms = int(round(segment.start * 1000))
        filter_parts.append(f"[{i}:a]adelay={delay_ms}|{delay_ms}[m{i}]")
        mix_inputs.append(f"[m{i}]")
    filter_parts.append(f"{''.join(mix_inputs)}amix=inputs={len(mix_inputs)}:duration=longest:normalize=0,alimiter=limit=1[mask]")
    run(["ffmpeg", "-y", *tone_inputs, "-filter_complex", ";".join(filter_parts), "-map", "[mask]", "-t", f"{duration:.3f}", str(mask)])
    run([
        "ffmpeg",
        "-y",
        "-i",
        str(original_audio),
        "-i",
        str(mask),
        "-filter_complex",
        f"[0:a][1:a]sidechaincompress=threshold=0.01:ratio={1 / max(volume, 0.001):.3f}:attack=20:release=250[out]",
        "-map",
        "[out]",
        "-c:a",
        "pcm_s16le",
        str(out_path),
    ])


def mix_audio(background: Path, dubbed: Path, out_path: Path, duration: float) -> None:
    run([
        "ffmpeg",
        "-y",
        "-i",
        str(background),
        "-i",
        str(dubbed),
        "-filter_complex",
        "[0:a][1:a]amix=inputs=2:duration=longest:normalize=0,loudnorm=I=-16:TP=-1.5:LRA=11[out]",
        "-map",
        "[out]",
        "-t",
        f"{duration:.3f}",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(out_path),
    ])


def mux_output(input_path: Path, final_audio: Path, output_path: Path, video_input: bool) -> None:
    if video_input:
        run([
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-i",
            str(final_audio),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(output_path),
        ])
    else:
        run(["ffmpeg", "-y", "-i", str(final_audio), "-c:a", "aac", str(output_path)])


def parse_srt(text: str) -> list[Segment]:
    blocks = re.split(r"\n\s*\n", text.strip())
    segments: list[Segment] = []
    for block in blocks:
        lines = [line.strip("\ufeff ") for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        time_line_index = next((i for i, line in enumerate(lines) if "-->" in line), -1)
        if time_line_index < 0:
            continue
        match = SRT_TIME_RE.search(lines[time_line_index])
        if not match:
            continue
        body = " ".join(lines[time_line_index + 1 :]).strip()
        if not body:
            continue
        index_text = lines[0] if time_line_index > 0 else str(len(segments) + 1)
        try:
            index = int(index_text)
        except ValueError:
            index = len(segments) + 1
        segments.append(
            Segment(
                index=index,
                start=srt_time_to_seconds(match.group("h"), match.group("m"), match.group("s"), match.group("ms")),
                end=srt_time_to_seconds(match.group("eh"), match.group("em"), match.group("es"), match.group("ems")),
                text=body,
            )
        )
    return segments


def srt_time_to_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")[:3]) / 1000.0


def extract_json(text: str) -> Any:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = min((p for p in [cleaned.find("{"), cleaned.find("[")] if p >= 0), default=-1)
        end = max(cleaned.rfind("}"), cleaned.rfind("]"))
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise RuntimeError(f"CoderAI chat did not return JSON:\n{text[:1000]}")


def file_data_uri(path: Path, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def write_api_audio_item(item: dict[str, Any], out_path: Path, session: requests.Session) -> None:
    for key in ("b64_wav", "b64_mp3", "b64_audio", "audio"):
        if item.get(key):
            raw = item[key]
            if isinstance(raw, str) and raw.startswith("data:"):
                raw = raw.split(",", 1)[1]
            out_path.write_bytes(base64.b64decode(raw))
            return
    if item.get("url"):
        response = session.get(item["url"], timeout=7200)
        response.raise_for_status()
        out_path.write_bytes(response.content)
        return
    raise RuntimeError(f"No audio payload found in API response item: {item.keys()}")


def choose_default_model(models: list[dict[str, Any]], capability: str) -> str | None:
    for model in models:
        if capability in (model.get("capabilities") or []):
            return model.get("id")
    return None


def env_default(service: str, field: str, fallback: str | None = None) -> str | None:
    prefix = SERVICE_ENV_PREFIXES[service]
    return os.environ.get(f"{prefix}_{field}") or fallback


def build_clients(args: argparse.Namespace) -> CoderAIClients:
    default = CoderAIClient(args.base_url, args.api_key)

    def service_client(service: str) -> CoderAIClient:
        base_url = getattr(args, f"{service}_base_url") or args.base_url
        api_key = getattr(args, f"{service}_api_key")
        if api_key is None:
            api_key = args.api_key
        return CoderAIClient(base_url, api_key)

    return CoderAIClients(
        default=default,
        transcribe=service_client("transcribe"),
        text=service_client("text"),
        voice=service_client("voice"),
        convert=service_client("convert"),
        stems=service_client("stems"),
    )


def client_label(client: CoderAIClient) -> str:
    return client.base_url


def assign_speakers(client: CoderAIClient, text_model: str, segments: list[Segment], max_speakers: int) -> None:
    payload = [
        {"id": s.index, "start": round(s.start, 3), "end": round(s.end, 3), "text": s.text}
        for s in segments
    ]
    system = "You assign dialogue subtitle segments to recurring speakers. Return only JSON."
    user = textwrap.dedent(
        f"""
        Assign each segment to one of at most {max_speakers} stable speaker ids.
        Use ids like speaker_01, speaker_02. Mark singing=true when the segment appears sung, lyrical, chanted, or is likely part of music.
        Return JSON as: {{"segments":[{{"id":1,"speaker":"speaker_01","singing":false}}]}}

        Segments:
        {json.dumps(payload, ensure_ascii=False)}
        """
    ).strip()
    try:
        data = client.chat_json(text_model, system, user, max_tokens=4096)
        by_id = {int(item["id"]): item for item in data.get("segments", [])}
        for segment in segments:
            item = by_id.get(segment.index, {})
            segment.speaker = sanitize_name(str(item.get("speaker") or segment.speaker))
            segment.is_singing = bool(item.get("singing", False))
    except Exception as exc:
        print(f"warning: speaker assignment failed, using automatic speakers: {exc}", file=sys.stderr)
        for i, segment in enumerate(segments):
            segment.speaker = f"speaker_{(i % max(1, max_speakers)) + 1:02d}"


def translate_segments(client: CoderAIClient, text_model: str, target_language: str, segments: list[Segment]) -> None:
    batch_size = 40
    system = "You translate dubbing scripts. Return only JSON."
    for start in range(0, len(segments), batch_size):
        batch = segments[start : start + batch_size]
        payload = [
            {
                "id": s.index,
                "source_text": s.text,
                "duration_seconds": round(s.duration, 3),
                "speaker": s.speaker,
                "singing": s.is_singing,
            }
            for s in batch
        ]
        user = textwrap.dedent(
            f"""
            Translate each segment to {target_language} for dubbing.
            Preserve meaning, tone, speaker intent, and song lyric style when singing=true.
            Keep the translation speakable within the provided duration. Prefer natural lip-sync/metric fit over literal word order.
            Return JSON as: {{"segments":[{{"id":1,"translation":"..."}}]}}

            Segments:
            {json.dumps(payload, ensure_ascii=False)}
            """
        ).strip()
        data = client.chat_json(text_model, system, user, max_tokens=4096)
        by_id = {int(item["id"]): str(item.get("translation", "")).strip() for item in data.get("segments", [])}
        for segment in batch:
            segment.translated = by_id.get(segment.index) or segment.text


def fit_translation_metric(client: CoderAIClient, text_model: str, target_language: str, segments: list[Segment]) -> None:
    system = "You adapt translated lines for dubbing timing and lip-sync. Return only JSON."
    for segment in segments:
        syllable_hint = max(2, int(segment.duration * 4.2))
        user = textwrap.dedent(
            f"""
            Rewrite this {target_language} dub line so it fits about {segment.duration:.2f} seconds.
            Aim for roughly {syllable_hint} syllables, preserve meaning, and keep it natural.
            If singing is true, keep lyric rhythm and rhyme when possible.
            Return JSON as: {{"translation":"..."}}

            Original: {segment.text}
            Current translation: {segment.translated}
            Singing: {segment.is_singing}
            """
        ).strip()
        try:
            data = client.chat_json(text_model, system, user, max_tokens=512)
            value = str(data.get("translation", "")).strip()
            if value:
                segment.translated = value
        except Exception as exc:
            print(f"warning: metric fitting failed for segment {segment.index}: {exc}", file=sys.stderr)


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip().lower())
    return cleaned[:48] or "speaker_01"


def create_voice_profiles(client: CoderAIClient, source_audio: Path, segments: list[Segment], workdir: Path, prefix: str) -> None:
    by_speaker: dict[str, list[Segment]] = {}
    for segment in segments:
        by_speaker.setdefault(segment.speaker, []).append(segment)
    for speaker, speaker_segments in by_speaker.items():
        selected = sorted(speaker_segments, key=lambda s: s.duration, reverse=True)[:6]
        start = max(0.0, selected[0].start - 0.05)
        end = selected[0].end + 0.05
        ref_audio = workdir / f"ref_{speaker}.wav"
        slice_audio(source_audio, start, end, ref_audio)
        transcript = selected[0].text.strip()
        voice_name = sanitize_name(f"{prefix}_{speaker}")
        print(f"creating voice profile {voice_name} from {start:.2f}-{end:.2f}s")
        client.create_voice(voice_name, ref_audio, transcript, f"Auto-extracted by tools/video_dubber.py for {speaker}")
        for segment in speaker_segments:
            segment.voice_name = voice_name
            segment.ref_audio = ref_audio


def generate_segment_audio(
    voice_client: CoderAIClient,
    convert_client: CoderAIClient,
    source_audio: Path,
    segments: list[Segment],
    workdir: Path,
    max_stretch: float,
    preserve_singing: bool,
) -> None:
    for n, segment in enumerate(segments, 1):
        raw = workdir / f"dub_raw_{segment.index:04d}.wav"
        fitted = workdir / f"dub_fit_{segment.index:04d}.wav"
        speed = 1.0
        if segment.translated:
            approx_chars_per_sec = len(segment.translated) / segment.duration
            if approx_chars_per_sec > 18:
                speed = min(1.35, approx_chars_per_sec / 16)
        print(f"[{n}/{len(segments)}] generating {segment.voice_name} {segment.duration:.2f}s")
        if preserve_singing and segment.is_singing:
            source_slice = workdir / f"sing_source_{segment.index:04d}.wav"
            slice_audio(source_audio, segment.start, segment.end, source_slice)
            try:
                convert_client.convert_voice(
                    source_slice,
                    segment.voice_name,
                    raw,
                    target_voice=segment.ref_audio,
                    f0_condition=True,
                    length_adjust=1.0,
                )
            except Exception as exc:
                print(f"warning: singing conversion failed for segment {segment.index}, falling back to cloned TTS: {exc}", file=sys.stderr)
                voice_client.clone_voice(segment.voice_name, segment.translated or segment.text, speed, raw)
        else:
            voice_client.clone_voice(segment.voice_name, segment.translated or segment.text, speed, raw)
        adjust_audio_timing(raw, segment.duration, fitted, max_stretch)
        segment.generated_audio = fitted


def write_artifacts(segments: list[Segment], output_base: Path) -> None:
    json_path = output_base.with_suffix(".segments.json")
    srt_path = output_base.with_suffix(".translated.srt")
    json_path.write_text(
        json.dumps([dataclasses.asdict(s) | {"ref_audio": str(s.ref_audio or ""), "generated_audio": str(s.generated_audio or "")} for s in segments], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = []
    for i, segment in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{seconds_to_srt(segment.start)} --> {seconds_to_srt(segment.end)}")
        lines.append(segment.translated or segment.text)
        lines.append("")
    srt_path.write_text("\n".join(lines), encoding="utf-8")


def seconds_to_srt(value: float) -> str:
    value = max(0.0, value)
    h = int(value // 3600)
    m = int((value % 3600) // 60)
    s = int(value % 60)
    ms = int(round((value - math.floor(value)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dub video/audio through CoderAI API while preserving music and effects.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Input video or audio file")
    parser.add_argument("-o", "--output", type=Path, help="Output media path")
    parser.add_argument("-l", "--target-language", required=True, help="Target dubbing language, e.g. Italian, Spanish, ja")
    parser.add_argument("--source-language", help="Optional source language hint for transcription")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="CoderAI API base URL")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="CoderAI bearer token; defaults to CODERAI_API_KEY")
    parser.add_argument("--audio-model", help="CoderAI transcription model id")
    parser.add_argument("--text-model", default=env_default("text", "MODEL"), help="CoderAI text model id for translation and dialogue analysis")
    parser.add_argument("--transcribe-base-url", default=env_default("transcribe", "BASE_URL"), help="Override CoderAI URL for /v1/audio/transcriptions")
    parser.add_argument("--transcribe-api-key", default=env_default("transcribe", "API_KEY"), help="Override bearer token for transcription")
    parser.add_argument("--transcribe-model", default=env_default("transcribe", "MODEL"), help="Alias for --audio-model; defaults to CODERAI_TRANSCRIBE_MODEL")
    parser.add_argument("--text-base-url", default=env_default("text", "BASE_URL"), help="Override CoderAI URL for /v1/chat/completions")
    parser.add_argument("--text-api-key", default=env_default("text", "API_KEY"), help="Override bearer token for text requests")
    parser.add_argument("--voice-base-url", default=env_default("voice", "BASE_URL"), help="Override CoderAI URL for /v1/audio/voices and /v1/audio/clone")
    parser.add_argument("--voice-api-key", default=env_default("voice", "API_KEY"), help="Override bearer token for voice cloning/profile requests")
    parser.add_argument("--convert-base-url", default=env_default("convert", "BASE_URL"), help="Override CoderAI URL for /v1/audio/convert")
    parser.add_argument("--convert-api-key", default=env_default("convert", "API_KEY"), help="Override bearer token for voice conversion")
    parser.add_argument("--stems-base-url", default=env_default("stems", "BASE_URL"), help="Override CoderAI URL for /v1/audio/stems")
    parser.add_argument("--stems-api-key", default=env_default("stems", "API_KEY"), help="Override bearer token for stem separation")
    parser.add_argument("--max-speakers", type=int, default=8, help="Maximum recurring speaker voices to infer")
    parser.add_argument("--voice-prefix", default="dub", help="Prefix for saved CoderAI voice profiles")
    parser.add_argument("--no-stems", action="store_true", help="Do not call /v1/audio/stems; use local ducking to preserve background")
    parser.add_argument("--stem-fallback", action="store_true", help="Ask CoderAI stems endpoint to use its ffmpeg fallback mode")
    parser.add_argument("--no-metric-fit", action="store_true", help="Skip second LLM pass for tighter metric/lip-sync adaptation")
    parser.add_argument("--no-singing-convert", action="store_true", help="Do not use /v1/audio/convert for singing segments")
    parser.add_argument("--duck-db", type=float, default=-14.0, help="Dialogue-region background ducking target in dB when stems are disabled/unavailable")
    parser.add_argument("--max-stretch", type=float, default=1.35, help="Maximum local time stretch/compress factor for generated lines")
    parser.add_argument("--keep-workdir", type=Path, help="Keep intermediate files in this directory")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    require_binary("ffmpeg")
    require_binary("ffprobe")
    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    video_input = is_video(input_path)
    output_path = args.output
    if output_path is None:
        suffix = ".mp4" if video_input else ".m4a"
        output_path = input_path.with_name(f"{input_path.stem}.dubbed.{sanitize_name(args.target_language)}{suffix}")
    output_path = output_path.expanduser().resolve()

    clients = build_clients(args)
    transcribe_models = clients.transcribe.list_models()
    text_models = clients.text.list_models()
    audio_model = args.audio_model or args.transcribe_model or choose_default_model(transcribe_models, "audio_transcription") or "whisper"
    text_model = args.text_model or choose_default_model(text_models, "text_generation")
    if not text_model:
        raise SystemExit("No text model found. Pass --text-model with a CoderAI chat model id.")

    work_context = tempfile.TemporaryDirectory(prefix="coderai-dub-") if args.keep_workdir is None else None
    workdir = args.keep_workdir or Path(work_context.name)  # type: ignore[union-attr]
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        source_audio = workdir / "source.wav"
        extract_audio(input_path, source_audio)
        total_duration = media_duration(source_audio)

        print(f"transcribing with {audio_model} via {client_label(clients.transcribe)}")
        segments = clients.transcribe.transcribe(source_audio, audio_model, args.source_language)
        segments = [s for s in segments if s.text.strip() and s.duration >= 0.08]
        if not segments:
            raise RuntimeError("No dialogue segments found in the input")

        print(f"assigning speakers and singing flags with {text_model} via {client_label(clients.text)}")
        assign_speakers(clients.text, text_model, segments, args.max_speakers)

        print(f"translating {len(segments)} segments to {args.target_language} via {client_label(clients.text)}")
        translate_segments(clients.text, text_model, args.target_language, segments)
        if not args.no_metric_fit:
            print("fitting translated lines to segment metrics")
            fit_translation_metric(clients.text, text_model, args.target_language, segments)

        run_prefix = sanitize_name(f"{args.voice_prefix}_{input_path.stem}_{int(time.time())}")
        print(f"creating voice profiles via {client_label(clients.voice)}")
        create_voice_profiles(clients.voice, source_audio, segments, workdir, run_prefix)
        generate_segment_audio(
            clients.voice,
            clients.convert,
            source_audio,
            segments,
            workdir,
            args.max_stretch,
            preserve_singing=not args.no_singing_convert,
        )

        dub_track = workdir / "dub_track.wav"
        build_dub_track(segments, total_duration, dub_track, workdir)

        background = workdir / "background.wav"
        stems = None
        if not args.no_stems:
            print(f"requesting CoderAI stem separation via {client_label(clients.stems)}")
            try:
                stems = clients.stems.separate_stems(source_audio, workdir, args.stem_fallback)
            except Exception as exc:
                print(f"warning: stems unavailable, using local dialogue ducking: {exc}", file=sys.stderr)
        if stems:
            _, instrumental = stems
            run(["ffmpeg", "-y", "-i", str(instrumental), "-t", f"{total_duration:.3f}", "-c:a", "pcm_s16le", str(background)])
        else:
            duck_background(source_audio, segments, background, workdir, args.duck_db)

        final_audio = workdir / "final_audio.m4a"
        mix_audio(background, dub_track, final_audio, total_duration)
        mux_output(input_path, final_audio, output_path, video_input)
        write_artifacts(segments, output_path)

        print(f"wrote {output_path}")
        print(f"wrote {output_path.with_suffix('.segments.json')}")
        print(f"wrote {output_path.with_suffix('.translated.srt')}")
        if args.keep_workdir:
            print(f"kept workdir {workdir}")
        return 0
    finally:
        if work_context is not None:
            work_context.cleanup()


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        raise SystemExit(130)
