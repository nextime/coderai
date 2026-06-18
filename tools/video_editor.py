#!/usr/bin/env python3
"""Browser-based video editor backed by CoderAI TTS + local ffmpeg.

Launches a small web interface that lets you:

  1. Pick a video and scrub it on a zoomable timeline (zoom all the way in to
     step single frames, or zoom out to see the whole clip).
  2. Generate natural-sounding AI speech (CoderAI /v1/audio/speech, Kokoro) and
     drop it at the exact playhead position. Voice gender (feminine/masculine)
     is chosen on the command line and can be overridden per clip in the UI.
  3. Select a start/stop region and accelerate it 2x / 4x / 8x / 16x / 32x /
     64x / 128x / 240x.
  4. Add a second audio track (music) with an independent volume control.
  5. Render and save the final result.

Everything is also configurable from the web UI (⚙ Configuration panel): the
CoderAI base URL + API key, the media directory videos are picked from, the
output directory, the TTS / STT models (chosen from the server's live model
list), and the default voice. Settings can be saved to a JSON config file and
reloaded later with `-c/--config`.

Source video / music can be picked from the server's media directory or
uploaded from the machine running the browser (so it works behind a reverse
proxy too). All AI work goes to the CoderAI server; all media work is done
locally with ffmpeg/ffprobe.

Requirements: ffmpeg + ffprobe on PATH, and the `requests` python package.
"""

from __future__ import annotations

import argparse
import array
import base64
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

try:
    import requests
except ImportError as exc:  # pragma: no cover
    raise SystemExit("This script requires requests: pip install requests") from exc


VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".flv"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".webm"}

# Curated Kokoro voices, grouped by gender, used to populate the UI selector.
VOICES = {
    "feminine": [
        ("af_sarah", "Sarah (US)"),
        ("af_heart", "Heart (US)"),
        ("af_bella", "Bella (US)"),
        ("af_nicole", "Nicole (US)"),
        ("af_sky", "Sky (US)"),
        ("bf_emma", "Emma (UK)"),
        ("bf_isabella", "Isabella (UK)"),
    ],
    "masculine": [
        ("am_michael", "Michael (US)"),
        ("am_adam", "Adam (US)"),
        ("am_eric", "Eric (US)"),
        ("am_liam", "Liam (US)"),
        ("am_onyx", "Onyx (US)"),
        ("bm_george", "George (UK)"),
        ("bm_lewis", "Lewis (UK)"),
    ],
}
DEFAULT_VOICE = {"feminine": "af_sarah", "masculine": "am_michael"}
SPEED_FACTORS = [2, 4, 8, 16, 32, 64, 128, 240]


def tts_emotions(model: str) -> list:
    """Emotions the configured TTS model can steer, or [] when unavailable.

    Looked up from the CoderAI backend's family map when importable; the editor
    only offers an emotion picker when this is non-empty."""
    try:
        from codai.api.tts_backends import family_emotions
        return family_emotions(model or "")
    except Exception:
        return []


def tts_styles(model: str) -> list:
    """Delivery styles (whisper/shout/tone/…) the model can steer, or []."""
    try:
        from codai.api.tts_backends import family_styles
        return family_styles(model or "")
    except Exception:
        return []


# --------------------------------------------------------------------------- #
# ffmpeg / ffprobe helpers
# --------------------------------------------------------------------------- #
def require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"Required binary '{name}' not found on PATH.")


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout[-2000:]}"
        )


def ffprobe(path: Path) -> dict:
    """Return duration (s), fps, width, height and whether an audio stream exists."""
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", str(path),
        ],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {out.stderr[:500]}")
    data = json.loads(out.stdout or "{}")
    duration = float(data.get("format", {}).get("duration", 0) or 0)
    fps = 30.0
    width = height = 0
    has_audio = False
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            width = int(stream.get("width", 0) or 0)
            height = int(stream.get("height", 0) or 0)
            rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/0"
            try:
                num, den = rate.split("/")
                if float(den) > 0 and float(num) > 0:
                    fps = float(num) / float(den)
            except (ValueError, ZeroDivisionError):
                pass
            if not duration:
                duration = float(stream.get("duration", 0) or 0)
        elif stream.get("codec_type") == "audio":
            has_audio = True
    return {
        "duration": round(duration, 4),
        "fps": round(fps, 4),
        "width": width,
        "height": height,
        "hasAudio": has_audio,
    }


def audio_waveform(path: Path, points: int = 1600) -> dict:
    """Return compact mono peak data for drawing a timeline waveform."""
    info = ffprobe(path)
    duration = max(0.0, float(info.get("duration", 0.0) or 0.0))
    if not info.get("hasAudio") or duration <= 0:
        return {"duration": duration, "peaks": []}
    proc = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
            "-ar", "8000", "-f", "s16le", "-",
        ],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"waveform extraction failed: {proc.stderr.decode('utf-8', 'ignore')[:500]}")
    samples = array.array("h")
    samples.frombytes(proc.stdout)
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        return {"duration": duration, "peaks": []}
    points = max(64, min(int(points or 1600), 12000))
    bucket = max(1, math.ceil(len(samples) / points))
    peaks = []
    for i in range(0, len(samples), bucket):
        chunk = samples[i:i + bucket]
        peaks.append(round(max(abs(v) for v in chunk) / 32768.0, 4))
    return {"duration": duration, "peaks": peaks}


def motion_graph(path: Path, points: int = 900) -> dict:
    info = ffprobe(path)
    duration = max(0.0, float(info.get("duration", 0.0) or 0.0))
    if not info.get("width") or duration <= 0:
        return {"duration": duration, "values": []}
    sample_fps = min(4.0, max(0.5, float(points or 900) / max(duration, 0.001)))
    w = 64
    h = 36
    proc = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(path), "-an",
            "-vf", f"fps={sample_fps:.4f},scale={w}:{h},format=gray",
            "-f", "rawvideo", "-",
        ],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"motion graph failed: {proc.stderr.decode('utf-8', 'ignore')[:500]}")
    frame_size = w * h
    frames = [proc.stdout[i:i + frame_size] for i in range(0, len(proc.stdout), frame_size)]
    frames = [f for f in frames if len(f) == frame_size]
    if len(frames) < 2:
        return {"duration": duration, "values": []}
    diffs = []
    prev = frames[0]
    for frame in frames[1:]:
        diff = sum(abs(a - b) for a, b in zip(frame, prev)) / (frame_size * 255.0)
        diffs.append(diff)
        prev = frame
    peak = max(diffs) or 1.0
    values = [round(min(1.0, d / peak), 4) for d in diffs]
    return {"duration": duration, "values": values, "sampleFps": sample_fps}


def srt_timestamp(t: float) -> str:
    t = max(0.0, t)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    if ms == 1000:
        s += 1
        ms = 0
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


_SRT_TIME_RE = None


def parse_srt(text: str) -> list[dict]:
    """Parse SRT text into [{start, end, text}] (seconds)."""
    import re
    pattern = re.compile(
        r"(\d{2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*"
        r"(\d{2}):(\d{2}):(\d{2})[,.](\d{1,3})"
    )
    segments: list[dict] = []
    blocks = re.split(r"\n\s*\n", text.strip())
    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        m = None
        body_start = 0
        for i, ln in enumerate(lines):
            m = pattern.search(ln)
            if m:
                body_start = i + 1
                break
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000.0
        end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000.0
        body = " ".join(lines[body_start:]).strip()
        if body:
            segments.append({"start": start, "end": end, "text": body})
    return segments


def build_srt(captions: list[dict], regions: list[dict]) -> str:
    """Render captions to SRT text, remapping times through speed regions."""
    items = []
    for c in captions:
        start = remap_time(float(c["start"]), regions)
        end = remap_time(float(c["end"]), regions)
        if end <= start:
            end = start + 0.5
        text = str(c.get("text", "")).strip()
        if text:
            items.append((start, end, text))
    items.sort(key=lambda x: x[0])
    out = []
    for i, (start, end, text) in enumerate(items, 1):
        out.append(f"{i}\n{srt_timestamp(start)} --> {srt_timestamp(end)}\n{text}\n")
    return "\n".join(out)


def atempo_chain(ratio: float) -> str:
    """Build an atempo filter chain (each stage limited to 0.5..2.0)."""
    if ratio <= 0:
        ratio = 1.0
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


# --------------------------------------------------------------------------- #
# Speed regions: timeline math
# --------------------------------------------------------------------------- #
def normalize_regions(regions: list[dict], duration: float) -> list[dict]:
    """Clip regions to [0, duration], sort, and drop overlaps/zero-length."""
    cleaned = []
    for r in regions:
        start = max(0.0, min(float(r["start"]), duration))
        end = max(0.0, min(float(r["end"]), duration))
        factor = float(r["factor"])
        if end - start > 1e-3 and factor > 1.0:
            cleaned.append({"start": start, "end": end, "factor": factor})
    cleaned.sort(key=lambda r: r["start"])
    out: list[dict] = []
    for r in cleaned:
        if out and r["start"] < out[-1]["end"]:
            r["start"] = out[-1]["end"]  # trim overlap; keep them disjoint
        if r["end"] - r["start"] > 1e-3:
            out.append(r)
    return out


def remap_time(t: float, regions: list[dict]) -> float:
    """Map a position in the source timeline to its position after speed-up."""
    new_t = t
    for r in regions:
        if t <= r["start"]:
            continue
        overlap = min(t, r["end"]) - r["start"]
        if overlap > 0:
            new_t -= overlap * (1.0 - 1.0 / r["factor"])
    return max(0.0, new_t)


def build_segments(regions: list[dict], duration: float) -> list[tuple]:
    """Cover [0, duration] with ordered (start, end, factor) segments."""
    segs: list[tuple] = []
    cursor = 0.0
    for r in regions:
        if r["start"] - cursor > 1e-3:
            segs.append((cursor, r["start"], 1.0))
        segs.append((r["start"], r["end"], r["factor"]))
        cursor = r["end"]
    if duration - cursor > 1e-3:
        segs.append((cursor, duration, 1.0))
    return [s for s in segs if s[1] - s[0] > 1e-3]


def normalize_cuts(cuts: list[dict], duration: float) -> list[dict]:
    """Clip cut regions to [0, duration], sort, and merge overlaps."""
    cleaned = []
    for c in cuts:
        start = max(0.0, min(float(c["start"]), duration))
        end = max(0.0, min(float(c["end"]), duration))
        if end - start > 1e-3:
            cleaned.append({"start": start, "end": end})
    cleaned.sort(key=lambda c: c["start"])
    out: list[dict] = []
    for c in cleaned:
        if out and c["start"] <= out[-1]["end"] + 1e-3:
            out[-1]["end"] = max(out[-1]["end"], c["end"])
        else:
            out.append(c)
    return out


def build_keep_segments(cuts: list[dict], duration: float) -> list[tuple[float, float]]:
    """Cover the portions of [0, duration] that remain after cuts."""
    segs: list[tuple[float, float]] = []
    cursor = 0.0
    for c in cuts:
        if c["start"] - cursor > 1e-3:
            segs.append((cursor, c["start"]))
        cursor = max(cursor, c["end"])
    if duration - cursor > 1e-3:
        segs.append((cursor, duration))
    return segs


def remap_cut_time(t: float, cuts: list[dict]) -> float:
    """Map source timeline time to the timeline after removed cut regions."""
    new_t = max(0.0, t)
    for c in cuts:
        if t <= c["start"]:
            break
        if t < c["end"]:
            return max(0.0, c["start"] - sum(x["end"] - x["start"] for x in cuts if x["end"] <= c["start"]))
        new_t -= c["end"] - c["start"]
    return max(0.0, new_t)


def remap_cut_regions(regions: list[dict], cuts: list[dict]) -> list[dict]:
    """Move source timeline regions onto the post-cut timeline."""
    out = []
    for r in regions:
        start = remap_cut_time(float(r["start"]), cuts)
        end = remap_cut_time(float(r["end"]), cuts)
        if end - start > 1e-3:
            nr = dict(r)
            nr["start"] = start
            nr["end"] = end
            out.append(nr)
    return out


# --------------------------------------------------------------------------- #
# Render pipeline
# --------------------------------------------------------------------------- #
class Editor:
    def __init__(self, cfg: "Config"):
        self.cfg = cfg
        self.session = requests.Session()
        if cfg.api_key:
            self.session.headers["Authorization"] = f"Bearer {cfg.api_key}"

    # ---- safe path resolution ------------------------------------------- #
    def _resolve(self, base: Path, name: str) -> Path:
        p = (base / name).resolve()
        if base.resolve() not in p.parents and p != base.resolve():
            raise ValueError("Path escapes the allowed directory")
        if not p.exists():
            raise FileNotFoundError(name)
        return p

    def media_file(self, name: str) -> Path:
        return self._resolve(self.cfg.media_dir, name)

    def external_file(self, path: str) -> Path:
        """Resolve an arbitrary audio/video file (e.g. a music track) from
        anywhere on disk. Relative paths are taken against the media dir."""
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = self.cfg.media_dir / path
        p = p.resolve()
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(path)
        if p.suffix.lower() not in AUDIO_EXTS and p.suffix.lower() not in VIDEO_EXTS:
            raise ValueError(f"Not an audio/video file: {p.name}")
        return p

    def save_upload(self, filename: str, stream, length: int) -> Path:
        """Stream an uploaded file (from the browser machine) to the uploads dir."""
        import re
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", Path(filename or "upload").name)[-100:] or "upload"
        if Path(safe).suffix.lower() not in (VIDEO_EXTS | AUDIO_EXTS):
            raise ValueError(f"Unsupported file type: {safe}")
        dest = self.cfg.upload_dir / f"{uuid.uuid4().hex}_{safe}"
        with dest.open("wb") as fh:
            remaining = length
            while remaining > 0:
                chunk = stream.read(min(1 << 20, remaining))
                if not chunk:
                    break
                fh.write(chunk)
                remaining -= len(chunk)
        return dest

    def resolve_ref(self, ref: str) -> Path:
        """Resolve a media reference from any of the supported sources:
        ``upload:<name>`` (browser upload), ``abs:<path>`` / absolute path
        (arbitrary server file), or a path relative to the media dir."""
        if ref.startswith("upload:"):
            return self._resolve(self.cfg.upload_dir, ref[len("upload:"):])
        if ref.startswith("abs:"):
            return self.external_file(ref[len("abs:"):])
        if Path(ref).expanduser().is_absolute():
            return self.external_file(ref)
        return self.media_file(ref)

    def refresh_auth(self) -> None:
        """Re-sync the HTTP auth header after the API key changes at runtime."""
        if self.cfg.api_key:
            self.session.headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        else:
            self.session.headers.pop("Authorization", None)

    def list_models(self) -> list[dict]:
        """Live list of models from the configured server (for the UI)."""
        return _list_models(self.cfg.base_url, self.cfg.api_key)

    # ---- TTS ------------------------------------------------------------- #
    def tts(self, text: str, voice: str, speed: float = 1.0, emotion: str = "",
            style: str = "") -> Path:
        if not self.cfg.tts_model:
            raise RuntimeError(
                "No TTS model configured. Pass --tts-model or make sure the "
                "CoderAI server exposes a Kokoro TTS model."
            )
        try:
            speed = max(0.25, min(4.0, float(speed) or 1.0))
        except (TypeError, ValueError):
            speed = 1.0
        body = {
            "model": self.cfg.tts_model,
            "input": text,
            "voice": voice or self.cfg.default_voice,
            "response_format": "wav",
            "speed": speed,
        }
        if emotion:
            body["emotion"] = emotion
        if style:
            body["style"] = style
        resp = self.session.post(
            f"{self.cfg.base_url}/v1/audio/speech", json=body, timeout=600
        )
        if not resp.ok:
            raise RuntimeError(f"TTS failed: {resp.status_code} {resp.text[:400]}")
        payload = resp.json()
        b64 = payload.get("audio") or payload.get("b64_wav") or payload.get("data")
        if not b64:
            raise RuntimeError("TTS response had no audio payload")
        out = self.cfg.tts_dir / f"tts_{uuid.uuid4().hex}.wav"
        out.write_bytes(base64.b64decode(b64))
        return out

    # ---- auto-captions (speech-to-text) --------------------------------- #
    def transcribe(self, name: str, language: str | None) -> list[dict]:
        if not self.cfg.stt_model:
            raise RuntimeError(
                "No transcription model configured. Pass --stt-model or expose a "
                "Whisper model on CoderAI."
            )
        media = self.resolve_ref(name)
        # Extract a compact audio track so uploads stay small / fast.
        tmp_audio = self.cfg.tts_dir / f"asr_{uuid.uuid4().hex}.mp3"
        run(["ffmpeg", "-y", "-i", str(media), "-vn", "-ac", "1", "-ar", "16000",
             "-b:a", "96k", str(tmp_audio)])
        try:
            with tmp_audio.open("rb") as fh:
                resp = self.session.post(
                    f"{self.cfg.base_url}/v1/audio/transcriptions",
                    data={
                        "model": self.cfg.stt_model,
                        "language": language or "",
                        "response_format": "srt",
                        "temperature": "0",
                    },
                    files={"file": (tmp_audio.name, fh, "audio/mpeg")},
                    timeout=1800,
                )
            if not resp.ok:
                raise RuntimeError(
                    f"Transcription failed: {resp.status_code} {resp.text[:400]}"
                )
            return parse_srt(resp.text)
        finally:
            tmp_audio.unlink(missing_ok=True)

    # ---- music generation ------------------------------------------------ #
    def generate_music(self, prompt: str, duration: float, model: str | None = None) -> Path:
        model_name = model or self.cfg.audio_model
        if not model_name:
            raise RuntimeError(
                "No music generation model configured. Pass --audio-model or configure "
                "a MusicGen/audio generation model in the UI."
            )
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("Music prompt is required")
        duration = max(1.0, min(float(duration or 10.0), 300.0))
        body = {
            "model": model_name,
            "prompt": prompt,
            "duration": duration,
            "response_format": "b64_wav",
        }
        resp = self.session.post(
            f"{self.cfg.base_url}/v1/audio/generate", json=body, timeout=1800
        )
        if not resp.ok:
            raise RuntimeError(f"Music generation failed: {resp.status_code} {resp.text[:400]}")
        payload = resp.json()
        item = (payload.get("data") or [{}])[0]
        ext = "wav"
        raw = item.get("b64_wav") or item.get("b64_mp3") or item.get("url") or ""
        if item.get("b64_mp3"):
            ext = "mp3"
        if not raw:
            raise RuntimeError("Music generation response had no audio payload")
        if raw.startswith("http://") or raw.startswith("https://"):
            got = self.session.get(raw, timeout=300)
            if not got.ok:
                raise RuntimeError(f"Could not download generated music: {got.status_code}")
            audio = got.content
            ctype = got.headers.get("Content-Type", "")
            ext = "mp3" if "mpeg" in ctype or raw.lower().endswith(".mp3") else "wav"
        else:
            if raw.startswith("data:"):
                raw = raw.split(",", 1)[1]
            audio = base64.b64decode(raw)
        out = self.cfg.upload_dir / f"musicgen_{uuid.uuid4().hex}.{ext}"
        out.write_bytes(audio)
        return out

    # ---- final render ---------------------------------------------------- #
    # Default transition length (seconds) used for fades to/from black across
    # gaps. Overlaps crossfade over the full overlap instead.
    TRANS = 0.5

    def _clip_specs(self, params: dict) -> list[dict]:
        """Resolve the playlist into [{path, offset, duration, hasAudio}].

        Accepts the new free-layout ``videoClips`` ([{name, offset}]) or the
        legacy contiguous ``videos`` / ``video`` forms."""
        specs = []
        layout = params.get("videoClips")
        if layout:
            for s in layout:
                p = self.resolve_ref(s["name"])
                info = ffprobe(p)
                specs.append({"path": p, "offset": max(0.0, float(s.get("offset", 0.0))),
                              "duration": float(s.get("duration", info["duration"])),
                              "trimStart": max(0.0, float(s.get("trimStart", 0.0))),
                              "hasAudio": info["hasAudio"]})
        else:
            names = params.get("videos") or (
                [params["video"]] if params.get("video") else [])
            off = 0.0
            for n in names:
                p = self.media_file(n)
                info = ffprobe(p)
                specs.append({"path": p, "offset": off, "duration": info["duration"],
                              "hasAudio": info["hasAudio"]})
                off += info["duration"]
        if not specs:
            raise ValueError("No video selected")
        specs.sort(key=lambda c: c["offset"])
        return specs

    def render(self, params: dict) -> Path:
        specs = self._clip_specs(params)
        tmp = Path(tempfile.mkdtemp(prefix="vedit_"))
        try:
            # Lay the clips out on one master timeline (gaps -> black/silence,
            # overlaps -> crossfade) so the rest of the pipeline sees a single
            # continuous clip with audio.
            if len(specs) == 1 and specs[0]["offset"] < 1e-3:
                base_in = self._normalize_single(specs[0], tmp)
            else:
                base_in = self._compose(specs, tmp)
            info = ffprobe(base_in)
            cuts = normalize_cuts(params.get("cutRegions", []), info["duration"])
            if cuts and not params.get("editsAfterCuts"):
                base_in, params = self._apply_cuts(base_in, info, cuts, params, tmp)
                info = ffprobe(base_in)
            regions = normalize_regions(params.get("speedRegions", []), info["duration"])

            base, base_has_audio, new_duration = self._apply_speed(
                base_in, info, regions, tmp
            )
            output = self._mix_and_mux(
                base, base_has_audio, new_duration, regions, params, specs[0]["path"]
            )
            return output
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def _normalize_single(self, spec: dict, tmp: Path) -> Path:
        """Re-encode one clip to a predictable pixel format / audio layout."""
        src = spec["path"]
        target = ffprobe(src)
        w = target["width"] or 1280
        h = target["height"] or 720
        fps = target["fps"] or 30.0
        seg = tmp / "seg_000.mp4"
        vf = (f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
              f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps:.5f}")
        # Honour the clip's trim: seek to trimStart and read only `duration`.
        ts = max(0.0, float(spec.get("trimStart", 0.0)))
        seek = ["-ss", f"{ts:.4f}"] if ts > 1e-3 else []
        limit = ["-t", f"{float(spec['duration']):.4f}"]
        cmd = ["ffmpeg", "-y"]
        if not spec["hasAudio"]:
            cmd += [*seek, *limit, "-i", str(src), "-f", "lavfi", "-i",
                    "anullsrc=r=44100:cl=stereo", "-map", "0:v:0", "-map", "1:a:0",
                    "-shortest"]
        else:
            cmd += [*seek, *limit, "-i", str(src), "-map", "0:v:0", "-map", "0:a:0"]
        cmd += ["-vf", vf, "-ar", "44100", "-ac", "2",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", str(seg)]
        run(cmd)
        return seg

    def _compose(self, specs: list[dict], tmp: Path) -> Path:
        """Composite clips onto a black canvas at their offsets.

        Gaps show black (with silence); overlapping clips crossfade (the later
        clip dissolves in over the earlier one); a clip bordered by a gap fades
        to/from black over ``TRANS`` seconds. Produces one clip with audio."""
        target = ffprobe(specs[0]["path"])
        w = target["width"] or 1280
        h = target["height"] or 720
        fps = target["fps"] or 30.0
        total = max(s["offset"] + s["duration"] for s in specs)
        n = len(specs)

        # Per-clip fade in/out (alpha) lengths from neighbour overlaps / gaps.
        fades = []
        for i, s in enumerate(specs):
            off, dur, end = s["offset"], s["duration"], s["offset"] + s["duration"]
            prev_end = specs[i - 1]["offset"] + specs[i - 1]["duration"] if i > 0 else 0.0
            overlap_prev = prev_end - off
            gap_before = off - prev_end
            if overlap_prev > 1e-3:
                fade_in = min(dur / 2, overlap_prev)
            elif gap_before > 1e-3:
                fade_in = min(dur / 2, self.TRANS)
            else:
                fade_in = 0.0
            gap_after = (specs[i + 1]["offset"] - end) if i < n - 1 else 0.0
            fade_out = min(dur / 2, self.TRANS) if gap_after > 1e-3 else 0.0
            fades.append((fade_in, fade_out))

        inputs: list[str] = []
        for s in specs:
            # Seek to the clip's trimStart and read only its duration so cut
            # regions and trims drop out of the source before compositing.
            ts = max(0.0, float(s.get("trimStart", 0.0)))
            if ts > 1e-3:
                inputs += ["-ss", f"{ts:.4f}"]
            inputs += ["-t", f"{float(s['duration']):.4f}", "-i", str(s["path"])]

        vchain = [f"color=c=black:s={w}x{h}:r={fps:.5f}:d={total:.4f}[bg]"]
        achain = [f"anullsrc=r=44100:cl=stereo:d={total:.4f}[abase]"]
        mix_a = ["[abase]"]
        acc = "[bg]"
        for i, s in enumerate(specs):
            off, dur, end = s["offset"], s["duration"], s["offset"] + s["duration"]
            fi, fo = fades[i]
            vf = (f"[{i}:v]scale={w}:{h}:force_original_aspect_ratio=decrease,"
                  f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps:.5f},format=yuva420p")
            if fi > 0.01:
                vf += f",fade=t=in:st=0:d={fi:.4f}:alpha=1"
            if fo > 0.01:
                vf += f",fade=t=out:st={dur - fo:.4f}:d={fo:.4f}:alpha=1"
            vf += f",setpts=PTS-STARTPTS+{off:.4f}/TB[v{i}]"
            vchain.append(vf)
            lbl = f"[o{i}]"
            vchain.append(
                f"{acc}[v{i}]overlay=eof_action=pass:"
                f"enable='between(t,{off:.4f},{end:.4f})'{lbl}"
            )
            acc = lbl
            if s["hasAudio"]:
                af = f"[{i}:a]aformat=sample_rates=44100:channel_layouts=stereo,asetpts=PTS-STARTPTS"
                if fi > 0.01:
                    af += f",afade=t=in:st=0:d={fi:.4f}"
                if fo > 0.01:
                    af += f",afade=t=out:st={dur - fo:.4f}:d={fo:.4f}"
                delay = int(round(off * 1000))
                if delay > 0:
                    af += f",adelay={delay}|{delay}"
                af += f"[a{i}]"
                achain.append(af)
                mix_a.append(f"[a{i}]")
        achain.append(
            f"{''.join(mix_a)}amix=inputs={len(mix_a)}:duration=longest:normalize=0[aout]"
        )

        filt = ";".join(vchain + achain)
        out = tmp / "composite.mp4"
        run(["ffmpeg", "-y", *inputs, "-filter_complex", filt,
             "-map", acc, "-map", "[aout]", "-t", f"{total:.4f}",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
             "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", str(out)])
        return out

    def _apply_cuts(self, video, info, cuts, params, tmp):
        duration = info["duration"]
        has_audio = info["hasAudio"]
        if not cuts:
            return video, params
        keep = build_keep_segments(cuts, duration)
        if not keep:
            raise ValueError("Cut regions remove the entire video")

        parts, vlabels, alabels = [], [], []
        for i, (s, e) in enumerate(keep):
            parts.append(
                f"[0:v]trim=start={s:.4f}:end={e:.4f},setpts=PTS-STARTPTS[v{i}]"
            )
            vlabels.append(f"[v{i}]")
            if has_audio:
                parts.append(
                    f"[0:a]atrim=start={s:.4f}:end={e:.4f},asetpts=PTS-STARTPTS[a{i}]"
                )
                alabels.append(f"[a{i}]")
        filt = ";".join(parts)
        filt += f";{''.join(vlabels)}concat=n={len(vlabels)}:v=1:a=0[v]"
        maps = ["-map", "[v]"]
        if has_audio:
            filt += f";{''.join(alabels)}concat=n={len(alabels)}:v=0:a=1[a]"
            maps += ["-map", "[a]"]
        out = tmp / "cut.mp4"
        cmd = ["ffmpeg", "-y", "-i", str(video), "-filter_complex", filt, *maps,
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
               "-pix_fmt", "yuv420p"]
        if has_audio:
            cmd += ["-c:a", "aac", "-b:a", "192k"]
        cmd.append(str(out))
        run(cmd)

        new_params = dict(params)
        if not params.get("editsAfterCuts"):
            new_params["speedRegions"] = remap_cut_regions(params.get("speedRegions", []), cuts)
            new_params["captions"] = remap_cut_regions(params.get("captions", []), cuts)
            tts = []
            for clip in params.get("ttsClips", []):
                nc = dict(clip)
                nc["time"] = remap_cut_time(float(clip.get("time", 0.0)), cuts)
                if clip.get("end") is not None:
                    nc["end"] = remap_cut_time(float(clip["end"]), cuts)
                tts.append(nc)
            new_params["ttsClips"] = tts
            musics = []
            for clip in params.get("musics", []):
                nc = dict(clip)
                nc["start"] = remap_cut_time(float(clip.get("start", 0.0)), cuts)
                if clip.get("end") is not None:
                    nc["end"] = remap_cut_time(float(clip["end"]), cuts)
                musics.append(nc)
            new_params["musics"] = musics
        return out, new_params

    def _apply_speed(self, video, info, regions, tmp):
        duration = info["duration"]
        has_audio = info["hasAudio"]
        new_duration = remap_time(duration, regions)
        if not regions:
            return video, has_audio, duration

        segs = build_segments(regions, duration)
        vparts, alabels, vlabels = [], [], []
        for i, (s, e, f) in enumerate(segs):
            vparts.append(
                f"[0:v]trim=start={s:.4f}:end={e:.4f},setpts=(PTS-STARTPTS)/{f}[v{i}]"
            )
            vlabels.append(f"[v{i}]")
            if has_audio:
                vparts.append(
                    f"[0:a]atrim=start={s:.4f}:end={e:.4f},asetpts=PTS-STARTPTS,"
                    f"{atempo_chain(f)}[a{i}]"
                )
                alabels.append(f"[a{i}]")

        filt = ";".join(vparts)
        filt += f";{''.join(vlabels)}concat=n={len(vlabels)}:v=1:a=0[v]"
        cmd = ["ffmpeg", "-y", "-i", str(video), "-filter_complex"]
        maps = ["-map", "[v]"]
        if has_audio:
            filt += f";{''.join(alabels)}concat=n={len(alabels)}:v=0:a=1[a]"
            maps += ["-map", "[a]"]
        base = tmp / "base.mp4"
        cmd += [filt, *maps, "-c:v", "libx264", "-preset", "veryfast",
                "-crf", "20", "-pix_fmt", "yuv420p"]
        if has_audio:
            cmd += ["-c:a", "aac", "-b:a", "192k"]
        cmd.append(str(base))
        run(cmd)
        return base, has_audio, new_duration

    def _mix_and_mux(self, base, base_has_audio, new_duration, regions, params, src):
        tts_clips = params.get("ttsClips", [])
        music = params.get("music")
        base_vol = float(params.get("baseVolume", 1.0))
        tts_vol = float(params.get("ttsVolume", 1.5))
        music_vol = float(params.get("musicVolume", 0.6))

        inputs = ["-i", str(base)]
        idx = 1
        filt_parts: list[str] = []
        mix_labels: list[str] = []
        if base_has_audio and base_vol > 0:
            filt_parts.append(f"[0:a]volume={base_vol:.3f}[bA]")
            mix_labels.append("[bA]")
        for clip in tts_clips:
            clip_path = self.cfg.tts_dir / clip["file"]
            if not clip_path.exists():
                continue
            new_t = remap_time(float(clip["time"]), regions)
            delay = int(round(new_t * 1000))
            inputs += ["-i", str(clip_path)]
            chain = []
            # Optional end: clamp the voice-over to the picked output window.
            end = clip.get("end")
            if end is not None:
                win = remap_time(float(end), regions) - new_t
                if win > 0.01:
                    chain.append(f"atrim=end={win:.4f}")
                    chain.append("asetpts=PTS-STARTPTS")
            if delay > 0:
                chain.append(f"adelay={delay}|{delay}")
            chain.append(f"volume={tts_vol:.3f}")
            filt_parts.append(f"[{idx}:a]{','.join(chain)}[t{idx}]")
            mix_labels.append(f"[t{idx}]")
            idx += 1
        # Music tracks: a list of clips, each placed at a picked start with an
        # optional end. Overlapping clips are crossfaded (the earlier one fades
        # out while the later one fades in across the overlap).
        musics = params.get("musics") or ([music] if music else [])
        mclips = []
        if music_vol > 0:
            for m in musics:
                ref = m.get("ref") or m.get("path") or m.get("file")
                if not ref:
                    continue
                try:
                    if m.get("ref"):
                        mp = self.resolve_ref(m["ref"])
                    elif m.get("path"):
                        mp = self.external_file(m["path"])
                    else:
                        mp = self.media_file(m["file"])
                except (FileNotFoundError, ValueError):
                    continue
                si = remap_time(float(m.get("start", 0.0)), regions)
                end_src = m.get("end")
                ei = (remap_time(float(end_src), regions)
                      if end_src is not None else new_duration)
                if ei - si > 0.05:
                    mclips.append({"path": mp, "si": si, "ei": ei})
        mclips.sort(key=lambda c: c["si"])
        for i, c in enumerate(mclips):
            si, ei = c["si"], c["ei"]
            local_len = ei - si
            fade_in = max(0.0, min(local_len, mclips[i - 1]["ei"] - si)) if i > 0 else 0.0
            fade_out = (max(0.0, min(local_len, ei - mclips[i + 1]["si"]))
                        if i < len(mclips) - 1 else 0.0)
            inputs += ["-stream_loop", "-1", "-i", str(c["path"])]
            chain = [f"atrim=end={local_len:.4f}", "asetpts=PTS-STARTPTS"]
            if fade_in > 0.01:
                chain.append(f"afade=t=in:st=0:d={fade_in:.4f}")
            if fade_out > 0.01:
                chain.append(f"afade=t=out:st={local_len - fade_out:.4f}:d={fade_out:.4f}")
            delay = int(round(si * 1000))
            if delay > 0:
                chain.append(f"adelay={delay}|{delay}")
            chain.append(f"volume={music_vol:.3f}")
            filt_parts.append(f"[{idx}:a]{','.join(chain)}[m{idx}]")
            mix_labels.append(f"[m{idx}]")
            idx += 1

        # Always include a silent bed so the mix has a stable length and at
        # least two inputs (amix needs >=1, but this also guarantees duration).
        inputs += ["-f", "lavfi", "-i",
                   f"anullsrc=r=44100:cl=stereo"]
        filt_parts.append(f"[{idx}:a]volume=0[sZ]")
        mix_labels.append("[sZ]")
        idx += 1

        out_dir = self.cfg.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output = out_dir / f"{src.stem}_edited_{stamp}.mp4"

        # Subtitles / captions (times remapped through the speed regions).
        captions = params.get("captions", [])
        burn = bool(params.get("burnCaptions", True))
        srt_path = None
        if captions:
            srt_path = output.with_suffix(".srt")
            srt_path.write_text(build_srt(captions, regions), encoding="utf-8")

        filt = ";".join(filt_parts)
        filt += (
            f";{''.join(mix_labels)}amix=inputs={len(mix_labels)}:"
            f"duration=longest:normalize=0[mix]"
        )

        video_map = "0:v"
        if srt_path and burn:
            esc = str(srt_path).replace("\\", "\\\\").replace(":", "\\:")
            filt += (
                f";[0:v]subtitles=filename='{esc}':force_style="
                f"'FontSize=22,Outline=1,Shadow=0,MarginV=26,"
                f"BorderStyle=1,PrimaryColour=&H00FFFFFF&'[v]"
            )
            video_map = "[v]"

        cmd = ["ffmpeg", "-y", *inputs]
        if srt_path and not burn:
            cmd += ["-i", str(srt_path)]
        cmd += [
            "-filter_complex", filt,
            "-map", video_map, "-map", "[mix]",
        ]
        if srt_path and not burn:
            # Soft subtitle track muxed alongside the video (mov_text for mp4).
            # The SRT is the next input after all audio inputs (index == idx).
            cmd += ["-map", f"{idx}:s:0", "-c:s", "mov_text"]
        cmd += [
            "-t", f"{new_duration:.4f}",
            "-c:v", ("libx264" if (srt_path and burn) else "copy"),
        ]
        if srt_path and burn:
            cmd += ["-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"]
        cmd += ["-c:a", "aac", "-b:a", "192k", str(output)]
        run(cmd)
        return output


# --------------------------------------------------------------------------- #
# HTTP server
# --------------------------------------------------------------------------- #
# Keys persisted to / loaded from a config file (-c/--config) and editable
# from the web Settings panel. host/port/no_browser stay command-line only.
SETTINGS_KEYS = (
    "media_dir", "output_dir", "base_url", "api_key",
    "voice", "voice_name", "tts_model", "stt_model", "audio_model", "video",
)

# Auto-loaded from / saved back to the current directory when -c is not given,
# so UI-configured settings (API key, models, …) persist across restarts.
DEFAULT_CONFIG_NAME = "video_editor.config.json"


def safe_session_name(name: str | None) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", (name or "default").strip())
    return safe.strip("._-") or "default"


class Config:
    """Mutable runtime configuration, shared by reference with the Editor.

    Built from a merged settings dict (built-in defaults < config file < CLI <
    live web edits). Everything here can be changed from the browser and saved
    back to a config file."""

    def __init__(self, settings: dict, config_path: str | None = None,
                 session_name: str | None = None):
        self.tts_dir = Path(tempfile.mkdtemp(prefix="vedit_tts_"))
        self.upload_dir = Path(tempfile.mkdtemp(prefix="vedit_up_"))
        self.config_path = Path(config_path).expanduser().resolve() if config_path else None
        self.session_name = safe_session_name(session_name) if session_name else None
        self.session_path = (Path.home() / ".cache" / "coderai" / "video_editor" /
                             "sessions" / f"{self.session_name}.json") if self.session_name else None
        self.session_asset_dir = (self.session_path.with_suffix("") if self.session_path else None)
        self.media_dir = Path(settings["media_dir"]).expanduser().resolve()
        self.output_dir = Path(settings["output_dir"]).expanduser().resolve()
        self.base_url = (settings.get("base_url") or "").rstrip("/")
        self.api_key = settings.get("api_key") or None
        self.gender = settings.get("voice") or "feminine"
        self.default_voice = settings.get("voice_name") or DEFAULT_VOICE.get(self.gender, "af_sarah")
        self.tts_model = settings.get("tts_model") or None
        self.stt_model = settings.get("stt_model") or None
        self.audio_model = settings.get("audio_model") or None
        self.default_video = settings.get("video") or None

    @property
    def session_enabled(self) -> bool:
        return self.session_path is not None

    def update(self, patch: dict) -> None:
        """Apply a partial settings update coming from the web UI (validated)."""
        if patch.get("media_dir"):
            d = Path(patch["media_dir"]).expanduser()
            if not d.is_dir():
                raise ValueError(f"Media directory not found: {d}")
            self.media_dir = d.resolve()
        if patch.get("output_dir"):
            self.output_dir = Path(patch["output_dir"]).expanduser().resolve()
        if patch.get("base_url"):
            self.base_url = patch["base_url"].rstrip("/")
        if "api_key" in patch:
            self.api_key = patch["api_key"] or None
        if patch.get("voice"):
            self.gender = patch["voice"]
        if "voice_name" in patch:
            self.default_voice = (patch["voice_name"]
                                  or DEFAULT_VOICE.get(self.gender, "af_sarah"))
        if "tts_model" in patch:
            self.tts_model = patch["tts_model"] or None
        if "stt_model" in patch:
            self.stt_model = patch["stt_model"] or None
        if "audio_model" in patch:
            self.audio_model = patch["audio_model"] or None

    def to_dict(self) -> dict:
        return {
            "media_dir": str(self.media_dir),
            "output_dir": str(self.output_dir),
            "base_url": self.base_url,
            "api_key": self.api_key,
            "voice": self.gender,
            "voice_name": self.default_voice,
            "tts_model": self.tts_model,
            "stt_model": self.stt_model,
            "audio_model": self.audio_model,
            "video": self.default_video,
        }

    def save(self, path: str | None = None) -> Path:
        """Persist current settings to a JSON config file usable with -c."""
        p = Path(path).expanduser().resolve() if path else self.config_path
        if not p:
            raise ValueError("No config path given (set one in the Save field).")
        if p.suffix == "":
            p = p.with_suffix(".json")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        self.config_path = p
        return p

    def load_session(self) -> dict | None:
        if not self.session_path or not self.session_path.is_file():
            return None
        try:
            payload = json.loads(self.session_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            raise RuntimeError(f"Could not load session {self.session_path}: {e}")
        self._restore_session_assets()
        return payload

    def save_session(self, state: dict) -> Path:
        if not self.session_path:
            raise ValueError("Session persistence is not enabled")
        self.session_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_session_assets(state)
        payload = {
            "version": 1,
            "name": self.session_name,
            "savedAt": time.time(),
            "state": state,
        }
        tmp = self.session_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.session_path)
        return self.session_path

    def _save_session_assets(self, state: dict) -> None:
        if not self.session_asset_dir:
            return
        upload_assets = self.session_asset_dir / "uploads"
        tts_assets = self.session_asset_dir / "tts"
        for clip in state.get("clips", []):
            ref = str(clip.get("name", ""))
            if ref.startswith("upload:"):
                src = self.upload_dir / Path(ref[len("upload:"):]).name
                if src.is_file():
                    upload_assets.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, upload_assets / src.name)
        for music in state.get("musics", []):
            ref = str(music.get("ref", ""))
            if ref.startswith("upload:"):
                src = self.upload_dir / Path(ref[len("upload:"):]).name
                if src.is_file():
                    upload_assets.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, upload_assets / src.name)
        for clip in state.get("tts", []):
            name = Path(str(clip.get("file", ""))).name
            if name:
                src = self.tts_dir / name
                if src.is_file():
                    tts_assets.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, tts_assets / src.name)

    def _restore_session_assets(self) -> None:
        if not self.session_asset_dir:
            return
        for dirname, dest in (("uploads", self.upload_dir), ("tts", self.tts_dir)):
            src_dir = self.session_asset_dir / dirname
            if not src_dir.is_dir():
                continue
            for src in src_dir.iterdir():
                if src.is_file():
                    shutil.copy2(src, dest / src.name)


def make_handler(editor: Editor):
    cfg = editor.cfg

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        _client_disconnect_errors = (
            BrokenPipeError,
            ConnectionResetError,
            ConnectionAbortedError,
        )

        def log_message(self, *a):  # quieter logs
            pass

        def handle(self):
            try:
                super().handle()
            except self._client_disconnect_errors:
                return

        # -- helpers ------------------------------------------------------- #
        def _json(self, obj, status=200):
            body = json.dumps(obj).encode("utf-8")
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except self._client_disconnect_errors:
                return

        def _read_json(self):
            length = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(length) or b"{}")

        def _send_file(self, path: Path, content_type=None):
            file_size = path.stat().st_size
            ctype = content_type or guess_type(path)
            rng = self.headers.get("Range")
            if rng and rng.startswith("bytes="):
                start_s, _, end_s = rng[6:].partition("-")
                start = int(start_s) if start_s else 0
                end = int(end_s) if end_s else file_size - 1
                end = min(end, file_size - 1)
                length = end - start + 1
                self.send_response(206)
                self.send_header("Content-Type", ctype)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                self.send_header("Content-Length", str(length))
                self.end_headers()
                with path.open("rb") as f:
                    f.seek(start)
                    remaining = length
                    while remaining > 0:
                        chunk = f.read(min(65536, remaining))
                        if not chunk:
                            break
                        try:
                            self.wfile.write(chunk)
                        except self._client_disconnect_errors:
                            return
                        remaining -= len(chunk)
            else:
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(file_size))
                self.end_headers()
                with path.open("rb") as f:
                    try:
                        shutil.copyfileobj(f, self.wfile)
                    except self._client_disconnect_errors:
                        return

        # -- reverse-proxy helpers ---------------------------------------- #
        def _public_prefix(self):
            """Path prefix this app is mounted under, per reverse-proxy headers.

            Returns e.g. '/editor' (no trailing slash) or '' when mounted at
            root. Lets the same server work behind `location /editor/ { … }`
            regardless of whether nginx strips the prefix."""
            p = (self.headers.get("X-Forwarded-Prefix")
                 or self.headers.get("X-Script-Name") or "")
            p = p.strip().rstrip("/")
            return p if p.startswith("/") else (("/" + p) if p else "")

        def _route(self, path):
            """Strip the forwarded prefix so internal routing is mount-agnostic."""
            pref = self._public_prefix()
            if pref and (path == pref or path.startswith(pref + "/")):
                path = path[len(pref):] or "/"
            return path

        # -- routing ------------------------------------------------------- #
        def do_GET(self):
            parsed = urlparse(self.path)
            route = self._route(parsed.path)
            qs = parse_qs(parsed.query)
            try:
                if route == "/":
                    self._send_html()
                elif route == "/api/config":
                    self._json(self._config_payload())
                elif route == "/api/videos":
                    self._json(self._list_media())
                elif route == "/api/settings":
                    self._json(self._settings_payload())
                elif route == "/api/session":
                    if not cfg.session_enabled:
                        self._json({"enabled": False, "state": None})
                    else:
                        payload = cfg.load_session()
                        self._json({
                            "enabled": True,
                            "name": cfg.session_name,
                            "path": str(cfg.session_path),
                            "state": (payload or {}).get("state"),
                            "savedAt": (payload or {}).get("savedAt"),
                        })
                elif route == "/api/models":
                    models = editor.list_models()
                    tts = detect_model(models, ("kokoro", "tts", "speech"))
                    stt = detect_model(models, ("whisper", "stt", "asr", "transcrib"))
                    audio = detect_model(models, ("musicgen", "audiogen", "audioldm", "audio_generation", "text-to-audio"))
                    self._json({"models": models, "modelIds": [m.get("id") for m in models if m.get("id")],
                                "detected": {"tts_model": tts, "stt_model": stt, "audio_model": audio}})
                elif route == "/api/probe":
                    if qs.get("ref"):
                        info = ffprobe(editor.resolve_ref(qs["ref"][0]))
                    elif qs.get("path"):
                        info = ffprobe(editor.external_file(qs["path"][0]))
                    else:
                        info = ffprobe(editor.media_file(qs["name"][0]))
                    self._json(info)
                elif route == "/api/waveform":
                    points = int((qs.get("points") or ["1600"])[0])
                    if qs.get("ref"):
                        wf = audio_waveform(editor.resolve_ref(qs["ref"][0]), points)
                    elif qs.get("path"):
                        wf = audio_waveform(editor.external_file(qs["path"][0]), points)
                    else:
                        wf = audio_waveform(editor.media_file(qs["name"][0]), points)
                    self._json(wf)
                elif route == "/api/motion":
                    points = int((qs.get("points") or ["900"])[0])
                    if qs.get("ref"):
                        mg = motion_graph(editor.resolve_ref(qs["ref"][0]), points)
                    elif qs.get("path"):
                        mg = motion_graph(editor.external_file(qs["path"][0]), points)
                    else:
                        mg = motion_graph(editor.media_file(qs["name"][0]), points)
                    self._json(mg)
                elif route == "/media":
                    self._send_file(editor.media_file(qs["name"][0]))
                elif route == "/upload":
                    self._send_file(
                        editor._resolve(cfg.upload_dir, Path(qs["name"][0]).name))
                elif route == "/tts":
                    self._send_file(cfg.tts_dir / Path(qs["name"][0]).name, "audio/wav")
                elif route == "/out":
                    self._send_file(
                        editor._resolve(cfg.output_dir, qs["name"][0]),
                        "application/octet-stream",
                    )
                else:
                    self._json({"error": "not found"}, 404)
            except FileNotFoundError as e:
                self._json({"error": f"not found: {e}"}, 404)
            except self._client_disconnect_errors:
                return
            except Exception as e:  # pragma: no cover - surfaced to UI
                self._json({"error": str(e)}, 500)

        def do_POST(self):
            parsed = urlparse(self.path)
            route = self._route(parsed.path)
            qs = parse_qs(parsed.query)
            try:
                if route == "/api/upload":
                    length = int(self.headers.get("Content-Length", 0))
                    fname = (qs.get("name") or ["upload"])[0]
                    saved = editor.save_upload(fname, self.rfile, length)
                    info = ffprobe(saved)
                    self._json({
                        "ref": f"upload:{saved.name}",
                        "label": Path(fname).name,
                        "duration": info["duration"],
                        "fps": info["fps"],
                        "hasVideo": info["width"] > 0,
                        "hasAudio": info["hasAudio"],
                    })
                elif route == "/api/tts":
                    body = self._read_json()
                    path = editor.tts(body["text"], body.get("voice", ""),
                                      body.get("speed", 1.0), body.get("emotion", ""),
                                      body.get("style", ""))
                    info = ffprobe(path)
                    self._json({
                        "file": path.name,
                        "url": f"tts?name={path.name}",
                        "duration": info["duration"],
                    })
                elif route == "/api/settings":
                    body = self._read_json()
                    cfg.update(body.get("settings", {}))
                    editor.refresh_auth()
                    saved = None
                    if body.get("save"):
                        saved = str(cfg.save(body.get("path") or None))
                    payload = self._settings_payload()
                    payload["saved"] = saved
                    self._json(payload)
                elif route == "/api/session":
                    body = self._read_json()
                    path = cfg.save_session(body.get("state", {}))
                    self._json({"saved": True, "path": str(path), "name": cfg.session_name})
                elif route == "/api/transcribe":
                    body = self._read_json()
                    segs = editor.transcribe(body["name"], body.get("language"))
                    self._json({"segments": segs})
                elif route == "/api/musicgen":
                    body = self._read_json()
                    path = editor.generate_music(
                        body.get("prompt", ""), body.get("duration", 10), body.get("model") or None
                    )
                    info = ffprobe(path)
                    self._json({
                        "ref": f"upload:{path.name}",
                        "label": body.get("label") or "generated music",
                        "file": path.name,
                        "url": f"upload?name={path.name}",
                        "duration": info["duration"],
                    })
                elif route == "/api/render":
                    body = self._read_json()
                    out = editor.render(body)
                    self._json({"output": str(out), "url": f"out?name={out.name}"})
                else:
                    self._json({"error": "not found"}, 404)
            except Exception as e:  # pragma: no cover
                import traceback
                traceback.print_exc()
                self._json({"error": str(e)}, 500)

        # -- payload builders --------------------------------------------- #
        def _config_payload(self):
            return {
                "gender": cfg.gender,
                "defaultVoice": cfg.default_voice,
                "voices": VOICES,
                "speedFactors": SPEED_FACTORS,
                "ttsModel": cfg.tts_model,
                "ttsEmotions": tts_emotions(cfg.tts_model),
                "ttsStyles": tts_styles(cfg.tts_model),
                "sttModel": cfg.stt_model,
                "audioModel": cfg.audio_model,
                "defaultVideo": cfg.default_video,
                "mediaDir": str(cfg.media_dir),
                "session": {
                    "enabled": cfg.session_enabled,
                    "name": cfg.session_name,
                    "path": str(cfg.session_path or ""),
                },
                "settings": self._settings_payload(),
            }

        def _settings_payload(self):
            return {
                "settings": cfg.to_dict(),
                "configPath": str(cfg.config_path or ""),
                "models": editor.list_models(),
            }

        def _list_media(self):
            videos, audios = [], []
            for p in sorted(cfg.media_dir.rglob("*")):
                if not p.is_file():
                    continue
                rel = str(p.relative_to(cfg.media_dir))
                if p.suffix.lower() in VIDEO_EXTS:
                    videos.append(rel)
                elif p.suffix.lower() in AUDIO_EXTS:
                    audios.append(rel)
            return {"videos": videos, "audios": audios}

        def _send_html(self):
            # Inject a <base> so the page's relative URLs resolve against the
            # public mount point (works at root or behind `location /editor/`).
            base = (self._public_prefix() or "") + "/"
            html = HTML.replace("<head>", f'<head>\n<base href="{base}">', 1)
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def guess_type(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".mp4": "video/mp4", ".webm": "video/webm", ".mkv": "video/x-matroska",
        ".mov": "video/quicktime", ".m4v": "video/x-m4v", ".avi": "video/x-msvideo",
        ".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4",
        ".aac": "audio/aac", ".flac": "audio/flac", ".ogg": "audio/ogg",
    }.get(ext, "application/octet-stream")


# --------------------------------------------------------------------------- #
# TTS model auto-detection
# --------------------------------------------------------------------------- #
def _model_items(payload) -> list[dict]:
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = payload.get("data") or payload.get("models") or payload.get("items") or []
    else:
        items = []
    out = []
    for item in items:
        if isinstance(item, str):
            out.append({"id": item})
        elif isinstance(item, dict):
            mid = item.get("id") or item.get("name") or item.get("model")
            if mid:
                x = dict(item)
                x["id"] = mid
                out.append(x)
    return out


def _list_models(base_url: str, api_key: str | None) -> list[dict]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    base = (base_url or "").rstrip("/")
    if not base:
        return []
    for suffix in ("/v1/models", "/models", "/api/models"):
        try:
            resp = requests.get(f"{base}{suffix}", headers=headers, timeout=15)
            if resp.ok:
                models = _model_items(resp.json())
                if models:
                    return models
        except (requests.RequestException, ValueError):
            continue
    return []


def detect_model(models: list[dict], keywords: tuple[str, ...]) -> str | None:
    scored = []
    for m in models:
        mid = str(m.get("id") or "")
        hay = " ".join(str(m.get(k, "")) for k in ("id", "name", "type", "task", "owned_by", "description")).lower()
        score = sum(1 for k in keywords if k in hay)
        if score:
            scored.append((score, mid))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored[0][1] if scored else None


# --------------------------------------------------------------------------- #
# Front-end (single page)
# --------------------------------------------------------------------------- #
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CoderAI Video Editor</title>
<style>
  :root{
    --bg:#0d1017; --panel:#161b26; --panel2:#1d2433; --line:#2a3245;
    --txt:#e6ebf5; --muted:#8a96ad; --accent:#5b8cff; --accent2:#37d9a0;
    --warn:#ffb454; --danger:#ff6b6b;
  }
  *{box-sizing:border-box}
  body{margin:0;font:14px/1.45 system-ui,Segoe UI,Roboto,sans-serif;
       background:var(--bg);color:var(--txt)}
  header{display:flex;align-items:center;gap:14px;padding:12px 18px;
         background:var(--panel);border-bottom:1px solid var(--line)}
  header h1{font-size:16px;margin:0;font-weight:650;letter-spacing:.3px}
  header .sub{color:var(--muted);font-size:12px}
  main{display:grid;grid-template-columns:1fr 7px var(--rightw,340px);gap:12px;padding:16px;
       max-width:1500px;margin:0 auto;height:calc(100vh - 49px);overflow:hidden}
  #splitter{cursor:col-resize;align-self:stretch;width:7px;border-radius:4px;
            background:var(--line);transition:background .15s}
  #splitter:hover,#splitter.drag{background:var(--accent,#5b8cff)}
  .col{display:flex;flex-direction:column;gap:16px;min-width:0;overflow:auto;min-height:0}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;
        padding:14px}
  .card h2{margin:0 0 10px;font-size:13px;text-transform:uppercase;
           letter-spacing:.6px;color:var(--muted)}
  select,input,textarea,button{font:inherit;color:var(--txt);
        background:var(--panel2);border:1px solid var(--line);border-radius:8px;
        padding:8px 10px}
  select,input[type=text],textarea{width:100%}
  textarea{resize:vertical;min-height:64px}
  button{cursor:pointer;background:var(--accent);border:none;font-weight:600}
  button:hover{filter:brightness(1.08)}
  button.ghost{background:var(--panel2);border:1px solid var(--line);font-weight:500}
  button.sm{padding:5px 9px;font-size:12px}
  button:disabled{opacity:.5;cursor:not-allowed}
  .row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
  .row.tight{gap:6px}
  label.field{display:block;margin:8px 0 4px;color:var(--muted);font-size:12px}
  video{width:100%;background:#000;border-radius:10px;max-height:46vh}
  .preview-stage{position:relative;background:#000;border-radius:10px;margin-top:10px;overflow:hidden}
  .preview-stage video{margin-top:0;border-radius:10px;display:block;cursor:pointer;
                       width:100%;height:auto}
  .preview-stage:fullscreen{display:flex;align-items:center;justify-content:center;border-radius:0}
  .preview-stage:fullscreen video{width:100%;height:100%;object-fit:contain;border-radius:0}
  .fs-timeline{position:absolute;left:0;right:0;bottom:0;height:92px;box-sizing:border-box;
               background:rgba(10,13,20,.82);border-top:1px solid rgba(255,255,255,.12);
               padding:6px 10px;display:none;z-index:6}
  .fs-timeline canvas{width:100%;height:100%;display:block;cursor:pointer}
  /* Floating preview docked at the top of the right column when scrolled away */
  #previewDock{position:sticky;top:0;z-index:8;margin-bottom:8px}
  #previewDock:empty{display:none}
  .preview-stage.docked{margin-top:0;box-shadow:0 8px 22px rgba(0,0,0,.55);
                        border:1px solid var(--line)}
  .preview-stage.docked video{width:100%;height:auto;max-height:34vh;object-fit:contain}
  .preview-stage.docked .fs-timeline{display:none!important}
  /* docked overlays: scale the subtitle down so it never covers the small video
     (zoom shrinks the box + reflows wrapping; transform is the fallback). */
  .preview-stage.docked .preview-caption{bottom:5%;line-height:1.15;
                                         text-shadow:0 1px 3px #000,0 0 2px #000;zoom:.5}
  .preview-stage.docked .preview-time{font-size:10px;padding:2px 5px;top:5px;right:5px}
  .preview-black{position:absolute;inset:0;background:#000;display:none;pointer-events:none}
  .preview-caption{position:absolute;left:5%;right:5%;bottom:7%;text-align:center;
                   font-weight:700;text-shadow:0 2px 5px #000,0 0 2px #000;
                   pointer-events:none;white-space:pre-wrap}
  .preview-time{position:absolute;right:8px;top:8px;background:rgba(0,0,0,.65);
                border:1px solid rgba(255,255,255,.18);border-radius:6px;
                padding:3px 7px;font-size:12px;font-variant-numeric:tabular-nums;
                pointer-events:none}
  /* timeline */
  .tl-controls{display:flex;gap:6px;align-items:center;flex-wrap:wrap;
               margin:10px 0 8px}
  .tc{font-variant-numeric:tabular-nums;background:var(--panel2);
      border:1px solid var(--line);border-radius:8px;padding:6px 10px;
      font-size:13px}
  input.tc{width:118px;padding:5px 8px}
  .tl-scroll{position:relative;overflow-x:auto;overflow-y:hidden;
             border:1px solid var(--line);border-radius:10px;background:#0a0d14}
  .tl-canvas{display:block;position:sticky;left:0;top:0}
  .tl-spacer{height:1px}
  .hover-preview{position:fixed;display:none;z-index:20;pointer-events:none;
                 background:#05070b;border:1px solid var(--line);border-radius:8px;
                 padding:4px;box-shadow:0 8px 24px rgba(0,0,0,.45)}
  .hover-preview video{display:block;width:180px;max-height:110px;border-radius:6px;background:#000}
  .hover-preview .time{font-size:11px;color:var(--muted);margin-top:3px;text-align:center}
  .zoom{display:flex;align-items:center;gap:8px;margin-left:auto}
  .zoom input[type=range]{width:160px}
  .list{display:flex;flex-direction:column;gap:6px;margin-top:8px;
        max-height:180px;overflow:auto}
  .item{display:flex;align-items:center;gap:8px;background:var(--panel2);
        border:1px solid var(--line);border-radius:8px;padding:6px 8px;font-size:12px}
  .item.sel{box-shadow:0 0 0 2px var(--accent,#5b8cff) inset;border-color:var(--accent,#5b8cff);
            background:rgba(91,140,255,.12)}
  .item .grow{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;
              white-space:nowrap}
  .pill{font-size:11px;padding:2px 7px;border-radius:99px;background:#23304a;
        color:#cfe0ff}
  .pill.spd{background:#3a2e16;color:#ffd9a0}
  .x{color:var(--danger);cursor:pointer;font-weight:700;padding:0 4px}
  .hint{color:var(--muted);font-size:12px;margin-top:6px}
  .status{font-size:12px;color:var(--muted);min-height:16px}
  .status.err{color:var(--danger)} .status.ok{color:var(--accent2)}
  .status.busy{color:#cfe0ff}
  .op-progress{display:none;margin-top:8px;background:#101722;border:1px solid var(--line);
               border-radius:999px;height:8px;overflow:hidden}
  .op-progress.on{display:block}
  .op-progress .bar{width:0%;height:100%;background:linear-gradient(90deg,var(--accent),var(--accent2));
                    border-radius:999px;transition:width .25s ease}
  .op-progress.indeterminate .bar{width:38%;animation:opSlide 1.1s ease-in-out infinite}
  @keyframes opSlide{0%{transform:translateX(-110%)}100%{transform:translateX(290%)}}
  .seg{display:flex;gap:8px}
  .seg>div{flex:1}
  input[type=range]{accent-color:var(--accent)}
  .save{background:var(--accent2);color:#06241a;width:100%;padding:11px;
        font-size:15px}
  a.dl{color:var(--accent2);font-weight:600}
  .badge{font-size:11px;color:var(--muted)}
</style>
</head>
<body>
<header>
  <h1>🎬 CoderAI Video Editor</h1>
  <span class="sub" id="subtitle"></span>
</header>
<main>
  <div class="col">
    <div class="card">
      <h2>Source playlist</h2>
      <label class="field">From the server media directory</label>
      <div class="row">
        <select id="videoSel" style="flex:1"></select>
        <button class="sm" id="addClip">＋ Add to timeline</button>
        <button class="ghost sm" id="reloadBtn" title="rescan media dir">↻</button>
      </div>
      <label class="field">…or upload from this computer</label>
      <input type="file" id="videoFile" accept="video/*">
      <div class="list" id="clipList"></div>
      <div id="previewHome">
        <div class="preview-stage" id="previewStage">
          <video id="player" controls preload="metadata"></video>
          <div class="preview-black" id="previewBlack"></div>
          <div class="preview-caption" id="previewCaption"></div>
          <div class="preview-time" id="previewTime">00:00:00.000</div>
          <div class="fs-timeline" id="fsTimeline"><canvas id="tlMini"></canvas></div>
        </div>
      </div>
      <label class="row" style="margin-top:8px;gap:6px;color:var(--muted)">
        <input type="checkbox" id="playOriginal" style="width:auto">
        Play original selected clip instead of timeline/final preview
      </label>
      <div class="tl-controls">
        <button class="ghost sm" id="frPrev">⟨ frame</button>
        <button class="ghost sm" id="secPrev">−1s</button>
        <button class="ghost sm" id="playBtn">▶︎ / ❚❚</button>
        <button class="ghost sm" id="secNext">+1s</button>
        <button class="ghost sm" id="frNext">frame ⟩</button>
        <button class="ghost sm" id="selFromPlay">select from playhead</button>
        <label class="row" style="gap:4px;color:var(--muted)">
          <input type="checkbox" id="followPlayhead" checked style="width:auto"> follow
        </label>
        <label class="row" style="gap:4px;color:var(--muted)" title="Lock all tracks: dragging any clip moves video, voice-over, music and captions together. Unlock to move a single piece.">
          <input type="checkbox" id="lockTracks" style="width:auto"> 🔒 lock
        </label>
        <span class="tc" id="tcode">00:00:00.000</span>
        <span class="tc" id="frameNo">f 0</span>
        <button class="ghost sm" id="rateDown">slower</button>
        <span class="tc" id="rateV">1×</span>
        <button class="ghost sm" id="rateUp">faster</button>
        <span class="zoom">
          <span class="badge">horizontal</span>
          <button class="ghost sm" id="zOut">−</button>
          <input type="range" id="zoom" min="-1000" max="1000" value="120">
          <button class="ghost sm" id="zIn">+</button>
        </span>
        <span class="zoom">
          <span class="badge">vertical</span>
          <button class="ghost sm" id="zyOut">−</button>
          <input type="range" id="zoomY" min="0" max="100" value="35">
          <button class="ghost sm" id="zyIn">+</button>
        </span>
      </div>
      <div class="tl-scroll" id="tlScroll">
        <canvas class="tl-canvas" id="tl" height="222"></canvas>
        <div class="tl-spacer" id="tlSpacer"></div>
      </div>
      <div class="hint">Click empty space to seek. <b>Drag</b> a
        <span style="color:#37d9a0">▮ voice-over</span>,
        <span style="color:#7aa2ff">▮ music</span> or
        <span style="color:#d9a0ff">▮ caption</span> bar to move it; drag a
        <span style="color:#ffd479">▮ video</span> bar to reorder the playlist;
        click a bar's <b>✕</b> to delete it. <b>Right-click</b> the timeline to
        set a shared start/end selection for speed, cuts, voice, music and captions.
        Video bars move <b>freely</b> (gaps
        play black, overlaps crossfade) and <b>snap</b> to neighbouring edges.
        Zoom in until single frames spread apart, then step with
        <b>⟨ frame</b> / <b>frame ⟩</b>. Click a voice/music clip and press
        <b>Ctrl/Cmd+C</b>, then click or hover a timeline position and press
        <b>Ctrl/Cmd+V</b> to duplicate it there.</div>
    </div>
  </div>

  <div id="splitter" title="Drag to resize"></div>
  <div class="col" id="rightCol">
    <div id="previewDock"></div>
    <details class="card" id="cfgCard">
      <summary style="cursor:pointer;color:var(--muted);font-size:13px;
               text-transform:uppercase;letter-spacing:.6px">⚙ Configuration</summary>
      <label class="field">CoderAI base URL</label>
      <input type="text" id="cfgBaseUrl" placeholder="http://127.0.0.1:8000">
      <label class="field">API key (optional)</label>
      <input type="password" id="cfgApiKey" placeholder="leave blank if none">
      <label class="field">Media directory (videos &amp; audio to browse)</label>
      <input type="text" id="cfgMediaDir">
      <label class="field">Output directory</label>
      <input type="text" id="cfgOutputDir">
      <div class="seg">
        <div>
          <label class="field">TTS model
            <button class="ghost sm" id="cfgRefreshModels" type="button"
              style="float:right;padding:1px 7px">↻ models</button></label>
          <input type="text" id="cfgTtsModel" list="modelList" placeholder="kokoro">
        </div>
        <div>
          <label class="field">STT model (optional)</label>
          <input type="text" id="cfgSttModel" list="modelList" placeholder="whisper">
        </div>
      </div>
      <label class="field">Music generation model (optional)</label>
      <input type="text" id="cfgAudioModel" list="modelList" placeholder="facebook/musicgen-small">
      <datalist id="modelList"></datalist>
      <label class="field">Default voice gender</label>
      <select id="cfgGender">
        <option value="feminine">feminine</option>
        <option value="masculine">masculine</option>
      </select>
      <div class="row" style="margin-top:10px">
        <button id="cfgApply" style="flex:1">Apply now</button>
      </div>
      <label class="field">Save to config file</label>
      <div class="row tight">
        <input type="text" id="cfgSavePath" style="flex:1"
               placeholder="video_editor.config.json">
        <button class="ghost sm" id="cfgSave">💾 Save</button>
      </div>
      <div class="hint">Reload later with <code>video_editor.py -c &lt;file&gt;</code>.
        “Apply now” updates the running session (and re-reads the media directory).</div>
      <div class="status" id="cfgStatus"></div>
    </details>

    <div class="card">
      <h2>① Voice over (AI TTS)</h2>
      <textarea id="ttsText" placeholder="Type what should be spoken…"></textarea>
      <label class="field">Voice</label>
      <select id="voiceSel"></select>
      <div class="row tight" style="margin-top:8px">
        <div style="flex:1">
          <label class="field">Speed</label>
          <select id="ttsSpeed"></select>
        </div>
        <div style="flex:1" id="ttsEmotionRow" hidden>
          <label class="field">Emotion</label>
          <select id="ttsEmotion"></select>
        </div>
        <div style="flex:1" id="ttsStyleRow" hidden>
          <label class="field">Delivery</label>
          <select id="ttsStyle"></select>
        </div>
      </div>
      <div class="row tight" style="margin-top:8px">
        <button class="ghost sm" id="ttsStartB">Set start</button>
        <input class="tc time-edit" id="ttsStartV" data-var="ttsStart" placeholder="playhead">
        <button class="ghost sm" id="ttsEndB">Optional stop</button>
        <input class="tc time-edit" id="ttsEndV" data-var="ttsEnd" placeholder="none">
        <button class="ghost sm" id="ttsClr">clear</button>
      </div>
      <label class="row" style="margin-top:8px;gap:6px;color:var(--muted)">
        <input type="checkbox" id="ttsAutoCap" checked style="width:auto">
        Automatically create subtitle/caption from this TTS text
      </label>
      <div class="row" style="margin-top:8px">
        <button id="ttsAdd" style="flex:1">＋ Speak from start</button>
      </div>
      <div class="row tight" style="margin-top:6px">
        <button class="ghost sm" id="ttsRegenAll" style="flex:1" title="Re-synthesize every voice-over with its saved text/voice/speed">↻ Regenerate all voice</button>
        <button class="ghost sm" id="ttsRegenCaps" style="flex:1" title="Regenerate every voice-over and rebuild the subtitles aligned to them">↻ Regen + re-sync subtitles</button>
      </div>
      <div class="hint">Start defaults to the playhead. Optional stop trims the
        voice-over and caption to that window; leave it unset to use the full TTS duration. Drag clips on the timeline to move them; click
        their ✕ to delete.</div>
      <div class="list" id="ttsList"></div>
    </div>

    <div class="card">
      <h2>② Speed-up region</h2>
      <div class="row tight">
        <button class="ghost sm" id="spStart">Set start</button>
        <input class="tc time-edit" id="spStartV" data-var="spStart" placeholder="—">
        <button class="ghost sm" id="spEnd">Set end</button>
        <input class="tc time-edit" id="spEndV" data-var="spEnd" placeholder="—">
      </div>
      <label class="field">Acceleration</label>
      <select id="spFactor"></select>
      <div class="row" style="margin-top:8px">
        <button id="spAdd" style="flex:1">＋ Add speed region</button>
      </div>
      <div class="list" id="spList"></div>
    </div>

    <div class="card">
      <h2>③ Cut / remove region</h2>
      <div class="row tight">
        <button class="ghost sm" id="cutUseSel">Use timeline selection</button>
        <button class="ghost sm" id="cutStart">Set start</button>
        <input class="tc time-edit" id="cutStartV" data-var="cutStartSel" placeholder="—">
        <button class="ghost sm" id="cutEnd">Set end</button>
        <input class="tc time-edit" id="cutEndV" data-var="cutEndSel" placeholder="—">
      </div>
      <div class="row" style="margin-top:8px">
        <button id="cutAdd" style="flex:1">＋ Remove selected region</button>
      </div>
      <div class="hint">Cut regions are removed from the rendered video and audio.</div>
      <div class="list" id="cutList"></div>
    </div>

    <div class="card">
      <h2>④ Music tracks</h2>
      <label class="field">From the server media directory</label>
      <select id="musicSel"></select>
      <label class="field">…or upload from this computer</label>
      <input type="file" id="musicFile" accept="audio/*,video/*">
      <label class="field">…or a server file path (absolute)</label>
      <input type="text" id="musicPath" placeholder="/path/to/track.mp3">
      <label class="field">…or generate with MusicGen</label>
      <textarea id="musPrompt" placeholder="Prompt: upbeat cinematic synthwave loop, warm bass, no vocals…"></textarea>
      <div class="row tight" style="margin-top:8px">
        <label class="badge">Duration (s)</label>
        <input class="tc" id="musDur" type="number" min="1" max="300" step="0.5" value="10">
        <button class="ghost sm" id="musGenerate">Generate music</button>
      </div>
      <div class="badge" id="musUpName" style="margin-top:4px"></div>
      <div class="row tight" style="margin-top:8px">
        <button class="ghost sm" id="musStartB">Set start</button>
        <input class="tc time-edit" id="musStartV" data-var="musStartSel" placeholder="playhead">
        <button class="ghost sm" id="musEndB">Set end</button>
        <input class="tc time-edit" id="musEndV" data-var="musEndSel" placeholder="to end">
        <button class="ghost sm" id="musClr">clear</button>
      </div>
      <label class="field">Music volume <span id="musVolV">60%</span></label>
      <input type="range" id="musVol" min="0" max="200" value="60">
      <div class="row" style="margin-top:8px">
        <button id="musAdd" style="flex:1">＋ Add music at selection</button>
      </div>
      <div class="hint">No end = play (looped) to the end. Overlapping tracks
        crossfade automatically. Drag on the timeline to move; click ✕ to delete.</div>
      <div class="list" id="musList"></div>
    </div>

    <div class="card">
      <h2>⑤ Captions / subtitles</h2>
      <textarea id="capText" placeholder="Caption text shown on screen…"></textarea>
      <div class="row tight" style="margin-top:8px">
        <button class="ghost sm" id="capStart">Set start</button>
        <input class="tc time-edit" id="capStartV" data-var="capStart" placeholder="—">
        <button class="ghost sm" id="capEnd">Set end</button>
        <input class="tc time-edit" id="capEndV" data-var="capEnd" placeholder="—">
      </div>
      <div class="row" style="margin-top:8px">
        <button id="capAdd" style="flex:1">＋ Add caption</button>
        <button class="ghost sm" id="capFromTts" title="Replace all subtitles with ones generated from the voice-overs">↻ Regenerate from voice-overs</button>
      </div>
      <div class="row" style="margin-top:8px">
        <button class="ghost sm" id="capTranscribe" title="Replace all subtitles by transcribing the original audio">⟲ Auto-caption (replace)</button>
      </div>
      <label class="row" style="margin-top:8px;gap:6px;color:var(--muted)">
        <input type="checkbox" id="capBurn" checked style="width:auto">
        Burn captions into the video (off = soft subtitle track + .srt sidecar)
      </label>
      <div class="list" id="capList"></div>
    </div>

    <div class="card">
      <h2>⑥ Mix &amp; render</h2>
      <label class="field">Original audio <span id="baseVolV">100%</span></label>
      <input type="range" id="baseVol" min="0" max="200" value="100">
      <label class="field">Voice-over volume <span id="ttsVolV">150%</span></label>
      <input type="range" id="ttsVol" min="0" max="300" value="150">
      <button class="save" id="saveBtn">💾 Render &amp; save</button>
      <div class="status" id="status"></div>
      <div class="op-progress" id="opProgress"><div class="bar" id="opProgressBar"></div></div>
    </div>
  </div>
</main>
<div class="hover-preview" id="hoverPreview">
  <video id="hoverVideo" muted preload="metadata"></video>
  <div class="time" id="hoverTime">00:00:00.000</div>
</div>

<div id="editOverlay" style="position:fixed;inset:0;background:rgba(0,0,0,.55);z-index:50;display:none;align-items:center;justify-content:center">
  <div class="card" style="width:min(560px,92vw);max-height:86vh;overflow:auto;margin:0">
    <h2 id="editTitle">Edit</h2>
    <label class="field">Text</label>
    <textarea id="editText" rows="3"></textarea>
    <div id="editVoiceWrap">
      <div class="row tight" style="margin-top:8px">
        <div style="flex:2"><label class="field">Voice</label><select id="editVoice"></select></div>
        <div style="flex:1"><label class="field">Speed</label><select id="editSpeed"></select></div>
      </div>
      <div class="row tight" style="margin-top:8px">
        <div style="flex:1" id="editEmotionWrap" hidden><label class="field">Emotion</label><select id="editEmotion"></select></div>
        <div style="flex:1" id="editStyleWrap" hidden><label class="field">Delivery</label><select id="editStyle"></select></div>
      </div>
    </div>
    <div class="row tight" style="margin-top:8px">
      <div style="flex:1"><label class="field">Start</label><input class="tc" id="editStart"></div>
      <div style="flex:1"><label class="field">End <span style="opacity:.6">(blank = full)</span></label><input class="tc" id="editEnd"></div>
    </div>
    <div class="row" style="margin-top:12px;gap:8px">
      <button id="editSave" style="flex:1">Save</button>
      <button id="editRegen" class="ghost" style="flex:1">↻ Re-generate</button>
      <button id="editCancel" class="ghost">Cancel</button>
    </div>
    <div class="hint" id="editHint"></div>
  </div>
</div>

<script>
const $ = s => document.querySelector(s);
const fmt = t => {
  t = Math.max(0, t||0);
  const h = Math.floor(t/3600), m = Math.floor(t%3600/60),
        s = Math.floor(t%60), ms = Math.round((t - Math.floor(t))*1000);
  return `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:`+
         `${String(s).padStart(2,'0')}.${String(ms).padStart(3,'0')}`;
};
const parseTime = v => {
  v = String(v||'').trim();
  if(!v) return null;
  if(/^\d+(\.\d+)?$/.test(v)) return +v;
  const parts=v.split(':').map(Number);
  if(parts.some(n=>Number.isNaN(n))) return NaN;
  if(parts.length===2) return parts[0]*60+parts[1];
  if(parts.length===3) return parts[0]*3600+parts[1]*60+parts[2];
  return NaN;
};
const api = async (url, opts) => {
  const r = await fetch(url, opts);
  const j = await r.json();
  if(!r.ok || j.error) throw new Error(j.error || ('HTTP '+r.status));
  return j;
};

const OP_STEPS = {
  tts: ['Queued voice generation','Sending text to TTS model','Waiting for generated audio','Probing audio duration','Adding voice clip'],
  transcribe: ['Queued transcription','Extracting clip audio','Sending audio to STT model','Receiving captions','Adding captions'],
  render: ['Queued render','Preparing timeline','Applying cuts and speed regions','Mixing audio and captions','Writing final video'],
  upload: ['Queued upload','Uploading media','Probing media','Adding to project'],
  clip: ['Queued clip add','Probing video','Adding to timeline','Loading timeline analysis'],
  music: ['Queued music add','Probing audio','Adding music track'],
  musicgen: ['Queued music generation','Sending prompt to MusicGen','Generating audio','Probing generated music','Ready to add'],
  media: ['Scanning media directory','Refreshing lists'],
  settings: ['Sending settings','Reloading configuration','Refreshing media'],
  models: ['Requesting model list','Updating model choices'],
};
let currentOp=null;
function setProgress(percent, indeterminate=false){
  const wrap=$('#opProgress'), bar=$('#opProgressBar');
  if(!wrap||!bar) return;
  wrap.classList.toggle('on', percent!=null || indeterminate);
  wrap.classList.toggle('indeterminate', !!indeterminate);
  if(percent!=null) bar.style.width=Math.max(0,Math.min(100,percent))+'%';
}
function startOp(kind, label, button){
  finishOp(false);
  const steps=OP_STEPS[kind]||[label||'Working'];
  const op={kind,label:label||steps[0],steps,idx:0,started:performance.now(),button:button||null,timer:null};
  if(op.button) op.button.disabled=true;
  currentOp=op;
  setStatus(steps[0]+'…','busy');
  setProgress(8,false);
  op.timer=setInterval(()=>{
    if(currentOp!==op) return;
    const elapsed=(performance.now()-op.started)/1000;
    const targetIdx=Math.min(steps.length-1, Math.floor(elapsed/2));
    if(targetIdx!==op.idx) op.idx=targetIdx;
    const soft=Math.min(88, 8 + elapsed*7);
    const stepped=8 + (op.idx/(Math.max(1,steps.length-1)))*72;
    setStatus(steps[op.idx]+'…','busy');
    setProgress(Math.max(soft, stepped), false);
  }, 350);
  return op;
}
function advanceOp(op, msg, percent){
  if(!op||currentOp!==op) return;
  if(msg) setStatus(msg+'…','busy');
  if(percent!=null) setProgress(percent,false);
}
function finishOp(success=true, msg='', cls='ok'){
  const op=currentOp;
  if(op){ clearInterval(op.timer); if(op.button) op.button.disabled=false; }
  currentOp=null;
  if(success){ setProgress(100,false); setTimeout(()=>{ if(!currentOp) setProgress(null,false); }, 700); }
  else setProgress(null,false);
  if(msg) setStatus(msg,cls);
}

let CFG=null, MEDIA={videos:[],audios:[]}, INFO={duration:0,fps:30};
let ppsExp=120;                 // horizontal zoom (log) -> pixels per second
let zoomY=35;                   // vertical timeline zoom, 0..100
let playRate=1.0;               // preview playback rate
let tts=[], speeds=[], cuts=[], spStart=null, spEnd=null;
let ttsStart=null, ttsEnd=null;              // pending voice-over selection
let caps=[], capStart=null, capEnd=null;     // captions + pending selection
let musics=[], musStartSel=null, musEndSel=null;   // music clips + pending sel
let musicUpload=null;                              // {ref,label} of an upload
let rangeSel={start:null,end:null,next:'start'};    // right-click shared range
let cutStartSel=null, cutEndSel=null;               // pending cut selection
let sessionEnabled=false, restoringSession=false, sessionDirty=false;
let drag=null, dragMoved=false, downT=null;  // timeline pointer interaction
let selectedTimelineItem=null, timelineClipboard=null; // copy/paste for audio clips
let clips=[];                   // ordered playlist: {name,duration,fps,offset}
let curIdx=-1;                  // index of clip currently loaded in <video>
let autoFollowPlayhead=true;
let hoverClip=-1, hoverSeekTimer=null, hoverIdleTimer=null;
let hoverPointerInside=false, lastHoverEvent=null, hoverLastMove=0;
const player=$('#player'), tl=$('#tl'), scroll=$('#tlScroll');
const previewBlack=$('#previewBlack'), previewCaption=$('#previewCaption'), previewTime=$('#previewTime');
const hoverPreview=$('#hoverPreview'), hoverVideo=$('#hoverVideo'), hoverTime=$('#hoverTime');
const previewStage=$('#previewStage'), fsTimeline=$('#fsTimeline'), tlMini=$('#tlMini');

// ---- preview player: click = play/pause, double-click = fullscreen ----
let _clickTimer=null, isFs=false;
function toggleFullscreen(){
  if(document.fullscreenElement) document.exitFullscreen && document.exitFullscreen();
  else if(previewStage.requestFullscreen) previewStage.requestFullscreen();
}
player.addEventListener('click', ()=>{
  if(playOriginalMode()) return;            // native controls handle it then
  if(_clickTimer) return;                   // a double-click may still arrive
  _clickTimer=setTimeout(()=>{ _clickTimer=null; playPause(); }, 220);
});
player.addEventListener('dblclick', e=>{
  e.preventDefault();
  if(_clickTimer){ clearTimeout(_clickTimer); _clickTimer=null; }
  toggleFullscreen();
});
document.addEventListener('fullscreenchange', ()=>{
  isFs=(document.fullscreenElement===previewStage);
  if(!isFs) fsTimeline.style.display='none';
});
// In fullscreen, reveal a miniature timeline only while hovering the bottom band.
previewStage.addEventListener('mousemove', e=>{
  if(!isFs) return;
  const r=previewStage.getBoundingClientRect();
  fsTimeline.style.display=(e.clientY > r.bottom-110) ? 'block' : 'none';
});
previewStage.addEventListener('mouseleave', ()=>{ fsTimeline.style.display='none'; });
tlMini.addEventListener('click', e=>{
  const r=tlMini.getBoundingClientRect();
  seekGlobal(outToSrc((e.clientX-r.left)/r.width*(outDur()||0)), false);
});
function drawMini(){
  const c=tlMini, W=c.clientWidth||1, H=c.clientHeight||1;
  if(c.width!==W) c.width=W;
  if(c.height!==H) c.height=H;
  const ctx=c.getContext('2d'); ctx.clearRect(0,0,W,H);
  ctx.fillStyle='#0a0d14'; ctx.fillRect(0,0,W,H);
  const dur=outDur()||1, X=t=>Math.max(0,Math.min(W,srcToOut(t)/dur*W));
  const vy=2, vh=H*0.46;
  ctx.fillStyle='#243a5e';
  for(const cl of clips){ const x=X(cl.offset); ctx.fillRect(x,vy,Math.max(1,X(cl.offset+cl.duration)-x),vh); }
  const strip=(items,gs,ge,y,col)=>{ ctx.fillStyle=col;
    for(const it of items){ const x=X(gs(it)); ctx.fillRect(x,y,Math.max(1,X(ge(it))-x),Math.max(3,H*0.13)); } };
  const sy=vy+vh+2;
  strip(tts,    c=>c.time,  c=>ttsEndOf(c), sy,            '#37d9a0');
  strip(caps,   c=>c.start, c=>c.end,       sy+H*0.17,     '#b48bff');
  strip(musics, m=>m.start, m=>musEndOf(m), sy+H*0.34,     '#ffb454');
  const px=X(headT); ctx.strokeStyle='#fff'; ctx.lineWidth=2;
  ctx.beginPath(); ctx.moveTo(px,0); ctx.lineTo(px,H); ctx.stroke();
}

// ---- floating preview: dock to the top of the right column when scrolled away ----
const previewHome=$('#previewHome'), previewDock=$('#previewDock');
const _leftCol=previewStage.closest('.col');
let _docked=false;
function setPreviewDocked(on){
  if(on===_docked) return;
  _docked=on;
  if(on){
    previewHome.style.minHeight=previewStage.offsetHeight+'px';  // keep left layout stable
    previewStage.classList.add('docked');
    previewDock.appendChild(previewStage);
  } else {
    previewStage.classList.remove('docked');
    previewHome.style.minHeight='';
    previewHome.appendChild(previewStage);
  }
  updatePreviewOverlays();   // re-fit the subtitle to the new video size
}
if('IntersectionObserver' in window){
  // Dock to the side player once the main player is less than 40% visible;
  // hand back to the main player when it's at least 40% visible again.
  new IntersectionObserver(es=>{
    for(const en of es){ if(!isFs) setPreviewDocked(en.intersectionRatio < 0.4); }
  }, {root:_leftCol, threshold:[0,0.4,1]}).observe(previewHome);
}

// ---- free-layout timeline (clips have independent offsets; gaps + overlaps) ----
let headT=0, playing=false, lastTs=0;     // master playhead clock
function rawDuration(){
  let end=0; for(const c of clips){ end=Math.max(end, c.offset+c.duration); }
  return end;
}
// Map a timeline instant through one ripple cut [s,e]: points before the cut
// stay put, points after slide left by its length, points inside collapse onto
// the cut's start (the content they sat on is gone).
function rippleTime(t,s,e,len){
  if(t<=s+1e-9) return t;
  if(t>=e-1e-9) return t-len;
  return s;
}
// Remap an interval; returns null when it lived entirely inside the cut.
function rippleRange(a,b,s,e,len){
  const na=rippleTime(a,s,e,len), nb=rippleTime(b,s,e,len);
  return (nb-na<1e-3)?null:{a:na,b:nb};
}
function applyCutsToClips(){
  if(!cuts.length) return;
  const nc=JSON.parse(JSON.stringify(cuts));
  nc.sort((a,b)=>a.start-b.start);
  const merged=[];
  for(const c of nc){
    if(merged.length&&c.start<=merged[merged.length-1].end+1e-3) merged[merged.length-1].end=Math.max(merged[merged.length-1].end,c.end);
    else merged.push(c);
  }
  // Ripple-delete each cut: drop the [s,e] window and slide everything after
  // it left by its length, so the removed region truly stops existing. Source
  // time (srcStart) is carried on each surviving video piece so the preview and
  // the export read the right frames after the gap is closed, and every other
  // track (voice-over, music, captions, speed regions) shifts with it to stay
  // aligned to the footage. Apply cuts from last to first so earlier cuts keep
  // their (still-valid) timeline coordinates.
  let cur=clips;
  for(let k=merged.length-1;k>=0;k--){
    const s=merged[k].start, e=merged[k].end, len=e-s;
    if(len<=1e-3) continue;
    // video clips: split around the cut, carrying source time on each piece
    const next=[];
    for(const c of cur){
      const ss=c.srcStart||0, ce=c.offset+c.duration;
      const leftEnd=Math.min(ce,s);            // surviving portion before the cut
      if(leftEnd>c.offset+1e-3)
        next.push({...c,offset:c.offset,duration:leftEnd-c.offset,srcStart:ss,waveform:null,motion:null});
      const rightStart=Math.max(c.offset,e);   // surviving portion after the cut
      if(ce>rightStart+1e-3)
        next.push({...c,offset:rightStart-len,duration:ce-rightStart,
                   srcStart:ss+(rightStart-c.offset),waveform:null,motion:null});
    }
    cur=next;
    // voice-over: shift its placement; drop a clip swallowed by the cut
    tts=tts.filter(t=>{
      if(t.end!=null){ const r=rippleRange(t.time,t.end,s,e,len); if(!r) return false; t.time=r.a; t.end=r.b; }
      else t.time=rippleTime(t.time,s,e,len);
      return true;
    });
    // speed regions, captions, music: clip/shift, drop if fully inside the cut
    speeds=speeds.map(x=>{const r=rippleRange(x.start,x.end,s,e,len);return r?{...x,start:r.a,end:r.b}:null;}).filter(Boolean);
    caps=caps.map(x=>{const r=rippleRange(x.start,x.end,s,e,len);return r?{...x,start:r.a,end:r.b}:null;}).filter(Boolean);
    musics=musics.map(m=>{
      if(m.end!=null){ const r=rippleRange(m.start,m.end,s,e,len); return r?{...m,start:r.a,end:r.b}:null; }
      return {...m,start:rippleTime(m.start,s,e,len)};
    }).filter(Boolean);
  }
  cur.sort((a,b)=>a.offset-b.offset);
  clips=cur;
  cuts=[];
}
function recompute(){ let end=0; for(const c of clips){ end=Math.max(end,c.offset+c.duration); } INFO.duration=end; }
function activeClipAt(t){
  let best=-1, bestOff=-1;
  for(let i=0;i<clips.length;i++){
    const c=clips[i];
    if(t>=c.offset-1e-4 && t<c.offset+c.duration-1e-4 && c.offset>=bestOff){
      best=i; bestOff=c.offset;
    }
  }
  return best;
}
function curClipFps(){ return (clips[curIdx]?.fps) || INFO.fps || 30; }
function globalTime(){ return headT; }
function playOriginalMode(){ return $('#playOriginal')?.checked; }
function followPlayheadMode(){ return $('#followPlayhead')?.checked; }
function lockTracksMode(){ return $('#lockTracks')?.checked; }
function updatePlayerControls(){
  player.controls=playOriginalMode();
  player.title=playOriginalMode()?'Original clip time':'Timeline preview time is shown in the toolbar';
}
function activeCaptionAt(t){
  for(let i=caps.length-1;i>=0;i--) if(t>=caps[i].start&&t<=caps[i].end) return caps[i].text;
  return '';
}
function updatePreviewOverlays(){
  updatePlayerControls();
  const gap=!playOriginalMode()&&activeClipAt(headT)<0;
  previewBlack.style.display=gap?'block':'none';
  previewCaption.textContent=playOriginalMode()?'':activeCaptionAt(headT);
  previewTime.textContent=fmt(playOriginalMode()?player.currentTime:headT);
  // Caption size is handled in CSS: the docked preview scales it down via `zoom`.
}
function loadClip(i, local, play){
  curIdx=i;
  const c=clips[i], ss=c.srcStart||0;
  player.src=mediaUrl(c.name);
  player.onloadeddata=()=>{ player.onloadeddata=null;
    player.playbackRate=playRate;
    player.currentTime=Math.max(0,ss+Math.min(Math.max(0,local), c.duration-0.04));
    if(play&&playing) player.play(); };
}
function seekGlobal(t, play){
  if(!clips.length){ headT=0; playing=false; updatePreviewOverlays(); return; }
  if(playOriginalMode()){
    const i=curIdx>=0?curIdx:0;
    curIdx=i; playing=!!play;
    if(!player.src) player.src=mediaUrl(clips[i].name);
    player.playbackRate=playRate;
    player.currentTime=Math.max(0,Math.min(t,clips[i].duration-0.04));
    if(play) player.play(); else player.pause();
    headT=clips[i].offset+player.currentTime;
    updatePreviewOverlays(); draw(); return;
  }
  headT=Math.max(0,Math.min(INFO.duration||0, t));
  autoFollowPlayhead=true;
  keepTimeVisible(headT);
  playing=!!play;
  const i=activeClipAt(headT);
  if(i>=0){
    const local=headT-clips[i].offset, ss=clips[i].srcStart||0;
    if(i!==curIdx) loadClip(i, local, play);
    else { player.playbackRate=playRate; player.currentTime=Math.max(0,ss+local); if(play) player.play(); else player.pause(); }
  } else { curIdx=-1; player.pause(); }   // in a gap → black / held
  updatePreviewOverlays();
}
function playPause(){
  if(playing){ playing=false; player.pause(); return; }
  playing=true;
  if(playOriginalMode()){
    if(curIdx<0&&clips.length) loadClip(0,0,true); else {player.playbackRate=playRate; player.play();}
    return;
  }
  if(headT>=INFO.duration-1e-3) headT=0;
  const i=activeClipAt(headT);
  if(i>=0){ if(i!==curIdx) loadClip(i, headT-clips[i].offset, true); else {player.playbackRate=playRate; player.play();} }
}

const RATE_STEPS=[0.125,0.25,0.5,0.75,1,1.25,1.5,2,3,4,8,16];
function setPlayRate(rate){
  playRate=RATE_STEPS.reduce((best,r)=>Math.abs(r-rate)<Math.abs(best-rate)?r:best,1);
  player.playbackRate=playRate;
  $('#rateV').textContent=(playRate>=1?playRate.toFixed(playRate%1?2:0):playRate.toFixed(3).replace(/0+$/,''))+'×';
  scheduleSessionSave();
}
function nudgeRate(dir){
  let i=RATE_STEPS.findIndex(r=>Math.abs(r-playRate)<1e-6);
  if(i<0) i=RATE_STEPS.findIndex(r=>r>playRate)-1;
  i=Math.max(0,Math.min(RATE_STEPS.length-1,i+dir));
  setPlayRate(RATE_STEPS[i]);
}

// ---- speed-aware time mapping: the pixel axis is OUTPUT (post-speed) time, so
// sped-up regions render shrunk and everything after them shifts earlier. ----
function normSpeeds(){
  const arr=speeds.filter(s=>s.end>s.start&&(+s.factor)>0)
    .map(s=>({start:+s.start,end:+s.end,factor:+s.factor})).sort((a,b)=>a.start-b.start);
  const out=[]; let last=0;
  for(const s of arr){ const st=Math.max(s.start,last), en=Math.max(st,s.end);
    if(en>st){ out.push({start:st,end:en,factor:s.factor}); last=en; } }
  return out;
}
function srcToOut(t){               // source seconds -> output seconds
  let o=t;
  for(const s of normSpeeds()){
    if(t<=s.start) break;
    o -= (Math.min(t,s.end)-s.start)*(1-1/s.factor);
  }
  return o;
}
function outToSrc(o){               // output seconds -> source seconds
  let prevSrc=0, prevOut=0;
  for(const s of normSpeeds()){
    const seg=s.start-prevSrc;
    if(o<=prevOut+seg) return prevSrc+(o-prevOut);
    prevOut+=seg; prevSrc=s.start;
    const reg=(s.end-s.start)/s.factor;
    if(o<=prevOut+reg) return s.start+(o-prevOut)*s.factor;
    prevOut+=reg; prevSrc=s.end;
  }
  return prevSrc+(o-prevOut);
}
const outDur = () => srcToOut(INFO.duration||0);

const fitPps = () => Math.max(0.05, (scroll.clientWidth-8)/Math.max(1,outDur()||1));
const zoomPps = () => Math.pow(10, 0.5 + (ppsExp/1000)*3.0);
const pps = () => Math.max(fitPps(), zoomPps());
function updateSubtitle(){
  $('#subtitle').textContent =
    `voice: ${CFG.gender} (${CFG.defaultVoice}) · TTS: ${CFG.ttsModel||'none'} · music: ${CFG.audioModel||'none'} · ${CFG.mediaDir}`;
}

// ---------- init ----------
(async function init(){
  CFG = await api('api/config');
  updateSubtitle();
  // voices
  const vs=$('#voiceSel');
  for(const g of ['feminine','masculine']){
    const og=document.createElement('optgroup'); og.label=g;
    for(const [id,label] of CFG.voices[g]){
      const o=document.createElement('option'); o.value=id;
      o.textContent=`${label} — ${id}`;
      if(id===CFG.defaultVoice) o.selected=true;
      og.appendChild(o);
    }
    vs.appendChild(og);
  }
  // voice-over speech speed
  for(const s of [0.5,0.75,0.9,1.0,1.1,1.25,1.5,2.0]){
    const o=document.createElement('option'); o.value=s;
    o.textContent=(s===1.0?'1.0× (normal)':s+'×'); if(s===1.0) o.selected=true;
    $('#ttsSpeed').appendChild(o);
  }
  // voice-over emotion — only shown when the TTS model advertises emotions
  const emos=CFG.ttsEmotions||[];
  if(emos.length){
    for(const e of emos){ const o=document.createElement('option'); o.value=e; o.textContent=e; $('#ttsEmotion').appendChild(o); }
    $('#ttsEmotionRow').hidden=false;
  }
  // voice-over delivery style (whisper/shout/tone/…) — gated on model support
  const styles=CFG.ttsStyles||[];
  if(styles.length){
    for(const s of styles){ const o=document.createElement('option'); o.value=s; o.textContent=s; $('#ttsStyle').appendChild(o); }
    $('#ttsStyleRow').hidden=false;
  }
  // speed factors
  for(const f of CFG.speedFactors){
    const o=document.createElement('option'); o.value=f; o.textContent=f+'×';
    $('#spFactor').appendChild(o);
  }
  // Optional auto-caption only when the server exposes a transcription model.
  const tb=$('#capTranscribe');
  tb.disabled=!CFG.sttModel;
  tb.title=CFG.sttModel?('Transcribe with '+CFG.sttModel)
    :'Disabled: start with --stt-model / configure a Whisper model to enable';
  const mgb=$('#musGenerate');
  mgb.disabled=!CFG.audioModel;
  mgb.title=CFG.audioModel?('Generate with '+CFG.audioModel)
    :'Disabled: configure a MusicGen/audio generation model to enable';
  fillSettings(CFG.settings);
  bindTimeInputs();
  sessionEnabled=!!(CFG.session&&CFG.session.enabled);
  // auto-detect models from server if none configured
  (async()=>{
    try{
      const r=await api('api/models');
      const d=r.detected||{};
      let changed=false;
      if(d.tts_model&&!$('#cfgTtsModel').value.trim()){ $('#cfgTtsModel').value=d.tts_model; changed=true; }
      if(d.stt_model&&!$('#cfgSttModel').value.trim()){ $('#cfgSttModel').value=d.stt_model; changed=true; }
      if(d.audio_model&&!$('#cfgAudioModel').value.trim()){ $('#cfgAudioModel').value=d.audio_model; changed=true; }
      if(changed){
        await api('api/settings',{method:'POST',headers:{'Content-Type':'application/json'},
          body:JSON.stringify({settings:gatherSettings(), save:false})});
        const st=await api('api/config');
        CFG.ttsModel=st.ttsModel; CFG.sttModel=st.sttModel; CFG.audioModel=st.audioModel;
        const tb=$('#capTranscribe'); tb.disabled=!CFG.sttModel;
        tb.title=CFG.sttModel?('Transcribe with '+CFG.sttModel):'Disabled: configure a Whisper/STT model to enable';
        const mgb=$('#musGenerate'); mgb.disabled=!CFG.audioModel;
        mgb.title=CFG.audioModel?('Generate with '+CFG.audioModel):'Disabled: configure a MusicGen/audio generation model to enable';
        updateSubtitle();
      }
    }catch(_){}
  })();
  await loadMedia();
  if(sessionEnabled && await restoreSession()){
    setStatus('Session restored: '+(CFG.session.name||'default'),'ok');
  } else if(CFG.defaultVideo && MEDIA.videos.includes(CFG.defaultVideo)){
    $('#videoSel').value=CFG.defaultVideo; await addClip(CFG.defaultVideo);
  }
})();

async function loadMedia(){
  const op=currentOp?null:startOp('media','Refreshing media');
  try{
  MEDIA = await api('api/videos');
  const vsel=$('#videoSel'); vsel.innerHTML='';
  for(const v of MEDIA.videos){
    const o=document.createElement('option'); o.value=v; o.textContent=v; vsel.appendChild(o);
  }
  const msel=$('#musicSel'); msel.innerHTML='<option value="">— none —</option>';
  for(const a of [...MEDIA.audios, ...MEDIA.videos]){
    const o=document.createElement('option'); o.value=a; o.textContent=a; msel.appendChild(o);
  }
  if(op) finishOp(true, 'Media list refreshed', 'ok');
  }catch(e){ if(op) finishOp(false, 'Media refresh failed: '+e.message, 'err'); throw e; }
}

// ---------- session persistence ----------
function gatherState(){
  return {
    clips:clips.map(c=>({name:c.name,label:c.label,duration:c.duration,fps:c.fps,
      offset:c.offset,srcStart:c.srcStart||0,srcDur:c.srcDur||c.duration,hasAudio:!!c.hasAudio})),
    tts:tts.map(c=>({time:c.time,end:(c.end!=null?c.end:null),file:c.file,url:c.url,
      text:c.text,voice:c.voice,dur:c.dur})),
    speeds:speeds.map(s=>({start:s.start,end:s.end,factor:s.factor})),
    cuts:cuts.map(c=>({start:c.start,end:c.end})),
    caps:caps.map(c=>({start:c.start,end:c.end,text:c.text})),
    musics:musics.map(m=>({ref:m.ref,label:m.label,start:m.start,
      end:(m.end!=null?m.end:null),duration:m.duration})),
    rangeSel:{...rangeSel},
    pending:{spStart,spEnd,ttsStart,ttsEnd,capStart,capEnd,musStartSel,musEndSel,
      cutStartSel,cutEndSel},
    view:{headT,ppsExp,zoomY,playRate,scrollLeft:scroll.scrollLeft,
      baseVol:+$('#baseVol').value,ttsVol:+$('#ttsVol').value,musVol:+$('#musVol').value,
      burnCaptions:$('#capBurn').checked,ttsAutoCap:$('#ttsAutoCap').checked,
      playOriginal:$('#playOriginal').checked,followPlayhead:$('#followPlayhead').checked,
      lockTracks:$('#lockTracks').checked,ttsSpeed:$('#ttsSpeed').value,
      rightw:parseInt(getComputedStyle(document.querySelector('main')).getPropertyValue('--rightw'))||340},
  };
}
function scheduleSessionSave(){
  if(!sessionEnabled||restoringSession) return;
  sessionDirty=true;
}

// ---------- undo / redo (timeline edits) ----------
let undoStack=[], redoStack=[];
const UNDO_MAX=80;
function snapshotState(){
  // Only the data fields — never the transient _audio / waveform / motion.
  return {
    clips:clips.map(c=>({name:c.name,label:c.label,duration:c.duration,fps:c.fps,
      offset:c.offset,srcStart:c.srcStart||0,srcDur:c.srcDur||c.duration,hasAudio:!!c.hasAudio})),
    tts:tts.map(c=>({time:c.time,end:(c.end!=null?c.end:null),file:c.file,url:c.url,
      text:c.text,voice:c.voice,dur:c.dur,speed:c.speed,emotion:c.emotion,style:c.style})),
    speeds:speeds.map(s=>({...s})),
    cuts:cuts.map(c=>({...c})),
    caps:caps.map(c=>({...c})),
    musics:musics.map(m=>({ref:m.ref,label:m.label,start:m.start,
      end:(m.end!=null?m.end:null),duration:m.duration})),
    headT,
  };
}
function pushUndo(){
  undoStack.push(snapshotState());
  if(undoStack.length>UNDO_MAX) undoStack.shift();
  redoStack=[];
}
function restoreState(s){
  clips=s.clips.map(c=>({...c,waveform:null,motion:null}));
  tts=s.tts.map(c=>({...c}));
  speeds=s.speeds.map(x=>({...x}));
  cuts=s.cuts.map(x=>({...x}));
  caps=s.caps.map(x=>({...x}));
  musics=s.musics.map(m=>({...m}));
  headT=s.headT||0;
  selectedTimelineItem=null;
  recompute();
  renderClips(); renderTts(); renderSpeeds(); renderCuts(); renderCaps(); renderMusic();
  updateSpLabels(); updateTtsSel(); updateCapLabels(); updateMusSel(); updateCutLabels();
  layout(); for(let i=0;i<clips.length;i++){ loadWaveform(i); loadMotion(i); }
  if(clips.length) seekGlobal(Math.min(headT,INFO.duration),false);
  draw(); scheduleSessionSave();
}
function undo(){
  if(!undoStack.length){ setStatus('Nothing to undo','err'); return; }
  redoStack.push(snapshotState());
  restoreState(undoStack.pop());
  setStatus('Undo','ok');
}
function redo(){
  if(!redoStack.length){ setStatus('Nothing to redo','err'); return; }
  undoStack.push(snapshotState());
  restoreState(redoStack.pop());
  setStatus('Redo','ok');
}
setInterval(async()=>{
  if(!sessionDirty||!sessionEnabled) return;
  sessionDirty=false;
  try{
    await api('api/session',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({state:gatherState()})});
  }catch(e){ sessionDirty=true; setStatus('Session save failed: '+e.message,'err'); }
}, 900);
setInterval(()=>{ if(sessionEnabled&&!restoringSession) sessionDirty=true; }, 3000);
async function restoreSession(){
  let r;
  try{ r=await api('api/session'); }catch(e){ setStatus('Session restore failed: '+e.message,'err'); return false; }
  const s=r.state; if(!s) return false;
  restoringSession=true;
  try{
    clips=(s.clips||[]).map(c=>({name:c.name,label:c.label||c.name,duration:+c.duration||0,
      fps:+c.fps||30,offset:+c.offset||0,srcStart:+c.srcStart||0,srcDur:+c.srcDur||(+c.duration||0),
      hasAudio:!!c.hasAudio,waveform:null,motion:null}));
    tts=(s.tts||[]).map(c=>({...c,url:c.url||('tts?name='+encodeURIComponent(c.file||''))}));
    speeds=s.speeds||[]; cuts=s.cuts||[]; caps=s.caps||[]; musics=s.musics||[];
    rangeSel=s.rangeSel||{start:null,end:null,next:'start'};
    const p=s.pending||{};
    spStart=p.spStart??null; spEnd=p.spEnd??null; ttsStart=p.ttsStart??null; ttsEnd=p.ttsEnd??null;
    capStart=p.capStart??null; capEnd=p.capEnd??null; musStartSel=p.musStartSel??null; musEndSel=p.musEndSel??null;
    cutStartSel=p.cutStartSel??null; cutEndSel=p.cutEndSel??null;
    const v=s.view||{}; ppsExp=v.ppsExp??ppsExp; zoomY=v.zoomY??zoomY; playRate=v.playRate??playRate; headT=v.headT??0;
    $('#zoom').value=ppsExp; $('#zoomY').value=zoomY;
    if(v.baseVol!=null) $('#baseVol').value=v.baseVol;
    if(v.ttsVol!=null) $('#ttsVol').value=v.ttsVol;
    if(v.musVol!=null) $('#musVol').value=v.musVol;
    if(v.burnCaptions!=null) $('#capBurn').checked=!!v.burnCaptions;
    if(v.ttsAutoCap!=null) $('#ttsAutoCap').checked=!!v.ttsAutoCap;
    if(v.playOriginal!=null) $('#playOriginal').checked=!!v.playOriginal;
    if(v.followPlayhead!=null) $('#followPlayhead').checked=!!v.followPlayhead;
    if(v.lockTracks!=null) $('#lockTracks').checked=!!v.lockTracks;
    if(v.ttsSpeed!=null&&$('#ttsSpeed')) $('#ttsSpeed').value=v.ttsSpeed;
    if(v.rightw) setSidebarWidth(v.rightw);
    recompute(); renderClips(); renderTts(); renderSpeeds(); renderCuts(); renderCaps(); renderMusic();
    updateSpLabels(); updateTtsSel(); updateCapLabels(); updateMusSel(); updateCutLabels();
    layout(); scroll.scrollLeft=v.scrollLeft||0; for(let i=0;i<clips.length;i++){ loadWaveform(i); loadMotion(i); }
    if(clips.length) seekGlobal(Math.min(headT,INFO.duration),false);
    setPlayRate(playRate); bindVolumeLabels(); draw();
    return true;
  } finally { restoringSession=false; }
}

// ---------- settings / configuration ----------
function modelId(m){ return typeof m==='string'?m:(m&&(m.id||m.name||m.model))||''; }
function fillModelList(models){
  const dl=$('#modelList'); dl.innerHTML='';
  for(const m of (models||[])){
    const id=modelId(m); if(!id) continue;
    const o=document.createElement('option'); o.value=id;
    const meta=typeof m==='object'?[m.type,m.task,m.owned_by].filter(Boolean).join(' · '):'';
    if(meta) o.label=id+' — '+meta;
    dl.appendChild(o);
  }
}
function fillSettings(s){
  const st=(s&&s.settings)||{};
  $('#cfgBaseUrl').value=st.base_url||'';
  $('#cfgApiKey').value=st.api_key||'';
  $('#cfgMediaDir').value=st.media_dir||'';
  $('#cfgOutputDir').value=st.output_dir||'';
  $('#cfgTtsModel').value=st.tts_model||'';
  $('#cfgSttModel').value=st.stt_model||'';
  $('#cfgAudioModel').value=st.audio_model||'';
  $('#cfgGender').value=st.voice||'feminine';
  $('#cfgSavePath').value=(s&&s.configPath)||'video_editor.config.json';
  fillModelList(s&&s.models);
}
function gatherSettings(){
  return {
    base_url:$('#cfgBaseUrl').value.trim(),
    api_key:$('#cfgApiKey').value,
    media_dir:$('#cfgMediaDir').value.trim(),
    output_dir:$('#cfgOutputDir').value.trim(),
    tts_model:$('#cfgTtsModel').value.trim(),
    stt_model:$('#cfgSttModel').value.trim(),
    audio_model:$('#cfgAudioModel').value.trim(),
    voice:$('#cfgGender').value,
    voice_name:$('#voiceSel').value||'',
  };
}
function setCfgStatus(m,c){const s=$('#cfgStatus');s.textContent=m;
  s.className='status'+(c?' '+c:'');}
async function applySettings(save){
  const path=$('#cfgSavePath').value.trim();
  const op=startOp('settings', save?'Saving settings':'Applying settings', save?$('#cfgSave'):$('#cfgApply'));
  setCfgStatus(save?'Saving…':'Applying…');
  try{
    const r=await api('api/settings',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({settings:gatherSettings(), save:!!save, path})});
    const st=r.settings||{};
    CFG.ttsModel=st.tts_model; CFG.sttModel=st.stt_model;
    CFG.audioModel=st.audio_model;
    CFG.gender=st.voice; CFG.defaultVoice=st.voice_name; CFG.mediaDir=st.media_dir;
    fillModelList(r.models);
    const tb=$('#capTranscribe');
    tb.disabled=!CFG.sttModel;
    tb.title=CFG.sttModel?('Transcribe with '+CFG.sttModel)
      :'Disabled: configure a Whisper/STT model to enable';
    const mgb=$('#musGenerate');
    mgb.disabled=!CFG.audioModel;
    mgb.title=CFG.audioModel?('Generate with '+CFG.audioModel)
      :'Disabled: configure a MusicGen/audio generation model to enable';
    updateSubtitle();
    advanceOp(op, 'Refreshing media after settings change', 78);
    await loadMedia();
    setCfgStatus(save?('Saved to '+r.saved):'Applied to running session.','ok');
    finishOp(true, save?'Settings saved':'Settings applied', 'ok');
  }catch(e){ setCfgStatus((save?'Save':'Apply')+' failed: '+e.message,'err'); finishOp(false, (save?'Save':'Apply')+' failed: '+e.message, 'err'); }
}
$('#cfgApply').onclick=()=>applySettings(false);
$('#cfgSave').onclick=()=>applySettings(true);
$('#cfgRefreshModels').onclick=async()=>{
  const op=startOp('models','Loading models',$('#cfgRefreshModels'));
  setCfgStatus('Loading models…');
  try{
    const r=await api('api/models'); fillModelList(r.models);
    const d=r.detected||{};
    if(d.tts_model&&!$('#cfgTtsModel').value.trim()){
      $('#cfgTtsModel').value=d.tts_model;
      setCfgStatus((r.models||[]).length+' model(s) available — TTS auto-set to '+d.tts_model+'. Apply to save.','ok');
    }
    if(d.stt_model&&!$('#cfgSttModel').value.trim()){
      $('#cfgSttModel').value=d.stt_model;
      setCfgStatus((r.models||[]).length+' model(s) available — STT auto-set to '+d.stt_model+'. Apply to save.','ok');
    }
    if(d.audio_model&&!$('#cfgAudioModel').value.trim()){
      $('#cfgAudioModel').value=d.audio_model;
      setCfgStatus((r.models||[]).length+' model(s) available — music model auto-set to '+d.audio_model+'. Apply to save.','ok');
    }
    if(!d.tts_model&&!d.stt_model&&!d.audio_model) setCfgStatus((r.models||[]).length+' model(s) available.','ok');
    finishOp(true, (r.models||[]).length+' model(s) loaded', 'ok');
  }
  catch(e){ setCfgStatus('Could not load models: '+e.message,'err'); finishOp(false, 'Could not load models: '+e.message, 'err'); }
};
$('#cfgGender').onchange=()=>{
  const g=$('#cfgGender').value;
  const def=(CFG.voices[g]&&CFG.voices[g][0])?CFG.voices[g][0][0]:'';
  const vs=$('#voiceSel');
  if(def&&[...vs.options].some(o=>o.value===def)) vs.value=def;
};

// A clip "ref" is one of: a media-dir-relative path, 'upload:<name>', or
// 'abs:<path>'. mediaUrl maps it to the right server route for <video>.
function mediaUrl(ref){
  if(ref.startsWith('upload:')) return 'upload?name='+encodeURIComponent(ref.slice(7));
  if(ref.startsWith('abs:'))    return 'media?name='+encodeURIComponent(ref);   // not previewable
  return 'media?name='+encodeURIComponent(ref);
}
async function uploadFile(file){
  setStatus('Uploading '+file.name+'…','busy'); setProgress(20,true);
  const r=await fetch('api/upload?name='+encodeURIComponent(file.name),
    {method:'POST', body:file});
  setProgress(70,false);
  const j=await r.json();
  if(!r.ok||j.error) throw new Error(j.error||('HTTP '+r.status));
  setProgress(90,false);
  return j;
}
async function addClipRef(ref, label, info){
  if(!info) info = await api('api/probe?ref='+encodeURIComponent(ref));
  pushUndo();
  // Append at the current end of the timeline; drag it anywhere afterwards.
  clips.push({name:ref, label:label||ref, duration:info.duration, fps:info.fps,
              offset:INFO.duration, srcStart:0, srcDur:info.duration,
              hasAudio:!!info.hasAudio, waveform:null});
  recompute(); renderClips();
  if(curIdx<0) seekGlobal(Math.max(0,INFO.duration-info.duration),false);
  layout();
  loadWaveform(clips.length-1);
  loadMotion(clips.length-1);
  scheduleSessionSave();
}
async function loadWaveform(i){
  const c=clips[i]; if(!c||!c.hasAudio) return;
  try{
    const r=await api('api/waveform?ref='+encodeURIComponent(c.name)+'&points=2200');
    if(clips[i]===c){ c.waveform=r.peaks||[]; draw(); }
  }catch(_){ c.waveform=[]; }
}
async function loadMotion(i){
  const c=clips[i]; if(!c) return;
  try{
    const r=await api('api/motion?ref='+encodeURIComponent(c.name)+'&points=900');
    if(clips[i]===c){ c.motion=r.values||[]; draw(); }
  }catch(_){ c.motion=[]; }
}
async function addClip(name){
  name = name || $('#videoSel').value;
  if(!name) return;
  const op=startOp('clip','Adding clip '+name,$('#addClip'));
  try{
    await addClipRef(name, name);
    finishOp(true, 'Added '+name, 'ok');
  }catch(e){ finishOp(false, 'Add clip failed: '+e.message, 'err'); }
}
function nudgeClip(i,d){
  pushUndo();
  clips[i].offset=Math.max(0, clips[i].offset+d);
  recompute(); renderClips(); layout(); draw();
  scheduleSessionSave();
}
function removeClip(i){
  pushUndo();
  clips.splice(i,1); recompute(); renderClips(); curIdx=-1;
  if(clips.length) seekGlobal(Math.min(headT,INFO.duration),false);
  else { player.pause(); player.removeAttribute('src'); player.load(); headT=0; }
  layout();
  scheduleSessionSave();
}
function renderClips(){
  const el=$('#clipList'); el.innerHTML='';
  clips.forEach((c,i)=>{
    const d=document.createElement('div'); d.className='item';
    d.innerHTML=`<span class="pill" title="start">${fmt(c.offset).replace(/^00:/,'')}</span>
      <span class="grow" title="${c.label}">${c.label}</span>
      <span class="badge">${fmt(c.duration).replace(/^00:/,'')}</span>
      <button class="ghost sm" data-l title="0.5s earlier">−</button>
      <button class="ghost sm" data-r title="0.5s later">＋</button><span class="x">✕</span>`;
    d.querySelector('[data-l]').onclick=()=>nudgeClip(i,-0.5);
    d.querySelector('[data-r]').onclick=()=>nudgeClip(i,0.5);
    d.querySelector('.x').onclick=()=>removeClip(i);
    el.appendChild(d);
  });
  scheduleSessionSave();
}

// ---------- timeline ----------
function layout(keepT=null){
  const centerT = keepT!=null ? keepT : globalTime();
  const total = Math.max(1, outDur()) * pps();
  $('#tlSpacer').style.width = total+'px';
  tl.height = timelineHeight();
  tl.width = scroll.clientWidth;
  if(keepT!=null) centerTimeline(centerT);
  draw();
  scheduleSessionSave();
}
function centerTimeline(t){
  scroll.scrollLeft=Math.max(0,srcToOut(t)*pps()-scroll.clientWidth/2);
}
function keepTimeVisible(t, margin=80){
  const x=srcToOut(t)*pps();
  if(x<scroll.scrollLeft+margin) scroll.scrollLeft=Math.max(0,x-margin);
  else if(x>scroll.scrollLeft+scroll.clientWidth-margin) scroll.scrollLeft=x-scroll.clientWidth+margin;
}

const LANE_STYLE={wave:{col:'#65e0ff',bg:'rgba(101,224,255,.10)',label:'audio'},
                  tts:{col:'#37d9a0',bg:'rgba(55,217,160,.22)',label:'voice'},
                  mus:{col:'#7aa2ff',bg:'rgba(122,162,255,.22)',label:'music'},
                  cap:{col:'#d9a0ff',bg:'rgba(217,160,255,.22)',label:'caption'},
                  vid:{col:'#ffd479',bg:'rgba(255,212,121,.18)',label:'video'}};
const yScale = () => 0.65 + (zoomY/100)*1.85;
const yv = n => Math.round(n*yScale());
const timelineHeight = () => yv(196)+58;
function lanes(){
  return {
    wave:{...LANE_STYLE.wave,y:yv(6),h:yv(42)},
    motion:{col:'#ff8b8b',bg:'rgba(255,139,139,.10)',label:'motion',y:yv(52),h:yv(18)},
    tts:{...LANE_STYLE.tts,y:yv(76),h:yv(22)},
    mus:{...LANE_STYLE.mus,y:yv(102),h:yv(22)},
    cap:{...LANE_STYLE.cap,y:yv(128),h:yv(22)},
    vid:{...LANE_STYLE.vid,y:yv(154),h:yv(22)},
  };
}
const rulerTop = () => yv(186);   // ticks/labels live below this y

function roundRect(ctx,x,y,w,h,r){
  r=Math.min(r,h/2,w/2); if(w<1)w=1;
  ctx.beginPath();
  ctx.moveTo(x+r,y); ctx.arcTo(x+w,y,x+w,y+h,r); ctx.arcTo(x+w,y+h,x,y+h,r);
  ctx.arcTo(x,y+h,x,y,r); ctx.arcTo(x,y,x+w,y,r); ctx.closePath();
}
// realW: keep the bar's real (uncompressed) length — speed regions speed up only
// the video, so voice-over/music/captions stay full-length and overlap it.
function laneBar(ctx,lane,start,end,off,p,text,closable,realW){
  const x=srcToOut(start)*p-off;
  const w=Math.max(3,(realW?(end-start):(srcToOut(end)-srcToOut(start)))*p);
  roundRect(ctx,x,lane.y,w,lane.h,5);
  ctx.fillStyle=lane.bg; ctx.fill();
  ctx.strokeStyle=lane.col; ctx.lineWidth=1.5; ctx.stroke();
  // crisp start/end edges
  ctx.fillStyle=lane.col;
  ctx.fillRect(x,lane.y,2,lane.h); ctx.fillRect(x+w-2,lane.y,2,lane.h);
  if(text){
    ctx.save(); roundRect(ctx,x,lane.y,w,lane.h,5); ctx.clip();
    ctx.fillStyle='#eaf2ff'; ctx.font='11px system-ui';
    ctx.fillText(text, x+6, lane.y+lane.h/2+4);
    ctx.restore();
  }
  // delete affordance (top-right ✕) when the bar is wide enough to hit
  if(closable && w>=20){
    ctx.save(); roundRect(ctx,x,lane.y,w,lane.h,5); ctx.clip();
    ctx.fillStyle='rgba(255,107,107,.92)';
    ctx.fillRect(x+w-15,lane.y,15,14);
    ctx.fillStyle='#fff'; ctx.font='bold 11px system-ui';
    ctx.fillText('✕', x+w-12, lane.y+11);
    ctx.restore();
  }
}
function revealSelectedItem(){
  if(!selectedTimelineItem) return;
  const map={tts:'#ttsList',mus:'#musList',cap:'#capList',speed:'#spList'};
  const el=document.querySelector((map[selectedTimelineItem.kind]||'')+' .item.sel');
  if(el) el.scrollIntoView({block:'nearest'});
}
function isSelectedTimelineItem(kind,i){
  return selectedTimelineItem&&selectedTimelineItem.kind===kind&&selectedTimelineItem.index===i;
}
function selectedStroke(ctx,lane,start,end,off,p,realW){
  const x=srcToOut(start)*p-off;
  const w=Math.max(3,(realW?(end-start):(srcToOut(end)-srcToOut(start)))*p);
  ctx.save(); ctx.strokeStyle='#ffffff'; ctx.lineWidth=2; ctx.setLineDash([4,3]);
  roundRect(ctx,x-2,lane.y-2,w+4,lane.h+4,6); ctx.stroke(); ctx.restore();
}
// The peak/motion arrays cover the whole source file; a trimmed clip only
// shows the [srcStart, srcStart+duration] slice of them, mapped across its bar.
function srcWindow(arr,c){
  const N=arr.length, sd=c.srcDur||c.duration||1, s=c.srcStart||0;
  const wStart=Math.max(0,Math.min(N-1,Math.floor((s/sd)*N)));
  const wEnd=Math.min(N,Math.max(wStart+1,Math.ceil(((s+c.duration)/sd)*N)));
  return {wStart,wEnd,wn:wEnd-wStart};
}
function drawWaveform(ctx,c,off,p,lane){
  const peaks=c.waveform;
  if(!peaks||!peaks.length||c.duration<=0) return;
  const x0=srcToOut(c.offset)*p-off, x1=srcToOut(c.offset+c.duration)*p-off;
  if(x1<0||x0>tl.width) return;
  const {wStart,wEnd,wn}=srcWindow(peaks,c), span=x1-x0;
  const mid=lane.y+lane.h/2, amp=lane.h/2-4;
  ctx.save();
  ctx.fillStyle=lane.bg; ctx.fillRect(Math.max(0,x0),lane.y,Math.min(tl.width,x1)-Math.max(0,x0),lane.h);
  ctx.strokeStyle=lane.col; ctx.globalAlpha=.88; ctx.beginPath();
  const sx=wStart+Math.max(0,Math.floor((Math.max(0,-x0)/span)*wn)-2);
  const ex=Math.min(wEnd,wStart+Math.ceil(((Math.min(tl.width,x1)-x0)/span)*wn)+2);
  const drawStep=Math.max(1, Math.floor((ex-sx)/Math.max(1,tl.width*1.5)));
  for(let i=sx;i<ex;i+=drawStep){
    let peak=0;
    for(let j=i;j<Math.min(ex,i+drawStep);j++) peak=Math.max(peak,peaks[j]||0);
    const x=x0+((i-wStart)/Math.max(1,wn-1))*span, h=peak*amp;
    ctx.moveTo(x,mid-h); ctx.lineTo(x,mid+h);
  }
  ctx.stroke(); ctx.globalAlpha=1; ctx.restore();
}
function drawMotion(ctx,c,off,p,lane){
  const values=c.motion;
  if(!values||!values.length||c.duration<=0) return;
  const x0=srcToOut(c.offset)*p-off, x1=srcToOut(c.offset+c.duration)*p-off;
  if(x1<0||x0>tl.width) return;
  const {wStart,wEnd,wn}=srcWindow(values,c), span=x1-x0;
  ctx.save();
  ctx.fillStyle=lane.bg; ctx.fillRect(Math.max(0,x0),lane.y,Math.min(tl.width,x1)-Math.max(0,x0),lane.h);
  const sx=wStart+Math.max(0,Math.floor((Math.max(0,-x0)/span)*wn)-2);
  const ex=Math.min(wEnd,wStart+Math.ceil(((Math.min(tl.width,x1)-x0)/span)*wn)+2);
  const step=Math.max(1, Math.floor((ex-sx)/Math.max(1,tl.width)));
  for(let i=sx;i<ex;i+=step){
    let v=0;
    for(let j=i;j<Math.min(ex,i+step);j++) v=Math.max(v,values[j]||0);
    const x=x0+((i-wStart)/Math.max(1,wn-1))*span, h=Math.max(1,v*lane.h);
    ctx.fillStyle=v<.18?'rgba(100,255,170,.65)':'rgba(255,139,139,.75)';
    ctx.fillRect(x,lane.y+lane.h-h,Math.max(1,step*span/wn),h);
  }
  ctx.restore();
}
function draw(){
  const ctx=tl.getContext('2d'), W=tl.width, H=tl.height;
  const p=pps(), off=scroll.scrollLeft, t0=off/p, t1=(off+W)/p;
  const L=lanes(), rt=rulerTop();
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle='#0a0d14'; ctx.fillRect(0,0,W,H);
  // speed regions: drawn shrunk to their output width, with bold hatching +
  // bright edges + a label pill so they're unmistakable on the timeline.
  speeds.forEach((s,i)=>{
    const x=srcToOut(s.start)*p-off, w=Math.max(2,(srcToOut(s.end)-srcToOut(s.start))*p);
    if(x+w<0||x>W) return;
    const sel=isSelectedTimelineItem('speed',i);
    ctx.save();
    ctx.fillStyle=sel?'rgba(255,180,84,.32)':'rgba(255,180,84,.22)'; ctx.fillRect(x,0,w,H);
    ctx.beginPath(); ctx.rect(x,0,w,H); ctx.clip();          // diagonal speed hatch
    ctx.strokeStyle='rgba(255,180,84,.38)'; ctx.lineWidth=2;
    for(let xx=x-H; xx<x+w; xx+=11){ ctx.beginPath(); ctx.moveTo(xx,H); ctx.lineTo(xx+H,0); ctx.stroke(); }
    ctx.restore();
    ctx.strokeStyle=sel?'#fff':'#ffb454'; ctx.lineWidth=sel?2.5:2;
    ctx.beginPath(); ctx.moveTo(x+1,0); ctx.lineTo(x+1,H); ctx.moveTo(x+w-1,0); ctx.lineTo(x+w-1,H); ctx.stroke();
    // label pill, clamped to stay on-screen and inside the region
    const txt='⏩ '+s.factor+'×';
    ctx.font='bold 11px system-ui';
    const tw=ctx.measureText(txt).width, pad=5, pw=tw+pad*2;
    let lx=Math.max(x+2, 2); lx=Math.min(lx, x+w-2-pw); if(lx<x+2) lx=x+2;
    ctx.fillStyle='#ffb454'; roundRect(ctx,lx,3,pw,16,4); ctx.fill();
    ctx.fillStyle='#231603'; ctx.fillText(txt, lx+pad, 15);
  });
  // pending speed selection
  if(spStart!=null){
    const a=srcToOut(spStart)*p-off, b=srcToOut(spEnd!=null?spEnd:globalTime())*p-off;
    ctx.fillStyle='rgba(255,180,84,.12)';
    ctx.fillRect(Math.min(a,b),0,Math.abs(b-a),H);
  }
  if(cutStartSel!=null){
    const a=srcToOut(cutStartSel)*p-off, b=srcToOut(cutEndSel!=null?cutEndSel:globalTime())*p-off;
    ctx.fillStyle='rgba(255,107,107,.20)'; ctx.fillRect(Math.min(a,b),0,Math.abs(b-a),H);
  }
  // shared right-click selection
  if(rangeSel.start!=null){
    const a=srcToOut(rangeSel.start)*p-off, b=srcToOut(rangeSel.end!=null?rangeSel.end:rangeSel.start)*p-off;
    ctx.fillStyle='rgba(255,255,255,.09)'; ctx.fillRect(Math.min(a,b),0,Math.max(2,Math.abs(b-a)),H);
    ctx.strokeStyle='rgba(255,255,255,.55)'; ctx.setLineDash([5,4]); ctx.strokeRect(Math.min(a,b)+.5,.5,Math.max(2,Math.abs(b-a)),H-1); ctx.setLineDash([]);
  }
  // ruler ticks (bottom strip)
  const frameDur=1/(curClipFps()||30), framePx=frameDur*p;
  const targets=[0.04,0.1,0.2,0.5,1,2,5,10,30,60,300];
  const step=targets.find(s=>s*p>=70)||600;
  ctx.strokeStyle='#1c2740'; ctx.beginPath();
  if(framePx>=5){
    for(let f=Math.floor(t0/frameDur); f*frameDur<=t1; f++){
      const x=f*frameDur*p-off; ctx.moveTo(x,H-12); ctx.lineTo(x,H);
    }
  }
  ctx.stroke();
  ctx.strokeStyle='#2a3245'; ctx.fillStyle='#8a96ad'; ctx.font='10px system-ui';
  ctx.beginPath();
  for(let s=Math.floor(t0/step)*step; s<=t1; s+=step){
    const x=s*p-off; ctx.moveTo(x,rt); ctx.lineTo(x,H-12);
    ctx.fillText(fmt(s).replace(/^00:/,''), x+3, H-2);
  }
  ctx.stroke();
  // ---- original audio waveform lane ----
  for(const c of clips){ drawWaveform(ctx,c,off,p,L.wave); }
  // ---- lightweight static/motion lane ----
  for(const c of clips){ drawMotion(ctx,c,off,p,L.motion); }
  // ---- video lane (the concatenated playlist; drag to reorder) ----
  for(const c of clips){
    laneBar(ctx,L.vid,c.offset,c.offset+c.duration,off,p,'🎞 '+c.label,true);
  }
  // ---- voice-over lane (normal speed: full length, overlaps sped video) ----
  for(let i=0;i<tts.length;i++){
    const c=tts[i]; laneBar(ctx,L.tts,c.time,ttsEndOf(c),off,p,'🔊 '+c.text,true,true);
    if(isSelectedTimelineItem('tts',i)) selectedStroke(ctx,L.tts,c.time,ttsEndOf(c),off,p,true);
  }
  // pending voice-over selection
  if(ttsStart!=null){
    const a=srcToOut(ttsStart)*p-off, b=srcToOut(ttsEnd!=null?ttsEnd:globalTime())*p-off;
    ctx.fillStyle='rgba(55,217,160,.18)';
    ctx.fillRect(Math.min(a,b),L.tts.y,Math.abs(b-a),L.tts.h);
  }
  // ---- music lane (each clip start → end, looped) ----
  for(let i=0;i<musics.length;i++){
    const m=musics[i];
    const end=musEndOf(m);
    laneBar(ctx,L.mus,m.start,end,off,p,
            '🎵 '+m.label+'  ('+$('#musVol').value+'%)',true,true);
    if(isSelectedTimelineItem('mus',i)) selectedStroke(ctx,L.mus,m.start,end,off,p,true);
    if(m.duration>0){            // loop boundary ticks (real spacing from the start)
      ctx.strokeStyle=L.mus.col; ctx.globalAlpha=.5; ctx.beginPath();
      const x0=srcToOut(m.start);
      for(let t=m.start+m.duration; t<end; t+=m.duration){
        const x=(x0+(t-m.start))*p-off; ctx.moveTo(x,L.mus.y); ctx.lineTo(x,L.mus.y+L.mus.h);
      }
      ctx.stroke(); ctx.globalAlpha=1;
    }
  }
  // pending music selection
  if(musStartSel!=null){
    const a=srcToOut(musStartSel)*p-off, b=srcToOut(musEndSel!=null?musEndSel:globalTime())*p-off;
    ctx.fillStyle='rgba(122,162,255,.18)';
    ctx.fillRect(Math.min(a,b),L.mus.y,Math.abs(b-a),L.mus.h);
  }
  // ---- caption lane (normal speed: full length, overlaps sped video) ----
  caps.forEach((c,i)=>{ laneBar(ctx,L.cap,c.start,c.end,off,p,'💬 '+c.text,true,true);
    if(isSelectedTimelineItem('cap',i)) selectedStroke(ctx,L.cap,c.start,c.end,off,p,true); });
  // pending caption selection
  if(capStart!=null){
    const a=srcToOut(capStart)*p-off, b=srcToOut(capEnd!=null?capEnd:globalTime())*p-off;
    ctx.fillStyle='rgba(217,160,255,.18)';
    ctx.fillRect(Math.min(a,b),L.cap.y,Math.abs(b-a),L.cap.h);
  }
  // pinned lane labels (stay at left edge while scrolling)
  ctx.font='10px system-ui';
  for(const k of ['wave','motion','tts','mus','cap','vid']){
    const lane=L[k];
    ctx.fillStyle='rgba(10,13,20,.8)'; ctx.fillRect(0,lane.y,46,12);
    ctx.fillStyle=lane.col; ctx.fillText(lane.label, 2, lane.y+10);
  }
  // playhead
  const px=srcToOut(globalTime())*p-off;
  ctx.fillStyle='#ff6b6b'; ctx.fillRect(px-1,0,2,H);
}
scroll.addEventListener('scroll', ()=>{draw();});
window.addEventListener('resize', layout);

// ---- draggable splitter between the main area and the right sidebar ----
const _main=document.querySelector('main'), _splitter=$('#splitter');
let _splitDrag=false;
function setSidebarWidth(w){
  const max=Math.max(280, _main.clientWidth-360);
  w=Math.max(280, Math.min(max, w));
  _main.style.setProperty('--rightw', w+'px');
}
_splitter.addEventListener('mousedown', e=>{
  _splitDrag=true; _splitter.classList.add('drag');
  document.body.style.cursor='col-resize'; e.preventDefault();
});
window.addEventListener('mousemove', e=>{
  if(!_splitDrag) return;
  setSidebarWidth(_main.getBoundingClientRect().right - e.clientX - 8);
  layout();                       // left column changed width → re-fit timeline + player
});
window.addEventListener('mouseup', ()=>{
  if(!_splitDrag) return;
  _splitDrag=false; _splitter.classList.remove('drag');
  document.body.style.cursor=''; layout(); scheduleSessionSave();
});

// ---------- timeline pointer interaction (seek / drag-move / delete) ----------
function ttsEndOf(c){ return (c.end!=null&&c.end>c.time)?c.end:c.time+(c.dur||0.4); }
// First start time at or after `anchor` where a clip of `dur` seconds fits
// without overlapping existing voice-over; slides past speech already there.
function firstFreeTtsSlot(anchor, dur){
  const iv=tts.map(c=>[c.time, ttsEndOf(c)]).sort((a,b)=>a[0]-b[0]);
  let t=anchor;
  for(const [s,e] of iv){
    if(e<=t+1e-3) continue;        // ends before our cursor → ignore
    if(s>=t+dur-1e-3) break;       // enough room before this clip → done
    t=e;                           // overlaps the window → jump past it
  }
  return t;
}
function musEndOf(m){ return (m.end!=null&&m.end>m.start)?m.end:INFO.duration; }

// ---------- timeline audio preview (voice-over + music while playing) --------
// Plays the voice-over and music tracks in sync with the playhead so the
// timeline can be auditioned without rendering. Preview gain is clamped to
// 100% (HTMLAudio can't amplify); the final render still honours the full
// volume sliders. Each tts/music item caches its own <audio> on `_audio`.
const previewAudios=new Set();
function clamp01(v){ return Math.max(0,Math.min(1,v)); }
function previewAudioFor(item, url){
  if(item._audio && item._audioUrl===url) return item._audio;
  if(item._audio){ try{item._audio.pause();}catch(_){} previewAudios.delete(item._audio); }
  const a=new Audio(); a.preload='auto'; a.src=url;
  item._audio=a; item._audioUrl=url; previewAudios.add(a);
  return a;
}
function drivePreviewAudio(a, want, desired, vol, loop){
  a.loop=!!loop;
  if(!want){ if(!a.paused) a.pause(); return; }
  a.volume=clamp01(vol);
  if(Math.abs(a.playbackRate-playRate)>1e-3) a.playbackRate=playRate;
  if(a.paused){
    if(isFinite(desired)) try{a.currentTime=Math.max(0,desired);}catch(_){}
    a.play().catch(()=>{});
  } else if(isFinite(desired) && Math.abs(a.currentTime-desired)>0.3){
    try{a.currentTime=Math.max(0,desired);}catch(_){}
  }
}
function syncTimelineAudio(){
  // Original-clip mode plays only the raw <video> audio; mute timeline tracks.
  const active=new Set();
  if(playing && !playOriginalMode()){
    const ttsVol=(+$('#ttsVol').value||0)/100, musVol=(+$('#musVol').value||0)/100;
    for(const c of tts){
      const want=headT>=c.time && headT<ttsEndOf(c);
      const a=previewAudioFor(c, c.url);
      drivePreviewAudio(a, want, headT-c.time, ttsVol, false);
      if(want) active.add(a);
    }
    for(const m of musics){
      const end=musEndOf(m), want=headT>=m.start && headT<end;
      const a=previewAudioFor(m, mediaUrl(m.ref));
      const into=headT-m.start;
      const desired=(m.duration>0)?(into % m.duration):into;
      drivePreviewAudio(a, want, desired, musVol, m.duration>0);
      if(want) active.add(a);
    }
  }
  // Pause anything no longer in-window (incl. deleted items still cached).
  for(const a of previewAudios) if(!active.has(a) && !a.paused) a.pause();
  // Keep the source video's own audio at the "Original audio" slider level.
  player.volume=clamp01((+$('#baseVol').value||0)/100);
}
function stopTimelineAudio(){ for(const a of previewAudios) if(!a.paused) a.pause(); }
function evCoords(e){
  const r=tl.getBoundingClientRect();
  const mx=e.clientX-r.left, my=e.clientY-r.top;
  // The pixel axis is output time; map back to source time for editing.
  return {mx,my,t:outToSrc((mx+scroll.scrollLeft)/pps())};
}
function timelineContainsEvent(e){
  const r=tl.getBoundingClientRect();
  return e.clientX>=r.left&&e.clientX<=r.right&&e.clientY>=r.top&&e.clientY<=r.bottom;
}
function renderHoverPreview(e){
  if(drag||!timelineContainsEvent(e)) { hideHoverPreview(); return; }
  const {mx,t}=evCoords(e);
  const i=activeClipAt(t);
  if(i<0){ hideHoverPreview(); return; }
  const c=clips[i], local=(c.srcStart||0)+Math.max(0,Math.min(c.duration-.04,t-c.offset));
  if(i!==hoverClip){ hoverClip=i; hoverVideo.src=mediaUrl(c.name); }
  hoverTime.textContent=fmt(t);
  const r=tl.getBoundingClientRect();
  const left=Math.max(8,Math.min(window.innerWidth-196,r.left+mx-90));
  const top=(r.top>130)?Math.max(8,r.top-126):Math.min(window.innerHeight-126,r.bottom+8);
  hoverPreview.style.left=left+'px';
  hoverPreview.style.top=top+'px';
  hoverPreview.style.display='block';
  hoverVisible=true;
  clearTimeout(hoverSeekTimer);
  hoverSeekTimer=setTimeout(()=>{ try{ hoverVideo.currentTime=local; }catch(_){} }, 30);
}
function showHoverPreview(e){
  if(drag||!timelineContainsEvent(e)){ hideHoverPreview(); return; }
  hoverPointerInside=true;
  lastHoverEvent={clientX:e.clientX,clientY:e.clientY};
  hoverLastMove=performance.now();
  renderHoverPreview(lastHoverEvent);
  clearTimeout(hoverIdleTimer);
  hoverIdleTimer=setTimeout(()=>{ hideHoverPreview(); }, 2000);
}
function hideHoverPreview(){
  hoverPreview.style.display='none'; hoverClip=-1;
  clearTimeout(hoverSeekTimer); clearTimeout(hoverIdleTimer);
}
function hitTest(mx,my){
  const p=pps(), off=scroll.scrollLeft;
  const L=lanes();
  const test=(s,e,L,realW)=>{
    if(my<L.y||my>L.y+L.h) return null;
    const x=srcToOut(s)*p-off, w=Math.max(3,(realW?(e-s):(srcToOut(e)-srcToOut(s)))*p);
    if(mx<x-3||mx>x+w+3) return null;
    return (w>=20&&mx>=x+w-15&&my<=L.y+14)?'close':'body';
  };
  // Speed regions: grab them by their top "⏩ N×" label pill (matches draw()).
  {
    const cx=tl.getContext('2d'); cx.font='bold 11px system-ui';
    for(let i=speeds.length-1;i>=0;i--){
      const s=speeds[i];
      const x=srcToOut(s.start)*p-off, w=Math.max(2,(srcToOut(s.end)-srcToOut(s.start))*p);
      if(x+w<0||x>tl.width) continue;
      const pw=cx.measureText('⏩ '+s.factor+'×').width+10;
      let lx=Math.max(x+2,2); lx=Math.min(lx,x+w-2-pw); if(lx<x+2) lx=x+2;
      if(mx>=lx-2&&mx<=lx+pw+2&&my>=1&&my<=21) return {kind:'speed',index:i,part:'body'};
    }
  }
  for(let i=tts.length-1;i>=0;i--){
    const part=test(tts[i].time,ttsEndOf(tts[i]),L.tts,true); if(part) return{kind:'tts',index:i,part}; }
  for(let i=musics.length-1;i>=0;i--){
    const part=test(musics[i].start,musEndOf(musics[i]),L.mus,true); if(part) return{kind:'mus',index:i,part}; }
  for(let i=caps.length-1;i>=0;i--){
    const part=test(caps[i].start,caps[i].end,L.cap,true); if(part) return{kind:'cap',index:i,part}; }
  for(let i=clips.length-1;i>=0;i--){
    const part=test(clips[i].offset,clips[i].offset+clips[i].duration,L.vid); if(part) return{kind:'vid',index:i,part}; }
  return null;
}
function removeItem(kind,i){
  if(kind==='vid'){ removeClip(i); return; }   // removeClip pushes its own undo
  pushUndo();
  if(kind==='tts'){ tts.splice(i,1); renderTts(); }
  else if(kind==='mus'){ musics.splice(i,1); renderMusic(); }
  else if(kind==='cap'){ caps.splice(i,1); renderCaps(); }
  if(selectedTimelineItem&&selectedTimelineItem.kind===kind&&selectedTimelineItem.index===i) selectedTimelineItem=null;
  draw();
}
function selectTimelineItem(kind,index){
  selectedTimelineItem=(['tts','mus','cap','speed'].includes(kind))?{kind,index}:null;
  // refresh the side-bar lists so the matching row is highlighted, then reveal it
  renderTts(); renderMusic(); renderCaps(); renderSpeeds();
  revealSelectedItem();
  draw();
}
function copySelectedTimelineItem(){
  if(!selectedTimelineItem){ setStatus('Select a voice or music clip on the timeline first.','err'); return false; }
  const {kind,index}=selectedTimelineItem;
  if(kind==='tts'&&!tts[index]) return false;
  if(kind==='mus'&&!musics[index]) return false;
  timelineClipboard={kind, clip:JSON.parse(JSON.stringify(kind==='tts'?tts[index]:musics[index]))};
  setStatus((kind==='tts'?'Voice-over':'Music')+' clip copied. Click the timeline and paste to duplicate.','ok');
  return true;
}
async function copyTimelineItemToSystemClipboard(){
  if(!copySelectedTimelineItem()) return;
  if(!navigator.clipboard?.writeText) return;
  try{ await navigator.clipboard.writeText(JSON.stringify({coderaiVideoEditorClip:timelineClipboard})); }
  catch(_){}
}
function pasteTimelineItem(at=null){
  if(!timelineClipboard){ setStatus('Copy a voice or music clip first.','err'); return false; }
  pushUndo();
  const kind=timelineClipboard.kind, src=JSON.parse(JSON.stringify(timelineClipboard.clip));
  const t=Math.max(0, at!=null?at:globalTime());
  if(kind==='tts'){
    const len=Math.max(0.05, ttsEndOf(src)-src.time);
    src.time=t; if(src.end!=null) src.end=t+len;
    tts.push(src); tts.sort((a,b)=>a.time-b.time);
    const idx=tts.indexOf(src); renderTts(); selectTimelineItem('tts',idx);
    setStatus('Voice-over duplicated at '+fmt(t),'ok');
  } else if(kind==='mus'){
    const len=Math.max(0.05, musEndOf(src)-src.start);
    src.start=t; if(src.end!=null) src.end=t+len;
    musics.push(src); musics.sort((a,b)=>a.start-b.start);
    const idx=musics.indexOf(src); renderMusic(); selectTimelineItem('mus',idx);
    setStatus('Music duplicated at '+fmt(t),'ok');
  }
  draw(); scheduleSessionSave();
  return true;
}
async function pasteTimelineItemFromSystemClipboard(at=null){
  if(navigator.clipboard?.readText){
    try{
      const obj=JSON.parse(await navigator.clipboard.readText()||'{}');
      const clip=obj&&obj.coderaiVideoEditorClip;
      if(clip&&['tts','mus'].includes(clip.kind)&&clip.clip) timelineClipboard=clip;
    }catch(_){}
  }
  return pasteTimelineItem(at);
}
function orderedRange(){
  if(rangeSel.start==null||rangeSel.end==null||Math.abs(rangeSel.end-rangeSel.start)<1e-3) return null;
  return {start:Math.min(rangeSel.start,rangeSel.end), end:Math.max(rangeSel.start,rangeSel.end)};
}
function rangeOr(start,end){
  const r=orderedRange();
  return {start:(start!=null?start:(r?r.start:globalTime())), end:(end!=null?end:(r?r.end:null))};
}
const timeVars={
  spStart:{get:()=>spStart,set:v=>spStart=v}, spEnd:{get:()=>spEnd,set:v=>spEnd=v},
  ttsStart:{get:()=>ttsStart,set:v=>ttsStart=v}, ttsEnd:{get:()=>ttsEnd,set:v=>ttsEnd=v},
  cutStartSel:{get:()=>cutStartSel,set:v=>cutStartSel=v}, cutEndSel:{get:()=>cutEndSel,set:v=>cutEndSel=v},
  capStart:{get:()=>capStart,set:v=>capStart=v}, capEnd:{get:()=>capEnd,set:v=>capEnd=v},
  musStartSel:{get:()=>musStartSel,set:v=>musStartSel=v}, musEndSel:{get:()=>musEndSel,set:v=>musEndSel=v},
};
function setTimeInput(id,val,placeholder){
  const el=$(id); if(!el) return;
  if(document.activeElement!==el) el.value=val!=null?fmt(val):'';
  el.placeholder=placeholder;
}
function bindTimeInputs(){
  for(const el of document.querySelectorAll('.time-edit')){
    const apply=()=>{
      const spec=timeVars[el.dataset.var]; if(!spec) return;
      const v=parseTime(el.value);
      if(v==null){ spec.set(null); }
      else if(Number.isNaN(v)){ setStatus('Invalid time: '+el.value,'err'); el.value=spec.get()!=null?fmt(spec.get()):''; return; }
      else spec.set(Math.max(0,Math.min(INFO.duration||v,v)));
      updateSpLabels(); updateTtsSel(); updateCutLabels(); updateCapLabels(); updateMusSel(); draw(); scheduleSessionSave();
    };
    el.addEventListener('change', apply);
    el.addEventListener('keydown', e=>{ if(e.key==='Enter'){ e.preventDefault(); el.blur(); } });
  }
}
function selectionFromPlayhead(){
  const t=globalTime();
  rangeSel={start:t,end:null,next:'end'};
  setStatus('Selection start '+fmt(t),'ok');
  scheduleSessionSave(); draw();
}
function setRangePoint(t){
  if(rangeSel.next==='start'||rangeSel.start==null){ rangeSel.start=Math.max(0,t); rangeSel.end=null; rangeSel.next='end'; }
  else { rangeSel.end=Math.max(0,t); rangeSel.next='start'; }
  const r=orderedRange();
  setStatus(r?('Timeline selection '+fmt(r.start)+' → '+fmt(r.end)):'Selection start '+fmt(rangeSel.start),'ok');
  scheduleSessionSave();
  draw();
}
// Sticky snapping: pull a clip's start/end onto nearby clip edges, 0, or the
// playhead when within ~8px, but otherwise let it move freely. Distances are
// measured in OUTPUT (on-screen) space so the ~8px pull stays consistent even
// across sped-up regions, where source seconds and screen pixels diverge.
function snapClip(i, off){
  const dur=clips[i].duration, thresh=8/pps();   // 8px expressed in output seconds
  const targets=[0, headT];
  for(let j=0;j<clips.length;j++){ if(j===i) continue;
    targets.push(clips[j].offset, clips[j].offset+clips[j].duration); }
  const oOff=srcToOut(off), oEnd=srcToOut(off+dur);
  let best=off, bestD=thresh;
  for(const tg of targets){
    const oTg=srcToOut(tg);
    if(Math.abs(oOff-oTg)<bestD){ best=tg; bestD=Math.abs(oOff-oTg); }          // start edge
    if(Math.abs(oEnd-oTg)<bestD){ best=tg-dur; bestD=Math.abs(oEnd-oTg); }      // end edge
  }
  return Math.max(0,best);
}
// Snapshot every track's positions so a locked drag can shift them all by one
// delta off their starting points (no cumulative drift), and remember the
// earliest start so the group can't be pushed below 0.
function captureTrackOrigins(){
  const clipsO=clips.map(c=>c.offset);
  const ttsO=tts.map(c=>({time:c.time,end:c.end}));
  const musO=musics.map(m=>({start:m.start,end:m.end}));
  const capO=caps.map(c=>({start:c.start,end:c.end}));
  const mins=[...clipsO,...ttsO.map(o=>o.time),...musO.map(o=>o.start),...capO.map(o=>o.start)];
  return {clips:clipsO,tts:ttsO,mus:musO,cap:capO,min:mins.length?Math.min(...mins):0};
}
// The start time of whatever piece the user grabbed (for edge snapping).
function grabOrigStart(drag){
  if(drag.kind==='tts') return drag.orig.time;
  if(drag.kind==='vid') return drag.orig.offset;
  return drag.orig.start;
}
function applyLockedMove(drag, dt){
  const L=drag.lock;
  let delta=dt;
  // snap the grabbed piece's start to 0 or the playhead, then clamp so nothing
  // in the group slides past the start of the timeline
  const gs=grabOrigStart(drag)+delta, thr=8/pps();
  for(const tg of [0, headT]){ if(Math.abs(gs-tg)<thr){ delta+=tg-gs; break; } }
  delta=Math.max(delta, -L.min);
  clips.forEach((c,i)=>{ c.offset=L.clips[i]+delta; });
  tts.forEach((c,i)=>{ c.time=L.tts[i].time+delta; if(L.tts[i].end!=null) c.end=L.tts[i].end+delta; });
  musics.forEach((m,i)=>{ m.start=L.mus[i].start+delta; if(L.mus[i].end!=null) m.end=L.mus[i].end+delta; });
  caps.forEach((c,i)=>{ c.start=L.cap[i].start+delta; c.end=L.cap[i].end+delta; });
  recompute();
}
tl.addEventListener('mousedown', e=>{
  hideHoverPreview();
  if(e.button===2){ e.preventDefault(); return; }
  const {mx,my,t}=evCoords(e);
  const hit=hitTest(mx,my);
  if(hit){
    if(hit.part==='close'){ removeItem(hit.kind,hit.index); e.preventDefault(); return; }
    selectTimelineItem(hit.kind,hit.index);
    drag={...hit, grabT:t}; dragMoved=false;
    if(hit.kind==='tts'){ const c=tts[hit.index]; drag.item=c; drag.orig={time:c.time,end:c.end}; }
    else if(hit.kind==='mus'){ const m=musics[hit.index]; drag.item=m; drag.orig={start:m.start,end:m.end}; }
    else if(hit.kind==='cap'){ const c=caps[hit.index]; drag.item=c; drag.orig={start:c.start,end:c.end}; }
    else if(hit.kind==='vid'){ drag.orig={offset:clips[hit.index].offset}; }
    if(lockTracksMode()) drag.lock=captureTrackOrigins();   // move every track as one
    drag._undo=snapshotState();   // committed on drop only if the drag actually moved
    e.preventDefault();
  } else { selectedTimelineItem=null; draw(); downT=t; }
});
tl.addEventListener('contextmenu', e=>{
  e.preventDefault();
  const {t}=evCoords(e);
  setRangePoint(Math.max(0,Math.min(INFO.duration||0,t)));
});
window.addEventListener('mousemove', e=>{
  if(drag){
    const {t}=evCoords(e), dt=t-drag.grabT;
    if(Math.abs(dt)>1e-4) dragMoved=true;
    if(drag.lock){ applyLockedMove(drag, dt); draw(); return; }
    if(drag.kind==='tts'){ const c=tts[drag.index]; const nt=Math.max(0,drag.orig.time+dt);
      if(drag.orig.end!=null) c.end=drag.orig.end+(nt-drag.orig.time); c.time=nt; }
    else if(drag.kind==='mus'){ const m=musics[drag.index]; const ns=Math.max(0,drag.orig.start+dt);
      if(drag.orig.end!=null) m.end=drag.orig.end+(ns-drag.orig.start); m.start=ns; }
    else if(drag.kind==='cap'){ const c=caps[drag.index]; const len=drag.orig.end-drag.orig.start;
      c.start=Math.max(0,drag.orig.start+dt); c.end=c.start+len; }
    else if(drag.kind==='vid'){ const c=clips[drag.index];
      c.offset=snapClip(drag.index, Math.max(0,drag.orig.offset+dt)); recompute(); }
    draw(); return;
  }
  // hover cursor feedback
  const {mx,my}=evCoords(e), h=hitTest(mx,my);
  tl.style.cursor = h ? (h.part==='close'?'pointer':'grab') : 'default';
  showHoverPreview(e);
});
tl.addEventListener('mouseenter', e=>{hoverPointerInside=true; showHoverPreview(e);});
tl.addEventListener('mouseleave', ()=>{hoverPointerInside=false; hideHoverPreview();});
scroll.addEventListener('scroll', hideHoverPreview);
window.addEventListener('mouseup', e=>{
  if(drag){
    // A drag that actually moved is one undoable step (snapshot taken at grab).
    if(dragMoved && drag._undo){
      undoStack.push(drag._undo);
      if(undoStack.length>UNDO_MAX) undoStack.shift();
      redoStack=[];
    }
    if(drag.lock){
      tts.sort((a,b)=>a.time-b.time); musics.sort((a,b)=>a.start-b.start); caps.sort((a,b)=>a.start-b.start);
      recompute(); renderClips(); renderTts(); renderMusic(); renderCaps(); layout();
      seekGlobal(Math.min(headT,INFO.duration),false); selectedTimelineItem=null; scheduleSessionSave();
      drag=null; draw(); return;
    }
    if(drag.kind==='vid' && dragMoved){ recompute(); renderClips(); layout();
      seekGlobal(Math.min(headT,INFO.duration),false); scheduleSessionSave(); }
    else if(drag.kind==='tts'){
      tts.sort((a,b)=>a.time-b.time); renderTts(); selectedTimelineItem={kind:'tts',index:tts.indexOf(drag.item)}; scheduleSessionSave(); }
    else if(drag.kind==='mus'){
      musics.sort((a,b)=>a.start-b.start); renderMusic(); selectedTimelineItem={kind:'mus',index:musics.indexOf(drag.item)}; scheduleSessionSave(); }
    else if(drag.kind==='cap'){ caps.sort((a,b)=>a.start-b.start); renderCaps(); scheduleSessionSave(); }
    // A click that didn't drag a voice-over / caption opens its editor.
    if(!dragMoved && (drag.kind==='tts'||drag.kind==='cap')) openTimelineEditor(drag.kind, drag.item);
    drag=null; draw(); return;
  }
  if(downT!=null){ seekGlobal(downT,false); downT=null; }
});
function tick(ts){
  if(!lastTs) lastTs=ts||0;
  const dt=((ts||0)-lastTs)/1000; lastTs=ts||0;
  if(playing){
    if(playOriginalMode()){
      if(curIdx>=0 && !player.paused) headT=clips[curIdx].offset+player.currentTime;
    } else if(curIdx>=0 && !player.paused && !player.ended){
      headT=clips[curIdx].offset+player.currentTime-(clips[curIdx].srcStart||0);
    } else {
      headT+=Math.max(0,dt);                 // advance through gaps / clip ends
    }
    updatePreviewOverlays();
    if(followPlayheadMode()) keepTimeVisible(headT);
    if(headT>=INFO.duration){ headT=INFO.duration; playing=false; player.pause(); }
    else {
      const i=activeClipAt(headT);
      if(i!==curIdx){
        if(i>=0) loadClip(i, headT-clips[i].offset, true);
        else { curIdx=-1; player.pause(); }
      }
    }
  }
  syncTimelineAudio();
  updateClock(); draw();
  if(isFs && fsTimeline.style.display!=='none') drawMini();
  requestAnimationFrame(tick);
}
function shortcutsEditableTarget(e){
  const tag=(e.target&&e.target.tagName||'').toLowerCase();
  return tag==='input'||tag==='textarea'||tag==='select'||e.target?.isContentEditable;
}
document.addEventListener('keydown', e=>{
  if(shortcutsEditableTarget(e)) return;
  const fs=!!document.fullscreenElement;
  const key=String(e.key||'').toLowerCase();
  if((e.ctrlKey||e.metaKey)&&key==='z'&&!e.shiftKey){ e.preventDefault(); undo(); }
  else if((e.ctrlKey||e.metaKey)&&(key==='y'||(key==='z'&&e.shiftKey))){ e.preventDefault(); redo(); }
  else if((e.ctrlKey||e.metaKey)&&key==='c'){ e.preventDefault(); copyTimelineItemToSystemClipboard(); }
  else if((e.ctrlKey||e.metaKey)&&key==='v'){
    e.preventDefault(); pasteTimelineItemFromSystemClipboard(hoverPointerInside&&lastHoverEvent?evCoords(lastHoverEvent).t:globalTime());
  }
  else if(e.key==='[' || (fs&&e.key==='ArrowDown')){ e.preventDefault(); nudgeRate(-1); }
  else if(e.key===']' || (fs&&e.key==='ArrowUp')){ e.preventDefault(); nudgeRate(1); }
  else if(e.key==='0'){ e.preventDefault(); setPlayRate(1); }
});
player.addEventListener('ratechange', ()=>{
  if(Math.abs(player.playbackRate-playRate)>1e-6) setPlayRate(player.playbackRate);
});
function updateClock(){
  const t=globalTime();
  $('#tcode').textContent=fmt(t);
  $('#frameNo').textContent='f '+Math.round(t*curClipFps());
}
requestAnimationFrame(tick);

// zoom + transport
$('#zoom').addEventListener('input', e=>{ ppsExp=+e.target.value; layout(globalTime()); });
$('#zIn').onclick=()=>{ppsExp=Math.min(1000,ppsExp+80);$('#zoom').value=ppsExp;layout(globalTime());};
$('#zOut').onclick=()=>{ppsExp=Math.max(-1000,ppsExp-80);$('#zoom').value=ppsExp;layout(globalTime());};
$('#zoomY').addEventListener('input', e=>{ zoomY=+e.target.value; layout(globalTime()); });
$('#zyIn').onclick=()=>{zoomY=Math.min(100,zoomY+10);$('#zoomY').value=zoomY;layout(globalTime());};
$('#zyOut').onclick=()=>{zoomY=Math.max(0,zoomY-10);$('#zoomY').value=zoomY;layout(globalTime());};
$('#frPrev').onclick=()=>seekGlobal(globalTime()-1/curClipFps(),false);
$('#frNext').onclick=()=>seekGlobal(globalTime()+1/curClipFps(),false);
$('#secPrev').onclick=()=>seekGlobal(globalTime()-1,false);
$('#secNext').onclick=()=>seekGlobal(globalTime()+1,false);
$('#playBtn').onclick=()=>playPause();
$('#selFromPlay').onclick=selectionFromPlayhead;
$('#playOriginal').onchange=()=>{playing=false;player.pause();seekGlobal(globalTime(),false);scheduleSessionSave();};
$('#followPlayhead').onchange=scheduleSessionSave;
$('#lockTracks').onchange=scheduleSessionSave;
$('#rateDown').onclick=()=>nudgeRate(-1);
$('#rateUp').onclick=()=>nudgeRate(1);
setPlayRate(1);
$('#addClip').onclick=()=>addClip();
$('#reloadBtn').onclick=()=>loadMedia();
$('#videoFile').onchange=async e=>{
  const f=e.target.files[0]; if(!f) return;
  const op=startOp('upload','Uploading '+f.name);
  try{ const r=await uploadFile(f);
    advanceOp(op, 'Adding uploaded video to timeline', 92);
    await addClipRef(r.ref, r.label, {duration:r.duration, fps:r.fps, hasAudio:r.hasAudio});
    finishOp(true, 'Added '+r.label, 'ok');
  }catch(err){ finishOp(false, 'Upload failed: '+err.message, 'err'); }
  e.target.value='';
};
$('#musicFile').onchange=async e=>{
  const f=e.target.files[0]; if(!f) return;
  const op=startOp('upload','Uploading '+f.name);
  try{ const r=await uploadFile(f);
    musicUpload={ref:r.ref, label:r.label};
    $('#musUpName').textContent='uploaded: '+r.label+' — set start/end, then Add';
    finishOp(true, 'Music uploaded.', 'ok');
  }catch(err){ finishOp(false, 'Upload failed: '+err.message, 'err'); }
  e.target.value='';
};

// ---------- TTS ----------
$('#ttsStartB').onclick=()=>{const r=orderedRange(); ttsStart=r?r.start:globalTime(); if(r) ttsEnd=r.end; updateTtsSel();draw();};
$('#ttsEndB').onclick=()=>{const r=orderedRange(); ttsEnd=r?r.end:globalTime(); if(r) ttsStart=r.start; updateTtsSel();draw();};
$('#ttsClr').onclick=()=>{ttsStart=ttsEnd=null;updateTtsSel();draw();};
function updateTtsSel(){
  setTimeInput('#ttsStartV',ttsStart,'playhead');
  setTimeInput('#ttsEndV',ttsEnd,'none');
}
$('#ttsAdd').onclick=async()=>{
  const btn=$('#ttsAdd');
  const text=$('#ttsText').value.trim();
  if(!text){ setStatus('Type some text first.','err'); return; }
  const sel=orderedRange();
  const explicit=(ttsStart!=null)||!!sel;     // did the user pin a start?
  const anchor=(ttsStart!=null)?ttsStart:(sel?sel.start:globalTime());
  // Only an explicit "Optional stop" truncates the clip; otherwise it plays the
  // full generated speech (a selection range only sets where it starts).
  const fixedEnd=(ttsEnd!=null&&ttsEnd>anchor)?ttsEnd:null;
  const op=startOp('tts','Generating voice',btn);
  try{
    advanceOp(op, 'Waiting for generated audio', 38);
    const speed=+($('#ttsSpeed')?.value)||1.0;
    const emotion=($('#ttsEmotionRow')&&!$('#ttsEmotionRow').hidden)?($('#ttsEmotion').value||''):'';
    const style=($('#ttsStyleRow')&&!$('#ttsStyleRow').hidden)?($('#ttsStyle').value||''):'';
    const res=await api('api/tts',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({text, voice:$('#voiceSel').value, speed, emotion, style})});
    advanceOp(op, 'Adding voice clip', 88);
    // No explicit start → drop it at the playhead, sliding past any speech
    // already there so voice-overs never overlap. A pinned start/range stays put.
    const dur=Math.max(0.05, res.duration || (fixedEnd!=null?fixedEnd-anchor:0) || 0.4);
    const at=explicit?anchor:firstFreeTtsSlot(anchor, dur);
    const end=(fixedEnd!=null)?(fixedEnd+(at-anchor)):null;
    pushUndo();
    tts.push({time:at, end, file:res.file, url:res.url, text, voice:$('#voiceSel').value, dur:res.duration});
    if($('#ttsAutoCap').checked){
      const capEnd=(end!=null&&end>at)?end:(at+Math.max(0.5,res.duration||1.5));
      caps.push({start:at,end:capEnd,text}); caps.sort((a,b)=>a.start-b.start);
      renderCaps();
    }
    tts.sort((a,b)=>a.time-b.time);
    ttsStart=ttsEnd=null; updateTtsSel();
    renderTts(); draw(); finishOp(true, 'Voice clip added at '+fmt(at), 'ok');
    $('#ttsText').value='';
  }catch(e){ finishOp(false, 'TTS error: '+e.message, 'err'); }
};
// Re-synthesize every voice-over from its saved settings, keeping its position.
async function regenAllVoice(op){
  for(let i=0;i<tts.length;i++){
    const c=tts[i];
    if(op) advanceOp(op, 'Voice '+(i+1)+' of '+tts.length, 8+(i/tts.length)*82);
    const res=await api('api/tts',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({text:c.text, voice:c.voice||CFG.defaultVoice, speed:c.speed||1.0,
        emotion:c.emotion||'', style:c.style||''})});
    const hadEnd=c.end!=null;
    c.file=res.file; c.url=res.url; c.dur=res.duration;
    if(c._audio){ try{c._audio.pause();}catch(_){} c._audio=null; c._audioUrl=null; }
    if(!hadEnd) c.end=null;          // full-speech clips track the new duration
  }
}
$('#ttsRegenAll').onclick=async()=>{
  if(!tts.length){ setStatus('No voice-overs to regenerate.','err'); return; }
  const op=startOp('tts','Regenerating all voice-overs',$('#ttsRegenAll'));
  pushUndo();
  try{
    await regenAllVoice(op);
    tts.sort((a,b)=>a.time-b.time); renderTts(); draw(); scheduleSessionSave();
    finishOp(true, tts.length+' voice-over(s) regenerated', 'ok');
  }catch(e){ finishOp(false, 'Regenerate failed: '+e.message, 'err'); }
};
$('#ttsRegenCaps').onclick=async()=>{
  if(!tts.length){ setStatus('No voice-overs to align subtitles to.','err'); return; }
  const op=startOp('tts','Regenerating voice + subtitles',$('#ttsRegenCaps'));
  pushUndo();
  try{
    await regenAllVoice(op);
    advanceOp(op,'Re-aligning subtitles',94);
    // Rebuild every subtitle straight from the (new) voice-overs so text and
    // timing line up exactly.
    caps=tts.map(c=>({start:c.time, end:ttsEndOf(c), text:c.text})).sort((a,b)=>a.start-b.start);
    tts.sort((a,b)=>a.time-b.time); renderTts(); renderCaps(); draw(); scheduleSessionSave();
    finishOp(true, 'Voice-overs regenerated; '+caps.length+' subtitle(s) re-aligned', 'ok');
  }catch(e){ finishOp(false, 'Failed: '+e.message, 'err'); }
};
function renderTts(){
  const el=$('#ttsList'); el.innerHTML='';
  tts.forEach((c,i)=>{
    const d=document.createElement('div'); d.className='item';
    if(isSelectedTimelineItem('tts',i)) d.classList.add('sel');
    d.innerHTML=`<span class="pill">${fmt(c.time)}</span>
      <span class="grow" title="${c.text.replace(/"/g,'')}">${c.text}</span>
      <button class="ghost sm">▶</button><span class="x">✕</span>`;
    d.querySelector('button').onclick=()=>new Audio(c.url).play();
    d.querySelector('.x').onclick=()=>{pushUndo();tts.splice(i,1);renderTts();draw();};
    d.querySelector('.grow').onclick=()=>openTimelineEditor('tts', c);
    d.querySelector('.grow').style.cursor='pointer';
    el.appendChild(d);
  });
  scheduleSessionSave();
}

// ---------- edit a voice-over / caption clicked on the timeline ----------
let editing=null;   // {kind:'tts'|'cap', item}
function fillVoiceSelect(sel, value){
  sel.innerHTML='';
  for(const g of ['feminine','masculine']){
    if(!CFG.voices||!CFG.voices[g]) continue;
    const og=document.createElement('optgroup'); og.label=g;
    for(const [id,label] of CFG.voices[g]){
      const o=document.createElement('option'); o.value=id; o.textContent=`${label} — ${id}`;
      og.appendChild(o);
    }
    sel.appendChild(og);
  }
  if(value!=null) sel.value=value;
}
function fillListSelect(sel, list, value){
  sel.innerHTML='';
  for(const v of (list||[])){ const o=document.createElement('option'); o.value=v; o.textContent=v; sel.appendChild(o); }
  if(value!=null) sel.value=value;
}
function fillSpeedSelect(sel, value){
  sel.innerHTML='';
  for(const s of [0.5,0.75,0.9,1.0,1.1,1.25,1.5,2.0]){
    const o=document.createElement('option'); o.value=s;
    o.textContent=(s===1.0?'1.0× (normal)':s+'×'); sel.appendChild(o);
  }
  sel.value=(value!=null?value:1.0);
}
function openTimelineEditor(kind, item){
  if(!item) return;
  editing={kind, item};
  const isTts=kind==='tts';
  $('#editTitle').textContent=isTts?'Edit voice-over':'Edit subtitle';
  $('#editText').value=item.text||'';
  $('#editVoiceWrap').hidden=!isTts;
  $('#editRegen').hidden=!isTts;
  $('#editHint').textContent=isTts
    ? 'Change the text/voice then Re-generate to resynthesize, or Save to just retime/relabel.'
    : 'Edit the subtitle text and timing, then Save.';
  if(isTts){
    // Built straight from CFG so the editor's full voice list and (if the model
    // supports them) emotion/delivery options always appear here.
    fillVoiceSelect($('#editVoice'), item.voice||CFG.defaultVoice);
    fillSpeedSelect($('#editSpeed'), item.speed||1.0);
    const emos=CFG.ttsEmotions||[], styles=CFG.ttsStyles||[];
    const ew=$('#editEmotionWrap'), sw=$('#editStyleWrap');
    ew.hidden=!emos.length;   if(emos.length)   fillListSelect($('#editEmotion'), emos, item.emotion||'');
    sw.hidden=!styles.length; if(styles.length) fillListSelect($('#editStyle'), styles, item.style||'');
    $('#editStart').value=fmt(item.time);
    $('#editEnd').value=(item.end!=null)?fmt(item.end):'';
  } else {
    $('#editStart').value=fmt(item.start);
    $('#editEnd').value=fmt(item.end);
  }
  $('#editOverlay').style.display='flex';
  $('#editText').focus();
}
function closeTimelineEditor(){ editing=null; $('#editOverlay').style.display='none'; }
$('#editCancel').onclick=closeTimelineEditor;
$('#editOverlay').addEventListener('mousedown', e=>{ if(e.target===$('#editOverlay')) closeTimelineEditor(); });
$('#editSave').onclick=()=>{
  if(!editing) return;
  const it=editing.item, text=$('#editText').value.trim();
  const s=parseTime($('#editStart').value), eRaw=$('#editEnd').value.trim();
  const e=eRaw?parseTime(eRaw):null;
  if(s==null||Number.isNaN(s)){ setStatus('Invalid start time.','err'); return; }
  if(e!=null&&(Number.isNaN(e)||e<=s)){ setStatus('End must be after start (or blank).','err'); return; }
  pushUndo();
  if(editing.kind==='tts'){
    it.text=text; it.time=Math.max(0,s); it.end=(e!=null)?e:null;
    it.voice=$('#editVoice').value;
    tts.sort((a,b)=>a.time-b.time); renderTts();
  } else {
    it.text=text; it.start=Math.max(0,s); it.end=(e!=null)?e:s+1.5;
    caps.sort((a,b)=>a.start-b.start); renderCaps();
  }
  draw(); scheduleSessionSave(); closeTimelineEditor();
  setStatus('Updated.','ok');
};
$('#editRegen').onclick=async()=>{
  if(!editing||editing.kind!=='tts') return;
  const it=editing.item, text=$('#editText').value.trim();
  if(!text){ setStatus('Type some text first.','err'); return; }
  const voice=$('#editVoice').value, speed=+($('#editSpeed').value)||1.0;
  const emotion=(!$('#editEmotionWrap').hidden)?($('#editEmotion').value||''):'';
  const style=(!$('#editStyleWrap').hidden)?($('#editStyle').value||''):'';
  const btn=$('#editRegen'); const op=startOp('tts','Re-generating voice',btn);
  try{
    advanceOp(op,'Waiting for generated audio',45);
    const res=await api('api/tts',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({text, voice, speed, emotion, style})});
    // Keep the clip where it is; swap in the new audio. If it had no explicit
    // stop (full-speech), keep it full; if it was trimmed, keep that window.
    pushUndo();
    const hadEnd=it.end!=null;
    it.text=text; it.voice=voice; it.speed=speed; it.emotion=emotion; it.style=style;
    it.file=res.file; it.url=res.url; it.dur=res.duration;
    if(it._audio){ try{it._audio.pause();}catch(_){} it._audio=null; it._audioUrl=null; }
    if(!hadEnd) it.end=null;
    tts.sort((a,b)=>a.time-b.time); renderTts(); draw(); scheduleSessionSave();
    finishOp(true,'Voice-over re-generated.','ok');
    closeTimelineEditor();
  }catch(err){ finishOp(false,'TTS error: '+err.message,'err'); }
};

// ---------- speed ----------
$('#spStart').onclick=()=>{const r=orderedRange(); spStart=r?r.start:globalTime(); if(r) spEnd=r.end; updateSpLabels();draw();};
$('#spEnd').onclick=()=>{const r=orderedRange(); spEnd=r?r.end:globalTime(); if(r) spStart=r.start; updateSpLabels();draw();};
function updateSpLabels(){
  setTimeInput('#spStartV',spStart,'—');
  setTimeInput('#spEndV',spEnd,'—');
}
$('#spAdd').onclick=()=>{
  const r=rangeOr(spStart,spEnd);
  if(r.start==null||r.end==null||r.end<=r.start){
    setStatus('Set a valid start and end first.','err'); return; }
  pushUndo();
  speeds.push({start:r.start,end:r.end,factor:+$('#spFactor').value});
  speeds.sort((a,b)=>a.start-b.start);
  spStart=spEnd=null; updateSpLabels(); renderSpeeds(); layout();
  setStatus('Speed region added: '+fmt(r.start)+' → '+fmt(r.end),'ok');
};
function renderSpeeds(){
  const el=$('#spList'); el.innerHTML='';
  speeds.forEach((s,i)=>{
    const d=document.createElement('div'); d.className='item';
    if(isSelectedTimelineItem('speed',i)) d.classList.add('sel');
    d.innerHTML=`<span class="pill spd">${s.factor}×</span>
      <span class="grow">${fmt(s.start)} → ${fmt(s.end)}</span><span class="x">✕</span>`;
    d.querySelector('.x').onclick=()=>{pushUndo();speeds.splice(i,1);renderSpeeds();layout();};
    el.appendChild(d);
  });
  scheduleSessionSave();
}

// ---------- cuts ----------
function updateCutLabels(){
  setTimeInput('#cutStartV',cutStartSel,'—');
  setTimeInput('#cutEndV',cutEndSel,'—');
}
function useRangeForCut(){
  const r=orderedRange();
  if(r){ cutStartSel=r.start; cutEndSel=r.end; updateCutLabels(); draw(); return true; }
  return false;
}
$('#cutUseSel').onclick=()=>{ if(!useRangeForCut()) setStatus('Right-click two timeline points first.','err'); };
$('#cutStart').onclick=()=>{const r=orderedRange(); cutStartSel=r?r.start:globalTime(); if(r) cutEndSel=r.end; updateCutLabels();draw();};
$('#cutEnd').onclick=()=>{const r=orderedRange(); cutEndSel=r?r.end:globalTime(); if(r) cutStartSel=r.start; updateCutLabels();draw();};
$('#cutAdd').onclick=()=>{
  const r=rangeOr(cutStartSel,cutEndSel);
  if(r.start==null||r.end==null||r.end<=r.start){ setStatus('Set a valid cut start and end first.','err'); return; }
  pushUndo();
  cuts.push({start:r.start,end:r.end});
  applyCutsToClips();
  headT=rippleTime(headT,r.start,r.end,r.end-r.start);   // keep playhead on its footage
  recompute(); headT=Math.min(headT,INFO.duration);
  cutStartSel=cutEndSel=null; updateCutLabels();
  renderCuts(); renderClips(); renderTts(); renderSpeeds(); renderCaps(); renderMusic();
  layout(globalTime()); draw();
  seekGlobal(headT,false);
  for(let i=0;i<clips.length;i++){ loadWaveform(i); loadMotion(i); }
  setStatus('Region '+fmt(r.start)+' → '+fmt(r.end)+' removed; tracks re-aligned','ok');
};
function renderCuts(){
  const el=$('#cutList'); el.innerHTML='';
  cuts.forEach((c,i)=>{
    const d=document.createElement('div'); d.className='item';
    d.innerHTML=`<span class="pill" style="background:#47252a;color:#ffd7d7">cut</span>`+
      `<span class="grow">${fmt(c.start)} → ${fmt(c.end)}</span><span class="x">✕</span>`;
    d.querySelector('.x').onclick=()=>{pushUndo();cuts.splice(i,1);renderCuts();draw();};
    el.appendChild(d);
  });
  scheduleSessionSave();
}

// ---------- captions ----------
$('#capStart').onclick=()=>{const r=orderedRange(); capStart=r?r.start:globalTime(); if(r) capEnd=r.end; updateCapLabels();draw();};
$('#capEnd').onclick=()=>{const r=orderedRange(); capEnd=r?r.end:globalTime(); if(r) capStart=r.start; updateCapLabels();draw();};
function updateCapLabels(){
  setTimeInput('#capStartV',capStart,'—');
  setTimeInput('#capEndV',capEnd,'—');
}
$('#capAdd').onclick=()=>{
  const text=$('#capText').value.trim();
  if(!text){ setStatus('Type caption text first.','err'); return; }
  const rr=rangeOr(capStart,capEnd); let s=rr.start, e=rr.end;
  if(s==null){ s=globalTime(); }
  if(e==null||e<=s){ e=s+Math.max(1.5, text.length*0.06); }
  pushUndo();
  caps.push({start:s,end:e,text}); caps.sort((a,b)=>a.start-b.start);
  capStart=capEnd=null; updateCapLabels(); $('#capText').value='';
  renderCaps(); draw();
  setStatus('Caption added at '+fmt(s),'ok');
};
$('#capFromTts').onclick=()=>{
  if(!tts.length){ setStatus('No voice-overs to caption yet.','err'); return; }
  pushUndo();
  // Regenerate ALL subtitles from the voice-overs, replacing any existing ones.
  caps=tts.map(c=>({start:c.time,end:ttsEndOf(c),text:c.text})).sort((a,b)=>a.start-b.start);
  renderCaps(); draw(); scheduleSessionSave();
  setStatus(caps.length+' subtitle(s) regenerated from voice-overs.','ok');
};
$('#capTranscribe').onclick=async()=>{
  if(!CFG.sttModel){ setStatus('Auto-caption needs a transcription model (start with --stt-model).','err'); return; }
  if(!clips.length){ setStatus('Add a video first.','err'); return; }
  const op=startOp('transcribe','Transcribing original audio',$('#capTranscribe'));
  try{
    pushUndo();
    const fresh=[]; let n=0;        // regeneration replaces the old captions
    for(let i=0;i<clips.length;i++){
      const clip=clips[i];
      advanceOp(op, 'Transcribing clip '+(i+1)+' of '+clips.length, 12 + (i/clips.length)*76);
      const r=await api('api/transcribe',{method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({name:clip.name})});
      for(const s of r.segments){
        fresh.push({start:s.start+clip.offset,end:s.end+clip.offset,text:s.text}); n++;
      }
    }
    caps=fresh.sort((a,b)=>a.start-b.start); renderCaps(); draw();
    finishOp(true, n+' caption(s) transcribed (replaced old).', 'ok');
  }catch(e){ finishOp(false, 'Transcription error: '+e.message, 'err'); }
};
function renderCaps(){
  const el=$('#capList'); el.innerHTML='';
  caps.forEach((c,i)=>{
    const d=document.createElement('div'); d.className='item';
    if(isSelectedTimelineItem('cap',i)) d.classList.add('sel');
    d.innerHTML=`<span class="pill" style="background:#2f2440;color:#e9d4ff">`+
      `${fmt(c.start).replace(/^00:/,'')}→${fmt(c.end).replace(/^00:/,'')}</span>`+
      `<span class="grow" title="${c.text.replace(/"/g,'')}">${c.text}</span>`+
      `<span class="x">✕</span>`;
    d.querySelector('.x').onclick=()=>{pushUndo();caps.splice(i,1);renderCaps();draw();};
    d.querySelector('.grow').onclick=()=>openTimelineEditor('cap', c);
    d.querySelector('.grow').style.cursor='pointer';
    el.appendChild(d);
  });
  scheduleSessionSave();
}

// ---------- music ----------
// Precedence: a generated/uploaded file, else a typed server path, else the
// media-dir dropdown. Each yields a {ref,label} resolvable server-side.
function musicSelection(){
  if(musicUpload) return {ref:musicUpload.ref, label:musicUpload.label};
  const path=$('#musicPath').value.trim();
  if(path) return {ref:'abs:'+path, label:path.split(/[\\/]/).pop()||path};
  const file=$('#musicSel').value;
  if(file) return {ref:file, label:file};
  return null;
}
function clearMusicUpload(){ if(musicUpload){ musicUpload=null; $('#musUpName').textContent=''; } }
$('#musicSel').onchange=clearMusicUpload;
$('#musicPath').addEventListener('input', ()=>{ if($('#musicPath').value.trim()) clearMusicUpload(); });
$('#musGenerate').onclick=async()=>{
  const prompt=$('#musPrompt').value.trim();
  if(!prompt){ setStatus('Type a music prompt first.','err'); return; }
  if(!CFG.audioModel&&!$('#cfgAudioModel').value.trim()){
    setStatus('Music generation needs an audio model in Settings.','err'); return;
  }
  const duration=Math.max(1,Math.min(300,+$('#musDur').value||10));
  const op=startOp('musicgen','Generating music',$('#musGenerate'));
  try{
    const r=await api('api/musicgen',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({prompt,duration,model:$('#cfgAudioModel').value.trim()||CFG.audioModel})});
    musicUpload={ref:r.ref, label:'generated: '+prompt.slice(0,60)};
    $('#musUpName').textContent='generated: '+prompt.slice(0,80)+' — set start/end, then Add';
    if(musStartSel==null) musStartSel=globalTime();
    if(musEndSel==null) musEndSel=musStartSel+(r.duration||duration);
    updateMusSel(); draw();
    finishOp(true, 'Generated '+Math.round(r.duration||duration)+'s music clip', 'ok');
  }catch(e){ finishOp(false, 'Music generation failed: '+e.message, 'err'); }
};
$('#musStartB').onclick=()=>{const r=orderedRange(); musStartSel=r?r.start:globalTime(); if(r) musEndSel=r.end; updateMusSel();draw();};
$('#musEndB').onclick=()=>{const r=orderedRange(); musEndSel=r?r.end:globalTime(); if(r) musStartSel=r.start; updateMusSel();draw();};
$('#musClr').onclick=()=>{musStartSel=musEndSel=null;updateMusSel();draw();};
function updateMusSel(){
  setTimeInput('#musStartV',musStartSel,'playhead');
  setTimeInput('#musEndV',musEndSel,'to end');
}
$('#musAdd').onclick=async()=>{
  const m=musicSelection();
  if(!m){ setStatus('Choose, upload, or type a music file first.','err'); return; }
  const op=startOp('music','Adding music track',$('#musAdd'));
  const rr=rangeOr(musStartSel,musEndSel);
  const start=rr.start;
  const end=(rr.end!=null&&rr.end>start)?rr.end:null;
  try{
    advanceOp(op, 'Probing selected music', 38);
    let dur=0; try{ dur=(await api('api/probe?ref='+encodeURIComponent(m.ref))).duration; }catch(_){}
    pushUndo();
    musics.push({ref:m.ref, label:m.label, start, end, duration:dur});
    musics.sort((a,b)=>a.start-b.start);
    musStartSel=musEndSel=null; updateMusSel(); renderMusic(); draw();
    finishOp(true, 'Music added at '+fmt(start), 'ok');
  }catch(e){ finishOp(false, 'Music add failed: '+e.message, 'err'); }
};
function renderMusic(){
  const el=$('#musList'); el.innerHTML='';
  musics.forEach((m,i)=>{
    const end=(m.end!=null&&m.end>m.start)?fmt(m.end).replace(/^00:/,''):'end';
    const d=document.createElement('div'); d.className='item';
    if(isSelectedTimelineItem('mus',i)) d.classList.add('sel');
    d.innerHTML=`<span class="pill" style="background:#1d3050;color:#cfe0ff">`+
      `${fmt(m.start).replace(/^00:/,'')}→${end}</span>`+
      `<span class="grow" title="${m.label.replace(/"/g,'')}">${m.label}</span>`+
      `<span class="x">✕</span>`;
    d.querySelector('.x').onclick=()=>{pushUndo();musics.splice(i,1);renderMusic();draw();};
    el.appendChild(d);
  });
  scheduleSessionSave();
}

// ---------- sliders ----------
const bind=(id,vid,suf)=>{const e=$(id);const f=()=>{$(vid).textContent=e.value+suf;draw();scheduleSessionSave();};e.oninput=f;f();};
let volumeLabelsBound=false;
function bindVolumeLabels(){
  $('#musVolV').textContent=$('#musVol').value+'%';
  $('#baseVolV').textContent=$('#baseVol').value+'%';
  $('#ttsVolV').textContent=$('#ttsVol').value+'%';
}
bind('#musVol','#musVolV','%'); bind('#baseVol','#baseVolV','%'); bind('#ttsVol','#ttsVolV','%');
$('#capBurn').onchange=scheduleSessionSave;
$('#ttsAutoCap').onchange=scheduleSessionSave;

// ---------- render ----------
$('#saveBtn').onclick=async()=>{
  const btn=$('#saveBtn');
  if(!clips.length){ setStatus('Add at least one video first.','err'); return; }
  const op=startOp('render','Rendering final video',btn);
  const body={
    videoClips:clips.map(c=>({name:c.name, offset:c.offset, duration:c.duration, trimStart:(c.srcStart||0)})),
    ttsClips:tts.map(c=>({time:c.time, end:(c.end!=null?c.end:null), file:c.file})),
    speedRegions:speeds.map(s=>({start:s.start,end:s.end,factor:s.factor})),
    cutRegions:[],
    musics:musics.map(m=>({ref:m.ref, start:m.start, end:(m.end!=null?m.end:null)})),
    captions:caps.map(c=>({start:c.start,end:c.end,text:c.text})),
    burnCaptions:$('#capBurn').checked,
    baseVolume:(+$('#baseVol').value)/100,
    ttsVolume:(+$('#ttsVol').value)/100,
    musicVolume:(+$('#musVol').value)/100,
  };
  try{
    advanceOp(op, 'Sending render job to ffmpeg', 18);
    const r=await api('api/render',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    finishOp(true, '', 'ok');
    $('#status').innerHTML=`✓ Saved to <code>${r.output}</code> · `+
      `<a class="dl" href="${r.url}" download>download</a>`;
  }catch(e){ finishOp(false, 'Render failed: '+e.message, 'err'); }
};
function setStatus(msg,cls){const s=$('#status');s.textContent=msg;
  s.className='status'+(cls?' '+cls:'');}
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Browser video editor: AI voice-over, speed regions, music, render."
    )
    # Overridable settings default to None so we can tell whether the user
    # passed them explicitly (CLI wins over the config file).
    parser.add_argument("video", nargs="?", default=None,
                        help="Video to preselect (path relative to the media dir)")
    parser.add_argument("-c", "--config", default=None,
                        help="Load settings from a JSON config file (saved from "
                             "the web UI). CLI flags override the file. "
                             f"When omitted, ./{DEFAULT_CONFIG_NAME} is auto-loaded "
                             "if present (and is the default Save target).")
    parser.add_argument("--media-dir", default=None,
                        help="Directory of videos/audio to browse (default: cwd)")
    parser.add_argument("--output-dir", default=None,
                        help="Where rendered files are written")
    parser.add_argument("--voice", choices=["feminine", "masculine"], default=None,
                        help="Default TTS voice gender")
    parser.add_argument("--voice-name", default=None,
                        help="Explicit Kokoro voice id (overrides --voice default)")
    parser.add_argument("--tts-model", default=None,
                        help="CoderAI TTS model id (auto-detected if omitted)")
    parser.add_argument("--stt-model", default=None,
                        help="CoderAI transcription model id for optional "
                             "auto-captioning of existing dialogue "
                             "(auto-detected if omitted; disabled if none)")
    parser.add_argument("--audio-model", default=None,
                        help="CoderAI MusicGen/audio generation model id "
                             "(auto-detected if omitted; disabled if none)")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8420)
    parser.add_argument("--no-browser", action="store_true",
                        help="Do not auto-open a browser tab")
    parser.add_argument("--session", nargs="?", const="default", default=None,
                        help="Enable realtime editor-state recovery, optionally named")
    args = parser.parse_args()

    # When -c is omitted, auto-load (and later save back to) a default config in
    # the current directory, so settings configured in the UI survive a restart
    # without the user having to remember to pass --config.
    if not args.config and Path(DEFAULT_CONFIG_NAME).is_file():
        args.config = DEFAULT_CONFIG_NAME
        print(f"ℹ  Loading settings from ./{DEFAULT_CONFIG_NAME} "
              "(pass --config to use a different file).", file=sys.stderr)

    require_binary("ffmpeg")
    require_binary("ffprobe")

    # Merge: built-in defaults < config file < explicit CLI flags / env.
    settings = {
        "media_dir": os.getcwd(),
        "output_dir": "video_editor_output",
        "base_url": os.environ.get("CODERAI_BASE_URL", "http://127.0.0.1:8000"),
        "api_key": os.environ.get("CODERAI_API_KEY"),
        "voice": "feminine",
        "voice_name": None,
        "tts_model": None,
        "stt_model": None,
        "audio_model": None,
        "video": None,
    }
    if args.config:
        cfg_path = Path(args.config).expanduser()
        if not cfg_path.is_file():
            raise SystemExit(f"--config file not found: {cfg_path}")
        try:
            loaded = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            raise SystemExit(f"Could not read config {cfg_path}: {e}")
        for k in SETTINGS_KEYS:
            if loaded.get(k) is not None:
                settings[k] = loaded[k]
    cli = {
        "media_dir": args.media_dir, "output_dir": args.output_dir,
        "base_url": args.base_url, "api_key": args.api_key,
        "voice": args.voice, "voice_name": args.voice_name,
        "tts_model": args.tts_model, "stt_model": args.stt_model,
        "audio_model": args.audio_model,
        "video": args.video,
    }
    for k, v in cli.items():
        if v is not None:
            settings[k] = v

    if not Path(settings["media_dir"]).expanduser().is_dir():
        raise SystemExit(f"Media directory not found: {settings['media_dir']}")

    # Auto-detect models only when not pinned by config/CLI.
    models = _list_models(settings["base_url"].rstrip("/"), settings.get("api_key"))
    if not settings["tts_model"]:
        settings["tts_model"] = detect_model(models, ("kokoro", "tts"))
    if not settings["stt_model"]:
        settings["stt_model"] = detect_model(models, ("whisper", "stt", "asr"))
    if not settings["audio_model"]:
        settings["audio_model"] = detect_model(
            models, ("musicgen", "audiogen", "audioldm", "audio_generation", "text-to-audio")
        )
    if not settings["tts_model"]:
        print("⚠  No TTS model found on the server. Voice-over will fail until you "
              "set one (--tts-model, a config file, or the web Settings panel).",
              file=sys.stderr)
    if not settings["stt_model"]:
        print("ℹ  No transcription model found — optional auto-caption of existing "
              "dialogue is disabled (captions from your voice-over text still work).",
              file=sys.stderr)
    if not settings["audio_model"]:
        print("ℹ  No audio generation model found — MusicGen music creation is disabled "
              "until you set one (--audio-model, config file, or Settings panel).",
              file=sys.stderr)

    cfg = Config(settings, config_path=args.config, session_name=args.session)
    tts_model, stt_model, audio_model = cfg.tts_model, cfg.stt_model, cfg.audio_model
    editor = Editor(cfg)
    handler = make_handler(editor)
    httpd = ThreadingHTTPServer((args.host, args.port), handler)

    url = f"http://{args.host}:{args.port}/"
    print(f"CoderAI Video Editor running at {url}")
    print(f"  media dir : {cfg.media_dir}")
    print(f"  output dir: {cfg.output_dir}")
    print(f"  TTS model : {tts_model or '(none)'}  voice: {cfg.default_voice} ({cfg.gender})")
    print(f"  STT model : {stt_model or '(none, auto-caption disabled)'}")
    print(f"  music gen : {audio_model or '(none, music generation disabled)'}")
    if cfg.config_path:
        print(f"  config    : {cfg.config_path}")
    if cfg.session_enabled:
        print(f"  session   : {cfg.session_name} ({cfg.session_path})")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        httpd.shutdown()
        shutil.rmtree(cfg.tts_dir, ignore_errors=True)
        shutil.rmtree(cfg.upload_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
