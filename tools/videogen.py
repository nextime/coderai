#!/usr/bin/env python3
"""
CoderAI VideoGen Studio

A small local web app for managing CoderAI character/environment profiles and
assembling multi-clip short movies with video generation, speech/lip-sync,
music, and sound effects.

The profile workflow intentionally mirrors tools/gen_township_fighters.py:
- local output has characters/<name>/meta.json + ref_XX.png
- local output has environments/<name>/meta.json + ref_XX.png
- generated profiles are created through /v1/characters/generate and
  /v1/environments/generate, fetched back from CoderAI, and saved locally
- local profiles can be uploaded/re-synced to CoderAI when reused

Run:
  python tools/videogen.py --base-url http://127.0.0.1:8776 --web-port 7790

Open:
  http://127.0.0.1:7790

Requirements:
  - CoderAI running
  - requests
  - ffmpeg/ffprobe on PATH for final assembly
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import mimetypes
import os
import queue
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any

try:
    import requests
except ImportError as exc:  # pragma: no cover
    raise SystemExit("This script requires requests: pip install requests") from exc

sys.stdout.reconfigure(line_buffering=True)

DEFAULT_BASE_URL = os.environ.get("CODERAI_BASE_URL", "http://127.0.0.1:8776")
DEFAULT_API_KEY = os.environ.get("CODERAI_API_KEY")
DEFAULT_OUT_DIR = "./videogen_output"


def log(*parts: Any) -> None:
    print(*parts, flush=True)


def safe_slug(value: str) -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "_", value)
    return value.strip("_-. ") or f"item_{uuid.uuid4().hex[:8]}"


def load_json_map(path: Path) -> dict[str, Any]:
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}


def session_token(out_dir: Path) -> str:
    path = out_dir / ".train_session"
    try:
        if path.exists():
            token = path.read_text(encoding="utf-8").strip()
            if token:
                return token
    except Exception:
        pass
    token = f"videogen-{uuid.uuid4().hex[:12]}"
    try:
        path.write_text(token, encoding="utf-8")
    except Exception:
        pass
    return token


def lora_jobs_file(out_dir: Path) -> Path:
    return out_dir / "lora_train_jobs.json"


def get_lora_job_id(out_dir: Path, lora_name: str) -> str | None:
    return load_json_map(lora_jobs_file(out_dir)).get(lora_name)


def set_lora_job_id(out_dir: Path, lora_name: str, job_id: str | None) -> None:
    path = lora_jobs_file(out_dir)
    data = load_json_map(path)
    if job_id:
        data[lora_name] = job_id
    else:
        data.pop(lora_name, None)
    try:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


def data_uri_for_file(path: Path, mime: str | None = None) -> str:
    if mime is None:
        mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def decode_data_uri(value: str) -> bytes:
    if value.startswith("data:"):
        value = value.split(",", 1)[1]
    return base64.b64decode(value)


def ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    raise RuntimeError("ffmpeg not found on PATH")


def ffprobe() -> str | None:
    return shutil.which("ffprobe")


def run_cmd(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{proc.stderr[-1200:]}")


def video_duration(path: Path) -> float:
    probe = ffprobe()
    if not probe:
        return 0.0
    proc = subprocess.run(
        [probe, "-v", "quiet", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        return float(proc.stdout.strip())
    except Exception:
        return 0.0


def concat_videos(paths: list[Path], out_path: Path) -> None:
    if not paths:
        raise RuntimeError("No clips to concatenate")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if len(paths) == 1:
        shutil.copy(paths[0], out_path)
        return
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        list_path = Path(handle.name)
        for path in paths:
            handle.write(f"file '{path.resolve()}'\n")
    try:
        # Re-encode to tolerate model outputs with slightly different stream metadata.
        run_cmd([
            ffmpeg(), "-y", "-f", "concat", "-safe", "0", "-i", str(list_path),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out_path),
        ])
    finally:
        try:
            list_path.unlink()
        except Exception:
            pass


def mux_background_audio(video_path: Path, audio_path: Path, out_path: Path, music_volume: float = 0.25) -> None:
    # Mix existing clip audio with generated background music/sfx.
    run_cmd([
        ffmpeg(), "-y", "-i", str(video_path), "-i", str(audio_path),
        "-filter_complex", f"[1:a]volume={music_volume}[bg];[0:a][bg]amix=inputs=2:duration=first:dropout_transition=2[a]",
        "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-shortest", str(out_path),
    ])


def extract_video_frames(video_path: Path, out_dir: Path, max_frames: int = 6) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = max(0.1, video_duration(video_path))
    count = max(1, int(max_frames))
    frames = []
    for idx in range(count):
        ts = duration * (idx + 0.5) / count
        out = out_dir / f"{video_path.stem}_frame_{idx:02d}.png"
        run_cmd([ffmpeg(), "-y", "-ss", f"{ts:.3f}", "-i", str(video_path), "-frames:v", "1", str(out)])
        if out.exists() and out.stat().st_size:
            frames.append(out)
    return frames


class CoderAIClient:
    def __init__(self, base_url: str, api_key: str | None = None, timeout: int = 7200):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"

    def _get(self, path: str, timeout: int = 60) -> dict[str, Any]:
        resp = self.session.get(f"{self.base}{path}", timeout=timeout)
        if not resp.ok:
            raise RuntimeError(f"GET {path} -> {resp.status_code}: {resp.text[:800]}")
        return resp.json()

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        resp = self.session.post(f"{self.base}{path}", json=body, timeout=self.timeout)
        if not resp.ok:
            raise RuntimeError(f"POST {path} -> {resp.status_code}: {resp.text[:1200]}")
        return resp.json()

    def _post_multipart(self, path: str, data: dict[str, Any], files: dict[str, Any]) -> dict[str, Any]:
        resp = self.session.post(f"{self.base}{path}", data=data, files=files, timeout=self.timeout)
        if not resp.ok:
            raise RuntimeError(f"POST {path} -> {resp.status_code}: {resp.text[:1200]}")
        return resp.json()

    def _patch(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        resp = self.session.patch(f"{self.base}{path}", json=body, timeout=self.timeout)
        if not resp.ok:
            raise RuntimeError(f"PATCH {path} -> {resp.status_code}: {resp.text[:800]}")
        return resp.json()

    def _delete(self, path: str) -> dict[str, Any]:
        resp = self.session.delete(f"{self.base}{path}", timeout=120)
        if not resp.ok:
            raise RuntimeError(f"DELETE {path} -> {resp.status_code}: {resp.text[:800]}")
        return resp.json()

    def list_models(self) -> list[dict[str, Any]]:
        return self._get("/v1/models").get("data", [])

    def list_by_capability(self, *caps: str) -> list[dict[str, Any]]:
        wanted = set(caps)
        out = []
        for model in self.list_models():
            got = set(model.get("capabilities") or [])
            if got.intersection(wanted):
                out.append(model)
        return out

    def list_characters(self) -> list[dict[str, Any]]:
        try:
            return self._get("/v1/characters").get("characters", [])
        except Exception:
            return []

    def list_environments(self) -> list[dict[str, Any]]:
        try:
            return self._get("/v1/environments").get("environments", [])
        except Exception:
            return []

    def list_voices(self) -> list[dict[str, Any]]:
        try:
            return self._get("/v1/audio/voices").get("voices", [])
        except Exception:
            return []

    def list_loras(self) -> list[dict[str, Any]]:
        try:
            return self._get("/v1/loras").get("loras", [])
        except Exception:
            return []

    def create_voice(self, name: str, description: str, transcript: str, audio_path: Path) -> dict[str, Any]:
        mime = mimetypes.guess_type(audio_path.name)[0] or "audio/wav"
        with audio_path.open("rb") as handle:
            return self._post_multipart(
                "/v1/audio/voices",
                {"name": name, "description": description, "transcript": transcript},
                {"audio": (audio_path.name, handle, mime)},
            )

    def extract_voice(self, name: str, description: str, transcript: str, media_path: Path) -> dict[str, Any]:
        mime = mimetypes.guess_type(media_path.name)[0] or "application/octet-stream"
        key = "video" if mime.startswith("video/") or media_path.suffix.lower() in {".mp4", ".mov", ".webm", ".mkv"} else "audio"
        return self._post("/v1/audio/voices/extract", {
            "name": name,
            "description": description,
            "transcript": transcript,
            key: data_uri_for_file(media_path, mime),
        })

    def train_lora(self, body: dict[str, Any]) -> dict[str, Any]:
        return self._post("/v1/loras/train", body)

    def lora_progress(self, job: str | None = None, session: str | None = None) -> dict[str, Any]:
        query = ""
        if job:
            query = f"?job={urllib.parse.quote(job)}"
        elif session:
            query = f"?session={urllib.parse.quote(session)}"
        try:
            return self._get(f"/v1/loras/progress{query}", timeout=30)
        except Exception:
            return {}

    def get_profile_images(self, kind: str, name: str) -> list[str]:
        plural = "characters" if kind == "character" else "environments"
        try:
            data = self._get(f"/v1/{plural}/{urllib.parse.quote(name)}")
            return [img.get("data", "") for img in data.get("images", []) if img.get("data")]
        except Exception:
            return []

    def create_profile(self, kind: str, name: str, description: str, prompt: str, model: str, n: int, width: int, height: int) -> dict[str, Any]:
        if kind == "character":
            path = "/v1/characters/generate"
        else:
            path = "/v1/environments/generate"
        return self._post(path, {
            "name": name,
            "description": description,
            "prompt": prompt,
            "model": model,
            "n": int(n),
            "width": int(width),
            "height": int(height),
        })

    def save_profile(self, kind: str, name: str, description: str, images: list[str]) -> dict[str, Any]:
        plural = "characters" if kind == "character" else "environments"
        return self._post(f"/v1/{plural}", {
            "name": name,
            "description": description,
            "images": [{"label": f"ref_{i:02d}", "data": im} for i, im in enumerate(images)],
        })

    def patch_profile(self, kind: str, name: str, description: str | None = None, add_images: list[str] | None = None) -> dict[str, Any]:
        plural = "characters" if kind == "character" else "environments"
        body: dict[str, Any] = {}
        if description is not None:
            body["description"] = description
        if add_images:
            body["add_images"] = [{"label": f"added_{i:02d}", "data": im} for i, im in enumerate(add_images)]
        return self._patch(f"/v1/{plural}/{urllib.parse.quote(name)}", body)

    def generate_image(self, body: dict[str, Any]) -> bytes:
        body = dict(body)
        body["response_format"] = "b64_json"
        data = self._post("/v1/images/generations", body)
        item = (data.get("data") or [{}])[0]
        raw = item.get("b64_json") or item.get("url") or ""
        return self._bytes_from_api_value(raw)

    def generate_video(self, body: dict[str, Any]) -> bytes:
        body = dict(body)
        body["response_format"] = "b64_mp4"
        data = self._post("/v1/video/generations", body)
        item = (data.get("data") or [{}])[0]
        raw = item.get("b64_mp4") or item.get("url") or ""
        return self._bytes_from_api_value(raw)

    def generate_audio(self, body: dict[str, Any]) -> bytes:
        body = dict(body)
        body["response_format"] = "b64_wav"
        data = self._post("/v1/audio/generate", body)
        item = (data.get("data") or [{}])[0]
        raw = item.get("b64_wav") or item.get("b64_mp3") or item.get("url") or ""
        return self._bytes_from_api_value(raw)

    def speech(self, model: str, text: str, voice: str, speed: float = 1.0) -> bytes:
        data = self._post("/v1/audio/speech", {
            "model": model,
            "input": text,
            "voice": voice,
            "speed": speed,
            "response_format": "wav",
        })
        raw = data.get("audio") or ""
        return self._bytes_from_api_value(raw)

    def _bytes_from_api_value(self, raw: str) -> bytes:
        if not raw:
            raise RuntimeError("API response did not contain media data")
        if raw.startswith("data:"):
            return decode_data_uri(raw)
        if raw.startswith("http://") or raw.startswith("https://"):
            with self.session.get(raw, timeout=self.timeout) as resp:
                resp.raise_for_status()
                return resp.content
        if raw.startswith("/v1/"):
            with self.session.get(f"{self.base}{raw}", timeout=self.timeout) as resp:
                resp.raise_for_status()
                return resp.content
        return base64.b64decode(raw)


class ProfileStore:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir

    def profile_dir(self, kind: str, name: str) -> Path:
        return self.out_dir / (kind + "s") / safe_slug(name)

    def save(self, kind: str, name: str, meta: dict[str, Any], images_b64: list[str]) -> Path:
        d = self.profile_dir(kind, name)
        d.mkdir(parents=True, exist_ok=True)
        clean_meta = dict(meta)
        clean_meta.setdefault("name", name)
        clean_meta.setdefault("created_at", int(time.time()))
        (d / "meta.json").write_text(json.dumps(clean_meta, indent=2), encoding="utf-8")
        for i, raw in enumerate(images_b64):
            if not raw:
                continue
            ext = ".png"
            try:
                payload = decode_data_uri(raw)
                if payload[:3] == b"\xff\xd8\xff":
                    ext = ".jpg"
                (d / f"ref_{i:02d}{ext}").write_bytes(payload)
            except Exception:
                pass
        return d

    def list_local(self, kind: str) -> list[dict[str, Any]]:
        base = self.out_dir / (kind + "s")
        if not base.exists():
            return []
        out = []
        for d in sorted(base.iterdir()):
            if not d.is_dir():
                continue
            meta_path = d / "meta.json"
            refs = sorted(list(d.glob("ref_*.png")) + list(d.glob("ref_*.jpg")) + list(d.glob("ref_*.jpeg")) + list(d.glob("ref_*.webp")))
            if not meta_path.exists() and not refs:
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
            except Exception:
                meta = {}
            name = meta.get("name") or d.name
            thumb = f"/media/{kind}s/{urllib.parse.quote(d.name)}/{urllib.parse.quote(refs[0].name)}" if refs else ""
            out.append({
                "name": name,
                "slug": d.name,
                "description": meta.get("description", ""),
                "prompt": meta.get("prompt", ""),
                "image_count": len(refs),
                "thumbnail": thumb,
                "local": True,
            })
        return out

    def load_images(self, kind: str, name: str) -> list[str]:
        d = self.profile_dir(kind, name)
        refs = sorted(list(d.glob("ref_*.png")) + list(d.glob("ref_*.jpg")) + list(d.glob("ref_*.jpeg")) + list(d.glob("ref_*.webp")))
        images = []
        for path in refs:
            images.append(data_uri_for_file(path, mimetypes.guess_type(path.name)[0] or "image/png"))
        return images

    def load_meta(self, kind: str, name: str) -> dict[str, Any]:
        p = self.profile_dir(kind, name) / "meta.json"
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}


def pick_model(models: list[dict[str, Any]], cap: str, override: str | None = None) -> str:
    if override:
        return override
    for model in models:
        if cap in (model.get("capabilities") or []):
            return model.get("id")
    return ""


class VideoGenApp:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.out_dir = Path(args.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "movies").mkdir(exist_ok=True)
        self.client = CoderAIClient(args.base_url, args.api_key)
        self.store = ProfileStore(self.out_dir)
        self.log_lines: list[str] = []
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.jobs: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()

    def emit(self, line: str) -> None:
        ts = time.strftime("%H:%M:%S")
        msg = f"[{ts}] {line}"
        log(msg)
        with self.lock:
            self.log_lines.append(msg)
            self.log_lines = self.log_lines[-1000:]
        self.log_queue.put(msg)

    def models_payload(self) -> dict[str, Any]:
        models = self.client.list_models()
        return {
            "models": models,
            "defaults": {
                "image_model": self.args.image_model or pick_model(models, "image_generation"),
                "video_model": self.args.video_model or pick_model(models, "video_generation"),
                "audio_model": self.args.audio_model or pick_model(models, "audio_generation"),
                "tts_model": self.args.tts_model or pick_model(models, "text_to_speech"),
            },
        }

    def profiles_payload(self) -> dict[str, Any]:
        chars = {p["name"]: p for p in self.store.list_local("character")}
        envs = {p["name"]: p for p in self.store.list_local("environment")}
        for item in self.client.list_characters():
            chars.setdefault(item.get("name", ""), {**item, "local": False})
        for item in self.client.list_environments():
            envs.setdefault(item.get("name", ""), {**item, "local": False})
        return {
            "characters": [v for k, v in sorted(chars.items()) if k],
            "environments": [v for k, v in sorted(envs.items()) if k],
            "voices": self.client.list_voices(),
            "loras": self.client.list_loras(),
        }

    def start_lora_job(self, payload: dict[str, Any]) -> str:
        job_id = f"lora-{uuid.uuid4().hex[:10]}"
        with self.lock:
            self.jobs[job_id] = {"status": "queued", "progress": 0, "kind": "lora"}
        thread = threading.Thread(target=self._lora_job, args=(job_id, payload), daemon=True)
        thread.start()
        return job_id

    def _lora_job(self, job_id: str, payload: dict[str, Any]) -> None:
        name = safe_slug(payload.get("name") or "movie_style")
        base_model = payload.get("base_model") or self.args.video_model or pick_model(self.client.list_models(), "video_generation")
        target = payload.get("target") or "video"
        media_paths = [Path(p.strip()) for p in (payload.get("media_paths") or "").splitlines() if p.strip()]
        style_prompt = payload.get("style_prompt") or f"{name} cinematic style"
        instance_prompt = payload.get("instance_prompt") or f"in {name} style, {style_prompt}"
        frames_per_video = int(payload.get("frames_per_video") or 6)
        try:
            if not base_model:
                raise RuntimeError("No base model configured for LoRA training")
            self._job_update(job_id, status="running", progress=5, message="collecting training media")
            train_dir = self.out_dir / "lora_training" / name
            frame_dir = train_dir / "frames"
            train_dir.mkdir(parents=True, exist_ok=True)
            images: list[str] = []
            scene_manifest = []
            for idx, path in enumerate(media_paths):
                if not path.exists() or not path.is_file():
                    self.emit(f"Skipping missing LoRA media: {path}")
                    continue
                ext = path.suffix.lower()
                scene_manifest.append({"path": str(path), "description": payload.get("scene_description") or ""})
                if ext in {".png", ".jpg", ".jpeg", ".webp"}:
                    images.append(data_uri_for_file(path, mimetypes.guess_type(path.name)[0] or "image/png"))
                elif ext in {".mp4", ".mov", ".webm", ".mkv"}:
                    self.emit(f"Extracting frames from {path.name}")
                    for frame in extract_video_frames(path, frame_dir / safe_slug(path.stem), frames_per_video):
                        images.append(data_uri_for_file(frame, "image/png"))
            if not images:
                raise RuntimeError("No image frames found. Add image files or videos to train from.")
            (train_dir / "manifest.json").write_text(json.dumps({"name": name, "style_prompt": style_prompt, "scenes": scene_manifest}, indent=2), encoding="utf-8")
            self.emit(f"Training {target} LoRA '{name}' from {len(images)} image/frame sample(s)")
            self._job_update(job_id, progress=25, message="training LoRA")
            train_body = {
                "name": name,
                "base_model": base_model,
                "train_base_model": payload.get("train_base_model") or None,
                "target": target,
                "images": images,
                "instance_prompt": instance_prompt,
                "steps": int(payload.get("steps") or 800),
                "rank": int(payload.get("rank") or 16),
                "learning_rate": float(payload.get("learning_rate") or 1e-4),
                "resolution": int(payload.get("resolution") or 512),
                "seed": int(payload.get("seed") or 42),
                "num_frames": int(payload.get("num_frames") or 1),
                "quantize_4bit": bool(payload.get("quantize_4bit", True)),
                "wait": False,
                "session": session_token(self.out_dir),
            }
            path = self._attach_or_start_lora(job_id, name, train_body)
            self.emit(f"LoRA ready: {name}" + (f" -> {path}" if path else ""))
            set_lora_job_id(self.out_dir, name, None)
            self._job_update(job_id, status="done", progress=100, message="done", name=name, path=path)
        except Exception as exc:
            self.emit(f"LoRA training failed: {exc}")
            self._job_update(job_id, status="error", error=str(exc), message=str(exc))

    def _attach_or_start_lora(self, ui_job_id: str, lora_name: str, train_body: dict[str, Any]) -> str | None:
        active_states = {"queued", "preparing", "training", "saving"}

        def kickoff() -> str:
            resp = self.client.train_lora({k: v for k, v in train_body.items() if v is not None})
            server_job = resp.get("job_id")
            if not server_job:
                raise RuntimeError(f"LoRA training did not return a job_id: {resp}")
            set_lora_job_id(self.out_dir, lora_name, server_job)
            return server_job

        server_job = get_lora_job_id(self.out_dir, lora_name)
        if server_job:
            progress = self.client.lora_progress(job=server_job)
            status = (progress.get("status") or "").strip()
            if status == "done" and progress.get("path"):
                self.emit(f"Re-attached: LoRA '{lora_name}' was already trained")
                return progress.get("path")
            if status in active_states:
                self.emit(f"Re-attached to running LoRA job for '{lora_name}'")
            else:
                self.emit(f"Previous LoRA job for '{lora_name}' was '{status or 'lost'}'; resubmitting to resume from checkpoint")
                server_job = kickoff()
        else:
            server_job = kickoff()

        started = time.time()
        resubmits = 0
        while True:
            if self._is_cancelled(ui_job_id):
                raise RuntimeError("cancelled")
            time.sleep(2.0)
            elapsed = int(time.time() - started)
            mm, ss = divmod(elapsed, 60)
            et = f"{mm}m{ss:02d}s" if mm else f"{ss}s"
            progress = self.client.lora_progress(job=server_job)
            status = (progress.get("status") or "").strip()
            if status == "done":
                return progress.get("path")
            if status == "error":
                raise RuntimeError(progress.get("message") or "LoRA training failed")
            if status in {"interrupted", "unknown", ""}:
                if resubmits < 2:
                    resubmits += 1
                    self.emit(f"LoRA job '{status or 'unknown'}' for '{lora_name}'; resubmitting resume #{resubmits}")
                    server_job = kickoff()
                    self._job_update(ui_job_id, progress=30, message=f"resuming after interruption ({et})")
                    continue
                raise RuntimeError(f"training {status or 'unknown'} and could not be resumed")
            total = progress.get("total") or train_body.get("steps") or 1
            step = progress.get("step") or 0
            if status == "queued":
                pct = 25
                msg = f"queued - waiting for GPU ({et})"
            elif status in {"preparing", "saving"} or not step:
                pct = 30 if status != "saving" else 94
                msg = f"{progress.get('message') or status or 'preparing'} ({et})"
            else:
                pct = 30 + int(64 * step / max(1, total))
                msg = progress.get("message") or f"training {step}/{total} ({et})"
            self._job_update(ui_job_id, progress=min(98, pct), message=msg, server_job=server_job)

    def start_voice_job(self, payload: dict[str, Any]) -> str:
        job_id = f"voice-{uuid.uuid4().hex[:10]}"
        with self.lock:
            self.jobs[job_id] = {"status": "queued", "progress": 0, "kind": "voice"}
        thread = threading.Thread(target=self._voice_job, args=(job_id, payload), daemon=True)
        thread.start()
        return job_id

    def _voice_job(self, job_id: str, payload: dict[str, Any]) -> None:
        name = safe_slug(payload.get("name") or "voice")
        description = payload.get("description") or ""
        transcript = payload.get("transcript") or ""
        media_path = Path(payload.get("media_path") or "")
        try:
            self._job_update(job_id, status="running", progress=5, message="validating reference media")
            if not media_path.exists() or not media_path.is_file():
                raise RuntimeError(f"Reference audio/video file not found: {media_path}")
            self.emit(f"Creating cloned voice profile '{name}' from {media_path.name}")
            self._job_update(job_id, progress=25, message="uploading reference media")
            if payload.get("extract", True):
                self.client.extract_voice(name, description, transcript, media_path)
            else:
                self.client.create_voice(name, description, transcript, media_path)
            self.emit(f"Voice profile ready: {name}")
            self._job_update(job_id, status="done", progress=100, message="done", name=name)
        except Exception as exc:
            self.emit(f"Voice job failed: {exc}")
            self._job_update(job_id, status="error", error=str(exc), message=str(exc))

    def start_profile_job(self, payload: dict[str, Any]) -> str:
        job_id = f"profile-{uuid.uuid4().hex[:10]}"
        with self.lock:
            self.jobs[job_id] = {"status": "queued", "progress": 0, "kind": payload.get("kind")}
        thread = threading.Thread(target=self._profile_job, args=(job_id, payload), daemon=True)
        thread.start()
        return job_id

    def _profile_job(self, job_id: str, payload: dict[str, Any]) -> None:
        kind = payload.get("kind") or "character"
        name = payload.get("name") or safe_slug(payload.get("description") or kind)
        name = safe_slug(name)
        description = payload.get("description") or ""
        prompt = payload.get("prompt") or description or name
        model = payload.get("model") or self.args.image_model
        n = int(payload.get("n") or (4 if kind == "character" else 3))
        width = int(payload.get("width") or (512 if kind == "character" else 768))
        height = int(payload.get("height") or 512)
        try:
            self._job_update(job_id, status="running", progress=5, message="starting")
            if payload.get("upload_only"):
                self.emit(f"Syncing local {kind} '{name}' to CoderAI")
                images = self.store.load_images(kind, name)
                if not images:
                    raise RuntimeError(f"No local images found for {kind} {name}")
                self.client.save_profile(kind, name, description or self.store.load_meta(kind, name).get("description", ""), images)
            else:
                if not model:
                    model = pick_model(self.client.list_models(), "image_generation")
                if not model:
                    raise RuntimeError("No image model configured or detected")
                self.emit(f"Generating {kind} '{name}' with {model} ({n} refs)")
                self._job_update(job_id, progress=15, message="generating references")
                self.client.create_profile(kind, name, description, prompt, model, n, width, height)
                self._job_update(job_id, progress=80, message="fetching profile images")
                images = self.client.get_profile_images(kind, name)
                if images:
                    self.store.save(kind, name, {
                        "name": name,
                        "description": description,
                        "prompt": prompt,
                        "model": model,
                        "kind": kind,
                    }, images)
            self.emit(f"Profile ready: {kind} '{name}'")
            self._job_update(job_id, status="done", progress=100, message="done", name=name)
        except Exception as exc:
            self.emit(f"Profile job failed: {exc}")
            self._job_update(job_id, status="error", error=str(exc), message=str(exc))

    def start_movie_job(self, payload: dict[str, Any]) -> str:
        job_id = f"movie-{uuid.uuid4().hex[:10]}"
        with self.lock:
            self.jobs[job_id] = {"status": "queued", "progress": 0, "movie": payload.get("title") or "movie"}
        target = self._movie_batch_job if int(payload.get("movie_count") or 1) > 1 else self._movie_job
        thread = threading.Thread(target=target, args=(job_id, payload), daemon=True)
        thread.start()
        return job_id

    def _movie_batch_job(self, job_id: str, payload: dict[str, Any]) -> None:
        count = max(1, int(payload.get("movie_count") or 1))
        outputs = []
        base_title = payload.get("title") or "untitled_movie"
        for idx in range(count):
            if self._is_cancelled(job_id):
                self._job_update(job_id, status="error", error="cancelled", message="cancelled")
                return
            variant = dict(payload)
            variant["movie_count"] = 1
            variant["title"] = f"{base_title}_variant_{idx + 1:02d}"
            self.emit(f"Rendering movie variant {idx + 1}/{count}")
            self._movie_job(job_id, variant)
            with self.lock:
                job = dict(self.jobs.get(job_id, {}))
            if job.get("status") == "error":
                return
            if job.get("output_url"):
                outputs.append(job["output_url"])
        self._job_update(job_id, status="done", progress=100, message=f"rendered {len(outputs)} movie variant(s)", outputs=outputs, output_url=outputs[-1] if outputs else None)

    def _movie_job(self, job_id: str, payload: dict[str, Any]) -> None:
        title = payload.get("title") or "untitled_movie"
        slug = safe_slug(title)
        movie_dir = self.out_dir / "movies" / f"{time.strftime('%Y%m%d_%H%M%S')}_{slug}"
        clip_dir = movie_dir / "clips"
        clip_dir.mkdir(parents=True, exist_ok=True)
        spec_path = movie_dir / "movie_spec.json"
        spec_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        clips = payload.get("clips") or []
        if not clips:
            self._job_update(job_id, status="error", error="No clips supplied")
            return
        try:
            models = self.client.list_models()
            video_model = payload.get("video_model") or self.args.video_model or pick_model(models, "video_generation")
            image_model = payload.get("image_model") or self.args.image_model or pick_model(models, "image_generation")
            audio_model = payload.get("audio_model") or self.args.audio_model or pick_model(models, "audio_generation")
            if not video_model:
                raise RuntimeError("No video model configured or detected")
            fps = int(payload.get("fps") or 8)
            width = int(payload.get("width") or 768)
            height = int(payload.get("height") or 432)
            default_frames = int(payload.get("num_frames") or 32)
            use_keyframes = bool(payload.get("use_keyframes"))
            selected_loras = self._selected_loras(payload)
            clip_paths: list[Path] = []
            total = len(clips)
            self.emit(f"Starting movie '{title}' with {total} clip(s)")
            for i, clip in enumerate(clips):
                if self._is_cancelled(job_id):
                    raise RuntimeError("cancelled")
                label = clip.get("title") or f"clip {i+1}"
                self.emit(f"Clip {i+1}/{total}: {label}")
                self._job_update(job_id, status="running", progress=int(5 + 80 * i / max(1, total)), message=f"rendering {label}")
                characters = [c for c in clip.get("characters", []) if c]
                environments = [e for e in clip.get("environments", []) if e]
                prompt = self._build_clip_prompt(payload, clip, characters, environments)
                body: dict[str, Any] = {
                    "model": video_model,
                    "prompt": prompt,
                    "negative_prompt": clip.get("negative_prompt") or payload.get("negative_prompt") or None,
                    "width": width,
                    "height": height,
                    "fps": int(clip.get("fps") or fps),
                    "num_frames": int(clip.get("num_frames") or default_frames),
                    "num_inference_steps": int(clip.get("steps") or payload.get("steps") or 25),
                    "guidance_scale": float(clip.get("guidance_scale") or payload.get("guidance_scale") or 7.0),
                    "mode": "t2v",
                    "seed": int(clip.get("seed") or random.randint(1, 2**31 - 1)),
                }
                if characters:
                    body["character_profiles"] = characters
                    body["character_strength"] = float(clip.get("character_strength") or payload.get("character_strength") or 0.75)
                if environments:
                    body["environment_profiles"] = environments
                if clip.get("camera_motion"):
                    body["camera_motion"] = clip.get("camera_motion")
                if selected_loras:
                    body["loras"] = selected_loras
                if clip.get("dialogues"):
                    body["dialogs"] = self._normalize_dialogues(clip.get("dialogues"))
                    body["lip_sync"] = bool(clip.get("lip_sync", True))
                    body["lip_sync_method"] = clip.get("lip_sync_method") or payload.get("lip_sync_method") or "wav2lip"
                    body["generate_subtitles"] = bool(clip.get("subtitles", True))
                    body["burn_subtitles"] = bool(clip.get("burn_subtitles", False))
                if clip.get("speech_text"):
                    body["tts_text"] = clip.get("speech_text")
                    body["tts_voice"] = clip.get("speech_voice") or payload.get("default_voice") or "af_sarah"
                    body["tts_speed"] = float(clip.get("speech_speed") or 1.0)
                    body["lip_sync"] = bool(clip.get("lip_sync", True))
                    body["lip_sync_method"] = clip.get("lip_sync_method") or payload.get("lip_sync_method") or "wav2lip"
                    body["add_audio"] = True
                    body["audio_type"] = "speech"
                if clip.get("music_prompt") or clip.get("sfx_prompt"):
                    body["add_audio"] = True
                    body["audio_type"] = clip.get("audio_type") or ("sfx" if clip.get("sfx_prompt") else "music")
                    body["audio_prompt"] = clip.get("music_prompt") or clip.get("sfx_prompt")
                if use_keyframes and image_model:
                    key_body = {
                        "model": image_model,
                        "prompt": prompt,
                        "size": f"{width}x{height}",
                        "steps": int(payload.get("keyframe_steps") or 24),
                        "character_profiles": characters or None,
                        "environment_profiles": environments or None,
                        "character_strength": float(payload.get("character_strength") or 0.75),
                    }
                    key_png = self.client.generate_image({k: v for k, v in key_body.items() if v is not None})
                    key_path = clip_dir / f"clip_{i+1:02d}_keyframe.png"
                    key_path.write_bytes(key_png)
                    body["mode"] = "ti2v"
                    body["init_image"] = "data:image/png;base64," + base64.b64encode(key_png).decode()
                mp4 = self.client.generate_video({k: v for k, v in body.items() if v is not None})
                clip_path = clip_dir / f"clip_{i+1:02d}_{safe_slug(label)}.mp4"
                clip_path.write_bytes(mp4)
                clip_paths.append(clip_path)
                self.emit(f"Rendered {clip_path.name} ({video_duration(clip_path):.1f}s)")
            self._job_update(job_id, progress=88, message="assembling clips")
            final_path = movie_dir / f"{slug}.mp4"
            concat_videos(clip_paths, final_path)
            if payload.get("soundtrack_prompt") and audio_model:
                self.emit("Generating final soundtrack")
                self._job_update(job_id, progress=93, message="generating soundtrack")
                duration = max(5.0, video_duration(final_path))
                audio = self.client.generate_audio({
                    "model": audio_model,
                    "prompt": payload.get("soundtrack_prompt"),
                    "duration": duration,
                    "temperature": float(payload.get("music_temperature") or 1.0),
                })
                audio_path = movie_dir / "soundtrack.wav"
                audio_path.write_bytes(audio)
                mixed_path = movie_dir / f"{slug}_mixed.mp4"
                mux_background_audio(final_path, audio_path, mixed_path, float(payload.get("music_volume") or 0.25))
                final_path = mixed_path
            rel = final_path.relative_to(self.out_dir).as_posix()
            self.emit(f"Movie complete: {final_path}")
            self._job_update(job_id, status="done", progress=100, message="done", output=rel, output_url=f"/media/{rel}")
        except Exception as exc:
            self.emit(f"Movie failed: {exc}")
            self._job_update(job_id, status="error", error=str(exc), message=str(exc))

    def _selected_loras(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        requested = payload.get("loras") or []
        if not requested:
            return []
        by_name = {item.get("name"): item for item in self.client.list_loras() if item.get("name")}
        out = []
        for item in requested:
            if isinstance(item, str):
                name, weight = item, float(payload.get("lora_weight") or 0.8)
            else:
                name, weight = item.get("name") or item.get("model"), float(item.get("weight") or payload.get("lora_weight") or 0.8)
            info = by_name.get(name, {})
            model = info.get("path") or name
            if model:
                out.append({"model": model, "weight": weight, "name": name})
        return out

    def _build_clip_prompt(self, movie: dict[str, Any], clip: dict[str, Any], characters: list[str], environments: list[str]) -> str:
        parts = []
        if movie.get("style"):
            parts.append(str(movie["style"]))
        if clip.get("shot"):
            parts.append(str(clip["shot"]))
        if clip.get("prompt"):
            parts.append(str(clip["prompt"]))
        if characters:
            hints = []
            for name in characters:
                meta = self.store.load_meta("character", name)
                hints.append(meta.get("description") or name)
            parts.append("Characters: " + "; ".join(hints))
        if environments:
            hints = []
            for name in environments:
                meta = self.store.load_meta("environment", name)
                hints.append(meta.get("description") or name)
            parts.append("Environment: " + "; ".join(hints))
        if clip.get("action"):
            parts.append("Action: " + str(clip["action"]))
        if clip.get("mood"):
            parts.append("Mood: " + str(clip["mood"]))
        return ", ".join(p for p in parts if p).strip()

    def _normalize_dialogues(self, dialogues: Any) -> list[dict[str, Any]]:
        if isinstance(dialogues, str):
            try:
                dialogues = json.loads(dialogues)
            except Exception:
                dialogues = []
        out = []
        for row in dialogues or []:
            if not isinstance(row, dict) or not row.get("text"):
                continue
            out.append({
                "character": row.get("character") or None,
                "voice": row.get("voice") or row.get("speaker") or None,
                "text": row.get("text"),
                "start_time": row.get("start_time") if row.get("start_time") not in ("", None) else None,
                "lip_sync": bool(row.get("lip_sync", True)),
                "lang": row.get("lang") or None,
                "speed": float(row.get("speed") or 1.0),
            })
        return out

    def _job_update(self, job_id: str, **updates: Any) -> None:
        with self.lock:
            job = self.jobs.setdefault(job_id, {})
            job.update(updates)
            job["updated_at"] = time.time()

    def _is_cancelled(self, job_id: str) -> bool:
        with self.lock:
            return bool(self.jobs.get(job_id, {}).get("cancel"))

    def cancel_job(self, job_id: str) -> None:
        self._job_update(job_id, cancel=True, message="cancelling")


HTML_PAGE = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CoderAI VideoGen Studio</title>
<style>
:root{--bg:#10141c;--panel:#18202d;--panel2:#202b3b;--ink:#eef3ff;--muted:#9fb0c7;--line:#314057;--accent:#42d6a4;--warn:#f5b461;--bad:#ff6b6b;--blue:#78a6ff}
*{box-sizing:border-box} body{margin:0;background:radial-gradient(circle at top left,#20344a 0,#10141c 38%,#0a0d13 100%);color:var(--ink);font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif} header{padding:28px 32px;border-bottom:1px solid var(--line);background:linear-gradient(135deg,rgba(66,214,164,.16),rgba(120,166,255,.08))} h1{margin:0;font-size:32px;letter-spacing:-.03em} h2{margin:0 0 14px;font-size:20px} h3{margin:12px 0 8px}.sub{color:var(--muted);margin-top:6px}.wrap{display:grid;grid-template-columns:340px 1fr;gap:18px;padding:18px}.card{background:rgba(24,32,45,.92);border:1px solid var(--line);border-radius:18px;padding:16px;box-shadow:0 14px 34px rgba(0,0,0,.22)} label{display:block;font-size:12px;color:var(--muted);margin:10px 0 5px} input,textarea,select{width:100%;border:1px solid var(--line);border-radius:12px;background:#0e141d;color:var(--ink);padding:10px 11px;font:inherit} textarea{min-height:90px;resize:vertical}.row{display:grid;grid-template-columns:1fr 1fr;gap:10px}.btn{border:0;border-radius:12px;padding:10px 13px;background:var(--accent);color:#062015;font-weight:800;cursor:pointer}.btn.secondary{background:#2a3548;color:var(--ink);border:1px solid var(--line)}.btn.warn{background:var(--warn);color:#241303}.btn.bad{background:var(--bad);color:#fff}.tabs{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px}.tab{padding:9px 13px;border-radius:999px;background:#18202d;border:1px solid var(--line);cursor:pointer;color:var(--muted)}.tab.active{background:var(--accent);color:#061b13;border-color:transparent;font-weight:800}.section{display:none}.section.active{display:block}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px}.profile{background:var(--panel2);border:1px solid var(--line);border-radius:14px;overflow:hidden}.profile img{width:100%;height:120px;object-fit:cover;background:#0e141d}.profile .p{padding:10px}.muted{color:var(--muted);font-size:13px}.clip{border:1px solid var(--line);border-radius:14px;padding:12px;background:#141c28;margin:12px 0}.clip-head{display:flex;justify-content:space-between;gap:10px;align-items:center}.dialogue{display:grid;grid-template-columns:1fr 1fr 2fr .7fr auto;gap:8px;margin:8px 0}.log{height:260px;overflow:auto;background:#070a0f;border:1px solid #253146;border-radius:14px;padding:12px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;white-space:pre-wrap}.pill{display:inline-block;padding:3px 8px;background:#111925;border:1px solid var(--line);border-radius:999px;color:var(--muted);font-size:12px;margin:2px}.out a{color:var(--accent)}@media(max-width:900px){.wrap{grid-template-columns:1fr}.dialogue{grid-template-columns:1fr}.row{grid-template-columns:1fr}}
</style>
</head>
<body>
<header><h1>CoderAI VideoGen Studio</h1><div class="sub">Manage reusable characters, environments, cloned voices, and multi-clip movies with realistic dialogue, lip-sync, music, and sound effects.</div></header>
<div class="wrap">
  <aside class="card">
    <h2>Connection</h2>
    <div class="muted" id="conn">Loading models...</div>
    <label>Image model</label><select id="image_model"></select>
    <label>Video model</label><select id="video_model"></select>
    <label>Audio/Music model</label><select id="audio_model"></select>
    <label>Default voice / cloned profile</label><input id="default_voice" list="voice_list" value="af_sarah"><datalist id="voice_list"></datalist>
    <hr style="border-color:var(--line);border-style:solid none none;margin:16px 0">
    <h2>Live Log</h2>
    <div class="log" id="log"></div>
    <div class="out" id="jobout"></div>
  </aside>
  <main class="card">
    <div class="tabs">
      <button class="tab active" data-tab="profiles">Profiles</button>
      <button class="tab" data-tab="movie">Movie Builder</button>
      <button class="tab" data-tab="loras">Style LoRAs</button>
      <button class="tab" data-tab="gallery">Gallery</button>
    </div>

    <section id="profiles" class="section active">
      <h2>Characters, Environments, and Voices</h2>
      <div class="row">
        <div class="card">
          <h3>Create Character</h3>
          <label>Name</label><input id="char_name" placeholder="alice">
          <label>Description</label><textarea id="char_desc" placeholder="Visual identity, age, clothing, face, silhouette..."></textarea>
          <label>Reference generation prompt</label><textarea id="char_prompt" placeholder="Consistent character sheet, front and side views..."></textarea>
          <div class="row"><div><label>Refs</label><input id="char_n" type="number" value="4"></div><div><label>Size</label><input id="char_size" value="512x512"></div></div>
          <button class="btn" onclick="createProfile('character')">Generate Character</button>
        </div>
        <div class="card">
          <h3>Create Environment</h3>
          <label>Name</label><input id="env_name" placeholder="old_library">
          <label>Description</label><textarea id="env_desc" placeholder="Location, architecture, lighting, atmosphere..."></textarea>
          <label>Reference generation prompt</label><textarea id="env_prompt" placeholder="Wide environment concept art..."></textarea>
          <div class="row"><div><label>Refs</label><input id="env_n" type="number" value="3"></div><div><label>Size</label><input id="env_size" value="768x512"></div></div>
          <button class="btn" onclick="createProfile('environment')">Generate Environment</button>
        </div>
      </div>
      <div class="card" style="margin-top:12px">
        <h3>Create Cloned Voice</h3>
        <div class="muted">Use a clean 5-30 second reference clip plus an exact transcript for realistic cloned dialogue. Audio or video files are accepted.</div>
        <div class="row"><div><label>Voice name</label><input id="voice_name" placeholder="alice_voice"></div><div><label>Reference audio/video path</label><input id="voice_media" placeholder="/path/to/reference.wav"></div></div>
        <label>Description</label><input id="voice_desc" placeholder="Warm, calm adult voice; close mic">
        <label>Exact reference transcript</label><textarea id="voice_transcript" placeholder="The exact words spoken in the reference clip. Required for best F5-TTS cloning."></textarea>
        <button class="btn" onclick="createVoice()">Create Cloned Voice</button>
      </div>
      <h3>Saved Characters</h3><div class="grid" id="chars"></div>
      <h3>Saved Environments</h3><div class="grid" id="envs"></div>
      <h3>Saved Voices</h3><div class="grid" id="voices"></div>
    </section>

    <section id="movie" class="section">
      <h2>Movie Builder</h2>
      <div class="row">
        <div><label>Title</label><input id="title" value="my_little_movie"></div>
        <div><label>Visual style</label><input id="style" value="cinematic, coherent character identity, natural motion, detailed lighting"></div>
      </div>
      <label>Style LoRAs to apply</label><select id="movie_loras" multiple size="5"></select>
      <div class="row"><div><label>LoRA weight</label><input id="movie_lora_weight" type="number" step="0.05" value="0.8"></div><div><label>Movie variants to produce</label><input id="movie_count" type="number" min="1" value="1"></div></div>
      <div class="row">
        <div><label>Width</label><input id="width" type="number" value="768"></div>
        <div><label>Height</label><input id="height" type="number" value="432"></div>
      </div>
      <div class="row">
        <div><label>FPS</label><input id="fps" type="number" value="8"></div>
        <div><label>Frames per clip</label><input id="num_frames" type="number" value="32"></div>
      </div>
      <div class="row">
        <div><label>Steps</label><input id="steps" type="number" value="25"></div>
        <div><label>Guidance</label><input id="guidance_scale" type="number" step="0.1" value="7.0"></div>
      </div>
      <label>Global negative prompt</label><input id="negative_prompt" value="flicker, morphing faces, extra limbs, low quality, unreadable text">
      <label><input id="use_keyframes" type="checkbox" style="width:auto"> Generate keyframe image before each video clip for stronger character/environment consistency</label>
      <label>Lip-sync method</label><select id="lip_sync_method"><option value="wav2lip">wav2lip</option><option value="sadtalker">sadtalker</option></select>
      <label>Final soundtrack prompt (optional; mixed under assembled movie)</label><textarea id="soundtrack_prompt" placeholder="tense orchestral pulse with soft percussion, no vocals"></textarea>
      <div id="clips"></div>
      <button class="btn secondary" onclick="addClip()">Add Clip</button>
      <button class="btn" onclick="startMovie()">Render Movie</button>
    </section>

    <section id="loras" class="section">
      <h2>Style LoRA Training</h2>
      <div class="muted">Train a reusable style LoRA from image/video scene references. Add stills or videos, describe the visual style, then choose the trained LoRA in Movie Builder to produce one or more movies in that style.</div>
      <div class="card" style="margin-top:12px">
        <div class="row"><div><label>LoRA name</label><input id="lora_name" placeholder="noir_rain_style"></div><div><label>Target</label><select id="lora_target"><option value="video">video</option><option value="image">image</option></select></div></div>
        <label>Training media paths, one per line</label><textarea id="lora_media" placeholder="/path/to/style_reference_01.png&#10;/path/to/scene_clip.mp4"></textarea>
        <label>Scene/style description</label><textarea id="lora_style" placeholder="Describe the scene language: lighting, lens, color palette, texture, movement, costumes, production design..."></textarea>
        <label>Instance prompt/token</label><input id="lora_instance" placeholder="in noir_rain_style cinematic style">
        <div class="row"><div><label>Steps</label><input id="lora_steps" type="number" value="800"></div><div><label>Rank</label><input id="lora_rank" type="number" value="16"></div></div>
        <div class="row"><div><label>Resolution</label><input id="lora_resolution" type="number" value="512"></div><div><label>Frames per video sample</label><input id="lora_frames" type="number" value="6"></div></div>
        <button class="btn" onclick="trainLora()">Train Style LoRA</button>
      </div>
      <h3>Available LoRAs</h3><div class="grid" id="loras_grid"></div>
    </section>

    <section id="gallery" class="section">
      <h2>Gallery</h2>
      <button class="btn secondary" onclick="loadGallery()">Refresh Gallery</button>
      <div class="grid" id="gallery_grid"></div>
    </section>
  </main>
</div>
<script>
let models=[], profiles={characters:[], environments:[], voices:[], loras:[]};
function $(id){return document.getElementById(id)}
function esc(s){return String(s||'').replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]))}
async function api(path, opts={}){let r=await fetch(path,{headers:{'Content-Type':'application/json'},...opts}); if(!r.ok) throw new Error(await r.text()); return await r.json()}
function fillSelect(sel, cap, def){let s=$(sel); s.innerHTML=''; let filtered=models.filter(m=>(m.capabilities||[]).includes(cap)); if(!filtered.length) filtered=models; for(let m of filtered){let o=document.createElement('option'); o.value=m.id; o.textContent=m.id; if(m.id===def) o.selected=true; s.appendChild(o)}}
async function loadModels(){let d=await api('/api/models'); models=d.models||[]; fillSelect('image_model','image_generation',d.defaults.image_model); fillSelect('video_model','video_generation',d.defaults.video_model); fillSelect('audio_model','audio_generation',d.defaults.audio_model); $('conn').textContent=`Connected: ${models.length} model(s)`}
async function loadProfiles(){profiles=await api('/api/profiles'); renderProfiles()}
function profileCard(p,kind){return `<div class="profile"><img src="${p.thumbnail||''}" onerror="this.style.display='none'"><div class="p"><b>${esc(p.name)}</b><div class="muted">${esc(p.description||'')}</div><span class="pill">${kind}</span><span class="pill">${p.image_count||0} refs</span><span class="pill">${p.local?'local':'server'}</span></div></div>`}
function voiceCard(v){return `<div class="profile"><div class="p"><b>${esc(v.name)}</b><div class="muted">${esc(v.description||'')}</div><span class="pill">cloned voice</span><span class="pill">${esc(v.audio_ext||'audio')}</span></div></div>`}
function loraCard(l){return `<div class="profile"><div class="p"><b>${esc(l.name)}</b><div class="muted">${esc(l.instance_prompt||l.path||'')}</div><span class="pill">${esc(l.target||'lora')}</span><span class="pill">rank ${esc(l.rank||'')}</span></div></div>`}
function renderVoiceList(){let voices=['af_sarah','af_sky','af_bella','am_adam','am_michael','en-US-JennyNeural',...(profiles.voices||[]).map(v=>v.name)].filter(Boolean); $('voice_list').innerHTML=[...new Set(voices)].map(v=>`<option value="${esc(v)}"></option>`).join('')}
function renderLoraList(){let opts=(profiles.loras||[]).map(l=>`<option value="${esc(l.name)}">${esc(l.name)}${l.target?' ('+esc(l.target)+')':''}</option>`).join(''); $('movie_loras').innerHTML=opts; $('loras_grid').innerHTML=(profiles.loras||[]).map(loraCard).join('')||'<div class="muted">No style LoRAs yet. Train one from media references.</div>'}
function renderProfiles(){$('chars').innerHTML=profiles.characters.map(p=>profileCard(p,'character')).join('')||'<div class="muted">No characters yet.</div>'; $('envs').innerHTML=profiles.environments.map(p=>profileCard(p,'environment')).join('')||'<div class="muted">No environments yet.</div>'; $('voices').innerHTML=(profiles.voices||[]).map(voiceCard).join('')||'<div class="muted">No cloned voices yet. Create one from a reference clip.</div>'; renderVoiceList(); renderLoraList(); renderClipSelectors()}
async function createProfile(kind){let isChar=kind==='character'; let name=$(isChar?'char_name':'env_name').value; let desc=$(isChar?'char_desc':'env_desc').value; let prompt=$(isChar?'char_prompt':'env_prompt').value||desc; let n=$(isChar?'char_n':'env_n').value; let [w,h]=($(isChar?'char_size':'env_size').value||'512x512').split('x').map(x=>parseInt(x,10)); let model=$('image_model').value; let d=await api('/api/profile/start',{method:'POST',body:JSON.stringify({kind,name,description:desc,prompt,model,n,width:w,height:h})}); watchJob(d.job_id);}
async function createVoice(){let d=await api('/api/voice/start',{method:'POST',body:JSON.stringify({name:$('voice_name').value,description:$('voice_desc').value,transcript:$('voice_transcript').value,media_path:$('voice_media').value,extract:true})}); watchJob(d.job_id)}
async function trainLora(){let d=await api('/api/lora/start',{method:'POST',body:JSON.stringify({name:$('lora_name').value,base_model:$('video_model').value,target:$('lora_target').value,media_paths:$('lora_media').value,style_prompt:$('lora_style').value,scene_description:$('lora_style').value,instance_prompt:$('lora_instance').value,steps:$('lora_steps').value,rank:$('lora_rank').value,resolution:$('lora_resolution').value,frames_per_video:$('lora_frames').value,quantize_4bit:true})}); watchJob(d.job_id)}
function options(items){return items.map(p=>`<option value="${esc(p.name)}">${esc(p.name)}</option>`).join('')}
function addClip(data={}){let idx=document.querySelectorAll('.clip').length+1; let div=document.createElement('div'); div.className='clip'; div.innerHTML=`<div class="clip-head"><h3>Clip ${idx}</h3><button class="btn bad" onclick="this.closest('.clip').remove()">Remove</button></div>
<label>Clip title</label><input class="c_title" value="${esc(data.title||'Shot '+idx)}">
<label>Shot/movie description</label><textarea class="c_prompt" placeholder="Describe the shot: camera, action, blocking, composition...">${esc(data.prompt||'')}</textarea>
<div class="row"><div><label>Characters</label><select class="c_chars" multiple size="5">${options(profiles.characters)}</select></div><div><label>Environments</label><select class="c_envs" multiple size="5">${options(profiles.environments)}</select></div></div>
<div class="row"><div><label>Camera motion</label><input class="c_camera" placeholder="zoom-in, pan-left, handheld..."></div><div><label>Mood/action</label><input class="c_action" placeholder="what happens in this clip"></div></div>
<label>Speech text (simple one-speaker; optional)</label><input class="c_speech" placeholder="Line spoken in this shot">
<div class="row"><div><label>Speech voice / cloned profile</label><input class="c_voice" list="voice_list" value="${esc($('default_voice').value||'af_sarah')}"></div><div><label>Speech speed</label><input class="c_speed" type="number" step="0.05" value="1.0"></div></div>
<label><input class="c_lipsync" type="checkbox" checked style="width:auto"> Lip-sync speech/dialogue</label>
<h3>Multi-character dialogue</h3><div class="dialogues"></div><button class="btn secondary" onclick="addDialogue(this)">Add Dialogue Line</button>
<label>Music prompt for this clip</label><input class="c_music" placeholder="short local music bed for this clip">
<label>Sound effects prompt for this clip</label><input class="c_sfx" placeholder="rain, footsteps, door creak, city ambience">
</div>`; $('clips').appendChild(div); renderClipSelectors(div)}
function renderClipSelectors(root=document){for(let s of root.querySelectorAll('.c_chars')){let vals=[...s.selectedOptions].map(o=>o.value); s.innerHTML=options(profiles.characters); for(let o of s.options) if(vals.includes(o.value)) o.selected=true} for(let s of root.querySelectorAll('.c_envs')){let vals=[...s.selectedOptions].map(o=>o.value); s.innerHTML=options(profiles.environments); for(let o of s.options) if(vals.includes(o.value)) o.selected=true} for(let s of root.querySelectorAll('.d_char')){let val=s.value; s.innerHTML='<option value="">(none)</option>'+options(profiles.characters); s.value=val}}
function addDialogue(btn){let box=btn.closest('.clip').querySelector('.dialogues'); let row=document.createElement('div'); row.className='dialogue'; row.innerHTML=`<select class="d_char"><option value="">(none)</option>${options(profiles.characters)}</select><input class="d_voice" list="voice_list" placeholder="voice/profile" value="${esc($('default_voice').value||'af_sarah')}"><input class="d_text" placeholder="dialogue text"><input class="d_start" placeholder="start s"><input class="d_speed" type="number" step="0.05" value="1.0"><button class="btn bad" onclick="this.parentElement.remove()">x</button>`; box.appendChild(row)}
function selected(sel){return [...sel.selectedOptions].map(o=>o.value)}
function collectMovie(){let clips=[...document.querySelectorAll('.clip')].map(c=>({title:c.querySelector('.c_title').value,prompt:c.querySelector('.c_prompt').value,characters:selected(c.querySelector('.c_chars')),environments:selected(c.querySelector('.c_envs')),camera_motion:c.querySelector('.c_camera').value,action:c.querySelector('.c_action').value,speech_text:c.querySelector('.c_speech').value,speech_voice:c.querySelector('.c_voice').value,speech_speed:c.querySelector('.c_speed').value,lip_sync:c.querySelector('.c_lipsync').checked,lip_sync_method:$('lip_sync_method').value,music_prompt:c.querySelector('.c_music').value,sfx_prompt:c.querySelector('.c_sfx').value,dialogues:[...c.querySelectorAll('.dialogue')].map(d=>({character:d.querySelector('.d_char').value,voice:d.querySelector('.d_voice').value,text:d.querySelector('.d_text').value,start_time:d.querySelector('.d_start').value,speed:d.querySelector('.d_speed').value,lip_sync:c.querySelector('.c_lipsync').checked}))})); return {title:$('title').value,style:$('style').value,image_model:$('image_model').value,video_model:$('video_model').value,audio_model:$('audio_model').value,default_voice:$('default_voice').value,lip_sync_method:$('lip_sync_method').value,width:+$('width').value,height:+$('height').value,fps:+$('fps').value,num_frames:+$('num_frames').value,steps:+$('steps').value,guidance_scale:+$('guidance_scale').value,negative_prompt:$('negative_prompt').value,use_keyframes:$('use_keyframes').checked,soundtrack_prompt:$('soundtrack_prompt').value,loras:selected($('movie_loras')).map(n=>({name:n,weight:+$('movie_lora_weight').value})),lora_weight:+$('movie_lora_weight').value,movie_count:+$('movie_count').value,clips}}
async function startMovie(){let d=await api('/api/movie/start',{method:'POST',body:JSON.stringify(collectMovie())}); watchJob(d.job_id)}
async function watchJob(id){$('jobout').innerHTML=`<p>Job <span class="pill">${id}</span></p>`; let timer=setInterval(async()=>{let j=await api('/api/job/'+id); $('jobout').innerHTML=`<p><span class="pill">${esc(j.status)}</span> ${j.progress||0}% ${esc(j.message||'')}</p>`+(j.output_url?`<p><a href="${j.output_url}" target="_blank">Open output</a></p>`:'')+(j.error?`<p style="color:var(--bad)">${esc(j.error)}</p>`:''); if(j.status==='done'||j.status==='error'){clearInterval(timer); loadProfiles(); loadGallery()}},1500)}
async function loadGallery(){let d=await api('/api/gallery'); $('gallery_grid').innerHTML=(d.items||[]).map(it=>`<div class="profile">${it.type==='video'?`<video src="${it.url}" controls style="width:100%;height:130px;background:#000"></video>`:`<img src="${it.url}">`}<div class="p"><b>${esc(it.name)}</b><br><a href="${it.url}" target="_blank">open</a></div></div>`).join('')||'<div class="muted">No media yet.</div>'}
function connectLog(){let es=new EventSource('/stream'); es.onmessage=e=>{let l=$('log'); l.textContent+=e.data+'\n'; l.scrollTop=l.scrollHeight}}
document.querySelectorAll('.tab').forEach(t=>t.onclick=()=>{document.querySelectorAll('.tab,.section').forEach(x=>x.classList.remove('active')); t.classList.add('active'); $(t.dataset.tab).classList.add('active')})
loadModels().then(loadProfiles).then(()=>addClip()); loadGallery(); connectLog();
</script>
</body>
</html>
"""


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def make_handler(app: VideoGenApp):
    class Handler(BaseHTTPRequestHandler):
        server_version = "VideoGen/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def _send(self, code: int, body: bytes, content_type: str = "text/plain") -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data: Any, code: int = 200) -> None:
            self._send(code, json.dumps(data).encode(), "application/json")

        def _read_json(self) -> dict[str, Any]:
            n = int(self.headers.get("Content-Length") or 0)
            if not n:
                return {}
            return json.loads(self.rfile.read(n).decode("utf-8"))

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            try:
                if path == "/":
                    self._send(200, HTML_PAGE.encode("utf-8"), "text/html; charset=utf-8")
                elif path == "/api/models":
                    self._json(app.models_payload())
                elif path == "/api/profiles":
                    self._json(app.profiles_payload())
                elif path.startswith("/api/job/"):
                    job_id = path.rsplit("/", 1)[1]
                    with app.lock:
                        self._json(app.jobs.get(job_id, {"status": "missing", "progress": 0}))
                elif path == "/api/gallery":
                    self._json({"items": self._gallery_items()})
                elif path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    with app.lock:
                        backlog = list(app.log_lines[-100:])
                    for line in backlog:
                        self.wfile.write(f"data: {line}\n\n".encode())
                    self.wfile.flush()
                    while True:
                        try:
                            line = app.log_queue.get(timeout=20)
                            self.wfile.write(f"data: {line}\n\n".encode())
                        except queue.Empty:
                            self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                elif path.startswith("/media/"):
                    rel = urllib.parse.unquote(path[len("/media/"):])
                    target = (app.out_dir / rel).resolve()
                    if not str(target).startswith(str(app.out_dir.resolve())) or not target.exists() or not target.is_file():
                        self._json({"error": "not found"}, 404)
                        return
                    ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
                    self._send(200, target.read_bytes(), ctype)
                else:
                    self._json({"error": "not found"}, 404)
            except BrokenPipeError:
                pass
            except Exception as exc:
                self._json({"error": str(exc)}, 500)

        def do_POST(self) -> None:
            path = urllib.parse.urlparse(self.path).path
            try:
                payload = self._read_json()
                if path == "/api/profile/start":
                    self._json({"job_id": app.start_profile_job(payload)})
                elif path == "/api/voice/start":
                    self._json({"job_id": app.start_voice_job(payload)})
                elif path == "/api/lora/start":
                    self._json({"job_id": app.start_lora_job(payload)})
                elif path == "/api/movie/start":
                    self._json({"job_id": app.start_movie_job(payload)})
                elif path.startswith("/api/job/") and path.endswith("/cancel"):
                    job_id = path.split("/")[-2]
                    app.cancel_job(job_id)
                    self._json({"ok": True})
                else:
                    self._json({"error": "not found"}, 404)
            except Exception as exc:
                self._json({"error": str(exc)}, 500)

        def _gallery_items(self) -> list[dict[str, Any]]:
            items = []
            for path in sorted(app.out_dir.rglob("*"), key=lambda p: p.stat().st_mtime if p.is_file() else 0, reverse=True):
                if not path.is_file():
                    continue
                ext = path.suffix.lower()
                if ext not in {".mp4", ".webm", ".mov", ".png", ".jpg", ".jpeg", ".webp", ".wav", ".mp3"}:
                    continue
                rel = path.relative_to(app.out_dir).as_posix()
                typ = "video" if ext in {".mp4", ".webm", ".mov"} else ("image" if ext in {".png", ".jpg", ".jpeg", ".webp"} else "audio")
                items.append({"name": path.name, "url": "/media/" + quote_rel(rel), "type": typ})
                if len(items) >= 80:
                    break
            return items

    return Handler


def quote_rel(path: str) -> str:
    return "/".join(urllib.parse.quote(part) for part in path.split("/"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CoderAI VideoGen Studio")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="CoderAI base URL")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="Bearer token for CoderAI")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Local output directory")
    parser.add_argument("--host", default="0.0.0.0", help="Web UI listen host (default: 0.0.0.0)")
    parser.add_argument("--web-port", type=int, default=7790, help="Local web UI port")
    parser.add_argument("--image-model", default="", help="Default image model")
    parser.add_argument("--video-model", default="", help="Default video model")
    parser.add_argument("--audio-model", default="", help="Default audio/music model")
    parser.add_argument("--tts-model", default="", help="Default TTS model, when direct speech endpoint is used")
    parser.add_argument("--browser", action="store_true", help="Open browser after startup")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    app = VideoGenApp(args)
    server = ThreadedHTTPServer((args.host, args.web_port), make_handler(app))
    url = f"http://{args.host}:{args.web_port}"
    log(f"VideoGen Studio running at {url}")
    log(f"CoderAI: {args.base_url}")
    log(f"Output: {Path(args.out_dir).resolve()}")
    if args.browser:
        import webbrowser
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("Stopping VideoGen Studio")


if __name__ == "__main__":
    main()
