#!/usr/bin/env python3
"""CoderAI Character Studio — turn real photos/videos into a reusable character, then
put that character into a completely new, generated video.

Everything AI is delegated to CoderAI endpoints; this script only orchestrates and
keeps local copies of the results:

  - POST /v1/characters/extract   detect faces in the supplied photos/videos and save
                                  the best crops as a named character profile
  - GET  /v1/characters           list saved profiles
  - GET  /v1/characters/{name}    fetch a profile's reference images
  - PATCH/DELETE /v1/characters/  prune bad references / remove a profile
  - POST /v1/audio/voices/extract clone the voice off the same footage, so the
                                  character can speak in their own voice
  - POST /v1/loras/train          optionally train an identity LoRA straight from
                                  the profile's crops (`character: <name>`), for
                                  models where IP-Adapter alone drifts
  - GET  /v1/loras/progress       poll / re-attach to that training job
  - POST /v1/video/generations    generate a new video with `character_profiles`
                                  (IP-Adapter, or the identity-keyframe bridge on
                                  models without one) plus any trained LoRAs
  - POST /v1/images/generations   optional identity portrait, as a quick sanity check

Two front-ends over the same core:

  CLI     python tools/character_studio.py demo --name alice \
              --source clips/alice.mp4 --source photos/alice1.jpg \
              --prompt "walking through a neon-lit Tokyo street at night"

Only use footage of people who agreed to it — your own actors, your own likeness,
or characters you generated. The profile is whoever you put in it.

  Web UI  python tools/character_studio.py web --browser
          (drop files, review the extracted crops, drop the bad ones, generate)

Python dependency: requests. Pillow is optional (used only for the contact sheet).
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import mimetypes
import os
import random
import re
import sys
import threading
import time
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable

try:
    import requests
except ImportError as exc:  # pragma: no cover - user environment check
    raise SystemExit("This script requires requests: pip install requests") from exc


DEFAULT_BASE_URL = os.environ.get("CODERAI_BASE_URL", "http://127.0.0.1:8000")
DEFAULT_API_KEY = os.environ.get("CODERAI_API_KEY")
DEFAULT_OUT_DIR = os.environ.get("CODERAI_CHARACTER_OUT", "character_output")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus"}

DEFAULT_NEGATIVE = (
    "different person, face morphing, distorted face, extra fingers, blurry, "
    "low quality, watermark, text"
)


def log(msg: str) -> None:
    print(msg, flush=True)


def safe_slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", (text or "").strip()).strip("._-")
    return slug or "character"


def decode_data_uri(value: str) -> bytes:
    if value.startswith("data:"):
        return base64.b64decode(value.split(",", 1)[1])
    return base64.b64decode(value)


def data_uri_for_file(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def kind_of(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in VIDEO_EXTS:
        return "video"
    if ext in IMAGE_EXTS:
        return "image"
    return ""


def expand_sources(sources: Iterable[str]) -> list[str]:
    """Expand files, directories and globs into a flat list of paths / URLs.

    URLs are passed through untouched — the server fetches them itself.
    """
    out: list[str] = []
    for raw in sources:
        item = str(raw).strip()
        if not item:
            continue
        if item.startswith(("http://", "https://", "data:")):
            out.append(item)
            continue
        path = Path(item).expanduser()
        if path.is_dir():
            for child in sorted(path.iterdir()):
                if child.is_file() and kind_of(child):
                    out.append(str(child))
            continue
        if any(ch in item for ch in "*?[") and not path.exists():
            for match in sorted(Path().glob(item)):
                if match.is_file() and kind_of(match):
                    out.append(str(match))
            continue
        if not path.is_file():
            raise SystemExit(f"Source not found: {item}")
        out.append(str(path))
    if not out:
        raise SystemExit("No usable source images or videos were found")
    return out


def split_sources(sources: Iterable[str]) -> tuple[list[str], list[str]]:
    """Split expanded sources into (images, videos) as base64/data-URI/URL strings."""
    images: list[str] = []
    videos: list[str] = []
    for item in sources:
        if item.startswith(("http://", "https://", "data:")):
            # Decide by extension when we can, default to image.
            guess = kind_of(Path(urllib.parse.urlparse(item).path)) if not item.startswith("data:") else ""
            if not guess and item.startswith("data:"):
                guess = "video" if item[5:].split(";", 1)[0].startswith("video/") else "image"
            (videos if guess == "video" else images).append(item)
            continue
        path = Path(item)
        payload = data_uri_for_file(path)
        (videos if kind_of(path) == "video" else images).append(payload)
    return images, videos


def contact_sheet(images: list[bytes], out_path: Path, columns: int = 5) -> Path | None:
    """Write a labelled grid of the reference crops. Silently skipped without Pillow."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    if not images:
        return None
    cell, pad, label_h = 192, 8, 18
    thumbs = []
    for raw in images:
        try:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            continue
        img.thumbnail((cell, cell))
        thumbs.append(img)
    if not thumbs:
        return None
    cols = max(1, min(columns, len(thumbs)))
    rows = (len(thumbs) + cols - 1) // cols
    width = cols * (cell + pad) + pad
    height = rows * (cell + label_h + pad) + pad
    sheet = Image.new("RGB", (width, height), (16, 20, 28))
    draw = ImageDraw.Draw(sheet)
    for idx, thumb in enumerate(thumbs):
        col, row = idx % cols, idx // cols
        x = pad + col * (cell + pad) + (cell - thumb.width) // 2
        y = pad + row * (cell + label_h + pad)
        sheet.paste(thumb, (x, y))
        draw.text((pad + col * (cell + pad), y + cell + 4), f"[{idx}]", fill=(200, 215, 235))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path


class CoderAIClient:
    def __init__(self, base_url: str, api_key: str | None = None, timeout: int = 7200):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"

    # ── plumbing ────────────────────────────────────────────────────────────
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

    def _patch(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        resp = self.session.patch(f"{self.base}{path}", json=body, timeout=300)
        if not resp.ok:
            raise RuntimeError(f"PATCH {path} -> {resp.status_code}: {resp.text[:800]}")
        return resp.json()

    def _delete(self, path: str) -> dict[str, Any]:
        resp = self.session.delete(f"{self.base}{path}", timeout=120)
        if not resp.ok:
            raise RuntimeError(f"DELETE {path} -> {resp.status_code}: {resp.text[:800]}")
        return resp.json()

    def _bytes_from_api_value(self, raw: str) -> bytes:
        if not raw:
            raise RuntimeError("API response did not contain media data")
        if raw.startswith("data:"):
            return decode_data_uri(raw)
        if raw.startswith(("http://", "https://")):
            with self.session.get(raw, timeout=self.timeout) as resp:
                resp.raise_for_status()
                return resp.content
        if raw.startswith("/v1/"):
            with self.session.get(f"{self.base}{raw}", timeout=self.timeout) as resp:
                resp.raise_for_status()
                return resp.content
        return base64.b64decode(raw)

    # ── models ──────────────────────────────────────────────────────────────
    def list_models(self) -> list[dict[str, Any]]:
        return self._get("/v1/models").get("data", [])

    # ── characters ──────────────────────────────────────────────────────────
    def list_characters(self) -> list[dict[str, Any]]:
        try:
            return self._get("/v1/characters").get("characters", [])
        except Exception:
            return []

    def get_character(self, name: str) -> dict[str, Any]:
        return self._get(f"/v1/characters/{urllib.parse.quote(name)}")

    def extract_character(self, name: str, description: str, images: list[str],
                          videos: list[str], max_images: int) -> dict[str, Any]:
        body: dict[str, Any] = {"name": name, "description": description, "max_images": int(max_images)}
        if images:
            body["images"] = images
        if videos:
            body["videos"] = videos
        return self._post("/v1/characters/extract", body)

    def prune_character(self, name: str, drop: list[int]) -> dict[str, Any]:
        return self._patch(f"/v1/characters/{urllib.parse.quote(name)}", {"remove_indices": sorted(set(drop))})

    def delete_character(self, name: str) -> dict[str, Any]:
        return self._delete(f"/v1/characters/{urllib.parse.quote(name)}")

    # ── voices ──────────────────────────────────────────────────────────────
    def list_voices(self) -> list[dict[str, Any]]:
        try:
            return self._get("/v1/audio/voices").get("voices", [])
        except Exception:
            return []

    def extract_voice(self, name: str, description: str, media: str,
                      is_video: bool, transcript: str = "") -> dict[str, Any]:
        body = {"name": name, "description": description, "transcript": transcript}
        body["video" if is_video else "audio"] = media
        return self._post("/v1/audio/voices/extract", body)

    # ── LoRAs ───────────────────────────────────────────────────────────────
    def list_loras(self) -> list[dict[str, Any]]:
        try:
            return self._get("/v1/loras").get("loras", [])
        except Exception:
            return []

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

    # ── generation ──────────────────────────────────────────────────────────
    def generate_image(self, body: dict[str, Any]) -> bytes:
        body = dict(body)
        body["response_format"] = "b64_json"
        data = self._post("/v1/images/generations", body)
        item = (data.get("data") or [{}])[0]
        return self._bytes_from_api_value(item.get("b64_json") or item.get("url") or "")

    def generate_video(self, body: dict[str, Any]) -> bytes:
        body = dict(body)
        body["response_format"] = "b64_mp4"
        data = self._post("/v1/video/generations", body)
        item = (data.get("data") or [{}])[0]
        return self._bytes_from_api_value(item.get("b64_mp4") or item.get("url") or "")


def pick_model(models: list[dict[str, Any]], cap: str, override: str | None = None) -> str:
    if override:
        return override
    for model in models:
        if cap in (model.get("capabilities") or []):
            return str(model.get("id") or "")
    return ""


class CharacterStudio:
    """Shared core: the CLI and the web UI both drive this."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.out_dir = Path(args.out_dir).expanduser()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.client = CoderAIClient(args.base_url, args.api_key)
        self.lock = threading.Lock()
        self.jobs: dict[str, dict[str, Any]] = {}

    # ── local mirror of a profile ───────────────────────────────────────────
    def char_dir(self, name: str) -> Path:
        return self.out_dir / "characters" / safe_slug(name)

    def mirror_profile(self, name: str) -> dict[str, Any]:
        """Pull the saved profile back down and write refs + contact sheet locally."""
        profile = self.client.get_character(name)
        cdir = self.char_dir(name)
        cdir.mkdir(parents=True, exist_ok=True)
        for stale in cdir.glob("ref_*.png"):
            stale.unlink()
        raw_images: list[bytes] = []
        for idx, img in enumerate(profile.get("images") or []):
            data = img.get("data") or ""
            if not data:
                continue
            blob = decode_data_uri(data)
            (cdir / f"ref_{idx:02d}.png").write_bytes(blob)
            raw_images.append(blob)
        sheet = contact_sheet(raw_images, cdir / "contact_sheet.png")
        (cdir / "meta.json").write_text(json.dumps({
            "name": profile.get("name", name),
            "description": profile.get("description", ""),
            "image_count": profile.get("image_count", len(raw_images)),
            "created_at": profile.get("created_at"),
        }, indent=2), encoding="utf-8")
        return {
            "name": profile.get("name", name),
            "description": profile.get("description", ""),
            "image_count": len(raw_images),
            "dir": str(cdir),
            "contact_sheet": str(sheet) if sheet else "",
            "images": [img.get("data", "") for img in (profile.get("images") or [])],
        }

    # ── step 1: extraction ──────────────────────────────────────────────────
    def extract(self, name: str, sources: list[str], description: str = "",
                max_images: int = 5, emit=log, with_voice: bool = False,
                transcript: str = "") -> dict[str, Any]:
        images, videos = split_sources(sources)
        emit(f"Extracting '{name}' from {len(images)} image(s) and {len(videos)} video(s)...")
        result = self.client.extract_character(name, description, images, videos, max_images)
        emit(f"Server saved {result.get('image_count')} reference image(s)")
        mirrored = self.mirror_profile(name)
        emit(f"References mirrored to {mirrored['dir']}")
        if mirrored.get("contact_sheet"):
            emit(f"Contact sheet: {mirrored['contact_sheet']}")
        else:
            emit("Contact sheet skipped (install Pillow to get one)")
        if with_voice:
            try:
                mirrored["voice"] = self.extract_voice_profile(
                    name, sources, emit=emit, transcript=transcript)
            except Exception as exc:
                emit(f"Voice clone failed ({exc}) — the character is still usable")
        return mirrored

    # ── step 1a: the voice off the same footage ─────────────────────────────
    def extract_voice_profile(self, name: str, sources: list[str], emit=log,
                              transcript: str = "") -> str | None:
        """Clone a voice profile from the first audio-bearing source.

        The face and the voice come from the same clip, so the character can speak
        in their own voice later (`--say`), instead of a stock TTS voice. Silently
        returns None when only stills were supplied — there is nothing to clone.
        """
        media, is_video = None, True
        for item in sources:
            if item.startswith(("http://", "https://")):
                if kind_of(Path(urllib.parse.urlparse(item).path)) == "video":
                    media = item
                    break
                continue
            if item.startswith("data:"):
                if item[5:].split(";", 1)[0].startswith(("video/", "audio/")):
                    media, is_video = item, item[5:].startswith("video/")
                    break
                continue
            path = Path(item)
            if kind_of(path) == "video" or path.suffix.lower() in AUDIO_EXTS:
                media = data_uri_for_file(path)
                is_video = kind_of(path) == "video"
                break
        if not media:
            emit("No video/audio source — skipping the voice clone (stills carry no voice)")
            return None
        voice = f"{safe_slug(name)}_voice"
        emit(f"Cloning voice '{voice}' from the source footage...")
        self.client.extract_voice(voice, f"voice of {name}", media, is_video, transcript)
        emit(f"Voice profile ready: {voice}")
        return voice

    # ── step 1b: an identity LoRA trained from the same crops ───────────────
    def _lora_jobs_path(self) -> Path:
        return self.out_dir / "lora_jobs.json"

    def _lora_jobs(self) -> dict[str, Any]:
        try:
            return json.loads(self._lora_jobs_path().read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _remember_lora_job(self, lora: str, job_id: str | None) -> None:
        """Persist the server-side job id so a restarted client re-attaches to a
        training already in flight instead of starting a second one."""
        jobs = self._lora_jobs()
        if job_id:
            jobs[lora] = {"job_id": job_id, "session": self.session_token(), "at": time.time()}
        else:
            jobs.pop(lora, None)
        self._lora_jobs_path().write_text(json.dumps(jobs, indent=2), encoding="utf-8")

    def session_token(self) -> str:
        """Stable per-out-dir token, so `?session=` recovery works across runs."""
        path = self.out_dir / "session.txt"
        try:
            token = path.read_text(encoding="utf-8").strip()
            if token:
                return token
        except Exception:
            pass
        token = f"charstudio-{uuid.uuid4().hex[:12]}"
        path.write_text(token, encoding="utf-8")
        return token

    def train_identity_lora(self, name: str, opts: dict[str, Any], emit=log) -> str | None:
        """Train a LoRA from a saved character profile's reference crops.

        The server pulls the images itself (`character: <profile>`), so this is the
        same set you reviewed and pruned — no re-upload. Returns the weights path.
        """
        lora_name = opts.get("lora_name") or f"{safe_slug(name)}_identity"
        models = self.client.list_models()
        target = (opts.get("target") or "video").strip()
        cap = "video_generation" if target == "video" else "image_generation"
        base_model = pick_model(
            models, cap,
            opts.get("base_model") or (self.args.video_model if target == "video" else self.args.image_model))
        if not base_model:
            raise RuntimeError(f"No {target} model available to train against — pass --base-model")

        profile = self.client.get_character(name)
        n_refs = int(profile.get("image_count") or 0)
        if not n_refs:
            raise RuntimeError(f"Character '{name}' has no reference images to train on")

        trigger = opts.get("trigger") or safe_slug(name)
        instance_prompt = opts.get("instance_prompt") or (
            f"a photo of {trigger} person" if target == "image" else f"{trigger} person")
        body = {
            "name": lora_name,
            "base_model": base_model,
            "target": target,
            "character": name,                 # server-side profile resolution
            "instance_prompt": instance_prompt,
            "steps": int(opts.get("steps") or 800),
            "rank": int(opts.get("rank") or 16),
            "learning_rate": float(opts.get("learning_rate") or 1e-4),
            "resolution": int(opts.get("resolution") or 512),
            "num_frames": int(opts.get("train_frames") or 1),
            "quantize_4bit": bool(opts.get("quantize_4bit", True)),
            "seed": int(opts.get("seed") or 42),
            "wait": False,
            "session": self.session_token(),
        }
        emit(f"Training LoRA '{lora_name}' on {base_model} ({target}) "
             f"from {n_refs} reference(s), {body['steps']} steps, rank {body['rank']}")
        emit(f"Trigger word: {trigger}  (put it in the scene prompt)")
        return self._attach_or_start_lora(lora_name, body, emit=emit,
                                          cancelled=opts.get("cancelled"))

    def _attach_or_start_lora(self, lora_name: str, body: dict[str, Any], emit=log,
                              cancelled=None) -> str | None:
        """Start (or re-attach to) a server-side training job and poll it to the end."""
        active = {"queued", "preparing", "training", "saving"}

        def kickoff() -> str:
            resp = self.client.train_lora(body)
            job_id = resp.get("job_id")
            if not job_id:
                raise RuntimeError(f"training did not return a job_id: {resp}")
            self._remember_lora_job(lora_name, job_id)
            return job_id

        known = (self._lora_jobs().get(lora_name) or {}).get("job_id")
        if known:
            progress = self.client.lora_progress(job=known)
            status = (progress.get("status") or "").strip()
            if status == "done" and progress.get("path"):
                emit(f"Re-attached: '{lora_name}' was already trained")
                self._remember_lora_job(lora_name, None)
                return progress.get("path")
            if status in active:
                emit(f"Re-attached to the running job for '{lora_name}'")
                job_id = known
            else:
                emit(f"Previous job was '{status or 'lost'}'; resubmitting (resumes from checkpoint)")
                job_id = kickoff()
        else:
            job_id = kickoff()

        started = time.time()
        resubmits = 0
        last_line = ""
        while True:
            if cancelled and cancelled():
                raise RuntimeError("cancelled")
            time.sleep(3.0)
            progress = self.client.lora_progress(job=job_id)
            status = (progress.get("status") or "").strip()
            elapsed = int(time.time() - started)
            mm, ss = divmod(elapsed, 60)
            et = f"{mm}m{ss:02d}s" if mm else f"{ss}s"
            if status == "done":
                self._remember_lora_job(lora_name, None)
                emit(f"LoRA '{lora_name}' trained in {et} -> {progress.get('path')}")
                return progress.get("path")
            if status == "error":
                self._remember_lora_job(lora_name, None)
                raise RuntimeError(progress.get("message") or "LoRA training failed")
            if status in {"interrupted", "unknown", ""}:
                if resubmits < 2:
                    resubmits += 1
                    emit(f"Job {status or 'lost'}; resubmitting to resume (#{resubmits})")
                    job_id = kickoff()
                    continue
                raise RuntimeError(f"training {status or 'unknown'} and could not be resumed")
            step, total = progress.get("step") or 0, progress.get("total") or body.get("steps") or 0
            line = (f"  {status} {step}/{total} ({et})" if step
                    else f"  {progress.get('message') or status} ({et})")
            if line != last_line:
                emit(line)
                last_line = line

    # ── step 2: a brand-new video with that character ───────────────────────
    def make_video(self, name: str, prompt: str, opts: dict[str, Any], emit=log) -> Path:
        models = self.client.list_models()
        video_model = pick_model(models, "video_generation", opts.get("video_model") or self.args.video_model)
        if not video_model:
            raise RuntimeError("No video model available — pass --video-model")
        image_model = pick_model(models, "image_generation", opts.get("image_model") or self.args.image_model)

        width = int(opts.get("width") or 768)
        height = int(opts.get("height") or 432)
        seed = int(opts.get("seed") or random.randint(1, 2**31 - 1))
        keyframe_identity = opts.get("keyframe_identity") or "auto"

        body: dict[str, Any] = {
            "model": video_model,
            "prompt": prompt,
            "negative_prompt": opts.get("negative_prompt") or DEFAULT_NEGATIVE,
            "width": width,
            "height": height,
            "fps": int(opts.get("fps") or 16),
            "num_frames": int(opts.get("num_frames") or 49),
            "num_inference_steps": int(opts.get("steps") or 25),
            "guidance_scale": float(opts.get("guidance_scale") or 5.0),
            "seed": seed,
            "mode": "t2v",
            # This is the whole point of the demo: the saved profile is resolved
            # server-side into IP-Adapter reference images.
            "character_profiles": [name],
            "character_strength": float(opts.get("character_strength") or 0.8),
            "keyframe_identity": keyframe_identity,
        }
        if keyframe_identity != "never" and image_model:
            body["keyframe_model"] = image_model
        if opts.get("keyframe_steps"):
            body["keyframe_steps"] = int(opts["keyframe_steps"])
        if opts.get("camera_motion"):
            body["camera_motion"] = opts["camera_motion"]
        say = (opts.get("say") or "").strip()
        if say:
            # A saved voice profile makes the server clone THAT voice for the line;
            # a bare id (e.g. af_sarah) is a stock TTS voice. Lip sync is applied to
            # the generated face either way.
            voice = (opts.get("voice") or f"{safe_slug(name)}_voice").strip()
            body["dialogs"] = [{"character": name, "voice": voice, "text": say,
                                "lip_sync": bool(opts.get("lip_sync", True)),
                                "speed": float(opts.get("speech_speed") or 1.0)}]
            body["lip_sync"] = bool(opts.get("lip_sync", True))
            if opts.get("lip_sync_method"):
                body["lip_sync_method"] = opts["lip_sync_method"]
            body["add_audio"] = True
            body["audio_type"] = "speech"
            emit(f"Dialog: '{say[:60]}' in voice '{voice}'"
                 + (" with lip sync" if body["lip_sync"] else ""))
        adapters = lora_specs(opts.get("loras"), float(opts.get("lora_weight") or 1.0))
        if adapters:
            body["loras"] = adapters

        emit(f"Video model: {video_model}  ({width}x{height}, seed {seed})")
        emit(f"Identity: character_profiles=['{name}'] strength={body['character_strength']} "
             f"keyframe_identity={keyframe_identity}")
        if adapters:
            emit("LoRAs: " + ", ".join(f"{a['id'][5:]}@{a['weight']}" for a in adapters))
        emit("Generating — this is the slow part...")
        started = time.time()
        mp4 = self.client.generate_video(body)

        videos_dir = self.out_dir / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{safe_slug(name)}_{safe_slug(prompt)[:48]}_{seed}"
        out_path = videos_dir / f"{stem}.mp4"
        out_path.write_bytes(mp4)
        (videos_dir / f"{stem}.json").write_text(json.dumps(body, indent=2), encoding="utf-8")
        emit(f"Wrote {out_path} ({len(mp4)/1e6:.1f} MB in {time.time()-started:.0f}s)")
        return out_path

    # ── optional: identity portrait, a fast check before the slow video ─────
    def make_portrait(self, name: str, prompt: str, opts: dict[str, Any], emit=log) -> Path | None:
        models = self.client.list_models()
        image_model = pick_model(models, "image_generation", opts.get("image_model") or self.args.image_model)
        if not image_model:
            emit("No image model available — skipping the identity portrait")
            return None
        emit(f"Rendering identity portrait with {image_model}...")
        body = {
            "model": image_model,
            "prompt": prompt,
            "size": f"{int(opts.get('width') or 768)}x{int(opts.get('height') or 432)}",
            "steps": int(opts.get("keyframe_steps") or 24),
            "character_profiles": [name],
            "character_strength": float(opts.get("character_strength") or 0.8),
        }
        # Only image-target LoRAs belong on the image endpoint.
        adapters = lora_specs(opts.get("image_loras"), float(opts.get("lora_weight") or 1.0))
        if adapters:
            body["loras"] = adapters
        png = self.client.generate_image(body)
        out_dir = self.out_dir / "portraits"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{safe_slug(name)}_{int(time.time())}.png"
        out_path.write_bytes(png)
        emit(f"Wrote {out_path}")
        return out_path

    # ── job bookkeeping (web UI) ────────────────────────────────────────────
    def _job_update(self, job_id: str, **updates: Any) -> None:
        with self.lock:
            job = self.jobs.setdefault(job_id, {"log": []})
            job.update(updates)
            job["updated_at"] = time.time()

    def _job_emit(self, job_id: str, line: str) -> None:
        stamped = f"[{time.strftime('%H:%M:%S')}] {line}"
        log(stamped)
        with self.lock:
            job = self.jobs.setdefault(job_id, {"log": []})
            job.setdefault("log", []).append(stamped)
            job["log"] = job["log"][-400:]
            job["message"] = line
            job["updated_at"] = time.time()

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self.lock:
            return dict(self.jobs.get(job_id) or {"status": "unknown"})

    def start_job(self, kind: str, target, payload: dict[str, Any]) -> str:
        job_id = f"{kind}-{uuid.uuid4().hex[:10]}"
        self._job_update(job_id, status="queued", progress=0, kind=kind, log=[])
        thread = threading.Thread(target=target, args=(job_id, payload), daemon=True)
        thread.start()
        return job_id

    def extract_job(self, job_id: str, payload: dict[str, Any]) -> None:
        emit = lambda line: self._job_emit(job_id, line)
        try:
            self._job_update(job_id, status="running", progress=10)
            name = (payload.get("name") or "").strip()
            if not name:
                raise RuntimeError("A character name is required")
            sources = [s for s in (payload.get("sources") or []) if s]
            if not sources:
                raise RuntimeError("Drop at least one photo or video")
            profile = self.extract(
                name, sources,
                description=payload.get("description") or "",
                max_images=int(payload.get("max_images") or 5),
                emit=emit,
                with_voice=bool(payload.get("with_voice")),
                transcript=payload.get("transcript") or "",
            )
            self._job_update(job_id, status="done", progress=100, result=profile)
        except Exception as exc:
            emit(f"FAILED: {exc}")
            self._job_update(job_id, status="error", error=str(exc))

    def train_job(self, job_id: str, payload: dict[str, Any]) -> None:
        emit = lambda line: self._job_emit(job_id, line)
        try:
            self._job_update(job_id, status="running", progress=10)
            name = (payload.get("name") or "").strip()
            if not name:
                raise RuntimeError("Pick a character profile to train from")
            payload = dict(payload)
            payload["cancelled"] = lambda: bool(self.get_job(job_id).get("cancel"))
            path = self.train_identity_lora(name, payload, emit=emit)
            self._job_update(job_id, status="done", progress=100, result={"path": path})
        except Exception as exc:
            emit(f"FAILED: {exc}")
            self._job_update(job_id, status="error", error=str(exc))

    def cancel_job(self, job_id: str) -> None:
        self._job_update(job_id, cancel=True, message="cancelling")

    def video_job(self, job_id: str, payload: dict[str, Any]) -> None:
        emit = lambda line: self._job_emit(job_id, line)
        try:
            self._job_update(job_id, status="running", progress=10)
            name = (payload.get("name") or "").strip()
            prompt = (payload.get("prompt") or "").strip()
            if not name or not prompt:
                raise RuntimeError("Both a character and a scene prompt are required")
            if payload.get("portrait"):
                self.make_portrait(name, prompt, payload, emit=emit)
                self._job_update(job_id, progress=30)
            path = self.make_video(name, prompt, payload, emit=emit)
            rel = path.relative_to(self.out_dir).as_posix()
            self._job_update(job_id, status="done", progress=100, output=rel,
                             output_url="/media/" + quote_rel(rel))
        except Exception as exc:
            emit(f"FAILED: {exc}")
            self._job_update(job_id, status="error", error=str(exc))

    def state_payload(self) -> dict[str, Any]:
        models = self.client.list_models()
        return {
            "characters": self.client.list_characters(),
            "loras": self.client.list_loras(),
            "voices": self.client.list_voices(),
            "models": [
                {"id": m.get("id"), "capabilities": m.get("capabilities") or []}
                for m in models
            ],
            "defaults": {
                "video_model": self.args.video_model or pick_model(models, "video_generation"),
                "image_model": self.args.image_model or pick_model(models, "image_generation"),
            },
            "base_url": self.args.base_url,
        }


def lora_specs(loras: Any, default_weight: float = 1.0) -> list[dict[str, Any]]:
    """Normalise "name", "name:0.8" or {"name":..,"weight":..} into request LoRAs.

    A registered LoRA is referenced server-side as `id: "name:<registered>"`.
    """
    out: list[dict[str, Any]] = []
    for item in loras or []:
        if isinstance(item, dict):
            name, weight = item.get("name") or item.get("id") or "", item.get("weight")
        else:
            name, _, raw = str(item).partition(":")
            weight = float(raw) if raw else None
        name = (name or "").strip()
        if not name:
            continue
        out.append({"id": f"name:{name}", "weight": float(weight if weight is not None else default_weight)})
    return out


def quote_rel(path: str) -> str:
    return "/".join(urllib.parse.quote(part) for part in path.split("/"))


HTML_PAGE = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CoderAI Character Studio</title>
<style>
:root{--bg:#10141c;--panel:#18202d;--panel2:#202b3b;--ink:#eef3ff;--muted:#9fb0c7;--line:#314057;--accent:#42d6a4;--warn:#f5b461;--bad:#ff6b6b;--blue:#78a6ff}
*{box-sizing:border-box}
body{margin:0;background:radial-gradient(circle at top left,#20344a 0,#10141c 38%,#0a0d13 100%);color:var(--ink);font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif}
header{padding:24px 32px;border-bottom:1px solid var(--line);background:linear-gradient(135deg,rgba(66,214,164,.16),rgba(120,166,255,.08))}
h1{margin:0;font-size:30px;letter-spacing:-.03em} h2{margin:0 0 12px;font-size:19px}
.sub{color:var(--muted);margin-top:6px;max-width:820px}
.wrap{display:grid;grid-template-columns:1fr 1fr;gap:18px;padding:18px;align-items:start}
.card{background:rgba(24,32,45,.92);border:1px solid var(--line);border-radius:18px;padding:16px;box-shadow:0 14px 34px rgba(0,0,0,.22)}
.card+.card{margin-top:18px}
label{display:block;font-size:12px;color:var(--muted);margin:10px 0 5px}
input,textarea,select{width:100%;border:1px solid var(--line);border-radius:12px;background:#0e141d;color:var(--ink);padding:10px 11px;font:inherit}
textarea{min-height:80px;resize:vertical}
.row{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.row3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px}
.btn{border:0;border-radius:12px;padding:10px 14px;background:var(--accent);color:#062015;font-weight:800;cursor:pointer;margin-top:12px}
.btn:disabled{opacity:.5;cursor:not-allowed}
.btn.secondary{background:#2a3548;color:var(--ink);border:1px solid var(--line)}
.btn.bad{background:var(--bad);color:#fff}
.drop{border:2px dashed var(--line);border-radius:14px;padding:22px;text-align:center;color:var(--muted);cursor:pointer;background:#0e141d}
.drop.hot{border-color:var(--accent);color:var(--ink)}
.files{margin-top:10px;display:flex;flex-wrap:wrap;gap:6px}
.pill{display:inline-block;padding:4px 9px;background:#111925;border:1px solid var(--line);border-radius:999px;color:var(--muted);font-size:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:10px;margin-top:10px}
.ref{background:var(--panel2);border:1px solid var(--line);border-radius:12px;overflow:hidden;position:relative;cursor:pointer}
.ref img{width:100%;height:120px;object-fit:cover;display:block;background:#0e141d}
.ref .idx{position:absolute;top:6px;left:6px;background:rgba(0,0,0,.65);border-radius:8px;padding:1px 7px;font-size:11px}
.ref.drop-me{outline:2px solid var(--bad);opacity:.55}
.ref.drop-me:after{content:"drop";position:absolute;bottom:6px;right:6px;background:var(--bad);color:#fff;border-radius:8px;padding:1px 7px;font-size:11px;font-weight:700}
.log{height:220px;overflow:auto;background:#070a0f;border:1px solid #253146;border-radius:14px;padding:12px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;white-space:pre-wrap;margin-top:12px}
.muted{color:var(--muted);font-size:13px}
video{width:100%;border-radius:14px;margin-top:12px;background:#000}
.chk{display:flex;align-items:center;gap:8px;margin-top:10px;color:var(--muted);font-size:13px}
.chk input{width:auto}
.warn{margin-top:12px;padding:10px 12px;border:1px solid var(--warn);border-radius:12px;color:var(--warn);font-size:12.5px;background:rgba(245,180,97,.07)}
@media(max-width:1000px){.wrap{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
  <h1>CoderAI Character Studio</h1>
  <div class="sub">Extract a character from real photos and videos, review the reference crops, then drop that same person into a brand-new generated scene. Every step is a plain CoderAI API call.</div>
</header>
<div class="wrap">
  <div>
    <div class="card">
      <h2>1 &middot; Extract from real footage</h2>
      <div class="row">
        <div><label>Character name</label><input id="name" placeholder="alice"></div>
        <div><label>Max reference images</label><input id="max_images" type="number" value="5" min="1" max="20"></div>
      </div>
      <label>Description (helps later prompts)</label>
      <input id="description" placeholder="woman, mid 30s, short dark hair, green eyes">
      <label>Source photos / videos</label>
      <div class="drop" id="drop">Drop files here, or click to choose<br><span class="muted">jpg &middot; png &middot; webp &middot; mp4 &middot; mov &middot; mkv &middot; webm</span></div>
      <input id="picker" type="file" multiple accept="image/*,video/*" style="display:none">
      <div class="files" id="filelist"></div>
      <div class="chk"><input id="with_voice" type="checkbox"><span>Also clone the voice from the same footage (needs a video/audio source)</span></div>
      <button class="btn" id="btn_extract">Extract character</button>
      <div class="warn">Files are base64-encoded into the request — keep source clips short (a few seconds is plenty; frames are sampled evenly).</div>
    </div>

    <div class="card">
      <h2>2 &middot; Reference crops</h2>
      <div class="row">
        <div><label>Profile</label><select id="charsel"></select></div>
        <div><label>&nbsp;</label><button class="btn secondary" id="btn_reload" style="margin-top:0">Reload</button></div>
      </div>
      <div class="muted" id="refnote">Pick a profile to see its references.</div>
      <div class="grid" id="refs"></div>
      <button class="btn bad" id="btn_prune">Drop selected references</button>
    </div>
  </div>

    <div class="card">
      <h2>3 &middot; Identity LoRA <span class="muted">(optional)</span></h2>
      <div class="muted">IP-Adapter alone drifts on long shots and profile angles. Training a small
      LoRA on the same crops locks the identity harder — minutes to hours, depending on the model.</div>
      <div class="row">
        <div><label>LoRA name</label><input id="lora_name" placeholder="(character)_identity"></div>
        <div><label>Trigger word</label><input id="trigger" placeholder="(character name)"></div>
      </div>
      <div class="row3">
        <div><label>Target</label><select id="train_target"><option value="video">video</option><option value="image">image</option></select></div>
        <div><label>Steps</label><input id="train_steps" type="number" value="800"></div>
        <div><label>Rank</label><input id="rank" type="number" value="16"></div>
      </div>
      <div class="row3">
        <div><label>Resolution</label><input id="train_resolution" type="number" value="512"></div>
        <div><label>Frames/sample</label><input id="train_frames" type="number" value="1"></div>
        <div><label>Learning rate</label><input id="learning_rate" type="number" step="0.00001" value="0.0001"></div>
      </div>
      <div class="chk"><input id="quantize_4bit" type="checkbox" checked><span>4-bit QLoRA base (needed to fit a large video model on one GPU)</span></div>
      <button class="btn" id="btn_train">Train identity LoRA</button>
      <button class="btn secondary" id="btn_cancel_train" style="margin-left:8px" disabled>Cancel</button>
    </div>
  </div>

  <div>
    <div class="card">
      <h2>4 &middot; New scene with that character</h2>
      <label>Scene prompt</label>
      <textarea id="prompt" placeholder="walking through a neon-lit Tokyo street at night, cinematic, shallow depth of field"></textarea>
      <div class="row">
        <div><label>Video model</label><select id="video_model"></select></div>
        <div><label>Keyframe image model</label><select id="image_model"></select></div>
      </div>
      <div class="row3">
        <div><label>Width</label><input id="width" type="number" value="768"></div>
        <div><label>Height</label><input id="height" type="number" value="432"></div>
        <div><label>Frames</label><input id="num_frames" type="number" value="49"></div>
      </div>
      <div class="row3">
        <div><label>FPS</label><input id="fps" type="number" value="16"></div>
        <div><label>Steps</label><input id="steps" type="number" value="25"></div>
        <div><label>Guidance</label><input id="guidance_scale" type="number" step="0.5" value="5"></div>
      </div>
      <div class="row3">
        <div><label>Identity strength</label><input id="character_strength" type="number" step="0.05" value="0.8"></div>
        <div><label>Keyframe identity</label><select id="keyframe_identity"><option value="auto">auto</option><option value="always">always</option><option value="never">never</option></select></div>
        <div><label>Camera motion</label><select id="camera_motion"><option value="">(none)</option><option>zoom-in</option><option>zoom-out</option><option>pan-left</option><option>pan-right</option><option>tilt-up</option><option>tilt-down</option><option>rotate</option></select></div>
      </div>
      <div class="row">
        <div><label>Apply LoRAs (ctrl-click for several)</label><select id="video_loras" multiple size="4"></select></div>
        <div><label>LoRA weight</label><input id="lora_weight" type="number" step="0.05" value="1.0"></div>
      </div>
      <label>Spoken line <span class="muted">(lip-synced, in the cloned voice)</span></label>
      <input id="say" placeholder="I told you I could ride">
      <div class="row3">
        <div><label>Voice</label><select id="voice"></select></div>
        <div><label>Speed</label><input id="speech_speed" type="number" step="0.05" value="1.0"></div>
        <div><label>Lip sync</label><select id="lip_sync"><option value="1">on</option><option value="0">off</option></select></div>
      </div>
      <label>Negative prompt</label>
      <input id="negative_prompt" placeholder="(server default)">
      <div class="chk"><input id="portrait" type="checkbox"><span>Render a still identity portrait first (fast sanity check before the slow video)</span></div>
      <button class="btn" id="btn_video">Generate video</button>
    </div>

    <div class="card">
      <h2>Progress</h2>
      <div class="log" id="log">Ready.</div>
      <div id="player"></div>
    </div>
  </div>
</div>
<script>
const ROOT_PATH="__ROOT_PATH__";   // mount prefix, injected server-side
const $=id=>document.getElementById(id);
let pending=[], state={characters:[],models:[],defaults:{}}, current=null, drops=new Set();

function esc(s){return String(s==null?'':s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
function logln(line){const el=$('log'); el.textContent+=(el.textContent?'\n':'')+line; el.scrollTop=el.scrollHeight}
function setLog(line){$('log').textContent=line}
async function api(path,opts){const r=await fetch(ROOT_PATH+path,Object.assign({headers:{'Content-Type':'application/json'}},opts||{})); const d=await r.json(); if(d.error) throw new Error(d.error); return d}

function readFile(file){return new Promise((res,rej)=>{const fr=new FileReader(); fr.onload=()=>res({name:file.name,data:fr.result}); fr.onerror=rej; fr.readAsDataURL(file)})}
function renderFiles(){$('filelist').innerHTML=pending.map(f=>`<span class="pill">${esc(f.name)}</span>`).join('')||'<span class="muted">No files selected.</span>'}
async function addFiles(list){for(const f of list){pending.push(await readFile(f))} renderFiles()}

$('drop').onclick=()=>$('picker').click();
$('picker').onchange=e=>addFiles(e.target.files);
$('drop').ondragover=e=>{e.preventDefault(); $('drop').classList.add('hot')};
$('drop').ondragleave=()=>$('drop').classList.remove('hot');
$('drop').ondrop=e=>{e.preventDefault(); $('drop').classList.remove('hot'); addFiles(e.dataTransfer.files)};

function selected(sel){return sel?[...sel.selectedOptions].map(o=>o.value):[]}
function options(list,sel){return list.map(v=>`<option ${v===sel?'selected':''}>${esc(v)}</option>`).join('')}
function capModels(cap){return state.models.filter(m=>(m.capabilities||[]).includes(cap)).map(m=>m.id)}

async function loadState(){
  state=await api('/api/state');
  const names=state.characters.map(c=>c.name);
  const keep=$('charsel').value;
  $('charsel').innerHTML=options(names, names.includes(keep)?keep:(state.characters[0]||{}).name);
  $('video_model').innerHTML=options(capModels('video_generation'), state.defaults.video_model);
  const voiceNames=(state.voices||[]).map(v=>v.name||v.id).filter(Boolean);
  const keptVoice=$('voice').value;
  $('voice').innerHTML='<option value="">(character\'s own voice)</option>'+options(voiceNames, keptVoice);
  const loraNames=(state.loras||[]).map(l=>l.name||l.id).filter(Boolean);
  const keptLoras=selected($('video_loras'));
  $('video_loras').innerHTML=loraNames.map(n=>`<option ${keptLoras.includes(n)?'selected':''}>${esc(n)}</option>`).join('')
    ||'<option disabled>no LoRAs trained yet</option>';
  $('image_model').innerHTML=options(capModels('image_generation'), state.defaults.image_model);
  if($('charsel').value) loadRefs($('charsel').value);
}

async function loadRefs(name){
  if(!name){$('refs').innerHTML=''; return}
  const d=await api('/api/character/'+encodeURIComponent(name));
  current=d; drops.clear();
  $('refnote').textContent=`${d.image_count} reference(s)${d.description?' — '+d.description:''}. Click a crop to mark it for removal (e.g. a bystander's face).`;
  $('refs').innerHTML=d.images.map((src,i)=>`<div class="ref" data-i="${i}" onclick="toggleDrop(${i})"><img src="${src}"><div class="idx">[${i}]</div></div>`).join('');
}
function toggleDrop(i){
  if(drops.has(i)) drops.delete(i); else drops.add(i);
  const el=document.querySelector('.ref[data-i="'+i+'"]');
  el.classList.toggle('drop-me', drops.has(i));
}

$('charsel').onchange=e=>loadRefs(e.target.value);
$('btn_reload').onclick=loadState;

$('btn_prune').onclick=async()=>{
  if(!current||!drops.size) return;
  await api('/api/character/'+encodeURIComponent(current.name)+'/prune',{method:'POST',body:JSON.stringify({drop:[...drops]})});
  logln('Dropped '+drops.size+' reference(s) from '+current.name);
  await loadRefs(current.name);
};

async function watch(job_id,done){
  const timer=setInterval(async()=>{
    const j=await api('/api/job/'+job_id);
    setLog((j.log||[]).join('\n'));
    if(j.status==='done'){clearInterval(timer); done(j)}
    if(j.status==='error'){clearInterval(timer); logln('Error: '+(j.error||'unknown'))
      $('btn_train').disabled=false; $('btn_cancel_train').disabled=true; $('btn_video').disabled=false}
  },1500);
}

$('btn_extract').onclick=async()=>{
  if(!pending.length){alert('Add at least one photo or video'); return}
  $('btn_extract').disabled=true; setLog('Uploading '+pending.length+' file(s)...');
  try{
    const d=await api('/api/extract',{method:'POST',body:JSON.stringify({
      name:$('name').value, description:$('description').value,
      max_images:+$('max_images').value||5, with_voice:$('with_voice').checked,
      sources:pending.map(f=>f.data)})});
    watch(d.job_id, async j=>{pending=[]; renderFiles(); await loadState(); if(j.result) {$('charsel').value=j.result.name; loadRefs(j.result.name)} $('btn_extract').disabled=false});
  }catch(e){logln('Error: '+e.message); $('btn_extract').disabled=false}
};

let trainJob=null;
$('btn_train').onclick=async()=>{
  const name=$('charsel').value;
  if(!name){alert('Extract or pick a character first'); return}
  $('btn_train').disabled=true; $('btn_cancel_train').disabled=false;
  setLog('Submitting training job...');
  try{
    const d=await api('/api/train',{method:'POST',body:JSON.stringify({
      name, lora_name:$('lora_name').value, trigger:$('trigger').value,
      target:$('train_target').value, base_model:$('video_model').value,
      steps:+$('train_steps').value, rank:+$('rank').value,
      resolution:+$('train_resolution').value, train_frames:+$('train_frames').value,
      learning_rate:+$('learning_rate').value, quantize_4bit:$('quantize_4bit').checked})});
    trainJob=d.job_id;
    watch(d.job_id, async()=>{$('btn_train').disabled=false; $('btn_cancel_train').disabled=true; trainJob=null; await loadState()});
  }catch(e){logln('Error: '+e.message); $('btn_train').disabled=false; $('btn_cancel_train').disabled=true}
};
$('btn_cancel_train').onclick=async()=>{
  if(!trainJob) return;
  await api('/api/job/'+trainJob+'/cancel',{method:'POST',body:'{}'});
  logln('Cancel requested — the run stops at the next poll (checkpoint is kept).');
};

$('btn_video').onclick=async()=>{
  const name=$('charsel').value;
  if(!name){alert('Extract or pick a character first'); return}
  if(!$('prompt').value.trim()){alert('Describe the new scene'); return}
  $('btn_video').disabled=true; setLog('Queued...');
  $('player').innerHTML='';
  try{
    const body={name, prompt:$('prompt').value,
      video_model:$('video_model').value, image_model:$('image_model').value,
      width:+$('width').value, height:+$('height').value, num_frames:+$('num_frames').value,
      fps:+$('fps').value, steps:+$('steps').value, guidance_scale:+$('guidance_scale').value,
      character_strength:+$('character_strength').value, keyframe_identity:$('keyframe_identity').value,
      camera_motion:$('camera_motion').value, negative_prompt:$('negative_prompt').value,
      loras:selected($('video_loras')), lora_weight:+$('lora_weight').value,
      say:$('say').value, voice:$('voice').value,
      lip_sync:$('lip_sync').value==='1', speech_speed:+$('speech_speed').value,
      portrait:$('portrait').checked};
    const d=await api('/api/video',{method:'POST',body:JSON.stringify(body)});
    watch(d.job_id, j=>{
      $('btn_video').disabled=false;
      if(j.output_url) $('player').innerHTML=`<video controls autoplay loop src="${ROOT_PATH+j.output_url}"></video><div class="muted" style="margin-top:8px">${esc(j.output)}</div>`;
    });
  }catch(e){logln('Error: '+e.message); $('btn_video').disabled=false}
};

loadState().catch(e=>logln('Error: '+e.message));
</script>
</body>
</html>
"""


def make_handler(studio: CharacterStudio):
    class Handler(BaseHTTPRequestHandler):
        server_version = "CoderAICharacterStudio/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:  # quieter console
            return

        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except BrokenPipeError:
                pass

        def _json(self, payload: Any, status: int = 200) -> None:
            self._send(status, json.dumps(payload).encode(), "application/json")

        def _public_prefix(self) -> str:
            """Path prefix this app is mounted under, per reverse-proxy headers.

            Returns e.g. '/character' (no trailing slash), or '' at the root. The
            page's fetches are built from it, so the same server works behind
            `location /character/ { ... }` whether or not nginx strips the prefix."""
            raw = (self.headers.get("X-Forwarded-Prefix")
                   or self.headers.get("X-Script-Name") or "")
            raw = raw.strip().rstrip("/")
            if not raw:
                return ""
            return raw if raw.startswith("/") else "/" + raw

        def _route(self, path: str) -> str:
            """Strip the forwarded prefix so internal routing is mount-agnostic."""
            prefix = self._public_prefix()
            if prefix and (path == prefix or path.startswith(prefix + "/")):
                path = path[len(prefix):] or "/"
            return path

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                return {}
            return json.loads(self.rfile.read(length) or b"{}")

        def _serve_media(self, rel: str) -> None:
            path = (studio.out_dir / urllib.parse.unquote(rel)).resolve()
            if not str(path).startswith(str(studio.out_dir.resolve())) or not path.is_file():
                self._json({"error": "not found"}, 404)
                return
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self._send(200, path.read_bytes(), ctype)

        def do_GET(self) -> None:
            path = self._route(urllib.parse.urlparse(self.path).path)
            try:
                if path in ("/", "/index.html"):
                    page = HTML_PAGE.replace("__ROOT_PATH__", self._public_prefix())
                    self._send(200, page.encode(), "text/html; charset=utf-8")
                elif path == "/api/state":
                    self._json(studio.state_payload())
                elif path.startswith("/api/character/"):
                    name = urllib.parse.unquote(path.split("/api/character/", 1)[1])
                    profile = studio.client.get_character(name)
                    self._json({
                        "name": profile.get("name", name),
                        "description": profile.get("description", ""),
                        "image_count": profile.get("image_count", 0),
                        "images": [img.get("data", "") for img in (profile.get("images") or [])],
                    })
                elif path.startswith("/api/job/"):
                    self._json(studio.get_job(path.rsplit("/", 1)[-1]))
                elif path.startswith("/media/"):
                    self._serve_media(path[len("/media/"):])
                else:
                    self._json({"error": "not found"}, 404)
            except BrokenPipeError:
                pass
            except Exception as exc:
                self._json({"error": str(exc)}, 500)

        def do_POST(self) -> None:
            path = self._route(urllib.parse.urlparse(self.path).path)
            try:
                payload = self._read_json()
                if path == "/api/extract":
                    self._json({"job_id": studio.start_job("extract", studio.extract_job, payload)})
                elif path == "/api/video":
                    self._json({"job_id": studio.start_job("video", studio.video_job, payload)})
                elif path == "/api/train":
                    self._json({"job_id": studio.start_job("train", studio.train_job, payload)})
                elif path.startswith("/api/job/") and path.endswith("/cancel"):
                    studio.cancel_job(path.split("/")[-2])
                    self._json({"ok": True})
                elif path.startswith("/api/character/") and path.endswith("/prune"):
                    name = urllib.parse.unquote(path[len("/api/character/"):-len("/prune")])
                    drop = [int(i) for i in (payload.get("drop") or [])]
                    studio.client.prune_character(name, drop)
                    studio.mirror_profile(name)
                    self._json({"ok": True})
                else:
                    self._json({"error": "not found"}, 404)
            except Exception as exc:
                self._json({"error": str(exc)}, 500)

    return Handler


# ── CLI ───────────────────────────────────────────────────────────────────────

def add_video_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prompt", required=True, help="Scene prompt for the NEW video")
    parser.add_argument("--negative-prompt", default="", help=f"Default: {DEFAULT_NEGATIVE!r}")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=432)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0, help="0 = random")
    parser.add_argument("--character-strength", type=float, default=0.8,
                        help="IP-Adapter scale for the character references")
    parser.add_argument("--keyframe-identity", choices=["auto", "always", "never"], default="auto",
                        help="Identity-keyframe bridge for video models without IP-Adapter")
    parser.add_argument("--keyframe-steps", type=int, default=0)
    parser.add_argument("--camera-motion", default="",
                        choices=["", "zoom-in", "zoom-out", "pan-left", "pan-right",
                                 "tilt-up", "tilt-down", "rotate"])
    parser.add_argument("--portrait", action="store_true",
                        help="Also render a still identity portrait (fast check before the slow video)")
    parser.add_argument("--say", default="",
                        help="Have the character speak this line, lip-synced, in their cloned voice")
    parser.add_argument("--voice", default="",
                        help="Voice profile or TTS voice id for --say (default: <character>_voice)")
    parser.add_argument("--no-lip-sync", action="store_true", help="Speak without lip sync")
    parser.add_argument("--speech-speed", type=float, default=1.0)
    parser.add_argument("--lora", action="append", default=[], metavar="NAME[:WEIGHT]",
                        help="Apply a trained LoRA to the video (repeatable, e.g. alice_identity:0.9)")
    parser.add_argument("--image-lora", action="append", default=[], metavar="NAME[:WEIGHT]",
                        help="LoRA for the still portrait / identity keyframe (repeatable)")


def video_opts(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "video_model": args.video_model,
        "image_model": args.image_model,
        "negative_prompt": args.negative_prompt,
        "width": args.width,
        "height": args.height,
        "num_frames": args.num_frames,
        "fps": args.fps,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "character_strength": args.character_strength,
        "keyframe_identity": args.keyframe_identity,
        "keyframe_steps": args.keyframe_steps,
        "camera_motion": args.camera_motion,
        "loras": args.lora,
        "image_loras": args.image_lora,
        "say": args.say,
        "voice": args.voice,
        "lip_sync": not args.no_lip_sync,
        "speech_speed": args.speech_speed,
    }


def add_train_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lora-name", default="", help="Output LoRA name (default: <character>_identity)")
    parser.add_argument("--target", choices=["video", "image"], default="video",
                        help="Pipeline the LoRA is trained for (default: video)")
    parser.add_argument("--base-model", default="", help="Model to train against (default: the studio's video/image model)")
    parser.add_argument("--trigger", default="", help="Trigger word (default: the character name)")
    parser.add_argument("--instance-prompt", default="", help="Override the training caption")
    parser.add_argument("--train-steps", type=int, default=800)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--train-resolution", type=int, default=512)
    parser.add_argument("--train-frames", type=int, default=1,
                        help="Frames per training sample (1 = stills, as most identity LoRAs are trained)")
    parser.add_argument("--no-quantize", action="store_true",
                        help="Train the base in bf16 instead of 4-bit QLoRA (needs a lot more VRAM)")


def train_opts(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "lora_name": args.lora_name,
        "target": args.target,
        "base_model": args.base_model,
        "trigger": args.trigger,
        "instance_prompt": args.instance_prompt,
        "steps": args.train_steps,
        "rank": args.rank,
        "learning_rate": args.learning_rate,
        "resolution": args.train_resolution,
        "train_frames": args.train_frames,
        "quantize_4bit": not args.no_quantize,
        "seed": getattr(args, "seed", 42) or 42,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CoderAI Character Studio — extract a character from real media, then generate a new video with them",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  # one shot: real footage in, new scene out\n"
            "  %(prog)s demo --name alice --source clips/alice.mp4 --source photos/ \\\n"
            "      --prompt 'walking through a neon-lit Tokyo street at night'\n\n"
            "  # step by step\n"
            "  %(prog)s extract --name alice --source clips/alice.mp4 --max-images 6\n"
            "  %(prog)s show alice\n"
            "  %(prog)s prune alice --drop 2 --drop 4      # a bystander got picked up\n"
            "  %(prog)s train alice --train-steps 1200     # identity LoRA from those crops\n"
            "  %(prog)s voice alice --source clips/alice.mp4  # clone her voice too\n"
            "  %(prog)s video alice --prompt 'riding a horse along a stormy beach' \\\n"
            "      --lora alice_identity:0.9 --say 'I told you I could ride'\n\n"
            "  # browser UI\n"
            "  %(prog)s web --browser\n"
        ),
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="CoderAI base URL")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="Bearer token for CoderAI")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Local output directory")
    parser.add_argument("--video-model", default="", help="Video model id (default: first with video_generation)")
    parser.add_argument("--image-model", default="", help="Image model id (default: first with image_generation)")

    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract", help="Extract a character profile from photos/videos")
    p_extract.add_argument("--name", required=True, help="Character profile name")
    p_extract.add_argument("--source", action="append", required=True, metavar="PATH|URL",
                           help="Photo, video, directory, glob or URL (repeatable)")
    p_extract.add_argument("--description", default="", help="Free-text description saved with the profile")
    p_extract.add_argument("--max-images", type=int, default=5, help="Max reference crops to keep")
    p_extract.add_argument("--voice", action="store_true",
                           help="Also clone the voice from the same footage (needs a video/audio source)")
    p_extract.add_argument("--transcript", default="",
                           help="Transcript of the reference audio (auto-transcribed when omitted)")

    sub.add_parser("list", help="List character profiles on the server")

    p_show = sub.add_parser("show", help="Download a profile's references locally + contact sheet")
    p_show.add_argument("name")

    p_prune = sub.add_parser("prune", help="Remove bad reference images by index")
    p_prune.add_argument("name")
    p_prune.add_argument("--drop", action="append", type=int, required=True,
                         help="0-based reference index to remove (repeatable)")

    p_delete = sub.add_parser("delete", help="Delete a character profile")
    p_delete.add_argument("name")

    p_train = sub.add_parser("train", help="Train an identity LoRA from a profile's reference crops")
    p_train.add_argument("name")
    p_train.add_argument("--seed", type=int, default=42)
    add_train_options(p_train)

    p_voice = sub.add_parser("voice", help="Clone a voice profile from a character's footage")
    p_voice.add_argument("name")
    p_voice.add_argument("--source", action="append", required=True, metavar="PATH|URL",
                         help="Video or audio carrying the voice (repeatable)")
    p_voice.add_argument("--transcript", default="")

    p_video = sub.add_parser("video", help="Generate a new video starring a saved character")
    p_video.add_argument("name")
    add_video_options(p_video)

    p_demo = sub.add_parser("demo", help="Extract, then generate a new video, in one go")
    p_demo.add_argument("--name", required=True)
    p_demo.add_argument("--source", action="append", required=True, metavar="PATH|URL")
    p_demo.add_argument("--description", default="")
    p_demo.add_argument("--max-images", type=int, default=5)
    p_demo.add_argument("--clone-voice", dest="clone_voice", action="store_true",
                        help="Also clone the voice from the same footage (use --say to speak with it)")
    p_demo.add_argument("--transcript", default="")
    p_demo.add_argument("--train", action="store_true",
                        help="Also train an identity LoRA from the crops and apply it to the video")
    add_video_options(p_demo)
    add_train_options(p_demo)

    p_web = sub.add_parser("web", help="Serve the browser UI")
    p_web.add_argument("--host", default="0.0.0.0", help="Listen host (default: 0.0.0.0)")
    p_web.add_argument("--web-port", type=int, default=7791, help="Listen port")
    p_web.add_argument("--browser", action="store_true", help="Open a browser after startup")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    studio = CharacterStudio(args)
    cmd = args.command

    if cmd == "web":
        server = ThreadingHTTPServer((args.host, args.web_port), make_handler(studio))
        url = f"http://{args.host}:{args.web_port}"
        log(f"Character Studio running at {url}")
        log(f"CoderAI: {args.base_url}")
        log(f"Output:  {studio.out_dir.resolve()}")
        if args.browser:
            import webbrowser
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            log("Stopping Character Studio")
        return 0

    if cmd == "list":
        profiles = studio.client.list_characters()
        if not profiles:
            log("No character profiles saved.")
            return 0
        for item in profiles:
            log(f"{item.get('name',''):<24} {item.get('image_count',0):>3} ref(s)  {item.get('description','')}")
        return 0

    if cmd == "show":
        info = studio.mirror_profile(args.name)
        log(f"{info['name']}: {info['image_count']} reference(s) -> {info['dir']}")
        if info.get("contact_sheet"):
            log(f"Contact sheet: {info['contact_sheet']}")
        return 0

    if cmd == "prune":
        studio.client.prune_character(args.name, args.drop)
        info = studio.mirror_profile(args.name)
        log(f"Dropped {len(set(args.drop))} reference(s); {info['image_count']} remain in '{info['name']}'")
        return 0

    if cmd == "delete":
        studio.client.delete_character(args.name)
        log(f"Deleted character '{args.name}'")
        return 0

    if cmd == "voice":
        voice = studio.extract_voice_profile(args.name, expand_sources(args.source),
                                             transcript=args.transcript)
        if not voice:
            return 1
        log(f"Use it with: video {args.name} --prompt '...' --say 'hello' --voice {voice}")
        return 0

    if cmd == "extract":
        studio.extract(args.name, expand_sources(args.source),
                       description=args.description, max_images=args.max_images,
                       with_voice=args.voice, transcript=args.transcript)
        log("Review the crops, drop any stray faces with `prune`, then run `video`.")
        return 0

    if cmd == "train":
        path = studio.train_identity_lora(args.name, train_opts(args))
        lora = args.lora_name or f"{safe_slug(args.name)}_identity"
        log(f"Apply it with: video {args.name} --prompt '...' --lora {lora}")
        return 0 if path else 1

    if cmd == "video":
        if args.portrait:
            studio.make_portrait(args.name, args.prompt, video_opts(args))
        studio.make_video(args.name, args.prompt, video_opts(args))
        return 0

    if cmd == "demo":
        log("[1/3] extracting the character from the supplied footage")
        profile = studio.extract(args.name, expand_sources(args.source),
                                 description=args.description, max_images=args.max_images,
                                 with_voice=args.clone_voice, transcript=args.transcript)
        if args.portrait:
            log("[2/3] identity portrait")
            studio.make_portrait(args.name, args.prompt, video_opts(args))
        else:
            log(f"[2/3] {profile['image_count']} reference(s) ready — skipping the portrait check")
        opts = video_opts(args)
        if args.train:
            log("[3/4] training an identity LoRA from those crops")
            lora = args.lora_name or f"{safe_slug(args.name)}_identity"
            studio.train_identity_lora(args.name, train_opts(args))
            opts["loras"] = list(opts.get("loras") or []) + [lora]
        total = 4 if args.train else 3
        log(f"[{total}/{total}] generating the new video")
        studio.make_video(args.name, args.prompt, opts)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
