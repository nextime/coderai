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
  - POST /v1/video/generations    generate a new video with `character_profiles`
                                  (IP-Adapter, or the identity-keyframe bridge on
                                  models without one)
  - POST /v1/images/generations   optional identity portrait, as a quick sanity check

Two front-ends over the same core:

  CLI     python tools/character_studio.py demo --name alice \
              --source clips/alice.mp4 --source photos/alice1.jpg \
              --prompt "walking through a neon-lit Tokyo street at night"

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
                max_images: int = 5, emit=log) -> dict[str, Any]:
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
        return mirrored

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

        emit(f"Video model: {video_model}  ({width}x{height}, seed {seed})")
        emit(f"Identity: character_profiles=['{name}'] strength={body['character_strength']} "
             f"keyframe_identity={keyframe_identity}")
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
        png = self.client.generate_image({
            "model": image_model,
            "prompt": prompt,
            "size": f"{int(opts.get('width') or 768)}x{int(opts.get('height') or 432)}",
            "steps": int(opts.get("keyframe_steps") or 24),
            "character_profiles": [name],
            "character_strength": float(opts.get("character_strength") or 0.8),
        })
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
            )
            self._job_update(job_id, status="done", progress=100, result=profile)
        except Exception as exc:
            emit(f"FAILED: {exc}")
            self._job_update(job_id, status="error", error=str(exc))

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

  <div>
    <div class="card">
      <h2>3 &middot; New scene with that character</h2>
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

function options(list,sel){return list.map(v=>`<option ${v===sel?'selected':''}>${esc(v)}</option>`).join('')}
function capModels(cap){return state.models.filter(m=>(m.capabilities||[]).includes(cap)).map(m=>m.id)}

async function loadState(){
  state=await api('/api/state');
  const names=state.characters.map(c=>c.name);
  const keep=$('charsel').value;
  $('charsel').innerHTML=options(names, names.includes(keep)?keep:(state.characters[0]||{}).name);
  $('video_model').innerHTML=options(capModels('video_generation'), state.defaults.video_model);
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
    if(j.status==='error'){clearInterval(timer); logln('Error: '+(j.error||'unknown'))}
  },1500);
}

$('btn_extract').onclick=async()=>{
  if(!pending.length){alert('Add at least one photo or video'); return}
  $('btn_extract').disabled=true; setLog('Uploading '+pending.length+' file(s)...');
  try{
    const d=await api('/api/extract',{method:'POST',body:JSON.stringify({
      name:$('name').value, description:$('description').value,
      max_images:+$('max_images').value||5, sources:pending.map(f=>f.data)})});
    watch(d.job_id, async j=>{pending=[]; renderFiles(); await loadState(); if(j.result) {$('charsel').value=j.result.name; loadRefs(j.result.name)} $('btn_extract').disabled=false});
  }catch(e){logln('Error: '+e.message); $('btn_extract').disabled=false}
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
            "  %(prog)s video alice --prompt 'riding a horse along a stormy beach'\n\n"
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

    sub.add_parser("list", help="List character profiles on the server")

    p_show = sub.add_parser("show", help="Download a profile's references locally + contact sheet")
    p_show.add_argument("name")

    p_prune = sub.add_parser("prune", help="Remove bad reference images by index")
    p_prune.add_argument("name")
    p_prune.add_argument("--drop", action="append", type=int, required=True,
                         help="0-based reference index to remove (repeatable)")

    p_delete = sub.add_parser("delete", help="Delete a character profile")
    p_delete.add_argument("name")

    p_video = sub.add_parser("video", help="Generate a new video starring a saved character")
    p_video.add_argument("name")
    add_video_options(p_video)

    p_demo = sub.add_parser("demo", help="Extract, then generate a new video, in one go")
    p_demo.add_argument("--name", required=True)
    p_demo.add_argument("--source", action="append", required=True, metavar="PATH|URL")
    p_demo.add_argument("--description", default="")
    p_demo.add_argument("--max-images", type=int, default=5)
    add_video_options(p_demo)

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

    if cmd == "extract":
        studio.extract(args.name, expand_sources(args.source),
                       description=args.description, max_images=args.max_images)
        log("Review the crops, drop any stray faces with `prune`, then run `video`.")
        return 0

    if cmd == "video":
        if args.portrait:
            studio.make_portrait(args.name, args.prompt, video_opts(args))
        studio.make_video(args.name, args.prompt, video_opts(args))
        return 0

    if cmd == "demo":
        log("[1/3] extracting the character from the supplied footage")
        profile = studio.extract(args.name, expand_sources(args.source),
                                 description=args.description, max_images=args.max_images)
        if args.portrait:
            log("[2/3] identity portrait")
            studio.make_portrait(args.name, args.prompt, video_opts(args))
        else:
            log(f"[2/3] {profile['image_count']} reference(s) ready — skipping the portrait check")
        log("[3/3] generating the new video")
        studio.make_video(args.name, args.prompt, video_opts(args))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
