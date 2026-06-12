# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Environment profile endpoints.

Saved environment profiles are named collections of reference images that
describe a setting, background, or location used to maintain visual
consistency of an environment across image/video generations.

POST   /v1/environments              – save / update an environment profile
POST   /v1/environments/extract      – extract from images and/or videos
POST   /v1/environments/generate     – generate from a text prompt
GET    /v1/environments              – list all saved profiles (no images)
GET    /v1/environments/{name}       – get a profile including base64 images
PATCH  /v1/environments/{name}       – update description / add / remove images
DELETE /v1/environments/{name}       – delete a profile
"""

import base64
import io
import json
import os
import subprocess
import tempfile
import time
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request


def _require_api_auth(request: Request) -> None:
    """Raise 401 if auth is enabled and the request carries no valid credential."""
    try:
        from codai.admin import routes as _admin_routes
        sm = _admin_routes.session_manager
    except Exception:
        return
    if sm is None:
        return
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        if sm.verify_token(auth[7:].strip()):
            return
    cookie = request.cookies.get("session", "")
    if cookie.endswith(".MUST_CHANGE"):
        cookie = cookie[:-12]
    if cookie and sm.validate_session(cookie):
        return
    raise HTTPException(
        status_code=401,
        detail={"message": "Invalid API key. Provide a valid Bearer token.",
                "type": "invalid_request_error", "code": "invalid_api_key"},
    )
from pydantic import BaseModel, ConfigDict

from codai.platform_paths import default_environments_dir, legacy_style_config_dir

router = APIRouter()

_ENVS_DIR: Optional[str] = None


def set_global_args(args):
    global _ENVS_DIR
    base = getattr(args, 'file_path', None) or str(legacy_style_config_dir())
    root = base if os.path.isdir(base) else (os.path.dirname(base) if base else str(legacy_style_config_dir()))
    _ENVS_DIR = os.path.join(root, 'environments')
    os.makedirs(_ENVS_DIR, exist_ok=True)


def set_global_file_path(path: str):
    pass  # not needed for environments


def _envs_dir() -> str:
    if _ENVS_DIR:
        return _ENVS_DIR
    d = str(default_environments_dir())
    os.makedirs(d, exist_ok=True)
    return d


def _env_dir(name: str) -> str:
    return os.path.join(_envs_dir(), name)


# ── Pydantic models ───────────────────────────────────────────────────────────

class EnvironmentImage(BaseModel):
    label: Optional[str] = None
    data: str                         # base64 image (with or without data: prefix)
    model_config = ConfigDict(extra="allow")


class EnvironmentSaveRequest(BaseModel):
    name: str
    description: Optional[str] = ""
    images: List[EnvironmentImage]
    model_config = ConfigDict(extra="allow")


class EnvironmentExtractRequest(BaseModel):
    name: str
    description: Optional[str] = ""
    images: Optional[List[str]] = None   # base64 source images
    videos: Optional[List[str]] = None   # base64/URL source videos
    max_images: Optional[int] = 5
    model_config = ConfigDict(extra="allow")


class EnvironmentGenerateRequest(BaseModel):
    name: str
    description: Optional[str] = ""
    prompt: str
    model: Optional[str] = None
    n: Optional[int] = 3
    steps: Optional[int] = None
    width: Optional[int] = 512
    height: Optional[int] = 512
    model_config = ConfigDict(extra="allow")


class EnvironmentPatchRequest(BaseModel):
    description: Optional[str] = None
    add_images: Optional[List[EnvironmentImage]] = None
    remove_indices: Optional[List[int]] = None
    model_config = ConfigDict(extra="allow")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save_environment(name: str, description: str, images: List[EnvironmentImage]) -> dict:
    edir = _env_dir(name)
    os.makedirs(edir, exist_ok=True)

    img_files = []
    for i, img in enumerate(images):
        raw = img.data
        if raw.startswith('data:'):
            _, b64 = raw.split(',', 1)
        else:
            b64 = raw
        img_bytes = base64.b64decode(b64)
        ext = '.png' if img_bytes[:4] == b'\x89PNG' else '.jpg'
        fname = f"ref{i:02d}{ext}"
        fpath = os.path.join(edir, fname)
        with open(fpath, 'wb') as f:
            f.write(img_bytes)
        img_files.append({'file': fname, 'label': img.label or f'ref{i}'})

    meta = {
        'name': name,
        'description': description,
        'images': img_files,
        'image_count': len(img_files),
        'created_at': int(time.time()),
    }
    with open(os.path.join(edir, 'meta.json'), 'w') as f:
        json.dump(meta, f)
    return meta


def _load_environment_meta(name: str) -> Optional[dict]:
    meta_path = os.path.join(_env_dir(name), 'meta.json')
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        return json.load(f)


def _load_environment_images(name: str) -> List[EnvironmentImage]:
    meta = _load_environment_meta(name)
    if not meta:
        return []
    edir = _env_dir(name)
    result = []
    for img_info in meta.get('images', []):
        fpath = os.path.join(edir, img_info['file'])
        if not os.path.exists(fpath):
            continue
        with open(fpath, 'rb') as f:
            raw = f.read()
        ext = img_info['file'].rsplit('.', 1)[-1]
        mime = 'image/png' if ext == 'png' else 'image/jpeg'
        b64 = base64.b64encode(raw).decode()
        result.append(EnvironmentImage(
            label=img_info.get('label'),
            data=f"data:{mime};base64,{b64}",
        ))
    return result


def _list_environments() -> list:
    d = _envs_dir()
    profiles = []
    for entry in os.scandir(d):
        if entry.is_dir():
            meta = _load_environment_meta(entry.name)
            if meta:
                profiles.append({k: v for k, v in meta.items() if k != 'images'})
    return sorted(profiles, key=lambda p: p.get('created_at', 0))


def _decode_source(data: str) -> bytes:
    if data.startswith('data:'):
        _, b64 = data.split(',', 1)
        return base64.b64decode(b64)
    if data.startswith('http://') or data.startswith('https://'):
        import urllib.request
        with urllib.request.urlopen(data, timeout=60) as r:
            return r.read()
    return base64.b64decode(data)


def _image_sharpness(img_bytes: bytes) -> float:
    try:
        import cv2
        import numpy as np
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
        return float(cv2.Laplacian(img, cv2.CV_64F).var())
    except Exception:
        return 0.0


def _normalise_image(img_bytes: bytes) -> bytes:
    """Convert any source image to a consistent PNG, resized to max 1024px on the long edge."""
    try:
        from PIL import Image as PILImage
        img = PILImage.open(io.BytesIO(img_bytes)).convert('RGB')
        img.thumbnail((1024, 1024), PILImage.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, 'PNG')
        return buf.getvalue()
    except Exception:
        return img_bytes


def _extract_frames_from_video(video_bytes: bytes, max_frames: int = 10) -> List[bytes]:
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, 'input.mp4')
        with open(in_path, 'wb') as f:
            f.write(video_bytes)

        probe = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=nw=1:nk=1', in_path],
            capture_output=True, text=True, timeout=30,
        )
        try:
            duration = float(probe.stdout.strip())
        except (ValueError, AttributeError):
            duration = 10.0

        frames_dir = os.path.join(tmpdir, 'frames')
        os.makedirs(frames_dir, exist_ok=True)
        fps_target = max(1, max_frames) / max(1.0, duration)
        subprocess.run(
            ['ffmpeg', '-y', '-i', in_path, '-vf', f'fps={fps_target:.4f}',
             '-frames:v', str(max_frames * 3), os.path.join(frames_dir, '%04d.png')],
            capture_output=True, timeout=120,
        )

        frame_files = sorted(os.listdir(frames_dir))
        frames = []
        for fname in frame_files:
            fpath = os.path.join(frames_dir, fname)
            with open(fpath, 'rb') as f:
                frames.append(f.read())
        return frames


def _select_best_frames(frames: List[bytes], max_count: int) -> List[bytes]:
    if not frames:
        return []
    scored = sorted(enumerate(frames), key=lambda t: _image_sharpness(t[1]), reverse=True)
    top = scored[:max(max_count * 2, 6)]
    top.sort(key=lambda t: t[0])
    step = max(1, len(top) // max_count)
    return [top[i][1] for i in range(0, len(top), step)][:max_count]


def resolve_environment_profiles(profile_names: List[str]) -> List[str]:
    """Resolve saved profile names → flat list of base64 image strings."""
    out = []
    for name in profile_names:
        for img in _load_environment_images(name):
            out.append(img.data)
    return out


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/v1/environments", summary="Create or replace an environment profile")
async def save_environment(req: EnvironmentSaveRequest, _auth=Depends(_require_api_auth)):
    """Save or update a named environment profile."""
    if not req.name or '/' in req.name or '..' in req.name:
        raise HTTPException(status_code=400, detail="Invalid environment name")
    if not req.images:
        raise HTTPException(status_code=400, detail="At least one reference image required")
    meta = _save_environment(req.name, req.description or '', req.images)
    return {"ok": True, "name": meta['name'], "image_count": meta['image_count']}


@router.get("/v1/environments", summary="List environment profiles")
async def list_environments(_auth=Depends(_require_api_auth)):
    """List all saved environment profiles (metadata only)."""
    return {"environments": _list_environments()}


@router.get("/v1/environments/{name}", summary="Get an environment profile")
async def get_environment(name: str, _auth=Depends(_require_api_auth)):
    """Get an environment profile including its reference images as base64."""
    meta = _load_environment_meta(name)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Environment '{name}' not found")
    images = _load_environment_images(name)
    return {
        "name": meta['name'],
        "description": meta.get('description', ''),
        "image_count": meta['image_count'],
        "created_at": meta['created_at'],
        "images": [img.model_dump() for img in images],
    }


@router.delete("/v1/environments/{name}", summary="Delete an environment profile")
async def delete_environment(name: str, _auth=Depends(_require_api_auth)):
    """Delete an environment profile."""
    edir = _env_dir(name)
    if not os.path.isdir(edir):
        raise HTTPException(status_code=404, detail=f"Environment '{name}' not found")
    import shutil
    shutil.rmtree(edir)
    return {"ok": True, "name": name}


@router.patch("/v1/environments/{name}", summary="Update an environment profile")
async def patch_environment(name: str, req: EnvironmentPatchRequest, _auth=Depends(_require_api_auth)):
    """Update an environment profile: description, add images, or remove images by index."""
    meta = _load_environment_meta(name)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Environment '{name}' not found")
    edir = _env_dir(name)

    img_files = list(meta.get('images', []))

    if req.remove_indices:
        for idx in sorted(set(req.remove_indices), reverse=True):
            if 0 <= idx < len(img_files):
                try:
                    os.unlink(os.path.join(edir, img_files[idx]['file']))
                except Exception:
                    pass
                img_files.pop(idx)

    if req.add_images:
        next_i = len(img_files)
        for img in req.add_images:
            raw = img.data
            if raw.startswith('data:'):
                _, b64 = raw.split(',', 1)
            else:
                b64 = raw
            img_bytes = base64.b64decode(b64)
            ext = '.png' if img_bytes[:4] == b'\x89PNG' else '.jpg'
            fname = f"ref{next_i:02d}{ext}"
            fpath = os.path.join(edir, fname)
            with open(fpath, 'wb') as f:
                f.write(img_bytes)
            img_files.append({'file': fname, 'label': img.label or f'ref{next_i}'})
            next_i += 1

    if req.description is not None:
        meta['description'] = req.description

    meta['images'] = img_files
    meta['image_count'] = len(img_files)
    with open(os.path.join(edir, 'meta.json'), 'w') as f:
        json.dump(meta, f)

    return {"ok": True, "name": name, "image_count": meta['image_count']}


@router.post("/v1/environments/generate", summary="Generate environment reference images")
async def generate_environment(req: EnvironmentGenerateRequest, request: Request):
    """
    Generate an environment profile from a text prompt.

    Calls the local image generation pipeline to produce `n` reference images,
    then saves them as a named environment profile.
    """
    if not req.name or '/' in req.name or '..' in req.name:
        raise HTTPException(status_code=400, detail="Invalid environment name")
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="A visual description prompt is required")

    n = max(1, min(8, req.n or 3))
    payload: dict = {
        "prompt": req.prompt,
        "n": n,
        "response_format": "b64_json",
        "size": f"{req.width or 512}x{req.height or 512}",
    }
    if req.model:
        payload["model"] = req.model
    if req.steps:
        payload["steps"] = req.steps

    try:
        import json as _json
        from codai.broker.asgi_bridge import execute_internal_request
        resp = await execute_internal_request(
            request.app,
            method="POST",
            path="/v1/images/generations",
            headers={"Content-Type": "application/json"},
            body=_json.dumps(payload).encode(),
        )
        if resp["status_code"] >= 400:
            try:
                detail = _json.loads(resp["body"]).get("detail", resp["body"].decode())
            except Exception:
                detail = resp["body"].decode()
            raise HTTPException(status_code=resp["status_code"], detail=f"Image generation failed: {detail}")

        images_data = _json.loads(resp["body"]).get("data", [])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation error: {e}")

    if not images_data:
        raise HTTPException(status_code=422, detail="No images were generated")

    env_images: List[EnvironmentImage] = []
    for i, item in enumerate(images_data):
        b64 = item.get("b64_json") or item.get("data", "")
        if not b64:
            continue
        if b64.startswith("data:"):
            header, b64_pure = b64.split(",", 1)
            mime = header.split(";")[0].split(":")[1] if ":" in header else "image/png"
        else:
            mime, b64_pure = "image/png", b64
        env_images.append(EnvironmentImage(
            label=f"generated_{i:02d}",
            data=f"data:{mime};base64,{b64_pure}",
        ))

    if not env_images:
        raise HTTPException(status_code=422, detail="Generated images could not be decoded")

    meta = _save_environment(req.name, req.description or req.prompt, env_images)
    return {"ok": True, "name": meta["name"], "image_count": meta["image_count"]}


@router.post("/v1/environments/extract", summary="Extract an environment from media")
async def extract_environment(req: EnvironmentExtractRequest):
    """
    Extract an environment profile from source images and/or videos.

    Whole frames are used as-is (no face cropping); the sharpest and most
    diverse frames are selected as reference images for the profile.
    """
    import asyncio

    if not req.name or '/' in req.name or '..' in req.name:
        raise HTTPException(status_code=400, detail="Invalid environment name")
    if not req.images and not req.videos:
        raise HTTPException(status_code=400, detail="Provide at least one image or video")

    max_per_source = max(1, req.max_images or 5)
    collected: List[bytes] = []

    if req.images:
        for src in req.images:
            try:
                img_bytes = await asyncio.get_event_loop().run_in_executor(
                    None, _decode_source, src)
                norm = await asyncio.get_event_loop().run_in_executor(
                    None, _normalise_image, img_bytes)
                collected.append(norm)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to process image: {e}")

    if req.videos:
        for src in req.videos:
            try:
                vid_bytes = await asyncio.get_event_loop().run_in_executor(
                    None, _decode_source, src)
                frames = await asyncio.get_event_loop().run_in_executor(
                    None, _extract_frames_from_video, vid_bytes, max_per_source * 4)
                best = await asyncio.get_event_loop().run_in_executor(
                    None, _select_best_frames, frames, max_per_source)
                normed = []
                for frame in best:
                    n = await asyncio.get_event_loop().run_in_executor(
                        None, _normalise_image, frame)
                    normed.append(n)
                collected.extend(normed)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to process video: {e}")

    if not collected:
        raise HTTPException(status_code=422, detail="No usable frames could be extracted")

    collected = collected[: req.max_images or 5]

    env_images = []
    for i, img_bytes in enumerate(collected):
        b64 = base64.b64encode(img_bytes).decode()
        mime = 'image/png' if img_bytes[:4] == b'\x89PNG' else 'image/jpeg'
        env_images.append(EnvironmentImage(
            label=f'extracted_{i:02d}',
            data=f'data:{mime};base64,{b64}',
        ))

    meta = _save_environment(req.name, req.description or '', env_images)
    return {"ok": True, "name": meta['name'], "image_count": meta['image_count']}
