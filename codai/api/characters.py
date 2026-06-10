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
Character profile endpoints.

Saved character profiles are named collections of reference images used to
maintain visual consistency of a character across image/video generations.

POST   /v1/characters              – save / update a character profile
POST   /v1/characters/extract      – extract character from images and/or videos
GET    /v1/characters              – list all saved profiles (no images)
GET    /v1/characters/{name}       – get a profile including base64 images
PATCH  /v1/characters/{name}       – update description / add / remove images
DELETE /v1/characters/{name}       – delete a profile
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
from pydantic import BaseModel, ConfigDict


def _require_api_auth(request: Request) -> None:
    """Raise 401 if auth is enabled and the request carries no valid credential."""
    try:
        from codai.admin import routes as _admin_routes
        sm = _admin_routes.session_manager
    except Exception:
        return  # auth subsystem unavailable — allow through
    if sm is None:
        return  # auth not configured on this instance

    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        token = auth[7:].strip()
        if sm.verify_token(token):
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

from codai.platform_paths import default_characters_dir, legacy_style_config_dir

router = APIRouter()

_CHARS_DIR: Optional[str] = None


def set_global_args(args):
    global _CHARS_DIR
    base = getattr(args, 'file_path', None) or str(legacy_style_config_dir())
    root = base if os.path.isdir(base) else (os.path.dirname(base) if base else str(legacy_style_config_dir()))
    _CHARS_DIR = os.path.join(root, 'characters')
    os.makedirs(_CHARS_DIR, exist_ok=True)


def set_global_file_path(path: str):
    pass  # not needed for characters


def _chars_dir() -> str:
    if _CHARS_DIR:
        return _CHARS_DIR
    d = str(default_characters_dir())
    os.makedirs(d, exist_ok=True)
    return d


def _char_dir(name: str) -> str:
    return os.path.join(_chars_dir(), name)


# ── Pydantic models ───────────────────────────────────────────────────────────

class CharacterImage(BaseModel):
    label: Optional[str] = None      # e.g. "front", "side", "close-up"
    data: str                         # base64 image (with or without data: prefix)
    model_config = ConfigDict(extra="allow")


class CharacterSaveRequest(BaseModel):
    name: str
    description: Optional[str] = ""
    images: List[CharacterImage]      # one or more reference images
    model_config = ConfigDict(extra="allow")


class CharacterExtractRequest(BaseModel):
    name: str
    description: Optional[str] = ""
    images: Optional[List[str]] = None   # base64 source images
    videos: Optional[List[str]] = None   # base64/URL source videos
    max_images: Optional[int] = 5        # max reference images to extract
    model_config = ConfigDict(extra="allow")


class CharacterGenerateRequest(BaseModel):
    name: str
    description: Optional[str] = ""
    prompt: str                            # visual description prompt
    model: Optional[str] = None           # image model to use
    n: Optional[int] = 3                  # number of reference images to generate
    steps: Optional[int] = None           # inference steps (None = model default)
    width: Optional[int] = 512
    height: Optional[int] = 512
    model_config = ConfigDict(extra="allow")


class CharacterPatchRequest(BaseModel):
    description: Optional[str] = None
    add_images: Optional[List[CharacterImage]] = None
    remove_indices: Optional[List[int]] = None   # 0-based indices to remove
    model_config = ConfigDict(extra="allow")


class CharacterProfile(BaseModel):
    name: str
    description: Optional[str] = ""
    image_count: int
    created_at: int
    images: Optional[List[CharacterImage]] = None  # only populated on GET /{name}
    model_config = ConfigDict(extra="allow")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save_character(name: str, description: str, images: List[CharacterImage]) -> dict:
    cdir = _char_dir(name)
    os.makedirs(cdir, exist_ok=True)

    img_files = []
    for i, img in enumerate(images):
        raw = img.data
        if raw.startswith('data:'):
            _, b64 = raw.split(',', 1)
        else:
            b64 = raw
        img_bytes = base64.b64decode(b64)
        # Detect PNG vs JPEG from magic bytes
        ext = '.png' if img_bytes[:4] == b'\x89PNG' else '.jpg'
        fname = f"ref{i:02d}{ext}"
        fpath = os.path.join(cdir, fname)
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
    with open(os.path.join(cdir, 'meta.json'), 'w') as f:
        json.dump(meta, f)
    return meta


def _load_character_meta(name: str) -> Optional[dict]:
    meta_path = os.path.join(_char_dir(name), 'meta.json')
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        return json.load(f)


def _load_character_images(name: str) -> List[CharacterImage]:
    meta = _load_character_meta(name)
    if not meta:
        return []
    cdir = _char_dir(name)
    result = []
    for img_info in meta.get('images', []):
        fpath = os.path.join(cdir, img_info['file'])
        if not os.path.exists(fpath):
            continue
        with open(fpath, 'rb') as f:
            raw = f.read()
        ext = img_info['file'].rsplit('.', 1)[-1]
        mime = 'image/png' if ext == 'png' else 'image/jpeg'
        b64 = base64.b64encode(raw).decode()
        result.append(CharacterImage(
            label=img_info.get('label'),
            data=f"data:{mime};base64,{b64}",
        ))
    return result


def _list_characters() -> list:
    d = _chars_dir()
    profiles = []
    for entry in os.scandir(d):
        if entry.is_dir():
            meta = _load_character_meta(entry.name)
            if meta:
                profiles.append({k: v for k, v in meta.items() if k != 'images'})
    return sorted(profiles, key=lambda p: p.get('created_at', 0))


def _decode_source(data: str) -> bytes:
    """Decode base64 or URL source into raw bytes."""
    if data.startswith('data:'):
        _, b64 = data.split(',', 1)
        return base64.b64decode(b64)
    if data.startswith('http://') or data.startswith('https://'):
        import urllib.request
        with urllib.request.urlopen(data, timeout=60) as r:
            return r.read()
    return base64.b64decode(data)


def _detect_faces_cv2(img_bytes: bytes):
    """
    Return list of (x,y,w,h) face rects, largest first.
    Tries MediaPipe (most accurate), then OpenCV DNN, then Haar cascade as fallback.
    Detections smaller than 2% of image area are discarded as false positives.
    Returns [] if no library is available or no plausible face is found.
    """
    try:
        import cv2
        import numpy as np
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return []
        ih, iw = img.shape[:2]
        img_area = ih * iw
        min_face_area = img_area * 0.02  # reject anything < 2% of image

        # ── Try MediaPipe first (most accurate, no model download needed) ──
        try:
            import mediapipe as mp
            mp_face = mp.solutions.face_detection
            with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as det:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = det.process(rgb)
            if results.detections:
                rects = []
                for d in results.detections:
                    bb = d.location_data.relative_bounding_box
                    x = int(bb.xmin * iw)
                    y = int(bb.ymin * ih)
                    w = int(bb.width * iw)
                    h = int(bb.height * ih)
                    if w * h >= min_face_area:
                        rects.append((x, y, w, h))
                if rects:
                    rects.sort(key=lambda r: r[2]*r[3], reverse=True)
                    return rects
        except ImportError:
            pass

        # ── Haar cascade fallback (stricter parameters to reduce false positives) ──
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        cascade = cv2.CascadeClassifier(cascade_path)
        # minSize scaled to image: at least 8% of the shorter dimension
        min_dim = int(min(iw, ih) * 0.08)
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=8,
            minSize=(max(40, min_dim), max(40, min_dim)),
        )
        if len(faces) == 0:
            return []
        rects = [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces
                 if int(w) * int(h) >= min_face_area]
        rects.sort(key=lambda r: r[2]*r[3], reverse=True)
        return rects
    except Exception:
        return []


def _crop_face(img_bytes: bytes, rect) -> Optional[bytes]:
    """Crop a face rect with generous padding (head-and-shoulders), return PNG bytes."""
    try:
        import cv2
        import numpy as np
        x, y, w, h = rect
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        ih, iw = img.shape[:2]
        side = max(w, h)
        # More padding on top to include hair/forehead, less at bottom
        pad_sides = int(side * 0.5)
        pad_top   = int(side * 0.7)
        pad_bot   = int(side * 0.4)
        x1 = max(0, x - pad_sides)
        y1 = max(0, y - pad_top)
        x2 = min(iw, x + w + pad_sides)
        y2 = min(ih, y + h + pad_bot)
        crop = img[y1:y2, x1:x2]
        ok, buf = cv2.imencode('.png', crop)
        return bytes(buf) if ok else None
    except Exception:
        return None


def _image_sharpness(img_bytes: bytes) -> float:
    """Return a sharpness score (Laplacian variance) for frame selection."""
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


def _extract_from_image(img_bytes: bytes) -> List[bytes]:
    """Return list of PNG crops from a single source image (one per detected face, or whole image)."""
    faces = _detect_faces_cv2(img_bytes)
    if faces:
        crops = [c for f in faces for c in [_crop_face(img_bytes, f)] if c]
        if crops:
            return crops
    # No face detected (or all detections filtered as false positives) — use whole image
    try:
        from PIL import Image as PILImage
        img = PILImage.open(io.BytesIO(img_bytes)).convert('RGB')
        buf = io.BytesIO()
        img.save(buf, 'PNG')
        return [buf.getvalue()]
    except Exception:
        return [img_bytes]


def _extract_frames_from_video(video_bytes: bytes, max_frames: int = 10) -> List[bytes]:
    """Extract diverse frames from a video using ffmpeg, return PNG byte lists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, 'input.mp4')
        with open(in_path, 'wb') as f:
            f.write(video_bytes)

        # Get duration
        probe = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=nw=1:nk=1', in_path],
            capture_output=True, text=True, timeout=30,
        )
        try:
            duration = float(probe.stdout.strip())
        except (ValueError, AttributeError):
            duration = 10.0

        # Extract evenly-spaced frames
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
    """Pick the sharpest frames, spaced across the video."""
    if not frames:
        return []
    scored = sorted(enumerate(frames), key=lambda t: _image_sharpness(t[1]), reverse=True)
    top = scored[:max(max_count * 2, 6)]
    top.sort(key=lambda t: t[0])  # restore temporal order
    step = max(1, len(top) // max_count)
    return [top[i][1] for i in range(0, len(top), step)][:max_count]


def resolve_character_profiles(profile_names: List[str]) -> List[str]:
    """Resolve saved profile names → flat list of base64 image strings."""
    out = []
    for name in profile_names:
        for img in _load_character_images(name):
            out.append(img.data)
    return out


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/v1/characters")
async def save_character(req: CharacterSaveRequest, _auth=Depends(_require_api_auth)):
    """Save or update a named character profile."""
    if not req.name or '/' in req.name or '..' in req.name:
        raise HTTPException(status_code=400, detail="Invalid character name")
    if not req.images:
        raise HTTPException(status_code=400, detail="At least one reference image required")
    meta = _save_character(req.name, req.description or '', req.images)
    return {"ok": True, "name": meta['name'], "image_count": meta['image_count']}


@router.get("/v1/characters")
async def list_characters(_auth=Depends(_require_api_auth)):
    """List all saved character profiles (metadata only, no images)."""
    return {"characters": _list_characters()}


@router.get("/v1/characters/{name}")
async def get_character(name: str, _auth=Depends(_require_api_auth)):
    """Get a character profile including its reference images as base64."""
    meta = _load_character_meta(name)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Character '{name}' not found")
    images = _load_character_images(name)
    return {
        "name": meta['name'],
        "description": meta.get('description', ''),
        "image_count": meta['image_count'],
        "created_at": meta['created_at'],
        "images": [img.model_dump() for img in images],
    }


@router.delete("/v1/characters/{name}")
async def delete_character(name: str, _auth=Depends(_require_api_auth)):
    """Delete a character profile."""
    cdir = _char_dir(name)
    if not os.path.isdir(cdir):
        raise HTTPException(status_code=404, detail=f"Character '{name}' not found")
    import shutil
    shutil.rmtree(cdir)
    return {"ok": True, "name": name}


@router.patch("/v1/characters/{name}")
async def patch_character(name: str, req: CharacterPatchRequest, _auth=Depends(_require_api_auth)):
    """Update a character profile: description, add images, or remove images by index."""
    meta = _load_character_meta(name)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Character '{name}' not found")
    cdir = _char_dir(name)

    img_files = list(meta.get('images', []))

    # Remove by index (highest first so indices stay valid)
    if req.remove_indices:
        for idx in sorted(set(req.remove_indices), reverse=True):
            if 0 <= idx < len(img_files):
                try:
                    os.unlink(os.path.join(cdir, img_files[idx]['file']))
                except Exception:
                    pass
                img_files.pop(idx)

    # Add new images
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
            fpath = os.path.join(cdir, fname)
            with open(fpath, 'wb') as f:
                f.write(img_bytes)
            img_files.append({'file': fname, 'label': img.label or f'ref{next_i}'})
            next_i += 1

    if req.description is not None:
        meta['description'] = req.description

    meta['images'] = img_files
    meta['image_count'] = len(img_files)
    with open(os.path.join(cdir, 'meta.json'), 'w') as f:
        json.dump(meta, f)

    return {"ok": True, "name": name, "image_count": meta['image_count']}


@router.post("/v1/characters/generate")
async def generate_character(req: CharacterGenerateRequest, request: Request):
    """
    Generate a character profile from a text prompt.

    Calls the local image generation pipeline to produce `n` reference images,
    then saves them as a named character profile.
    """
    if not req.name or '/' in req.name or '..' in req.name:
        raise HTTPException(status_code=400, detail="Invalid character name")
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

    char_images: List[CharacterImage] = []
    for i, item in enumerate(images_data):
        b64 = item.get("b64_json") or item.get("data", "")
        if not b64:
            continue
        if b64.startswith("data:"):
            header, b64_pure = b64.split(",", 1)
            mime = header.split(";")[0].split(":")[1] if ":" in header else "image/png"
        else:
            mime, b64_pure = "image/png", b64
        char_images.append(CharacterImage(
            label=f"generated_{i:02d}",
            data=f"data:{mime};base64,{b64_pure}",
        ))

    if not char_images:
        raise HTTPException(status_code=422, detail="Generated images could not be decoded")

    meta = _save_character(req.name, req.description or req.prompt, char_images)
    return {"ok": True, "name": meta["name"], "image_count": meta["image_count"]}


@router.post("/v1/characters/extract")
async def extract_character(req: CharacterExtractRequest):
    """
    Extract a character profile from source images and/or videos.

    Images and videos are analysed for faces; the best crops are saved as
    reference images for the named character profile.
    """
    import asyncio

    if not req.name or '/' in req.name or '..' in req.name:
        raise HTTPException(status_code=400, detail="Invalid character name")
    if not req.images and not req.videos:
        raise HTTPException(status_code=400, detail="Provide at least one image or video")

    max_per_source = max(1, (req.max_images or 5))
    collected: List[bytes] = []

    # --- images ---
    if req.images:
        for src in req.images:
            try:
                img_bytes = await asyncio.get_event_loop().run_in_executor(
                    None, _decode_source, src)
                crops = await asyncio.get_event_loop().run_in_executor(
                    None, _extract_from_image, img_bytes)
                collected.extend(crops[:max_per_source])
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to process image: {e}")

    # --- videos ---
    if req.videos:
        for src in req.videos:
            try:
                vid_bytes = await asyncio.get_event_loop().run_in_executor(
                    None, _decode_source, src)
                frames = await asyncio.get_event_loop().run_in_executor(
                    None, _extract_frames_from_video, vid_bytes, max_per_source * 4)
                best = await asyncio.get_event_loop().run_in_executor(
                    None, _select_best_frames, frames, max_per_source * 2)
                face_crops: List[bytes] = []
                for frame in best:
                    crops = await asyncio.get_event_loop().run_in_executor(
                        None, _extract_from_image, frame)
                    face_crops.extend(crops)
                collected.extend(face_crops[:max_per_source])
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to process video: {e}")

    if not collected:
        raise HTTPException(status_code=422, detail="No usable frames could be extracted")

    # Deduplicate and trim
    collected = collected[: req.max_images or 5]

    # Build CharacterImage list from raw PNG bytes
    char_images = []
    for i, img_bytes in enumerate(collected):
        b64 = base64.b64encode(img_bytes).decode()
        mime = 'image/png' if img_bytes[:4] == b'\x89PNG' else 'image/jpeg'
        char_images.append(CharacterImage(
            label=f'extracted_{i:02d}',
            data=f'data:{mime};base64,{b64}',
        ))

    meta = _save_character(req.name, req.description or '', char_images)
    return {"ok": True, "name": meta['name'], "image_count": meta['image_count']}
