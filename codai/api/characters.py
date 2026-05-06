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
maintain visual consistency of a character across multiple video generations.

POST   /v1/characters              – save / update a character profile
GET    /v1/characters              – list all saved profiles (no images)
GET    /v1/characters/{name}       – get a profile including base64 images
DELETE /v1/characters/{name}       – delete a profile
"""

import base64
import json
import os
import time
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

router = APIRouter()

_CHARS_DIR: Optional[str] = None


def set_global_args(args):
    global _CHARS_DIR
    base = getattr(args, 'file_path', None) or os.path.expanduser('~/.coderai')
    root = base if os.path.isdir(base) else (os.path.dirname(base) if base else os.path.expanduser('~/.coderai'))
    _CHARS_DIR = os.path.join(root, 'characters')
    os.makedirs(_CHARS_DIR, exist_ok=True)


def set_global_file_path(path: str):
    pass  # not needed for characters


def _chars_dir() -> str:
    if _CHARS_DIR:
        return _CHARS_DIR
    d = os.path.expanduser('~/.coderai/characters')
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


def resolve_character_profiles(profile_names: List[str]) -> List[str]:
    """Resolve saved profile names → flat list of base64 image strings."""
    out = []
    for name in profile_names:
        for img in _load_character_images(name):
            out.append(img.data)
    return out


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/v1/characters")
async def save_character(req: CharacterSaveRequest):
    """Save or update a named character profile."""
    if not req.name or '/' in req.name or '..' in req.name:
        raise HTTPException(status_code=400, detail="Invalid character name")
    if not req.images:
        raise HTTPException(status_code=400, detail="At least one reference image required")
    meta = _save_character(req.name, req.description or '', req.images)
    return {"ok": True, "name": meta['name'], "image_count": meta['image_count']}


@router.get("/v1/characters")
async def list_characters():
    """List all saved character profiles (metadata only, no images)."""
    return {"characters": _list_characters()}


@router.get("/v1/characters/{name}")
async def get_character(name: str):
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
async def delete_character(name: str):
    """Delete a character profile."""
    cdir = _char_dir(name)
    if not os.path.isdir(cdir):
        raise HTTPException(status_code=404, detail=f"Character '{name}' not found")
    import shutil
    shutil.rmtree(cdir)
    return {"ok": True, "name": name}
